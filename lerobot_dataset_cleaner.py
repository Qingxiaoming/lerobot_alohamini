#!/usr/bin/env python3
"""
LeRobot Dataset Cleaner & Merger (v3.0-aware)
=============================================
可靠地清洗、降采样、合并 LeRobot v3.0（以及部分旧版）数据集。

特性：
- clean: 删除指定 episode、删除 frozen 帧、降采样 FPS、重新组织 chunk
- merge: 合并多个数据集，支持自动 FPS 对齐；底层调用 LeRobot 官方 aggregate_datasets
          以保证视频合并正确。
- 重新计算 stats.json（与 LeRobot 内部 RunningQuantileStats 一致，包含 quantiles）。
- 输出格式为 LeRobot v3.0，每个 episode 独占一个 chunk（data/chunk-{ep:03d}/file-000.parquet），
  从而消除 data/meta mismatch。

使用 conda 环境运行（确保已安装 pyarrow / lerobot）：
    conda run -n lerobot_alohamini python lerobot_dataset_cleaner.py ...
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.datasets.compute_stats import RunningQuantileStats, estimate_num_samples
from lerobot.datasets.utils import serialize_dict


# =============================================================================
# Helpers
# =============================================================================

def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_ffmpeg(cmd: list, check: bool = True):
    full_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + cmd
    try:
        subprocess.run(full_cmd, check=check, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\nFFmpeg error:\n{e.stderr}")
        raise


def get_video_frame_count(video_path: Path) -> int:
    if not video_path.exists():
        return 0
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        return int(out)
    except Exception:
        return 0


def get_video_fps(video_path: Path) -> float:
    if not video_path.exists():
        return 0.0
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        num, den = out.split("/")
        return float(num) / float(den)
    except Exception:
        return 0.0


def get_video_duration(video_path: Path) -> float:
    if not video_path.exists():
        return 0.0
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def get_video_info(video_path: Path) -> Optional[Tuple[int, int, int]]:
    """一次 ffprobe 调用返回 (frame_count, width, height)。失败返回 None。"""
    if not video_path.exists():
        return None
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets,width,height",
        "-of", "json",
        str(video_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        data = json.loads(out)
        stream = data["streams"][0]
        count = int(stream["nb_read_packets"])
        w = int(stream["width"])
        h = int(stream["height"])
        return count, w, h
    except Exception:
        return None


def copy_video_segment(input_path: Path, output_path: Path, start_frame: int, num_frames: int, fps: float):
    if not input_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = start_frame / fps
    duration = num_frames / fps
    cmd = [
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", str(input_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-an",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def concat_video_segments(segment_paths: List[Path], output_path: Path, fps: float):
    valid = [p for p in segment_paths if p.exists() and p.stat().st_size > 0]
    if not valid:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(valid) == 1:
        shutil.copy2(valid[0], output_path)
        return
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in valid:
            f.write(f"file '{p.absolute()}'\n")
        concat_list = f.name
    try:
        cmd = [
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-an",
            str(output_path),
        ]
        run_ffmpeg(cmd)
    finally:
        os.unlink(concat_list)


def downsample_video(input_path: Path, output_path: Path, source_fps: float, target_fps: float):
    if not input_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ratio = int(round(source_fps / target_fps))
    cmd = [
        "-i", str(input_path),
        "-vf", f"select=not(mod(n\\,{ratio})),setpts=N/FRAME_RATE/TB",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(target_fps),
        "-an",
        str(output_path),
    ]
    run_ffmpeg(cmd)


# =============================================================================
# Frozen frame detection
# =============================================================================

def find_frozen_frames(actions: np.ndarray, threshold: float = 0.001,
                       min_consecutive_frames: int = 3) -> Set[int]:
    """
    检测需要删除的 frozen 帧。

    规则：若连续 `min_consecutive_frames` 帧的 action L2 变化量 < threshold，
    只保留该段的第一帧，删除其余帧。

    返回：需要删除的帧索引（0-based）。
    """
    if len(actions) < 2:
        return set()

    diffs = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    frozen = diffs < threshold

    to_remove = set()
    i = 0
    while i < len(frozen):
        if frozen[i]:
            start = i
            while i < len(frozen) and frozen[i]:
                i += 1
            end = i - 1  # 最后一个 frozen diff 的索引
            num_frames = end - start + 1  # frozen 帧的数量（不包含参考帧）
            if num_frames >= min_consecutive_frames:
                # 删除该段中除第一帧外的所有 frozen 帧
                for idx in range(start + 2, end + 2):
                    if idx < len(actions):
                        to_remove.add(idx)
        else:
            i += 1
    return to_remove


def keep_ranges(total_frames: int, remove_indices: Set[int]) -> List[Tuple[int, int]]:
    if not remove_indices:
        return [(0, total_frames)]
    sorted_remove = sorted(remove_indices)
    ranges = []
    start = 0
    for idx in sorted_remove:
        if idx > start:
            ranges.append((start, idx - start))
        start = idx + 1
    if start < total_frames:
        ranges.append((start, total_frames - start))
    return ranges


# =============================================================================
# Dataset I/O
# =============================================================================

def load_info(root: Path) -> dict:
    return load_json(root / "meta" / "info.json")


def save_info(root: Path, info: dict):
    save_json(root / "meta" / "info.json", info)


def load_tasks(root: Path) -> pd.DataFrame:
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        return pd.read_parquet(tasks_path)
    return pd.DataFrame({"task_index": []}, index=pd.Index([], name="task"))


def save_tasks(root: Path, tasks: pd.DataFrame):
    path = root / "meta" / "tasks.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(path)


def read_all_episodes(root: Path) -> Dict[int, pd.DataFrame]:
    """
    读取 data/ 下所有 parquet，按 episode_index 分组，返回 {old_ep_idx: DataFrame}。
    兼容 v3 分片（多 episode 每文件）以及旧版 episode_xxx.parquet。
    """
    data_dir = root / "data"
    files = sorted(data_dir.glob("**/*.parquet"))
    if not files:
        return {}

    tables = [pq.read_table(f) for f in files]
    combined = pa.concat_tables(tables)
    df = combined.to_pandas()
    # 优先按全局 index 排序；若无 index 则按 episode_index + frame_index
    if "index" in df.columns:
        df = df.sort_values("index")
    else:
        df = df.sort_values(["episode_index", "frame_index"])

    episodes = {}
    for ep_idx, group in df.groupby("episode_index", sort=True):
        episodes[int(ep_idx)] = group.reset_index(drop=True)
    return episodes


def build_target_data_schema(source_schema: pa.Schema) -> pa.Schema:
    """保留原始数据列（去掉可能残留的 data/chunk_index、data/file_index）。

    LeRobot v3.0 的数据 parquet 只包含特征列和索引列；chunk/file 映射保存在
    meta/episodes.parquet 中，因此不要写入 data/chunk_index、data/file_index。
    """
    base_cols = [n for n in source_schema.names if n not in ("data/chunk_index", "data/file_index")]
    fields = [pa.field(n, source_schema.field(n).type) for n in base_cols]
    return pa.schema(fields)


def write_data_parquet(df: pd.DataFrame, schema: pa.Schema, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # 保证列顺序与 schema 一致
    df = df[[f.name for f in schema if f.name in df.columns]]
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def write_episodes_parquet(rows: List[dict], schema: pa.Schema, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in schema.names:
        if col not in df.columns:
            df[col] = None
    df = df[[f.name for f in schema]]
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def build_episodes_schema(info: dict, has_videos: bool) -> pa.Schema:
    fields = [
        pa.field("episode_index", pa.int64()),
        pa.field("tasks", pa.list_(pa.string())),
        pa.field("length", pa.int64()),
        pa.field("data/chunk_index", pa.int64()),
        pa.field("data/file_index", pa.int64()),
        pa.field("dataset_from_index", pa.int64()),
        pa.field("dataset_to_index", pa.int64()),
    ]
    if has_videos:
        for key in info.get("features", {}):
            if info["features"][key].get("dtype") == "video":
                fields += [
                    pa.field(f"videos/{key}/chunk_index", pa.int64()),
                    pa.field(f"videos/{key}/file_index", pa.int64()),
                    pa.field(f"videos/{key}/from_timestamp", pa.float64()),
                    pa.field(f"videos/{key}/to_timestamp", pa.float64()),
                ]
    fields += [
        pa.field("meta/episodes/chunk_index", pa.int64()),
        pa.field("meta/episodes/file_index", pa.int64()),
    ]
    return pa.schema(fields)


# =============================================================================
# Video discovery / extraction
# =============================================================================

def discover_video_files(root: Path, info: dict) -> Dict[str, Dict[int, Path]]:
    """
    尝试发现每个 camera / episode 对应的视频文件。
    支持：
      1. v3 旧版 per-episode: videos/{key}/chunk-*/episode_*.mp4
      2. v3 标准 per-episode: videos/{key}/chunk-{ep:03d}/file-000.mp4
    返回：{camera_key: {episode_index: Path}}。
    """
    result: Dict[str, Dict[int, Path]] = {}
    video_dir = root / "videos"
    if not video_dir.exists():
        return result
    for key in info.get("features", {}):
        if info["features"][key].get("dtype") != "video":
            continue
        cam_dir = video_dir / key
        if not cam_dir.exists():
            continue
        cam_files = sorted(cam_dir.rglob("*.mp4"))
        mapping: Dict[int, Path] = {}
        for f in cam_files:
            # 尝试从文件名解析 episode index
            if "episode_" in f.name:
                try:
                    ep = int(f.stem.split("episode_")[-1])
                    mapping[ep] = f
                except Exception:
                    pass
            elif f.name.startswith("file-"):
                # 标准 v3 file-000.mp4，需要配合目录 chunk-{idx}
                try:
                    chunk_idx = int(f.parent.name.split("-")[-1])
                    file_idx = int(f.stem.split("-")[-1])
                    # 若 chunk 与 episode 一一对应，则 chunk_idx 即 episode index
                    mapping[chunk_idx] = f
                except Exception:
                    pass
        if mapping:
            result[key] = mapping
    return result


def has_any_video(root: Path, info: dict) -> bool:
    files = discover_video_files(root, info)
    return any(bool(v) for v in files.values())


def extract_video_keep_ranges(
    input_path: Path,
    output_path: Path,
    keep_ranges: List[Tuple[int, int]],
    source_fps: float,
    target_fps: float,
):
    """用单个 ffmpeg filter 保留指定帧范围，重新编码为 H.264。

    keep_ranges 是 **目标 FPS** 下的帧范围；函数会根据 source_fps/target_fps
    自动缩放到源视频帧号。这样即使源视频 FPS 与目标不一致（例如 30fps 源被合并到
    15fps 数据集）也能对齐。
    """
    if not input_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = get_video_frame_count(input_path)

    # 缩放到源视频帧号
    ratio = source_fps / target_fps if target_fps > 0 and source_fps > 0 else 1.0
    if abs(ratio - 1.0) > 1e-6:
        scaled_ranges: List[Tuple[int, int]] = []
        for start, num in keep_ranges:
            if num <= 0:
                continue
            src_start = int(round(start * ratio))
            src_end = int(round((start + num) * ratio))
            scaled_ranges.append((src_start, max(0, src_end - src_start)))
        keep_ranges = scaled_ranges

    if keep_ranges == [(0, total_frames)]:
        shutil.copy2(input_path, output_path)
        return

    # 构建 select 表达式：between(n\,start\,end) 之和
    parts = []
    for start, num in keep_ranges:
        if num <= 0:
            continue
        end = start + num - 1
        parts.append(f"between(n\\,{start}\\,{end})")
    if not parts:
        return
    select_expr = "+".join(parts)

    cmd = [
        "-i", str(input_path),
        "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(target_fps),
        "-an",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def extract_or_copy_episode_video(
    input_root: Path,
    output_root: Path,
    camera_key: str,
    old_ep_idx: int,
    new_ep_idx: int,
    fps: float,
    keep_ranges: Optional[List[Tuple[int, int]]] = None,
    always_reencode: bool = False,
):
    """
    将输入数据集中某 episode 的视频提取/拷贝到输出数据集的新位置。
    目前仅支持 per-episode 视频文件；sharded v3 视频暂不支持按 episode 裁剪。
    """
    discovered = discover_video_files(input_root, load_info(input_root))
    cam_map = discovered.get(camera_key, {})
    old_path = cam_map.get(old_ep_idx)

    out_dir = output_root / "videos" / camera_key / f"chunk-{new_ep_idx:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "file-000.mp4"

    if old_path is None or not old_path.exists():
        return False

    source_fps = get_video_fps(old_path)
    if source_fps <= 0:
        source_fps = fps

    if keep_ranges is None:
        keep_ranges = [(0, get_video_frame_count(old_path))]

    # 如果源视频 FPS 与目标不一致，必须重新编码/裁剪，不能直接复制
    if (
        not always_reencode
        and abs(source_fps - fps) < 1e-6
        and keep_ranges == [(0, get_video_frame_count(old_path))]
    ):
        shutil.copy2(old_path, out_path)
        return True

    extract_video_keep_ranges(old_path, out_path, keep_ranges, source_fps, fps)
    return True


# =============================================================================
# Statistics
# =============================================================================

def sample_video_frames(video_path: Path, max_samples: int = 50) -> Optional[np.ndarray]:
    """从视频中均匀采样若干帧，返回 (N, C, H, W) uint8 数组。

    使用 ffmpeg 的 select filter 直接抽取指定帧，避免 PyAV 逐帧解码整个视频。
    """
    if not video_path.exists():
        return None
    info = get_video_info(video_path)
    if info is None:
        return None
    total, width, height = info
    if total == 0 or width == 0 or height == 0:
        return None

    idxs = [int(i) for i in np.linspace(0, total - 1, min(max_samples, total))]
    if not idxs:
        return None

    # 构建 select 表达式，例如 select='eq(n\,0)+eq(n\,100)+...'
    parts = [f"eq(n\\,{i})" for i in idxs]
    select_expr = "+".join(parts)

    cmd = [
        "-i", str(video_path),
        "-vf", f"select='{select_expr}'",
        "-vsync", "0",
        "-f", "image2pipe",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]

    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + cmd,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"\nFFmpeg error in sample_video_frames:\n{e.stderr.decode('utf-8', errors='ignore')}")
        return None

    raw = proc.stdout
    frame_size = height * width * 3
    n_frames = len(raw) // frame_size
    if n_frames == 0:
        return None

    arr = np.frombuffer(raw, dtype=np.uint8)[: n_frames * frame_size]
    arr = arr.reshape(n_frames, height, width, 3)
    arr = np.transpose(arr, (0, 3, 1, 2))  # (N, C, H, W)
    return arr


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    """与 LeRobot 官方一致的图像降采样策略。"""
    _, height, width = img.shape
    if max(width, height) < max_size_threshold:
        return img
    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def compute_video_feature_stats(root: Path, feature_key: str, info: dict,
                                skip_video_stats: bool = False) -> Optional[dict]:
    """计算某个 video feature 的 per-channel 统计量（归一化到 [0,1]）。
    
    使用与 LeRobot 官方一致的策略：
    - 根据数据集大小自动估计采样数量
    - 图像降采样到最大 150px
    - 使用 RunningQuantileStats 计算分位数
    """
    if skip_video_stats:
        print(f"  Skipping video stats for {feature_key}")
        return None
    
    total_eps = info["total_episodes"]
    video_paths = []
    for ep_idx in range(total_eps):
        video_path = root / "videos" / feature_key / f"chunk-{ep_idx:03d}" / "file-000.mp4"
        if video_path.exists():
            video_paths.append(video_path)
    
    if not video_paths:
        return None
    
    # 使用官方策略估计采样数量
    total_frames = sum(get_video_frame_count(vp) for vp in video_paths)
    num_samples = estimate_num_samples(total_frames, min_num_samples=100, max_num_samples=10000)
    samples_per_video = max(1, num_samples // len(video_paths))
    
    print(f"  Video stats [{feature_key}]: {len(video_paths)} videos, ~{samples_per_video} frames/video")
    
    sampled = []
    for video_path in tqdm(video_paths, desc=f"Video stats [{feature_key}]", leave=False):
        frames = sample_video_frames(video_path, samples_per_video)
        if frames is not None:
            # 应用图像降采样
            frames = np.stack([auto_downsample_height_width(f) for f in frames])
            sampled.append(frames)
    
    if not sampled:
        return None
    
    arr = np.concatenate(sampled, axis=0).astype(np.float32) / 255.0
    
    # 使用 RunningQuantileStats 计算统计量
    reshaped = arr.transpose(0, 2, 3, 1).reshape(-1, arr.shape[1])
    running_stats = RunningQuantileStats()
    running_stats.update(reshaped)
    stats = running_stats.get_statistics()
    
    # 转换为 per-channel 格式 (3, 1, 1)
    per_channel = {}
    for key, value in stats.items():
        if key == "count":
            per_channel[key] = value
        else:
            per_channel[key] = value.reshape(-1, 1, 1)
    
    return per_channel


def recompute_stats(root: Path, info: dict, skip_video_stats: bool = False) -> dict:
    """使用 LeRobot 的 RunningQuantileStats 重新计算 stats.json，并补充 video feature 统计。
    
    Args:
        root: 数据集根目录
        info: info.json 内容
        skip_video_stats: 是否跳过视频特征统计
    """
    features = info.get("features", {})
    numeric_dtypes = {"float32", "float64", "int32", "int64"}

    accumulators: Dict[str, RunningQuantileStats] = {}
    shapes: Dict[str, Tuple[int, ...]] = {}

    data_files = sorted((root / "data").glob("**/*.parquet"))
    for f in tqdm(data_files, desc="Computing stats"):
        table = pq.read_table(f).to_pandas()
        for key, ft in features.items():
            if ft.get("dtype") not in numeric_dtypes:
                continue
            if key not in table.columns:
                continue
            values = table[key].values
            if len(values) == 0:
                continue
            shape = tuple(ft.get("shape", (1,)))
            shapes[key] = shape
            if len(shape) == 1 and shape[0] == 1:
                arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
            else:
                arr = np.stack([np.asarray(v, dtype=np.float64) for v in values])
            if key not in accumulators:
                accumulators[key] = RunningQuantileStats()
            accumulators[key].update(arr)

    stats = {}
    for key, rs in accumulators.items():
        try:
            s = rs.get_statistics()
        except ValueError:
            # 样本不足 2 个时回退
            continue
        stats[key] = {k: np.asarray(v) for k, v in s.items()}

    # 为 video feature 计算 per-channel 统计
    if not skip_video_stats:
        for key, ft in features.items():
            if ft.get("dtype") == "video":
                video_stats = compute_video_feature_stats(root, key, info)
                if video_stats is not None:
                    stats[key] = video_stats
    else:
        print("  Skipping video feature stats calculation")

    return stats


# =============================================================================
# Core clean operation
# =============================================================================

def clean_dataset(
    input_path: Path,
    output_path: Path,
    delete_episodes: Optional[List[int]] = None,
    remove_frozen: bool = False,
    frozen_threshold: float = 0.001,
    frozen_min_frames: int = 3,
    downsample_fps: Optional[float] = None,
    drop_video_features_if_missing: bool = False,
    min_episode_length: int = 0,
    skip_video_stats: bool = False,
) -> Path:
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not (input_path / "meta" / "info.json").exists():
        raise FileNotFoundError(f"No info.json found in {input_path}")

    info = load_info(input_path)
    source_fps = float(info.get("fps", 30.0))
    target_fps = downsample_fps if downsample_fps is not None else source_fps
    if target_fps <= 0:
        raise ValueError(f"Invalid target fps: {target_fps}")

    ratio = int(round(source_fps / target_fps)) if target_fps != source_fps else 1

    print(f"\nLoading episodes from {input_path}")
    episodes = read_all_episodes(input_path)
    print(f"  Found {len(episodes)} episodes, source fps={source_fps}, target fps={target_fps}")

    # 删除指定 episode
    to_delete = set(delete_episodes or [])
    if to_delete:
        print(f"  Deleting episodes: {sorted(to_delete)}")
    kept_old_eps = [ep for ep in sorted(episodes.keys()) if ep not in to_delete]

    if not kept_old_eps:
        raise ValueError("No episodes left after deletion.")

    # 删除比例警告
    total_eps = len(episodes)
    deleted_count = len(to_delete)
    if deleted_count > 0:
        delete_ratio = deleted_count / total_eps
        print(f"  Deletion ratio: {deleted_count}/{total_eps} ({delete_ratio*100:.1f}%)")
        if delete_ratio > 0.3 and skip_video_stats:
            print("  ⚠️  WARNING: Deleting more than 30% of episodes with --skip-video-stats!")
            print("     Video statistics may be significantly outdated. Consider running without --skip-video-stats.")

    # 视频处理
    videos_present = has_any_video(input_path, info)
    camera_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    if camera_keys and not videos_present:
        print(f"  WARNING: features declare {len(camera_keys)} video keys but no video files were found.")
        if not drop_video_features_if_missing:
            print("  Video features will be dropped so the dataset remains loadable. "
                  "Use --keep-video-features-if-missing to retain them (dataset will fail to load).")
        for key in camera_keys:
            info["features"].pop(key)
        camera_keys = []
        info["video_path"] = None

    # 准备输出目录
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 复制 tasks
    tasks = load_tasks(input_path)
    save_tasks(output_path, tasks)

    # 确定数据 schema
    sample_schema = pq.read_schema(next(iter((input_path / "data").glob("**/*.parquet"))))
    target_data_schema = build_target_data_schema(sample_schema)

    ep_schema = build_episodes_schema(info, has_videos=bool(camera_keys and videos_present))

    new_info = dict(info)
    new_info["codebase_version"] = "v3.0"
    new_info["fps"] = target_fps
    new_info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    if camera_keys and videos_present:
        new_info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    else:
        new_info["video_path"] = None
    new_info["total_episodes"] = len(kept_old_eps)
    new_info["total_frames"] = 0
    new_info["splits"] = {"train": f"0:{len(kept_old_eps)}"}

    episode_rows = []
    global_frame_idx = 0
    total_removed_frozen = 0
    skipped_short = 0
    new_ep_idx = 0

    for old_ep_idx in tqdm(kept_old_eps, desc="Cleaning episodes"):
        df = episodes[old_ep_idx].copy()
        original_len = len(df)

        # 降采样（帧跳过）
        if ratio > 1:
            df = df.iloc[::ratio].reset_index(drop=True)

        # 删除 frozen 帧
        keep = None
        if remove_frozen and "action" in df.columns and len(df) >= 2:
            actions = np.stack([np.asarray(a, dtype=np.float64) for a in df["action"].values])
            frozen = find_frozen_frames(actions, frozen_threshold, frozen_min_frames)
            if frozen:
                total_removed_frozen += len(frozen)
                keep_mask = ~df.index.isin(frozen)
                df = df[keep_mask].reset_index(drop=True)
                keep = keep_ranges(original_len if ratio == 1 else len(df) + len(frozen), frozen)

        length = len(df)
        if length < min_episode_length:
            skipped_short += 1
            continue

        # 重新编号
        df["episode_index"] = new_ep_idx
        df["frame_index"] = np.arange(length, dtype=np.int64)
        df["timestamp"] = (np.arange(length, dtype=np.float64) / target_fps).astype(np.float32)
        df["index"] = np.arange(global_frame_idx, global_frame_idx + length, dtype=np.int64)

        # 写数据 parquet（chunk/file 映射仅保存在 episodes 元数据中）
        data_path = output_path / "data" / f"chunk-{new_ep_idx:03d}" / "file-000.parquet"
        write_data_parquet(df, target_data_schema, data_path)

        # 处理视频（per-episode 模式）
        for cam in camera_keys:
            if videos_present:
                extract_or_copy_episode_video(
                    input_path,
                    output_path,
                    cam,
                    old_ep_idx,
                    new_ep_idx,
                    target_fps,
                    keep_ranges=keep,
                    always_reencode=remove_frozen,
                )

        # episodes 元数据行
        row = {
            "episode_index": new_ep_idx,
            "tasks": [tasks.index[task_idx] for task_idx in df["task_index"].unique()]
            if "task_index" in df.columns and len(tasks) > 0
            else [],
            "length": length,
            "data/chunk_index": new_ep_idx,
            "data/file_index": 0,
            "dataset_from_index": global_frame_idx,
            "dataset_to_index": global_frame_idx + length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        if camera_keys and videos_present:
            for cam in camera_keys:
                row[f"videos/{cam}/chunk_index"] = new_ep_idx
                row[f"videos/{cam}/file_index"] = 0
                row[f"videos/{cam}/from_timestamp"] = 0.0
                row[f"videos/{cam}/to_timestamp"] = length / target_fps
        episode_rows.append(row)

        global_frame_idx += length
        new_ep_idx += 1

    # 写 episodes.parquet
    ep_path = output_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    write_episodes_parquet(episode_rows, ep_schema, ep_path)

    new_info["total_episodes"] = len(episode_rows)
    new_info["total_frames"] = global_frame_idx
    new_info["splits"] = {"train": f"0:{len(episode_rows)}"}

    # 若重新编码过视频，统一标注为 H.264，避免 info.json 与实际视频 codec 不一致
    if remove_frozen and camera_keys and videos_present:
        for cam in camera_keys:
            video_info = new_info["features"][cam].setdefault("info", {})
            video_info.update({
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.crf": 18,
                "video.preset": "fast",
                "video.g": 2,
            })

    save_info(output_path, new_info)

    # 重新计算 stats
    print("\nRecomputing statistics...")
    stats = recompute_stats(output_path, new_info, skip_video_stats=skip_video_stats)
    save_json(output_path / "meta" / "stats.json", serialize_dict(stats))

    print(f"\nDone: {output_path}")
    print(f"  Kept episodes: {len(episode_rows)} | Frames: {global_frame_idx}")
    if skipped_short:
        print(f"  Skipped short episodes (<{min_episode_length} frames): {skipped_short}")
    if remove_frozen:
        print(f"  Frozen frames removed: {total_removed_frozen}")
    return output_path


# =============================================================================
# Robust merge (handles malformed / hybrid v2.1-v3.0 datasets)
# =============================================================================

def resolve_data_file(root: Path, chunk_index: int, file_index: int) -> Optional[Path]:
    """根据 metadata 中的 chunk/file index 找到真实的数据 parquet 文件。

    兼容标准 v3 命名（file-000.parquet）和部分旧版/损坏命名（episode_000000.parquet）。
    """
    chunk_dir = root / "data" / f"chunk-{chunk_index:03d}"
    if not chunk_dir.exists():
        return None

    # 标准 v3
    standard = chunk_dir / f"file-{file_index:03d}.parquet"
    if standard.exists():
        return standard

    candidates = list(chunk_dir.glob("*.parquet"))
    if not candidates:
        return None

    # 若目录里只有一个 parquet，直接返回
    if len(candidates) == 1:
        return candidates[0]

    # 尝试从文件名 episode_XXXXXX 解析 episode index 进行匹配
    # 适用于 backup_15fps 的 episode_{chunk}00000 命名
    for c in candidates:
        if "episode_" in c.name:
            try:
                ep_num = int(c.stem.split("episode_")[-1])
                # chunk_index 对应 episode_num // 100000 的启发式
                if ep_num // 100000 == chunk_index:
                    return c
            except Exception:
                continue
    return candidates[0]


def resolve_video_file(root: Path, camera_key: str, chunk_index: int, file_index: int) -> Optional[Path]:
    f = root / "videos" / camera_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
    return f if f.exists() else None


def load_source_episodes(root: Path) -> Tuple[dict, pd.DataFrame, List[dict]]:
    """加载源数据集的 info、tasks 和 episodes 元数据行。"""
    info = load_info(root)
    tasks = load_tasks(root)

    # 读取 v3 episodes parquet
    ep_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if ep_files:
        df = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
    elif (root / "meta" / "episodes.parquet").exists():
        df = pd.read_parquet(root / "meta" / "episodes.parquet")
    else:
        df = pd.DataFrame()

    rows = df.to_dict("records") if not df.empty else []
    return info, tasks, rows


def extract_video_segment_copy(src: Path, dst: Path, from_ts: float, to_ts: float, target_fps: float):
    """从源视频按时间戳裁剪一段，并统一输出为 target_fps 的 H.264。

    使用重编码而非 stream copy，因此能正确处理源视频 FPS 与目标不一致、
    或源文件包含多个 episode 拼接的情况。
    """
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, to_ts - from_ts)
    if duration <= 0:
        return False

    cmd = [
        "-ss", str(from_ts),
        "-t", str(duration),
        "-i", str(src),
        "-vf", f"fps={target_fps},setpts=N/FRAME_RATE/TB",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(target_fps),
        "-an",
        str(dst),
    ]
    try:
        run_ffmpeg(cmd)
        return dst.exists() and dst.stat().st_size > 0
    except subprocess.CalledProcessError:
        return False


def merge_datasets_command(input_paths: List[Path], output_path: Path, target_fps: Optional[float] = None,
                           skip_video_stats: bool = False):
    """合并多个 LeRobot 数据集，输出为规范 v3.0 per-episode 格式。

    不依赖 LeRobotDataset/aggregate_datasets，因此能处理：
      - 数据文件命名不规范（episode_xxx.parquet）
      - video.fps 元数据不一致
      - 多 episode 共享一个 data/video 文件（会根据 metadata 的 timestamp 裁剪视频）
      
    对于单数据集格式转换，直接拷贝原 stats.json 以保持准确性。
    """
    input_paths = [Path(p).expanduser().resolve() for p in input_paths]
    output_path = Path(output_path).expanduser().resolve()

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 先统一 FPS（必要时降采样整个数据集）
    sources: List[Path] = []
    temp_dirs: List[Path] = []
    for p in input_paths:
        info = load_info(p)
        fps = float(info.get("fps", 30.0))
        if target_fps is not None and abs(fps - target_fps) > 1e-6:
            print(f"Downsampling {p.name} from {fps}fps to {target_fps}fps...")
            tmp_out = Path(tempfile.mkdtemp(prefix=f"cleaned_{p.name}_"))
            clean_dataset(
                p,
                tmp_out,
                downsample_fps=target_fps,
            )
            sources.append(tmp_out)
            temp_dirs.append(tmp_out)
        else:
            sources.append(p)

    # 用第一个源的特征定义作为基准
    base_info = load_info(sources[0])
    base_fps = float(base_info.get("fps", 15.0))
    if target_fps is not None:
        base_fps = target_fps

    camera_keys = [k for k, v in base_info.get("features", {}).items() if v.get("dtype") == "video"]

    # 合并 tasks
    all_tasks = pd.concat([load_tasks(p) for p in sources], ignore_index=False)
    unique_tasks = all_tasks.index.unique()
    merged_tasks = pd.DataFrame(
        {"task_index": range(len(unique_tasks))},
        index=pd.Index(unique_tasks, name="task"),
    )
    save_tasks(output_path, merged_tasks)

    # 准备 info
    new_info = dict(base_info)
    new_info["codebase_version"] = "v3.0"
    new_info["fps"] = base_fps
    new_info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    new_info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4" if camera_keys else None
    new_info["total_episodes"] = 0
    new_info["total_frames"] = 0
    new_info["splits"] = {"train": "0:0"}

    # 确定数据 schema（从第一个非空源读取）
    sample_file = None
    for p in sources:
        f = next(iter((p / "data").glob("**/*.parquet")), None)
        if f:
            sample_file = f
            break
    if sample_file is None:
        raise FileNotFoundError("No data parquet files found in source datasets.")
    target_data_schema = build_target_data_schema(pq.read_schema(sample_file))
    ep_schema = build_episodes_schema(new_info, has_videos=bool(camera_keys))

    episode_rows = []
    global_frame_idx = 0
    new_ep_idx = 0

    for src_root in sources:
        info, tasks_src, ep_rows = load_source_episodes(src_root)
        src_fps = float(info.get("fps", base_fps))
        fps_ratio = src_fps / base_fps
        print(f"\nMerging from {src_root.name}: {len(ep_rows)} episodes")

        # task 映射：源 task_index -> 合并后 task_index
        task_map = {}
        for src_tidx in tasks_src["task_index"]:
            task_str = tasks_src.index[src_tidx]
            dst_tidx = merged_tasks[merged_tasks.index == task_str]["task_index"].values[0]
            task_map[src_tidx] = dst_tidx

        for ep_meta in tqdm(ep_rows, desc=f"Copy episodes from {src_root.name}"):
            old_ep_idx = int(ep_meta["episode_index"])

            # 加载数据
            chunk_idx = ep_meta["data/chunk_index"]
            file_idx = ep_meta["data/file_index"]
            data_file = resolve_data_file(src_root, chunk_idx, file_idx)
            if data_file is None:
                print(f"  WARNING: data file not found for episode {old_ep_idx} (chunk={chunk_idx}, file={file_idx})")
                continue

            df = pq.read_table(data_file).to_pandas()
            df = df[df["episode_index"] == old_ep_idx].copy()
            if len(df) == 0:
                print(f"  WARNING: no frames for episode {old_ep_idx} in {data_file}")
                continue

            # 若 fps 不同，按时间对齐降采样（这里假设已经预处理过，仅做保险）
            if abs(fps_ratio - 1.0) > 1e-6:
                # 按目标 fps 选取最近帧
                df = df.iloc[::int(round(fps_ratio))].reset_index(drop=True)

            length = len(df)
            df["episode_index"] = new_ep_idx
            df["frame_index"] = np.arange(length, dtype=np.int64)
            df["timestamp"] = (np.arange(length, dtype=np.float64) / base_fps).astype(np.float32)
            df["index"] = np.arange(global_frame_idx, global_frame_idx + length, dtype=np.int64)
            if "task_index" in df.columns:
                df["task_index"] = df["task_index"].map(task_map).fillna(0).astype(np.int64)

            # 写数据
            data_path = output_path / "data" / f"chunk-{new_ep_idx:03d}" / "file-000.parquet"
            write_data_parquet(df, target_data_schema, data_path)

            # 处理视频
            row = {
                "episode_index": new_ep_idx,
                "tasks": [],
                "length": length,
                "data/chunk_index": new_ep_idx,
                "data/file_index": 0,
                "dataset_from_index": global_frame_idx,
                "dataset_to_index": global_frame_idx + length,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            if "tasks" in ep_meta and ep_meta["tasks"]:
                if isinstance(ep_meta["tasks"], (list, tuple)):
                    row["tasks"] = [str(t) for t in ep_meta["tasks"]]
                else:
                    row["tasks"] = [str(ep_meta["tasks"])]
            elif len(tasks_src) > 0 and "task_index" in df.columns:
                row["tasks"] = [str(tasks_src.index[int(t)]) for t in df["task_index"].unique()]

            for cam in camera_keys:
                chunk_idx = int(ep_meta.get(f"videos/{cam}/chunk_index", 0))
                file_idx = int(ep_meta.get(f"videos/{cam}/file_index", 0))
                from_ts = float(ep_meta.get(f"videos/{cam}/from_timestamp", 0.0))
                to_ts = float(ep_meta.get(f"videos/{cam}/to_timestamp", from_ts + length / src_fps))

                src_video = resolve_video_file(src_root, cam, chunk_idx, file_idx)
                dst_dir = output_path / "videos" / cam / f"chunk-{new_ep_idx:03d}"
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_video = dst_dir / "file-000.mp4"

                if src_video is not None:
                    extract_video_segment_copy(src_video, dst_video, from_ts, to_ts, base_fps)
                else:
                    print(f"  WARNING: video not found for episode {old_ep_idx}, camera {cam}")

                row[f"videos/{cam}/chunk_index"] = new_ep_idx
                row[f"videos/{cam}/file_index"] = 0
                row[f"videos/{cam}/from_timestamp"] = 0.0
                row[f"videos/{cam}/to_timestamp"] = length / base_fps

            episode_rows.append(row)
            global_frame_idx += length
            new_ep_idx += 1

    # 写 episodes 元数据
    ep_path = output_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    write_episodes_parquet(episode_rows, ep_schema, ep_path)

    new_info["total_episodes"] = new_ep_idx
    new_info["total_frames"] = global_frame_idx
    new_info["splits"] = {"train": f"0:{new_ep_idx}"}

    # merge 中视频被重编码为 H.264，统一更新 feature info
    if camera_keys:
        for cam in camera_keys:
            video_info = new_info["features"][cam].setdefault("info", {})
            video_info.update({
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.crf": 18,
                "video.preset": "fast",
                "video.g": 2,
            })

    save_info(output_path, new_info)

    # 统计计算：单数据集转换时直接拷贝原 stats.json，多数据集合并时重新计算
    print("\nHandling statistics...")
    if len(input_paths) == 1 and not target_fps:
        # 单数据集格式转换，直接拷贝原 stats.json
        src_stats_path = input_paths[0] / "meta" / "stats.json"
        if src_stats_path.exists():
            print("  Copying original stats.json (single dataset format conversion)")
            shutil.copy2(src_stats_path, output_path / "meta" / "stats.json")
        else:
            print("  Recomputing stats (original not found)")
            stats = recompute_stats(output_path, new_info, skip_video_stats=skip_video_stats)
            save_json(output_path / "meta" / "stats.json", serialize_dict(stats))
    else:
        # 多数据集合并或 FPS 转换，重新计算 stats
        print("  Recomputing stats (multi-dataset merge or FPS conversion)")
        stats = recompute_stats(output_path, new_info, skip_video_stats=skip_video_stats)
        save_json(output_path / "meta" / "stats.json", serialize_dict(stats))

    print(f"\nMerge complete: {output_path}")
    print(f"  Episodes: {new_ep_idx} | Frames: {global_frame_idx}")

    # 清理临时目录
    for d in temp_dirs:
        shutil.rmtree(d, ignore_errors=True)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LeRobot Dataset Cleaner & Merger (v3.0-aware)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 删除 episode 9、11，删除 frozen 帧，并丢弃清洗后短于 30 帧的 episode
  python lerobot_dataset_cleaner.py clean ~/lerobot/local_merged_dataset_v4 \\
      --delete-episodes 9 11 --remove-frozen --min-episode-length 30 \\
      --output ~/lerobot/local_merged_dataset_v5

  # 仅降采样到 15fps
  python lerobot_dataset_cleaner.py clean ~/lerobot/dataset30fps --downsample-fps 15 --output ~/lerobot/dataset15fps

  # 合并 3 个数据集（自动对齐到 15fps）
  python lerobot_dataset_cleaner.py merge ~/lerobot/ds1 ~/lerobot/ds2 ~/lerobot/ds3 \\
      --target-fps 15 --output ~/lerobot/merged
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    clean_p = subparsers.add_parser("clean", help="清洗数据集")
    clean_p.add_argument("dataset", type=str, help="输入数据集路径")
    clean_p.add_argument("--delete-episodes", type=int, nargs="+", default=[],
                         help="要删除的 episode 索引")
    clean_p.add_argument("--remove-frozen", action="store_true",
                         help="删除 frozen 帧")
    clean_p.add_argument("--frozen-threshold", type=float, default=0.001)
    clean_p.add_argument("--frozen-min-frames", type=int, default=3)
    clean_p.add_argument("--downsample-fps", type=float, default=None)
    clean_p.add_argument("--keep-video-features-if-missing", action="store_true",
                         help="即使视频文件缺失也保留 video feature（会导致 LeRobotDataset 加载失败，仅在后续补视频时使用）")
    clean_p.add_argument("--min-episode-length", type=int, default=0,
                         help="清洗后长度小于该值的 episode 会被丢弃（默认 0，不丢弃）")
    clean_p.add_argument("--skip-video-stats", action="store_true",
                         help="跳过视频特征统计计算（大幅加速，约 10x）")
    clean_p.add_argument("--output", type=str, required=True)

    merge_p = subparsers.add_parser("merge", help="合并数据集")
    merge_p.add_argument("datasets", type=str, nargs="+", help="输入数据集路径")
    merge_p.add_argument("--target-fps", type=float, default=None)
    merge_p.add_argument("--skip-video-stats", action="store_true",
                         help="跳过视频特征统计计算（仅在多数据集合并或 FPS 转换时生效）")
    merge_p.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    if args.command == "clean":
        clean_dataset(
            input_path=Path(args.dataset),
            output_path=Path(args.output),
            delete_episodes=args.delete_episodes,
            remove_frozen=args.remove_frozen,
            frozen_threshold=args.frozen_threshold,
            frozen_min_frames=args.frozen_min_frames,
            downsample_fps=args.downsample_fps,
            drop_video_features_if_missing=not args.keep_video_features_if_missing,
            min_episode_length=args.min_episode_length,
            skip_video_stats=args.skip_video_stats,
        )
    elif args.command == "merge":
        merge_datasets_command(
            input_paths=[Path(p) for p in args.datasets],
            output_path=Path(args.output),
            target_fps=args.target_fps,
            skip_video_stats=args.skip_video_stats,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
