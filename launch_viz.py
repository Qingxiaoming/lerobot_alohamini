#!/usr/bin/env python3
"""
LeRobot Dataset Viz Launcher
============================
命令行交互式选择器：列出本地数据集和 episodes，调用官方 lerobot-dataset-viz
打开 Rerun 可视化。

用法：
    python launch_viz.py
    python launch_viz.py --dataset-dir ~/.cache/huggingface/lerobot
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def get_dataset_dir() -> Path:
    """获取数据集根目录。"""
    if len(sys.argv) > 1 and sys.argv[1] in ("-d", "--dataset-dir"):
        return Path(sys.argv[2]).expanduser().resolve()
    return Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def find_datasets(base_dir: Path) -> list[Path]:
    """查找所有 LeRobot 数据集目录。"""
    datasets = []
    if not base_dir.exists():
        return datasets
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and (item / "meta" / "info.json").exists():
            datasets.append(item)
    return datasets


def get_num_episodes(dataset_path: Path) -> int:
    """读取所有元数据分片，返回数据集的 episode 数量。"""
    return len(list_episodes(dataset_path))


def list_episodes(dataset_path: Path) -> list[int]:
    """返回所有实际存在的 episode 索引，支持多个 chunk/file 分片。"""
    episodes_dir = dataset_path / "meta" / "episodes"
    episode_indices: set[int] = set()

    for episodes_path in sorted(episodes_dir.glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(episodes_path, columns=["episode_index"])
        episode_indices.update(int(index) for index in df["episode_index"].tolist())

    if episode_indices:
        return sorted(episode_indices)

    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    return list(range(info.get("total_episodes", 0)))


def launch_viz(dataset_path: Path, episode_index: int):
    """调用官方 lerobot-dataset-viz。"""
    cmd = [
        "lerobot-dataset-viz",
        "--repo-id", dataset_path.name,
        "--root", str(dataset_path),
        "--episode-index", str(episode_index),
    ]
    print(f"\n🎬 启动可视化: episode {episode_index}")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("\n❌ 错误：找不到 'lerobot-dataset-viz' 命令。")
        print("请确保已安装依赖：pip install 'lerobot[dataset_viz]'")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 可视化命令退出码: {e.returncode}")


def input_number(prompt: str, max_val: int | None = None) -> int | None:
    """读取用户输入的数字，支持 q 退出。"""
    while True:
        try:
            value = input(prompt).strip().lower()
            if value in ("q", "quit", "exit"):
                return None
            num = int(value)
            if max_val is not None and not (0 <= num < max_val):
                print(f"请输入 0 到 {max_val - 1} 之间的数字")
                continue
            return num
        except ValueError:
            print("请输入数字，或输入 q 退出")


def input_episode_index(prompt: str, episode_indices: list[int]) -> int | None:
    """读取 episode 编号，并确保该编号在数据集中真实存在。"""
    valid_indices = set(episode_indices)
    while True:
        value = input(prompt).strip().lower()
        if value in ("q", "quit", "exit"):
            return None
        try:
            episode_index = int(value)
        except ValueError:
            print("请输入 episode 编号，或输入 q 退出")
            continue
        if episode_index not in valid_indices:
            print("该 episode 不存在，请输入上面范围内的有效编号")
            continue
        return episode_index


def main():
    dataset_dir = get_dataset_dir()
    print(f"扫描数据集目录: {dataset_dir}")

    datasets = find_datasets(dataset_dir)
    if not datasets:
        print(f"未在 {dataset_dir} 找到 LeRobot 数据集")
        return

    print(f"\n找到 {len(datasets)} 个数据集:\n")
    for i, ds in enumerate(datasets):
        num_eps = get_num_episodes(ds)
        print(f"  [{i}] {ds.name:40s} ({num_eps} episodes)")

    ds_idx = input_number("\n选择数据集编号 (q 退出): ", len(datasets))
    if ds_idx is None:
        return

    selected_dataset = datasets[ds_idx]
    episode_indices = list_episodes(selected_dataset)
    num_episodes = len(episode_indices)
    print(f"\n已选择: {selected_dataset.name}")
    if not episode_indices:
        print("该数据集还没有可查看的 episode")
        return
    print(f"共有 {num_episodes} 个 episodes ({episode_indices[0]} ~ {episode_indices[-1]})")

    while True:
        ep_idx = input_episode_index(
            f"\n输入要查看的 episode 编号 ({episode_indices[0]}-{episode_indices[-1]}, q 退出): ",
            episode_indices,
        )
        if ep_idx is None:
            break
        launch_viz(selected_dataset, ep_idx)

    print("\n再见!")


if __name__ == "__main__":
    main()
