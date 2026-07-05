import pandas as pd
from pathlib import Path

# 数据集路径
dataset_path = Path("/home/yan/.cache/huggingface/lerobot/local_dataset_f1")

# 读取 episodes 元数据
episodes_df = pd.read_parquet(dataset_path / "meta/episodes/chunk-000/file-000.parquet")

print("=" * 80)
print(f"数据集: {dataset_path.name}")
print(f"总 Episodes: {len(episodes_df)}")
print(f"总帧数: {episodes_df['length'].sum()}")
print("=" * 80)

# 查看每个摄像头的视频文件信息
video_keys = ["observation.images.chest", "observation.images.forward", 
              "observation.images.wrist_left", "observation.images.wrist_right"]

print("\n各摄像头视频文件分布:")
print("-" * 80)

for video_key in video_keys:
    cam_name = video_key.split('.')[-1]
    
    # 统计该摄像头实际存在的视频文件数量
    video_dir = dataset_path / "videos" / video_key / "chunk-000"
    if video_dir.exists():
        mp4_files = sorted(video_dir.glob("*.mp4"))
        print(f"\n{cam_name}: {len(mp4_files)} 个视频文件")
        for f in mp4_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")
    else:
        print(f"\n{cam_name}: 目录不存在")

    # 查看每个 episode 是否有该摄像头的数据
    chunk_col = f"videos/{video_key}/chunk_index"
    if chunk_col in episodes_df.columns:
        # 统计有多少个 episode 有该摄像头的数据
        valid_episodes = episodes_df[chunk_col].notna().sum()
        print(f"  有视频数据的 episodes: {valid_episodes}/{len(episodes_df)}")
        
        # 查看视频文件到 episode 的映射
        file_episodes = {}
        for idx, row in episodes_df.iterrows():
            if pd.notna(row[chunk_col]):
                chunk_idx = int(row[chunk_col])
                file_idx = int(row[f"videos/{video_key}/file_index"])
                key = f"chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
                if key not in file_episodes:
                    file_episodes[key] = []
                file_episodes[key].append(idx)
        
        print(f"  视频文件 -> episodes 映射:")
        for file_path, eps in sorted(file_episodes.items()):
            print(f"    {file_path}: episodes {min(eps)}-{max(eps)} ({len(eps)}个)")
    else:
        print(f"  没有视频元数据")

# 检查是否有缺失摄像头数据的 episodes
print("\n" + "=" * 80)
print("检查每个 Episode 的摄像头覆盖情况:")
print("=" * 80)

missing_camera_episodes = []
for idx, row in episodes_df.iterrows():
    missing_cameras = []
    for video_key in video_keys:
        chunk_col = f"videos/{video_key}/chunk_index"
        if chunk_col in episodes_df.columns and pd.isna(row[chunk_col]):
            missing_cameras.append(video_key.split('.')[-1])
    
    if missing_cameras:
        missing_camera_episodes.append((idx, missing_cameras))
        print(f"Episode {idx}: 缺少 {', '.join(missing_cameras)} 摄像头数据")

if not missing_camera_episodes:
    print("所有 Episodes 都有完整的摄像头数据")
else:
    print(f"\n共 {len(missing_camera_episodes)} 个 episodes 缺少摄像头数据")