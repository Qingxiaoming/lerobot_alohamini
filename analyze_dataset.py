#!/usr/bin/env python
"""分析数据集动作分布，帮助诊断机械臂右偏问题"""

import numpy as np

from lerobot.datasets import LeRobotDataset


def analyze_action_distribution():
    # 加载数据集
    dataset = LeRobotDataset(
        repo_id="local_merged_dataset_v8",
        root="/home/yan/.cache/huggingface/lerobot/local_merged_dataset_v8",
        video_backend="torchcodec",
    )
    
    print(f"数据集总帧数: {len(dataset)}")
    print(f"动作特征名称: {dataset.meta.info['features']['action']['names']}")
    
    # 收集右臂动作数据
    right_arm_actions = []
    
    for i in range(min(1000, len(dataset))):
        sample = dataset[i]
        action = sample["action"]
        right_arm_actions.append(action[7:14])  # 右臂7个关节
    
    right_arm_actions = np.array(right_arm_actions)
    
    # 动作名称索引对应
    action_names = dataset.meta.info['features']['action']['names']
    right_arm_names = action_names[7:14]
    
    print("\n" + "="*60)
    print("右臂各关节动作统计:")
    print("="*60)
    for i, name in enumerate(right_arm_names):
        mean_val = right_arm_actions[:, i].mean()
        std_val = right_arm_actions[:, i].std()
        min_val = right_arm_actions[:, i].min()
        max_val = right_arm_actions[:, i].max()
        median_val = np.median(right_arm_actions[:, i])
        skewness = (mean_val - median_val) / std_val if std_val > 0 else 0
        
        print(f"\n{name}:")
        print(f"  均值:    {mean_val:>10.2f}")
        print(f"  中位数:  {median_val:>10.2f}")
        print(f"  标准差:  {std_val:>10.2f}")
        print(f"  最小值:  {min_val:>10.2f}")
        print(f"  最大值:  {max_val:>10.2f}")
        print(f"  偏度(mean-median/std): {skewness:>10.4f}")
        
        # 检查分布对称性
        range_total = max_val - min_val
        range_from_mean_to_min = mean_val - min_val
        range_from_mean_to_max = max_val - mean_val
        
        if range_from_mean_to_max > range_from_mean_to_min * 1.5:
            print(f"  ⚠️  分布偏右（右侧范围更大）")
        elif range_from_mean_to_min > range_from_mean_to_max * 1.5:
            print(f"  ⚠️  分布偏左（左侧范围更大）")
        else:
            print(f"  ✓ 分布相对对称")
    
    # 分析训练数据中纸巾位置的分布
    # 通过查看wrist_yaw和wrist_roll来判断抓取方向
    print("\n" + "="*60)
    print("抓取方向分析:")
    print("="*60)
    
    shoulder_pan = right_arm_actions[:, 0]  # shoulder_pan - 影响左右位置
    wrist_yaw = right_arm_actions[:, 4]  # wrist_yaw - 影响末端朝向
    wrist_roll = right_arm_actions[:, 5]  # wrist_roll
    gripper = right_arm_actions[:, 6]  # gripper
    
    print(f"肩pan均值:      {shoulder_pan.mean():.2f}, 标准差: {shoulder_pan.std():.2f}")
    print(f"腕部yaw均值:    {wrist_yaw.mean():.2f}, 标准差: {wrist_yaw.std():.2f}")
    print(f"腕部roll均值:   {wrist_roll.mean():.2f}, 标准差: {wrist_roll.std():.2f}")
    print(f"夹爪均值:       {gripper.mean():.2f}, 标准差: {gripper.std():.2f}")
    
    # 判断是否有明显的方向偏斜
    pan_threshold = 5
    if shoulder_pan.mean() > pan_threshold:
        print(f"\n⚠️  肩pan均值({shoulder_pan.mean():.2f})偏右，可能导致抓取偏向右侧")
    elif shoulder_pan.mean() < -pan_threshold:
        print(f"\n⚠️  肩pan均值({shoulder_pan.mean():.2f})偏左，可能导致抓取偏向左侧")
    else:
        print(f"\n✓ 肩pan分布相对对称")
    
    # 分析状态观测数据
    print("\n" + "="*60)
    print("状态观测数据统计:")
    print("="*60)
    
    # 直接查看stats.json中的统计
    stats = dataset.meta.stats['observation.state']
    state_names = dataset.meta.info['features']['observation.state']['names']
    
    print("\n右臂状态统计:")
    for i in range(7):
        idx = 7 + i
        name = state_names[idx]
        print(f"{name}:")
        print(f"  均值: {stats['mean'][idx]:.2f}, 标准差: {stats['std'][idx]:.2f}")
        print(f"  范围: [{stats['min'][idx]:.2f}, {stats['max'][idx]:.2f}]")
    
    # 对比动作和状态的统计差异
    print("\n" + "="*60)
    print("动作 vs 状态统计对比:")
    print("="*60)
    
    action_stats = dataset.meta.stats['action']
    
    for i in range(7):
        idx = 7 + i
        action_mean = action_stats['mean'][idx]
        state_mean = stats['mean'][idx]
        diff = action_mean - state_mean
        
        print(f"\n{right_arm_names[i]}:")
        print(f"  动作均值: {action_mean:>10.2f}")
        print(f"  状态均值: {state_mean:>10.2f}")
        print(f"  差异:     {diff:>10.2f}")
        
        if abs(diff) > 5:
            print(f"  ⚠️  动作和状态均值差异较大")


if __name__ == "__main__":
    analyze_action_distribution()