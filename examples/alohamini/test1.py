#!/usr/bin/env python3
"""
通过 ZMQ 桥接将 AlohaMini 所有关节平滑移动到各自的中点位置。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import zmq

# 关节名称（与 Genie Sim 中的字段名一致）
REGULAR_JOINTS = (
    "arm_left_shoulder_pan.pos",
    "arm_left_shoulder_lift.pos",
    "arm_left_elbow_flex.pos",
    "arm_left_wrist_flex.pos",
    "arm_left_wrist_yaw.pos",
    "arm_left_wrist_roll.pos",
    "arm_right_shoulder_pan.pos",
    "arm_right_shoulder_lift.pos",
    "arm_right_elbow_flex.pos",
    "arm_right_wrist_flex.pos",
    "arm_right_wrist_yaw.pos",
    "arm_right_wrist_roll.pos",
)
GRIPPERS = ("arm_left_gripper.pos", "arm_right_gripper.pos")
ARM_JOINTS = REGULAR_JOINTS + GRIPPERS
BASE_FIELDS = ("x.vel", "y.vel", "theta.vel")
LIFT_FIELD = "lift_axis.height_mm"

# 目标中点值（根据表格映射到对应的动作字段）
# 注意：这里填写的数值是表格中的“中点”列，单位应与模拟环境匹配。
# 如果模拟使用弧度或百分比，请按实际转换。
MID_TARGETS = {
    "arm_left_shoulder_pan.pos": 2054,
    "arm_left_shoulder_lift.pos": 2137,
    "arm_left_elbow_flex.pos": 1977.5,
    "arm_left_wrist_flex.pos": 2033.5,
    "arm_left_wrist_yaw.pos": 2066.5,
    "arm_left_wrist_roll.pos": 2047.5,
    "arm_left_gripper.pos": 2733.5,
    "arm_right_shoulder_pan.pos": 2135,
    "arm_right_shoulder_lift.pos": 2061,
    "arm_right_elbow_flex.pos": 1952.5,
    "arm_right_wrist_flex.pos": 2180,
    "arm_right_wrist_yaw.pos": 2058,
    "arm_right_wrist_roll.pos": 2047.5,
    "arm_right_gripper.pos": 2768,
}


def receive_latest(socket: zmq.Socket, timeout_ms: int) -> dict[str, Any]:
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    if not dict(poller.poll(timeout_ms)).get(socket):
        raise TimeoutError(f"No Genie Sim observation received for {timeout_ms / 1000:g}s")
    latest = json.loads(socket.recv_string())
    while True:
        try:
            latest = json.loads(socket.recv_string(flags=zmq.NOBLOCK))
        except zmq.Again:
            return latest


def smoothstep(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def move_all_joints(
    command_socket: zmq.Socket,
    observation_socket: zmq.Socket,
    action: dict[str, float],
    targets: dict[str, float],
    duration_s: float,
    rate_hz: float,
    timeout_ms: int,
) -> None:
    """将 action 中所有在 targets 里的关节平滑移动到目标值，其他字段保持不变。"""
    # 记录起始值
    starts = {joint: action[joint] for joint in targets.keys() if joint in action}
    if not starts:
        raise ValueError("No matching joints found in action dictionary.")

    steps = max(1, round(duration_s * rate_hz))
    period_s = 1.0 / rate_hz

    for step in range(1, steps + 1):
        started = time.monotonic()
        alpha = smoothstep(step / steps)
        for joint, target in targets.items():
            if joint in action:
                action[joint] = starts[joint] + (target - starts[joint]) * alpha
        command_socket.send_string(json.dumps(action))
        receive_latest(observation_socket, timeout_ms)
        time.sleep(max(0.0, period_s - (time.monotonic() - started)))

    # 最后精确赋值一次
    for joint, target in targets.items():
        if joint in action:
            action[joint] = target
    command_socket.send_string(json.dumps(action))
    receive_latest(observation_socket, timeout_ms)


def hold(
    command_socket: zmq.Socket,
    observation_socket: zmq.Socket,
    action: dict[str, float],
    duration_s: float,
    rate_hz: float,
    timeout_ms: int,
) -> None:
    deadline = time.monotonic() + duration_s
    period_s = 1.0 / rate_hz
    while time.monotonic() < deadline:
        command_socket.send_string(json.dumps(action))
        receive_latest(observation_socket, timeout_ms)
        time.sleep(period_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-ip", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=5555)
    parser.add_argument("--observation-port", type=int, default=5556)
    parser.add_argument("--move-seconds", type=float, default=2.0,
                        help="插值持续时间（秒）")
    parser.add_argument("--hold-seconds", type=float, default=2.0,
                        help="到达中点后保持的时间（秒），0 表示不保持")
    parser.add_argument("--rate", type=float, default=15.0,
                        help="命令发送频率（Hz）")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="接收观测超时（秒）")
    parser.add_argument("--yes", action="store_true",
                        help="直接执行，不询问确认")
    args = parser.parse_args()

    if min(args.move_seconds, args.rate, args.timeout) <= 0 or args.hold_seconds < 0:
        parser.error("move-seconds, rate, timeout 必须为正，hold-seconds 不能为负")

    if not args.yes:
        print("此脚本将移动所有关节到中值位置。")
        print("请确保模拟正在运行且 ZMQ 端口正确。")
        answer = input("输入 'sim' 继续: ")
        if answer != "sim":
            raise SystemExit("已取消。")

    timeout_ms = math.ceil(args.timeout * 1000)
    context = zmq.Context()
    command_socket = context.socket(zmq.PUSH)
    command_socket.setsockopt(zmq.CONFLATE, 1)
    command_socket.connect(f"tcp://{args.sim_ip}:{args.command_port}")
    observation_socket = context.socket(zmq.PULL)
    observation_socket.setsockopt(zmq.CONFLATE, 1)
    observation_socket.connect(f"tcp://{args.sim_ip}:{args.observation_port}")

    try:
        # 获取当前状态
        observation = receive_latest(observation_socket, timeout_ms)
        required = (*ARM_JOINTS, LIFT_FIELD)
        missing = [f for f in required if f not in observation]
        if missing:
            raise KeyError(f"观测缺少字段: {missing}")

        # 构建动作字典：当前关节值 + 底盘速度置零 + 升降轴保持当前
        action = {field: float(observation[field]) for field in required}
        action.update(dict.fromkeys(BASE_FIELDS, 0.0))

        # 检查目标是否与动作字段匹配
        valid_targets = {k: v for k, v in MID_TARGETS.items() if k in action}
        if not valid_targets:
            raise ValueError("MID_TARGETS 中没有与动作字段匹配的项，请检查名称。")

        print("开始将所有关节移动到中点...")
        move_all_joints(
            command_socket,
            observation_socket,
            action,
            valid_targets,
            args.move_seconds,
            args.rate,
            timeout_ms,
        )

        if args.hold_seconds > 0:
            print(f"保持中点位置 {args.hold_seconds}s...")
            hold(
                command_socket,
                observation_socket,
                action,
                args.hold_seconds,
                args.rate,
                timeout_ms,
            )

        print("完成：所有关节已到达中点。")
    except KeyboardInterrupt:
        print("\n用户中断，机器人将停在当前姿态。")
    finally:
        command_socket.close(linger=0)
        observation_socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
