# AlohaMini2Pro × Genie Sim 命令清单

最后更新：2026-07-24

以下命令来自本轮实际操作，并补充了后续最常用的检查与停止命令。

## 变量与路径

```bash
# 本机 LeRobot
cd /home/yan/桌面/doing/lerobot_alohamini

# 本机 Genie Sim
cd /home/yan/genie_sim

# 真机
ssh pi5@192.168.88.2
cd ~/lerobot_alohamini
conda activate lerobot_alohamini
```

## Docker 状态

```bash
docker ps
docker ps --filter name=geniesim3
```

进入容器：

```bash
docker exec -it \
  -u 1000:1000 \
  -e HOME=/home/isaac-sim \
  -w /workspace \
  geniesim3 bash
```

## 构建 Genie Sim ROS

```bash
docker exec \
  -u 1000:1000 \
  -e HOME=/home/isaac-sim \
  -w /workspace \
  geniesim3 \
  bash -lc 'geniesim ros build dev'
```

## 启动 Isaac Sim AlohaMini2Pro

前台启动：

```bash
docker exec -it \
  -w /workspace/source/geniesim \
  geniesim3 bash -lc '
source /workspace/devel/setup.bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_alohamini2pro \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false
'
```

后台启动：

```bash
docker exec \
  -w /workspace/source/geniesim \
  geniesim3 bash -lc '
source /workspace/devel/setup.bash
nohup ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_alohamini2pro \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false \
  >/tmp/geniesim_alohamini_gui.log 2>&1 &
echo $!
'
```

注意：工作目录必须是 `/workspace/source/geniesim`。从 `/workspace` 启动会报：

```text
./assets not found and geniesim_assets is not installed
```

查看启动日志：

```bash
docker exec geniesim3 \
  tail -n 160 /tmp/geniesim_alohamini_gui.log
```

确认场景进程：

```bash
docker exec geniesim3 bash -lc '
pgrep -af "ros2 launch genie_sim_bringup app.launch.py|genie_sim_engine_isaacsim.py|genie_sim_render_node"
'
```

## 停止 Isaac Sim 场景

```bash
docker exec geniesim3 bash -lc '
pkill -f "^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 launch genie_sim_bringup app.launch.py scene:=scene_flat_alohamini2pro" || true
sleep 5
pkill -f "^/workspace/devel/lib/genie_sim_render/genie_sim_render_node" || true
'
```

检查是否有遗留：

```bash
docker exec geniesim3 bash -lc '
pgrep -af "genie_sim_engine_isaacsim.py|genie_sim_render_node|ros2 launch genie_sim_bringup app.launch.py|rviz2" || true
'
```

## 启动 LeRobot 仿真桥

需要先启动 Isaac Sim 场景：

```bash
docker exec -it \
  -w /workspace \
  geniesim3 bash -lc '
source /workspace/devel/setup.bash
ros2 run genie_sim_engine alohamini_lerobot_bridge.py \
  --rate 15 \
  --joint-ranges \
  /workspace/devel/share/genie_sim_engine/config/alohamini2pro_joint_ranges.json
'
```

后台运行：

```bash
docker exec \
  -w /workspace \
  geniesim3 bash -lc '
source /workspace/devel/setup.bash
nohup ros2 run genie_sim_engine alohamini_lerobot_bridge.py \
  --rate 15 \
  --joint-ranges \
  /workspace/devel/share/genie_sim_engine/config/alohamini2pro_joint_ranges.json \
  >/tmp/alohamini_lerobot_bridge.log 2>&1 &
'
```

查看桥日志：

```bash
docker exec geniesim3 \
  tail -n 80 /tmp/alohamini_lerobot_bridge.log
```

停止桥：

```bash
docker exec geniesim3 bash -lc '
pkill -f "^python3 /workspace/devel/lib/genie_sim_engine/alohamini_lerobot_bridge.py" || true
pkill -f "^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 run genie_sim_engine alohamini_lerobot_bridge.py" || true
'
```

## ROS 节点与话题检查

```bash
docker exec geniesim3 bash -lc '
source /workspace/devel/setup.bash
ros2 node list | sort
'
```

```bash
docker exec geniesim3 bash -lc '
source /workspace/devel/setup.bash
ros2 topic list | sort
'
```

只读取一帧关节状态：

```bash
docker exec geniesim3 bash -lc '
source /workspace/devel/setup.bash
timeout 5 ros2 topic echo /joint_states \
  --once \
  --qos-reliability best_effort
'
```

检查发布者/订阅者：

```bash
docker exec geniesim3 bash -lc '
source /workspace/devel/setup.bash
ros2 topic info /joint_states --verbose
ros2 topic info /joint_command --verbose
'
```

检查四相机：

```bash
docker exec geniesim3 bash -lc '
source /workspace/devel/setup.bash
ros2 topic list | grep -E "front_camera|chest_camera|left_camera|right_camera"
'
```

相机频率：

```bash
docker exec geniesim3 bash -lc '
source /workspace/devel/setup.bash
timeout 8 ros2 topic hz /genie_sim/front_camera_rgb/image_raw
'
```

## 查看桌面窗口

检查 Isaac Sim / RViz 窗口是否存在：

```bash
DISPLAY=:0 xwininfo -root -tree \
  | grep -Ei 'Isaac Sim|Omniverse|RViz'
```

RViz 仅用于 ROS 可视化，不是 Isaac Sim GUI。若临时需要：

```bash
docker exec -w /workspace geniesim3 bash -lc '
source /workspace/devel/setup.bash
rviz2 -d /workspace/devel/share/genie_sim_bringup/rviz/view_robot.rviz
'
```

RViz 中鼠标放在机器人上按 `F` 可聚焦；严格恢复视角使用
`Panels → Views → Reset`。

## 真机标定文件

树莓派查看 follower 标定：

```bash
cat ~/.cache/huggingface/lerobot/calibration/robots/alohamini/AlohaMiniRobot.json
```

查看自然姿态：

```bash
cat ~/.cache/huggingface/lerobot/calibration/robots/alohamini/AlohaMiniRobot.natural_pose.json
```

保存当前自然姿态：

```bash
conda activate lerobot_alohamini
cd ~/lerobot_alohamini
python -m lerobot.robots.alohamini.natural_pose save \
  --robot_model alohamini2pro \
  --arm both
```

不要用原始 `Goal_Position=2048` 代替自然姿态。

## 启动真机 Host

登录真机：

```bash
ssh pi5@192.168.88.2
conda activate lerobot_alohamini
cd ~/lerobot_alohamini
```

正常双向 Host：

```bash
python -m lerobot.robots.alohamini.lekiwi_host \
  --robot_model alohamini2pro
```

只发布观测、不接受动作：

```bash
python -m lerobot.robots.alohamini.lekiwi_host \
  --robot_model alohamini2pro \
  --observation_only
```

跳过升降回零：

```bash
python -m lerobot.robots.alohamini.lekiwi_host \
  --robot_model alohamini2pro \
  --observation_only \
  --no_home_lift
```

若询问是否使用现有标定，直接按 Enter；不要输入 `c`：

```text
Press ENTER to use provided calibration file ...
```

后台启动 observation-only Host：

```bash
nohup bash -c "
printf '\n' |
python -m lerobot.robots.alohamini.lekiwi_host \
  --robot_model alohamini2pro \
  --observation_only
" >/tmp/alohamini_observation_host.log 2>&1 </dev/null &
```

查看 Host：

```bash
pgrep -af 'lerobot.robots.alohamini.lekiwi_host'
tail -n 120 /tmp/alohamini_observation_host.log
ss -ltnp | grep -E ':5555|:5556'
```

停止 Host：

```bash
pkill -f '^python -m lerobot.robots.alohamini.lekiwi_host'
```

## 真机单向镜像到仿真

启动顺序：

1. Isaac Sim 场景。
2. Genie Sim LeRobot 桥。
3. 树莓派 observation-only Host。
4. 本机镜像脚本。

本机运行：

```bash
cd /home/yan/桌面/doing/lerobot_alohamini
conda activate lerobot_alohamini
python examples/alohamini/mirror_physical_to_sim.py \
  --physical-ip 192.168.88.2
```

默认只同步双臂和夹爪 14 维。同步底盘和升降：

```bash
python examples/alohamini/mirror_physical_to_sim.py \
  --physical-ip 192.168.88.2 \
  --include-base-lift
```

停止镜像：按 `Ctrl+C`。

## 仿真逐关节全范围验收

先启动 Isaac Sim 场景和 LeRobot 仿真桥，不需要连接真机。

完整扫描 14 个双臂/夹爪关节：

```bash
cd /home/yan/桌面/doing/lerobot_alohamini
conda activate lerobot_alohamini
python examples/alohamini/validate_sim_joint_ranges.py --yes
```

普通关节依次执行：

```text
0 → -100 → 0 → 100 → 0
```

夹爪依次执行：

```text
50 → 0 → 50 → 100 → 50
```

仅扫描一个或多个指定关节：

```bash
python examples/alohamini/validate_sim_joint_ranges.py \
  --joint arm_left_shoulder_pan.pos \
  --joint arm_left_gripper.pos
```

调整插值时间、保持时间和误差阈值：

```bash
python examples/alohamini/validate_sim_joint_ranges.py \
  --duration 3 \
  --hold 1 \
  --tolerance 2
```

随时按 `Ctrl+C` 停止。报告保存到：

```text
outputs/sim_joint_validation/
```

2026-07-25 首次完整验收结果：

```text
70/70 checkpoints passed
maximum normalized error: 0.057214
worst checkpoint: arm_left_shoulder_lift.pos at -100
```

报告：

```text
outputs/sim_joint_validation/
  alohamini_sim_joint_validation_20260725-201308.json
  alohamini_sim_joint_validation_20260725-201308.md
```

数值通过只表示目标值与 `/joint_states` 反馈一致。穿模、自碰撞、方向和夹爪几何
仍需观察 Isaac Sim GUI，并在 Markdown 报告中手动勾选。

2026-07-25 第二次复测改为以保存的 `AlohaMiniRobot.natural_pose` 为基线，
不再把其他关节留在归一化零位；同时反转了左右夹爪映射：

```text
70/70 checkpoints passed
maximum normalized error: 0.065824
worst checkpoint: arm_right_shoulder_lift.pos at its natural-pose target
```

新报告：

```text
outputs/sim_joint_validation/
  alohamini_sim_joint_validation_20260725-203240.json
  alohamini_sim_joint_validation_20260725-203240.md
```

默认基线已内置到脚本，也可使用最新真机自然姿态文件覆盖：

```bash
python examples/alohamini/validate_sim_joint_ranges.py \
  --baseline-pose /path/to/AlohaMiniRobot.natural_pose.json
```

## LeRobot 连接仿真

仿真桥在本机 `5555/5556` 运行时，现有客户端使用：

```text
remote_ip=127.0.0.1
robot_model=alohamini2pro
```

例如策略评估：

```bash
conda activate lerobot_alohamini
cd /home/yan/桌面/doing/lerobot_alohamini
python examples/alohamini/evaluate_bi.py \
  --remote_ip 127.0.0.1 \
  --policy_path outputs/train/<policy>/checkpoints/last/pretrained_model
```

实际参数以脚本 `--help` 为准：

```bash
python examples/alohamini/evaluate_bi.py --help
python examples/alohamini/record_bi.py --help
python examples/alohamini/replay_bi.py --help
```

## 网络与端口排查

确认树莓派连通：

```bash
ping -c 3 192.168.88.2
ssh -o BatchMode=yes -o ConnectTimeout=5 pi5@192.168.88.2 'hostname'
```

树莓派端口：

```bash
ssh pi5@192.168.88.2 \
  'ss -ltnp | grep -E ":5555|:5556" || true'
```

容器桥进程：

```bash
docker exec geniesim3 \
  pgrep -af alohamini_lerobot_bridge
```

注意：物理 Host 与仿真桥都默认使用 `5555/5556`。它们在不同 IP 上可同时运行；
同一 IP 上不能占用同一端口。

## 代码检查

```bash
cd /home/yan/桌面/doing/lerobot_alohamini
conda activate lerobot_alohamini
ruff check \
  src/lerobot/robots/alohamini/lekiwi_host.py \
  examples/alohamini/mirror_physical_to_sim.py
ruff format --check \
  src/lerobot/robots/alohamini/lekiwi_host.py \
  examples/alohamini/mirror_physical_to_sim.py
git diff --check
git status --short
```

Genie Sim 工作树：

```bash
git -C /home/yan/genie_sim status --short
git -C /home/yan/genie_sim diff --check
```

## 一键检查是否全部停止

本机镜像：

```bash
pgrep -af mirror_physical_to_sim || true
```

树莓派 Host 与端口：

```bash
ssh pi5@192.168.88.2 '
pgrep -af "lerobot.robots.alohamini.lekiwi_host" || true
ss -ltnp | grep -E ":5555|:5556" || true
'
```

容器进程：

```bash
docker exec geniesim3 bash -lc '
pgrep -af "genie_sim_engine_isaacsim.py|genie_sim_render_node|app.launch.py|alohamini_lerobot_bridge|rviz2" || true
'
```
