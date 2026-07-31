# AlohaMini2Pro × Genie Sim：当前状态、操作与遗留问题

最后更新：2026-07-24

这份文档用于继续 AlohaMini2Pro、LeRobot 与 Genie Sim / Isaac Sim 的集成。
开始新会话时先阅读两个仓库各自的 `AGENTS.md`，并检查工作树；不要覆盖无关改动。

## 路径与机器

- LeRobot：`/home/yan/桌面/doing/lerobot_alohamini`
- Genie Sim：`/home/yan/genie_sim`
- Genie Sim Docker：`geniesim3`
- Docker 内工作区：`/workspace`
- 必须从 Docker 内 `/workspace/source/geniesim` 启动场景，因为这里有 `./assets`
- 真机树莓派：`pi5@192.168.88.2`
- 真机 LeRobot：`/home/pi5/lerobot_alohamini`
- 真机 Conda 环境：`lerobot_alohamini`
- 真机 ID / 标定文件名：`AlohaMiniRobot`

## 本轮结束时的进程状态

- Isaac Sim AlohaMini2Pro 场景和 GUI 已停止。
- RViz 已关闭。
- 本机真机→仿真镜像已停止。
- 树莓派 observation-only Host 已停止。
- Genie Sim LeRobot ZMQ 桥已停止。
- 树莓派和容器的 `5555/5556` 临时监听均已释放。
- 两个遗留的 OVRTX 渲染节点也已停止。

## 已完成并验证

### 模型与场景

- AlohaMini2Pro 已导入 Genie Sim，保留五个相机 link。
- 训练/推理接口只启用四个相机：
  - `forward`
  - `chest`
  - `wrist_left`
  - `wrist_right`
- 后置相机仍在 USD 中，但不发布到当前 LeRobot 四相机接口。
- 四路图像均验证为 `640×480×3`，不是黑屏或相机位于模型内部。
- ROS `/joint_states` 与 `/joint_command` 已打通。
- 18 个自由度：底盘虚拟 x/y/yaw、升降轴、左右各 7 个臂/夹爪关节。

### LeRobot 通信桥

Genie Sim 中新增：

```text
/home/yan/genie_sim/source/geniesim_ros/src/ros_ws/src/
  genie_sim_engine/scripts/alohamini_lerobot_bridge.py
```

它复用真机 `LeKiwiHost` 的 ZMQ 协议：

- 命令：TCP `5555`
- 观测：TCP `5556`
- 普通关节：`[-100, 100]`
- 夹爪：`[0, 100]`
- `x.vel/y.vel`：机体坐标系 m/s
- `theta.vel`：deg/s
- `lift_axis.height_mm`：mm

现有 `LeKiwiClient`、录制脚本和推理脚本不需要新增 simulator robot class。
端到端验证已通过：

- 收到 18 个状态字段。
- 收到四张 640×480 图像。
- 安全保持动作成功到达 `/joint_command`。
- 真机→仿真单向镜像稳定约 29.5 Hz。

### 仿真关节全范围数值验收

新增：

```text
examples/alohamini/validate_sim_joint_ranges.py
```

2026-07-25 已在不连接真机的情况下完成 14 个双臂/夹爪关节扫描：

- 普通关节：`0 → -100 → 0 → 100 → 0`
- 夹爪：`50 → 0 → 50 → 100 → 50`
- 共 70 个检查点，`70/70` 数值通过。
- 最大归一化误差：`0.057214`。
- 最差检查点：`arm_left_shoulder_lift.pos`，目标 `-100`，反馈
  `-99.942786`。
- 测试未中断，未出现反馈跳变。

报告：

```text
outputs/sim_joint_validation/alohamini_sim_joint_validation_20260725-201308.json
outputs/sim_joint_validation/alohamini_sim_joint_validation_20260725-201308.md
```

这只证明桥接映射、驱动目标和关节反馈一致；GUI 穿模、自碰撞、方向与夹爪几何仍需
人工确认，不能因为 `70/70` 就直接写入最终 URDF limits。

随后根据用户观察做了两项修正：

- 左右夹爪的 `sim_low/sim_high` 已交换，修复开合方向相反。
- 验收脚本改用用户保存的 `AlohaMiniRobot.natural_pose` 作为完整基线。未测试关节
  保持自然姿态，每个关节测试后也回到自身自然姿态值，不再回到全关节归一化 `0`。

第二次复测报告：

```text
outputs/sim_joint_validation/alohamini_sim_joint_validation_20260725-203240.json
outputs/sim_joint_validation/alohamini_sim_joint_validation_20260725-203240.md
```

第二次仍为 `70/70` 数值通过，最大误差 `0.065824`。需要用户根据本次 GUI 观察确认
夹爪方向和穿模情况，之后才能决定是否修改普通关节范围。

### 真机关节标定映射

真机 follower 标定来自：

```text
~/.cache/huggingface/lerobot/calibration/robots/alohamini/AlohaMiniRobot.json
```

换算关系：

```text
sim_rad = (raw_tick - 2048) * 2*pi / 4096
```

生成的 Genie Sim 配置：

```text
/home/yan/genie_sim/source/geniesim_ros/src/ros_ws/src/
  genie_sim_engine/config/alohamini2pro_joint_ranges.json
```

真机保存的自然姿态：

```text
~/.cache/huggingface/lerobot/calibration/robots/alohamini/
  AlohaMiniRobot.natural_pose.json
```

该自然姿态已发送到仿真，用户目视确认姿态一致。真机手动拖动机械臂时，
仿真双臂和夹爪也已成功实时跟随。

## 常用操作

### 1. 构建 Genie Sim ROS 工作区

```bash
docker exec -u 1000:1000 \
  -e HOME=/home/isaac-sim \
  -w /workspace \
  geniesim3 \
  bash -lc 'geniesim ros build dev'
```

### 2. 启动 Isaac Sim AlohaMini2Pro GUI

必须使用正确工作目录，否则会报 `./assets not found`：

```bash
docker exec -w /workspace/source/geniesim geniesim3 bash -lc '
source /workspace/devel/setup.bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_alohamini2pro \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false
'
```

后台启动：

```bash
docker exec -w /workspace/source/geniesim geniesim3 bash -lc '
source /workspace/devel/setup.bash
nohup ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_alohamini2pro \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false \
  >/tmp/geniesim_alohamini_gui.log 2>&1 &
'
```

查看日志：

```bash
docker exec geniesim3 tail -n 120 /tmp/geniesim_alohamini_gui.log
```

如果 GUI 内部整片灰，重启上述场景进程；不要用 RViz 代替 Isaac Sim GUI。

### 3. 启动 Genie Sim LeRobot 桥

```bash
docker exec -w /workspace geniesim3 bash -lc '
source /workspace/devel/setup.bash
ros2 run genie_sim_engine alohamini_lerobot_bridge.py \
  --rate 15 \
  --joint-ranges \
  /workspace/devel/share/genie_sim_engine/config/alohamini2pro_joint_ranges.json
'
```

未传 `--joint-ranges` 时会退回 `[-pi, pi]` 占位映射，只适合通信检查。

### 4. 在树莓派启动真机 Host

普通双向 Host：

```bash
ssh pi5@192.168.88.2
conda activate lerobot_alohamini
cd ~/lerobot_alohamini
python -m lerobot.robots.alohamini.lekiwi_host \
  --robot_model alohamini2pro
```

只读取真机、绝不接收动作的 Host：

```bash
python -m lerobot.robots.alohamini.lekiwi_host \
  --robot_model alohamini2pro \
  --observation_only
```

默认启动会执行升降轴下行回零。若某次测试不允许回零，再加：

```text
--no_home_lift
```

当前 Host 可能提示：

```text
Press ENTER to use provided calibration file ...
```

直接按 Enter 使用现有 `AlohaMiniRobot.json`；不要输入 `c`，否则会重新采集标定。

### 5. 真机单向控制仿真

先启动：

1. Isaac Sim 场景。
2. Genie Sim LeRobot 桥。
3. 树莓派 `--observation_only` Host。

然后在本项目根目录运行：

```bash
conda activate lerobot_alohamini
python examples/alohamini/mirror_physical_to_sim.py \
  --physical-ip 192.168.88.2
```

默认只同步双臂和夹爪 14 个字段，不同步底盘与升降轴。确认安全后才使用：

```text
--include-base-lift
```

### 6. 停止临时通信进程

本机镜像前台运行时按 `Ctrl+C`。

树莓派：

```bash
ssh pi5@192.168.88.2 \
  'pkill -f "^python -m lerobot.robots.alohamini.lekiwi_host"'
```

Genie Sim 桥：

```bash
docker exec geniesim3 bash -lc '
pkill -f "^python3 /workspace/devel/lib/genie_sim_engine/alohamini_lerobot_bridge.py" || true
pkill -f "^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 run genie_sim_engine alohamini_lerobot_bridge.py" || true
'
```

检查端口：

```bash
ssh pi5@192.168.88.2 'ss -ltnp | grep -E ":5555|:5556" || true'
```

## 本轮代码改动

LeRobot 仓库：

- `src/lerobot/robots/alohamini/lekiwi_host.py`
  - 新增 `--observation_only`
  - 新增 `--no_home_lift`
- `examples/alohamini/mirror_physical_to_sim.py`
  - 真机观测到仿真动作的单向镜像

注意：`lekiwi_host.py` 已通过 SCP 同步到树莓派：

```text
/home/pi5/lerobot_alohamini/src/lerobot/robots/alohamini/lekiwi_host.py
```

因此树莓派工作树中该文件现在可能显示为本地修改。

Genie Sim 仓库：

- AlohaMini2Pro robot package、mesh、xacro/provider。
- `scene_flat_alohamini2pro.yaml`
- `alohamini_lerobot_bridge.py`
- `alohamini2pro_joint_ranges.json`
- `docs/alohamini_lerobot_bridge.md`
- AlohaMini 专用 joint classification。
- CMake/package dependency 安装项。

## 已知问题与下一步

### 1. URDF 物理参数仍是占位值

- 机械臂关节原始 `effort="0"`、`velocity="0"` 已为仿真可运行而放宽。
- 当前刚度、阻尼、最大力能够稳定运行，但不等于真机辨识结果。
- 惯量和质量来自导入模型，尚未通过称重、CAD 或系统辨识复核。

这些不会阻塞通信接口和策略调用，但会影响动力学真实性与 sim-to-real。

### 2. 碰撞体仍需简化

当前大量 STL 同时用于 visual/collision。应生成简化碰撞体并检查：

- 自碰撞误报。
- 夹爪接触稳定性。
- 物体抓取摩擦和穿透。
- CPU/GPU 物理计算开销。

### 3. 夹爪极限只验证了自然姿态

夹爪方向已反转，`0/100` 数值目标与反馈已通过；仍需用户确认反转后的实际开合方向、
最大张开时 mesh 是否穿模，以及是否真实对应真机行程。

### 4. Host 标定检查存在集合不一致

双臂标定保存在一个 JSON 中，但 `left_bus.is_calibrated` 会拿左总线电机集合与整份
双臂标定集合比较，因此正常标定也可能提示重新选择。当前做法是按 Enter 沿用现有标定。
后续应在 `LeKiwi.is_calibrated` 中分别按左右总线过滤后比较，消除误提示。

### 5. Observation-only 仍执行通用连接配置

`--observation_only` 不处理命令，也不执行 watchdog 写停止动作；但 `robot.connect()`
仍会连接相机、调用 `configure()`，默认还会让升降轴回零。若需要严格寄存器只读模式，
应进一步拆分硬件连接与配置流程。

### 6. 底盘与升降镜像未实测

本轮只验证双臂和夹爪 14 维。底盘机体坐标转换已在代码中实现，但真机驱动仿真底盘、
升降轴高度跟随仍应在空旷、安全条件下单独测试。

### 7. Isaac Sim GUI 灰屏

曾出现 GUI 窗口内容整片灰，但物理进程仍在。重启场景后恢复。若复现，应记录：

- `/tmp/geniesim_alohamini_gui.log`
- GPU 显存与利用率。
- 窗口是否仍存在。
- ROS `/joint_states` 是否仍发布。

不要因为窗口灰屏直接判断物理仿真已经停止。

### 8. USD 虚拟底盘 visual reference 警告

启动日志中反复出现 `root_x_link`、`root_y_link` 的 unresolved reference prim path：

```text
configuration/robot_physics.usd@</visuals/root_x_link>
configuration/robot_physics.usd@</visuals/root_y_link>
```

机器人主体仍能显示且关节测试能运行，但应修复虚拟底盘 link 的 visual overlay 生成，
避免缓存重组或其他后端加载时出现缺失。

### 9. 训练策略上真机的边界

通信字段、形状、单位和 ZMQ 协议已经兼容；因此同一个 LeRobot policy/client 可以切换
仿真 Host 与真机 Host。仍未验证：

- 实际任务训练是否成功。
- 仿真相机与真机相机的视觉域差异。
- 动力学、延迟、摩擦和标定误差。
- 策略输出在真机上的安全性。

真机首次推理必须使用低速、限幅、急停和人工监护，不能把“通信兼容”当成
“策略已可安全部署”。
