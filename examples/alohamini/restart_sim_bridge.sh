#!/usr/bin/env bash

set -euo pipefail

CONTAINER="geniesim3"
HOST_CONFIG="/home/yan/genie_sim/source/geniesim_ros/src/ros_ws/src/genie_sim_engine/config/alohamini2pro_joint_ranges.json"
CONTAINER_CONFIG="/workspace/devel/share/genie_sim_engine/config/alohamini2pro_joint_ranges.json"
LOG_FILE="/tmp/alohamini_lerobot_bridge_manual.log"

if [[ ! -f "${HOST_CONFIG}" ]]; then
    echo "Config not found: ${HOST_CONFIG}" >&2
    exit 1
fi

echo "[1/3] Syncing joint-range config..."
docker cp "${HOST_CONFIG}" "${CONTAINER}:${CONTAINER_CONFIG}"

echo "[2/3] Stopping the old bridge..."
docker exec "${CONTAINER}" bash -lc '
pkill -f "^python3 /workspace/devel/lib/genie_sim_engine/alohamini_lerobot_bridge.py" || true
pkill -f "^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 run genie_sim_engine alohamini_lerobot_bridge.py" || true
rm -f /tmp/alohamini_lerobot_bridge_manual.log
'

echo "[3/3] Starting the bridge..."
docker exec \
    -u 1000:1000 \
    -e HOME=/home/isaac-sim \
    -w /workspace \
    "${CONTAINER}" \
    bash -lc "
source /workspace/devel/setup.bash
nohup ros2 run genie_sim_engine alohamini_lerobot_bridge.py \
    --rate 15 \
    --joint-ranges '${CONTAINER_CONFIG}' \
    >'${LOG_FILE}' 2>&1 </dev/null &
"

sleep 1
docker exec "${CONTAINER}" bash -lc '
if pgrep -af "^python3 /workspace/devel/lib/genie_sim_engine/alohamini_lerobot_bridge.py"; then
    echo "Bridge restarted successfully."
else
    echo "Bridge failed to start. Log follows:" >&2
    tail -n 80 /tmp/alohamini_lerobot_bridge_manual.log >&2
    exit 1
fi
'
