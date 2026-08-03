#!/usr/bin/env python3

"""Automated simulation rollout evaluation for AlohaMini policies.

Reuses the exact ``lerobot-rollout`` pipeline (``build_rollout_context`` +
``create_strategy``) so evaluation is identical to deployment.  Per episode:

1. reset the simulator task (``reset-task --execute``) and wait until the
   task state reports ``settled=true``;
2. run the policy for ``--episode-duration-s`` seconds;
3. read the generation-latched terminal task result
   (``/genie_sim/task_state/result``) and classify
   success / failure / timeout;
4. reset the policy chunk queue and start the next episode.

Only one ZMQ observation consumer may be connected at a time: do not run
``lerobot-rollout`` or ``inspect_sim_action.py`` concurrently.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import lerobot.robots.alohamini  # noqa: F401  — registers alohamini_client robot type
from lerobot.configs import parser as lr_parser
from lerobot.rollout import RolloutConfig, build_rollout_context, create_strategy
from lerobot.utils.process import ProcessSignalHandler


@lr_parser.wrap()
def _parse_rollout_config(cfg: RolloutConfig) -> RolloutConfig:
    """Parse a RolloutConfig through the exact lerobot-rollout CLI path."""
    return cfg


def build_rollout_config(rollout_args: list[str]) -> RolloutConfig:
    """Parse rollout CLI args via the same parser the deployment CLI uses."""
    saved_argv = sys.argv
    sys.argv = ["eval_sim.py", *rollout_args]
    try:
        return _parse_rollout_config()
    finally:
        sys.argv = saved_argv


def docker_exec(container: str, cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", container, "bash", "-lc", cmd],
        capture_output=True,
        text=True,
        timeout=120,
    )


def reset_task(container: str) -> None:
    """Run the task reset and wait for the simulator to settle."""
    result = docker_exec(
        container,
        "source /workspace/devel/setup.bash && "
        "ros2 run genie_sim_task_runtime alohamini_task_control.py reset-task --execute",
    )
    if result.returncode != 0:
        raise RuntimeError(f"reset-task failed: {result.stderr[-500:]}")
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        probe = docker_exec(
            container,
            "source /workspace/devel/setup.bash && "
            "timeout 4 ros2 topic echo /genie_sim/task_state/settled --field data --once",
        )
        if "true" in probe.stdout.lower():
            return
        time.sleep(0.5)
    raise RuntimeError("task did not settle within 20 s after reset")


def read_task_result(container: str) -> str:
    """Return 'pending' / 'success' / 'failure' from the task-state topic."""
    probe = docker_exec(
        container,
        "source /workspace/devel/setup.bash && "
        "timeout 4 ros2 topic echo /genie_sim/task_state/result --field data --once",
    )
    text = probe.stdout.strip().lower()
    for token in ("success", "failure", "pending"):
        if token in text:
            return token
    return "pending"


def poll_task_result(container: str, timeout_s: float) -> str:
    """Wait for a terminal task result, returning 'timeout' if none arrives."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = read_task_result(container)
        if result in ("success", "failure"):
            return result
        time.sleep(1.0)
    return "timeout"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an AlohaMini policy in the Genie Sim via task success rate."
    )
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-duration-s", type=float, default=60.0)
    parser.add_argument("--result-timeout-s", type=float, default=60.0)
    parser.add_argument("--task", type=str, default="pickup1")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--remote-ip", type=str, default="127.0.0.1")
    parser.add_argument("--robot-model", type=str, default="alohamini2pro")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--container", type=str, default="geniesim3")
    parser.add_argument("--out", type=Path, default=Path("eval_sim_results.json"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and print the rollout config, then exit without connecting.",
    )
    args = parser.parse_args()

    rollout_args = [
        "--strategy.type=base",
        "--inference.type=sync",
        "--robot.type=alohamini_client",
        f"--robot.remote_ip={args.remote_ip}",
        "--robot.id=eval_sim",
        f"--robot.robot_model={args.robot_model}",
        f"--policy.path={args.policy_path}",
        f"--task={args.task}",
        f"--device={args.device}",
        f"--fps={args.fps}",
        "--duration=0",
        "--return_to_initial_position=false",
    ]
    cfg = build_rollout_config(rollout_args)
    if args.dry_run:
        print(json.dumps({"rollout_config": str(cfg), "num_episodes": args.num_episodes}, indent=2))
        return

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    ctx = build_rollout_context(cfg, shutdown_event)
    strategy = create_strategy(cfg.strategy)
    strategy.setup(ctx)

    outcomes: list[str] = []
    try:
        for episode in range(1, args.num_episodes + 1):
            print(f"[eval] episode {episode}/{args.num_episodes}: resetting task...", flush=True)
            reset_task(args.container)
            ctx.runtime.cfg.duration = args.episode_duration_s
            strategy.run(ctx)
            outcome = poll_task_result(args.container, args.result_timeout_s)
            outcomes.append(outcome)
            print(f"[eval] episode {episode} outcome: {outcome}", flush=True)
            # Clear the policy chunk queue between episodes.
            ctx.policy.inference.reset()
    finally:
        strategy.teardown(ctx)

    counts = {label: outcomes.count(label) for label in ("success", "failure", "timeout")}
    summary = {
        "policy_path": args.policy_path,
        "num_episodes": len(outcomes),
        "counts": counts,
        "success_rate": counts["success"] / len(outcomes) if outcomes else 0.0,
        "outcomes": outcomes,
    }
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
