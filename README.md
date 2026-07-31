# lerobot_alohamini

Shared software layer for the AlohaMini product line, built on HuggingFace LeRobot. Supports both the full AlohaMini robot (dual-arm + mobile base + lift) and the AM-ARM200 arm.

> Haven't assembled your hardware yet? Start here: [AlohaMini](https://github.com/liyiteng/alohamini) · [AM-ARM200](https://github.com/liyiteng/AM-ARM)

## Updates
- **[2025-05-21]** Add support for AM-ARM200 and AlohaMini 2 / 2 Pro
- **[2025-04-10]** Compatible with LeRobot 0.5.2

## Documentation

Start with setup, then follow the workflow for your hardware. The repository-wide
[documentation index](DOCS.md) also links LeRobot references, module documentation, and internal
runbooks.

### Recommended Path

1. [Install](docs/source/alohamini/installation.mdx) — prepare the environment, serial port permissions, and Hugging Face login.
2. Pick your robot workflow:
   - [AM-ARM200](docs/source/alohamini/am-arm200.mdx) — single-arm workflow on one PC: calibration, teleoperation, dataset recording, training, and evaluation.
   - [AlohaMini 1 / 2 / 2 Pro](docs/source/alohamini/alohamini.mdx) — dual-arm workflow with Pi + PC: calibration, teleoperation, dataset recording, training, and evaluation.

### References

| Reference | Use it for |
|-----------|------------|
| [AlohaMini Documentation](docs/source/alohamini/index.mdx) | Public documentation home and workflow selection |
| [Hardware Profiles](docs/source/alohamini/profiles.mdx) | `--arm_profile` and `--robot_model` flag meanings |
| [Command Reference](docs/source/alohamini/commands.mdx) | Setup, host, teleoperation, recording, training, evaluation, and common checks |
| [Debug Tools](docs/modules/examples/debug-tools.md) | Low-level motor, wheel, lift axis, servo ID, phase, midpoint, torque, and scripted-action debug functions |

---

## Team & Contact

AlohaMini is created by **Li Yiteng** and **Wu Zhiyong**.

- Email: liyiteng+github@gmail.com
- WeChat: liyiteng

## Acknowledgements

- [LeRobot](https://github.com/huggingface/lerobot) — the software stack this repository targets
- [ALOHA](https://tonyzhaozh.github.io/aloha/) — the bimanual teleoperation paradigm
- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) — pioneered the low-cost open arm design pattern
