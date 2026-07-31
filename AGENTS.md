This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`docs/internal/agents/user-support-guide.md`](./docs/internal/agents/user-support-guide.md)**
> (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).
>
> **Documentation index → [`DOCS.md`](./DOCS.md)**

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — Public HF documentation (`.mdx` files), including hardware guides and tutorials.
  Built separately via `docs-requirements.txt` and CI workflows.
- **`docs/modules/`** — Module-owned references, example guides, and short policy package READMEs.
- **`docs/internal/`** — Agent guides, runbooks, handoffs, and dated change notes that are not
  published on the documentation site.
- **`docs/maintainers/`** — Documentation build and publishing instructions.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `README.md`, `DOCS.md`, and `AGENTS.md`.

## Documentation Rules

Treat documentation location and indexing as part of the change:

- **Public documentation** belongs in `docs/source/`. Every user-facing page must be reachable from
  `docs/source/_toctree.yml`; AlohaMini public guides belong in `docs/source/alohamini/`.
- **Module-owned documentation** belongs in `docs/modules/`. A README may remain beside source code
  only when module discovery or GitHub rendering requires it; prefer a symlink to the canonical file
  in `docs/modules/`.
- **Internal documentation** belongs in `docs/internal/`:
  - reusable operational procedures go in `runbooks/`;
  - point-in-time state and next-session context go in `handoffs/`;
  - repair, enhancement, migration, and validation records go in `change-notes/`.
- **Maintainer instructions** for building or publishing documentation belong in
  `docs/maintainers/`.
- Do not add fix logs, enhancement logs, session notes, generated reports, or module-specific manuals
  to the repository root. Root-level Markdown documentation is limited to `README.md`, `DOCS.md`,
  `AGENTS.md`, and compatibility links such as `CLAUDE.md`. GitHub community-health files belong in
  `.github/`.
- Name dated internal records `YYYY-MM-DD-<scope>-<slug>.md`. State the affected scope, observed
  problem, change, verification evidence, and remaining limitations.
- When adding, moving, or renaming a document, update all affected indexes, `_toctree.yml` entries,
  relative links, and source README symlinks in the same change.
- Preserve commands, paths, option names, topic names, configuration keys, and code blocks exactly
  when reorganizing or translating documentation. Explicitly label host-, container-, simulator-, and
  real-robot-only procedures.
- Before handing off a documentation change, check internal links and symlink targets, run
  `git diff --check`, and build `docs/source/` with the documented `doc-builder` command when the
  documentation dependencies are available.

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
