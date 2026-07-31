# Documentation

The repository documentation is divided by audience and publication status.

| Location | Purpose | Published in the documentation site |
| --- | --- | --- |
| [`source/`](source/) | Public LeRobot and AlohaMini user documentation | Yes |
| [`modules/`](modules/) | Documentation owned by a code module or example | No |
| [`internal/`](internal/) | Agent guides, operational runbooks, handoffs, and dated change notes | No |
| [`maintainers/`](maintainers/) | Documentation build and maintenance instructions | No |

Start from the repository-level [`DOCS.md`](../DOCS.md) for a task-oriented index.

## Public Documentation

Public pages live under [`source/`](source/) and must be registered in
[`source/_toctree.yml`](source/_toctree.yml). AlohaMini has its own public section under
[`source/alohamini/`](source/alohamini/).

## Supporting Documentation

Module documentation is centralized under [`modules/`](modules/) while source-local README files may
remain as symlinks when package discovery or GitHub rendering requires them.

Internal documents are indexed from [`internal/README.md`](internal/README.md). Stable procedures belong
in `internal/runbooks/`; point-in-time status belongs in `internal/handoffs/`; dated fix and enhancement
records belong in `internal/change-notes/`.

## Maintaining the Site

See [`maintainers/building-docs.md`](maintainers/building-docs.md) for build, preview, navigation, and
authoring instructions.
