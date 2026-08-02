# AlohaMini structured episode result

Date: 2026-08-01

## Scope

- `examples/alohamini/record_sim.py`
- `tests/robots/test_alohamini_record_sim.py`

## Observed problem

The simulator already published authoritative C3-L1 task truth, but the
LeRobot collection loop accepted an attempt by searching `episode.log` for the
human-formatted substring `data: success`. That log spelling did not correlate
the result to the collector's attempt identity or simulator reset generation.

## Change

The capture command now passes its existing `attempt_id` into the Genie Sim
episode wrapper. For complete `run` attempts, `record_sim.py` requires a
version-1 `episode_result.json` whose task, attempt, generation, status,
timestamps, and metrics satisfy the shared task-runtime contract. Only a
correlated `success` with a zero capture exit status is eligible for dataset
conversion. Missing, malformed, mismatched, failed, or nonzero-exit attempts
are discarded, and a valid structured result is preserved in the session
JSONL for diagnosis.

Partial `pick` and `place` diagnostics retain their explicit
`--allow_partial_episode` behavior and do not claim full-task success.

## Verification

- `uv run pytest -p no:cacheprovider tests/robots/test_alohamini_record_sim.py -q`
  was unavailable because that bare uv environment does not contain pytest;
  `conda run -n lerobot_alohamini python -m pytest -p no:cacheprovider
  tests/robots/test_alohamini_record_sim.py -q` — 7 passed.
- `uv run ruff ...` was unavailable for the same environment reason;
  `conda run -n lerobot_alohamini ruff check examples/alohamini/record_sim.py
  tests/robots/test_alohamini_record_sim.py` — passed.
- `git diff --check` — passed.

## Remaining limitations

- This is static and unit-test evidence only; a fresh live C3-L1 run must still
  prove the container-installed producer and host-side collector handshake.
- Timeout before a terminal simulator result is represented by the capture
  command's nonzero exit and absence of a terminal result file.
