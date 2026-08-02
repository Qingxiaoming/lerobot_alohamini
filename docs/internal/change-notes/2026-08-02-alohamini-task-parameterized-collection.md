# AlohaMini task-parameterized collection

Date: 2026-08-02

## Scope

- `examples/alohamini/record_sim.py`
- `tests/robots/test_alohamini_record_sim.py`

## Observed problem

The simulator collector was hardwired to the C3-L1 task: it always ran
`alohamini_c3_l1_capture.sh`, never forwarded a task selection into the
container, and `load_episode_result` rejected any structured result whose
`task_id` was not exactly `"c3_l1"`. The second vertical slice
(`alohaminipro_Fruits`) therefore could not be collected.

## Change

`record_sim.py` is now task-parameterized:

- New `--task` argument (`c3_l1` default, `alohaminipro_fruits` supported).
- `capture_attempt` forwards `GENIESIM_TASK` and `GENIESIM_TASK_ID` through
  `docker exec -e`, so the episode script runs the selected task plugin and
  `wait-result` writes the matching task id into `episode_result.json`.
- `load_episode_result` / `episode_succeeded` validate the result against the
  requested task id instead of the hardcoded `c3_l1`.

## Verification

- `conda run -n lerobot_alohamini python -m pytest -p no:cacheprovider
  tests/robots/test_alohamini_record_sim.py -q` — pass; 10 tests (new
  env-passthrough and task-id parameterization cases).
- `conda run -n lerobot_alohamini ruff check examples/alohamini/record_sim.py
  tests/robots/test_alohamini_record_sim.py` — pass.
- `git diff --check` — pass.

## Remaining limitations

- Static/unit evidence only; a live `--task alohaminipro_fruits` collection
  run should save a fruits dataset episode with a matching task id.
- The episode/capture shell entry names still say `c3_l1`; renaming them is a
  compatibility decision for the runtime refactor's Phase 8, not part of this
  change.
