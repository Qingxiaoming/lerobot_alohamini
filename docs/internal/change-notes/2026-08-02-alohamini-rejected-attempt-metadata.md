# AlohaMini rejected-attempt metadata

Date: 2026-08-02

## Scope

- `examples/alohamini/record_sim.py`
- `tests/robots/test_alohamini_record_sim.py`

## Observed problem

Rejected attempts were recorded in the session JSONL with only
`attempt_id` and `reason`. When `wait-result` produced a structured
`episode_result.json`, the reset generation was only available nested inside
that record; attempts rejected before `wait-result` (for example camera-ready
timeouts with zero frames) carried no stream-count evidence at all.

## Change

Failure rows now mirror the reset generation at the top level whenever a
valid `episode_result.json` is present, and always include a `raw_metadata`
summary (`frame_count`, `action_count`, `state_count`,
`dropped_incomplete_batches`, `dropped_incomplete_batches_after_ready`) when
the raw recorder metadata is readable. Successful rows also expose
`reset_generation` at the top level for consistent session-JSONL querying.
A `load_raw_metadata` helper keeps malformed or missing metadata tolerant.

## Verification

- `conda run -n lerobot_alohamini python -m pytest -p no:cacheprovider
  tests/robots/test_alohamini_record_sim.py -q` — pass; 9 tests (2 new).
- `conda run -n lerobot_alohamini ruff check examples/alohamini/record_sim.py
  tests/robots/test_alohamini_record_sim.py` — pass.
- `git diff --check` — pass.

## Remaining limitations

- Static and unit-test evidence only; the next live run should show the
  richer rejected-attempt rows in `session-*.jsonl`.
- Attempts rejected before the recorder wrote `raw/metadata.json` still have
  no stream counts to report.
