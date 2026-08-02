# AlohaMini simulator conversion failure retry

Date: 2026-07-31

## Scope

- `examples/alohamini/record_sim.py`
- `tests/robots/test_alohamini_record_sim.py`

## Observed problem

The raw camera recorder correctly marked a C3-L1 attempt invalid when any
post-ready timestamp lacked one of the four synchronized camera images.
`RawEpisode` raised `ValueError`, but the collection loop re-raised that
conversion exception and terminated the complete long-running session.

The retained incident bundle contained 319 complete four-camera groups and
five incomplete groups. Recorder and episode processes exited cleanly and the
four written camera files had equal complete-frame counts; this was a strict
synchronization rejection, not a writer-queue or disk-space failure.

## Change

The zero-tolerance camera contract remains unchanged: any incomplete camera
batch discards the entire attempt. Raw-bundle validation failures now:

1. write `conversion_failed`, exception type, and message to the session JSONL;
2. retain the raw bundle only with `--keep_failed`, otherwise delete it;
3. continue with the next attempt instead of terminating the collection loop.

`--max_attempts` remains the bounded session-level stop condition.
Actual LeRobot dataset creation, image encoding, or episode-write exceptions
still abort and retain the raw bundle; those failures can indicate dataset
corruption and must not be hidden as ordinary camera-drop retries.

## Verification

- `conda run -n lerobot_alohamini python -m pytest -p no:cacheprovider
  tests/robots/test_alohamini_record_sim.py -q` — 6 passed.
- Added a two-attempt regression: the first conversion raises the same camera
  synchronization `ValueError`, is logged and deleted, and the second attempt
  is saved and finalized.
- `ruff check` passed for both modified Python files.
- `git diff --check` passed.

## Remaining limitations

- The raw recorder currently counts incomplete timestamp batches but does not
  preserve the missing camera name or timestamp. Add that diagnostic separately
  if per-camera root-cause analysis is needed.
- Reaching `--max_attempts` without enough valid episodes still ends with the
  existing collection-shortfall error by design.
