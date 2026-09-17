# Bounded runtime-settlement failed-hook diagnostics

TASK-32562; test-only change after db4247b8db. Frozen files and hashes: `/private/tmp/uat-failed-hook-hashes.json`.

`observe_runtime_settlement` now snapshots at most 64 original list/tuple hook references and a monotonic start time without wrapping or changing any hook. On an original `_settle_stage` exception it walks at most 64 traceback frames, requires exact `original_stage.__code__` identity, and reads only that frame's `hook` local. It accepts that object only by `is` identity against the bounded original hook snapshot. `failed_hook` uses the same bounded drain qualname as the existing candidate-hook identity label; it does not claim whether close or drain threw. `stage_elapsed` records elapsed stage duration. Unknown/out-of-bound hook identity stays null.

Metadata failures cannot replace the original stage error, including cancellation raised while obtaining metadata. Existing diagnostic-stop failure signaling and original function restoration remain unchanged. No product code, native gate, callback, equality call, lock, deadline, retry, or unrelated diagnostic changed.

Verification:
- RED: four cases failed at missing metadata and replacement of the original error by metadata failure, `/private/tmp/uat-failed-hook-red-final.log`, 0.81s. A preceding test-only class-definition mistake was corrected before this valid RED.
- Additional metadata cancellation RED: 1 failed / 1 passed, `/private/tmp/uat-failed-hook-metadata-red.log`, 0.72s, before protecting the original stage error from a metadata BaseException.
- Final complete diagnostic module: **22 passed in 0.81s**, `/private/tmp/uat-failed-hook-final.log`.
- Native `_settle_stage` regressions prove exact second false-drain hook; throwing-close and cancelled-drain original identities/closed bookkeeping; 65th hook unclassified with no observer equality callbacks; metadata error/cancellation preserves the original native failure. Existing privacy, lease-snapshot, nonblocking, successful-result, cancellation, restoration, and write-failure tests pass.
- Ruff clean. Bandit baseline102/current116, exactly14 new test assertions and zero new nonassert finding. `git diff --check` clean. No new native build or CI run.

Replay: activate the project venv, then `python -m pytest Tests/Backup_Recovery/test_thread_diagnostics.py -q`.

This prepares evidence for the unresolved Windows Library settlement refusal. It does not identify the actual failing Windows owner until the next existing native run emits the new metadata. No commit/push performed.
