# Preview snapshot failure diagnostic

Implemented only in `Tests/Backup_Recovery/thread_diagnostics.py` and its existing `test_thread_diagnostics.py`; no product, runner, deadline, retry, or guard change. TASK-32562 scope authorized by root.

Exact Windows 081 evidence still identifies only primary-core included→unavailable during pre-maintenance rediscovery. `_CoreAdapter.validate` catches OSError/ValueError/SQLite errors and returns core_validation_unavailable; this result does not identify the original snapshot failure. No claim that the 081 cause is now proven.

The existing capture observer now temporarily wraps only `_PreviewScope.sqlite_target`. Success returns the original result untouched and emits nothing. Failure retains the existing bounded exception metadata (class, errno/winerror, basename/function/line frames) and a reason only for exact builtin ValueError/OSError with a single exact string from: preview_sqlite_changed, preview_sqlite_unavailable, preview_sqlite_limit, preview_sqlite_write_failed. Unknown messages become null. The wrapper never reads source values, file contents, paths, frame locals, or snapshot state. It retains at most eight shared diagnostic records, rethrows the same BaseException with a bare raise, and restores the exact original method at stop.

Failure-only metadata serialization/write failures are best-effort and cannot replace the snapshot exception or add a deferred stop failure. The small explicit broad catch is annotated for that original-error precedence. Existing inventory/capture observer behavior and cleanup-error policy are unchanged. No tracing, callback wrapping, retries, or new locks were added.

Native proof before edits: `/private/tmp/uat-preview-snapshot-diagnostic-probe.py` and `.log` reuse the existing real concurrent ordinary-note fixture, fixed 20-second child deadline. Genuine source-set change raises ValueError at sqlite_target's final before/after check; the original primary adapter still returns unavailable/config-only with core_validation_unavailable. Fresh stable discovery remains equal to the original successful scope. This is mechanism proof, not attribution of the preserved Windows case.

TDD: `/private/tmp/uat-capture-snapshot-red.log` records 5 failed/4 passed in 2.45s before helper implementation: metadata records absent in four error cases plus the native concurrent-note case. Four original-error controls already passed without instrumentation. First full GREEN: 31 cases. Final expanded helper suite: **33 passed in 2.32s**, `/private/tmp/uat-capture-snapshot-final.log`, including six metadata/write failure combinations (OSError, RuntimeError, KeyboardInterrupt), exact original error/terminal traceback, cancellation, unknown-message privacy, eight-record limit, success identity/no emission, exact method restoration, and the real concurrent-note adapter result. The native test imports/reuses the existing script rather than duplicating the fixture.

Ruff and git diff --check pass. Bandit baseline/current: B105 1/1 unchanged; B101 test assertions 115/134; no new non-assert findings. Hash/static receipt: `/private/tmp/uat-capture-snapshot-hashes.json`. Product sources and actual UAT fixtures were untouched. No native Windows pass claim; a fresh existing selection must produce the missing causal evidence.

Frozen hashes:
- thread_diagnostics.py: 059931cd11b345a8417546678310001c4806851d372e01dd96d7199456781b31
- test_thread_diagnostics.py: 091d61da80b2bee7dc1b88e4f297c59c3b6d1522a97115db875db7a73e8bdb5a
