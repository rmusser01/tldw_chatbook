# Ordinary-app later rollback: preserved-log target change

Read-only review, 2026-09-14. Current source HEAD d58650f42ebc1ee6a0f3b3aa9220ce14e312d1e7. The six relevant UI/planner/later/restart/launcher/diagnostics files have no diff against immutable d3082e. No UAT profile was opened, modified, or recovered by this investigation.

## Finding and limit

The ordinary application can invalidate a later-rollback review by writing its own intentionally excluded, preserved application log. This is a demonstrated source-level/native-file mechanism. It is consistent with the actual keyboard failure but not a conclusive reconstruction of its unretained old in-memory fingerprint or exception frame.

The UAT observer reports review 10:52:42 → confirm 10:58:07; only control-work and tldw_cli_app.log metadata changed, including UI event_loop_stall logging at 03:52:47/56. Config/core/sidecars showed no timestamp change. The displayed plan preserves app.log. No new journal or pending record was created.

## Exact mechanism

- `restore_plan.py:_paths` (323–339) includes all preserved paths. `_observed` (218) records file identity, mode, size, mtime/ctime, and content hash; `_fingerprint` (256) retains those observations. The bounded existing service/control exception does not apply to application logs.
- `later_rollback.py:execute_rollback` calls `recheck_targets(approved_plan)` at 1900 before opening the old copy, creating a workspace, or entering native maintenance. A log append therefore produces `target_changed` with no new journal.
- A preserved log is correctly protected. Excluding its content/metadata from the fingerprint would weaken reviewed target preservation; this is not recommended.

Disposable native probe: `/private/tmp/test_later_preserved_log_review.py`; result `/private/tmp/uat-later-preserved-log-review-final.log`: **1 passed in 0.66s**. It uses installed `_Diagnostics.discover`, private native directory/file creation, a real append, the real fingerprint and actual `execute_rollback` entry. Before append the review recheck succeeds; afterward owner/ID/path/status and inode remain the same but execution refuses exact `target_changed` before control access. The minimal typed plan intentionally does not claim an authenticated full rollback lifecycle. An initial probe assertion comparing all diagnostics metadata was corrected because discovery correctly updates log metadata; no product assertion was weakened.

## Existing flow asymmetry

Initial replacement calls `_requires_recovery_restart` from review/start, and `_sync_replacement_host` (2074) offers the recovery handoff and disables ordinary-app review. Later `_review_rollback` (764) and `_start_rollback` (874) lack equivalent guards and execute inside the ordinary app.

`app.py:request_recovery_restart` (18020) already rejects unsettled editors/running recovery, performs ordinary guarded shutdown, and retains the request only after successful shutdown. `recovery_restart.py` passes only nonsecret path hints into the existing fixed fresh interpreter. `launcher.py:recovery_app` constructs the minimal Textual host and RecoveryService without TldwCli's normal composition/logging setup. This removes the ordinary app's log writer before the new review; it is not a guarantee against independent external writers, which existing fingerprints must continue to detect.

## Smallest supported correction proposal (not implemented)

Require the same existing recovery handoff **before later-rollback review**, and reject stale/programmatic ordinary-app start as initial replacement does. Clear passwords and any reviewed plan; obtain a new inventory, copy-password review, acknowledgments, new safety-copy password, and consequence confirmation in RecoveryApp. Do not transport passwords, approved plans, authority, or target fingerprints across shutdown.

The existing `RecoveryRestart(None, target)` already permits the handoff. Current child on_mount always opens inspect/replace, so a small nonsecret copies-navigation hint (optionally selected copy ID, validated again by the child) would improve continuation without adding a new recovery mode. Alternatively reuse the request unchanged and require reselecting Recovery copies after restart. The latter has the smallest implementation surface but loses navigation continuity. No changes to service/planner/maintenance checks are needed to address this demonstrated flow problem.

Required verification before acceptance: ordinary-app later review cannot run service preview before handoff; stale start cannot run service execution; shutdown refusal leaves ordinary app usable; no secrets/plan cross restart; child reselects and freshly reviews actual committed copy; complete later rollback through existing native path succeeds, with all changed preserved-file refusals retained. Use actual normal-app logging in an isolated lifecycle; do not query live SQLite while replacement is pending.
