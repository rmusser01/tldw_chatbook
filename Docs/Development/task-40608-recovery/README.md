# TASK-40608 recovery checkpoint

This directory preserves source patches and continuation state for the owner's crash-recovery request on 2026-11-16. The initial source checkpoint is `03be283bb0229b904bea306fd05583ae589fe91f`; draft PR: https://github.com/rmusser01/tldw_chatbook/pull/3575. These patch files preserve an earlier integration draft; consult the live code and current result below before using them. Do not apply them wholesale over newer code.

## Required outcome

Keep every existing storage, permission, currentness, approval, persistence, and retirement check. Both unchanged README and approved-edit journeys must finish below five seconds in every candidate of three sequential pairs, also retaining the original 30% comparison. Order: B1,C1,C2,B2,B3,C3. Subtract only actual measured human approval wait. The final pairs have not started; TASK-40608 remains In Progress.

The accepted architecture uses one contained backend runtime/store/controller/domain/permission owner, finite IPC values, a readonly parent projection adapter, and the real UI attachment. Production process selection remains disabled. Tasks 60 and 61 are complete; Task 62 is active; Task 63's complete mounted-caller qualification is conditional.

## Current result and next action

**2026-11-17 interruption recovery:** session-view/current-input publication, the readonly projection view, and the accepted-turn test are now installed in the working implementation. Their first run, `task62-accepted-process-a1`, cannot qualify them: after the interruption its after JSON is entirely NUL-filled, and its outer/inner XML and log are partly NUL-filled. The surviving log prefix shows two passes and two failures, but the failure details and normal after proof are unavailable. Preserve those files and repeat the same four checks in a fresh run. All 2,836 live input hashes and all frozen copies still match the pre-run manifest; its SHA-256 is `24382a132d6e16d9ea4044acd2210afbad8db615b470a46a7f5d4c777224d4ed`. The actual-user census at `2026-11-17T15:18:24.6553750Z` found no surviving workers. No accepted provider/terminal result is claimed. The last fully observed passing checkpoint below was pushed as `571af6c263` before the interruption.

The correction passed `task62-plan-process-a10`: both the actual source/selector regression and unchanged process pre-ACK test passed. The process reached the original commit plan in **2.484 seconds**, reused the actual projection reference, and positively retired the original pre-ACK cancellation. Repeated association checks entered the native getter **zero times**; selector/cache/environment/CWD and definition changes were refused. Normal after receipt: `2026-11-17T06:32:33.8797249Z`, SID ending 1002, all 2,835 hashes unchanged, zero workers. XML: 2 tests, 0 failures/errors, 108.175 seconds including isolated fixture setup/cleanup. Runner SHA-256: `c2b2de436897d43161e91347642cddd813aef8c4ca11f701f3c724bff25c422d`; manifest SHA-256: `75d3f704347e2344f8c1b542c1f3c0de2ccb573b12eb84b6d71863fb737b4edc`.

Next, qualify the installed session-view/current-input sidecar and real accepted ACK/provider/terminal-persistence test. The 2.484-second pre-plan result is not a whole journey, UI, or final three-pair speed qualification. Diagnostic timing logs remain temporary.

The preceding diagnostic and its failure remain evidence:

`task62-plan-process-a9` failed its unchanged ten-second pre-plan wait. The normal after receipt at `2026-11-17T06:06:41.5852226Z` recorded the actual Windows user (SID ending 1002), all 2,835 frozen source hashes unchanged, and zero remaining workers. XML: one test, one failure, zero errors, 79.509 seconds. Runner SHA-256: `2a434967759e934b2ff9e06ffb34afb268a4e3b327a173e371e4b9ae85e132e2`; manifest SHA-256: `3f4acc5a07c7a09a4a06835b27edc47d41035b929a8894bc427b7469dd398289`.

The final required configuration read consumed 8.093 seconds elapsed and 7.890625 seconds thread CPU before cancellation. Its observed stack was `open_handle → _parent → stat → raw.scope → config_participants.operation → guarded getter → backend.require_sources → presentation.require_association → _require_mcp_attachment_source → configuration currentness callback`. This is a sampled location, not a complete cost profile. Source inspection confirms that the newly added backend association check invokes guarded `config.get_user_data_dir` repeatedly; each call enters storage admission. The original local presentation association checks object/definition identity without reacquiring directory ownership. Resolve this boundary while retaining all selector/config/source drift and original native checks; then repeat the same actual-process test. Do not widen its deadline or warm the fixture.

Earlier contained-process startup/session/permission/lifecycle verification passed. Decision (6), commit-plan encoding (3), and projection (2) focused checks passed in the recorded combined run. The real accepted provider/tool/terminal path, independent FULL proof, UI attachment, and whole-journey performance remain open. Source-only review and cancelled work are not successful timing evidence.

The checkpoint also removes trailing whitespace in the initial plan and one extra EOF blank line in `test_console_execution_inherited_display.py`; refresh that test's hash for any next frozen run.

## Pending patches

- `task62-session-view-publication-draft/incremental.diff` preserves seven-file session-row/current-input publication work, including the new readonly view. Its worker hunk predates the latest diagnostics; merge surgically. The manifest records original/proposed hashes, not current live hashes.
- `task62-parent-transcript-adapter-draft/proposed/execution_projection.py` is the source-reviewed UI adapter draft; its report and contract explain the original-widget checks.
- `task62-readonly-projection-view-*` preserves its contract/review. The module itself is already present in the session-view patch.
- `task62-accepted-process-turn-test-draft/proposal.diff` extends the real-process test with a controlled loopback provider, original ACK, terminal persistence, and cleanup. It depends on the pending session-view/current-input APIs and has not run.
- `task-62-process-integration-plan.md` preserves the immediate plan. The canonical larger plan and ADRs remain in their normal repository locations.

Patch headers were made repository-relative; the new-file header uses `/dev/null`. Source hunks are unchanged. `snapshot.json` records the copied artifacts. Local test logs, private profiles/databases, repeated source snapshots, and machine-specific raw receipts are intentionally not published. They remain in the local `.superpowers/sdd/2026-11-13-task-40608-sub-five` and external verification lab; this directory does not claim to replace their evidence.

## Coordination and verification

Use one actual-Windows-user native launcher at a time. Freeze source and tests before each run; inspect the normal after receipt for unchanged hashes and no surviving workers before releasing the freeze. A forced cleanup is not success. Targeted checks only; the owner has not requested a full-suite sweep. Preserve failures and rejected experiments.

Keep draft PR #3575 synchronized at coherent checkpoints, including updated recovery notes for unfinished changes. Leave auto-merge off. Do not describe a recovery checkpoint as completion, passing verification, or a speedup.
