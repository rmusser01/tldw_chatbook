# TASK-38614 scoped fix-round review

**Spec compliance: Approved for the reviewed fix. Quality: Approved.** I1 is resolved. New findings: **0 Critical / 0 Important / 0 Minor / 0 new CannotVerify**. CV1 and CV2 remain two explicit historical evidence qualifications, with the dispositions below; neither is an unresolved product finding.

Scope: FIXBASE `5f03c716fe4b1356678a61d578c39d687b1a1fe0` to HEAD `98e80353153e9042c8e0030da9755a55c003806b`, exactly two paths. Read the complete initial review, the full report appendix after its unchanged 9786-byte prefix, and the complete 6934-byte fix package, SHA256 `e7f4bc81b8e253b263a5a23572df220231cd1f26ee3c7ad24a26d137ae96f20b`. This is the requested I1/fix/evidence follow-up, not a second whole-fix audit.

## I1 disposition — resolved

- `tldw_chatbook/UI/Console_Modules/console_spend_projection.py:1074` now passes `self.refresh` directly to the existing scheduler. The production factory binds this to the async native screen method and `screen.run_worker` (`:907`–`:908`), so no native coroutine exists before Textual enters the worker. The surrounding currentness, successful-publication and changed-value conditions, nonexclusive mode, publication group and error behavior are unchanged. The nearby context-read scheduling is unchanged. No adapter or new owner is needed for the actual production callback contract.
- The new parametrized test, `Tests/UI/test_console_sync_worker_admission.py:132`, uses the real mounted screen, controller/store and `ConsoleContextReadSnapshot.for_screen`. It warms real context, changes the actual context key with a user message, and calls the original `_refresh` body. Its profiler observes the original scheduler's return and cancels that actual Worker before entry; it does not replace the scheduler. Assertions distinguish cancelled worker non-entry from admitted original-code entry, Worker success and visible new-session tab publication. RuntimeWarning capture and GC remain inside the observation scope and after host cleanup.
- The RED receipt genuinely reaches that warning assertion: `task-1-round1-alias-red.log:85` records `ChatScreen._sync_native_console_chat_ui was never awaited`; the cancelled case fails while the admitted case passes (`:173`, 1 failed / 1 passed in 8.26s). `task-1-round1-alias-green.log:7` records both passing in 7.37s after the one-line correction. The final unchanged-source control run also passes both cases.

## New-change quality checks

- The older regression's idle predicate now uses `getattr(..., False)` for the optional control-bar replay field (`Tests/UI/test_console_sync_worker_admission.py:103`). This matches production's established absent-field meaning; it writes no guard, changes no deadline and removes no behavior assertion. The retained covering failure is specifically the previous direct attribute lookup (`task-1-round1-covering.log:107`), not a refresh failure. The final run includes the amended test and its live-error-propagation control.
- Reviewed the factory/callback reachability already established in I1 and the affected test callback providers. Production construction supplies an async method. The existing shared-context test helpers also supply async refresh callbacks; their coroutine-only scheduler fakes are not used as proof of this publication boundary. The new real-Textual negative and positive cases provide that proof.
- Read `task-1-round1-alias-audit.json` as the writer's follow-up scope record, not as an independent replacement for the previous source review. Its alias classification is consistent with the paths examined during that review. No new broader scheduling rewrite is present in the two-path diff.
- `task-1-round1-static-final.json` records zero introduced per-file Ruff findings and zero formatter/edit overlaps for the overall 11 changed paths, with protected changes empty and the staged baseline preserved. Existing lint/format debt is not claimed clean. Root separately verifies current commit/index/worktree pins and the original 279 identities/staged61 in `root-round1-delivery-audit.json`.

## Evidence dispositions

**CV1 — historical uncertainty retained; current-source evidence now sufficient.** The appendix explicitly retracts the unproved claim that the old 14-pass invocation followed the wiring formatting correction. That older run's launch epoch remains CannotVerify; it is not retroactively assigned final hashes. For the new runs, I read `task-1-round1-run.py`: it writes the `.before.json` source/argv record before launching the command, then records exit, elapsed time and post-run hashes. I independently compared the final-controls and final-public before files, receipt before/after maps, and current hashes for all 13 recorded paths; all agree. Their actual raw logs record:

- `task-1-round1-final-controls.log:9`: **4 passed in 22.32s**, exit 0; all three worker/alias cases plus the mounted workflow. Command elapsed 25.09s.
- `task-1-round1-final-public.log:6`: **1 passed in 6.84s**, exit 0; original public workflow. Command elapsed 8.37s.

Exact command arrays include strict parent RuntimeWarning/PytestUnraisableExceptionWarning settings; the workflow retains its previously reviewed child-side RuntimeWarning capture/GC/assertion. These are fresh, source-pinned scoped results, not a substitute claim that the whole old 14-case invocation ran on these bytes.

**CV2 — unchanged historical qualification, no blanket exoneration.** The earlier partial controller 55F/157P, separate native 7F/12P and bootstrap 1F/6P invocations remain adverse and separately qualified. The appendix does not call them green or claim historical baseline passes. Round1 also retains its own **1 failed / 11 passed** covering invocation (`task-1-round1-covering.log:201`) and explains the optional-field test observation correction. Its product hashes match final bytes; only that later test predicate differs. The fresh final controls verify the amended test, while the recorded covering invocation remains a failure. No full-suite conclusion follows.

## Assessment

**I1 closed; the complete warning fix's independent code gate is approved based on the original review plus this scoped correction.** No new blocking or nonblocking code finding was found in the fix round. Root-owned task/documentation closeout remains outside this review.

No tests or Git commands were rerun; only source, logs and receipts were read, with read-only hash comparisons. This report is the only reviewer write.

Skipped: another whole-fix review, successful test repeats, full suite and provider/network runs. Remaining limits: old14 launch-time identity and historical adverse cohorts retain CV1/CV2; no broader clean-suite claim is supported.
