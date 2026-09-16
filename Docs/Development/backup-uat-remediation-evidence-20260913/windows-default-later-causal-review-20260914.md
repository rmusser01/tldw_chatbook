# Windows 081 default later workflow: bounded causal review

Read-only review of test_later_rollback_credential_ui.py and the supplied Windows 081 artifact. The child driver has no diff from 081 to current HEAD. Both later-child.log and later-parent-failure.json.log match their artifact SHA256 entries. No source, fixture, SQLite, keyboard session or Linux logs were changed/accessed beyond the authorized source/artifact reads.

**Important correction:** the ninth/final child-log line is `later_rollback_complete`, reporting `state=succeeded` and `restoration_validated=True`. Thus the second actual rollback reached successful validation; this is not evidence that it stalled after changed-acknowledgement review. The parent recorded the original subprocess TimeoutExpired. No child exception or thread snapshots identify the remaining wait.

After the final checkpoint (driver line202), the remaining sequence is:

- recovery_copies via asyncio.to_thread (line204), with no separate local timeout; actual retained-copy enumeration and assertions that both copies are verified;
- wait for the new copy's UI button (90 seconds on Windows), select it, and assert review acknowledgements/reset state;
- start_copy_inspection and service.wait (180 seconds on Windows), assert success/archive_verified, then verify_sealed credential policy;
- exit app.run_test and asyncio.run. RecoveryApp.on_unmount (launcher144–145) awaits service.close; service.close (recovery_service1301 onward) cancels unfinished work then waits for its executor. This is an existing possible wait boundary, not a demonstrated cause.

The final checkpoint is emitted before these remaining assertions. It does not prove enumeration, sealed-copy readback, UI teardown or executor shutdown completed. An exception during those steps can also enter asynchronous cleanup before the outer child exception handler runs, so the missing failure log cannot exclude an earlier pending exception.

## Smallest next diagnostic

Keep all test/product deadlines and assertions. In this existing child only, reuse observe_threads with a unique `later-threads.log` prefix and interval10, plus observe_admission(..., native_calls=False) using `later-admission.log` and its existing interval5. Start them after the existing installed-origin/import assertions and before main executes; keep them active through run_test teardown and asyncio.run/default-executor shutdown. Register each acquired observer immediately within an ExitStack as `stop_observer(stop)` so later setup failure cleans up earlier observers, reverse cleanup still runs, and any underlying exception/cancellation remains primary. Preserve existing record_failure behavior. No wrapper around product callbacks or native guards is required.

Add only fixed stage labels and elapsed monotonic seconds to flushed child output: child start, copies enumeration start/end, new-copy UI selection start/end, inspection start/end, sealed verification end, app-context exit start/end and asyncio.run completion. Existing checkpoint output can include the same monotonic elapsed value; do not print operation IDs, passwords, paths, payloads or exception text in new metadata. A fixed append-only *.log is also collector-compatible. The existing JSON/JSONL checkpoint files are not collected by run_platform_product._collect_safe_logs (line702 selects *.log), explaining why their stored wall timestamps are unavailable here.

Threads observer retains four rolling samples, each bounded to32 threads/64 code frames, without source lines or locals; admission records provide bounded numeric topology/timing and original calls. Periodic files survive the parent killing the child, whereas finally-only output cannot. Reuse unique prefixes to avoid the fatal observer's O_EXCL collision. A killed writer can leave the latest periodic JSON partial; flushed fixed phase lines remain useful independently.

## Budget interpretation and acceptance

The observed parent result establishes the child exceeded its existing900-second Windows subprocess bound. It does not measure the individual steps: these collected checkpoint lines have no timestamps. The enclosing test's2400-second cap includes the separately earned replacement and the child; no evidence here shows that outer cap fired first. Local90/180/360-second waits are limits, not measured durations, and summing their maxima cannot establish either expected runtime or which step consumed the900 seconds. Do not increase them on this evidence.

This probe should identify whether the final interval is ordinary bounded progress, native copy admission/enumeration, inspection, a UI wait, or shutdown/exception cleanup. No product correction is justified until that distinction is recorded.
