# Windows 238564 support independent review

**42 native passed;317 product cases yielded313 passes and4 failures, with zero skips/errors/missing outcomes.** Exact run34912765803/job104203707615 tested clean source `238564276d7efc84e0954ace75f4c6e62493db3c`. Product JUnit duration2680.336s; native1.647s. The execution step began2026-09-15 00:21:54UTC. Monitoring used bounded60s backoff without rerun/cancel. No source edits, app tests or Linux access were performed.

All **163 indexed artifact SHA256 values** verify. Two installed receipts each contain2867 files and2475 comparable Python files, all matching the source receipt and independently matching exact Git blobs (2443 CRLF-normalized,32 exact; zero mismatch). Fixture0 wheel `00b52985a2cc44d404278e3baa8d806eeb83bdfae4aa9794fc08e9487d79cd57`; fixture1 `06b58335058a51c4bd22c922a9211d34f366b0c98a52cd6978c44209c3d57812`. These are verified available receipts, not a fresh remote installed-directory census. Unselected workflow jobs are not pytest skips.

## Required case accounting

All12 native SQLite-inside-config cases passed: cached, reopened, getter, pause_before, pause_during, error, cancel, error_selector, cancel_selector, selector_after_open, failed_close, custom (4.846–6.755s each). All12 persona-observation-budget cases passed. The latter are deterministic mocked-clock/child harness tests:6 platform/headroom projections,2 exhausted-budget refusals,3 retained completion-marker requirements,1 original timeout preservation. Their Linux-labeled cases executed on Windows as unit tests; they do not constitute an actual Linux Persona lifecycle run.

| Actual installed case | Result | Duration |
|---|---|---:|
| Console mounted capture/resumed writes | PASS |283.716s|
| Settings mounted capture/resumed writes | FAIL: admission_timeout |377.946s|
| Library→Settings mounted capture/resumed writes | FAIL: outer subprocess420s timeout |420.300s|
| Fresh Recovery UI handoff | FAIL: parent subprocess70s timeout |111.873s including fixture/test overhead|
| Fresh Recovery UI later mode | FAIL: restart-process observation timeout |70.008s|

No other failures occurred. Full exact node/timing accounting is preserved in case-accounting.json and bounded-summary.json.

## Console actual success, historical snapshot failure preserved

The installed body completes boot readiness, real note save, Complete preview, successful coherent archive creation, native maintenance release, fresh resumed note write/live readback, and archive assertions (pre-capture note present, post-capture note absent), normal teardown and no blocked network attempts. Five inventories show no deltas/unavailable dependencies. Runtime cache retirement reduces8 leases to the startup lease, then startup retirement completes and0 leases remain. No snapshot-refusal event is recorded.

This pass does not identify the prior bd6/15c WAL writer, prove that prior races were fixed, or authorize weaker opened-state checks. It is a current successful run of the same actual contract. Slowest completed group671.2871ms/20roots.

## Settings: failed Scheduler drain remains exact

Settings reached F4 and Complete preview. Its two inventories are unchanged. Native capture failed at `capture_service.capture:220 → admission._admit:600 → _lock:173 → _check:188`, original **AdmissionTimeout**, followed by the actual embedded assertion at line83.

The runtime observer identifies **SchedulerLoop._maintenance_drain**,29.954s, then `RecoveryRequired(runtime_work_not_settled)` at `_settle_stage:54`; startup_retired=false and caches never reach retirement. Initial state had10 leases/one startup and zero pending/core/raw operations. At failure the storage snapshot is **unavailable**, which must not be read as zero. Active observations show trace thread8120 initializing_enter27.688s and heartbeat7536 scope0.094s.

The four snapshots identify progressing real roles: heartbeat7536 goes from initializer wait through native authority to post-acquisition scope; trace8120 is `LegacyTraceMaintenance.run_batch:753`, moving from repository admission into the actual Notes SQLite-handle reopen, then another initializer wait; DB-size3140 waits in configoperation343 while resolving the **Prompts** path. This differs from bd6 Library's current_graph_epoch/Notes-size ordering. There is no fixed reverse lock cycle or exact permanent owner leak proved by these snapshots.

Slowest completed group: `_admit:541`,21roots,86.7255436s wall versus0.296875s own-thread CPU and87.203125s process CPU. Inclusive token cost is not additive. This shows prolonged wall delay while other process work occurs; it does not isolate its scheduling/OS cause.

## Library: producer settlement succeeded; later observation expired

Library loaded, F4 completed, and both initial inventories are unchanged. Unlike Settings, **all observed producer stages succeed**, caches retire8→1, startup_retired=true, and runtime resume records0 leases/pending/operations. There is no recorded failed hook, recovery exception or body failure before parent420s termination; service_close was not reached in retained entry phases.

Capture thread2268 progresses across the four final snapshots: native `_admit:653` publication-root proof; then `_discover_capture_inventory` at capture_service221, config source/registry proof; then native readonly Workspaces validation (`recovery_operations.py:148`, `_WorkspacesAdapter.validate`) opening its private SQLite artifact. Thus it passed the earlier exclusive lease acquisition and reached under-maintenance discovery. The terminal parent timeout is **not** evidence of Scheduler drain failure or another completed admission_timeout result. Native checks remain in progress, and the full snapshot validation/capture/resumed-write assertions are unexecuted or unproved.

Its slowest completed group `_admit:617` is48.5852155s wall/1.0625s own CPU/49.296875s process CPU for21roots. The bounded log does not assign the remaining elapsed interval to one owner or establish the reason work did not finish. No timeout/guard change is proposed.

## Restart failures: different observation boundaries

`test_actual_handoff_execs_fresh_recovery_ui[handoff]` fails in the original `subprocess.run(...timeout=70)` before `wait_for_restart`; the111.873s JUnit duration includes fixture/test overhead and is not a changed child timeout. File output is used, so the former inherited-pipe-EOF cause cannot be assumed. Its log has no fresh completion marker; ordinary startup was still logged, but no precise retained child stack isolates the outstanding work. No unrelated Buddy warning is assigned causality.

`[later]` reaches the existing restart process observer with only0.813s left in the shared70s window. It raises `TimeoutError(restart_process_timeout)` after psutil wait, preserving the native timeout chain and exact-process cleanup. The retained output contains **retired and reopened**, emitted after the fresh installed Recovery UI's mode/empty-copy-list/no-prior-plan assertions, service-close, package-origin, config-preservation and no-network assertions. That proves the fresh assertion body reached its marker; it does **not** prove normal process exit before the deadline. The final teardown/exit cause remains unidentified. This is not permission to count the test as a pass or extend its limit.

## Provenance and next-scope limits

This run predates the newer05e DB-status/TTS correction and cannot validate it. The separate05e focused evidence and Linux acceptance remain independent. Relative to bd6, Console now passes, Settings now fails, Library reaches later native validation but times out, and two restart observations fail; changing outcomes do not by themselves prove new defects or fixes. Preserve all four failures and their exact limits. No new product patch follows from this artifact review alone.

Evidence root `/private/tmp/uat-windows-238564-support/backup-platform-windows-2022-py3.12-support-diagnostic-238564276d7efc84e0954ace75f4c6e62493db3c`; verifier outputs `/private/tmp/uat-windows-238564-support-verify.log` and `...-blobs.log`; JSON receipts in `/private/tmp/uat-windows-238564-support/{verification,git-blob-verification,case-accounting,bounded-summary}.json`.

Full preserved JUnit failures in `/private/tmp/uat-windows-238564-support/junit-failures/`: `d061c3e25d6c54c2.txt` handoff, `f339fa553c45a2ce.txt` later, `fba1b81bd111b8be.txt` Settings, `344f17c907ca7f60.txt` Library. Bounded per-route logs reside under artifact `test-logs/product-pytest/test_mounted_console_complete_{0,1,2}/home`; restart-output.log files under `test_actual_handoff_execs_fres{0,1}`.
