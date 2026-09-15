# Windows 4244 Settings / Library causal review

Read-only review; no product, test, fixture or keyboard changes and no Linux access. Parent independently verified104 artifact hashes, two installed receipts and2474 Python source matches per receipt. Relevant left_rail.py and storage_admission.py have no diff from tested4244 to current worktree. Counts supplied by parent: native42 pass; product161 =158 pass/3 fail. The third handoff failure is outside this review.

## Settings: navigation failure before backup

JUnit fba1b81bd111b8be.txt identifies two linked exceptions: pilot.press('f4') raises WaitForScreenTimeout while waiting for pending widget messages; during run_test exit Textual raises its stored timer exception, NoMatches in ConsoleLeftRail._sync_progress_count (left_rail2556), querying #console-agent-progress. Only MOUNTED_BEFORE_SETTINGS is printed. No Settings backup inventory, loop-profile, recovery-failure or runtime-settlement file exists. This failure never establishes a backup admission attempt.

The rail on_mount630–632 starts a0.5s progress timer when the callback exists; compose2502 creates the progress button under that same condition; on_unmount2548 stops the parent timer. The callback2552–2562 queries the descendant unconditionally. Therefore an absent descendant during a still-active or queued callback is an established failure. Whether this happened during recompose/navigation or teardown, and whether it caused or followed the initial Pilot wait, is not timestamped. Do not call the initial navigation timeout a backup lease failure or claim the timer alone caused it.

Smallest reproduction: the existing mounted rail fixture in Tests/UI/test_console_left_rail.py already exercises actual rail.recompose (line386). Reproduce the real progress callback/timer across the descendant-removal interval, plus a queued callback during Settings navigation/parent teardown. Verify the callback resumes updating a new live button and preserves refresh-count behavior. Only after reproducing the lifecycle would a narrow missing-target/lifetime correction be justified; no broad exception catch, timer/deadline increase or backup guard change is supported.

## Library: changed SQLite preview, correctly refused

JUnit344f17c907ca7f60.txt ends with backup failed/capturing, issues review_required, review_issues scope_changed. The new capture observer supplies the exact precursor: preview_sqlite_changed from storage_admission._PreviewScope.sqlite_target1530 during capture's rediscovery (capture_service180), following a successful first preview (line77).

That exact branch compares the bytes copied with the expected size and the opened descriptor's final(dev,ino,size,mtime_ns,ctime_ns) with the initial state. It is the post-copy check, not the later pathname recheck and not an atime comparison. The record does not identify main versus WAL, which field/count changed, or which writer caused it. Concurrent write/checkpoint is a plausible mechanism, not a measured actor attribution.

The resulting inventory shows primary Notes included→unavailable, changed dependencies/shared group, two excluded SQLite transient members disappearing, and dependent rows losing their core dependency. This is a concrete causal chain to scope_changed. There is no runtime settlement or admission_timeout event; refusal occurs before capture acquires maintenance. Existing test_thread_diagnostics.py889 already proves a real concurrent-note refusal can create this classification; it does not prove that exact writer caused this run.

Smallest next discrimination, if needed before deciding remediation: retain only a fixed main/WAL role and booleans naming which existing post-copy comparisons failed (count/dev/ino/size/mtime/ctime), plus elapsed copy time, in an approved test observer. Never record paths, field values, contents or unrestricted locals. Pair with a deterministic native concurrent owner write/checkpoint at this same boundary using the existing synthetic fixture. Preserve the refusal and coherent-copy invariant. These logs do not justify accepting an unstable image, suppressing scope_changed, adding automatic retry, or moving preview/capture authority boundaries.

## Admission and loop measurements

| Case | Completed groups | Inclusive group wall total / maximum | Completed tokens | Inclusive token wall total / maximum |
|---|---:|---:|---:|---:|
| Settings |2116/2116,0 errors|8.059s /37.98ms|4520/4520,0 errors|16.827s /823.80ms|
| Library |2283/2283,0 errors|10.291s /339.57ms|4849/4849,0 errors|19.434s /339.53ms|

The last16 detailed group calls in each case all completed without errors, all caller pause_requested702, each with one namespace/root/unique root and max depth14. Settings wall range3.087–13.256ms, thread CPU0 at the recorded Windows granularity, process CPU0–15.625ms. Library wall3.061–14.076ms, thread/process CPU0–15.625ms. Zero quantized CPU is not proof of zero work; process CPU includes other threads. Totals are inclusive, overlap across methods/threads, and cannot be summed into end-to-end savings. Last16 shapes do not establish every earlier topology.

Library's bounded loop sample spans5.085s with4.297s thread CPU. Its two Canvas policy watcher/config-read calls account for3.781s inclusive thread CPU; tray-height adjustment69 calls totals0.234s and dispatch-method368 calls0.047s. The observer dropped257028 calls under its bound, so this is partial attribution, not a complete exclusive profile. It supports significant synchronous config-read work in that window, not an infinite tray/reconcile feedback loop or the SQLite writer's identity. Settings has no loop sample because it failed before the observer's capture-start attachment point.

These4244 failures do not reproduce b560's two EX lease admission timeouts. Short completed groups here cannot explain or retroactively disprove those older failures. The existing lexical-overlap and fresh-second-observation reviews remain applicable: no evidence warrants replacing native reads with caches or weakening fresh identity/authority checks.

Evidence root: /private/tmp/uat-windows-4244-support/backup-platform-windows-2022-py3.12-support-diagnostic-4244ce3e7af4c14018a2c1f615d255c88964767e/test-logs/product-pytest/test_mounted_console_complete_{1,2}/home. JUnit excerpts are in /private/tmp/uat-windows-4244-support/junit-failures/. No new performance/product fix is proven by this review.
