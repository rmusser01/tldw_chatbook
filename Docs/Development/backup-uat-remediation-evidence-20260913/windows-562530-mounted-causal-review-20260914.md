# Windows 562530 mounted failures — causal review

Read-only source/artifact review. No repository/fixture changes, live SQLite, network, tests, remote runs, Linux logs or agents. Parent verified109 artifact hashes, two installed receipts with2474 Python Git blob matches each. Reported results: native42 pass; product165 =161 pass/4 fail/0 skip in2357.521s. Handoff70s timeout is outside the three mounted cases below and predates the separately reviewed file-output/descendant observer correction.

## Console: the preview changed, but the writer is not identified

The first inventory is complete. During capture's second discovery, _PreviewScope.sqlite_target raises preview_sqlite_changed at storage_admission1530. That branch rejects copied-byte-count inequality or final opened-descriptor(dev,ino,size,mtime_ns,ctime_ns) inequality; it is not an atime check. Primary Notes changes included→unavailable, two transient SQLite rows disappear, shared groups change, and seven dependencies lose their primary Notes target. capture_service185 then raises CaptureReviewRequired/scope_changed before maintenance.

This is the same exact mechanism as4244 Library, now in Console. It does not identify main versus WAL, changed comparison field, or a writer/checkpointer. Retained stacks show conversation-list reads, AgentRuns count reads and synchronous config reads, not a recorded Notes write coincident with the copy. Do not assign those readers as the writer. The native concurrent-note regression establishes a possible mechanism, not this run's actor. Scope refusal preserves the stable-copy contract.

## Settings: native lease timeout, different from4244's missing button

F4 and preview completed; both inventories are unchanged and complete. The original terminal exception is AdmissionTimeout: capture_service220 → Admission._admit600 → exclusive lease wait. No missing-button error or runtime-settlement record appears. Capture thread1980 waits for ordinary leases; samples show pause observer7140 in pause_requested702 → _groups364 → _tokens with Windows handle/ACL work. The main thread alternates Textual height/dispatch/reconcile work, then is idle in its final sample.

Unlike4244, the new aggregate timing proves one completed _groups call took88.844s (2209/2209 complete,0 errors, total106.335s). However its individual detail was evicted by the last16 bound: retained calls are later completed21-root pause_requested scans around66–74ms. The outlier's caller, thread CPU and process CPU cannot be recovered or equated with the sampled pause invocation. Thus a long scan is measured, and delayed pause observation is a supported dependency candidate, but the exact held owner and scan timing relative to the60s admission wait remain unproved. Absence of settlement is consistent with no recorded producer settlement, not proof of a particular retained owner or forgotten-close defect.

Its early loop profile spans7.328s/6.281s thread CPU, including three Canvas policy config reads at5.594s inclusive thread CPU. It is a bounded earlier window, not the whole88.844s interval. Nested timings cannot be summed; these records do not prove an endless UI feedback loop.

## Library: owners retired; capture still doing held discovery at420s

Both completed inventories are unchanged and complete; no preview_sqlite_changed or recovery-failure record. Five stage beginnings precede successful cache retirement: live leases8→1, then startup retirement leaves0, with0 pending/ordinary/raw operations. Notes' before-retirement predicate is eligible/current/nontransactional; it does not remain stuck. The observer records startup_retired=True.

The event named runtime_resume is emitted BEFORE awaiting the original resume, not after readmission. Its unrecognized_runtime_issue label can represent None (not in the observer whitelist); it is not proof of an actual runtime error. Startup reacquirer7948 is subsequently in Admission._admit577 waiting for the capture gate. That is downstream exclusion while capture owns maintenance, not a blocker preventing capture from proceeding.

The four final rolling samples show capture4228 progressing through Admission._publication_roots653 (realpath/native root proof), then session._discover_capture_inventory at capture_service221. The final sample is Agents/recovery.py76: connect_private_sqlite('recovery.operations.agent_logs', item.path, read_only=True), through private directory/native security verification. The source function and line distinguish this owner despite the bounded traceback retaining only basename recovery.py. It is the nearest observed unfinished discovery owner, not evidence that its SQL/schema/read is permanently blocked. The main loop is idle in the last two samples. A third completed inventory and archive result are absent when the parent420s subprocess deadline kills the child.

A retained final _admit group check took46.086s wall,0.234s worker-thread CPU and46.281s process CPU. Its own CPU is a tiny portion of the elapsed interval; process CPU includes other threads. This cannot be attributed as46s of pure group algorithm work or46s of native ACL CPU. It supports heavy concurrent process activity/scheduling effects but does not isolate the cause. Unlike4244 Library, this case passed preview and producer retirement and reached held capture discovery.

## Bounded measurement summary

| Case | Groups completed/errors | Inclusive group total / max | Tokens completed/errors | Inclusive token total / max |
|---|---|---|---|---|
| Console |2051/0|8.096s /94.59ms|4345/0|16.465s /271.49ms|
| Settings |2209/0|106.335s /88.844s|5668/0|119.432s /5.380s|
| Library |2323/0|58.709s /46.086s|5363/0|114.310s /2.512s|

These totals overlap across nested methods and threads. Console retains one-root groups; Settings retains21-root groups; Library's last16 include registration growth14→21 roots and the final group. Bounds omit older detailed topology, and Windows CPU quantization can round short calls to0. Compact extracted numeric evidence is /private/tmp/uat-windows-562530-mounted-causal-summary.json.

## One minimal next probe

If one follow-up observation is authorized, retain ONE slowest completed _groups detail in the existing admission observer, alongside its current rolling16, using its existing fixed caller/thread/wall/threadCPU/processCPU/root-count/token-aggregate schema. Settings currently retains a numeric88.844s maximum but discards exactly the caller/CPU detail needed to explain it. This bounded additional record addresses that concrete loss without more product calls, locks, per-file output, unrestricted locals or paths. It may still leave Console writer identity and the final Library owner's duration unresolved; do not expand the probe into multiple new mechanisms or claim all causes become known.

No code fix is justified here: do not skip the second native observation, cache authority, weaken stable-copy checks, increase deadlines, or force-retire owners. The earlier rejected lexical-overlap and observation-reuse proposals remain rejected. These results must remain distinct from4244 and from subsequent product/test corrections.

Evidence: /private/tmp/uat-windows-562530-support/backup-platform-windows-2022-py3.12-support-diagnostic-562530e4b5d7876ecf503b3455a38af264d6cf87/test-logs/product-pytest/test_mounted_console_complete_{0,1,2}/home, plus sibling /private/tmp/uat-windows-562530-support/junit-failures/.
