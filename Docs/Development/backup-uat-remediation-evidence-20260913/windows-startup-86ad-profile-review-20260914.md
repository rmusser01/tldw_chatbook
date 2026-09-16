# Exact86ad Windows startup diagnostic

The diagnostic step executed, but the candidate did not pass: pinned dev passed in52.87s (38.66s test call), candidate hit the unchanged60s timeout, and the diagnostic shell exited1. A successful/continued workflow-step label must not be reported as successful candidate acceptance. The separate primary Windows run also timed out in all four cases.

## Provenance and observation limits

Existing run34864871343, `/private/tmp/uat-windows-startup-86ad9993dc-full.log`, SHA256339ac4068ae471e43d65547cb0dcbd08ff26b04eed11c158efb2d14afc1a8525. Diagnostic lines2424–2450 use pinned dev4631b60f8dd9623fc55bf16f4a37e29fcb1240c7 and the candidate workspace, with the same fresh-process test_claim_authority_survives_screen_recompose_and_not_window_selection and observer. Dev result is2626; candidate Timeout2691/2923, shell exit1 at line2924. No new execution or repository changes.

Eight complete JSON records were recovered: four dev, four candidate; every source_matches is true and every observer_errors list empty. Extracted data: `/private/tmp/uat-windows-startup-86ad-profiles.json`. Same-field comparison: `/private/tmp/uat-windows-startup-86ad-profile-comparison.json`. The observer is byte-unchanged between5af and86ad. It profiles the enabling/main thread and exports only the20 largest inclusive and20 largest self-time product entries at each sample. Missing functions are unreported, not zero. Periodic snapshots are cumulative and taken while work is active; async lifetimes, live nested entries and observer overhead prevent treating their inclusive totals as additive CPU or clean wall time. In particular current admission_authority35.48s exceeding acquire_storage34.11s is not evidence of35.48s additional delay.

## Comparable observed results

| Measurement | Earlier5af | Current86ad |
|---|---:|---:|
| Dev actual result |1 pass57.10s|1 pass52.87s|
| Candidate actual result |Timeout/exit1|Timeout/exit1|
| Last candidate profiler elapsed |59.922s|60.922s|
| Dev constructor, own log |0.607s|0.581s|
| Candidate constructor, own log |23.634s|22.779s|
| acquire_storage calls / inclusive time |249 /36.740s|260 /34.114s|
| raw._scope calls / inclusive time |331 /23.356s|343 /31.120s|
| bootstrap._registry calls / inclusive time |1275 /21.243s|1331 /18.871s|
| bootstrap._records calls / inclusive time |1327 /9.874s|1389 /25.110s|
| admission_authority calls / inclusive time |249 /10.666s|261 /35.477s|
| Windows open_handle calls / self time |133414 /4.768s|138576 /4.917s|
| Windows security calls / self time |73454 /3.597s|76625 /3.682s|
| Config load calls / inclusive time |75 /11.030s|79 /17.259s|

Current dev sample clocks:13.703/31.156/49.047/56.406s; candidate14.235/28.860/44.828/60.922s. Observer clocks include more than the pytest-reported call; their difference from test totals is not a changed60s ceiling. Current dev import7.591s versus candidate5.117s also cautions against attributing all elapsed difference to imports. Dev passes with its recorded preexisting unverified-Windows-privacy warning; this is a baseline comparison, not justification to revert candidate native protections.

The candidate retains a substantial measured native-admission cost relative to the passing dev baseline. Neither the modest constructor reduction nor the higher call counts establishes improvement or regression caused by the latest patch: these are different live endpoint snapshots of incomplete workloads, not equal completed work. No Windows speedup follows from this table.

## Exact timeout boundary and changed-method coverage

Current candidate timeout stack2871–2922 is initial ChatScreen.compose_content → _build_console_settings_summary_state → active settings readiness → session/default settings → _ensure_console_chat_store → Canvas enabled policy → guarded config load → native raw/storage/bootstrap/private-directory qualification/cleanup. This is still initial Console composition, like the earlier5af instrumented timeout, not evidence of completed GGUF behavior assertions. The current bounded list retains17 _read_canvas_enabled calls/13.845775s inclusive and only0.000094s self time. That identifies the finite caller chain where native guarded work accumulates; it does not prove an individual native call blocks, a lock deadlocks, or all17 calls belong to this one summary invocation.

Neither `_global_context_policy_overrides` nor its parser appears in any bounded list for either revision. Its actual Windows call count is therefore unavailable. The new9→1 lifetime remains supported by the previously reviewed local native per-call proof only. Total260 acquisitions cannot be used to confirm or refute that per-method reduction. The failed stack is instead inside settings readiness/Canvas policy before this summary builder reaches its later context-control work.

No safe additional lifetime or removed check is established here. Existing independent policy reads may be at different callback/effect boundaries; prior native late-record/registry proofs prohibit sharing stale permission observations. The evidence supports retaining the unresolved startup failure, its current exact caller chain and all native guards. It does not support caching, timeout changes, startup-wide scopes or inferring a new optimization from the top20 aggregate alone.
