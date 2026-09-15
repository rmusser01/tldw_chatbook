# Exact2031d Windows native-close audit

Read-only review of run34940059565, exact2031d0c5261c2b4781aba40752f228da9d19b4eb. No apps/tests, remote operations, or repository edits.

**59native tests pass;58product cases finish with55passes/3failures, no skips/errors/missing outcomes.** All33 indexed artifact SHA256 entries match. The exact clean/private-tracked-head source receipt covers16,696files; eight relevant test/driver/helper digests independently match Git. Installed receipt contains2,867files and2,475package Python files matching source/exact Git (2,443CRLF/32exact), zero mismatches. WheelSHA256d6d10449452cf1a7ab0581e3025cdbdfa5659a259655fa87aef3ce4bc6eb4e75. This verifies retained receipts, not a remote live installation.

| Case | Outcome | JUnit seconds |
|---|---|---:|
|Native guidance/config-save coherence|TimeoutExpired, existing45s child bound|50.295|
|Native display pairs/fresh later send guard|TimeoutExpired, existing45s child bound|45.046|
|Bounded section covered-native|Assertion at child94|7.778|
|Installed Settings capture/resumed writes|Passed|173.297|
|Installed Library capture/resumed writes|Passed|136.302|

The first guidance JUnit duration includes setup/timeout processing; it is not a50s configured limit. Both **actual separate** guidance-phases.log files contain import_begin, import_complete, construct_begin, construct_complete, mounted. Neither contains assertions_complete or shutdown_complete. Unlike a523's embedded command strings, these are positive emitted phase receipts: both app contexts entered and the failure was not solely waiting for initial mount. The remaining interval includes screen/store setup and the assertion sequence; without intermediate receipts the exact operation/count/output failure is unlocalized. Timeout kills and no assertion traceback do not prove the assertions passed.

Covered-native terminates at real child line94: `assert not section._reconcile_scheduled`. Earlier lines88–93 pass: expected viewport height,section height6,scroll_y4,hint displayed,viewportfocusable,and actual changed-demand allocation request. Thus restored geometry/focus assertions succeed but the test's immediate no-pending-reconcile assertion fails. The subsequent .04s quietness check is not reached. This evidence alone cannot determine whether a benign queued final pass or a persistent rescheduling defect remains; parent owns that bounded diagnosis. Seven other bounded-section parameter cases pass; do not waive the failed native case.

## Two actual installed workflows complete

Settings constructs25.687s/mountyield49.453s and service_close begins/completes124.015s. Library constructs24.906s/mountyield47.313s and service_close begins/completes131.625s. Each has five completed inventory observations: unchanged scope, no unavailable dependencies. Each runtime sequence has five settlement starts, cache retirement leases8→1 and resume0, withpending/operations/raw_operations0. Both execute unchanged assertions for complete capture, coherent archived pre-capture note and excluded post-capture note, ordinary resumed native note writes/readback, and Canvas enabled before capture/after resume. Neither records an operation failure. This is current positive installed qualification, separate from prior a523 Library420s failure.

Loop instrumentation is limited: Settings sample5.956661s/1.875threadCPU with202,369dropped calls; Library5.044650s/.609375CPU with21,422dropped calls. The cooperative stop can exceed its nominal5s window and retained call lists are incomplete. These windows are not whole-lifecycle profiles, nor proof of global absence of repeated layout callbacks. Successful actual workflows establish acceptance for those cases; do not attribute all timing differences to one patch.

Startup run34939974091 still has the separately documented four60s UI timeouts. This native-close result is partial success, not whole-PR qualification. Exact counts, case identities, emitted phases, source/Git checks, and artifact-index hash are in `/private/tmp/uat-2031d-native-independent-summary.json`.
