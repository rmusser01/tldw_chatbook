# Shared served Console and mobile qualification

Task reviews and whole-feature review, including their scoped corrections, are
complete. This packet separates source/unit evidence, actual served-browser
evidence, and physical phone/VPN qualification. The final code review covers
the recorded source; physical hardware and VPN checks remain open.

## Actual served-browser evidence

The fixture starts the production TLS server and actual retained `TldwCli`
AppService child against one contained canonical profile. A controlled provider
replaces external network generation; native admission, prompt acceptance,
execution ownership, persistence, projections and command handling remain active.
The profile is selected before genuine public NoData cutover and app startup.

- Fresh unseeded profile: the terminal prompt entered the provider, terminal close
  and phone reload retained PID/session/incarnation/run, an explicit phone Send
  queued and ran once, terminal reattachment retained the child, and exactly four
  expected user/assistant rows persisted. The optional tokenizer directory stayed
  absent. Raw epoch: `fresh-twelfth-r159.log`; fixture `task11-served-xayehjr3`.
- Interaction epoch: actual terminal mouse focus, multiline paste and resize
  retained the original input Thread object. Canvas disable retained Console
  identity and draft. Unauthenticated, missing-CSRF and wrong-origin commands
  returned 401. Raw epoch: `interaction-lifecycle-second.log`; successful fixture
  `task11-served-019xgrz5`. This is one passing browser case; that same command's
  separate lifecycle case failed and is not counted as passing.
- Historical cold phone command failure, repaired by R160 below: Create before full mount, and immediate
  Send after post-mount Create, can return 409 `preview_sqlite_changed`. The native
  cutover observer attempts stable byte copies of live SQLite/WAL files while
  the actual app writes them. `cold-gate-trace.log` and
  `cold-cutover-origin.json` preserve the exact pass-through traceback and source
  attribution. No automatic retry or admission bypass is used. Earlier retained
  journey success applies only to its witnessed timing.

Raw logs and source epochs are retained under
`.superpowers/sdd/2026-10-25-shared-console-mobile-supervision-implementation/task-11-evidence/`.
Fixture artifacts remain in private OS temporary directories named above; the
local final manifest supplies absolute paths and hashes. Counts from different
source epochs are not summed into a release pass count.

## Automated source and component checks

R156 qualifies existing module-owned SQLite openers without a new role or broad
inventory waiver. R157 routes both Notes boot probes through the existing
read-only, must-exist native store. R158 selects all server storage paths from
one authoritative atomic config snapshot. R159 avoids constructing a tokenizer
manager when native pinned observation proves default mappings absent.

R160 uses native live read-only transactions for ordinary completion checks while
keeping preview/maintenance copies strict. The deterministic WAL regression first
failed (`r160-wal-before.log`), then passed with a witnessed native writer inside
the read transaction. The focused accounting/provenance epoch passed 12 tests in
64.96 s. A subsequent connected epoch passed three close/cancellation cases and
failed the restored adapter's missing digest workspace; that failure is retained.
After adding the same prepaid native workspace, the connected fourth epoch
passed three tests in 194.93 s, including genuine authenticated restoration and
unchanged captured-SQLite source/schema controls. All 12 source hashes match
before/after that epoch. Structural negative controls separately passed 10 tests.
These counts describe distinct epochs, not one aggregate pass count.

The original inventory epoch had a source-unchanged Dreams capture census mismatch.
FIX1/R165 adds the exact already documented B40 Dreams call to the test census;
the focused inventory file then passed all76 tests, including its negative controls.
Production roles and checker policy were unchanged. Three
model-limit tests expect 4096 while unchanged production code explicitly defaults
to 32000; the test, token counter and model-capability sources match the original
feature base. These are source classifications, not claims that the original
base was separately executed. Final scoped counts and dispositions are recorded below.

R161 removes only the two unused float timestamp fields from live worker
projections. Native fleet clocks and the strict control protocol are unchanged.
A real control-message regression first failed on `invalid_payload_value`; the
repaired strict-wire and actual two-chat/two-worker epoch passed two tests in
61.48 s. That journey witnessed guidance consumption, exact worker/main controls,
a supported question and unchanged desktop selection/draft. Its attempted
`fs_list` was unadvertised and refused, so that epoch does **not** establish
successful local-tool execution or approval. The stronger admitted-tool journey
is tracked separately.

## Cold phone and process ownership

The R160 corrected cold/lifecycle epoch passed five tests in 271.62 s with all
15 source hashes unchanged. A fresh unseeded profile opened authenticated `/m`
without attaching a terminal, clicked the shipped New chat control, then sent
immediately when the shipped composer enabled. Exactly one Create and one Send
were submitted; the original Send receipt became accepted, one provider call
produced the exact reply, two message rows persisted, and the child exited 0.
No private mount/readiness signal or command retry was used. The preceding
200-only HTTP oracle failure is retained: native pending Send correctly returns
202 and the original receipt is queried.

Distinct fresh-profile process cases recorded clean shutdown (child 71096, exit
0), control-channel EOF (71778, exit 0), and deliberate child crash (72498, exit
-9). For parent death, only the owned standalone server was killed. Child 73438
remained alive with the execution fence: replacement acquisition refused, then
succeeded after that child was physically absent. This is physical absence
proof, not a collected child exit code. Raw results and source bindings are in
`r160-cold-lifecycle-second*`; private fixture roots are `task11-served-n9l7appu`,
`task11-served-oap0jslw`, `task11-served-hr9finst`, `task11-served-jfrhrdig`, and
`task11-served-5b2wpr5m` respectively.

The replacement ordering extension passed separately in the fourth connected
epoch (the other two browser cases in that command failed in their fixture
drivers). With child 87969 and its creation time still live, a preloaded owned
process invoked `ConsoleRuntime.ensure_agent_bridge`. The actual execution-owner
acquisition refused before run-store construction, orphan reconciliation, store
factory or provider factory; all four counters remained zero. Native read-only
run and receipt rows had identical before/after hashes, and the original gateway
count remained one. The child subsequently retired and the fence could be
acquired. Evidence: `task11-served-360_7urc/replacement-result.json` and
`connected-fleet-transport-parent-fourth*`; all 21 source bindings matched.

## Physical phone and VPN

**UNVERIFIED.** No physical iPhone/Android device or VPN reconnect was available to
this fixture. Chromium's narrow viewport is browser emulation. Keyboard reach,
VPN interruption/rejoin and worker targeting on a physical phone are not claimed.

## Later connected browser epochs

`fleet-transport-fifth` passed two cases in 118.73 s with all 21 bindings unchanged.
The two-chat/two-worker journey approved exactly one native `character_search`
round and parsed its successful nonempty result. A separate authenticated phone
context saw the same supported question and disconnected before the original
phone answered. Worker guidance reached the next provider entry; exact worker
and main controls retained the desktop selection, draft and input Thread.
The transport case interrupted actual browser POSTs before admission, after
acceptance and after effect. Original-ID queries returned 404 only for the
unadmitted command and accepted receipts for the other two. Reload did not replay;
exactly two provider calls and four message rows persisted. Earlier duplicate
fixture clicks and a Playwright handler signature error remain failed epochs.

`security-sixth` passed the mobile-off case and failed the enabled case because
the fixture F12 key collided with the incumbent Canvas action. After moving the
fixture-only native review trigger to F9, `security-seventh` passed in 62.45 s
with all 24 bindings unchanged. It witnessed stale epoch 409, native clock ticket
expiry 410 without an accepted receipt, old terminal send/resize refusal after
takeover, retained unsupported review with no supported SubmitDecision control,
bounded projection overflow before phone connection and while an actual socket
was left unread, convergence after phone reconnect, revocation 401 and absolute
session expiry 401. The unsupported review is explicitly produced through the
native runtime seam; it does not establish a natural project-folder trigger.
The unread socket is not an OS backpressure or memory-throughput benchmark.
`proxy-first` separately passed the cold CTA through the trusted TLS proxy.
These are source-bound epochs, not an aggregate release count.

## Final source-bound qualification

The full new browser file first completed **10 passed, 4 failed in 733.44 s**
with 50 unchanged bindings. Its failures exposed the cold-selection race and
three fixture/interaction failures; this is not an all-green release epoch.
R162 repaired explicit Create/Open selection: a page-memory intent follows only
its original command receipt and an exact same-owner live row after refresh.
Navigation, a newer action or owner replacement cancels it. Normal target
confirmation still gates the composer; no mutation is replayed.

The portable client file then passed **60 tests in 61.82 s**. Its unrelated late
receipt branch was not delivered in that epoch. A separate **12 passed in8.78s**
both-engine run attempted a response-body witness, but independent review found
that its delivery event was never awaited. That run does not prove the unrelated
receipt arrived; this remains deferred Minor T11-R05. The affected real-browser run recorded **4 passed,
1 failed in 318.07 s**: direct/proxy cold Create→Send, two chats/two workers with
rendered exact guidance, and enabled security/canonical Library/inert text passed.
Its transport case failed before a third Create; without a witnessed click/request,
that historical cause remains **UNCLASSIFIED**. A subsequent pass-through
pointer/click/HTTP-observed transport run passed **1 test in 57.57 s**, with all
three loss boundaries, original queries, two provider calls, four persisted rows
and child exit0. That later pass does not retrospectively explain the failure.
All these R162 epochs retained 51 identical before/after bindings each.

The native/control batch recorded **324 passed, 6 failed in 334.93 s**, with
60 unchanged bindings. Five failures were test fixtures: two obsolete reset-UI
variants, an unwired test runtime app and two setup-rollback nodes whose config
selector changed after binding. The remaining failure is the unchanged Dreams
capture census mismatch. Test-only R163/R164 corrections retained real API,
projection, private-profile guards and rollback assertions. The first corrected
run was **6 passed, 1 failed in 4.60 s** (a test property-setter mistake); its
single corrected receipt rerun passed in **0.54 s**. Final reset controls passed
in Chromium and WebKit with initially enabled approvals, immediate revocation
while metadata was withheld/unavailable, zero decision tickets/effects and
recovery only from current metadata.

The connected migration batch passed **28 tests in 494.95 s**, all 57 bindings
unchanged: genuine public original/authenticated-restored completion, interrupted
import Finish/Rollback and re-import, populated conflicts, cross-config selection,
unknown-schema refusal, physical source claims, imported checkpoint discard and
native automatic-authority holds. The interruption cases inject an exception and
restart the service. Separate existing physical-process cases actually exit73
at staged/unwitnessed creation, then verify startup quarantine and witnessed-inode
recovery versus unsettled refusal. Those two cases plus the four final reset
controls passed **6 tests in 24.66 s**, all 54 bindings unchanged.

Final scoped static checks cover 26 changed/new Python files: no introduced Ruff
or formatter findings, no native Python3.12 AST warnings, JavaScript syntax and
`git diff --check` pass. Whole-file Ruff/formatter commands still return1 for
106 lint findings and 193 formatting changes outside this task's changed source;
all 106 lint exclusions match original Task11 BASE source under the same rules.
No rule suppression or broad cleanup was applied. Eight earlier scoped formatting
changes have exact native AST-equivalence mappings to the recorded browser bytes;
connected checks ran after formatting. Counts from distinct epochs are never summed.

The local handoff is
`.superpowers/sdd/2026-10-25-shared-console-mobile-supervision-implementation/task-11-implementer-report.md`
and `task-11-final-manifest.json` in that directory, with exact commands, source
hashes, raw failures, private fixture paths and process exits. These local evidence
artifacts are not packaged runtime data. Task11 original review and scoped FIX1
rereview are complete; final whole-feature review and scoped correction are complete.
Physical phone/VPN qualification remains **UNVERIFIED**;
the full repository test suite was not run.


## FIX1 qualification and native warning diagnosis

The original positive retention log (`task11-served-7c34hc1e`) includes an Evals
configuration `storage_scope_not_enrolled` error. The original positive fleet log
(`task11-served-atr_cfcv`) includes that error, two native run-log disable warnings
with `ConsoleScratchSpaceUnavailable`, and a workerB PreToolUse engine error
that the native policy converts to a fail-closed denial. Successful
`character_search` does not establish private scratch/log authority, and the
older unadvertised `fs_list` refusal proves no successful filesystem execution.
Evals falls back to defaults and records its persistence error; exact feature-BASE
failure origin was not executed or established. T11-R06 remains deferred, and
no fixture directory was seeded to hide it.

The same original fleet sequence was observed with finite Python3.12 local code
monitoring in `r03-local-websocket-witness` (1 passed in72.39s;37 unchanged source
bindings; fixture `task11-served-nqbnd_60`, physical child exit0). Native methods,
callbacks, admission and deadlines were unchanged; own monitoring cleanup was
verified. Twenty-three native log append calls returned successful record numbers
before disposal. The runtime then closed hooks and tombstoned scratch while
work was still unwinding; subsequent unrelated-chat/workerB log writes became
inactive and workerB PreToolUse was denied by the closed hook engine. The saved
439-event timeline identifies runtime disposal as the ordering boundary in this
current reproduction. It does not manufacture a missing timeline for the earlier
positive epoch or prove an independently executed baseline origin.

That original sequence's accepted Stop/correction receipts were insufficient
proof of exact effects or worker retirement. FIX1 extends the same journey to
observe those boundaries before whole-app shutdown. The final affected actual
browser epoch passed **4 tests in 248.70 s**, with all 37 source bindings unchanged:
exact fleet effects, cold Create→immediate Send directly and through the trusted
TLS proxy, and all three original-ID transport-loss boundaries. Each owned child
physically exited0. The fleet fixture `task11-served-pk18t9je` (child95004) records
27 successful native log writes, 10 open/nonblocked hook outcomes, and all four
original run returns before app disposal. WorkerB Stop retires its actual Thread
while workerA and chatA remain live; both exact main corrections reach the next
provider phase before main Stop. WorkerA then finishes with its canonical durable
result and dead original Thread; chatA produces exactly its original prompt/reply
once. No scratch/run-log disable or closed-hook warning occurs in this epoch.
The Evals startup error remains disclosed above. Wrapper/global-profile diagnostic attempts are
retained as instrumentation-invalid or timing-perturbed, not production RED or
passing qualification. Physical phone/VPN qualification remains UNVERIFIED.


R01 now distinguishes list requests begun before registration of the exact
owner-bound host intent, including a list started during host bootstrap. Such a
list cannot cancel the newer intent; a genuinely later owner replacement still
cancels it. R166 prevents identical answerability advertisements on the same
WebSocket from creating a stale-round rejection/reset loop. The first false
advertisement still revokes controls immediately; current metadata, a new socket
or different targets advertise afresh. Native rejection and confirmation remain
unchanged. The full existing portable browser file passed **66 tests in70.67s**
on final client/test bytes, across Chromium and WebKit, with37 unchanged bindings.
The earlier connected **26-test** epoch includes the existing withheld-metadata
reset controls. The exact documented B40 Dreams census correction passed all
**76 inventory tests in47.14s**, including strict negative controls, on final
formatted bytes with38 unchanged bindings. The enclosing Counter block has a
recorded native Python3.12 AST-equivalence proof; unrelated formatting stays
untouched. Its earlier76-test result within a90-test epoch remains historical. These distinct epochs are not summed.

Three actual-cover attempts remain recorded: the first incorrectly expected a
completed worker in a live-only list; the second used a pre-Send conversation ID
of None for the durable worker endpoint; the third passed with the actual
persisted conversation identity. Neither failure is classified as a production
regression. The older missing-third-Create transport failure remains unclassified;
the current transport case witnesses each Create ticket/request/applied receipt
and current native row before the corresponding Send. T11-R05's unawaited
unrelated-receipt observer and T11-R06's Evals persistence error remain deferred
minor findings. FIX1 evidence is appended to the local implementation report and
recorded in `task-11-fix1-final-manifest.json`. Scoped rereview approved all four
Important findings with no new findings at commit `2356c7ae94c1`; final whole-feature
review and its scoped correction are complete. Physical phone/VPN remains **UNVERIFIED**.


## Final whole-feature correction

Unresolved command outcomes now affect their original owner/chat/incarnation and
applicable run, worker or decision round. They remain visible with their original
identity for review and continue to query the original ID. A missing response and
missing receipt retain the original action/target; neither grants permission to
repeat it. Independently confirmed unrelated work remains controllable. Exact
current Stop stays available under the existing native idempotent contract;
disconnect, reset, stale targets and unadmitted state still revoke controls.
Create/Open also recheck uncertainty against the actual host returned by bootstrap.

Predecessor receipts are explicitly historical, including reused worker handles
and main-run receipts viewed beneath a worker. The late unrelated-receipt negative
now awaits both successful response delivery and rendered client processing before
checking selection/draft. These close the current WF-M01/WF-M02 (earlier T10-F05/
T11-R05) gaps without changing the historical qualification of their older epochs.

Final whole-feature cover passes90 portable Chromium/WebKit cases,2 integrated
native old-owner receipt/projection cases and1 actual TLS transport-loss journey
in one source-frozen93-case epoch. The earlier4-case epoch separately covers three
native exact Stop controls under receipt pressure plus the transport journey;
counts overlap and are not summed. Full-file Ruff/formatter on both changed Python
files, native AST, JavaScript syntax and diff check pass without exclusions.
New report/manifest: whole-feature-fix-implementer-report.md and
whole-feature-fix-final-manifest.json in the existing local SDD directory. Earlier
reports/manifests remain frozen. Scoped rereview approved all three findings; Evals R06 and original
warning/third-Create origin limits remain disclosed. Physical phone/VPN remains
**UNVERIFIED**. No full repository test sweep was run.

The same whole-feature reviewer approved WF-I01, WF-M01 and WF-M02 at
`22bae10d3dd6`, with no new findings. The local
`whole-feature-fix-rereview-findings.md` records the scoped verdict; the original
review retains its stated coverage and CV-01–CV-03 qualifications. Completion
metadata does not change the tested executable source.
