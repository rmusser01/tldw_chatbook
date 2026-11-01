# Shared served Console and mobile qualification

Task11 implementation and bounded qualification are recorded below; independent review remains pending. This packet separates source/unit evidence,
actual served-browser evidence, and physical phone/VPN qualification. It does not
mark the release or Task36769 complete.

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

The inventory has a source-unchanged Dreams capture census mismatch. Three
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
receipt branch was not delivered in that epoch. A corrected response-body witness
proved actual browser delivery in a separate **12 passed in 8.78 s** both-engine
run on final test bytes. The affected real-browser run recorded **4 passed,
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
artifacts are not packaged runtime data. Independent Task11 and whole-feature
review remain pending. Physical phone/VPN qualification remains **UNVERIFIED**;
the full repository test suite was not run.
