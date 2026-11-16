# Windows Send latency implementation plan

> **For agentic workers:** Use superpowers:executing-plans. Root owns integration and all native testing/timing. Agents implement only their explicitly assigned files; no concurrent native runs.

**Goal:** Qualify PR3534 below six seconds from actual Enter/Send action to provider-adapter entry on Windows, preserving feedback and saved-turn correctness.

**Architecture:** Extend the existing finite configuration/preparation owners where measured redundant work occurs. Keep every effect-time permission check and physical native retirement. No lease spans provider I/O or awaits.

**Tech Stack:** Python, Textual, SQLite, existing native storage admission.

**Spec:** ../specs/2026-10-06-console-send-preparation-architecture-design.md, amended by the owner's 2026-11-15 six-second merge gate and authorization to refactor.

ADR required: yes, amend existing ADR126 for bounded configuration-read assembly and ADR231 for promoted manual preparation polls.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md; backlog/decisions/231-console-polling-and-full-state-reconciliation.md; existing ADR230 continues to govern Send preparation ownership.
Reason: reuse existing native operations and display owners without changing durable dispatch or authorization.

## Constraints and review focus

- Each first and later Send sample, for both routes, must be below six seconds; no average substitutes for a failed sample. Provider response time and application boot are separate.
- Native runs remain sequential. Diagnostic instrumentation and concurrent host load are recorded; uninstrumented measurements qualify release.
- Preserve all public path getters, custom overrides, per-getter additive failures and no memo publication after failure.
- Configuration admission/retirement errors propagate; do not retry a partly retired read or treat it as an empty successful deny list.
- Custom getters remain callable under the normal configuration read scope. Keep path selection, source drift, parent posture, pause and cancellation checks.
- Save failure keeps the draft. Waiting for reply, Streaming and owned shutdown must remain verified for Enter and Send.

## Task 1 — Attribute the complete event-to-dispatch path

Files: Tests/Performance/test_console_send_wall_clock.py; external diagnostic plugin and isolated native receipts.

- [x] Route the existing full-app diagnostic through actual Enter and button events using the production task factory.
- [x] Record original-source first-send failure (86.908 seconds) and focused native RED (26 admissions for 13 database paths).
- [x] Complete the diagnostic with explicit ready-composer preconditions and per-owner inclusive cost attribution. All three sends, persisted responses and trace settlement must finish; preserve failures as evidence.

## Task 2 — Group related configuration reads

Files: tldw_chatbook/Utils/sensitive_paths.py; Tests/Utils/test_sensitive_config_read_scope.py; ADR126.

Interface: _sensitive_db_paths() -> tuple[Path, ...] retains its signature and all 13 actual getters. One config_participants.operation(config) owns the synchronous assembly; nested getters reuse its checked admission.

- [x] Native RED asserts original output parity and at most two storage acquisitions, and fails at 26.
- [ ] Add custom/failed getter, no-partial-memo, changed selection and failed retirement controls appropriate to the selected scope.
- [x] Enclose the existing loop in the existing operation; retain per-getter try/_debug and let scope-level failures escape.
- [ ] Run the focused native regression, affected config/companion/run-log controls and lint. Expected: all pass without changing native checks or deadlines.
- [ ] Compare complete uninstrumented first/later sends. Keep only a justified change; further measured causes require a scoped plan amendment before implementation.

## Task 3 — Coordinator experiment, deferred after comparison

Disposition: removed from PR3534. The prototype passed its concurrency controls
and integration checks, but did not establish a whole-Send speed benefit. With
the same diagnostic periodic-poll deferral, the original coordinator measured
16.367/19.647/15.186s and the prototype 23.115/23.230/17.878s; host variation
prevents a precise regression claim. Both fail the release gate. Preserve the
prototype patch/test in task artifacts for separate review. The following is the
historical experimental plan, not selected production behavior.

Files: tldw_chatbook/Backup_Recovery/storage_admission.py; tldw_chatbook/Backup_Recovery/participants.py; Tests/Backup_Recovery/test_core_operation_coordinator_io.py; ADR126.

Interface: existing _check_operation still invokes the original native path proof. It may return a short-lived binding snapshot for an immediate in-memory recheck at the consuming critical section. No native result is cached for another operation. Acquisition scope/control derivation remains fresh before and after counting its lease.

- [ ] Run held-original-I/O REDs for core access/getter/nested operation/initializer and both settled admission observations. Another worker must acquire the coordinator before native proof is released. Direct checks must refuse registration, lease, repository binding and path changes during that proof.
- [ ] Separate current in-memory custody checks from original native path proof. Revalidate the exact bindings afterward; preserve accepted-operation behavior during pause and all native retirement.
- [ ] Move outer caller critical sections and both admission scope/control observations off native I/O. Pending acquisitions and counted leases stay registered throughout. Recheck attempt/captured hold identity and names atomically with counting or publication; retain retry/refusal semantics.
- [ ] Run the new controls and existing initialization, core-lifetime, path-replacement and evidence-reuse tests. Expected: all pass with unchanged safety budgets.
- [ ] Repeat the complete full-app profile and ordinary sequential timings. Investigate remaining measured work only if the six-second gate still fails.

ADR required: yes, scoped clarification of existing ADR126's raw native-proof pattern for core and admission proof. No new coordinator, connection pool or cache.

## Task 3b — Group declared permission-store members

Files: tldw_chatbook/Backup_Recovery/raw_participants.py and focused MCP/raw
admission tests. Shared-preparation lane owns these files; root owns tests and
integration. Controller/display lane remains source review until its bounded
proposal is reviewed.

ADR required: yes, amend existing ADR126 for this installed owner.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: use the existing related-path admission contract for the permission
participant without creating a new snapshot, lease lifetime or authority cache.

- [ ] Add an original-call native control that requires one direct admission
  carrying all declared permission members; confirm RED on three admissions.
- [ ] In the existing raw admission branch, select only the checked MCP route
  and installed participant owner mcp.permissions. Group admission_paths using
  the existing related_paths argument; preserve all pins, preflight and separate
  canonical observation lease.
- [ ] Update the old scalar count oracle only alongside full-member and retirement
  assertions. Verify changed/outside member, pause, failed close and custom/source
  routes; run affected tests sequentially.
- [ ] Measure the original complete Send route and retain only demonstrated
  improvement with unchanged correctness.

Cross-pass preview reuse is not selected: ordinary permission.load publishes an
observation which the existing owned payload helper omits. Grouping admission
preserves that behavior without broadening the preview contract.

## Task 3c — Preserve progress without repeating routine preparation reads

Files: existing console_send_start, composer_run_controls, config_sync,
chat_screen and console_spend_projection owners, plus focused UI controls.
Controller/display lane owns these files; root owns integration/native runs.

ADR required: yes, bounded amendment to existing ADR231.
ADR path: backlog/decisions/231-console-polling-and-full-state-reconciliation.md.
Reason: extend the accepted manual preparation display route through the same
turn's promoted preparation until its own provider request, while preserving FULL.

- [ ] Establish RED for original promoted preparation polling and the retained
  task's SendStart after the unrelated three-second display window is pruned.
- [ ] Preserve the existing unpromoted route. Qualify promoted work through the
  original task-to-preparation map, current store preparation, completed FULL
  source and resident input/policy revisions. Unknown or changed owners stay FULL.
- [ ] Keep speech/H3 handling, draft synchronization, original tabs/transcript
  publication and immediate feedback. Defer routine retrieval, character,
  readiness/config and rail reads on this qualified route. Read only an existing
  same-owner context display without scheduling its refresh; missing evidence
  falls back to FULL. Keep existing cadence work owed and the poll alive.
- [ ] Preserve explicit FULL dominance and post-await currentness. Verify saved
  draft/new edits, provider entry across an await, pending decisions, settings/
  runtime/Canvas/roleplay changes, other sessions, cancellation and terminal
  handoff through the existing producers. No manual sync substitutes for them.
- [ ] Run both event routes on final integrated sources; report quiet individual
  action-to-adapter samples separately from diagnostic and counterfactual runs.

## Task 3d — Retain one canonical admission during permission-scope construction

Files: raw_participants, mcp_source_participants and focused permission tests.
Shared-preparation lane owns these files; root owns native verification.

ADR required: yes, amend existing ADR126.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: nine repeated canonical admissions occur inside one synchronous raw-scope
construction. Retain one existing lease while keeping every fresh witness read.

- [ ] Add a native original-call regression requiring two total admissions per
  installed permission read: one canonical observer and one full member group.
  Confirm RED on the grouped candidate's ten admissions, with original witnesses,
  payload publication and retirement assertions unchanged.
- [ ] Derive the installed canonical path using the existing pure binding check,
  acquire its lease under the existing acquisition, and pass a construction-local
  retained map through the private MCP selection/binding/member/participant gates.
  Every selected_path call still invokes original observed/_witnesses; no witness
  or selection result is memoized. Keep state inactive until original publication.
- [ ] Transfer the early lease into the existing state's leases/holds when state
  is constructed; scope cleanup owns it before that point. Keep separate related-
  member admission, since canonical-only authority does not cover main/tmp/bak.
  Recovered sources must not pre-admit unused canonical sidecars. Active operation
  checks continue using existing mcp_observation with no new owner or TLS registry.
- [ ] Verify source/config/recovery changes, custom sources, preflight pause,
  missing/exact-file member enrollment, exceptional construction and uncertain
  native close. Keep fresh final checks, refusal boundaries and physical cleanup.
- [ ] Measure original whole Send after integration with Task3c; no admission
  count or instrumented duration substitutes for the six-second merge gate.

## Task 3e — Separate MCP declaration from live validation

Files: raw_participants, mcp_source_participants, focused permission/member and
filesystem-wait tests. Shared-preparation lane owns implementation; root runs native tests.

ADR required: yes, amend existing ADR126.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: structural metadata access currently repeats recovery-file validation
inside one synchronous constructor; make the remaining validation boundaries explicit.

- [x] Establish RED for the five structural witness derivations per installed
  generation-selected MCP construction: member declaration, history classification
  and participant registration/helper validation. Test local/context/permission
  sources under their common recovery-selection contract.
- [x] Reuse only this construction's already validated owner/selected/installed
  declaration through pure helpers and existing participant registration. Keep
  the custom/uninstalled routes unchanged; no witness cache, new owner or TLS state.
- [x] Preserve initial selection, fresh pre-lock, post-wait preflight and late
  publication checks, active custody/parent checks, readability approval,
  file-open and checked-read final publication. Drift may refuse at the next
  explicit boundary; no payload access may precede that validation.
- [ ] Verify changed source/config/generation, source-mutex pause, exact-file
  sidecars, recovered destinations, construction failure and physical uncertain
  close, plus original permission behavior. Extend existing canonical retention
  and related-member admission to the same generation-selected source family;
  canonical and selected members remain independently admitted (two acquisitions).

## Task 3f — Build resident configuration references once per capture

Files: Chat/console_configuration_preparation.py and its focused configuration
capture tests. Controller/display lane owns implementation; root integrates.

ADR required: no new ADR.
ADR path: backlog/decisions/230-console-send-preparation-and-io-ownership.md.
Reason: remove repeated metadata construction within an existing synchronous
capture, preserving owners, callbacks, awaits and all freshness/refusal boundaries.

- [x] Establish RED for repeated original reference construction/classification
  on the successful stock capture; preserve custom and cold-trust paths.
- [x] Share the local successful classification with capture assembly. Do not
  retain it across calls/awaits or remove any preparing_poll_record recheck.
- [ ] Verify missing/cold proofs, custom factories/loaders, bound/unbound/replaced
  MCP readers and changed-owner/config refusals, then integrated UI/Send controls.

## Task 4 — Qualify and integrate

Files: existing feedback/shutdown tests, task34601.1 and Docs/Development/console-optimization-review-list.md.

- [ ] Run both event routes sequentially on the final integrated source with real private-profile storage. Record each cold/warm action-to-adapter sample and host conditions.
- [ ] Run current targeted feedback, save-failure/draft, permission and physical-shutdown checks. Expected: existing budgets and contracts pass.
- [ ] Update the deferred optimization list and implementation evidence. Obtain independent review and fix material findings.
- [ ] Push with lease protection, resolve current-head review threads and pass PR CI. Re-arm auto-merge only after the six-second gate passes. Follow queue rebases through CI and actual merge.

## Execution record

Baseline ccb453d001 on dev cc6006fcf593: PR auto-merge disabled. The first full-app Enter sample was 86.908 seconds on a loaded host; it is a failure, not a release qualification. The diagnostic's second send did not establish readiness and is being corrected before using warm samples. The focused original-function native count is 26 admissions for 13 getters.

Ruling: preserve native authorization through the existing operation rather than add stock-function provenance machinery. Custom getters are still invoked, now within a documented synchronous configuration read scope; extensions must not wait on another thread that needs the same configuration locks.

Configuration batch native controls: four pass in41.85s (pr3534-config-scope-green-1). The completed diagnostic pr3534-six-second-costs-enter-3 records Enter-to-adapter37.060/33.032/26.771s, three saved user/assistant pairs and complete traces. It separately reports1.430s awaiting the original onboarding publication before the second eligible Send. No producer or capture gate was bypassed. First-send core checks entered with the coordinator held1553times, totaling4.794s inside those nested diagnostic spans; these figures do not establish whole-path savings. The six-second gate remains unmet.

Coordinator RED: nine failures and one pass in12.96s. All ten new controls pass
after the change. The first integrated run exposed an inappropriate use of the
reuse-selector helper as final hold authority; the final check now tests actual
hold readiness/stop/error. Existing paused-startup readmission then reproduced
the reviewer's subclass finding (three failures, one pass) and was corrected with
the existing pause-owned subclass check. Both ordinary and cancelled native
startup readmission, real app startup/capture handoff, and shutdown checks pass.

Remaining qualification repairs before release:
- Compare eleven core-lifetime failures with exact dev cc600. Ten use WinSock
  select on process pipes; one assumes the managed ChaChaNotes cursor survives
  its owner's documented close. If baseline confirms these, use existing
  pipe_ready and a distinct caller-owned cursor without weakening exclusion.
- Add a regression for unavailable sensitive-context resolution. The public
  is_sensitive_path predicate must fail closed, including after a changed config
  selection; the failed assembly must still propagate internally and never memoize.
- A symlink privilege failure in the existing sensitive-path test is an explicit
  host limitation. Preserve symlink coverage on capable platforms rather than
  change machine privileges or replace the native check with a mock.

The exact dev cc600 baseline reproduced twelve failures and passed the config
retarget test. Eleven failures were the existing lifetime fixture issues; the
twelfth was the host's unavailable symlink privilege. Reusing the existing
cross-platform pipe observer and creating a distinct caller-owned cursor
preserves the original exclusion assertions and time bounds. The public
sensitive predicate now refuses an unavailable inventory; internal assembly
still raises without memo publication. The combined lifetime, coordinator and
sensitive-config controls pass: 76 tests in 149.44s
(pr3534-lifetime-and-sensitive-green-1), with unchanged source throughout.

Benchmark correction: the previous test wrapped gateway.resolve_for_send to
choose streaming. This fails the production native-method identity check and
uses the resolver's default worker path, omitting the owned preparation runner
and its bounded interpreter handoff. Those elapsed/census receipts describe
that custom resolver route and cannot qualify stock Send. The diagnostic now
sets actual session streaming state and asserts the original provider resolver
methods remain installed. Custody/stage observers and the controlled adapter
seam were separately audited; no additional identity bypass was found.

The measured recurring display work remains a candidate, not an implemented
optimization: readiness configuration, character context and context-input
refreshes recur during the whole send/completion phase. Counted phase work is
not all pre-adapter work, and thread spread alone does not prove simultaneous
contention. Re-measure the corrected original route before expanding polling
behavior or introducing additional preparation batching.

Corrected stock-resolver quiet baseline with the original coordinator:
38.259/55.197/54.747s (pr3534-original-coordinator-stock-enter-1). The identical
source with diagnostic-only periodic display deferral measured
16.367/19.647/15.186s (pr3534-coordinator-control-periodic-enter-1). Both completed
three saved replies and trace settlement with unchanged sources. This is
directional evidence under variable host load, not an exact saving or release
qualification. Task3c selects a bounded production route rather than the
diagnostic's blanket periodic deferral. All samples still fail six seconds.

Permission-member RED: the original scope makes three scalar admissions for
main/tmp/bak; the new regression requires one original admission carrying all
three and verifies physical retirement first (pr3534-permission-group-red-1,
one intended failure in14.76s). No scope or native deadline changed.

Task3c original-callback RED: three intended failures in88.65s; the exact custody
reader and no-refresh display reader were absent, and original promoted polls
never qualified. The first candidate passes all four focused checks in146.30s,
including Enter and button with original post-save history held. Independent
review identified a custom terminal-status getter not covered by source guards;
the candidate now requires that existing getter to remain stock. Added promoted
input/settings, pending FULL, provider, Stop and terminal transition checks await
their separate run.

Task3d RED: two intended failures in15.54s, ten acquisitions instead of two and
no retained canonical custody at the pre-State construction boundary. Candidate
verification passes fifteen of eighteen checks in204.45s, including every new
count/construction/drift case, both original permission witness/context counts,
preflight pause, exact-file refusal, changed configuration and corrupt backup.
Three related-path fixtures fail during their initial enrollment before the
changed permission route; those remain unresolved. Exact dev also fails its
post-close0.1s maintenance check. A separate original dev exception-chain probe
confirms Windows1314 behind the moved-alias failure; the test now uses the existing
real directory-alias helper. Original dev confirms the MCP subprocess pipe
select failures; existing pipe_ready replaces those calls at unchanged bounds.

The integrated quiet stock Enter comparison still FAILS the merge gate:
59.957/45.006/33.006s (pr3534-integrated-preparation-stock-enter-1), with all three
saved replies/trace settlement and unchanged sources. The functional held-state
qualification does not establish ordinary-app eligibility or a latency saving.
Next diagnostic observes original poll-source/qualification returns without
replacing those functions, before selecting another change.

The passive original-return diagnostic confirms ordinary-app promoted polling
qualifies (pr3534-ordinary-poll-qualification-1). Its resident registry is present;
the missing-registry hypothesis is disproved. Instrumented Send samples are
31.504/32.755/24.018s with complete replies and traces. Qualification is therefore
not the remaining main cause. Separate the retained display cost from preparation
using the existing diagnostic-only periodic deferral before another product edit.
These are attribution runs, not release timing evidence.

Three further native construction controls pass in22.64s
(pr3534-canonical-construction-edges-1): source drift while the source mutex is
held, pause cancellation during that wait, and actual uncertain descriptor close
preserving native exclusion against an independent maintainer. The extended UI
transition run failed both cases before its intended edges: settings Apply is
deliberately disabled during the run, and this early preparation state has no
visible Stop action. Correct those probes to exercise supported configuration
publication and exact custody cancellation; do not claim visible Stop coverage
from that cancellation control. Existing Stop/shutdown qualification remains due.

Current-source diagnostic periodic deferral still takes22.528/20.873/19.955s
(pr3534-current-periodic-counterfactual-1); all three replies and trace settlements
complete, with original non-display guards and restoration intact. Deferring polls
alone cannot qualify this candidate. The original-return census records89/78/59
completed promoted polls and11/10/14 FULL-bodied returns. Repeated eligibility
performs resident metadata checks, with no direct native read found in the stock
source graph; count alone does not establish its CPU cost. Next sparse census
measures original native acquisition/witness boundaries and the synchronous
eligibility envelope, without changing product functions.

The next related-file/transition selection passes8/9 in132.86s
(pr3534-related-and-promoted-edges-2): all seven related-path controls and button
custody cancellation pass. The earlier enrollment failures did not recur, so their
cause remains unproved. Enter fails while closing the read-only modal before its
intended subsequent transition probes. The fixture's class-only open predicate
does not establish mounted readiness; preserve the close policy and time bound
while observing that precondition and any failure state.

The sparse census on unchanged integrated source completes75.464/65.291/71.017s
sends under heavy host load, with zero observer errors and restored original
functions (pr3534-sparse-native-and-enter-transition-1). Native acquisition starts
are349/324/352 across threads; their inclusive wall times overlap and are not
additive savings. Witness construction repeats nine times per permission scope:
45/54/54 of85/93/93 total witness derivations. Existing helper/source review
identifies five structural derivations that can be separated from explicit live
validation; Task3e selects that bounded refactor. Eligibility itself consumes
4.953/5.047/5.125 seconds of main-thread CPU across2176/2257/2194 calls. Task3f
selects the demonstrated duplicate resident classification within each capture,
while retaining every callback/await recheck. Readiness projection still performs
27/24/32 native reads; its explicit FULL demand producers remain under investigation.

Task3e/3f RED completes seven intended failures in56.35s with unchanged sources
(pr3534-live-gates-and-reference-red-1). All six ordinary/recovered MCP cases
complete original load, active checks, physical retirement and historical-byte
controls before reporting nine construction witnesses instead of four. Three
successful captures each build/classify references twice; stale-owner refusals
remain correct. Required PR census validation and four exact workflow contract
tests pass; hosted CI and final Windows qualification remain outstanding.

The current native MCP lifetime/refusal selection passes all27 cases in426.49s
(pr3534-mcp-lifetime-negatives-1), including uncertain-close exclusion. The Enter
transition probe now reaches modal closure, changed configuration, pending FULL
and provider handback, then fails its immediate assertion after Task.done before
the original done callback retires custody. Its corrected wait includes that
retirement within the same60-second bound; terminal qualification awaits rerun.

The first integrated Task3e/3f selection completes58passes/8failures in669.07s
with unchanged sources (pr3534-live-gates-and-reference-green-1). The new original
capture count and all four MCP construction gates pass. Three recovered count
cases omit the existing missing-parent admission from their expected member
group; the fixture now includes it and asserts no directory creation. A stale
ephemeral-MCP expectation and four older cold-receipt assertions/timeouts are
being compared with exact dev before attribution. Recovery approval, source/config/
generation drift, pause, sidecar exclusion and uncertain-close controls pass.

Independent review identifies one removed cheap native-posture check at direct
participant registration. Its new mid-construction demotion regression must fail
on the candidate, then the original refusal must be restored without rebuilding
recovery witnesses. The existing before-scope demotion control does not cover
this interval. No release qualification is claimed from this intermediate run.

The original pinned-posture check is restored, without an additional recovery
witness. Its exact mid-construction regression first failed, then passed in the
23-case integrated construction/UI selection (189.38s,
pr3534-construction-and-ui-green-2). Both promoted UI routes complete their
configuration, FULL-demand, provider and custody-retirement transitions. Button
cancellation uses the exact custody task; this is not visible Stop coverage.
Exact dev cc600 reproduces the stale ephemeral-MCP parity expectation. The
corrected parity fixture preserves field equality and fresh-source controls.

The latest uninstrumented stock Enter run still fails the merge gate:
77.244/44.322/22.795s (pr3534-live-gates-stock-enter-1). All three replies and
traces finish with unchanged source. Saved-to-trace-reservation intervals are
57.907/24.986/13.658s. Pre-run host CPU samples were37.0–41.2 percent; they do
not describe host load during the sends or support a comparison with older runs.

The first readiness-publication diagnostic is invalid for attribution. Its
foreground target resolved to the shared contextlib.helper code object and
filled the2048-row cap within1.4s of every action, before durable acceptance.
The sends complete38.200/42.321/26.778s and observer state is restored, but those
truncated rows cannot select a product change. Resolve decorated targets to
their exact original bodies, verify source/qualname and add a shared-wrapper
collision self-check before the next native diagnostic.

Read-only Windows source review narrows the finite DB grouping candidate.
The existing process-group predicate already excluded unsupported hosts before
its mechanical rename. NativeAgentSources.activate owns source execution and
connections but does not establish core_operation, while Windows maintenance
workers already guard their execution and Chat DB connection. Removing that
platform gate is therefore neither a proven batching fix nor a selected change.
If attribution identifies AgentRuns bootstrap, consider its synchronous initial
row/lifecycle cluster under the existing exact-DB operation and connection owner,
preserving independent transactions and best-effort lifecycle errors. Do not
span provider I/O, custom callbacks or awaits with that native scope.

Windows cold-receipt oracle follow-up (pr3534-cold-native-oracles-green-1):
the native_managed case passes in64.23s, including unchanged receipt identity,
trust publication and physical retirement. The native case hits its original
70-second child deadline before finishing; the combined run is1pass/1failure
in139.75s with unchanged source. Keep that timeout unresolved; the supported-host
ordinary-tree oracle correction alone does not qualify this route's performance.

The corrected original-body readiness census completes52.808/31.480/32.610s
(pr3534-readiness-attribution-enter-2,221.45s overall). All replies/traces finish;
sources stay unchanged. Local events remain below caps, with zero observer errors
and complete restoration; callback wall overhead is0.108/0.069/0.072s. There are
21/18/15 readiness refreshes but only1/1/0 changed publications, both from owner-key
changes. No value/policy/pricing/model-cap/tokenizer change fires. In send2, the
sole publication's concurrent FULL already owes its replay and returns immediately;
send3 starts9 FULL enqueues with no readiness publication. This disproves changed
readiness publication as the repeated demand source in these samples. Do not
implement the proposed same-owner publication cadence change from this evidence.

Foreground rows in that diagnostic are absent because its identity filter reads
an instance dictionary slot while the real screen exposes the controller through
a property. Readiness rows remain valid; the foreground envelopes do not. Use the
benchmark's already captured actual controller/bridge owners, with an absent-slot
regression and required-stage coverage checks before the next run. Never infer
zero foreground work from the empty filtered census.

The exact-owner foreground diagnostic is valid (pr3534-readiness-attribution-enter-3):
49.756/53.543/28.351s; all four required original targets are observed in each
phase, with zero missing targets, caps or source errors and complete restoration.
Ten postcommit effects are seen per send. Trace-request assembly takes about
0.057s; identity/owner/workspace/history publication together takes about
0.54/0.76/0.50s, and the checkpoint transition0.35–0.62s. They are not the large
post-save stall. Provider-entry effect to the original agent-reply body spans
6.795/7.303/4.396s; that route awaits the existing guarded maintenance call creating
the automatic-work chain and enters the checked-generation wrapper's fresh
response-rule resolution before the observed original body. The whole interval
cannot be assigned solely to maintenance admission. Original bridge-preparation entry to adapter spans
18.727/22.754/6.496s. These are nested elapsed regions, not additive savings.
The three provider/coroutine spans intentionally remain open at the adapter cutoff.

### Task3g investigation — consolidate facts inside one existing execution gate

Files under review: Backup_Recovery/activation.py and focused native activation
controls. Shared-preparation lane owns the new test file; root owns native RED,
integration and the decision to implement after independent contract review.

ADR required: yes, amend existing126 before product implementation.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: per-call recovery fact capture is a security/service contract; every
existing guarded entry and actual owner approval must remain fresh.

Source review finds five control-record reads and four registry reads in one
successful execution_scope(('config', owner), path): the source-scope check plus
startup and activation checks repeated for both owners. Keep every launch and
nested entry gate, exact retained lease check and final effect ordering. Select
no cross-call cache and no broader lease lifetime. The final reviewed candidate
changes only activation_permission's stock startup route: one fresh control/
registry read serves startup and the unchanged activation-policy body for that
owner. Keep execution_scope and its source-scope read entirely unchanged. A
two-owner gate targets three control reads and two registry reads. A replaced
startup binding/code retains the original callback then fresh activation reads.
Pending intent can be published while a lease is held; each owner's fresh call
retains its original refusal. Keep every actual generation required/approval read.
The narrower change also benefits scalar callers and avoids an execution-loop
bypass or new source-scope API; the earlier two-control-read variant is unselected.

- [x] Establish ordinary and approved-recovered native original-call count REDs,
  with real allowed results and physical/borrowed-lease retirement qualified.
- [x] Review paired-record loss, pending intent, source/config drift, shared
  namespaces, custom callbacks, revocation, read failures and physical close.
- [x] Amend ADR126 and implement only the reviewed finite gate; run its native
  negative controls and the original recovered-agent launch cases sequentially.
- [x] Repeat original complete Send measurements; keep the six-second merge gate.

Current feedback/shutdown qualification (pr3534-feedback-stop-shutdown-current-1):
5pass/1failure in180.63s. Both actual Enter/button feedback-state cases pass
through held preparation and provider wait; visible Stop retains the draft and
the native-helper shutdown control confirms physical cleanup. Button also passes
the first-feedback/input100ms control. Enter fails the unchanged first-feedback
budget at156.25ms loop CPU/168.77ms wall. This is an unresolved speed failure,
not an oversubscribed-host wall-only miss. Compare the exact case with clean
dev before choosing a UI change; physical terminal verification remains separate.

The exact Enter first-frame control also fails on clean dev cc600
(pr3534-feedback-enter-dev-baseline-1): 529.10ms wall/234.375ms loop CPU,
97.65s overall. This establishes a pre-existing failure, not an exemption from
the immediate-feedback requirement or a measured PR speed improvement. A narrow
original-function observer will attribute action-to-first-frame before any UI edit.

Task3g RED1/RED2 are invalid as count comparisons: all four cases fail during
fixture setup before the monitored gate (43.61/36.23s). Adding the existing
Windows default-owner context did not fix the cause. The actual child traceback
shows TOMLDecodeError: Invalid hex value at line4 column15: the reused setup
writes an unescaped Windows data path into a TOML basic string. Correct only the
new fixture's path serialization with Path.as_posix(), retain native ownership,
all original checks and bounds, and rerun before implementation.

Task3g RED3 is valid (pr3534-execution-gate-read-red-3,52.95s): ordinary and
approved-recovered scopes both return allowed, preserve the borrowed lease and
physical exclusion, retire and reopen successfully, then fail exactly at five
control/four registry reads instead of three/two. Their gate durations are
0.0579/0.1150s; these small isolated fixtures are not whole-Send savings. Both
next-owner pending-intent and paired-history-loss refusal controls pass. Proceed
with the reviewed scalar-only change and custom-startup compatibility controls.

The additional native-save/shutdown selection passes all nine cases
(pr3534-native-save-shutdown-current-1,47.87s) on frozen integrated source.
Repeated close/dispose cancellation retains actual native work through retirement;
cancelled saves preserve original/newer drafts and do not dispatch; a normal save
releases native ownership before provider wait; unrelated submit tails do not
extend the native drain; bounded UI reconciliation retires during shutdown for
both native variants. This is targeted lifetime evidence, not whole-app exit or
physical-terminal verification.

Task3g native integration (pr3534-execution-gate-read-green-1,131.14s): all
seven new regression/negative cases pass, as do71 existing paired-history,
shared-owner, corruption, safe-startup and registry controls. One existing
private-control-root test fails because its chmod(0755) fixture does not make a
Windows directory's DACL public. Compare that exact test with dev and use the
existing native ACL fixture for this negative control; do not weaken the refusal.
Independent source review found no actionable issue in the scalar patch.

### Task3h — own immediate-feedback imports at Console mount

ADR required: no; implement the existing immediate-feedback contract
within ADR097's unchanged boot ratchets and ADR231's existing display lifecycle.
ADR path: backlog/decisions/097-boot-budget-ratchets.md;
backlog/decisions/231-console-polling-and-full-state-reconciliation.md.
Reason: a normal module dependency moves to its existing component mount;
no new owner, scheduler, cache, native permission or dispatch boundary is added.

The valid first-frame trace passes narrowly at97.907ms wall/93.75ms loop CPU;
32.786ms is cold acknowledgement import (including7.995ms sent_draft), versus
0.405ms actual request handling. The existing idle pre-import pass is routes-first
and races immediate Send; moving those routes again would revisit an earlier
Library regression. Import only the acknowledgement dependency at original
ChatScreen.on_mount entry, creating no acknowledgement/flight/native owners.
Keep the existing lazy helper for direct/unmounted callers. Both modules' screen
imports are function-local, so this introduces no import cycle.

- [x] Establish a fresh-process original Enter RED proving the mounted Console
  still owes acknowledgement imports before action, without warming the fixture.
- [x] Add only the existing-helper import at mount; update stale lifecycle prose.
- [x] Run both original100ms feedback/input cases and import-cycle controls.
- [x] Run the unchanged ui_ready module census. Expect two added modules; if the
  ratchet fails, remove equivalent unnecessary boot cost before adopting. Do not
  raise limits or hide work behind an after-refresh race.
- [ ] Verify whole-Send timing on integrated source; this addresses first paint,
  not a claimed solution to the multi-second dispatch delay.

The first scalar timing launch was refused before any app process because the
caller pre-created the runner-owned result directory. Keep it excluded. The
corrected quiet run (pr3534-scalar-activation-stock-enter-2,172.17s) completes all
three saved responses/traces on unchanged source:27.7965/27.0485/21.4130s from
actual Enter to the stock provider's controlled final adapter. Every sample fails
the six-second gate. Pre-run host CPU samples48.2–56.7 percent do not characterize
the sends or establish a causal comparison with older runs. The scalar refactor
qualifies reduced read duplication, not release performance.

Task3h original Enter RED is valid: both acknowledgement modules are absent after
the original mounted-ready wait, before any helper call/action. The combined
selection passes the actual Windows unsafe-DACL refusal plus all five exact CI
target/job controls (pr3534-feedback-import-red-acl-green-1,1fail/6pass,51.47s).
The one-line mount import is implemented for native feedback and unchanged
startup-budget verification; it creates no acknowledgement or flight owner.

Task3h first integrated selection (pr3534-mounted-feedback-and-boot-budget-1,
210.82s) passes Enter's unchanged feedback/input budget and the import-cycle
control. Button misses the100ms wall gate narrowly (78.125ms loop CPU), so it
remains unresolved. The unchanged ui_ready census reports1043 against1040;
new residents versus the pinned snapshot are the two acknowledgement modules,
provider_setup_facts and windows_files. Compare clean dev before attributing all
three excess modules to this PR. Select the existing ADR097 remedy: defer
DictionaryPicker and ConsoleScopePickerModal to their explicit open handlers,
including DictionaryPicker's Personas edge and both ScopePicker edges. These
widgets do not participate in mount, readiness, publication proofs or Send.
This targets two-module payback without raising either boot/pre-import limit;
verify actual residency and the existing modal-action controls afterward.

Clean dev cc600 itself measures1041 against1040 on Windows
(pr3534-ui-ready-dev-baseline-1,153.52s), with provider_setup_facts/windows_files
added versus the pinned snapshot. Pay back that existing one-module excess too:
defer chat_events_console_dictionaries from its sole ChatScreen eager import to
the same explicit attach/detach workers. Its four functions have no mount,
readiness, registration or source-witness role; shared dependencies are already
resident. Together these are local import deferrals for dictionary/scope actions,
not new startup scheduling. Verify unchanged boot and pre-import budgets plus
existing scope-modal, dictionary-modal and dictionary-service controls.

Task3g's original seven recovered-agent launch states pass after serializing the
Windows TOML fixture path with Path.as_posix(). The new regression consumes that
same fixture directly; its expanded child bytes are unchanged by this cleanup.
The private-control-root chmod failure reproduces on clean dev. Using the existing
exact-path Windows DACL fixture makes the original public-read refusal pass,
without changing the native posture checks. Both fixes are test portability only.

Task3h payback verification (pr3534-feedback-import-payback-1,228.58s) passes
both original feedback/input cases, the unchanged ui_ready census (1037/1040),
and first-send warmup (94/94). Enter/button Sending frames are91.064/96.574ms;
input frames12.336/13.609ms. Pending USER frames are separately168.971/181.001ms.
These are supplied headless frames, not physical terminal flush. Sources and HEAD
remain unchanged. The separate registry pre-import pass fails560/558 modules;
investigate its added module set against dev before attributing it to this change.
No budget rises. The six-second whole-Send gate remains open.

The three modal-action checks fail identically on clean dev cc600
(pr3534-modal-actions-dev-baseline-1,38.47s): raw_source_selection_changed occurs
in fixture/config setup before their changed action imports. Apply the existing
per-node bootstrap_profile convention to those three tests, leaving their actions,
assertions and source-selection checks intact; native rerun is pending. The other
four dictionary picker/service controls already pass.

### Task3i — use single-widget lookup for recovery display controls

ADR required: no.
ADR path: N/A; this is a mechanical UI lookup change within ADR231's existing
recovery publication and currentness contract.
Reason: the current original-source py-spy capture attributes8.16 sampled
main-thread seconds to archive-recovery sync in its60-second window. Its
single-surface lookup alone spends5.38 in DOMQuery materialization. The installed
Textual query_one API already provides ID lookup and node-change invalidation.
Use that existing API; add no view-level cache, new lifecycle or authority token.
The sampled window includes first Send, settlement and part of the next Send;
these inclusive sample weights are neither CPU seconds nor predicted savings.

Root owns archive_recovery.py and integrated native verification. Replace the
three single-target query/truthiness/first sequences in _surface, _confirmation
and sync with query_one and the same missing-node result. Preserve the existing
duck-typed surfaces, actual mounted/attached checks, post-await currentness,
decision/receipt ownership and all multiple-result queries. Existing mounted
card tests cover retiring nodes, missing controls, remount, source replacement
and app exit; no mirrored implementation test or timing-budget change is needed.

- [x] Attribute the original UI cost in a complete, source-stable Send capture.
- [x] Review Textual lookup/invalidation semantics and the existing lifetime tests.
- [x] Apply only the three lookup substitutions; run the affected mounted cases.
- [ ] Repeat quiet first/later Enter and button sends sequentially on integration.

Task3h pre-import correction: DictionaryPicker's package imports three additional
Persona modules. Deferring its eager edge moved those modules into the later
registry pass. Restore the original DictionaryPicker imports and instead defer
the action-only character picker, whose package/dependencies remain resident.
Retain the scope picker and dictionary-service deferrals. Expected counts are
1040 ui_ready,557 registry-pass and94 first-send modules; verify all unchanged
ratchets and the actual character-picker actions/lifetimes before qualification.

Task3i and picker integration (pr3534-recovery-lookup-and-picker-actions-1,
259.03s): all25 targeted mounted-action and physical-lifetime cases pass on
unchanged source. The three modal tests now reach and pass their actual actions.
Recovery controls remain correct during partial removal, unmount, chat replacement,
queued old events and app exit. All seven picker native ownership cases pass,
including repeated cancellation and borrowed connections. Independent review of
the three lookup substitutions found no actionable issue. No whole-Send gain is
claimed until the unprofiled integrated run completes.

The exact clean-dev registry-pass control confirms557/558 modules and
427289/427830 LOC (pr3534-preimport-pass-dev-baseline-1,37.89s). This establishes
the original DictionaryPicker deferral's marginal regression; the replacement
remains subject to the unchanged budgets on integrated source.

The unprofiled integrated run (pr3534-query-lookup-enter-and-budgets-1,395.15s)
records24.473/21.642/21.889s, all three turns saved and traces settled, with
unchanged source. Every sample still fails six seconds. All unchanged startup
budgets now pass:1040/1040 ui_ready,557/558 registry-pass modules and94/94
first-send warmup. Button feedback passes; Enter misses100ms at115ms wall with
93.75ms loop CPU. Keep that failure open, without changing the threshold.

Current periodic-only discriminator (pr3534-current-periodic-discriminator-1)
records14.464/14.706/12.182s with69/58/61 routine Console polls deferred until
adapter entry. Explicit FULL requests and all native/action authority stay live;
guards have no reported identity exceptions and callbacks restore. All turns
save and settle. This isolates only routine Console demand and cannot qualify
production latency; neither this run nor the quiet run meets the merge gate.

Extend Task3i with one additional existing presence-only query in session-tab
sync: untyped query_one/NoMatches for the workflow line, followed by the same
controller check and select_chat callback only when found. Keep all lock/await
rechecks and callback ordering. Do not change the left-rail multi-match loop.
The current first-feedback trace remains valid and misses narrowly at101.529ms
wall/78.125ms CPU: acknowledgement imports are resident, on_key takes0.708ms,
paint takes14.103ms including7.374ms tabs, and the natural compositor starts
at83.822ms. This motivates the same measured lookup cleanup, not a new scheduler
or weakened paint budget. Run the existing tab-lock/source/custom-adapter controls
and both original feedback cases on the changed source.

Task3i workflow lookup selection (pr3534-tab-lookup-feedback-1,236.52s):
both original feedback/input cases pass, at87.932ms Enter and95.554ms button;
pending USER frames are175.165/163.705ms. Both session-controller cases pass.
Four checks fail before their intended lifetime assertions: the screen-pump
fixture cannot establish its initial quiet state, and the three tab-boundary
observers reject the original replay worker's bound async method or overflow
their event cap. The current dev source already uses that bound-method worker.
Compare the quiet failure against dev; correct only the observer's valid worker
shape and add bounded failure diagnostics without growing time/event limits.
ADR required:no; these are test-oracle/diagnostic corrections, not lifecycle changes.

The periodic-disabled30-second py-spy capture (pr3534-pyspy-periodic-1) records
one19.3047s first Send on unchanged source, including a12.4880s saved-to-trace
interval. It defers93 routine polls while explicit FULL and native work remain
live. The sampler exits normally and restores its hook; the original scenario
then fails because the composer is still blocked before the second Send. This
is incomplete attribution only, not three-turn correctness or latency evidence.
Inspect the finite foreground work and the post-turn readiness state separately.

### Task3j — share the scalar paired-generation control observation

ADR required:no new ADR; apply ADR126's existing fresh scalar-decision rule,
already used by Task3g, to the generation-witness reader.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: `_paired_witnesses` calls the stock startup reader and then rereads the
same control records and registry for one synchronous decision. Use the existing
startup-from-controls helper only when its defining startup callback remains
stock. Custom or replaced startup callbacks retain their original route.

Shared preparation lane owns generation_witnesses.py and an off-tree focused
regression patch; root owns integration and all native runs. Keep source-scope
admission, every raw construction gate, required generation records and all later
fresh witness calls unchanged. Add no cache or retained authority across calls.

- [ ] Establish native RED for duplicate original control/registry observations
  in one decision; qualify separate decisions, paired histories and custom paths.
- [ ] Share only that decision's detached observations through the existing
  helper and preserve refusal/error behavior and custom callback signatures.
- [ ] Run original witness/history/source-drift controls and focused lint.
- [ ] Include this change in quiet integrated timings; the sampled startup share
  is under one second across two workers and is not an additive saving estimate.

Task3j disposition: deferred, no product change. The approved-history count
control proves2/2 observations instead of1/1 and both custom startup controls
pass. The ordinary test's namespace-exclusion fixture was invalid; its corrected
artifact remains external. Source review then finds a real policy distinction:
startup's _records omits activation-file reads and normalizes base-control
failures, whereas the later full read exposes specific activation failures.
RAG and recovery UI consume those distinctions. A direct shared full read would
change them. Preserve the proposal for review instead of adding retry or error
projection complexity to this latency fix.

### Task3k — remove preview permission forwarding scopes

ADR required:yes, bounded clarification of the existing checked-read contract.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: use the existing captured permission source and owned payload reader
for one preview observation; do not retain data or admission between reads.

The normal preview getter forwards through nested guarded load methods. Select
the existing _CapturedSources.read_permission_payload/_load_in_owned_scope
route only under its existing stock/currentness qualification. Add a private
keyword-only observe_publication=False option. The preview opts in to the same
permission-publication observation inside the original mutation fence; other
snapshot callers retain the default. Original custom fallback load already
observes and must not be observed twice. Preserve producer lifetime, optional
failure policy, final native check/receipt, source currentness and retirement.

Baseline lane owns off-tree patches for MCP/permission_store.py,
MCP/console_snapshot.py, Chat/console_tool_surface_preview.py and a distinct
focused test. Root applies tests first and owns native verification.

- [x] Native RED: one actual preview payload/observation with three forwarding
  scopes; pin the selected one-scope behavior only after native retirement.
- [x] Implement the opt-in observation and qualified preview call; retain all
  original construction gates, declared members and custom fallbacks.
- [ ] Verify observation, changed sources, cancellation, failed reads and late
  permission revocation through original controls, then integrated Send timing.

Task3k original native RED (pr3534-preview-forwarding-red-1,98.60s): enabled
and killed cases reach only the final3-versus1 scope failure after proving one
payload/observation, original result, two leases and physical retirement. All four
missing/corrupt/custom/default-publication controls pass. Source hashes match.

Task3i observer qualification (pr3534-tab-native-tripwire-and-quiet-1,149.24s):
all three actual tab-owner drift cases and the deliberate Windows native-I/O
tripwire pass with the4096-event cap unchanged. Windows metadata observation is
limited to its original warm-settings purpose; source/lease/descriptor and
current-owner retirement checks remain intact. The remaining screen-pump case
fails its pre-test quiet predicate on dev and integration. Its bounded receipt
shows an idle message pump and pending native readiness/hook workers, not a held
tab lock. Before changing production behavior, use the existing event-loop-only
wait for this passive predicate instead of repeated Pilot message-pump barriers.
Keep the exact predicate and ten-second deadline; qualify the real held-lock case
and existing maintenance-drain case. ADR required:no, test observation only.

Task3k integrated GREEN (pr3534-preview-forwarding-green-and-pump-1,309.57s):
all38 cases pass with source_changed empty and HEAD fixed. The eight focused
controls qualify one scope, one payload/publication, custom fallback, missing
and corrupt reads, default nonpublication, final native check after publication,
and refusal after source retargeting. The entire original preview preparation
suite preserves cancellation, late revocation and callback checks. Independent
source review finds no issue. Append the new file to admission lane2 without
reordering prior targets or changing deadlines; quiet integrated timings remain.

The same run qualifies Task3i's passive quiet wait: both the actual tab holder
against the screen message pump and original coalesced maintenance drain pass.
Only the test's wait mechanism changed; its predicate and ten-second deadline
remain. The Windows-only metadata calibration is a local native qualification;
it does not add a new Windows CI job.

### Task3l — group the cold sensitive-input assembly

ADR required:no new ADR; extend ADR126's bounded configuration-read assembly.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: FOLLOWUP29's original capture locates repeated configuration reads in
one synchronous raw-input miss. The13 DB getters already share one scope;
single-file, skill-trust and direct-child rules still acquire separately.

Root owns Utils/sensitive_paths.py and Tests/Utils/test_sensitive_config_read_scope.py.
Keep the first live _raw_inputs_key read and memo-hit route unchanged. On a miss,
construct the existing input tuple inside one config_participants.operation.
Keep all original getters, per-getter failure behavior and native checks. Perform
the original final live key check and memo publication only after the new scope
has physically retired. No new cache, dropped getter or moved key authority.
Custom getters execute synchronously under the same configuration lock contract
already used by the DB getter group. A failure cannot publish a partial memo.

- [x] Establish original native RED for one cold assembly's duplicate admissions,
  with result parity and actor-local restoration before the count assertion.
- [x] Group only the existing tuple and qualify original/custom/failure controls,
  changed environment and retirement refusal without enlarging budgets.
- [ ] Retain only after integrated timing and correctness checks.

Task3k quiet qualification still fails the required performance:
pr3534-preview-forwarding-enter-1 records32.222/13.468/12.971s;
pr3534-preview-forwarding-button-1 records15.310/13.526/16.133s.
All six turns save, reply and settle their traces; sources remain unchanged.
No sample meets six seconds. Whole-Send savings are not established by these
unpaired variable runs, and the PR remains unarmed.

Task3l RED (pr3534-cold-input-assembly-red-1,21.12s) fails only the count:
16 admissions exceed6, after output parity, actor-local restoration and current
memo-key checks. The changed-environment no-memo control passes. GREEN
(pr3534-cold-input-assembly-green-1,91.75s) passes52 targeted cases, with the
existing Windows symlink-privilege skip. All6 configuration-scope controls and
4 CI target contracts pass; source/head stay fixed. The count is now at most6.
The count test does not independently establish native HANDLE closure; the
existing config-operation retirement contract and separate lifetime controls
remain responsible for that guarantee. Independent review finds no production
issue. Scoped lint passes except three unchanged UI diagnostics outside this
change; the new preview test's formatting is corrected.

Regenerate the derived remote worker after the sensitive-source change. Its
existing import-closure check gives the expected RED naming only _raw_inputs;
add that lazy laptop-only builder to the existing refusal allowlist. Supported
remote callers provide their detached context and never enter it. Run the whole
remote-bundle test file, retaining its stdlib-only and byte-reproducibility checks.

Task3l bundle integration passes16 cases with the existing POSIX-only fcntl
skip in8.15s (pr3534-sensitive-bundle-green-1); source/head remain fixed.
Quiet Enter (pr3534-cold-input-enter-1,130.86s) completes all three saved and
settled turns at17.421/14.209/13.512s, source_changed empty. The six-second gate
still fails; the cold grouping has not established a whole-path saving.

### Task3m — retain completed promoted display publication

ADR required: yes (amend the existing bounded route).
ADR path: backlog/decisions/231-console-polling-and-full-state-reconciliation.md.
Reason: omit unchanged tab/transcript pipelines only after their original
successful publication while retaining current action, recovery and poll owners.

1. Correct transcript-gate fixtures to complete the mounted original pipeline;
   an intentionally swallowed downstream exception is not publication. Add
   failed/cancelled publication controls retaining the last completed gate.
2. Reproduce the unchanged promoted Enter journey repeating tabs/transcript after
   the real saved USER echo is mounted, under its existing observation bound.
3. Extract one shared complete transcript gate into Console_Modules/transcript.py;
   move token publication to actual success. Include current widget identity and
   visible citation counts, whose existing producer explicitly invalidates it.
4. Extend the completed FULL witness inside the existing composer/screen owners.
   Preserve original callback qualification, retained helper effects and every
   post-await currentness check. Recovery synchronization remains per-pass.
   Advance the existing fifteen-call backstop once and use the existing promoted
   long branch for its forced pass; short passes remain deferred.
5. Verify both actual input routes, missing/changed publication refusal, custom
   readers, settings/draft invalidation, FULL dominance, final/background handoff,
   feedback and retirement. Then run quiet whole-Send samples sequentially.

The integration owner applies and verifies all product changes. The UI lane
prepares an off-tree patch in chat_screen.py, composer_run_controls.py and
Console_Modules/transcript.py. The separate response-rule lane owns only
ResponseRuleStore.effective_rules. Neither lane runs native/app tests. Count
reductions do not establish the six-second gate; final saved-source measurements
must still qualify both Enter and Send. No generic invalidation system or new
cache/lifecycle owner is selected.

### Task3n — group each effective response-rule read

ADR required:no new ADR; apply ADR126 finite core-operation ownership within
ADR232's existing synchronous rule resolver.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md;
backlog/decisions/232-console-learned-response-rules.md.
Reason: qualified warm Send3 in pr3534-foreground-intervals-1 executes two distinct
ResponseRuleStore.effective_rules calls. Each enters three core operations and
three storage acquisitions for scope bindings; active rules require additional
revision reads. The calls span0.185578 and0.113843s. Their results remain separate
because a real await/effect intervenes; no cross-call cache is permitted.

Inside each existing self._lock, retain register_context first, then wrap only
the original binding/revision assembly and resolution in one existing
participants.core_operation(self.repository.db). Preserve individual SQLite
transactions, all public reads, precedence, KeyError handling, native currentness,
borrowed connections and return only after successful operation retirement.
No connection pool, combined transaction or rule verdict cache is introduced.
Root integrates; baseline lane prepares off-tree tests/product for review.

- [x] Native RED proves redundant actual acquisitions with empty and inherited
  active rules, output/precedence and owner-restoration checks.
- [x] Group the original synchronous body and qualify separate-call freshness,
  temporary/memory rules, borrowed handles, failure and existing write contention.
- [ ] Include the verified change in final whole-Send measurements; do not infer
  a six-second result from these small inclusive durations or counts.

Task3n RED (pr3534-rule-assembly-red-and-cold-receipt-1) reaches every
output/freshness/retirement prerequisite, then fails counts [3,4] and [6,5]
against [1,1]. Both custom-reader error controls pass. The four rule cases run
before the separate cold-receipt controls; source hashes stay unchanged.
Applied the reviewed finite core-operation scope and enrolled the new file in
the existing second admission CI lane; native GREEN remains pending.

Task3n GREEN (pr3534-response-rule-assembly-green-1,94.68s):52 pass across
the four new native assembly cases, original rule-store and write-contention
files and exact second-admission CI target contract. The count is one admission
for each fresh empty/inherited assembly; custom failures, separate-call mutation,
temporary rules, borrowed handles, physical retirement and existing cancellation
and write-contention controls pass. Sources/HEAD stay unchanged. Whole-Send
measurement waits for the separate UI implementation; no total saving claimed.

Task3m RED (pr3534-promoted-publication-red-1,66.86s) reproduces the repeated
promoted display pass after the real saved USER echo. Both failed/cancelled
publication controls see the token replaced prematurely. Six corrected mounted
gate/recovery positives pass. Two context fixtures still omit a downstream
status collaborator and fail before qualification; correct those fixtures
before using them as evidence. All product and test hashes remain unchanged.

Task3m fixture qualification (pr3534-publication-fixture-controls-1,58.09s):
all10 context cases now pass after supplying their original controller status.
The strengthened cancellation control identifies its exact task/future and the
original refresh coroutine for the held widget, then fails only the premature
token assertion as intended. No source drift. Citation invalidation retains a
separate original-producer control before product integration.

The original citation producer control passes in27.46s
(pr3534-citation-publication-control-1): while the exact original refresh waits,
discovery invalidates the old gate and requests FULL; the next original pipeline
mounts Cited sources(2). Retain that behavior after moving publication. Added
the gate file and two context controls to the existing second admission CI lane;
the two real promoted input journeys remain in the existing UI census.

Task3m integrated GREEN (pr3534-promoted-publication-integrated-1,126.17s):
26 pass across the four Task3n native cases, nine completed-gate/citation cases,
ten context cases, both actual promoted Enter/button journeys and the CI target
contract. The original saved echo, Sending, short-pass omission, natural
fifteenth-call backstop, custom/settings/draft refusal, across-await FULL and
provider/custody retirement all pass. Independent source review found no blocker
in the final7392C65A product packet. Source hashes and HEAD are unchanged.
Quiet Enter/button measurements follow sequentially; six-second acceptance is
still open. The separate missing-publication refusal supplement is pending.

Task3m/3n quiet qualification still fails the merge requirement. Sequential
pr3534-completed-publication-enter-1 reports15.428174/13.162604/12.198086s
(122.42s total run); button-1 reports14.938608/11.796831/10.885252s
(118.04s total). All six turns save/reply/settle and sources remain unchanged.
These are actual app actions to a controlled provider adapter, not physical
terminal or live-provider response measurements. Unpaired host variation means
no precise saving is attributed to the two changes. Keep auto-merge disabled.

The earlier foreground capture also spans action-to-save: qualified cold/warm
5.128610/5.820550s, dominated by action-to-controller3.544980/4.269331s.
Worker native activity overlaps that interval but lacks producer/awaiter links;
do not assign it to Send merely by time or thread. The next external diagnostic
adds original configuration/read producers and their issued-read/callback IDs,
plus routine UI and eligibility/capture counts. Existing source guards, local
event caps, retirement and adapter cutoff remain; no product callback, timer or
deadline changes. Inspect current route use and exact producer work before
selecting another consolidation. See presave-attribution.md and FOLLOWUP35.

Missing-publication supplement (pr3534-missing-display-publication-1,157.83s)
passes both Enter/button journeys with their original20s shared observation
window. Removing only completed display proof makes the original routine path
run FULL/tabs/config; actual successful publication requalifies the still-held
turn with its saved row mounted and zero provider calls. The selected task is
preserved with asyncio.timeout; wait_for(coroutine) would create another Task
and invalidate that observer identity. Product source is unchanged from the
quiet timing pair. This closes Task3m plan step5 without enlarging a deadline.

Cold-receipt checkpoint (pr3534-cold-receipt-checkpoint-1): native now passes
in47.25s with its original70s child limit. Mounted trust replacement reaches
MODEL but fails the preceding receipt-type prerequisite in61.94s. Current dev
713fa also fails that case, earlier at its first-entry wait (63.01s); it does not
reproduce the exact later assertion. Source comparison establishes the relevant
publication branches are unchanged. Builtin-only catalogs do not publish a cold
trust receipt, and Windows has no ordinary-tree bridge publication. The fixture
seeded a managed skill only for native_managed. Extend that existing unapproved
managed-skill prerequisite to mounted_queued and mounted_trust_replaced so those
receipt-dependent cases actually exercise the original cross-platform publisher.
Keep all receipt identity, queued replacement/refusal and70s assertions unchanged.
ADR required:no, fixture prerequisite only under the existing publication contract.

Current integrated feedback/retirement qualification
(pr3534-completed-display-feedback-shutdown-1,199.13s): all15 cases pass,
including actual Enter/button acknowledgement, Waiting for reply/Streaming,
stopped-draft retention, repeated cancellation and physical native cleanup.
The original headless run-feedback frames are88.803/71.022ms; pending-user
frames are147.763/137.353ms. These are supplied compositor frames, not physical
terminal measurements. Source hashes and HEAD stay fixed. This establishes
feedback and retirement on the Task3m/3n source; six-second dispatch remains open.

### Task3o — validate a nested installed MCP source once

ADR required: yes, amend the existing finite raw-source contract.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: the exactly linked initial read_sources worker spends2.704/4.023s on
warm observations, including three nested catalog entries. Those entries each
perform check -> selection -> check on the same installed source: three fresh
binding witnesses and two parent checks before one unchanged method body.
This is repeated validation, not three JSON reads or three new acquisitions.
Instrumented spans include scheduling; they do not establish removable latency.

1. Retain the first original full check in raw.scope. Qualify only an installed
   generation-selected MCP participant, the exact same source/route and native
   path, no explicit selected_read, and a boolean writing flag.
2. Recheck resident participant/source/selected identity and live_state under
   the existing coordinator. The just-checked participant selection must equal
   the raw state's selection; enforce the requested write scope and all leases.
   Then join the existing operation without the two duplicate binding reads.
3. Keep original method bodies, custom/uninstalled/history/explicit-selection
   fallbacks, effect-time file checks, final checked-read validation and physical
   retirement. Do not cache a witness or extend any native owner.
4. Establish original-function RED counts after real payload/custody/retirement
   prerequisites, then qualify source/path/recovery drift, write and participant
   refusal, accepted pause behavior and existing native cleanup controls.
5. Run quiet Enter and Send samples sequentially on the integrated result.
   This small shared-boundary change supersedes reconsidering the rejected
   approximately213-line initial-catalog bypass (OPT28/73); that bypass stays
   rejected. Keep the six-second merge requirement unchanged.

Root owns integration and all native runs. The admission lane supplies only the
raw_participants.py patch; the baseline lane supplies off-tree tests and source
attribution. Initial source review supports one fresh witness/parent check per
join, with unchanged scope, acquisition and payload-read counts.

Task3o RED2 (pr3534-mcp-nested-join-red-2,14.45s) reaches all real payload,
same-operation/no-new-admission, parent-FD, exact-lease and native-hold retirement
prerequisites for local/context/permission sources, then fails only the final
3-witness/2-parent count against1/1. Sources and HEAD remain unchanged. RED1 was
an invalid retirement prerequisite: the configuration fixture still owned its
startup lease. RED2 uses the existing test suite's actual startup-owner close
after seeding, before the measured operation; physical cleanup assertions remain.
The reviewed26-line shared-boundary patch is integrated and the combined ten-case
file is enrolled in the existing second admission CI lane. Native GREEN pending.

Task3o first integration run (pr3534-mcp-nested-join-green-1,105.10s):36 pass,
including all ten new cases, and two older pause controls fail only their
nonempty acquisition observer. Source hashes are unchanged. The observers wrap
recovery_activation.acquire_storage, which construction-local admission no
longer calls; earlier payload, pause, drain and retirement assertions pass.
The same original two controls pass on dev713fa in34.14s
(pr3534-dev-mcp-pause-controls-1). Observe the original acquisition function body
on the exact issuing source/thread instead, retaining the nonempty and no-late-
acquisition assertions. This repairs test evidence for the changed route; it
does not loosen the pause contract. Whole-Send measurements use unchanged
production code while that separate observer correction is prepared.

Task3o quiet measurements still fail: pr3534-nested-entry-enter-1 records
20.856190/16.715343/16.122192s (161.74s run), followed sequentially by
button-1 at16.130445/14.254132/12.016223s (125.97s run). All six real app sends
save/reply/settle with unchanged sources and full private defaults. These are
headless action-to-controlled-adapter timings, not physical terminal/live-provider
response measurements. No improvement is claimed: the preceding unpaired samples
were lower. The reduced proof count alone does not meet the owner’s six-second
requirement. Keep the PR unarmed and investigate the linked post-save owners.

Task3o final targeted integration (pr3534-mcp-nested-join-integrated-2,107.41s)
passes all38 cases after the observer correction. Original function-body
observation covers both acquisition aliases, direct constructor calls, the exact
worker and main-thread maintenance queries. Nonempty/no-after-pause assertions,
payloads, source refusals and native cleanup all remain. Sources/HEAD stay fixed.
This qualifies correctness and repeated-proof reduction, not dispatch latency.

### Current-dev checkpoint verification (2026-11-16)

ADR required: no. ADR path: existing ADR126/ADR230.
Reason: correct fixture prerequisites for existing configuration and shutdown
contracts, without changing product authority or timing thresholds.

The branch is rebased onto dev71e49709 at8ebf1b93 and published, with auto-merge
off. Census2149 passes. Integrated feedback/shutdown gives15 passes and one
queued-reconcile flag failure after50ms. Exact dev cases pass; sparse original
callback observation on this branch also passes both parameters with unchanged
sources and restored monitoring. Neither pass explains the earlier failure.
Make the callback-safety test explicitly deliver its actual queued callback
through Textual after shutdown-owner entry, using the existing two-second
readiness budget and unchanged50ms observation/assertions. This qualifies the
shutdown interleaving, not natural compositor latency.

Current CI finds17 Tool Recovery failures at a configuration-source mismatch.
The same listing and flush controls pass on dev; branch controls refuse the
retargeted collection-time source. Retain the existing private bootstrap profile
only for real guarded consumers, preserving tmp_path workspaces, original
assertions and fail-closed product behavior. The leased-cat control independently
fails on Windows dev while reading a CRLF file; do not attribute that failure to
this branch. Verify the corrected files and affected shutdown controls together
after their independent checks, then publish the qualified corrections.

Post-save diagnostic1 completes all three turns in136.45s with fixed sources.
Action-to-adapter18.922/14.852/17.467s is instrumented, not a release timing.
Foreground bodies are observed, but the first observer does not link the
production FreshContextExecutor wrapping of to_thread or eager retained tasks.
Do not infer worker causality from proximity; qualify those exact existing
edges before using worker intervals to select another consolidation.

Checkpoint fixture verification: the three tool files plus shutdown produce
86 passes/2 failures in57.70s. All17 hosted configuration failures now reach
their intended assertions. The remaining leased-cat failure is also on dev
(native file metadata/length validation); no product change is selected here.
The other failure is the newly explicit initial no-pending-layout precondition:
the native tree can schedule another reconcile after the shared three pauses.
Wait for that real callback to retire, bounded separately at2s, before capturing
the callback under test; never clear its flag manually.

Perf Guard's compiled source adapter lacks the new negative display-publication
seam and claims every query_one selector exists. Supply only the absent-proof
result and reject absent widgets, without inventing a runtime or positive proof.
Its complete source-ownership file plus all shutdown cases pass27/27 in20.12s
(pr3534-perf-shutdown-fixture-green-1). All existing assertions remain. The50ms
shutdown observation now starts after confirmed delivery, so it tests callback
safety rather than natural scheduling latency. Scoped F lint and diff check pass;
independent review finds no blocking issue. Whole-Send speed remains unqualified.

Post-save diagnostic2 completes all three turns in157.32s with fixed sources.
The corrected observer links8/9/8 selected workers and three eager retained tasks
per Send with no missing/ambiguous handoffs. Cold collection integrity passes;
warm unmatched/reused spans still limit whole-phase attribution. Instrumented
action-to-adapter20.552/15.014/19.896s is not a quiet release measurement.

Checkpoint follow-up: hosted Tool Recovery reaches its next step after the 17
original fixes and exposes 125 configuration-selection failures in Git/review
fixtures. Bind a real per-test selected config via the existing support helper;
keep independent HOME/XDG and use that helper again in the two deliberate
source-reselection controls. One shared fixture boundary per module replaces
per-test decorators. Native verification pr3534-git-config-fixtures-green-1 has
37 passes and four Windows failures reproduced unchanged on dev in
pr3534-dev-git-platform-controls-1: POSIX path expectation, absent killpg,
POSIX-script execution and a 30s truncated-diff subprocess timeout. Review-source
native setup separately lacks symlink privilege on both branches; hosted Linux
must verify its complete suite. The three representative Git controls fail on
the unpatched branch and pass after correction. F lint and diff check pass.

The fixture-only correction is committed as 74e66ed713. Hosted Python 3.12 and
3.14 both pass the complete Git/review step; their next step is under review.

### Task3p — one explicit acquisition for a worker's declared sources

ADR required: yes.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: consolidate the measured five independent acquisitions at worker entry
without turning primary-only related paths into implicit execution authority.

The exact ledger/provider worker prefixes spend more than 99% of their observed
body-entry delay inside native admission. Existing related_paths already checks
the complete selected set before/after counting one native hold, but grants an
execution selection only for its primary. Introduce an explicit grouped entry
that records and verifies every declared member's selection. Keep ordinary
acquire_storage and its primary-only companion contract unchanged.

1. Add a focused native regression: five real declared sources use one counted
   acquisition, every member obtains its actual execution context, unrelated
   paths refuse, and physical exclusion/retirement remain observable. Run RED
   against the current shared guard before applying the implementation.
2. The storage lane prepares _acquire_execution_storage(paths, attempt): one original
   _Acquisition owns member-selection observations, existing related-path
   admission, final selection/liveness/pause checks and all failure cleanup.
   Store immutable member selections on the existing exact StorageLease;
   its execution_context still re-observes the requested member and live hold.
3. Root integrates the shared RecoveryAdmissionGuard: pre-acquire only distinct
   missing paths as one group, register one close, and keep the existing ordered
   per-source admit_source/execution_scope/finalize calls. A single missing path
   retains acquire_storage. Keep the group's _Acquisition pending through the
   original owner loop; check at the former new-path boundaries and again for
   every missing member, then retire it before publishing ExecutionState or
   yielding payload. This preserves refusal of a pause between approvals.
   Nested or transferred states retain their parent leases; a child owns only
   its new group. No payload runs before all checks.
4. Qualify undeclared/retargeted paths, selector/recovery drift, member scope,
   pause during acquisition, refusal between owner approvals, nested/transfer
   lifetime and failure/uncertain-close cleanup. Keep per-owner control reads
   fresh; no cached authorization, source-scope bypass or extended worker life.
5. Run affected Agent/MCP/RAG admission controls and final quiet Enter/button
   measurements sequentially. Counts do not establish the six-second gate.

This deliberately moves admission of all missing sources before the first owner
approval. It may do additional read-only preflight before an earlier owner would
refuse, but does not permit an effect or bypass an approval. Preserve the family
exception contract. Root owns integration and every native/app/test run; storage
and verification lanes supply independent off-tree patches with distinct files.

Native RED (pr3534-execution-group-count-red-1, 14.78s) proves all five real
member contexts, payload, physical exclusion, pending-owner retirement and
maintenance reopening before failing the final count: five acquisitions/tokens
instead of one. Native GREEN on the integrated storage/runtime implementation
passes all 13 count/runtime/per-owner controls in 69.01s, with fixed sources and
HEAD. The group retains one pending preflight through approvals and verifies
its exact process/thread/task owner before issuing member authority. Existing
between-owner pending and paired-history-loss refusals remain green. Full
group-specific negatives subsequently pass in
pr3534-execution-group-negatives-1: 48 passed, one POSIX-only skip in 207.85s,
with unchanged sources and HEAD. This includes all 23 new group controls,
ordinary related-path refusal, remote bundle validation and both bundle-size
ratchets. Affected family controls and final quiet button verification remain
pending.

Task3p first quiet Enter run (pr3534-execution-group-enter-1, 136.42s) completes
all three saved turns at 16.050/14.417/15.480s, with fixed sources and full private
defaults. These headless action-to-controlled-adapter timings still fail the
six-second gate; the acquisition-count improvement is not sufficient.

### Task3q — qualify direct fresh Windows security-descriptor reads

ADR required: yes.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: replace a native descriptor retrieval primitive while preserving the
current conservative owner/ACL projection and per-call freshness contract.

History review finds Claude's 4921fadd28 native-query implementation and
158cb6992a notification reuse on the earlier integration line, but neither is
an ancestor of the actual PR3050 merge 588389cfe2. The October31 reconciliation
explicitly ported selected behavior while keeping current-dev native admission.
Historical claims that its native optimizations merged overstate the surviving
tree. No specific recorded rejection of the descriptor retrieval was found.
Its old speed measurements cannot establish current performance.

1. Reuse only the fresh NtQuerySecurityObject retrieval idea. Preserve the
   CURRENT _decode_security/_acl_mode semantics, byte-only decoding cache,
   descriptor-size limits, native handle birth/retirement and all source gates.
   Do not replay old token-owner, snapshot-tree or change-notification changes.
2. Establish a native RED count against the current security method after
   same-handle GetSecurityInfo/current-decoder projection and actual cleanup
   prerequisites. Add live file/directory and ACL-mutation parity controls.
3. Query OWNER|DACL into a bounded call-local buffer; bounded resize retries
   handle changing descriptor length. Require valid self-relative bytes before
   decoding. Keep ordinary large-descriptor decoding uncached and metadata
   refusal above the existing 4096-byte bound. All errors remain fail closed.
4. Adapt existing acquisition/free test doubles and native call observers to
   the actual primitive while preserving descriptor freshness, exact handle
   identity, native close/uncertain-retirement, error and projection assertions.
5. Root integrates the independently prepared windows_files.py and test packets;
   native runs remain sequential. Run existing security/metadata/path controls,
   then quiet full-default Enter and button sends. Retain the six-second gate.

Microsoft's documented user-mode name, self-relative return, READ_CONTROL access
and 64KiB NTFS descriptor bound support the API contract:
https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/nf-ntifs-zwquerysecurityobject.
Actual projection equivalence and performance still require current native tests.

Native primitive qualification: pr3534-security-query-red-1 (10.48s) passes
same-handle projection, descriptor-allocation retirement and positive native
handle-close checks, then fails only the intended final count: zero direct NT
queries and three GetSecurityInfo calls. The identical test passes with the
replacement in pr3534-security-query-green-1 (3.15s), observing three fresh NT
queries and zero GetSecurityInfo calls. Full ACL/size/error controls and complete
quiet sends remain pending; this count evidence alone does not qualify merge.

Separate hosted-CI correction 82c1a57863 shortens only the sensitive-database
accessor's explanatory docstring and regenerates the remote bundle. Both existing
bundle-size ratchets, bundle tests and F lint pass; its size budget is unchanged.

Complete quiet Task3q measurements still fail: Enter
15.577/11.852/13.677s (pr3534-nt-query-enter-1, 130.67s), button
15.228/16.161/13.994s (pr3534-nt-query-button-1, 133.47s). Every turn saved,
replied and settled; sources/HEAD remained fixed. These runs do not establish a
causal whole-Send gain. Current quiet Enter stages place controller entry at
3.168/2.962/4.047s and save completion at 5.064/4.415/5.542s; trace reservation
starts at 15.003/11.416/13.128s. Repeated native ancestry work remains under
investigation; no cross-owner permission cache is selected.

Native security/metadata qualification passes all 75 cases in 7.14s
(pr3534-security-query-controls-2). The first run had 71 passes and four new
file-fixture setup failures. A repeated original failing case with locals proves
SetSecurityInfo strips the creation ACL's OI|CI flags on files while preserving
owner, revision, masks and trustees. The fixture now establishes its explicit
noninheriting file ACL before the existing exact-restoration helper, asserting
projection equality across that setup. Shared restoration checks and product
code are unchanged. Native large-ACL resize and family integration remain pending.

Family comparison: pr3534-execution-family-controls-1 has four passes and eight
failures in 122.28s. The eight unchanged selectors also fail on clean dev71e in
pr3534-dev-execution-family-controls-1 (78.72s). MCP/RAG common setup emits
unescaped Windows paths in TOML, so those cases fail before family execution.
The Agent baseline stops at the same malformed fixture class; the candidate
already has that spelling corrected and exposes an unsynchronized request for
cancellation before native entry. Its comparison needs byte-identical corrected
fixtures, not a claim that the two failures prove the same behavior. MCP's native
worker cancellation case already waits for actual entry and still needs separate
analysis of the pre-entry timing failure. No product failure is waived.

The first native leaf census (pr3534-native-leaf-costs-1, 203.95s) completes all
three turns with original bindings/callbacks restored and all 14 targets found.
Its overall collection is invalid: detailed caller-chain and unfinished-span
caps overflow, and Python3.12 lacks local unwind events for expected missing-path
exceptions. Do not use its inclusive times as latency attribution. Uncapped
original-body start counts show 586/693/614 registry reads and 701/814/730
control-record reads across worker threads during the instrumented send windows.
A narrower two-reader start-count census will identify their consumers without
low-level duration/exception bookkeeping. No new product optimization is selected
from the incomplete trace.

### CI boundary correction — configuration owns the finite read scope

ADR required: yes (cross-module API ownership).
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: the finite assembly is correct in behavior, but Utils must not import
the domain-layer Backup_Recovery participant directly. Hosted architecture
verification identifies both new imports in sensitive_paths.

Expose one synchronous `config.configuration_read_scope()` facade over config's
existing participant owner. Replace only the two direct imports/with boundaries
in sensitive_paths; keep custom getters, native checks, exception propagation
and cache publication ordering unchanged. Do not change layer allowlists.
Verify the layer contract, the original finite-read native regression and its
failure/custom-source controls. The facade must not span awaits or provider IO.

### Task3r — register the existing display reader at composition

ADR required: yes (cross-module resource ownership).
ADR path: backlog/decisions/231-console-polling-and-full-state-reconciliation.md;
existing ADR356 forbids implementation-identity authority.
Reason: hosted CI exposes new code/default introspection, and the mounted
original-source regression proves a transparent callback wrapper incorrectly
disables current display reuse.

1. Keep ConsoleContextReadSnapshot as the sole cache. Compose one display reader
   with the actual Context/spend controller; bind it once to that controller and
   screen. A normal read records its actual snapshot, config projection and visit.
2. A refresh-free read requires that same registered controller/resource pair,
   current attachment, controller/store/session and current published snapshot.
   Recheck after reading; copying records cannot recreate registration. Preserve
   custom constructors without a resource and their ordinary callback route.
3. Remove this branch's code/default identity gates. Replace their mutation-count
   tests with resource/currentness, transparent-wrapper and copied-owner controls;
   retain existing FULL, configuration-read, feedback and lifecycle requirements.
4. Defer Task3g startup read reuse: retain the public startup_permission callback
   followed by fresh activation observations. Do not move its identity checks
   into another helper or raise architecture pins. Keep the shared evaluation
   helper required by storage admission. Record this deferral in the review list.
5. Run the mounted regression and existing affected controls sequentially; then
   qualify package identity/layer and existing projection-size ratchets. Count
   and ownership evidence alone cannot establish the six-second merge gate.

Original-source RED pr3534-context-display-resource-red-1 fails only the final
same-value assertion after transparent wrapping (58.10s): original publication,
zero config/live-policy calls and no scheduled work all pass. Sources and HEAD
remain fixed throughout that run.

Integrated ownership/facade qualification passes53 cases with one POSIX-only
skip in197.50s (pr3534-context-owner-and-scope-green-1), fixed sources/HEAD.
This includes both mounted display-resource cases, original finite config-read
and failure controls, identity/layer/ChatScreen-size ratchets and regenerated
worker bundle integrity/size. Existing FULL/config/feedback controls remain next.

The narrower original-body read census is valid
(pr3534-control-read-counts-1,176.83s). One complete warm action-to-adapter window
contains746 control-record and629 registry reads across main and worker threads.
224 acquisitions each read both inputs before and after counting, accounting for
448 of each total. Consumer chains include53 SQLite opens and14 readiness-config
assemblies. These are whole-window start counts, not per-call timestamps, paired
invocations or evidence that each read is on the foreground/post-save path.
All recorded phases remain within caps and restore original monitoring bindings.
Select further work from caller/lifetime evidence rather than assigning these
aggregate counts a causal latency share.

### Task3s — readiness uses the existing publication coordinator

ADR required: no new ADR.
ADR path: existing backlog/decisions/231-console-polling-and-full-state-reconciliation.md.
Reason: correct one publication caller that bypasses the existing cadence owner.

The successful readiness refresh directly schedules FULL, including while an
original FULL awaits that same read. Context and appearance snapshots already
publish through config_sync.publish_console_presentation. Establish a causal
regression using the original readiness refresh and cadence/settlement hooks;
then route this single completion through that existing helper. Keep all native
reads, currentness, expiry, settings convergence, cancellation and physical
retirement unchanged. Verify later changed publication, stopped-poll settlement,
independent decision state and the original idle custom callback. Retain mounted
FULL/config/feedback controls; this scheduling test cannot establish native speed.

The execution-count fixture also probes its actual retained namespace. An
ordinary unbound lease does not claim the unrelated registered profile namespace;
the old hard-coded maintenance probe was not evidence of exclusion. Preserve
all original count/refusal and lifetime assertions and deadlines, requiring
actual expected namespace membership plus full retained execution-context equality.

Task3s RED is valid in pr3534-acquisition-publication-red-1 (26.79s combined):
the original finite read retires and the awaiting projection renders before the
final assertion finds one redundant FULL. The idle custom-callback control passes.
Native Windows acquisition separately passes real payload, hold exclusion and
retirement before observing2 records/2 registry/2 scope derivations and no evidence
reuse. That count establishes opportunity, not the safety of metadata-only reuse.

CI composer-size correction (existing ADR231; no new ADR): move the four resident
FULL source/publication observation helpers into console_spend_projection, retaining
custody/preparation eligibility in composer_run_controls. Keep dependencies lazy,
all original predicates and defining-module references, and the existing size
ceilings. Qualify preparing/FULL, compiled-source, response-rule and observer
controls. This mechanical ownership correction makes no speed claim.

### Task3t — fresh Windows acquisition derivation reuse

ADR required: yes.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md,
Fresh Windows acquisition evidence amendment.
Reason: qualify existing hold-local proof with complete fresh Windows observations.

1. Retain the meaningful native count RED and production one-second settling rule.
   Qualify native same-size edits with suppressed timestamps and newly added
   directory entries with restored times against original full refusal.
2. Add one platform observation seam to existing _Evidence. Windows uses the
   current checked batch for metadata/exact SD plus bounded fresh file digests and
   sorted directory membership. POSIX keeps its original observations. Reuse
   existing native volume queries and directory enumeration, no duplicate owners.
3. Couple every finite Windows observation to an actual serving-hold StorageLease.
   Positive retirement releases it; uncertain native close retains that token and
   exact failed births, refuses fallback, and preserves native exclusion. Keep
   metadata refusal separate from existing SQLite close-failure recovery.
4. Enable only the acquisition branch after two original positive confirmations;
   keep _ordinary_hold Windows-excluded. Preserve original dependency, source,
   epoch, namespace, pending/count, pause/cancellation and final currentness gates.
   Unsupported/oversized evidence uses original derivation after proven retirement.
5. Independently review frozen product/tests, then run targeted native unchanged
   counts, content/membership/ACL/path/epoch changes, limits/volume, cancellation,
   uncertain-close and existing metadata/runtime/group controls. Existing POSIX
   evidence checks remain required on supported CI platforms.
6. Run quiet actual cold/warm Enter and button measurements sequentially, then
   final integrated feedback/shutdown checks. Every sample must be below six
   seconds before the PR can be armed or merged; no count replaces this gate.

Readiness and mechanical-owner qualification passes41 targeted controls in64.70s
(pr3534-readiness-controls-and-notes-reason-1). Its separate Notes diagnostic
fails with the original explicit writable_filesystem_unsupported reason: the
Notes adapter is intentionally read-only on Windows. The Linux config-source
fixture correction still needs hosted verification; no product guard is changed.

Integrated mounted verification pr3534-promoted-owner-integration-1 passes all6
cases in173.87s: original Enter/button promoted preparation, inherited defining
source fallback, both display-resource controls and response-rule close/resume.
The hosted Enter timeout is not reproduced on these integrated sources; its local
case completes in54.10s with the original deadlines. This is functional evidence,
not a quiet Send timing result or proof of the prior hosted timeout's cause.

Matched corrected family fixtures pass7/8 on both current product and dev71e;
the remaining MCP worker-cancel case fails before entry on current and times out
on dev. Preserve that unresolved case. Correcting the execution namespace probe
then passes all7 startup/next-owner/refusal controls in
pr3534-notes-execution-fixtures-green-1. Its8 Notes failures are separately
identified as unsupported Windows writes, not waived as passing controls.

The native timestamp/content premises pass both original cases in23.55s
(pr3534-windows-content-premises-1): a same-size registry edit with suppressed
timestamps and a new pending entry with restored directory times leave the old
metadata/SD projection equal, yet the original full acquisition correctly refuses.
Restoration, real payload and physical retirement also pass. This rules out
metadata-only reuse; it does not qualify the proposed fresh-content path.

Mounted feedback/shutdown verification passes11 cases in259.66s, including both
input routes'100ms feedback and complete Waiting/Streaming journeys, both shutdown
variants, terminal settlement and transcript publication controls. Two inherited
FULL replay observers incorrectly require Worker._work to be a coroutine, although
both actual producers pass a bound async method. Correct only that observer to
require the original bound method and exact screen; all37 assertion ASTs and5s/15s
deadlines remain unchanged. Both cases then pass in72.39s after a conflict-free
rebase onto dev3917e2d0cb (pr3534-rebased-replay-observer-1). Product code is unchanged
by that test correction. New quiet timings and final integration remain required.

Task3t review rejects new callable/code identity gates. Call the configured
platform capability directly, declining unsupported optional evidence to the full
route only after positive retirement. Keep actual hold/filesystem/currentness
checks; do not raise identity ratchets or attest private implementation helpers.

Quiet rebased measurements remain above the gate: Enter16.539/11.812/12.932s
(pr3534-rebased-ready-enter-1,126.27s) and button13.708/11.782/10.263s
(pr3534-rebased-ready-button-1,105.33s). All six turns save, reply and settle;
full-default profiles and source/HEAD remain fixed per run. This does not establish
a causal improvement over prior unmatched runs. Continue Task3t qualification.

Incoming dev's two raw-shell ownership cases failed in hosted fixture setup before
shell behavior because the isolated config was not selected. Selecting that
fixture's actual config before real WorkspaceDB/registry construction preserves
all assertions and makes both cases pass in8.53s on rebased Windows
(pr3534-rebased-raw-shell-fixture-1). Hosted confirmation remains required.

Task3t initial qualification passes83 cases in101.51s
(pr3534-fresh-evidence-initial-1), with source/HEAD fixed: the unchanged original
count regression now observes no control/registry/scope re-derivation for a settled
bound acquisition, while observing fresh native batches, actual exclusion and
retirement. Both timestamp premises, existing metadata/security controls and
mounted/unmounted Context ownership controls pass. Independent review required
complete read length equal to same-handle size and a final acquisition check after
optional confirmation I/O; both are present in the qualified candidate.

The11 additional native lifetime/fallback controls pass in92.19s
(pr3534-fresh-evidence-lifetime-1). They cover changed contents/membership after
confirmation, cancellation after available/unavailable confirmation, incomplete
reads, retained live exclusion on uncertain close at all three observation phases,
readonly volume, optional descriptor refusal and bounded aggregate content.
The EOF oracle permits harmless stale epochs but requires the original full
derivation on all three attempted acquisitions; cache eviction is not the contract.

Quiet candidate Enter remains above the gate at11.227/9.896/8.264s
(pr3534-fresh-evidence-enter-1,94.88s). All turns save, reply and settle. Do not
rerun the button merely to repeat a known failing gate; measure both routes after
the next material improvement. Current two-reader census is valid
(pr3534-fresh-evidence-read-counts-1,91.99s), with no lost events or unrestored
bindings. Warm send2 contains388 control-record starts and278 registry starts;
all registry work is on worker threads. These remain whole-window counts, not
foreground durations or a causal whole-Send latency attribution.

CI compatibility correction (existing ADR231/356; no new ADR): the new display
reader captured an attach visit before mount, breaking the unchanged original
Context wiring test with NoActiveAppError (pr3534-unmounted-context-red-1,51.29s).
Capture optional reuse registration only for the actual installed snapshot on a
mounted, attached screen; always retain the ordinary factory/input callback route.
That unchanged test and both mounted resource controls pass in the83-case batch.

CI first-Send import correction — ADR required:no; ADR path:N/A. The dev rebase
combines two previously independent warm-up additions and exceeds the unchanged
94-module limit at95, reproduced natively in pr3534-patch-import-budget-red-1
(19.61s). Move the optional patch parser and its limits from the workspace executor
and protocol module imports into their existing patch-only branches. Keep every
parser, validation condition, constant and normal Send warm-up entry unchanged.
Verify the original budget, ordinary/harness/library/review no-UI-import controls,
and local/SSH patch parsing and size/target checks. This mechanical lazy-import
correction preserves existing boundaries and cannot itself qualify six-second Send.

### Task3u — one fresh controls assembly per activation caller

ADR required:yes.
ADR path:backlog/decisions/126-complete-local-backup-and-recovery.md,
Explicit same-call activation controls amendment; existing ADR356 applies.
Reason: consolidate repeated validation behind an explicit shared API and replace
the withdrawn implementation-identity customization mechanism with deny-only input.

1. Preserve the qualified current census. Its warm window has86 activation calls
   and47 paired-witness calls with duplicate startup/control reads; acquisition
   contributes only12/12 of388/278. Do not optimize unrelated bound-path batching.
2. Establish real ordinary/recovered count REDs with actual permission, payload,
   native exclusion and physical retirement before the final count assertion.
3. Add read_activation_controls with one fresh full controls/registry observation
   and the existing startup decision. Return local facts or None; no cache/owner
   framework. Run an optional explicit deny-only restriction before fresh reads.
4. Consume those facts in each activation owner and paired-witness call. Forward
   restriction through execution_scope and, if exposed, bypass witness derived
   reuse/publication for that call. Keep invalid-owner validation, source scope,
   native leases and all live required/allowed checks unchanged. Keep standalone
   startup_permission unchanged, including its permissive inspection contract.
5. Replace code-mutation customization tests with explicit denial, raising and
   Allow-then-mutate controls. Retain between-owner pending/pair loss, recovered
   approvals, shared roots, source drift, cancellation and retirement assertions.
6. Use distinct product/test file ownership and independent source review. Root
   integrates and runs all native tests/timings sequentially. Verify the shared
   consumers, relevant architecture guards and current CI, then quiet cold/warm
   Enter/button sends. Merge remains forbidden until every sample is below6s.

Task3u original-source RED is qualified in pr3534-activation-controls-red-1
(4 failures,30.19s): both ordinary and approved execution observe5 controls/4
registry reads; both paired-witness cases observe2/2. Each fails only its final
count assertion after actual payload, native exclusion and physical retirement.
The unchanged count cases now accompany the independently reviewed shared-reader
candidate. This is not yet a whole-Send speed result.

The optional-parser fixture follow-up passes14 native cases in103.21s
(pr3534-patch-review-fixtures-green-1): real review-send child databases close after
asyncio's executor retires, and executor fixtures select their actual isolated
configuration before use. Matched clean dev3917 reproduces the two child cleanup
failures but passes the three executor cases, so only the latter interaction is
introduced by this branch. No cleanup exception or assertion is suppressed.

Current-head CI also exposes a three-line config size overrun after the dev
rebase. Condense the existing lock-probe docstring from seven lines to four;
config.py is8837 lines and its executable AST is unchanged. Retain the8837 ceiling.
The workflow selection correction adds the six new regression files and restores
the missing execution_source_groups entry in the exact selection contract;
timeouts and all other workflow semantics stay unchanged.

Task3u native qualification passes21 cases in125.82s
(pr3534-activation-controls-green-1), including the unchanged four count cases,
explicit veto/error and Allow-then-change checks, next-owner freshness, original
callback signatures, physical retirement and permissive startup inspection.
Quiet Enter remains above the gate at12.698/9.372/7.847s
(pr3534-shared-controls-enter-1,108.38s), with fixed source and HEAD. The later
instrumented run measures7.844/5.494/4.829s; it is diagnostic, not quiet acceptance
or causal proof of improvement. All three original adapter windows are bounded,
with no observer errors/overflows and restored bindings. Inclusive spans overlap.

Integrated verification pr3534-shared-controls-integration-1 passes411 cases in
278.02s: whole execution-source-group ownership/cancellation/retirement controls,
exact workflow selection and node resolution, unchanged module-size and identity
ratchets, and the second pre-mount Context failure reported by hosted UI20.
All changed/new Python files pass Ruff F checks; git diff --check is clean.
Native Task3t lifetime controls and Task3u activation controls remain qualified
as recorded above. POSIX derived-witness behavior still requires hosted CI.

### Task3v — avoid loading backup catalogs to read a restore marker

ADR required:no.
ADR path:N/A; existing ADR126's read-only marker and native storage rules apply.
Reason: mechanical dependency correction preserving public APIs, SQL, refusal
semantics and actual storage ownership; no new cache or service boundary.

1. Preserve the current diagnostic: first Send imports Personal_Context.recovery
   for the no-secure-keyring marker lookup; its inclusive import is0.541s, with
   DB.recovery_operations0.536s. The marker reader does not use backup catalogs.
2. Add a fresh-interpreter regression exercising actual unavailable-keyring
   status and non-creation of an absent database before its import assertion.
3. Move the existing marker reader into repository_keys beside its caller;
   retain the recovery module's public import and identical readonly SQL/error
   behavior. Do not alter backup adapters, encryption or key selection.
4. Verify the original regression, restored-key reason/readonly cases, relevant
   backup behavior and static boundaries, then repeat quiet Windows Send.
   Keep all native runs sequential and the six-second merge gate unchanged.

Task3v original-source regression fails only the final import assertion after
actual locked status and absent-database non-creation pass (4.61s). The moved
reader has identical AST except the existing equivalent marker constant name.
The original regression, restored marker reasons across restarts, readonly keyring
fallback cases and updated CI selection pass150 cases in27.41s. Quiet Enter then
measures6.276/5.742/4.696s (pr3534-marker-quiet-enter-1,61.43s), with source and HEAD
fixed. Cold Send still misses six seconds; no release gate is satisfied.

GitHub reports the published a676555a66 checkpoint conflicts with current dev
5e3c8b2bd4 and has scheduled no new pull_request runs. Preserve a local checkpoint,
rebase onto current dev, resolve the actual conflicts and reverify affected
boundaries before pushing. Auto-merge remains off throughout integration.

The rebase onto5e3c8b2bd4 is complete atb34dc71753. Only the UI census conflicted:
retain dev's2147 entries in order, then append the three PR nodes and set the
measured floor to2150. The census checker and142 CI contracts pass (28.52s).
Push uses an exact lease againsta676555a66; auto-merge remains off.

### Task3w — separate remote result types from SSH execution imports

ADR required:no.
ADR path:N/A; existing ADR181's transport/status contracts remain unchanged.
Reason: mechanical extraction of unchanged data types, preserving legacy public
exports and all transport behavior; no new execution or authority boundary.

1. Use the current cold diagnostic: remote status imports transport and the
   remote worker bundle solely to access its enum/result types (0.231s inclusive).
2. Add a fresh-process regression that exercises actual cache transitions and
   response classification, then requires transport implementation to remain cold.
3. Move the enum and two result dataclasses unchanged into a dependency-leaf
   models module. Import them from status; re-export them from the original
   transport module. Keep native transport registrations, protocol and bundle
   content untouched.
4. Verify the original RED, all cache/classification cases, legacy type exports
   and relevant transport tests. Preserve CI selection and measure both Windows
   Send routes sequentially; no release qualification from import counts alone.

The rebased quiet button run completes all three saved replies and traces at
6.748/5.817/5.635s (pr3534-dev5e3-quiet-button-1, 64.53s). Source and HEAD remain
fixed; the cold sample fails the release gate. Task3w's original-source regression
then fails only its final dependency assertion after actual cache transitions and
response classification pass (pr3534-remote-status-import-red-1, 2.92s).
The three extracted type ASTs and all remaining transport behavior ASTs are
unchanged. Native transport registration and the generated worker remain intact.
The bash-based transport suite is selected for hosted Linux CI, alongside the
portable cache/import regressions; it is not claimed as Windows SSH coverage.

Task3w passes 191 targeted cases (16.04s), but its first quiet Enter run measures
24.000/25.114/23.576s. The exact pre-extraction comparison measures
7.802/5.877/4.947s. A second original-source run was inadvertently labelled
pr3534-remote-types-quiet-enter-2 after a CRLF-sensitive restoration assertion
failed; its 9.227/6.796/6.204s is original-source evidence, not candidate evidence.
The exact candidate was subsequently restored using normalized text equality.
Keep all receipts; neither comparison proves a causal timing change.

### Task3v CI ownership correction

Current-head hosted CI reproduces two missed ownership obligations: the moved
marker reader still names the recovery module's SQLite owner, and the persistence
census still points to the old function location. Add a module-owned read-only
marker policy with no backup/capture authority, select it at the marker reader,
and relocate the two inventory rows. Keep all SQL, path validation, source-mode
preservation and error behavior. First prove writable creation is refused by the
new owner, then verify the original marker cases and both exact CI failures.
ADR required: no new ADR. Existing ADR029's module-owned SQLite registration and
ADR126's explicit persistence inventory apply; this does not grant new authority.

The new owner refuses writable creation after the original-source missing-owner
RED (2.34s). Both reported CI contracts, the complete SQLite inventory checks,
original restored-marker/readonly cases and unchanged module-size ratchets pass
278 cases in 275.87s (pr3534-marker-owner-green-1). The 4035-line SQLite ceiling is
unchanged; the existing notification-owner rationale was shortened without
changing that owner's policy. Scoped Ruff F checks pass.

The integrated quiet Enter run measures 7.739/5.405/4.693s
(pr3534-remote-types-owner-quiet-enter-1, 66.00s), with fixed source and HEAD;
all three saved replies and traces complete. Cold Send still fails six seconds.
Independent source review finds no critical/important blocker in the type move,
read-only policy, public exports, inventories or CI selection. It ran no tests
and makes no speed claim. The current Linux 3.12 tool job exhausted its existing
15-minute job budget after checkout consumed 11m21s; dependency installation took
21s and earlier tool groups passed. Do not widen test or job timeouts for that run.
