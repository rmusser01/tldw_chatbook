# PR3411 CI and review blocker implementation plan

> For agentic workers: use subagent-driven-development, one source writer at a time, with independent spec and quality gates.

Goal: make PR3411 eligible to merge while preserving the shared checkout and unrelated changes.
Architecture: repair existing receiving owners and their tests; retain strict source, permission, transaction and physical-return contracts. Use the existing ignored source export and temporary Git indexes; never check out, stash, reset or clean the shared worktree.
Tech stack: Python3.12+, Textual8, SQLite, GitHub Actions.
Spec: the user's PR3411 blocker request, existing ADR285 and the exact CI/review receipts in .superpowers/pr-prep/pr3411-ci-2026-11-05/.

ADR required: no new ADR for routine fixes or direct implementation of existing decisions; amend ADR285 before any SSH proof/pooling boundary choice and before the approved detached-normalization helper correction.
ADR path: backlog/decisions/285-native-phased-agent-workflows.md (existing); ADR029 private SQLite, ADR150/161 UI and ADR230 hook-read policy remain applicable.
Reason: preserve established owners; no new storage/schema/permission owner is authorized.

Global constraints: original dev/e086 HEAD, index c5 and294 protected file identities are immutable. Only the topic branch is pushed; no merge or auto-merge. Current Qodo review and all resolved threads are required before queue eligibility. Targeted local suites only; original failed evidence stays intact. Do not raise ratchets or waive guards without their existing explicit exception procedure and evidence. Root owns Git, network, Backlog and verification execution; source writers never create subagents.

- [x] Prerequisites: capture294-file preservation baseline, current head71851/base29592, queue on,7 review threads and7 failed job logs; independent read-only CI/review audits.
- [x] Task1: workflow UI callable admission. In UI/Console_Modules/workflows.py and Widgets/Console/console_workflow_{setup,status}.py replace eager run_worker coroutine arguments with bound async methods or functools.partial preserving captured inputs, groups, exclusive mode and error paths. Add real Textual pre-entry cancellation/replacement/refusal controls with RuntimeWarning and PytestUnraisableExceptionWarning as errors; retain positive entry/publication. Example: run_worker(partial(self._load_line, chat), group="workflow-line-read", exclusive=True). Gate this focused delta independently before the next writer.
- [x] Task2: SSH handshake fallback. Diagnose actual failed master launch, finite check and original process/source custody; reject nonzero/False/error-text cleanup guesses. Choose a source-qualified proof or conservative existing direct-transport route before editing, amend ADR285 first, and test successful fallback plus escaped/ambiguous master retention and verified reuse. Files: Chat/console_ssh_custody.py, Tools/remote_workspace_transport.py and existing SSH/native tests; no new executor/owner/protocol.
- [x] Task3: private SQLite CI. Use existing private-SQLite owners in Backup_Recovery/detached_layout.py, matching exact paths/validation/backup semantics. Correct C99–C102 expected inventory range and canonical disposition text; add exactly C103/B41 for the checked private destination reopen/backup, preserve exclusive creation, restricted staged source, cancellation/deadline and DELETE finalization, and qualify only the exact validated dynamic owner call with mutation controls. Retain the census and literal-owner checks. Run the entire affected inventory file and detached-layout recovery tests. Gate rollback/close/path behavior and existing ADR029 ownership.
- [x] Task4: performance ratchets. Repay import costs through existing sync-operation and view-only first-use boundaries. Retain the original module-bound authority enum/refusal types, coordinator import at Runtime construction and foreign source-record timing: an initial qualification-leaf candidate saved no UI modules and was rejected at review. Keep the original predicate and negative lookup controls. Replace the two new ancestor bare-type CSS subjects with token-backed authored classes, explicitly verify the narrowed framework-descendant match set, and rebuild generated CSS/bundles using their builders. Inspect exact settings/Canvas preimport growth; use existing dated owner governance for exact changed-owner size rows, with no spare budget or boot/CSS/preimport cap changes. Run the exact failed boot/census/CSS/payload/module guards; do not silently repin floors.
- [x] Task5: affected CI fixture compatibility. Reproduce runtime-ownership, viewless-hook, macOS owned-hook and wizard failures from their logs. Reuse actual store/runtime/private-profile/service wiring and complete legacy observer protocol; keep real accepted/revoked/retained outcomes. The original PostCompact resolver sees a catalog composed before the owned MCP connection; compose only within original captured permissions and rebuild the same provider group. The wizard has a real visible-error/Back race: publish its existing validated classification before the status-row mount await, with a deterministic held-mount regression. Repair shared send fixtures through explicit file-backed/direct-mode/original-trust readiness without changing cheap generic memory/cold defaults; complete character/runtime doubles and real admitted-root metadata fixtures; update actual workflow modal inventories, responsive token-backed tiers and stale launcher assertions. The fenced UAT failure text is the verified missing-README negative control (74 positive support tests and all four receipts passed); it needs no change. Retain every negative source/cleanup control. Run only these affected files/nodes and preserve source-root/epoch receipts.
- [x] Task6: remaining verified review items. Add accurate public annotations in the reported modules, preserve passive read snapshots rather than write transactions, use the existing migration transaction helper only if initialization/FULL semantics remain equivalent, and validate sandbox spelling without provisioning/rejecting supported paths. Add allowed body-free correlation context only under existing diagnostic/privacy policy and update its official inventory if needed. Review each claim against existing contracts, not the bot's suggested code.
- [ ] Final gate: scoped source/test review after each task, one whole combined fix review; required CI and current-head Qodo review, resolve only verified addressed/inapplicable threads, validate queue eligibility and up-to-date state. Push through the existing topic safely, never arm or merge without a request. Verify294 protected bytes/index/HEAD again and complete TASK-38944 with exact limits.

- [ ] Task14: close the remaining exact hosted CI selectors and the separate Goal
  and ordinary child-start defects through the reviewed [remaining CI repair plan](2026-11-06-pr3411-remaining-ci-repairs.md).
  Use coherent existing-owner batches and their independent gates; keep unresolved
  hosted diagnostics open. Fresh dev synchronization and scoped review precede
  bounded diagnostic publication; final whole-branch review follows all gates.

Interface/overlap scan: Task1 UI files also receive Task4 class names and Task6 annotations, so writers run sequentially and later diffs preserve prior fixes. Task2 transport is outside the worker bundle closure; Task4 checks and rebuilds the actual bundled security-helper changes. Task3 SQLite helper changes consume existing ADR029 policy and inventory, and Task5 fixtures must use those original owners. Task4 preserves original native qualification and record-capture boundaries; its ordinary sync/view deferrals must preserve actual operation behavior, cold first-send/import budgets and current test lookup seams before Task5. No task may introduce permissive fallback, copied-source trust or teardown proof. Task2 is an explicit diagnosis/ruling gate before its implementation; all other scopes are evidence-pinned.


Task 4 ownership clarification: ADR required: yes (amend existing). ADR path: backlog/decisions/285-native-phased-agent-workflows.md, “PR3411 deferred receipt implementation with finite storage ownership.” Reason: defer the receipt implementation through its existing operation boundary while retaining exact original storage ownership and eager native source bindings. Readiness requires real handle/borrowed/fallback/close-failure and substitution controls plus an independent gate; numeric startup budgets alone are insufficient.

Task4 final gate: SPEC PASS / QUALITY APPROVE C0 I0 M0. Original coordinator/config/capacity timing retained; one earlier receipt-source tuple added under the prior ADR285 amendment. Native receipt scope, source-substitution, borrowed-handle, retry-close and fan-out controls:56 passed. Final numeric/CI contracts:6 passed. Earlier adjacent performance/CSS:41 passed; generated bundle and exact size/slack:33 passed. Mounted production Setup/History/Status styles at two sizes; final clean Close passed. Lint baseline806/current805, zero new. Old aggregate Close timeout has no proven cause; the actual-click precondition is now explicit. Existing census1683/floor1682 unchanged. Task5 fixture corrections remain separate.

Task5 scope refinement before cursor implementation: actual mounted wake regression and passive original-writer evidence identify missing selected-cursor advancement when a standalone operational mixin attempt first inserts its pending assistant. ADR required: yes (amend existing). ADR path: backlog/decisions/286-console-prompt-mix-ins.md; ADR100 governs the unchanged three-state cursor. Apply reviewed amendment before production; test original selected, branch/before-first, retry, title, held publication, reaffirmation and rollback paths using existing owned SQLite worker, then the original mounted four-surface oracle. No new schema or terminal rewrite.

Task5 original-owner leak ruling: ADR required: no new ADR; existing ADR285 applies. Passive exact allocation evidence identifies two source-less run_ordinary_preparation reservations whose callbacks never enter a native source. Refuse after existing local checks and before borrowing/reserving/enqueueing; all native partial-entry and SSH uncertain-retirement paths remain unchanged. Actual full-capacity regression RED precedes this two-line fix, followed by source-less cancellation/real drain and mounted main-agent navigation verification.

Task5 architecture refinements before production: ADR required: yes, amend existing ADR285; existing ADR097/230 constrain currentness metadata/import budgets and ADR166 governs archive recovery. The independently reviewed resident currentness/platform alias allocation preserves generic retained cleanup and other-controller selection and refuses changed sources before optional ordinary fallback. The independently reviewed archive interoperability allocation uses the existing issued authority, two closed descriptors and opaque claim for reachable preparation, with original consent, policy and physical return retained. Apply these ADR amendments before production, then RED/GREEN and native mutation/source/retirement gates; no budget or permission exception.


## Task5 closing evidence — 2026-11-06

ADR required: yes, amendments to existing ADR285 and ADR286 were independently
reviewed and applied before their production changes. ADR166 and ADR100 retain
archive authority and cursor ownership. Reused the original ordinary preparation,
issuer/descriptor records, mixin transaction and native terminal/retirement paths.

Final local gates retain exact source receipts; they are separate runs:

- 39 unchanged startup, preimport, cold-send, cycle and CI-target checks passed.
- 17 cold/module/getter controls passed, including zero foreign getter calls and
  exact issued-authority release; all 13 original archive compatibility cases
  then passed on the final source. Five final owner checks passed with zero slack.
- 27 currentness controls passed, including actual partial/native planning. Eight
  original planning, cancellation, consent and approved-execution checks passed.
- Six combined mounted manual/cleanup/navigation checks and eight real dialog
  action cases passed without instrumentation. The original three-second stream
  and two-second modal bounds remain. A prior timeout's cause remains unassigned;
  the independently reviewed approval-session setup order preserves the scenario.
- Cursor controls retained the mounted transcript/payload/SQL/leaf agreement and
  ten file-backed cursor cases. Sourceless preparation and original main-agent
  controls passed; no missing callback was treated as physical return.
- Ruff/format comparison covered all 82 changed Python targets: zero new issues
  or formatting debt; inherited findings were retained (1121 before, 1119 after).

The original failing archive callback control left a durable running-only row
despite worker return. Its existing failure handler now writes the required
error terminal; a reopened SQLite check and actual turn retirement prove it.
The first-use archive package stays cold at first paint, preserves its witness
across a genuine parent reload and refuses foreign parent/child bindings before
metadata. Its authority property's getter code is qualified before native use.

No full-suite or new provider replay is claimed. Earlier partially failing,
timed-out and baseline fixture runs remain adverse evidence; none is relabeled
as a full green suite. Task6 and current-head hosted CI/review gates remain open.


### Task6 privacy-prefix refinement before production

ADR required: no new ADR; implement existing ADR029 local-private-data boundary.
Actual targeted privacy checks expose preexisting dynamic values in diagnostics
whose existing constant-prefix contract requires category-only messages. Repair
every actual offending call selected by that same guard, preserving all callback
fallbacks, state returns, user-error behavior and permission checks. Narrow only
the corresponding live metadata-expression contracts and remeasure those live
diagnostic owners; never relax the constant-message guard or historical pins.
Use original callback/sentinel controls and the real private sink checks.


### Task6 qualified evidence and remaining UI blocker — 2026-11-06

Public annotations retain callable admission and type-only domain imports. Existing
manual migration rollback envelopes and deferred/caller-owned read transactions
remain intact. Sandbox lexical validation reuses the existing validator and keeps
missing paths and symlink spelling without creating directories.

Meaningful pre-fix failures preceded sandbox/correlation and privacy changes.
Final Sandbox/Reference controls:67 passed. Native planning, source substitution,
read-only task authority and original child-log controls:10 passed. DB/migration,
passive import and live diagnostic checks passed; the sole privacy failure in that
37-case run was subsequently closed by17 passing privacy/live-inventory controls.
Another31 worker-admission, cancellation and UI diagnostic cases passed with
RuntimeWarning and PytestUnraisableExceptionWarning treated as errors. Eight
exact corresponding owner checks passed with no spare budget. These are separate
runs, not one full suite.

The privacy correction follows the original category-only guard:10 actual
violations in six existing owners now omit dynamic values. Original state
identity, dropped/retained rules, committed workspace survival, backfill retries
and user-visible UI errors remain tested. Only10 live metadata-field contracts
are narrowed and six live diagnostic digests updated; historical evidence stays.

Scoped Ruff/format covered106 Python targets:1431 inherited findings became1428,
with no new semantic finding or formatting debt. Raw reports retain six existing
BLE001 catches whose only spelling change removes unused exception bindings;
matching preserves their existing path/rule/message/scope/count. No source lint
waiver or changed exception behavior was added.

During qualification a broader UI run found the original stale-review
journey failing after its fresh second approval. Its isolated reproduction shows
real completed/drained task and phase records; final result presentation did not
arrive before the existing deadline. Diagnose using original owners and retain
all approval/accounting/terminal assertions and bounds before the final gate.
The broader300-second run timed out and is not passing-suite evidence.


### Evidence-based stale-journey fixture ruling before correction

ADR required:no; routine test-budget alignment. A passive original-owner probe
shows fresh approval accepted at0.842s and native provider entry6.60s; at
8.24s the modal remains subscribed with its owned read active and another
refresh pending. No source, accounting or recovery error was observed. These
samples establish progress, not eventual publication or global native drain. The default8-second wait includes cold native
phase preparation and execution, rather than measuring UI publication alone.
The same function's ordinary approval-to-result journey already uses20seconds
(test_console_workflow_command.py1065–1069). Align only the stale branch's
equivalent final wait with that existing phase-operation budget. This supersedes
the earlier blanket retention of that one fixture deadline; all other bounds,
numeric performance guards and functional/security assertions stay intact.
No product speedup or latency-SLO claim follows. Independent review precedes
application; an uninstrumented stale journey and ordinary approval control
must then pass with the original accounting/decision/receipt assertions.


The reviewed one-keyword alignment subsequently passed the uninstrumented stale
and ordinary approval journeys:2 passed in72.43s, including all original stale
refusal, refreshed canonical decision, accounting and duplicate-command oracles.
The five complete affected privacy service files passed136 tests in27.37s.
The initial whole Personas editor run had three legacy fixture failures and15
passes. Reviewed canonical row/default and exact profile-marker corrections
then closed those failures; the final formatted file passes all18 cases.


Task6 closing gate:SPEC PASS / QUALITY APPROVE, no open findings. Final complete
Personas file18 passed after pure AST/literal-preserving formatting;136 affected
service tests and the original stale/ordinary approval pair passed separately.
Final106-file Ruff/format comparison has no new semantic finding or formatting
debt (1431 inherited detections before,1428 after; six unchanged catch spellings
matched independently). The only changed deadline aligns the stale journey with
its existing20-second native phase-operation budget as recorded above.
Task6 is complete; actual current-dev union and hosted CI/review remain open.


### Final-union native-hook fixture refinement before correction

ADR required:no; preserve the existing native plugin preparation contract.
The actual rebased12-case hook gate passed10 and failed2; isolated reproduction
confirmed both. One wrapper rejects the existing scratch_manager keyword before
its source-replacement assertion. The other observer holds the first admission
check, but a passive original-caller trace shows the new preordinary model check
(`_native_skill_configuration` -> `PluginService.admit`) arrives before native
hook preparation. No wrong runtime was established.

Forward the exact original scratch manager/snapshot in both legacy callback
wrappers. Select only the proven earlier model-check producer using a test-local
ContextVar around the captured original controller._native_skill_configuration,
with exact argument delegation and finally-reset. Its checks still call the
original checker without the test hold; all other marked or unmarked native-hook
checks remain held and recorded. Independent review rejected an unapplied
marker-only default because it could make direct refusal assertions vacuous.
Direct/unqualified compatibility probes retain their original observation. Retain every opaque, owner, capacity,
cancellation, source-substitution and physical-return assertion. No product
body, native record, permission or broader preordinary admission policy changes.
Independent review precedes application; then qualify the complete affected
native-hook file and the four typed-refusal/read-failure union controls.


The complete affected hook gate then exposed the same preordinary observer
problem in four additional rows and a stale qualification-function replacement.
Keep the original qualifier and source records intact: simulate missing killpg
only within bounded capability-selection contexts. Compatibility fallback keeps
its original ownerNone/cold-resource oracles. Related command/legacy engines
restore capability before real command execution; MCP and workflow-history
controls restore it before fixture/runtime cleanup. Five analogous old controls
actually failed at the changed-source guard before correction. These are
capability-input simulations, not live Windows cleanup qualification.


### Remaining final fixture contracts before correction

Two artificial private capacity/activity replacements are now caught by the
ordinary-turn source-integrity guard before the later hook-specific refusal.
Assert its exact RuntimeError instead of a later accepted=False result; retain
the no-hook, original capacity/activity cleanup checks and strengthen zero
provider/no preparation-read checks. Existing ADR285 requires changed-source
refusal; no production error catch or permission exception is justified.

A fallback MCP case selected capability absence only after its runtime had
latched qualified-native requirements. Construct that single indirect fixture
under the simulated missing capability, then use its original real MCP peer
and fallback assertions. The existing dependency order starts the peer after
runtime construction; client start/close uses terminate/kill, not killpg.
Restore the bounded platform context before inner hooks/peer cleanup and
before outer task/runtime cleanup, including failures. Other cases keep their
original qualified startup. Never rewrite the latched flag or source records.


The remaining MCP refusal was diagnosed through original catch, lifecycle and
catalog observations: currentness stays true and cleanup is empty, but the real
tool is narrowed away by a still-pending synchronous snapshot with empty MCP
maxima. An original async-capture attempt retained the same pending empty
snapshot in this unqualified mode and failed; that attempted caller change is
removed. Reuse the existing canonical fixture setup: load the actual original
MCP.server module and configure the test-owned local service's original default
manifest reader before normal submit capture. Its policy enforcer already has
the unchanged default None value. No maxima, registry, permission floor, source
record or target-resolution guard is edited. Unknown custom metadata remains
fail-closed; qualification covers the actual canonical manifest source.


### Final local union gate — 2026-11-06

Actual dev b2c618 + reviewed topic91a63b merged into tree190eb6 with only the
three reviewed hook/Settings/census resolutions; current Task6 rows remain exact.
All generated CSS reproduces from the merged source, with no generated drift;
the SSH worker bundle reproduces. UI census1701 retains the dev1700 floor.
Separate stable-source runs pass:39 startup/first-send/import checks;13 focused
CI/size/CSS/worker guards;21 workflow geometry/resize cases;2 selector checks;
all35 affected native-hook/typed-owner cases;9 final refusal/capability/real
command/MCP controls. The canonical unqualified MCP caller also passed alone.
All original required-save/source/cleanup negatives remain; no native Windows
qualification or full local suite/provider replay is claimed.
Final110-file Ruff/format comparison retains1431→1428 inherited detections with
no new semantic finding or format debt; six unchanged catch spellings were
independently matched and raw reports retained. All local source/review findings
are closed. Hosted current-head CI, Qodo review and thread resolution remain
pending before Task38944 is Done or merge readiness can be claimed.


## Additional review verification and correction — 2026-11-06

The user's additional review extends the existing TASK-38944 acceptance criteria.
Source before this batch is f9b59ea342132bd111ab69ca91550fca2a12b49f.
Read-only triage reports and bounded probes live beside the existing CI receipts
under .superpowers/pr-prep/pr3411-ci-2026-11-05/feedback-*.
All original global constraints and source custody/permission gates remain binding.

ADR required: no new ADR for child signaling, typed pre-entry refusal, terminal
state guards, literal notifications or standalone migration transaction repair.
Amend existing ADR285 before changing payload charge/retention, exact schema
variant construction/validation, native hook group custody or first-send config
persistence policy. ADR029 retains private-data/schema recovery boundaries;
ADR126 retains storage ownership; ADR150/161 retain UI patterns.

- [x] Task8: repair child-start signaling and typed tool pre-entry refusals through
  original AgentService worker/operation owners. A finally must cover source and
  connection setup as well as child body. Distinguish proven no-provider-entry
  refusal from attempted source/unwind uncertainty; preserve genuinely unknown
  physical returns. Regression-first, including cancellation/retirement, exact
  exception transport, prompt subsequent calls and retained-unknown controls.
- [x] Task8b: qualify the real cold-core nonexecution contract using original
  in-place observers and exercised main/future-thread positive controls. Only
  budget/capacity declaration residency changes its test/documentation contract;
  keep production identities/capture timing, compatible module/schema aliases
  and numerical startup budgets. Amend existing ADR285 before the test change.
  Verify targeted cold/import guards on3.12 and private3.14; independent gate
  before Task9. Design/brief: feedback-task8b-design.md and
  feedback-task8b-brief.md in this plan's existing PR preparation artifacts.
- [x] Task9: repair exact workflow schema handling and first-open catalog cost.
  Reproduce ANALYZE and genuine optional receipt-table creation failure with
  file-backed reopen. Preserve rejection of extra user objects, malformed required
  DDL, wrong stamps and unreviewed legacy/staged recovery schemas. Reuse trusted
  canonical/legacy variants without reconstructing all duplicates on every cold
  first open. Check migration runner separately from standalone SQL execution;
  preserve the runner's FULL rollback envelope and make the standalone migration
  atomic if its independent failure is confirmed. Independent spec/quality gate.
- [x] Task10: close workflow terminal transitions and accounting lifecycle. Trace
  every direct/CAS status writer before changing the common state boundary; allow
  documented removal, stale CAS and recovery cases only. Reclaim unused reserved
  slack only after conclusive settlement, with exact joint result/checkpoint
  charges, existing durable evidence and tombstones retained. Verify repeated
  workflow availability and existing explicit guarded payload removal. No silent
  deletion of retained history or unresolved reservations. Gate independently.
- [x] Task11: persist first-send completion across unrelated config writes through
  the existing locked config owner while retaining live-lifetime identity checks.
  Amend ADR285 narrowly before behavior. Repair literal exception notifications
  with markup=False through actual render tests, and make first-send refusal copy
  action-neutral at its shared producer and exact-string receiver. Gate independently.
- [x] Task12: settle native hook clean-exit PGID reuse safely while preserving
  qualified descendant custody and ordinary hook semantics. Verify the actual
  lifetime/proof boundary before selecting a fix; no numeric liveness or successful
  exit may impersonate original-group identity. Record a narrow ADR amendment
  before any custody change and run real process/descendant/cleanup controls.
- [x] Task13: source-identity, duplication and interface cleanup.
  Document the compatibility cost of wrapping/replacement and the fact that
  same-process identity is not a hostile-code security boundary. Evaluate the
  repeated wave/helper blocks against existing original-source contracts before
  mechanical consolidation; preserve every original validation and mutation
  control. Gate this cleanup independently, then execute Task14 below. Final
  integration, publication and current-head CI/Qodo/thread readiness follow Task14.

Interface/overlap scan:

| Tasks | Shared owner/interface | Required ordering and check |
| --- | --- | --- |
| 8 / 9 | AgentService operation-owned DB scope / AgentRunsDB open | Signal every setup outcome; keep actual connection unwind and FULL migration ownership. |
| 9 / 10 | workflow schema / store CAS and payload accounting | Schema handling lands first; no invented DDL or data-loss retention fallback. |
| 8 / 12 | native source return / hook group physical custody | A typed unused refusal cannot stand in for an entered hook's retirement. |
| 11 / 13 | first-send original-source identity / helper qualification | Durable per-key update after quiescence does not bless a replaced active receiver. |
| 8 / 13 | AgentService definitions / closed source records | Preserve receiving owners and actual refusal/unknown controls through any cleanup. |
| 9 / 13 | exact trusted schema constructors / definition records | Performance cache may reuse trusted values, never accept arbitrary live schema. |
| 10 / 12 | terminal/accounting closure / physical group return | No reclaim or terminal success on unknown cleanup. |

Each task agrees with its regression-first scope; claims awaiting live proof stay
explicitly unqualified. Tasks are serial writers, including their focused tests;
no later task starts with open Critical/Important findings from its gate.


## Task8b implementation notes — scoped gate passed

ADR required: yes (existing amendment).
ADR path: backlog/decisions/285-native-phased-agent-workflows.md,
“PR3411 resident currentness without loading workflow execution.”
Reason: clarify declaration residency while preserving the accepted original
source identities and early capture boundary. The reviewed sentence was added
before changing the cold test.

The existing fleet-message cold test now observes original Python entry codes
in place on the importer and future Thread workers, with process-wide SQLite
audit admissions. It verifies the exact checkout, resolves all eleven targets
(including deliberately loaded post-cold owners), and exercises genuine
capacity/owner/operation plus SQLite positives on the main and joined worker
threads. Compatible cold modules, public schemas and WorkOrigin aliases remain
asserted; production definitions, capture timing and numeric budgets are
unchanged. The amended test uses an explicitly selected private child profile
and Source path. No wrapper, lazy factory, module hiding or class move is used.

Evidence and limitations are recorded in the existing PR preparation artifacts
as feedback-task8b-report.md and its uniquely named bounded receipts. Task8's
completed state is preserved. Task8b passed its independent specification and
quality gate with no open findings at private checkpoint ab9cf0dde19e;
feedback-task8b-review.md records the scoped review. Hosted CI and task-wide
acceptance remain open. Task9 follows.

Task9 historical qualification also covers the three exact v30 results in the 2026-11-06 historical amendment to ADR-285. It requires full stamps and receipt metadata, authentic historical fixtures for the five upgrade tests, and qualification through all recovery readers. The completed independent gate is recorded in the closing section below.


## Task9 closing gate and Task10 handoff — 2026-11-06

Task9 passed independent scoped specification and quality re-review at private
checkpoint 10129398ba92 with no open findings. Exact catalog/metadata/stamp
admission, local and live consistent read views, canonical precomputation and
standalone atomicity retain their reviewed boundaries. The actual primary 3.12
combined full-profile node passed 105.59 seconds with original 45-second child
limits and both coherent 83-owner/70-file archives. Both readers passed 45
profile/history cases each and 67 legacy controls passed. Private failure output
was verified through five intended nested-pytest failures and a success control.
Earlier mixed-engine full-media timeouts remain unattributed; no full-suite or
new hosted-CI claim follows. Evidence: feedback-task9-report.md and
feedback-task9-rereview1.md in the existing private preparation artifacts.

Task10 implements the user-selected v30 policy: preserve requests, plans,
decisions, accounting and history; reclaim only positively settled result-pair
charge and expose confirmed removal of existing saved results/checkpoints/window
payloads. ADR required: yes, amend existing. ADR path:
backlog/decisions/285-native-phased-agent-workflows.md. Reason: charged-storage
and explicit removal contract. The reviewed amendment is applied before code;
no v31/full-body erasure or cap increase is selected. Task10 remains pending
implementation and its independent native/UI/archive evidence gate.


## Task10 closing gate and Task11 handoff — 2026-11-06

Task10 passed independent specification and quality re-review at private
checkpoint 6019f04f0c5a with no open Critical or Important findings. The final
phase-owned claim/slot proof keeps legitimate result-only histories readable
after first native preparation, preserves full legacy reservations and refuses
missing or foreign native custody. Terminal/same-chat control, exact complete
pair charging, scoped owned deletion and original consent/replay remain covered
by the earlier task and fix-round evidence. Results, checkpoints and window
payloads can be explicitly removed; plans, decisions, accounting, reservation
identities and native Change Review snapshots remain. Immutable-only saturation
and accessible-chat limits remain. No schema, authorizer, permission or cap
exception was introduced.

Final primary 3.12/macOS evidence includes 67 core, 23 domain, genuine
partial-resume compatibility, restricted retry-copy checks and three separately
completed installed capture/restore/rebackup states. Original phase limits
remain. The interrupted archive aggregate and at-kill inventory gap, first
retained capture watchdog failure and earlier unattributed EOF/setup failures
remain adverse evidence. No full suite, 3.14 or hosted-green claim follows.
Evidence: feedback-task10-report.md and feedback-task10-rereview2.md in the
existing preparation artifacts. The additional startup/return qualification is
now Task14s in the remaining-CI plan; it is not a Task10 product fix.

Task11 follows the reviewed first-send correction. ADR required: yes, amend
existing. ADR path: backlog/decisions/285-native-phased-agent-workflows.md.
Reason: narrowly allow a locked one-key completion-fact merge after original
physical quiescence, preserving unrelated native settings and all live source
identities. Apply the reviewed amendment before code. Update both direct and
received promotion producers to carry the original six-field source tuple; keep
one pending fact/task/fence and distinguish publisher, waiter and disposer
cancellation through physical writer return. Use the existing literal settings
transaction's actual locked path/generation and retain failure for bounded
ordinary/disposal retry. Verify real mounted Send/direct completion, restart,
intervening settings, source/path refusal, retry and cancellation. Repair literal
Undo notifications with markup=False and the shared neutral publication refusal
producer/receiver through actual Toast/Undo/RawCLI controls. Task11 local implementation and independent approval are complete; see the
closing gate below.


## Task11 closing gate and Task12 handoff — 2026-11-06

Task11 is complete locally at private checkpoint403656671ab1e4a5c3c18cd52d798f2bf961cf3e
(base72ac, initialf227, scopedR1). Independent SPEC PASS / QUALITY APPROVE in
feedback-task11-rereview1.md closes the publisher exception/cancellation finding
and original confirmation Toast qualification, with no open scoped findings.
Both receipt producers retain their original six-field source, pending completion
merges through the native locked literal transaction after quiescence, and the
existing task/fence carries physical writer return and bounded retry. Review/Chat
errors render literally and shared publication refusal is action-neutral.

Final R1 evidence includes12 focused cancellation/waiter/disposal/retry cases and
the unchanged original confirmation Toast node. Original actual direct/mounted
Send, native survivor/RawCLI/Undo, current-source restart and rendered feedback
controls retain their exact receipt epochs and applicability in
feedback-task11-report.md. Two-file scoped Ruff has27 inherited findings and zero
added; range formatting is unchanged. Unattributed earlier RawCLI/Toast timeouts,
the allowed RawCLI persistence diagnostic, UNKNOWN retirement negative and lost
read-only inventory session remain recorded; no full-suite, provider, Windows,
whole-branch or hosted-green claim follows. Public task ACs remain open.

Task12 follows the independently reviewed direct-process ownership design.
ADR required: yes, amend existing.
ADR path: backlog/decisions/285-native-phased-agent-workflows.md.
Reason: keep exclusive non-reaping child identity through the last destructive
group signal, then consume its exact status and separately prove group absence.
The exact approved amendment is applied before code. Preserve original native
grant/argv/parent/session topology and ordinary clean-hook behavior. Distinguish
still-owned pending resource reaping from lost ownership: ECHILD ends every later
numeric-PID action, including stdlib cleanup; late resource return cannot invent
physical success. Qualify native SIGCHLD disposition/flags and child watcher,
fair bounded stdin/dual-output capture and true EOF, actual granted engine
commands/descendants/cancellation and post-reap uncertainty with original caps.

The local Docker Python3.12 Linux-arm64 image is environment preparation only;
actual original Linux worker/native qualification remains required alongside the
recorded Darwin ABI/host checks. Root coordinates its isolated container and
dependencies. Reuse run_hooks/legacy_hook_ownership with no new helper process,
executor, permission, runtime setting or broad unsupported-macOS fallback.
One Source writer, targeted serial tests, preserved failures, final freeze and
independent spec/quality gate precede Task13.


### Task12 compatibility/finalization clarification — before wiring

Independent SPEC PASS / QUALITY APPROVE in
feedback-task12-compat-design-review.md approves keeping the shared structural
POSIX gate and native attachment unchanged, with private actual-worker ABI/host
qualification before process_attempted/Popen. Exact Python/Darwin patch receipt
provenance is not an ABI discriminator. Supported ABI/runtime classes require
actual source/physical receipts; inspect the actual running loop/effective
watcher (3.12/3.13 policy-owned, 3.14 loop-owned), native flags and stable host.
Conclusive prelaunch refusal applies only to that target; already-started
siblings keep their obligations. No attached ordinary fallback.

After its original exact consume, the same finalizer may poll non-destructively
within the original cleanup deadline: presence/EPERM remains pending and later
ESRCH may prove group absence. No signalling capability reopens. Lost wait or
source ownership, other missing proofs and retained expiry uncertainty remain
unresolved. Preserve replacement-remains/disappears, ECHILD/source-loss/expiry
and actual delayed-descendant controls. Existing ADR285 is amended before
behavioral wiring; no new ADR/owner/executor/permission. Linux C layout now has a
Source-free receipt, not an engine qualification; private3.14, Intel and actual
original Linux engine gates remain explicit.


### Task12 retained-source clarification — before owner seam

Corrected source-seam design has independent SPEC PASS / QUALITY APPROVE in
feedback-task12-cleanup-source-design-rereview1.md. Existing ADR285 is amended
before changing the owner seam. Preserve strict launch/handoff; private retained
cleanup compares exact precreation original identities and full session registry/
identity, separately qualifies finite original checker/helper definitions before
foreign invocation, and retains accepted recovery owner/path observations. Only
expected disposal/close/revocation, original permission-target unpublication and
handoff cancellation are excluded lifecycle states; other registration loss is
not an exception. Callback identity cannot invoke foreign equality.

Actual per-chat close can remove the session after bounded grace without an
explicit legacy drain. This seam grants no session-removal exemption. Qualify
that real case conservatively; any required existing close/drain correction
needs its own concrete gate before behavior. Intermediate native process/IO/
descendant/host/cancellation controls and expected source-loss REDs are not final
implementation or cross-platform approval.


### Task12 selected-chat close correction — approved before behavior

ADR required: yes, amend existing. ADR path: backlog/decisions/285-native-phased-agent-workflows.md. Reason: retain the exact selected session through original legacy capture/process/host retirement using its existing ticket and close coroutine. Corrected design R1 has independent SPEC PASS / QUALITY APPROVE in feedback-task12-session-close-design-rereview1.md; apply its exact amendment before this behavior.

Reuse existing close fences/ticket/state and initial two-second grace. Capture and validate original runtime/store/session/full identity before readiness and ticket consumption. Add only active per-session capture counts to the existing issuer and a monotonic construction-start witness to the existing runtime. Seal genuine never-attached absence under the original nonblocking hook lock after both fences; never adopt a later issuer. Poll qualified selected metadata asynchronously, with no default-pool wait or await under a lock. Pending keeps the same coroutine/ticket; unknown retains session/fences and cannot delete. Other-chat and ordinary paths retain their owners.

Regression-first controls: real hook outliving grace, launch fence boundary, capture/reservation and host/write tails, repeated cancellation, other-chat first attachment/continued work, sealed absence versus partial/removed/replaced definitions, full identity/ticket/generation checks after event retirement and direct finalizer entry, nonblocking lock/pool saturation and existing global shutdown. Current conservative close control reproduces session removal with retained opaque cleanup_pending; it is not a success proof. Final full-source platform/source/physical tests and independent implementation gate remain required before Task13.


Task12 construction-association clarification: independent scoped R2 SPEC PASS / QUALITY APPROVE in feedback-task12-session-close-design-rereview2.md supersedes only the proposed construction Boolean. One original runtime field advances False to True to its completed original (engine, issuer, permissions) tuple under the existing producer lock; genuine ordinary completion uses issuer None. Compare retained originals before candidate calls and pin tuple identity in existing close state. Partial/error stays True, no reconstruction; saved absence never adopts later attachment. Before-close coherent graph substitution controls preserve the private witness; tuple identity detects replacement after pinning. Arbitrary hostile private-witness rewriting is outside the supported contract. The exact append-only ADR285 addendum precedes association classification; all R1 subject/capture/fence/poll/physical/cancel/other-chat controls remain.


## Task12 closing gate and Task13 handoff — 2026-11-06

Task12 is complete locally through private checkpoints ec67c57a09ee,71e6518b47d4 and884d91a81630. Scoped R2 SPEC PASS / QUALITY APPROVE in feedback-task12-rereview2.md closes all six initial/R1 findings. Native direct-child wait ownership, signal-before-consume, bounded group absence, true EOF versus local IO, source/lifecycle separation and original host/write obligations are retained. Selected-chat close reuses its original ticket/coroutine/fences, exact subject and original producer witness, with asynchronous selected-custody retirement. Registration loss and late ESRCH cannot publish success. Original ordinary behavior and structural qualification remain distinct.

Final production qualification includes 193 primary tests, 21 disposable-host tests and 29 cases on each of macOS Python 3.13/3.14, Rosetta, local Linux ARM and emulated AMD64, all on 214b9a76. Test-only R2 preserves every production byte and test body, adding two individual skips and their import. Its current 27 fleet passes and policy result of one ordinary pass/four native skips cover that metadata difference. Root audited all 41,224 baseline blobs/modes, exact deltas/rich maps and 77 receipt/log/map histories with all 74 prior histories intact. Earlier failed predicates/fixtures, the ignored warning option and emulation timing evidence remain historical. No full suite, provider replay, native Windows, physical Intel, actual PID turnover or hosted-green claim follows. Public task acceptance remains open.

Task13 follows the already independently approved cleanup/interface design. ADR required: yes, amend existing. ADR path: backlog/decisions/285-native-phased-agent-workflows.md. Reason: preserve original receiving/capture/physical-return contracts while sharing pure construction and explicitly retaining four reviewed lazy DB receiver dependencies. Exact approved ADR and dated layer-map note are applied before code; the four reasons/retirement conditions must also appear in the PR explanation before any allowlist row is added. No additional symbol/use is authorized by scanner success. Preserve six/seven/nine-field distinctions, original captures/bootstrap and source wrapping limits; use only existing Utils/ToolExecutor/Skills_Interop owners. Re-measure all integrated size/cold/lookup/template/app-entry guards with original caps, actual native/SQLite/negative and profiler controls, then fresh independent spec/quality gate. Task14 and whole-branch/current-head CI/Qodo/thread/up-to-date gates remain later.

### Task13 integrated extraction refinement — before additional code, 2026-11-07

ADR required: yes, amend existing. ADR path: backlog/decisions/285-native-phased-agent-workflows.md. Reason: record the exact defining, compatibility, capture and receiving owners before the integrated size/layer repairs. Independent integrated R1 design gate: SPEC PASS / QUALITY APPROVE in feedback-task13-integrated-addendum-rereview1.md, qualification-gated only.

The shared seven-field recorder candidate failed its mandatory before-consumer replacement control; retain the original Chat primitive and local collectors. The three Tool projection copies remain an explicitly unmet cleanup objective unless an already independently enrolled collector exists; create no bootstrap authority to remove them. Keep the separately approved adjacent wave guard deletion.

Remove the integrated Task9 seventh upward edge by physically moving the two exact bounded catalog/metadata readers and canonical helper limits into existing DB/recovery_sqlite, with old Backup aliases and explicit per-caller patch boundaries. Move only unrelated cohesive controller skill bodies behind old full-signature lazy entries and eight hook-review operations as direct runtime class entries; keep every selected-close source, lock, fence, witness, host/native Future and actual physical-return owner unchanged. Preserve named old globals at each original access, including after suspension.

Use one narrowly reviewed private Media/local_media_reading_schema.py pure-data leaf for the existing SQL tuple, with the service's old pre-class alias and unchanged functions/globals/collectors. Count its real one-module cold cost under unchanged caps. Move Personal SQL declarations to existing repository_models with exact bytes/order/aliases; move ConfigLoadFailure to already imported startup_errors using its old config module name inside the NamedTuple class body, with independent 3.12/3.13/3.14 annotation/constructor/hints/pickle checks. Keep the video factory's old entry and a failure-time reader of its actual app globals; do not capture the logger early or reimport app.

Record exact original/current source and metadata; preserve all failed recorder, selector/plugin-environment, native timeout and observer attempts. Matched prior/current held-tool failure is not successful dispatch/retirement; the original progress/refusal and physical outcome still need qualification. Current alias-receiver rerun passed unchanged cap after an earlier timeout; both receipts remain. Run negative/genuine-original positive subject checks, Task12 source/close regressions, every producing/receiving size/slack row, exact new/stale layers/census/entry/template and all original cold budgets. No upward pin, fifth row, compensating unrelated deferral, schema/permission/custody policy or completion claim. Fresh implementation spec/quality gate remains required before Task14.

### Task13 last size decision — explicit bounded amendment before code

ADR required: amend existing ADR285; no new storage/schema/runtime authority decision. ADR path: backlog/decisions/285-native-phased-agent-workflows.md. Independent conditional SPEC PASS/QUALITY APPROVE: feedback-task13-trace-notes-supplement-review.md.

This supersedes ONLY the initial no-upward-size instruction for Chat/console_trace_maintenance_runtime.py. Root verified its exact228lines/SHA d3be9e8f104b137f395a4d0709a9e45b542f69e04df226c8a68d9e0201f0edfb and byteidentity to81b7. Retain code unchanged; explicitly date the one193→228 pin and narrow ratchet-header explanation, no spare. This is35linesofsize debt, not an unchanged-budget pass or user waiver. The authorization is the requested autonomous blocker fix and SDD cap-ruling procedure; applicableAGENTS/CLAUDE contain no user-imposedno-increase constraint. Rejected worker/proof moves and readable setup alternatives remain preserved. No other upward size/layer/cold/performance/timeout/census/concurrency change. Retire only after separate coherent source/physical/timing-safe extraction and lower exactpin. All trace source/parking/import/cancel/unknown/physical/cold tests remain required.

Notes code follows the approved small existing-state-owner design. Root baselines Notes/notes_device_state_store.py and Tests/App/test_notes_sync_boot_gate.py before edits. Old wiring entry/signature/live patches and Path/exists guard stay before lazy Notes import; only actual read-onlySELECT/documentation move to the existing state owner with same_connect flags, query-only OperationalError catch and finalclose/error/race behavior. No startup34/34growth, schema initialization, new logical owner, writableconnector or cache. Run focused genuine missing/configured/empty/corrupt/patch/connect/close/startup/cold controls and all final integrated gates. Task13 remains incomplete; no Task14/push/merge.

### Task13 private observer acquisition correction — before Source fixture adoption

ADR required: amend existing ADR285 test-observer contract; path backlog/decisions/285-native-phased-agent-workflows.md. Independent design SPEC PASS/QUALITY APPROVE: feedback-task13-observer-early-acquisition-review.md. Use the smaller early ID acquisition/empty-callback qualification in cooperative trusted-startup private child, retain exclusive ownership, and register existing adapters/masks only at original arming points. Known conflicting worker/global mask refuses before callback exchange; this does not attest unseen C-thread/audit/local-code state. Preserve setup/import-failure ownership and primary/all cleanup errors; armed teardown still requires real producer return, same-lock admission fence/drain and final alias reset. No product authority, new framework/module or cap change. PREP implementation, concurrent-foreign and cleanup controls, combined code gate, explicit Root adoption, actual Source held/alias/owner-anchor nodes and fullblob audit remain required before Task13 C0/Task14. All earlier failed/passing receipts stay distinct.


### Task13 local closing gate and Task14a handoff — 2026-11-07

Task13 is locally complete at private code checkpoint02163c4f7e4dd179d943357511e36698078aa456, following prerequisite metadata d433. Independent whole-task SPEC PASS/QUALITY APPROVE (feedback-task13-final-review.md), plus scoped EOF R2 C0. Root audited all41224baseline blobs/modes, exact65 whole-task changed paths and v5 evidence; code checkpoint adds59paths beyond the prerequisite documents. Final core7b4b865fa987f7c81170f2524b67cc3a6242894e9b16500dedc638e5e45fb1e8, production8ed71e67e69e403793f52fb6d6141b2186772f250f66cf03d47d36f76b27c6bc. The only post-test change removed two surplus app.py EOF newline bytes with identical full AST/location data. All actual native and guard runs stay attributed to their recorded2ef/0233 or earlier exact epochs; they are not relabeled new-byte runs.

Actual Source alias, actual-owner anchor, unrelated overlap and later unchanged held-return controls pass their original assertions and caps. First Source held failure and failure-only PREP snapshot remain adverse; cause and timing stability unresolved. The sole dated193→228 trace-size debt and optional unchanged Tool projections remain explicit. No fullsuite/provider/Windows/hosted/currentQodo or merge-readiness claim. Full original707 and feedback39 literal ruling bodies are retained; the latter frozen cutoff is now copied durably beside the original archive.

Task14a begins next under the separately approved remaining-CI plan: ADR required no; ADR path N/A, existing ADR285 source/publication boundaries retained. Modify only the generated CompiledSourceScreen/make_screen query fixture in Tests/Performance/test_console_tab_sync_source_ownership.py. Read the current original native tab-sync/workflow query and config replay callers first; reproduce one current failing selector, add the smallest truthful query representation, then the entire24-case existing parameterization (all21assigned hosted failures plus3existing positives; noxfail, as verified against publishedf9/preflight/current), preserving cancellation/replay/drain/foreign-source/import-absence oracles. Use preflight formatter baseline0539d72e... only after identity recheck; private-profile scoped runner, exactcode/maps/receipts, focused static/format and independent gate. No product fix or broad fixture weakening is selected. Root owns checkpoint/publication and all5public TASK-38944 ACs stay open.


Task14a is locally approved: one truthful emptyworkflow-row query fixture,21originalRED and all24existingGREEN, zero newRuff and inheritedformat-only debt retained. Sourcecase inventory was corrected against publishedf9/preflight/current; original21failureIDs and all85ledgerassignments unchanged. Task14b follows its reviewed preflight: four existing native attention preparations and both actualpickerconfig reads, only two testfiles, originalUIjourneys/caps/sourceguards and separate received/stale-source controls. No newADR/productfallback selected.

Task14b is locally approved after the separately recorded third-file observer
scope: independent SPEC PASS / QUALITY APPROVE in feedback-task14b-review.md.
All eleven unique required cases have passing Source receipts at their recorded
epochs; the final original managed-demand node passed with held native initializer
and physical cleanup. No production/shared helper or cap changed. The earlier
corrected attempt's fallback/RuntimeError remains unexplained and material; no
repeatability claim follows. Root verifies the full tree and checkpoint before
Task14c's one-file compaction fixture opt-in. Its nineteen affected helper cases
and twenty-three existing source/lifetime controls retain their original profiles,
cold paths, forecasts, rollback assertions and finite cleanup. ADR required no,
path N/A; existing ADR285. Public task acceptance and current-head CI remain open.

Task14c is locally approved in feedback-task14c-review.md (SPEC PASS / QUALITY
APPROVE C0). Existing compaction setup now establishes original trust, attachment,
stop and canonical MCP readiness; actual received tasks and the original first-send
publisher are awaited before dependent programmatic sends. One tooltip expectation
uses the original Estimated qualifier; one queued caller reuses the helper. All42
required cases have exact passing node receipts, with the tooltip-only cross-epoch
composition explicitly preserved. Production8ed71e67 is unchanged; core0e9a211c.
The source-backed scope extensions, setup-only diagnostics and earlier unknown
exception origins remain documented, with no reliability claim. Root full-tree
audit/checkpoint follows before Task14d's already reviewed native/custom read-hold
split. ADR required no, path N/A; existing ADR285. Public acceptance remains open.

Task14d local closing gate (2026-11-07): independent whole-D SPEC PASS / QUALITY
APPROVE / C0 and final same-epoch31/31 focused cases, normal exit0/175.545s,
no warnings or skipped cases. Exactly four test files; production remains
8ed71e67. The explicit supported-manifest/getter-refusal and legacy API contract
corrections, finite physical-return/caller-cancellation checks and inherited
counter/observer limits are documented in the remaining-CI plan and local-gates
report. Every earlier adverse receipt stays retained. Root full-tree audit and
checkpoint precede Task14e; all five public acceptance criteria remain open.

Task14e local qualification: ten exact selectors pass unchanged, including nine
original private-profile children, with inherited validated runtime/unraisable
warning policy and normal61.132s final return. Independent local C0 review is
feedback-task14e-review.md; hosted cause and public ACs remain open. The reviewed
intermediate checkpoint route now obtains fresh-dev/current-head CI evidence
before unresolved hosted diagnoses or later plan repairs are claimed closed.
