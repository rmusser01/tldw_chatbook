# PR3411 rebuild implementation plan

> Execute inline in the existing checkout. The user requested no subagents, new
> worktree, hidden source export, automatic commits/resets/stashes/pushes,
> publication or merge. Stage only this task's exact paths for review, preserving
> all pre-existing staged entries.

**Goal:** Replace PR3411's orchestration while preserving its approved native
coding-agent feature and actual runtime safety boundaries.

**Architecture:** One app-owned coordinator delegates typed child lifecycle to
existing AgentService/FleetCoordinator/ExecutionOwner. Inject task narrowing
through LoopDeps and real final invocation. Keep durable history, explicit phase
approval and freshly validated resume. Separate historical effect uncertainty
from actual live-worker custody.

**Task:** [TASK-39147](../../../backlog/tasks/task-39147%20-%20Rebuild-PR3411-native-phase-orchestration-on-existing-runtime.md)

**Spec:** [replacement design and salvage manifest](../specs/2026-11-07-native-phase-orchestration-rebuild-design.md)

ADR required: yes

ADR path: [ADR-356](../../../backlog/decisions/356-native-phase-orchestration-runtime-ownership.md)

Reason: replace cross-module lifecycle/recovery contracts, preserving ADR-285 product requirements.

## Constraints

- Work in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook`; preserve unrelated
  dirty/index content and current branch. PR head/base and earlier checkout differ.
- No function/code/default/closure identity, stack inspection or self-attestation
  for orchestration authority. Actual root, permission, budget, revision and
  physical-resource checks remain mandatory.
- Keep plan bounds 128 KiB / six execution phases / twelve tasks / 64 footprint
  entries per task; use current finite cumulative settings and no counter refill.
- Read existing architecture and applicable ADRs before each boundary change.
  Read design-language/component-patterns before touching UI or CSS.
- Targeted tests only, with existing private-profile isolation. No full sweep.
- Keep the feature disabled by default until complete acceptance. No replacement
  launch path may bypass current permission, resource or storage owners.

## 1. Provenance and ownership

- [x] Verify GitHub head `c52accf`, base `a0ae30b`, checkout `e08627a`; create
  TASK-39147 In Progress before adding its implementation plan.
- [x] Observe old run interrupted; inspect relevant processes before source edits.
- [x] Write ADR, design, salvage/removal manifest and this plan in repository paths.
- [x] Before each edit compare current file bytes/diff with the observed starting
  version. If another writer changes the same source, stop that edit and record
  the conflict; continue independent mapping. Never reset or copy an entire tree.

## 2. Pure runtime and injected restriction seam

**Files:** `Agents/agent_runtime.py`, `Agents/agent_service.py`, targeted runtime
and service tests under `Tests/Agents/`; architecture text when verified.

**Interface:** retain `LoopDeps.guard_tool_calls(list[ToolCall]) -> dict[str,str]`
and its existing name/call-id verdict semantics and once-per-batch hook execution.
Add repeatable `LoopDeps.validate_tool_call(ToolCall) -> None`, default no-op;
exceptions refuse. Service injects the existing run-bound task restriction through
this callback. The pure loop knows no workflow module. The bridge's existing guard
runs effectful legacy hooks, so repeated freshness checks must not replay it.

- [x] Add a failing behavior test that ordinary pure-loop execution never imports
  workflow policy. Add guard refusal controls before preparers, after hook argument
  replacement, and after approval-time revocation, including runtime-intercepted
  spawn/discovery tools. No source-attestation assertions.
- [x] Replace direct `check_current_task_call` imports/calls in the pure loop with
  the injected restriction at those effect boundaries. Skip preparers for denied
  calls. Preserve correlation mapping and fail-closed exceptions.
- [x] Inject workflow validation alongside the existing service guard; retain the actual
  provider invocation check after waits and mutations. Do not double-charge or
  turn restriction into approval. Ordinary no-guard calls keep their result shape.
- [x] Run focused runtime/hook/service/scope cases; lint and inspect the diff.

## 3. Native child lifecycle

**Files:** `Agents/agent_service.py`, existing `fleet_coordinator.py`,
`execution_capacity.py`, `agent_models.py` only where a shared typed result belongs;
existing child launch/failure/cancellation tests.

**Interface:** preserve ordinary spawn return compatibility. Workflow coordinator
consumes typed native accepted/refused/completed outcomes; runtime owns resources,
not an implementation identity tuple or an inspecting coordinator frame.

- [x] Trace every caller of `_compose_native_child`, `_launch_prepared_child`,
  `_launch_prepared_child_result`, `launch_workflow_task` and settlement callbacks.
- [x] Pin signal-setup failure, thread-start failure, pre-commit cancellation,
  accepted-child cancellation and late physical cleanup through real native owners.
- [x] Move provisional resource cleanup and accepted child custody into the
  existing native launch owner; use existing cancellation/events and resource
  observations. Refusal-before-start and cleanup-unproven remain distinct.
- [x] Remove corresponding workflow function/caller introspection and obsolete
  identity-preserving tests only after those behavior controls pass.
- [x] Verify ordinary inline/threaded spawn, routing, budgets, hooks, lineage,
  result delivery and disposal through the affected targeted tests.

## 4. Thin coordinator and ordinary shared-tree integration

**Files:** replace `Chat/console_workflow_runs.py`, retire/reduce preparation/wave
modules and their callers, simplify `console_ordinary_tree.py`; selectively adapt
`workflow_task_scope.py`, ConsoleRuntime/controller/bridge and existing UI modules.

- [x] Keep coordinator commands start/approve/pause/stop/resume/revise/get and
  passive subscriptions where UI depends on them. Their bodies orchestrate native
  operations and store transitions, never inspect Python source or callers.
- [x] Preserve exact initial/per-phase approval, idempotent acceptance and control
  revision checks; launch only dependency-ready bounded tasks using one binding.
- [x] Acquire complete shared-tree claims before native capacity, releasing
  unlaunched claims on capacity refusal. Retain accepted claims through real drain.
- [x] Preserve ordinary chat and survivor-child conflict handling without workflow
  self-attestation. Verify inactive/disabled workflow ordinary sends.
- [x] Adapt retained UI commands and Settings only through existing patterns;
  preserve callable Textual worker admission and safe Rich rendering.

## 5. Recovery, history and quota

**Files:** `Agents/workflow_run_service.py`, `DB/workflow_runs.py`,
`Workspaces/worktree_activity.py`, Console startup and affected tests/migrations.

- [x] Reproduce closed uncertain recovery blocking unrelated scratch using actual
  store/activity APIs under private-profile tests, without extracted functions.
- [x] Separate historical unresolved effects from native live/unproven worker
  protection. Add a supported explicit reconciliation/release operation using
  current startup audit and current control revision; reject genuinely live work.
- [x] Keep history readable, resume explicit and fresh, and known/unknown charges
  honest. Verify unrelated work becomes available after completed reconciliation.
- [x] Verify result and change-window removal releases quota exactly once and
  retains history/accounting. Preserve terminal-state and transactional migration
  fixes from the current PR after exact source/hunk comparison.
- [x] Allocate a new migration only if schema changes are necessary; never rewrite
  a shipped migration or silently reconstruct live authority from durable rows.

## 6. Acceptance and removal

- [x] Prove the native Console journey listed in the design, including failed
  verification, approved repair, independent review and restart/history.
- [x] Verify shared-tree conflict, stale writes, unrelated dirty content, limits,
  permission/revision changes, cancellation and physical drain with targeted tests.
- [x] Delete now-unused coordination/attestation code and imports after caller
  search; replace implementation-count governance with behavioral requirements.
- [x] Recheck independent fix manifest against exact current PR hunks. Document
  retained fixes and remaining limitations; do not claim all salvage is certified.
- [x] Run affected lint/format checks and self-review; update architecture and
  task notes with exact commands/results. Keep unfinished acceptance boxes open.
- [x] Verify original unrelated files/index remain preserved. Mark Done only when
  all acceptance and DoD items pass. Leave publication/merge for user authorization.

## Evidence log

2026-11-07: Git/PR/source ownership mapping and tracked design completed. No
implementation or application acceptance result is claimed by this initial entry.
Task/ADR IDs were checked against 847 local branch/remote ref tips and registered
worktree tasks; TASK-39147 and ADR-356 were allocated beyond observed maxima.

### Runtime boundary slice, 2026-11-07

Implemented `LoopDeps.validate_tool_call` and native AgentService injection in the
existing checkout. Removed the pure loop's workflow import. Repeated validation
surrounds preparation, mutations and actual dispatch, including runtime tools;
the effectful batch guard is not replayed. Denied calls skip preparers and the
batch guard, while allowed siblings retain normal review. Final hook validation
cannot widen scope. Existing provider-side checks remain. Three direct-loop test
fixtures now explicitly inject workflow policy instead of relying on an implicit
module import. No coordinator/recovery attestation removal is claimed yet.

Evidence:

- New boundary file first failed all eight cases for the intended missing injected
  behavior. The additional final-hook-mutation test separately failed by invoking
  the outside-scope call before the final validation correction.
- Final targeted command (current checkout source, not the PR export):
  `.venv/bin/python -m pytest Tests/Agents/test_agent_runtime_tool_validation.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_agent_runtime_review_hook.py Tests/Agents/test_hooks_v2_tool_pipeline.py Tests/Agents/test_workflow_task_scope.py Tests/Agents/test_workflow_launch_freshness.py -q --tb=short --show-capture=no --timeout=45`
  → **221 passed in 90.80s**. Private-profile setup comes from existing conftest.
- `ruff check` passes for the new boundary tests and changed scope fixture.
  Per-file comparison against HEAD introduces no lint diagnostics in runtime or
  service; their existing 21 and 42 findings remain. The new test file passes
  `ruff format --check`; `git diff --check` passes. No unrelated lint cleanup.
- Before staging this task's exact paths, HEAD/index match the starting snapshot
  and all pre-existing unrelated dirty files match their recorded bytes. Staging
  preserved every pre-existing index entry outside this task's ten paths; all
  unrelated dirty bytes and HEAD remain unchanged. Only task paths were staged;
  there is no commit or publication. The full pre-existing staged diff has an
  unrelated whitespace finding in `css/screen_modal_file_notes_history.tcss:303`;
  it was preserved. The task's own staged paths pass the whitespace check.

Not run: full suite, full application/provider journey or restart acceptance.
Next ordered work is section 3: typed native child lifecycle, including salvage
of the PR head's admission/start-signal failure handling that is absent from this
earlier checkout. Do not copy the whole newer AgentService over this checkout.
TASK-39147 remains In Progress; only this runtime boundary slice is verified.

### Continued native ownership replacement (work in progress)

- Native admission now includes guard-factory failures in the existing unstarted
  child teardown. The new factory case first failed by aborting the parent; the
  native admission/child-launch cohort then passed **31 tests in 21.30s**.
- Ordinary fleet regression: **142 passed, four deselected in 49.31s**. All four
  isolated-worktree cases fail at the same nodes against HEAD loaded in memory;
  the first two report `operation_path_outside_scope`. The bootstrap marker and
  closing DB fixture expose the actual bodies instead of profile-switch failures.
  No full-suite result or isolated-worktree success is claimed.
- `ConsolePreparationRead` exposes its actual owned worker handle. Workflow read
  consumers no longer fingerprint the reader or inspect its source ContextVar.
  The transparent-runner control first failed at the old fingerprint check;
  handles, hook preparation and policy cohort passed **31 tests in 1.38s**.
  Removed the obsolete reader function/class fingerprint table.
- Native wave launch, close, release and return checks now use explicit wave
  enrollment, native launch results, actual snapshot tickets and physical drain.
  Deleted the unused caller-frame launch association, all wave implementation
  fingerprints, and terminal configuration callback closure authentication.
  Transparent wrappers around launch/release/open/run each completed one actual
  native child, checkpoint and resource retirement. Further acceptance is ongoing.
- Removed history predicate implementation attestation, preserving actual
  authorization, binding, manifest, verifier and checkpoint checks. History and
  native child-launch cohort: **54 passed in 55.60s**.
- Three complete native probes passed: overlapping bounded tasks in one capture
  wave, held tool-return before provider unbind, and capture-off source lifetime.
  The cancellation-after-baseline probe remains under diagnosis: its cancellation
  observer also fails against HEAD. Retain this failure separately from the three
  positive probes; no cancellation success is claimed.
- The native-preparation file had stale tests assuming a task had a primary model
  lifeline before launch. Exact HEAD execution reproduced those failures. Tests
  now observe the real preparation registry, child owner/client and native frame
  finalizer. The first partial source-loader baseline after removing reader
  fingerprints was incompatible and is not baseline evidence; the corrected
  baseline loads all seven changed producer modules from HEAD in memory.
- Shutdown revealed a real ordering gap: downstream disposal tombstoned scratch
  before an unlaunched execution preparation's closing callback returned. The
  current fix waits the registered finite preparation task before downstream
  disposal. Its final verification is still running; do not mark acceptance Done.

The native read retains its actual callback result even if final selection
validation refuses further work; retirement still requires the actual producer
and retirement futures. Callback/source-scope errors remain distinct. The final
coordinator/preparation simplification, recovery reconciliation, quota acceptance,
full native correction/restore journey, lint and preservation review remain open.

### Native child lifecycle work in progress, 2026-11-07

Resumed after the user correctly pointed out that the first checkpoint was not
the end of the rebuild. Continue inline until acceptance, not merely until a slice
passes. No active old writer was observed on resumption.

- Native launch/return now uses a registered launch entry and exact pending result
  under the existing authorization lock, not caller frames or function identity.
  Wave membership is passed explicitly and validated against runtime registration.
- Child admission uses a thread-safe `Future`; setup failures publish refusal and
  cannot leave an unanswered admission wait. Source scopes return before native
  root retirement. Actual thread entry on a failed `start()` keeps capacity and
  budget; an actually unstarted thread refunds its original reservation.
- Removed source/code/default/closure and bytecode checking from the native launch,
  capacity, phase-commit and task-binding paths. Preserved current owner, selected
  binding, activity claim, reservation, run lineage, revision and ledger validation.
  Old preparation/wave/coordinator attestation still needs replacement.
- Red/green controls cover ordinary and workflow transparent wrappers, guard setup
  failure, mutation/revocation and retained actual worker capacity. The binding,
  launch, policy and task-scope cohort passed **108 tests in 66.23s** before the
  final dead-table cleanup and added post-entry-start-error control.
- A broader targeted fleet run was interrupted after repeated failures. The first
  failing native concurrent-child node also fails against Git HEAD sources loaded
  in memory (no exported source tree), with the same `raw_source_selection_changed`
  profile-admission failure. Existing fleet tests lacked the bootstrap-profile
  marker used by native source-bound tests. Their DB fixture also did not close
  its parent connection. Kept the real guard, pinned the private bootstrap profile,
  and closed the fixture DB; three representative fleet cases then passed in 2.27s.
  The complete affected fleet file is being rerun with `--maxfail=3`; do not claim
  its earlier interrupted run as passing or dismiss all its failures as baseline.

Current implementation destination remains this checkout. New lifecycle changes
are unstaged after the first ten staged paths. Do not overwrite those staged
changes with HEAD while continuing. Next: complete targeted fleet verification,
then replace coordinator/preparation/wave attestation and finish recovery lifecycle.


### Native ownership, history controls and v30 recovery, 2026-11-07

Implementation continues inline. No subagent, replacement checkout, commit or
publication was used. Actual resource checks replace implementation/caller proofs
in native child binding, phase preparation, wave and coordinator paths. The three
workflow modules have no code/default/closure/frame authority checks. Ordinary
tree catalogs retain actual providers/spec maps/roots and actual storage-scope
return, accepting transparent wrappers with workflows both disabled and enabled.
The unused controller/bridge planning and ordinary composition fingerprint tables
were removed after caller search. Native wake disposal checks its registered
controller/resource owner rather than receiver implementation.

The full native correction journey passed again after v30: planning, failed native
verification, approved repair with parallel edits, fresh passing check, independent
review, final report, Stop preserving dirty files, complete backup, isolated restore
and rebackup. Command:
`.venv/bin/python -m pytest Tests/Chat/test_workflow_native_journeys.py::test_native_correction_keeps_failed_check_and_fresh_passing_handoff -q --tb=short --show-capture=no --timeout=300 --basetemp=/tmp/pr3411-native-complete-v30`
→ **1 passed in 138.56s**. Receipt and original/restored logs are under that dedicated
basetemp. This uses real native workers/tools/files and a deterministic provider,
not a remote provider or performance certification.

Recovery now distinguishes a current actual native worker from prior uncertain
effects. Explicit review releases only its recovery overlay; a live claim stays.
The initial immutable-decision marker failed a full-quota test: it could prevent
the review needed to permit result removal. ADR356 now specifies the additive v30
fixed-size review revision, which records review without rewriting unknown charges
or native retirement. Constructor/standalone/staging catalogs and the narrow SQLite
migration allowlist include it. Old v25–29 histories default to no review.

Evidence so far (targeted commands only):

- `Tests/Agents/test_workflow_run_service.py` recovery-close/full-quota/live-worker
  nodes → **3 passed in 3.49s**. Full-quota control first failed as intended with
  `workflow_payload_capacity` under the decision-payload approach.
- `Tests/DB/test_workflow_runs_migration.py Tests/Backup_Recovery/test_workflow_recovery_schema.py -k 'not complete_backup'`
  → **33 passed, 1 deselected in 7.15s**, including forced v30 rollback and exact
  constructor/standalone catalog equivalence. The initial staging failure exposed
  SQLite's table-specific `quick_check` authorizer callback; its allowlist is now
  limited to installed workflow migration tables.
- Service/history/recovery cohort (`test_workflow_run_service`, DB `test_workflow_runs`,
  `test_workflow_attempts`, `test_workflow_evidence_retention`,
  `test_workflow_admission_boundaries`, Chat `test_workflow_recovery`,
  `test_workflow_activity_primitives`) → **120 passed in 50.22s**.
- `Tests/Chat/test_workflow_native_return.py` replaces obsolete code/frame return
  attestation probes with six real finite callback outcomes and three private-process
  failed native source/storage/finish exits → **9 passed in 38.84s**. Failed exits
  retain their actual owner/claim/preparation; no false positive drain is reported.
- Ordinary shared-tree child/late lifecycle/conflict/cancellation cohort →
  **12 passed, 46 deselected in 103.38s**. Transparent provider/scope wrappers with
  workflows off/on plus native scope body/cleanup/handle outcomes → **7 passed in
  9.92s**. One intentionally failing old source-gate control retained private fixture
  resources and was terminated explicitly (PID34821), not counted as a pass.
- History UI at 140×42 and 80×24 completed cancellation, stale confirmation,
  explicit review and payload removal with no agent/provider execution. Their two
  tests passed in the combined 13-pass ordinary/UI cohort. Native SVGs were visually
  inspected at both sizes through the browser; labels and close/refresh controls
  remain readable and reachable using existing component classes, with no CSS edit.
- Planning native cohort → **12 passed, 2 failed** initially: budget-zero and changed
  duplicate repair-command refusals retained unused primary slots. Preparation now
  distinguishes unchanged native admission rows from committed/unreadable returns.
  Both exact cases plus the launch-vs-retirement race → **3 passed in 31.82s**.
- An ordinary direct-send hook-initialization failure also reproduces at HEAD loaded
  in memory (no export/worktree); it is not called a passing regression. The received
  killed-MCP node passes by itself in current source (**1 passed in 10.27s**), while
  one combined ordering run failed. Keep that order-sensitive limit visible.

Remaining work: final affected concurrency/permission/UI cohorts, obsolete
implementation-only probe retirement, final lint/format/self-review and preservation
check, then task acceptance/notes. No full suite or live external-provider test.

### Further functional qualification and self-review (still in progress)

- Final runtime/permission/hook/task scope/freshness/native-child/platform/schema
  cohort: **256 passed in 125.81s** (`/tmp/pr3411-native-runtime-permission-final.txt`).
- Native preparation/wave/coordinator/read handles/finite outcomes: **39 passed in
  122.57s** (`/tmp/pr3411-native-ownership-final.txt`).
- Mounted controls/history and token governance: **14 passed in 148.78s**
  (`/tmp/pr3411-ui-final.txt`). A later overly broad metadata-worker recovery guard
  made a harmless history refresh block review. Removed that classification;
  actual execution/preparation/wave guards remain. Recovery/UI, passive read,
  failed native scopes and live-worker guards then passed **9 tests in 53.56s**
  (`/tmp/pr3411-recovery-controls-final.txt`).
- Expanded concurrency/partial cohort initially had **14 failed, 22 passed** in
  665.02s. Preserve that failure log separately. Corrections: known pre-capture
  activity refusal returns its unused wave; retained-custody probes must not run
  successful fixture shutdown; remaining-work probes must use an actually resolved
  configured provider; direct commit/control fences require actual resources and
  a typed control intent, never a caller-frame proof.
  Qualified exact failed cases: native wait/unproven publication **5 passed in
  95.90s**; remaining resolved target + failed owned-close probe **2 passed in
  39.11s**; remaining boundary/cancellation/owned-close cohort **5 passed in
  111.41s**; resource-fence/metadata cohort **8 passed in 19.88s**. The latter's
  passive-read expectation was superseded by the corrected 9-test recovery cohort.
- V30 recovery catalog now uses static DDL transformation, adding no SQLite
  connection at declaration import. Exact migration/staging/archive cohort remains
  **33 passed, 1 deselected in 7.07s**.
- Salvaged narrow analyzed-archive support from the pinned PR's exact SQLite
  statistics descriptors: application catalogs stay exact; only genuine stat1/stat4
  metadata are recognized, and statistics deletion is authorized only during the
  installed workflow table-rebuild step. Native/current + v24 analyzed-history
  upgrade controls first failed `unsupported_schema`; those and existing AgentRuns
  schema/authorizer controls now pass **26 tests in 7.34s**. Other PR schema families
  and unrelated UI fixes were not copied wholesale.
- Baseline attempt after adding more HEAD-loaded modules was invalid because the
  observer required a ModuleSpec origin. It is not baseline evidence. The in-memory
  loader now supplies its exact local origin; no source checkout/export was created.

Self-review has identified a remaining lifecycle issue before Done: canceled
planning with unknown token accounting may retain an admitted primary even though
its provider/lifeline/tool threads have physically returned. ADR356 now explicitly
separates this returned-resource disposition from required history publication.
`Tests/Chat/test_workflow_recovery_cancellation.py` is the new actual native
cancellation/review/shutdown acceptance control. Fix this through native resource
return and existing owned history/ledger operations, preserving unknown charges
and genuine failed-scope/live-worker protection. Do not stop at this checkpoint.

Still required: complete that control, retire remaining obsolete implementation-only
probes/adapt affected callers, finish exact salvage notes, final lint/format/self-review
and preservation/index check, then acceptance/task hygiene. No full suite or external
provider run was authorized or performed.

### Ordered next lifecycle correction

The new cancellation control is valid RED: `workflow_not_settled` follows a canceled
planning call with an uncertain token reservation; its native lifeline cleanup and
driver/tool threads have returned, yet the primary is still admitted. Log:
`/tmp/pr3411-cancelled-planner-recovery-red.txt` (1 failed, private exit17).

Implement the smallest native disposition for this known accounting refusal:

1. In the planning driver, recognize the actual producer's
   `AutomaticWorkRefused('workflow_not_settled')` only alongside positive actual
   read retirement, `_planning_physical_proof`, native scope/handle return and an
   uncertain accounting snapshot. Unknown/failed physical scopes retain custody.
2. Add an owned `WorkflowRunService` operation for this actual registered prepared
   planning object/read/owner, not a saved ID or detached bool. Under the existing
   FULL ledger transaction, verify the current owner and original planning attempt;
   preserve charges, mark an accepted native attempt `review_required`, preserve
   unknown retirement, close its actual active interval and publish recovery status.
   Do not repeat the refused settlement or declare model/accounting success.
3. Install the same bounded folder-scoped recovery overlay as startup before
   settling the physical claim. Finish the actual native owner once, release only
   its actual configuration/intake/primary slots and retain a typed native return
   receipt separate from successful accounting settlement. Propagate the original
   refusal/cancellation after physical retirement; shield required cleanup from
   repeated caller cancellation.
4. `coordinator.reconcile` may remove the retained nonlive run only after checking
   that actual return receipt, original driver return and absent live resources,
   then the owned revision-checked review. A passive refresh must not block it.
5. Include interrupted planning `result_bytes` in supported payload quota removal;
   preserve generation/token charges and plan/accounting history. Re-run the new
   native cancellation/review/normal-shutdown control and actual failed-scope/live
   worker controls, then adapt the superseded old unknown-usage-retains-owner test.

This realizes ADR356's existing lifecycle/recovery requirement rather than adding
another executor or caller/implementation attestation layer. Keep parent task open.


### Final qualification and review (2026-11-07)

The chronological checkpoints above retain their original source/failure context;
these results supersede earlier incomplete checks without converting failed runs
into passes.

- The complete native correction/repair/independent-review/final-report journey,
  including dirty Stop and real backup/restore/rebackup, passed again: **1 passed
  in 142.02s**, `/tmp/pr3411-native-complete-final.txt` and
  `/tmp/pr3411-native-complete-final/`.
- Returned planning with unknown usage first failed with an admitted owner after
  physical return. Stop, repeated caller cancellation and reversed accounting
  clocks now preserve unknown charges, release actual returned resources, allow
  unrelated scratch, support review and remove interrupted planning payload quota.
  The qualified native controls plus known-usage Stop and final-metadata
  cancellation pass **6 tests in 137.19s**
  (`/tmp/pr3411-terminal-cancellation-qualified.txt`).
- Generic terminal controls now refuse reopening an old history. A separate
  resource-checked native-return transition may reveal uncertain usage after Stop;
  explicit recovery review has its own closing CAS and works for older closed or
  stopped uncertainty. Storage/service/migration/archive cohort, including complete
  backup/restore/rebackup: **83 passed in 86.60s**
  (`/tmp/pr3411-terminal-review-storage-final.txt`). Earlier terminal reopening and
  terminal-review REDs are retained separately.
- Native admission/child launch/repeatable runtime validation/read handles/wrapped
  coordinator/setup-worker cohort: **48 passed in 70.43s**
  (`/tmp/pr3411-native-admission-final.txt`). A wrapper test passed alone but hit
  its 10-second wait under parallel test load; its bounded wait is now 30 seconds
  and it passed in the final cohort. No provider timeout was removed.
- Mounted history review/removal at 140x42 and 80x24, callable setup-worker refusal
  and design-token governance: **11 passed in 94.79s**
  (`/tmp/pr3411-history-ui-final.txt`). Setup worker RED reproduced a leaked
  unawaited coroutine and disabled retry; the existing callable/partial admission
  pattern fixes both without a new task engine, stylesheet or dependency.
- Full settlement run had **2 failed, 17 passed**; both failures were the Stop
  controls qualified in the later six-test cohort after the terminal transition
  fix. Earlier implementation-only tests have been removed/replaced: changing a
  callback after its actual completed operation is not proof of a live resource;
  normal source/storage/operation failure tests remain. A delivered final metadata
  cancellation is recorded before publication validation, and a post-return error
  stays available as its cause instead of retaining resources.
- Activity file had **1 failed, 15 passed**; the retained-fleet-fence probe is still
  being qualified separately. Keep this limit open until the actual first failure
  is understood. Native failed source/storage/operation scopes still have isolated
  controls that do not claim successful shutdown or manufacture drain.
- Lint compared to HEAD in memory has **zero new diagnostics** across task-owned
  Python files (324 inherited findings remain). Format only changed regions;
  preserve inherited formatter churn. Task-only `git diff --check` passes.
- Pre-staging preservation: original HEAD, every pre-existing unrelated index
  entry and all 297 recorded unrelated dirty files remain unchanged. Exact owned
  path manifest: `/tmp/pr3411-rebuild-owned-paths.json`. No source tree export,
  worktree, agent, commit, reset, stash, push, publication or merge was created.

Final work: qualify the fleet-fence probe, final changed-region format/self-review,
task notes/status and exact task-only staging. No full suite, remote model/service
or cross-platform hardware qualification was performed. Existing isolated-worktree
baseline failures and the order-sensitive killed-MCP ordinary test remain outside
this selected shared-binding rebuild; they are not passing evidence.


The fleet-fence diagnostic exposed a real prelaunch cleanup omission, not a reason
to weaken physical-return checks. A fenced native fleet with no child launched now
selects the existing cancellation/cleanup path before dispatch. The returned phase
is awaiting explicit resume (UI: paused), retains exact charges, and releases its
unused native frame/configuration/intake. Stop/fenced controls pass **2 tests in
23.64s** (`/tmp/pr3411-native-fence-green.txt`); real failed source/storage/operation
exit controls still pass **3 tests** in the paired run. The prior paired run's one
failure was only its stale `awaiting_resume` UI label assertion; backend state is
now checked separately from its existing `paused` projection.

Bounds, policy, live launch freshness and change-window retention pass **61 tests
in 88.96s** (`/tmp/pr3411-bounds-freshness-windows-final.txt`). Ordinary actual
received turns with provider/source/writer wrappers, workflows both off and on,
pass **4 tests in 13.35s** (`/tmp/pr3411-ordinary-wrapped-final.txt`). The early
writer-wrapper RED stalled its fixture teardown after reporting failure; only its
identified private runner was terminated. It is not retirement/passing evidence.

Self-review traced the new seams through runtime validation, provider dispatch,
launch admission, finite returns, coordinator commands, history controls, SQLite
migration/archive validation and ordinary sends. No workflow production receiver
uses function/code/default/closure/caller-frame identity. Actual session/binding,
configuration, registry/tool maps, permission, revision, budget and native resource
checks remain. The ordinary module's unused-in-workflow function-record utility is
still exported for unrelated existing first-send/SSH consumers; those contracts
were not silently rewritten. No new dependencies or license changes occurred.
The architecture, ADR, exact salvage dispositions and a concrete testing lesson
are updated. Final-source native journey and task-only staging remain pending
until their actual outputs are verified.


### Initial acceptance closeout (before follow-up review)

The final-source native Console acceptance passed **1 test in 117.99s**
(`/tmp/pr3411-native-complete-accepted.txt`). Its actual original/restored histories,
provider/tool receipts, dirty-file assertions and backup/restore/rebackup artifacts
are under `/tmp/pr3411-native-complete-accepted/`. This follows the final fleet-fence,
terminal/recovery and cancellation changes; no production edit followed it.
All task acceptance criteria are supported by the qualified cohorts above. A full
suite was neither requested nor run. No remote-provider or unrelated physical
archive-lineage certification is implied.

Final review found no unresolved issue in the selected rebuild scope. New lint
findings: zero; changed-region formatting and task diff whitespace checks pass.
Inherited lint/format findings, isolated-worktree baseline failures and the
order-sensitive killed-MCP test remain documented limits. The last task hygiene
operation is exact-path staging, preserving the initial index and unrelated dirty
bytes; no commit or publication is authorized.


Staging completed for exactly 64 task-owned paths. Final post-stage verification
(`/tmp/pr3411-final-preservation.json`) confirms unchanged HEAD, zero changes to
pre-existing unrelated index entries or dirty bytes, no unstaged task paths, clean
task staged whitespace, Done status and all eight acceptance criteria checked.
No commit, push or publication occurred. Source, notes and review evidence are
ready in the existing checkout.


## Review follow-up: four approved corrections

The earlier completion was superseded by four targeted review
reproductions: a known-usage clock failure retains a returned planner; unreviewed
legacy closed history loses its fence; a replacement folder inherits a historical
fence; an arbitrary partial Library constructor runs with bounded protection.

ADR required: no new ADR
ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
Reason: implement the already-approved resource ownership and recovery contracts.

Use test-first corrections in the existing checkout. Scope: planning preparation
and its owned history journal, recovery protection projection, ordinary Library
construction registration, directly affected tests/fixtures and these task/docs.
Preserve known token charges, explicit review, actual failed-scope/live-worker
retention, native bounded read-only concurrency and unrelated staging. No function,
code, closure, caller or descriptor identity may authorize the correction.

- [x] Add/run failing regressions for all four reviewed cases.
- [x] Implement returned-accounting, explicit-review and saved-root corrections.
- [x] Register the native Library construction resource with its actual runtime;
      unregistered constructors stay opaque before they execute.
- [x] Verify targeted regressions and surrounding lifecycle/native/ordinary cases.
- [x] Record evidence and self-review; update TASK-39147 and stage exact paths.

The returned planner now uses its actual producer error and positive native
resource proof for clock, SQL and post-commit delivery failures. The journal no
longer requires unknown token usage. It preserves a committed plan/completion and
known charges; an uncommitted settlement remains review-required with unknown
retirement. Repeated cancellation still joins the required recording operation.
Real failed source/storage/operation scopes still retain their native owner.
History offers explicit review for `recovery_required` even with known accounting.

Recovery protection selection now includes unreviewed legacy closed histories;
the existing immutable review marker distinguishes actual review. A saved plan's
directory-chain identity is checked before projecting its folder fence. A renamed
directory, replacement at the old path or retargeted symlink confers no claim over
the replacement. Unrelated scratch remains available.

Library construction is a runtime-owned callable resource registered when the
Console runtime is constructed. Bounded admission captures that resource and
checks its current runtime/app association again before invocation. Arbitrary
partials are opaque before their initialization writes. A late callback swap is
refused before construction. Native read-only tools and a transparent wrapper of
the registered callable remain bounded. Deleted the unused Library function/code/
defaults/closure snapshot; no implementation-identity authority was restored.

Evidence (cohorts overlap; do not add them as unique totals):

- Initial REDs: actual known-usage clock, SQL and post-commit failures retained
  returned planning owners, replacement directories inherited fences, and a custom
  partial constructor wrote under bounded protection
  (`/tmp/pr39147-four-fixes-red.txt`). Two initial legacy-history fixtures lacked
  prior native-owner recovery and the symlink footprint used an alias; neither
  fixture failure counted as a behavioral RED. Corrected legacy and canonical
  symlink cases then failed for their intended protection assertions: **3 failed**
  (`/tmp/pr39147-recovery-fixtures-red.txt`).
- Lifecycle/recovery controls: **18 passed in 148.42s**
  (`/tmp/pr39147-planning-recovery-green.txt`), including all three real failed-scope
  controls, repeated cancellation and invalid clocks.
- Strengthened final recovery/accounting/Library cases: **24 passed in 77.60s**
  (`/tmp/pr39147-regressions-final.txt`). Checks include exact settled token charge
  10, preserved native completion/plan after commit, uncertainty before commit,
  explicit closing review, native read-only/wrapped construction and late swap.
- Mounted known-accounting history RED: the real recovery button stayed disabled
  (`/tmp/pr39147-history-known-red.txt`). All three wide/narrow/known mounted
  history cases passed after the visibility correction in the connected UI run.
- Runtime-owned controller construction and captured direct/RAG selection passed
  **2 tests in 4.62s** with the existing bootstrap-profile marker applied by the
  test invocation, without changing source or production guards.
- The design detector returned no findings (`/tmp/pr39147-ui-detector.json`).

The connected UI module run is not a clean acceptance claim: **76 passed, 24 failed,
1 xfailed** (`/tmp/pr39147-history-library-ui.txt`). Fourteen failures reject old
per-test config reselection before the Library builder, one inventories an
unchanged `attachments_transferred` field, and the others concern mounted custody/
Chat-start fixtures. The selected native factory case passes with the proper
bootstrap profile. The unchanged staged versions of all five affected production
modules were then executed in memory, with no source export or checkout/index
mutation: all four representative custody/field/Chat-start cases reproduced their
same failures in **22.67s** (`/tmp/pr39147-staged-ui-baseline.txt`). Remaining UI
parameterizations are not certified as passing or individually baselined.

The first current-source complete journey exceeded its existing 35s correction
wait under concurrent UI/test load; the expected `awaiting_result_review` appeared
about three seconds later and the rig retired normally
(`/tmp/pr39147-native-journey.txt`). The isolated rerun keeps the original deadlines.
It passed **1 test in 156.23s** (`/tmp/pr39147-native-journey-alone.txt`): actual
native workers/tools, failed verification, user-edit-preserving correction,
independent review, final handoff, dirty Stop, backup/restore/rebackup and restart
history. Artifacts are under `/tmp/pr39147-fixed-native-journey-alone/`.

Self-review traced each changed seam through its real caller, actual resource
return, SQLite transaction and history/UI projection. Final per-file lint delta
against HEAD has zero new findings (324 inherited findings retained); changed
regions format cleanly and task-only whitespace checks pass. No dependency,
configuration option, migration or implementation-identity authority was added.
All 13 follow-up paths belong to the original 64-path TASK-39147 manifest.

Final follow-up accounting/recovery verification passed **17 tests in 41.69s**
(`/tmp/pr39147-accounting-recovery-final.txt`). Before staging, all unrelated
initial dirty/untracked file hashes, the entire initial index and HEAD matched
`/tmp/pr39147-fix-start.json`. The final exact-path stage and preservation audit
are recorded in `/tmp/pr39147-fix-final-preservation.json`. Only the 13 follow-up
paths are updated in the existing task staging; no commit or publication occurs.


## Second follow-up: journal retry, shutdown and stale recovery ownership

Status: Verified. The re-review reproduced secondary journal failure retaining
an opaque native claim, shutdown refusing a positively returned unreviewed planner,
and replacement roots inheriting planning-only or already-installed fences.
ADR required: no new ADR
ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
Reason: direct corrections to the existing native-return and historical ownership
contract; no new schema, storage family or alternate executor.

Test first in the existing checkout. Separate physical return from retryable
accounting recording, preserve native/unknown charges and failed-scope protection,
and use actual filesystem identity for historical denial. Planning-only history
without a saved plan has no persisted ownership or mutating execution to fence;
retain its accounting review without claiming a folder from its pathname. Validate
existing overlays at admission, outside the owner lock, and wake waiters when an
obsolete overlay retires. Verify complete shutdown and later explicit review,
then update notes and stage only the exact task changes.


The native return receipt now captures the actual preparation, phase, tracked
producer error, owner and claim after positive reader/lifeline/tool/storage drain,
configuration exit and intake/primary-token return. Durable admission and known
charges pass to recovery; a captured current folder overlay replaces the opaque
claim before accounting recording. A recording error leaves a retryable receipt,
not a running owner. Explicit review retries with the displayed revision. A
committed recording whose delivery failed is idempotent; reviewed history cannot
be reopened by the retry. Shutdown discards only receipt-proven nonlive entries,
leaving review markers and native/accounting rows untouched for startup recovery.

Historical projection requires a saved plan/root digest. Read-only planning with
no saved plan remains reviewable without claiming a folder by its saved pathname.
Existing overlays revalidate directory chains at prelaunch, blocker queries, wave
admission and coordinator readiness. Only stale historical scopes retire; other
current roots and every actual native claim remain. Metadata checks and observer
notifications run outside the admission lock; no new poller or schema is added.

RED evidence: the initial targeted batch had **17 failures in 47.54s**
(`/tmp/pr39147-second-fix-red.txt`). Fifteen caught their intended bugs; two live
symlink controls incorrectly expected an independent canonical replacement target
to conflict with the original scoped claim. Corrected only that expectation and
retained assertions that the live owner/claim survive and stale history retires.
A mixed-root regression separately failed as intended
(`/tmp/pr39147-second-fix-mixed-red.txt`). A post-commit recording-delivery case
also reproduced the leak against the unchanged staged production modules executed
only in memory (`/tmp/pr39147-second-fix-delivery-red.txt`); no source export or
checkout/index mutation was used for this comparison.

The first combined correction run passed **42 tests in 94.17s**
(`/tmp/pr39147-second-fix-green.txt`). Additional actual post-commit delivery,
stale revision/repaired journal and failed source/storage/operation controls passed
**5 tests in 31.01s** (`/tmp/pr39147-second-fix-delivery-scope-green.txt`). These
cohorts overlap; do not sum them as unique totals. Final regressions and complete
native journey qualification follow below after their actual processes finish.

Final-source focused regressions passed **50 tests in 119.18s**
(`/tmp/pr39147-second-fix-regressions-final.txt`), including all prior accounting/
recovery and ordinary Library controls. The complete native correction/Stop/
backup/restore/rebackup journey passed **1 test in 114.59s**
(`/tmp/pr39147-second-fix-native.txt`), with original/restored artifacts under
`/tmp/pr39147-second-fix-native/`. No production edit followed these final tests.

Self-review traced both recording callers, the actual return proof, revision-
checked retry, idempotent post-commit delivery, shutdown, all historical admission
readers and mixed-root pruning. No new implementation-identity references, storage
migration, dependency or configuration option were added. Per-file lint comparison
against HEAD has zero introduced findings (324 inherited findings retained), and
changed-region formatting and exact task whitespace checks pass. The existing
TASK-39147 testing lesson now records the actual double-write failure and unreviewed
shutdown incident. Earlier broad UI/test/platform qualifications remain unchanged;
no full suite or remote-provider run was performed.

Exact staging updates 11 existing task-owned paths. The pre-stage snapshot is
`/tmp/pr39147-second-fix-start.json`; final index/file preservation is recorded in
`/tmp/pr39147-second-fix-final-preservation.json`. No commit, reset, stash, push,
replacement PR or merge is part of this follow-up.


## Third follow-up: uncertain metadata, frozen planning input and closing retry

Status: Completed for the three fixes with targeted regression evidence. The
complete native journey was unqualified at that close; the later UAT section
records its functional qualification. Review reproduced a
transient EIO discarding an unchanged folder fence, a same-value system-prompt
save invalidating native unwind, and a lost closing-review response stranding
known-accounting history.
ADR required: no new ADR
ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
Reason: direct corrections to existing ownership/input/recovery contracts, with no
new schema, dependency or execution authority.

Test first in the current checkout. Preserve existing historical denial when
metadata is unreadable; use positive metadata or confirmed absence for retirement.
Freeze the admitted prompt value rather than authenticate cleanup with a live
session-settings object. Make reviewed terminal acknowledgement read-only and
idempotent while retaining revision/native-liveness checks and removed payloads.
Verify equivalent and changed prompt saves during both running and stopping,
real failed-scope controls, journal retry and original/recovered native journeys.
Keep unrelated staging/bytes unchanged and stage exact task files only.


The fence correction retains an established scope when the full chain is unreadable.
A fallback `lstat` may positively prove that the root changed or disappeared; if
that metadata is also unavailable, ownership remains unresolved and protected.
Checks stay outside the admission lock and never retire real native claims. Added
regular-file replacements alongside missing/directory/symlink controls so the
conservative error path does not turn confirmed removal into a permanent barrier.

Planning now captures only the admitted prompt text. It no longer authenticates
native unwind with a replaceable session-settings instance. Equivalent and changed
prompt saves while running or stopping preserve the exact provider input, cleanup,
intake release and later settings, without weakening compiled system/seed checks.
Closing review first recognizes the immutable review marker, while still checking
the displayed revision and real liveness. A delivered-again acknowledgement leaves
closed/removed state, review revision and payload removal unchanged for both known
and unknown accounting; it does not reopen history or issue execution authority.

All initial regressions failed for their intended behavior: **14 failed in 86.42s**
(`/tmp/pr39147-third-fix-red.txt`). The first corrected cohort passed **35 tests in
112.88s** (`/tmp/pr39147-third-fix-green.txt`), including persistent/one-shot EIO,
permission and incomplete metadata, four actual running/stopping prompt saves,
known/unknown lost closing delivery, stale retry and removed-history acknowledgement.
Broader final controls and the native journey are recorded below when complete.

Final-source regressions passed **71 tests in 382.26s**
(`/tmp/pr39147-third-fix-regressions-final.txt`): activity primitives, all recovery/
cancellation and startup-history tests, opaque ordinary Library construction and
all three actual failed-source/storage/operation controls. No production change
followed that run. The 14 new cases first failed at the intended boundaries; the
additional regular-file replacement cases preserve positive retirement controls.

Two isolated complete native-journey attempts did not qualify: **1 failed in
29.98s** (`/tmp/pr39147-third-fix-native.txt`) and **1 failed in 38.63s**
(`/tmp/pr39147-third-fix-native-rerun.txt`). Both exceeded the unchanged five-second
initial planning wait, before correction/execution/backup assertions. In the first,
the expected state was published just before timeout; in the second, provider
entry occurred about 4.74 seconds into the wait. Both completed normal retirement
and reopened storage. These are failed timing evidence, not complete-journey
passes; the earlier 114.59s pass belongs to the preceding source revision.

Self-review traced all prompt capture/dispatch/return callers, root pruning and
admission readers, closing review and coordinator retry/removal. No implementation-
identity check, migration, dependency or configuration option was introduced.
Per-file lint comparison against HEAD has zero new findings (324 inherited), and
changed-region format and exact task whitespace checks pass. Existing broad UI,
archive-lineage and platform limitations remain; no full suite or external
provider was run.


The unchanged staged pre-fix versions of the three production modules were then
loaded only in child-process memory for comparison, without a source export or
checkout/index mutation (`/tmp/pr39147-third-fix-baseline-native.py`). That complete
journey also failed: **1 failed in 87.86s**
(`/tmp/pr39147-third-fix-native-baseline.txt`). It passed initial planning but
exceeded the existing 35-second correction wait, reached the expected review
state about nine seconds afterward, and retired normally. This demonstrates a
separate pre-fix timing failure; it does not reproduce or exonerate the current
initial-planning timeout. No deadline or test outcome was relaxed. Final native
end-to-end qualification is explicitly outstanding, while all 71 targeted tests
and actual failed-scope controls pass.

Exact staging updates 10 existing task-owned paths listed in
`/tmp/pr39147-third-fix-paths.json`. The start snapshot is
`/tmp/pr39147-third-fix-start.json`; final index/file preservation is recorded in
`/tmp/pr39147-third-fix-final-preservation.json`. The audit compares every unrelated
index entry and all initial dirty/untracked file hashes, as well as HEAD. No
commit, reset, stash, push, replacement PR, merge or unrelated task update is part
of this follow-up.

## End-to-end and UAT follow-up

User authorized complete end-to-end verification and UAT after the third fix.
ADR required: no
ADR path: backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
Reason: verification of the approved contract; no new product behavior planned.

1. Preserve the current index and dirty/untracked bytes in
   `/tmp/pr39147-uat-start.json`; run the unchanged complete native correction,
   Stop, backup, isolated restore and rebackup journey in a private profile.
2. Investigate any timing or functional failure using actual state/resource
   evidence. Keep original failures separate from diagnostic/qualified runs.
3. Drive the mounted Console's setup, plan/phase approvals, result acceptance,
   prompt editor, Stop, history recovery and removal controls. Check both wide
   and narrow layouts and inspect actual screenshots.
4. Exercise lost closing-review delivery with known/unknown accounting through
   real confirmation controls; verify safe retry, preserved charges, removed
   payloads, no live owners and normal shutdown.
5. Record exact outcomes and remaining limits in the task and evidence; stage
   only verification changes for TASK-39147, auditing unrelated bytes and index.


Functional end-to-end verification passed **1 test in 217.49s**, including actual
failed/fresh verifier evidence, correction, independent review, final acceptance,
Stop, preserved dirty Git state, coherent backup, isolated restore with identical
durable facts/no live workers, and rebackup. This uses explicit UAT bounds of 240s
overall and 60s status waits in a temporary runner; production source/budgets and
backup/restore watchdogs were unchanged. The original 130s process-watchdog failure
and the 35s correction-wait failure remain failed timing evidence. Host load was
49–56 on 18 logical CPUs; no latency claim is made.

Final mounted prompt/closing acceptance passed **4 tests in 163.59s**. An additional
mounted two-phase approval/checkpoint/result-acceptance journey passed **1 test in
56.37s**, using actual native children and original widgets. The original command/
setup UAT passed, while two original approval tests stopped before workflow actions
at tab selection/setup presentation. Extra probe setup mistakes and their corrected
completion/cancellation waits are qualified in the evidence, not called regressions.

UAT found an existing **P2 prompt-editor layout defect at 80×24**: Apply/Clear/Cancel
are clipped below the dialog. The widget/stylesheet are unchanged from HEAD. The
editor passed at 140×42, and workflow recovery/removal passed at 80×24. No production
fix was bundled into this verification-only follow-up. The evidence report is
`Docs/superpowers/qa/2026-11-07-task-39147-end-to-end-uat.md`, with its captured SVG.
No full suite, external provider, human-terminal or other-platform certification.

Exact verification staging updates five task-owned documents/evidence paths. The
start snapshot is `/tmp/pr39147-uat-start.json`; the final preservation audit is
`/tmp/pr39147-uat-final-preservation.json`. All production source, unrelated staged
entries/file bytes and HEAD are preserved. No implementation-identity checks were
restored; there is no commit, push, reset, stash or merge.


## Prompt-editor clipping correction (AC19)

ADR required: no new ADR
ADR path: N/A; existing ADR150/161 modal/component rules apply.
Reason: routine geometry correction using the canonical native scrolling and
trailing docked-action pattern; no service or authority contract changes.

The user authorized fixing the 80×24 UAT finding, adding regression coverage,
rerunning UAT, updating TASK-39147 and exact-path staging. The existing dialog now
scrolls, its existing action row docks at the bottom, and Library status updates
scroll into view. The modular CSS source and regenerated app bundle carry the
same one-line dock rule. Caller ids and Apply/Clear/Cancel outcomes are retained.

The new regression first failed because Apply was not painted at 80×24 (**1 failed
in 11.41s**). The first corrected-layout regressions passed
**7 tests in 111.83s**; the final painted-label/click/resize/whole-feedback/focus
regressions passed **7 tests in 204.41s** after the scroll-timing correction. Six existing prompt persistence/Library controls passed
**6 tests in 83.43s**, and all six passed again against the final code in
**213.53s**. Full mounted prompt UAT at both sizes and during running/Stop
passed **4 cases in 379.34s**, including physical Cancel/Clear and native shutdown.
Those UAT cases predate only a final Library-feedback scroll-timing correction:
a stronger painted-text check exposed the last feedback line behind the footer
when scrolling measured Static before its new height was laid out. The scroll now
waits until after layout refresh. Both final-code narrow mounted UAT cases then
passed **2 tests / 2 deselected in 254.35s**, including running and Stop/recovery,
physical Apply/Cancel/Clear and actual native cleanup/shutdown. This overlaps the
initial four cases; deadlines were unchanged. Exact commands/logs, before/after
images and all qualifications are in the UAT report.

The full CSS build picked up unrelated staged portrait declarations and generated
rules with unavailable tokens. These produced a separate parsing-failure packet;
the two unrelated generated sheets were restored to starting index bytes. The
prompt-specific bundle matches its modular source. Broader CSS checks finished
16 passed / 3 failed / 20 deselected: portrait-sheet drift, a canonical-class scan
timeout, and dimension debt in four unchanged modules. None is called green or
absorbed into this fix. New test lint and touched-file formatting pass; four
existing modal lint findings remain unchanged; the design detector returned [].

The first UAT packet and diagnostic startup-tab failure remain separate. The final
probe waits for Console's initial blank tab before selecting the seeded chat.
Capture checks the actual painted controls; title-only SVG exports were rejected.
No full-suite, external-provider, other-platform or full backup-journey rerun was
needed for this scoped UI correction. The initial snapshot is
`/tmp/pr39147-prompt-layout-start.json`; exact owned paths are recorded in
`/tmp/pr39147-prompt-layout-paths.json`; final preservation/staging is audited in
`/tmp/pr39147-prompt-layout-final-preservation.json`. No commit, push, reset, stash,
worktree or unrelated staged-source edit.
