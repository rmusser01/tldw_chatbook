# Personal Context pinned-dev conflict refresh implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore PR #2862's integration with pinned dev after new upstream changes, preserving the reviewed memory behavior and native qualification limits.

**Architecture:** Merge the fixed dev commit into the existing isolated feature branch without rewriting history. Resolve the Settings split by composing both existing selector families and preserve both lesson additions. Qualify the actual combined source; earlier receipts remain attributed to their original commits.

**Tech Stack:** Native macOS Python >=3.12 (installed 3.12.11), Textual >=8.0.0,<9, SQLite, existing pytest/Ruff and CSS generators.

**Spec:** Docs/superpowers/specs/2026-09-25-personal-context-memory-evolution-design.md; TASK-25907.23 acceptance criteria; ADR-193 Unit A; ADR-097, ADR-150 and ADR-161.

ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md; backlog/decisions/193-native-v2-profile-compatibility-and-admission.md
Reason: Direct integration of existing owners and contracts; no new architecture, dependency, schema or policy is selected. Resolve actual canonical ADR filenames when linking this plan from the task.

## Global Constraints

- Python >=3.12; Textual >=8.0.0,<9. Use the existing native interpreter and installed dependencies.
- Work only in /Users/macbook-dev/.codex/worktrees/personal-context-memory-pr/tldw_chatbook on codex/personal-context-memory-dev.
- Feature pin: e3527ac883e32c94087293212ddc9d5223505adf. Dev pin: dfee4bf4c66ec2656a6c4ea2667edb63cc38a445. Do not chase later dev movement during this qualification.
- No full repository test sweep. Run native targeted tests for affected owners and required architecture/style/import/generated guards.
- Do not raise size, method, style, CSS, descriptor, import or deadline limits to obtain a pass. Distinguish incoming independently approved ratchet changes from local repairs; inspect their exact before/after values and retain stricter existing values where the same guard overlaps.
- Use synthetic private profiles and temporary storage. No production profile, keyring, provider or source calls.
- V2, Forget, source-opening, schema/AAD, disclosure and companion-server activation remain closed. No PR merge into dev.
- Preserve the independent retirement branch, primary checkout and foreign changes. No stash/reset/prune/gc, no force push, no manual generated stylesheet edits.
- Reuse existing owners and helpers; no speculative abstractions, extra dependencies or normalization of source prose/blank lines to evade size guards.

## Task 1: Integrate pinned dev and resolve owner conflicts

**Files:**
- Modify: tldw_chatbook/css/build_css.py, existing Settings ScreenOwnedSplit only.
- Modify: backlog/docs/lessons-testing-evidence.md, preserve both existing conflict sides.
- Generate through the canonical builder: tldw_chatbook/css/* generated outputs and their existing manifest metadata.
- Test: Tests/UI/test_widget_css_consolidation.py, Tests/UI/test_css_bundle_sync_guard.py, Tests/Architecture/test_module_size_ratchet.py, Tests/Architecture/test_screen_size_ratchet.py, Tests/Architecture/test_library_modules_size_ratchet.py.

**Interfaces:**
- Consumes: existing ScreenOwnedSplit(modules, sheets, prefixes, pinned), immutable feature/dev pins and the canonical CSS build entry point.
- Produces: one combined Settings ownership rule retaining personal-context and Hooks selectors, both lessons' exact bodies, and a conflict-free candidate with both parents preserved.

- [x] Step 1: Verify the branch, root, clean baseline and existing linked-worktree identity. Record both parent SHAs and the conflict preview; the native baseline receipts apply only to e3527ac883.
- [x] Step 2: Start the reversible local merge:

```bash
git -c gc.auto=0 merge --no-commit --no-ff dfee4bf4c66ec2656a6c4ea2667edb63cc38a445
git diff --name-only --diff-filter=U
```

Expected unmerged paths: build_css.py and lessons-testing-evidence.md. Unexpected conflicts require source inspection and a plan amendment before resolution.

- [x] Step 3: Retain the existing two source modules and pinned tokens, with this combined prefix tuple:

```python
prefixes={
    "settings": (
        "settings", "personal-context", "console-hooks", "hook-review", "-wide-viewport"
    )
},
```

Preserve upstream's Hooks ownership comment and the existing memory/RecoveryPassphraseDialog ownership comment. For every lessons conflict concatenate the complete feature body and the complete incoming body, separated by one blank line; remove only Git conflict markers.

- [x] Step 4: Inspect the automatically merged runtime/controller/screen changes against both parents. Confirm the new Hooks admission does not bypass the existing preparation/maintenance, fork publication and private source ownership guards. Compare all pre-refresh ratchet constants against the merged versions and the incoming ADR-097 exception provenance; no local relaxation.
- [x] Step 5: Regenerate using the existing native builder:

```bash
python tldw_chatbook/css/build_css.py
```

Then run the targeted CSS ownership and architecture families, retaining XML/import/source receipts. Use the existing private-profile harness and native interpreter; failed checks enter root-cause diagnosis and a scoped plan amendment before code repair. Do not relabel a single passing rerun as resolution of an unknown intermittent failure.
- [x] Step 6: Check conflict markers, exact lesson-body preservation and whitespace. Commit the conflict-free integration only after source/guard evidence is recorded; report any still-open native verification accurately in the task report.

## Task 2: Repay inherited strict preimport payload debt

**Files:** Existing shared Chunking package/engine export owners and exact
import callers in the Library/rechunk and Evals screen/panel/runner owners;
select the minimal paths after tracing their current import/use contracts.
Existing native preimport/import/behavior guards and focused contract controls.

**Interfaces:** Consumes Task 1's candidate and immutable native comparison:
feature e3527ac883 = 540 modules / 406,000 LOC; dev dfee4bf4c6 = 557 / 411,886;
combined = 541 / 406,612. The pass-wide feature→combined module-set delta is
only `UI.Screens.settings_hooks`; published feature already exceeds the strict
500 / 378,740 pins by 40 modules / 27,260 LOC. Produces an actual combined
source proving all strict 500 / 378,740 / 123,319 limits without exceptions.

- [x] Trace real dependency callers before source repair. Shared Chunking
  eagerness starts in `Chunking/__init__.py` legacy exports: Library rechunk
  `library_rechunk_service` imports `Chunk_Lib`, RAG `chunking_service` imports
  its compatibility exceptions/functions, and `chunking_lab_screen` imports
  package submodules `lab_models`/`lab_preflight`. The current pass bills 44
  Chunking modules / 17,604 LOC. A Library-only local import cannot repay the
  pass because the later Chunking Lab route still executes the package init.
  Evals currently bills 27 engine modules / 7,062 LOC and 14 UI modules /
  9,329 LOC; determine the existing compose/use seams before deferring them.
- [x] Follow existing project PEP 562 lazy-export patterns only where they
  preserve `__all__`, `dir`, public object identity, direct/from/star imports
  and real use behavior. Use existing Evals compose/use sites; retain route
  order, screen identity, state ownership, source/monkeypatch seams and limits.
  No new framework, screen hierarchy, prewarm trick or timing waiver. If a
  public contract must change, escalate the concrete design before editing.
- [x] Add focused red import/use controls, make only diagnosed existing-owner
  deferrals, and run affected behavior/import/static guards under native
  synthetic profiles. Tiny incoming Settings Hooks laziness remains distinct
  from inherited debt and must preserve its event/query/class identity seams.
- [x] Prove the exact candidate with the strict native census and source/import
  hashes. The Task 1 strict red is load-bearing: it is not parked or waived,
  and Task 3 publication cannot proceed until it passes. ADR required: no;
  existing `backlog/decisions/097-boot-budget-ratchets.md` governs direct
  import-cost deferral; any changed public contract needs a separate decision.

## Task 3: Qualify and publish the actual combined candidate

**Files:**
- Modify: backlog/tasks/task-25907.23 - Integrate-Personal-Context-memory-improvements-onto-current-dev.md, owned plan/notes/criteria/status sections through Backlog CLI.
- Modify: backlog/docs/personal-context-memory-roadmap.md, current integration checkpoint.
- Test: current Console ownership/fork/maintenance, Hooks admission/run/review, private-profile coverage, Settings/provenance/Next Send, CSS/theme/boot/import guards, changed provider setup/readiness and diagnostic inventory owners.

**Interfaces:**
- Consumes: Task 1's combined candidate and exact pinned dev; existing public PR #2862 against dev and the saved qualification runner/receipt format.
- Produces: source-bound native receipts, reviewed integration diff, truthful PR description and current tracker status.

- [x] Step 1: Select exact test modules from the automatically merged owner diff, including Tests/Chat/test_console_hook_admission.py, test_console_local_review_hook.py, test_console_chat_controller.py, test_console_runtime_lifetime.py, test_console_fork_mutation_fences.py, test_console_fork_transition_census.py and trace preparation/recovery controls. Add the existing memory UI/context/import groups and Tests/test_private_profile_coverage.py; record selection/exclusions before execution.
- [x] Step 2: Run the selection natively with separate receipts for exact private children, current import paths, source hashes and descriptor categories. Run ./scripts/preflight.sh with the existing pinned public Mermaid inputs. Check changed Python parsing/scoped lint, full lint/format for newly authored files, task readability/IDs, local document links and whitespace.
- [ ] Step 3: Review the scoped integration package against both pins. If a reviewer finds a concrete defect, repair it in its shared owner and rerun the covering checks without budget/timeout waivers. Preserve historical failures and any unavailable-history skip explicitly.
- [ ] Step 4: Rewrite the local PR-description draft around the actual candidate, native outcomes and closed gates; verify it before publication. Push normally to the existing PR branch and update the description within the user's ongoing PR authorization. If automatic approval review rejects an exact export, retain the concrete draft and explain that rejection before requesting approval.
- [ ] Step 5: Verify remote head, base, description and current checks. Close all criteria and mark Done through Backlog CLI only when verified; mirror the roadmap milestone and publish documentation-only bookkeeping. Retain original qualified receipts/worktrees and keep activation/merge outside this scope.

### Task 1 diagnosed qualification amendment — 2026-09-30

Fresh native targeted run (`/private/tmp/memory-dev-refresh-20260930/task1.xml`)
passed 166 cases and failed only the unchanged size pins: controller 29,349 /
29,210 (+139), ChatScreen 25,259 / 25,198 (+61). All CSS checks passed.
The automatically merged Hook review/dispatch and shared approval notification
bodies cause this growth. Preserve their original public/class and instance
monkeypatch seams while mechanically relocating Hook view functions (including
the typed draft dispatcher) into the existing `UI/Console_Modules/hooks.py`.
Keep controller state on its owner; move Hook approval notification, engine
lookup, admission read and sibling refusal bodies into the small
`Chat/console_run_hooks.py` helper, with live canonical controller name reads.
No signatures, policy, timing, custody or mutation order change. Retain all
existing line/method pins; cover the move with existing Hook UI/admission/review,
runtime ownership/lifetime, preparation and fork tests plus one seam regression.
Retain stricter feature preimport constants 500 / 378,740 / 123,319, while keeping
the incoming ADR-097 exception ledger as upstream history; no local raise.
Record source hashes before handover. ADR required: no; existing ADR-097,
ADR-150/161 and ADR-193 apply; this is a mechanical existing-owner correction.

Owner-placement refinement: sibling approval refusal belongs to the existing
`Chat/console_interrupt_rounds.py` permission-round owner, not the Hook helper;
retain the controller's canonical module alias. The five moved Hook view bodies
alone leave ChatScreen 25,208 / 25,198 (10 over). Also mechanically relocate
its incoming-changed `_build_console_workbench_state` projection to existing
`UI/Console_Modules/wiring.py`, preserving its class alias and live screen
`build_console_workbench_state` lookup. No new state, policy or UI values.
Governance paths: `backlog/decisions/150-design-token-system-and-design-language.md`,
`backlog/decisions/161-component-pattern-library.md`, and
`backlog/decisions/193-native-v2-profile-compatibility-and-admission.md`.

### Task 1 deterministic harness qualification amendment

`covering-hooks.xml` records 329 passed / 96 failed / 1 expected failure.
All 96 failures are `raw_source_selection_changed`: 44 local-review cases
construct a real `LocalToolProvider` whose `_default_specs` reads the admitted
config; 52 Workbench cases construct/mount the real app, likewise reading the
collection-bound config. The per-test environment redirect changes its selected
root before these reads; failures occur before the moved Hook bodies. Opt these
two source-bound modules into the existing `bootstrap_profile` marker, retaining
a synthetic private profile for their interpreter lifetime and the existing
network/keyring guards. No production admission bypass or timeout/FD relaxation.
Rerun both exact modules and the covering selection. Pin new recipient modules
at their exact measurements in the existing module ratchet; do not raise any
existing source or recipient row. The inherited strict preimport red belongs to
new Task 2 and must pass before Task 3 qualification/publication.

### Task 1 exact local-review fixture amendment

The bootstrap-profile requalification passed 115 and exposed 18 narrower
failures. Five local-review failures come from shared `_bare_controller` using
`object.__new__` without the real controller's empty `store` and
`_character_read_guards`; initialize both in that existing test owner. Four
real runtime-policy/watchlist cases explicitly change TLDW_CONFIG_PATH after
collection admission, still failing raw_source_selection_changed. Use existing
`Tests.private_profile.private_profile_test` on those exact cases with their
request fixture and retain each child's admitted private root instead of
redirecting it again. Preserve original behavior assertions, configuration
admission, isolated DB owners, and all deadlines/FD limits. Record exact native
red-to-green controls before broad covering requalification. Workbench's nine
remaining failures and typed-review expectation remain under immutable-parent
contract diagnosis before any assertion repair. ADR required: no; fixture
ownership repair follows existing ADR-193 admission without production changes.

### Task 1 immutable Workbench contract qualification amendment

Both immutable native parents reproduce all nine remaining Workbench failures
(`feature-paired-workbench.xml`, `dev-paired-workbench.xml`, with exact source
imports and private profiles). Preserve current canonical UX: explicitly open
Inspect through its existing action before visible-text checks (ADR-083/077);
retain specific recovery labels/tooltips supplied by structured readiness and
separately prove generic blocker fallback (ADR-095); account for the existing
agent-fleet section before the staged tray (ADR-137/083); include the implemented
Alt+C Context rail footer binding (ADR-031). Keep all substantive text, ordering,
action and readiness assertions. One real lifecycle defect remains: during
partial DOM teardown/recomposition, `_adapt_console_workspace_to_width` finds
the grid then unguarded rail/handle queries raise NoMatches. Move those same
four queries inside its existing QueryError guard, preserving the catch type,
responsive logic, state, line caps and timing. Add a runnable partial-grid red
control, then prove it and the direct-improve recovery journey green. The Hook
review test receives typed ToolReviewDecision; use existing normalize_tool_review
to assert both proceed verdict and approved provenance, while retaining denied
Hook refusal and the single approval round. ADR required: no; direct fixes of
existing accepted owners/contracts, with no policy or public interface change.

Exact control refinement: `workbench-controls-green.xml` passes 11 and narrows
two fixture races. Direct-improve queries the composer immediately after waiting
only for its shell, while native asynchronous recompose temporarily removes that
composer. Wait for the required composer through the existing selector helper
with its unchanged two-second bound. Inspect tests must open only when the
existing right rail reports hidden: bootstrap-profile cases can retain an open
layout, and unconditional toggle closes it. Keep all recovery/readiness checks
and production logic unchanged. Scoped Ruff formatting of moved bodies and
new recipient measurements precedes their final qualification; no existing
recipient cap changes.

Further narrowed fixture evidence: `workbench-narrow-controls.xml` reveals the
missing composer is the screen's fail-safe fallback, not merely a wait race:
`_UnavailableResolutionGateway` omits `cached_context_window`. Inherit the
existing `_ReadyResolutionGateway` offline metadata implementation and override
only the intended failing resolve method. Blocked Inspect's Alt+I action correctly
defers to its setup modal; prepare the underlying Inspect layout through the
existing explicit rail-preference opener, preserving the modal/focus guard.
These are test-double/view-preparation corrections, not production relaxation.
Retain the exact selector wait and all readiness/action text assertions.

Final staged hygiene finds only two incoming EOF blank-line defects in
Tests/Chat/test_console_provider_support.py and
Tests/LLM_Calls/test_hosted_provider_engine_handler.py. Remove only excess
EOF newlines, proving identical ASTs and substantive token streams. Preserve
the tested freeze separately and refreeze final sources; production and all
selected covering-test bytes remain identical. This is whitespace hygiene,
not prose removal for caps or an additional behavior change.

### Task 1 review fix round 1: explicit resource retirement

Review of local merge215f438403 found behavioral passes with unproven native
resource retirement. Exact weak-owner/source-free native14-case control passes
all assertions but retains per-case runs.db/agent_runs.db, injected chacha.sqlite
connections (UI and worker threads), and constructor instance locks. The three
Hooks-review increments are exact .instance.lock handles, not HookPermissions.
The helper has58calls across24files; preserve its callers and the factory's
constructor-only/borrowed-owner boundary. Shared-helper retirement placement is
under controller decision before source edits there. Independently fix the
local-review bridge's owned AgentRunsDB with finally close after progress/run
retirement. Fix receipt-service runtime with finally release of the hydration
barrier then await existing runtime.dispose; preserve5s barriers and original
assertions. Prove exact owned native paths absent after teardown, not merely
below warn-only growth200. Distinguish bounded borrowed interpreter owners:
workspace_file_roots default registry, selected config bootstrap lease, and
Logging_Config's documented process-lifetime crash stream. No production leak
inference, GC cleanup substitute, limit increase, profile/admission bypass or
Task2 source work. ADR required: no; explicit test-owned retirement implements
existing Console runtime/native193 custody boundaries.

### Task 1 review fix round 1: shared exact-owner cleanup amendment

The58-call/24-file helper census found two synchronous runtime-construction
bodies: test_agent_bridge_is_built_from_the_sibling_run_store_and_memoized and
test_agent_bridge_is_absent_without_a_durable_run_store. Preserve these bodies
as synchronous with explicit no-running-loop controls, requesting one opt-in
async owner-retirement fixture. The existing synchronous root autouse drain
requests that fixture before yield only for asyncio-marked items or that exact
opt-in fixture in request.fixturenames; unrelated synchronous tests acquire no
loop. Because outer synchronous teardown runs first, it delegates the entire
runtime -> helper-injected DB -> constructor DB/instance lock -> path drain
to the requested async fixture, including inline-import callers. Imports in
cleanup remain lazy and conditional. Existing runtime.dispose and DB quiescence
retire workers before SQLite owner handles. Capture constructor lock status/
handle exactly; Utils/instance_lock.py documents handle.close as OS release and
has no separate release helper. Preserve the factory's constructor-only boundary
for databases/locks even if app attributes are replaced. The fleet recovery
autouse fixture currently closes databases before the root async finalizer;
delegate its resource retirement to the existing shared helper/factory ledgers.
The seed-only _seed_done_primary_with_subagents also creates an AgentRunsDB,
returning IDs only: close it in finally at its existing helper, preserving all
assertions and returned IDs. Cover original14 resource-red cases, both sync
constructors, seed siblings, inline async/helper siblings and multi-app/borrowed
factory controls with exact native post-teardown path census and order events.
Diagnostic weak-owner generations must distinguish reused object IDs, and no
diagnostic strong references may replace retirement. Prove bounded borrowed
workspace registry/config lease/crash stream by exact live owner/native paths.
No production changes, global async autouse conversion, loop fallback, GC-as-
cleanup, assertion weakening or guard/cap changes. ADR required: no; mechanical
test-owner cleanup implements existing native/runtime custody contracts.

Task1 caller refinement: synchronous tree-walk characterization calls the
shared hydration _fixture_app, which injects a real DB but disables agent_runtime
and exercises only flattening. Keep the body synchronous. Split helper runtime
retirement from its synchronous exact-DB drain, mirroring factory custody:
async finalizer retires both runtime ledgers before either DB drain; sync
finalizer drains exact injected and constructor DB/lock owners without a loop.
Include the tree-walk control and transitive-helper census; no async fallback.

Task1 native order control enrolls only the three observed new admission reds:
agent bridge sibling-store memoization, absent-then-durable bridge construction,
and hydration tree-walk characterization. load_settings is bound during module
collection to the private bootstrap profile; per-case isolate_test_environment
redirects that source, so _RawParticipant's exact selected-path check refuses
before app construction. Mark these cases bootstrap_profile using the existing
source-bound private profile fixture; preserve guards, bodies and loop controls.
The first verification-control attempts also had a missing asyncio import and
mistakenly counted existing event_loop_policy as an actual loop fixture; their
red receipts remain, with the corrected precise runner/event_loop check.

### Task1 resource review: persisted sidebar synchronous scheduling correction

Native persisted ui_state.toml controls reproduce three warning-as-error reds:
ChatScreen.__init__ -> _load_sidebar_state reactive assignment -> initial
watch_sidebar_state -> _schedule_sidebar_state_save -> Textual set_timer ->
Timer._start create_task(_run_timer()) without a running loop. Exact allocation
stacks and immutable AST method equality at feature e352, dev dfee, merge215 and
current distinguish this inherited shared-owner failure from Hook growth. The
scheduler's callers are watch_sidebar_state (reactive sidebar loading, toggles,
expand/collapse/reset) and handle_reset_settings's explicit same-value reset.
No reusable running-loop predicate exists; preserve dirty/revision tracking then
use stdlib asyncio.get_running_loop's RuntimeError boundary before any timer
creation. Existing mounted debounce, worker persistence and quit flush remain
unchanged. This is a production shared-owner correction, not timer mocking or
warning suppression. Add a deterministic persisted-state synchronous control,
retain all three original sync bodies/no-loop checks, and cover actual mounted
restore, debounce, cancellation and flush owners. Keep screen/method/import/FD/
deadline caps unchanged. ADR required: no; routine bug fix preserves existing
sidebar/runtime boundaries and ADR097 direct paydown/ADR193 admission contracts.

Four selected real helper siblings (mount-claim ordering, marks+runs seeding,
launch-vs-screen hydration with two apps, and source-unmarked auto-async automatic
work pause) each reproduced raw_source_selection_changed before app construction
in native execution. Enroll exactly these observed nodes bootstrap_profile,
preserving real admission guards. The98 transitive caller census finds95async
bodies (six rely on existing pytest asyncio_mode=auto, which marks collected
items before fixtures), and only the3documented synchronous bodies.

Task1 sidebar refinement after focused sibling execution: hydration constructs
ChatScreen on a running asyncio loop before app.run_test installs Textual's
active_app context. Persisted restoration arms the same sidebar timer; its
_tick and _stop_all fail with LookupError(active_app), disrupting app shutdown
and contaminating that run's resource census. Preserve dirty/revision tracking
and defer timer creation while not is_mounted, then retain the running-loop
boundary. Add a real async-unmounted persisted-state regression (no timer fake,
new loop, or mount hierarchy), and verify mounted debounce lands and immediate
quit flush persists. Rerun siblings in a fresh interpreter before attributing
residual resource growth. No foreign-thread close or factory ownership expansion.

Task1 exact worker-owner retirement refinement: fresh corrected siblings passed
all six bodies but five resource finalizers failed. Isolated seeded-wake case
reproduces five surviving native AgentRuns DB/WAL/SHM handles: generation3 retains
one asyncio_0 connection after MainThread.close. Creation stack is wiring's
_seed_console_fleet_history -> bare asyncio.to_thread(wake.seed_from_marks) ->
pending_wake_conversation_ids. The same coordinator's recover/row worker calls
already use base_db.run_owned_db_call and retire their asyncio_4/1 handles;
there is no evidence for foreign-thread or factory close-all. Reuse that existing
finite operation-owned helper at the exact UI seed scheduler, preserving custom
wake mocks without a DB getter via None (the helper's existing custom-owner path).
No new interface, limit change, or coordinator/public identity change.
Hydration independently opens a factory WorkspaceDB worker handle through
workspace._read_console_workspace_scope -> bare to_thread(get_workspace_scope).
Its read/write shared persistence twins are called by the existing scope selector
at5437/5497; use the same helper with their exact registry db, keeping existing
in-memory guards and custom-owner behavior. Existing operation_owned_connection
closes only newly opened current-worker handles; borrowed transactions remain
owned. Preserve all assertions, qualify isolated red-to-green and the six sibling
population, custom wake seam and actual scope persistence. Final 22 resource
population, mounted/sidebar lifetime and static caps follow on final bytes.

Task1 owner-control fixture refinement: the wrapper control run passes41 bodies;
new queued UI control and the existing getter-less wiring control assign partial
store/controller doubles through canonical ChatScreen runtime properties. Their
session doubles have no id; final runtime disposal raises AttributeError before
the exact DB/lock drain, contaminating subsequent finalizers. Preserve all
behavior assertions and use the existing monkeypatch fixture for these exact
canonical assignments so original app-owned objects restore before async owner
cleanup (monkeypatch is set up after the delegated root cleanup fixture).
No production disposer weakening or extra factory ownership. Qualify these two
controls in a fresh interpreter before attributing any residual handles.

Task1 control restoration ordering correction: the fresh two-control run proves
other root autouse fixtures request monkeypatch before the async owner fixture;
its general undo therefore runs too late. Use explicit test-local try/finally to
restore captured canonical store/controller before the body returns, then close
standalone original/replacement DBs. Do not rely on fixture-order assumptions;
retain monkeypatch for dispatch replacement and all original behavior assertions.

Task1 captured-None finite seeding amendment: seed_from_marks(database=None)
means re-read the current bridge, so the initial same-line wrapper cannot freeze
None across a queued None-to-DB replacement. Add one private finite-seed method
on the existing coordinator: capture once, return0 when absent before enqueue,
otherwise use run_owned_db_call(db,seed_from_marks,database=db). Wiring resolves
that method at invocation and preserves its existing zero-arg to_thread fallback
for custom doubles. Recovery keeps its already captured ledger._db unchanged.
Add exact queued-replacement and captured-None UI controls; no default/sentinel
public contract change. No real functools deduplication exists, so no import move.
Fresh dispatch evidence further identifies hydration's remaining Workspace handle
at _resume_console_workspace_conversation -> retrieval scope resolver ->
chat_rag_events.resolve_scope_for_session cached branch -> bare to_thread bound
get_workspace_scope. Reuse run_owned_db_call with the exact captured registry_db
there; the fresh-scope branch owns/closes a separate short-lived DB already and
is unchanged. The approved workspace read/write twins remain their own direct
qualification. Rerun hydration and auto siblings fresh before further attribution.
Exact getter-less wiring node's native raw_source_selection_changed is enrolled
via existing bootstrap_profile; keep config admission guards intact.

Task1 correction to cached/fresh ownership diagnosis: the new two-branch native
control passes cached=True with exact borrowed MainThread identity intact;
fresh=False fails worker_leases and retains four native Workspace handles plus
its admission lease. Source902-932 shows _read_fresh_workspace_scope_sync uses
registry_service.db.connection() directly, not a separate short-lived DB; the
earlier claim was incorrect. Reuse run_owned_db_call for this fresh branch as
well, with the captured registry_db and same bound service passed to its existing
reader. Registry service db is assigned once by __init__ at640 (no later service
assignment in source); both callbacks read that same service-owned DB. Leave
fresh JSON/version/existence/fail-closed parsing and custom/in-memory branches
unchanged. No synthetic foreign-thread close or factory custody expansion.

Task1 final finite-read refinement: valid native historical control is red only
for the closed-cache case (registered MainThread connection survives); the
borrowed active transaction control passes. The original first control attempt
used a diagnostic os.open replacement and tripped the repository's capability
identity guard; that diagnostic was removed, no guard bypass was made, and the
valid red receipt is task1-history-owner-red-valid. Place existing
operation_owned_connection around historical_snapshot's uncached derive call.
The read remains permitted by the existing read admission boundary during late
UI callbacks; only its newly acquired connection retires. Cache hits, actual
snapshot data, borrowed transaction and pause/refusal behavior remain intact;
no runtime resurrection/reattach or broad lifecycle fence.
Use the approved existing coordinator private finite-seed method with a concrete
getter-less receiver accommodation: zero-arg to_thread fallback only when no
callable getter exists; native getter-present None returns0 before enqueue.
Wiring late-imports the canonical class and calls that method unbound, preserving
live class/seed lookup and the existing custom test seam. This fits2399 after
formatter checks without compression or import/prose/blank stripping.

Task1 formatted wiring paydown: Ruff's required local-import separator makes
wiring2400 against2399. Replace the module's sole functools.partial wrapper with
its equivalent zero-argument lambda over the per-invocation wake local (never
reassigned after callback capture), and remove the now-unused local import.
This is actual dependency/indirection removal, not prose/blank stripping; keep
callback execution deferred, canonical class lookup at worker invocation and
all scheduling/custom/queued assertions. Extend queued control to replace the
canonical class method after scheduling and verify the same receiver reaches
that live method. Source is then frozen before final native ownership checks.

Task1 Textual Worker correction: revoke the proposed zero-arg lambda. Installed
Textual8.2.8 Worker._run_async accepts coroutine functions, partial.func that is a
coroutine function, or an awaitable; a normal lambda returning a coroutine is
rejected before invocation. No lambda-version run is qualification evidence.
Retain the lazy async partial and move the existing seed retry/error metadata
orchestration into the same coordinator private finite-seed owner, removing the
redundant nested wiring wrapper. This is substantive ownership/dependency reuse
and preserves cancel-before-start allocation timing. Canonical class is imported
and its method resolved when scheduling; the real bound seed remains read at
execution and DB captured before finite offload. Recovery stays unchanged.
Verify real native Textual Worker execution/cancellation alongside custom,
queued bridge replacement, captured-None and live class lookup at scheduling.

Task1 diagnostic inventory refinement after configured static guard: the existing
history-seed exception handler moved from wiring to the existing coordinator's
private finite seed owner. The persistent inventory correctly detects one
TASK494 owner disappearing and one TASK492 warning added; native statement
review against HEAD shows identical exception_type-only interpolation and
message, with formatting only. No user text, path, secret, sink, or new diagnostic
was introduced. Rebuild Docs/security/production-diagnostic-inventory.json for
this reviewed owner relocation and rerun the exact inventory guard. Preserve
original red receipt (245pass/1fail/1historical skip). No production-byte change,
limit change, diagnostic-contract broadening, or new ADR is needed.

Task1 native worker/finalizer-order refinement: final frozen28-case population
passed bodies but retained an injected chacha.sqlite FD after registry0. A
source-free attribution rerun instead retained private config ChaChaNotes
DB/WAL/SHM: actual creator was app_feature_glue backfill -> config lazy getter
construction/seeding on asyncio_4. The root isolation finalizer resets config
while canceled Textual thread work can still be running; Worker cancellation
is not native executor completion (installed Textual8.2.8 Worker._run_threaded
and existing app_lifecycle1327-1356 document normal Runner.close joins).
The original injected FD1 remains open evidence until final joined population
proves retirement, not inferred to be this separate config race.
Record deterministic held native Textual worker red control, then make existing
retire_test_app_owners explicitly depend on isolate_test_environment, encoding
async owner teardown before original config reset/environment restoration.
Dispose both runtime ledgers, then public loop.shutdown_default_executor using
existing WORKER_CANCELLATION_GRACE_SECONDS3.0 only for actual nonempty app-owner
ledgers. Its timeout warning is a hard custody failure: do not subsequently
retire DB/locks/paths. No private executor manipulation, unbounded join, new
config reset path, GC cleanup, factory custody expansion, production policy or
limit change. Preserve sync-body/no-loop controls and unrelated async cases
without app owners; qualify fixture dependency order/admission and exact28-case
resource population on refrozen bytes. ADR required:no; direct existing native
worker ownership implementation, with ADR193 source admission preserved.

Task1 executor failure-chain refinement: corrected native held-worker red is
1body pass/1teardown error: config reset preceded callback completion with
private config DB/WAL/SHM retained. Explicit isolation dependency/public3s join
passes that exact control and unrelated sync/no-loop2. Direct timeout control
alone cannot prove the later isolation finalizer: actual fixture-chain fault
injection confirms join warning raises, yet original isolate finalizer closes
its seeded config DB/cache. Preserve this exact expected-failing chain receipt.
Set one request-node uncertainty flag before join, clear only after success;
existing isolate finalizer refuses prompt/config retirement while flag remains
(timeout/cancellation), without a second reset path, helper/framework, or broad
DB custody. Direct finite timeout/cancel controls use separate request node so
intentional caught unit faults do not poison the real successful fixture chain.
Repeat actual chain fault injection to prove seeded config native handles,
constructor DBs and exact lock stay owned/open after refusal; normal native
worker/timeout/cancel/no-app controls and joined affected population follow.
No-app async spy preserves normal Runner.close positional300s join while rejecting
app-owner3s; its prior TypeError is an observation-control fault, preserved rather
than credited as cleanup success. Diagnostic config getter closure bug affected
order-trace/pre-GC/first held-control receipts only, corrected by capturing the
original function; original frozen28 red and initial attribution28 red remain
valid. No GC invocation was added or used as a fix; limits/production bytes stay.


Task1 final sidebar source-lifetime fixture refinement: final native joined
sidebar selection passes13 real sidebar parent cases/11 exact private children,
but the standalone IO-lifetime case fails before its original IO assertions:
_LIFETIME creates object.__new__(ChatScreen) without Textual _is_mounted,
so existing Widget.is_mounted raises AttributeError at the shared scheduler.
Its preexisting set_timer fake already models mounted scheduling. Initialize
only screen._is_mounted=True in that bare fixture; preserve every cancellation,
flush/latest-revision assertion and35s/10s deadline. This completes intended
fixture state, not actual mounting; the separate real native mounted cases
supply debounce/quit behavior proof. Rerun that exact parent and its native
subprocess, refreeze test bytes with all production hashes unchanged. No new
production edit, timer fake, warning filter or limit change. ADR required:no;
test-only direct completion of existing lifetime control.


Task1 early runtime retirement failure refinement: controller source review
finds the uncertainty flag is set after factory/attached runtime disposal,
although ConsoleRuntime.dispose may raise or cancel before executor join.
Detect the exact nonempty app-owner ledgers and set that same request-node
flag before the first runtime await; clear only after successful public bounded
executor join. Preserve runtime-before-DB/lock/path order, synchronous no-loop
and unrelated async gates,3s deadline, original isolation dependency/reset.
Add exact RuntimeError/cancellation early-disposal red controls proving flag
and no DB/lock/path drain, then relocate only those existing statements and
prove actual fixture-chain config refusal under each fault. This is the same
test custody correction, no production change/new framework or limit change.
Strict import measurement remains its fresh native registry walk, not snapshot:
record actual prewarmed baseline module paths/hashes to explain unchanged541/
406612 despite Console source edits. ADR required:no, existing native owner
retirement/source admission contracts are unchanged.


## Task 2 approved scope amendments (canonical-copy reconciliation)

The Backlog plan and private brief were amended before source edits. This canonical copy was reconciled afterward on 2026-09-30; it does not claim earlier amendment timing.

Follow-up trace: Chunking Lab catalog imports AUTO_SENTINEL from Chunking/auto_selection.py, which still eagerly reaches planner/templates/runtime. Add this exact owner and the whole Chunking Lab screen to the RED entry-point controls, then defer planner/classifier/template aliases at their existing use sites. Native census must prove this is actual whole-pass shedding. Evals convention binding is paused: source proves changed subclass-MRO and same-handler-name collision dispatch; retain original decorators pending controller decision.


Controller-approved ordinary-use expansion: defer optional MCP visual imports at UI/MCP_Modules/mcp_workbench.py existing compose/secondary-mode/query/handler use sites; keep permission DTOs and every current Message definition and binding unchanged. Preserve lazy public aliases and runtime get_type_hints using qualified self-owner annotations where necessary. STTS/Personas remain unchanged because STTS has type-bound decorators to profile-library-defined Messages. After a fresh serial census, defer only needed Evals execution/sample-bench/inspector dependencies at existing use sites. No event extraction, new hierarchy, dispatcher, source stripping, cap increases, prewarm or route changes. ADR097 governs same-contract deferral. Enroll only the six demonstrated source-bound Chunking/Library test modules with the existing bootstrap_profile marker after exact BASE reproduction.


MCP cap-preserving amendment approved by controller: retain workbench6744 cap; move only existing pure _import_summary/_import_severity bodies verbatim into UI/MCP_Modules/mcp_profile_form.py, retain their lazy original aliases and monkeypatch reads, and add an exact measured recipient row to test_module_size_ratchet.py. Use one private self-module reference with cached PEP562 exports and qualified actual-use/annotation names, preserving all message definitions, handlers and permission DTOs/gates. No generic facade or dispatcher.


Measured-needed Evals phase: MCP census is494modules/382478LOC, leaving3738LOC total debt. Defer only evals_screen existing execution aliases (character/word runners), sample_bench, inspector classes and skill_eval_launch helpers; defer library_rail sample_bench until existing compose use; defer skill_eval.runner judge/prompts/scoring/simulation/static dependency aliases until existing run/parser use while keeping estimate_calls/max_estimate_calls eager for panel class constants. Preserve exact real aliases via existing cached PEP562 pattern, annotation owner lookups, all original Message classes/decorators/callables, and caller patch seams. Add fresh-process import controls plus actual synthetic Evals worker/UI, annotation and export controls; serial fresh census determines success. No source/module extraction or behavior changes.


Final measured-needed Evals execution seam:484modules/378882LOC remains142LOC over. Existing evals_screen skill_eval.subject imports (SubjectError, subject_from_directory, subject_from_store) are worker-only and the195LOC owner has no other Evals eager importer. Add those exact aliases to the existing lazy map; preserve validation, error class and caller patch identity. No new owner or contract change.


Fixture reconciliation extension: exact BASE Evals-owner overlay reproduced raw_source_selection_changed in both test_evals_screen and test_evals_character_run_e2e before relevant behavior (evals-base-red receipt). Controller authorizes existing bootstrap_profile enrollment for only these two demonstrated modules; keep all admission/network/keyring guards and test body assertions unchanged.

### Task 3: Reviewed qualification before publication

ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/193-native-v2-profile-compatibility-and-admission.md
Reason: execute existing integration qualification and publication boundaries; no new storage, authority, interface or runtime decision.

BASE: 1b076bde0f8bdf50c4241233cca0ec40a42c6a1c. Task1 and Task2 are independently reviewed. Task2 current native invoice is483modules/378681LOC/105452max-route; limits500/378740/115452. Two nonblocking warnings remain explicitly recorded (unchanged vendor regex and intentional59LOC margin).

Phase A: record the exact current targeted selection/exclusions/reuse before native execution; consume /private/tmp/memory-dev-refresh-20260930/task3-selection-draft.json and frozen Task1/Task2 receipts. Qualify current joined resource-owner population using the existing source-free native probe, collecting the actual current cases rather than relabeling historical32. Run current memory/Console/UI/context/provider/Hook/boot/generated/style/task and private-profile checks for changed owners; preserve all failures and exact child/source/import/descriptor evidence. No full sweep or parallel heavy UI/worker/import census. Preflight uses the eight verified pinned public source inputs at /private/tmp/memory-failure-repair-20260929/mermaid-inputs; verification receipt task3-offline-inputs.json is not a generated-preflight pass.

Write a concrete local PR-description draft with actual source pins, outcomes, warnings, reuse and closed gates. Update qualification notes through Backlog CLI and the roadmap without advancing open criteria/status. Commit only authorized Task3 changes, freeze source/evidence, and report for independent task and final broad review. Do not publish yet.

Phase B: after independent task and whole-branch review, resume the original Task3 implementer for normal existing-branch push and truthful PR-body update under ongoing authorization. Verify remote head/base/body/current checks; only then close criteria/status through Backlog CLI and publish metadata-only bookkeeping with exact source-identity proof. No dev merge/activation/force push/housekeeping. Exact automatic export rejection preserves the concrete draft and requires its stated reason to be reported before an exact approval question.


### Task3 native finite transcript-reader correction

Task3 current-byte resource correction: resources.xml executes34 bodies successfully but reports4 teardown errors, first at the two-app hydration equivalence case. Native owner events prove AgentRunsDB closes at runtime disposal then a queued transcript projection reopens its MainThread handle via change_review_marker_messages. Trace all callers and probe sibling resume_marker_messages before correction. Add targeted RED borrowed/reopened native controls in the existing finite-retirement test module, then wrap only proven finite durable-read owners in existing operation_owned_connection, preserving records, fallbacks, borrowed transactions, in-memory/custom behavior and all caps/deadlines. Repeat focused GREEN/durable-marker controls and current joined resources; update source freezes and rerun affected import/inventory guards after actual source changes. ADR required: no new ADR. ADR path: backlog/decisions/097-boot-budget-ratchets.md; existing finite operation-owned connection contract. Reason: enforce the existing finite-read lifetime contract without new authority, storage, runtime disposal policy or framework. No publication or criteria/status advancement. Canonical diagnostic inventory also reports Task2 MCP workbench47-call digest drift. Canonical --statements and exact lookup-normalized AST counters prove unchanged logger calls/arguments and persistent sinks; regenerate only the source-derived MCP owner digest and verify the inventory guard.


### Task3 exact source-admission fixture enrollment

Task3 exact fixture admission correction: native wake-attribution records RecoveryRequired(raw_source_selection_changed) from read_hooks_config_snapshot and early Hooks unavailable refusal before unchanged wake-token PermissionError validation. marker-green separately reaches the same source-admission rejection from Internal_Prompts before durable marker generation. Enroll only test_the_wake_exemption_never_outlives_its_turn and test_resume_re_derives_the_summary_row_byte_identical with the existing bootstrap_profile marker. Preserve every original body/assertion, PermissionError contract, Hook/wake gates, deadlines and source identity. Run the exact controls and full affected runtime-lifetime module, plus the planned finite-reader/resource groups. ADR required: no. ADR path: N/A. Reason: existing synthetic profile fixture enrollment, no production behavior or authority change.


### Task3 automatic-library fixture refinement

Task3 automatic-library fixture refinement: interrupted incoming qualification preserves255passes/3failures including original300s provider-event timeout. Exact native first-two controls and immutable BASE reproduce RecoveryRequired(raw_source_selection_changed) through Hook config admission before submit-task registration/provider resolution. Controller authorizes existing bootstrap_profile module enrollment only for Tests/Chat/test_console_automatic_library_preparation.py, whose64 original test bodies share synthetic controller/store/provider setup. Preserve every body/assertion, source/network/keyring/Hook/permission guard and300s deadline; no production change. Run the whole module clean, then remaining incoming owners with exact deduplication and retain interrupted evidence. Any distinct later failure requires its own attribution. ADR required: no; existing synthetic source-admission fixture contract, N/A new ADR.


### Task3 diagnostic snapshot refresh

Task3 diagnostic snapshot refresh: read-only native warm census repeats1033/cap1033 versus diagnostic snapshot1030, adding only Chat.console_chat_persistence, Chat.console_run_hooks and UI.Console_Modules.hooks. App import repeats678/cap686 versus snapshot669 (+22/-13 exact owners retained). These are diagnostic module-name baselines, not equality/LOC gates. Controller authorizes canonical unforced scripts/update_boot_budget_snapshots.py --only ui-ready --only import-weight under native private profiles, preserving before/after lists and original warnings. No force, cap, route, deadline or source-deferral change. Verify only the two snapshots change and rerun exact covering guards on final metadata. An over-cap measurement remains RED. ADR required: no new ADR; existing ADR097 deliberate snapshot writer governs.


### Task3 closed-loop control refinement

Task3 closed-loop control refinement: exact admitted immutableBASE and current native observation prove the emergency-detachment fixture closes while hook_admission_reason is still awaiting asyncio.to_thread after20zero-delay ticks. Replace only that polling with existing held.wait under unchanged outer300s pytest deadline, preserving all original emergency assertions; disclose this one body exception and retain remaining63 bodies unchanged. Add real-task registered-COMMITTING and earlier pending-Hook RED controls for native ContextVar custody and sys.unraisablehook, asserting cleanup before GC. Trace existing begin_shutdown/_detach_closed_submit_tasks and maintenance-call ownership before choosing a production correction. Public Python3.12 Task.get_context is available, not a policy waiver. No production edit until validRED and controller owner-contract ruling; no warning/deadline suppression or GC-as-fix. ADR required: no for fixture/RED controls; any newly discovered boundary change requires explicit scope review.


### Task3 closed-loop owner correction

Task3 closed-loop owner correction, controller approved after5native REDs: relocate existing active-submit registration/final unregister to submit_draft whole diagnostic wrapper, preserving original Hook-before-capture refusal, live-state publication, immutable request identity and cancellation order. Existing detachment owner finalizes only exact captured closed-loop submit coroutines in native Python3.12 Task.get_context; retain pending Task state/destroyed diagnostic and exclusive-preparation/maintenance custody. Diagnostic scope handles GeneratorExit alone with existing monitor.close(timeout=0), preserving owner-context token reset and original awaited normal success/error/cancellation drain. No foreign loop/worker adoption or new cleanup framework. Keep all caps/deadlines/prose. Qualify all5RED variants, original86module cases, diagnostic normal/error/cancel and monitor seal/drain, then fresh complete affected Console selection on frozen final source (initial1214passes no longer qualify this changed ownership contract), plus incoming/Hook/lifetime/resources/static/size/import/generated guards. ADR required: no new ADR. ADR paths: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/197-console-hook-configuration-review.md; backlog/decisions/097-boot-budget-ratchets.md. Reason: direct enforcement of accepted turn custody and unchanged consent/budget boundaries.


### Task3 diagnostic fixture enrollment

Task3 diagnostic fixture enrollment: exact immutableBASE diagnostics-base reproduces Collector app-import RecoveryRequired(raw_source_selection_changed) and Hook-before-sink refusal. Controller authorizes only Tests/Chat/test_console_send_diagnostics.py existing bootstrap_profile marker; preserve all13 original bodies, sink/privacy/normal error/cancellation assertions and every source/admission/network/keyring guard. Include the whole owner module in final current-source qualification with deduplication; carry5closed-loop nativeGREEN variants and actual monitor seal/drain/foreign-context proof. Preserve original BASE/setup failures and unchanged29210controller cap. ADR required: no; existing fixture source-admission contract, no production policy change.


### Task3 Hook regression fixture enrollment

Task3 Hook regression fixture enrollment: current native and immutable BASE hooks-regression-attribution/base reproduce read_hooks_config_snapshot RecoveryRequired(raw_source_selection_changed) before the intended fake Hook callback. Preserve incoming-frozen934passes/1failure/14deselected as failed partial evidence. Controller authorizes only Tests/Chat/test_console_run_hooks_regressions.py existing bootstrap_profile module marker; preserve all14 original body ASTs, Hook refusal/retry/permission/privacy assertions, source admission/network/keyring guards and timeouts. Run the full module clean, then fresh exact incoming coverage with explicit deduplication. No production gate change or wider enrollment. ADR required: no; existing synthetic source-admission fixture contract, N/A new ADR.


### Task3 obsolete close-call fixture reconciliation

Task3 obsolete close-call fixture reconciliation: incoming-final retains932passes/1failure/14deselected; exact immutableBASE close-api-base reproduces AttributeError for the removed ConsoleChatController.close_session API. Controller authorizes replacing only this call in test_console_send_gate_queue_race.py with existing Tests.Chat.console_close_helpers.close_controller_session, whose real lifecycle revision/begin/finalize contract applies to this already-quiesced test with no active tasks/workers. Preserve raw cancellation-before-removal observation [(session.id,True)], empty-store assertion and all six sibling bodies; no production shim, bypass or policy change. Run the full7case owner and remaining14incoming modules. Reuse only the exact932canonical cases from eight completed modules while every production/dependency byte is unchanged; retain failed batch status and record exact14deselected IDs/reasons. Any production correction invalidates reuse. ADR required: no new ADR; direct existing two-phase close fixture contract under ADR094 turn lifetime.


### Task3 queue fixture admission

Task3 queue fixture admission: close-api-final records the corrected close case passing and a distinct sibling failing before its commit barrier. Exact queue-admission-current and immutable queue-admission-base reproduce guarded raw_source_selection_changed at Hook config snapshot before queue/commit behavior. Controller approves only test_console_send_gate_queue_race.py existing bootstrap_profile module enrollment. Preserve six original async body ASTs, corrected synchronous close assertions, all10/30/300s waits and transaction/queue/raw-order/source/network/keyring guards. Run full7owner then exact14remaining incoming modules. Production/dependency bytes remain fixed for932prefix reuse; any production correction invalidates it. Retain actual pre/post chronology and14exactdeselected IDs/reasons. ADR required:no; existing synthetic admitted-profile fixture, no production policy change.


### Task3 queue immutable-custody fixture reconciliation

Task3 queue immutable-custody fixture reconciliation: queue-owner-final records3passes/1failure; admitted immutableBASE queue-custody-base reproduces Queued prompt has no frozen custody request from the unhealthy-recovery fixture direct registry.admit setup. Controller authorizes only replacing that setup call with await controller.queue_prompt, same session/text/expected_revision. The existing method captures real immutable configuration/custody and retains Hook/maintenance/staged-rider/revision gates. Preserve real drain and every original recovery/zero-provider/transaction assertion and wait; no production change. Run all7without maxfail; disclose both obsolete-close and queue-custody body exceptions, with remaining five original bodies and all assertions unchanged. ADR required:no new ADR. ADR path: backlog/decisions/098-visible-bounded-console-prompt-queue.md. Reason: direct accepted owning-session immutable queue-custody contract.


### Task3 turn-context fixture enrollment

Task3 turn-context fixture enrollment under controller conditional ruling: incoming-remainder records3passes/1failure/13deselected. Exact native turn-context-current and immutableBASE turn-context-base both prove read_hooks_config_snapshot RecoveryRequired(raw_source_selection_changed) before intended frozen-argument identity assertions. Test module bytes equalBASE. Enroll only Tests/Chat/test_console_turn_execution_context.py with existing bootstrap_profile marker; preserve all55 original bodies/decorators/assertions and all admission/source/network/keyring guards. Run complete owner then remaining13 incoming modules. No generic bypass, production, API or body correction is authorized by this pattern. ADR required:no new ADR; existing synthetic admitted-profile fixture contract.


### Task3 turn-library fixture enrollment

Task3 turn-library fixture enrollment under controller conditional ruling: incoming-remainder13 records1failure/13deselected at immediate capture ordering. Exact native turn-library-current and immutableBASE turn-library-base both prove read_hooks_config_snapshot RecoveryRequired(raw_source_selection_changed) before intended configuration/authority/RAG/provider behavior; original module bytes equalBASE. Enroll only Tests/Chat/test_console_turn_library_authority.py with existing bootstrap_profile marker. Preserve all18 original test body/decorator ASTs, ordering/authority/privacy assertions, source/network/keyring/admission guards and deadlines. Qualify complete owner before remaining12incoming modules. No production, API or body repair under this ruling. ADR required:no; existing synthetic admitted-profile fixture contract.


### Task3 Library fixture contract reconciliation

Task3 Library fixture contract reconciliation, controller-approved after full current and admitted immutableBASE both19passes/13failures. Native library-policy-observed proves fake coordinator Allowed/Automatic disagrees with real temporary holder Never/Blocked, correctly narrowed at handoff; this is existing Console ADR079 destination disclosure, not gated Personal Context disclosure. Seed only three disclosure fixtures real session holders from each own already-selected coordinator.snapshot before capture, preserving every disclosure/endpoint/owner/settlement assertion and Blocked negative. In only two queue fixtures, replace obsolete configuration-count expectations with admission capture/no dequeue recapture, adding exact queued configuration identity and fresh execution Library policy/gateway observations; preserve recovery-after-claim/provider-boundary/privacy assertions and all waits. Report five body exceptions and unchanged remainder. Clarify ADR079 broad queue-capture sentence narrowly: Library authority and resolved destination after dequeue, immutable screen configuration at queue handoff under ADR094. No production edit/new ADR/policy change. Existing ADRs: backlog/decisions/079-console-library-conversation-authority.md; backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md. Run complete32 after serial Console19, retain all REDs and source-bound proof. Production bytes remain fixed for prefix reuse.


### Task3 Library observation boundary refinement

Task3 Library observation refinement within the same two approved queue bodies: turn-library-qualified30passes/2AttributeErrors came from a newly authored attempt to read custody_request on the intentionally body-free public snapshot; turn-library-final30passes/2identity failures came from final ConsoleTurnExecutionContext.__post_init__ deliberately detaching configuration. Observe exact callback object identity at existing _capture_turn_library_authority input through a wrapper calling the real owner unchanged, and assert full-value equality in the final provider context. Recovery input must be distinct after claim; queued input must be the admission object. Preserve policy revisions9/11, gateway/count/reservation/attempt/privacy assertions and all deadlines. No production/API expansion; existing ADR094 immutable custody and complete-context detachment remain intact. Retain both failed attempted observations and rerun complete32 under a fresh tag. ADR required:no newADR; existing ADR079/094 contracts.


### Task3 provider CAS fixture private-profile correction

Task3 provider CAS fixture private-profile correction, controller-approved after exact three fixture functions/five cases fail on current native and immutableBASE at get_atomic_config_snapshot with raw_source_selection_changed before CAS behavior. The fixtures explicitly reselect TLDW_CONFIG_PATH, so bootstrap marker alone is insufficient. Use existing Tests.private_profile.private_profile_test plus request and seed identical TOML into Path(config_module.get_cli_config_path()) already selected by each unique native child, retaining env assignment to that same path. Preserve all original CAS/conflict/privacy/generation/readback assertions, TOML, parameter IDs and300sdeadline. No source rebinding/admission bypass/production or profile-helper change, and no parent config writes. Capture exact five child/native/source receipts without inherited external proof plugins; full changed owner and remaining12covering selection must pass, with438pass/1fail kept as failed partial evidence. ADR required:no newADR; direct existing private-profile source custody under ADR193.


### Task3 native splash continuation diagnosis

Task3 native splash continuation diagnosis: current joined resources-observed-joined retains33passes/1ScreenStackError at app.on_splash_screen_closed after awaited native widget removal, with all34owned/unexplained counters zero. The original Pilot timeout did not recur and remains unattributed. Source-free isolated hydration passes without any splash callback, so it is observation only. Controller authorizes a deterministic native existing-owner RED in test_splash_initial_screen_preimport.py: deliver the actual Closed message, hold continuation after real splash removal, then allow real Textual shutdown to drain screens before release. Record public is_running and diagnostic private state, preserve actual cleanup and original deadlines. Trace all callers and existing startup shutdown guards before any production correction; no fixture splash suppression or new lifecycle state. ADR required:no for regression diagnosis; existing app-exit lifetime under ADR094 and unchanged ADR097 budgets.


### Task3 native splash continuation repair

Task3 native splash continuation repair, controller-approved after two deterministic native REDs at the exact ScreenStackError. Actual real shutdown properties are public is_running True→False, private _shutting_down False, stack0. Wrap the entire existing post-removal continuation in if self.is_running; inline only the sole main_ui_widgets temporary at its for-loop use. Normalized continuation AST, calls/filter/logging/order and every existing comment remain equivalent, while one added guard line and one removed single-use assignment keep app.py5712under unchanged5712cap. No list.remove rewrite, logging/prose/blank deletion, new owner/import/private lifecycle flag or deadline change. Use existing ADR036 application composition lifetime and ADR097 ratchets, no newADR. Run native regression GREEN with supported junit_family=xunit1, full existing splash/no-splash and skip-on-keypress owners, fresh size/import/strict census/joined resource population. Any App production change invalidates prior932prefix reuse; rerun exact incoming population and assess other receipts only by precise unchanged contract/caller proof. Keep every failed joined/observer/RED receipt and unresolved earlier navigation timeout causal concern. Amend before source and do not mutate App while another native selection is importing/running it.


### Task3 navigation-continuity fixture enrollment

Task3 navigation-continuity fixture enrollment under the controller conditional ruling: incoming-remainder11 retains379passes/1failure/1existingxfail/13deselected. Exact navigation-source-current and immutableBASE navigation-source-base both fail at _build_test_app→load_settings→raw_source_selection_changed before app construction or custody/privacy behavior. All nine original test function ASTs and module bytes equalBASE before marker. Enroll only Tests/UI/test_console_turn_navigation_continuity.py with existing bootstrap_profile module marker; preserve all bodies/assertions/5second event waits and every admission/network/keyring guard. Run complete owner and later full incoming selection on repaired App bytes. Retain unchanged runtime-ownership TASK32873 xfail as limitation, not a new waiver/pass. ADR required:no newADR; existing synthetic-profile source custody under ADR193.


### Task3 exact splash-owner fixture enrollment

Task3 exact splash-owner fixture enrollment, controller-approved after current splash-owner-final31fails/2passes and immutableBASE splash-owner-base31identicalfails/1pass (new admitted native regression is the extra pass). Enroll only Tests/UI/test_splash_initial_screen_preimport.py and Tests/UI/test_splash_skip_on_keypress.py with existing bootstrap_profile module markers; preserve existing asyncio marker, all original15+6function ASTs, settings,31failing IDs/assertions/guards and deadlines. Every failure is raw_source_selection_changed before intended behavior. Qualify all33current cases with supported junit_family=xunit1 for regression state properties; no body/production/admission bypass. ADR required:no newADR; existing synthetic profile source custody under ADR193. App repair remains governed by ADR036/097 and exact5712cap.


### Task3 navigation gateway fixture contract

Task3 exact navigation gateway fixture API repair: navigation-owner-final and admitted immutableBASE navigation-admitted-base both reproduce missing _TwoChunkGateway.cached_context_window during real screen mounting. ChatScreen active context estimates call this synchronous metadata method; existing _ReadyResolutionGateway.cached_context_window uses only its settings argument and pure resolve_context_window, no self fields or metadata/network calls. Import the already-loaded helper beside _configure_native_ready_console and bind its existing method on only _TwoChunkGateway. Preserve shared _StallingWakeGateway, all nine original test bodies/assertions/barriers/custody/privacy/deadlines. Verify inherited function identity/offline implementation, scoped import Ruff, complete nine owner without maxfail, then full incoming. ADR required:no newADR; fixture-only repair of existing gateway contract, no production fallback.


### Task3 finite metadata worker retirement

Task3 native finite metadata retirement, controller-approved after current navigation resource control retains13 owned native handles and admitted immutableBASE retains16. Actual worker stacks identify ChatPersistenceService._require_workspace_scope registry get_workspace and ConsoleAgentBridge.resolve_run_log_target metadata lookup; independently called run_log_available/load_run_log_page/load_run_log_text share _owning_run_id_for_log. Add real-worker fresh, stale raw-closed cache and active borrowed transaction RED controls for each finite owner, covering valid/unknown workspace and primary/drill/unknown/mismatched run metadata. Then reuse only existing operation_owned_connection around those three reads/resolution boundaries with local imports. Preserve original results, exceptions, materialization, registry/API identity, live borrowed transactions, memory/custom behavior and all guards/caps/deadlines. No global DB sweeping, controller expansion, new helper, cleanup layer or GC change. The separate original3s navigation timeout and aggregate274descriptor growth remain unresolved; these controls do not attribute the entire population. This production delta invalidates prior Console dependency-byte reuse as well as the already-invalid932incoming prefix. Require fresh full affected Console and incoming selections, current expanded joined resource ownership and impacted consumer/import/static/generated controls. ADR required:no newADR. ADR paths: backlog/decisions/126-complete-local-backup-and-recovery.md (fresh-versus-borrowed source-thread retirement); backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/097-boot-budget-ratchets.md. Direct accepted custody correction does not qualify independent Complete product work. Retain all native failed observations; PhaseB remains gated.


### Task3 exact run-log fixture admission

Task3 run-log fixture admission, controller-approved after metadata-owner-covering177passes/1failure and immutableBASE run-log-owner-base26passes/8failures. RunLogWriter.bind reaches guarded source admission before writing its synthetic log; raw_source_selection_changed prevents original availability/privacy/truncation behavior. A collection-only existing bootstrap_profile marker control restores34/34original bodies. Enroll only Tests/Chat/test_console_agent_tool_result_cap.py with that existing module marker, preserving all34body ASTs/assertions, primary/drill/unknown log behavior, privacy/revocation/truncation and source/admission/storage/network guards. No broader fixture or production change. Qualify the complete covering owner selection on actual amended source; retain all failures and marker-only diagnostic control separately. ADR required:no newADR; existing synthetic-profile custody under backlog/decisions/193-native-v2-profile-compatibility-and-admission.md. PhaseB remains gated and original3s timeout/aggregate274FD warning remain separately unresolved.


### Task3 native probe metadata observation

Task3 observe-only probe correction, controller-approved after expanded resources-metadata-final41passes/1bodyfailure/1teardownerror. The immutable Task1 observer reads connection.in_transaction across a live shared registry before real close; a concurrently closed handle raises ProgrammingError inside the observer and prevents native close. Native extracted observer RED reproduces this before delegation. Preserve the original probe/hash and use a separate Task3 copy changing only tuple snapshots and narrow read-only closed-state metadata, following its existing raw-state ProgrammingError pattern. All17assertion ASTs, limits, ownership definitions, real-close ordering and call arguments remain identical; native GREEN proves exactly-once delegation, unchanged return and application-close exception identity. This is instrumentation qualification, not native retirement evidence. Rerun the entire expanded86case joined population on the copy, preserving the failed receipt and original30s/3s timeouts plus274FD warning independently. No production closing-order/cap/deadline/GC or owner-boundary change. ADR required:no newADR; source-free observation correction only.


### Task3 exact Agent-rail fixture contracts

Task3 exact Agent-rail fixture repair, controller-approved after metadata-owner-final205passes/1admissionfailure, immutableBASE11passes/39failures, admitted current47passes/3failures and admittedBASE1pass/2failures for the three distinct cases. Enroll only Tests/UI/test_console_agent_rail.py with existing bootstrap_profile marker. In only test_drilldown_render_path_rejects_record_from_other_conversation and test_drilldown_step_text_grows_with_a_configured_cap_above_eighty add the existing15sibling-shaped local fake subagent_counts(self,conversation_ids):return{} API, with no fields, new helper or production fallback. One BASE cross-conversation fake happens to pass due background timing; current exact WorkerFailed plus identical source and sibling BASE failure remain explicit. In only test_modal_log_predicate_uses_turn_token_without_metadata_reads strengthen the existing same100x0.01 settled predicate to require mounted modal and nonzero Next geometry before its original click; optionally assert delivery. Native observation proves initial0x0region yields Pilot offset0,0/background hit and clickFalse despite true target gate. Preserve every original assertion AST, all remaining body/decorator ASTs, source/storage/network guards and original wait budgets. Record exactly three body exceptions and qualify full actual-source50 plus whole covering selection. The run-log companion module preserves33original test functions producing34cases, correcting earlier34body shorthand. No production or UI behavior change; original3s timeout/274FD finding and observer contamination remain separate. ADR required:no newADR; existing fixture API and mounted-layout synchronization under ADR036, source admission under ADR193.


### Task3 mounted Agent-rail fake API completion

Task3 exact mounted Agent-rail fake API completion: retain metadata-owner-qualified211passes/1failure and full owner42passes/8failures; seven current failures are missing subagent_counts while one stale-generation WorkerCancelled is distinct. Exact admitted immutableBASE eight-fake selection has4passes/4identicalmissingAPIfailures, with background timing making other omissions pass; all eight original body/decorator ASTs equalBASE and install local fake on a mounted real rail whose shared refresh requires the batched API. Controller-approved after exact native/current/BASE and shared-caller proof: add only the existing field-free subagent_counts(self,conversation_ids):return{} method to local _FakeBridge in test_agent_section_lines_render_brackets_literally_not_escaped, test_agent_section_falls_back_to_historical_snapshot_when_live_is_idle, test_agent_section_prefers_live_snapshot_over_historical_when_present, test_drilldown_falls_back_to_overview_after_conversation_switch, test_console_agent_fleet_token_total_sums_live_handles, test_cancel_console_agent_fleet_row_delegates_to_the_bridge, test_drilldown_shows_a_live_childs_steps_before_they_reach_the_db, test_drilldown_still_prefers_the_persisted_steps_once_they_exist. Preserve all17 existing current count methods including custom behavior, every original assertion/decorator and other body ASTs; total authorized body exceptions become11 including prior2API+1modal. No shared fake framework, production fallback, source/admission/network guard or deadline change. Full50 and complete covering required; separate stale-worker/descriptor/timeout findings remain open. ADR required:no newADR; existing fixture API under ADR036/193. Separately approved twelfth body exception: in test_log_probe_settles_stale_or_cancelled_generation capture the exact newly scheduled generation2 worker before yielding and await only that worker, rather than broad host.workers.wait_for_complete. Deterministic private admittedBASE held-after-publication control retains1failure/1pass with canceled generation1 still registered alongside generation2; isolated current2/BASE2 passes are observations only. Preserve intentional old-worker cancellation, actual lifecycle/teardown retirement, original3second barrier/outerdeadline and calls A/A plus publication True and every original assertion. Qualify deterministic native owner-wait GREEN and complete50/whole covering. No production worker policy/suppression or cap change.


### Task3 native borrowed-WAL evidence contract

Task3 native borrowed-WAL evidence correction, controller-approved after native Python3.12.11/SQLite3.49.1 stdlib and actual WorkspaceDB controls. Three sequential worker native closes preserve the same main borrower/transaction and stable fourth same-inode WAL database descriptor; original final owning MainThread close retires all descriptors. Real application control also removes the worker connection/lease and preserves exact main connection/lease. Existing ADR126 fresh-versus-borrowed ownership is binding. Preserve the copied probe exact <=3 and all17assertions/caps; classify only this proven private diagnostic count assumption as defective, never a green batch or runtime cleanup requirement. Run complete joined86 without maxfail and retain exact per-node outcomes, borrower/startup identities, zero owned/unexplained paths, stable deferred native custody and original final-owning-close proof. Any other assertion failure remains unresolved. No assertion removal, limit increase, forcedGC/OSclose, borrower reset, journaling change or cleanup. Repository200session-growth sentinel plus fresh complete incoming and Console must pass; original274aggregate growth/3s timeout remain separately attributed, not explained by this one descriptor. PhaseA may finish DONE_WITH_CONCERNS with the exact retained private diagnosticRED for independent task and broad review, never all-resource-green, publication or tracker closure. ADR required:no newADR; existing backlog/decisions/126-complete-local-backup-and-recovery.md and094/097 govern unchanged ownership/caps.


### Task3 exact beginner-onboarding fixture state

Task3 exact beginner-onboarding fixture seed, controller-approved after paired native current and immutableBASE prior-real-send controls. The actual modal is mounted but quiet with no sync pending, empty transcript, guidance_dismissed False and cached/persisted console.onboarding.first_send_completed True from the preceding successful send. Existing production global completion semantics are correct. In only Tests/UI/test_console_workbench_contract.py::test_console_empty_transcript_exposes_beginner_activation_actions seed first_send_completed=False in app_config.console.onboarding before host construction, then persist only chat_defaults and console.onboarding through existing persist_seeded_config dotted-section support. Preserve every original assertion, wait and decorator; no generic fixture reset, full-console write, production behavior or timing change. Run paired prior-real-send/beginner GREEN and final complete incoming (which covers fullWorkbench); no redundant fullWorkbench run. Retain original incoming277failures, two callback warnings and failed observer separately. ADR required:no newADR; existing readiness/onboarding contract from TASK2154.7, no ownership or activation change.


### Task3 exact Settings fixture contracts

Task3 controller-approved exact stale Settings fixture correction, native current/immutable BASE failures retained: change only the two failure_status_text expected suffix constants from F8 to the canonical F3 Logs binding (leave arbitrary old-F8 redaction samples untouched); read canonical css/features/_settings.tcss with existing design_token_preamble, resolve_variable_definitions and isolate_local_variables, preserving all original literal width30/6fr/2fr, height1fr/min-height0 and inline-width negative assertions; extend the Speech provider source-pin regex/docstring to recognize existing _path alongside input/select/switch while preserving source/table equality and nonempty/extra-row negatives. The real _path reads self.state and helper methods. No production/CSS/table edit, new resolver or guard relaxation. Exact native GREENs precede final covering. ADR required:no newADR; direct existing ADR031 shortcut and ADR150/161 design-token contracts; task23109 search evidence. Other source-admission, rendered-index, new-file, session-rendering and callback findings remain separate and unapproved.


### Task3 final Settings source and search contracts

Task3 controller-approved exact remaining Settings/search repairs after current/immutable BASE and source/caller attribution, with detailed private controller-repair-proposals.md. Add only23 persisted search rows (15 rendered plus8 source-pinned provider path fields) and two documented transient-control exclusions (theme preset target; web backend editor selector), keeping all category floors/completeness/labels/negatives. Index owner is on actual Settings preimport route; normal-format40LOC projection leaves19LOC but actual native unchanged-cap census is required. Enroll exactly225 clean named functions plus six explicitly diagnosed ownership/search/seed functions with existing bootstrap_profile, no whole-module marker. Exact13 selected-path functions use existing private_profile_test+request with14 native children and asyncio on four original sync wrappers; keep the same TOML/bytes and choose Path(config_module.get_cli_config_path()), deriving backups from that path. Separate new-file fixture uses its own selected synthetic child path and unlinks only helper-created config before original inspect/validate/save/no-backup behavior. Preserve all original assertion ASTs except four explicit stale expectation owners: Providers tuple adds enabled/keep_count snapshot fields; Schedules branch verifies existing writable global gate while other readonly negatives stay; two search-copy owners change three expected count literals2to3 for existing My Profile privacy match, same ranking/focus. Two exact pre-host seeds set only background_effects.scope=transcript and huggingface.api_key_env_var=HUGGINGFACE_API_KEY in their respective named tests, preserving save-map/credential assertions. No guard, source rebinding, factory reset, helper, dependency, production policy or cap/deadline change. Native children inherit no external proof plugins; final complete incoming/Console/expanded joined86 remain required. ADR required:no newADR; direct existing TASK23109 search contract, ADR119 snapshot preference ownership, ADR019/032 Settings global briefing gate and ADR193 profile admission. Independent tests/notes/status remain unchanged. Exact function inventories are retained in task3 evidence; all former failed batches stay failed.

Exact bootstrap owners (225 clean +6): test_compact_input_edge_renders_under_real_bundle, test_conversation_settings_return_clean_deep_link_focuses_exact_provider_credential, test_conversation_settings_return_continuation_survives_fresh_settings_screen, test_conversation_settings_return_is_single_flight_and_retries_after_failed_navigation, test_conversation_settings_return_keeps_mounted_credential_out_of_transfer_surfaces, test_conversation_settings_return_preserves_explicit_unselected_model, test_conversation_settings_return_preserves_same_provider_draft_and_discloses_fields, test_conversation_settings_return_save_failure_retains_draft_and_handoff, test_conversation_settings_return_save_shows_typed_continuation, test_conversation_settings_return_stay_settles_exact_handoff, test_conversation_settings_return_without_saving_cancel_allows_retry, test_conversation_settings_return_without_saving_is_single_flight_on_confirm, test_conversation_settings_save_focuses_primary_return_above_compact_fold, test_detail_row_folds_long_config_keys, test_disabled_save_revert_carry_text_annotation, test_discovered_model_save_crash_status_is_plain_language, test_every_category_renders_the_state_banner, test_field_search_enter_focuses_the_field, test_field_search_finds_and_focuses_folder_files_tree_controls, test_field_search_folder_files_tree_width_guides_to_custom_widths_when_disabled, test_field_search_surfaces_category_and_names_the_field, test_filter_clears_after_opening_a_match, test_filter_matches_owned_config_keys, test_filter_placeholder_names_categories, test_filter_rank_tiers_keep_relative_order, test_filter_word_boundary_match_outranks_substring, test_footer_entries_advertise_save_revert_where_draft_model_acts, test_footer_entries_advertise_test_where_it_acts, test_footer_entries_drop_save_revert_where_no_draft_model_exists, test_footer_entries_drop_test_hint_where_no_test_action_exists, test_footer_entries_privacy_keeps_test_and_raw_cli_save_revert, test_inspector_overflow_hint_matches_body_overflow, test_invalid_video_gen_draft_never_invokes_save_worker, test_local_scope_note_is_pinned_outside_scrollable_body, test_model_discovery_crash_status_is_plain_language_without_raw_exception, test_model_thinking_visibility_has_search_guidance_and_device_ownership, test_numeric_labels_carry_units, test_probe_settings_endpoint_counts_models_and_normalizes_path, test_probe_settings_endpoint_maps_transport_failures, test_probe_settings_endpoint_reports_http_status_and_invalid_url, test_provider_navigation_conflict_discard_explicitly_applies_staged_target, test_provider_navigation_conflict_requires_review_discard_or_return, test_save_revert_pair_hidden_on_non_draft_categories, test_search_ambiguous_theme_disambiguates_with_scope, test_search_description_tier_match_still_lands_on_the_field, test_search_enter_focuses_reduce_motion, test_search_finds_reduce_motion_with_scope_text, test_search_landing_expands_enclosing_collapsibles, test_search_landing_on_disabled_field_explains_instead_of_no_op, test_search_next_segment_is_dropped_on_short_terminals, test_search_token_keeps_its_pre_existing_intra_category_landing, test_settings_active_category_uses_explicit_nav_marker, test_settings_advanced_config_blocks_invalid_toml_and_redacts_secret, test_settings_advanced_config_blocks_non_mapping_toml_on_save, test_settings_advanced_config_guided_path_buttons_escape_raw_toml, test_settings_advanced_config_keeps_safety_actions_before_raw_editor, test_settings_advanced_config_shows_raw_editor_and_safety_actions, test_settings_advanced_config_uses_editor_owned_scroll_region, test_settings_appearance_focused_input_keeps_typed_text_visible, test_settings_appearance_library_reader_controls_round_trip_all_destinations, test_settings_appearance_preview_checks_draft_without_theme_or_save, test_settings_appearance_renders_guided_defaults_and_validates, test_settings_appearance_revert_restores_loaded_values, test_settings_appearance_save_signals_live_console_refresh, test_settings_category_navigation_is_grouped_for_scan, test_settings_category_rail_renders_no_hidden_status_rows, test_settings_category_search_escape_clears_filter, test_settings_category_search_normalizes_oversized_control_input, test_settings_category_search_reveals_domain_matches, test_settings_category_search_uses_plain_standard_input_widgets, test_settings_category_selection_updates_detail_and_inspector, test_settings_config_path_delegates_to_shared_accessor, test_settings_config_path_validates_env_override, test_settings_console_background_effects_save_nested_config, test_settings_console_background_fps_rejects_out_of_range_save, test_settings_console_background_workbench_loaded_scope_mounts_as_transcript, test_settings_console_background_workbench_loaded_scope_save_shows_fallback, test_settings_console_background_workbench_loaded_scope_unrelated_save_falls_back, test_settings_console_background_workbench_raw_scope_unrelated_save_includes_fallback, test_settings_console_background_workbench_scope_falls_back_to_transcript, test_settings_console_behavior_clean_staged_feedback_shows_workbench_warning, test_settings_console_behavior_clean_state_does_not_show_staged_feedback, test_settings_console_behavior_default_undo_does_not_show_staged_feedback, test_settings_console_behavior_display_name_revert_restores_loaded_value, test_settings_console_behavior_focus_auto_scrolls_to_field_guide, test_settings_console_behavior_focus_reveals_full_guide_when_purpose_starts_flush_with_bottom_fold, test_settings_console_behavior_owns_background_effect_settings, test_settings_console_behavior_rejects_invalid_tool_result_display_chars, test_settings_console_behavior_rejects_overwide_cjk_display_name_atomically, test_settings_console_behavior_renders_background_effect_controls, test_settings_console_behavior_renders_global_default_controls, test_settings_console_behavior_revert_button_works_with_input_focus, test_settings_console_behavior_revert_discards_draft, test_settings_console_behavior_revert_restores_global_defaults, test_settings_console_behavior_saves_display_name_exactly, test_settings_console_behavior_saves_max_parallel_runs, test_settings_console_behavior_saves_paste_threshold, test_settings_console_behavior_stages_save_and_revert, test_settings_console_behavior_uses_batched_save_adapter, test_settings_console_guided_save_revert_enable_only_when_dirty, test_settings_detail_shows_state_banner_and_structured_rows, test_settings_diagnostics_invalid_config_source_does_not_duplicate_error, test_settings_diagnostics_test_shortcut_runs_validate_and_reload, test_settings_diagnostics_unexpected_config_path_errors_are_not_masked, test_settings_diagnostics_validate_and_reload_config_actions, test_settings_domain_category_contracts_are_explicit_about_mutation_scope, test_settings_domain_category_renders_read_only_owner_contract, test_settings_domain_defaults_group_toggle_expands_and_collapses, test_settings_domain_group_expands_when_restored_to_domain_category, test_settings_enum_select_value_clamps_case_and_unknown_values, test_settings_generation_controls_allow_anthropic_max_thinking_effort, test_settings_generation_controls_allow_openai_none_reasoning_effort, test_settings_inspector_boundary_is_structured_without_duplicate_copy, test_settings_inspector_has_no_write_blocked_contradiction, test_settings_inspector_uses_category_specific_guidance, test_settings_invalid_input_keeps_error_tint_while_focused, test_settings_keeps_dormant_library_width_through_resize_mode_and_generation, test_settings_keyboard_category_focus_survives_selection_recompose, test_settings_library_rag_inspector_uses_shortened_terse_guidance, test_settings_library_rag_renders_guided_defaults_and_validates, test_settings_library_rag_reranker_warning_shown_for_a_warning_triggering_draft, test_settings_library_rag_save_does_not_touch_app_config_and_persists_profile, test_settings_library_rag_sync_clamps_invalid_select_values, test_settings_long_detail_and_inspector_panes_are_scrollable_containers, test_settings_manual_sync_dialog_readable_fallback_when_counts_unloaded, test_settings_manual_sync_run_requires_confirmation_with_pending_counts, test_settings_mixed_conflicts_display_and_resolve_same_adoption_review, test_settings_mount_triggers_at_most_one_post_mount_recompose, test_settings_navigation_context_preselection_does_not_create_provider_draft, test_settings_navigation_context_preserves_existing_provider_draft_values, test_settings_navigation_provider_context_tolerates_missing_provider_values, test_settings_notes_adoption_controls_resolve_and_resume_enrollment, test_settings_on_key_treats_detached_screen_focus_as_absent, test_settings_optional_int_defaults_load_invalid_values_as_blank, test_settings_ownership_record_falls_back_without_crashing, test_settings_paste_toggle_keeps_keyboard_focus_after_refresh, test_settings_privacy_and_diagnostics_label_unsupported_mutations_as_wip, test_settings_privacy_security_recovery_actions_navigate_to_existing_categories, test_settings_privacy_security_renders_guided_redacted_posture, test_settings_privacy_security_test_shortcut_runs_privacy_check, test_settings_privacy_shortcut_passes_stable_config_snapshot_to_worker, test_settings_profile_enum_select_clamps_saved_values, test_settings_provider_api_key_focus_style_has_no_underline, test_settings_provider_blank_select_value_is_not_treated_as_provider, test_settings_provider_category_blocks_empty_manual_provider_save, test_settings_provider_category_does_not_save_unedited_effective_defaults, test_settings_provider_category_lists_console_supported_catalog, test_settings_provider_category_preserves_existing_endpoint_key, test_settings_provider_category_rejects_invalid_credential_env_var, test_settings_provider_category_rejects_out_of_range_model_profile, test_settings_provider_category_renders_catalog_select_with_visible_value, test_settings_provider_category_renders_local_api_key_setup_without_revealing_secret, test_settings_provider_category_saves_and_clears_local_api_key, test_settings_provider_category_saves_credential_env_var, test_settings_provider_category_saves_exact_provider_model_pair, test_settings_provider_category_saves_llamacpp_endpoint, test_settings_provider_category_saves_provider_defaults_without_sampling, test_settings_provider_category_updates_existing_non_normalized_provider_section, test_settings_provider_category_uses_effective_console_source, test_settings_provider_connect_block_precedes_collapsed_generation_defaults, test_settings_provider_custom_value_uses_manual_field_for_unknown_provider, test_settings_provider_endpoint_save_blocks_blank_provider, test_settings_provider_endpoint_uses_url_safe_input_for_url_values, test_settings_provider_endpoint_validation_blocks_bad_url, test_settings_provider_guided_save_revert_enable_only_when_dirty, test_settings_provider_keyless_local_provider_does_not_report_missing_env_var, test_settings_provider_manual_entry_promotes_known_provider_to_catalog_select, test_settings_provider_model_defaults_appear_before_reference_copy, test_settings_provider_model_discovery_controls_render_for_eligible_provider, test_settings_provider_model_discovery_saves_selected_runtime_models, test_settings_provider_model_discovery_shows_ambiguous_provider_recovery, test_settings_provider_model_profile_none_values_render_as_blank_inputs, test_settings_provider_model_switch_does_not_save_unedited_profile, test_settings_provider_model_switch_loads_selected_model_profile, test_settings_provider_navigation_context_focuses_api_key_field, test_settings_provider_navigation_context_uses_one_presentation_identity, test_settings_provider_openai_endpoint_placeholder_uses_provider_context, test_settings_provider_picker_current_alias_preserves_connection_drafts, test_settings_provider_picker_enter_provider_id_focuses_manual_field, test_settings_provider_picker_filter_clear_restores_current_highlight, test_settings_provider_picker_filtered_selection_uses_provider_lifecycle, test_settings_provider_picker_geometry_is_bounded_and_non_overlapping, test_settings_provider_picker_initial_known_provider_enter_is_noop_for_drafts, test_settings_provider_picker_initial_unknown_provider_is_selected_exactly, test_settings_provider_picker_no_match_is_honest_with_manual_action, test_settings_provider_picker_persistence_alias_saves_canonical_provider, test_settings_provider_picker_persistence_alias_uses_catalog_lifecycle, test_settings_provider_picker_rejects_unsupported_manual_id_without_losing_drafts, test_settings_provider_picker_saved_unknown_activation_is_exact_noop, test_settings_provider_picker_search_keeps_draft_endpoint_and_api_key, test_settings_provider_picker_supported_manual_alias_uses_catalog_lifecycle, test_settings_provider_repopulation_is_not_a_user_edit, test_settings_provider_revert_restores_provider_dependent_placeholders, test_settings_provider_route_echoes_do_not_survive_widget_lifecycle, test_settings_provider_save_button_works_with_endpoint_input_focus, test_settings_provider_streaming_and_enums_prevent_invalid_input, test_settings_provider_switch_does_not_save_stale_endpoint, test_settings_provider_switch_resets_staged_model_for_each_provider_transition, test_settings_provider_switch_selects_provider_default_model, test_settings_provider_switch_updates_inspector_readiness, test_settings_provider_test_toast_states_success, test_settings_provider_text_inputs_do_not_trigger_footer_shortcuts, test_settings_restore_state_ignores_malformed_values, test_settings_saves_each_mistral_entry_to_its_distinct_owner, test_settings_screen_resume_refreshes_cached_source_rows, test_settings_screen_resume_skips_refresh_while_manual_sync_run_in_flight, test_settings_state_round_trip_preserves_active_work_without_aliasing, test_settings_tab_focus_and_enter_select_categories, test_settings_user_emptied_context_window_still_refuses_the_save, test_slash_refocus_selects_existing_filter_text, test_splash_screen_category_opens_without_crashing, test_state_banner_is_pinned_outside_detail_scroll, test_state_banner_leads_with_persistence_badge, test_state_banner_text_has_exactly_one_state_segment, test_sync_rows_recompose_mid_navigation_still_focuses_target_field, test_t_hint_uses_each_categorys_real_verb, test_theme_and_splash_appear_in_settings_sidebar, test_theme_category_opens_without_crashing, test_theme_category_settles_without_recompose_storm, test_theme_user_edit_does_not_remount_editor, test_threshold_field_has_focused_guidance, test_video_gen_filter_exposes_category_and_enter_opens_existing_panel, test_video_generation_category_is_final_domain_default_and_count_matches_rail, test_workspaces_banner_names_reversal_paths, test_workspaces_unselected_card_shows_hint_not_blank, test_settings_ownership_records_cover_categories_and_runtime_boundaries, test_settings_domain_categories_are_grouped_and_have_ownership_records, test_settings_category_search_filters_and_enter_opens_first_match, test_settings_category_search_reports_ranked_matches_and_enter_target, test_settings_console_behavior_saves_global_defaults, test_settings_navigation_context_can_preselect_provider_category_target.

Exact private-profile owners (13 + separate new-file): test_settings_advanced_config_backup_load_never_clobbers_unsaved_typing, test_settings_advanced_config_load_backup_reports_decode_failure, test_settings_advanced_config_loads_backup_preview_without_saving, test_settings_advanced_config_saves_atomically_with_backup, test_settings_appearance_defaults_ascii_glyphs_off_in_fresh_config, test_settings_appearance_reads_ascii_glyphs_from_fresh_config, test_settings_diagnostics_combined_helper_skips_reload_when_invalid, test_settings_diagnostics_combined_helper_validates_once, test_settings_diagnostics_results_include_config_source_and_redact_errors, test_settings_diagnostics_strictly_reports_corrupt_toml, test_settings_schedules_gate_disables_retry_after_live_apply_failure, test_settings_schedules_gate_is_painted_and_persists_recovery_action, test_settings_schedules_gate_reports_durable_cache_publish_failure, test_settings_advanced_config_new_file_save_reports_no_backup.


### Task3 actual navigation sync completion and fixture custody

Task3 controller-approved exact navigation fixture refinement after valid native current/immutable BASE held-worker REDs. In only test_hidden_completion_and_background_approval_reconcile_by_session retain original direct sync request then wait while public app.workers has worker.node is reopened and group console-sync, yielding asyncio.sleep(0) under unchanged outer300s. Native WorkerManager membership retains predecessor until task completion and registers exclusive successor before removing predecessor; therefore follow actual manager membership rather than await canceled Worker.wait or suppress cancellation. No private task, new helper, flag mutation, timer or deadline. Retain original pilot.pause, render privacy/session/card/attention assertions and all3/5/8/10 waits. Move only existing fallback approval resolve/join into finally inside live run_test context; retain exact5second join, normal settlement and original exception propagation. Qualification requires held GREEN with actual manager wait observed before release and injected original assertion failure proving thread retirement while app live and no warnings; normal incoming warning gate remains. No production marshal or sync change; native3s pre-intervention timeout and old fullbatch failures remain distinct. ADR required:no newADR; existing ADR094 runtime/task custody and ADR036 app composition, fixture-only completion/retirement correction.


### Task3 same-selected-source fixture cache refresh

Task3 controller-approved exact five-fixture public reload refinement after private-profile15child run10passed5failed: same selected path keeps the intentional startup config cache, so overwriting original TOML must use public load_settings(force_reload=True) before original app/read. Add exactly one config_module.load_settings(force_reload=True) after unchanged TOML write and same-path envassignment in three scheduledgate functions and two freshappearance functions (6cases), no private cache reset, guard bypass, helper/runtime/bytes/assertion/deadline change. Default guarded reload_bootstrap follows force. Retain failed15child batch and run all15again once; exact native child/interpreter/source receipts without inherited proof plugins. ADR required:no newADR; existing ADR193 admission and config loader force-reload contract. Exact owners: test_settings_schedules_gate_is_painted_and_persists_recovery_action, test_settings_schedules_gate_reports_durable_cache_publish_failure, test_settings_schedules_gate_disables_retry_after_live_apply_failure, test_settings_appearance_reads_ascii_glyphs_from_fresh_config, test_settings_appearance_defaults_ascii_glyphs_off_in_fresh_config.


### Task3 existing Console search label alignment

Task3 controller-approved two existing Console Settings search label literals after native exact23 run22passed1failed. Entire rendered settings_screen.py remains immutableBASE-identical; source1270/1284 names Wall-clock (seconds) and Per-tool-call (seconds), while BASE index415-420 and current index contain the obsolete extra word limit. Replace only those two label constants in settings_search_index.py. Missing-row assertion previously short-circuited this unchanged containment guard. Preserve every search assertion, field ID, category floor, original deadline, all import/LOC caps; zero source LOC/import delta. ADR required:no newADR; direct existing settings field-search contract. Retain failed23 receipt, run fresh23 and affected native invoice after final formatting.


### Task3 exact native resource snapshot identity

Task3 controller-approved exact private resource-boundary inode observation for final joined86. Keep the copied task3_resource_probe.py bytes and all17assertions/order/limits unchanged. A separate observation-only companion calls original handles exactly once and returns the identical list object/values, independently recording fd/device/inode only when the path still matches; closed/reused/mismatched fds are explicit observation races, never accepted custody. Per-node phase/caller/line and monotonic snapshot sequence bind the failing <=3 snapshot. Native owned-temp-fd controls prove delegate-once/result identity+values/original exception identity and missing/reused/mismatched receipts; no application close, GC, cleanup, capability substitution, borrowed-owner mutation or new allowance. Final resource runtime RED and separately labeled offline exact-expression audit remain required. ADR required:no newADR; direct evidence of existing ADR126 fresh-versus-borrowed custody. All original failed receipts remain.


### Task3 memory-owner observation classification

Task3 controller-approved private memory-owner observation refinement after resources-metadata-complete83bodypasses/3privateobserverbodyfailures/50private<=3teardownerrors. Real WorkspaceDB and AgentRunsDB memory constructors deliberately omit maintenance participants under participants.py351-410/453-477/501-520. Preserve original Task1 probe and immutable pre-refinement Task3 copy/hash. Change only the two copied close-observation comprehensions to () if self.is_memory_db else tuple(self._maintenance_participant.connections.items()), with explicit memory-owner classification: empty participant rows do not mean zero native memory connections. File-backed missing participants still fail. Preserve all17assertions/order/<=3/caps, exactly-once original close arguments/results/exceptions, no application/test/GC/cleanup/borrower mutation. Native real memory controls must prove borrowed transaction preservation through finite operations and rollback plus original close cache retirement; retain already-closed/borrowed/application-error delegation controls. Then fresh complete joined86 without maxfail, exact inode/borrower/startup custody and offline original-expression audit. Prior83/3/50 remains failed observer evidence; fresh Console/incoming reuse requires unchanged exact application/test bytes. ADR required:no newADR; existing ADR126 and094/097 custody/evidence only.


### Task3 historical design and current qualification status

Task3 bounded documentation audit: distinguish the memory evolution design original inspected owner-table baseline from completed TASK25907.1-.4 first-release plan/runtime history; mark only their historical executable-plan/runtime-start review rows complete with existing roadmap links. Preserve accepted contracts and closed future gates. In roadmap preserve prior TASK25907.23 publication as historical and identify reopened pinned-dev PhaseA review/publication as pending; no current task AC/status closure. Documentation-only, no newADR, implementation or rerun.


### Task3 review fix round1: queue capture test name

Task3 independent review fix round1: rename only Tests/Chat/test_console_turn_library_authority.py::test_queued_configuration_and_policy_capture_only_after_dequeue to test_queued_configuration_captures_at_enqueue_and_policy_after_dequeue so the name reflects existing enqueue configuration capture and dequeue policy capture. Search exact references and preserve historical receipts/names. Preserve all bodies/arguments/decorators/assertions/fixtures/deadlines; only wrap the longer function signature if required by existing formatter. Prove entire module AST equality after substituting exactly this function identifier. Run only the actual renamed native case and scoped lint/format/reference checks. Reuse earlier covering receipts solely under unchanged production bytes and pure name equivalence, never global byte identity. Preserve immutable PhaseA062b report/brief/draft/freeze copies before appending fix evidence. ADR required:no newADR; routine test-name maintenance. No descriptor probe change, publication, criteria/status closure, other reviewer correction or broad rerun. Scoped fix re-review and broad review remain controller gates.


### Task3 final whole-branch finding fix wave

Task3 final whole-branch finding wave, FIX_BASE 0a4fecb511558264a7e65dee0c025fe08b43b4b4, controller rulings83-85: repair only shared PersonalContextProvenanceDetails content-free changed-message/Reload loss across collapse/screen suspend, retaining literal/private metadata clearing and generation/currentness/lease/pending-task fences. Trace all panel/modal/Settings callers; add mounted RED then deterministic hide/resume/Reload and stale-result/invalidation negatives. Add one bounded native production profile_search regression with independently fixed expected identities, equal scores and more than20 candidates; no search algorithm/priority change. ADR required:no newADR. ADR paths: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/097-boot-budget-ratchets.md. Reason: routine repair and regression coverage of the accepted provenance and lexical-search contracts; no new boundary. Read design language before UI change. Park optional private sibling-helper type-hint introspection (no established caller/contract); no eager-import/loader correction. Preserve all prior qualification artifacts, runtime resourceRED43, unattributed timeouts/274growth, xfail/history skip, vendor/legacy noise, all caps/deadlines/closed gates and AC6 open. Amend only plan through Backlog CLI, not criteria/status. Run native affected provenance/Settings/currentness/design/guard/census controls and actual production search boundary, with exact source/import receipts and bounded unchanged-owner reuse; no repeated allConsole/incoming/full suite absent new doubt. Commit scoped final wave and final-fix-report for exactly one controller scoped re-review. No dev integration, publication, activation, descriptor-probe correction or Done claim.


### Task3 PhaseB actual current-dev integration

Task3 actual PhaseB current-dev integration after final whole-branch and sole final-wave scoped approval, controller rulings86-87: BASE52ab8c5990c0fe4edeffbaa91241c25e114081eb, exact incoming dev ef831d9f383f58a806fe54d61a6fd75678ec73c0. Normal non-rewriting merge into existing feature branch; preserve both testing-lesson incidents, regenerate CSS through canonical source builder. Preserve real uncorrected transcript8270/cap8265 native RED, then only the approved five temporary eliminations (three assertion-only turn locals, immediate Header presentation local, intermediate desired_keys list) under frozen plain-string row/ordinary Header call contracts; retain hashing/constructor-lookup evaluation-order bounds and all incoming Live-output copy/comments. No cap/prose/deadline/guard relaxation. Finalize exact integrated-source collection, deduplication and child custody for the seven serial groups in private phaseb-readonly/cap-proposal/native-selection-proposal.json before execution; all22 incoming production/18 test modules plus stdio fixture, changed Console/approval/output/schema/close/retirement owners and memory-tool/Canvas live-wire/Next Send/barrier negatives require actual current-source evidence. Native Python3.12.11 synthetic private profiles, unchanged probes and exact joined86 no-maxfail;43 historical private count errors remain runtimeRED, any different failure separately diagnosed. Fresh strict/source/derived/diagnostic invoice required; canonical diagnostic writer only after source/sink/argument proof, no scanner/privacy exemption or speculative diet. Reuse52ab provenance only with exact unaffected owner/caller/import proof; prior Console/resource dependency reuse invalidated by incoming runtime changes. ADR required:no newADR. ADR paths: backlog/decisions/205-console-partial-tool-output.md; backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/126-complete-local-backup-and-recovery.md; backlog/decisions/197-console-hook-configuration-review.md; backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/150-design-token-system-and-design-language.md; accepted Personal Context authority/privacy design. Reason: integrate accepted owners and a routine temporary-elimination cap repair without new architecture. Preserve every failed historical receipt, existing noise/xfail/historyskip/fragile margins and closed future release gates; AC6 MUST stay open and InProgress. No production profile/provider/keyring, newdependency/framework, V2/Forget/schema-AAD/disclosure/source-opening/server activation, full repository sweep, dev-branch merge, forcepush or housekeeping. Write phase-b-integration-report and new local full-memory-benefit PR draft while preserving historical draft, commit clean integrated source and freeze all source/results/limits, then STOP before push/PR edit/status closure for one controller scoped integration gate. Any additional source conflict/deficit needs a named diagnosis and controller ruling before changes.


### Task3 PhaseB exact tool-fixture admission repair

Task3 PhaseB Ruling88 fixture-source admission amendment: exact native integrated Group1 retained589pass/285callfail/all874setup+teardownpass, immutable52ab three-owner25pass/324raw_source_selection_changed, and unchanged-source existing-bootstrap collection control349pass establish pre-behavior fixture admission omissions. Enroll ONLY the241 named functions/285cases in private phase-b/admission-marker-proposal.json SHA256 0aa3df97cbec7bf10849a106ef73eb5327fa6fe428eaf15522ecd878d148ad68: Tests/Agents/test_local_tool_provider.py167, Tests/Chat/test_anthropic_native_tools.py35, Tests/Chat/test_cohere_native_tools.py39. Add only @pytest.mark.bootstrap_profile and sole pytest import in Anthropic. Preserve every original function body/signature/decorator/assertion, input/source/storage/network guard, fixture data and deadline; no module marker, runtime/cache/profile helper or source selection change. The39 extra BASE failures belong to five functions already enrolled upstream (one renamed), not ordering drift; no edits there. Whole-file AST equivalence after removing only approved markers/import and fresh native349 actual-source covering required. Production bytes fixed; prior failed874/BASE/control receipts remain historical, and current cap/derived group286pass/2skip/3headroomwarnings may be reused only by exact unchanged production/guard/dependency proof. Separate MCP loop warning is not resolved or authorized for cleanup. ADR required:no newADR; routine fixture enrollment under existing ADR126 selected-source and ADR094/097/197 custody/guard contracts. AC6 remains open/In Progress; no publication or status closure.


### Task3 PhaseB explicit MCP fixture loop ownership

Task3 PhaseB Rulings89-90: retain source-free weakref/original-factory observation as bounded diagnosis only, with possible observer timing perturbation and no causal closure of original InvalidFD-1 unraisable. Exact mcp-loop-fixture-proposal.json SHA256 0a7a2d84c7b18ee622c3a37a700f8fc572001e2942d5aee3c002045825d0dab4 names24 test functions/25 direct dummy main_loop allocations in Tests/Agents/test_mcp_tool_provider.py. Native current89 and immutable52ab88 pass while25 direct loops remain open after their body and24 remain alive/unclosed after existing fixture teardown. Replace only those25 expressions with existing function-scoped event_loop fixture and add that argument to the24 functions. No fixture/helper/global wrapper or production loop/GC/deadline/assertion change; all five explicit running/closed-loop owners remain exact. The two providers in test_compose_catalog_rejects_same_id_with_changed_definition now borrow one fixture loop; neither submits to it and both real catalog compositions still use asyncio.run. Preserve original catalog/definition assertions. Existing fixture shutdown may run queued callbacks and shutdown_asyncgens; all dummy loops have no submitted work, and native after-fixture closed-state evidence must verify actual owner retirement. Whole-file normalized AST proof, fresh complete89 MCP cases, and later fresh full874 final-source Group1 required. Original warning remains unattributed even if later runs pass. ADR required:no newADR; routine test-owned resource correction under existing ADR094/126/197, no production custody change. AC6 stays open/In Progress; no publication/status closure.


### Task3 PhaseB final UI admission and exact static contract corrections

Task3 PhaseB Rulings91-92 after no live native process: exact UI proposal SHA256 6a4b6eb67c80efcbe093e496f74ed04259b1b0926badefa3ba7fd015c82c6806 enrolls only81 named functions/94cases with existing @pytest.mark.bootstrap_profile: Tests/UI/test_console_assistant_turn.py4; test_console_button_routing.py20; test_console_transcript_window_reconcile.py12; test_console_transcript_pruning.py9; test_console_transcript_two_sided_window.py14; test_console_transcript_tail_follow.py5; test_console_thinking_disclosures.py15; test_console_turn_file_card_factory.py2. Current92pass/94source-refusal and immutable52ab94same failures precede original behavior; exact81-function collection-only restoration186pass/zero errors/warnings685.33s proves existing admitted fixture path. Preserve every original body/signature/decorator/assertion/import and all source/privacy/network/keyring/deadline guards, no module-wide marker or source rebinding. Entire-file normalized AST equality and fresh full186 final-source cases required; previous control stays predecessor evidence. Exact static proposal SHA256 2264080709709c9fd0f665b08a1ce210b4f495b5ee4e4a161719d0ba0b2cd8fa permits only local import sorting in MCP/client.py492, same bound logger.error(template,stage,error_type,origin) to logger.log("ERROR",template,stage,error_type,origin) in UI/Console_Modules/session.py2963, and terminal rerender except pass to bare return at2982. Candidate SHAs 515da21b290139983572562d1379a447a50ba1367c58907e08e35a3b37ec92c6 and 8d4e8e372b7ce5a284de8aa2e39b7d8d05eae40005c3cb37bc64bf7e4a63843f are frozen. Actual installed Loguru delegates both methods identically to _log with same options/message/args/kwargs and caller depth; preserve fixed text, classification arguments/evaluation order/bound context, no exception content/body/keyword capture/raw logger substitution. Import ordering is bounded to actual inert initialization, not arbitrary monkeypatch equivalence. Terminal return preserves implicit None and all existing comments. Normalized whole-owner AST proof, identical diagnostic count/sink topology and exact ERROR/template/argument normalization precede canonical inventory writer for only demonstrated method/digest change; no scanner/privacy/review-ledger exemption, unrelated lint fix, suppression, prose stripping, new helper/dependency or cap raise. Fresh native8 close/private/once-twice-rerender controls within full186, MCP result/progress controls within full874, all already-required final1576 Console/874 core/joined86 and affected cap/strict/diagnostic invoices on final bytes; prior Group7 becomes predecessor where changed sources matter. ADR required:no newADR; mechanical existing-contract/fixture correction under ADR205/094/126/197/097/150 and accepted Personal Context privacy. AC6 remains open/In Progress; retain private43runtimeRED, original sporadic InvalidFD warning and older timeouts/growth without causal closure. Stop before publication/PR edit/status closure for scoped integration gate.


### Task3 PhaseB exact approval fixture contract corrections

Task3 PhaseB Rulings93-98: after all live native diagnostics stop, compose only exact whitelisted approved fixture changes, preserving each standalone candidate hash and original source snapshots outside repository. Ruling93: five MCPToolProvider construction formatting collapses in Tests/Agents/test_mcp_tool_provider.py, entire AST identical; fresh five cases and scoped format. Completed874/89 are predecessor-byte evidence reused only with whole-AST/production identity, source-line/debug metadata changed. Ruling94: Tests/UI/test_console_mcp_approval.py existing CSS helper reuses canonical exact grouped/minified selector and token resolution helpers, preserving all four original CSS test bodies and27/auto/1fr/6/3; two typed ToolReviewDecision assertions preserve verdict/refusal while strengthening approved/denied/no-answer provenance; ReadGap subclasses existing ApprovalDecisions with unchanged real lock/revocation/get interception/3s assertions. Ruling95: only owning-run human-wait test existing5s/0.01 predicate observes received AND human_input_wait_active; all original assertions/joins unchanged, eight held current/52 controls prove registration-before-publication and mount-before-wait with original worker retirement. Ruling96: exact wake body passes original rig local_chat_conversation_service identity to its replacement thread app before ConsoleRuntime creation, plus only that existing bootstrap marker; real guarded storage service and deadlines unchanged, current/52 source-exceptionRED and candidateGREEN retained. Ruling97: six headless readiness bodies strengthen only existing seven _settle predicates to actual answerable map/projection/claimable payload/attention/expected notices, preserving64 subsequent original behavior/privacy assertions and3/default5s limits; existing bootstrap enrollment only these six functions. Original marker-only8fail remains distinct from seven candidateGREEN cases on both sources. Ruling98: only existing two-caller headless _build_console_app gateway instance binds already-loaded _ReadyResolutionGateway.cached_context_window pure fallback, same method identity/no self fields/network, preserving shared fake and all original test bodies; only two direct functions receive existing bootstrap. Exact whitelisted composed source/AST/import proofs required; no additional MCP/headless markers or timer/layout repair implied. Fresh four CSS/two typed/ReadGap, human-wait/headless covering controls and complete affected Console qualification after final bytes; prior full1542pass34fail remains non-green. Preserve original private43 runtimeRED, oldtime/growth/InvalidFD and baseline manual-unread warning without causal closure. ADR required:no newADR; routine existing fixture/API/custody contracts under ADR067/094/195/197/097/150. AC6 stays open/In Progress; no cap/deadline/assert waiver, production source, publication or status closure.


### Task3 PhaseB approval readiness and incoming disclosure geometry

Task3 PhaseB Rulings99-101 after completed Group4 and no live native process. Ruling99 enrolls exactly17 named functions/18cases with existing bootstrap_profile only, current source/member/body/decorator hashes verified against original refusal and exact current/immutable52 marker controls. Names: Tests/UI/test_console_headless_approval.py: test_a_headless_round_announces_through_the_app_not_the_screen, test_a_round_armed_with_a_view_attached_does_not_double_announce, test_attaching_a_view_mounts_a_round_armed_while_detached, test_two_headless_rounds_each_mount_in_turn, test_unified_router_never_invokes_legacy_type_setters; Tests/UI/test_console_mcp_approval.py: test_a_route_with_nothing_pending_can_decline_to_warn, test_alt_a_focuses_the_pending_approval_decision_select, test_alt_a_notifies_when_nothing_is_pending, test_alt_a_reaches_the_card_at_80_columns_with_inspector_closed, test_approved_definitive_tool_stays_mounted_until_real_terminal, test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css, test_descriptor_effects_reach_the_mounted_production_approval_card, test_finishing_card_is_not_counted_and_keyboard_focuses_the_card, test_local_provider_refuses_write_after_revoke_then_arm, test_local_same_name_finishing_rows_complete_by_call_id_out_of_order, test_single_row_fast_buttons_have_nonzero_geometry_and_do_not_overlap_under_bundled_css, test_the_approval_route_reaches_a_pending_skill_install_card. Preserve original bodies, assertions, imports, real source/authority/privacy guards; no module-wide enrollment. Ruling100 changes only _wait_for_production_console_ready in Tests/UI/test_console_mcp_approval.py: retain existing lookup, await its original shielded timer only when retained, then always retain existing call_next barrier and all seven original readiness assertions with unchanged10s waits. Remove obsolete assert timer must still be retained. Comment explicitly says no retained timer is not proof of completion. Native exact callback observation showed same reconciled screen and four successful real callbacks before old assertion; failed profiler/timeout receipts remain diagnostic failures. Ruling101 repairs the proven incoming TASK18920 denial disclosure margin in existing source tldw_chatbook/css/components/_agentic_terminal.tcss only: .deny-reason selector becomes ChatApprovalCard Collapsible.deny-reason (+28bytes); unchanged tokens/declarations and existing collapsed/expanded title rules, no global rules/values. Both actual constructors remain covered. Two original mounted geometry fixtures deliberately account for the accepted extra one-line disclosure: core-content bounds remain batch8/single7, independently require collapsed disclosure height1 and all margins0; effective TOTAL bounds become batch9/single8. This is NOT original total-height assertion equivalence and must not be described as all old caps unchanged. Preserve all original width/nonoverlap/visibility/action/privacy/deadline assertions. Before CSS edit run actual two revised geometry tests RED on original CSS, then canonical CSS build and actual GREEN geometry/denial/decision/currentness covering controls. Fresh full affected MCP/headless owners, complete Console/UI/joined86, style/token/CSS608090/caps/import invoices remain required; no source/module/method/FD/import/deadline ratchet changes, no paydown/blank stripping/framework/dependency. Previous native CSS607130/608090 headroom960 is predecessor evidence, +28 is projected until actual writer/invoice. Group4 native74/58children and Group6 native4/8children qualify their exact predecessor bytes; any reuse across this narrowly matched denial stylesheet delta requires precise unchanged Python/test/helper and unaffected-owner proof, never global source identity. ADR required:no newADR; direct accepted TASK18920/ADR067/094/195/097/150 contract corrections. AC6 remains hard/open/In Progress; retain private43 runtimeRED, old timeouts/growth/InvalidFD and all new failed diagnostic/candidate/geometry receipts. No publication, PR edit or status closure.


### Task3 PhaseB exact revoked-child fixture retirement

Task3 PhaseB fixed-pin and exact revoked-child fixture amendment, controller rulings102-104 and final source authorization: finish native qualification at exact dev ef831d9f383f58a806fe54d61a6fd75678ec73c0 despite later observed dev2ee6ea78345eedd9ee2bc6509bb66bbff24dff58; no latest-dev-qualified/merge-ready/Done claim or moving-base remerge. Preserve full Console1575pass/1fail and interpreter-shutdown interruption, prematurely started/interrupted UI run and actual serial186GREEN separately; original full-run worker remains unattributed. Sixteen native current/immutable52 synthetic controls prove only fixture readiness/custody and typed metadata. Apply exactly candidatea944e2c976c1fc6369e1c8fc2d31264ac3af2ebf3addeb13d2e8f6f719c181ca to Tests/UI/test_console_mcp_approval.py::test_a_revoked_childs_card_can_no_longer_execute_its_tool: fixed minimum0.2sleep becomes existing sibling mounted-readiness3s/0.01 polling, explicitly changed timing semantics. Retain production approval30s and move the original single join3s into finally with the existing idempotent per-run revoke/future-arm fence, covering precondition errors and pre-registration late arms. Preserve five original substantive assertions, arguments/decorators, stale click, real gate/provider resultfalse and no-file checks. Use canonical normalize_tool_review and ToolReviewValue local imports/annotation to reject typed proceed and assert canceled/unanswered approval_decision is None; no authored denial manufactured. All136other function ASTs and all production/helper/CSS bytes unchanged; no further source/cleanup/cap/deadline edits. Re-freeze3130production/94selectedtests/4843supporting paths; run original actual one-case and fresh complete1576Console no-maxfail through actual natural exit. Reuse Groups1/3/4/5/6/7 only with exact unaffected Python/CSS/selected-test/helper/caller proofs, never whole-final-byte relabel. Current joined86bodypass/48private count teardownRED differs from historical43; all remain runtime RED, offline later expression evaluations are not successful runtime teardown assertions. Preserve old timeouts/274growth/InvalidFD and all failed/observer receipts. ADR required:no newADR; routine exact fixture correction under existing ADR067/094/195/197/097/126, accepted approval authority/privacy and unchanged typed decision contract. AC6 remains hard/open and task In Progress; no publication/PR edit/status closure or future activation. Final report/normal merge commit/new local PR draft stop for scoped integration gate.


### Task 3 CI follow-up: existing consumer census and test-owned coroutine

CI follow-up (2026-10-01, Ruling 110): published ad5a92b3e7e7d8c0483f355c040616a5cc15440e reproduces the exact resume-consumer census failure and completed-route unawaited coroutine warning from run 36872650671. Add only the existing llama.cpp consumer literal to EXPECTED_RESUME_HANDOFF_CONSUMERS and close the fresh coroutine after CancelHarness.run_worker records its worker call. Preserve every original test body/assertion, hedge, production owner and deadline. Verify both changed modules (8 cases) plus existing navigation success/error/cancellation order controls (3 cases), surfacing warnings as errors; check scoped lint/format and exact production identity. Keep all prior qualification artifacts immutable, current 48 resource teardown RED and historical concerns explicit, AC2/3/6 open and task In Progress. Commit for one scoped gate before publication. GitHub DIRTY against 2ee6 differs from the controller's clean pinned merge-tree; no latest-dev merge is authorized by that mismatch. ADR required: no new ADR; routine test census/coroutine-fixture maintenance implementing TASK-31808, backlog/decisions/114-llamacpp-lab-console-connection-authority.md and backlog/decisions/020-automatic-model-catalog-refresh.md without changing runtime contracts.

CI static completion (Ruling 111): in the same cancel-route test owner, remove its unused SimpleNamespace import and empty import-group separator, and apply only Ruff's two required continuation/wrapper call collapse hunks. Preserve every function/assertion/decorator AST and existing behavior; no production or additional cleanup. Recheck complete changed-owner lint/format, production identity and all 11 native cases on exact final bytes with warnings as errors. ADR required: no new ADR; mechanical unused-import/format maintenance within the existing CI repair. All criteria/status and retained qualification limits remain unchanged.
