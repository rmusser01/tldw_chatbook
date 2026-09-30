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

- [ ] Step 1: Verify the branch, root, clean baseline and existing linked-worktree identity. Record both parent SHAs and the conflict preview; the native baseline receipts apply only to e3527ac883.
- [ ] Step 2: Start the reversible local merge:

```bash
git -c gc.auto=0 merge --no-commit --no-ff dfee4bf4c66ec2656a6c4ea2667edb63cc38a445
git diff --name-only --diff-filter=U
```

Expected unmerged paths: build_css.py and lessons-testing-evidence.md. Unexpected conflicts require source inspection and a plan amendment before resolution.

- [ ] Step 3: Retain the existing two source modules and pinned tokens, with this combined prefix tuple:

```python
prefixes={
    "settings": (
        "settings", "personal-context", "console-hooks", "hook-review", "-wide-viewport"
    )
},
```

Preserve upstream's Hooks ownership comment and the existing memory/RecoveryPassphraseDialog ownership comment. For every lessons conflict concatenate the complete feature body and the complete incoming body, separated by one blank line; remove only Git conflict markers.

- [ ] Step 4: Inspect the automatically merged runtime/controller/screen changes against both parents. Confirm the new Hooks admission does not bypass the existing preparation/maintenance, fork publication and private source ownership guards. Compare all pre-refresh ratchet constants against the merged versions and the incoming ADR-097 exception provenance; no local relaxation.
- [ ] Step 5: Regenerate using the existing native builder:

```bash
python tldw_chatbook/css/build_css.py
```

Then run the targeted CSS ownership and architecture families, retaining XML/import/source receipts. Use the existing private-profile harness and native interpreter; failed checks enter root-cause diagnosis and a scoped plan amendment before code repair. Do not relabel a single passing rerun as resolution of an unknown intermittent failure.
- [ ] Step 6: Check conflict markers, exact lesson-body preservation and whitespace. Commit the conflict-free integration only after source/guard evidence is recorded; report any still-open native verification accurately in the task report.

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

- [ ] Trace real dependency callers before source repair. Shared Chunking
  eagerness starts in `Chunking/__init__.py` legacy exports: Library rechunk
  `library_rechunk_service` imports `Chunk_Lib`, RAG `chunking_service` imports
  its compatibility exceptions/functions, and `chunking_lab_screen` imports
  package submodules `lab_models`/`lab_preflight`. The current pass bills 44
  Chunking modules / 17,604 LOC. A Library-only local import cannot repay the
  pass because the later Chunking Lab route still executes the package init.
  Evals currently bills 27 engine modules / 7,062 LOC and 14 UI modules /
  9,329 LOC; determine the existing compose/use seams before deferring them.
- [ ] Follow existing project PEP 562 lazy-export patterns only where they
  preserve `__all__`, `dir`, public object identity, direct/from/star imports
  and real use behavior. Use existing Evals compose/use sites; retain route
  order, screen identity, state ownership, source/monkeypatch seams and limits.
  No new framework, screen hierarchy, prewarm trick or timing waiver. If a
  public contract must change, escalate the concrete design before editing.
- [ ] Add focused red import/use controls, make only diagnosed existing-owner
  deferrals, and run affected behavior/import/static guards under native
  synthetic profiles. Tiny incoming Settings Hooks laziness remains distinct
  from inherited debt and must preserve its event/query/class identity seams.
- [ ] Prove the exact candidate with the strict native census and source/import
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

- [ ] Step 1: Select exact test modules from the automatically merged owner diff, including Tests/Chat/test_console_hook_admission.py, test_console_local_review_hook.py, test_console_chat_controller.py, test_console_runtime_lifetime.py, test_console_fork_mutation_fences.py, test_console_fork_transition_census.py and trace preparation/recovery controls. Add the existing memory UI/context/import groups and Tests/test_private_profile_coverage.py; record selection/exclusions before execution.
- [ ] Step 2: Run the selection natively with separate receipts for exact private children, current import paths, source hashes and descriptor categories. Run ./scripts/preflight.sh with the existing pinned public Mermaid inputs. Check changed Python parsing/scoped lint, full lint/format for newly authored files, task readability/IDs, local document links and whitespace.
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
