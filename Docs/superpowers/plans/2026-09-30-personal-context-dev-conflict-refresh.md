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
