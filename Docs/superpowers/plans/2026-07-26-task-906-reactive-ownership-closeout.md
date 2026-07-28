# TldwCli Reactive Ownership Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce the exact final `TldwCli` reactive contract, exercise every changed registered destination in the full production app, and prove the same ownership/resource contract from a clean installed wheel.

**Architecture:** `TldwCli` retains only application-lifecycle reactives `current_tab` and `splash_screen_active`; destination/session state lives in the owners established by TASK-647–652 and TASK-904–905. One source AST sentinel and one installed-package AST sentinel derive the class-body descriptors exactly. Installed execution occurs outside the checkout with private state roots and a normally constructed production app.

**Tech Stack:** Python 3.11+, Textual production app, Python AST, pytest/pytest-asyncio, build/wheel/sdist, pip target installs, Ruff.

**Backlog:** [TASK-906](../../../backlog/tasks/task-906%20-%20Close-TldwCli-reactive-ownership-with-installed-distribution-sentinels.md)

**Specification:** [TldwCli Reactive State Decomposition Design](../specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md)

**Depends on:** TASK-647, TASK-648, TASK-649, TASK-650, TASK-651, TASK-652, TASK-904, TASK-905

**ADR required:** yes

**ADR path:** `backlog/decisions/032-immutable-installed-distribution-assets.md`; `backlog/decisions/033-application-session-state-ownership.md`

**Reason:** ADR-033 defines the final application ownership boundary and ADR-032 requires the installed artifact—not the source checkout—to be the final release gate.

---

## Execution and Test Boundary

This task runs only:

- normal production `TldwCli` tests under `Tests/ProductionApp/`;
- direct app-independent state/resolver/request-mapping tests outside
  `Tests/UI/`;
- static AST tests;
- the installed-distribution test.

Do not run, cite, or include `Tests/UI` collections, because its conftest
imports surrogate app/widget harnesses. Do not claim a raw repository-wide
`pytest` result. The authorized suite is explicitly listed below.

## File Structure

- Modify `tldw_chatbook/app.py`: delete the already-no-op
  `watch_current_tab()` and any final removed-name residue.
- Modify `Tests/test_application_state_ownership.py`: exact 61-descriptor disposition and full production-source dynamic access guard.
- Create `Tests/ProductionApp/test_reactive_ownership_maturity.py`: full-app route and owner maturity sentinel.
- Modify `Tests/ProductionApp/conftest.py`: only if needed to make private environment isolation complete; never add an alternate app fixture.
- Modify `Tests/Packaging/test_installed_distribution.py`: installed AST, source-exclusion, and production-app maturity probes.
- Modify the approved spec and TASK-647–652/TASK-904–906 files only during
  final reconciliation.

## Task 1: Start TASK-906 and Freeze the Exact Final Contract

- [x] Move TASK-906 In Progress and add its task-local plan:

```bash
backlog task edit 906 -s "In Progress"
backlog task edit 906 --plan $'ADR required: yes\nADR path: backlog/decisions/032-immutable-installed-distribution-assets.md; backlog/decisions/033-application-session-state-ownership.md\nReason: ADR-033 defines the final root owners and ADR-032 requires clean installed-artifact proof.\n\n1. Enforce the exact TldwCli reactive set.\n2. Run every affected registered route in the production app.\n3. Extend installed-wheel ownership and maturity probes.\n4. Run the authorized integrated gate and reconcile TASK-647–652 and TASK-904–906.'
```

- [x] Add one AST helper that finds class-body `reactive(...)` assignments on
  `TldwCli`, including annotated assignments, and require exactly:

```python
{"current_tab", "splash_screen_active"}
```

- [x] Define `RETIRED_TLDW_REACTIVES` from the other 59 reviewed names in the
  specification and scan every production Python file for:
  class descriptors, assignments/deletes, root `app.<name>` access, constant
  `getattr`/`setattr`/`delattr`, string-key access, and handler
  `reactive_attr` values.
- [x] Scope the guard to `TldwCli`/application-root access so legitimate
  destination fields such as `MediaWindow.media_active_view` remain allowed.
- [x] Delete the already-no-op `watch_current_tab()`. Navigation remains the
  only writer of canonical `current_tab`; do not recreate its retired
  view-toggling body.
- [x] Run:

```bash
pytest Tests/test_application_state_ownership.py -q
```

Expected: PASS only when the exact two-descriptor contract and all 59 removals hold.

## Task 2: Exercise Every Changed Registered Destination

- [x] In `test_reactive_ownership_maturity.py`, construct a normal `TldwCli`
  and navigate, in fresh-screen mode, through:
  `llm`, `chat`, `personas`, `library`, `media`, `search`, `ingest`,
  `mcp`, `evals`, and `settings`.
- [x] Assert each route resolves to its registered production screen and the
  intended owner is mounted. Exercise one safe local state/action on each
  screen; then navigate away/back to detect removed-name access during save,
  restore, resume, unmount, and fresh construction.
- [x] Assert the app instance has no retired reactive/companion attributes and
  that snapshots contain only allowlisted primitives, not prompt bodies,
  records, widgets, workers, services, or removed names.
- [x] Add an AST-based source test that scans every file in
  `Tests/ProductionApp/` and rejects class bases ending in `App`/`Screen`,
  imports or calls of `SimpleNamespace`/`MagicMock`, calls to
  `object.__new__(TldwCli)`, unbound `TldwCli` method calls, imports from the
  two legacy test-harness modules, and fixtures returning an app substitute.
  Do not use raw substring rejection that would fail on this guard's own
  pattern declarations.
- [x] Run:

```bash
pytest Tests/ProductionApp Tests/test_application_state_ownership.py -q
```

Expected: PASS; record exact counts and duration.

## Task 3: Extend the Installed-Distribution Probe

- [x] In `Tests/Packaging/test_installed_distribution.py`, extend the existing
  copied-source build fixture and installed child probe; do not replace it
  with a source-checkout smoke.
- [x] Before the child starts, write a minimal private config with splash and
  first-run UI disabled under the child `TLDW_CONFIG_PATH`.
- [x] Add both `CHECKOUT_ROOT` (the real worktree) and
  `BUILD_SOURCE_ROOT` (the copied temporary build input) to the child
  environment. Resolve both strictly before launch. Assert:
  - the imported package root is under the pip `--target` directory;
  - no `sys.path` entry resolves inside either excluded source root;
  - after all probe imports and the production-app run, every loaded
    `tldw_chatbook` or `tldw_chatbook.*` module with `__file__` or package
    `__path__` resolves under the installed target and never under either
    excluded source root;
  - installed `app.py` has exactly the two reviewed class-body reactives;
  - no installed production source contains an application-root access to a
    retired name;
  - packaged CSS/config/eval/templates/licenses and console entry points still
    satisfy the existing resource contract.
- [x] Use the already-created real `TldwCli` in the child probe with
  `app.run_test()`, wait for the registered Home screen, navigate to one
  affected destination such as Chat or Models, and exit cleanly. Do not define
  an installed test app or screen.
- [x] Preserve the before/after hash assertion proving the installed target is
  immutable.
- [x] Run focused source and production-app sentinels, then commit the
  TASK-906 source/test candidate before any release build:

```bash
pytest Tests/ProductionApp/test_reactive_ownership_maturity.py Tests/test_application_state_ownership.py -q
git add tldw_chatbook/app.py Tests/ProductionApp/test_reactive_ownership_maturity.py Tests/ProductionApp/conftest.py Tests/test_application_state_ownership.py Tests/Packaging/test_installed_distribution.py
git commit -m "test(state): enforce installed reactive ownership (task-906)"
```

- [x] Confirm the committed source/test scope is clean, then run:

```bash
git diff --exit-code -- tldw_chatbook Tests Packaging pyproject.toml MANIFEST.in
git diff --cached --exit-code -- tldw_chatbook Tests Packaging pyproject.toml MANIFEST.in
pytest Tests/Packaging/test_installed_distribution.py -q
```

Expected: PASS after building one sdist and one wheel, installing with
`--no-deps` outside the checkout, and running the installed probes.
The child must fail if the active editable-install finder supplies even one
loaded package module from the checkout or copied build source.

## Task 4: Run Static and Authorized Integrated Gates

- [x] Confirm all implementation/test changes are committed before the release
  build while ignoring the unrelated `.superpowers/sdd` files:

```bash
git diff --exit-code -- tldw_chatbook Tests Packaging pyproject.toml MANIFEST.in
git diff --cached --exit-code -- tldw_chatbook Tests Packaging pyproject.toml MANIFEST.in
```

- [x] Run compile and formatting/lint checks over the final changed scope:

```bash
python -m compileall -q tldw_chatbook Tests/ProductionApp Tests/State Tests/Provider Tests/Library/test_server_ingest_request.py Tests/test_application_state_ownership.py Tests/Packaging/test_installed_distribution.py
python -m ruff check tldw_chatbook/app.py tldw_chatbook/UI/LLM_Management_Window.py tldw_chatbook/UI/MediaWindow_v2.py tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Screens/provider_model_resolution.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/media_screen.py tldw_chatbook/UI/Screens/search_screen.py tldw_chatbook/UI/Screens/mcp_screen.py tldw_chatbook/UI/Screens/evals_screen.py tldw_chatbook/Utils/log_widget_manager.py tldw_chatbook/Event_Handlers Tests/ProductionApp Tests/State Tests/Provider Tests/Library/test_server_ingest_request.py Tests/test_application_state_ownership.py Tests/Packaging/test_installed_distribution.py
python -m ruff check --ignore F841 tldw_chatbook/UI/Screens/settings_screen.py
python -c 'import json, subprocess, sys; p = subprocess.run([sys.executable, "-m", "ruff", "check", "--select", "F841", "--output-format", "json", "tldw_chatbook/UI/Screens/settings_screen.py"], capture_output=True, text=True); findings = json.loads(p.stdout); assert findings == [], findings'
python -m ruff format --check tldw_chatbook/UI/LLM_Management_Window.py tldw_chatbook/UI/MediaWindow_v2.py tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Screens/media_screen.py tldw_chatbook/UI/Screens/search_screen.py tldw_chatbook/UI/Screens/mcp_screen.py tldw_chatbook/Utils/log_widget_manager.py tldw_chatbook/Event_Handlers/LLM_Management_Events tldw_chatbook/Event_Handlers/worker_events.py tldw_chatbook/Event_Handlers/media_events.py tldw_chatbook/Event_Handlers/collections_tag_events.py tldw_chatbook/Event_Handlers/worker_handlers/misc_worker_handler.py tldw_chatbook/Event_Handlers/ingest_events.py Tests/ProductionApp Tests/State Tests/Provider Tests/Library/test_server_ingest_request.py Tests/test_application_state_ownership.py
git diff --check
```

The latest-`dev` reconciliation removed
`Event_Handlers/sidebar_events.py` and
`worker_handlers/chat_worker_handler.py`, so the format gate omits those dead
paths rather than restoring them. The pre-task format exceptions are
`app.py`, `chat_screen.py`, `provider_model_resolution.py`,
`settings_screen.py`, `personas_screen.py`, `library_screen.py`,
`evals_screen.py`, `Chat_Events/chat_events.py`, `conv_char_events.py`, and
`Tests/Packaging/test_installed_distribution.py`; do not mass-format them.
The JSON assertion records the stronger latest-`dev` baseline:
`settings_screen.py` has zero F841 findings.

- [x] Run the authorized integrated suite:

```bash
pytest Tests/ProductionApp Tests/State/test_pending_handoff_store.py Tests/Provider/test_provider_model_resolution.py Tests/Library/test_server_ingest_request.py Tests/test_application_state_ownership.py Tests/Packaging/test_installed_distribution.py -q
```

Expected: PASS. Record exact pass/skip/warning counts and duration. This is the
only integrated claim for the tranche.

## Task 5: Reconcile TASK-647–652 and TASK-904–906 and Commit Closeout Documentation

- [x] Re-read TASK-647–652 and TASK-904–906 with
  `backlog task <id> --plain`.
  Confirm each earlier task is Done, each acceptance criterion is checked, its
  Implementation Notes contain exact evidence, and no final gate regressed an
  earlier invariant.
- [x] Check TASK-906 acceptance criteria only after the fresh source,
  production-app, installed-wheel, static, and integrated evidence is recorded.
- [x] Update the approved specification status to `Implemented` and add the
  final task/plan links only after TASK-647–652 and TASK-904–905 are Done and
  TASK-906 has all
  implementation, verification, ADR, and documentation evidence complete.
- [x] Add TASK-906 Implementation Notes with the exact source,
  production-app, installed-wheel, static, and integrated commands; result
  counts and durations; modified files; ADR links; and any deviations. Re-read
  the task, then mark it Done only when no placeholder text remains:

```bash
backlog task 906 --plain
backlog task edit 906 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 -s Done
```

- [x] Commit final documentation/task reconciliation:

```bash
git add Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md 'backlog/tasks/task-647 - Restore-LLM-destination-actions-and-retire-the-dead-app-button-dispatcher.md' 'backlog/tasks/task-648 - Move-provider-selection-to-Settings-Console-sessions-and-a-typed-handoff.md' 'backlog/tasks/task-649 - Retire-the-unreachable-legacy-Chat-composition.md' 'backlog/tasks/task-650 - Remove-legacy-Chat-root-reactive-and-worker-state.md' 'backlog/tasks/task-651 - Remove-legacy-CCP-and-prompt-root-state.md' 'backlog/tasks/task-652 - Remove-duplicate-Media-root-state-and-stop-mutation-bubbling.md' 'backlog/tasks/task-904 - Remove-retired-Notes-Search-Ingest-Tools-and-Evals-root-state.md' 'backlog/tasks/task-905 - Retire-unreachable-TLDW-API-worker-context-and-handlers.md' 'backlog/tasks/task-906 - Close-TldwCli-reactive-ownership-with-installed-distribution-sentinels.md'
git commit -m "docs(state): close reactive ownership tranche (task-906)"
```

Before staging, run `git status --short backlog/tasks` and confirm the explicit
TASK-647–652 and TASK-904–906 sets are the only task scope being committed.
