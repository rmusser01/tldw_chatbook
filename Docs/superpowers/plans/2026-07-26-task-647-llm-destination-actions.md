# LLM Destination Action Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every visible Models action truthful and destination-owned, remove the unsupported custom Transformers launch UI, and delete the dead `TldwCli` button dispatcher and LLM root view state.

**Architecture:** `LLMManagementWindow` owns navigation and an allowlisted action registry. UI lookup stays inside that mounted window; app-owned process handles and workers are passed explicitly to existing handlers where required. Supported controls stop their event before dispatch, while the unimplemented Transformers server-launch block is removed rather than simulated.

**Tech Stack:** Python 3.11+, Textual messages/workers/reactives, pytest/pytest-asyncio, AST ownership checks, Ruff.

**Backlog:** [TASK-647](../../../backlog/tasks/task-647%20-%20Restore-LLM-destination-actions-and-retire-the-dead-app-button-dispatcher.md)

**Specification:** [TldwCli Reactive State Decomposition Design](../specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md)

**Depends on:** None

**ADR required:** yes

**ADR path:** `backlog/decisions/026-application-session-state-ownership.md`; `backlog/decisions/011-chatbook-workbench-ui-system.md`

**Reason:** The accepted ADRs already require destination-owned view state and actions; this task implements that boundary without creating a new decision.

---

## Execution and Test Boundary

Activate the existing environment and verify that imports resolve to this worktree:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate
python -c "import pathlib, tldw_chatbook; print(pathlib.Path(tldw_chatbook.__file__).resolve())"
```

The printed path must be under `.worktrees/privacy-lifecycle-eval-wheel-hardening`.

Mounted behavior belongs in `Tests/ProductionApp/`, not `Tests/UI/`. The first
slice creates a `Tests/ProductionApp/conftest.py` that establishes private
`HOME`, `USERPROFILE`, XDG, data, config, and temp roots before application
imports. Tests construct the normal `TldwCli`, use `app.run_test()`, and query
the registered `LLMScreen`/`LLMManagementWindow`. Do not import or request
anything from `Tests.textual_test_harness`, `Tests.textual_test_utils`, or the
surrogate fixtures in `Tests/UI/conftest.py`.

Direct AST and handler-function tests are allowed. An action callback recorder
or injected exception on the real mounted window is allowed; an `App`,
`TldwCli`, `Screen`, or destination substitute is not.

## File Structure

- Modify `tldw_chatbook/UI/LLM_Management_Window.py`: own action registration, event stopping, bounded recovery, truthful button state, and the retained Transformers model-operations output.
- Modify `tldw_chatbook/Event_Handlers/LLM_Management_Events/*.py`: normalize supported action call signatures so UI queries target the destination and lifecycle work uses the explicit app.
- Modify `tldw_chatbook/Utils/log_widget_manager.py`: target the mounted destination for Transformers model-operation output instead of querying through the app root.
- Delete `tldw_chatbook/Event_Handlers/llm_nav_events.py`: remove the duplicate root-reactive navigation path.
- Modify `tldw_chatbook/Event_Handlers/tab_initializers/misc_tab_initializers.py` and `__init__.py`: remove the unused LLM initializer that writes root state.
- Modify `tldw_chatbook/app.py`: remove LLM event-map imports, `button_handler_map`, `_build_handler_map()`, the no-op root `on_button_pressed()`, `llm_active_view`, `_initial_llm_view`, and `watch_llm_active_view()`.
- Create `Tests/ProductionApp/conftest.py`: private environment setup only; no app fixture or alternate application.
- Create `Tests/ProductionApp/test_llm_destination_actions.py`: mounted production navigation/action behavior.
- Modify `Tests/test_application_state_ownership.py`: structural dispatcher and root-owner guards.

## Task 1: Start TASK-647 and Freeze the Button Census

- [ ] Move TASK-647 to In Progress and add the task-local implementation plan, including the ADR fields above, before changing production code:

```bash
backlog task edit 647 -s "In Progress"
backlog task edit 647 --plan $'ADR required: yes\nADR path: backlog/decisions/026-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md\nReason: Existing ADRs assign view state and actions to the mounted destination.\n\n1. Freeze the visible Models button census.\n2. Register supported actions on LLMManagementWindow.\n3. Remove the unsupported Transformers launch block.\n4. Delete the root dispatcher and duplicate LLM state.\n5. Run production-app, structural, and static gates.'
```

- [ ] Add an AST census test that enumerates actionable `Button` IDs composed directly by `LLMManagementWindow`, partitions navigation IDs from action IDs, and requires each visible action to be in the destination allowlist or the explicit removal set.
- [ ] Put exactly these unsupported IDs in the removal set:
  `transformers-browse-script-button`,
  `transformers-start-server-button`, and
  `transformers-stop-server-button`.
- [ ] Run the census before implementation:

```bash
pytest Tests/ProductionApp/test_llm_destination_actions.py -q -k census
```

Expected: FAIL because action registration is absent and the unsupported block is still composed.

## Task 2: Route Supported Actions Through the Mounted Destination

- [ ] Add one destination-local action handler for non-navigation buttons. It must call `event.stop()` before lookup or await, ignore unknown IDs, and support sync or async registered functions.
- [ ] Normalize the existing provider-specific maps to one `(window, app, event)` contract. UI queries use `window.query_one()`; notifications, worker launch, process handles, and `call_from_thread` use the explicit production app.
- [ ] Do not add a controller, generic event bus, or second process owner. Existing provider modules may stay separate.
- [ ] On an exception, identify only the action ID and exception category, restore the corresponding start/stop controls to a truthful state, and show bounded recovery copy without rendering command arguments, paths, model IDs, or subprocess output.
- [ ] Add mounted tests that:
  - navigate from `LLMScreen` to a second real view and observe only `LLMManagementWindow.active_view`;
  - invoke a safe Transformers list-local-models validation path exactly once;
  - fault-inject one registered action on the actual mounted window and verify bounded recovery plus stopped propagation;
  - prove an unknown action ID is ignored.
- [ ] Run:

```bash
pytest Tests/ProductionApp/test_llm_destination_actions.py -q -k "navigation or action or failure"
```

Expected: PASS.

## Task 3: Remove the Unsupported Transformers Server Block

- [ ] Delete the custom server-script, interpreter, host, port,
  additional-arguments, browse-script, start, and stop controls from the
  Transformers view.
- [ ] Keep and locally route the models-directory browse, list-local-models, and download-model controls.
- [ ] Keep `#transformers-log-output`, move/relabel it as the retained model
  operations output beside the list/download controls, and update
  `log_widget_manager.py` plus the Transformers handlers so all UI queries and
  writes target the mounted `LLMManagementWindow`. Remove the app-root
  `_update_transformers_log()` forwarding helper.
- [ ] Remove any orphan selector, helper, or test reference associated only with the deleted launch block. Do not implement a Transformers server process lifecycle.
- [ ] Exercise retained list and download failure/success output paths and
  assert they update the mounted output without `QueryError`.
- [ ] Run the census and a real mounted Transformers view assertion:

```bash
pytest Tests/ProductionApp/test_llm_destination_actions.py -q -k transformers
```

Expected: PASS, with all three unsupported IDs absent and supported controls present.

## Task 4: Delete Root Dispatch and Duplicate LLM View State

- [ ] In `app.py`, remove `button_handler_map`, `_build_handler_map()`, the root no-op `on_button_pressed()`, LLM map imports, `llm_active_view`, `_initial_llm_view`, and `watch_llm_active_view()`.
- [ ] Delete `llm_nav_events.py` and remove its package/import references.
- [ ] Remove `LLMTabInitializer` from the unused initializer facade. Help text must be populated by `LLMManagementWindow` mount/view activation, not a root-state initializer.
- [ ] Extend the AST guard to reject:
  - a `TldwCli` descriptor or root access named `llm_active_view`;
  - `button_handler_map`, `_build_handler_map`, and constant `reactive_attr` dispatch strings;
  - production import of `llm_nav_events`.
- [ ] Run:

```bash
pytest Tests/ProductionApp/test_llm_destination_actions.py Tests/test_application_state_ownership.py -q
```

Expected: PASS.

## Task 5: Verify and Close TASK-647

- [ ] Run focused static gates:

```bash
python -m compileall -q tldw_chatbook/UI/LLM_Management_Window.py tldw_chatbook/Event_Handlers/LLM_Management_Events tldw_chatbook/Event_Handlers/tab_initializers tldw_chatbook/Utils/log_widget_manager.py tldw_chatbook/app.py
python -m ruff check tldw_chatbook/UI/LLM_Management_Window.py tldw_chatbook/Event_Handlers/LLM_Management_Events tldw_chatbook/Event_Handlers/tab_initializers tldw_chatbook/Utils/log_widget_manager.py tldw_chatbook/app.py Tests/ProductionApp/conftest.py Tests/ProductionApp/test_llm_destination_actions.py Tests/test_application_state_ownership.py
python -m ruff format --check tldw_chatbook/UI/LLM_Management_Window.py tldw_chatbook/Event_Handlers/LLM_Management_Events tldw_chatbook/Event_Handlers/tab_initializers tldw_chatbook/Utils/log_widget_manager.py Tests/ProductionApp/conftest.py Tests/ProductionApp/test_llm_destination_actions.py Tests/test_application_state_ownership.py
git diff --check
```

- `app.py` is a verified pre-task Ruff-format baseline exception; do not
  mass-format it. It remains covered by Ruff lint, compile, focused tests, and
  diff hygiene.

- [ ] Commit code and tests:

```bash
git add tldw_chatbook/UI/LLM_Management_Window.py tldw_chatbook/Event_Handlers/LLM_Management_Events tldw_chatbook/Event_Handlers/tab_initializers/misc_tab_initializers.py tldw_chatbook/Event_Handlers/tab_initializers/__init__.py tldw_chatbook/Utils/log_widget_manager.py tldw_chatbook/app.py Tests/ProductionApp/conftest.py Tests/ProductionApp/test_llm_destination_actions.py Tests/test_application_state_ownership.py
git add -u tldw_chatbook/Event_Handlers/llm_nav_events.py
git commit -m "refactor(llm): own Models actions at destination (task-647)"
```

- [ ] Re-read TASK-647, check every acceptance criterion only against fresh evidence, add concise Implementation Notes including the accepted ADRs and exact test counts, then mark Done:

```bash
backlog task 647 --plain
backlog task edit 647 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 -s Done
```

- [ ] Commit the completed task file separately:

```bash
git add 'backlog/tasks/task-647 - Restore-LLM-destination-actions-and-retire-the-dead-app-button-dispatcher.md'
git commit -m "docs(backlog): close Models action ownership (task-647)"
```
