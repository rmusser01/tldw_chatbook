# Gate 1 Core Product Loop Screen Adaptation

Date: 2026-05-06
Task: TASK-10.5
Status: verified

## Scope

Gate 1 adapts the three core-loop screens into usable Textual destination layouts:

- Home as a Command Center with status, scope, attention queue, active work, selected-item inspector, next-best action, and recent work regions.
- Console as an Agent Workbench around the existing chat surface, with staged context, transcript, run inspector, and composer contract regions.
- Library as a Source Workbench with actionable mode controls while preserving source snapshot and Console handoff behavior.

This gate intentionally does not replace `ChatWindowEnhanced` internals and does not implement full Library-native Search/RAG. Those remain required Gate 1.5 and Gate 1.6 follow-ups.

## Walkthrough

Home mounted with mixed active work and exposed `#home-dashboard-grid`, `#home-attention-queue`, `#home-active-work-region`, `#home-inspector`, `#home-next-actions-region`, and `#home-recent-work-region`. The selected-item inspector showed the active work target, and existing runtime controls remained clickable through `Tests/UI/test_home_screen.py`.

Console mounted as an agentic shell and exposed `#console-workspace-grid`, `#console-staged-context-tray`, `#console-transcript-region`, `#console-run-inspector`, and `#console-composer-region`. Existing live-work source readiness and pending launch cards still render inside the new shell, and key live-work button routing remains covered.

Library mounted with `#library-mode-bar`, `#library-source-browser`, `#library-source-detail`, and `#library-source-inspector`. Clicking `#library-mode-search` stayed on Library and changed the detail copy to `Search/RAG mode`, with Console handoff still available through the existing `Use in Console` action when sources exist.

## Functional Result

The QA pass verified usable mounted behavior rather than selector presence only:

- Home control clicks still call the app runtime hooks for approve, reject, pause, resume, retry, details, and Console handoff.
- Console preserves existing chat handoff, shell-bar, live-work readiness, pending launch rendering, and W+C live-work primary action routing.
- Library mode selection is clickable in the mounted app and does not leave the Library route or break source snapshot, study, or destination-shell behavior.

## Verification

Red regression:

- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py --tb=short`
- Result before implementation: `3 failed`, covering missing Home, Console, and Library Gate 1 contracts.

Focused implementation checks:

- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_home_core_loop_uses_dashboard_regions_and_selected_item_inspector Tests/UI/test_home_screen.py --tb=short`
- Result: `22 passed, 1 warning`
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_shell_bar.py Tests/UI/test_console_live_work_handoffs.py::test_console_renders_pending_launch_context Tests/UI/test_console_live_work_handoffs.py::test_console_renders_source_readiness_summary_without_pending_launch Tests/UI/test_console_live_work_handoffs.py::test_console_wc_live_work_action_button_routes_run_details --tb=short`
- Result: `37 passed, 1 warning`
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_library_core_loop_modes_are_actionable_without_leaving_library Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_destination_shells.py --tb=short`
- Result: `91 passed, 1 warning`

## Defects

- Initial Home adaptation rendered all Active Work controls, but later controls were outside the visible parent region in mounted tests. Fixed by giving the Home dashboard row deterministic height and compact control hit targets.
- Initial Library mode buttons existed but were out of bounds because the test harness did not load app TCSS and the mode label consumed the full row width. Fixed with inline mode-bar sizing so mode controls stay reachable in mounted tests.

No P0 or P1 defect remains in the verified Gate 1 scope.

## Residual Risk

- `ChatWindowEnhanced` remains the legacy Console internals inside the new Console shell. Gate 1.5 must decompose or replace it with Console-native components.
- Library `Search/RAG` mode is selectable and honest, but full Library-native retrieval, evidence, citation, and handoff workflow remains Gate 1.6.
- Other destinations are not adapted by this gate and remain owned by later Gate 2 and Gate 3 plans.

## Exit Decision

Pass for Gate 1. Home, Console, and Library now expose the approved core product-loop screen regions with mounted interaction coverage. Continue with Gate 1.5 for Console internals and Gate 1.6 for Library-native Search/RAG before broad Gate 2 destination rewrites.
