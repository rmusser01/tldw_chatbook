# Workflows Screenshot QA Notes

Date: 2026-05-10
Branch: `codex/screen-qa-workflows`
Backlog task: TASK-14.8
Commit: 5e8f92a15b05b512aa42e1ec2818c67fde1b3d3a
Screen: Workflows
Viewport: 2050x1240
Launch method: `tldw-serve --host 127.0.0.1 --port 8831` with isolated HOME/XDG profile and `[general] default_tab = "workflows"`
Screenshot method: Playwright-controlled headless Chrome against textual-web
Fallback reason: none

## Baseline Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/workflows/baseline-2026-05-10-workflows.png`
- Defects: Workflows rendered as loose content instead of a clear workbench. The title/purpose/mode copy consumed excess vertical space, panes lacked explicit column titles, vertical boundaries were weak, the inspector did not summarize workflow state, and the disabled Console launch path was harder to scan than approved destination screens.

## Interaction Smoke

- Goal: Verify the no-active-run recovery path is understandable and blocks Console launch.
- Steps: Open Workflows as the default tab in textual-web with no active workflow run available; inspect the procedure list, detail pane, inspector pane, and disabled Console launch action.
- Result: The final screen exposes the blocked state, owner/recovery copy, next action, and disabled Console launch affordance without requiring hidden context.

## Fixes

- Summary: Converted Workflows to the approved destination workbench pattern: compact destination header, mode strip, full-height Procedure Library / Run Detail / Run Inspector columns, divider rails, explicit inspector state, approval summary, Console state, and next action. Added mounted regressions for the visual contract and active-run status handling.

## Final Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/workflows/final-2026-05-10-workflows.png`
- User approval: approved in chat on 2026-05-10

## Verification

- Commands:
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_workflows_screen_matches_approved_procedure_columns Tests/UI/test_console_live_work_handoffs.py::test_workflows_destination_keeps_console_launch_disabled_without_active_run Tests/UI/test_console_live_work_handoffs.py::test_workflows_destination_routes_latest_active_run_to_console --tb=short`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_live_work_handoffs.py::test_workflows_destination_keeps_console_launch_disabled_without_active_run Tests/UI/test_console_live_work_handoffs.py::test_workflows_destination_routes_latest_active_run_to_console Tests/UI/test_destination_shells.py --tb=short`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_workflows_screen_matches_approved_procedure_columns --tb=short`
  - `git diff --check`
- Results: Initial targeted red/green covered missing Workflows contract. Final focused verification passed: 81 passed / 1 warning for destination shells and Workflows handoff tests; 1 passed / 1 warning for visual parity contract; `git diff --check` passed.

## Residual Risks

- None recorded.
