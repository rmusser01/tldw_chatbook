# Home Screenshot QA Notes

Date: 2026-05-09
Branch: `codex/screen-qa-home`
Backlog task: TASK-14.2
Commit: pending
Screen: Home
Viewport: 2048x1280
Launch method: `tldw-serve --host 127.0.0.1 --port 8772`
Screenshot method: Python Playwright browser screenshot against textual-web
Fallback reason: none

## Baseline Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/home/baseline-2026-05-08-playwright-home.png`
- Defects: Home had weak/no visible pane structure, oversized empty regions, verbose readiness rows, and a follow-up area that read like a large blank canvas instead of a compact dashboard.

## Interaction Smoke

- Goal: Verify the Home primary next-best action leads to a real module workflow.
- Steps: Opened Home after splash, clicked `Import Library sources` in the attention queue / next-best action area, and captured the resulting screen.
- Result: Navigation opened Library and surfaced Library source/import/search affordances.
- Path: `Docs/superpowers/qa/product-maturity/screen-qa/home/interaction-2026-05-09-playwright-home-primary-action-library.png`

## Fixes

- Summary: Compacted the Home readiness summary, bounded the Home dashboard with terminal-style workbench treatment, added explicit vertical pane dividers, constrained the follow-up row, clarified the default selected action in the inspector, exposed keyboard hints, and replaced vague next-action/recent-work copy with actionable dashboard copy.

## Final Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/home/final-2026-05-09-playwright-home-polish2.png`
- User approval: approved in-session on 2026-05-09

## Verification

- Commands: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_home_screen.py::test_home_screen_compacts_multi_module_readiness_summary Tests/UI/test_home_screen.py::test_home_dashboard_uses_bordered_terminal_panes Tests/UI/test_home_screen.py::test_home_followup_row_stays_compact_below_dashboard_grid --tb=short`
- Results: 3 passed, 1 warning
- Commands: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_home_screen.py::test_home_screen_compacts_multi_module_readiness_summary Tests/UI/test_home_screen.py::test_home_empty_state_inspector_explains_selected_primary_action Tests/UI/test_home_screen.py::test_home_next_actions_offer_distinct_followup_choices Tests/UI/test_home_screen.py::test_home_recent_work_empty_state_sets_expectation --tb=short`
- Results: 4 passed, 1 warning
- Commands: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_home_screen.py::test_home_empty_state_inspector_explains_selected_primary_action Tests/UI/test_home_screen.py::test_home_recent_work_available_state_points_to_resume_paths Tests/UI/test_home_screen.py::test_home_followup_row_stays_compact_below_dashboard_grid --tb=short`
- Results: 3 passed, 1 warning
- Commands: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_home_screen.py Tests/Home/test_dashboard_state.py Tests/Home/test_active_work_adapter.py Tests/UI/test_unified_shell_phase2_home_adapter.py --tb=short`
- Results: 75 passed, 8 warnings
- Commands: `git diff --check`
- Results: passed

## Residual Risks

- PR review may still request follow-up adjustments before merge.
