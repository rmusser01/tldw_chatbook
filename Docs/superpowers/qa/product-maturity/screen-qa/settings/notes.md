# Settings Screenshot QA Notes

Date: 2026-05-11
Branch: `codex/screen-qa-settings`
Backlog task: TASK-14.12
Commit: PR #306 merge `6baee11e59039340c10e2027567f62922929e968`
Screen: Settings
Viewport: 2050x1240 textual-web browser capture
Launch method: `tldw-serve --host 127.0.0.1 --port 8832` with isolated HOME/XDG config and `default_tab = "settings"`
Screenshot method: Playwright Chromium screenshot of textual-web
Fallback reason: none

## Baseline Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/settings/baseline-2026-05-11-settings.png`
- Defects: Settings rendered as an under-structured work area with no clear column ownership, oversized empty space, weak boundaries, and a visually ambiguous large-paste toggle.

## Interaction Smoke

- Goal: Verify Settings exposes a real app-level preference action and does not route runtime-specific MCP/ACP controls into global Settings.
- Steps: Opened Settings from the running app, inspected the global/console behavior areas, verified the large-paste Console setting is visible and toggleable, and verified the Appearance action routes to the customization surface.
- Result: Smoke path passed in focused mounted tests and the final screenshot shows the toggle as a readable button rather than an unreadable checkbox glyph.

## Fixes

- Summary: Converted Settings to the approved compact shell with a narrow settings-section column and wider preference-detail and scope-inspector columns. Added explicit column dividers, concise header/mode copy, a readable Console large-paste toggle, and a regression that verifies the left column remains narrower than the other two.

## Final Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/settings/final-2026-05-11-settings.png`
- User approval: approved in Codex thread after actual screenshot review

## Verification

- Commands:
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py::test_settings_destination_uses_three_column_workbench_contract Tests/UI/test_destination_shells.py::test_settings_appearance_action_routes_to_customize_surface Tests/UI/test_destination_shells.py::test_settings_console_paste_collapse_toggle_reflects_and_persists_config --tb=short`
- Results: CSS build completed; focused Settings tests passed (`4 passed`, one existing requests dependency warning).

## Residual Risks

- Settings remains a destination shell for global preferences. Deeper category navigation and additional persisted preference controls remain later product-depth work, outside this screenshot QA pass.
