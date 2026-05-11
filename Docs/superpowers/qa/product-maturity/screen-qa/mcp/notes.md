# MCP Screenshot QA Notes

Date: 2026-05-10
Branch: `codex/screen-qa-mcp`
Backlog task: TASK-14.9
Commit: 5ddb311ed56bb750290ab9daf413fe552da3ce37
Screen: MCP
Viewport: 2050x1240
Launch method: `tldw-serve --host 127.0.0.1 --port 8832` with isolated HOME/XDG profile and `[general] default_tab = "mcp"`
Screenshot method: Playwright-controlled headless Chrome against textual-web
Fallback reason: none

## Baseline Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/mcp/baseline-2026-05-10-mcp.png`
- Defects: MCP rendered the destination title as a large empty header block, detached purpose/mode copy below it, and constrained the workbench to the upper portion of the viewport. The three functional columns existed, but the layout did not match the compact approved destination workbench pattern and wasted substantial vertical space.

## Interaction Smoke

- Goal: Verify the default local MCP overview and blocked action path are visible without configured remote servers.
- Steps: Open MCP as the default tab in textual-web with the isolated runtime profile; inspect the source/server/scope controls, overview detail payload, disabled action select, payload box, and disabled Run Action affordance.
- Result: The final screen keeps the local MCP overview readable, exposes server/scope controls, and honestly disables action execution when no action is available.

## Fixes

- Summary: Added MCP to the approved compact destination shell and workbench TCSS selector groups, regenerated the modular stylesheet, and added a focused MCP visual parity assertion for compact shell rows and full-height workbench panes.

## Final Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/mcp/final-2026-05-10-mcp.png`
- User approval: approved in chat on 2026-05-10

## Verification

- Commands:
  - python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_mcp_uses_visible_server_detail_readiness_layout_without_overflow Tests/UI/test_destination_visual_parity_correction.py::test_mcp_unavailable_or_local_default_state_keeps_workbench_geometry Tests/UI/test_destination_visual_parity_correction.py::test_mcp_forced_loading_state_stays_inside_workbench --tb=short
  - python tldw_chatbook/css/build_css.py
  - `git diff --check`
- Results: Focused MCP visual parity verification passed: 3 passed / 1 warning. CSS regenerated successfully; `git diff --check` passed.

## Residual Risks

- None recorded.
