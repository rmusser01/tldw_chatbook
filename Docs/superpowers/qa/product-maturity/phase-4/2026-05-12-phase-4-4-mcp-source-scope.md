# Phase 4.4 MCP Source Scope And Action Readiness

Date: 2026-05-14
Branch: `codex/phase4-4-mcp-readiness`
Backlog task: TASK-11.4
Screen: MCP

## Scope

Clarify MCP as a server-first control surface:

- Local and server scope remain explicit.
- Tools stay nested under the selected server inventory instead of flattening into generic settings.
- Runtime actions expose ready, empty, unavailable, and policy-denied states in the MCP readiness pane.
- Existing MCP destination route aliases and Unified MCP panel behavior stay compatible.

## Automated Evidence

- Command: `python -m pytest -q Tests/UI/test_destination_shells.py::test_mcp_destination_labels_server_first_workbench_columns Tests/UI/test_unified_mcp_panel.py::test_render_overview_section_shows_scannable_summary_before_raw_json Tests/UI/test_screen_navigation.py::test_main_navigation_copy_and_order --tb=short`
- Result: `3 passed, 1 warning in 8.23s`
- Command: `python -m pytest -q Tests/UI/test_unified_mcp_panel.py::test_unified_mcp_panel_switches_between_local_and_server_views Tests/UI/test_unified_mcp_panel.py::test_unified_mcp_panel_exposes_scope_and_section_controls_for_server_context Tests/UI/test_unified_mcp_panel.py::test_unified_mcp_panel_exposes_local_inventory_runtime_actions Tests/UI/test_destination_shells.py::test_mcp_destination_embeds_unified_mcp_management_panel Tests/UI/test_destination_shells.py::test_mcp_destination_labels_server_first_workbench_columns --tb=short`
- Result: `5 passed, 1 warning in 9.18s`
- Command: `python -m pytest -q Tests/UI/test_unified_mcp_panel.py Tests/UI/test_destination_shells.py::test_mcp_destination_embeds_unified_mcp_management_panel Tests/UI/test_destination_shells.py::test_mcp_destination_labels_server_first_workbench_columns --tb=short`
- Result: `15 passed, 1 warning in 8.32s`
- Command: `python -m pytest -q Tests/UI/test_unified_mcp_panel.py::test_unified_mcp_panel_explains_empty_and_policy_blocked_actions Tests/UI/test_destination_shells.py::test_mcp_destination_labels_server_first_workbench_columns --tb=short`
- Result: `2 passed, 1 warning in 5.08s`
- Command: `python -m pytest -q Tests/UI/test_unified_mcp_panel.py Tests/UI/test_destination_shells.py::test_mcp_destination_embeds_unified_mcp_management_panel Tests/UI/test_destination_shells.py::test_mcp_destination_labels_server_first_workbench_columns Tests/UI/test_shell_product_model_visibility.py::test_navigation_exposes_explicit_overflow_hint --tb=short`
- Result: `16 passed, 1 warning in 10.06s`
- Command: `python -m pytest -q Tests/UI/test_unified_mcp_panel.py Tests/UI/test_destination_shells.py::test_mcp_destination_embeds_unified_mcp_management_panel Tests/UI/test_destination_shells.py::test_mcp_destination_labels_server_first_workbench_columns Tests/UI/test_shell_product_model_visibility.py::test_navigation_exposes_explicit_overflow_hint Tests/UI/test_screen_navigation.py::test_main_navigation_copy_and_order --tb=short`
- Result: `17 passed, 1 warning in 9.53s`
- Command: `git diff --check`
- Result: passed

## Screenshot Evidence

- Capture method: textual-web at `http://127.0.0.1:8846` using Playwright-controlled Chromium, following the project CDP/browser QA runbook. The capture script rejected black screenshots before accepting evidence.
- Viewport: `2050x1240` browser viewport, device scale factor `1`.
- Screenshot: `Docs/superpowers/qa/product-maturity/phase-4/mcp-source-scope-final-real-viewport-2026-05-14.png`
- Status: approved by the user in chat on 2026-05-14.
- Notes:
  - The approved capture shows top navigation, the MCP purpose/mode rows, three clearly separated workbench columns, visible Source/Section controls, blocked-action recovery copy, and the bottom Textual status bar.
  - Earlier intermediate captures are superseded by the approved final capture because they either used forced CDP viewport metrics, exposed an intermediate layout issue, or failed screenshot validation.

## Broader Replay Status

- Command: `python -m pytest -q Tests/UI/test_product_maturity_phase1_empty_setup_states.py::test_clean_run_setup_and_runtime_blockers_expose_recovery_copy Tests/UI/test_unified_shell_phase6_first_time_replay.py Tests/UI/test_unified_shell_phase6_power_user_replay.py Tests/UI/test_unified_shell_phase6_nielsen_closeout.py --tb=short`
- Result: `3 failed, 4 passed, 1 warning in 23.37s`
- Notes: failures are outside the MCP screen pass and appear tied to stale copy/timing expectations in Personas, Settings, and a power-user replay wait condition.

## Residual Risks

- This slice clarifies existing MCP scope and readiness states. It does not implement new MCP server runtime management or policy editing flows.
- Future draggable resizing should build on the visible column divider handles added here; no drag behavior is implemented in this slice.
