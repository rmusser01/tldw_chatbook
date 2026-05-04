# Phase 4.1 MCP Destination Service Adoption

Task: `TASK-5.1`
Branch: `codex/unified-shell-phase4-mcp-destination-service`

## Goal

Turn the top-level MCP destination from a disabled placeholder into a real service-backed control surface by adopting the existing `UnifiedMCPPanel` inside the primary shell.

## Implementation Summary

- Replaced the MCP destination placeholder panel with the existing `UnifiedMCPPanel`.
- Preserved the `tools_settings` route as an MCP alias through the same `MCPScreen` destination.
- Added MCP screen state save and restore for the Unified MCP view state.
- Added runtime-backend refresh delegation so the embedded panel reloads context when runtime source changes.
- Added tooltip copy to the shared Unified MCP action button so the newly exposed action remains self-explanatory in destination-level tooltip checks.

## Verification

- Baseline focused command before changes: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_unified_mcp_panel.py -q`
- Baseline focused result: `44 passed, 3 warnings in 19.61s`.
- Red result: `test_mcp_destination_embeds_unified_mcp_management_panel` failed because no `UnifiedMCPPanel` was mounted on `MCPScreen`.
- Green command after implementation: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_unified_mcp_panel.py -q`
- Green result: `44 passed, 1 warning in 15.64s`.
- Tracking red result: `test_mcp_destination_service_adoption_tracking_evidence_exists` failed because this evidence file did not exist.

## QA Walkthrough Notes

- Environment: focused Textual mounted-window tests using the repo virtualenv.
- Entry path: top navigation `MCP` destination and legacy `tools_settings` alias.
- Visual check: MCP still shows the destination header and ownership copy, then exposes the Unified MCP panel with source, server, scope, section, action, payload, and result controls.
- Functional result: the top-level MCP destination now reaches the existing Unified MCP service seam instead of requiring users to know the legacy Tools Settings route.
- Alias result: `tools_settings` still resolves to `MCPScreen`, preserving compatibility while no longer behaving like global Settings.
- Recovery result: if no Unified MCP service is available in the app session, the panel shows its existing unavailable status inside the real MCP management surface rather than a separate disabled shell placeholder.

## Residual Risk

- This slice adopts the existing Unified MCP panel; it does not expand MCP server capabilities, policy behavior, or action execution semantics.
- Console source readiness still marks MCP as not wired for live-work context because this slice is destination-service adoption, not Console MCP launch/follow integration.
- Manual full-app keyboard walkthrough remains part of later Phase 4/6 closeout; this slice is verified through focused mounted-window QA.
