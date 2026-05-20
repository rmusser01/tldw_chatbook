# Console Workspace Context Rail QA

Date: 2026-05-20
Branch: `codex/workspace-console-context-panel`
Backlog task: TASK-64
Screen: Console

## Scope

Implement the read-only Console workspace context rail from the workspace operating context plan:

- Split the Console left rail into `Staged Context` and `Convos & Workspaces`.
- Render missing-service, no-active-workspace, and active-workspace states from pure display-state logic.
- Keep workspace switching and new conversation creation disabled until later slices wire those actions.
- Preserve the existing terminal-native three-column Console workbench and composer behavior.

This slice does not implement workspace switching, conversation persistence, workspace-scoped Library filtering, sync transfer, or server handoff.

## Automated Evidence

- Command: `python -m pytest -q Tests/Workspaces/test_workspace_display_state.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_destination_visual_parity_correction.py::test_console_uses_three_pane_workbench_and_visible_composer Tests/UI/test_destination_visual_parity_correction.py::test_core_default_empty_or_blocked_states_keep_workbench_geometry Tests/UI/test_console_internals_decomposition.py::test_console_workbench_weights_transcript_as_primary_region Tests/UI/test_console_internals_decomposition.py::test_console_workbench_panes_have_visible_terminal_frames Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions --tb=short`
- Result: `16 passed, 8 warnings in 17.94s`
- Command: `git diff --check`
- Result: passed with no output

## Screenshot Evidence

- Capture method: actual running app via `textual-web` on localhost using Playwright-controlled Chromium/CDP.
- Viewport: `2050x1240` browser viewport, device scale factor `1`.
- Screenshot: `Docs/superpowers/qa/product-maturity/phase-4/actual-visual-captures/console-workspace-context-rail-2026-05-20-disabled-actions.png`
- Review-fix screenshot: `Docs/superpowers/qa/product-maturity/phase-4/actual-visual-captures/console-workspace-context-rail-2026-05-20-review-fixes.png`
- Image check: `2050 x 1240`, RGB PNG.
- User approval: approved in the product UX review thread.

## QA Walkthrough Notes

- The approved capture shows the Console with the existing top navigation and mode/status rows intact.
- The left column is visibly split into two framed panels: `Staged Context` above `Convos & Workspaces`.
- The workspace rail shows an active workspace, authority/sync/runtime rows, and a conversation row without hiding Library or Notes content.
- `Change workspace` and `New conversation` are visible but disabled, with recovery copy that names the later implementation slice instead of implying the actions work now.
- The transcript remains the primary center column and the run inspector remains the narrower right column.
- Review fix replay removes the redundant left-rail parent frame while keeping distinct framed `Staged Context` and `Convos & Workspaces` sections.

## Residual Risks

- Workspace switching and new workspace conversation creation are intentionally not wired in this slice.
- Active workspace state currently comes from the local registry display state; cross-device/server handoff and ACP package transfer remain later slices.
- The rail is designed for later live actions and resizable panes, but this slice does not add drag resizing.
