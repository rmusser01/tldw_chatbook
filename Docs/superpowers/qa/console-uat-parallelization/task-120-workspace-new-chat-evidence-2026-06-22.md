# TASK-120 Workspace New Chat CDP Evidence

Date: 2026-06-23

## Scope

Rendered Textual-web/CDP evidence for Console workspace-scoped new chat creation.

## Environment

- Branch: `codex/task-120-workspace-new-chat`
- HEAD: rebased `codex/task-120-workspace-new-chat` on `origin/dev`
- Compared source branch: `codex/console-workspace-new-chat`
- Server URL: `http://127.0.0.1:8767/?fontsize=12`
- Isolated HOME: `/tmp/tldw-task-120-rebased-cdp/home`
- Isolated config: `/tmp/tldw-task-120-rebased-cdp/config`
- Isolated data: `/tmp/tldw-task-120-rebased-cdp/data`
- Isolated cache: `/tmp/tldw-task-120-rebased-cdp/cache`
- Launch command: `/usr/bin/env PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-120-workspace-new-chat HOME=/tmp/tldw-task-120-rebased-cdp/home XDG_CONFIG_HOME=/tmp/tldw-task-120-rebased-cdp/config XDG_DATA_HOME=/tmp/tldw-task-120-rebased-cdp/data XDG_CACHE_HOME=/tmp/tldw-task-120-rebased-cdp/cache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/tldw-serve --host 127.0.0.1 --port 8767`

## Evidence

- `Docs/superpowers/qa/console-uat-parallelization/task-120-workspace-new-conversation-initial-cdp-2026-06-22.png`
  - Shows Console's Default workspace rail with `New conversation`, `Chat 1 [active]`, `Storage: local`, `Sync: Off`, `File tools: Off in Default workspace`, and server handoff marked `Not configured`.
- `Docs/superpowers/qa/console-uat-parallelization/task-120-workspace-new-conversation-after-click-cdp-2026-06-22.png`
  - Shows the post-click state with `Chat 2 [active]` and `Chat 1 [open]` in the Default workspace rail while local storage, disabled sync/file tools, and unavailable handoff copy remain visible.

## Verification

- `.venv/bin/python -m pytest -q Tests/Workspaces/test_workspace_display_state.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_list_reserves_two_line_rows_with_margin Tests/UI/test_console_native_chat_flow.py::test_console_new_chat_tab_appears_in_workspace_conversation_rail Tests/UI/test_console_native_chat_flow.py::test_console_new_chat_tab_promotes_active_native_session_in_workspace_rail Tests/UI/test_console_native_chat_flow.py::test_console_workspace_rail_new_conversation_creates_default_workspace_session Tests/UI/test_console_native_chat_flow.py::test_console_workspace_rail_new_conversation_stays_scoped_to_active_workspace Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_row_switches_native_session Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_switch_restores_transcript_messages --tb=short`
  - Result: `38 passed, 8 warnings in 28.77s`
- `git diff --check`
  - Result: no output
- `file Docs/superpowers/qa/console-uat-parallelization/task-120-workspace-new-conversation-initial-cdp-2026-06-22.png Docs/superpowers/qa/console-uat-parallelization/task-120-workspace-new-conversation-after-click-cdp-2026-06-22.png`
  - Result after conversion: PNG image data, 1280 x 720 for both screenshots.

## Approval

- State: captured, pending user/PR approval.

## Residual Risk

- The rendered evidence covers Default workspace creation. Cross-workspace visibility boundaries are covered by mounted regression tests in `Tests/UI/test_console_native_chat_flow.py`.
