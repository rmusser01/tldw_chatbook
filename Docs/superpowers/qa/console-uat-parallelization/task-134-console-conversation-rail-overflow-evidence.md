# TASK-134 Console Conversation Rail Overflow Evidence

Date: 2026-06-25

## Scope

Rendered Textual-web/CDP evidence for the Console workspace conversation rail overflow fix.

## Environment

- Branch: `codex/console-uat-findings-fixes`
- Evidence capture base HEAD: `ec8a3659`
- Final closeout commit: commit containing this evidence note
- Server URL: `http://127.0.0.1:8765/?fontsize=12`
- Isolated HOME: `/tmp/tldw-task-134-cdp-0029/home`
- Isolated config: `/tmp/tldw-task-134-cdp-0029/config`
- Isolated data: `/tmp/tldw-task-134-cdp-0029/data`
- Isolated cache: `/tmp/tldw-task-134-cdp-0029/cache`
- Workspace DB: `/tmp/tldw-task-134-cdp-0029/home/.local/share/tldw_cli/default_user/tldw_chatbook_workspaces.db`
- Chat DB: `/tmp/tldw-task-134-cdp-0029/home/.local/share/tldw_cli/default_user/tldw_chatbook_ChaChaNotes.db`
- App log: `/tmp/tldw-task-134-cdp-0029/home/.local/share/tldw_cli/default_user/tldw_cli_app.log`
- Launch command: `/usr/bin/env PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook HOME=/tmp/tldw-task-134-cdp-0029/home XDG_CONFIG_HOME=/tmp/tldw-task-134-cdp-0029/config XDG_DATA_HOME=/tmp/tldw-task-134-cdp-0029/data XDG_CACHE_HOME=/tmp/tldw-task-134-cdp-0029/cache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/tldw-serve --host 127.0.0.1 --port 8765`

## Seed Data

- Default workspace was created through `LocalWorkspaceRegistryService.ensure_default_workspace()`.
- `ChatPersistenceService.create_conversation(..., scope_type="workspace", workspace_id=DEFAULT_WORKSPACE_ID)` created 36 saved workspace conversations.
- Seeded titles include `Overflow Topic NN` and four searchable conversations: `Search Needle 07`, `Search Needle 17`, `Search Needle 27`, and `Search Needle 35`.

## Evidence

- `Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-rail-overflow-cdp-2026-06-25.png`
  - Shows the Default workspace with `Conversations (37)`, a bounded conversation list, `New conversation`, and the lower `Storage local` row visible above the composer.
- `Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-search-cdp-2026-06-25.png`
  - Shows real UI search interaction for `Search Needle`, four matching workspace conversations, `4 matches`, an active `Clear` control, and `New conversation` still available below the filtered list.

## Verification

- `.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py -k "workspace_conversation or workspace_rail or workspace_switch or default_workspace or conversation_search" --tb=short`
  - Result: `45 passed, 91 deselected, 1 warning in 38.43s`
- `.venv/bin/python -m pytest -q Tests/UI/test_console_session_settings.py::test_console_left_rail_body_scrolls_below_fixed_header_without_settings_summary Tests/UI/test_console_internals_decomposition.py::test_console_left_rail_sections_use_available_space --tb=short`
  - Result: `2 passed, 1 warning in 5.18s`
- `git diff --check`
  - Result: no output
- `file Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-rail-overflow-cdp-2026-06-25.png Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-search-cdp-2026-06-25.png`
  - Result: PNG image data, 1280 x 720 for both screenshots.

## ADR Check

- ADR required: no
- ADR path: N/A
- Reason: bounded/collapsible/searchable rail presentation and local UI preference state only; no schema, sync policy, ownership, provider/runtime, service contract, or security boundary change.

## Approval

- State: captured, pending user/PR approval.

## Residual Risk

- Provider-response UAT is not claimed here; the captured running app shows the real provider setup blocked state because no API key was configured in the isolated environment.
- CDP evidence covers the Default workspace. Cross-workspace search, collapse, and selection boundaries are covered by focused mounted regressions.
