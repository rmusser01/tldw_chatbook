# Console Grouped Conversation Browser QA

Date: 2026-06-27

## Verification Commands

- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_chat_conversation_scope_service.py Tests/Chat/test_server_chat_conversation_service.py -k "mirror_report_does_not_add_local_marks or payload_does_not_forward_local_marks" -q`
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_conversation_local_marks_service.py Tests/Workspaces/test_console_conversation_browser_state.py -q`
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_workspace_context_rail.py -q`
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_chat_store.py Tests/Chat/test_chat_conversation_service.py Tests/Chat/test_chat_conversation_scope_service.py Tests/Chat/test_server_chat_conversation_service.py -q`
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k "conversation_browser or workspace_conversation" -q`
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat Tests/Workspaces Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py -q`

## Screenshot Evidence

- Default grouped browser: `Docs/superpowers/qa/console-grouped-conversation-browser/default-grouped-browser.png`
- Active workspace expanded: `Docs/superpowers/qa/console-grouped-conversation-browser/active-workspace-expanded.png`
- Search across collapsed groups: `Docs/superpowers/qa/console-grouped-conversation-browser/search-collapsed-group-match.png`
- Starred section: `Docs/superpowers/qa/console-grouped-conversation-browser/starred-section.png`
- Long list bounded with readiness rows: `Docs/superpowers/qa/console-grouped-conversation-browser/long-list-readiness-reachable.png`

## Capture Notes

- Captured from a real `tldw-serve` Textual-web session at `http://127.0.0.1:8768`.
- The capture used an isolated profile under `/private/tmp/tldw-console-browser-cdp-20260627-task6` with `TLDW_CONFIG_PATH`, `HOME`, and `XDG_*` set away from the user's real app state.
- Fixture data included real `WorkspaceDB`, `CharactersRAGDB`, workspace memberships, local starred marks, and persisted chat conversations.
- The search evidence typed `Needle Beta` into `#console-workspace-conversation-search`; the DOM text confirmed the query and the screenshot shows the Beta workspace match exposed while browsing all workspaces.

## Notes

- Stars are stored in `conversation_local_marks`.
- Stars are local-only and do not create `sync_log` rows.
- Default workspace conversations render under Chats but keep Default workspace scope.
