---
id: TASK-334
title: Repair broken chat_with_provider call sites (latent ImportError)
status: Done
assignee: []
created_date: '2026-07-20 18:45'
labels: [tech-debt, bug]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The unified `chat_with_provider` dispatcher was removed, but three call sites still import it — `UI/Tools_Settings_Window.py:3361`, `Tools/code_audit_tool.py:124`, and `MCP/server.py:157` — a latent `ImportError` waiting to fire on those code paths. The real dispatcher is `chat_api_call()` (`Chat/Chat_Functions.py:646`), with provider-specific `chat_with_<provider>()` functions in `LLM_Calls/LLM_API_Calls.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All four call sites are repointed to `chat_api_call()` with `streaming=False` via `extract_response_content()` helper
- [x] #2 The affected code paths import and run without `ImportError`
- [x] #3 A grep for `chat_with_provider` returns no live references (only dead stubs in `MCP/tools.py` deferred to follow-up)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The `chat_with_provider` imports were LIVE (not dead) — all four sites repointed to `chat_api_call(..., streaming=False)` + new `extract_response_content` helper (Chat/Chat_Functions.py). Sites: MCP/server.py `chat_with_llm` (was an unguarded ImportError crashing `TldwMCPServer.__init__`); Tools/code_audit_tool.py `_request_llm_analysis` (on FileAuditSystem; soft-failing → deception analysis dead); UI/Tools_Settings_Window.py `_test_chat_connection` + `_test_all_api_keys` (buttons always reported failure). Kwarg map: provider→api_endpoint, messages→messages_payload, temperature→temp, drop timeout, add streaming=False; content extracted via extract_response_content (never-raises, always-str). A repo-wide test guards against any dead `chat_with_provider` import returning. MCP/tools.py's NotImplementedError stubs left intentional; MCPTools.chat_with_character deferred to a follow-up (it also depends on the dead save_conversation_from_messages). Files: Chat/Chat_Functions.py, MCP/server.py, Tools/code_audit_tool.py, UI/Tools_Settings_Window.py + tests.
<!-- SECTION:NOTES:END -->
