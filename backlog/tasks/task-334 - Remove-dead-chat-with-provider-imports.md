---
id: TASK-334
title: Remove dead chat_with_provider imports (latent ImportError)
status: To Do
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
- [ ] #1 The three dead imports are removed or repointed to `chat_api_call()` / `chat_with_<provider>()`
- [ ] #2 The affected code paths import and run without `ImportError`
- [ ] #3 A grep for `chat_with_provider` returns no live references
<!-- AC:END -->
