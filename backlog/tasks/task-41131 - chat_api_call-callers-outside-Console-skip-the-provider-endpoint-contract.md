---
id: TASK-41131
title: "chat_api_call callers outside Console skip the provider endpoint contract"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- security
- providers
priority: medium
references:
- tldw_chatbook/Chat/Chat_Functions.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 endpoint stream fu/r14-endpoint 6cb8e34461 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-40300 made Console's send, readiness, registry and project identity refuse every endpoint the provider endpoint contract refuses (zone ids, leading-hyphen hosts, 127.1, IDN, query or fragment). The same configured endpoint still reaches the network from every other caller of the unified dispatcher.

**Evidence.** `chat_api_call` (`Chat/Chat_Functions.py:1029`) applies no contract check (`endpoint_refusal` is not referenced in the file). On the endpoint branch only `chat_with_llama` checks it (`LLM_API_Calls_Local.py:899-910`). Callers outside Console reported by the endpoint stream: `Chat_Functions.chat` (legacy), `Chat/document_generator.py`, `AgentService`'s default chat call, Dreams, Summarization, Local_Ingestion, Research, Evals, Library, WebSearch, MCP sampling and the Personal Context interview.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every provider call that resolves a local or custom endpoint is refused before any network I/O when the contract refuses the endpoint, with the contract's reason
- [ ] #2 The check sits where all callers route (for example chat_api_call or the provider handlers), not in each caller
- [ ] #3 Tests cover at least the dispatcher plus two non-Console callers with the refused shapes, red before the change
<!-- AC:END -->
