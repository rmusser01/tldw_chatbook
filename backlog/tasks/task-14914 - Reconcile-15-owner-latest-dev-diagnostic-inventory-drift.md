---
id: TASK-14914
title: Reconcile 15-owner latest-dev diagnostic inventory drift
status: To Do
assignee: []
created_date: '2026-08-11 04:38'
updated_date: '2026-08-11 04:47'
labels:
  - testing
  - baseline
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-3796 final review independently reproduced a stale persistent-diagnostic inventory on exact origin/dev b4c5105eda4c38a5b5446d7982c36d5fefaec8a1. The generated-versus-stored baseline differs across exactly 15 unrelated owner paths: tldw_chatbook/Agents/agent_service.py; tldw_chatbook/Chat/console_agent_bridge.py; tldw_chatbook/Chat/console_chat_controller.py; tldw_chatbook/Chat/console_chat_store.py; tldw_chatbook/Chat/console_context_compaction.py; tldw_chatbook/Chat/console_provider_gateway.py; tldw_chatbook/MCP/client.py; tldw_chatbook/MCP/local_server_tools.py; tldw_chatbook/MCP/prompts.py; tldw_chatbook/MCP/server.py; tldw_chatbook/RAG_Search/fusion.py; tldw_chatbook/RAG_Search/simplified/rag_service.py; tldw_chatbook/RAG_Search/simplified/search_service.py; tldw_chatbook/UI/Screens/chat_screen.py; and tldw_chatbook/UI/Screens/library_screen.py. The detached-base manifest diff is 40 additions/26 deletions with SHA-256 2a17c75f2756f03c38d24d209a673797fbd720800cea7c268a70cddf1af559d3; owner files are 485 to 487, TASK-492 calls 1,167 to 1,201, TASK-494 calls 6,962 to 6,979, and persistent-sink topology remains six files. Review and reconcile this exact current-dev incident under ADR-029 without allowing TASK-3796 to bless unrelated drift.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every generated-versus-stored delta for the recorded 15 owner paths is reviewed under ADR-029
- [ ] #2 Unsafe private values or exception details in the recorded delta are repaired without changing unrelated production behavior
- [ ] #3 The persistent-diagnostic inventory is regenerated with only reviewed owner changes and unchanged six-file sink topology
- [ ] #4 The focused architecture checker and regression coverage pass without constructing a test application
<!-- AC:END -->
