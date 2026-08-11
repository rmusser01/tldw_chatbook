---
id: TASK-15103
title: Reconcile 17-owner latest-dev diagnostic inventory drift
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-11 04:38'
updated_date: '2026-08-11 13:47'
labels:
  - testing
  - baseline
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-3796 final review independently reproduced a stale persistent-diagnostic inventory on exact origin/dev 6d72f15f8332b6469a5d644d409b80914634a8dd. The generated-versus-stored baseline differs across exactly 17 unrelated owner paths: tldw_chatbook/Agents/agent_service.py; tldw_chatbook/Chat/console_agent_bridge.py; tldw_chatbook/Chat/console_chat_controller.py; tldw_chatbook/Chat/console_chat_store.py; tldw_chatbook/Chat/console_context_compaction.py; tldw_chatbook/Chat/console_provider_gateway.py; tldw_chatbook/MCP/client.py; tldw_chatbook/MCP/local_server_tools.py; tldw_chatbook/MCP/prompts.py; tldw_chatbook/MCP/server.py; tldw_chatbook/RAG_Search/fusion.py; tldw_chatbook/RAG_Search/simplified/rag_service.py; tldw_chatbook/RAG_Search/simplified/search_service.py; tldw_chatbook/UI/Console_Modules/session.py; tldw_chatbook/UI/Screens/chat_screen.py; tldw_chatbook/UI/Screens/library_screen.py; and tldw_chatbook/app.py. The detached-base Git-patch manifest diff is 44 additions/30 deletions with SHA-256 b77bd95ccc84d3bac066e0971a8bc24e20fdb58bef9b762d5ba77aa6399db4dd; owner files are 485 to 487, TASK-492 calls 1,167 to 1,200, TASK-494 calls 6,962 to 6,986, and persistent-sink topology remains six files. Review and reconcile this exact current-dev incident under ADR-029 without allowing TASK-3796 to bless unrelated drift. This record moved from the later-claimed TASK-14914 to TASK-15103 during the TASK-3796 PR rebase because exact add-commit provenance established that dev's visual-compaction task claimed TASK-14914 first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Unsafe private values or exception details in the recorded delta are repaired without changing unrelated production behavior
- [ ] #2 The persistent-diagnostic inventory is regenerated with only reviewed owner changes and unchanged six-file sink topology
- [ ] #3 The focused architecture checker and regression coverage pass without constructing a test application
- [ ] #4 Every generated-versus-stored delta for the recorded 17 owner paths is reviewed under ADR-029
<!-- AC:END -->

## Implementation Plan

Detailed executable plan:
`Docs/superpowers/plans/2026-08-11-task-15103-diagnostic-inventory-reconciliation.md`

1. Freeze the exact 17-owner incident in a schema-validated multiset ledger
   with per-group provenance and disposition evidence.
2. Replace the shallow TASK-15103 review path with a ledger-driven guard that
   reuses the existing alias-aware diagnostic extractor and mutation-tests all
   supported message/capture forms.
3. Add direct real-production-function privacy sentinels and repair unsafe
   Agents, Chat, MCP, RAG, UI, and application diagnostics in reviewed batches.
4. Regenerate the production diagnostic manifest only after all source and
   ledger gates pass, then prove the boundary rejects unknown data, forged
   summaries, extra owners, classification changes, and sink changes.
5. Rebase onto the latest `dev`, compare the complete call population for all
   17 owners, rerun only touched-function tests/static gates, complete
   independent review, and close the task.

ADR required: no

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: this task enforces the existing ADR-029 privacy boundary without
changing persistent-sink ownership, storage, or the metadata policy.
