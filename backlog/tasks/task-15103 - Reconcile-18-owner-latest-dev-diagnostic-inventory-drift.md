---
id: TASK-15103
title: Reconcile 18-owner latest-dev diagnostic inventory drift
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-11 04:38'
updated_date: '2026-08-11 15:59'
labels:
  - testing
  - baseline
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Latest-dev stop-gate revalidation on exact `origin/dev` `85863257dd7a30b16451f8f32e0c7142dd1d5273` confirms the generated-versus-stored baseline contains exactly 18 unrelated owner paths: `tldw_chatbook/Agents/agent_service.py`; `tldw_chatbook/Chat/console_agent_bridge.py`; `tldw_chatbook/Chat/console_chat_controller.py`; `tldw_chatbook/Chat/console_chat_store.py`; `tldw_chatbook/Chat/console_context_compaction.py`; `tldw_chatbook/Chat/console_provider_gateway.py`; `tldw_chatbook/MCP/client.py`; `tldw_chatbook/MCP/local_server_tools.py`; `tldw_chatbook/MCP/prompts.py`; `tldw_chatbook/MCP/server.py`; `tldw_chatbook/RAG_Search/fusion.py`; `tldw_chatbook/RAG_Search/simplified/rag_service.py`; `tldw_chatbook/RAG_Search/simplified/search_service.py`; `tldw_chatbook/UI/Console_Modules/session.py`; `tldw_chatbook/UI/Screens/chat_screen.py`; `tldw_chatbook/UI/Screens/library_screen.py`; `tldw_chatbook/app.py`; and, as the 18th path, `tldw_chatbook/UI/Screens/settings_screen.py`. The intervening visual-evaluation work changes `console_provider_gateway.py` without changing its diagnostic population or sink set and adds no diagnostic owner. The detached canonical `--write` Git-patch manifest diff remains 46 additions/32 deletions with SHA-256 `286f4acecbe504571b2cfed82078bd7763b40db2fac8609af8a76e72ef5e99fb`; stored/generated totals remain owner files 485/487, TASK-492 calls 1,144/1,180, TASK-494 calls 6,962/6,987, and persistent-sink files 6/6. Review and reconcile this exact current-dev incident under ADR-029 without allowing prior TASK-3796 review or the apparent metadata shape of the new settings diagnostic to bless unrelated drift. This record moved from the later-claimed TASK-14914 to TASK-15103 during the TASK-3796 PR rebase because exact add-commit provenance established that dev's visual-compaction task claimed TASK-14914 first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Unsafe private values or exception details in the recorded delta are repaired without changing unrelated production behavior
- [ ] #2 The persistent-diagnostic inventory is regenerated with only reviewed owner changes and unchanged six-file sink topology
- [ ] #3 The focused architecture checker and regression coverage pass without constructing a test application
- [ ] #4 Every generated-versus-stored delta for the recorded 18 owner paths is reviewed under ADR-029
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed executable plan:
`Docs/superpowers/plans/2026-08-11-task-15103-diagnostic-inventory-reconciliation.md`

1. Freeze the exact 18-owner incident in a schema-validated multiset ledger
   with per-group provenance and disposition evidence.
2. Replace the shallow TASK-15103 review path with a ledger-driven guard that
   reuses and hardens the existing alias-aware diagnostic extractor and
   mutation-tests all supported message/capture forms. Task 2 explicitly owns
   the proven alias/scoping gap for aliases introduced or mutated in `try`,
   `for`, `while`, `with`, and `match`, including conservative control-flow
   joins, shadowing, reassignment, and mutation coverage.
3. Add direct real-production-function privacy sentinels and repair unsafe
   Agents, Chat, MCP, RAG, UI, and application diagnostics in reviewed batches.
4. Regenerate the production diagnostic manifest only after all source and
   ledger gates pass, then prove the boundary rejects unknown data, forged
   summaries, extra owners, classification changes, and sink changes.
5. Rebase onto the latest `dev`, compare the complete call population for all
   18 owners, rerun only touched-function tests/static gates, complete
   independent review, and close the task.

ADR required: no

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: this task enforces the existing ADR-029 privacy boundary without
changing persistent-sink ownership, storage, or the metadata policy.
<!-- SECTION:PLAN:END -->
