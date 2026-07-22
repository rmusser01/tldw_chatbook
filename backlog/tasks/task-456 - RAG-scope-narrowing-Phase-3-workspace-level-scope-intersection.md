---
id: TASK-456
title: 'RAG scope narrowing Phase 3: workspace-level scope + intersection'
status: In Progress
assignee: []
created_date: '2026-07-22 02:09'
labels:
  - rag
  - scope
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 3 (final) of the RAG scope narrowing program (spec Docs/superpowers/specs/2026-07-21-rag-scope-narrowing-design.md §3 layering; plan Tasks 12-14). Adds workspace-level scope on top of Phases 1-2: a workspace defines an in-scope document set that bounds all retrieval for conversations inside it (the 'hunt X' / 'sales reports' workflow). Storage in the workspace registry DB (Workspace_DB) with FK cascade; the conversation∩workspace intersection is enforced end-to-end; the conversation picker's universe is restricted to the workspace's in-scope items (D3); the header chip tooltip shows the 'conversation A ∩ workspace B → N' breakdown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A workspace scope can be set/cleared via a Console workspace-area entry point (stored in Workspace_DB, dropped on workspace delete)
- [ ] #2 Retrieval for a conversation inside a scoped workspace is bounded by the conv∩workspace intersection on all paths; no-overlap short-circuits to EMPTY with an honest notice
- [ ] #3 The conversation picker inside a scoped workspace offers only workspace-in-scope items (D3)
- [ ] #4 The chip/row reflect the effective (post-intersection) state with an intersection tooltip; zero DB on compose/recompose
- [ ] #5 QA captures reviewed and approved by owner before merge
<!-- AC:END -->
