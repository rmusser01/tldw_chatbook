---
id: TASK-35700
title: >-
  Library RAG: every hybrid search raises TypeError on query_embedding since
  859c76f104
status: To Do
assignee: []
created_date: '2026-10-20 01:47'
updated_date: '2026-10-22 18:22'
labels:
  - library
  - rag
  - console
  - p1
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since dev 859c76f104 (2026-10-11, tasks 34389-34394), EnhancedRAGServiceV2.search forwards query_embedding= to RAGService.search, whose signature has no such parameter. The activation guard's signature.bind_partial then raises TypeError: got an unexpected keyword argument 'query_embedding'.

Impact: with the default hybrid_basic profile, every Library RAG search through the shared service fails. Console Automatic Library pauses every send with 'Library search failed; your message was not sent' (chip 'Blocked — Library error'), and Retry fails the same way. The Library screen's own RAG search is likely affected too (not checked).

Found live during TASK-33621.20 on 2026-10-19 (head rebased onto dev ce4aa4a231). A probe on automatic_search_outcome logged the TypeError raised through RAG_Search/generation.py:409 (invoke) and RAG_Search/activation.py:234 (call, inspect.bind_partial). The send-harness tests use Library doubles, so nothing caught it. Workaround seen live: a keyword profile ([rag.service] profile = "bm25_only") avoids the hybrid engine call.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A hybrid Library RAG search through the real EnhancedRAGServiceV2 returns results or an empty outcome instead of raising
- [ ] #2 A caller's precomputed query embedding is either used by the base search or not forwarded; it is never an unexpected keyword argument
- [ ] #3 A test drives the real service class (not a double) through a hybrid search and fails on dev ce4aa4a231
- [ ] #4 Live: a Console Automatic Library send on the default profile with an empty Library reaches the provider
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-10-22 (TASK-33621.20 rebase onto dev 7700c63a68): dev appears to have fixed this under TASK-35992. 9206d1d395 stops EnhancedRAGServiceV2.search forwarding query_embedding to RAGService.search (a non-None value now raises ValueError), and adds Tests/Library/test_enhanced_rag_search_contract.py, which drives the real EnhancedRAGServiceV2 through semantic, hybrid and keyword searches (AC#1-#3; not yet checked to fail on ce4aa4a231). Live on the rebased TASK-33621.20 head with the default hybrid_basic profile and an empty Library, a Console Automatic send logged 'Performing hybrid search ... Search completed ... found 0 results' and reached the provider, which replied (AC#4; evidence uxrev/evidence/bd5/L66/app-round1.log). Left To Do for the owner to confirm and close.
<!-- SECTION:NOTES:END -->
