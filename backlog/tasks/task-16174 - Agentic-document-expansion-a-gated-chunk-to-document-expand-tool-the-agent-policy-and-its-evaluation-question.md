---
id: TASK-16174
title: >-
  Agentic document expansion: a gated chunk-to-document expand tool, the agent
  policy, and its evaluation question
status: To Do
assignee: []
created_date: '2026-08-14 02:40'
labels:
  - rag
  - agents
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The owner asked (2026-08-14) whether the retrieval stack can go BM25 + vector -> re-rank -> an agent that searches over whole documents. This task records the honest state of each layer and scopes the missing one.

State of the three layers, as measured by the RAG server-port programme:

- Hybrid (BM25 + vector) SHIPS and is MEASURED. The engine's keyword leg is rank-fair and tiered (TASK-15700); fusion weighting landed in the fusion/weighting arcs; the golden-set instrument reports it per cell.
- Re-ranking is CONSTRUCTED BUT UNMEASURED. A reranking stage exists in config and profiles, but 'cross_encoder' is explicitly NOT an implemented strategy (RAG_Search/config_profiles.py:352-356 says so in a comment), and TASK-3502 (reranker follow-ups: provider/model selection, cost surface, re-review residuals) is still open. No golden-set cell isolates reranking's contribution today.
- The agentic layer is UNBUILT. RAGSearchTool (Tools/rag_search_tool.py:13) is agent-callable, so an agent can already issue retrieval queries. The chunk-to-document linkage also exists: retrieval rows carry a source id and the full text is reachable (DB/Client_Media_DB_v2.py get_media_by_id family; the PRF probe fetched documents this way). What does NOT exist is any tool that lets an agent EXPAND a hit into its document or NAVIGATE within one: there is no expand/read-document tool in the catalog, and sibling/parent-chunk inclusion was deferred at P3. So an agent can retrieve, but it cannot follow a promising chunk into the rest of the document except by issuing more blind queries.

The capability the owner described is therefore one missing tool plus a policy for when to use it, plus an evaluation question the current instrument cannot answer.

Evaluation is the part most likely to be underestimated. The golden set scores RETRIEVAL (does the target document reach top-k), and an agentic expansion loop's whole value is at the ANSWER level: did following the chunk into the document produce a better answer for the same or less spend. That is P3 grader territory - a different instrument, not a new cell in this one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The catalog gains one gated document-expansion tool that, given a retrieval hit, returns the surrounding or whole document text under an explicit size budget, and it is OFF by default like every other gateable builtin
- [ ] #2 The tool's contract is stated in terms the agent can act on: what identity it takes (the retrieval row's source identity), what it returns for each source type, and what it does when the document is larger than the budget
- [ ] #3 An agent policy is written and testable: the conditions under which expansion is worth its tokens (e.g. a high-ranked hit whose chunk is truncated or label-only) versus re-querying, with at least one test exercising each branch
- [ ] #4 Media and conversation seam rows' label-only problem is addressed or explicitly scoped out: today those rows carry no document text ('Matched media - {type}'), so an expansion tool is the only way an agent sees their content
- [ ] #5 The evaluation question is answered with a decision, not an assumption: either an answer-level (P3 grader) measurement is defined and run for the expansion loop, or the task records why retrieval-level scoring is sufficient and what that leaves unmeasured
- [ ] #6 The relationship to the re-ranking gap is recorded: whether the expansion loop presumes a working reranker (TASK-3502, cross_encoder unimplemented) or is independent of it
<!-- AC:END -->
