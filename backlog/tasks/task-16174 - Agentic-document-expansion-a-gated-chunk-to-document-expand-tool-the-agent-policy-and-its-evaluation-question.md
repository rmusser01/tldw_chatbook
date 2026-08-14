---
id: TASK-16174
title: >-
  Agentic document expansion: a gated chunk-to-document expand tool, the agent
  policy, and its evaluation question
status: To Do
assignee: []
created_date: '2026-08-14 02:40'
updated_date: '2026-08-14 06:29'
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
- The agentic layer is UNBUILT. RAGSearchTool (Tools/rag_search_tool.py:13) is agent-callable, so an agent can already issue retrieval queries. The chunk-to-document linkage also exists: retrieval rows carry a source id and the full text is reachable (DB/Client_Media_DB_v2.py get_media_by_id family; the PRF probe fetched documents this way). What does NOT exist is any tool that lets an agent EXPAND a hit into its document or NAVIGATE within one: there is no expand/read-document tool in the catalog. So an agent can retrieve, but it cannot follow a promising chunk into the rest of the document except by issuing more blind queries.

**Sibling/parent inclusion is worse than deferred: it is an INERT, USER-REACHABLE SURFACE.** (Corrected 2026-08-14 at review; an earlier draft of this filing placed it at P3 and called it deferred. Both halves were wrong.)

- WRONG PHASE. The programme's own phasing puts "granularity router, sibling/parent inclusion" under **P2 Retrieval intelligence** (Docs/superpowers/specs/2026-08-07-rag-port-p0-foundations-design.md:37). P3 is Answer trust (graders, citations, faithfulness). Only the EVALUATION question below is P3-grader territory; the inclusion feature is not, and the two must not be conflated again.
- WRONG STATE. The knobs are not merely unimplemented -- they are shipped, switchable and dead. `SearchConfig.include_parent_docs` / `parent_size_threshold` / `parent_inclusion_strategy` exist (RAG_Search/simplified/config.py:559-561) and are set to True / "size_based" by THREE shipped profiles: hybrid_enhanced_rag (config_profiles.py:310-312), hybrid_full_rag (:347-349) and research_rag (:524-526). Nothing in tldw_chatbook/ READS any of the three (grep-verified: the only occurrences are the definition and those three profile assignments). A user who selects one of those profiles has parent inclusion switched on, and it silently does nothing.

That is a notch worse than the reranker's "constructed but unmeasured": this one is not even wired. It bears directly on AC#1/#2 below, because a new expansion tool would otherwise become a THIRD overlapping surface (the inert config knobs, the profiles that set them, and the tool) for one capability.

The capability the owner described is therefore one missing tool plus a policy for when to use it, plus an evaluation question the current instrument cannot answer -- and a decision about the inert knobs that must not be postponed a second time.

Evaluation is the part most likely to be underestimated. The golden set scores RETRIEVAL (does the target document reach top-k), and an agentic expansion loop's whole value is at the ANSWER level: did following the chunk into the document produce a better answer for the same or less spend. That is P3 grader territory - a different instrument, not a new cell in this one.

**MEASURED UPDATE 2026-08-14 (TASK-16071 Task 2): AC#4's premise TRIPLED, and it is now measured in both directions.** AC#4 was filed against the label-only problem as a known but unquantified hazard. TASK-16071 replaced the plain four-seam path's fixed-order concatenation with a rank-fair interleave, and the PRF probe's price line measured the same top-M window before and after:

- **BEFORE (concatenating merge): 39 of 211 fed rows label-only (18%).**
- **AFTER (rank-fair merge): 113 of 211 label-only (54%).**

Same corpus, same k, same M, same 211 fetches (one read per fed row) - only the merge changed. The mechanism is the point: media and conversation rows are exactly the rows that carry no document text ('Matched media - {type}', 'Matched conversation - N messages'), and a rank-fair rotation puts them into the top-M slots that a full notes seam used to monopolise. **The merge changes WHAT a top-M consumer sees, not merely the order it sees it in.** Any consumer of this path - RAG Answer evidence, a PRF-style feedback loop, a future re-ranker, and the expansion tool this task proposes - now receives a window in which the majority of rows are self-describing labels rather than content.

Consequences for this task, stated so AC#4 is not evaluated against the old 18%:

- The label-only share is no longer a minority case to be handled defensively; at 54% it is the dominant case, which strengthens 'addressed' and weakens 'explicitly scoped out' as an acceptable resolution of AC#4.
- It raises the value of the tool in AC#1/#2 (without expansion, a majority of what an agent is shown is a label) and simultaneously raises its expected call volume, which is a cost input to the AC#3 policy - expanding every label-only row is a different budget from expanding the occasional one.
- The 'one read per fed row' price the PRF probe paid is the same read an expansion tool would pay. That probe's numbers are a usable cost baseline for AC#3/#5, measured on this corpus.

Source: `.superpowers/sdd/2026-08-14-rag-four-seam-cross-ranking/task-2-report.md` and the staleness marker in `Tests/RAG_Eval/README.md`'s PRF section.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The catalog gains one gated document-expansion tool that, given a retrieval hit, returns the surrounding or whole document text under an explicit size budget, and it is OFF by default like every other gateable builtin
- [ ] #2 The tool's contract is stated in terms the agent can act on: what identity it takes (the retrieval row's source identity), what it returns for each source type, and what it does when the document is larger than the budget
- [ ] #3 An agent policy is written and testable: the conditions under which expansion is worth its tokens (e.g. a high-ranked hit whose chunk is truncated or label-only) versus re-querying, with at least one test exercising each branch
- [ ] #4 Media and conversation seam rows' label-only problem is addressed or explicitly scoped out: today those rows carry no document text ('Matched media - {type}'), so an expansion tool is the only way an agent sees their content
- [ ] #5 The evaluation question is answered with a decision, not an assumption: either an answer-level (P3 grader) measurement is defined and run for the expansion loop, or the task records why retrieval-level scoring is sufficient and what that leaves unmeasured
- [ ] #6 The relationship to the re-ranking gap is recorded: whether the expansion loop presumes a working reranker (TASK-3502, cross_encoder unimplemented) or is independent of it
- [ ] #7 The inert parent-inclusion config surface is resolved, not left standing: include_parent_docs / parent_size_threshold / parent_inclusion_strategy (config.py:559-561, set by three shipped profiles) are either WIRED to the expansion work or RETIRED from config and from those profiles. A user-switchable knob that silently does nothing must not survive alongside a new tool that does the same job
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Linkage from TASK-16071 Task 2 (2026-08-14): AC#4's label-only premise is now MEASURED in both directions and has TRIPLED — 39/211 fed rows label-only (18%) under the pre-16071 concatenating merge, 113/211 (54%) under the rank-fair interleave, same corpus/k/M/fetch count, only the merge changed. Mechanism: media and conversation rows are precisely the rows carrying no document text, and a rank-fair rotation puts them into the top-M slots a full notes seam used to monopolise — the merge changes WHAT a top-M consumer sees, not just the order. At 54% the label-only case is dominant rather than a minority, which strengthens 'addressed' and weakens 'explicitly scoped out' as an AC#4 resolution, raises the expected call volume feeding the AC#3 policy budget, and supplies a measured per-row cost baseline (one read per fed row) for AC#3/#5. Full measurement in .superpowers/sdd/2026-08-14-rag-four-seam-cross-ranking/task-2-report.md; Tests/RAG_Eval/README.md's PRF section carries a staleness marker with the same figures.
<!-- SECTION:NOTES:END -->
