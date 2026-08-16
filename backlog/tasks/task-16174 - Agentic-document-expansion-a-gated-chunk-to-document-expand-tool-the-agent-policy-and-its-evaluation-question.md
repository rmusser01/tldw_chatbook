---
id: TASK-16174
title: >-
  Agentic document expansion: a gated chunk-to-document expand tool, the agent
  policy, and its evaluation question
status: Done
assignee:
  - '@claude'
created_date: '2026-08-14 02:40'
updated_date: '2026-08-15 23:40'
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

Source (repo-reachable): `Tests/RAG_Eval/README.md`'s PRF section prose and its TASK-16071 arc section — both carry these figures inline. (The session's own SDD notes were gitignored and are gone with the worktree; do not cite them.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The catalog gains one gated document-expansion tool that, given a retrieval hit, returns the surrounding or whole document text under an explicit size budget, and it is OFF by default like every other gateable builtin
- [x] #2 The tool's contract is stated in terms the agent can act on: what identity it takes (the retrieval row's source identity), what it returns for each source type, and what it does when the document is larger than the budget
- [x] #3 An agent policy is written and testable: the conditions under which expansion is worth its tokens (e.g. a high-ranked hit whose chunk is truncated or label-only) versus re-querying, with at least one test exercising each branch
- [x] #4 Media and conversation seam rows' label-only problem is addressed or explicitly scoped out: today those rows carry no document text ('Matched media - {type}'), so an expansion tool is the only way an agent sees their content
- [x] #5 The evaluation question is answered with a decision, not an assumption: either an answer-level (P3 grader) measurement is defined and run for the expansion loop, or the task records why retrieval-level scoring is sufficient and what that leaves unmeasured
- [x] #6 The relationship to the re-ranking gap is recorded: whether the expansion loop presumes a working reranker (TASK-3502, cross_encoder unimplemented) or is independent of it
- [x] #7 The inert parent-inclusion config surface is resolved, not left standing: include_parent_docs / parent_size_threshold / parent_inclusion_strategy (config.py:559-561, set by three shipped profiles) are either WIRED to the expansion work or RETIRED from config and from those profiles. A user-switchable knob that silently does nothing must not survive alongside a new tool that does the same job
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Executing the four-task plan at Docs/superpowers/plans/2026-08-15-rag-agentic-expansion.md, bound by the design spec at Docs/superpowers/specs/2026-08-15-rag-agentic-expansion-design.md.

1. Task 1 (Phase K, AC#7): retire include_parent_docs / parent_size_threshold / parent_inclusion_strategy from SearchConfig and the three profiles that set them; make RAGConfig.from_dict filter unknown search keys (warn + drop) so saved TOML survives the retirement. Gate: RAG_EVAL=1 pytest Tests/RAG_Eval/ reads 105/105 (+0.000).
2. Task 2 (Phase T, AC#1/#2/#4): new gated builtin expand_document (ExpandDocumentTool, gate key expand_document_enabled, risk-tagged -> floored to ask) with per-type fetch, budget, chunk-centred window and offset continuation; label-only media/conversation rows are first-class cases.
3. Task 3 (Phase P, AC#3): pure expand_hint helper wired into Agents/library_rag_tool_provider.py's _project_row so the policy is in the payload the agent actually sees, plus the policy sentence in the tool description.
4. Task 4 (Phase E, AC#5/#6): author + RUN an answer-level oracle check (tool-OFF vs tool-ON, fact oracles that appear only in media/conversation bodies), record the table honestly, re-run the gate, close all seven ACs on evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Linkage from TASK-16071 Task 2 (2026-08-14): AC#4's label-only premise is now MEASURED in both directions and has TRIPLED — 39/211 fed rows label-only (18%) under the pre-16071 concatenating merge, 113/211 (54%) under the rank-fair interleave, same corpus/k/M/fetch count, only the merge changed. Mechanism: media and conversation rows are precisely the rows carrying no document text, and a rank-fair rotation puts them into the top-M slots a full notes seam used to monopolise — the merge changes WHAT a top-M consumer sees, not just the order. At 54% the label-only case is dominant rather than a minority, which strengthens 'addressed' and weakens 'explicitly scoped out' as an AC#4 resolution, raises the expected call volume feeding the AC#3 policy budget, and supplies a measured per-row cost baseline (one read per fed row) for AC#3/#5. Figures are carried inline by `Tests/RAG_Eval/README.md`'s PRF section prose and its TASK-16071 arc section — cite those, not the arc's gitignored session notes.

PRODUCTION SITE (final review 2026-08-14): the label-only cost is not probe-only. `Agents/library_rag_tool_provider.py` (`:216-219` mode="rag", `:250-252` cut to _MAX_TOP_K=10) is the DEFAULT Console Library retrieval tool; under a plain profile its 10-row window went from up-to-ten text-bearing note rows to a ~4/3/3 rotation whose media/conversation rows carry label snippets only. That is this task's strongest concrete case: an agent's evidence window is where the fetch belongs.
---

## Implementation Notes — arc complete (2026-08-15)

Four phases, four commits, one follow-on: **K** (retire the inert knobs) →
**T** (the gated tool) → **P** (the policy, wired into the payload, plus
**3b** the identity that makes it actionable) → **E** (the answer-level
oracle run + closure). The gated retrieval suite read
`[rag-eval baselines] PASSED: No regression. 105 metric(s) within 0.05 of
baseline.` with **all 105 cells at (+0.000)** at both ends of the arc — no
retrieval behaviour moved, by construction: the capability is a pull-based
agent tool invoked *after* retrieval, never engine-side inclusion during it.

**The headline, measured rather than assumed.** Eight fixed questions over
the RAG-eval fixture corpus, each carrying a fact oracle grep-verified to
appear only inside one media/conversation document's **body** (never a
title — the agent sees titles — never another document, never the question
text). The real agent loop (`AgentService` → `chat_api_call` →
`api.anthropic.com`, `claude-haiku-4-5`, temperature 0) answered each
question twice, tool-OFF then tool-ON, scored by mechanical oracle
inclusion with **no LLM grader**:

| | tool-OFF | tool-ON |
|---|---|---|
| oracle-fact inclusion | **0/8** | **7/8** |

$0.177 for the run (+$0.022 smoke). The single ON miss is a **retrieval**
failure present in both arms — both of `q6`'s searches returned
`status: "empty"`, so there was never a row to expand — so conditioned on
the target row being returned the ON arm answered **7 of 7**. Four OFF
questions retrieved the label-only row and still scored zero, saying the
mechanism out loud unprompted: *"I can only see that it's a media document
and cannot access its full contents through the search results."* Full
method, isolation proof, per-question attribution and disclosed limits:
`Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/report.md`.

### The seven ACs, against evidence

- **#1 — gated tool, OFF by default.** `ExpandDocumentTool`
  (`tldw_chatbook/Tools/document_expansion_tool.py`) registered as one
  `GateableTool` row in `Agents/tool_catalog.py`'s `_GATEABLE_BUILTINS`
  behind `[tools] expand_document_enabled`. Pinned by
  `test_tool_is_gated_off_by_default`, and demonstrated live: the OFF
  arm's built-in catalog is `['calculator', 'get_current_datetime']` and
  the ON arm's is `['calculator', 'expand_document',
  'get_current_datetime']` — the tool appears because the gate flipped,
  not because the harness added it. The Settings-side switch is derived
  from the same table (`builtin_tool_gate.all_tool_gates`), so it appeared
  with zero UI wiring.
- **#2 — an actionable contract.** `execute(source_type, source_id,
  chunk_id="", chunk_start=None, offset=0, max_chars=None, note_id=…,
  media_id=…, doc_id=…, **provenance)` returning the SAME keys on every
  branch — `status | source_type | source_id | title | text | total_size |
  window{start,end} | truncated | next_offset` — for note / media /
  conversation / prompt and for `not_found` / `unsupported` / `error`. Over
  budget it returns a window (chunk-centred when `chunk_start` is known,
  else the head) plus `next_offset`, so a long document is navigable
  without re-querying. Eleven contract tests, RED-first.
- **#3 — a testable policy, in the payload the agent actually reads.**
  `Library/library_expand_policy.expand_hint` is pure and returns
  `{expandable, reason}` ∈ `label_only` / `truncated_snippet` /
  `text_bearing`, or `None` for a row with nothing to expand;
  `Agents/library_rag_tool_provider._project_row` attaches it. The
  decision rule also sits in the tool's own `description`, pinned verbatim
  by `test_tool_description_states_the_policy_verbatim`. 16 policy tests +
  39 provider tests cover each branch, the sealing loop's byte budget, and
  hostile rows.
- **#4 — the label-only problem addressed, not scoped out.** Media and
  conversation rows are the tool's first-class acceptance cases, not
  afterthoughts: the contract works from exactly what such a row carries
  (`source_type` + `source_id`, empty `chunk_id`). Closed in the field —
  every one of the 7 successful live expansions was a label-only row, and
  the 54%-label-only window this task was filed against is now openable.
- **#5 — the evaluation question answered with a run.** The oracle run
  above, committed whole (`questions.toml`, `oracle_run.py`, `report.md`,
  `run-artifacts.json`). Retrieval-level scoring was NOT sufficient and
  the task does not claim it was. Disclosed limits: N=8, one corpus, one
  model, one run per arm; oracle inclusion ≠ answer quality; one route.
- **#6 — the recorded sentence on reranking.** *The expansion tool is
  INDEPENDENT of reranking: it consumes final retrieval rows whatever
  produced them, and presumes nothing about TASK-3502's unimplemented
  `cross_encoder`.* It reads `source_type`/`source_id` off a row that any
  ordering could have produced, so a reranker landing later changes which
  rows are offered, never whether they can be opened. Carried in the
  tool's module docstring so it cannot drift from the code.
- **#7 — the inert surface retired, not left standing.**
  `include_parent_docs` / `parent_size_threshold` /
  `parent_inclusion_strategy` deleted from `SearchConfig` and from all
  three shipped profiles (nine writes). Grep-verified zero occurrences in
  `tldw_chatbook/` beyond comments recording the retirement, and only the
  compat test in `Tests/`. `RAGConfig.from_dict` now filters unknown
  `[search]` keys with a logged notice, so a user's saved TOML still
  carrying them loads instead of raising `TypeError` — the compat hazard
  verified at plan time.

### Decisions and trade-offs

- **Retire, not wire (AC#7).** Pre-registered before any code: the knobs
  get wired only if the capability lands engine-side. It did not — the
  capability is pull-based, budgeted, gated and per-hit — and engine-side
  inclusion would move retrieval, which the 105/105 gate forbids. Wiring
  them here was out of scope by construction.
- **A module constant, not a new `[tools]` key, for the budget.**
  `DEFAULT_MAX_CHARS = 8000` / `HARD_MAX_CHARS = 32000`. A knob whose only
  consumer is a default is exactly the dead surface Phase K just retired.
- **`chunk_id` is an INDEX, not an offset** (`f"{doc_id}_chunk_{i}"`), so
  the window anchor is `chunk_start` — a real chunk-metadata key. No
  index→offset guessing path exists, deliberately.
- **Identity rides the hint's own precondition** (Task 3b). A row with
  nothing to expand carries neither hint nor identity, so verdict and
  identity cannot drift apart and no raw id leaks on an unsupported row.
  Cost: +97.5 B/row cumulative over the pre-arc payload (2585 → 3560 B on
  a normal 10-row payload, 1.22% of the 32 KiB ceiling). Identity is never
  truncated by the sealing loop — a halved id is a wrong id, not a smaller
  one.
- **The friction is disclosed, not designed around.** `risk_tags =
  ("reads",)` floors the tool to *ask*, so each call raises an approval
  card until the user sets Allow. The live run recorded exactly **7
  approval rounds in the ON arm and 0 in the OFF arm** — the gate is real,
  and a user will feel it.
- **The oracle run measures the `plain` route.** That is the route that
  emits label-only rows; under `hybrid` the engine returns chunk text and
  there is nothing label-only to expand. Pre-registered in
  `questions.toml`, not chosen after seeing a number.

### Two suspects checked and NOT closed

Task 3b pre-registered two ways expansion could misfire. **Neither
manifested in this run, and neither is refuted by it** — the `plain`
route emits no chunked rows and real database ids, so it cannot exercise
either. They remain the first things to check if the tool is measured on a
semantic/hybrid route, and are carried by a follow-up task:

1. `chunk_start` is absent from the projected payload, so a chunked row
   would expand from the document HEAD rather than around the match.
2. A semantic `source_id` can be a vector-store point id whose
   `note_id`/`doc_id` fallbacks are likewise absent from the payload, so a
   declared-expandable row could return `not_found`.

### Files

Added: `tldw_chatbook/Tools/document_expansion_tool.py`;
`tldw_chatbook/Library/library_expand_policy.py`;
`Tests/Tools/test_document_expansion_tool.py`;
`Tests/Library/test_library_expand_policy.py`;
`Tests/RAG_Search/test_search_config_compat.py`;
`Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/` (`questions.toml`,
`oracle_run.py`, `report.md`, `run-artifacts.json`).

Modified: `tldw_chatbook/RAG_Search/simplified/config.py`;
`tldw_chatbook/RAG_Search/config_profiles.py`;
`tldw_chatbook/Agents/tool_catalog.py`;
`tldw_chatbook/Agents/library_rag_tool_provider.py`;
`tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py` (a `_TOOL_COPY` row —
without it first-run setup ships a blank row for the new tool);
`Tests/Agents/test_builtin_tool_gate.py`,
`Tests/Agents/test_library_tool_provider.py`,
`Tests/UI/test_mcp_workbench.py` (four hard-coded arity literals now
derived from `len(_GATEABLE_BUILTINS)`);
`tldw_chatbook/Agents/builtin_tool_gate.py` and
`tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py` (docstrings that hard-coded
"the 7 rows"/"all nine gates" — the counts are now derived, so the eighth
built-in did not falsify them); `Tests/RAG_Eval/README.md`;
`Docs/Development/CHUNKING-IMPLEMENTATION-COMPLETE.md`,
`-SUMMARY.md`, `CHUNKING-IMPROVE-1.md` (each claimed this feature was
implemented; each now carries a correction).

### Verification

- Gate: `[rag-eval baselines] PASSED: No regression. 105 metric(s) within
  0.05 of baseline.`, 105/105 cells `(+0.000)`; `Tests/RAG_Eval` 307
  passed.
- Batteries: `Tests/Library Tests/Tools Tests/Agents Tests/RAG_Search` =
  **4313 passed, 29 skipped, 0 failed** (skips are pre-existing
  optional-dependency guards).
- Collection sweep over every test file this branch touched vs merge-base
  `8727a2861`: 6 files, 348 tests collected, 0 errors.
- Ruff clean on every touched file.
<!-- SECTION:NOTES:END -->
