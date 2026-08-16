---
id: TASK-3502
title: >-
  Reranker follow-ups: provider/model selection, cost surface, and two re-review
  residuals
status: Done
assignee: []
created_date: '2026-08-07 20:36'
updated_date: '2026-08-16 23:11'
labels:
  - rag
  - settings
dependencies:
  - TASK-3170
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-3170's Task 4 fixed the reranker factory so a reranking-enabled profile actually constructs and runs a reranker (it had never worked before -- a double-strategy TypeError meant reranking silently never activated in production). That fix surfaced follow-on gaps that were explicitly left out of Task 4's scope: Settings ▸ RAG's 'Enable reranking' toggle creates a bare RerankingConfig that defaults to provider=openai, model=gpt-3.5-turbo with no way to pick a different provider/model and no cost disclosure before enabling it, even though a single search can now issue up to 15 provider calls for LLM-driven reranking (pointwise scores each candidate individually). Two smaller, already-diagnosed residuals from Task 4's re-review rounds are folded in here rather than filed separately: (a) the reranking_degraded disclosure tag's cache-safety fix (copy-not-mutate) has no dedicated test coverage for the Pairwise/Listwise strategies, where the copy semantics actually matter differently than Pointwise; (b) BaseReranker's last_rerank_failures/last_rerank_total counters are instance state on a shared singleton reranker and are racy under concurrent search() calls -- diagnostic-only corruption (the disclosed count could be wrong), not a correctness bug in the reranked results themselves.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings ▸ RAG's Reranking fold lets the user choose the reranking provider and model, not just a bare on/off toggle defaulting to openai/gpt-3.5-turbo
- [x] #2 Enabling reranking discloses, before the user commits, that reranking issues one provider call per candidate result (up to Rerank results many) and therefore has a real per-search cost
- [x] #3 A regression test drives the real PairwiseReranker and ListwiseReranker strategies through the reranking_degraded copy-not-mutate path and confirms neither poisons a cached SearchResult
- [x] #4 BaseReranker's per-call failure counters are safe under concurrent search() calls on the shared reranker singleton, or the diagnostic disclosure is scoped so a race cannot misattribute one search's failures to another's disclosed tag
- [x] #5 (note-a) The reranking_skipped/reranking_degraded disclosure tags have a real UI consumer: a user whose reranking silently skipped or degraded can see that it did
- [x] #6 (note-b) A row no model rescored does not render "| reranked" -- the row-level claim matches what actually happened to that row
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-16-reranker-followups-design.md
Plan: Docs/superpowers/plans/2026-08-16-reranker-followups.md
T1 engine honesty (AC#3, AC#4, note-b) -> T2 UI honesty (AC#1, AC#2, note-a) -> T3 closure + cross_encoder filing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made the now-real reranker honest: it has controls, it discloses its cost
before you commit, it says so when it degrades, and it no longer claims to
have rescored rows it did not. Two commits on
`feat/rag-3502-reranker-followups` off dev `2b1d1817f`, spec
`Docs/superpowers/specs/2026-08-16-reranker-followups-design.md`, plan
`Docs/superpowers/plans/2026-08-16-reranker-followups.md`.

**T1 -- engine honesty (`f44bf0c33`): AC#3, AC#4, note-(b).**

- AC#4, the counter race, fixed by SCOPING not locking (the owner's ruling
  against clever-unstable fixes on a diagnostic path). `rerank()` now returns
  a frozen `RerankOutcome(results, failed, total)` with a `.degraded`
  property -- one entry point, no list-returning variant alongside it, so no
  caller can take the results while dropping the counts. `total` is per-result
  for pointwise/listwise and per-COMPARISON for pairwise. The instance
  attributes `last_rerank_failures`/`last_rerank_total`, `_record_rerank_
  outcome`, AND Pairwise's own `_pairwise_comparisons_failed/_total` (the same
  race one level down) are all removed; pairwise threads a per-call
  `_ComparisonTally` down the merge-sort instead. Exactly ONE production
  consumer existed -- `enhanced_rag_service_v2.py:322-323` -- and it now reads
  the returned counts; the disclosure string is unchanged, so the existing
  pinned tag text still holds. Pinned by an interfering-write test landing the
  concurrent write exactly where a competing `rerank()` would land it, plus a
  structural pin that no per-call state survives on the instance.
- AC#3 produced a **finding worth more than the fix it did not need**: neither
  `PairwiseReranker` nor `ListwiseReranker` MUTATES -- but neither COPIES
  either. They only reorder; the rows handed back are the caller's (the
  cache's) own objects, and on Listwise's two failure paths the returned list
  IS the caller's list object. So the entire no-mutation contract rests on
  `_tag_first_result`'s copy at the disclosure site, and the strategies
  contribute nothing to it. Closed by coverage rather than a production
  change, with the sharper fact named; teeth proven by restoring the old
  in-place `results[0].metadata[key] = value`, which turns all four AC#3 tests
  red.
- note-(b): `RerankingResult` gains `scored: bool`, set `False` at both
  failure constructors; `_apply_scores` gives an unscored row its ORIGINAL
  score and a metadata copy WITHOUT the `rerank_score` stamp, so a failed row
  resolves to `vector_similarity` instead of rendering `" | reranked"`. No
  fabricated scores -- a partly degraded search honestly mixes kinds row by
  row. The flag rides the reranker's result cache, so a cache HIT re-applies
  the same partial-failure honesty. Contract comment at
  `Library/library_rag_score_kinds.py` updated (the stamp is PER ROW, not per
  rerank call) and the "what a REAL reranked row carries" pin extended with
  its converse.

**T2 -- UI honesty (`4594c0a8c`): AC#1, AC#2, note-(a).**

- AC#1: `reranker_provider` field -> adapter read/write (blank-means-default,
  mirroring `reranker_model`) -> a fold `Select` whose options are ENUMERATED
  from `Chat_Functions.API_CALL_HANDLERS` (set-equality asserted twice, never
  hand-listed). Two design calls: the `"openai (default)"` row carries the
  provider NAME rather than a blank sentinel (a blank would make
  anthropic->openai a silent no-op), and **the handler compares the EFFECTIVE
  provider before staging** -- without that, merely MOUNTING the category
  dirties the draft, because the Select resolves a blank loaded value to
  `"openai"` and posts `Select.Changed` at mount. Caught by a written-to-fail
  test; the task-15740 family again (an app's own rewrite staged as a user
  edit).
- AC#2: `Static#settings-library-rag-reranker-cost-disclosure` under the
  toggle, visible with reranking OFF, never a tooltip: "Reranking scores each
  result with a separate {provider} call - up to {top_k} calls per search,
  billed at that provider's rates." Not static text -- re-rendered from the
  staged draft, and `_sync_library_rag_widgets` passes its OWN values, so a
  profile PREVIEW discloses the browsed profile's cost rather than the active
  one's. No pricing estimator, per spec.
- note-(a): the tag SURVIVES end to end with no plumbing added --
  `_tag_first_result` -> metadata -> `_semantic_row` copies the whole metadata
  block into `provenance` (semantic AND hybrid legs) -> `LibraryRagResultRow.
  from_result` copies provenance wholesale; pinned parametrized over both
  tags. The consumer `library_rag_reranking_notice(rows)` is appended LAST
  into the EXISTING coverage-note channel (one quiet line, not a second
  competing one) and scans EVERY row, because the engine tags position 0 of
  ITS list while scope filters run afterwards. Detail is collapsed, clamped to
  120 chars and markup-escaped.

**Trade-offs and fences.** No live provider call anywhere: every reranker test
fakes `chat_api_call` and plants a fake key so the real path executes instead
of short-circuiting on the credential check. No `cross_encoder`, no
profile-default changes. The gate held at `PASSED: No regression. 105
metric(s) within 0.05 of baseline.` in T1 and again at close-out -- no cell
moved, despite the stamping change touching the semantic path.

**Filed at close-out (full ID-safety sweep; true max 16865 across 152 remote
refs + 132 worktrees, leapfrogged):** TASK-16965 implement-or-retire
`cross_encoder` with a gated-instrument measurement (the quality question this
arc could not answer -- an LLM reranker is unmeasurable on a local
deterministic instrument, a local cross-encoder is not; retire pre-registered
as an acceptable outcome), and TASK-17065 the provider-coverage gap the new
picker exposed (the Select offers 29 providers; `BaseReranker._call_llm_impl`,
`reranker.py:187-206`, hand-rolls credentials for four and never touches
`resolve_provider_api_key` -- note-(a)'s notice currently only DISCLOSES that
failure).

**Files.** Production: `RAG_Search/reranker.py`, `RAG_Search/simplified/
enhanced_rag_service_v2.py`, `Library/library_rag_score_kinds.py`,
`Library/library_rag_state.py`, `UI/Screens/settings_library_rag_defaults.py`,
`UI/Screens/settings_rag_profile_adapter.py`, `UI/Screens/settings_screen.py`
(the note-(a) consumer lives in `library_rag_state.py`'s existing
coverage-note channel, so the panel widget itself needed no change). Tests:
`Tests/RAG_Search/test_reranker_degraded_paths.py` (new, 11 tests),
`Tests/RAG_Search/test_reranker_construction.py`,
`Tests/UI/test_settings_rag_profile_adapter.py`,
`Tests/UI/test_settings_rag_profile_region.py`,
`Tests/UI/test_settings_library_rag_defaults.py`,
`Tests/UI/test_product_maturity_gate16_library_search_rag.py`,
`Tests/Library/test_library_rag_state.py`,
`Tests/Library/test_library_local_rag_search_service.py`. Docs:
`Docs/User_Guide/settings/rag.md`, `Docs/User_Guide/library/search-and-rag.md`
(stamps state verification was by mounted-widget/state tests, NOT a live TUI
walkthrough), plus the spec and plan under `Docs/superpowers/`.

---

*Original final-review scope note (2026-08-07), both items now closed above:*
(a) the reranking_skipped/reranking_degraded disclosure tags had ZERO UI
consumers -- disclosure was metadata/log-only, so a Hybrid Full user with a
dead reranker credential saw normal-looking results; (b) partial pointwise
failure stamped rerank_score = original_score on failed rows, so a 14/15-failed
rerank still rendered " | reranked" on rows never actually rescored --
conservative in direction (no fabricated score), but an over-claim about what
happened to those rows.
<!-- SECTION:NOTES:END -->
