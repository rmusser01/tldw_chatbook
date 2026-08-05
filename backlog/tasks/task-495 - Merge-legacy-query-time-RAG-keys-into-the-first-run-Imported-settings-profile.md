---
id: TASK-495
title: >-
  Merge legacy query-time RAG keys into the first-run "Imported settings"
  profile
status: Done
assignee:
  - '@claude'
created_date: '2026-07-23 02:00'
updated_date: '2026-07-25 14:35'
labels:
  - rag
  - profiles
  - followup
dependencies:
  - task-502
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from SP2b (task-487, PR for config-resolution unification) final review. `ensure_imported_profile` snapshots the active-profile resolution (built-in `hybrid_basic` + env on a true first run), which is correct for the SP1 index-fingerprint invariant (pre-SP2b ingestion built the collection from the built-in profile, so SP1 adopted under the built-in fingerprint). But it does NOT capture a user's hand-tuned NON-fingerprint-affecting query-time legacy keys (`[AppRAGSearchConfig.rag.search].default_top_k`, `score_threshold`, `include_citations`, reranking settings). Those are silently discarded on import.

Enrich the first-run snapshot: merge such hand-set query-time legacy keys onto the built-in base (they do not affect the SP1 fingerprint, so the index invariant is unaffected — only merge NON-index-determining fields to keep the fingerprint equal to SP1's adoption). Do NOT merge embedding-model / chunk fields from legacy keys (that would change the fingerprint and orphan the legacy collection).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The "Imported settings" first-run profile preserves a user's hand-set query-time legacy keys (top_k, score_threshold, citations, reranking) from `[AppRAGSearchConfig.rag.*]`.
- [x] #2 The imported profile's SP1 fingerprint still equals SP1's adopted legacy-collection fingerprint (only non-index-determining fields merged) — a test asserts this holds with legacy query-keys set.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a private helper in active_config.py that reads the raw [AppRAGSearchConfig.rag.search] / [AppRAGSearchConfig.rag.processor] dicts via get_cli_setting and returns only the KEYS LITERALLY PRESENT (not defaulted) among the allow-listed query-time fields: search.default_top_k, search.score_threshold, search.include_citations, processor.enable_reranking, processor.reranker_model, processor.reranker_top_k.
2. In ensure_imported_profile(), after snapshot = resolve_active_rag_config(), merge those hand-set values onto snapshot.search in place, before wrapping snapshot in the ProfileConfig. Never touch embedding/chunking/vector_store.distance_metric fields (fingerprint-determining, per collection_fingerprint._index_fields) -- confirmed exhaustively against that module.
3. TDD in Tests/RAG/test_first_run_import.py (already covers ensure_imported_profile): RED tests for (a) hand-set legacy keys preserved in the imported profile, (b) fingerprint invariance with those same keys set (mirrors existing test_imported_fingerprint_matches_sp1_adoption pattern), (c) no legacy keys set -> import unchanged vs today, (d) legacy embedding/chunk keys set -> NOT merged, fingerprint still equal.
4. Implement, get GREEN, run full Tests/RAG/ suite for regressions.
5. Update task-495 ACs + Implementation Notes, set Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `_hand_set_legacy_query_time_keys()` / `_merge_legacy_query_time_keys()` to
`tldw_chatbook/RAG_Search/simplified/active_config.py` and call the latter from
`ensure_imported_profile()` right after `snapshot = resolve_active_rag_config()`,
before the snapshot is wrapped in the new `ProfileConfig`.

`_hand_set_legacy_query_time_keys()` reads the raw
`get_cli_setting("AppRAGSearchConfig", "rag", {})` dict directly and checks
literal key **presence** (`key in search`/`key in processor`), not "differs
from default" — an explicit value equal to the dataclass default is still
honored, and nothing is synthesized for an absent key. Only a fixed
allow-list is ever read: `_LEGACY_SEARCH_KEYS = (default_top_k,
score_threshold, include_citations)` from `[AppRAGSearchConfig.rag.search]`
and `_LEGACY_PROCESSOR_KEYS = (enable_reranking, reranker_model,
reranker_top_k)` from `[AppRAGSearchConfig.rag.processor]` (per the TOML
mapping documented in `simplified/config.py`'s `EXAMPLE_TOML_CONFIG`).
Embedding/chunking/`vector_store.distance_metric` legacy keys are never read
by this path at all — confirmed against `collection_fingerprint.
_index_fields`' exhaustive list of fingerprint-input fields, so this merge
can never move `fingerprint_collection()`'s output.

TDD: extended `Tests/RAG/test_first_run_import.py` (the existing
`ensure_imported_profile` test file) with 4 tests — preserved-hand-set-keys
(RED before the change), fingerprint-invariance-with-those-keys-set (AC #2,
already held pre-change since nothing was merged yet, kept as a regression
guard), unchanged-with-no-legacy-keys (byte/asdict-equal to today's
behavior), and legacy-index-determining-keys-NOT-merged (RED before the
change: `default_top_k` wasn't merged; embedding/chunk/distance_metric
assertions passed throughout since that path never touches them). All 12
tests in the file pass; full `Tests/RAG/` suite: 541 passed, 8 skipped (was
~537 passed, 8 skipped before this change — net +4 new tests, no
regressions).

Files modified:
- `tldw_chatbook/RAG_Search/simplified/active_config.py`
- `Tests/RAG/test_first_run_import.py`
<!-- SECTION:NOTES:END -->
