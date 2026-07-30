---
id: TASK-1451
title: >-
  Gate the RAG_Search autouse HF-model warm-up fixture and default HuggingFace to offline in tests
status: In Progress
assignee: []
created_date: '2026-07-30 09:05'
labels:
  - testing
  - performance
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/RAG_Search/conftest.py` carried a session-scoped **autouse** fixture that, whenever transformers is installed (CI installs it for every job), ran `transformers.utils.move_cache()` and downloaded/loaded `hf-internal-testing/tiny-bert` — network-touching work paid by every session that touches the directory, including fully-mocked ones, and multiplied per worker once pytest-xdist lands. The conftest also force-set `TRANSFORMERS_OFFLINE=0` at import, making test runs network-dependent by default. Found by the 2026-07-30 test-suite audit (driver #10).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] The transformers warm-up fixture no longer runs for sessions that only use mocked embeddings; suites that load real models request it explicitly
- [ ] HuggingFace env defaults to offline in tests; enabling `TLDW_RUN_REAL_EMBEDDINGS` (or `TLDW_TEST_ALLOW_HF_DOWNLOADS`) restores downloads; an externally-set value always wins
- [ ] `pytest Tests/RAG_Search` outcome set is unchanged vs the serial baseline (junit diff)

## Implementation Plan

1. Replace the forced `TRANSFORMERS_OFFLINE=0` with offline-by-default `setdefault`s keyed off the existing real-embeddings env gate
2. Drop `autouse=True` from the session fixture; rename to `real_transformers_session`
3. Have `test_embeddings_real_integration.py` (the only real-model suite in the directory) request it via a lazy autouse shim gated on `TLDW_RUN_REAL_EMBEDDINGS`
4. Verify: `pytest Tests/RAG_Search` before/after, junit outcome diff empty

## Implementation Notes

Offline-by-default via `setdefault` (external env always wins); the warm-up fixture
(torch cpu default + move_cache + tiny-bert preload — the meta-tensor guard) is now
`real_transformers_session`, requested lazily from the real-integration module only
when `TLDW_RUN_REAL_EMBEDDINGS` is on, so chromadb-only tests in that file don't pay
it either. Grep confirms no other module referenced the old fixture name. Modified:
`Tests/RAG_Search/conftest.py`, `Tests/RAG_Search/test_embeddings_real_integration.py`.
