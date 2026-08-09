---
id: TASK-3998
title: >-
  Eval-harness fingerprint must record the load-bearing stack
  (transformers/torch/chromadb), not sentence-transformers
status: Done
assignee: []
created_date: '2026-08-09 14:48'
updated_date: '2026-08-09 17:33'
labels:
  - rag
  - eval
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the P1 final review (TASK-3894): the harness's real embedding path is Embeddings_Lib._HFEmbedder -> transformers.AutoModel + torch, with chromadb doing ANN retrieval; none of those three is fingerprinted, while sentence-transformers -- which is not on the load path -- is. This breaks in both directions: upgrading torch/transformers/chromadb shifts numerics with NO fingerprint change, producing a false REGRESSED hunt; upgrading sentence-transformers alone produces a pointless ENVIRONMENT_CHANGED re-stamp with no actual numeric change behind it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fingerprint includes transformers, torch, and chromadb versions
- [x] #2 The sentence-transformers key's retention or removal is decided and documented
- [x] #3 Baselines are re-stamped in the same commit with both old and new fingerprints shown
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-09-rag-port-hybrid-fusion-fixes.md (Task 2) and Docs/superpowers/specs/2026-08-09-rag-port-hybrid-fusion-fixes-design.md for the fingerprint-keys design (transformers/torch/chromadb compared, sentence-transformers informational).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented per the plan (Docs/superpowers/plans/2026-08-09-rag-port-hybrid-fusion-fixes.md Task 2) and design doc's TASK-3998 section.

current_fingerprint() (Tests/RAG_Eval/harness/baseline_io.py) now returns six compared keys: model, transformers, torch, chromadb, corpus_sha256, platform. All three new versions come from a new _package_version(name) helper wrapping importlib.metadata.version with an "absent" fallback (never raises -- extras gate guarantees presence in gated runs, but the always-on tests exercise this without the gate). sentence_transformers moved out of the compared dict entirely: a new _informational_stamp() helper produces it, and _metadata() now writes it under a new, non-compared metadata["environment_info"] key, separate from the compared metadata["environment"] block. Nothing in _environment_diff/environment_mismatch needed to change -- they only ever look at metadata["environment"], so putting sentence_transformers anywhere else already excludes it from comparison. Documented the split as a fifth numbered decision in the module docstring, matching the file's existing style (each decision already had one).

Tests (Tests/RAG_Eval/test_baseline_io.py): added 3 new always-on tests (non-empty new-key values; sentence_transformers recorded in environment_info but absent from environment, exercised via real current_fingerprint() with no override so it proves the production split; a baseline stamped in the old 4-key shape reads as ENVIRONMENT_CHANGED naming exactly {sentence_transformers, transformers, torch, chromadb} when compared against a real current fingerprint). Updated exactly ONE existing test in place: test_fingerprint_carries_exactly_the_four_documented_keys -> renamed test_fingerprint_carries_exactly_the_six_documented_keys, asserting the new 6-key set instead of the old 4-key set (the key set itself IS the change this task makes). No other existing test was touched -- the shared _fingerprint() test helper (used by ~25 other tests as an opaque, self-consistent stamp/compare pair) was deliberately left in its pre-existing 4-key shape, since those tests only need internal consistency between their own stamp and compare calls, not agreement with current_fingerprint()'s real shape.

Verified true RED-before-GREEN by temporarily swapping baseline_io.py back to its pre-change (git HEAD) content, confirming exactly the 4 new/changed tests failed (25 pre-existing tests still passed unmodified), then restoring the new implementation and re-confirming 29/29 green.

Same-commit re-stamp: `RAG_EVAL=1 RAG_EVAL_UPDATE_BASELINES=1 pytest Tests/RAG_Eval/` -- all 60 gated metrics printed `(+0.000)` deltas (baseline JSON diffs confirm the "metrics" blocks are byte-identical; only metadata.environment/environment_info/report_only.latency changed). Old fingerprint: {model: all-MiniLM-L6-v2, sentence_transformers: 5.4.1, corpus_sha256: cbf911c98c..., platform: darwin}. New fingerprint: {model: all-MiniLM-L6-v2, transformers: 5.6.2, torch: 2.11.0, chromadb: 1.5.8, corpus_sha256: cbf911c98c... (unchanged), platform: darwin} + environment_info.sentence_transformers: 5.4.1. A subsequent plain gated run (no update) against the freshly stamped baselines printed `PASSED: No regression. 60 metric(s) within 0.05 of baseline.` Full Tests/RAG_Eval/ battery: 142 passed (4 skipped when ungated, as expected).

Also updated Tests/RAG_Eval/README.md's "Fingerprint semantics" section, which described the old 4-key shape and would otherwise have gone stale the moment this landed.

Files: Tests/RAG_Eval/harness/baseline_io.py, Tests/RAG_Eval/test_baseline_io.py, Tests/RAG_Eval/baselines/{hybrid,plain,semantic}.json, Tests/RAG_Eval/README.md.
<!-- SECTION:NOTES:END -->
