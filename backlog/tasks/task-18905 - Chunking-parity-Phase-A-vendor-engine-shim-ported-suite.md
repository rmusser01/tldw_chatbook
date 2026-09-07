---
id: TASK-18905
title: 'Chunking parity Phase A: vendor engine + Chunk_Lib shim + ported suite'
status: Done
assignee: []
created_date: '2026-08-19 09:30'
updated_date: '2026-08-19 09:30'
labels:
  - chunking
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase A (PR 1) of the Chunking Engine Parity sub-project: vendor `tldw_server`'s Chunking engine at `dev` @ `385afa95` into `tldw_chatbook/Chunking/engine/` via a manifest-driven sync script; write the three phase-1 shims (`testing`, `config`, `prompt_loader`); rewrite `Chunk_Lib.py` as the compatibility shim (legacy signatures preserved, exception aliases, constants re-exported, flat chunk-dict contract at the DB seam, shim-enforced tiktoken); port the upstream test suite plus golden parity fixtures verified with test mode explicitly disabled; add call-site characterization tests including the DB round-trip.

Plan: `Docs/superpowers/plans/2026-08-19-chunking-engine-parity.md` Tasks 1–6.
Spec: `Docs/superpowers/specs/2026-08-18-chunking-engine-parity-design.md` (§5, §6.2, §10).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `VENDOR_MANIFEST.toml` pins repo + branch + SHA (`dev` @ `385afa95`) with the exact 35-file vendored list; `sync_chunking_engine.py` is idempotent, SHA-verifying (never syncs from an unverified local path), and fails loudly on local modifications
- [x] #2 Licence obligations met: upstream GPLv3 headers preserved, `tldw_server` LICENSE shipped at `Chunking/engine/LICENSE`, licence recorded in the manifest, `"tldw_chatbook.Chunking.engine" = ["LICENSE"]` added to pyproject `license-files`
- [x] #3 The three shims exist (`testing`, `config`, `prompt_loader`); the engine imports and constructs a `Chunker` on a base install (no sklearn/nltk/transformers present)
- [x] #4 `Chunk_Lib.py` is a shim: legacy signatures work unchanged, exception aliases (`LanguageDetectionError`→`LanguageNotSupportedError`, `MemoryLimitError`→`InvalidInputError`) hold, constants (`DEFAULT_CHUNK_OPTIONS`, `MAX_CHUNK_SIZE_*`, `MAX_DOCUMENT_SIZE_*`, `ensure_nltk_data`) re-exported, module-level `chunk_xml` restored, and a `tokens` request that would word-approximate raises a clear tiktoken error (no silent fallback)
- [x] #5 Upstream suite ported into `Tests/Chunking/` (41 files; endpoint/async/propositions tests excluded per spec §10.1) and passes; offset/overlap property tests also pass with test mode explicitly disabled
- [x] #6 Golden parity fixtures generated from the server engine with test mode disabled and asserted byte-for-byte by `test_golden_parity.py`
- [x] #7 Call-site characterization tests pass, including the DB round-trip (top-level `start_char`/`end_char` populate the `UnvectorizedMediaChunks` columns, not NULL)
- [x] #8 `tiktoken` and `defusedxml` are declared core dependencies; `Tests/Performance/test_app_import_weight.py` stays green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Sync script + manifest + vendored tree + licence (plan Task 1)
2. The three shim modules (plan Task 2)
3. `Chunk_Lib.py` rewrite as compatibility shim (plan Task 3)
4. Port the upstream test suite + conftest production-path marker + golden fixtures (plan Task 4)
5. Call-site characterization tests + DB round-trip pin (plan Task 5)
6. Phase close-out: targeted suites + import-weight guard (plan Task 6)
<!-- SECTION:PLAN:END -->

## Implementation Notes

Commits `1c981dab5..5e12ada3b`: vendored the engine at `dev@385afa95` (35 files, manifest-driven sync, idempotent + SHA-verifying), added the three phase-1 shims, rewrote `Chunk_Lib.py` as the compat shim, ported the upstream suite (40 files at the pin — the brief's "41" counted an older upstream snapshot; endpoint/async/propositions excluded per §10.1), generated golden parity fixtures with test mode disabled, and pinned the DB round-trip contract.

- **Throughput margin (AC evidence)**: 2 MB corpus, words 400/100 — legacy regex 0.030s vs engine 0.11s; sentences 0.012s vs 0.06s. ~3–4x slower on the chunking step alone, but absolute cost is sub-0.2s at book scale. Chunker-construction cost empirically negligible: 600 chunks incl. per-call strategy construction in 0.11s. Margin: accepted.
- **Licence follow-up (review I-1)**: `LICENSES/GPL-3.0-only.txt` now also ships in the vendored subtree (GPLv3 §4); pyproject entry extended to `["LICENSE", "LICENSES/GPL-3.0-only.txt"]`.
- **Spec nuances**: XML ingestion parity is tracked separately (task-18941); `rolling_summarize` keeps its dual contract (token-based defaults in the engine, words-based at the chatbook shim seam).
- Full detail: `.superpowers` reports for the phase; ADR `backlog/decisions/071-vendored-chunking-engine-parity.md`.
