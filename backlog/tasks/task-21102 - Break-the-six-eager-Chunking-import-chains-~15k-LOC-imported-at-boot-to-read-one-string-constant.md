---
id: TASK-21102
title: >-
  Break the six eager Chunking import chains - ~15k LOC imported at boot to read
  one string constant
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 04:16'
labels:
  - performance
  - startup
  - imports
  - chunking
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21102).

~15k LOC of `Chunking/` (incl. 28/38 vendored engine modules, a real `import langdetect`, an
nltk `find_spec` path scan, and the Internal_Prompts package) is imported eagerly through SIX
entry points. The first is `Local_Ingestion/local_file_ingestion.py:172` importing
`ENGINE_VERSION` - a string literal (`Chunk_Lib.py:150`). The others:
`Library/ingest_preflight.py:26`, `Library/web_clip_request.py:27`, `RAG_Search/__init__.py:21`,
`RAG_Admin/local_rag_admin_service.py:17-18`, `app.py:1997-2007`. Fixing only app.py buys
nothing - all six must break.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `import tldw_chatbook.app` no longer executes the Chunking package (nor langdetect) - pinned by a test asserting `"tldw_chatbook.Chunking" not in sys.modules` after importing the app module
- [x] #2 All six entry points are converted (constant inlined or lazily accessed; PEP-562 re-exports where a facade is needed); chunking behavior at first real use is unchanged
- [x] #3 Warm `python -X importtime` before/after numbers recorded in the task
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline: teed importtime + residency probe in isolated env (scratch HOME/XDG/TLDW_CONFIG_PATH); baseline runs of the test files covering the six modules.\n2. One source of truth for ENGINE_VERSION: new stdlib-only tldw_chatbook/chunking_engine_version.py (outside the Chunking package -- importing any Chunking submodule executes the package init and the whole engine); Chunk_Lib re-imports it so the package surface and the identity test keep the same object.\n3. Convert entry points: local_file_ingestion imports the pin from the new module (fixes chains via ingest_capabilities/ingest_preflight/web_clip_request too); RAG_Search/__init__ becomes a PEP-562 lazy facade preserving the stub-on-ImportError fallback; Media/local_media_reading_service defers ChunkingService to its sole call site; RAG_Admin/local_rag_admin_service defers AUTO_SENTINEL/get_chunking_service to use sites; app.py's two error-type imports move into an except-time accessor used only by the sole handler (~4048), preserving the caught types.\n4. Red-first guard test in Tests/Packaging following test_extras_import_closure.py's subprocess pattern: after import tldw_chatbook.app, no tldw_chatbook.Chunking* module and no langdetect in sys.modules.\n5. Re-run residency trace to catch any straggler edges; fix until guard is green.\n6. Verify: Tests/Chunking, Tests/Local_Ingestion, Tests/Library (preflight/web_clip/rechunk/ingest), Tests/RAG_Admin, Tests/Media, Tests/RAG, Tests/Packaging, App import tests; full --collect-only sweep; after importtime, A/B any red against base 3c3c919fc.\n7. Record importtime table + notes; commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Review fix round (2026-08-23)

Adversarial review confirmed all headline claims but returned one Major + one Minor; both fixed.

**MAJOR-1 (except-clause matcher ran for EVERY exception):** `except _template_resolution_errors() as exc:` evaluated the lazy imports for any exception reaching the ingest-dispatch handler, (a) replacing an unrelated in-flight error with `ModuleNotFoundError` on Chunking-broken installs and (b) importing ~39 Chunking modules mid-exception-handling on healthy ones. Fix in `app.py` `_template_resolution_errors()`: return `()` when `"tldw_chatbook.Chunking" not in sys.modules` (no template error can be in flight if the defining package never imported), and wrap the imports in `try/except Exception: return ()` so a broken env lets the ORIGINAL exception propagate. New subprocess-isolated `Tests/App/test_template_error_lazy_matching.py` (3 tests, red-first: the two guard tests failed on the pre-fix code with exactly the review's two outcomes - `OUTCOME:replaced-by:ModuleNotFoundError` and the 39-module side-import list - and the third anti-overcorrection test pins that both named error types still match once Chunking is resident, so a degenerate always-`()` mutation also reddens).

**MINOR-2 (facade widened the deps-absent surface):** the first facade cut stubbed all 10 re-exports; base's eager fallback defined only `EmbeddingsService`/`ChunkingService`/`IndexingService` (+ `RAGService` alias) and left the other 6 undefined, so `from tldw_chatbook.RAG_Search import create_rag_service` raised ImportError - which `Tests/RAG/test_rag_dependencies.py`'s `check_rag_services` uses as feature detection (it would have flipped False->True on deps-absent installs). Fix in `RAG_Search/__init__.py`: `_STUB_ON_FAILURE` frozenset restores the exact base split - the 4 base names degrade to stubs, all others raise AttributeError (chained from the ImportError) so from-imports raise ImportError. New subprocess-isolated `Tests/RAG/test_rag_search_facade.py` (2 tests, red-first: the deps-absent test failed on the all-stubs facade at `RAGConfig`).

**Re-verification:** fix-round tests 7 passed (3+2 new, 2 closure guards); Tests/App 169 passed / 8 failed - failure set byte-identical to base (Actor_Packs crash); facade-consumer batch 365 passed / 1 failed / 10 skipped (the 1 identical on base); mixed RAG_Admin+Media+Packaging batch 234 passed / 8 failed / 42 errors - set identical to baseline; Local_Ingestion 348/2/3 and Library 117/1 unchanged; collect-only 55,050 (+5 new tests), same 33 pre-existing errors. `Tests/RAG/test_rag_dependencies.py` itself contains no test functions (0 collected; it is a diagnostic script module) - its `check_rag_services` contract is pinned by the new facade test instead.
<!-- SECTION:NOTES:END -->
