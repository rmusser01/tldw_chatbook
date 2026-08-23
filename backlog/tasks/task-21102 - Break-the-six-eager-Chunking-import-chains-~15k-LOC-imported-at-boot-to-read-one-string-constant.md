---
id: TASK-21102
title: >-
  Break the six eager Chunking import chains - ~15k LOC imported at boot to read
  one string constant
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 03:55'
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
Broke every eager import chain that executed the Chunking package (~15k LOC: first-party shim + vendored engine + langdetect attempt + nltk find_spec scan) during `import tldw_chatbook.app`. Verified via an import-hook edge tracer, not just the pinned survey lines - dev had moved and one extra chain existed.

**One source of truth for ENGINE_VERSION:** new stdlib-only (zero-import) module `tldw_chatbook/chunking_engine_version.py` holds the pin string. It must live OUTSIDE `Chunking/` because importing any `Chunking.*` submodule executes the package `__init__` and with it the whole engine. `Chunking/Chunk_Lib.py` now re-imports it (same object, so `Tests/Local_Ingestion/test_engine_version_stamp.py::test_engine_version_reexported_from_package`'s identity assertion holds), and the persist seam imports the pin module directly. No vendored file touched (vendored code is only `Chunking/engine/`; Chunk_Lib is first-party).

**Conversions (7 chains, the survey's 6 + 1 found on my base):**
1. `Local_Ingestion/local_file_ingestion.py` - `ENGINE_VERSION` now from the pin module. This alone also freed `Library/ingest_preflight.py`, `Library/web_clip_request.py`, and `Library/ingest_capabilities.py` (they import only chunking-free names from it).
2. `RAG_Search/__init__.py` - PEP-562 lazy facade (`__getattr__` + cache + `__dir__`); the legacy stub-on-ImportError fallback is preserved per-name (constructor raises the same "RAG services not available" ImportError, now at first use). This also removed the eager `.simplified` tree (~70 modules incl. `chunking_service`, `enhanced_chunking_service`, `parent_child_adapter`).
3. `Media/local_media_reading_service.py` - `ChunkingService` import moved into `_chunk_text` (its sole use).
4. `RAG_Admin/local_rag_admin_service.py` - `get_chunking_service` moved into `__init__` (exact falsy-or semantics preserved); `AUTO_SENTINEL` moved into `_decorate_template_record`.
5. `RAG_Admin/template_validation.py` (the extra chain, via `rag_admin_scope_service` <- `RAG_Admin/__init__` <- app.py) - the three `Chunking.engine.regex_safety` helpers became lazy pass-through wrappers.
6. `app.py` - the two template-error imports became `_template_resolution_errors()`, evaluated in the except clause of the sole handler (`except _template_resolution_errors() as exc:`); by the time a template error can be in flight, `_ingest_job_options` has already imported the raising modules, so the caught types are the identical class objects.

**Guard (red-first):** `Tests/Packaging/test_chunking_import_closure.py`, following TASK-21104's subprocess-isolated pattern: asserts no `tldw_chatbook.Chunking*` and no `langdetect` in `sys.modules` after `import tldw_chatbook.app`, plus anti-vacuity checks that the converted modules are still in the closure; second test pins the pin-module's chunking-free import + Chunk_Lib identity. Red before the fix (40 Chunking modules listed), green after. Honest scope: langdetect/nltk are not installed in this venv, so that half bites only on envs that have them; the module-residency assertion is what pins the boot path.

**Warm `python -X importtime` (isolated HOME/XDG/TLDW_CONFIG_PATH, runs 1/2/3; logs in test-logs/):**
| metric | before | after |
| app cumulative (run 3, warm) | 810.8 ms (runs: 1178/784/811) | 730.9 ms (runs: 749/727/731) |
| tldw_chatbook.Chunking* modules resident | 43 | 0 |
| Chunking self-time sum | 11.4 ms | 0 |
| total modules after app import | 1831 | 1757 |
(`Internal_Prompts` also left the boot closure - it was only reached through Chunk_Lib.)

**Tests (all counts read, teed to test-logs/; base = 3c3c919fc):**
- Guard: 2 passed (red-first verified).
- Tests/Chunking minus test_sync_script: 547 passed, 3 failed, 39 skipped, 1 xfailed, 1 error - failure set IDENTICAL to base baseline (semantic/golden-cjk/template-rag + transformers-offline error; missing optional deps in this venv).
- Tests/Chunking/test_sync_script.py: 8 passed (base: 8 passed). NOTE: an intermediate red here was self-inflicted - my first full-suite run hit the Bash timeout mid-test and left the test's "# local edit" marker in the vendored `engine/constants.py`; restored Edit-based, rerun green.
- Tests/Local_Ingestion (minus numpy-dependent parakeet file, uncollectable on base too): 348 passed, 2 failed, 3 skipped - identical to base.
- Library preflight/web_clip/rechunk: 117 passed, 1 failed (egress redirect; in base baseline).
- RAG_Admin + Media(3 files) + RAG/test_chunking_service + Packaging + engine_version_stamp: 234 passed (base 232, +2 = new guard), 8 failed, 42 errors - failure/error sets byte-identical to base (packaging build-env errors, nltk-dependent chunking_service tests, rag-admin wiring reds from the known Actor_Packs crash).
- Tests/App: 166 passed, 8 failed - A/B'd on a throwaway base worktree: identical 8 failures (known pre-existing Actor_Packs `'NoneType' object has no attribute 'execute_query'` crash).
- Facade consumers (Library rag-mode/media-chunk-tool/student-story/local-rag-search, RAG fusion/citation-capture/parent-child/ingestion-indexing): 365 passed, 1 failed, 10 skipped - the 1 failure identical on base.
- Tests/Performance/test_app_import_weight.py: 3 passed, 3 skipped.
- Full collect-only: 55045 collected (base 55043, +2), 33 collection errors - error set identical to base (optional deps absent from venv).

**Files:** new `tldw_chatbook/chunking_engine_version.py`, `Tests/Packaging/test_chunking_import_closure.py`; modified `app.py`, `Chunking/Chunk_Lib.py`, `Local_Ingestion/local_file_ingestion.py`, `RAG_Search/__init__.py`, `Media/local_media_reading_service.py`, `RAG_Admin/local_rag_admin_service.py`, `RAG_Admin/template_validation.py`.

**Deliberately not touched:** `Library/library_rechunk_service.py:42-44` still imports Chunk_Lib/template_runtime at module scope - it is not in the app import closure (verified by residency probe) and legitimately uses the engine; `Widgets/chunk_preview_modal.py`, `media_details_widget.py`, `Local_Ingestion/XML_Ingestion.py` likewise closure-absent.
<!-- SECTION:NOTES:END -->
