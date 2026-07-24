---
id: TASK-257
title: Defer optional-feature imports: chromadb, web-search chain, PDF/document processors (~550ms)
status: Done
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, startup]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three eager chains for features unused by default: Chroma_Lib module-scope get_safe_import('chromadb') pulls chromadb→OTel→gRPC→protobuf (~154ms) via RAG_Admin; Tools/__init__ imports WebSearchTool → Article_Extractor_Lib module-scope playwright/trafilatura/dateparser (~197ms) though web_search_enabled defaults False (executor gate at tool_executor.py:643 is already correct, and the SAME FILE already fixed pandas with find_spec); app.py:124's direct submodule import bypasses Local_Ingestion's own lazy __init__, loading pymupdf/onnxruntime (~170ms) + Document lib (~59ms) for optional pdf/ebook extras. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P2 C2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 chromadb imports only when ChromaDBManager is instantiated
- [x] #2 Web-search chain imports only when the tool is enabled/used (find_spec probe for availability)
- [x] #3 Per-format ingestion processors import at dispatch time
- [x] #4 Measured app-import delta reported (expected >400ms); optional-deps availability semantics unchanged (extras absent still degrade gracefully)
<!-- AC:END -->

## Implementation Plan

1. Read the current state of `Embeddings/Chroma_Lib.py` (another session's RAG
   program had touched this area recently) before changing anything.
2. Chain 1 (chromadb): defer `get_safe_import('numpy')`/`get_safe_import('chromadb')`
   into `_ensure_numpy()`/`_ensure_chromadb()` helpers called from
   `ChromaDBManager.__init__`, preserving the existing unavailable-feature
   placeholders.
3. Chain 2 (web-search): convert `Tools/__init__.py`'s eager optional-tool
   imports to a PEP 562 lazy `__getattr__` re-export layer (mirroring
   `Local_Ingestion/__init__.py`/`tldw_api/__init__.py`); defer
   `Web_Scraping/Article_Extractor_Lib.py`'s module-scope playwright/
   trafilatura imports behind `find_spec` probes + `_ensure_playwright()`/
   `_ensure_trafilatura()` helpers, mirroring that file's existing pandas
   `find_spec` deferral.
4. Chain 3 (PDF/document/ebook/audio/video): move
   `Local_Ingestion/local_file_ingestion.py`'s five per-format processor
   imports (process_pdf/process_document/process_ebook/LocalAudioProcessor/
   LocalVideoProcessor) off module scope into `_ensure_*()` helpers invoked
   from `parse_local_file_for_ingest()`'s per-`file_type` dispatch branches.
5. Write `Tests/Utils/test_optional_import_deferral.py`: isolated-subprocess
   `sys.modules` assertions for all five deferred modules, functional smoke
   per chain, and a graceful-absence (simulated missing dependency) case.
6. Measure app-import wall time/module count/`sys.modules` residency
   before (detached `origin/dev` worktree) vs after, in a scratch HOME/XDG
   sandbox; report deltas.
7. Run the full affected test suites and confirm no existing test files
   were modified.

## Implementation Notes

**Chain 1 -- chromadb (`Embeddings/Chroma_Lib.py`)**: module-scope
`numpy = get_safe_import('numpy')` / `chromadb = get_safe_import('chromadb')`
(and the `Settings`/`ChromaError`/`InvalidDimensionException`/`Collection`/
`QueryResult` names derived from `chromadb`) are now `None`/unavailable-feature
placeholders at import time, resolved by `_ensure_numpy()`/`_ensure_chromadb()`
helpers (mirroring `Embeddings_Lib.py`'s `_ensure_torch()` pattern) called at
the top of `ChromaDBManager.__init__`. The entire rest of the 1,231-line file
is inside `ChromaDBManager` methods, reachable only after `__init__`, so no
other module-scope code needed changes. This chain's `_ensure_*()` work was
already present, uncommitted, in the worktree when this session started
(apparently a prior partial run of this exact task) -- it was reviewed,
verified correct against the current file state, and left as-is.

**Chain 2 -- web-search (`Tools/__init__.py` + `Web_Scraping/Article_Extractor_Lib.py`)**:
`Tools/__init__.py`'s eager `try/except` imports of `WebSearchTool`/
`ReadFileTool`/`ListDirectoryTool`/`WriteFileTool`/`RAGSearchTool`/
`CreateNoteTool`/`SearchNotesTool`/`UpdateNoteTool` are replaced with a PEP 562
`__getattr__` + `_SUBMODULE_BY_NAME` mapping (same pattern as
`Local_Ingestion/__init__.py`/`tldw_api/__init__.py`); `Tool`/`ToolExecutor`/
`DateTimeTool`/`CalculatorTool`/`get_tool_executor`/`reload_tool_executor` stay
eager (lightweight, always-needed). `tool_executor.get_tool_executor()` was
already unaffected -- it imports each optional tool directly from its own
submodule, gated by per-tool config flags, never through this package's names.
`Article_Extractor_Lib.py`'s module-scope `try: from playwright... / import
trafilatura` blocks are replaced with `find_spec`-based `PLAYWRIGHT_AVAILABLE`/
`TRAFILATURA_AVAILABLE` probes (mirroring the file's existing pandas
deferral) plus `_ensure_playwright()`/`_ensure_trafilatura()` helpers, called
at the top of `scrape_article()` (guards `async_playwright`/`TimeoutError`/
`trafilatura.extract`/`extract_metadata` use) and `recursive_scrape()` (guards
its direct `async_playwright()` call). `TimeoutError` is intentionally
shadowed at module scope (unchanged pre-existing behavior) so `except
TimeoutError` clauses transparently pick up playwright's real exception once
ensured. The `_ensure_*()` guards no-op when the name is already bound to
something other than `None` (e.g. a test's `unittest.mock.patch`), so the
existing `Tests/Web_Scraping/test_article_extractor.py` mocks of
`async_playwright` continue to work unmodified.

**Chain 3 -- PDF/document/ebook/audio/video (`Local_Ingestion/local_file_ingestion.py`)**:
the five per-format processor imports moved from module scope into
module-level `None` placeholders + `_ensure_process_pdf()`/
`_ensure_process_document()`/`_ensure_process_ebook()`/
`_ensure_local_audio_processor()`/`_ensure_local_video_processor()` helpers,
called from `parse_local_file_for_ingest()`'s matching `file_type` branch.
Plain local `from .X import Y` statements inside each branch were considered
and rejected: two existing tests
(`Tests/Local_Ingestion/test_parse_url_routing.py::test_video_url_routes_to_audio_video_branch`,
`Tests/Local_Ingestion/test_ingest_parse_worker.py`'s `process_pdf`
monkeypatch, plus a third `monkeypatch.setattr(lfi, "LocalAudioProcessor",
...)` in the same file) patch these names as **module attributes** of
`local_file_ingestion`, which requires the name to already exist at module
scope -- the `_ensure_*()`-with-module-global pattern (matching the chromadb/
playwright precedent in this same task) satisfies both the laziness goal and
existing-test patchability: each `_ensure_*()` no-ops when its name is
already bound to something other than the placeholder, so a patched/mocked
value is never clobbered.

**Tests**: new `Tests/Utils/test_optional_import_deferral.py` (16 tests) --
isolated-subprocess `sys.modules` assertions (whole-set + one parametrized
test per module) after `import tldw_chatbook.app`, using a scratch HOME/XDG
sandbox (never the real `~/.config/tldw_cli`); functional smoke per chain
(real `ChromaDBManager` construction resolving real chromadb, `Tools`
PEP 562 `__getattr__` resolving `WebSearchTool` + full `__all__`/unknown-name
checks, `parse_local_file_for_ingest` processing a plaintext file with none of
the five processors touched, a PDF-branch wiring check via a mocked
`process_pdf`); two graceful-absence cases (simulated broken playwright via
`builtins.__import__` interception -- mirrors the file's existing pandas-guard
test pattern -- and simulated broken chromadb via a faked `get_safe_import`).
All existing tests were left read-only (`git diff origin/dev --diff-filter=M
--name-only -- Tests/` is empty).

**Measured deltas** (5x `import tldw_chatbook.app` in a fresh interpreter per
run, scratch HOME/XDG, cwd = tree under test so the editable install resolves
locally; baseline = detached `origin/dev` worktree at the same commit this
branch started from, removed after measurement):

| | baseline (before) | after | delta |
|---|---|---|---|
| wall time (steady-state avg, runs 2-5) | ~1.542s | ~0.980s | **~562ms faster** |
| wall time (all 5 runs incl. cold run 1) | ~1.774s | ~0.999s | ~775ms faster |
| module count | 3,294 | 1,992 | **-1,302 modules (-39.5%)** |
| `chromadb`/`playwright`/`trafilatura`/`pymupdf`/`dateparser` resident? | all 5 loaded | **none loaded** | -- |

Exceeds the >400ms AC #4 target. `fitz` (pymupdf's import name) was never
separately resident in either run (pymupdf's own top-level package name
covers it). Optional-deps availability semantics are unchanged: each chain's
existing unavailable-feature fallback path (ImportError with an install hint
for chromadb/ChromaDBManager; error-dict degrade for
scrape_article/scrape_and_summarize_multiple; internal per-module
try/except in PDF_Processing_Lib/Document_Processing_Lib/Book_Ingestion_Lib/
audio_processing/video_processing) is untouched -- only *when* the real
import is attempted moved, not *whether* it degrades gracefully when absent.

**Files modified**: `tldw_chatbook/Embeddings/Chroma_Lib.py`,
`tldw_chatbook/Tools/__init__.py`,
`tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py`,
`tldw_chatbook/Local_Ingestion/local_file_ingestion.py`.
**Files added**: `Tests/Utils/test_optional_import_deferral.py`.
