# Follow-up 163 — slim eager heavy imports at app startup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. One cohesive import-slimming task (backlog 163). Branch `claude/followups-startup-perf` off dev b8344550. Anchors exact at branch point; grep symbols, lines drift.

**Goal:** `import tldw_chatbook.app` no longer eagerly loads torch/transformers/numpy/nltk (~1,800 modules, most of the boot weight); they load on first real embedding/summarization use instead. Guarded by a subprocess regression test.

**Measured baseline (branch point):** `import tldw_chatbook.app` = ~4.8s / 6,518 modules, with `torch` (1,026), `transformers` (256), `scipy` (526), `sklearn` (190), `pandas` (293), `nltk` (240) all present.

**Root causes (traced via a meta-path finder, exact):**
1. **torch/transformers/numpy:** `app.py:182 from .RAG_Admin.local_rag_admin_service import LocalRAGAdminService` → `Embeddings/Chroma_Lib` → `Embeddings/__init__` → `Embeddings/Embeddings_Lib.py:53-55` does module-scope `numpy = get_safe_import('numpy')`, `torch = get_safe_import('torch')`, `transformers = get_safe_import('transformers')`. `get_safe_import` → `check_dependency` → `__import__(name)` (a REAL import), so all three load at boot.
2. **nltk/scipy/sklearn/pandas:** `Web_Scraping/Article_Extractor_Lib.py:99 from ...LLM_Calls.Summarization_General_Lib import analyze` (module scope) → pulls `Chunking/Chunk_Lib` and the summarization stack (nltk/scipy/sklearn/pandas). (~731ms of the boot.)

**Global Constraints:** explicit-path staging (NEVER `git add -A`); Fable 5 co-author line; RED-first (the regression test is the RED); NO behavior/feature change — torch/transformers/numpy must still be available when an embedding IS created; venv pytest with isolated HOME. Test command: `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`.

## Task (single cluster)

**Files:**
- Modify: `tldw_chatbook/Embeddings/Embeddings_Lib.py` (module-scope `get_safe_import` at :53-55; 13 `torch.` uses + 15 transformers/AutoModel uses)
- Modify: `tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py` (:99 eager `analyze` import; grep its `analyze(` call sites)
- Create: `Tests/Performance/test_app_import_weight.py` (or the closest existing perf-test dir — `ls Tests/ | grep -i perf`)

**Approach — Embeddings_Lib (the torch/transformers/numpy edge):**
- Replace the three module-scope `get_safe_import` calls (:53-55) with lazy access. Two acceptable patterns — pick the one that fits the existing code and document it:
  - (a) Module-level `torch = None` / `transformers = None` / `numpy = None` placeholders + a `_ensure_torch()` / `_ensure_transformers()` / `_ensure_numpy()` helper (each `global X; if X is None: X = get_safe_import(...)` — or a direct guarded import) called at the TOP of every method/function that dereferences the name. This is the F3 `transcription_service._ensure_torch_import` pattern — mirror it. OR
  - (b) `functools.cache`d module-level `_torch()` accessor functions returning the module, and rewrite the ~28 use sites from `torch.foo` to `_torch().foo`.
  - Prefer (a) (smaller diff at use sites) unless the code structure makes (b) cleaner. Whichever: EVERY current `torch.`/`transformers.`/`numpy.` dereference must be preceded (in its call path) by the ensure/accessor so a genuine embedding creation still works. Read all 28 use sites; miss none.
  - Availability checks (e.g. "is torch installed") must use `importlib.util.find_spec` (cheap, no import), NOT the eager `get_safe_import`/`__import__` — so a probe at boot doesn't reload torch. (F3 OCR-self-heal lesson.)
- Keep `require_dependency`/error messaging behavior identical when a dep is genuinely missing (an embedding attempt without torch installed must still raise the same clear error, now at first-use rather than import).

**Approach — Article_Extractor_Lib (the nltk edge):**
- Move `from ...LLM_Calls.Summarization_General_Lib import analyze` (:99) from module scope into the function(s) that call `analyze(` (grep the call sites). Same as F3's Document/PDF processor deferral.

**Scope extension (authorized 2026-07-11 by plan owner after live trace):** the dominant remaining boot weight is `nltk` (240 mods) + its transitive deps `scipy` (526) + `sklearn` (190), all pulled by `Chunking/Chunk_Lib.py` at module scope: (1) bare `import nltk` (:31-32, inside the top-level try/except) and (2) the module-scope `ensure_nltk_data()` CALL at :108, which additionally performs a **network punkt download at import time**. Defer both: derive `NLTK_AVAILABLE` from `importlib.util.find_spec("nltk")` (no import); make `sent_tokenize`/`nltk` lazily loaded via an `_ensure_nltk()` helper called at the top of every nltk use site (`_adaptive_chunk_size_nltk` :575, `semantic_chunk` :652, and the two `sent_tokenize(` calls :581/:664 — grep to confirm); and move the `ensure_nltk_data()` invocation out of module scope into first-use (idempotent guard flag), so a genuine chunk still downloads/finds punkt exactly as before but boot does not. Killing nltk-at-boot removes scipy+sklearn automatically (they are nltk-transitive, confirmed by meta-path trace). This is the crux of AC #1; the original file-scope list is extended to include `tldw_chatbook/Chunking/Chunk_Lib.py`. If `pandas` (or any other forbidden module) is still resident after the nltk fix, trace its REAL importer (a finder that fires on exec, not the `find_spec` probe) and defer it if it lives in Embeddings/Web_Scraping/LLM_Calls/Chunking/Local_Ingestion; otherwise report BLOCKED with the chain.

**Regression test (`Tests/Performance/test_app_import_weight.py`):**
- Subprocess-based (the import must be measured in a FRESH interpreter — `sys.modules` is process-global): `subprocess.run([sys.executable, "-c", SNIPPET], ...)` where SNIPPET imports `tldw_chatbook.app` then asserts `not any(m.split('.')[0] in {"torch","transformers","nltk","scipy","sklearn","pandas","docling","torchvision"} for m in sys.modules)`, exiting non-zero with the offenders on failure. The module-absence assertion is the real guard; add a generous wall-time bound (e.g. < 4.0s) only as a catastrophic-regression catch (don't make it flaky).
- RED: this test FAILS on the current code (torch/nltk present). Verify before implementing.
- Second test: assert the deferral didn't break availability — importing `Embeddings_Lib` then calling its `_ensure_torch()` (or accessor) returns the real torch module (skip if torch not installed in the env via `find_spec`).

**Verification & feature-regression gate:**
- The regression test green (no heavy modules at boot); measure and record the new `import tldw_chatbook.app` time + module count in the report (expect a large drop).
- Feature regression: run `Tests/Embeddings/` (or wherever embedding-factory tests live — `ls Tests/ | grep -i embed`), `Tests/RAG*/` if present, `Tests/Chunking/`, and anything covering `Article_Extractor`/summarization — these prove embeddings/summarization still WORK (torch loads on first use). `python -c "import tldw_chatbook.app"` clean.

Commit: `perf(startup): defer torch/transformers/numpy and the summarization stack out of app boot (163)`

## Gate

Combined: the new perf test + `Tests/Embeddings/` + `Tests/Chunking/` + `Tests/RAG_Search/` (adjudicate what exists) + `python -c "import tldw_chatbook.app"`. Mark backlog 163 Done (tick ACs). PR to dev; merge only on explicit user authorization. (No visual QA — no UI change.)
