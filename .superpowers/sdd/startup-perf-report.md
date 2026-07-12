# Task 163 — slim eager heavy imports at app startup — Report

Branch: `claude/followups-startup-perf`
Plan: `Docs/superpowers/plans/2026-07-11-followups-startup-perf.md`

## Status: COMPLETE (AC #1 satisfied) — after the authorized scope extension

> **NOTE:** This top section documents commit 1 (torch/transformers/numpy),
> which left nltk/scipy/sklearn/pandas as a "Blocked" out-of-scope chain. The
> coordinator then AUTHORIZED extending scope into `Chunking/Chunk_Lib.py`;
> commit 2 closed that chain. **See the "UPDATE — nltk/scipy/sklearn/pandas
> deferral" section at the bottom of this file for the final state.** All
> eight HEAVY_MODULES are now absent at boot (6,519 → 3,291 modules); AC #1 is
> satisfied. The "Blocked"/"do not mark Done" language below is superseded by
> that UPDATE.

torch and transformers were eliminated from `import tldw_chatbook.app` in
commit 1 (1,026 + 256 = 1,282 heavy modules); nltk/scipy/sklearn/pandas were
eliminated in commit 2. See "Blocked: remaining chain" below for the original
trace (now resolved) and the UPDATE section for the fix.

## Deferral pattern chosen

Pattern (a) from the plan: module-level `torch = None` / `transformers = None`
/ `numpy = None` placeholders + `_ensure_torch()` / `_ensure_transformers()`
/ `_ensure_numpy()` helpers (`global X; if X is not None: return X; X =
get_safe_import(...)`), called at the top of every method that dereferences
the name — mirrors `Local_Ingestion/transcription_service.py`'s
`_ensure_torch_import()`/`_ensure_qwen2audio_imports()` precedent exactly.

For the `analyze` / `chunk_for_embedding` / `pandas` eager imports, the fix
is the plan's second pattern: move the import from module scope into the
function(s) that actually call it (mirrors
`Local_Ingestion/document_processor.py`'s deferral precedent, per
`Tests/Local_Ingestion/test_ingest_import_weight.py`, which pre-existed this
task and validates the same pattern for the OCR/local-ingestion chain).

## Files changed and every use-site class guarded

### `tldw_chatbook/Embeddings/Embeddings_Lib.py` (ROOT CAUSE 1)

Replaced the module-scope `numpy = get_safe_import('numpy')` / `torch = ...`
/ `transformers = ...` (was lines 53-55) with `torch/transformers/numpy/
AutoModel/AutoTokenizer = None` placeholders and `_ensure_torch()` /
`_ensure_transformers()` / `_ensure_numpy()`.

Every `torch.` / `transformers.` / `AutoModel.` / `AutoTokenizer.` / `np.`
dereference site (28 total, all read and traced) is now preceded by the
matching `_ensure_*()` call in its call path:

- `Tensor` type alias: was `torch.Tensor` or `Any` depending on availability
  at import time — now unconditionally `Any`. Verified this is *only* ever
  used in annotations (deferred at runtime by this module's pre-existing
  `from __future__ import annotations`) and in the `PoolingFn = Callable[[Tensor,
  Tensor], Tensor]` alias (a plain typing construct, not a runtime dispatch
  target — no `isinstance(x, Tensor)` anywhere in the file).
- `_masked_mean()` (module function): `_ensure_torch()` added at top; inlined
  `torch.nn.functional.normalize(...)` instead of relying on a module-level
  `normalize` alias that could no longer be resolved at import time (`normalize`
  and its `create_unavailable_feature_handler(...)` fallback were removed —
  nothing else referenced them; confirmed via grep across `tldw_chatbook/` and
  `Tests/`).
- `_HuggingFaceEmbedder.__init__`: `_ensure_torch()` + `_ensure_transformers()`
  added at the very top, plus an explicit `if torch is None or transformers is
  None: raise ImportError(...)` reusing the exact same message `_build()`
  already used — `_build()` gated construction of this class before, but this
  makes the class safe to construct directly too (defense in depth; no test
  or caller does this today, confirmed via grep). All `AutoTokenizer.from_pretrained`,
  `AutoModel.from_pretrained` (x3, including the two meta-tensor-reload retry
  paths), `torch.float16/float32/cuda.is_available/device/backends.mps` sites
  inside `__init__` are downstream of this guard.
- `_HuggingFaceEmbedder._forward`: `_ensure_torch()` added at top (guards
  `torch.inference_mode()`).
- `_HuggingFaceEmbedder.embed`: `_ensure_torch()` added right before
  `torch.cat(...)` (the loop above it already calls `_forward`, which
  self-guards, but this covers the empty-`vecs` edge too).
- `_HuggingFaceEmbedder.close`: `_ensure_torch()` added before
  `torch.cuda.is_available()/empty_cache()`.
- `_openai_embedder`'s inner `_embed()`: `_ensure_numpy()` added at top
  (guards `np.asarray`/`np.float32`).
- `EmbeddingFactory._build()` (huggingface branch): `_ensure_torch()` +
  `_ensure_transformers()` added before the `if torch is None or transformers
  is None: raise ImportError(...)` gate (unchanged message/semantics).
- `EmbeddingFactory.embed()` (empty-`texts` branch): `_ensure_numpy()` added
  before `np.empty(...)`.

Availability checks use `get_safe_import` (a real, cached `__import__` via
`optional_deps.check_dependency`) rather than `importlib.util.find_spec`,
matching the plan's pattern (a): `_ensure_torch()` etc. *are* the "first real
use" points, called only from inside methods that are about to actually
build/use a model — never from a cheap probe path — so there was no bare
availability-only site that needed the `find_spec`-no-import treatment in
this file.

Removed the now-unused `create_unavailable_feature_handler` import (was only
used for the `normalize`/`AutoModel`/`AutoTokenizer` placeholders that are
gone). Left `check_dependency`/`require_dependency` in the import list
unchanged — they were already unused pre-existing imports before this task
and are out of scope to clean up.

**Verified end-to-end** (not just import-absence): constructed a real
`EmbeddingFactory`, called `.embed(['hello world'])`, which triggered
`_ensure_torch()`/`_ensure_transformers()`, downloaded
`mixedbread-ai/mxbai-embed-large-v1` from HF Hub, and produced a real
`(1, 1024)` embedding — proving the deferral doesn't break the actual
feature, only *when* torch/transformers load.

### `tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py` (ROOT CAUSE 2)

- Moved `from ...LLM_Calls.Summarization_General_Lib import analyze` (was
  line 99) into `scrape_and_summarize_multiple()`, right before its one call
  site (inside the `if summarize_checkbox: ... if content:` branch, so it's
  only imported when there's actually content to summarize).
- **Additional eager edge found and fixed** (same file, in scope): the
  module's own `try: import pandas as pd; PANDAS_AVAILABLE = True; except
  ImportError: ...` block (was lines 61-66) is used by exactly one function,
  `parse_csv_urls()` (`pd.read_csv`, `pd.errors.EmptyDataError`). Converted
  `PANDAS_AVAILABLE` to a cheap `importlib.util.find_spec("pandas") is not
  None` probe (per the plan's explicit "availability checks must use
  find_spec, not get_safe_import" instruction) and moved `import pandas as
  pd` into `parse_csv_urls()` itself. Verified `parse_csv_urls()` still works
  end-to-end against a real temp CSV file.

### `tldw_chatbook/Embeddings/Chroma_Lib.py` (additional eager edge, in
scope — same directory as Embeddings_Lib.py)

Traced why `torch`/`nltk` still loaded after the two plan-specified fixes:
`RAG_Admin/local_rag_admin_service.py:14-15` does `try: from
..Embeddings.Chroma_Lib import ChromaDBManager except Exception: ChromaDBManager
= None` at module scope (a legitimate optional-fallback import, left
untouched — not the problem). `Chroma_Lib.py` itself, independently of
`Embeddings_Lib.py`, did its own module-scope:
- `from tldw_chatbook.Chunking.Chunk_Lib import chunk_for_embedding` (pulls
  nltk/scipy/sklearn/pandas via `Chunk_Lib.py:31 import nltk`)
- `from tldw_chatbook.LLM_Calls.Summarization_General_Lib import analyze`
  (same nltk/scipy/sklearn/pandas stack via `Summarization_General_Lib` ->
  `Chunk_Lib`)

Both have exactly one call site each (confirmed via grep for
`chunk_for_embedding(` / `analyze(`): `process_and_store_content()` and
`situate_context()` respectively. Moved both imports into those methods.
`Chroma_Lib.py`'s own `numpy = get_safe_import('numpy')` /
`chromadb = get_safe_import('chromadb')` (module scope) were **left as-is**:
verified `import chromadb` alone only pulls `numpy` (not in the plan's guard
set, and not worth chasing since chromadb needs it regardless), and
deferring chromadb itself across a ~1,150-line file with no single call-site
chokepoint would be a much larger, higher-risk change out of proportion to
this task — documented here as a deliberate scope decision, not an oversight.

### `tldw_chatbook/Web_Scraping/WebSearch_APIs.py` (additional eager edge,
in scope — same directory as Article_Extractor_Lib.py)

Traced a *second*, independent boot path into the summarization stack:
`app.py:67` -> `Event_Handlers/Chat_Events/chat_streaming_events.py:20` ->
`Event_Handlers/worker_events.py:27 from ..Tools import get_tool_executor` ->
`Tools/__init__.py:17 from .web_search_tool import WebSearchTool` ->
`Tools/web_search_tool.py:12 from ..Web_Scraping.WebSearch_APIs import
perform_websearch` -> `WebSearch_APIs.py:70 from
...Summarization_General_Lib import analyze` (module scope). One call site
(`search_result_relevance()`, inside its relevance-scoring loop). Moved the
import to the top of that function.

## Before / after measurement

Same methodology both times (fresh, never-before-used `HOME`/`XDG_DATA_HOME`
dirs, single-shot `import tldw_chatbook.app`, `sys.modules` count):

| | import time | module count | torch | transformers | numpy | nltk/scipy/sklearn/pandas |
|---|---|---|---|---|---|---|
| **Pre-fix** (dev b8344550) | ~5.65s | 6,519 | loaded | loaded | loaded | loaded |
| **Post-fix** (this branch) | ~4.11s | 4,659 | **not loaded** | **not loaded** | loaded (chromadb needs it) | still loaded (blocked, see below) |

Repeated runs (warmer disk cache, same isolated-but-reused HOME):
pre-fix ~4.8s-5.7s / 6,518-6,519 modules; post-fix ~2.1s-4.5s / 4,658-4,660
modules. Module count is the stable signal (deterministic module-for-module,
~28% reduction); wall time is noisy across runs (cold-cache subprocess
overhead) but consistently lower post-fix.

`docling`/`torchvision` were absent both before and after (never triggered
by this chain).

## Blocked: remaining chain (nltk/scipy/sklearn/pandas)

After fixing all four in-scope files above, `find_spec`/`sys.modules` still
showed nltk/scipy/sklearn/pandas loaded by a plain `import tldw_chatbook.app`.
Traced (meta-path/`__import__` tracer, exact, branch point) to a **fifth**,
structurally different chain that is **not** in Embeddings/Web_Scraping/
LLM_Calls scope:

```
app.py:182            from .RAG_Admin.local_rag_admin_service import LocalRAGAdminService
RAG_Admin/__init__.py:3  from .local_rag_admin_service import LocalRAGAdminService
RAG_Admin/local_rag_admin_service.py:11
                       from ..Chunking.chunking_interop_library import get_chunking_service
Chunking/__init__.py:6   from .Chunk_Lib import (...)     <- package __init__ runs
                                                              unconditionally for ANY
                                                              import under tldw_chatbook.Chunking,
                                                              including chunking_interop_library
Chunking/Chunk_Lib.py:31  import nltk                      <- module scope, real import
  -> nltk/metrics/association.py: from scipy.stats import fisher_exact   (pulls scipy)
  -> nltk/classify/scikitlearn.py: from sklearn.feature_extraction import DictVectorizer  (pulls sklearn)
  -> (pandas is pulled the same way, confirmed independently: import tldw_chatbook.Chunking.Chunk_Lib
     alone loads {nltk, scipy, sklearn, pandas, numpy})
```

`RAG_Admin/local_rag_admin_service.py:11`'s import of
`chunking_interop_library` is legitimate application logic (RAG_Admin
genuinely needs the chunking service) — the actual problem is that
`Chunking/__init__.py` unconditionally re-exports from `Chunk_Lib` at
package-init time, and `Chunk_Lib.py:31` does a bare module-scope `import
nltk` (in a `try/except ImportError`, but still a real eager import).
Neither `RAG_Admin/` nor `Chunking/` is in this task's authorized scope
(Embeddings/Web_Scraping/LLM_Calls only), so per instructions I stopped here
without editing either file.

**Recommendation for a follow-up task**: the fix is small and mirrors the
exact pattern used four times in this commit — either (a) make
`Chunk_Lib.py`'s `import nltk` lazy (module-level `nltk = None` +
`_ensure_nltk()`, called from the ~2-3 functions that actually call
`sent_tokenize`/etc.), or (b) stop `Chunking/__init__.py` from eagerly
re-exporting `Chunk_Lib` symbols at package-init time (defer the re-export or
have `chunking_interop_library.py` import `Chunk_Lib` directly instead of via
the package `__init__`). Either would let
`test_app_import_does_not_load_full_heavy_dependency_set` (see below) go
green without touching anything already fixed in this commit.

## Test evidence

New file: `Tests/Performance/test_app_import_weight.py` (subprocess-based,
fresh interpreter per the plan's RED-first/module-absence strategy).

- `test_app_import_does_not_load_torch_or_transformers` — **hard assertion**,
  the actual guard for this task's achieved scope. RED-verified: fails on
  pre-fix code (`git stash` back to the original 4 files) with `['torch',
  'transformers']` loaded; GREEN on the fix.
- `test_app_import_stays_well_under_pre_fix_baseline` — catastrophic-regression
  tripwire (module count < 5,200, wall time < 8s, both generous). RED on
  pre-fix (6,519 modules / ~5.6s), GREEN post-fix (4,659 modules / ~2-4.5s).
- `test_app_import_does_not_load_full_heavy_dependency_set` — the plan's
  literal full-guard-set assertion, marked `xfail(strict=True)` with the
  chain above as the reason, so it documents the exact remaining gap, keeps
  the suite green, and will loudly XPASS-fail (prompting marker removal) if
  a follow-up fixes the `Chunking` chokepoint. `xfail` both before and after
  this task's fix (expected — the full set was never going to pass without
  the `Chunking` fix).
- `test_ensure_torch_resolves_real_torch_when_installed`,
  `test_ensure_transformers_resolves_real_transformers_when_installed`,
  `test_ensure_numpy_resolves_real_numpy_when_installed` — availability-
  preserved checks (skip via `find_spec` if the dep isn't installed). RED on
  pre-fix (`AttributeError: no attribute '_ensure_torch'` — the helpers
  didn't exist yet), GREEN post-fix.

Run: `HOME=<isolated> XDG_DATA_HOME=<isolated> .venv/bin/python -m pytest
Tests/Performance/test_app_import_weight.py Tests/Performance/test_app_startup_performance.py
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`
→ **11 passed, 1 xfailed**.

Feature-regression suites (prove embeddings/summarization/chunking/web-search
still work, same isolated-HOME venv pytest):
- `Tests/RAG_Search/` → 54 passed, 11 skipped
- `Tests/Chunking/` + `Tests/RAG_Admin/` + `Tests/Web_Scraping/` → 98 passed, 1 skipped
- `Tests/RAG/` → 221 passed, 8 skipped
- `Tests/Web_Scraping/` + `Tests/Web_Scraping_Interop/` → 75 passed
- `Tests/Utils/test_optional_deps.py` + `Tests/UI/Embeddings/` → 107 passed
- `Tests/test_enhanced_rag.py` + `Tests/test_embeddings_datatable_fix.py` → 4 passed, 6 skipped
- `Tests/Local_Ingestion/test_ingest_import_weight.py` (pre-existing F3
  precedent for this exact pattern) → 3 passed
- `Tests/Tools_Interop/` (WebSearch_APIs boot path) → 10 passed
- `python -c "import tldw_chatbook.app"` → clean, no heavy-module regression
  beyond the documented gap
- Grepped `Tests/` for any monkeypatching of the module-level `analyze` /
  `chunk_for_embedding` / `AutoModel` / `AutoTokenizer` names that the local-
  import conversion could have broken — none found.

**Zero test failures** across all of the above (only pre-existing skips, for
tests requiring network/real models/optional deps not installed in this
env).

## Concerns / follow-ups

1. **Primary concern**: task 163's regression test as literally specified in
   the plan (full 8-module guard set) does not pass — see "Blocked" above.
   torch/transformers (the two heaviest, ~1,282 of ~1,860 modules) are fully
   eliminated; nltk/scipy/sklearn/pandas remain via the `Chunking/__init__.py`
   chokepoint, tracked via the `xfail` test and this report.
2. `numpy` loads at boot regardless (chromadb depends on it) — not part of
   the plan's guard set and not chased further; flagged in the report for
   transparency.
3. Backlog task 163 should **not** be marked Done until either (a) a
   follow-up task fixes the `Chunking` chokepoint and the `xfail` is
   promoted to a hard assertion, or (b) the acceptance criteria are
   explicitly narrowed to "torch/transformers eliminated" with the
   nltk/scipy/sklearn/pandas gap split into a separate task.

---

# UPDATE — nltk/scipy/sklearn/pandas deferral (scope extension authorized 2026-07-11)

The coordinator authorized extending scope into `tldw_chatbook/Chunking/Chunk_Lib.py`
to close the "Blocked" chain above. Done in a second commit on top of the first.

## What was fixed (`tldw_chatbook/Chunking/Chunk_Lib.py`)

1. **Deferred the module-scope `import nltk`.** Replaced the `try: import nltk;
   from nltk.tokenize import sent_tokenize; NLTK_AVAILABLE = True except
   ImportError: ...` block with:
   - `NLTK_AVAILABLE = importlib.util.find_spec("nltk") is not None` (cheap
     probe, no import; added `import importlib.util` at top).
   - Module-level `nltk = None` + `_ensure_nltk()` helper (`global nltk,
     sent_tokenize; if nltk is not None: return; if not NLTK_AVAILABLE: return
     None; import nltk as _nltk; from nltk.tokenize import sent_tokenize as
     _st; nltk = _nltk; sent_tokenize = _st`) — mirrors the `_ensure_torch()`
     pattern from commit 1.
   - The regex fallback is now a named `_sent_tokenize_fallback(text)` bound to
     module-level `sent_tokenize` up front, so the name always exists and
     behaves identically when nltk is absent; `_ensure_nltk()` rebinds it to
     the real nltk tokenizer on first successful import. Single-arg signature
     preserved exactly (callers pass `language=` only on the real-nltk path;
     the resulting `TypeError` on the fallback is caught by their existing
     try/except, same as before).
2. **Moved the module-scope `ensure_nltk_data()` CALL off boot.** The old
   `ensure_nltk_data()` at module scope did a **punkt network download at
   import time**. Removed the top-level call; made the function idempotent via
   a module-level `_nltk_data_ready = False` flag (set True only on a
   successful find/download) and it now calls `_ensure_nltk()` internally, so
   punkt is found/downloaded on first real chunk instead of at import.
3. **Guarded every nltk use site.** `_ensure_nltk()` + `ensure_nltk_data()`
   added at the top of `_adaptive_chunk_size_nltk` (before its `sent_tokenize`)
   and `_semantic_chunking` (before its two `sent_tokenize` calls). Grep
   confirmed these are the only `sent_tokenize(`/`nltk.` use sites in the file.
   `language_chunkers.py` already imports nltk lazily inside a method
   (`get_safe_import('nltk')` in `__init__`, `from nltk.tokenize import
   sent_tokenize` inside `tokenize_sentences`) — not a boot edge, left as-is.

Test touched: `Tests/Utils/test_startup_polish_regressions.py::test_nltk_download_false_is_not_logged_as_success`
directly calls `Chunk_Lib.ensure_nltk_data()`; added a `monkeypatch.setattr(
Chunk_Lib, "_nltk_data_ready", False)` so the now-idempotent function still
exercises the real find/download path regardless of test ordering.

## pandas

pandas was **also** an nltk-transitive dependency — deferring nltk removed it
too (no independent executing importer). Verified with a finder that fires on
module *execution* (real `__import__`, not `find_spec`): a plain `import
tldw_chatbook.app` triggers **zero** executing imports of pandas/nltk/scipy/
sklearn. So `pandas` is kept in the test's forbidden set (`HEAVY_MODULES`) and
passes. The `find_spec`-based `PANDAS_AVAILABLE` probe added to
`Article_Extractor_Lib.py` in commit 1 was never the resident importer — the
resident pandas was the nltk-transitive one, now gone.

`numpy` stays ALLOWED (out of `HEAVY_MODULES`): pulled by chromadb/pymupdf,
comparatively light, legitimate boot dependency.

## Residency table (fresh interpreter, isolated HOME, `import tldw_chatbook.app`)

| module | original (dev) | after commit 1 (torch/transformers) | after commit 2 (nltk) |
|---|---|---|---|
| torch | loaded | **absent** | absent |
| transformers | loaded | **absent** | absent |
| nltk | loaded | loaded | **absent** |
| scipy | loaded | loaded | **absent** |
| sklearn | loaded | loaded | **absent** |
| pandas | loaded | loaded | **absent** |
| docling | absent | absent | absent |
| torchvision | absent | absent | absent |
| numpy | loaded | loaded | loaded (allowed) |
| **module count** | **6,519** | 4,659 | **3,291** |
| **import time** | ~5.65s | ~4.11s | **~1.48s** |

Net: **6,519 → 3,291 modules (−49.5%)**, ~5.65s → ~1.48s (fresh-dir,
apples-to-apples; time is noisy but consistently far lower). All eight
HEAVY_MODULES now absent at boot — **AC #1 satisfied**.

## Behavior-identical verification (original vs lazy-nltk, side by side)

Same script run against stashed-original and new `Chunk_Lib.py`:

| | original | new |
|---|---|---|
| nltk loaded after `import Chunk_Lib` | **True** | **False** |
| words chunking | 11 chunks | 11 chunks |
| adaptive chunking (nltk path + LookupError→non-punkt fallback) | 1 chunk | 1 chunk |
| semantic chunking | LookupError | LookupError |
| nltk/scipy/sklearn loaded after real chunk | True | True |

The only difference is the intended one (nltk no longer at import; loads on
first chunk that needs it). The semantic-chunking `LookupError` is a
**pre-existing** repo bug unrelated to this task — `ensure_nltk_data()`
downloads the legacy `punkt` resource but the installed nltk 5.x's
`sent_tokenize` requires `punkt_tab`; `_semantic_chunking`'s inner
`sent_tokenize(..., language='english')` (in its `except LookupError` handler)
re-raises unguarded. Reproduces identically on the original code, so it is out
of scope here — flagged as a documented follow-up (change the two `punkt`
strings to `punkt_tab`, or add `nltk.download('punkt_tab')`).

## RED evidence for the extended test

`test_app_import_does_not_load_full_heavy_dependency_set` was promoted from
`xfail` to a hard assertion (asserts `loaded_heavy == []` over the full
`HEAVY_MODULES` set). RED-verified: `git stash push -- Chunking/Chunk_Lib.py`
(revert only the nltk fix, keep the test), run → FAILS with `import
tldw_chatbook.app eagerly loaded heavy modules: ['nltk', ...]`. Restore →
PASSES. `MAX_MODULE_COUNT` tightened 5,200 → 4,000 (boot is now ~3,291).

## Updated test results

- `Tests/Performance/test_app_import_weight.py` → **6 passed** (was 5 passed +
  1 xfailed; the full-set guard is now a passing hard assertion).
- Feature-regression: `Tests/Chunking/` + `Tests/RAG/` + `Tests/RAG_Admin/` +
  `Tests/RAG_Search/` + `Tests/Web_Scraping/` + `Tests/Utils/test_startup_polish_regressions.py`
  + `Tests/Utils/test_optional_deps.py` → **404 passed, 20 skipped, 0 failed**.
- End-to-end: real words + adaptive chunking work (adaptive exercises
  `_ensure_nltk()` + `ensure_nltk_data()` + real `sent_tokenize`); nltk absent
  at boot, present after first chunk.
- `python -c "import tldw_chatbook.app"` clean; residency confirms all eight
  HEAVY_MODULES absent, numpy present.

## Remaining concern (documented follow-up, not blocking)

Semantic chunking (`method='semantic'`) is broken by the pre-existing
`punkt` vs `punkt_tab` nltk-5.x mismatch in `ensure_nltk_data()` /
`_semantic_chunking` — reproduces on original code, unrelated to this task.
Recommend a small follow-up to update the punkt resource name.
