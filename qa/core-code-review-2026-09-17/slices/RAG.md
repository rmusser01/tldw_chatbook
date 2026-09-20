# RAG — tldw_chatbook/RAG_Search/ (46 files), 27572 lines

Worktree: /Users/macbook-dev/Documents/GitHub/tldw-review @ d8fb4053f9 (read-only). Report written incrementally.

## Coverage

Honest summary first: **~12 900 of 27 572 lines read line-by-line (47%)**; another ~7 400 lines sampled by symbol cluster around a specific question; the remainder scanned mechanically (pattern greps, symbol outlines, reachability greps). The brief's "read every file over 600 lines in full" rule is met for 6 of the 14 such files; the 8 misses are named below with the reason. Depth was spent preferentially on the files where a finding turned out to live, and on establishing which parts of this package are reachable at all — that reachability question consumed a large share of the run and is itself three of the findings.

| file | lines | coverage |
|---|---|---|
| simplified/rag_service.py | 4426 | **read in full** (run 1) |
| ingestion_indexing.py | 1780 | **read in full** (4 passes: 1-450, 450-900, 900-1400, 1400-1780) |
| simplified/vector_store.py | 1691 | **read in full** (run 1) |
| reranker.py | 1430 | sampled — 1-170 (module contract + `RerankingConfig`), full symbol outline, 182-360 (`BaseReranker`), 360-830 (`_call_llm_impl`, `PointwiseReranker`, `PairwiseReranker`), 830-1100 (`ListwiseReranker`, cross-encoder loading). **Not read: 1100-1430** (`CrossEncoderReranker` body, factories). |
| config_profiles.py | 1384 | sampled — 1-240, 760-1384 read line-by-line; **250-760 (the 13 built-in profile definitions) scanned by grep only** (`RAGConfig()` + `embedding.model` assignments). |
| simplified/simple_cache.py | 1224 | **read in full** (outline + 116-340, 340-620, 620-930, 930-1224) |
| simplified/embeddings_wrapper.py | 1168 | sampled — 1-220, 220-470, 470-660, 648-760, 760-1000. **Not read: 1000-1168** (`MockEmbeddingsService`, `normalize_embeddings`, module tail). |
| simplified/config.py | 980 | **read in full** (1-200, 200-560, 560-980) |
| pipeline_functions_simple.py | 914 | mechanical only — outline + targeted greps. Deprioritized after establishing production-unreachability (see that finding). |
| pipeline_loader.py | 777 | mechanical only — importer/reachability greps + 80-90, 340-410, 600-615, 740-755. Same reason. |
| pipeline_builder_simple.py | 766 | sampled — 375-400 (`_rrf_merge_parallel_results`), 505-550 (`BUILTIN_PIPELINES`), 580-590, 660-670. Same reason. |
| simplified/active_config.py | 654 | sampled — 1-120 (module contract, pointer/marker helpers) + full-file greps for config reads, swallowed returns, mkdir, imports. **Not read: 120-654.** |
| recovery.py | 637 | sampled — 1-120 (`projection_ready`, `_rows`, `_written_rows`) + full-file SQL/exception greps. **Not read: 120-637.** |
| simplified/enhanced_rag_service_v2.py | 626 | sampled — 1-360 (imports, `_tag_first_result`, `__init__`, `_configure_reranker`, `from_profile`, `search` head), 470-530 (`index_batch_optimized`, experiments). **Not read: 360-470, 530-626.** |
| model_recovery.py | 603 | sampled — 1-80 (`_closure`, the O_NOFOLLOW file walk) + greps. **Not read: 80-603.** |
| parallel_processor.py | 585 | **read in full** |
| local_citation_capture.py | 557 | mechanical only |
| simplified/enhanced_rag_service.py | 509 | mechanical only (outline + importer greps) |
| parent_child_adapter.py | 463 | mechanical only (docstring + importer greps) |
| eval/regression.py | 455 | sampled — 30-50, 190-200, 390-435 (`_get_baseline_path`, `_save_atomic`) |
| simplified/health_check.py | 453 | sampled — outline, 49-70, 90-130, 425-453 (enough to establish reachability + the `init_health_checker` global) |
| activation.py | 433 | sampled — 120-175 (`source_paths` tail, `_identity`, `_Execution`, `execution`), 256-292 (`guarded`/`async_guarded`) |
| simplified/enhanced_indexing_helpers.py | 423 | mechanical only (import + `generate_embeddings_batch` resolution greps) |
| generation.py | 420 | sampled — 1-50 (`register_service`, `_SERVICES` weakness, `_same_collection`) |
| fusion.py | 370 | sampled — 225-370 (`resolve_hybrid_alpha`, `_shipped_rrf_k`, `resolve_rrf_k`) |
| semantic_availability.py | 361 | mechanical only (+ 155-165 read for the profile-switch cache comment) |
| simplified/circuit_breaker.py | 352 | mechanical only |
| simplified/citations.py | 323 | mechanical only |
| simplified/indexing_helpers.py | 319 | sampled — 1-60, 150-319 (`store_documents_batch` in full; the embedding-failure padding) |
| eval/gating.py | 309 | sampled — 40-90, 115-160 |
| eval/metrics.py | 275 | mechanical only |
| chunking_service.py | 255 | **read in full** |
| simplified/collection_indexes.py | 232 | mechanical only |
| simplified/search_service.py | 202 | **read in full** |
| pipeline_integration.py | 184 | **read in full** |
| simplified/rag_factory.py | 166 | mechanical only (+ `:52` read) |
| backfill.py | 155 | mechanical only |
| simplified/collection_fingerprint.py | 131 | mechanical only (+ `_index_fields` referenced from two findings) |
| RAG_Search/__init__.py | 130 | sampled — 80-130 (the PEP 562 lazy `__getattr__`) |
| enhanced_chunking_service.py | 121 | **read in full** |
| simplified/__init__.py | 102 | **read in full** |
| pipeline_types.py | 81 | mechanical only |
| simplified/db_connection_pool.py | 65 | **read in full** |
| search_modes.py | 45 | mechanical only |
| simplified/data_models.py | 30 | mechanical only |
| eval/__init__.py | 6 | **read in full** |

Whole-package mechanical sweeps run over all 46 files (these are the basis for several "verified-fine" rows): SQL + `fetchall`/`LIMIT`; `except Exception` followed by `pass`/`return <falsy>`; bare `except`; mutable default arguments; `re.compile` placement; `run_worker`/`@work`; loguru-vs-stdlib-logging; raw HTTP clients; `get_cli_setting` call counts; `atomic_file_ops` usage; importer/reachability greps for every module in the package.

## Findings

### P1 [D1] — The Library Search/RAG semantic query runs ChromaDB's synchronous `query()` directly on the Textual event loop; the first search of a session freezes the UI for ~212 ms
- Where: `tldw_chatbook/RAG_Search/simplified/rag_service.py:1592` (`search_with_citations`) and `:1600` (`search`), both inside `async def _semantic_search` (L1552). Store side: `tldw_chatbook/RAG_Search/simplified/vector_store.py:470` `ChromaVectorStore.search` (sync) and `:575` `search_with_citations` (sync), whose `client` property (`vector_store.py:280`) lazily imports `chromadb` and opens `PersistentClient` on first use.
- Caller chain (traced, not inferred): `UI/Screens/library_screen.py:34871` `@work(exclusive=True, group="library_rag_search") async def _execute_library_rag_search` (a **coroutine** worker → Textual event loop, not a thread) → `Library/library_rag_service.py:94 run_library_rag_search` → `Library/library_local_rag_search_service.py:885` `await rag_service.search(..., search_type="semantic")` → `rag_service.search` → `_semantic_search` → the sync store call above.
- Evidence:
  - `grep -n "async def _execute_library_rag_search" -B1 tldw_chatbook/UI/Screens/library_screen.py` → `34871: @work(exclusive=True, group="library_rag_search")` / `34872: async def _execute_library_rag_search(`
  - Measured, isolated env, 20 000 chunks × dim 384 persisted Chroma collection, fresh interpreter, a 5 ms asyncio ticker running alongside:
    ```
    COLD first search_with_citations on the loop: 212 ms (20 results)
    WARM subsequent search_with_citations:        3.5 ms
    max event-loop tick lag observed:             215 ms (5 ms target tick)
    ```
    (script: build store with `ChromaVectorStore(...).add()` ×20 batches, then `asyncio.run` a ticker + two `vs.search_with_citations(q,"query",20)` calls; full command in "Left UNVERIFIED"/notes below)
- Why it matters: the whole UI (spinner, keystrokes, the search's own cancel path) is stalled for the full store call; the cold cost scales with collection size because it includes the HNSW index load, and every scoped search multiplies it — `library_local_rag_search_service.py:893-901` issues **one store query per source type** in a Python `for` loop, all of them on the loop.
- Asymmetry that shows this is an oversight rather than a decision: the sibling calls on the very same path are deliberately offloaded — `library_local_rag_search_service.py:1092 service = await asyncio.to_thread(get_shared_rag_service)` ("First-time construction … runs in `asyncio.to_thread` -- never on the UI event loop") and `:1122 stats = await asyncio.to_thread(get_stats)` ("ChromaDB-backed stats can touch disk; keep it off the event loop"). The embedding half of `_semantic_search` is offloaded too (`embeddings_wrapper.py:747 await asyncio.to_thread(native_worker(...))`). Only the query is not.
- Recommended correction: wrap the two store calls in `_semantic_search` in `await asyncio.to_thread(...)` (one `functools.partial` each, ~6 lines). That is the single choke point — every semantic caller routes through `_semantic_search`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found (`rg -n "to_thread" Tests/RAG*` → no test asserts the sync shape)
- Already covered: none

### P1 [D1] — the RAG search cache leaks its memory accounting on every TTL prune and eventually stops caching permanently; every search after that is a cold miss
- Where: `tldw_chatbook/RAG_Search/simplified/simple_cache.py:1055-1088` (`_prune_expired_async`) and `:1090-1126` (`prune_expired`) delete entries from `self._cache` **without** decrementing `self._current_memory_bytes`. The sync twin `_prune_expired_sync` (`:1128-1150`) does it correctly — it calls `_update_memory_sync()` at `:1147`. The counter is then the gate on `put_async`'s eviction loop (`:642-644`) and on its give-up branch (`:646-651`, `logger.warning("Entry too large for cache") ; return`).
- Live path: `RAGService.__init__` (`rag_service.py:816`) constructs `SimpleRAGCache` with the default `max_memory_mb=100.0`; the production search path uses `get_async`/`put_async` only (stated at `simple_cache.py:748-752` and re-verified there), and `get_async:378-381` is what fires `_prune_expired_async`.
- Evidence — reproduced, isolated env, `max_memory_mb=1.0`, `ttl_seconds=0.2` (only to shorten the clock; nothing else changed):
  ```
  round 1: entries in cache =  0, _current_memory_bytes =    637.5 KB (cap 1024 KB)
  round 2: entries in cache =  0, _current_memory_bytes =   1020.0 KB (cap 1024 KB)
  round 3: entries in cache =  0, _current_memory_bytes =   1020.0 KB (cap 1024 KB)
  ... (rounds 4-6 identical)

  after the drift, is 'final' actually cached? -> False
  entries now: 0  _current_memory_bytes: 1020.0 KB
  get_async('final') returns: None  <-- cache is dead
  ```
  (script: 6 rounds of 5 `put_async` + `sleep(0.25)` + one `get_async` to fire the prune, then one more `put_async`/`get_async`. The counter reports 1020 KB of resident entries while the cache holds **zero**.)
- Time to failure under the shipped defaults: one cached entry for a `top_k=10` search with ~2 KB of text per chunk measures 31.9 KB via `_deep_getsizeof`, so the 100 MB cap is reached after ~3 200 leaked entries — about **32 full-cache prune cycles**, and a prune fires at most once per `_prune_interval = min(ttl/2, 1800)` = 30 minutes. A long-lived TUI session (this app runs for days) gets there; a short one does not. There is no recovery: nothing on the async path ever recomputes `_current_memory_bytes`, so once it ratchets up it stays up for the life of the process, and it only ever goes down by an eviction's share.
- Second-order effects before total failure: the effective cache budget shrinks monotonically with every expiry (so the cache degrades progressively, not only at the end), and `get_metrics()["memory_usage_percent"]` / `log_gauge("cache_memory_estimate_mb", …)` report a number that is wrong in the same direction the whole time.
- Recommended correction: make `_prune_expired_async` and `prune_expired` decrement the counter the way `_prune_expired_sync` already does — accumulate the pruned entries' sizes and subtract, or (simplest) recompute from the surviving entries. The two public prune bodies are also byte-identical duplicates of each other (`:1055-1088` vs `:1090-1126`), so the fix belongs in one shared body, not three.
- A companion inconsistency worth fixing in the same change: the async path sizes entries with `_deep_getsizeof` (`:604` region, a real object-graph walk) while the sync path uses `_estimate_entry_size` (`:970-1002`, a flat 1 KB per result), and both mutate the same `_current_memory_bytes`. Mixed use would make the accounting drift a second way. Production only uses the async path today, so this one is latent.
- Size: S · ADR: no · Confidence: verified (reproduced)
- Pinning test: none — `grep -rn "_current_memory_bytes" Tests/` finds no assertion on the counter after a prune.
- Already covered: none. (TASK-15701 covered the sync twins' *cache key*, a different defect in the same file.)

### P2 [D1] — `ChromaVectorStore.search` converts every store failure into "no results"; the Library UI then renders the verified-empty-index message
- Where: `tldw_chatbook/RAG_Search/simplified/vector_store.py:568-571` (`except Exception as e: … return []`); same shape at `:1527` (stats, benign).
- Evidence: read; the `except` block logs at ERROR and `return []`. `search_with_citations` (`:602`) calls `search` and therefore inherits it. Downstream, `Library/library_local_rag_search_service.py:907` treats `not raw_results` plus a zero `get_collection_stats` count as "Index empty"; with a corrupted/locked Chroma directory `get_collection_stats` also fails → `_semantic_index_is_empty` returns False → the user gets the generic "0 results" outcome for what is actually a broken store.
- Why it matters: a broken vector store is indistinguishable from an empty corpus at the UI; the user re-indexes (or gives up) instead of seeing the real error.
- Recommended correction: let the exception propagate (the callers already have a `except Exception` → "Retrieval failed / Retry" recovery outcome at `library_rag_service.py:152`), or return a sentinel the caller can distinguish. Keep the `return []` only for the "collection does not exist yet" case.
- Size: M · ADR: no · Confidence: inferred (the swallow is verified by reading; the UI-message consequence is traced through code, not reproduced live)
- Pinning test: none
- Already covered: none

### P2 [D1] — a profile switch retires the shared RAG service without ever calling `close()`, and a module global in `health_check.py` pins the retired instance past garbage collection
- Where: `tldw_chatbook/RAG_Search/ingestion_indexing.py:483-489` (`set_shared_rag_service` / `reset_shared_rag_service` — reassign the global, no `close()`), against `tldw_chatbook/RAG_Search/simplified/rag_service.py:850 init_health_checker(self)` → `simplified/health_check.py:437-440` (`_health_checker = RAGHealthChecker(rag_service)`, a module-level **strong** reference; `health_check.py:63 self.rag_service = rag_service`).
- Shipped callers of the retirement path: `UI/Screens/settings_rag_profile_adapter.py:471` (Settings ▸ RAG save) and `RAG_Search/simplified/active_config.py:373` (`set_active_profile`).
- Evidence (isolated env, `create_config_for_testing()` → mock embeddings + in-memory store):
  ```
  service 1 still alive after del + gc: True
    held by health_check._health_checker.rag_service: True
    its ThreadPoolExecutor shut down: False
  after a SECOND service is built, service 1 alive: False
    _health_checker now points at service 2: True
  ```
- Why it matters: `RAGService.close()` shuts down the per-service `ThreadPoolExecutor` (`rag_service.py:855`, up to 8 threads) and releases the embeddings handle, the Chroma client, and the DB connection pools. A profile switch drops the only *intended* reference and calls none of that, so the previous profile's loaded embedding model and open Chroma client stay resident until the next service is built (which is what finally overwrites `_health_checker` and lets GC run). The module already knows this rule and applies it on the *other* retirement path: `ingestion_indexing.py:225-250 _close_discarded_rag_service` closes a race-losing build and documents "a discarded build is therefore not actually resource-free". The main retirement path skips it.
- Recommended correction: have `set_shared_rag_service` close the instance it displaces, using the existing `_close_discarded_rag_service` helper and its documented rule (call it OUTSIDE `_shared_service_lock`, since `close()` blocks on `ThreadPoolExecutor.shutdown`). Separately, `init_health_checker` should hold a `weakref` (or the global should be dropped with the service) — nothing reads it anyway (see the next finding).
- Size: M · ADR: no · Confidence: verified
- Pinning test: none (`grep -rn "reset_shared_rag_service" Tests/` finds isolation fixtures, none asserting close-on-retire)
- Already covered: none

### P2 [D1] — a misconfigured reranker provider reports `UnboundLocalError: cannot access local variable 'response'` instead of the real error
- Where: `tldw_chatbook/RAG_Search/reranker.py:620-634` (`PointwiseReranker._score_result`): `response = await self._call_llm(prompt)` is inside the `try`, and the `except (json.JSONDecodeError, ValueError, KeyError)` handler's log line reads `len(response)` and `content_fingerprint(response)` — both unbound when `_call_llm` itself raised.
- Evidence (isolated env, `_call_llm` patched to raise the exact error the dispatcher produces):
  ```
  $PY -c "... patch PointwiseReranker._call_llm -> raise ValueError('Unsupported API endpoint: foo') ...
           await r._score_result('q', res, 0)"
  → RAISED UnboundLocalError: cannot access local variable 'response' where it is not associated with a value
  ```
  That `ValueError` is real, not invented: `Chat/Chat_Functions.py:1073` raises `ValueError(f"Unsupported API endpoint: {endpoint_display}. Valid endpoints: …")` whenever `API_CALL_HANDLERS` has no handler for the configured `model_provider`, and `BaseReranker._call_llm` (`reranker.py:301-330`) re-raises it after exhausting `max_retries`.
- Why it matters: the result row still degrades correctly (`asyncio.gather(..., return_exceptions=True)` at `:539` catches it and the row is counted failed), but the ONLY diagnostic the user and the log get is `Failed to score result 0: cannot access local variable 'response'` — the message that would have named the misconfigured provider and listed the valid ones is destroyed. This is a setup-error path, which is exactly where a useful message matters most, and the same `reranker.py` header records TASK-17065 fixing a previous version of "reranking silently reached 0 of the 29 providers".
- Recommended correction: initialise `response = ""` before the `try`, or narrow the handler to wrap only `json.loads`/`float` (the parse errors it is for) and let a call failure propagate to `gather`, where it is already handled and logged with its own type and message.
- Size: S · ADR: no · Confidence: verified (reproduced)
- Pinning test: `Tests/RAG_Search/test_reranker_degraded_paths.py` exists and pins the degraded-rerank contract; it does not cover a raising `_call_llm` at this call site (the repro above goes red today).
- Already covered: none

### P2 [D1] — `BaseReranker._cache` is an unbounded dict with no size limit, TTL, or eviction, on a process-lifetime singleton
- Where: `tldw_chatbook/RAG_Search/reranker.py:188` (`self._cache = {} if config.cache_results else None`, and `RerankingConfig.cache_results` defaults to `True` at `:176`); written at `:568` (`PointwiseReranker`) and `:1282` (`CrossEncoderReranker`).
- Evidence: `grep -n "_cache" tldw_chatbook/RAG_Search/reranker.py` → 13 hits: one construction, two membership tests, two reads, two writes, and the `_get_cache_key`/`_cross_encoder_cache_key` helpers. **No eviction, no cap, no TTL, no `clear()` anywhere in the file.** The owning reranker is built once in `EnhancedRAGServiceV2._configure_reranker` (`enhanced_rag_service_v2.py:215-231`) and lives for the life of the shared service.
- Why it matters: every distinct (query, result-id-set) adds a `List[RerankingResult]` of up to `top_k_to_rerank` (default 20) entries, each of which can carry the model's `reasoning` text when `include_reasoning` is on. Nothing ever removes them. Contrast `SimpleRAGCache` in the same package, which has `max_size`, a TTL, *and* a memory cap for the same kind of payload. Reranking is opt-in (`SearchConfig.enable_reranking` defaults to `False` and a profile must carry a `reranking_config`), so this only bites users who turned it on — which is why it is P2 and not P1.
- Recommended correction: reuse the bounded cache that already exists rather than adding another one — an `OrderedDict` with `max_size` and `move_to_end`, or `functools.lru_cache`-style bounding, is a ~5-line change at `:188`/`:568`/`:1282`. (Fix `SimpleRAGCache`'s accounting first — see the P1 above — if it is to be reused directly.)
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P2 [D3] — `parallel_processor.py` (585 lines) is a dead subsystem: the only thing it exports that runs is the `ProcessingConfig` dataclass; the pipeline it exists for is documented as "not implemented"
- Where: `tldw_chatbook/RAG_Search/parallel_processor.py` (whole file). Its two constructors are called at `simplified/enhanced_rag_service_v2.py:201` (`create_embedding_processor`) and `:204` (`create_chunking_processor`), under the default `enable_parallel_processing=True`.
- Evidence:
  - `grep -rn "embedding_processor|generate_embeddings_batch|EmbeddingBatchProcessor|process_documents_parallel" tldw_chatbook/ Tests/` → the only assignments of `self.embedding_processor` are `enhanced_rag_service_v2.py:197,201`; **no read of it anywhere**. `self.chunking_processor` is read exactly once, at `:486`, inside a branch whose body is `logger.debug("Parallel batch-indexing pipeline is not implemented; using base optimized path")`. The `generate_embeddings_batch` hits in `rag_service.py:88/1152` and `enhanced_indexing_helpers.py:196/213` resolve to `simplified/indexing_helpers.py:93`, a **different** function.
  - The module's own docstring at `enhanced_rag_service_v2.py:474-484` records that the parallel branch was removed in task-247 because it was "broken in both directions".
- Why it matters, concretely (three live costs for zero function):
  1. **Misleading INFO log on every RAG service construction**: `create_chunking_processor` → `ChunkingBatchProcessor.__init__` → `BatchProcessor._determine_worker_count()` (`parallel_processor.py:133-148`) emits `Using N workers (CPUs: M)`, then `enhanced_rag_service_v2.py:205` emits `Initialized parallel processors with N workers`. Nothing parallel ever runs.
  2. **A live circular-import edge** — see the next finding.
  3. It is untested-and-broken code that a future contributor could re-enable: `ChunkingBatchProcessor.chunk_documents_batch` (`:457`) submits a **closure** (`process_document`, defined inside the method) to `ProcessPoolExecutor.submit` (`:196`); closures are not picklable, so that path could never have run. `EmbeddingBatchProcessor.generate_embeddings_batch` (`:298`) also confuses two index spaces: on a failed batch it does `failed_indices.extend(range(i, min(i + batch_size, total)))` at `:353` where `i` is an index into `tasks` (batches), not into `texts`, and the successful embeddings of surviving batches are appended positionally, so any failure silently shifts every later embedding onto the wrong text.
- Recommended correction: delete `parallel_processor.py` except `ProcessingConfig` (its only live consumer is `config_profiles.ProfileConfig.processing_config`), move that dataclass to `RAG_Search/simplified/config.py` next to the other config dataclasses, and drop the two `create_*_processor` calls and the `enable_parallel_processing` flag from `EnhancedRAGServiceV2`. Deletion, not repair — nothing asks for the feature.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none for the processors themselves; `Tests/RAG_Search/test_reranker_construction.py` patches `select_profile_for_experiment`/`record_experiment_result` on a fake manager but asserts nothing about parallel processing.
- Already covered: none

### P2 [D3] — `import tldw_chatbook.RAG_Search.parallel_processor` raises ImportError from a circular import; the module is only importable because every shipped caller happens to import `simplified` first
- Where: `tldw_chatbook/RAG_Search/parallel_processor.py:28` `from .simplified.data_models import IndexingResult` → executes `simplified/__init__.py:35` `from .enhanced_rag_service_v2 import EnhancedRAGServiceV2` → `enhanced_rag_service_v2.py:30` `from ..parallel_processor import (create_embedding_processor, …)` on a partially-initialized module.
- Evidence: `cd <worktree> && source env.sh && PYTHONPATH=<worktree> $PY -c "import tldw_chatbook.RAG_Search.parallel_processor"` →
  ```
  File ".../RAG_Search/simplified/enhanced_rag_service_v2.py", line 30, in <module>
      from ..parallel_processor import (
  ImportError: cannot import name 'create_embedding_processor' from partially initialized module
  'tldw_chatbook.RAG_Search.parallel_processor' (most likely due to a circular import)
  ```
- Why it matters: this is the *third* instance of the exact cycle this package has already been burned by twice — `enhanced_rag_service_v2.py:36-55` documents task-21160 deferring `config_profiles`, and `:64-84` documents deferring `reranker`, both after `task-21102` made `RAG_Search/__init__` lazy and unmasked the latent order dependency. The remaining edge is live today; any new module that imports `parallel_processor` before `simplified` (a script, a test, a CLI entry point) fails at import.
- Recommended correction: if the module survives at all (see the previous finding, which deletes most of it), defer `IndexingResult` the same way its two siblings are deferred — `if TYPE_CHECKING:` for the annotation plus a function-body import at the one runtime use (`parallel_processor.py:480`). If the module is deleted, the cycle goes with it.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (no test imports `parallel_processor` first — which is exactly why this survived)
- Already covered: none

### P2 [D3] — `simplified/health_check.py` (453 lines) is constructed on every RAG service build and read by nothing
- Where: `tldw_chatbook/RAG_Search/simplified/health_check.py` (whole file); constructed at `simplified/rag_service.py:850`; the only reader is `RAGService.get_health_status()` (`rag_service.py:4341-4348`).
- Evidence: `grep -rn "get_health_status" tldw_chatbook/ Tests/` → exactly two hits, both inside `rag_service.py` (the method definition and its one-line body). `grep -rn "RAGHealthChecker|HealthStatus|ComponentHealth" tldw_chatbook/ Tests/` outside `health_check.py` → **zero**. Not exported from `simplified/__init__.py` or `RAG_Search/__init__.py` (`grep -n health` on both → no match).
- Why it matters: 453 lines of unexercised code that nonetheless runs its constructor on every service build and installs the module global behind the leak above. Its `get_health_sync()` (`:425-430`) would also spin up a fresh event loop with `asyncio.new_event_loop()` + `run_until_complete` — calling it from the Textual event loop thread would raise; nothing calls it, so that is latent rather than live.
- Recommended correction: delete the module and `RAGService.get_health_status()`, or wire it to the RAG admin surface if the diagnostics are actually wanted. Do not leave it half-attached.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P2 [D3] — `pipeline_integration.py` (184 lines) has zero importers anywhere, and one of its branches imports a module that no longer exists
- Where: `tldw_chatbook/RAG_Search/pipeline_integration.py` (whole file); the stale import is `:122` `from ..Event_Handlers.Chat_Events.chat_rag_events_simplified import perform_search_with_pipeline`.
- Evidence:
  - `grep -rn "pipeline_integration|PipelineManager|get_pipeline_manager|reload_all_pipelines|get_available_pipeline_ids" tldw_chatbook/ Tests/` (excluding the file itself) → **no output**. Not re-exported from `RAG_Search/__init__.py` either.
  - `$PY -c "import importlib; importlib.import_module('tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events_simplified')"` → `ModuleNotFoundError: No module named 'tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events_simplified'`. The module was renamed to `chat_rag_events.py`, which still carries the old name in its header comment (`:1`) and in `logger.bind(module="chat_rag_events_simplified")` (`:70`). `perform_search_with_pipeline` lives at `chat_rag_events.py:383`.
- Why it matters: this is the exact "function-body import of a module that no longer exists, which mocked tests never catch" shape from the review brief, and it survives only because the entire module is unreachable. The sibling modules of the same subsystem (`pipeline_builder_simple`, `pipeline_functions_simple`, `pipeline_types`, `pipeline_loader`) ARE live via `Event_Handlers/Chat_Events/chat_rag_events.py:54-55`, so this is a stranded integration layer, not a dead subsystem.
- Recommended correction: delete the file. If any of `PipelineManager`'s legacy-mode mapping is wanted, it belongs in `pipeline_loader.py` next to the loader it wraps.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P2 [D3] — `pipeline_loader.py` (777 lines) is reachable only through the dead `pipeline_integration.py`; the whole legacy `perform_*` pipeline path hangs off an entry point nothing in production calls
- Where: `tldw_chatbook/RAG_Search/pipeline_loader.py` (777), `pipeline_integration.py` (184); downstream `pipeline_builder_simple.py` (766) and `pipeline_functions_simple.py` (914) are reached only via `Event_Handlers/Chat_Events/chat_rag_events.py`'s `perform_plain_rag_search` / `perform_full_rag_pipeline` / `perform_hybrid_rag_search`.
- Evidence (each a grep over `tldw_chatbook/` with the defining file excluded):
  - `PipelineLoader|get_pipeline_loader|get_pipeline_function` → 3 hits, **all** in `pipeline_integration.py` (`:13`, `:22`, `:47`), which itself has zero importers (previous finding).
  - `perform_plain_rag_search|perform_full_rag_pipeline|perform_hybrid_rag_search` → definitions at `chat_rag_events.py:177/225/279`; the only `await` call sites are `chat_rag_events.py:1959/1972/1989`, all inside `get_rag_context_capture_for_chat` (`:1811`).
  - `get_rag_context_capture_for_chat` → called only by `get_rag_context_for_chat` (`:2054`); `get_rag_context_for_chat` repo-wide (all file types, excluding `Tests/`, `Docs/`, `backlog/`) → definition plus two docstring mentions and **no caller**. Its only callers are `Tests/RAG/test_rag_ui_integration.py` and `Tests/RAG/test_rag_dependencies.py`.
  - The live Console/Library retrieval path goes elsewhere: `UI/Console_Modules/retrieval.py:36` and `Chat/console_runtime.py:2476` reach `Library/library_rag_service.run_library_rag_search` → `Library/library_local_rag_search_service` → `RAGService.search`. `capture_console_staged_evidence_for_chat` (`chat_rag_events.py:1659`) authorizes already-staged evidence and runs no pipeline.
- Limit of the method (stated honestly): `pipeline_loader` resolves pipeline functions **by name** from `tldw_chatbook/Config_Files/rag_pipelines.toml` (`:41 function = "perform_hybrid_rag_search"`), so a name-based registry is in play — but the only code that consults that registry is `pipeline_integration.py`, which nothing imports. I did not execute the app to confirm.
- Why it matters: ~2 600 lines of this slice (plus `pipeline_types.py`) exist to serve a chat-RAG entry point no shipped surface calls, while the surface that IS live uses a different engine path entirely. That is a large maintenance surface and a standing source of "which RAG path is the real one?" confusion — `pipeline_functions_simple.py` and `rag_service.py` already carry cross-referencing comments explaining that they are two implementations of the same fusion.
- A concrete cost, if the path IS live: `chat_rag_events.py:346 resolve_hybrid_alpha(hybrid_alpha)` is called with `hybrid_alpha=None` by default, and `pipeline_builder_simple.py:393 resolve_rrf_k(merge_config.get("rrf_k"))` gets `None` because `BUILTIN_PIPELINES["hybrid"]`'s merge step config sets `alpha` (pinned at `chat_rag_events.py:362-365`) but never `rrf_k`. Each `None` sends `fusion.py` into `resolve_active_rag_config()`. Measured, isolated env, warm caches:
  ```
  resolve_active_rag_config():                    10.38 ms/call (warm)
  resolve_hybrid_alpha(None)+resolve_rrf_k(None): 19.11 ms per hybrid search
  with explicit values (the rag_service path):     0.0004 ms
  ```
  `RAGService._hybrid_search` (`rag_service.py:1325-1326`) passes both explicitly and pays 0.4 µs; the pipeline path pays ~19 ms of config re-reading per query, on the event loop. This matches the sibling reviewer's finding that a cached `get_cli_setting`/`load_settings` still costs ~11 ms per call.
- Recommended correction: decide the path's fate first. If it is dead, delete `pipeline_integration.py`, `pipeline_loader.py`, `Config_Files/rag_pipelines.toml`, the three `perform_*` functions and `get_rag_context_for_chat` together (they only keep each other alive). If it is meant to stay, pin `rrf_k` into `BUILTIN_PIPELINES["hybrid"]`'s merge config beside `alpha` — a one-line fix that removes half the 19 ms.
- Size: L (a deletion this size crosses `RAG_Search` ↔ `Event_Handlers` and needs an owner decision) · ADR: no — but see `backlog/docs/library-decomposition-recipe.md` §1 for the per-subsystem PR shape if it is deleted · Confidence: verified for every grep above; **inferred** for the conclusion "unreachable", because of the name-based TOML registry.
- Pinning test: `Tests/RAG/test_rag_ui_integration.py:147-166` calls `get_rag_context_for_chat` directly and asserts its behaviour — it pins the helper, not any product path.
- Already covered: none. (`pipeline_builder_simple.py:375-386` records that TASK-3501 "intentionally retained this pipeline materializer; do not refactor it speculatively" — that ruling is about not refactoring the fusion, not about the path's reachability.)

### P2 [D4a] — RAG profile JSON is written with a bare truncate-then-write while `Utils/atomic_file_ops.atomic_write_json` exists and is used by 10 other modules
- Where: `tldw_chatbook/RAG_Search/config_profiles.py:652` (`_save_one`, the single choke point every profile save goes through: `save_profile` ← `clone_profile`, `create_custom_profile`, `rename_profile`, `active_config.ensure_imported_profile`, the Settings screen's save path). Same shape at `:822` (legacy-blob migration), `:1096` (experiment config), `:1234` (experiment results), and `pipeline_loader.py:746`.
- Evidence:
  - `sed -n '648,653p' config_profiles.py` → `with self._definition_write(), open(self._profile_path(profile.id), "w") as f: json.dump(profile.to_dict(), f, indent=2, default=str)`
  - `grep -rln "atomic_write_json" tldw_chatbook/` → 10 modules (`Model_Artifacts/service.py`, `Skills_Interop/skill_trust_store.py`, `Chatbooks/local_chatbook_service.py`, `UI/LLM_Management/vllm_profiles.py`, `Web_Server/artifact_share_manifest.py`, …). `Utils/atomic_file_ops.py:199 atomic_write_json` does temp-file + `os.fsync` (`:103`) + `os.replace`.
- Why it matters: a crash, a power loss, or ENOSPC between `open(..., "w")` (which truncates immediately) and the end of `json.dump` leaves a zero-length or half-written profile file. `_load_custom_profiles` (`:761`) catches the resulting `JSONDecodeError`, logs `Failed to load profile <name>`, and **skips** it — the user's saved RAG profile silently disappears from the picker, and the previous good content is already gone.
- Recommended correction: `_save_one` → `atomic_write_json(self._profile_path(profile.id), profile.to_dict(), indent=2, default=str)` (check that helper's kwargs; it may need a `json.dumps` + `atomic_write_text`). Canonical home already exists: `Utils/atomic_file_ops.py`. The three non-profile call sites are lower value (`:1096`/`:1234` write experiment artefacts from a subsystem that never runs — see the dead-subsystem finding).
- Size: S · ADR: no · Confidence: inferred (the non-atomic write and the skip-on-parse-failure loader are both verified by reading; the crash-window loss is not reproduced)
- Pinning test: `Tests/RAG/test_config_profiles.py` covers save/load round-trips but nothing about torn writes.
- Already covered: none

### P2 [D4a] — the RAG embeddings wrapper re-rolls ONE row of `Embeddings_Lib`'s bare-id→HF-path table; the other 13 rows (including `RAGConfig`'s own default model and the Settings placeholder) go to the Hub unqualified
- Where: `tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py:141-143` (`_BARE_HF_MODEL_ID_ALIASES`, one entry) applied at `:432` inside `_build_config`. The canonical table it duplicates a row of is `tldw_chatbook/Embeddings/Embeddings_Lib.py:936-1000` (`get_common_embedding_models()`), plus `:923-930` (`get_default_embedding_config()`).
- Evidence (isolated env):
  ```
  canonical table rows whose ID is bare but whose HF path is org-prefixed: 14
    'mxbai-embed-large-v1' -> 'mixedbread-ai/mxbai-embed-large-v1'   covered by RAG alias table: False
    'e5-small-v2'          -> 'intfloat/e5-small-v2'                 covered by RAG alias table: False
    'all-MiniLM-L6-v2'     -> 'sentence-transformers/all-MiniLM-L6-v2'  covered by RAG alias table: True
    'bge-base-en-v1.5'     -> 'BAAI/bge-base-en-v1.5'                covered by RAG alias table: False
    ... (14 rows, 1 covered)
  ```
  and the mechanism, calling `_build_config` directly:
  ```
  'mxbai-embed-large-v1'   -> model_name_or_path='mxbai-embed-large-v1'
  'all-MiniLM-L6-v2'       -> model_name_or_path='sentence-transformers/all-MiniLM-L6-v2'
  'bge-base-en-v1.5'       -> model_name_or_path='bge-base-en-v1.5'
  RAGConfig() default embedding model: mxbai-embed-large-v1
  ```
- Why it matters: `embeddings_wrapper.py`'s own comment states the consequence for exactly this shape — a bare id "silently 404-ing into the dim=768 default". The uncovered ids are not hypothetical: `simplified/config.py:290 EmbeddingConfig.model` defaults to `"mxbai-embed-large-v1"`, `UI/Screens/settings_library_rag_defaults.py:61` carries the same default, and `UI/Screens/settings_screen.py:19019` shows `placeholder="e.g. mxbai-embed-large-v1"` — so the string a user is most likely to type into the embedding-model field is one of the 13 the alias table does not cover, while `Embeddings_Lib` has known its correct HF path all along. Every shipped built-in profile overrides the model with an id that happens to be covered or already prefixed (`config_profiles.py:260,282,304,326,348`), which is why this has not been hit in the default flow.
- Recommended correction: D4 sub-case (a) — the helper exists and is ignored. Replace `_BARE_HF_MODEL_ID_ALIASES` with a lookup into `Embeddings_Lib.get_common_embedding_models()` (`model_name_or_path` of the matching row), keeping the existing rule that only the HTTP-facing id is rewritten and never `self.model_name` (which determines the collection fingerprint — see `collection_fingerprint._index_fields`). Canonical home: `Embeddings/Embeddings_Lib.py`.
- Size: S · ADR: no · Confidence: verified for the divergence and for the unqualified id `_build_config` emits; **inferred** for the 404 itself (no network in this review).
- Pinning test: none found for the alias table's coverage.
- Already covered: none (task-640 AC#7 added the one-row table; nothing covers the other 13)

### P3 [D3] — the RAG A/B-testing subsystem has no production entry point
- Where: `tldw_chatbook/RAG_Search/config_profiles.py:85-115` (`ExperimentConfig`), `:1093-1146` (`start_experiment`), `:1153-1180` (`select_profile_for_experiment`), `:1182-1210` (`record_experiment_result`), `:1212-1249` (`end_experiment`); wiring at `simplified/enhanced_rag_service_v2.py:499-518`.
- Evidence: `grep -rn "\.start_experiment\(|\.end_experiment\(" tldw_chatbook/` → only `enhanced_rag_service_v2.py:504` and `:513` (the methods' own bodies calling the manager). `grep -rn "start_experiment" Tests/` → `Tests/Backup_Recovery/test_rag_manager_maintenance.py`, `Tests/Backup_Recovery/test_rag_definition_credentials.py` only. No UI, CLI, config key, or event handler ever calls `EnhancedRAGServiceV2.start_experiment`.
- Why it matters: ~250 lines plus a per-search branch (`enhanced_rag_service_v2.py:323`) that can never be true, and an on-disk `rag_profiles/experiments/` tree that is never written. Runtime cost is zero (the branch is guarded by `self._current_experiment and user_id`), so this is a maintenance-surface finding, not a performance one. Note the tests DO exercise it, so "delete" needs the tests deleted with it.
- Recommended correction: delete, or file it as intentionally-parked and say so in the module docstring. Do not "fix" it.
- Size: M · ADR: no · Confidence: verified
- Pinning test: `Tests/Backup_Recovery/test_rag_manager_maintenance.py` exercises the manager's experiment methods directly — it pins the helpers, not any product behaviour.
- Already covered: none

### P3 [D3] — `parallel_processor.py:31` guards an INTERNAL import with `try/except ImportError` and installs a silent no-op stub
- Where: `tldw_chatbook/RAG_Search/parallel_processor.py:31-36`
- Evidence: `grep -n "log_gauge" tldw_chatbook/Metrics/metrics_logger.py` → `185: log_gauge = default_metrics.log_gauge` — a plain module-level name that is always present. The `except ImportError:` arm defines `def log_gauge(...): pass`.
- Why it matters: `metrics_logger` is not an optional dependency; the only way this except can fire is a real breakage (a rename, a circular import), and when it does every `log_gauge` call in this module becomes a silent no-op instead of an error. Same shape flagged in the brief's D3 list.
- Recommended correction: fold `log_gauge` into the unguarded import on `:27`. (Moot if the module is deleted.)
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D3] — `config_profiles.quick_profile` is dead
- Where: `tldw_chatbook/RAG_Search/config_profiles.py:1374`
- Evidence: `grep -rn "quick_profile" tldw_chatbook/ Tests/` → one hit, the definition itself. Not in any `__all__` or re-export (`RAG_Search/__init__.py` does not name it).
- Why it matters: minor; a dead public-looking helper invites use of a path nothing exercises.
- Recommended correction: delete.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D3] — `EmbeddingsServiceWrapper.create_embeddings` hashes the whole batch to probe a cache whose key space it cannot match; `embeddings_cache_hit` can never fire
- Where: `tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py:516-546` (the `cache_key` computation and `is_cached = hasattr(self.factory, "_cache") and cache_key in getattr(self.factory, "_cache", {})`).
- Evidence: `Embeddings/Embeddings_Lib.py:645` → `self._cache: "OrderedDict[str, CacheRecord]" = OrderedDict()`, keyed by **model id** (`self._cache[model_id_to_use]` at `:826`). The wrapper's `cache_key` is a 32-char hex digest of the text batch. Measured, isolated env:
  ```
  after 3 IDENTICAL batches of 32 texts:
    _cache_hits   = 0
    _cache_misses = 3
  cache-key computation cost: 0.014 ms per batch of 32
  ```
- Why it matters: `get_metrics()` (`:824-847`) always reports `cache_hit_rate: 0.0`, and `log_gauge("embeddings_cache_hit_rate", …)` always emits 0 — a permanently-false metric, which is worse than no metric when someone later tunes embedding throughput from it. (It is also the consumer of `health_check.py:112`'s `cache_hit_rate` field.) The CPU cost is negligible (0.014 ms/batch), so this is a correctness-of-diagnostics finding, not an efficiency one.
- Recommended correction: delete the `cache_key` block, the `_cache_hits`/`_cache_misses` counters, and the `cache_hit_rate` field — the underlying factory does not expose a per-batch cache to measure. `hashlib.md5(...)` at `:521` also lacks `usedforsecurity=False` and would raise on a FIPS build; deleting the block removes that too.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D3] — `simple_cache._make_key` advertises xxhash, which is not a dependency of this project and is not installed; every key pays a failed import instead
- Where: `tldw_chatbook/RAG_Search/simplified/simple_cache.py:195` (docstring "Uses xxhash for better performance than MD5"), `:327-335` (the `try: import xxhash / except ImportError:` fallback).
- Evidence: `grep -rn "xxhash" tldw_chatbook/` → 4 hits, all inside `simple_cache.py`. `grep -rn "xxhash" pyproject.toml` → no match (not a required dep, not in any optional group). Isolated env: `importlib.util.find_spec('xxhash')` → `False`; a failed `import xxhash` costs **0.021 ms** per attempt (a failed import is not cached in `sys.modules`, so the finder chain re-runs every call), against a `hashlib.md5` of the same string at roughly a microsecond.
- Why it matters: the fast path is dead in every install, the docstring states the opposite, and the per-key cost is ~15× the hash it was meant to replace. The absolute number is small (a handful of `_make_key` calls per search), so this is a correctness-of-documentation finding, not a hot-path one.
- Recommended correction: delete the try/except and the docstring line, keeping `hashlib.md5(..., usedforsecurity=False)`; or add `xxhash` to an optional group and hoist the import to module scope behind a module-level flag so the failure is paid once.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D4b] — process-RSS-in-MB is re-rolled in 5 places with 3 different divisor spellings; two of the copies are a byte-identical 5-second-TTL cache
- Where: `RAG_Search/simplified/vector_store.py:255-279` and `RAG_Search/simplified/embeddings_wrapper.py:373-397` (verbatim twins, `psutil.Process().memory_info().rss / (1024 * 1024)` behind an identical `self._memory_cache` dict with `ttl: 5.0`); plus `Metrics/metrics_logger.py:170` (`/(1024**2)`), `Metrics/metrics.py:231` (`/(1024**2)`), `Widgets/detailed_progress.py:332` (`/1024/1024`).
- Evidence: `grep -rn "memory_info().rss" tldw_chatbook/` → the 5 hits above; `ls tldw_chatbook/Utils/ | grep -i "mem|resource|psutil"` → no existing helper.
- Why it matters: no behavioural drift (all five compute the same number), so this is consistency only — but the two RAG copies carry a caching policy (5 s TTL) that will drift the moment one is tuned.
- Recommended correction: D4 sub-case (b), no helper exists. Canonical home: `Metrics/metrics_logger.py`, which already owns the `process_memory_mb` gauge — export a `process_memory_mb(ttl: float = 5.0)` there and have the two RAG classes call it.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D4b] — two `# Exponential backoff` comments in this slice sit over linear sleeps
- Where: `tldw_chatbook/RAG_Search/reranker.py:311` and `:319` (`await asyncio.sleep(1 * retries)  # Exponential backoff`), `tldw_chatbook/RAG_Search/simplified/rag_service.py:1984` (`await asyncio.sleep(0.1 * retry_count)  # Exponential backoff`).
- Evidence: read; `grep -rn "Exponential backoff" tldw_chatbook/` shows the same mislabel outside the slice too (`Web_Scraping/Article_Extractor_Lib.py:552`), while `Evals/eval_errors.py:498-556` implements a real one.
- Why it matters: the comment is the only statement of intent; a reader tuning retries will trust it. With `max_retries = 3` the reranker waits 1+2+3 = 6 s where the comment implies 1+2+4.
- Recommended correction: either fix the sleeps (`base * 2 ** retries`) or fix the comments. If the backoff is actually wanted, `Evals/eval_errors.py`'s `retry_with_backoff` is the existing implementation — no new helper needed.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D4b] — "strip to a non-empty string or None" is re-rolled 5 times under 5 names across 4 packages
- Where: `RAG_Search/simplified/config.py:156 _cleaned_path_setting`, `Chat/chat_conversation_service.py:27 _clean_text`, `Chat/trajectory.py:1044 _optional_text`, `MCP/server_target_store.py:391 _normalize_optional_text`, `MCP/unified_control_models.py:507 _text_or_none`.
- Evidence: read all five; bodies are `if value is None: return None / text = str(value).strip() / return text or None` — byte-identical modulo the local variable name (`server_target_store`'s uses `normalized`). No behavioural drift.
- Why it matters: honestly, very little — it is four lines. The cost is five names for one concept, which makes the pattern un-greppable.
- Recommended correction: D4 sub-case (b). If consolidated at all, `Utils/` is the home and one name wins. A reasonable disposition is "leave it": a shared import for four lines is not obviously cheaper than the copies.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

---

## Candidate dispositions

| candidate (file:line pattern) | disposition |
|---|---|
| dup_shape/dup_verbatim `_identity@RAG_Search/activation.py:141` (+ `MCP/activation.py:25`, `Agents/activation.py:24`) | **confirmed** — byte-identical in all three. Already reported by a sibling reviewer; not re-derived. I additionally checked the three modules' `guarded`/`execution` bodies: those have genuinely diverged (different arguments, different source-gathering), so `_identity` is the only true clone, not the tip of a larger one. |
| dup_verbatim `_get_cached_memory_info@vector_store.py:255` ≡ `embeddings_wrapper.py:373` | **confirmed** → finding "process-RSS-in-MB re-rolled in 5 places" (P3 D4b) |
| dup_verbatim `_cleaned_path_setting@simplified/config.py:156` (+ 4 siblings) | **confirmed** → finding "strip to non-empty-or-None re-rolled 5 times" (P3 D4b) |
| dup_shape `reload_all_pipelines@pipeline_integration.py:175` | **confirmed, but the real finding is bigger** — the whole file has zero importers → finding "pipeline_integration.py (184 lines) has zero importers" (P2 D3). The shape-match against `Utils/Utils.py:637 cleanup_temp_files` is a coincidence of body shape (two-line delegator), not a duplication. |
| except_exception_pass `ingestion_indexing.py:1365` | **retired** — `reset_ingestion_indexer()`'s `try: _indexer.stop() except Exception: pass`. Test-support teardown; a failure to stop a daemon thread has nothing to report and no data path. |
| except_exception_pass `vector_store.py:1527` | **retired** — inside `get_memory_usage`'s psutil probe; a stats-only path, and the surrounding function returns a partial dict by design. |
| except_exception_pass `embeddings_wrapper.py:862` | **retired** — same shape: the `torch.cuda.memory_allocated()` probe in `get_memory_usage`. Stats only. |
| except_exception_return `fusion.py` (1) | **retired** — `_shipped_rrf_k`'s `except Exception: return DEFAULT_RRF_K` around a *module-constant import*, documented at `fusion.py:283-292` as "a last resort must itself be total". Correct as written. |
| except_exception_return `parent_child_adapter.py` (1), `recovery.py` (1) | **not examined** in depth — both files sampled only (see Coverage). |
| `vector_store.py:568 except Exception: … return []` (search) | **confirmed** → P2 finding. Sharpened by a same-package precedent: `simplified/search_service.py:196-201` refuses this exact shape in a comment citing task-2271 ("must surface as an error, never as a silent '0 results'"). |
| function_body_import (22 files) | **mostly retired** — the ones I checked are deliberate circular-import breaks documented in place (`enhanced_rag_service_v2.py:36-84` for `config_profiles`/`reranker`, `ingestion_indexing.py:352-356` for `active_config`, `active_config.py:20-29`, `reranker.py:950-966` for the torch-pulling `sentence_transformers`). Two real ones: `reranker.py:207 import hashlib` inside `_get_cache_key` although `hashlib` is imported at `:25`, and `rag_service.py:3227 import asyncio` although imported at `:8`. Both P3-trivial; folded into the coverage notes rather than given their own finding. |
| inline_truncate `vector_store.py:642`, `:1394` (`[:300] + "..."`) | **retired as a duplication candidate** — both are citation-snippet construction inside `_create_citations_from_result`, the same function's two store implementations. No `Utils` truncation helper exists that they bypass. (`SearchConfig.snippet_max_chars = 240` is a *different*, user-facing budget applied elsewhere; the 300 here is the citation text, not the snippet.) |
| legacy_markers (21 files) | **retired** — sampled `chunking_service.py` (11 markers), `enhanced_chunking_service.py` (6), `active_config.py` (34), `parent_child_adapter.py` (25). In every case the "legacy" word is documenting a *deliberately preserved* compatibility contract (the legacy chunk-param validation messages, the legacy `[AppRAGSearchConfig.rag.search]` TOML keys, the legacy parent/child return shape), not dead code. |
| mutable_class_attr `eval/gating.py:53,72,178`, `eval/regression.py:49,78,105` (`model_config`) | **retired** — every one is `model_config = {"frozen": True}`, pydantic v2's documented class-level configuration, and it is load-bearing here (it makes the result models immutable). Not a shared-mutable-state hazard. |
| os_replace_no_atomic `config_profiles.py:833` | **retired at that line, confirmed elsewhere** — `:833` is a plain rename of an already-written file to `custom_profiles.json.migrated`, not a write-then-replace, and its own comment explains why `os.replace` (not `Path.rename`) is correct on Windows. The real finding is one function away: `_save_one` (`:648-653`) does a bare truncate-then-write → P2 D4a. |
| raw_mkdir `config_profiles.py:233`, `:1093` | **retired** — both are inside an `acquire_storage(...)` admission scope (`:232` directly, `:1093` via `_definition_write(selected)` at `:1091`). |
| raw_mkdir `vector_store.py:226`, `pipeline_builder_simple.py:586`, `pipeline_loader.py:163`, `eval/gating.py:155`, `eval/regression.py:195` | **partly retired** — `vector_store.py:226` is inside `with acquire_storage(self.persist_directory)` at `:225`. The other three were **not examined** (two are in the production-unreachable pipeline modules; `eval/` is test-harness-only). |
| raw_1024x1024 (21 hits) | **split** — the two `_get_cached_memory_info` copies are the D4b finding above. `model_recovery.py:63,67` are **retired**: they are `os.read(fd, 1024 * 1024)` read-buffer sizes and an `8 * 1024 * 1024` manifest cap, not MB conversions. The rest are MB formatting inside log lines; no shared helper exists and inventing one is not worth a PR. |
| seed_name `_identity@activation.py:141` | see the first row. |
| tempfile_no_secure `eval/regression.py:409` | **retired** — it is `tempfile.mkstemp` (the *secure* API; `mktemp` is the unsafe one) followed by `Path(temp_path).replace(target_path)` in `_save_atomic`. Worth noting as context for the `config_profiles` finding: this module hand-rolls an atomic write (without `os.fsync`) while `config_profiles` does no atomic write at all, and neither uses `Utils/atomic_file_ops.py`. |
| try_import_guard `parallel_processor.py:31` (`log_gauge`) | **confirmed** → P3 finding (internal import guarded, dead fallback). |
| try_import_guard `parallel_processor.py:19`, `embeddings_wrapper.py:22`, `vector_store.py:9`, `rag_service.py:41`, `indexing_helpers.py:12`, `enhanced_indexing_helpers.py:16` (numpy) | **retired** — numpy is a genuine `embeddings_rag` optional dep and each guard sets a `NUMPY_AVAILABLE` flag the module actually branches on. |
| try_import_guard `simple_cache.py:329` (`xxhash`) | **confirmed** → P3 finding (advertised dep that is not a dep and is not installed). |
| try_import_guard `reranker.py:40` | **retired** — the numpy guard; `sentence_transformers` is deferred properly at `:950`. |
| try_import_guard — all remaining `handlers=['Exception']` rows (`fusion.py:240/293/345`, `ingestion_indexing.py:205/355/699/999/1030`, `rag_service.py:1904/2116/2588/4361`, `active_config.py:241/588`, `simplified/config.py:102`, `pipeline_*`) | **retired** — each is a documented "config/telemetry must never break search/ingest" boundary with a stated fallback, and `ingestion_indexing.py:699` deliberately re-raises `RAGActivationRequired` before its broad handler. |
| except_exception_return `pipeline_loader.py:752`, `collection_indexes.py:138/216`, `semantic_availability.py:357`, `active_config.py:93/399/447/654`, `ingestion_indexing.py:374/391/1310`, `simplified/config.py:135`, `vector_store.py:704/833/1008`, `embeddings_wrapper.py:783` | **not individually examined** beyond the sweep that produced the list; `vector_store.py:833` (`add_documents`) is **retired** — `grep -rn "add_documents" tldw_chatbook/` finds no production caller (only `Tests/RAG_Search/test_embeddings_integration.py` and a conftest fake), so its swallow cannot reach a shipped index write. The real indexing path (`indexing_helpers.store_documents_batch:270`) awaits `rag_service._store_chunks` and converts a raise into `IndexingResult(success=False)`, which `ingestion_indexing.index_entries` then refuses to record in the indexing DB — so the item is re-indexed next run. That is the correct behaviour, and it retires the parent's "swallowed failed index write" concern. |

## Verified-fine
Things that look like smells in this slice and are not, with the evidence:
- **`fetchall()` with no LIMIT on chunk tables** — does not exist here. `grep -rniE "\"SELECT |'SELECT |SELECT .* FROM" tldw_chatbook/RAG_Search/` returns 11 hits; every one either carries `LIMIT ? OFFSET ?` (`ingestion_indexing.py:1505,1527`), is a bounded single-row read (`recovery.py:264-274`), or is an `id IN (SELECT value FROM json_each(?))` pushdown fragment inside an already-LIMITed FTS query (`rag_service.py:2334/2425/2775/4192`).
- **Duplication with `Chunking/Chunk_Lib.py` (ADR-078)** — already resolved and documented. `RAG_Search/chunking_service.py:1-15` records that its independent regex splitter was *deleted* (chunking-engine-parity task 7) and everything routes through `Chunk_Lib.improved_chunking_process`; `enhanced_chunking_service.py:1-24` records the same for the structure-aware path, which now delegates to `parent_child_adapter`. What remains in both wrappers is a deliberate legacy-contract shim (three validation messages + a flat 0-based `chunk_index`), pinned by `Tests/RAG/test_chunking_service.py`.
- **URL fetches bypassing `Utils/egress.py`** — none in this slice. `grep -rnE "httpx\.|requests\.|urllib|aiohttp|urlopen" tldw_chatbook/RAG_Search/` returns exactly one hit, `activation.py:15 from urllib.parse import urlsplit` (parsing, not fetching). Every remote call the package makes goes through `Chat/Chat_Functions.chat_api_call` (`reranker.py:48`), and `reranker.py:361-371` documents that credential resolution is deliberately delegated there.
- **Per-search config reads in the Library path** — `RAGService._hybrid_search` (`rag_service.py:1325-1326`) passes resolved values to `resolve_hybrid_alpha`/`resolve_rrf_k`, measured at **0.0004 ms**. Only the (separately-reported) legacy pipeline path passes `None`. `resolve_active_rag_config()` itself is called at service construction and from Settings screens, not per query.
- **`SimpleRAGCache._deep_getsizeof` on every put** — looks like an expensive object-graph walk on the hot path. Measured on a 20-result entry: **0.054 ms**, and a full `put_async` including eviction is **0.090 ms**. Not worth changing (the *accounting* bug above is a separate matter).
- **The three `activation.py` modules** — parallel by design, not copy-paste. Only `_identity` is byte-identical (see dispositions); `guarded`, `execution` and `_sources` have materially different bodies in each package.
- **`RAG_Search/eval/` (1 045 lines)** — imported only by `Tests/RAG_Eval/*`. This is deliberate (the package docstring at `eval/__init__.py:2-6` says so) and it is a real evaluation harness, not dead code. Worth knowing that it ships in the wheel with no production caller, but that is a packaging judgement, not a defect.
- **Stdlib `logging` in `chunking_service.py` and `parent_child_adapter.py`** — not a violation of the brief's rule (which is about loguru *and* stdlib logging in the *same* file). Neither imports loguru, and 46 files repo-wide are stdlib-logging-only, so this is a repo-wide convention gap, not a RAG_Search one.
- **No Textual anti-patterns in this package** — `grep -rn "run_worker|@work" tldw_chatbook/RAG_Search/` → zero hits; no `query_one`, no reactives, no mutable class attributes on widgets. `grep -rnE "def .*=\s*(\[\]|\{\})"` → zero mutable default arguments. All three `re.compile` calls are at module scope (`reranker.py:1021`, `eval/regression.py:35`, `collection_fingerprint.py:33`).
- **`import tldw_chatbook.RAG_Search.simplified.rag_service` import cost** — measured with `-X importtime` under the isolated env: **409 ms total**. numpy is loaded eagerly (28.5 ms) but torch, chromadb, sentence_transformers and transformers are **not** — the deferred-import discipline in this package works. The cost is almost entirely upstream: `tldw_chatbook.config` 181 ms, `tiktoken_runtime` 76 ms, `Chunking.Chunk_Lib` 53 ms, `DB.Client_Media_DB_v2` 37 ms. Nothing to fix inside this slice.
- **The embeddings failure-padding in `indexing_helpers.generate_embeddings_batch`** — appending zero vectors for failed chunks *and* recording their indices looks like it would store garbage; it does not. `store_documents_batch:225-235` filters every index in `failed_indices_set` out before storage, so the padding exists only to keep positional alignment. Correct as written. (Contrast `parallel_processor.EmbeddingBatchProcessor`, the dead twin, which gets exactly this wrong.)

## Retired
Candidates I raised during the run and then retired, with the evidence:
- **"Per-search `get_cli_setting` in the Library semantic path"** — traced and retired: the Library path resolves config once at service construction. Only the legacy pipeline path re-reads it (kept, as a sub-point of the pipeline finding, with its measurement).
- **"`IngestionIndexer._run` closes the indexing DB after every batch, so the next batch gets a closed handle"** — retired: `DB/RAG_Indexing_DB.close()` (`:242-253`) closes only the calling thread's held connection and sets `self._thread_local.conn = None`; `_held_connection()` (`:177-200`) transparently reopens. No defect.
- **"`os.replace` in `config_profiles.py:833` is a hand-rolled atomic write"** — retired (it is a post-write rename); replaced by the real `_save_one` finding.
- **"`eval/regression.py:409` uses an insecure temp file"** — retired (`mkstemp`, not `mktemp`).
- **"`add_documents`'s swallowed failure corrupts the index"** — retired: no production caller.
- **"pydantic `model_config` rows are mutable class attributes"** — retired as pydantic convention, per the brief's instruction.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| The uncovered bare HF model ids (`mxbai-embed-large-v1` et al.) actually 404 against the Hub | no network in this review; `HF_HUB_OFFLINE=1` is set by the isolated env | `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && HF_HUB_OFFLINE=0 PYTHONPATH=$WT $PY -c "from huggingface_hub import HfApi; HfApi().model_info('mxbai-embed-large-v1')"` (expect `RepositoryNotFoundError`; compare with `'mixedbread-ai/mxbai-embed-large-v1'`) |
| The 212 ms cold-search UI freeze is visible as a stall in the running app | the brief forbids running the app | drive the TUI per `.claude/skills/verify/SKILL.md`: `tmux -L verify new-session -d …`, open Library ▸ Search, run the first semantic query of the session against a populated Chroma collection, and `capture-pane` the spinner across the call |
| `ChromaVectorStore.search`'s `return []` renders as the empty-index message rather than an error in the Library UI | traced through `library_local_rag_search_service.py:907` by reading; not driven live | same tmux recipe, with the Chroma `persist_directory` made unreadable (`chmod 000`) before the query |
| The cache-accounting leak reaches the shipped 100 MB cap in a realistic session | reproduced at `max_memory_mb=1.0`; the shipped-cap timeline (~32 prune cycles ≥30 min apart) is arithmetic, not observation | run the repro at `max_memory_mb=100.0` with `ttl_seconds=0.2` and ~3 200 puts, asserting `cache._current_memory_bytes` climbs monotonically while `len(cache._cache)` does not |
| The legacy pipeline path (`perform_*`, `pipeline_loader`, `pipeline_builder_simple`, `pipeline_functions_simple`) is genuinely unreachable in production | grep-verified from every angle I could find, but the path resolves functions **by name** from `Config_Files/rag_pipelines.toml`, so a dynamic entry could exist that grep cannot see | `cd <worktree> && source <SCRATCH>/env.sh && $PY -m pytest Tests/RAG/test_scope_pipeline_enforcement.py -q` for the pinning surface, then instrument: add a `logger.error("REACHED")` at `chat_rag_events.py:1959` and drive a Console chat turn with RAG enabled via the tmux recipe |
| `parent_child_adapter.py` (463) and `recovery.py` (637) `except Exception: return` sites | sampled only (see Coverage) — I ran out of depth budget on these two after the higher-yield files | read `parent_child_adapter.py` and `recovery.py:120-637` in full |
