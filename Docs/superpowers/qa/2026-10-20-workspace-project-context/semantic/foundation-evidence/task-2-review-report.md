### Spec Compliance

- ❌ Issues found: a configured, keyless OpenAI-compatible local endpoint is rejected (`tldw_chatbook/Workspaces/project_index/semantic.py:68`, `tldw_chatbook/Embeddings/Embeddings_Lib.py:266-295`); a configured Hugging Face `cache_dir` is silently ignored during offline snapshot selection and omitted from compatibility identity (`tldw_chatbook/Embeddings/Embeddings_Lib.py:338-351`, `tldw_chatbook/Workspaces/project_index/semantic.py:117`); and an already-issued operation does not pause when the selected catalog destination/configuration changes (`tldw_chatbook/RAG_Search/projection_scope.py:124-143`). These are Important findings below.
- ⚠️ Cannot verify from this Task 2 diff: actual semantic generation, ready publication, search disclosure limits, UI behavior, 1 GiB accounting, retrieval quality and scale belong to Tasks 3/4. The controller should qualify those against the later implementation; `Tests/RAG/test_project_offline_factory.py:50-180` proves only real local loading and model reuse with synthetic text.

### Strengths

- `tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py:207-245` validates an issued carrier before Library source inference or constructor effects; `tldw_chatbook/RAG_Search/activation.py:151-158,193-199` routes project calls before default Library admission and retains the `None` route.
- `tldw_chatbook/RAG_Search/projection_scope.py:144-188,190-218` has finite child pins, current scope/state checks, closed-parent rebind checks and a serialized wrapper replacement (`tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py:327-345`). The async/native transfer and repeated cancellation path are exercised in `Tests/RAG/test_project_projection_scope.py:176-274`; the production cached-model reuse path is exercised in `Tests/RAG/test_project_offline_factory.py:50-180`.
- `tldw_chatbook/Embeddings/Embeddings_Lib.py:338-390,784-793` requires an existing, closure-checked local snapshot and disables dynamic fallback for project wrappers (`tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py:419-427`). The narrow Windows descriptor adapter change at `tldw_chatbook/RAG_Search/model_recovery.py:28` retains the pinned-directory approach.
- `tldw_chatbook/Embeddings/Embeddings_Lib.py:618-656` removes raw endpoint and exception text from normal HTTP success/retry/failure logs and propagated request errors. `Tests/RAG/test_embeddings_lib_logging.py:73-116` exercises those installed paths with synthetic credentials.

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- `tldw_chatbook/Workspaces/project_index/semantic.py:68`, `tldw_chatbook/Embeddings/Embeddings_Lib.py:266-295`: selected configured `OpenAICfg` is schema-validated before project credential resolution. Its `api_key` validator runs before the later-declared `base_url` field, so `values.data.get("base_url")` is empty. With no `OPENAI_API_KEY` and no configured key, a valid local OpenAI-compatible endpoint is classified as invalid configuration before `_target` can allow a keyless base URL. The current configured-endpoint test supplies both a key and an environment key (`Tests/RAG/test_project_projection_scope.py:433-446`), missing this case. Make the schema's keyless-base-URL validation work at a model-level or another correct validation point, then test the actual selected catalog path without a key.
- `tldw_chatbook/Embeddings/Embeddings_Lib.py:338-351`, `tldw_chatbook/Workspaces/project_index/semantic.py:117,129-137,157-165`: the offline resolver uses `HF_HUB_CACHE`/the global default for a Hub model ID even when the selected `HFModelCfg.cache_dir` points at a different existing cache. It may report setup required despite a complete configured snapshot, or silently choose a different artifact from the global cache. Omitting `cache_dir` from the settings fingerprint also hides this change from approval/compatibility checks. Honor and validate the selected cache root, or explicitly reject unsupported custom roots; bind the resolved artifact in the effective compatibility contract and add a two-cache test.
- `tldw_chatbook/RAG_Search/projection_scope.py:124-143`, `tldw_chatbook/Workspaces/project_index/semantic.py:193-209`: issuance compares the current catalog candidate once, but each subsequent `execution()` only rereads persisted semantic state. If the configured endpoint or model changes while a carrier remains live, the old wrapper can still embed and send text without pausing for renewed destination approval. Recompare current safe catalog identity at each call boundary, before model/provider work; test a mid-carrier config change. Credential rotation can remain a fresh-wrapper concern as specified.
- `tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py:1180-1210`, `tldw_chatbook/Embeddings/Embeddings_Lib.py:1026-1043`: project callers are required to close model/native work before carrier retirement, but `wrapper.close()` swallows a factory/model close exception and logs its raw text. A failed `rec["close"]()` can therefore leave a model borrower live while caller code proceeds to `carrier.close()`, and the log may reveal sensitive provider or path detail. For project wrappers, propagate a classified close failure and retain/report incomplete retirement without raw exception text; prove the injected failing-close path.

#### Minor (Nice to Have)

- `.superpowers/sdd/2026-10-20-workspace-project-semantic-index/task2-receipts/task2-scope-final-32-parent.log:1-12`, `.superpowers/sdd/2026-10-20-workspace-project-semantic-index/task2-receipts/task2-final-28-parent.log:19-20`: the retained passing output still contains inherited pytest-asyncio loop-scope and Pydantic `Field(exclude_if)` deprecation warnings, plus one unavailable Windows symlink test skip. These are correctly disclosed as existing evidence limits, but the warning output is not pristine; clean up the underlying test configuration/schema warning in its owning work, and do not count the skipped alias case as covered.

### Assessment

**Task quality:** Needs fixes

**Reasoning:** The central issued-carrier and offline-factory path is substantial and has credible focused receipts, including actual cached-model reuse. The four Important edge cases leave supported endpoint/cache choices, mid-operation consent, and physical close failure handling below the Task 2 contract.

- Check: reviewed the single committed `7ba0029d..2160d5b3` diff package and focused unchanged code only for the named validator-order, cache-root, live-configuration and retirement risks; no tests rerun or Git mutations. `controller-task2-evidence-index.json` reports all 16 committed hashes matching and retained parent/pre-run receipts; `task2-scope-final-32-parent.log:18`, `task2-composition-final-33-parent.log:16` and `task2-final-28-parent.log:19-20` show the cited passing selections and limitations.
