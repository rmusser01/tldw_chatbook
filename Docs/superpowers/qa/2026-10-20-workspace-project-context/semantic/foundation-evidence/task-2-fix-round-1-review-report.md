- **Selected keyless OpenAI-compatible endpoint** — ADDRESSED. `tldw_chatbook/Embeddings/Embeddings_Lib.py:271` validates after `base_url` is parsed; `:581` refuses a default OpenAI endpoint without a key before opening a session. `tldw_chatbook/Workspaces/project_index/semantic.py:64` validates the selected catalog entry and `:154` resolves the actual keyless custom target. `Tests/RAG/test_project_projection_scope.py:500` exercises the selected keyless catalog path.
- **Configured HF cache root and artifact identity** — ADDRESSED. `tldw_chatbook/Embeddings/Embeddings_Lib.py:331` selects `cfg.cache_dir` ahead of the global cache and `:368` checks the snapshot closure. `tldw_chatbook/Workspaces/project_index/semantic.py:115` fingerprints the selected cache root; `:171` resolves and fingerprints the actual snapshot. `tldw_chatbook/RAG_Search/projection_scope.py:203` requires exact artifact compatibility for rebind. `Tests/RAG/test_project_projection_scope.py:531` uses distinct complete cache roots.
- **Live-carrier catalog change** — ADDRESSED. `tldw_chatbook/RAG_Search/projection_scope.py:125` checks persisted state, then `:148` checks current selected catalog identity after the finite store/fence has closed; `execution()` reaches that check at `:172` before embedding work. `tldw_chatbook/Workspaces/project_index/semantic.py:190` compares the selected provider, model, destination, and safe configuration. `Tests/RAG/test_project_projection_scope.py:376` changes the configured endpoint during a live carrier and observes refusal. Existing `tldw_chatbook/RAG_Search/model_recovery.py:544` checks retained local model closure/receipt before encode.
- **Failed project model close and retirement** — ADDRESSED. `tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py:1195` roots the actual wrapper before close, retains its factory on failure, emits a classified error without the raw cause, and settles only after positive close. `tldw_chatbook/Workspaces/project_index/service.py:169` denies affected semantic use while pending, `:184` roots the wrapper in its operation, and `:1457` reports pending shutdown without indefinite wait. `tldw_chatbook/RAG_Search/projection_scope.py:98` prevents the failed carrier from retiring; keyword admission remains independent in `tldw_chatbook/Workspaces/project_index/service.py:137`. `Tests/RAG/test_project_projection_scope.py:578` injects an actual failing factory record closer, checks sanitization and GC retention, exercises keyword search, then retries and retires.

### New Breakage in the Fix Diff

None. Critical: 0; Important: 0; Minor: 0.

### Out-of-Scope Observations

None.

### Verdict

**Fix round:** All findings addressed, no new Critical/Important breakage. Open findings: none.

Check: `task2-receipts/task2-fix1-key-final-parent.log` reports 7 passed; `task2-fix1-catalog-final-parent.log` 2 passed; `task2-fix1-close-final-parent.log` 1 passed; `task2-fix1-composition-final-parent.log` 1 passed. These are implementer runs inspected as receipts; no test was rerun for this review.
