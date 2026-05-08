# Gate 1.6 / Phase 3.8 Library-Native Search/RAG QA

## Scope

Gate 1.6 verifies that Library owns the deliberate Search/RAG workflow instead of only linking to the legacy `search` route. The verified slice covers Library source scope, query controls, retrieval status, evidence/results, citations/snippets, selected-result inspector state, Library-to-Console staged evidence, Console-initiated Library RAG, and recoverable blocked states when the Library RAG service is unavailable.

Out of scope: replacing `SearchRAGWindow`, full Workspaces/Collections adapters, full Library-native answer composition, and full ChatWindowEnhanced replacement.

## Walkthrough

- Library mounted Search/RAG mode remains inside the Library shell via `#library-search-rag-panel`, `#library-rag-source-scope`, `#library-rag-query-input`, `#library-rag-run-query`, `#library-rag-results`, `#library-rag-inspector`, and `#library-rag-use-in-console`.
- Empty query state keeps `#library-rag-run-query` and `#library-rag-use-in-console` disabled with durable recovery copy instead of a transient notification.
- A fake local retrieval service renders evidence at `#library-rag-result-0`, snippet text at `#library-rag-result-snippet-0`, and citation labels at `#library-rag-result-citations-0`.
- Selecting `#library-rag-select-result-0` updates `#library-rag-selected-result` and enables `#library-rag-use-in-console`.
- Console staged evidence preserves `query`, `source_id`, `chunk_id`, `snippet`, `citations`, `score`, `runtime_backend`, `source_authority`, and `source_selector_state` in the live-work payload.
- Console can invoke Library RAG from `#console-library-rag-query-input` and `#console-run-library-rag`; successful retrieval stages evidence at `#console-live-work-payload-source-id`, while unavailable retrieval renders blocked recovery in `#console-run-inspector`.

## Functional Result

Users can complete the Gate 1.6 local workflow without knowing the legacy Search/RAG destination:

1. Open Library.
2. Switch to Search/RAG mode.
3. Enter a query against visible local Library source scope.
4. Review evidence with snippets and citations.
5. Select a usable result.
6. Stage that result into Console with source authority and review recovery copy preserved.
7. Start from Console and request Library RAG, receiving either staged evidence or an explicit blocked state with owner and next action.

## Verification

- Red test: `test_gate16_library_search_rag_evidence_is_tracked` failed before this evidence file existed.
- `$PY -m pytest -q Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_search_handoffs.py Tests/UI/test_console_live_work_handoffs.py --tb=short` -> `90 passed, 1 warning`.
- `$PY -m pytest -q Tests/Library/test_library_rag_state.py Tests/Library/test_library_rag_service.py Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py Tests/UI/test_search_handoffs.py Tests/UI/test_ux_audit_smoke.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_disabled_action_recovery_tooltips.py --tb=short` -> `142 passed, 8 warnings`.

## Defects

- No P0/P1 defects remain in the mounted Library Search/RAG and Console staged-evidence seams covered by this gate.

## Residual Risk

- Workspaces and Collections are visible in the Library source model but still need deeper adapters in later gates.
- The legacy `SearchRAGWindow` remains reachable as a compatibility route; full replacement is intentionally deferred.
- Console-initiated Library RAG stages retrieved evidence and blocked states only; it does not yet rewrite the active chat draft or perform full conversational answer synthesis.
- Manual verification with a real indexed local corpus remains useful after the next indexing/parity slice; this gate uses deterministic mounted services for QA evidence.
