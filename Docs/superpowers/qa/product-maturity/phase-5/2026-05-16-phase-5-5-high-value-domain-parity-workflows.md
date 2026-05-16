# Phase 5.5 High-Value Domain Parity Workflows QA

Date: 2026-05-16
Status: verified

## Scope

Phase 5.5 verifies the highest-value domain parity seam selected for this slice: Library/Search/RAG evidence handed off to Console live work. The workflow must preserve local behavior while keeping server-backed evidence source-honest.

Out of scope for this gate: new remote RAG endpoints, write sync, broad domain orchestration, Schedules/Workflows run-control services, ACP runtime launch, and UI redesign.

## Workflow Check

Verified by mounted Library workflow coverage:

1. Open Library.
2. Switch to Search/RAG mode.
3. Run a Library Search/RAG query through the app-owned retrieval service.
4. Select a retrieved evidence row.
5. Send the evidence to Console live work.
6. Confirm local evidence keeps a `local:library-rag:<id>` target.
7. Confirm server-backed evidence uses a `server:library-rag:<id>` target and preserves `source_authority=server`.

## Source Authority Coverage

The high-value parity behavior is intentionally narrow:

- Local Library RAG results still stage as local-owned Console evidence.
- Server-backed Library RAG results now stage as server-owned Console evidence.
- `runtime_backend`, `source_authority`, and `source_selector_state` remain aligned in the payload.
- The selected evidence remains reachable from Console even when the result is not local-owned.

## Recovery Boundary

The existing Library/Search/RAG recovery states remain authoritative:

- Missing retrieval service remains blocked with explicit recovery copy.
- Policy-denied server retrieval remains blocked through the existing persistent recovery state.
- The fix does not silently fall back from server-owned evidence to local-owned evidence.

## Verification Commands

Focused commands run for this gate:

```bash
python -m pytest -q Tests/UI/test_product_maturity_gate16_library_search_rag.py::test_library_search_rag_selected_result_launches_console_live_work Tests/UI/test_product_maturity_gate16_library_search_rag.py::test_library_search_rag_server_result_launches_server_console_live_work --tb=short
```

Result: `2 passed, 1 warning`.

Full focused verification for this slice:

```bash
python -m pytest -q Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/Library/test_library_rag_service.py Tests/UI/test_product_maturity_phase5_server_parity_plan.py --tb=short
```

Result: `19 passed, 8 warnings`.

```bash
git diff --check
```

Result: pass.

## Functional Defects

No P0/P1 functional defects remain in the verified Phase 5.5 scope.

Accepted residual functional gaps:

- Phase 5.5 does not implement a remote RAG endpoint or server retrieval client.
- Schedules/Workflows run-control services remain explicit Phase 5 residuals.
- ACP runtime launch remains an explicit Phase 5 residual.

## UX Defects

No P0/P1 UX defects remain in the verified Phase 5.5 scope.

Accepted residual UX risks:

- The selected slice changes Console handoff authority only; it does not add new visible UI controls.
- Deeper server availability display for each Library Search/RAG provider remains future work.

## Visual/UI Defects

No P0/P1 visual/UI defects remain in the verified Phase 5.5 scope.

No actual screenshot approval is required for this slice because the visible Textual layout was not changed. The change is payload authority and reachability at the Library/Search/RAG to Console handoff seam.

## Result

Phase 5.5 passes for the implemented gate scope because server-backed Library/Search/RAG evidence is no longer mislabeled as a local Console target, while existing local handoff behavior and recovery boundaries remain intact.
