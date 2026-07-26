---
id: TASK-638
title: Sweep remaining stale dependency flag reads search_rag_window
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 18:00'
updated_date: '2026-07-26 02:56'
labels:
  - followup
  - uat
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-657 fixed the lazy embeddings_rag gate at EmbeddingFactory, but two raw DEPENDENCIES_AVAILABLE reads in search_rag_window.py retain the stale-flag anti-pattern (cosmetic banner/guard shows deps-missing to users who have them). Also: the 657 test module's skipif over-covers test_manually_forced_unavailable_is_still_honored, which needs no extras - losing CI coverage on no-extras environments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 search_rag_window.py dependency reads route through the lazy ensure path
- [x] #2 The forced-unavailable invariant test runs on no-extras environments
- [x] #3 Existing optional_deps tests stay green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate the real search_rag_window.py (tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py; UI/SearchRAGWindow.py is a re-export shim) and confirm the two raw DEPENDENCIES_AVAILABLE.get('embeddings_rag') reads (on_mount banner, _start_indexing_run guard).
2. Lift task-657's private Embeddings_Lib._embeddings_rag_available() re-probe logic into a public tldw_chatbook/Utils/optional_deps.py::lazy_embeddings_rag_available() (single seam), and make Embeddings_Lib delegate to it (reload-safe via sys.modules).
3. TDD: add tests for the shared seam + a pristine-registry SearchRAGWindow regression test (skipif needs real extras) proving no false missing-deps banner; run to confirm RED against current code.
4. Route both search_rag_window.py call sites through lazy_embeddings_rag_available(); update existing "missing deps" tests to force the underlying checker (not just the flag) since the lazy re-probe would otherwise silently flip a forced-False back to True on a dev machine with the extras installed.
5. Restructure Tests/RAG/test_lazy_embeddings_rag_dependency_check.py's module-level skipif into a per-test marker so test_manually_forced_unavailable_is_still_honored (and the pristine-registry sanity check) run on no-extras environments.
6. Sweep Tests/ for other places forcing embeddings_rag False without forcing the checker against this window; fix any that break.
7. Run gates: Tests/UI/test_search_rag_window.py, Tests/Embeddings/, Tests/RAG/, Tests/Utils/test_optional_deps.py.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Lifted task-657's private Embeddings_Lib._embeddings_rag_available() re-probe logic into a public tldw_chatbook/Utils/optional_deps.py::lazy_embeddings_rag_available() (single seam: trusts a True flag reading, re-runs check_embeddings_rag_deps() on a False reading rather than trusting a stale negative). Embeddings_Lib._embeddings_rag_available() now delegates to it via sys.modules (reload-safe, monkeypatch-honoring), and the dead _current_dependencies_available() helper it used to need was removed.

Routed both raw DEPENDENCIES_AVAILABLE.get('embeddings_rag') reads in tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py (on_mount's missing-deps banner + _start_indexing_run's guard) through the shared seam. Under the default lazy dependency-checking mode this fixes a real UAT-class bug: a pristine (never-checked) registry previously showed the missing-deps banner and refused Start Indexing even when the embeddings_rag extras were genuinely importable.

Restructured Tests/RAG/test_lazy_embeddings_rag_dependency_check.py's module-level `pytestmark` skipif into a per-test `_NEEDS_REAL_EXTRAS` marker applied only to the two tests that construct a real EmbeddingFactory/RAG service; test_manually_forced_unavailable_is_still_honored and test_registry_starts_pristine_false_before_any_lazy_check now run unconditionally (no-extras environments included). Added two new tests exercising lazy_embeddings_rag_available() directly.

TDD: added a new TestLazyEmbeddingsDependencyGate class in Tests/UI/test_search_rag_window.py (pristine-registry + real deps -> search/indexing enabled) which failed against the pre-fix code (RED), then passed after the search_rag_window.py fix (GREEN). Updated the two existing "missing deps" tests (in test_search_rag_window.py, plus two more discovered via a repo-wide sweep: Tests/UI/test_disabled_action_recovery_tooltips.py and Tests/UI/test_product_maturity_phase1_empty_setup_states.py) to force check_embeddings_rag_deps() itself, not just the DEPENDENCIES_AVAILABLE flag -- on a dev machine where the extras are genuinely installed, the new lazy re-probe would otherwise silently flip a forced-False flag back to True and break these "unavailable" assertions.

Verified: Tests/UI/test_search_rag_window.py 41 passed (39 baseline + 2 new); Tests/Embeddings/ 5 passed; Tests/RAG/ 578 passed, 8 skipped (baseline 576/8, +2 new tests, 0 new skips); Tests/Utils/test_optional_deps.py 25 passed; the three sweep-discovered test files all green. One unrelated pre-existing failure (test_unified_shell_phase5_recovery_taxonomy.py::test_service_backed_policy_destinations_use_async_workers_without_asyncio_run) confirmed present on a clean stash (unrelated to this change).

Files: tldw_chatbook/Utils/optional_deps.py, tldw_chatbook/Embeddings/Embeddings_Lib.py, tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py, Tests/RAG/test_lazy_embeddings_rag_dependency_check.py, Tests/UI/test_search_rag_window.py, Tests/UI/test_disabled_action_recovery_tooltips.py, Tests/UI/test_product_maturity_phase1_empty_setup_states.py.
<!-- SECTION:NOTES:END -->
