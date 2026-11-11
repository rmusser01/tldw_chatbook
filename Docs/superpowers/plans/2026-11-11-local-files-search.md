# Local Files Keyword Search Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans or superpowers:subagent-driven-development. Track the steps with checkboxes.

**Goal:** Search one selected folder by name and content, inspect matching lines, and ask for an answer with file citations from the native Local Files view.

**Architecture:** The Workspace project index issues a finite UI search from its existing binding authority and returns live file receipts for subsequent actions. The view owns query selection and navigation. The cited answer adapter uses the existing answer pipeline with a file evidence bundle.

**Tech Stack:** Python 3.12+, Textual 8, SQLite FTS5, Chatbook's existing model dispatch.

**Spec:** `Docs/superpowers/specs/2026-11-11-local-files-search-design.md`

- ADR required: yes
- ADR paths: `backlog/decisions/241-workspace-project-context-and-folder-bound-index.md` and `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`
- Reason: Local Files introduces user-initiated file evidence and file citations.

## Global Constraints

Use explicitly configured local Workspace bindings. Keep the cache separate from Library. Files remain authoritative. Read and preview through native Capture. Worker results belong to a single screen visit and request. Reuse existing Textual components and token-backed styles. Verify targeted changes under a private test profile.

## Review Focus

- A matching file is replaced with identical text. Capture identity must reject its prior receipt.
- Binding authority or the index manifest changes after search. Preview and answer evidence must refuse the prior receipt.
- A new query supersedes a held search. It must own the result pane.
- A filename-only hit with no excerpt must remain discoverable, while a query with no text evidence must issue zero answer calls.
- A hosted answer destination or evidence changes after approval. The pending request must refuse dispatch.

## 1. Issue search and evidence from the existing folder index

- [x] Add real-index tests to `Tests/Workspaces/test_project_index_service.py` for a filename hit, a content hit, replacement with identical text, and a binding revision change. Run the new cases and verify the missing user-facing API failure.
- [x] Add `ProjectIndexService.search_binding(key, query, *, limit=10) -> BindingSearchResult` and `validate_binding_hits(result, hits) -> tuple[Path, ...] | None` in `Workspaces/project_index/service.py`, with receipt types in `models.py`. Reuse the existing ranker, bounded native Capture, and scope/revision/manifest checks. Store no raw captured text beyond the bounded public excerpt. Keep Console results and permissions intact.
- [x] Run the new cases and the project index tool tests. Review and commit the search slice with the design, plan, and ADR-241 update.

## 2. Ground answers in the selected file evidence

- [ ] Add tests for empty evidence, a valid file citation, changed source evidence, and exact provider approval. Assert on the dispatched request at the answer adapter boundary.
- [ ] Refactor `Library/library_rag_answer_service.py` around its existing evidence-consuming answer function. Keep the Library entry point and its payload behavior. Add `Workspaces/project_index/file_answers.py` to build a private file evidence bundle and dispatch against the approved provider and evidence.
- [ ] Run the new answer tests and the Library answer service and citation regressions. Update ADR-024 and commit.

## 3. Mount the native view and command

- [ ] Add a mounted test in `Tests/UI/test_local_files_screen.py` for folder choice, results, preview, and a held search superseded by a second query.
- [ ] Add `UI/Screens/local_files_screen.py`, register its lazy route in `UI/Navigation/screen_registry.py`, and expose a native-only palette command in `app_command_providers.py`. Use existing Workspace selection and Settings navigation. Resolve selected result actions and citations through the same receipt validator.
- [ ] Verify the mounted flow at 80×24 and a wider layout, existing project index Settings/Console coverage, and scoped static checks. Update `Docs/Architecture/project-indexing.md` and the user guide. Commit the view.

## Semantic follow-up

Reconcile PR #3237 against `dev` separately. When the backend is available, expose Keyword, Semantic, and Hybrid for the selected binding through the same user search and citation path. Keep keyword independent of embedding readiness. Qualify real offline retrieval and report timings before integration.
