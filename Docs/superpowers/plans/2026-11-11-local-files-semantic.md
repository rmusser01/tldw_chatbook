# Local Files Semantic Search Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans or superpowers:subagent-driven-development. Track the steps with checkboxes.

**Goal:** Use the backend from PR #3237 for semantic and hybrid retrieval in Local Files Search and Ask.

**Architecture:** Reconcile the branch's scoped semantic generations and embedding admission with current `dev`. Add an explicit retrieval mode to the Local Files entry point and use the existing projection, ranking, fusion, and live Capture. The Console tool retains its existing hybrid behavior.

**Tech Stack:** Python 3.12, Textual 8, SQLite FTS5, existing optional local embeddings.

**Spec:** `Docs/superpowers/specs/2026-11-11-local-files-search-design.md`

## Global Constraints

Reuse PR #3237's native retirement, per-folder model and destination consent, generation checks, and storage budgets. Keyword requests make no embedding call. Scope search to one selected local folder. Keep model work outside the UI loop and outside SQL/registry transactions.

## Review Focus

- An unavailable model or generation gives explicit semantic status. Hybrid can still report keyword results.
- Changing mode or folder while a search is held discards its result.
- Revoking semantic consent or replacing its generation invalidates previously returned semantic evidence before preview, reveal, or answer.
- Exact names and words remain available in Hybrid using the backend's existing rank fusion.
- New model or destination settings require the existing per-folder setup and consent.

## Backend reconciliation

- [x] Verify PR #3237 and the current base head. Resolve draft conflicts against current `dev` in the separate local branch.
- [x] Preserve current CI topology while verifying the project index contracts and repository guards. Record the actual qualification of the reconciled backend.

## Search and Ask integration

- [x] Add test coverage for Keyword without model work, Semantic without keyword fallback, and Hybrid with existing fallback and fusion. Include revocation of a semantic receipt after selection.
- [x] Add a `mode` argument to `ProjectIndexService.search_binding` for `keyword`, `semantic`, or `hybrid`. Snapshot the exact binding before projecting the query. Reuse the PR's existing model and vector phases and then recheck the captured binding before ranking or publishing. Retain the semantic generation identity in the file receipt when semantic candidates contributed.
- [x] Add the three modes to `LocalFilesScreen`, defaulting to Keyword. Bind mode to the request and show semantic status independently. Use the selected mode for Ask and clear a stale request when mode changes.
- [x] Verify the existing semantic tests, new mode coverage, the mounted Local Files flow, and an offline model query with the qualified installed model. The real-app offline MiniLM lifecycle passed in 280.03 seconds. Per-query latency remains unmeasured on the combined branch.

- ADR required: yes
- ADR path: `backlog/decisions/241-workspace-project-context-and-folder-bound-index.md`
- Reason: the existing binding and generation contracts now cover an explicit user-selected retrieval mode and reusable file receipts.
