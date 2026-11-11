---
id: TASK-40130
title: Reuse semantic project indexing in Local Files
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
type: feature
dependencies:
  - TASK-40129
references:
  - backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
  - Docs/superpowers/plans/2026-11-11-local-files-semantic.md
---

## Description

Let a person find files by meaning or use keyword and semantic retrieval together in Local Files. Reuse the draft project index backend and its per-folder configuration.

## Acceptance Criteria

- [ ] Local Files offers Semantic and Hybrid retrieval for one selected folder with clear model/index readiness.
- [ ] A semantic query and answer use the current compatible generation. Keyword search works with embeddings unavailable.
- [ ] Stale or revoked semantic evidence cannot be previewed or sent to an answer provider.
- [ ] Existing Console project search and Library retrieval retain their behavior.

## Implementation Plan

1. Reconcile PR #3237's backend with current dev and qualify affected contracts.
2. Add explicit retrieval mode and semantic evidence receipts to the Local Files entry point.
3. Wire the mode selector for Search and Ask and verify the mounted workflow.

- ADR required: yes
- ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
- Reason: semantic mode selection and evidence retention extend the existing binding authority contract.

## Implementation Notes

- Reconciled the backend from draft PR #3237 at `dee2f1edc2` onto `dev` at `ae9f9e8d21` in local commit `ffb8aea641`. The reconciliation preserves current CI structure and records the Windows project index contract, diagnostic inventory, query plan, and Textual guard checks.
- Local Files now selects Keyword, Semantic, or Hybrid for Search and Ask. The query worker reuses the binding's model consent, projection, exact cosine ranking, and reciprocal rank fusion. Keyword skips the embedding leg. The view distinguishes unavailable Semantic and Hybrid's keyword fallback. Semantic receipts validate the original generation as well as the live file, index scope, and refresh revision.
- The combined tests passed deterministic search and generation revocation, plus mounted flows at 80×24 and 120×40. The offline MiniLM real-app lifecycle passed in 280.03 seconds with zero HTTP attempts. It exercised paraphrase retrieval, failed-refresh recovery, restart, Disable, Clear and detach. The shared test environment lacks SentenceTransformers, so this test used the existing qualification environment with the optional embedding dependencies installed. Per-query latency was not measured in this run.
- The remote draft PR remains unchanged. ADR-241 governs this delivery.
