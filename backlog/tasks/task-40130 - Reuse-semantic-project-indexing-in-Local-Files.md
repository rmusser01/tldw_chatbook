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
4. Compare the six writer lifecycle failures against clean dev, fix regressions within the existing ADR-241 contract, and run the full suite.
5. Keep status reads read-only while projecting the scope and job revision a mutation will reconcile. Verify that another authority change invalidates an earlier observation.

- ADR required: yes
- ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
- Reason: semantic mode selection and evidence retention extend the existing binding authority contract.

## Implementation Notes

- Reconciled the backend from draft PR #3237 at `dee2f1edc2` onto `dev` at `ae9f9e8d21` in local commit `ffb8aea641`. The reconciliation preserves current CI structure and records the Windows project index contract, diagnostic inventory, query plan, and Textual guard checks.
- Local Files now selects Keyword, Semantic, or Hybrid for Search and Ask. The query worker reuses the binding's model consent, projection, exact cosine ranking, and reciprocal rank fusion. Keyword skips the embedding leg. The view distinguishes unavailable Semantic and Hybrid's keyword fallback. Semantic receipts validate the original generation as well as the live file, index scope, and refresh revision.
- The combined tests passed deterministic search and generation revocation, plus mounted flows at 80×24 and 120×40. The offline MiniLM real-app lifecycle passed in 280.03 seconds with zero HTTP attempts. It exercised paraphrase retrieval, failed-refresh recovery, restart, Disable, Clear and detach. The shared test environment lacks SentenceTransformers, so this test used the existing qualification environment with the optional embedding dependencies installed. Per-query latency was not measured in this run.
- The remote draft PR remains unchanged. ADR-241 governs this delivery.
- Clean dev at `ae9f9e8d21` passed five of the six Windows writer lifecycle cases. The supervisor-dispatch fixture timed out on dev and the semantic branch. A new deterministic test isolated the additional service-lock dependency on the semantic branch. All six lifecycle cases and the three focused allocation checks passed after the fixes.
- Writer SQL completion now skips the reader handoff while the service lock is held. Explicit Resume can replace a durable Building claim after acquiring its native lifetime and verifying the exact retired claim. The lifecycle fixtures now follow the allocation gate order, tolerate bounded busy responses, and wait for native retirement when asserting terminal states.
- The full pytest collection found 135,059 tests and 13 errors. Clean dev reproduced the same 13 errors in the same environment. This blocks a green full-suite result.
- The old exclusion-floor test reproduced a Refresh failure on the semantic branch and passed on clean dev. Status and management now project the exact scope that the fenced mutation will reconcile without writing the registry witness or index control on a read. Each registry revision advances the epoch and job revision separately so a later unseen edit invalidates an older observation. The original case, new regression case, and neighboring authority and semantic settings cases passed.
- Both project index SQLite connections use the registered literal owner. The peer preflight inventory documents its private connection helper. The two late-publication tests now await the writer's actual retirement before checking its publication result.
- The reader retirement test now checks the actual failed reader's retained pin. It permits the semantic preflight to acquire and retire its own pin while requiring the failed reader to keep admission and block reuse.
- After rebasing onto dev e10ce63fe8, fixed the project index import directions within ADR-241 and recorded both derived SQLite connection sites in the backup owner census. Moved the Local Files navigation code into existing helpers to preserve the Settings and command provider budgets. The import, size, owner, native index, and mounted navigation checks passed.
- The two mounted Settings theme cases now save screenshots in their pytest temporary profiles. Both passed in a checkout without the original qualification directory.
- Local Files reads the workspace registry through the existing application service accessor. The UI lookup census and mounted Local Files flows passed.
- The real-registry search test now waits for Ready through transient native contention. The real-registry and live-file replacement tests passed.
- Corrected the legacy migration fixture to remove all v4-only tables when constructing v1. Detached reclamation now tests the competing Enable with a scope captured before the native gate is held. Both v1 cases, v3 migration, and detached reclamation passed.
- The two real loopback embedding tests use the existing embeddings_rag dependency check. They skip with an explicit reason in the default environment and both passed in the qualified environment with the extra installed.
