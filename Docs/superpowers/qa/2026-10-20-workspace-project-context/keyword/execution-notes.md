# Optional folder-bound keyword index execution

Status: in progress. This file records actual implementation and qualification; planned checks are not passing evidence.

Scope: explicit per-local-folder keyword indexing; preserve saved Library controls and scopes. Semantic indexing remains a separate deliverable. ADR: [240](../../../../../backlog/decisions/240-workspace-project-context-and-folder-bound-index.md). [Implementation plan](../../../plans/2026-10-20-workspace-project-keyword-index.md).

Implementation base: 43054ffa6f02ef4ffc57c0e49ce5b169a5dd5e23. Branch: codex/workspace-project-keyword-index. Managed Windows worktree and existing Python 3.12.10 environment; dependencies reused without installs. Original checkout and user data are untouched. All indexing fixtures/corpora are synthetic and isolated.

## Evidence

- Unmodified Workspace DB and registry baseline: 90 passed, 1 warning in 162.31s. Targeted selection only. Existing pytest-asyncio setup and Pydantic deprecation warnings remain known limits.
- K1 storage/authority: commit8bdb4be36d independently approved for spec and quality. Final amended selection25passed1inheritedwarning46.74s. Covering183passed10platformskips4failed422.43s; two K1 failures corrected, two untouched census failures reproduced on exact33fa199 export2failed148.23s. New-file Ruff/format/whitespace pass;25existing Ruff findings unchanged. [Report](task-1-report.md), [review](task-1-review.md), [final focused output](k1-final-focused.log), [baseline reproduction](k1-baseline-census.log).
- K2 scanner, K3 lifecycle, K4 tool search, K5 Settings/release: pending.

## Qualification limits

No full-suite, other-platform, clean-install, live-user-folder, semantic, hosted-CI or merge claim is made. Exact task evidence and inherited limitations will be recorded as each gate completes.

## Rulings I made

Execution decisions and their costs will be exported in chronological order from this delivery's ledger before closure.
