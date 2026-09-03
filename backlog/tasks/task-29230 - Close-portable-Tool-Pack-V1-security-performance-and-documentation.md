---
id: TASK-29230
title: Close portable Tool Pack V1 security performance and documentation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-02 21:32'
updated_date: '2026-09-02 21:32'
labels:
  - tool-packs
  - security
  - performance
  - documentation
dependencies:
  - TASK-29229
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close portable Tool Pack V1 with explicit privacy, maximum-bound, architecture-ownership, documentation, and whole-feature verification evidence so the policy-only trust boundary is enforced and understandable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recursive privacy tests prove manifests, reviews, archives, notifications, logs, and stable failures exclude credentials, commands, arguments, environment, endpoints, paths, tool prose/schema, workspace/Persona state, receipts, approvals, and runtime-install data; executable/plugin/skill fields are rejected.
- [x] #2 Maximum supported tool, server, fallback, JSON, ZIP, permission-store, and receipt bounds succeed; every one-over case fails early with a deterministic category; non-gating benchmark evidence records elapsed time and peak memory.
- [x] #3 Structural performance tests prove export/import review captures each authority source once and performs no per-tool disk or network work.
- [x] #4 Architecture tripwires preserve workspace guard-protocol isolation, Settings lifecycle-only ownership, MCP rule-edit ownership, Actor Pack independence, and explicit registration or exclusion for every permission-addressable namespace.
- [x] #5 User-facing Settings and MCP design documentation explains deterministic policy-only packs, review and exact mapping, unbound import, first bind, editing, removal/tombstones, receipt degradation, uncertain outcomes, Windows publication limits, and the future Tools+Skills/plugin boundary.
- [x] #6 The complete targeted Tool Pack matrix, scoped static/hygiene checks, and final code/security review pass with no unresolved Critical or Important findings; the repository-wide suite remains opt-in.
- [x] #7 ADR-107 remains linked as the governing decision, implementation notes record final evidence and any deviations, and all closeout task hygiene is complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add recursive forbidden-key and sentinel tests over review objects, canonical archive bytes, stable errors, notifications, and logs; prove unknown executable/plugin/skill/runtime-install fields are rejected.
2. Add exact-limit and one-over tests for 2,000 tools, 256 servers, 257 fallbacks, and near-limit archive/store/receipt payloads; record non-gating elapsed-time and peak-memory evidence and assert one capture per source with no per-tool I/O.
3. Add architecture tests pinning the workspace guard protocol, Settings-versus-MCP ownership, Actor Pack separation, and complete permission-addressable namespace classification.
4. Update `Docs/Design/User_Settings.md` and `Docs/Design/MCP.md` with the complete V1 user contract and explicit non-goals, including separate Windows publication support and future Tools+Skills/plugin installation.
5. Run the plan's complete targeted Tool Pack matrix, scoped Ruff and diff checks, the applicable UI detector, and a whole-feature self/security review; remediate every actionable finding and rerun affected gates.
6. Check all acceptance criteria, add concise implementation notes and final evidence, and mark the task Done only after every Definition-of-Done condition is satisfied.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: ADR-107 already defines the V1 policy-only trust boundary, limits, ownership, Windows publication claim, and separation from future executable Tools+Skills/plugin packs; this task adds enforcement and documentation without changing those decisions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Added recursive policy-only privacy ratchets, exact maximum/one-over capacity tests, non-gating performance measurements, one-capture/no-network assertions, and architecture ownership tripwires for Tool Pack V1.
- Documented the complete Tool Profile lifecycle in the canonical Settings and MCP design guides, including exact mapping, inert import, first bind, tombstone removal, degraded receipts, uncertain outcomes, the separate Windows publication claim, and the future executable Tools+Skills/plugin boundary.
- The initial closeout matrix exposed latent Console lifecycle regressions. Fixed frozen capture admission, default request routing, shared summary budgeting, accepted-but-undispatched temporary recovery settlement, regenerate failure-row placement, and assistant-edit clearing of paired thinking/continuation provenance. Updated narrow persistence fakes and stale ADR-090 continuation expectations accordingly.
- Security review reconfirmed exact-schema rejection, strict bounded archive reads without extraction, one complete inventory snapshot per operation, safe Ask/Deny fallbacks, exact destination reconciliation, fixed lifecycle/store/workspace lock order, receipt non-authority, captured-profile propagation, and no executable, skill, plugin, connection, credential, or runtime-install path in V1. No unresolved Critical or Important findings remain.
- Verification: the complete scoped matrix passed **1,515 tests** in **230.24s**; focused controller/persistence/recovery/export coverage passed **379 tests**; continuation/thinking coverage passed **78 tests**; and the final limit suite passed **4 tests**. Scoped Ruff and `git diff --check` passed. The only warning was the repository's existing Requests dependency-version warning. The repository-wide suite remained opt-in and was not run; an exploratory selection outside the task matrix exposed nine failures in source/test pairs unchanged from the branch base (one promotion fake and eight image-generation fixtures), so they were not folded into this closeout.
- Maximum-pack benchmark evidence: export **0.561s / 3,909,437 bytes peak**, import **0.283s / 4,786,607 bytes peak** for 2,000 tools, 256 servers, and 257 fallbacks. These measurements are diagnostic, not gating thresholds.
- ADR check: no new decision was required. [ADR-107](../decisions/107-portable-tool-use-packs.md) remains the governing V1 architecture; ADR-090 governs the paired thinking/continuation clearing correction found during verification. The UI detector was not applicable because this closeout changed no UI implementation files; the affected surfaces were exercised by the passing Textual matrix.
