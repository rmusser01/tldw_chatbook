---
id: TASK-1601
title: Bind application Sync graph to the runtime server context provider
status: Done
assignee:
  - '@codex'
created_date: '2026-07-31 14:02'
updated_date: '2026-08-01 15:59'
labels:
  - architecture
  - sync
  - lifecycle
  - packaging
dependencies:
  - TASK-1538
references:
  - backlog/decisions/036-application-service-composition-lifecycle.md
  - backlog/decisions/033-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-31-application-sync-runtime-provider-design.md
  - >-
    Docs/superpowers/plans/2026-07-31-application-sync-runtime-provider-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the app-composed Sync graph follow the application-owned runtime server selection, client-cache, shutdown, repository, and in-memory key-cache authorities instead of retaining a private compatibility provider or detached empty cache.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The app-composed ServerSyncService retains the exact TldwCli server_context_provider and sync_state_repository identities while public from_config compatibility remains available.
- [x] #2 SyncScopeService, LocalFirstSyncService, and ManualSyncControlService retain the exact application-owned server service, repository, local-first service, and initially empty dataset-key cache identities required by their contracts.
- [x] #3 Sync operations resolve clients lazily through the shared provider without a service-local client cache, and application shutdown remains the sole owner of provider client cleanup.
- [x] #4 The missing production local apply store remains explicitly blocked and is not replaced with an in-memory or test-only implementation in this provider-ownership tranche.
- [x] #5 Focused source checks, real production TldwCli tests, and an offline installed-wheel production-app probe verify the ownership graph without a test, surrogate, simplified, or locally redefined application.
- [x] #6 ADR-036 and the provider-migration audit use a numeric-safe semantic and AST inventory and record the verified residual app-level compatibility-provider count, expected to be 31 on the reviewed baseline and re-derived after rebases.
- [x] #7 Focused Sync, runtime-policy, ProductionApp, Packaging, full-suite, static, formatting, and diff-hygiene verification passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/036-application-service-composition-lifecycle.md
Reason: TASK-1601 changes application runtime-provider ownership, a public Sync factory contract, shared in-memory key ownership, and shutdown ownership under the existing ADR-036 composition policy.

1. Rebase onto current dev, re-derive the provider inventory, verify the focused baseline, and amend ADR-036 before implementation.
2. Add a RED numeric-service audit regression, correct the semantic matcher, and reconcile the shared provider audit as this tranche's single audit owner.
3. Add RED direct tests, forward sync_state_repository through the provider-aware factory, and close compatibility-test clients in finally.
4. Add a RED late-key-mutation test and retain the exact initially empty application dataset-key cache.
5. Add narrow source, real TldwCli lifecycle, and offline installed-wheel sentinels; switch only app-composed Sync to server_context_provider; reconcile the residual audit inventory.
6. Run focused, RuntimePolicy, ProductionApp, Packaging, installed-wheel, static, full-suite, rebase, and post-rebase gates; self-review and close the Backlog task only with exact evidence.
7. For PR review remediation, preserve a safe non-secret OS/runtime environment baseline in installed-wheel child processes, bring every cited touched test callable up to the requested annotation/docstring standard, rerun the affected gates, reply to and resolve every review thread, then rebase onto the latest dev and repeat the post-rebase smoke.

Detailed executable plan: Docs/superpowers/plans/2026-07-31-application-sync-runtime-provider-ownership.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the approved ADR-036 application-composition slice. `TldwCli` now constructs `ServerSyncService` through the exact runtime `server_context_provider` while forwarding the exact `sync_state_repository`; the provider-aware public factory preserves that repository without constructing or caching a client. `LocalFirstSyncService` now retains an explicitly supplied empty dataset-key mapping, so Local-first and Manual Sync observe the same process-memory-only cache. The production local apply store remains `None`, and public `from_config(...)` compatibility remains available with deterministic provider cleanup in its test.

Verification uses production classes/functions only: direct Sync contracts, the real mounted `TldwCli`, and the offline installed wheel's production app. The final numeric-safe AST inventory is `{'total': 31, 'sync': 0}` versus `{'total': 32, 'sync': 1}` on `origin/dev@64ebed833`. Full-suite evidence before the final rebase was `24,932 passed, 170 skipped, 114 warnings` in `13,767.16s`; the final post-rebase Sync, Manual Sync, RuntimePolicy, ProductionApp, and Packaging gate passed `501 tests` in `248.39s`, including the installed-wheel probe. The latest-dev Parakeet/STT overlap passed `73 tests`; focused Sync (`60`), Manual Sync (`9`), the no-surrogate sentinel, compile, Ruff lint/format, and diff hygiene also pass.

Deviations were verification-driven and documented rather than hidden: latest dev required stale test-contract reconciliation for the public character-import outcome, TTS export allowlist, direct Footer/Library production-function tests, and RAG lock timing; no test or simplified App was introduced. Self-review also removed a Briefings endpoint diagnostic exposure. Independent PR review and Qodo then found that the installed-wheel child environment had become too synthetic: it now preserves an explicit safe `PATH`/locale/Windows-runtime allowlist while continuing to exclude host credentials and proxy settings. All six cited touched tests now carry concrete annotations and Google-style contracts; the combined review gate passed `11 tests`.

The final rebase refreshed the reviewed diagnostic inventory after the newly landed Parakeet/STT work (`3 passed`, 432 owner files, 1,068 TASK-492 calls, 6,660 TASK-494 calls, four sink files unchanged). The new Parakeet preflight diagnostic uses a fixed message and is rejected by the persistent sink's metadata-only filter; the supporting privacy matrix passed `16 tests`. Concurrent dev merges repeatedly collided with branch Backlog IDs; TASK-1652 was reopened, deterministically reconciled, sentinel-verified, and reclosed. The later File Notes documentation claimant moved from the already occupied TASK-1713 identity to TASK-1721.

ADR required: yes
ADR path: `backlog/decisions/036-application-service-composition-lifecycle.md`
Reason: This implementation changes the application runtime-provider, repository, memory-only secret-cache, public factory, and shutdown ownership boundaries governed by ADR-036.

Core modified files: `tldw_chatbook/app.py`, both Sync services, focused Sync/ProductionApp/Packaging/runtime-audit tests, ADR-036, the provider migration audit, this design/plan, and the task record.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked.
- [x] #2 Implementation follows the approved plan or documents deviations.
- [x] #3 Focused and full automated verification passes.
- [x] #4 Ruff lint and format checks pass for changed Python files.
- [x] #5 ADR-036, migration audit, task notes, and relevant documentation are current.
- [x] #6 Self-review finds no unresolved correctness, privacy, lifecycle, packaging, or test-contract issue.
- [x] #7 No regression introduces a test, surrogate, simplified, or locally redefined application.
- [x] #8 Diff hygiene and installed-distribution verification pass.
- [x] #9 Task status is set to Done only after all preceding items are complete.
<!-- DOD:END -->
