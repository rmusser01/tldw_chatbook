---
id: TASK-1601
title: Bind application Sync graph to the runtime server context provider
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-31 14:02'
updated_date: '2026-07-31 14:27'
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
- [ ] #1 The app-composed ServerSyncService retains the exact TldwCli server_context_provider and sync_state_repository identities while public from_config compatibility remains available.
- [ ] #2 SyncScopeService, LocalFirstSyncService, and ManualSyncControlService retain the exact application-owned server service, repository, local-first service, and initially empty dataset-key cache identities required by their contracts.
- [ ] #3 Sync operations resolve clients lazily through the shared provider without a service-local client cache, and application shutdown remains the sole owner of provider client cleanup.
- [ ] #4 The missing production local apply store remains explicitly blocked and is not replaced with an in-memory or test-only implementation in this provider-ownership tranche.
- [ ] #5 Focused source checks, real production TldwCli tests, and an offline installed-wheel production-app probe verify the ownership graph without a test, surrogate, simplified, or locally redefined application.
- [ ] #6 ADR-036 and the provider-migration audit use a numeric-safe semantic and AST inventory and record the verified residual app-level compatibility-provider count, expected to be 31 on the reviewed baseline and re-derived after rebases.
- [ ] #7 Focused Sync, runtime-policy, ProductionApp, Packaging, full-suite, static, formatting, and diff-hygiene verification passes.
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

Detailed executable plan: Docs/superpowers/plans/2026-07-31-application-sync-runtime-provider-ownership.md
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 All acceptance criteria are checked.
- [ ] #2 Implementation follows the approved plan or documents deviations.
- [ ] #3 Focused and full automated verification passes.
- [ ] #4 Ruff lint and format checks pass for changed Python files.
- [ ] #5 ADR-036, migration audit, task notes, and relevant documentation are current.
- [ ] #6 Self-review finds no unresolved correctness, privacy, lifecycle, packaging, or test-contract issue.
- [ ] #7 No regression introduces a test, surrogate, simplified, or locally redefined application.
- [ ] #8 Diff hygiene and installed-distribution verification pass.
- [ ] #9 Task status is set to Done only after all preceding items are complete.
<!-- DOD:END -->
