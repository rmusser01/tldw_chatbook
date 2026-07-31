---
id: TASK-1601
title: Bind application Sync graph to the runtime server context provider
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-31 14:02'
updated_date: '2026-07-31 14:06'
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
- [ ] #6 ADR-036 and the provider-migration audit describe the corrected Sync boundary and the verified residual count of 31 executable app-level compatibility-provider calls.
- [ ] #7 Focused Sync, runtime-policy, ProductionApp, Packaging, full-suite, static, formatting, and diff-hygiene verification passes.
<!-- AC:END -->

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
