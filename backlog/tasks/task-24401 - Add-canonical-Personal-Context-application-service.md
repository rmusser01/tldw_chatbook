---
id: TASK-24401
title: Add canonical Personal Context application service
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-29 18:01'
labels: []
dependencies: []
references:
  - Docs/superpowers/plans/2026-08-28-personal-context-01-core-chatbook-local.md
documentation:
  - backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create Chatbook's single authorized service boundary for Personal Context lifecycle, encrypted local runtime policy, workspace scope mapping, safe exports, and destructive local-profile operations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 One PersonalContextService is the authorized lifecycle boundary for profile creation, canonical workspace scopes, semantic-key collision handling, immutable create/update/archive/restore/tombstone transitions, expiry filtering, and optimistic conflicts
- [ ] #2 Runtime enablement, per-scope agent authority, local workspace mappings, privacy modes, profile lock state, and 24-hour Undo before-images remain encrypted and peer-local while disabled or locked profiles fail closed without deleting records
- [ ] #3 Local removal requires standalone-copy confirmation when applicable; explicit plaintext export excludes keys, raw drafts, Undo data, and peer-local receipts; passphrase-encrypted recovery export round-trips eligible profile data without logging content
- [ ] #4 Targeted tests prove lifecycle/state transitions, key-collision and stale-version behavior, encrypted local policy/Undo, locked and disabled status, destructive confirmation, validated export destinations, export exclusions, and recovery round-trip
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect Shared Core lifecycle models, the encrypted repository seams, existing runtime-policy services, and validated export/path patterns; identify only the minimal repository extensions needed for scopes and encrypted Undo.
2. Write focused RED tests for profile/scope/record lifecycle, semantic-key collisions, expiry, local runtime authority, locked/disabled behavior, Undo, removal confirmation, plaintext export exclusions, and passphrase recovery round-trip.
3. Implement the canonical PersonalContextService mutation choke point and encrypted peer-local runtime policy, extending the repository only for canonical scopes and bounded encrypted Undo artifacts.
4. Implement bootstrap and explicit validated export services with no plaintext logs or implicit destinations; preserve the root wheel's embedded Shared Core contract.
5. Run targeted service/repository/packaging/private-owner tests, Ruff and diff checks, independent specification and code-quality review, then record evidence and close the task.

ADR required: no — ADR-102 already governs the single service boundary, encrypted local policy, export privacy, key custody, and destructive lifecycle.

ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
<!-- SECTION:PLAN:END -->
