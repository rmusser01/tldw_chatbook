---
id: TASK-24402
title: Add My Profile to canonical Settings
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-29 19:17'
updated_date: '2026-08-29 19:25'
labels:
  - personal-context
  - settings
dependencies: []
references:
  - Docs/superpowers/plans/2026-08-28-personal-context-01-core-chatbook-local.md
documentation:
  - >-
    backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the encrypted local Personal Context Profile in Chatbook's canonical F9 Settings surface so users can understand profile state, manage records and local privacy controls, perform safe exports and local removal, and reach every promised action at supported terminal sizes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 My Profile is registered in canonical Settings search/navigation and renders available, empty, disabled, locked, and error states without exposing profile content
- [ ] #2 One app-owned PersonalContextService is shared across fresh Settings visits and remains injectable for isolated tests and later Console use
- [ ] #3 Users can add, edit, archive, restore, and delete eligible profile records through PersonalContextService with clear confirmation and conflict recovery
- [ ] #4 Users can change runtime enablement, record syncability and visibility, and per-scope agent authority while exact runtime state and disabled reasons remain legible without relying on color
- [ ] #5 Plaintext and encrypted recovery exports require explicit validated destinations and warnings; local profile removal requires the service confirmation contract, offers an explicit fresh-profile path after destruction, and exposes no delete-everywhere control in this local-only phase
- [ ] #6 The panel and its working key bindings remain contained and operable at supported narrow terminal widths with production consolidated CSS
- [ ] #7 Targeted Settings, Personal Context, CSS bundle, and private SQLite compatibility tests plus bounded scratch-profile live verification pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED service/presentation-contract tests for a content-safe Settings snapshot, scope-authority reads, service-owned manual record creation, and explicit reinitialization after a locally destroyed standalone profile; add RED production-shaped Settings tests for category registration, state rendering, CRUD/conflict handling, authority/runtime controls, export/removal confirmations, honest shortcuts, and narrow containment.\n2. Wire one lazy app-owned PersonalContextService for production and preserve explicit Settings constructor injection for tests; keep crypto/database work out of compose and category search paths.\n3. Add the minimal PersonalContextService read/manual-create APIs required by the UI and a fail-closed explicit fresh-storage-generation transition after confirmed local destruction, then implement a standalone PersonalContextSettingsPanel whose immutable snapshots load in a fenced background worker and whose mutations always delegate to the service.\n4. Register My Profile in the canonical Settings summary, Data & Privacy group, search/guidance/ownership/state contracts, mount only the new panel from the detail pane, and wire only category-scoped working actions and confirmations; do not expose Delete Everywhere in this local-only phase.\n5. Add the Personal Context CSS source to the consolidated build manifest, rebuild generated CSS, and verify supported narrow widths with the production CSS harness and explicit text-labeled available, empty, disabled, locked, removed, error, focus, and destructive states.\n6. Run targeted Settings, Personal Context, CSS, app composition, packaging, and private-SQLite tests plus Ruff/format/diff checks; perform an isolated scratch-profile live persistence check with before/after real-profile fingerprints; run independent specification, code-quality, and UI finish review before closeout.\n\nADR required: no — ADR-102 already governs the app-owned service boundary, encrypted local policy, Settings lifecycle controls, export privacy, destructive local removal, and explicit new-profile creation.\n\nADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
<!-- SECTION:PLAN:END -->
