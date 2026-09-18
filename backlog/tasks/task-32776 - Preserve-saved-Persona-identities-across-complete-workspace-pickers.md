---
id: TASK-32776
title: Preserve saved Persona identities across complete workspace pickers
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 08:04'
updated_date: '2026-09-18 08:33'
labels:
  - ui
  - settings
  - design-system
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Workspace Persona controls must distinguish saved identities from control choices and offer every locally saved Persona without losing existing defaults.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Valid saved Persona IDs including none and auto remain distinct from None and automatic-create controls during creation, recomposition and default editing.
- [x] #2 Both workspace Persona selectors offer the complete local catalog beyond the default 100 records, with literal accurate labels for selected identities.
- [x] #3 A failed list does not mislabel an independently readable saved Persona as unavailable; unavailable/deleted identities cannot be silently rebound.
- [x] #4 Explicit None, cancellation, preserved tool profiles and explicit read-write confirmation retain their existing persistence semantics.
- [x] #5 Targeted regressions and four native size/theme journeys verify visible keyboard choices, exact private persistence and clean lifecycle with unchanged default profiles.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/079-workspace-assistant-defaults.md and backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md; presentation ADR-150/161. Reason: repair existing picker identity and catalog requests within established local service, memory confirmation and registry contracts. 1. Pin valid-ID collisions, complete catalog and failed-list selected-identity recovery with real-service mounted tests. 2. Use non-string control choices and request the complete in-memory local catalog; preserve authoritative saved-identity lookup. 3. Update existing control-choice test fixtures and run targeted creation/default/Settings regressions plus scoped lint. 4. Run private native keyboard journeys in four size/theme cells with real persistence, inspect captures, update ledger and draft PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved saved Persona IDs with typed control choices, complete local catalog requests and authoritative selected-ID fallback in the shared picker and canonical Settings. Existing registry/profile/memory contracts remain unchanged (ADR-079/139; presentation ADR-150/161; no new ADR). Nine real-service regressions plus related/fixture-corrective, lifecycle and governance runs establish 81 distinct passing cases. Converted old creation tests to private-profile processes and replaced two fixed async-review delays with bounded state waits. Native run002 PID74075 passed dark/light x 80x24/170x48 with 20 inspected captures, exact source hashes and reopened database checks after each mutation; exit0, eleven healthy databases, lock reacquisition and unchanged default fingerprints. First native run correctly rejected a project fixture inside protected data; sibling fixture correction and failed-run lifecycle retained. Scoped static checks pass with 114 unchanged Settings diagnostics. Independent review found no blocker. QA: Docs/superpowers/qa/2026-09-18-workspace-persona-identity/README.md. Updated completion ledger, remaining Persona audit and typed-ID lesson. Cold auto-provisioning remains the next confirmed repair; no full suite or provider calls.
<!-- SECTION:NOTES:END -->
