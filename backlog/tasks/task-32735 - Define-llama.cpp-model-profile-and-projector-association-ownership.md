---
id: TASK-32735
title: Define llama.cpp model profile and projector association ownership
status: To Do
assignee: []
created_date: '2026-09-17 17:32'
updated_date: '2026-09-17 17:32'
labels:
  - llamacpp
  - catapult-review
  - models
  - design
dependencies:
  - TASK-32727
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Decide how exact managed artifacts, reusable tuning and companion projectors relate without making local paths a persistent model identity.

### Scope

One ADR and focused spec for model-to-profile/runtime references and projector companion authority, including source modes, leases, matching evidence, removal and snapshot compatibility.

### Explicit exclusions

No persisted external model/projector paths, automatic filename-based compatibility, changes to Console sampling ownership or file copying during profile selection.

### Architecture gate

ADR required: yes. ADR path: N/A for this scoping record; this design task publishes a canonical decision before dependent code. Reason: new durable associations and companion-artifact ownership extend ADR-025, ADR-119 and ADR-165.

### Affected areas

- `backlog/decisions/`
- `Docs/superpowers/specs/`
- `tldw_chatbook/Model_Artifacts/service.py`
- `tldw_chatbook/Model_Artifacts/leases.py`
- `tldw_chatbook/LLM_Management/llamacpp_profiles.py`
- `tldw_chatbook/LLM_Management/snapshot_admission.py`

### Validation contract

Review the ADR/spec against every acceptance criterion and the existing ownership ADRs. Check Markdown links and dependency ordering. The focused executable plan must name concrete module interfaces, test paths/commands and live-evidence gates. This design-only ticket does not need application tests.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The ADR defines exact managed artifact reference to profile/runtime association, explicit save/remove behavior, invalidation and recovery for missing/deleted references without persisting external source paths.
- [ ] #2 A versioned bounded private association store and concurrency/migration policy are specified separately from tuning-only profile payloads.
- [ ] #3 Projector selection has its own exact verified identity and managed lease for the entire process lifetime; partial acquisition, failed spawn, stale cleanup and stubborn-process outcomes are defined.
- [ ] #4 Explicit repository/bundle/revision evidence may suggest pairing; filename similarity and vision hints never prove compatibility, and uncertain pairing requires explicit selection.
- [ ] #5 The spec and focused executable plan pin public interfaces and tests for external session-only selection, raw projector conflicts, snapshot compatibility and measured multimodal qualification.
<!-- AC:END -->
