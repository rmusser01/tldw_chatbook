---
id: TASK-32727
title: Define llama.cpp runtime identity acquisition and rollback ownership
status: To Do
assignee: []
created_date: '2026-09-17 17:26'
updated_date: '2026-09-17 17:26'
labels:
  - llamacpp
  - catapult-review
  - runtimes
  - design
dependencies:
  - TASK-32721
  - TASK-32722
  - TASK-32723
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish executable provenance and lifecycle rules before adding durable runtime management.

### Scope

One canonical ADR and focused runtime spec covering custom registration, bounded capability probes, managed release acquisition, private executable storage, exact process leases, qualification, selection and rollback.

### Explicit exclusions

No runtime downloader implementation, auto-update service, source builds, broad platform support claims, or generalized launcher framework.

### Architecture gate

ADR required: yes. ADR path: N/A for this scoping record; this design task must allocate and publish a canonical backlog/decisions decision before dependent code starts. Reason: new executable trust, storage and lifecycle authority.

### Affected areas

- `backlog/decisions/`
- `Docs/superpowers/specs/`
- `tldw_chatbook/LLM_Management/`
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py`
- `tldw_chatbook/Model_Artifacts/fetch.py`

### Validation contract

Review the ADR/spec against every acceptance criterion and the existing ownership ADRs. Check Markdown links and dependency ordering. The focused executable plan must name concrete module interfaces, test paths/commands and live-evidence gates. This design-only ticket does not need application tests.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A canonical ADR and linked spec define opaque runtime identity, custom executable path privacy, change detection, installed/executable/model-qualified states and bounded version/help/device evidence.
- [ ] #2 The acquisition contract specifies allowed upstream sources and redirects, release/build provenance, publisher verification versus locally computed integrity, archive/path/size limits, interruption recovery and atomic installation.
- [ ] #3 The lifecycle contract prevents replacement/deletion of an in-use runtime from first release, retains a last-known-good candidate, defines explicit rollback and preserves exact process/model leases and snapshot compatibility.
- [ ] #4 A finite initial OS/architecture/backend support matrix and actual qualification requirements are recorded; unsupported or unavailable combinations have explicit status, never filename-based compatibility claims.
- [ ] #5 A focused executable plan names modules/interfaces, storage versions/bounds, tests, migration and recovery behavior, and compares reuse of existing download primitives against a separate runtime store.
<!-- AC:END -->
