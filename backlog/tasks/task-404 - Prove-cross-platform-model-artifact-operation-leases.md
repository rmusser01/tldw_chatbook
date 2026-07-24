---
id: TASK-404
title: Prove cross-platform model artifact operation leases
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:01'
updated_date: '2026-07-24 01:13'
labels:
  - stt
  - artifacts
  - architecture
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - Docs/superpowers/plans/2026-07-23-model-artifact-operation-leases.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the reusable interprocess shared and exclusive lease primitive that must protect immutable model artifacts before the shared artifact service is implemented.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single typed lease API acquires shared load leases and exclusive mutation leases with bounded non-blocking retry and explicit timeout errors.
- [ ] #2 Multiple spawned processes can hold shared leases for the same root and dependency artifacts while an exclusive delete attempt remains blocked.
- [ ] #3 Normal close and forced process termination release every held lease automatically, with no PID-file or stale-lock cleanup path.
- [ ] #4 Lease sets acquire artifacts in stable ID order and release safely after partial acquisition failure.
- [ ] #5 Native Windows, macOS, and Linux process tests prove shared/shared compatibility, shared/exclusive exclusion, idle residency, and crash release.
- [ ] #6 The selected package and version, platform behavior, license, and failure result are documented; failure on any required OS blocks the artifact-core task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 mandates a proven cross-process shared/exclusive lease primitive before the artifact lifecycle may be implemented.

1. Pin portalocker==3.2.0 and add failing tests for the typed single-artifact lease API.
2. Implement opaque lock identities, bounded timeout and cancellation handling, deterministic release, and canonical multi-artifact lease sets with rollback.
3. Prove shared/shared compatibility, shared/exclusive exclusion, idle root-and-dependency residency, partial rollback, and crash release with spawn-process tests.
4. Add a focused native GitHub Actions matrix for Ubuntu, macOS, and Windows and its workflow-shape regression.
5. Document the selected primitive and complete this task only after all three native jobs pass for the same commit.

Detailed plan: Docs/superpowers/plans/2026-07-23-model-artifact-operation-leases.md
<!-- SECTION:PLAN:END -->
