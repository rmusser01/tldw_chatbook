---
id: TASK-594
title: Build shared model artifact descriptors and lifecycle
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 01:02'
updated_date: '2026-07-29 03:52'
labels:
  - stt
  - artifacts
  - architecture
dependencies:
  - TASK-505
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - Docs/superpowers/specs/2026-07-28-shared-model-artifact-core-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the provider-neutral artifact foundation for immutable GGUF and ONNX model revisions, dependency readiness, installed inventory, leases, and safe deletion, with STT as the first consumer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typed descriptors represent immutable root and dependency revisions, variants, formats, sizes, SHA-256 values, licenses, runtime compatibility, platform constraints, and provenance classes.
- [x] #2 Installed versions use immutable directories and atomic active and root-readiness records; a dependency closure is never loadable until every exact revision is verified.
- [x] #3 Canonical dependency-closure fingerprints are stable and participate in artifact handles, lease sets, and resident-model identity.
- [x] #4 ModelArtifactService is the sole managed-store writer and exposes installed inventory without importing inference runtimes.
- [x] #5 Mutation and deletion use TASK-505 leases; deletion of a root or loaded dependency is refused while leased and succeeds after release or process death.
- [x] #6 Crash reconciliation, rollback, path containment, disk accounting primitives, and dependency-free lifecycle tests pass without network access.
- [x] #7 Content-addressed deduplication and LLM artifact migration are not introduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-07-28-shared-model-artifact-core.md

ADR required: no
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: TASK-594 directly implements the accepted provider-neutral artifact boundary without changing it.

1. Add typed immutable descriptor contracts and canonical closure fingerprints with strict portable validation.
2. Add verified same-filesystem promotion, fast inventory, and exact disk accounting under lifecycle/target leases.
3. Add strict versioned readiness and active records, dependency closure activation, and leased handles.
4. Add lease-safe deletion and crash reconciliation without deleting corrupt payload or abandoned staging.
5. Verify runtime-import boundaries, update lease documentation, run focused tests/static checks, and complete task hygiene.

Implementation remains one production module and explicitly excludes download clients, UI, inference runtimes, catalogs, deduplication, LLM migration, and first-run-wizard work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a direct, one-production-module artifact core: typed immutable descriptors and canonical closure identity; verified same-filesystem promotion; installed inventory and exact disk accounting; strict atomic readiness/active state; dependency-closure activation and acquisition; lease-safe deletion; and crash reconciliation with rollback and path containment.

Lease contract: reuses TASK-505 with lifecycle-first writer ordering, sorted artifact keys, exact shared closure verification, and shared closure leases retained for the complete artifact-handle/resident-model lifetime.

Verification: 277 offline Model_Artifacts tests passed with no warnings; Ruff lint and format checks passed; mypy passed for 3 source files; compileall and git diff --check passed.

Deliberate non-goals: downloader, Textual UI, inference/runtime integration, model catalog, content-addressed deduplication, LLM migration, and first-run wizard implementation.

ADR required: no. ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md. Reason: direct implementation of the accepted ADR-025 provider-neutral boundary without changing it.

Files modified or added across TASK-594 at a high level: shared artifact service and package exports; focused descriptor/lifecycle/process tests; artifact-core design and implementation plan; operation-lease documentation; and linked Backlog task metadata.

Retained gate caveat: evidence is local/macOS only and is not cross-platform proof. TASK-505 remains open for native Windows/Linux and final matrix qualification.
<!-- SECTION:NOTES:END -->
