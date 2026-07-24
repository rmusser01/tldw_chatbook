---
id: TASK-507
title: Build shared model artifact descriptors and lifecycle
status: To Do
assignee: []
created_date: '2026-07-24 01:02'
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
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the provider-neutral artifact foundation for immutable GGUF and ONNX model revisions, dependency readiness, installed inventory, leases, and safe deletion, with STT as the first consumer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typed descriptors represent immutable root and dependency revisions, variants, formats, sizes, SHA-256 values, licenses, runtime compatibility, platform constraints, and provenance classes.
- [ ] #2 Installed versions use immutable directories and atomic active and root-readiness records; a dependency closure is never loadable until every exact revision is verified.
- [ ] #3 Canonical dependency-closure fingerprints are stable and participate in artifact handles, lease sets, and resident-model identity.
- [ ] #4 ModelArtifactService is the sole managed-store writer and exposes installed inventory without importing inference runtimes.
- [ ] #5 Mutation and deletion use TASK-505 leases; deletion of a root or loaded dependency is refused while leased and succeeds after release or process death.
- [ ] #6 Crash reconciliation, rollback, path containment, disk accounting primitives, and dependency-free lifecycle tests pass without network access.
- [ ] #7 Content-addressed deduplication and LLM artifact migration are not introduced.
<!-- AC:END -->
