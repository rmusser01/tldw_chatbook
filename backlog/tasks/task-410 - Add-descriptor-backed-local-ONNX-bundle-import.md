---
id: TASK-410
title: Add descriptor-backed local ONNX bundle import
status: To Do
assignee: []
created_date: '2026-07-24 01:03'
labels:
  - stt
  - artifacts
  - import
  - onnx
dependencies:
  - TASK-406
  - TASK-408
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Support offline import of known Parakeet and VAD ONNX bundles while rejecting unknown or modified graphs until isolated arbitrary-graph validation is separately designed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import accepts only a bundle manifest that resolves to an existing Chatbook catalog descriptor and rejects unknown descriptors or graph variants.
- [ ] #2 Every required graph and external-data file is a contained regular file and matches the descriptor byte size and SHA-256 before activation.
- [ ] #3 Missing files, extra required relationships, absolute paths, traversal, symlinks, irregular files, and modified graphs fail with stable artifact errors.
- [ ] #4 The full root and dependency closure is copied through staging, revalidated after copy, fingerprinted, and made loadable only through the root readiness record written last.
- [ ] #5 No untrusted ONNX graph is parsed in the UI process or resident inference worker, and no provider is allowed to fetch a missing file implicitly.
- [ ] #6 Tests cover Parakeet v2, v3, VAD, multi-file bundles, offline success, corruption, containment failures, interruption, and rollback.
<!-- AC:END -->
