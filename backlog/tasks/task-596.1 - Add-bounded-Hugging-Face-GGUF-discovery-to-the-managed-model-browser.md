---
id: TASK-596.1
title: Add bounded Hugging Face GGUF discovery to the managed model browser
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-01 21:51'
updated_date: '2026-08-01 22:00'
labels:
  - stt
  - artifacts
  - ui
  - security
dependencies:
  - TASK-595
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-08-01-task-596-model-artifact-browser-design.md
  - >-
    Docs/superpowers/specs/2026-08-01-task-596-1-remote-model-discovery-design.md
parent_task_id: TASK-596
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users explicitly find and download remote GGUF models through the shared managed-model flow without implying that arbitrary models are runtime-compatible or independently verified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening Remote performs no network request; explicit search or exact repository submission runs off the Textual event loop with bounded, generation-fenced results.
- [ ] #2 A selected repository resolves to an immutable commit and offers only LFS-backed single GGUF files or complete bounded GGUF shard sets with recorded sizes and SHA-256 digests.
- [ ] #3 A selected candidate reaches the existing managed preflight, consent, download, verification, and installation flow; configured Hugging Face credentials support gated or private repositories without being persisted or forwarded across origins.
- [ ] #4 Remote models are labeled Local integrity recorded, remain unassigned and inactive, and are never presented as runtime-compatible, transcription-ready, or eligible for automatic routing.
- [ ] #5 Known license metadata is shown; missing license metadata is recorded as NOASSERTION with a pinned source-review page and requires explicit acknowledgment before download.
- [ ] #6 Focused adapter, GGUF grouping, Textual, redirect-security, and managed-acquisition tests cover the flow without adding native or platform-specific dependencies; Windows and Linux gates remain required when runners are available.
<!-- AC:END -->
