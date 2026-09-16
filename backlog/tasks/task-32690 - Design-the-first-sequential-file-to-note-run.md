---
id: TASK-32690
title: Design the first sequential file-to-note run
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-16 04:19'
updated_date: '2026-09-16 04:23'
labels:
  - workflows
  - design
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-09-16-workflows-first-run-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the smallest first execution delivery for the approved local file, prompt,
llama.cpp, human review and Note workflow on merged dev, with explicit lifecycle
guarantees and no accidental restoration of withdrawn SQLite ownership machinery.
This task produces a reviewed design, not an implementation or runtime claim.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved five-step flow has explicit inputs outputs permission and cancellation behavior.
- [x] #2 Reuse and exclusions are grounded in merged-dev and parked-source evidence without restoring withdrawn SQLite infrastructure.
- [ ] #3 Execution lifetime and restart guarantees are explicitly selected by the user before implementation planning.
- [ ] #4 The written design records ADR applicability and targeted automated and live acceptance evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, amend existing ADR138 after the user selects the execution-lifetime contract; no new ADR number yet. ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md. Reason: the proposed staged execution delivery changes restart/recovery guarantees; ADR125 and ADR036 remain unchanged. 1. Inspect immutable merged-dev657f70ffe7 and parked8aa1987af9 source plus the latest server reference. 2. Separate reusable authoring, expression and domain-service paths from withdrawn execution ownership and recovery machinery. 3. Present session-bound versus restart-resumable execution, including concrete data-loss/uncertain-effect tradeoffs, and obtain the user decision. 4. Finalize the narrowly scoped design and approved ADR amendment, self-review exact boundaries and verification criteria, then request written-design approval before implementation planning.
<!-- SECTION:PLAN:END -->
