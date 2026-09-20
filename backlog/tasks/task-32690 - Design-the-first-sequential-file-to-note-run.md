---
id: TASK-32690
title: Design the first sequential file-to-note run
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:19'
updated_date: '2026-09-16 05:04'
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
- [x] #1 The approved five-step flow has explicit inputs outputs permission and cancellation behavior.
- [x] #2 Reuse and exclusions are grounded in merged-dev and parked-source evidence without restoring withdrawn SQLite infrastructure.
- [x] #3 Execution lifetime and restart guarantees are explicitly selected by the user before implementation planning.
- [x] #4 The written design records ADR applicability and targeted automated and live acceptance evidence.
- [x] #5 The user reviews the final written design before implementation planning begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes; the user selected session-bound option A and the lifetime amendment is recorded in existing ADR-138. ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md. Reason: staged execution defers durable run/wait/recovery guarantees; ADR-125 and ADR-036 remain unchanged. 1. Inspect immutable merged-dev 657f70ffe7 and parked 8aa1987af9 source plus the server reference. 2. Separate reusable authoring, expression and domain-service paths from withdrawn execution ownership and recovery machinery. 3. Record the user-selected session-bound lifetime, including review-data loss and uncertain Note commits. 4. Specify the five operation subsets, captured authority, bounded model integration, off-loop Notes policy/transaction path, cancellation/drain ordering and targeted/live evidence; distinguish required qualification from existing API capability. 5. Self-review the written design and ADR amendment, verify documentation/task hygiene, then obtain written-design approval before implementation planning. This task remains design-only; no runtime implementation or production/live-profile tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The user approved option A and then the final written design. Recorded the session-bound first delivery in Docs/superpowers/specs/2026-09-16-workflows-first-run-design.md and ADR-138; pinned source reuse/exclusions, operation contracts, timing/byte bounds, physical cancellation and Note readback requirements. The design changes no runtime code or storage ownership and makes no live-execution claim. Documentation diff checks and Backlog ID/path validation passed. The follow-on implementation plan is Docs/superpowers/plans/2026-09-16-workflows-first-run.md under the approved existing ADR; no new ADR is required for direct implementation.
<!-- SECTION:NOTES:END -->
