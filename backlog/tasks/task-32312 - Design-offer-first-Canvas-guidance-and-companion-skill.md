---
id: TASK-32312
title: Design offer-first Canvas guidance and companion skill
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-11 04:29'
updated_date: '2026-09-11 04:37'
labels:
  - canvas
  - design
  - skills
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-09-10-canvas-guidance-skill-design.md
  - backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define how Console offers Canvas before spending tokens on authoring, and how a focused guide tool and optional inline skill help the assistant create compatible displays using the existing Canvas runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design records the approved offer-first behavior and artifact-scoped consent.
- [x] #2 The spec defines bounded guide loading, inline skill packaging, runtime compatibility, recovery, and targeted verification.
- [x] #3 A canonical ADR records the guide-tool and packaging decision and is linked from the spec and task.
- [ ] #4 Independent spec review is complete and the written spec is presented for user review before implementation planning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md
Reason: Introduces a read-only guide-tool contract and distinguishes packaged authoring documentation from imported skill trust; preserves ADR-121, ADR-124, and ADR-009.
1. Explore Canvas, tool discovery, skill invocation/trust, and applicable ADRs on origin/dev (completed).
2. Clarify offer-first consent, compare approaches, and present behavior, architecture, and verification sections (user approved).
3. Write Docs/superpowers/specs/2026-09-10-canvas-guidance-skill-design.md and ADR-149 with linked task/review records.
4. Obtain independent spec-document review; fix planning-blocking findings and check documentation links, task IDs, and diff whitespace.
5. Commit only design artifacts, present the written spec for user review, and wait before creating an implementation plan.
Visual companion: not needed; design questions concern behavior/contracts, not visual layout. No browser mockups or application code are part of this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Documented the user-approved offer-first behavior, scoped canvas_guide(topic) contract, shared packaged guides, optional trusted inline $canvas skill, bounded repair, and targeted verification. No application code or runtime assets changed.
Created Docs/superpowers/specs/2026-09-10-canvas-guidance-skill-design.md, backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md, and Docs/superpowers/reviews/2026-09-10-canvas-guidance-skill-spec-review.md; updated the ADR index and this task. ADR-149 records the new contract and retains ADR-121/124/009.
Independent spec review approved round 1 with no issues or advisory findings. Documentation validation passed: 17 local links, no unresolved placeholders, a unique ADR-149 entry, and the repository task-ID/Windows-path guard across 3693 task files. Cross-ref/worktree allocation checked 207 remote refs and 329 worktrees; an additional all-ref numeric census agreed on ADR maximum 148 before this allocation. Recheck IDs before merge.
Implementation Plan steps 1-4 are complete. Step 5 is in progress: the written spec will be presented after the design-only commit, and the task remains In Progress pending the written-spec review gate. Application tests, browser trials, and model behavior trials belong to implementation and were not run or claimed by this documentation task.
<!-- SECTION:NOTES:END -->
