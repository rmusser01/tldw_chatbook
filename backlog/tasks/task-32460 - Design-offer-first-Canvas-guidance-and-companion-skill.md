---
id: TASK-32460
title: Design offer-first Canvas guidance and companion skill
status: Done
assignee:
  - '@codex'
created_date: '2026-09-11 04:29'
updated_date: '2026-09-11 04:50'
labels:
  - canvas
  - design
  - skills
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-09-10-canvas-guidance-skill-design.md
  - backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md
  - Docs/superpowers/plans/2026-09-10-canvas-guidance-skill-implementation.md
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
- [x] #4 Independent spec review is complete and the written spec is presented for user review before implementation planning.
- [x] #5 A reviewed implementation plan maps the approved spec to concrete files, integration checks, and execution steps.
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
6. After written-spec approval, inventory exact callers and write Docs/superpowers/plans/2026-09-10-canvas-guidance-skill-implementation.md with focused executable steps.
7. Obtain independent plan review, verify document references and acceptance coverage, record the result, commit the documentation, and hand off execution.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the approved Canvas design and implementation plan. The user approved each design section, then approved the written spec after design commit 323ae22c5a. The plan preserves existing Canvas ownership/runtime boundaries and adds only the scoped documentation tool, offer-first guidance, packaged examples, optional trusted inline skill, and targeted verification.
Deliverables: Docs/superpowers/specs/2026-09-10-canvas-guidance-skill-design.md; Docs/superpowers/plans/2026-09-10-canvas-guidance-skill-implementation.md; backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md; the spec/plan review records under Docs/superpowers/reviews; the ADR index and this task. ADR-149 preserves ADR-121/124/009. Metadata now records written-spec approval.
Independent spec review and independent plan review each approved round 1 with no blocking or advisory findings. The plan identifies exact provider/catalog/guidance/projection callers, reuses the existing Mermaid guide and browser/wheel harnesses, and distinguishes model behavior from scripted plumbing tests.
Verification: 30 local links across five documents resolve; four Python plan excerpts parse (syntax only); primary integration/test paths exist; the repository task-ID/Windows-path guard passed across 3693 task files; whitespace checks passed. No product code changed, so application tests, runtime/browser qualification, and model behavior trials were not run or claimed by this documentation task. Those are explicit implementation acceptance work.
All documentation-plan steps are complete and the reviewed plan is ready for execution handoff. A separate implementation Backlog task will be created when execution starts. No new generalized lesson was needed. Only these design/planning artifacts are included in this delivery.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

PR #2613 rebased onto dev at 8dd282bad9. This Canvas task originally used TASK-32312; the older upstream task retains that ID. The Canvas records were created at 04:29/04:53 UTC on September 11, after upstream add commits 4df384ae6d (03:47 UTC) and 1744d805eb (04:07 UTC). The local/remote ref and worktree sweep found maximum ID 32457; this record is now TASK-32460. Canvas document references move with it; historical live evidence and temporary paths retain their original names.

A concurrent merge (PR #2608, be380a1a6f) used the initially selected replacement TASK-32458 before Canvas was published. A fresh sweep found maximum 32459; the Canvas design moved to TASK-32460 before push.
