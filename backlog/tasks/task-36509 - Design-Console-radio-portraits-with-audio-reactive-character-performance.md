---
id: TASK-36509
title: Design Console radio portraits with audio-reactive character performance
status: Done
created_date: 2026-10-23 22:24
assignee:
- '@codex'
labels:
- console
- design
- character-chat
- tts
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- backlog/decisions/242-console-radio-portrait-performance.md
- Docs/superpowers/reviews/2026-10-23-console-radio-portraits-design-review.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
updated_date: 2026-10-23 23:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the first version of a living character portrait in the Console sidebar, combining radio-screen presentation, TTS-synchronized mouth movement, blinking and subtle idle motion. Include generating performance frames from an existing portrait and importing ready-made artwork, with review before activation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A written design captures the agreed radio framing, independent mood and speech animation, silence-aware TTS playback behavior, accessibility and graceful fallback.
- [x] #2 The design defines the six-frame generation/import/preview/Apply workflow, safe ownership and persistence, compatibility, and provider capability boundaries.
- [x] #3 A linked ADR records the new performance-artwork and speech-timing contracts, alternatives and consequences.
- [x] #4 The spec has been checked for contradictions, placeholders, broken local links and scope, with future automated and live verification specified.
- [x] #5 The user has reviewed the written spec and acknowledged material review corrections before implementation planning begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/242-console-radio-portrait-performance.md
Reason: Adds durable character performance artwork, a cross-module playback-timing contract and long-lived Console presentation. Existing ADR-074, ADR-083, ADR-144, ADR-150 and ADR-161 remain governing boundaries.

1. Inspect existing Console expressions, visual identity, image-edit capability and TTS playback owners.
2. Write the agreed first-version design and ADR; specify concrete fallback, data ownership, compatibility and verification behavior.
3. Self-review the documents, validate local links and Backlog metadata, and commit only the design artifacts.
4. Present the written spec for user review. Keep implementation planning and product changes pending that review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The user approved the revised design on 2026-10-23. Wrote the Console radio-portrait spec, source-anchored six-finding review, accepted design-only ADR-242 and its index entry. The reviewed contracts cover source-bound six-pose artwork, local SQLite ownership, shared import/generation review, speech ownership/output timing, silence gating, cache invalidation, motion preferences and radio framing.

Created Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md and TASK-36542 through TASK-36550 in dependency order. All nine implementation tasks are To Do, with six unchecked outcome-based ACs each and no premature task Implementation Plans/Notes. The plan defines shared names/signatures, file ownership, meaningful red/green assertions, targeted checks and real provider/device qualification. Planning self-review settled monotonic remove/reapply revision identity, revision lookup across source changes, motion-token order and deterministic DAC publication tests.

Validation: all 5510 repository task files are readable; all nine new IDs are unique in their source buckets, dependencies point backward, all 54 feature ACs remain unchecked, local document links resolve and all 15 Python declaration/assertion blocks parse. Spec/review/ADR/plan linkage and whitespace checks pass. Existing unrelated TASK-34991/TASK-35028 ID collisions remain untouched. No generalized lesson was added because no new repository trap beyond documented ones was established.

Documentation/planning only: no product code, migration, provider call, live audio, product test suite, push or merge. Design ACs and user review are complete; this Done status closes design only. Plan review and execution-method choice precede implementation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved design and six review corrections are recorded in the spec and ADR-242. The implementation plan and nine atomic To Do tasks (TASK-36542–TASK-36550) are ready for review; product implementation has not started.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
