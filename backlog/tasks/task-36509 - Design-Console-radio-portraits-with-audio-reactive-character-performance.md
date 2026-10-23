---
id: TASK-36509
title: Design Console radio portraits with audio-reactive character performance
status: In Progress
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
- [ ] #5 The user has reviewed the written spec before implementation planning begins.
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
Wrote Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md and
backlog/decisions/242-console-radio-portrait-performance.md; added ADR-242 to the
decision index. Covers six aligned mouth/blink frames per expression, generated
or imported into one reviewed draft, atomic local SQLite attachment storage,
source/revision fencing, radio framing, audio-output ownership/timing, motion
preferences and deterministic plus live qualification requirements. Original
artwork and portable export formats remain unchanged; performance data is
explicitly profile-local in this version. ADR-242 is Proposed pending written
spec review, with existing ADR-074/083/144/150/161 preserved.

Self-review clarified manual-reaction precedence during historical replay and
the static-poster fallback for expressions lacking controllable mouth frames.
Documentation link/placeholder/whitespace checks pass. Python 3.12 task-file
validation passes across all 5501 files; the new task passes isolated ID checks.
The repository-wide ID guard fails on existing untracked/tracked collisions for
TASK-34991 and TASK-35028, both present before this design task. They were not
changed. The system Python was too old for the path validator; rerunning with
the existing Python 3.12 virtual environment succeeded.

Documentation-only work: no product implementation, migration, image generation,
live audio, full test suite or implementation plan. User review remains pending;
AC #5 is intentionally unchecked and this design task remains In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
