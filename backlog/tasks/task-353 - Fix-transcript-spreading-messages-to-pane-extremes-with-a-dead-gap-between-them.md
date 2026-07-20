---
id: TASK-353
title: Fix transcript spreading messages to pane extremes with a dead gap between them
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After the first send, the user message renders at the very top of the transcript pane and the 'Assistant [failed]' + System rows at the very bottom, with ~40 empty rows between them. The pattern persists after more messages, across reloads of the persisted conversation, and in the selected-message state - it is layout, not scroll position or streaming anchoring.

**Repro:** Send one message in a fresh Console conversation and let it complete/fail. Observe the user message pinned top and the response pinned bottom with a large void between.

**Verifier note:** Real, but narrower than reported: j3-63/67 show the void sits immediately after the inline-image row and spans ~36-40 rows — matching the image row widget's max_height 40 clamp (console_transcript.py:932); non-image messages stack contiguously (j3-67 bottom run of 5 rows). So this is an inline-image row-height defect from task-215/PR #626 (post-dates every June-verified transcript capture), not a general transcript layout regression, and no ledger item covers image-row geometry — reviewer's suspected_regression is wrong but the defect is genuine and unrecorded. P2 appropriate: any conversation containing an image becomes unscannable.

**Source:** Console UX expert review 2026-07-20 (finding j3-transcript-giant-gap-layout; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J3 attachments journey. Evidence: `j3-63-response-final.png`, `j3-67-shift-enter.png`, `j3-79-view-modal.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Messages should stack contiguously (top-anchored or bottom-anchored), so the conversation reads as a chronological flow
<!-- AC:END -->
