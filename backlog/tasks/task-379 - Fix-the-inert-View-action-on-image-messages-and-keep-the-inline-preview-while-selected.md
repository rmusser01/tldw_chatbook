---
id: TASK-379
title: Fix the inert View action on image messages and keep the inline preview while selected
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Message actions (Copy, Edit, Save as..., regenerate, continue, rate, delete, View, Save Image) only appear after clicking the message - nothing signals messages are clickable. On an image message, clicking 'View' produced no visible result (no modal, no expansion) in repeated attempts; additionally, the moment the message enters its selected/actions state the inline pixel preview disappears, so the 'View' affordance sits next to an image that just vanished.

**Repro:** Open a conversation containing an image message. Click the message: action strip appears and the inline pixels vanish. Click 'View': nothing observable happens.

**Verifier note:** View is a silent 3-mode cycle (pixels→graphics→hidden, chat_screen.py:2680-2688) with NO feedback of the resulting mode; graphics mode needs a terminal graphics protocol that textual-serve/xterm.js cannot deliver, so the first click legitimately looks inert in the review harness (partly tool artifact), and the graphics-render-failure case has no fallback (only import failure falls back, console_transcript.py:903-914). The 'preview vanishes on selection' is most plausibly the same oversized-image-row/scroll defect as j3-transcript-giant-gap (row derivation keeps image rows when selected, console_transcript.py:738-770 — no hide mechanism exists). The pre-click discoverability sub-point overlaps known-open open-message-action-affordance. Net-new piece: zero state feedback on the View cycle. P2→P3.

**Source:** Console UX expert review 2026-07-20 (finding j3-view-action-inert-preview-hides; P3, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J3 attachments journey. Evidence: `j3-79-view-modal.png`, `j3-77-click-img-label.png`, `j3-70-reopened-conv.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 View should open an enlarged render (or visibly toggle something)
- [ ] #2 Selection should not hide the preview
- [ ] #3 Message affordances should be hinted before click
<!-- AC:END -->
