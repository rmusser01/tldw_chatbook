---
id: TASK-298
title: Console transcript follows the stream tail (anchor), without yanking readers
status: Done
assignee: ['@claude']
created_date: '2026-07-18 00:40'
labels: [console, ux]
dependencies: [task-259]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed during task-259's live-rig A/B (pre-existing on base, not a 259 regression): when a streamed reply grows taller than the transcript viewport, the view never scrolls — the stream finishes below the fold and the user sees a frozen top-of-message while tokens arrive off-screen. Standard terminal/chat UX is tail-follow with reader courtesy: pinned to the newest content while the user is at the bottom, released the moment they scroll up, re-engaged when they return to the bottom — and an explicit send always jumps to the tail. Textual 8's built-in anchor implements exactly the pin/release/re-engage lifecycle, so the transcript should engage it at mount and re-anchor on new user messages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A streamed reply taller than the viewport keeps the newest content visible while the reader is at the bottom
- [x] #2 A reader who scrolled up is never yanked by stream growth; returning to the bottom re-engages follow
- [x] #3 A new user message (a send) jumps to the tail even from a scrolled-up position; mere tail-message updates do not
- [x] #4 Verified live on the streaming rig
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Engage Textual's anchor on ConsoleTranscript at mount (pin/release/re-engage handled natively).
2. Re-anchor in set_messages when a NEW user-role message appears at the tail (send semantics); track the tail id so tick re-sets of the same list never re-anchor.
3. Harness tests for all three behaviors + the no-re-anchor tick case; live-rig verification by controller.

Live-rig results (2026-07-18, llama.cpp gateway + textual-serve/playwright, same recipe as task-259): 40-line stream now finishes with the TAIL visible (base A/B froze at 36 with the rest below the fold); 150-line run captured FOLLOW-IN-MOTION mid-stream (view tracking 112->150 with Stop active, through a spawned sub-agent tool flow); finalize keeps the newest content on screen. AC#2/#3 (no-yank + send-re-anchor) pinned deterministically by the harness tests (xterm can't drive scroll reliably).
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Two-line mechanism, no scroll bookkeeping of our own: `on_mount` calls `self.anchor()` (Textual keeps the view at the bottom on content growth, releases on user scroll, re-engages when the user returns to the bottom — verified against the 8.2.7 source: `release_anchor`/`_check_anchor`); `set_messages` calls `anchor()` again only when the tail message id CHANGES to a user-role message (a send), tracked via `_tail_message_id` so the 0.2s streaming tick re-setting the same list can never re-anchor a scrolled-up reader. The transcript performs no programmatic scrolling anywhere (selection mounts an inline action row without moving the viewport), so no anchor conflicts exist.

Tests: Tests/UI/test_console_transcript_tail_follow.py — 4 (mount-anchor + follow growth past the viewport, scrolled-up reader not yanked by assistant growth, send re-anchors from scrolled-up, same-tail tick re-set stays put).

Files: Widgets/Console/console_transcript.py, Tests/UI/test_console_transcript_tail_follow.py.
<!-- SECTION:NOTES:END -->
