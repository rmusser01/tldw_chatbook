---
id: TASK-361
title: Fix broken Console reflow on live terminal shrink
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux, keyboard]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live-resizing a healthy 900x620 session down to 700x480 left: rail expanded to full width, transcript and inspector gone entirely (cold start at the same size keeps them), stale inspector text fragments overlaying the header rows, the screen title replaced by a stuck nav tooltip 'Open the live agent Console.', and no composer. Growing to 1400x900 restored panes and composer correctly, but the tooltip fragment kept overpainting the header border through subsequent reflows.

**Repro:** Open at 900x620, hover/click a nav tab label, then resize the browser viewport to 700x480 and wait 2.5s: compare with a fresh 700x480 session. Resize up to 1400x900 and note the persistent tooltip fragment across rows 2-3.

**Verifier note:** Evidence j6-b02-live-700x480.png vs j6-b05 cold start confirms divergence: rail full-width, transcript/inspector regions gone, no composer, and the nav tooltip 'Open the live agent Console.' stuck over the header — a mounted-overlay leftover surviving full repaint, so app/framework-level rather than a paint artifact. chat_screen.py has no resize handling, so divergence is Textual reflow + stale overlay state. Medium confidence because the only exercisable resize path was browser-viewport via textual-serve (journey's own 'blocked' note); native SIGWINCH may differ. Not covered by any ledger item. P2 appropriate (recoverable by growing the window).

**Source:** Console UX expert review 2026-07-20 (finding j6-live-resize-broken-reflow; P2, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J6 keyboard-only/small-terminal journey. Evidence: `j6-b02-live-700x480.png`, `j6-b05-cold-700x480.png`, `j6-b03-live-1400x900.png`, `j6-b04-back-900x620.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reflow after a live resize converges to the same layout as a cold start at that size
- [ ] #2 Overlays/tooltips are re-rendered or dismissed on resize instead of leaving artifacts over chrome
<!-- AC:END -->
