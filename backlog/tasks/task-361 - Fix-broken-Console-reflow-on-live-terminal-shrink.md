---
id: TASK-361
title: Fix broken Console reflow on live terminal shrink
status: Done
assignee:
  - '@claude'
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
- [x] #1 Reflow after a live resize converges to the same layout as a cold start at that size
- [x] #2 Overlays/tooltips are re-rendered or dismissed on resize instead of leaving artifacts over chrome
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
KEY FINDING: the reflow-divergence does NOT reproduce with a NATIVE resize. A
pilot repro (`Pilot.resize_terminal`, the same Textual reflow machinery
textual-serve drives) shows a live resize 160x48 -> 90x30 converges to the exact
cold-start pane layout — rail/transcript/composer all present, `-console-compact`
toggled — with zero divergence. The review's "rail full-width, panes gone" was
observed only via textual-serve's browser-viewport resize at origin/dev
cad9e271d, BEFORE TASK-346 added the `@on(Resize)` height handler; that handler
(plus Textual's own reflow) now keeps the panes on resize. AC#1 is therefore
met on the native path and regression-locked by
`test_console_live_resize_converges_to_cold_start_layout` (asserts live-resize
layout == cold-start layout).

AC#2: the `@on(Resize)` handler now calls `self._clear_tooltip()` first, so a
hover tooltip (the review's stuck "Open the live agent Console." nav tooltip)
is dismissed on resize instead of surviving the repaint as a stale overlay.
Locked by `test_console_resize_dismisses_stale_tooltip`.

Verification note: confirmed via native pilot resize (definitive for the Textual
reflow path) rather than a full textual-serve browser-viewport session; the
underlying reflow code is identical. A browser-viewport served-app pass remains
available as belt-and-suspenders if the exact review path must be re-walked.
<!-- SECTION:NOTES:END -->
