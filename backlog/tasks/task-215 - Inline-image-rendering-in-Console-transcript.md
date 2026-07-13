---
id: TASK-215
title: Inline image rendering in Console transcript
status: To Do
assignee: []
created_date: '2026-07-13 09:30'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 of Console attachments (PR #621) ships a placeholder chip (🖼 label) for image messages. This fast-follow renders images inline in the transcript: pixel mode (rich-pixels) and terminal-graphics mode (textual-image), with a Toggle View action, porting the proven rendering from ChatMessageEnhanced (_render_pixelated/_render_regular/_render_fallback + [chat.images] render-mode config and terminal_overrides). Must respect the transcript's row-signature reconcile loop (0.2s streaming timer) — image rows must not re-render every tick.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Image messages render inline (pixels or TGP per config/terminal) with the chip as fallback
- [ ] #2 Toggle View switches modes per message without remounting the transcript
- [ ] #3 Streaming reconcile does not rebuild image rows on unrelated ticks (row-signature stable)
- [ ] #4 Unmounted-row rendering stays safe (no NoActiveAppError; Content-based fallback path)
<!-- AC:END -->
