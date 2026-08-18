---
id: task-18315
title: Keyboard selection mode follow-ups from the whole-branch review
status: To Do
assignee: ['@Robert']
created_date: '2026-08-18'
labels: [console, selection, keyboard]
dependencies: [task-18156]
priority: medium
---

## Description (the why)

The phase-5 whole-branch review triaged four non-blocking findings to a
follow-up rather than spot patches. They cluster around the mode's
stale-state edges and one deliberate-but-surprising motion semantic.

## Acceptance Criteria (the what)

- [x] Row-destruction cleanup is eager (fixed on PR #1813 after Qodo independently confirmed): `_cancel_selection_if_row_removed` (or its successor) also clears `_kb_selection_row`/`_kb_anchor`/`_kb_end` and hides the hint, so a replaced row never leaves the docked hint lingering until the next keypress
- [x] `h`'s motion candidate carries (fixed on PR #1813 after Qodo independently confirmed) the same len(text) upper clamp as every other motion, so a stale `_kb_end` after a streaming shrink self-heals on `h` too
- [ ] A decision is recorded (and implemented or explicitly declined) on `o`-swap crossing semantics: forward motions after `o` currently clamp to anchor-1 instead of crossing like vim — either cross or document the clamp as the contract in ADR-068
- [ ] `_RecordingPromptQueue.presentation_for` honors (or asserts on) its kwargs instead of hardcoding the idle shape, closing the latent test trap
- [ ] Enter-outside-mode coverage asserts the message-selection toggle actually fires, not merely that no menu opens
