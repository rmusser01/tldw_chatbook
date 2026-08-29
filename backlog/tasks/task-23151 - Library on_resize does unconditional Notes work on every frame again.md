---
id: TASK-23151
title: Library on_resize does unconditional Notes work on every frame again
status: To Do
assignee: []
created_date: '2026-08-28'
labels:
  - performance
  - library
  - regression
priority: high
dependencies: []
---

## Description

The 2026-08-02 ratchet `test_library_note_fifty_same_side_resize_sequences_do_zero_notes_work`
asserts that a resize which does not cross a layout band does **zero** Notes work. It now measures
**300** calls to `_apply_library_notes_stage_visibility` across 50 resizes (both parametrised
initial sizes). This is a genuine production regression, not a stale test: the ratchet is
correct and must stay at `== 0`.

This is the same defect class TASK-23025 eliminated from the resize path days earlier — per-frame
work reaching the DOM on frames that changed nothing — so it also erodes a fix the 2026-08-27
performance review just shipped.

## Acceptance Criteria

- [ ] A same-side resize sequence that crosses no layout band performs zero Notes stage-visibility
  work, with the existing ratchet unchanged and still asserting `== 0`
- [ ] Band-crossing resizes still apply stage visibility exactly once per crossing
- [ ] The emergency-return path added by the introducing commit keeps its behaviour (a regression
  test covers the narrow-emergency case it was added for)

## Evidence

`tldw_chatbook/UI/Screens/library_screen.py:7284` calls `_apply_library_notes_stage_visibility()`
**before** the `if compact == self._library_notes_compact: ... return` early-out at `:7285-7287`
that makes same-side resizes a no-op. 50 resizes x 6 call sites = 300. The same commit added the
call to `_update_library_notes_responsive_state` (`:7237`).

Introduced by `6161bd1fe19` (2026-08-26) "feat(library): add narrow emergency return path", on dev
via merge `6bed8d6f59` (PR #2124). Reproduces standalone, so it is not test pollution.
