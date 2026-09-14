---
id: TASK-32461
title: 'Library Prompts: three more dirty vetoes refuse without a word'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-12 00:10'
updated_date: '2026-09-14 15:28'
labels:
  - library
  - prompts
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32393 fixed the Escape/Back seam: a dirty Prompts editor now says "Unsaved Prompt changes — Save or Discard changes first." instead of refusing in silence. Three sibling vetoes read the same flag and still say nothing, so the same click-does-nothing-says-nothing defect survives on three other surfaces.

All three refuse on `_flush_library_prompt_save()` returning False (which is exactly `not _prompts_state.dirty`) and return without notifying: the prompt-row switch (`library_screen.py` ~25675, pressing another prompt row while the open one is dirty), select-mode entry (`library_prompts_controller.py` ~1433, pressing "Select" while dirty), and the entry-reconcile path (~33622, a deep link into a prompt arriving while the editor is dirty). The rail-row switch and the app-level navigation guard already notify (`library_screen.py:21405`, `:10485`), so the copy and the pattern exist — these three were simply never wired to them.

The background reconcile path is the one that needs a judgement rather than a copy-paste: a toast raised by something the user did not just press may be noise, so decide whether it explains, defers, or stays silent by design and record which.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pressing another prompt row while the open prompt is dirty states why the switch was refused, on the same line as the next step
- [x] #2 Pressing Select while the open prompt is dirty states why it was refused
- [x] #3 The entry-reconcile veto's behaviour is decided and recorded in the task (explain, defer, or deliberately silent), and matches what ships
- [x] #4 Each wired refusal is covered by a test that fails if it becomes a silent no-op again
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify the three filed line anchors against the current files (dev moved).
2. Wire the prompt-row switch and Select-mode entry to the existing _notify_prompt_dirty_veto().
3. Entry reconcile: per the controller's ruling, explain and name the blocked target (no queue).
4. Red-first pins for all three, mutation-tested by dropping each notify.
5. Docs: Docs/User_Guide/library/prompts.md + stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All three sibling vetoes now speak; none of them changed what they refuse.

- **Prompt-row switch** (`library_screen.py`, `handle_library_prompt_row`) and
  **Select** (`library_prompts_controller.py`,
  `handle_library_prompts_select`) call the existing
  `_notify_prompt_dirty_veto()` before returning — the same
  `LIBRARY_PROMPT_DIRTY_VETO_COPY` Back/Escape, the rail-row switch and the
  app-level navigation guard already raise. No third variant.
- **Deep link** (`library_screen.py`, `_open_library_item_by_id`'s `"prompt"`
  branch) raises a target-naming variant instead: the notifier grew one
  optional `blocked_target` kwarg and formats the new
  `LIBRARY_PROMPT_ENTRY_DIRTY_VETO_COPY` when it is set. Nothing is queued —
  see `## Decision`. The target is `Prompt <id>`, the editor's own
  unresolved-name shape, because no caller passes a display name to that seam.

Line anchors: all three had moved since filing (25675→25847, 1433→1439,
33622→33803) and were re-verified before editing.

Deviation from the brief's "touch `library_screen.py` at exactly one place":
the entry-reconcile seam lives in that file too (the filed `~33622` anchor is
`library_screen.py`, not the controller), so it is three one-line-ish,
non-structural touches there — the veto seam, the deep-link seam, and the
mechanical delegator whose signature had to carry the new kwarg.

Evidence: `Tests/UI/test_library_prompt_dirty_vetoes.py` (3 new pins) went
3 red → 3 green, and each pin was mutation-tested by deleting its own notify
(1 failed, 2 passed, each time the matching one). Live at 235x52 on a scratch
profile: the row-switch and Select refusals both raise the toast (captures
`01-row-switch-veto.txt`, `02-select-veto.txt`). The deep-link refusal is not
reachable by hand — every route to a prompt deep link is itself vetoed while
the editor is dirty — so it is covered by the mounted-screen pin only.

Modified: `tldw_chatbook/UI/Screens/library_screen.py`,
`tldw_chatbook/UI/Library_Modules/library_prompts_controller.py`,
`tldw_chatbook/UI/Library_Modules/screen_constants.py`,
`Tests/UI/test_library_prompt_dirty_vetoes.py` (new),
`Docs/User_Guide/library/prompts.md`.
<!-- SECTION:NOTES:END -->

## Decision

**AC#3 — the entry-reconcile veto explains** (the controller's call, recorded
verbatim; revisitable):

> RULING on AC#3 (mine, so you do not stall): the entry-reconcile veto
> **explains**. A deep link into a prompt that silently evaporates is the same
> click-does-nothing defect one layer further out — the user pressed something
> somewhere to cause it. Name the blocked target and the way out in one line
> (e.g. the pending prompt's title plus "save or discard the open prompt
> first"). Do NOT build a queue that applies the deep link after the save —
> that is machinery for a case nobody has asked for.

What ships matches: `_open_library_item_by_id`'s prompt branch raises
`LIBRARY_PROMPT_ENTRY_DIRTY_VETO_COPY` — "Can't open Prompt 7 — Save or
Discard the open Prompt first." — and still returns `None`, so the link is
dropped exactly as before and nothing is queued. The target is named by id
rather than title because no caller passes a display name down to that seam
(the editor's own unresolved-name fallback has the same shape).
