---
id: TASK-32257
title: >-
  Library Notes: Import selected items unavailable gives no reason at the
  control
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 16:11'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The disabled control states that it is unavailable and nothing else, so there is no way to learn what would make it available.

What makes this a defect rather than a nit is that the correct grammar already exists four panels away on the same screen: the sync setup's disabled radio reads "Unavailable - server sync-folder capability not installed", and Session Git's disabled Commit reads "Stage at least one session note to commit". Disabled controls carrying their reason as text is one of this screen's genuine strengths; this control is the exception.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The disabled Import control carries its reason as text at the control
- [x] #2 The reason names what would make it available
- [x] #3 Covered by a test asserting the reason string in the disabled state
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the disabled Import control: _disabled_action_label puts the reason on the tooltip only.\n2. Move the reason into the label text for both disabled primaries (Check selection shares the helper and the defect).\n3. RED test asserting the reason string in the rendered disabled label; fix; GREEN.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The reason lived on the tooltip alone, so the control read "Import selected items unavailable" and nothing else -- the one exception on a screen whose disabled sync radio and Session Git Commit already carry theirs.

Approach: the blocker goes in the label, through the shared `_disabled_action_label` helper rather than at the reported control, so **Check selection** -- which had the same defect through the same helper -- is fixed by the same line. Grammar follows the screen: "Import selected items unavailable — Choose how to handle the folder name collision". The tooltip is unchanged.

Verified live in the shell at 235x52: the selection phase paints "Check selection unavailable — Choose a source first" at the control (capture 22-live-picker-selected.txt).

Files: `tldw_chatbook/Widgets/Library/library_note_import_canvas.py`, `Tests/UI/test_library_notes_wave_import_ux.py`, `Tests/Widgets/Library/test_library_note_import_canvas.py`.
<!-- SECTION:NOTES:END -->
