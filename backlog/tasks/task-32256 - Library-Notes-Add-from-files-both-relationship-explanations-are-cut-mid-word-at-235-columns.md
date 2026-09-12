---
id: TASK-32256
title: >-
  Library Notes Add from files: both relationship explanations are cut mid- word
  at 235 columns
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
Residual of task-32125, which correctly moved the two relationships to sibling buttons under their own descriptions (C cap 26, D cap 21, with only Back to Notes pinned below). The descriptions themselves are truncated mid-word at 235x52, so the one screen whose entire job is to explain the difference between "Import once" and "Keep a folder synced" delivers neither explanation in full -- to a first-timer, at the point of an irreversible-feeling choice.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both relationship descriptions render complete at 235x52
- [x] #2 Neither description is cut mid-word at any supported width: below the width where they fit they wrap or disclose
- [x] #3 Covered by a test asserting the full description text at 235x52
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the mid-word cut at 235x52 against the cited captures and a rendered probe.\n2. If it reproduces, make the descriptions wrap; if not, pin the full text with a render-level test and report the non-reproduction.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
NOT REPRODUCED, then pinned.

Both relationship descriptions render complete at 235x52. Checked against the evidence this task cites -- every chooser capture in the critique session (`notes-crit2/C/caps/26-add-from-files.txt`, `D/caps/21-add-from-files.txt`, `C/caps/72-afl.txt`, `D/caps/68-relationship.txt`, `P/caps/04-addfiles.txt`) carries both sentences in full, and a grep across all four assessors caps for a truncated instance returns nothing. Re-checked live on this branch at 235x52 (capture `20-live-chooser-235x52.txt`): both complete.

The mechanism rules it out too. Neither Static carries a nowrap/ellipsis rule, and nothing in the screen sheet gives one to a bare Static under `#notes-sync-body`; measured, the widgets grow from 1 line at 112 cells to 2 at 59 and 3 at 39. They wrap, which is what AC#2 asks for, rather than clipping.

So AC#1 and AC#2 were already true and are now pinned by AC#3 -- a parametrized render test at 235 (the wide terminal), 113 (the share the reader pane actually gets there) and 60 (the narrow floor) asserting the exact sentences and that `width x lines` covers the text at every one. The likeliest origin of the report is a capture read through a column cut rather than the app.

Files: `Tests/UI/test_library_notes_wave_import_ux.py`.
**Fix round 1 (review finding 6).** The pin asserted on `renderable.plain`, the
SOURCE text, which truncation never touches -- mutation-tested, a clipped
description passed at 235 and 113 and failed only at 60. It now joins the
widget's own rendered strips and refuses an ellipsis, and under the same
mutation (nowrap + ellipsis + a 40-cell width) it fails at all three widths.
<!-- SECTION:NOTES:END -->
