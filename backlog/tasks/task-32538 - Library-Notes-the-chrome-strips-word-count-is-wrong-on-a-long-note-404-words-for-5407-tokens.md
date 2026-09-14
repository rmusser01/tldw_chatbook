---
id: TASK-32538
title: >-
  Library Notes: the chrome strip's word count is wrong on a long note ("404
  words" for 5,407 tokens)
status: Done
assignee:
  - '@claude'
created_date: '2026-09-13 06:46'
updated_date: '2026-09-14 14:45'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor B, persona Alex, Edit workflow on the 35 KB seeded note. D3. New with task-32143 / PR #2615 (the strip did not exist at critique #2).

**What happened.** Open "Very long note — scaling laws digest" (DB: 35,204 chars, 5,407 `\S+` tokens, 363 lines). Chrome strip: "404 words · 1:1", then "404 words · 362:1" after Ctrl+End, then "407 words · 363:22" after typing four words (B 31, 33). 404 is the character length of the list's first row title ("Markdown showcase"). Info's own count was not exercised. Captures: B 31, 33.

**Cause.** INFERRED: `_note_word_count` (`tldw_chatbook/UI/Library_Modules/library_notes_controller.py:3136`, `re.finditer(r"\S+")`) is correct on its input and the strip repaints whatever value it is fed (`tldw_chatbook/Widgets/Library/library_notes_canvas.py:3020-3040`), so the fed value is wrong — the strip is being handed the wrong text on open, and the per-edit delta then accumulates on top of it. Docs contradicted: notes.md's chrome strip "N words · L:C".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The chrome strip's word count equals _note_word_count of the open note's body on load and after every edit, verified on the 35 KB fixture (≈5,400)
- [x] #2 A regression test opens a multi-thousand-word note through the production open path and asserts the strip value is the body's count, not the length of a list-row title
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live (power profile, 235x52) and headless through the row-press path with the strip feed instrumented.
2. Compare the strip against _note_word_count of the loaded body and against assessor B's own captures 31/33.
3. Pin the production open path in Tests/UI/test_library_notes_w4_data_truth.py (short note then a 5,400-word note; caret move keeps the long note's count).
4. Fix only what the reproduction proves; update the guide stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Investigated; **no product change** — the reported count is not reproducible
and the strip was already correct.

Evidence. Live on the seeded power profile (235x52), opening "Markdown
showcase" (441 words) and then, straight after it, "Very long note — scaling
laws digest": the strip read `441 words · 1:1` and then `5,427 words · 1:1`.
The database row for that note is 37,519 characters and exactly 5,427
`\S+` tokens, so the painted value equals `_note_word_count(body)` (AC#1).
Resizing to 100x30 — a repaint that carries no count and reads back the last
one fed — kept `5,427 words · 1:1`. The two strip pins in the new file also
pass unchanged on origin/dev 4631b60f8d, which is the direct proof there was
nothing to fix. Critique #3's own captures read "5,404 words · 1:1" and
"5,407 words · 363:22"; the reported "404 words" is those numbers with the
thousands separator dropped in the reading.

Pinned instead of changed (AC#2): `Tests/UI/test_library_notes_w4_data_truth.py`
opens a 3-word note and then a 38 KB / 5,400-word note through the production
row-press path and asserts the strip equals
`f"{_note_word_count(body):,} words · 1:1"`, and that a caret move afterwards
never paints the short note's count. A stale or truncated count is now a test
failure rather than a judgement call. No rescan was introduced — the strip's
0.7 µs repaint budget (task-32143) is untouched.

Files: `Tests/UI/test_library_notes_w4_data_truth.py` (new),
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
