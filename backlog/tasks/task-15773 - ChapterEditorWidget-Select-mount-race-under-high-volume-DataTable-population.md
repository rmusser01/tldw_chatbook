---
id: TASK-15773
title: ChapterEditorWidget/Select mount race under high-volume DataTable population
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - bug
  - stts
  - flake
priority: low
---

## Description

Found and documented as an out-of-scope dodge in task-15478's Review round 3
(input-latency burn-down). While testing chapter-detection on very large
pastes, an unrelated, pre-existing race in `ChapterEditorWidget`/`Select`'s
mount sequence tripped when the chapter table populated a very large number
of rows in one reactive update — observed once in a full-file test run with
`_make_large_book` producing ~999 chapters for a 3M-word book (2000
words/chapter density).

Task-15478 worked around it by reducing `_make_large_book`'s chapter density
(2000 -> 60,000 words per chapter), which made the flake reproduce 0/4
afterward in its own suite — a real dodge, not a fix. The race itself (what
in `ChapterEditorWidget`'s mount sequence loses ordering when `Select` is
populated with a very large row count in one shot) is still there and
unowned.

## Acceptance Criteria

- [ ] The `ChapterEditorWidget`/`Select` mount-sequence race is reproduced
      deterministically (e.g. via the original higher-density
      `_make_large_book` shape, or a targeted stress test that populates the
      table with hundreds+ of rows in one reactive update)
- [ ] Root cause is identified (an ordering assumption between the table
      populate and the Select's own mount/options-set) and fixed at the
      source, not worked around by capping row counts in tests
- [ ] A regression test pins the fix at a row count that reproduced the race
      before the fix
- [ ] `Tests/UI/test_speech_audiobook_chapter_detection.py` stays green,
      including at its original (pre-workaround) chapter density if restored
