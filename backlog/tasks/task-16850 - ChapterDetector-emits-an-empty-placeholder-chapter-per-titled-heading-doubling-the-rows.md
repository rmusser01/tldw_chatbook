---
id: TASK-16850
title: 'ChapterDetector emits an empty placeholder chapter per titled heading, doubling the rows'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - bug
  - tts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-existing detector defect made newly visible by TASK-15773's table fix (PR #1710,
review residual 5); code re-read at dev `ee741cf10` confirms the shape unchanged.
`ChapterDetector.detect_chapters` (`tldw_chatbook/TTS/audiobook_generator.py:141-231`):

- For every titled heading match it appends a **placeholder** `Chapter` with
  `content=""` ("Will be filled later", `~:188-197`).
- When the *next* heading arrives, the accumulated body is appended as a **separate**
  untitled `Chapter {n}` entry (`~:168-180`) — the preceding placeholder is never
  back-filled.
- Only the **final** placeholder ever gets filled, by the "don't forget the last
  chapter" branch (`~:203-208`, `if chapters and not chapters[-1].content`).

Net effect, probed by the review on a 13-header book: **25 chapters**, alternating
`title='Chapter 1', content_len=0` placeholders with unnamed body rows — roughly 2x the
true count. The chapter table (now truthful post-15773) shows the doubled list, the
first selected row is an empty placeholder (which is why the editor preview shows
nothing on landing), and downstream audiobook generation iterates phantom empty
chapters bearing the real titles while the actual prose sits in untitled rows.

Fix direction: back-fill the pending titled placeholder with the accumulated body when
the next heading (or EOF) arrives, instead of appending the body as a separate chapter —
one titled Chapter per heading, numbering contiguous. Re-baseline
`Tests/UI/test_speech_audiobook_chapter_detection.py` counts deliberately (they
currently pin the doubled behavior implicitly via totals).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A document with N titled headings detects exactly N chapters, each carrying its own title AND its body text (born-red test on the 13-header shape)
- [ ] #2 No zero-content chapter is emitted for a heading that has body text; the untitled/edge shapes (preamble before the first heading, headerless documents) keep their current behavior, stated in the test
- [ ] #3 Detection and editor suites re-baselined and green, with the count changes justified in the notes
<!-- AC:END -->
