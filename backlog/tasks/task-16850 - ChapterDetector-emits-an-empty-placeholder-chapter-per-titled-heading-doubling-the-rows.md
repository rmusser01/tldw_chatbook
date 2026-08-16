---
id: TASK-16850
title: 'ChapterDetector emits an empty placeholder chapter per titled heading, doubling the rows'
status: Done
assignee:
  - '@claude'
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
- [x] #1 A document with N titled headings detects exactly N chapters, each carrying its own title AND its body text (born-red test on the 13-header shape)
- [x] #2 No zero-content chapter is emitted for a heading that has body text; the untitled/edge shapes (preamble before the first heading, headerless documents) keep their current behavior, stated in the test
- [x] #3 Detection and editor suites re-baselined and green, with the count changes justified in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline the chapter suites at HEAD (`c8b951616`): detection UI suite, editor
   population-race + tuple-order suites, audiobooks client suite.
2. Probe the bug at HEAD: 13-header synthetic book through
   `ChapterDetector.detect_chapters` — expect ~25 alternating empty/full rows
   (review residual 5 shape).
3. Write a born-red unit suite `Tests/TTS/test_chapter_detector.py`: 13 headings
   → exactly 13 chapters, each titled AND carrying the body that follows ITS
   heading; edge pins for preamble ("Chapter 0" row, current behavior),
   headerless document ("Chapter 1" single row, current behavior), whitespace-only
   input ("Full Content" fallback), and back-to-back headings (empty chapter kept
   for the bodiless heading — the title must not be dropped). Run: red at HEAD.
4. Restructure `detect_chapters` to the classic scan: keep ONE open chapter at a
   time (`current_title` = the pending heading's title, or None for preamble); on
   a new heading, close the open chapter with the body accumulated since its
   heading; at EOF close the last. No placeholder append, no back-fill branch.
5. Green the born-red suite; re-run the baseline suites; correct any pins that
   encoded the doubled count (per task description); ruff on touched files.
6. Task notes + Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restructured `ChapterDetector.detect_chapters`
(`tldw_chatbook/TTS/audiobook_generator.py`) to the classic scan: exactly one
chapter is "open" at a time — `current_title` holds the pending heading's title
(None while still in preamble); a new heading closes the open chapter with the
body accumulated since its own heading, and EOF closes the last one the same
way. The placeholder append ("Will be filled later") and the back-fill-only-
the-last branch are gone.

**Born-red evidence**: at base `c8b951616`, the 13-header probe (the 15773
review's shape) detected **25** chapters alternating `title='Chapter 1'
content_len=0` / `title='Chapter 1' content_len=57` — exactly the review's
residual-5 finding. The new suite `Tests/TTS/test_chapter_detector.py` ran 5
failed / 4 passed at that HEAD (the 4 passers are the edge-preservation pins,
green by design); 9/9 green post-fix, with the pairing asserted (each chapter's
content == the body following ITS heading), not just non-emptiness.

**Edge decisions** (each probed at the base sha before restructuring, pinned in
the test docstrings per AC #2):
- Preamble before the first heading: kept as the historical untitled
  `number=0` / "Chapter 0" row.
- Headerless document: kept as one `number=1` / "Chapter 1" chapter (the old
  EOF branch's numbering — NOT the "Full Content" fallback, which fires only
  on whitespace-only input and is also kept).
- Back-to-back headings / trailing bodiless heading: the bodiless heading
  keeps an empty chapter — dropping it would silently eat a pasted title, and
  consumers already tolerate `content == ""` (pre-fix output was full of empty
  placeholders; the editor's duration estimate handles 0 words). AC #2's
  guarantee is scoped to headings that HAVE body text.
- Positions: a merged chapter spans heading line → last body line (the old
  placeholder pinned start=end=heading line; the old body rows already used
  the heading line as start).
- Duration/word-count estimates: nothing to recompute in the detector — it
  never set `estimated_duration`; both the editor preview
  (`chapter_editor_widget._update_chapter_preview`) and
  `AudioBookProgress.estimated_duration` derive from content at use time, and
  content is now correctly attached.

**Re-baselining (AC #3)**: no existing pin actually encoded the doubled count.
`Tests/UI/test_speech_audiobook_chapter_detection.py` asserts
`len(detected_chapters) >= 1` / truthiness / stub-chapter lists, and the
editor suites use synthetic `_chapters(n)` fixtures — so **zero count
corrections were needed**; the whole pre-fix baseline set (13 tests) passes
unchanged post-fix, plus the 9 new detector tests (22 total). Broader sweep:
`Tests/TTS/` + audiobook/STTS UI suites = 4105 passed; the 9 failures there
(audio_cpp managed-integration, app-ownership drain, connection-error copy,
request-admission ×4, stts-capability ×2) were re-run against the pristine
base detector via an edit-safe file swap and fail identically at HEAD —
pre-existing on the base, unrelated to this change.

**Files**: `tldw_chatbook/TTS/audiobook_generator.py` (detector restructure),
`Tests/TTS/test_chapter_detector.py` (new born-red suite).
<!-- SECTION:NOTES:END -->
