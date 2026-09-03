---
id: TASK-28021
title: >-
  Library media viewer - match count and highlight undercount multiple matches
  per line
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
updated_date: '2026-09-02 21:07'
labels:
  - library
  - bug
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
find_content_matches reports a line at most once and the raw-mode highlighter styles only the first occurrence per line (library_media_viewer_state.py:377-381; library_media_content.py:43-53), so Match 3 of 7 undercounts dense lines; rendered mode gets no highlights at all (sync_search targets the raw widget only).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The match count reflects all occurrences
- [ ] #2 All occurrences on a line are highlighted in raw mode
- [ ] #3 Rendered mode either highlights matches or states why it cannot
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RECON (not started — split by AC): AC#2 (highlight ALL occurrences per line in raw mode) is CONTAINED to library_media_raw_view.py (_hit_for_line returns all offsets; loop stylize in render_line) + flip Tests/Library/test_library_media_raw_view.py::test_highlight_styles_only_the_first_occurrence_on_a_wrapped_line. AC#3 (rendered/Markdown mode) is DOCUMENT-ONLY ('states why it cannot'): sync_search targets the raw widget only; Textual Markdown exposes no per-source-line span-styling API. AC#1 (coherent 'Match X of Y' occurrence count) is NOT contained: find_content_matches returns LINE indices (each line once); making the count occurrence-accurate while Prev/Next stay per-line yields an incoherent X-of-Y in different units. Coherent count forces per-occurrence navigation -> change the primitive's return type (tuple[int,...] -> (line,col)) rippling through ~8 sites/4 files (screen memo+advance+scroll, viewer compose x2, sync signatures, set_match_lines/_hit_for_line, _status_text) + flip Tests/Library/test_library_media_viewer_state.py + test_library_media_content.py + test_library_media_reader_match_nav_t22209.py. Recommend own PR.
<!-- SECTION:NOTES:END -->
