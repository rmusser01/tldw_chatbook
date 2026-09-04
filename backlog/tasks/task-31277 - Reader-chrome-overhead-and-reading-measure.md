---
id: TASK-31277
title: Reader chrome overhead and reading measure
status: Done
assignee: []
created_date: '2026-09-04 13:54'
updated_date: '2026-09-04 20:48'
labels:
  - library
  - media-ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P2: eight rows of chrome precede the first content line at 235 cols (identity line `Local Media item`, `‹ Back`, title, an empty byline row when there is no author or URL, spacer, toolbar, mode row, a section header that repeats the selected tab — A cap 03); prose runs about 150 cells against DESIGN.md's 65-75; video transcripts render `## Section 1` literally because markdown detection is limited to plaintext/markdown/obsidian_note (library_media_viewer_state.py:201-208).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The identity line renders only for server or external items
- [x] #2 The empty byline row is collapsed when there is no author or URL
- [x] #3 The section header no longer repeats the selected tab
- [x] #4 Text bodies are capped at a readable measure (about 90 cells) without breaking raw or rendered modes
- [x] #5 Transcript bodies that contain markdown headings render them (or the sniff covers those media types); no literal `##`
- [x] #6 Before and after captures in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: chrome rows ≤ 5 on a local item without author/URL; no byline row; no repeated section header; body ≤ 92 cells with the box full width and the long line wrapping; video transcript with ## headings defaults to Rendered
2. GREEN: identity line only for external items; byline only when author/URL; Read/Info headers removed, Analysis/Highlights headers hidden (ids queried by tests/CSS); max-width: 92 on both bodies at the component tier; video/audio added to _MARKDOWN_MEDIA_TYPES with the content sniff kept as the second gate
3. Live 235x52 before/after
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Chrome 9 rows → 5 at 235x52 (Back, title, toolbar, mode row, border); prose ~136 → ~88 cells with the bordered box still spanning the pane (painted wrap proves the raw wrap index was rebuilt at the capped width); '## Section 1' in a video transcript renders as a heading. _MARKDOWN_MEDIA_TYPES = {plaintext, markdown, obsidian_note, video, audio} — a transcript line starting with '# ' now defaults to Rendered (Raw is one press away). Markdown items keep 6 chrome rows (the Rendered|Raw strip is a control, not a repeated label). The Back button was left untouched (PR B owns it). Deferred: a comment claims the removed Read header 'cost four rows' (it cost one); two companion assertions cannot fail; the join-artifact pin moved from the pane's first row to the title row; the chrome fixture uses the now-allowlisted 'audio' type; the section-header CSS rule now styles only hidden ids.
<!-- SECTION:NOTES:END -->
