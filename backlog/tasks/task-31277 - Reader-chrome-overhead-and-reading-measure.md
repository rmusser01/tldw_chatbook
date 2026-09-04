---
id: TASK-31277
title: Reader chrome overhead and reading measure
status: To Do
assignee: []
created_date: '2026-09-04 13:54'
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
- [ ] #1 The identity line renders only for server or external items
- [ ] #2 The empty byline row is collapsed when there is no author or URL
- [ ] #3 The section header no longer repeats the selected tab
- [ ] #4 Text bodies are capped at a readable measure (about 90 cells) without breaking raw or rendered modes
- [ ] #5 Transcript bodies that contain markdown headings render them (or the sniff covers those media types); no literal `##`
- [ ] #6 Before and after captures in the notes
<!-- AC:END -->
