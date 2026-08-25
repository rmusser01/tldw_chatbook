---
id: TASK-22209
title: >-
  Media reader match-navigation: one document pass per click
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - library
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22209).

Pre-existing, still live (the markdown re-parse half was fixed; this is the text half).
Per Prev/Next click: `_advance_library_media_content_match`
(`library_screen.py:33451-33484`) rebuilds the viewer state (full content copy), runs
`find_content_matches` over the whole document, then `sync_match_index -> sync_search ->
build_raw_content_renderable` runs a SECOND full `find_content_matches` plus a Rich `Text`
rebuild with up to 3 appends per line over the entire document
(`Widgets/Library/library_media_content.py:16-53`, `:112-134`). 3-4 O(document) passes per
click; noticeable on multi-MB transcripts.

## Acceptance Criteria

- [ ] Match navigation performs at most one O(document) scan per click (match list cached keyed on content identity + query; the renderable patches highlight styles rather than rebuilding, or its rebuild is measured and accepted)
- [ ] Measured before/after on a multi-MB document
- [ ] TASK-21134's layout=False mitigation on the search refresh is preserved
