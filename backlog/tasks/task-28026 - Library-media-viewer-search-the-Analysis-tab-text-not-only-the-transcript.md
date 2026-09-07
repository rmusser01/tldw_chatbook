---
id: TASK-28026
title: 'Library media viewer - search the Analysis tab text, not only the transcript'
status: Done
assignee: []
created_date: '2026-09-02 06:46'
updated_date: '2026-09-02 19:56'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the 2026-09-01 Library media UX critique (task-28011 split): in-item content search (find_content_matches over viewer.content) searches the transcript ONLY. When the user is reading an analysis in the Reader's Analysis tab, the search box does not find text in the analysis. For the sequential-review flow (reading analyses), searching within the analysis is as useful as searching the transcript. Extend the search corpus to include the active tab's text (analysis when the Analysis tab is showing), with matching/highlighting/scroll targeting that tab. task-28011 delivered the keyboard match-navigation (Enter advances, wrapping, footer-advertised) for the content corpus; this task is the remaining corpus-breadth half.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Searching while the Analysis tab is active finds and highlights matches in the analysis text
- [x] #2 Match count, prev/next, and Enter-advance operate over whichever tab's text is being searched
- [x] #3 Switching tabs re-scopes the search corpus without stale highlights
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Analysis tab is now searchable. When an analysis exists, _compose_analysis renders it in the SAME LibraryMediaContentSearchControls + LibraryMediaContentBody (raw mode) the Read tab uses, so the in-item find bar highlights matches in the analysis text (was a plain non-searchable Static). The screen's corpus (_library_media_content_matches) is mode-aware -- analysis text in the Analysis tab, transcript elsewhere -- with reader mode added to the memo key so a tab switch can't serve the other tab's matches. handle_library_media_reader_mode clears the search query/index/memo on a real tab switch (no stale highlights). Match count, Prev/Next, and Enter-advance all follow the active tab via the shared corpus. Tests: analysis-tab-is-searchable (status 'Match 1 of 2'), switching-tabs-clears-the-search; updated 3 existing analysis tests to read the analysis from the content body's .content (the non-empty analysis no longer uses #library-media-viewer-analysis-text -- that id remains for the empty/generating states). Test trap: the full test-file context recomposes the analysis tab via a background refresh, so wait for #library-media-analysis-edit to settle + poll the status rather than a fixed-pause query_one. Files: Widgets/Library/library_media_viewer.py, UI/Screens/library_screen.py, Tests/UI/test_library_shell.py.
<!-- SECTION:NOTES:END -->
