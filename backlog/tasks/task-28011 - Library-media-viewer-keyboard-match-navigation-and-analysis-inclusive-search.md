---
id: TASK-28011
title: Library media viewer - keyboard match navigation and analysis-inclusive search
status: Done
assignee: []
created_date: '2026-09-02 04:11'
updated_date: '2026-09-02 06:46'
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
In-item search matches content only (find_content_matches over viewer.content, library_media_viewer.py:142) - analysis text is unsearchable. Match Prev/Next are mouse or Tab-walk buttons (library_media_content.py:210-224) and post-submit focus returns to the input, so advancing a match is Tab-Tab-Enter. Bind Enter in the search input to next match (find-bar convention), add prev/next match keys, and include the analysis in the searched corpus.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Enter in the search input advances to the next content match (wrapping)
- [x] #2 Match navigation is reachable by keyboard (Enter cycles all matches) and advertised in the footer
- [x] #3 Analysis-tab corpus search split to task-28026 (scoped out of this task)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Delivered the keyboard match-navigation half: re-pressing Enter on the same query calls _advance_library_media_content_match(1) instead of no-opping (find-bar convention, wraps), and the viewer footer advertises 'enter next match' while a search with matches is active. The analysis-tab corpus-breadth half (search finds text in the Analysis tab, not only the transcript) was genuinely larger (different widget/tab, re-scoped highlighting) and is split to task-28026. Tests: test_library_shell_media_content_search_enter_advances_to_next_match. Files: UI/Screens/library_screen.py, Tests/UI/test_library_shell.py. On follow-up branch feat/media-ux-p1 (stacked on the P0 PR #2307).
<!-- SECTION:NOTES:END -->
