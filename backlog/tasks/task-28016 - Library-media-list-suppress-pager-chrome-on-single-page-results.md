---
id: TASK-28016
title: Library media list - suppress pager chrome on single-page results
status: Done
assignee: []
created_date: '2026-09-02 04:11'
updated_date: '2026-09-02 21:06'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With three items, the list still renders the full pager including the lines Already on the first page and No more results - permanent noise on any single-page list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Single-page results show no pager status noise
- [x] #2 The pager appears when a second page exists
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a single_page flag to LibraryPagerDisplay (fresh, non-loading, one page) and used it in the media canvas _compose_pager to drop the 'Page 1 of 1' counter and the boundary reasons ('Already on the first page.', 'No more results.') on single-page results -- keeping the item range and the (disabled) controls so nothing shifts once a second page exists. Files: Library/library_pager_state.py, Widgets/Library/library_media_canvas.py; tests in Tests/Library/test_library_pager_state.py + Tests/UI/test_library_media_side_by_side.py. Note: the range line ('1-3 of 3') is retained (it is informative, not the noise the task named); AC#2 is satisfied as a regression guard -- multi-page pagers are unaffected.
<!-- SECTION:NOTES:END -->
