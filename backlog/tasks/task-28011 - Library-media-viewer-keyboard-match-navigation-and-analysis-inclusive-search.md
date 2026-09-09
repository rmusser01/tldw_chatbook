---
id: TASK-28011
title: Library media viewer - keyboard match navigation and analysis-inclusive search
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
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
- [ ] #1 Enter in the search input advances to the next match
- [ ] #2 Prev and next match are reachable by keyboard and advertised
- [ ] #3 Search finds text in the analysis section, not just content
<!-- AC:END -->
