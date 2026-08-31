---
id: TASK-25710
title: Home content recents stream + resume banner
status: Done
assignee: []
created_date: '2026-08-30 23:39'
updated_date: '2026-08-31 02:10'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Merge conversations/notes/media into Home's Recent section from the content snapshot pipeline, promote the resume banner to the newest content item (media becomes a resume kind), retire the limit-1 resume queries, add a row Open control (spec 2026-08-29 §1-2)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Recent section shows mixed content recents newest-first capped at 8
- [x] Banner resumes newest content item incl. media with relative age
- [x] Conversation banner/row opens that conversation in Console via nav context
- [x] Note/media rows open their Library views
- [x] limit-1 resume seam queries retired
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Content recents via content-snapshot pipeline (limit-8, merged in pure dashboard_state); banner promoted incl. media kind with age suffix; limit-1 _home_resume_fields retired; row Open control + resume dispatch route by prefixed id; conversation resume deep-links via ADR-079 seam. Controller-implemented per ledger ruling; subagent review gate applied.
<!-- SECTION:NOTES:END -->
