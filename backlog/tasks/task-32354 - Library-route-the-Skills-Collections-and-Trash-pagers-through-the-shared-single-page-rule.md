---
id: TASK-32354
title: >-
  Library: route the Skills, Collections and Trash pagers through the shared
  single-page rule
status: To Do
assignee: []
created_date: '2026-09-11 06:17'
labels:
  - library
  - ux
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 100x30 the Skills canvas spends four lines on '1-2 of 2 / Page 1 of 1 / Already on the first page. / ○ Previous ○ Next' for two items; at 60x24 that is 4 of 18 rows and pushes the second skill off-screen (A caps 60/61). PROVEN: library_pager_layout (task-28016/32104) is imported by the media, conversations and prompts canvases only; library_skills_canvas.py composes its own _compose_pager. Pinned by test_skills_canvas_renders_exact_pager_and_source_wide_trust_count — a design decision to reverse deliberately. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Skills, Collections and Trash render no pager chrome when everything fits on one page, like Media/Conversations/Prompts
- [ ] #2 The existing pin is updated to the shared rule
<!-- AC:END -->
