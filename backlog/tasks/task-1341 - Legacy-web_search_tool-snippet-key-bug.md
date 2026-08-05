---
id: TASK-1341
title: Legacy web_search_tool snippet-key bug
status: To Do
assignee: []
created_date: '2026-08-05 16:47'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during task-1340 review: Tools/web_search_tool.py:113 reads item.get('snippet') but perform_websearch normalizes body text to top-level 'content' (snippet only under metadata), so the legacy tool renders 'No description available' for every real result. The new web_search core (Tools/web_tool_impls.py) fixed this with snippet||content||default; apply the same one-line fix to the legacy tool.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Legacy web_search_tool renders result body text from real perform_websearch payloads,Tests pin the real normalized shape
<!-- AC:END -->
