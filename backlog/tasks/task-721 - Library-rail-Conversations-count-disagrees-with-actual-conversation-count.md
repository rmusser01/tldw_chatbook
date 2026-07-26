---
id: TASK-721
title: Library rail Conversations count disagrees with actual conversation count
status: Done
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - library
  - bug
  - investigation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With six non-deleted conversations in the chat DB the Library rail showed Conversations (1) (captures cap-21/24). The count appears to exclude workspace-scoped conversations, disagreeing with the Console browser and reading as data loss. Root cause untraced. Finding m5.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The counting rule is identified and documented
- [x] #2 The displayed count matches what the Conversations view actually lists
- [x] #3 If workspace-scoped conversations are intentionally excluded the label says so
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the count: Library snapshot -> list_conversations(scope_type="all") -> search_conversations_page.
2. Red cross-client test in the task-179 regression suite; fix; verify scoped listings keep the filter.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rule identified: search_conversations_page ALWAYS filtered on the reading handle's client_id. The UAT's "Conversations (1)" vs 6 real rows: the app (CLI_APP_CLIENT_ID) saw only its own 1 conversation; the 5 seeded rows carried client "uat-seed" and were hidden from the Library Browse count/list, while the Console workspace browser lists membership rows with no client filter - a genuine cross-surface split-brain that would also hide server-synced or otherwise foreign-client rows. Fix: the browse-everything scope ("all") now skips the client_id clause (documented in the query builder); scoped (global/workspace) listings keep the historical client filter, locked by a companion test. Tests: 2 new in Tests/Library/test_library_conversations_visibility.py; visibility+service (48) and library-hub+DB (206) suites green.
<!-- SECTION:NOTES:END -->
