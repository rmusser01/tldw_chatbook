---
id: TASK-33261
title: 'PERF-02: Conversation search runs its FTS MATCH once per candidate conversation
  (TASK-278 regression)'
status: To Do
created_date: 2026-09-28 18:01
labels:
- performance
- database
- search
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
search_conversations_page (DB/ChaChaNotes_DB.py ~11759) now filters with a correlated EXISTS(SELECT ... messages_fts MATCH ...). SQLite re-runs the full-text query once per candidate conversation. Measured at 150k messages: about 3 s for a rare term and about 70 s for a common one; the DB bench shows 1,124 ms versus 7 ms for the uncorrelated IN form. This is the TASK-278 rewrite (Done) regressing. It hits Ctrl+K History search, Library > Conversations search and the Console browser search. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-02; every issue with file:line is listed under PERF-02 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Conversation content search evaluates the messages_fts MATCH once per query, not once per candidate conversation
- [ ] #2 A plan-pin test captured with sqlite_stat1 absent fails on the correlated form and passes on the fix
- [ ] #3 Existing conversation-search behaviour tests pass unchanged (title/id matching, workspace scoping, deleted-message exclusion)
- [ ] #4 A seeded benchmark shows content search at 150k messages completing in under 100 ms
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
