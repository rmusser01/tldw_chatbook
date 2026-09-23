---
id: TASK-32917
title: Gate server audio diagnostics by administrator role
status: In Progress
assignee: []
created_date: '2026-09-23 20:15'
updated_date: '2026-09-23 20:20'
labels: []
dependencies: []
references:
  - backlog/decisions/178-server-audio-diagnostic-admin-boundary.md
  - 'https://github.com/rmusser01/tldw_server/pull/2968'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Match Chatbook connected-mode audio diagnostics to the server administrator authorization contract so ordinary users see an explicit admin-required outcome.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Passive STT health remains usable for ordinary connected users.
- [ ] #2 Warm STT health and streaming diagnostics require server admin identity before dispatch.
- [ ] #3 A server 403 for either diagnostic becomes an explicit admin-required result.
- [ ] #4 Targeted connected-mode tests cover admin and nonadmin outcomes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-09-23-server-audio-diagnostic-admin-parity.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR: https://github.com/rmusser01/tldw_chatbook/pull/2822. Targeted service/scope tests: 15 passed, 1 existing from_config bootstrap failure excluded; Ruff/compileall/diff checks passed.
<!-- SECTION:NOTES:END -->
