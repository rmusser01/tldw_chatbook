---
id: TASK-21145
title: 'First-chat handoff: no jargon interception, actionable send-blocked state'
status: To Do
assignee: []
created_date: '2026-08-25 06:14'
labels:
  - ux
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings H-2, H-3 (findings.md section H): on a fresh profile the first message triggers a 'Project instructions need a folder' dialog exposing raw no_eligible_binding; with a broken provider the composer says 'Send blocked - finish provider setup to continue' with no way to reach that setup, and validation can sit 30s+ with no error or cancel.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typing a first message on a fresh profile is never intercepted by the project-instructions folder dialog
- [ ] #2 The send-blocked state offers a working affordance that opens provider setup
- [ ] #3 Provider validation surfaces a terminal result (success or actionable error) within a bounded time, with the run cancellable
- [ ] #4 No raw internal error codes are shown to the user in this flow
<!-- AC:END -->
