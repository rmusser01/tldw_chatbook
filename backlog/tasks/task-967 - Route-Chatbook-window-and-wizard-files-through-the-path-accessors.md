---
id: TASK-967
title: Route Chatbook window and wizard files through the path accessors
status: To Do
assignee: []
created_date: '2026-07-27 18:06'
labels:
  - config
  - chatbooks
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While completing TASK-865's sweep, the Chatbook window and wizard files were found bypassing the config path accessors and composing user-data and config paths directly. This is the same drift class the audit exists to close: a literal that is correct today and silently wrong the moment the app resolves that path differently. Deliberately left out of TASK-865's scope so the sweep could land, and recorded here rather than lost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Chatbook window and wizard files derive their paths from the accessors,No hardcoded ~/.config/tldw_cli or ~/.local/share/tldw_cli literal remains in those files,A test derives its expected path the same way the app does rather than re-spelling a literal
<!-- AC:END -->
