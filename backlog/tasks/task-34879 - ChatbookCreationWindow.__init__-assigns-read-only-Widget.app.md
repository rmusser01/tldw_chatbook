---
id: TASK-34879
title: ChatbookCreationWindow.__init__ assigns read-only Widget.app
status: Done
assignee: []
created_date: '2026-09-19 21:20'
updated_date: '2026-09-20 06:04'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-existing bug found during wide-tier wave 2: ChatbookCreationWindow.__init__ tries to assign self.app which is a read-only property on Textual Widget — the class is un-constructible in its current form. Documented in .superpowers/wave2-report.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ChatbookCreationWindow can be constructed and mounted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in PR #2751 (commit 481f4e217b): removed the self.app assignment and app_instance parameter; class is constructible again. Merged to dev as 9929b48be52.
<!-- SECTION:NOTES:END -->
