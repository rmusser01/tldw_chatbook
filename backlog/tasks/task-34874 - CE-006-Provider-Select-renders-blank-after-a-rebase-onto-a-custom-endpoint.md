---
id: TASK-34874
title: 'CE-006: Provider Select renders blank after a rebase onto a custom endpoint'
status: Done
assignee: []
created_date: '2026-09-13 16:35'
updated_date: '2026-09-20 05:05'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Post-#2668 UAT re-run (Docs/superpowers/qa/console-custom-endpoints-uat-2026-09-13, captures 04b-06b): after creating/switching onto a custom-ep entry the modal's rebase completes without crashing, but the Provider Select renders blank and its dropdown lists no options until the modal is reopened. Cosmetic-but-confusing state right at the feature's moment of success.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Provider Select shows the entry display name immediately after post-Create auto-switch,Dropdown lists all providers incl. entries after the switch,Regression test drives the real rebase chain
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in PR #2736 (commit 7d181dc197): provider picker options now refresh before the post-Create switch value assignment. Merged to dev as 030fd9935d0.
<!-- SECTION:NOTES:END -->
