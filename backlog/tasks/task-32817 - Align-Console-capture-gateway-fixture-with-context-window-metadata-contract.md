---
id: TASK-32817
title: Align Console capture gateway fixture with context-window metadata contract
status: To Do
assignee: []
created_date: '2026-09-18 19:12'
labels:
  - test-health
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32815 targeted archive verification found both restore/resume/send variants fail while mounting because CapturingGateway lacks cached_context_window. Both exact cases reproduce on saved source 2cc702b85c, before the footer guard. Preserve the real resume/send assertions while restoring the provider fixture contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Console capture gateway fixture supports the current context-window metadata contract with truthful deterministic test data.
- [ ] #2 Both archive restore/resume/send variants pass with their original history, unrelated draft and persistence assertions intact.
- [ ] #3 Targeted fixture consumers are checked and the saved-source failure evidence is linked.
<!-- AC:END -->
