---
id: TASK-21252
title: Support secure Actor Pack import on Windows
status: To Do
assignee: []
created_date: '2026-08-24 00:11'
labels:
  - actor-packs
  - windows
  - security
dependencies: []
references:
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide native Windows staging, review, activation, and authenticated cleanup for Actor Pack imports without weakening the private-directory and link-traversal protections established for supported platforms.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Windows user can import, review, and activate a valid Actor Pack through the existing service contract.
- [ ] #2 Staging authority is verified using documented Windows ownership and ACL rules before any candidate is trusted or removed.
- [ ] #3 Reparse points and other link-like filesystem objects are rejected or handled without traversal, and ambiguous candidates are left untouched.
- [ ] #4 Startup cleanup removes only bounded candidates whose authenticity and private staging authority have both been proven on Windows.
- [ ] #5 Failure paths return stable Actor Pack error codes and do not delete user-controlled or unverifiable data.
- [ ] #6 Native Windows CI covers valid import, interrupted staging recovery, ACL denial, reparse-point attacks, and non-destructive cleanup refusal.
- [ ] #7 The governing ADR and user-facing platform support documentation describe the Windows security model and limitations.
<!-- AC:END -->
