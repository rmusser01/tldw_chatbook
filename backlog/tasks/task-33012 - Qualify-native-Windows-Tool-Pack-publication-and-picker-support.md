---
id: TASK-33012
title: Qualify native Windows Tool Pack publication and picker support
status: To Do
assignee: []
created_date: '2026-09-27 15:13'
labels:
  - tool-packs
  - windows
  - security
dependencies: []
references:
  - backlog/decisions/107-portable-tool-use-packs.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide native Windows picker and secure create-only publication support for policy-only Tool Packs under ADR-107.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Native import and export pickers return centrally validated paths and report recoverable cancellation or unsupported operations.
- [ ] #2 Publication creates complete archives only at absent reviewed destinations, rejecting overwrite, reparse points and stale destination identity.
- [ ] #3 Native Windows verification covers successful publication, replacement races, existing targets, ACL denial, interruption and reparse-point attacks.
- [ ] #4 A canonical ADR records the Windows publication security model and user documentation reports verified support accurately.
<!-- AC:END -->
