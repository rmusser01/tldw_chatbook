---
id: TASK-22511
title: Model shell_exec over the armed raw CLI
status: To Do
assignee: []
created_date: '2026-08-27 04:53'
labels:
  - console
  - tools
  - security
  - agents
dependencies:
  - TASK-18926
  - TASK-22509
references:
  - backlog/decisions/093-raw-and-virtual-cli-execution-boundaries.md
  - Docs/superpowers/specs/2026-08-26-raw-and-virtual-cli-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a model request the same dangerous one-shot host-shell executor as the Console user command, but only through an unmistakable command-visible approval boundary with no persistent silent grant.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One shell_exec model tool accepts command, shell selector, optional absolute initial directory, and a timeout that cannot exceed 300 seconds
<!-- AC:END -->
