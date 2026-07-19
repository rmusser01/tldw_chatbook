---
id: TASK-293
title: MCP bridge hardening follow-ups from Phase 5 reviews
status: To Do
assignee: []
created_date: '2026-07-17 19:18'
labels:
  - mcp
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Non-blocking items from PR #675 final/pre-merge reviews: (1) wedged-loop >5s race can still double-record a bridge failure — strict exactly-once needs loop-side dedupe (execute_hub_tool skipping its record when its future was already cancelled); (2) execute_hub_tool's pre-try ValueError (non-local/builtin key) leaves zero records (unreachable via production compose today); (3) session approvals are name-keyed — hash-keying them would extend the rug-pull guard's reach to mid-run definition changes; (4) kill-switch denial records decision=denied with error=None, indistinguishable from a permission deny — pass error detail; (5) review hook gates against the unfiltered provider catalog — a skill named mcp__* could trigger a spurious approval card / persist an always-allow for a never-invoked MCP tool; (6) audit read_recent(200) disk I/O on every workbench sync — gate to audit-mode-active; (7) EntrySelected index TOCTOU vs background resync.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each item fixed or explicitly declined with a reason recorded in this task
<!-- AC:END -->
