---
id: TASK-1344
title: 'Local agent tools phase 3c: web-research skill'
status: In Progress
assignee: []
created_date: '2026-08-05 22:07'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md §2.6. Plan: Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3c.md. Includes skill-runner local-tool narrowing (disclosed spec deviation).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Skill-run subagents can be narrowed against local tools (never granted beyond the parent's allow-list)
- [ ] #2 A skill declaring web_search/web_fetch in allowed-tools gets exactly those (plus requested builtins); undeclared skills behave as before
- [ ] #3 web-research skill definition parses and passes trust scanning
- [ ] #4 Install documentation exists
- [ ] #5 All new tests pass
<!-- AC:END -->


## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3c.md
<!-- SECTION:PLAN:END -->
