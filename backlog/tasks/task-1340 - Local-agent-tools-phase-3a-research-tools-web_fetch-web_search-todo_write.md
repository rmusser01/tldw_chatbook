---
id: TASK-1340
title: 'Local agent tools phase 3a: research tools (web_fetch/web_search/todo_write)'
status: In Progress
assignee: []
created_date: '2026-08-05 15:09'
updated_date: '2026-08-05 15:10'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md (phase 3a). Plan: Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3a.md. ADRs: 032, 033.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 web_fetch refuses private/loopback/link-local targets and non-http(s) schemes, including on redirect hops
- [ ] #2 web_fetch enforces redirect cap, timeout, byte caps, per-domain rate limit, and TTL cache
- [ ] #3 web_search delegates to perform_websearch with bounded per-result size
- [ ] #4 todo_write mutates per-session state and renders in the transcript
- [ ] #5 Agent system prompt hints at find_tools/load_tools discovery
- [ ] #6 All new tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3a.md
<!-- SECTION:PLAN:END -->
