---
id: TASK-1343
title: 'Local agent tools phase 3b-ii: git tools (read-only port)'
status: In Progress
assignee: []
created_date: '2026-08-05 20:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md §2.5. Plan: Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3b-ii.md. ADR-033 (process boundary). Port of tldw_server git_module.py @ 5605b9d9, async-to-sync.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 git subprocess wrapper enforces fixed-argv allowlist (subcommands + -C/--no-pager only), sanitized env, timeout, 1 MB output cap
- [ ] #2 git_status/git_diff/git_log/git_blame/git_branches work against workspace-confined repos
- [ ] #3 Non-repo paths, git-unavailable, and disallowed invocations return model-actionable errors (no raw exceptions)
- [ ] #4 Injection attempts (flag smuggling via args, path escapes) are refused
- [ ] #5 All new tests pass
<!-- AC:END -->


## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3b-ii.md
<!-- SECTION:PLAN:END -->
