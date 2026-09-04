---
id: TASK-31450
title: >-
  Console Inspect rail Environment redesign: git/PR/CI/tasks/agents panel
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-04 20:30'
labels:
  - console
  - inspector
  - ux
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebuild the Console Inspect rail around a Codex-style Environment panel so a
user doing agentic development can answer "what changed, is CI green, which
task is this, what are my agents doing?" without leaving the app. Owner-approved
spec: `Docs/superpowers/specs/2026-09-04-console-inspector-environment-redesign-design.md`;
implementation plan: `Docs/superpowers/plans/2026-09-04-console-inspector-environment-redesign.md`.
ID leapfrogged to 31450 after a sweep (dev max 31383, branch-name max 26042).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Inspect rail shows an Environment section with live git working-tree change counts, branch (ahead/behind, worktree marker), and an execution-target row, bound to the active workspace root; non-git workspaces show one quiet empty row
- [ ] #2 PR and CI-checks rows appear via the gh CLI when available and silently hide when gh is absent/unauthenticated/no-PR; failures keep last good data with a stale marker and back off after 3 consecutive errors
- [ ] #3 A Tasks section shows the branch-linked backlog task (with AC progress) or status counts, expanding to a scrollable In-Progress-first list; absent without a backlog/ dir
- [ ] #4 The agent fleet section renders in the Inspect rail (moved from the left rail) with its existing behavior intact
- [ ] #5 Row actions work: Changes→Change Review, Commit-or-push→Change Review in working-tree mode, PR open-in-browser, PR/checks/task add-to-composer; all keyboard-reachable
- [ ] #6 Environment/Tasks collapse state persists per workspace; no new I/O on the 0.2s tick; zero new boot cost while the rail is collapsed
- [ ] #7 Both-seams test coverage (projection + screen wiring) and a live 80x24 tmux verification with captures in Implementation Notes
<!-- AC:END -->

## Implementation Plan

See `Docs/superpowers/plans/2026-09-04-console-inspector-environment-redesign.md`
(14 TDD tasks: pure state → gatherers → persistence/composition → orchestration/wiring → docs+live verification).
