---
id: TASK-16801
title: 'Change review: git commit/push/PR modes'
status: To Do
assignee: []
created_date: '2026-08-15'
labels:
  - console
  - change-review
  - git
dependencies:
  - TASK-1972
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Change review (TASK-1972 and the turn file card that presents it) only
shows the working-tree diff a turn produced — there is no way to act on
it with git from inside the app, and reviewing more than one file at a
time means opening each turn's diffs one at a time. When the workspace
happens to be a real git repository, a user who wants to commit, push, or
open a pull request for an agent's changes has to leave Console and drop
to a shell for everything past inspecting the diff.

This is V2 of the turn file card design
(`Docs/superpowers/specs/2026-08-15-console-turn-file-review-design.md`,
"Out of scope" section), which names two related additions together:
a sidebar multi-file review view (see across several changed files at
once, instead of one card/turn at a time), and contextual `current` /
`commit` / `push` / PR actions that only appear when the workspace is a
git repository and each action's own precondition is met (a configured
remote for push, a supported git host for PR creation, etc.).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A sidebar multi-file review view lets the user see and navigate several of a turn's (or several turns') changed files at once, rather than only one card/turn at a time
- [ ] #2 When the workspace is a git repository, change review offers contextual actions for `current` (working tree), `commit`, `push`, and opening a PR
- [ ] #3 Each action is offered only when its own precondition holds (e.g. push requires a configured remote, PR creation requires a supported git host) and explains why it is unavailable otherwise, rather than failing silently or with a raw error
- [ ] #4 Commit and push never run without an explicit confirmation step, consistent with the no-silent-destructive-action precedent already established for revert (TASK-1845/TASK-1972/TASK-1974)
- [ ] #5 A workspace that is not a git repository, or one where change tracking is degraded, shows none of the git-contextual modes rather than a broken or erroring control
- [ ] #6 Commit and push are exercised end-to-end against a real local git repository in tests (a temp repo with a local bare remote), not only against mocked git calls
<!-- AC:END -->
