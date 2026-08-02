---
id: TASK-1970
title: 'Change review: ShadowRepoService — hardened per-root shadow git'
status: To Do
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - workspaces
  - change-review
  - agents
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Foundation for Agent Change Review. A service owning one shadow git repo per CANONICAL root path (symlinks resolved), GIT_DIR under the app data dir, `core.worktree` pointing at the root — the user's own `.git` is never touched. Every invocation passes explicit `--git-dir`/`--work-tree` with `GIT_*` env scrubbed. Init must pin the four configs that break on real machines: app-local `user.name`/`user.email` (commit fails without identity), `commit.gpgsign=false` (a global gpgsign signs or prompts on every snapshot), empty `core.hooksPath` (global husky-style hooks must not fire), `gc.auto=0`. Provides snapshot-commit, numstat, per-file diff/show, and a per-root lock (in-process + flock) with index.lock retry.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A snapshot commit of a scratch root succeeds in a scratch HOME configured with NO git identity, `commit.gpgsign=true`, and a global `core.hooksPath` whose hook would fail the run — all three hazards proven by test
- [ ] #2 The same root reached through a symlink and directly resolves to ONE shadow repo
- [ ] #3 No file or directory is ever created inside the tracked root (asserted by tree comparison)
- [ ] #4 Two concurrent snapshot calls on one root serialize instead of failing on index.lock
- [ ] #5 `shutil.which('git')` absent -> the service reports unavailable; nothing raises
- [ ] #6 Forced excludes (.git/, node_modules/, .venv/, __pycache__/, build dirs) live in info/exclude and are honored
<!-- AC:END -->
