---
id: TASK-31201
title: Recompose census needs an anti-slack guard like the size ratchet
status: To Do
assignee: []
created_date: '2026-09-02 15:14'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final review of the library decomposition foundation: Tests/UI/test_library_recompose_ratchet.py pins a ceiling only; headroom drift happened twice before (107->80, 74->63). Mirror test_budget_is_not_left_slack_after_a_wave.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Census pin has a slack guard with a documented tolerance
- [ ] #2 Guard is mutation-tested (headroom injected -> fails)
<!-- AC:END -->

## Renumbering provenance

Filed as `TASK-27019` on this branch (2026-09-02 15:14). Merging
`origin/dev` (2026-09-03) surfaced a collision with dev's own
`TASK-27019` ("Document Personal Context Profile for Chatbook users and
developers", `created_date: 2026-09-01 14:45`, status Done, PR #2311) --
per the 2026-08-21 owner rule (TASK-19601) and its precedent already
recorded in `backlog/docs/lessons-backlog-hygiene.md` ("2026-09-02,
Personal Context documentation closeout": that same id had already
displaced one earlier collision, an MCP task renumbered to
`TASK-28228`), the OLDER arrival keeps the id and this, the younger
claimant, renumbers. `preflight.sh`'s backlog-task-ids check caught the
duplicate at merge time.

Renumbered `TASK-27019` -> `TASK-31201`, derived from a fresh sweep of
`refs/remotes/*` plus every local worktree (true max at rename time:
31200; see the sweep command in `lessons-backlog-hygiene.md`), not the
local worktree's own max (31001), per that same lesson file's warning
against trusting a single ref's view. No code, test, or other task file
referenced `task-27019`/`TASK-27019` for this task specifically (it was
freshly filed, not yet implemented), so the file rename plus this
frontmatter `id:` update are the only provenance-affected artifacts.
