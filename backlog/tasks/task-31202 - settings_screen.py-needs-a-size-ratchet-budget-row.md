---
id: TASK-31202
title: settings_screen.py needs a size-ratchet budget row
status: To Do
assignee: []
created_date: '2026-09-02 19:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec 2026-09-01 non-goal follow-up: the ratchet that let library_screen triple also has no settings row; settings_screen.py was 15,922 lines at the 2026-08-02 doctrine baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Budget row added at measured values
- [ ] #2 Mutation-checked (dummy method -> fails)
<!-- AC:END -->

## Renumbering provenance

Filed as `TASK-27020` on the wave-2 branch (2026-09-02 19:25). Merging
`origin/dev` (2026-09-03) surfaced a collision with dev's own
`TASK-27020` ("Webhooks fire needs-approval lifecycle event when a run
pauses for human approval"), an unrelated, already-filed task -- per the
2026-08-21 owner rule (TASK-19601) and the precedent recorded in
`backlog/docs/lessons-backlog-hygiene.md`, the OLDER arrival keeps the
id and this, the younger claimant, renumbers.

Renumbered `TASK-27020` -> `TASK-31202`, derived from a fresh sweep of
`refs/remotes/*` plus every local worktree at merge time (true max:
31201, the foundation's own `TASK-27019` -> `TASK-31201` renumber
already landed on `dev`), not this branch's own prior max (27021), per
the lesson file's warning against trusting a single ref's view. No code
or test referenced `task-27020`/`TASK-27020` (freshly filed, not yet
implemented); the SDD ledger references in
`.superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio/`
(`task-1-report.md`) were updated to the new id in the same merge
commit. The archival `review-*.diff` snapshots in that same directory
are frozen `git diff` captures of specific historical commit ranges and
were deliberately left unedited -- rewriting their content to a
different task number would make them no longer match the named commit
range, the same reasoning that keeps git log messages immutable.
