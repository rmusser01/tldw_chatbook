---
id: TASK-21241
title: >-
  TASK-21118 residue - unlocked mutation-generation increment and a stale
  Settings active marker
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - console
  - workspaces
  - concurrency
  - ux
dependencies: []
priority: low
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; the two Minors
ledgered when TASK-21118 merged (PR #2010, `736359202`).

TASK-21118 took the Console keystroke path from ~1.25 synchronous Workspace-DB reads per key
(measured: 20 keys → **25** `ensure_default_workspace` + **25** `get_active_workspace`) to
**0 + 0**, by memoizing the screen's workspace-context resolution against a
`mutation_generation` counter bumped by all seven workspace-record mutators. Staged-launch
evidence-bundle parsing went from 11 parses across 3 keys to ≤1. Two Minors were accepted:

1. `Workspaces/registry_service.py:228` increments `self._mutation_generation` with a bare
   `+=` — an unlocked read-modify-write. Every mutator runs on the UI thread today, so a lost
   increment is not reachable; nothing enforces that. A lost increment would serve a stale
   workspace context to the keystroke path until the next mutation.
2. The Settings "(active)" marker is not repainted in the edge state the memo never heals, so
   it can name a workspace that is no longer active. Cosmetic, but it is the one place the
   user is told which workspace is active.

## Acceptance Criteria

- [ ] A lost `mutation_generation` increment is impossible by construction, or the UI-thread-only constraint is enforced rather than assumed
- [ ] The Settings active-workspace marker matches the actual active workspace in the edge state where the memo is not re-read
- [ ] TASK-21118's keystroke measurement (0 registry calls across 20 keys) does not regress
