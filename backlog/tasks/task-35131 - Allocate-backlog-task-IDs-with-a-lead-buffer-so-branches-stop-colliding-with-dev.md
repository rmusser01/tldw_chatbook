---
id: TASK-35131
title: Allocate backlog task IDs with a lead buffer so branches stop colliding with
  dev
status: In Progress
assignee: []
created_date: 2026-10-16 09:15
labels:
- tooling
- backlog-hygiene
dependencies: []
priority: high
updated_date: 2026-10-16 16:35
---

## Description

dev mints task IDs continuously (~2/hour across active sessions). Any branch that picks "next free ID" at task-creation time collides with dev's later minting before the PR lands, and the `Backlog task IDs are unique` required check evicts the PR from the merge queue. This happened FOUR consecutive times on the UI-perf effort's PRs (#3083 and #3119, 2026-10-14/15): ids 34948-34952, then 34949-34951, then 34991/34992, then 35026/35027 — each renumbered to "next free", each re-collided within hours. The eviction that finally stuck used a +30 lead buffer (35058/35059) and survived ~15h of dev minting. The class of failure is structural: allocation races minting. This task removes the race.

## Acceptance Criteria

- [ ] A helper (e.g. `scripts/next_backlog_task_ids.py N [--buffer B]`) prints N free task IDs starting at `max(existing across tasks/completed/archive) + buffer` (default buffer 30), verifying each is unused in all three directories the CLI resolves ids in
- [ ] The helper is documented in `backlog/docs/` so agents and humans reach for it instead of eyeballing `ls backlog/tasks/`
- [ ] `backlog/docs/lessons-backlog-hygiene.md` gains the incident entry (four evictions, dates, PR numbers, the +30 fix) per the lessons format — incident, not just the rule
- [ ] The backlog-hygiene guidance tells task creators to allocate with the helper (or manually with a lead buffer) whenever creating tasks on a branch that will ride the merge queue

## Implementation Plan

1. Write `scripts/next_backlog_task_ids.py` (stdlib-only like the other derived-artifact checkers): scan `backlog/tasks`, `backlog/completed`, `backlog/archive/tasks`, print N ids from max+buffer, `--buffer` flag, refuse to print an id that exists anywhere
2. Add tests next to the other checker tests (tmp-dir fixture with planted task files at the max, assert printed ids skip them and honor the buffer)
3. Write the lessons entry with the incident record from PRs #3083/#3119
4. Update the hygiene doc's task-creation guidance

## Implementation Notes

(added when implemented)
