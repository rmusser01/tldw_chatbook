---
id: TASK-35131
title: Allocate backlog task IDs with a lead buffer so branches stop colliding with
  dev
status: Done
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

- [x] A helper (`scripts/next_backlog_task_ids.py N [--buffer B]`) prints N free task IDs starting at `max(existing across tasks/completed/archive) + buffer` (default buffer 30), verifying each is unused in all three directories the CLI resolves ids in
- [x] The helper is documented in `backlog/docs/` so agents and humans reach for it instead of eyeballing `ls backlog/tasks/`
- [x] `backlog/docs/lessons-backlog-hygiene.md` gains the incident entry (four evictions, dates, PR numbers, the +30 fix) per the lessons format — incident, not just the rule
- [x] The backlog-hygiene guidance tells task creators to allocate with the helper (or manually with a lead buffer) whenever creating tasks on a branch that will ride the merge queue

## Implementation Plan

1. Write `scripts/next_backlog_task_ids.py` (stdlib-only like the other derived-artifact checkers): scan `backlog/tasks`, `backlog/completed`, `backlog/archive/tasks`, print N ids from max+buffer, `--buffer` flag, refuse to print an id that exists anywhere
2. Add tests next to the other checker tests (tmp-dir fixture with planted task files at the max, assert printed ids skip them and honor the buffer)
3. Write the lessons entry with the incident record from PRs #3083/#3119
4. Update the hygiene doc's task-creation guidance

## Implementation Notes

- `scripts/next_backlog_task_ids.py`: stdlib-only CLI (same constraint family as the derived-artifacts checkers). Scans both claim surfaces (filename prefix and frontmatter `id:`) across `tasks/`, `completed/`, and `archive/tasks/` — the same three directories the uniqueness guard resolves ids in — then prints N ascending ids starting at `max + buffer` (default 30, derived from the incident: ~15h of headroom at dev's observed ~2 ids/hour mint rate). A skip loop guards ids that appear concurrently after the scan. `--buffer 0` is rejected with a message naming the race it would reintroduce.
- The lesson follows the doc's Incident/What-to-do shape and records the full eviction chain (34948-34952 -> 34949-34951 -> 34991/34992 -> 35026/35027, PRs #3083/#3119, the 94-minute final collision) — the evidence, not just the rule. The "What to do" points at the helper.
- Tests: `Tests/Scripts/test_next_backlog_task_ids.py`, 9 tests — all-three-directory + both-surface scanning (incl. frontmatter-only claims from hand-edited renames), buffer lead, sparse-high-claim raises the max, dotted-subtask stems count via their integer part, empty tree, default-buffer pin, invalid-argument exits. 9/9 pass; the full `Tests/Scripts/` directory green alongside the existing checker tests.
- Dogfooding: this task itself was filed at id 35131 (max 35100 + 31) and the branch rides the queue with that lead.
- ADR required: no — a helper script plus documentation; no storage/sync/contract change.
