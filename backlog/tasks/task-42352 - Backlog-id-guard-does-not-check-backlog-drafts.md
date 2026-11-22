---
id: TASK-42352
title: "Backlog id guard does not check backlog drafts"
status: To Do
assignee: []
created_date: 2026-11-22 16:37
updated_date: 2026-11-22 16:37
labels:
- follow-up
- backlog
- tooling
priority: low
references:
- scripts/check_backlog_task_ids.py
- scripts/next_backlog_task_ids.py
- scripts/preflight.sh
- .github/workflows/derived-artifacts.yml
- backlog/drafts
- backlog/archive/drafts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Proposed by round 20's backlog fixer (fu/r20-backlog, 2026-11-22). Read at 59cc99e61b.

**Why.** `scripts/check_backlog_task_ids.py` scans `backlog/tasks`, `backlog/completed` and `backlog/archive/tasks` (`TASK_DIRS`, `:41-45`), and `scripts/next_backlog_task_ids.py` allocates from the same three. Neither looks at `backlog/drafts/` or `backlog/archive/drafts/`. Round 20 found draft-38943 sharing its number with TASK-38943, which dev's TASK-39147 cites; no check had noticed. Drafts are their own Backlog.md series (`DRAFT-<n>`), so the tools do not confuse them, but people cite bare numbers, and a draft promoted later needs a free task id. The draft was renumbered to DRAFT-42312 (5083c52f7b).

One equal pair already exists and is legitimate history: the archived DRAFT-1 and TASK-1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The guard reports two draft files in `backlog/drafts/` or `backlog/archive/drafts/` that hold the same draft id
- [ ] #2 The guard reports a draft whose number equals a task id, except pairs listed in an explicit, commented allowlist that starts with DRAFT-1 and TASK-1
- [ ] #3 The id allocator never hands out a number that a draft holds
- [ ] #4 Tests against scratch directories show each new report red before the change, and the guard passes on the current tree
- [ ] #5 The check runs where the existing id check runs: `./scripts/preflight.sh` and the required CI job
<!-- AC:END -->
