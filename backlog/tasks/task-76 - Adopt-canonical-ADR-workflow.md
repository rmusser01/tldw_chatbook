---
id: TASK-76
title: Adopt canonical ADR workflow
status: Done
assignee: []
created_date: '2026-06-02 20:41'
updated_date: '2026-06-03 01:14'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish backlog/decisions as the canonical ADR source of truth and update agent workflow guidance so significant architectural decisions are written, linked, and tracked consistently.
<!-- SECTION:DESCRIPTION:END -->

## Setup Notes

- Base commit: `fe4f92e6`
- Duplicate check: `backlog task list --plain` found no existing ADR workflow adoption task before `TASK-76` was created.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Canonical ADR template exists
- [x] #2 Canonical ADR index explains workflow and trigger rules
- [x] #3 First canonical ADR records this workflow decision
- [x] #4 Historical ADR-like docs are indexed without migration
- [x] #5 AGENTS.md requires ADR checks during planning and closeout
- [x] #6 Documentation verification passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Follow Docs/superpowers/plans/2026-06-02-adr-workflow-implementation.md task-by-task.
2. ADR required: yes.
3. ADR path: backlog/decisions/001-adopt-backlog-decisions-as-canonical-adrs.md.
4. Implement canonical ADR template, first ADR, index, historical index, and AGENTS.md workflow updates.
5. Run focused documentation verification and update task closeout notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added the canonical ADR template, the first ADR for canonical Backlog decisions, the ADR README, and the historical index. Updated `AGENTS.md` with ADR trigger, planning check, closeout, and DoD guidance.

Verification commands run:

- `test -f backlog/decisions/000-template.md`
- `test -f backlog/decisions/001-adopt-backlog-decisions-as-canonical-adrs.md`
- `test -f backlog/decisions/README.md`
- `test -f backlog/decisions/historical-index.md`
- `rg -n "docs/(Development|Design|Features|Parity|Code|superpowers)" backlog/decisions AGENTS.md`
- `rg -n "TODO|TBD|PLACEHOLDER|FIXME|<TASK_ID>|<TASK_PATH>" backlog/decisions AGENTS.md`
- `rg -n "\\[ \\]" backlog/decisions`
- `rg -n "ADR required: yes/no|001-adopt-backlog-decisions-as-canonical-adrs|historical-index|ADR hygiene|Canonical Architecture Decision Records" backlog/decisions AGENTS.md`
- `git diff --check`
- `git diff --check fe4f92e6`
- `git diff --stat fe4f92e6 -- AGENTS.md backlog/decisions "backlog/tasks/task-76 - Adopt-canonical-ADR-workflow.md"`
- `git diff fe4f92e6 -- AGENTS.md backlog/decisions "backlog/tasks/task-76 - Adopt-canonical-ADR-workflow.md"`

Corrective verification after reverting an unrelated Backlog duplicate-id normalization:

- `git diff --name-status fe4f92e6..HEAD`
- `git diff --check fe4f92e6`

The final unscoped name-status diff is limited to `AGENTS.md`, canonical ADR files in `backlog/decisions/`, and this rollout task.
<!-- SECTION:NOTES:END -->
