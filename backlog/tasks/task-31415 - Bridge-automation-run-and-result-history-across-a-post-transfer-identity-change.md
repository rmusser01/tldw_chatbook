---
id: TASK-31415
title: >-
  Bridge automation run and result history across a post-transfer identity
  change
status: To Do
assignee: []
created_date: '2026-09-04 22:40'
labels:
  - scheduling
  - sync
  - correctness
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A definition's execution history becomes invisible when the definition changes owner.

`ScheduledTasksDB.list_automation_runs` and `list_automation_results` each filter on a single `definition_id`, but a run or result carries whichever id the side that PRODUCED it used: a locally-executed run carries the local row id, while a mirrored one carries the SERVER's id (`upsert_automation_results_from_server` copies `definition_id` verbatim — it has no local id to translate it to). After a to-server transfer, `adopt_server_definition_identity` links the row by setting `server_id` while keeping the local `id`, so one definition now legitimately answers to two ids — and the history queries know only one of them at a time. Runs and results recorded before the transfer therefore do not appear under the id the history is fetched with after it.

The to-local direction is worse: that leg creates a fresh local row (`from_server_pending`), so the server-side history does not follow the definition at all.

The definition LOOKUP half of this problem is already solved and is the model for the fix: `index_definitions_by_id` (`UI/Screens/scheduling/unified_rows.py:388`) indexes each row under BOTH id spaces, which is how PR-6 live round 1's defect (3) was fixed — mark-solved had been refusing for exactly the server-owned rows it targeted, because eligibility keyed on the local id while a synced result carried the server id. The QUERY half was never bridged; the handoff spec parks "Run-history sync-down (server_id column reserved)" in its section 12 out-of-scope list, and the redesign surfaced the history rows on the pane without changing how they are fetched.

Ledger trail: `backlog/docs/spec-2026-08-31-schedules-handoff-parity.md` sections 9 and 12; the PR-6 live record in `backlog/tasks/task-18940`'s progress log (defect 3 and its fix).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A transferred definition's run history shows runs recorded under both its local id and its server id
- [ ] #2 The same holds for its results, in the inbox and in the pane's results rows
- [ ] #3 Identity resolution is one shared seam, not a bridge duplicated at each call site
- [ ] #4 A definition that has never transferred returns exactly the rows it returns today, with no extra rows and no changed ordering
- [ ] #5 A to-local transfer leaves the definition's prior server-side history reachable, or the task records why that is server-side work and names the gap in the user guide
<!-- AC:END -->
