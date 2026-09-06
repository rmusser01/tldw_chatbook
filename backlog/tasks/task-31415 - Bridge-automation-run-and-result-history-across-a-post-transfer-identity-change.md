---
id: TASK-31415
title: >-
  Bridge automation run and result history across a post-transfer identity
  change
status: Done
assignee:
  - '@claude'
created_date: '2026-09-04 22:40'
updated_date: '2026-09-06 14:07'
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
- [x] #1 A transferred definition's run history shows runs recorded under both its local id and its server id
- [x] #2 The same holds for its results, in the inbox and in the pane's results rows
- [x] #3 Identity resolution is one shared seam, not a bridge duplicated at each call site
- [x] #4 A definition that has never transferred returns exactly the rows it returns today, with no extra rows and no changed ordering
- [x] #5 A to-local transfer leaves the definition's prior server-side history reachable, or the task records why that is server-side work and names the gap in the user guide
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a private `_definition_id_aliases(owner_id, definition_id)` helper to ScheduledTasksDB: given either a definition's local id or server_id, look up the row and return the distinct {id, server_id} set; fall back to (definition_id,) when no row matches.
2. Route list_automation_runs and list_automation_results' definition_id filters through the helper (IN clause instead of equality), preserving ORDER BY exactly.
3. Investigate whether the same id-space ambiguity reaches the results count/unread-count path used by the inbox and pane badge (count_automation_results / count_unread_results); bridge it through the same helper if warranted, and check the workbench call sites that already hand-roll a dual-id-space loop (_definition_results_query, _definition_unread_result_ids) for regressions.
4. AC#5: grep the to-local (from_server_pending) transfer path for any persisted link from the new local row back to the origin server/mirror id; bridge it if the data exists locally, otherwise record the gap in the task file and the user guide's transfer section.
5. Add DB-layer tests: cross-identity bridge (revert-checked), never-transferred regression gate (pinned ordering), unknown-id/no-cross-bleed. Run the scheduling DB test module + Tests/Scheduling/ + Tests/UI/test_schedules_workbench.py.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added ScheduledTasksDB._definition_id_aliases(owner_id, definition_id): given either a definition's local id or server_id, resolves the definition row and returns the distinct {id, server_id} set (falls back to (definition_id,) when no row matches, so history for a deleted/unknown definition is unaffected). list_automation_runs and list_automation_results now filter `definition_id IN (aliases)` through this helper instead of `= ?`, with ORDER BY unchanged.

Scope grew beyond the two named functions: count_automation_results (and count_unread_results, which delegates to it) carries the identical id-space ambiguity for the results side and feeds the pane's unread badge and the definition-scoped "mark all read" fan-out (_fetch_definition_detail_counts, _definition_unread_result_ids in schedules_workbench.py) -- so it is bridged through the same helper too, one shared seam for both runs and results (AC#3). count_automation_runs is untouched: automation_runs is local-only (never synced from the server, per its own docstring), so a run's definition_id is always the local id and there is no ambiguity to bridge.

Bridging count_automation_results forced touching two workbench call sites the task said to leave alone: `_definition_results_query` and `_definition_unread_result_ids` were hand-rolling their OWN dual-id-space bridge (loop over local_id/server_id, merge/sum) -- exactly the "duplicated bridge" AC#3 asks to eliminate. Left as-is once the DB layer also bridges, they break in two different, verified ways: test_definition_results_query_merges_local_and_server_id_spaces failed (total double-counted, 4 instead of 2) when only list_automation_results was bridged and count stayed exact-match; test_definition_unread_result_ids_merges_both_id_spaces would fail the opposite way (silently drops results -- a definition-scoped "Mark all read" that never actually clears some post-transfer unread rows) if count stayed exact-match while list was bridged, because the caller's dynamic `limit=unread_total` becomes inconsistent with the now-broader list query. Bridging both DB functions and collapsing those two workbench methods from a loop-and-merge to one call each (the seam already returns the union) is the only combination that keeps both pre-existing pinned tests true; verified by running the base (pre-fix) file against the tests and back. No other of the "6 workbench call sites" were touched.

AC#5 ruling: NOT bridged, recorded as server-side/data-does-not-exist-locally. Traced create_local_copy_from_mirror (the from_server_pending leg): the new local row is a fresh INSERT with a fresh uuid4 id and no server_id (never set), and the field dict built there has no column carrying the origin mirror's id or server_id -- the only place that link ever existed was payload["server_definition_id"] inside the release's pending_mutations row, which sync_engine._push_definition_release deletes via delete_pending_mutation() the moment the release lands. sync_mapping (the table that WOULD carry such a link) is wired for reminder_task only, never automation_definition. The old mirror row is never deleted -- it survives as an archived row still answering to its own (id, server_id) -- so the prior history is not lost, just permanently detached from the new local row with nothing in the schema to reattach it. Documented in Docs/User_Guide/schedules.md's "Moving a task between this device and the server" section.

Files: tldw_chatbook/Scheduling/db/scheduled_tasks_db.py (_definition_id_aliases + 3 call sites), tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py (_definition_results_query, _definition_unread_result_ids simplified to single calls), Tests/Scheduling/test_scheduled_tasks_db.py (3 new tests), Docs/User_Guide/schedules.md (AC#5 gap note).
<!-- SECTION:NOTES:END -->
