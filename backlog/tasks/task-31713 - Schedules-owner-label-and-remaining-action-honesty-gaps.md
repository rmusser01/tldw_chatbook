---
id: TASK-31713
title: >-
  Schedules owner-label consistency + remaining action-honesty gaps
status: To Do
assignee: []
created_date: '2026-09-05 12:05'
labels: [scheduling, ux, honesty]
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Minor/polish findings from the schedules single-surface UAT (findings
Minor 12, 17, 24 (narrowed), Polish 30), still present at dev tip
`da2fbdbc2`. The #2422 remediation PR fixed the sync-pull path's honest
copy (a server missing `/results` now says so instead of surfacing
`scheduled_task_not_found`, `sync_engine.py:1403-1449`) and the invisible-
editor/escape-focus/blank-Select-on-activation defects — this task is the
residue that survives that fix.

Concrete findings:
- Three owner-label formats coexist in one list, none of them the
  spec's dim `⇅ server` suffix: definition rows get a bracket PREFIX
  (`"[This device] <name>"` / `"[<server id>] <name>"`,
  `automation_name_cell` in `definition_detail.py:340-359`), reminder rows
  get a parenthetical SUFFIX only when server-scoped
  (`" (server: <id>)"`, `_queue_owner_suffix` in `task_detail.py:431-451`)
  and **no label at all** for a local reminder.
- The 60-second next-run ticker's re-render calls `table.clear()` then
  repopulates every row (`schedules_workbench.py:1436-1449`) — Textual's
  `DataTable.clear()` resets the table's horizontal scroll to column 0,
  so a user reading a truncated subtitle is yanked back to column 0 once
  a minute.
- A definition's server-side `Run now` still surfaces the server client's
  raw exception text on failure — `_run_automation_now_server`
  (`schedules_workbench.py:3888-3920`) catches `ServerClientError`
  generically and notifies `f"Failed to run '{name}': {exc}"` with no
  capabilities gate beforehand, unlike the (now-fixed) results-pull path
  which checks `_automation_capabilities_available()` first and turns a
  missing-endpoint 404 into "This server does not provide the results
  inbox (server too old)." A server too old for
  `/definitions/{id}/run` still surfaces its raw 404 text through
  Run-now.
- The Conflicts and Results pushed views still carry large amounts of dead
  vertical space relative to their content (a handful of rows in a
  full-height pane).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Reminder and definition rows use the same owner-label format (prefix or suffix, not both), and a local row's labeling behavior is consistent between the two primitives
- [ ] #2 The 60-second (or any periodic) row re-render preserves the DataTable's current horizontal scroll position instead of resetting it to column 0
- [ ] #3 A server-side `Run now` against a server that does not implement the run endpoint reports honest copy (e.g. "This server does not support running automations on demand") instead of the raw client exception text, mirroring the results-pull fix
- [ ] #4 The Conflicts and Results pushed views make reasonable use of vertical space instead of leaving most of the pane empty for a handful of rows
<!-- AC:END -->
