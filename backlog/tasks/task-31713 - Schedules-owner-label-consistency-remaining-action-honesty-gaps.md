---
id: TASK-31713
title: Schedules owner-label consistency + remaining action-honesty gaps
status: Done
assignee: []
created_date: '2026-09-05 12:05'
updated_date: '2026-09-06 06:44'
labels:
  - scheduling
  - ux
  - honesty
dependencies: []
priority: medium
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
- [x] #1 Reminder and definition rows use the same owner-label format (prefix or suffix, not both), and a local row's labeling behavior is consistent between the two primitives
- [x] #2 The 60-second (or any periodic) row re-render preserves the DataTable's current horizontal scroll position instead of resetting it to column 0
- [x] #3 A server-side `Run now` against a server that does not implement the run endpoint reports honest copy (e.g. "This server does not support running automations on demand") instead of the raw client exception text, mirroring the results-pull fix
- [x] #4 The Conflicts and Results pushed views make reasonable use of vertical space instead of leaving most of the pane empty for a handful of rows
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented all four ACs. Owner-label consistency (#1): `automation_name_cell`
(definition_detail.py) used to always bracket-prefix even a local row
(`"[This device] name"`); reminders show nothing for local and a
parenthetical `" (server: <id>)"` suffix only when server-scoped
(`_queue_owner_suffix`). Rewrote `automation_name_cell` to match that
exact convention (bare for local, `" (server: <id>)"` suffix otherwise,
folding in `_definition_owner_label`'s existing pending-sync qualifier)
-- `_definition_owner_label` itself untouched (still reused correctly by
the definition detail pane's own unbracketed "Runs on" row). Updated
every pinned test asserting the old bracket format across
test_schedules_unified_list.py/test_schedules_automations_tab.py; the
D8 escaping test's premise (a bracket PREFIX around a URL server id being
markup-eaten) is now structurally impossible since owner labels no
longer use brackets -- updated its docstring/assertion accordingly while
keeping its still-valid bracket-in-NAME/Model-cell escaping coverage.
DataTable scroll preservation (#2): `_render_table`'s `table.clear()`
unconditionally reset `scroll_x` to 0 every render pass, including the
60s ticker's own tick=True re-render which changes no column layout.
Capture `scroll_x` before `clear()`, restore after repopulating rows.
Pinned with a real-overflow test (long title forces genuine horizontal
scroll, not a synthetic scroll_x assignment) -- revert-checked against a
throwaway bare `table.clear()`. Run-now honesty (#3): mirrored the
already-shipped results-pull fix (SyncEngine._pull_results) exactly --
gate on the SAME `_automation_capabilities_available()` before the
network call, and catch `ServerClientNotFoundError` specifically on the
call itself for the "mid-rollout server" case, both giving the same
honest copy instead of raw exception text.
`MockSchedulingServiceMixin.sync_engine` defaults to `None` across the
whole test suite, so the gate treats `None` as "skip" (fail-open,
matching `_automation_capabilities_available`'s own documented
philosophy) rather than crashing every mocked run-now test. Two new
tests cover both honesty paths (capabilities-absent, route-404),
revert-checked against a throwaway pre-fix copy. Dead space (#4):
ConflictsTab/ResultsTab's DataTable was `height: 1fr` (dominating a
full-height pushed pane with blank background below a handful of rows)
while the detail pane below was capped at a small fixed `max-height`
regardless of free space. Bounded the table to `height: auto; max-height:
15` (independently scrollable beyond that, nothing lost) and flipped the
detail pane to `height: 1fr` so it uses the freed space -- the
content-rich half of each screen. Pinned with geometry tests on both
widgets (bare harness + CSS_PATH=BUNDLED_STYLESHEET, per the lesson that
BUNDLED_CSS-declared widget defaults only resolve through
ConsolidatedCSSApp's `_get_default_css` override, not a bare `App`).

Modified: tldw_chatbook/UI/Screens/scheduling/definition_detail.py,
tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py,
tldw_chatbook/UI/Screens/scheduling/conflicts_tab.py,
tldw_chatbook/UI/Screens/scheduling/results_tab.py (+ generated CSS
bundle files via build_css.py). Tests:
Tests/UI/test_schedules_unified_list.py,
Tests/UI/test_schedules_automations_tab.py,
Tests/UI/test_schedules_workbench.py, Tests/UI/test_schedules_results_tab.py.
<!-- SECTION:NOTES:END -->
