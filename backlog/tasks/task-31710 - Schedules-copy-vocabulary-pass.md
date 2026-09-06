---
id: TASK-31710
title: Schedules copy/vocabulary pass
status: Done
assignee: []
created_date: '2026-09-05 12:05'
updated_date: '2026-09-06 05:38'
labels:
  - scheduling
  - ux
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Minor/polish copy findings from the schedules single-surface UAT
(`qa/schedules-uat-single-surface` session, findings Minor 14 and Polish
28/29), still present at dev tip `da2fbdbc2` (post the #2422 remediation).
The workbench mixes multiple names for the same concept, describes itself
inaccurately, and renders one literal double-hyphen where the copy meant an
em dash.

Concrete findings:
- Three vocabularies for one list of things — *scheduled task* / *automation*
  / *task* / *definition* — appear across the `Create ▾` chooser, toasts,
  and pane titles; most visibly the two detail panes are titled `"Task
  Detail"` (`tldw_chatbook/UI/Screens/scheduling/task_detail.py:618`) and
  `"Definition Detail"` (`tldw_chatbook/UI/Screens/scheduling/
  definition_detail.py:607`) for what the rest of the screen calls one
  unified queue.
- The screen's header subtitle reads `"When jobs, watchlists, and
  workflows run."` (`schedules_workbench.py:584` and `:4443`) — none of
  those three nouns names what the screen actually lists (reminders +
  recurring questions), and watchlist projections are explicitly out of
  scope per the redesign spec.
- The server-transfer confirm dialog's own body text renders a literal
  `--` where an em dash was intended: `"...the transfer -- nothing goes
  dark while this is only queued."` (`schedules_workbench.py:2243`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The two detail-pane titles, the `Create ▾` chooser copy, and save toasts converge on one consistent vocabulary for "the thing in the queue" (or the split is intentional and stated once, not scattered across 3+ wordings)
- [x] #2 The header subtitle names what the screen actually lists (reminders and recurring questions), not watchlists or workflows
- [x] #3 The transfer confirm dialog's body text renders a real em dash instead of a literal `--`, and any other schedules-screen user-facing string with the same double-hyphen pattern is corrected
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify anchors: task_detail.py:618 "Task Detail", definition_detail.py:607 "Definition Detail", schedules_workbench.py:584/:4443 subtitle, :2243 transfer-confirm em-dash.
2. Grep the schedules screen for the same double-hyphen pattern beyond the one anchor.
3. Pick one vocabulary per primitive, respecting task-23106's locked "scheduled task" noun (AST-guarded in schedules_workbench.py/task_detail.py/reminder_form.py) and the "Recurring Question" noun already established by automation_definition_form.py's own titles; converge the Create chooser modal, the two detail-pane titles, and the save toasts onto those two names instead of a third/fourth wording.
4. Rewrite the header subtitle to name reminders/scheduled-tasks + recurring questions, not jobs/watchlists/workflows.
5. Replace every literal "--" em-dash typo in schedules-screen user-facing strings with a real em dash.
6. Update the tests that pinned the old copy; add coverage where missing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Converged the schedules screen's vocabulary onto the two names already established elsewhere: "Scheduled task" (task-23106's locked, AST-guarded noun for the reminder primitive) and "Recurring Question" (automation_definition_form.py's own "New/Edit Recurring Question" titles). Changed: new_task_choice_modal.py's "Reminder..." button/copy -> "Scheduled task..."; definition_detail.py's "Definition Detail" pane title + empty-state -> "Recurring Question Detail"/"Select a recurring question..."; schedules_workbench.py's automation-save toasts "Automation {verb}." -> "Recurring question {verb}.". Left task_detail.py's "Task Detail" title unchanged (already reads as short for "Scheduled Task Detail", no AST-guard conflict). Header subtitle (schedules_workbench.py:589/:4458, plus shell_destinations.py's nav registry entry and schedules_screen.py's docstring) now reads "When scheduled tasks fire and recurring questions run." AST-swept 6 literal "--" em-dash typos (schedules_workbench.py x5, results_tab.py x1) to real em dashes -- confirmed via a one-off AST scan (docstrings/comments excluded) that these were the only schedules-screen offenders.

Updated pinned tests: test_schedules_automations_tab.py (toast wording), test_destination_visual_parity_correction.py (pane-title tuple).

Modified: tldw_chatbook/UI/Screens/scheduling/forms/new_task_choice_modal.py, definition_detail.py, schedules_workbench.py, results_tab.py, tldw_chatbook/UI/Navigation/shell_destinations.py, tldw_chatbook/UI/Screens/schedules_screen.py, Tests/UI/test_schedules_automations_tab.py, Tests/UI/test_destination_visual_parity_correction.py.

--- Fix round (review REQUEST-CHANGES) ---

AC#3 sweep scope gap: the original pass scoped the `--`->em-dash sweep to `.py` files under `UI/Screens/scheduling/` only, missing 8 genuinely user-facing strings in `tldw_chatbook/Scheduling/services/scheduling_service.py` (three module-level transfer/cancel reason constants, three field_error/reachability messages, two `ResolveOutcome.reason` strings -- all reach the user via `row.show_error`/`outcome.errors[...]["message"]` on this same screen). Fixed all 8; re-ran the AST sweep script across the whole `Scheduling/` tree, 0 remaining hits.

Docs/User_Guide/schedules.md was stale for AC#1/AC#2 (title, intro sentence, and the Create chooser's "Reminder…" button copy) -- CLAUDE.md requires the matching User_Guide page be updated alongside a screen's UI copy change. Updated the three stale strings and added a dated "Copy synced with code" note.

Modified (this round): tldw_chatbook/Scheduling/services/scheduling_service.py, Docs/User_Guide/schedules.md.
<!-- SECTION:NOTES:END -->
