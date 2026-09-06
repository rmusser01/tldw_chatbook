---
id: TASK-31710
title: >-
  Schedules copy/vocabulary pass
status: To Do
assignee: []
created_date: '2026-09-05 12:05'
labels: [scheduling, ux, copy]
priority: low
dependencies: []
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
- [ ] #1 The two detail-pane titles, the `Create ▾` chooser copy, and save toasts converge on one consistent vocabulary for "the thing in the queue" (or the split is intentional and stated once, not scattered across 3+ wordings)
- [ ] #2 The header subtitle names what the screen actually lists (reminders and recurring questions), not watchlists or workflows
- [ ] #3 The transfer confirm dialog's body text renders a real em dash instead of a literal `--`, and any other schedules-screen user-facing string with the same double-hyphen pattern is corrected
<!-- AC:END -->
