---
id: TASK-35563
title: The Schedules queue chips show which filter is selected
status: To Do
created_date: 2026-10-18 17:42
dependencies: []
labels:
- model-config-redesign
- follow-up
- schedules
priority: low
references:
- tldw_chatbook/css/features/_scheduling.tcss
- tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py
- Tests/UI/test_schedules_keyboard_map.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's live capture phase (2026-10-18) and confirmed in triage to predate Phase 9: it reproduces at the phase base 111f8e48fc.

The Queue's filter chips (All, Active, Paused, Completed) paint alike. The selected chip differs only by one step of text colour, (223,223,223) against (224,224,224). `SchedulesWorkbench._set_queue_chip` sets `variant='primary'`, but the rule '#scheduling-queue-chips .scheduling-queue-chip' sets every chip's background and border. App CSS outranks Textual's Button variant styles, so the variant shows nothing. The rule dates to 48d01048ae (2026-09-03).

Live evidence: Phase 9 captures 08a and 08c, and the triage's 08c base/head pair (byte-identical chip rows).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The selected queue chip is visibly distinct from the other three at 211x44 and 235x52, with at least 3:1 contrast between the selected and unselected chip fills
- [ ] #2 A focused chip still shows the focus style
- [ ] #3 A painted test pins that the selected chip's background differs from the others
<!-- AC:END -->
