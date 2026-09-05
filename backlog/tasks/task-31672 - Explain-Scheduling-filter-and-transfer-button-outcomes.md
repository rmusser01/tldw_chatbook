---
id: TASK-31672
title: Explain Scheduling filter and transfer button outcomes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:09'
updated_date: '2026-09-05 18:33'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give Scheduling controls truthful outcome tooltips, including their initial mounted states, without altering action behavior or dynamic lifecycle guidance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Queue filter, reminder transfer, and automation transfer and run controls explain their outcomes.
- [x] #2 Automation lifecycle default tooltip is present before selection and existing pause resume and lock-state updates remain correct.
- [x] #3 The Scheduling destination outcome audit and full affected Scheduling behavior files pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve RED evidence from unchanged Scheduling destination audit:10initial missing tooltips,9persistently missing. 2. Add a focused mounted regression in Tests/UI/test_schedules_workbench.py for initial control tooltips and lifecycle configured/paused/locked/default-restored states. 3. Add only truthful tooltip keyword defaults in schedules_workbench.py, task_detail.py, definition_detail.py: filter scheduled queue state, cancel/retry ownership transfer, request an immediate automation run, initial pause. Preserve handlers and dynamic updates. 4. Run the unchanged Scheduling destination outcome audit and complete Tests/UI/test_schedules_workbench.py, Tests/UI/test_schedules_unified_list.py, Tests/UI/test_schedules_transfer_actions.py (definition detail/owner regressions live in workbench file). 5. Scoped lint/format checks, parent review, record evidence, mark done and commit. ADR required: no. ADR path: N/A. Reason: routine truthful-copy repair preserving behavior and transfer/lifecycle ownership.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added10tooltip defaults only: fourqueuefilters, fourownershiptransfercontrols, automationRunNow and initialPause. Copy explains existingoutcomes; cancellation isqualified and immediateexecution is a request. Existing dynamic Pause/Resume and lock-reason updates areunchanged, pinnedby mounted regression acrossconfigured,paused,lockedandunlockedstates. Newregression failedbeforeimplementation; initial verification exposed an incorrecttestcall to apply_lifecycle withoutdefinitionid, corrected tocanonicalAPI. Finalexactgate: pytest Tests/UI/test_schedules_workbench.py Tests/UI/test_schedules_unified_list.py Tests/UI/test_schedules_transfer_actions.py test_destination_action_buttons_explain_their_outcome[schedules] -o addopts=;179passed143.82s, report /private/tmp/tldw-31672-scheduling-serial.xml. Parallel rerun observed3existingtransferfixtures with transientemptytable; all3passedisolated andfinalserialgate, reported toparent for separate readinessdiagnosis insteadofchangingbehavior. Rufflint and changedrangeformat/diffchecks passed; Impeccableclarify usedpreciseexistingterms, detectorreportednone; parentindependentreview noblockingfindings. ADRrequired:no, routinecopyonlyrepair; nohandlers/layoutchanged.
<!-- SECTION:NOTES:END -->
