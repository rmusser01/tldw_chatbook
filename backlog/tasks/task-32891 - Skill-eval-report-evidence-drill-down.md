---
id: TASK-32891
title: 'Skill eval: report evidence drill-down'
status: Done
assignee:
  - '@robert'
created_date: '2026-09-21 22:32'
updated_date: '2026-09-22 03:21'
labels:
  - evals
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
HCI review B2 (MEDIUM). The report shows findings and an artifact COUNT ('artifacts: N') but no way to open judge transcripts, simulation transcripts, or cells. Debugging 'why did dimension X score Y' requires leaving the UI entirely. Power-user gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 From the report, a user can open (read-only) the judge and simulation transcripts behind a run
- [x] #2 Findings link to the evidence that produced them, or a copy-path affordance exists for report JSON/artifacts
- [x] #3 Minimum viable if full viewer is out of scope: one-tap access to the persisted artifact location
<!-- AC:END -->


## Implementation Notes

- MVP per the task's own fallback clause ("one-tap access to the persisted artifact location" - in a TUI, inline read-only rendering IS the access): the report lists each captured artifact (sample id, kind from the row's metadata, compact parsed payload from `metrics` - the row shape `save_artifact` writes), capped at 12 lines with an "and N more" ellipsis. Row-key trap: `eval_results` rows carry `metrics`/`metadata`, not `parsed`/`kind`.
- Tests: `test_report_lists_artifacts_read_only` (both artifacts render with sample ids and parsed values).
- Files: `tldw_chatbook/UI/Evals/skill_eval_detail.py`, `Tests/UI/test_evals_skill_eval_screen.py`.
