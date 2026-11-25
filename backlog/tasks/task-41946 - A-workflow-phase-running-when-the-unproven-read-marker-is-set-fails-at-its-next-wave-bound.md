---
id: TASK-41946
title: "A workflow phase running when the unproven-read marker is set fails at its next wave boundary"
status: Done
assignee: []
created_date: '2026-11-19 07:30'
updated_date: '2026-11-25 14:00'
labels:
- follow-up
- workflows
- needs-decision
priority: low
references:
- tldw_chatbook/Chat/console_workflow_runs.py
- backlog/decisions/285-native-phased-agent-workflows.md
dependencies:
- TASK-41645
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41645 kept Stop/Pause/Resume/Accept/Remove working under the marker, but a phase already running when the marker is set still fails at its next wave boundary: the task in flight completes, the mid-phase publish is refused, the phase settles `failed`, and the chat stays held until restart. ADR-285 records this as a known limit. Whether the phase should keep running is a product choice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's choice (keep running, or fail at the boundary as now) is recorded in ADR-285
- [x] #2 A test pins the chosen behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Owner decision 2026-11-25: keep the current behaviour (fail at the next wave boundary; chat held until restart, consistent with TASK-40716). ADR required: yes, ADR-285's known-limit line becomes a dated owner decision.

1. Check whether TASK-41645 already committed a test pinning exactly this; if so cite or extend it.
2. Otherwise add a journey to `Tests/Chat/test_workflow_failed_phase_release.py`: a two-task phase whose task-1 provider call is in flight when another chat's read sets the marker; assert task-1 finished, task-2 never ran, the phase settled `failed`, the run stays held and a new turn is refused, and shutdown reports the held run. Negative control by temporary mutation.
3. ADR-285 text.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner decision 2026-11-25: keep the current behaviour. No product code changed; ADR-285 records the decision and a new test pins it.

**No existing pin.** TASK-41645 did not commit one: its notes say the boundary failure was seen "in a probe (not committed)". Its nearest test, `test_detail_view_stops_a_running_phase_under_an_unproven_read`, holds the phase at the wave call and Stops it, so that phase never reaches a wave boundary under the marker.

**ADR-285.** In the TASK-41645 section, the "Known limit, not changed" line is now a dated owner decision citing this task and the TASK-40716 rule (ADR-134), naming the pinning test.

**Test.** `test_a_phase_running_when_the_marker_is_set_fails_at_its_next_wave_boundary` (`Tests/Chat/test_workflow_failed_phase_release.py`, journey `phase_running_under_unproven_read`). A two-task phase (task-2 depends on task-1) is approved; task-1's provider call is held in flight; another chat's read loses a storage handle, which sets the unproven-read marker (`_unproven_other_read`); task-1 is released. Asserts: provider calls are exactly `planning, task-1`; the phase settled with outcome `failed` (a wrapper on `settle_execution_phase`); the stored results are one `completed` (task-1); the run stays in `coordinator._runs`; a new turn in the chat is refused; shutdown reports `workflow_original_metadata_shutdown_unresolved`.

**Negative control.** Mutation: the wave loop's mid-phase `_publish` swallows `workflow_return_unproven`, so the phase keeps running under the marker. Result: red, task-2 entered the provider (`G_PROVIDER_ENTRY task-2`) and the provider-calls assertion failed. Restored; run against the final test.

**Evidence (2026-11-25, base origin/dev 2959b48314, machine load average 35-57).**
- `Tests/Chat/test_workflow_failed_phase_release.py` whole: base 16 passed at `-n 4` and serially; head 17 passed at `-n 4` and serially.
- Light-tier selection (`scripts/ci/tier_policy.select_changes`): that file plus the base set (`test_console_approval_scope_journeys`, `test_approval_timeout_policy`, `test_smoke`), which this branch does not touch; the base set passed whole on the TASK-41950 branch (same code).
- `ruff check tldw_chatbook --select F` and `Tests/lint/ruff_gate.py` pass; the new lines are `ruff format` clean (the file has older unformatted lines). `./scripts/preflight.sh` run at this head.

Files: `Tests/Chat/test_workflow_failed_phase_release.py`, `backlog/decisions/285-native-phased-agent-workflows.md`.
<!-- SECTION:NOTES:END -->
