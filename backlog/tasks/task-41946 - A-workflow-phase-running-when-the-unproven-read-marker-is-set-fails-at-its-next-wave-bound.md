---
id: TASK-41946
title: "A workflow phase running when the unproven-read marker is set fails at its next wave boundary"
status: Done
assignee: []
created_date: '2026-11-19 07:30'
updated_date: '2026-11-25 14:40'
labels:
- follow-up
- workflows
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
Owner decision 2026-11-25: keep the current behaviour (fail at the next publication; chat held until restart under the TASK-41320 rule, consistent with TASK-40716). ADR required: yes, ADR-285's known-limit line becomes a dated owner decision.

1. Check whether TASK-41645 already committed a test pinning exactly this; if so cite or extend it.
2. Otherwise add a journey to `Tests/Chat/test_workflow_failed_phase_release.py`: a two-task phase whose task-1 provider call is in flight when another chat's read sets the marker; assert task-1 finished, task-2 never ran, the phase settled `failed`, the run is retained (reconcile refuses it as restart-only) and a new turn is refused. Negative control by temporary mutation.
3. ADR-285 text.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner decision 2026-11-25: keep the current behaviour. No product code changed. ADR-285 records the decision, and a new test pins it.

**No existing pin.** TASK-41645 did not commit one: its notes say the boundary failure was seen "in a probe (not committed)". Its nearest test, `test_detail_view_stops_a_running_phase_under_an_unproven_read`, holds the phase at the wave call and Stops it, so that phase never reaches a publication under the marker.

**ADR-285.** In the TASK-41645 section, the "Known limit, not changed" line is now a dated owner decision.
- **What it records.** Nothing in the phase loop checks the marker before the publication that follows a wave. So the work in flight, and any wave the phase launches before its next publication, finishes. A phase between waves still launches its next one. The phase fails at that publication, and no later wave runs.
- **What it cites.** It names TASK-41320's "Reconciling retained execution runs" rule (an unproven metadata read is released only by a restart) as the reason the chat stays held, consistent with TASK-40716, and names the test. The first round cited TASK-40716 alone, which covers tool cleanups.

**Test.** `test_a_phase_running_when_the_marker_is_set_fails_at_its_next_wave_boundary` (`Tests/Chat/test_workflow_failed_phase_release.py`, journey `phase_running_under_unproven_read`).
- **Setup.** A two-task phase (task-2 depends on task-1) is approved, and task-1's provider call is held in flight. Another chat's read loses a storage handle, which sets the unproven-read marker (`_unproven_other_read`). Then task-1 is released.
- **What it asserts:**
  - the provider calls are exactly `planning, task-1`;
  - the phase settled with outcome `failed` (a wrapper on `settle_execution_phase`), and the stored results are one `completed`;
  - this run is retained: it is still in `coordinator._runs` with status `recovery_required`;
  - reconcile refuses it with `workflow_return_unproven`, and the Console names a restart (`_restart_only`);
  - a new turn in the chat is refused with "A turn is already preparing.".
- **What it does not pin.** The closing `_close_unresolved(..., "workflow_original_metadata_shutdown_unresolved")` comes from the other chat's marker alone (coordinator close checks the metadata cells before the runs). It is required teardown, not evidence that shutdown reports this run.

**Negative control.** The wave loop's mid-phase `_publish` was made to swallow `workflow_return_unproven`, so the phase keeps running under the marker. The test went red: task-2 entered the provider (`G_PROVIDER_ENTRY task-2`) and the provider-calls assertion failed. Restored, and run against the final test.

**Evidence (2026-11-25, base origin/dev 2959b48314; machine load average 19-57 from other sessions).**
- `Tests/Chat/test_workflow_failed_phase_release.py` whole: base 16 passed at `-n 4` and serially. This head (review round) 17 passed at `-n 4` and serially, as did the first round.
- Light-tier selection (`scripts/ci/tier_policy.select_changes`): that file plus the base set (`test_console_approval_scope_journeys`, `test_approval_timeout_policy`, `test_smoke`), 52 passed at this head.
- `ruff check tldw_chatbook --select F` and `Tests/lint/ruff_gate.py` pass. The new lines are `ruff format` clean; the file has older unformatted lines. `./scripts/preflight.sh` under a scratch HOME: all derived-artifact checks passed.

Not changed, out of this task's scope: the held chat's refusal copy ("A turn is already preparing.") does not say that a restart clears it.

Files: `Tests/Chat/test_workflow_failed_phase_release.py`, `backlog/decisions/285-native-phased-agent-workflows.md`.
<!-- SECTION:NOTES:END -->
