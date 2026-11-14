---
id: TASK-40908
title: Start measured projection intervals after background sync settlement
status: Done
created_date: 2026-11-14 05:53
assignee:
- '@codex'
updated_date: 2026-11-14 17:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The run-start coalescing test advances its clock before settling background synchronization. Another real FULL can then serve the expired interval before the task whose projection count the test asserts, causing the loaded UI9 failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The measured explicit sync starts its advanced-clock interval after existing background sync settlement, preserving the exact projection counts, successful render checks, overlapping request proof, and existing deadlines.
- [x] #2 A real background FULL at the old boundary reproduces the CI two-versus-three failure; the corrected test passes the same control and still fails when coalescing or freshness is broken.
- [x] #3 Focused affected tests, static checks, and independent review pass; the normal loaded UI9 lane qualifies without production, isolation, selector, resource, or workflow topology changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: Test-only scheduling/observation boundary correction within existing cadence and CI contracts.
1. Retain the exact live UI9 failure and reproduce a real competing FULL serving the advanced-clock request before the measured owner.
2. Advance the test clock only after its existing settled-sync wait, immediately before the measured original sync; retain every behavioral assertion and real late overlapping request.
3. Run the controlled case, meaningful negative controls and affected cadence/projection tests, lint and independent review; qualify normal UI9 before Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The existing settled_sync helper now advances the simulated clock after its existing 30-second background-settlement wait, immediately before the measured original FULL. Both expired-interval calls use that boundary. All 13 original assertions, test names, render-success checks, real overlapping late request, deadlines and profile isolation remain unchanged (AST checked). No production, selector, workflow, topology or resource change. ADR-359 applies; no new ADR.

Eleventh full gate 43557511010 UI9 job132335367866 failed the final projection count (2 versus 3); 150 other tests, including all three repaired Notes journeys, passed in687.10s. Retained /tmp/tldw-ci-eleventh-ui9-failure.log. A temporary real-background-FULL probe at the sixth helper entry reproduces the same count failure in6.51s, then the corrected exact test passes7.05s. Its observation records show the old background FULL served the advanced clock interval first; after the correction it sees the prior interval and the measured owner advances afterward. Logs /tmp/tldw-ci-projection-sync-boundary-{red,green}.log and /tmp/tldw-ci-projection-boundary-observations.log; probe /tmp/tldw_ci_projection_boundary_probe.py. The probe runs actual production sync; it neither fabricates cadence results nor suppresses background work.

Both affected modules passed35 cases in169.92s with the probe limited to this exact test (/tmp/tldw-ci-projection-sync-modules.log). Meaningful temporary negatives still fail: bypassing coalescing fails3 versus0 in6.66s; perpetual coalescing fails1 versus2 in6.74s at the first expired interval (/tmp/tldw-ci-projection-sync-negative-{coalescing,freshness}.log). Commands use the repository Python3.12 venv, PYTHONPATH=/tmp and the named temporary PYTEST_PLUGINS; no probe is committed.

Independent projection_interval_review found no blockers: after the settled predicate there is no yield before FULL takes its in-progress guard. Ruff, changed-range formatter and whitespace pass. Test names and all original assertions match HEAD. Updated the incident-based testing lesson. Loaded normal UI9 qualification remains pending, so AC3 and status stay In Progress.

Evidence limit: the first unmodified local run failed an earlier successful-render assertion (False) in7.00s, not the historical CI count (/tmp/tldw-ci-projection-sync-original.log). An observation-only trace passed36.20s but changes scheduling. Neither result is claimed as the exact CI reproduction or as proof this boundary correction resolves that distinct observation. The controlled exact-count reproduction and successful affected modules support the correction; next loaded CI still governs qualification.
Loaded normal PR qualification passed on final head 258ebec33d68a46a4f5d5a20838cce8dd6e2f627: run 43612634465 attempt 1, UI9 job 132502103856, runner ghq-test-a5585397. All 151 shard tests passed in 680.66s; job 16:38:39–16:51:14Z (12m35s). The full required gate passed and PR #3529 merged at 2026-11-14T17:22:30Z as d46f753d8287cffd48706f912bfd838dc8756edb. Raw run/jobs and complete logs: /tmp/tldw-ci-success-43612634465. Final merge tree equals the qualified head tree. This completes AC3; no additional code change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
