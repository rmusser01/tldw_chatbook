---
id: TASK-40848
title: Wait for the endpoint fixture configuration revision before opening settings
status: Done
created_date: 2026-11-14 02:46
assignee:
- '@codex'
updated_date: 2026-11-14 04:08
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The endpoint round-trip CI test can open settings while an older configuration presentation read is pending, causing a legitimate refusal and then querying the Console as though it were a modal. Establish the fixture's new endpoint revision before exercising the real opener.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The shared endpoint fixture waits for its new checked configuration revision using the existing bounded helper; the real opener and all endpoint, draft and persistence assertions remain.
- [x] #2 A delayed real old-revision refresh no longer causes the observed missing-modal failure, while an unavailable current revision still fails within the existing bound.
- [x] #3 Affected tests, independent review and focused static checks pass; the normal self-hosted UI27 lane qualifies without production, isolation, selector or timeout changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: Test fixture readiness correction within the existing projection and CI contracts.
1. Reproduce the missing-modal error by delaying the real revision-zero configuration refresh and observe its captured/current revision and real opener verdict.
2. Reuse the production presentation owner in the shared fixture to wait for the newly set endpoint revision through the existing 15-second helper; assert the real round-trip opens succeed.
3. Run all shared-fixture journeys under the controlled delay, a stuck-current negative control, owner-fence regressions, static checks and independent review; qualify the loaded normal UI27 lane.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The shared _console_on_9200 fixture now uses the existing _sync_console_control_bar_now helper after replacing the endpoint, so the checked presentation belongs to the new settings revision before the real opener runs. The round-trip test asserts both real opens. The final test diff is one import, one existing-helper call, a why comment and two opener assertions. All original assertions, function names, selectors, private-profile boundaries and deadlines remain; no production or workflow change. Updated the existing pending-display lesson. ADR-359 applies; no new ADR is required.

Evidence: ninth normal gate43539534151 UI27 job132278678047 failed the round-trip test with NoMatches for the model-change control on ChatScreen;231 other tests passed (/tmp/tldw-ci-ninth-ui27-failure.log). The unchanged exact test passed locally12.06s. Delaying only the real revision-zero refresh0.5s reproduced the identical NoMatches, with captured revision0/current1 and real preparation returning False (/tmp/tldw-ci-endpoint-cold-red.log; /tmp/tldw-ci-endpoint-cold-observations.log). The final helper passed all5 shared-fixture journeys in39.86s with an event-conditioned real old-read hold; every case logged revision0/current1 before the original refresh (/tmp/tldw-ci-endpoint-held-green.log; /tmp/tldw-ci-endpoint-held-observations.log). Holding real publication30s fails at the unchanged15-second helper bound (/tmp/tldw-ci-endpoint-final-negative.log). Independent review verified pending-owner, expiry, closed/disposed refusal, real actions and unchanged cleanup; no blockers. Ruff passes; changed ranges are formatted. Full-file formatting reports pre-existing untouched sections, which remain unchanged. Whitespace, backlog IDs/readability and AST assertion/function retention pass.

Additional evidence limit: an optional run of Tests/UI/test_console_settings_context_owner.py fails8 pre-existing stub cases because ChatScreen.__new__ lacks _nodes; running that unchanged module alone with no temporary plugin reproduces the same8 failures/2passes in7.62s (/tmp/tldw-ci-endpoint-owner-baseline.log). The affected fixture's5 journeys passed separately. No unrelated stubs were changed. Loaded qualification: tenth normal gate43548543910 UI27 job132306455375 passed all232 tests in548.04s, including the original endpoint round-trip15.65s, on ghq-test-f18fcb06. Job2026-11-14T03:53:49Z–04:04:17Z (10m28s). Log /tmp/tldw-ci-endpoint-live.log. All acceptance criteria are complete; this is lane qualification, not successful full-CI timing (UI26 failed separately).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
