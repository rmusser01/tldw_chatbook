---
id: TASK-40938
title: Order the new CSS carrier after its recorded build in staleness test
status: Done
created_date: 2026-11-14 05:56
assignee:
- '@codex'
updated_date: 2026-11-14 17:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The new bundled-CSS carrier fixture assumes its creation receives a later filesystem timestamp than the recorded input. On a coarse clock both writes share one tick, so the test does not establish the newer-than-build scenario it claims.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The gained-carrier test establishes a newer carrier timestamp than the recorded build and retains its stale verdict and exact gained-declaration reason assertions.
- [x] #2 The original failure reproduces with coarse real-file timestamps; the corrected CSS module passes that control and a disabled declaration detector still fails.
- [x] #3 Static checks and independent review pass, and the normal loaded UI24 lane qualifies without production or workflow changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: Test fixture timestamp precondition for existing CSS staleness policy, no production contract change.
1. Retain eleventh-run UI24 failure and reproduce the equal timestamp with actual files.
2. Reuse the existing timestamp helper after writing the new carrier, leaving production and assertions unchanged.
3. Verify the full affected CSS module under coarse writes, retain a declaration-detector negative control, lint/review, and qualify normal UI24.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reused the existing _bump_mtime(carrier) immediately after writing the new unmanifested carrier. This establishes the newer-than-recorded-build condition used by the existing CSS discovery policy. Both original assertions and all test names are AST-identical. Production detection, filesystem isolation, selectors and workflow resources/topology are unchanged; no new helper or dependency. ADR-359 applies, no new ADR.

Eleventh full gate43557511010 UI24 job132335367901 failed assert stale for the gained-carrier test;369 other tests passed648.85s (/tmp/tldw-ci-eleventh-ui24-failure.log). The normal unchanged exact test passed1.01s locally. A temporary plugin performs real writes and assigns a shared whole-second filesystem tick, while leaving later explicit os.utime calls intact; it reproduces the exact original false stale verdict in1.10s. The corrected full CSS module passes all19 cases in2.04s under that control. Logs /tmp/tldw-ci-css-carrier-{original,controlled-red,controlled-green}.log; probe /tmp/tldw_ci_css_carrier_probe.py.

The meaningful negative replaces only the actual bundled-declaration regex with a never-matching regex, still using the coarse real-file fixture. The corrected exact test fails its original assert stale in1.25s (/tmp/tldw-ci-css-carrier-negative.log; /tmp/tldw_ci_css_carrier_negative.py). This preserves declaration-detection coverage. Temporary plugins are outside the repository and are not committed.

Independent css_carrier_review found no blockers. Ruff, changed-range formatter, whitespace and AST assertion/name checks pass. Updated the testing-evidence lesson. The documented production policy still excludes equal/backdated unmanifested carriers; this change qualifies the newer-than-build scenario and does not broaden that policy. Normal loaded UI24 qualification remains pending, so AC3 and status stay In Progress.
Loaded normal PR qualification passed on final head 258ebec33d68a46a4f5d5a20838cce8dd6e2f627: run 43612634465 attempt 1, UI24 job 132502103939, runner ghq-test-3327ac58. All 370 tests passed in 648.15s; job 16:53:46–17:06:05Z (12m19s). The full required gate passed and PR #3529 merged at 2026-11-14T17:22:30Z. Raw run/jobs and logs: /tmp/tldw-ci-success-43612634465. This completes AC3; no production or workflow change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
