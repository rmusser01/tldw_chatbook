---
id: TASK-40968
title: Qualify near-expiry display rendering without coordinator contention
status: Done
created_date: 2026-11-14 15:47
assignee:
- '@codex'
updated_date: 2026-11-14 17:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The near-expiry display test treats every declined render as an expiry defect, although the real checked renderer must also decline a busy storage coordinator. Preserve the freshness regression while explicitly establishing its uncontended render precondition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The mounted near-expiry case still proves warm-up refreshes the aging display and the immediate render accepts that fresh display without weakening production checks, clocks, or deadlines.
- [x] #2 A controlled real coordinator overlap reproduces the prior assertion; the corrected test passes the same overlap and still detects disabled refresh-ahead or stale-proof acceptance.
- [x] #3 Focused regressions, lint, and independent review pass; the normal loaded UI9 gate qualifies before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: Test-only precondition repair within the existing checked-display and isolated-runner contracts.
1. Preserve the exact live failure and observe the real renderer under deterministic coordinator overlap; distinguish reproduced interleaving from unobserved historical CI cause.
2. Establish the coordinator-free synchronous render precondition with a 30-second timed acquisition matching the existing setup wait budget and preserve all original freshness/render/backoff assertions.
3. Run affected mounted tests and meaningful refresh/expiry negative controls, independent review and static checks; qualify the normal PR gate after rebase.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The mounted expiry-only test now asserts that warm-up refreshed the timestamp, acquires the real storage coordinator with a 30-second timeout, advances its existing clock and runs the original synchronous render under that lock, then releases it in finally. No production code, expiry threshold, original render/backoff assertion, test selector, workflow topology, runner resource or isolation change. The scoped fixture remains an idle gateway-only harness; it is not a general active-render locking helper. ADR-359 applies; no new ADR required.

Evidence: normal PR gate43565526841 at29827102be failed only UI9 test_a_warm_up_rereads_a_display_near_expiry_so_its_render_accepts_it (150 other shard cases passed); downstream required aggregate failed. Original local exact test passed6.97s. A real thread holding storage_admission._lock for250ms immediately after original warm() returns reproduced the assertion on the rebased branch in7.23s; tracing the original check recorded refusal line913 with proof age0.05s. This proves a valid interleaving, not which predicate failed in the uninstrumented historical CI log. Source helper and production checks were traced before implementation.

The same controlled overlap passes the final timed-acquisition version in6.37s. Disabling refresh-ahead (REFRESH_AHEAD_SHARE=1.0) still fails the new freshness assertion, at1000.0 versus clock1000.99, in6.00s. Holding the real coordinator31s fails the explicit acquisition assertion after its30s bound (37.12s total with startup/teardown). No retry can refresh away the defect. The initial yielding setup used the existing until() helper; the31s control exposed its predicate-before-deadline late-success behavior, so it was replaced by timed acquisition before commit.

Affected projection and checked-display modules passed69 cases in178.26s; the exact case was rerun with the final timed-acquisition line and both controls above. Independent review found no blockers, checked physical warm-read retirement, bounded same-thread acquisition, real reentrant validation, idle-harness lock order, absence of awaits under the lock, and finally cleanup. Ruff check/format, whitespace, backlog ID/readability checks pass. Task remains In Progress until the normal loaded UI9 gate qualifies.

Receipts: /tmp/tldw-ci-warm-{contention-rebased-red,bounded-green,bounded-refresh-negative,busy-bounded-negative,focused}.log; temporary probes /tmp/tldw_ci_warm_{contention,negative,busy}.py. No probe is committed. Rebased cleanly onto dev1c14f0e942 before the correction; the Qodo exception already authorized for PR3529 applies to this independently reviewed repair.
Loaded normal PR qualification passed on final head 258ebec33d68a46a4f5d5a20838cce8dd6e2f627: run 43612634465 attempt 1, UI9 job 132502103856, runner ghq-test-a5585397. All 151 tests passed in 680.66s; job 16:38:39–16:51:14Z (12m35s). The required aggregate passed at 17:22:07Z, 1h28m09s after workflow creation; final cleanup completed at 17:23:40Z. PR #3529 merged at 17:22:30Z as d46f753d8287cffd48706f912bfd838dc8756edb; its tree equals the qualified head. Complete successful-run evidence: /tmp/tldw-ci-success-43612634465. This completes AC3 and the task; no further source modification.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
