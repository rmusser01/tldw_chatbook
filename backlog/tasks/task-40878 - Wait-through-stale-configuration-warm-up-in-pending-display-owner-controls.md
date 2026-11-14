---
id: TASK-40878
title: Wait through stale configuration warm-up in pending display owner controls
status: In Progress
created_date: 2026-11-14 04:08
assignee:
- '@codex'
updated_date: 2026-11-14 04:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The loaded UI26 owner-control test fails during readiness setup when a real configuration warm-up returns a transient refusal before the existing bounded render wait can continue. Preserve the owner/source mutation assertions while establishing a current display.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The fixture waits through a transient stale warm-up before requiring the real rail render, within its existing 30-second deadline and unchanged profile isolation.
- [x] #2 A delayed real publication reproduces the old warm-up assertion and passes both owner-control cases after the fix; persistent unreadiness still fails the existing bound.
- [ ] #3 Focused tests, static checks and independent review pass; the normal loaded UI26 lane qualifies without production or workflow topology changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: Test fixture readiness correction within existing projection and CI contracts.
1. Record the live assertion and reproduce a legitimate false warm-up using the actual projection with delayed resumption.
2. Extend the fixture's existing readiness loop to include the warm-up result without changing the deadline, assertions, selectors or isolation.
3. Run both affected owner/source controls under the controlled timing, retain a persistent-unready negative control, lint and independent review; qualify normal UI26.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The fixture now treats the real configuration warm-up and rail render as consecutive readiness predicates in its existing 30-second loop. A transient false warm-up retries before sync; reader exceptions and cancellation still propagate. Both complete original test functions, mutation controls, assertions, decorators and private-profile isolation are AST-identical. No production, selector, deadline, workflow or resource change. The shared native helper was not substituted because it uses a different 15-second bound and control-bar/lock contract. Updated the existing testing-evidence lesson; ADR-359 applies, no new ADR.

Evidence: tenth full gate43548543910 UI26 job132306455432 failed the current-owner/saved-generation test at assert await projection.warm();154 other tests passed (/tmp/tldw-ci-tenth-ui26-failure.log). Unmodified exact local test passed14.57s. Temporary probe /tmp/tldw_ci_pending_owner_probe.py requests a real refresh and delays the actual settled wait's resumption beyond the real one-second display lifetime: production warm returned false at age1.106s, reproducing the same assertion (/tmp/tldw-ci-pending-owner-controlled-red.log; /tmp/tldw-ci-pending-owner-observations.log). Both corrected cases passed26.26s under that one-time real expiry (/tmp/tldw-ci-pending-owner-controlled-green.log). Persistent real expiry fails the existing deadline assertion at about30.6s, including the last attempt,42.69s total with child startup/teardown (/tmp/tldw-ci-pending-owner-negative.log).

Independent review found no blockers; confirmed owner/source controls, successful warm before real sync, propagated exceptions/cancellation, and the different existing helper contract. Ruff, changed-range format, whitespace, backlog ID/readability6571files pass. Whole-file format has untouched existing drift. Evidence limits: the probe demonstrates a legitimate refusal schedule, not which owner/expiry condition caused uninstrumented CI. The unchanged deadline checks completed retries and does not interrupt a never-settling warm-up. Loaded normal UI26 qualification remains pending; task stays In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
