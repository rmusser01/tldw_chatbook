---
id: TASK-40723
title: Keep self-hosted Python imports working in sanitized child processes
status: Done
created_date: 2026-11-13 18:37
assignee:
- '@codex'
updated_date: 2026-11-13 19:18
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first full local CI gate fails because setup-python installs under the runner work directory and sanitized child processes lose access to the installed dependencies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A setup-python interpreter imports installed pytest after LD_LIBRARY_PATH is removed and HOME changes.
- [x] #2 Fresh disposable runners use the canonical tool-cache location while private-profile isolation remains unchanged.
- [x] #3 The failing project-chat UAT is rerun and the corrected full-gate timing is tracked separately from the failed attempt.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: Repair the guest image toolchain within the accepted runner boundary.

1. Reproduce the sanitized-child import failure against the same official Python archive.
2. Configure the guest runner tool cache at /opt/hostedtoolcache and add an actual sanitized-child qualification assertion.
3. Publish a verified trusted base image, recycle old runners after jobs finish/cancel, rerun the failed UAT and normal PR gate. Preserve first-attempt timings.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Corrected the disposable guest tool-cache location to /opt/hostedtoolcache using runner .env AGENT_TOOLSDIRECTORY and a runner-owned cache directory. The exact official Python archive reproduced missing pytest when LD_LIBRARY_PATH was removed at its relocated path; canonical placement fixed the sanitized child. Removed the incomplete manually seeded proof interpreter after the next attempt exposed a missing pip executable; official setup-python now performs complete runtime installation. Application isolation fixtures unchanged.

Added reproducible guest customization and qualification matching production cache:pip, plus child checks with LD_LIBRARY_PATH removed and HOME changed for pinned3.12.11 and current3.12. Independent review, YAML/shell validation and whitespace checks passed. Live qualification43487941053 succeeded. On full gate43487945840, job132103333488 passed Run project-chat UAT (2m34s), complete evidence validation, the missing-evidence negative control, and artifact retention. No current gate failures at19:16Z; remaining unrelated suites continue. Failed/cancelled attempts43482084755 and43485256104 are retained separately and are not successful full-gate timings.

ADR required:no new ADR; direct repair within ADR-359. Full local suite not run. Bookkeeping is recorded locally without pushing a new head that would restart healthy CI; overall merge and full-gate duration remain under the active monitor.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
