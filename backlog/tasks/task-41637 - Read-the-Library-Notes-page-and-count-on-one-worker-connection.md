---
id: TASK-41637
title: Read the Library Notes page and count on one worker connection
status: In Progress
created_date: 2026-11-18 04:15
assignee:
- '@codex'
labels:
- library
- performance
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3610
updated_date: 2026-11-18 04:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cold Library opens a second owner for the already-initialized profile Notes database. Its page and total then use separate worker connections. Use one admitted read for the desktop Library snapshot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A native Library snapshot returns the original paginated Notes records and authoritative active count using one worker connection. A count failure retains the original unknown-total fallback.
- [x] #2 The existing local policy, current profile binding, custom scope and memory behavior, mutation owner, and finite connection retirement stay effective.
- [x] #3 Retain every paired sealed macOS PTY run with cold and warm exact count paint, full observation windows, host load and ordinary Quit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Retain the exact 75994498f7 baseline, call timings, and all three disposable alternating Terminal pairs. Add a failing regression through the existing Library snapshot entry point with a real file-backed Notes scope.
2. Let the stock snapshot run its existing page and count queries on the same finite worker. Use the matching initialized profile database when the original user owner has not been built. Keep direct Notes calls, mutations, policy and optional-count degradation on their existing contracts.
3. Verify real connection entry and close, exact pagination, denied access, and custom fallback. Track the final source in three paired 120×50 Terminal runs and keep all results. Run affected guards and record the evidence.
ADR required: no
ADR path: backlog/decisions/076-library-lifecycle-progressive-disclosure.md and backlog/decisions/250-application-services-bundle.md
Reason: This composes existing local read operations inside the Library snapshot and retains the same services, publication, authority, and database lifetimes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The stock file-backed Library snapshot uses the existing page and count queries on one finite worker connection. The matching profile database serves reads before a separate Notes user owner exists. Existing per-user owners, policy checks, mutation paths, custom callbacks and unknown-count fallback retain their contracts.

The original native regression opened two connections. The final four cases passed with one open for the matching profile, exact paging and count, count-failure degradation, denied access without opening SQLite, and the original route for a different profile owner. All 14 ordinary Notes profile-admission failures reproduced on the exact baseline. The 16 Notes cases passed in the existing bootstrap-profile mode, and all ten sandbox-blocked worker checks passed with native child inspection.

All six final PTY runs retained the full windows and retired every scoped SQLite descriptor. The candidate halved Notes SQLite preparations. Median warm paint moved from 127 ms to 78 ms, cold paint moved from 988 ms to 993 ms, and the worst cold loop gap rose from 417 ms to 575 ms. The original and candidate runs, including the disposable prototype, are recorded in [the QA report](../../Docs/QA/console-library-responsiveness/README.md#local-notes-page-and-total).

The follow-up remains local while CI evaluates the published PR commit. Existing [ADR-076](../decisions/076-library-lifecycle-progressive-disclosure.md) and [ADR-250](../decisions/250-application-services-bundle.md) cover the snapshot and service boundaries.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
