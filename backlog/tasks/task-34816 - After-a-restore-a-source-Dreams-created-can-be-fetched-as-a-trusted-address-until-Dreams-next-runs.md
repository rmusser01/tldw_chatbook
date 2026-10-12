---
id: TASK-34816
title: >-
  After a restore, a source Dreams created can be fetched as a trusted address
  until Dreams next runs
status: In Progress
assignee: []
created_date: '2026-10-12 05:39'
updated_date: '2026-10-12 19:39'
labels:
  - dreams
  - security
  - backup-recovery
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dreams marks each watch source it creates so that the address, which came from a web search and not from the user, is fetched under the stricter public-address rules. The mark lives with the source, but the list of which sources Dreams created lives in the Dreams database, and that database is not part of a backup while the sources' store is. After a restore that brings back an unmarked source from an older build, an ordinary start does not re-check the marks (doing so would open the sources' store on the start-up path); with Dreams on, the next daily cycle does, and with Dreams off nothing does until Dreams is switched. Until then such a source, if still active, is fetched as if the user had typed its address, which is the weaker rule. This is pinned by a test and stated in the code. Found in the review of PR #2921 (round 4, item B4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A restored source that Dreams created is never fetched under the user-entered rule, whether Dreams is on or off, from the first fetch after the restore
- [ ] #2 An ordinary start does not open the sources' store on the start-up path to achieve this, or the measured cost is recorded and accepted
- [ ] #3 The Dreams database is either included in backups or the task records why the stores it describes do not depend on it after a restore
- [ ] #4 A test restores an unmarked, active, Dreams-created source with Dreams off and shows its next fetch is refused for a private address
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes; amend existing207 provenance,196 Dreams and126 recovery contracts before code.
Enforce conservative durable lineage at the first restored fetch even with Dreams disabled, including missing/malformed legacy marks. Preserve explicitly user-entered intranet intent, qualify Dreams optional/custom recovery ownership or eliminate the cross-store authority dependency, and test real restore of an active unmarked source with Dreams off/private fetch refusal. Avoid startup source-store opening; any necessary cost requires measured qualification. Periodic backfill or a documented timeout window is not a first-fetch authority fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Current source update: author0c941ba2ec984d960ba83376aeace0e830452bbc, 113 nonmerge commits/88 feature paths after actual merged Goal9595facbbb. Independent source review preserves the real shared Toast fix and stronger reconcile/suspension work; remaining integrated proof and actual post-Buddy/latest-dev publication gate are pending. Prior source/plan/qualification claims remain historical. No product change or completion is claimed here.

PR2921 current private integration qualification (2026-10-12):
Subscriptions v3 conservatively normalizes missing/malformed legacy lineage before the first fetch, independent of Dreams startup/on/off state. Actual restore of unmarked active sources with Dreams off refuses private-address fetch; explicit stored user-entered authority survives. Optional Dreams recovery, exact populated historical catalog upgrades and inactive restored authority are qualified. Unmarked intranet rows need explicit user edit or existing allowlist; absence never proves intent. ADR-207/196/126 apply.
Full source-bound outcomes, original failures, native owners and tradeoffs: .superpowers/sdd/2026-10-11-pr2921-current-dev-integration/task-1-report.md; external pr-review-wave2/pr2921-current-integration. Local implementation is ready for independent review. Actual-base restack, complete independent review and hosted required gates remain pending; preserve In Progress and the existing acceptance checklist until the full DoD is met.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->
