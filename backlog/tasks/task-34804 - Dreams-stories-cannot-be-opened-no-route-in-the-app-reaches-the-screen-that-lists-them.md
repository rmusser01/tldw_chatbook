---
id: TASK-34804
title: >-
  Dreams stories cannot be opened: no route in the app reaches the screen that
  lists them
status: In Progress
assignee: []
created_date: '2026-10-12 02:33'
updated_date: '2026-10-12 14:38'
labels:
  - dreams
  - navigation
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dreams writes a daily collection of stories and the user guide says they appear on the Artifacts screen, where a story opens for feedback, Keep, Track, Watch, Ingest, Export, Dive and the goals editor. Since the Artifacts destination was folded into Library (ADR-172, September 2026) the artifacts route resolves to the Library screen, the old Artifacts screen is never mounted anywhere in the product, and the Library artifact catalog knows only chatbooks and reports. So a user who enables Dreams gets cycles that spend the daily search and model budgets and stories that cannot be listed or opened, and every action that starts from a story is unreachable. The test suite did not catch it because the UI test harness maps the artifacts destination to the old screen itself. Found by reading and by executing the route lookup during the review of PR 2921; not yet confirmed in a running app. It is already true on dev for the Dreams phases that have shipped.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With Dreams enabled and a collection present, the user can reach the story list from the app's own navigation and open a story
- [ ] #2 Every action the user guide lists for a story is reachable from there, including the goals editor
- [ ] #3 The Dreams UI tests mount the same screen the app's navigation mounts, so an unreachable surface fails a test
- [ ] #4 The user guide and the Settings hints name the real location and key
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use actual production Library routing/catalog/controller to expose Dreams stories, all existing actions and goals editor; update tests to mount actual destination and guide/Settings real location/key. Preserve shell authority/layout/token/owner contracts and current PC/Goal. Direct ADR172/196 integration, no new ADR.

Source update admits34815 persisted degradation reasons alongside the real Library-owned story/list/goals navigation; compose existing token-backed views, preserve all original actions and avoid a second artifact shell.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Source author follow-up admitted into the user-authorized complete PR2921 review/remediation. Historical source description/notes retained. Current integration plan targets frozen updated author6c074d6eed/v6; actual post-Buddy dev and fresh author lease remain required before implementation. No product change or completion claimed.

Current source update: author0c941ba2ec984d960ba83376aeace0e830452bbc, 113 nonmerge commits/88 feature paths after actual merged Goal9595facbbb. Independent source review preserves the real shared Toast fix and stronger reconcile/suspension work; remaining integrated proof and actual post-Buddy/latest-dev publication gate are pending. Prior source/plan/qualification claims remain historical. No product change or completion is claimed here.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->
