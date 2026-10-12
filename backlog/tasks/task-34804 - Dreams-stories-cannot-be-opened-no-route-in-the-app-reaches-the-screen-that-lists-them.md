---
id: TASK-34804
title: 'Dreams stories cannot be opened: no route in the app reaches the screen that
  lists them'
status: To Do
created_date: 2026-10-12 02:33
labels:
- dreams
- navigation
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
