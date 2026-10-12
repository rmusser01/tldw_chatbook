---
id: TASK-34815
title: Dreams does not show why a daily cycle stopped short
status: In Progress
assignee: []
created_date: '2026-10-12 05:39'
updated_date: '2026-10-12 19:39'
labels:
  - dreams
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a Dreams cycle cannot do everything it set out to do (the search budget ran out, a provider failed, Dreams was switched off part-way), it records the reasons on that day's collection. Nothing in the app reads that record, so a user who gets fewer stories than usual, or none, has no way to learn why. The user guide now says this plainly. Found in the review of PR #2921 (round 4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A day's collection that recorded a reason for stopping short shows that reason where the user reads the collection
- [ ] #2 A collection that completed normally shows nothing extra
- [ ] #3 The text shown is the recorded reason in plain words, never rendered as markup
- [ ] #4 The user guide describes where the reason appears
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no; presentation of an existing recorded outcome under existing Dreams/Library decisions.
Compose an existing-token plain-text view of persisted degradation_notes in the real Library collection-reading path, alongside34804 navigation. Preserve ordinary completion wording and literal runtime names. Test stopped-short collection detail and ordinary completion, update the actual guide location. No new cycle/retry policy or claim that a deleted empty collection still exists.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Current source update: author0c941ba2ec984d960ba83376aeace0e830452bbc, 113 nonmerge commits/88 feature paths after actual merged Goal9595facbbb. Independent source review preserves the real shared Toast fix and stronger reconcile/suspension work; remaining integrated proof and actual post-Buddy/latest-dev publication gate are pending. Prior source/plan/qualification claims remain historical. No product change or completion is claimed here.

PR2921 current private integration qualification (2026-10-12):
The actual Library collection reader shows the persisted degradation_notes literally, including empty/failed collections, and adds no normal-completion noise. Recorded reason rendering and real reader navigation pass in final Library qualification. No new cycle policy or shell was added; guide describes the location.
Full source-bound outcomes, original failures, native owners and tradeoffs: .superpowers/sdd/2026-10-11-pr2921-current-dev-integration/task-1-report.md; external pr-review-wave2/pr2921-current-integration. Local implementation is ready for independent review. Actual-base restack, complete independent review and hosted required gates remain pending; preserve In Progress and the existing acceptance checklist until the full DoD is met.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->
