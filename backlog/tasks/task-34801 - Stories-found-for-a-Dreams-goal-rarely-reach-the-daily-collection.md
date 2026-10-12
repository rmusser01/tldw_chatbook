---
id: TASK-34801
title: Stories found for a Dreams goal rarely reach the daily collection
status: In Progress
assignee: []
created_date: '2026-10-12 01:45'
updated_date: '2026-10-12 05:02'
labels:
  - dreams
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Dreams cycle searches for the user's goals as well as topics, and feedback on a goal-found story steers that goal's future queries. The ranking that picks the day's stories scores a candidate only by word overlap with topics, so a candidate found by a goal query scores zero unless it happens to share a word with a topic. Goal-found stories therefore seldom make the collection, and the goal feedback loop this feature adds seldom has anything to act on. Noted in the review of PR 2921; ranking was left unchanged there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A candidate found by a goal query competes for the day's collection on a footing the goal's owner would expect, without displacing every topic story
- [ ] #2 The effect on a mixed profile of topics and goals is measured before and after and recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Measure actual mixed goal/topic daily selection before and after; apply the smallest goal-result opportunity rule while retaining topic results, private-goal exclusion, generating provenance and shared budgets. Qualify real discovery/cycle story outcomes and required regression membership. Existing ADR196; no new ADR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Source author follow-up admitted into the user-authorized complete PR2921 review/remediation. Historical source description/notes retained. Current integration plan targets frozen updated author6c074d6eed/v6; actual post-Buddy dev and fresh author lease remain required before implementation. No product change or completion claimed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->
