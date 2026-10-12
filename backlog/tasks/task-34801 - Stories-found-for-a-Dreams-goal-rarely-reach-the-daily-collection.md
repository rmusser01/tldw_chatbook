---
id: TASK-34801
title: Stories found for a Dreams goal rarely reach the daily collection
status: To Do
created_date: 2026-10-12 01:45
labels:
- dreams
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
