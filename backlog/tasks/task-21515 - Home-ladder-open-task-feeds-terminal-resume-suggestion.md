---
id: TASK-21515
title: Home ladder open-task feeds + terminal resume suggestion
status: Done
assignee: []
created_date: '2026-08-31 02:22'
updated_date: '2026-08-31 02:32'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Feed pending/failed eval runs and read-it-later count into the next-best-action ladder; terminal start-console becomes Resume last conversation when one exists (spec 2026-08-29 §4)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ladder suggests reviewing eval runs when pending/failed runs exist,Ladder suggests read-it-later when queue non-empty,Terminal suggestion deep-links the newest conversation via nav context,running eval runs never counted,Default adapter and missing services degrade to no suggestion
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ladder gains review_eval_runs + review_read_later branches (pending/failed only, never running); terminal suggestion becomes Resume last conversation when a recent conversation exists and deep-links it via the ADR-079 seam; providers degrade quietly through the thread-worker snapshot refresh. Controller-implemented per ledger ruling; subagent review gate applied.
<!-- SECTION:NOTES:END -->
