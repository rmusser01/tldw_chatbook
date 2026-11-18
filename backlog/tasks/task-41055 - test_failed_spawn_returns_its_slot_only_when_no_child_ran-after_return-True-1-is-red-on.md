---
id: TASK-41055
title: "test_failed_spawn_returns_its_slot_only_when_no_child_ran[after_return-True-1] is red on dev"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- agents
priority: low
references:
- Tests/Agents/test_agent_service_review_containment.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Agents/test_agent_service_review_containment.py::test_failed_spawn_returns_its_slot_only_when_no_child_ran[after_return-True-1]` fails on dev ecad404bd4 (2026-11-15) (serially, at `-n 4` and under the bootstrap profile) with "DID NOT RAISE RuntimeError". The other four params pass.

TASK-36950.3's review recorded the same node red at its base on 2026-11-13. The test pins when a failed spawn may give its review slot back: only when no child ran. If the product is wrong, a failed spawn can free a slot that a child still holds, or keep one that nothing uses. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes say whether the product or the test is wrong, and name the commit
- [ ] #2 All five params pass on dev; a fixed product defect is shown red first
<!-- AC:END -->
