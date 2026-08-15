---
id: TASK-16325
title: Persist structured claims and enable follow-up Q&A over run evidence
status: To Do
assignee:
  - '@robert'
created_date: '2026-08-15 05:15'
labels:
  - research
dependencies:
  - TASK-16322
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Runs currently would store only a final markdown blob. Persist claims with source id and verbatim quote and confidence as a JSON artifact so follow-up questions can be answered from stored evidence without re-spending on search, mirroring tldw_server follow_up_json bounded seed contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Extracted claims with source id and quote and confidence are persisted as a run artifact in JSON,Follow-up questions are answered from stored evidence without new searches when evidence suffices,Insufficient evidence triggers an explicit fallback to a new search or run rather than a fabricated answer,The seed shape is bounded (outline plus key claims plus unresolved questions) matching the server follow-up contract,Tests cover retrieval and the fallback boundary
<!-- AC:END -->
