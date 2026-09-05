---
id: TASK-31740
title: Bind dictionary send integration fixtures to durable conversation authority
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:48'
updated_date: '2026-09-05 19:55'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore truthful dictionary send integration evidence where the current fixtures assign only an in-memory conversation ID and durable turn admission refuses both provider and agent sends.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both provider and agent dictionary sends reach their intended dispatch branch and preserve exact dictionary expansion assertions using current durable conversation authority.
- [x] #2 No production persistence guard or expected output is relaxed, and the complete integration file passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; test-only current-contract repair. 1. Diagnose the pre-extraction two-test durable-commit failure and compare the working world-info fixture binding. 2. Bind the existing conversation via its canonical store/persistence contract and retain exact dictionary assertions. 3. Run the complete integration file and extraction-adjacent tests; lint and obtain root review before a separate scoped commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The existing conversation path in chat_persistence_service.py commit_durable_turn requires a matching durable console_conversation_library_policy row and hydrated authority; the old fixture assigned only persisted_conversation_id. Inserted the session policy candidate and hydrated it before send, selected agent_runtime through the canonical config seam, and installed the capture gateway factory before mounting. Exact expanded payload and raw transcript assertions are unchanged. Both original failures reproduced before extraction; full file2passed7.88s, then included in184passed109.41s combined verification. Root reviewed this fixture repair with no actionable findings. No new ADR: test-only contract alignment.
<!-- SECTION:NOTES:END -->
