---
id: TASK-38602
title: Stop legacy chat error paths from returning errors as replies
status: To Do
created_date: 2026-11-02 14:29
labels:
- bug
- chat
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Legacy chat returns raw exception text as the assistant reply so callers cannot distinguish failure and provider error bodies flow into chat content; DocumentGenerator also bypasses API_CALL_HANDLERS with a stale provider dict that rejects newer providers. See bug-sweep-2026-11-02-dev.md module summary for Chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Legacy chat re-raises Chat errors instead of returning exception text as the reply
- [ ] #2 DocumentGenerator dispatches through chat_api_call so all registered providers work
- [ ] #3 Tests cover both paths
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
