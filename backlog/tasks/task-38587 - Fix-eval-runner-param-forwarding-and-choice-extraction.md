---
id: TASK-38587
title: Fix eval runner param forwarding and choice extraction
status: Done
created_date: 2026-11-02 14:27
labels:
- bug
- evals
priority: medium
updated_date: 2026-11-03 02:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
stop_sequences is silently dropped by the _call_llm whitelist so generation evals run past their intended stop, and _extract_choice fabricates choice A whenever nothing matches which biases accuracy. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 stop_sequences reaches chat_api_call
- [x] #2 Failed choice extraction counts as an extraction error instead of defaulting to choice A
- [x] #3 Dead call_kwargs construction removed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
stop whitelisted and forwarded to chat_api_call; failed extraction raises ChoiceExtractionError and counts as an error sample; dead call_kwargs blocks removed. Commit 68466e79fc.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
