---
id: TASK-38583
title: Stop LogProbRunner from fabricating predictions
status: Done
created_date: 2026-11-02 14:26
labels:
- bug
- evals
priority: high
updated_date: 2026-11-03 02:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task_type logprob runs always predict choice 0 because every logprob is minus infinity while real paid API calls are made and discarded, and minus Infinity is persisted into metrics. See bug-sweep-2026-11-02-dev.md section 1.4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unsupported logprob providers produce error samples instead of always choosing choice 0
- [x] #2 No paid request result is discarded
- [x] #3 No minus Infinity value is persisted into eval metrics
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
_get_text_logprob raises typed unsupported without making any LLM call; samples become error samples; persist boundary sanitizes non-finite floats so metrics stay strict-JSON. Commit c0ad0d1cca.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
