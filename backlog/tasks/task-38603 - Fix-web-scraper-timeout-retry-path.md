---
id: TASK-38603
title: Fix web scraper timeout retry path
status: To Do
created_date: 2026-11-02 14:29
labels:
- bug
- scraping
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The timeout branch retries with no backoff and never raises MaxRetriesExceededError after the final attempt, and retries set to zero returns an empty string silently. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Timeout branch sleeps between attempts and raises after the final one
- [ ] #2 retries below 1 is rejected or coerced
- [ ] #3 Test covers timeout exhaustion
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
