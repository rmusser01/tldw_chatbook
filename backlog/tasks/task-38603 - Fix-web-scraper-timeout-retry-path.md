---
id: TASK-38603
title: Fix web scraper timeout retry path
status: Done
created_date: 2026-11-02 14:29
labels:
- bug
- scraping
priority: low
updated_date: 2026-11-03 02:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The timeout branch retries with no backoff and never raises MaxRetriesExceededError after the final attempt, and retries set to zero returns an empty string silently. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Timeout branch sleeps between attempts and raises after the final one
- [x] #2 retries below 1 is rejected or coerced
- [x] #3 Test covers timeout exhaustion
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Timeout branch sleeps between attempts and raises MaxRetriesExceededError after the final one; retries below 1 rejected with the config key named. Commit c9bcdac2cb.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
