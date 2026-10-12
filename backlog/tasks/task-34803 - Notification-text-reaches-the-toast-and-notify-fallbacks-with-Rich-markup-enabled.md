---
id: TASK-34803
title: >-
  Notification text reaches the toast and notify fallbacks with Rich markup
  enabled
status: In Progress
assignee: []
created_date: '2026-10-12 01:46'
updated_date: '2026-10-12 05:02'
labels:
  - notifications
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The shared notification helper passes a notification's title and message, which are often runtime text such as a page or story title, to the toast and to the notify fallback without turning markup off or escaping. The repo's rule for runtime text is to pass markup off for notify and to escape text shown in toasts, because text containing markup characters can raise at render time. The markup census does not see this site because the text arrives in a variable. Dreams tracked-update notices are one more producer of such text; watchlist alerts already were. Found in the review of PR 2921; the helper is shared, so it was left alone there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A notification whose title or message contains square-bracket markup renders literally and never raises, through the toast path and both notify fallbacks
- [ ] #2 Notifications that rely on markup today, if any, are identified and keep their appearance
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Read every caller of shared notification helper/dispatcher, audit intentionally formatted producers; make runtime titles/messages literal at actual toast and both notify fallbacks with square-bracket controls. Preserve deliberate producer appearance and existing notification affinity/worker ownership. No ADR required for presentation bug fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Source author follow-up admitted into the user-authorized complete PR2921 review/remediation. Historical source description/notes retained. Current integration plan targets frozen updated author6c074d6eed/v6; actual post-Buddy dev and fresh author lease remain required before implementation. No product change or completion claimed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->
