---
id: TASK-34803
title: Notification text reaches the toast and notify fallbacks with Rich markup enabled
status: To Do
created_date: 2026-10-12 01:46
labels:
- notifications
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
