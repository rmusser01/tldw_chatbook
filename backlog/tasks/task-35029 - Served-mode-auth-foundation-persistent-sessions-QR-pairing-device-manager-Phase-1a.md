---
id: TASK-35029
title: 'Served-mode auth foundation: persistent sessions, QR pairing, device manager
  (Phase 1a)'
status: In Progress
created_date: 2026-10-15 17:49
updated_date: 2026-10-15 17:49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements spec 2026-10-15 §3: persistent device sessions, /auth/devices, QR pairing, tldw-serve --pair
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Restarting the served process keeps phone sessions logged in
- [ ] #2 QR pairing works from /auth/devices and --pair
- [ ] #3 Logout and per-device/all-device revocation work
- [ ] #4 Web_Server README incident response updated
- [ ] #5 ADR-233 linked
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR: backlog/decisions/233-served-web-session-persistence-and-device-pairing.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
