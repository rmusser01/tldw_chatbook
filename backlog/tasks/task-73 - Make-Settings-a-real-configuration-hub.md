---
id: TASK-73
title: Make Settings a real configuration hub
status: To Do
labels:
- settings
- configuration
- ux
- product-maturity
- console
dependencies: []
priority: high
documentation:
- Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Settings the trustworthy global configuration hub for Chatbook, with real load, validate, save, revert, recovery, and cross-screen verification paths instead of static summaries or placeholder affordances.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings ownership is explicit for global defaults, persisted config, runtime status, and destination-owned workflows.
- [ ] #2 Providers and Models can configure every Console-supported `chat_api_call()` provider without contradictory Settings and Console readiness states.
- [ ] #3 Console behavior defaults can be configured, saved, reverted, and verified from Settings.
- [ ] #4 Storage, Privacy, Diagnostics, and Advanced Config provide safe validation/recovery paths without leaking secrets.
- [ ] #5 Server, sync, workspace, handoff, and domain settings expose honest defaults/status without moving runtime ownership out of the owning destinations.
- [ ] #6 Actual CDP/Textual-web screenshot QA is captured and approved for every changed Settings category.
- [ ] #7 QA walkthrough verifies Settings functionality and cross-screen behavior works, not just that the screen renders or is clickable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Unit, mounted, and integration tests relevant to changed behavior pass
- [ ] #3 Static analysis and diff hygiene checks pass
- [ ] #4 Actual app QA walkthrough completed with screenshots
- [ ] #5 User approval recorded for visible Settings changes
- [ ] #6 Documentation and task notes updated
- [ ] #7 Task status moved to Done after implementation notes are added
<!-- DOD:END -->
