---
id: TASK-35836
title: Console mount and resume refreshes on the UI loop wait out a config rebuild
status: To Do
created_date: 2026-10-20 19:16
dependencies:
- TASK-35645
labels:
- console
- performance
- flaky
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While a config rebuild holds the config locks, a Console refresh that runs at mount or on a screen resume and reads config on a cold cache waits for the rebuild on the UI loop. TASK-35645's pin caught one: the command popup's skill-candidate discovery (a console-skill-discovery worker started at mount and on some resumes) built the local skill trust service, which read config, while the pin held a rebuild (2 of 10 whole-file runs under 4-way parallel load). The mount path schedules about 25 refreshes and none has been listed for config reads. TASK-35645 kept the full UI sync, its transcript reconcile and seven post-mount refresh entry points from waiting, and its pin waits for mount-time skill discovery to finish before holding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every refresh the Console mount and screen resume schedule is listed with whether it reads config on the UI loop, using the recording-lock diagnostic from TASK-35645
- [ ] #2 No listed refresh waits for a config rebuild, shown by a test that holds the rebuild lock from the start of the Console mount
- [ ] #3 Action reads stay live, as TASK-35645 decided
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
