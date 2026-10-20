---
id: TASK-35645
title: Console screen refreshes on the UI loop wait out a config rebuild
status: To Do
created_date: 2026-10-20 00:23
dependencies:
- TASK-35444
labels:
- console
- performance
- flaky
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While a config rebuild holds the config locks (after a save or reload invalidates the cache), any Console refresh on the UI loop that reads config on a cold cache waits for the rebuild to finish, and the whole app stops responding for that time. A real rebuild takes milliseconds, but the response-rules phase test holds one for seconds to prove the UI never waits, and it fails in CI whenever a background refresh reads config during that hold (TASK-35444: 7 of 31 runs on 2026-10-19). TASK-34415 made the control refresh defer instead of waiting, and TASK-35444 moved three more refresh reads onto that deferral. Found one at a time, more remain: the transcript's turn-file-cards setting (get_cli_setting in ConsoleTranscript._reconcile_rows) and the full UI sync's core-state push at its start. Each fix has exposed the next read. The general approach is an owner decision: defer screen by screen, as TASK-34415 did, or have config reads during a rebuild return the last loaded config instead of waiting, which also changes what action reads such as a send see during a rebuild.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Which approach to take (screen-by-screen deferral or reads that do not wait) is decided by the owner and recorded
- [ ] #2 No Console refresh on the UI loop waits for a config rebuild, shown by a test that holds the rebuild lock across a full UI sync including a transcript row reconcile
- [ ] #3 Reads that a send or other action relies on still see the configuration the owner's decision requires
- [ ] #4 The response-rules phase test passes 20 consecutive pull-request runs
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
