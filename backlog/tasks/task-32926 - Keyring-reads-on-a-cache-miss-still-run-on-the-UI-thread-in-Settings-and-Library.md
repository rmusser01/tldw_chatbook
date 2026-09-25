---
id: TASK-32926
title: Keyring reads on a cache miss still run on the UI thread in Settings and Library
status: To Do
created_date: 2026-09-23 19:58
labels:
- performance
- linux
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32921/32922/32924 cache keyring reads so repeated UI-path lookups share one round trip, but a cache miss still reads the OS keyring synchronously on the Textual event loop. On Linux a locked gnome-keyring can block that read on an unlock prompt. Remaining UI-thread sites (Qodo review on #2820): Settings Privacy & Security builds skill-trust posture inside compose(); Settings sync scope and Library screen helpers resolve the active server context (auth token -> principal id) synchronously; the Image/Video Gen panels resolve backend secrets in compose().
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening Settings Privacy & Security never reads the OS keyring on the UI thread; trust posture renders as pending and fills in from a worker
- [ ] #2 Settings sync scope and Library server-scope resolution never read the OS keyring on the UI thread
- [ ] #3 Image and Video Gen panels compose from configuration loaded off the UI thread, including after save, revert and Test
- [ ] #4 A test with a keyring backend that blocks proves each surface stays responsive
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
