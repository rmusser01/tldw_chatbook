---
id: TASK-701
title: TLDW_CONFIG_PATH does not isolate the runtime-policy state file
status: Done
assignee:
  - '@claude'
created_date: '2026-07-26 13:59'
updated_date: '2026-07-26 18:30'
labels:
  - config
  - testing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The runtime-policy state file (which records whether the app is running local or server) is resolved from a hardcoded home path, not from the active config path. A scratch or test profile launched via TLDW_CONFIG_PATH therefore shares the real user's runtime-policy state: switching a test profile to server mode silently leaves the user's own app in server mode afterwards. This undermines the isolated-profile recipe used to verify UI changes, and makes local/server behaviour untestable without touching real state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A profile launched with TLDW_CONFIG_PATH reads and writes its own runtime-policy state
- [x] #2 Switching modes in an isolated profile leaves the default profile's mode untouched
- [x] #3 The default location is unchanged when no override is set
<!-- AC:END -->
