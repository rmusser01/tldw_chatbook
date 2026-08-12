---
id: TASK-15674
title: Honor TLDW_CONFIG_PATH when persisting startup defaults
status: To Do
assignee: []
created_date: '2026-08-12 06:35'
labels:
  - bug
  - config
  - privacy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent app startup under an isolated config profile from writing normalized default keys into the user's default config file. This was reproduced during generated-video player UAT: the profile remained isolated for reads, but startup appended defaults to the unrelated real config; the exact pre-run file was restored from a validated snapshot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Starting the real app with `TLDW_CONFIG_PATH` pointing to a scratch profile leaves the default user config byte-for-byte unchanged.
- [ ] #2 Defaults needed by the isolated run are written only to the effective profile path if persistence is required.
- [ ] #3 A regression test uses distinct profile and decoy default configs and proves no cross-profile write.
- [ ] #4 Existing no-override startup persistence behavior remains covered.
- [ ] #5 No config values or credentials are emitted in diagnostics.
<!-- AC:END -->
