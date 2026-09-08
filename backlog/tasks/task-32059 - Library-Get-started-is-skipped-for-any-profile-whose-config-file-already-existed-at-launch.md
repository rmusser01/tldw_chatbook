---
id: TASK-32059
title: >-
  Library Get started is skipped for any profile whose config file already
  existed at launch
status: To Do
assignee: []
created_date: '2026-09-08 18:24'
labels:
  - library
  - onboarding
  - ux
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
coerce_library_lifecycle returns EXPANDED when the stored lifecycle is absent and the profile was not created in the current run, so a user who completes setup, quits and relaunches before visiting Library never sees the documented compact Get started rail (both scratch profiles skipped it for the same reason). Inferred from the code path; not yet observed on a real first run. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A profile that completes first-run setup and relaunches before visiting Library still sees Get started
- [ ] #2 The lifecycle is persisted at profile creation rather than defaulting from the absence of a key
- [ ] #3 A test covers the quit-after-setup-then-relaunch sequence
<!-- AC:END -->
