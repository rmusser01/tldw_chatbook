---
id: TASK-32059
title: >-
  Library Get started is skipped for any profile whose config file already
  existed at launch
status: Done
assignee: []
created_date: '2026-09-08 18:24'
updated_date: '2026-09-08 20:03'
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
- [x] #1 A profile that completes first-run setup and relaunches before visiting Library still sees Get started
- [x] #2 The lifecycle is persisted at profile creation rather than defaulting from the absence of a key
- [x] #3 A test covers the quit-after-setup-then-relaunch sequence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test: create a profile in run 1 (config created, Library never opened), relaunch, assert the lifecycle is still UNKNOWN (Get started).
2. Stamp [library.rail_state] lifecycle = "unknown" from TldwCli.__init__ when first_profile_created_this_session() is true.
3. Keep the shared test factory's returning-profile fiction complete (the sandbox creates a profile per test).
4. Docs stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
coerce_library_lifecycle reads an absent [library.rail_state] lifecycle as EXPANDED for any profile the current run did not create, so a user who completed first-run setup, quit, and relaunched before ever opening Library never saw Get started. TldwCli.__init__ now stamps 'unknown' when first_profile_created_this_session() is true (guarded, and it never overwrites a stored value). The literal string avoids importing the Library package at boot for one value. The shared test factory pretends a sandbox-created profile is a returning one, so that fiction now also covers the stamp: save_setting_to_cli_config is patched off disk and the key is popped from the snapshot, keeping the Expanded default under test. Files: tldw_chatbook/app.py, Tests/UI/app_factory.py, Tests/UI/test_library_crit8_polish_shell.py, Docs/User_Guide/library.md.
<!-- SECTION:NOTES:END -->
