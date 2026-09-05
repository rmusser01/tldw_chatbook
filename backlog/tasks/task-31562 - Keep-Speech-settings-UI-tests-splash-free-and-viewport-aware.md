---
id: TASK-31562
title: Keep Speech settings UI tests splash-free and viewport-aware
status: Done
assignee: []
created_date: '2026-09-05 01:50'
updated_date: '2026-09-05 01:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore deterministic Speech and TTS settings coverage after the production startup splash and taller panel layout made valid interaction tests time out or click outside the viewport.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The production Settings boundary test mounts Settings deterministically without waiting for the splash animation.
- [x] #2 Speech action tests bring Save, Restore Defaults, and Revert controls into the test viewport while preserving the intended focused-field interaction.
- [x] #3 The three reported Speech settings regressions pass in a targeted pytest run.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Disable the production splash only within the pushed-screen boundary test using the established settings-test patch.
2. Add a small test helper that scrolls a target action into view before Pilot clicks it without stealing field focus.
3. Update the affected Speech action tests and run targeted pytest plus Ruff.

ADR required: no
ADR path: N/A
Reason: This is test-harness maintenance for existing Settings behavior and does not change application architecture or runtime contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the Speech settings test harness to disable the production splash in the pushed-screen boundary test and added viewport-aware action clicking for the taller Speech and TTS panel. Save/Revert coverage explicitly restores input focus without scrolling before each Pilot click. Verified the three reported regressions pass (3/3) and Ruff passes. Modified: Tests/UI/test_settings_speech_tts_panel.py. ADR required: no; test-only maintenance preserves existing runtime contracts.
<!-- SECTION:NOTES:END -->
