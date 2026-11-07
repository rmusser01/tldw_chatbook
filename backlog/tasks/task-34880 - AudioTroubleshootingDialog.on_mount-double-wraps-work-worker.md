---
id: TASK-34880
title: AudioTroubleshootingDialog.on_mount double-wraps @work worker
status: Done
assignee: []
created_date: '2026-09-19 21:21'
updated_date: '2026-09-20 06:04'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-existing bug: AudioTroubleshootingDialog.on_mount calls self.run_worker on a method already decorated with @work, which always raises WorkerError — the dialog's audio device scan never runs in production either. Documented in .superpowers/wave2-report.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dialog mounts and scans audio devices without WorkerError
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in PR #2751 (commit cb5841fd99): on_mount calls the @work method directly; _get_devices_safe moved to a thread worker. Merged to dev as 9929b48be52.
<!-- SECTION:NOTES:END -->
