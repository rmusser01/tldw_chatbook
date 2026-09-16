---
id: TASK-32699
title: Keep filtered file picker controls visible at compact sizes
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-16 06:07'
labels:
  - ui
  - file-picker
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The provider-recovery native review found that at 80x24 the GGUF filter leaves the filename field about one column wide and pushes Cancel outside the dialog. Keyboard Escape still cancels, but the controls are not visibly usable. Evidence: Docs/superpowers/qa/2026-09-16-ingest-recovery/picker-80.svg. Existing folder-only compact behavior is recorded in TASK-32665.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 80x24 in both themes the filtered file picker shows a usable filename field, filter, Open and Cancel together with a loaded listing.
- [ ] #2 Typing or pasting a file path, changing the filter, selecting and cancelling work through actual mounted controls.
- [ ] #3 Compact-wide resize retains typed path, selection, current filter and focus without recreating controls; neighboring folder and save pickers retain their behavior.
- [ ] #4 Token-backed production CSS, targeted tests and isolated native captures qualify the layout without broad test sweeps or model operations.
<!-- AC:END -->
