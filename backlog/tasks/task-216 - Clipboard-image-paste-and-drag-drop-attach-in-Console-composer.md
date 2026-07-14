---
id: TASK-216
title: Clipboard image paste and drag-drop attach in Console composer
status: In Progress
assignee: ['@claude']
created_date: '2026-07-13 09:30'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console attachments Phase 1 (PR #621) supports the file picker only. Add clipboard image paste (terminal capability permitting) and drag-drop of files onto the composer, both routing through the existing attachment_core.process_attachment_path pipeline and the per-session pending-attachment state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pasting an image stages it as the pending attachment with the composer indicator
- [ ] #2 Dropping a supported file on the composer behaves like picking it in the file dialog (inline vs attachment routing preserved)
- [ ] #3 Unsupported/oversized paste/drop shows the same validation toasts as the picker path
<!-- AC:END -->
