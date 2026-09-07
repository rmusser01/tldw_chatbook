---
id: TASK-1514
title: >-
  EnhancedFileDialog Select is unpinned against the bundle-tier width collapse
status: To Do
assignee: []
created_date: '2026-07-30 14:00'
labels:
  - evals
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Noted while fixing task-1479 (2026-07-30). The vendored fspicker's filename input collapsed to ~5 cells because `features/_conversations.tcss` ships a bare `Select { width: 100%; }` that outranks any widget DEFAULT_CSS through the generated bundle. The fix pinned `FileSave/FileOpen InputBar` widgets at bundle tier, but `Widgets/enhanced_file_picker.py`'s `EnhancedFileDialog` composes its own `#file-filter` Select and is not pinned against the same rule — a plausible latent layout bug in the Enhanced picker family (same defect class previously fixed for MCP/approval-card Selects).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] EnhancedFileDialog's footer widgets render at sane widths under the real CSS bundle (painted-geometry test)
- [ ] Any collapse found is fixed at the same bundle tier with a scoped selector
<!-- AC:END -->
