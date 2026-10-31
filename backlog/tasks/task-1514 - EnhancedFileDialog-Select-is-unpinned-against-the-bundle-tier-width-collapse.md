---
id: TASK-1514
title: >-
  EnhancedFileDialog Select is unpinned against the bundle-tier width collapse
status: Done
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
- [x] EnhancedFileDialog's footer widgets render at sane widths under the real CSS bundle (painted-geometry test)
- [x] Any collapse found is fixed at the same bundle tier with a scoped selector
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

Root cause retired rather than pinned: the app-wide bare 'Select { width: 100%; }' rule that collapsed the dialog was removed from css/features/_conversations.tcss by TASK-33003.1 (retirement note at _conversations.tcss:293-300); the vendored FileSave/FileOpen InputBar Select pinning lives at bundle tier in css/components/_dialogs.tcss:29-30.
