---
id: TASK-32602
title: Preserve Prompt editor focus across terminal resize
status: To Do
assignee: []
created_date: '2026-09-15 06:26'
updated_date: '2026-09-15 06:44'
labels:
  - library
  - prompts
  - focus
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Prompt continuity audit found that resizing a focused Advanced message field from 170x48 to 80x24 leaves it below the viewport. A later wide transition can restore the Create rail row instead of the live editor focus. Stable-size Basic and Advanced layout fixes do not address this shared resize restoration path. Evidence and the removed local-scroll experiment are recorded in Docs/superpowers/qa/2026-09-14-prompt-continuity/README.md. Reproduce with the new continuity journey before its Basic return: focus the Advanced message field, resize to 80x24 and back to 170x48, and assert the same focused widget plus its compositor-painted text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A focused Prompt text field stays visibly focused across 170x48 to 80x24 and back without changing its content or identity.
- [ ] #2 A newer explicit focus move wins over deferred resize restoration, including a move outside the Prompt editor.
- [ ] #3 Resize remains free of source reloads and preference writes, with production-CSS and native evidence.
<!-- AC:END -->
