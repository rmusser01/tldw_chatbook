---
id: TASK-225
title: Escape save path toast in Save Image worker
status: Done
assignee:
  - '@claude'
created_date: '2026-07-13 11:15'
updated_date: '2026-07-16 20:31'
labels:
  - console
  - tech-debt
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Save Image worker's success toast notify(f"Image saved to {target}") interpolates the config-derived save path unescaped into a markup-rendering surface. Not user-file-derived (config + generated filename), so low risk, but a [chat.images].save_location containing markup-like tokens would render wrong. Escape it like the sibling toasts fixed in f1824513.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Save-path toast escapes the interpolated path per repo convention
- [x] #2 A path containing markup-like tokens displays literally
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both Save Image toasts (single 'Image saved to …' and multi 'Saved N images to …') now escape_markup the interpolated path, matching the sibling toasts from f1824513. Tests exercise the real _save_console_message_image worker over a bare ChatScreen with a markup-token directory (sav[e]dir) and assert the opening bracket is escaped (Rich only needs the opening bracket neutralized). Sweep 1018/69/0.
<!-- SECTION:NOTES:END -->
