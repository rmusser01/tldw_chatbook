---
id: TASK-488
title: >-
  Restore non-character workbench modes (dictionaries/lore/personas) across
  navigation
status: To Do
assignee: []
created_date: '2026-07-23 15:30'
labels:
  - roleplay
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-434. TASK-434 restores the Personas workbench selection + preview across a navigation round-trip, but only for CHARACTERS mode; restore is gated to characters because the row-render + mode-widget-visibility (PreviewPane.display, Try-It .display, _render_dictionary_rows/_render_lore_rows/_refresh_profile_rows_worker) live only in _apply_mode, which the restore path does not call. Non-character modes therefore fall back to the default characters view on return. Extend restoration so dictionaries/lore/personas modes restore their mode, list, selection, and center too — via a light 'render current mode's list + sync widget visibility' helper that does NOT switch_mode (which would reset sort/tag/page/selection).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Returning to Personas after a dictionaries/lore/personas round-trip restores that mode with a populated list, the prior selection, correct center pane, and correct Preview/Try-It widget visibility
<!-- AC:END -->
