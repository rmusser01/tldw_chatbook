---
id: TASK-31223
title: Footer - never advertise keys the focused widget will swallow
status: Done
assignee: []
created_date: '2026-09-03 22:31'
updated_date: '2026-09-04 00:31'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique P1: the footer shortcut branch keys off screen state, not focus; with focus in an Input it advertised '] next in set | m | R' while keystrokes were inserted as text (a stray ] corrupted the filter to a zero-match list). Keyboard-first brand: the footer is the instrument and must not lie.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With focus in a text input, the footer reflects typing context instead of advertising swallowed action keys
- [x] #2 Walk keys shown in the footer always work when shown
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped in PR #2359. _library_footer_shortcuts_for_current_state drops single-printable-key entries while an Input/TextArea holds focus, keeps esc/enter/F-keys + informational chips, announces the swap ('typing in field'). Re-applied on context FLIPS only from on_descendant_focus, routed through _apply_library_notes_footer_context (Qodo catch: the generic registration overwrote the Notes editor footer). Live-verified both directions.
<!-- SECTION:NOTES:END -->
