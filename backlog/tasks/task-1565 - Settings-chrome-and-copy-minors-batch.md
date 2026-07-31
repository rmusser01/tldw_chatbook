---
id: TASK-1565
title: 'Settings: chrome and copy minors batch'
status: Done
assignee: []
created_date: '2026-07-31 02:00'
labels: [settings, ux, P3]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Grouped P3 findings from the 2026-07-31 critique, each small but shipped:
- "Tokens: -- |" footer fragment intermittent across categories, always with a dangling pipe.
- Breadcrumb advertises "accounts"; no accounts category exists.
- Sidebar does not auto-scroll to the selected category (selection can be entirely off-viewport).
- "Duration 1.5" / "Animation 1.0" unitless on Splash Screen.
- RAG: "Clone it, then Set active, to edit." comma cadence; red Delete on a read-only built-in profile; description flyout overlaps the rail's lower rows.
- Guided-path chips ("Providers | Console | Privacy") disagree with sidebar names ("Providers & Models", "Console Behavior", "Privacy & Security").
- Two permanently visible filter-help lines say the same thing.
- Splash gallery mixes label casing and shows raw ids ("TLDW CHATBOOK v2.0 (digital_rain)").
- "WIP" appears in user-facing copy in six categories.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed item fixed or explicitly declined with a reason in notes.
- [x] #2 No footer chrome renders dangling separators.
- [x] #3 Category names are referenced identically everywhere they appear.
<!-- AC:END -->

## Implementation Notes

Fixed: idle footer token placeholder "Tokens: -- |" -> "Tokens: --" (live
format never had the pipe); breadcrumb + docstring drop the nonexistent
"accounts"; rail auto-scrolls the selected category into view
(`scroll_visible` after refresh); splash "Duration (s)" / "Animation speed
(x)" units; RAG read-only guidance comma cadence ("Clone it, then press Set
active to edit the clone."); Advanced Config guided chips renamed to match
the sidebar exactly ("Providers & Models", "Console Behavior", "Privacy &
Security"); duplicate filter-help line removed (the live status line
carries the same guidance); user-facing "WIP" phrasing replaced
("read-only here", "not available yet", "Read-only" label) with pinned
tests updated.

Declined with reasons: splash gallery casing/raw ids (display names are the
cards' own metadata; normalizing would misrepresent card identity);
RAG description-flyout overlap and the red Delete on a read-only built-in
profile (both need RAG-pane layout/state work better scoped with task-1345's
RAG follow-ups than this chrome batch).
