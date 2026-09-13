---
id: TASK-32303
title: >-
  Console RAG settings modal still renders the ✓/○ selection pair — decide
  whether the Library glyph legend is product-wide
status: Done
assignee: []
created_date: '2026-09-11 00:54'
updated_date: '2026-09-11 17:29'
labels:
  - console
  - ux
  - critique-9
  - decision-needed
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Library glyph legend (task-32235) moved selection to ☑/☐ and kept ○ for blocked/disabled. `Widgets/Console/console_rag_settings_modal.py` still mirrors the old ✓/○ pair, so the same state reads differently on the Console screen. Product decision: adopt the Library legend app-wide, or record that Console keeps its own.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A decision is recorded (product-wide legend or Console-specific), and if product-wide the modal uses the shared constants
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate the Library legend constants (task-32235) and the Console modal's own pair.
2. Red-first: move the modal's pins to the shared constants.
3. Swap the literals for the shared constants; follow any copy that names the glyphs.
4. Record the decision verbatim; docs + stamp; live-verify the modal at 235x52.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
``console_rag_source_toggle_label`` now reads ``LIBRARY_GLYPH_SELECTED`` /
``LIBRARY_GLYPH_UNSELECTED`` from ``Library/library_shell_state.py`` -- the
same constants Library's own scope toggles read (``scope_toggle_label``), not a
second copy of the glyphs -- so a source toggle paints ☑/☐ and "○" keeps the
one meaning task-32235 left it: blocked or disabled. Scope is this modal, not
Console as a whole: the per-conversation library-access modal's RADIO pair is
still ●/○ and is filed as task-32464 (review round 1, F1). This modal has no
glyph-marked blocked state of its own (its only disabled control is the Run
button, which carries no marker), so nothing else moved. The function's own
docstring, which quoted "✓ Notes" / "○ Prompts", follows the change, and so
does the Console guide's description of the modal.

**Evidence.** The three existing pins were moved to the shared constants
FIRST and measured red against the old literals, then green. A new pin reads
the PAINTED strips of each toggle at 235x52 and refuses both retired glyphs.
``test_source_scope_survives_a_screen_state_round_trip`` is red before and
after this branch (``'ChatScreen' object has no attribute '_session'``) --
verified at base.

**Deviation:** no live capture. The Console's Library-search modal is opened
from a control bar that is locked until first-run provider setup completes,
and this host has no provider credentials, so the 235x52 painted assertion in
the mounted harness stands in for it.

**Files:** ``tldw_chatbook/Widgets/Console/console_rag_settings_modal.py``,
``Tests/UI/test_console_rag_settings_modal.py``,
``Docs/User_Guide/console/context-and-rag.md``.
<!-- SECTION:NOTES:END -->

## Decision

<!-- SECTION:DECISION:BEGIN -->
Console RAG adopts Library's legend: the RAG settings modal reads the shared
constants; Library stays as shipped.

Scope note (review round 1, F1): "adopts the legend" is this modal, not every
Console surface. `Widgets/Console/console_library_access_modal.py:45,53` still
paints ○ for an unselected RADIO, which under the Library legend reads as
blocked; that glyph pair is a separate decision, filed as task-32464 and
deliberately not touched here.
<!-- SECTION:DECISION:END -->
