---
id: TASK-1375
title: Harden SettingsURLInput zero-width-break cursor math
status: Done
assignee: []
created_date: '2026-08-05 23:38'
updated_date: '2026-08-05 23:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique minor: SettingsURLInput's zero-width-break trick for textual-web is clever but fragile — cursor-math bugs will present to users as 'settings is broken'. Audit the cursor/selection math against the inserted zero-width characters and pin behavior with tests (or replace with a less fragile wrapping approach if one exists).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cursor left/right, Home/End, selection, and deletion behave correctly across zero-width-break positions,Unit tests cover cursor math at and around inserted break characters,No visual change in normal terminal rendering
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit cursor/selection/deletion math empirically (Pilot-driven render_line inspection). 2. No live bugs found: value stays raw so editing ops are correct; _display_index mapping verified; zero-cell break keeps scroll math intact. 3. Pin behavior with comprehensive unit tests in Tests/UI/test_settings_url_input.py. ADR required: no
ADR path: N/A
Reason: routine UX hardening, no architectural decision
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audit-first outcome: the audit found NO live cursor-math bugs, so no widget code changed. The trick is sound by construction: `SettingsURLInput.value` stays raw (no break chars ever enter it), so every editing operation (cursor left/right, Home/End, selection, backspace/delete, validation) runs on raw positions in base `Input`. The only raw->display mapping is `_textual_web_safe_url_display_index`, used solely for the two stylize calls in `render_line`; it was verified correct at every raw index (the break is treated as belonging to the preceding scheme text, so the cursor/selection boundary after a scheme maps just past the break). Scroll and hardware-cursor math (`Input._cursor_offset`, `Strip.crop`) is cell-based and unaffected because `cell_len("\u200b") == 0` — that invariant was verified against the pinned rich/Textual (8.2.7) and is now pinned by a test as a tripwire.

Evidence gathering: Pilot-driven inspection of rendered strips (styled-cell diffs) for cursor at raw 4/5/6/end, selections (3,8)/(0,5), backspace/delete at and around the break boundary, Home/End, and a scrolled narrow-width render; all behaved correctly.

Hardening delivered as a comprehensive pinning suite so a future regression (or a rich/Textual change giving the break non-zero cell width) fails tests instead of presenting as "settings is broken": new `Tests/UI/test_settings_url_input.py` with 12 tests covering display insertion (single/multiple/uppercase/non-URL), exhaustive display-index mapping, the zero-cell invariant, rendered-text/no-visual-change equivalence, cursor stylize cells around the break, selection spans across the break, deletion at/around break positions, Home/End/arrow stepping, scrolled rendering consistency, and password-mode bypass.

Modified/added files:
- `Tests/UI/test_settings_url_input.py` (new; imports the widget read-only from `UI/Screens/settings_screen.py` — no production code changes required)

Tests: `Tests/UI/test_settings_url_input.py` 12 passed.
<!-- SECTION:NOTES:END -->
