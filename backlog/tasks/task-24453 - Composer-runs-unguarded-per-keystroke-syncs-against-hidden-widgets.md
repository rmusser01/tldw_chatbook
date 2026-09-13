---
id: TASK-24453
title: Composer runs unguarded per-keystroke syncs against hidden widgets
status: Done
assignee: []
created_date: '2026-08-29'
labels:
  - performance
  - ui
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Typing in the Console composer costs 20.6 ms of CPU per keystroke on an empty conversation,
with 35.3 `query_one` calls, 8.4 `Static.update()` calls and 31.8 `set_class` calls per key.

`Widgets/Console/console_composer_bar.py::_sync_collapsed_presentation` runs three times per
keystroke. Each run performs 4 `query_one` lookups, a `Static.update()` and 4 `set_class`
calls -- on the collapsed row, which is `display:none` for the entire time the user is typing.
It is unconditional: there is no early return on unchanged state and no cached widget handles.

Other unguarded per-key repeats: `_apply_draft_height` (4 `query_one`), `_refresh_visible_draft`
(2 `query_one` + 2 `Static.update`), `_console_command_popup_or_none` (2 `query_one`), and
`_sync_raw_cli_state` (2 `query_one`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `_sync_collapsed_presentation` performs no DOM work when its inputs are unchanged since the previous call
- [x] #2 No content update or class mutation is applied to the collapsed presentation row while the composer is expanded
- [x] #3 `query_one` calls per keystroke in the Console composer are reduced by at least half against the pre-change baseline
- [x] #4 CPU per keystroke on an empty Console conversation improves measurably in an interleaved A/B
- [x] #5 Collapsing and expanding the composer, and the raw-CLI danger styling, still behave correctly
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Guarded five unconditional per-keystroke syncs in `console_composer_bar.py`, each with a
signature over exactly the state its body reads -- the same fingerprint pattern
`main_navigation._update_overflow_hints` already uses. All signatures reset in `on_mount`, so a
remounted composer always re-applies to fresh widgets.

- `_sync_collapsed_presentation` (ran 3x/key): signature + it no longer touches the collapsed
  row at all while the composer is expanded, because that row is `display:none` then. `_collapsed`
  is in the signature, so collapsing always misses the cache and repaints before the row is seen.
- `_apply_draft_height`: was calling `self.refresh(layout=True)` on EVERY keystroke, which is
  what forced a whole-screen relayout per keypress. Now guarded on (row_count, recovery_rows,
  composer_height); `_apply_collapsed_geometry` clears the signature since it overwrites the
  geometry.
- `_sync_improvement_recovery`: `visible` is False for a whole ordinary typing session; the guard
  also skips the `_refresh_visible_draft` it chained into.
- `_sync_raw_cli_state`: status row content/geometry guarded on `active`.
- `_sync_hidden_input`: writes the mirror only when the canonical payload changed.

Measured A/B on the real app:
- `query_one` per keystroke 35.4 -> 12.3 (-65%)
- `Static.update` per keystroke 8.5 -> 2.4 (-72%)
- `set_class` per keystroke 31.8 -> 17.9 (-44%)
- `refresh(layout=True)` invalidations per keystroke 11.7 -> 6.7 (-43%)
- CPU per keystroke ~21.8 -> ~19.7 ms

Note on what did NOT move: `Screen._refresh_layout` stays at ~1.04 per keystroke. Textual
coalesces invalidations into one layout per frame, and `_refresh_visible_draft` still legitimately
invalidates once per key because the draft text genuinely changed. Removing invalidations below
that floor cannot remove the layout itself.

Modified: `tldw_chatbook/Widgets/Console/console_composer_bar.py`.
<!-- SECTION:NOTES:END -->
