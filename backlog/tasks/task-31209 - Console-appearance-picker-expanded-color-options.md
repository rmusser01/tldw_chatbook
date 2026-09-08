---
id: TASK-31209
title: 'Console appearance picker: expanded color options'
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-03 12:00'
labels:
  - console
  - ui
dependencies:
  - TASK-31207
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-31207 ships the appearance picker with a fixed ~14-swatch palette. Expand
the color choices: a larger curated palette and/or custom hex entry (validated,
stored as `#rrggbb` so existing sanitization keeps working). Keep the picker
usable in a narrow terminal — consider a scrollable/paginated swatch region
rather than one long row.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Users can choose from more colors than the initial fixed palette (extended palette and/or validated custom hex entry)
- [x] #2 Stored values remain `#rrggbb`; existing sanitization treats any legacy or invalid value as before (unset), and rows render identically for previously-set appearances
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Extend `CONSOLE_APPEARANCE_PALETTE` with deeper/darker siblings of the
   base hues plus uncovered wheel segments.
2. Custom hex entry in the picker: a small Input that live-applies valid
   canonical hex (leading `#` optional, case-insensitive) and ignores junk
   or mid-typing input without disturbing the current selection.
3. Keep the swatch strip usable at the 64-cell modal: scrollable
   `HorizontalScroll` strip of 3-cell swatches.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Palette grew 13 -> 21 swatches (Rose, Ruby, Gold, Lime, Emerald, Deep
  Blue, Purple, Fuchsia); all canonical `#rrggbb`, unique, still
  dark-panel-readable.
- New `normalize_console_appearance_hex_input` (Chat/console_appearance.py):
  optional `#`, case-insensitive, returns None for anything not exactly six
  hex digits, so the picker's `Input.Changed` handler can ignore mid-typing
  input and never clobber a chosen color; the explicit "none" swatch stays
  the only way to clear.
- The swatch strip became a `HorizontalScroll` (22 swatches no longer fit one
  60-cell row); shrinking swatches to two cells was tried and rejected --
  content width hit zero under layout and crashed cell chopping. Selected
  swatch highlight is a background tint, not a border, for the same reason.
- Custom hex values pass the same `#rrggbb` validation on read; older stored
  palette colors are unaffected (format-only validation, forward-compatible
  by design from task-31207).
- Tests: hex normalization table, palette-size floor, live-apply +
  junk-ignored modal test (`test_console_appearance_picker.py`), plus all
  prior appearance suites green (53 across appearance/switcher; 413 across
  the full touched-suite sweep).
- Modified: `Chat/console_appearance.py`,
  `Widgets/Console/console_appearance_picker_modal.py`.
<!-- SECTION:NOTES:END -->
