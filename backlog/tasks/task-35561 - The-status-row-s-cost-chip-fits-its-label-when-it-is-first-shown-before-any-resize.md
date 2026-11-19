---
id: TASK-35561
title: The status row's cost chip fits its label when it is first shown, before any resize
status: To Do
created_date: 2026-10-18 17:42
dependencies: []
labels:
- model-config-redesign
- follow-up
- console
priority: low
references:
- tldw_chatbook/Widgets/Console/console_status_chips.py
- Tests/Chat/test_console_status_chips_cost.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's live capture phase (2026-10-18) and confirmed in triage to predate Phase 9: it reproduces at the phase base 111f8e48fc.

In some launches the first 211x44 frame shows the cost chip cut mid-word, for example "Context 0% · Curren". After a resize it shows "Context 0%", whole. The chip keeps its full label because its fit runs while the chip is not yet laid out: its `virtual_region.x` is still 0. `ConsoleStatusChips._refit_after_layout`'s place key leaves out the chip's own placement, so the layout that finally places it does not refit it. A trace showed the chip laid out at x=179 about 0.3 s after both scheduled refits had run.

`test_the_cost_chip_fits_when_it_is_shown_after_layout` passes because in its small harness the first refresh already places the chip.

Live evidence: Phase 9 capture 11a, and the triage's 11a captures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In a full Console at 211x44 and 235x52 the cost chip's first painted label fits whole, without needing a resize
- [ ] #2 A deterministic test reproduces the late placement (the chip shown while it is not yet laid out) and fails before the fix
- [ ] #3 The fix costs at most one extra refit per placement change and keeps the existing refit-count pins true or updates them on purpose
<!-- AC:END -->
