---
id: TASK-24620
title: Console voice chip starves the composer draft at narrow widths
status: Done
assignee:
  - @zcode
created_date: '2026-08-30'
updated_date: '2026-08-30'
labels:
  - console
  - defect
  - ux
priority: high
dependencies: []
---

## Description (the why)

Follow-up to TASK-24415 (PR #2214), which flagged this exact same-class
suspect in its Implementation Notes — now confirmed by the user having hit
it live. The dictation voice chip (`#console-voice-status`) sizes itself
against the composer's FULL width reserving only `VOICE_CHIP_MIN_WIDTH`
(24 cells):

```python
total_width = self.size.width or self.VOICE_CHIP_MAX_WIDTH * 2
available = max(0, total_width - self.VOICE_CHIP_MIN_WIDTH)
width = min(self.VOICE_CHIP_MAX_WIDTH, available)
```

It ignores the left cluster (18), the actions row (25), and the draft
entirely. With the chip's 51-cell executor-wait copy or any long state
message, a ~100-cell composer row gives the chip up to 53 cells and the
`1fr` draft (min_width 0) the remainder — a few columns, or zero. Exactly
the starvation shape TASK-24415 fixed for `#console-send-disabled-reason`,
in the one sibling the earlier task measured but did not reach.

The mic-live state itself is not lost when the chip must yield: the Dictate
button's label carries it ("Dictating"), so the chip is advisory detail
(partial transcript, elapsed, wait copy) under the draft floor — the same
priority TASK-24415 established.

## Acceptance Criteria

- [x] With a dictation chip state rendered and a narrow composer (80-col
      app), the visible draft keeps non-zero width; the chip clamps to a
      live-row-derived budget or hides below a legible floor.
- [x] At wide widths the chip still renders within its 53-cell ceiling and
      the draft keeps its 32-cell floor.
- [x] A resize re-derives the chip budget (a shrink retracts the chip, not
      the draft).
- [x] Direct unit tests cover the budget helper's branches; a mounted
      geometry test pins the integration (the TASK-24415 test pattern).
- [x] Live verification in a real terminal at 80 columns with a chip state
      showing: the draft remains visible (error-state chip hidden, typed
      draft and caret visible; failure feedback arrived via the rail
      panel, not the starved chip).

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: layout bug fix inside the existing composer widget, mirroring
TASK-24415's reviewer-approved pattern; no contract or boundary decision.

1. RED: mounted geometry test at narrow width with a chip state — draft
   region collapses (reproduce).
2. Add `_voice_chip_width_cap` mirroring `_send_reason_width_cap`
   (row − left cluster − actions(attachment-aware) − draft floor; hide
   below a legibility floor); apply it in `set_voice_status`; cache the
   last status inputs and replay from `on_resize`.
3. Unit-test the cap branches; live-verify at 80 cols.

## Implementation Notes

Fixed 2026-08-30, TDD (RED reproduced the user-hit bug: draft at 2 columns
beside a 53-cell chip; even at wide widths the draft got 28 < the 32 floor).
88 dictation-streaming tests, 47 popup/composer/dictation tests green;
live-verified at 80 columns.

- **Approach**: `_voice_chip_width_cap` mirrors TASK-24415's
  `_send_reason_width_cap` -- budget = row − left cluster − actions −
  `ADVISORY_MARGIN_ALLOWANCE`(2) − draft floor; below a 12-cell legible
  remainder the chip hides (the Dictate button's label carries the
  mic-live state, and dictation failures surface through the rail panel).
  `set_voice_status` applies it, caches its last inputs
  (`_voice_status_last`, class attr for fixture safety) and its APPLIED
  width (`_voice_chip_last_width` -- the cache, not `region`, because
  region is stale until the next layout pass), and `on_resize` replays.
- **Preparing is exempt**: the busy/preparing chip is action feedback for
  a Dictate press; the 80-column busy-parakeet tests pin the WHOLE copy
  visible, so preparing keeps its legacy full-width sizing (it already
  collapses chrome via `_sync_full_width_voice_presentation`). Every other
  state uses the budgeted cap.
- **Two-strip interplay fixed**: the reason strip's cap now subtracts the
  chip's cached width (chip has priority) and both caps subtract the 2
  separator-margin cells measured on the laid-out row -- without these,
  the two advisory strips each budgeted the full remainder and jointly
  starved the draft to 30 < 32 at 160 cols.
- **Pre-existing dev regression repaired**: TASK-24415 (PR #2214) broke
  `test_busy_parakeet_mic_stays_reachable_and_cancels_at_80_columns` on
  dev (verified failing on plain origin/dev) -- its final assert required
  the reason strip displayed at a 77-cell row where the 24415 budget is
  legitimately zero. The assert now pins the actual restore contract
  (`_voice_full_width_preparing` cleared, chip-width cache reset); that
  suite was not in PR #2214's verification runs.
- Files: `tldw_chatbook/Widgets/Console/console_composer_bar.py`,
  `Tests/UI/test_console_composer_reason_width.py` (+3 mounted, +4 unit),
  `Tests/UI/test_console_dictation_streaming.py` (stale assert updated to
  the post-24415 contract).
- ADR: not required (layout fix mirroring an already-reviewed pattern).
