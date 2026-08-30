---
id: TASK-24620
title: Console voice chip starves the composer draft at narrow widths
status: In Progress
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

- [ ] With a dictation chip state rendered and a narrow composer (80-col
      app), the visible draft keeps non-zero width; the chip clamps to a
      live-row-derived budget or hides below a legible floor.
- [ ] At wide widths the chip still renders within its 53-cell ceiling and
      the draft keeps its 32-cell floor.
- [ ] A resize re-derives the chip budget (a shrink retracts the chip, not
      the draft).
- [ ] Direct unit tests cover the budget helper's branches; a mounted
      geometry test pins the integration (the TASK-24415 test pattern).
- [ ] Live verification in a real terminal at 80 columns with a chip state
      showing: the draft remains visible.

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

(added after implementation)
