---
id: TASK-16220
title: Console grid blows out horizontally at 120 columns with the Inspector rail open
status: To Do
assignee: []
labels:
  - bug
  - console
  - layout
priority: high
---

## Description

At 120x30 with the Inspector (right) rail open, the Console workspace grid's
fr columns resolve to absurd widths -- measured live: left rail `3fr` -> 354,
main `13fr` -> 1534, right rail `4fr` -> 472, i.e. **every fr unit resolved to
118 (the full content width) instead of sharing it**. The shell is unusable at
that size: the transcript starts ~400 columns off-screen. Four
`test_console_shell_regions.py` size2 rows and the two
`test_console_rail_width_budget.py` size rows pin exactly this and are red on
dev.

**Bisected**: green at `64bb15091` (the 120x30 baseline's own commit, 10/10),
first bad commit `7dbbc401b` (TASK-2154 UX remediation) -- the same commit
that raised the left rail's min-width 24 -> 30 (LY-07), added the
force-collapse bands (left < 100, right < 150 unless explicitly opened), and
the main min-width waiver (LY-11).

**What is ruled out, measured**:
- Not the config: `stack_collapsed_rail_labels` false everywhere.
- Not stylesheet loss: the harness is `ConsolidatedCSSApp` (real CSS), and the
  grid's computed styles read `overflow=(hidden,hidden)`, `layout=horizontal`.
- Not raw Textual fr mechanics: a minimal repro of the exact structure
  (Horizontal, overflow hidden, children 13 / 3fr min30 / 13fr min0 /
  4fr min34 / 11, handles display-none, 120 wide) lays out CORRECTLY
  (30/56/34). Something the production children add -- classes, frames, or
  the rails' own content constraints -- flips the solver into per-child fr
  resolution.
- The grid's `min-height: 20` CSS is inline-overridden to 0 (a4c0d4f49) and
  its height resolves 1fr -> 8 rows; the vertical squeeze is separate from
  the horizontal blowout.

Repro: `pytest Tests/UI/test_console_shell_regions.py -k size2` (4 fail);
the exact-mins arithmetic and probe transcripts are in the task-15791 notes.

## Acceptance Criteria

- [ ] At 120x30 with the Inspector rail open, every workspace-grid child's region fits inside the 120-column viewport
- [ ] The fr-blowout mechanism is identified and stated (what makes production's children resolve fr per-child when a minimal replica shares correctly)
- [ ] The size2 rows of test_console_shell_regions.py and both size rows of test_console_rail_width_budget.py pass
- [ ] Degradation at 100-149 columns with both rails open is a stated design rule (which rail yields), not an accident of the solver
