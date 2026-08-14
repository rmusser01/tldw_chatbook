---
id: TASK-16220
title: >-
  Console grid blows out horizontally at 120 columns with the Inspector rail
  open
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-08-14 23:38'
labels:
  - bug
  - console
  - layout
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 120x30 with the Inspector (right) rail open, the Console workspace grid's
fr columns resolve to absurd widths -- measured live: left rail `3fr` -> 354,
main `13fr` -> 1534, right rail `4fr` -> 472, i.e. **every fr unit resolved to
118 (the full content width) instead of sharing it**. The shell is unusable at
that size: the transcript starts ~400 columns off-screen. Four
`test_console_shell_regions.py` size2 rows pin this geometry directly. The two
`test_console_rail_width_budget.py` size rows are also red, on a later stale
12-cell label oracle for the now-deliberate 13-cell label-plus-gutter contract.

**Bisected**: green at `64bb15091` (the 120x30 baseline's own commit, 10/10),
first bad commit `7dbbc401b` (TASK-2154 UX remediation) -- the same commit
that raised the left rail's min-width 24 -> 30 (LY-07), added the
force-collapse bands (left < 100, right < 150 unless explicitly opened), and
the main min-width waiver (LY-11).

**What is ruled out, measured**:
- Not the config: `stack_collapsed_rail_labels` false everywhere.
- Not stylesheet loss: the harness is `ConsolidatedCSSApp` (real CSS), and the
  grid's computed styles read `overflow=(hidden,hidden)`, `layout=horizontal`.
- Not production descendants: the earlier healthy minimal repro used
  `13fr min0` for the Transcript. An exact three-minimum repro (`3fr min30 /
  13fr min56 / 4fr min34`) inside the real 118-column content box reproduces
  the production blowout exactly (354/1534/472). Textual 8.2.7 min-clamps all
  three fractions; with no fraction left unresolved it returns the original
  118 columns as the `fr` unit. The 120-column automatic Inspector-open path
  bypasses the compact override that would waive the Transcript minimum.
- The grid's `min-height: 20` CSS is inline-overridden to 0 (a4c0d4f49) and
  its height resolves 1fr -> 8 rows; the vertical squeeze is separate from
  the horizontal blowout.

Repro: `pytest Tests/UI/test_console_shell_regions.py -k size2` (4 fail);
the exact-minimum arithmetic and probe transcripts are in the task-15791 notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 120x30 with the Inspector rail open, every workspace-grid child's region fits inside the 120-column viewport
- [ ] #2 The fr-blowout mechanism is identified and stated (all three minimums exhaust Textual's fractional pool, and the automatic Inspector-open path bypasses the compact override)
- [ ] #3 The size2 rows of test_console_shell_regions.py and both size rows of test_console_rail_width_budget.py pass
- [ ] #4 Degradation at 100-149 columns with both rails open is a stated design rule (which rail yields), not an accident of the solver
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Refine ADR-043 for Inspector-first compact priority and responsive focus handoff.
2. Add pure rail-priority, reveal-update, and width-band contracts with RED-to-GREEN tests.
3. Thread one width authority through existing Console compose/current/resize paths and preserve keyboard focus.
4. Prove production-hierarchy containment at 120 columns and refresh the stale label-width oracle.
5. Run focused verification, self-review, update docs/task evidence, and close through Backlog CLI.

Detailed plan: [2026-08-14 Console grid horizontal blowout implementation plan](../../Docs/superpowers/plans/2026-08-14-console-grid-horizontal-blowout.md)

- ADR required: yes
- ADR path: [ADR-043: Console rail compact-collapse yields to explicit toggles](../decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md)
- Reason: TASK-16220 refines the durable conflict policy between two persisted rail preferences in the 100-149-column compact band.
<!-- SECTION:PLAN:END -->
