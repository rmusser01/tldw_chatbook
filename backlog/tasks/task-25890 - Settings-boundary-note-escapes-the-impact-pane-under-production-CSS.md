---
id: task-25890
title: Settings boundary-note escapes the impact pane under production CSS
status: Done
assignee: []
created_date: '2026-08-31'
labels:
  - settings
  - css
  - dev-red
priority: medium
---

## Description (the why)

At 140x42 under the PRODUCTION stylesheet, `#settings-boundary-note`
(classes `settings-detail-row`) renders at y=45 — **outside**
`#settings-impact-pane` (y 5..40). The workbench-geometry contract that
should catch this (`test_runtime_and_settings_default_states_preserve_
workbench_geometry[settings-...]`) had never actually tested it: its
`DestinationHarness` loads only the consolidated widget-defaults sheets, no
app bundle, so it asserted geometry no user ever sees.

## Evidence

Found during TASK-25812 (agentic CSS split-by-screen), 2026-08-31, and
**verified pre-existing**: the branch base `49e648b7d1` fails identically
when the same test is pointed at the production stylesheet — this is not a
split regression, it is a masked production condition.

| harness CSS | result |
|---|---|
| consolidated only (as shipped) | green — but vacuous |
| production bundle (branch base) | **red, marker at y=45 vs pane 5..40** |
| production bundle + split sheets | red, identically |

The test now runs under the production stylesheet set with the settings
param marked `xfail(strict=True)` citing this task — when the geometry is
fixed, the strict xfail flips loudly and the mark comes off.

## Acceptance Criteria (the what)

- [x] `#settings-boundary-note` stays inside `#settings-impact-pane` at
      140x42 under the full production stylesheet (bundle + split sheets)
- [x] The `xfail` mark on the settings param of
      `test_runtime_and_settings_default_states_preserve_workbench_geometry`
      is removed in the same change
- [x] Verified against a live capture, not only the harness — the harness
      masked this once already

## Notes

Sibling hazard worth sweeping separately: any other geometry contract using
`DestinationHarness` asserts against a CSS-less mount. The ACP param of the
same test passes under production CSS, so the vacuity is not universal —
but it is structural.

## Renumbering provenance

Filed 2026-08-31 as task-25814; renumbered to task-25890 the same day when
preflight caught a filename collision with dev's older
`task-25814 - Console-send-is-blocked-before-provider-dispatch...` (the
2026-08-21 owner rule: the older arrival keeps the id). The 25890 id was
verified free across all 72 remote branches before claiming. The xfail
reason string in `test_destination_visual_parity_correction.py` was updated
in the same commit.

## Implementation Notes

Closed by TASK-33003.7 (model-config Phase 3, branch model-config-p3,
2026-09-29), which absorbed this task. Root cause, measured under the
production stylesheet at 140x42: the Scope Inspector's `2fr` share left a
25-column pane (18-cell text column), and a blank row after every inspector
row added 7 more rows, so the Overview inspector held 40 rows of content in a
16-row window and the note sat at y=55 (pane 5..40). At 211x44 the same blank
rows cut the note's last line at the fold. Fix, in
`css/features/_settings.tcss`: the scoped blank-row override is deleted,
and the pane keeps the spec's Inspector width (36) as a `min-width` floor
from 134 columns. A static floor starved the detail pane below that (Network
CA path 7 editable cells at 120 columns), so `settings_screen.py` sets the
floor class from `SETTINGS_INSPECTOR_FLOOR_MIN_WIDTH`, next to the existing
compact-workbench width sync. The strict xfail is removed; live captures at
140x42, 211x44 and 235x52 are in `qa/model-config-p3-2026-09-28/task-7/`.
Details in TASK-33003.7.
