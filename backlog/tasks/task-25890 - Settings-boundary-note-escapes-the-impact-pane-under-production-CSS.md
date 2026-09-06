---
id: TASK-25890
title: Settings boundary-note escapes the impact pane under production CSS
status: To Do
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

- [ ] `#settings-boundary-note` stays inside `#settings-impact-pane` at
      140x42 under the full production stylesheet (bundle + split sheets)
- [ ] The `xfail` mark on the settings param of
      `test_runtime_and_settings_default_states_preserve_workbench_geometry`
      is removed in the same change
- [ ] Verified against a live capture, not only the harness — the harness
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
