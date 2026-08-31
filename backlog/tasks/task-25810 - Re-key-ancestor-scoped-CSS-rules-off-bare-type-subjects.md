---
id: task-25810
title: Re-key ancestor-scoped CSS rules off bare-type subjects
status: To Do
assignee: []
created_date: '2026-08-30'
labels:
  - performance
  - css
priority: high
---

## Description (the why)

Textual indexes every CSS rule under its **rightmost selector only**
(`RuleSet._post_parse`: `selector_set.selectors[-1]`). A rule written as
`#prompt-variables-actions Button:disabled` is therefore filed under
`Button` and becomes a candidate for **every Button in the app**, each of
which runs full selector matching before rejecting it.

On ChatScreen (502 nodes, 4,380 rules) this is not a rounding error:
`Button` carries 188 rules, 180 of them ancestor-scoped, against 110 live
buttons — 20,680 candidate considerations, **71% of all candidate work on
the screen**. Across the common type keys, **93.2% of candidate work
(24,487 of 26,279) comes from ancestor-scoped rules that cannot match the
node considering them.**

This compounds with TASK-25811: the outgoing-screen restyle pays this
overhead too.

## Evidence

Measured on dev `0ef6f3fd4e`; full method in
`Docs/Design/2026-08-30-holistic-perf-review.md` §2.

A/B on a real full-screen `stylesheet.update` (median of 7, warm):

| arm | time |
|---|---:|
| baseline | 101.1 ms |
| ancestor-scoped bare-type rules removed from the index | 40.8 ms |
| **delta** | **60.4 ms (60%)** |

The ablated arm renders wrong styles — it prices the work, it is not the
proposed change. **60% is an upper bound**: re-keying moves a rule to a
narrower key rather than deleting it, so the intended widgets still
evaluate it.

## Acceptance Criteria (the what)

- [ ] The widest offenders are re-keyed so their subject carries a class
      only the intended widgets have (start with `Button`: 180 scoped
      rules, 110 live instances)
- [ ] Measured full-screen `stylesheet.update` on ChatScreen improves
      against the 101.1 ms baseline, reported with the same median-of-7
      method rather than a single sample
- [ ] The realised saving is stated as measured, and explicitly compared
      with the 60% upper bound — a shortfall is expected and fine, an
      unexplained match is suspicious
- [ ] No visual regression: the re-keyed rules still apply to exactly the
      widgets they applied to before (assert computed styles, not
      selector text)
- [ ] A guard fails new CSS whose subject selector is a bare common type
      (`Button`, `Static`, `Input`, `Vertical`, `Horizontal`) with an
      ancestor qualifier — otherwise this regrows, as the boot CSS budget
      did three cycles running

## Notes

Do not chase every key. `Widget` (1 rule, 502 nodes) and `Checkbox`
(0 live) are noise. The value is concentrated in `Button` and `Static`,
which together are 24,200 of the 26,279 considerations.
