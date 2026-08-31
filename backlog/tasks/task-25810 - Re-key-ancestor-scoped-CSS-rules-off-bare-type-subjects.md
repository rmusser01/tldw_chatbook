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

## Partial implementation landed: ancestor rejection in the fast path

A cheaper route to most of this win was found and implemented, **without
touching a single CSS rule or widget markup**.

`tldw_chatbook/Utils/textual_css_fastpath.py` already owns candidate
construction for `Stylesheet.apply`. It now also **rejects candidates whose
leading compound names an ancestor the node does not have**: a rule like
`#panel Button` is skipped for every Button that has no `#panel` ancestor,
before upstream runs full selector matching on it.

Interleaved A/B (four pairs, filter toggled in place, median of five
full-screen `stylesheet.update` calls per arm) on a 502-node Console:

| arm | `update(screen)` | samples |
|---|---:|---|
| filter off | 105.0 ms | 103.5 / 104.0 / 105.0 / 108.3 |
| filter on | **66.2 ms** | 64.8 / 65.8 / 66.2 / 66.8 |
| **delta** | **−38.8 ms (−37%)** | ranges do not overlap |

That is 37% of the 60% upper bound this task measured, for none of the
markup churn.

The filter is conservative by construction: a rule survives unless EVERY one
of its selector sets states an unmet requirement, and any shape that cannot
be decided cheaply reports "no requirement". A leading TYPE selector is
deliberately undecidable — matching it against an ancestor needs MRO walking,
which is the cost being avoided.

**Both guards were mutation-tested.** Deleting the one-compound guard (which
would demand an ancestor for `#thing.foo`, whose id is on the SUBJECT) fails
the per-node fidelity tour. Note the first version of the new unit test did
NOT catch that mutation — its `Button.foo` case starts with a TYPE selector
and returns None either way. Cases led by an id/class (`#thing.foo`,
`.foo.bar`) were added, and now the unit test fails on that mutation in
0.66 s rather than only via the 9 s full-app tour.

## What remains for this task

The CSS-level re-keying is still worth doing and is NOT superseded:

- [ ] The filter only helps rules with an id/class-led ancestor. Rules led by
      a TYPE (`Widget Button`) are still evaluated for every Button
- [ ] Re-keying moves the rule out of the `Button` bucket entirely, which
      also shrinks the candidate SET (the filter still builds it, then
      discards)
- [ ] The lint in the original ACs is still the thing that stops regrowth

Re-measure the remaining headroom against the new **66.2 ms** baseline, not
the original 101.1 ms.

## Further lead, not taken here

`apply()` calls `_ordered_candidates` **before** upstream's own per-node
cache check (`Stylesheet.apply` keys a cache on the node's pseudo-class
signature when `update_nodes` passes one). On a cache hit the candidate
build — including the ancestor walk — is pure overhead.

The measured −37% is already **net** of that, on the real
`stylesheet.update` path which does pass a cache, so this is upside rather
than a correction. Deliberately not taken: reproducing upstream's cache-key
construction here would duplicate a private contract that
`test_upstream_apply_still_has_the_shape_the_fastpath_assumes` would then
have to pin as well. Worth measuring the hit rate before deciding it is
worth that coupling.
