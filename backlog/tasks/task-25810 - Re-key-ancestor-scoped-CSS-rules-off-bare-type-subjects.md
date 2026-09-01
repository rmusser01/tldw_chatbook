---
id: task-25810
title: Re-key ancestor-scoped CSS rules off bare-type subjects
status: Done
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

(An earlier version of this task said the overhead compounds with
TASK-25811's outgoing-screen restyle. **TASK-25811 was retracted as a
measurement artifact** in the same PR -- settled windows show zero style
work on the outgoing screen -- so that interaction does not exist and must
not be used to prioritise this work.)

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
- [x] A guard fails new CSS whose subject selector is a bare common type
      (`Button`, `Static`, `Input`, `Vertical`, `Horizontal`) with an
      ancestor qualifier — otherwise this regrows, as the boot CSS budget
      did three cycles running.
      *Done 2026-08-31 as a RATCHET, not a lint:
      `test_ancestor_scoped_bare_type_rule_count_is_a_ratchet` in
      `Tests/Performance/test_textual_css_fastpath.py`, pinned at measured
      274 + 10 slack = 284, never raised, with an anti-vacuity floor of
      150. Counts the PARSED stylesheet, not .tcss text — a text regex is
      how the 08-29 dead-CSS sweep went wrong. Both assertions
      mutation-tested (MAX below actual fails naming offenders; floor
      above actual fails as hollow-census). Already runs per-PR: the file
      is in `perf-guard.yml`. When re-keying lands, LOWER the constant.*

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

**That captures roughly 62% of the measured upper bound**, for none of the
markup churn: a 37-point reduction against a possible ~60-point one
(38.8 ms of a 60.3 ms available saving; the two arms were measured in
separate runs with slightly different baselines, 105.0 vs 101.1 ms, so
compare the percentages rather than the absolute milliseconds).

An earlier revision of this task said "37% of the 60% upper bound", which
conflated a 37% *reduction* with capturing 37% *of the bound*. It
understated what the filter achieved and overstated the remaining
opportunity -- roughly a third of the bound is left, not two thirds.

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

## The number to quote

The −37% above is a synthetic full-screen `stylesheet.update`. On a real,
settled Console → Library navigation (three interleaved pairs):

| arm | apply time per switch |
|---|---:|
| filter off | 72.7 ms |
| filter on | **53.9 ms** |
| **delta** | **−18.8 ms (−26%)** |

**Quote 26%, not 37% — and it is the reduction of a switch's CSS-APPLY
time, not of switch wall time** (Qodo, PR #2258; the paired wall-time delta
was not separately measured — in wall terms it removes ~19 ms per
navigation). Remaining CSS re-keying headroom should be measured against
these numbers, not the synthetic ones.

## Sizing of the remaining re-key (2026-08-31) — CLOSED by owner decision 2026-08-31

Owner accepted the recommendation below ("25810 close"). The re-key
scope is closed; the delivered value is the filter plus the ratchet.

Measured what the outstanding Button re-keying would buy NOW, with the
shipped filter installed, interleaved arms (4 pairs, median of 7 updates
per arm, non-overlapping ranges):

| arm | `stylesheet.update`, 500 nodes |
|---|---:|
| filter ON, index untouched | 77.9 ms (77.7–79.3) |
| filter ON + Button rules modelled as re-keyed | 57.6 ms (57.2–61.6) |
| **remaining value of Button re-keying** | **~20.3 ms synthetic** |

Scaling by the measured real-navigation ratio, that is roughly **8–10 ms
per screen switch**.

Then sized the churn by grouping the 184 ancestor-scoped Button rules by
their leading scope:

- **102 distinct ancestor scopes**; the largest holds 5 rules, the top 20
  cumulate to only 35%. **There is no concentrated slice** — each scope is
  one compose-site edit (buttons gain a class) plus its selector edits, so
  capturing even half the win touches ~50 sites across the whole app.
- 74 of the 184 rules are TYPE-led (`Widget Button`-shaped): the filter
  cannot reject those today, and re-keying them costs the same markup churn.

**Recommendation: close the re-keying scope as not worth the churn** (owner
call). ~8–10 ms per switch, diffused over ~100 tiny groups app-wide, is a
mega-diff with real visual-regression surface for a small win — the shape
of change the stability-over-quick-wins ruling rejects. What this task has
delivered stands on its own: the filter (~62% of the original bound,
zero markup churn) and the never-rise ratchet at 284 that stops regrowth.
If the ratchet ever forces a re-key, do it per-offender at that moment.

(Method note: the first, non-interleaved run of this measurement said
27.9 ms; interleaving corrected it to 20.3. Same lesson as every other
number this review.)
