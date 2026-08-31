# Holistic performance review — 2026-08-30

Sixth holistic performance review of tldw_chatbook. Same commissioning
prompt as the 2026-08-22, 08-24, 08-27 and 08-29 reviews.

* **Pin:** dev `0ef6f3fd4e` (405 commits since the 08-29 pin `bc1e26ce60`).
* **Machine:** darwin 24.6.0, Python 3.12.11, Textual 8.x.
* **Method:** measured, not read. Every number below states how it was
  taken; the ones that were wrong the first time say so.

---

## 0. Ratchet baseline (ADR-097)

Run on the pin, before any change:

| guard | measured | limit | headroom | verdict |
|---|---|---|---|---|
| boot import weight | 634 | 660 | 26 | green |
| `_ui_ready` census | 969 | **972** | **3** | green, at the edge |
| boot CSS bytes | **878,333** | 860,000 | **−18,333** | **RED** |
| pre-import payload | 488 / 372,041 LOC | 500 / 380,000 | 12 / 7,959 | green, tight |

Two governance observations:

1. **The `_ui_ready` constant is 972, but ADR-097's table says 970.** It was
   raised deliberately by the owner on 2026-08-29 (`6fac5dbf95`, "raise
   ui-ready census ratchet 970->972 for tls_trust (PR #2223, ADR-097
   deliberate refresh)"). The decision was made and the cause named — but
   ADR-097 requires an **exception-ledger row in the same commit**, and the
   ledger still reads *"(none granted yet)"*. The ledger is therefore
   inaccurate about the one raise that has occurred. Cheap to fix; left as
   a finding because an audit trail that silently misses entries is worse
   than no audit trail.
2. The CSS ratchet has been red since before this review and is the same
   file flagged as an owner call in the 08-27 and 08-29 reviews. It is now
   quantified in time (below) rather than only in bytes.

---

## 1. CSS parsing is ~191 ms in a single boot hit, and one file is 28% of it

### What the ratchet says vs what it measures

The CSS byte guard asserts *"Every one of these bytes is parsed before first
paint"* but prices **bytes**, which nobody had converted into time. Measured:

| source | bytes | rules | cold parse |
|---|---:|---:|---:|
| `tldw_cli_modular.tcss` | 671,467 | 3,273 | 119.4 ms |
| `widget_defaults_scoped.tcss` | 95,077 | 392 | 15.7 ms |
| `widget_defaults_self.tcss` | 93,110 | 408 | 14.7 ms |
| `screen_css_scoped.tcss` | 15,216 | 111 | 4.3 ms |
| `screen_css_self.tcss` | 2,769 | 13 | 1.2 ms |
| **total** | **877,639** | **4,197** | **155.3 ms** |

In the running app, instrumenting `Stylesheet.parse` directly:

```
call 1: 191.3 ms      <- the whole bundle, once
call 2:   1.0 ms      <- _parse_cache hits
call 3:   0.9 ms
call 4:   1.0 ms
```

So it is **one 191 ms hit at boot**, not repeated work. Against a cold start
of ~1.7 s (module import measured separately at ~1.15 s, three cold runs:
1.20 / 1.08 / 1.15 s), CSS parse is **≈11–14% of cold boot**.

### Attribution, by ablation

`components/_agentic_terminal.tcss` cannot be parsed standalone — it
references variables the bundle defines earlier — so its cost was measured
by removing its segment from the bundle and re-parsing, three cold
interpreters per arm:

| bundle | rules | cold parse |
|---|---:|---:|
| full | 3,273 | 122.9 / 124.3 / 137.4 ms |
| without `_agentic_terminal.tcss` | 2,022 | 69.0 / 70.0 / 70.0 ms |
| **attributable** | **1,251 rules** | **≈54 ms** |

One file is **283,119 B (32% of boot CSS bytes), 1,251 rules (38% of the
bundle's rules), and ≈54 ms (≈28% of the boot CSS parse)**. It also supplied
+12,902 B of the +23,613 B growth that pushed the ratchet red.

### The measurement trap this produced

**In-app timing of CSS parse is unreliable and was wrong by 250×.** A fresh
`Stylesheet` inside a booted app re-parsed 671 KB in "0.48 ms" — 1.4 GB/s,
which is impossible for a Python tokenizer. Clearing the two module-level
caches I could find (`parse_selectors`, `is_id_selector`) did not change it.
The same content in a **cold interpreter** takes **121 ms** (5.5 MB/s, a
believable rate). The in-app number was reported to me by my own probe and
looked plausible enough to build on.

*Rule: parse/boot costs must be measured in a cold subprocess. If a rate
implies >100 MB/s of Python text processing, the measurement is wrong.*

The instrumented real `Stylesheet.parse` (191 ms) independently confirms the
cold number and refutes the in-app one.

---

## 2. Method notes / things that did NOT survive

* **A route-visiting loop that measured nothing.** A probe for the
  `_parse_cache` LRU(64) cliff wrapped `app.switch_screen(route)` in
  `except Exception: continue`; every route raised, no screen was visited,
  and it still printed a confident `FINAL sources: 14 / HEADROOM 46`.
  Discarded. The 08-29 review had already disproved this hypothesis
  (47 sources, 17 headroom, zero parse calls on warm switches).
  *A loop whose body can silently skip every iteration must assert that it
  did work before reporting a total.*

---

## 2. 93% of per-node CSS candidate work comes from rules that cannot match

### Mechanism (verified in Textual's source, not inferred)

`RuleSet._post_parse` indexes each rule under **its rightmost selector
only**:

```python
selector = selector_set.selectors[-1]      # textual/css/model.py
if selector_type == type_type:
    add_selector(selector.name)
```

So `#prompt-variables-actions Button:disabled` is filed under `Button`, and
becomes a candidate for **every Button in the app** — full selector matching
runs on each before it is rejected.

### Measured on ChatScreen (502 nodes, 4,380 rules)

| type key | rules | ancestor-scoped | live nodes | considerations | re-keyable |
|---|---:|---:|---:|---:|---:|
| `Button` | 188 | 180 | 110 | 20,680 | 19,800 |
| `Static` | 16 | 15 | 220 | 3,520 | 3,300 |
| `Input` | 38 | 34 | 6 | 228 | 204 |
| `Vertical` | 13 | 12 | 68 | 884 | 816 |
| others | — | — | — | 967 | 367 |
| **total** | | | | **26,279** | **24,487 (93.2%)** |

`Button` alone supplies **71% of all candidate work on the screen**.

### Price, by A/B rather than by inference

Candidate count is not automatically milliseconds, so it was A/B'd: remove
the 290 ancestor-scoped bare-type rules from the narrowing index and re-time
a real full-screen restyle (median of 7, after a warm-up):

| arm | `stylesheet.update` |
|---|---:|
| baseline | **101.1 ms** |
| ancestor-scoped bare-type rules ablated | **40.8 ms** |
| **delta** | **60.4 ms (60%)** |

The ablated arm renders wrong styles — it exists only to price the work, and
is not a proposed change. **60% is an upper bound**: re-keying moves a rule
to a narrower key rather than deleting it, so the intended widgets still
evaluate it. For rules scoped to one panel the realised saving should be
close to the bound; for broadly-scoped ones it will be less.

**Fix direction:** give the subject its own class. `#prompt-variables-actions
Button` → a `.prompt-variables-action` class on those buttons. Costs a class
attribute per widget and removes the rule from 100+ unrelated buttons'
candidate sets. This is a convention, and is worth a lint that fails new CSS
whose subject is a bare common type.

---

## 3. Leaving a screen costs more than building the one you asked for

Instrumented `Stylesheet.apply` across an ordinary Console → Library →
Console → Library navigation. Reproducible to the call across three runs
(332 / 389 / 1,577 / 384 every time):

| navigation | screen built | nodes | applies | apply ms | wall ms |
|---|---|---:|---:|---:|---:|
| → Library (1st) | LibraryScreen | 96 | 332 | 105.0 | 301.6 |
| → Console (1st) | ChatScreen | 207 | 389 | 79.9 | 160.1 |
| **→ Library (2nd)** | LibraryScreen | 96 | **1,577** | **540.0** | **1,003.2** |
| → Console (2nd) | ChatScreen | 207 | 384 | 76.4 | 103.4 |

CSS apply is **50–72% of switch wall time**. The 2nd Library visit costs
**4.7× the applies of the 1st** — deterministically.

Attributing each apply to the screen its node belongs to explains it:

```
library#2: total_applies=1577
    1124 applies under ChatScreen#4992     <- the screen being LEFT
     247 applies under LibraryScreen#8608  <- the screen being BUILT
```

**71% of the switch's style work is spent restyling the outgoing screen**, at
~5.4 applies per node on a 207-node ChatScreen. The first Library visit is
cheap only because the screen it left was the small splash.

Live instance counts also climb across the same navigation
(`ChatScreen: 1 → 2`, `LibraryScreen: 1 → 2`), consistent with the
fresh-screen-per-switch behaviour already filed as TASK-24452 — but the new
part is that a retained, no-longer-current instance is still absorbing style
applies.

Findings 2 and 3 compound: the outgoing-screen restyle is itself paying the
93% candidate overhead from finding 2.

### 3a. Root cause: the outgoing screen is resumed, then suspended

Tracing `Screen._on_screen_resume` / `_on_screen_suspend` with instance
identity across one Console → Library navigation:

```
RESUME  ChatScreen#7872      <- the screen being LEFT, resumed first
SUSPEND LibraryScreen#8560   <- an older RETAINED Library instance
RESUME  LibraryScreen#8624   <- the incoming screen
SUSPEND ChatScreen#7872      <- the outgoing screen, suspended right after
```

The outgoing screen is **resumed at the start of a navigation that is about
to replace it**, and suspended moments later. That resume runs
`_on_screen_resume -> dom.update_node_styles -> app.update_styles` over its
whole 207-node subtree.

Attributing the outgoing screen's 1,107 applies by call stack:

| applies | trigger |
|---:|---|
| 499 (45%) | `_on_screen_resume` → `update_node_styles` → `app.update_styles` |
| 306 (28%) | `widget.mount` → `_compose` (widgets mounted INTO the screen being left) |
| 116 (10%) | `widget.update_styles` → `update_node_styles` |
| 57 | `stylesheet.update_nodes` |
| 31 | `widget.mount` → `app._register` |

Every one of those is discarded when the screen is suspended and replaced.

Note the older retained `LibraryScreen#8560` still participating in the
lifecycle — retained instances are not inert.

### 3b. Two probe bugs found here, both worth the rule

* **A navigation helper that polled on node count returned before the
  switch.** `go()` waited "until the screen has >40 nodes", but the CURRENT
  screen already did, so it returned immediately and a later probe reported
  `INCOMING == OUTGOING`. Any probe that navigates must wait for the screen
  **identity** to change, and assert it did.
* **An attribution probe classified the incoming screen as "elsewhere"**,
  producing 334/1,377 and appearing to refute the 1,107 figure. Re-running
  with explicit OUTGOING / INCOMING / RETAINED roles confirmed the original
  1,107 (71.6%). *When two measurements disagree, the bug is usually in the
  newer one's bucketing, not in the phenomenon.*

---

## 4. Postscript: the ratchets moved again during this review

The baseline in §0 was taken at pin `0ef6f3fd4e`. Re-running the same two
guards after rebasing onto dev **21 commits later, the same day**:

| guard | at pin | +21 commits | limit |
|---|---:|---:|---:|
| boot CSS bytes | 878,333 | **879,439** | 860,000 (already breached) |
| `_ui_ready` census | 969 (headroom 3) | **970 (headroom 2)** | 972 |

The CSS breach deepened by 1,106 B and the `_ui_ready` census consumed a
third of its remaining headroom, in a single day's ordinary merge traffic
and without either guard being touched.

This is precisely the consumption pattern ADR-097 was written to stop, and
it is still running. It also sharpens TASK-25812: the CSS ratchet cannot be
brought back under its limit by a one-off trim if routine traffic adds
~1 KB/day to the same path — the fix has to move something structurally off
the pre-first-paint leg, not shave it.

---

## 5. Implemented: ancestor rejection in the CSS fast path

Finding §2 measured a 60% upper bound from re-keying ancestor-scoped rules
onto classes — invasive, touching ~180 CSS rules and the widget markup that
carries the classes. A cheaper route gets most of it with no CSS or markup
change at all.

`tldw_chatbook/Utils/textual_css_fastpath.py` already owns candidate
construction. It now rejects candidates whose leading compound names an
ancestor the node does not have, before upstream evaluates them.

Interleaved A/B, four pairs, filter toggled in place, median of five
full-screen `stylesheet.update` calls per arm, 502-node Console:

| arm | `update(screen)` | samples |
|---|---:|---|
| filter off | 105.0 ms | 103.5 / 104.0 / 105.0 / 108.3 |
| filter on | **66.2 ms** | 64.8 / 65.8 / 66.2 / 66.8 |
| **delta** | **−38.8 ms (−37%)** | ranges do not overlap |

Interleaved rather than blocked, so drift (GC, thermal, cache warmth) cannot
be attributed to whichever arm ran second.

### Why it is safe

A rule survives unless **every** one of its selector sets states a
requirement that is unmet. Anything not cheaply decidable reports "no
requirement" — including a leading TYPE selector, because matching a type
against an ancestor needs MRO walking, which is the cost being avoided.

### Both guards were mutation-tested, and the first one had a hole

* Deleting the one-compound guard — which would demand an *ancestor*
  `#thing` for `#thing.foo`, whose id is on the SUBJECT — fails the per-node
  fidelity tour across three real screens.
* **The new unit test did not initially catch that mutation.** Its
  `Button.foo` case begins with a TYPE selector, so it returns `None` with
  or without the guard. Shapes *led* by an id/class (`#thing.foo`,
  `.foo.bar`) were added; the unit test now fails on that mutation in 0.66 s
  instead of only via the 9 s full-app tour.

* A third guard covers the runtime case: ancestor names are recomputed on
  every apply, never cached, because ancestors gain and lose classes
  constantly (`-active`, `hidden`, ...). Caching them is the obvious
  "optimisation" here and would silently drop styles that have just become
  applicable. `test_filter_follows_a_class_added_to_an_ancestor_at_runtime`
  asserts both directions (a scoped rule becomes a candidate when an
  ancestor gains the class, and stops being one when it loses it); caching
  the set fails it, and the fidelity tour independently.

*A test that looks thorough can still be blind to the exact mutation it was
written for. Mutating it is the only way to find out.*

### Not superseded

CSS-level re-keying (TASK-25810) remains worth doing: the filter does not
help rules led by a TYPE, and re-keying additionally shrinks the candidate
*set* rather than building it and discarding. Remaining headroom should be
measured against the new **66.2 ms** baseline.

### 5a. Verification: a paired-arm run, because a failure list attributes nothing

The fidelity tour covers Chat, Library and Settings. A large share of this
repo's CSS-sensitive tests exercise Watchlists, Schedules, Personas and
workbench snapshots — surfaces that tour never visits — so it is strong
evidence *for the three screens it walks*, and not more.

The 82 CSS-sensitive UI test files (1,105 tests) were therefore run **twice**,
identically, ~33 minutes per arm: once with the filter installed, once with
the implementation reverted.

| arm | failed | passed |
|---|---:|---:|
| filter ON | 39 | 1,064 |
| filter OFF (baseline) | 40 | 1,063 |

| comparison | result |
|---|---|
| **broken by the filter** | **none** |
| pre-existing (fail in both arms) | 39 |
| passed with, failed without | 1 |

That last row is **not** a fix. `test_active_reveal_queue_retains_only_
identity_across_target_and_rail_removal` fails **4 out of 4** runs in
isolation *with the filter installed*; it passed in the ON arm through
ordering luck. Claiming the filter fixed it would have been the easiest
sentence to write and the wrong one.

Everything that worried me on the ON arm — `test_workbench_visual_snapshots`,
the eight `test_destination_visual_parity_correction` failures, the CSS
contract and build-integrity tests, the rail width budgets — fails
identically without the change.

*A list of failures in a suite you have never run clean says nothing about
your change. Two earlier long sweeps in this cycle produced exactly that and
had to be discarded; only the paired arms answered the question.*
