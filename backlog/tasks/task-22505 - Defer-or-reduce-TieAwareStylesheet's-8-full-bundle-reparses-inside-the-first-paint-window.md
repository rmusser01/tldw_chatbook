---
id: TASK-22505
title: >-
  Defer or reduce TieAwareStylesheet's 8 full bundle reparses inside the first-paint window
status: In Progress
assignee:
  - rmusser01
created_date: '2026-08-26'
labels:
  - performance
  - startup
  - css
priority: medium
dependencies: []
---

## Description

Source: close-out of the 2026-08-24 holistic performance review's burn-down (29 tasks,
TASK-22200..22228, all merged 2026-08-25/26). Evidence: `Docs/Design/2026-08-24-holistic-perf-review.md` plus the originating task's
Implementation Notes.

Measured by TASK-22222's permanent counter: `TieAwareStylesheet`'s tie-breaker lowering arms
14 times and causes **8 actual full reparses** of the ~834 KB boot bundle, ALL inside the
first mount (none in the following second). A/B against a neutered arm: 20 vs 12
`Stylesheet.parse()` calls, identical fresh and warm. These are warm/parse-cached so far
below the module docstring's 125-380 ms cold price, but 8 full passes over the bundle during
first paint had never been quantified before and sit squarely in the TTI window this review
was commissioned over.

Related: boot-parsed CSS is now budgeted at 860,000 B (measured 833,841, already +20 KB
since the review) — the TASK-21115 ratchet counts SOURCES and pushes bytes into these eager
sheets by design; the two budgets price the same trade in different currencies.

## Acceptance Criteria

- [ ] The reparse count during first mount is reduced (batch the base-class offers, defer arming past first paint, or reparse once at the end) with the counter proving it
- [ ] The tie-breaker correctness the mechanism exists for is preserved: the TieAware staleness tests stay green and a computed-style spot-check matches before/after
- [ ] Boot-to-`_ui_ready` measured before/after with the interleaved A/A-controlled method (an honest wash is an acceptable outcome)


## Findings (2026-10-11 measurement study, dev `9f8533686c` + this branch)

Premise re-verified, then measured; the safe reduction lever is exhausted
at the current design, and the one available code change would reintroduce
the defect class TASK-21115 closed. Left In Progress for an owner decision.

- **Fresh counts (headless boot to `_ui_ready`, this branch)**: **16 arms
  -> 20 total `Stylesheet.parse()` calls** (vs TASK-22222's 14 -> 8 extra
  on 2026-08-25; the bundle has grown). Every arm lands AFTER the first
  parse (`self._rules` non-empty), so no arm is free.
- **Interleaved A/A TTI (3x each, separate processes, same profile,
  armed-vs-neutered alternating)**: armed median **11.074 s** vs neutered
  **10.742 s** -- the arming costs **~330 ms (~3%)** of boot-to-ready in
  this environment, i.e. ~40 ms per warm reparse over the ~834 KB bundle
  (consistent with the counter docstring's "well under the 125-380 ms
  cold figure"; Textual's parse cache absorbs the unchanged sources, the
  residue is the rules-list rebuild over 4,734 rules).
- **Why the obvious batchings do not reduce anything**: `_require_parse`
  is sticky, so arms already coalesce at every `apply()` boundary -- the
  16 -> 8 gap IS that coalescing. Flushing "once at the end of mount"
  only helps for arms landing between two applies, and each of the 8
  reparses precedes an `apply()` that must see the lowered tie-breakers:
  they are correctness-load-bearing during boot compose, not waste.
- **Why "defer arming past first paint" was rejected**: the arms exist so
  a consolidated subclass's rules are not shadowed by a base rule carrying
  a stale tie-breaker (the TASK-21115 measured failure shape). Deferring
  the rearm past `_ui_ready` reintroduces exactly that stale parse inside
  the boot window -- where the app composes its consolidated screens --
  and a one-shot post-ready reparse does not re-apply already-computed
  styles, so a boot-time tie would ship wrong geometry until an unrelated
  refresh. The only correct deferral needs an upstream-grade incremental
  re-parse (re-tie only the lowered sources' rules in place), which is an
  upstream/Textual change or a `reparse` override that mutates
  `_rules` per source -- a redesign, not a deferral.
- **Recommended next step (owner decision)**: either accept the ~330 ms
  as priced correctness (close as won't-fix-with-evidence), or file the
  incremental re-tie redesign (override `reparse` to rebuild only the
  lowered sources' `RuleSet`s and splice them into `_rules`, keeping
  `_rules_map = None`), which removes all 8 boot reparses at ~40 ms each
  without any deferral window.
