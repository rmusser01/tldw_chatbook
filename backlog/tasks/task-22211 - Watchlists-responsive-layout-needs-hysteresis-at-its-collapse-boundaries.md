---
id: TASK-22211
title: >-
  Watchlists responsive layout needs hysteresis at its collapse boundaries
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
labels:
  - performance
  - watchlists
  - ux
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22211).

New with PR #2063. `UI/Watchlists_Modules/region_layout.py:132-175`:
`resolve_effective_layout` applies bare width thresholds with no `previous` state, and
`on_resize` recomputes per Textual Resize event. Crossing 145 columns by ONE cell flips
the right rail: region factory + mount/remove pair per flip
(`watchlists_workbench.py:226-309`), repeated per Resize during a drag. This is the
documented sub-2-cell width-flap trap; the Library media reader carries the fix
(`LAYOUT_HYSTERESIS_WIDTH = 4`, `Library/library_media_reader_state.py:16`, `:341-355`)
and Watchlists does not. Aggravator (medium confidence): `_available_layout_width` prefers
`workbench.size.width` (`watchlists_collections_screen.py:2999`), which is
scrollbar-sensitive — a scrollbar toggle at the boundary could flap the layout with no
user resize.

## Acceptance Criteria

- [x] Repeated +/-1-cell width changes at a collapse boundary cause no mount/remove churn (hysteresis test at the boundary, both directions)
- [x] The width source is not flappable by a scrollbar toggle, or a code-level guard absorbs sub-hysteresis changes (the repo rule: never trust a CSS-only guard)
- [x] Approach consistent with the Library reader's hysteresis precedent

## Implementation Plan

1. Red-first pure-function tests on `resolve_effective_layout` (extend
   `Tests/Watchlists/test_watchlists_responsive_layout.py`): a new
   `previous: RegionLayout | None` keyword threads the previously resolved
   effective layout; +/-1-cell oscillation at each collapse boundary (read
   145/144, management 108/107, and the second read boundary 115/114) is
   stable in both directions; expansion requires clearing the boundary by
   `LAYOUT_HYSTERESIS_WIDTH = 4` (the Library reader precedent,
   `Library/library_media_reader_state.py:16`); two boundaries near each
   other compose per-region (RIGHT_RAIL stays collapsed while LEFT_RAIL's
   own expand boundary is evaluated with the freed width accounted);
   `previous=None` (first-ever resolve) is byte-identical to today's
   behavior; hysteresis result's collapsed set is always a superset of the
   bare resolve's (overflow safety: hysteresis can only suppress
   *expansion*, never suppress a collapse, so `required_width <= width`
   whenever the bare resolver achieves it — a pane is never held open at a
   width where it cannot fit).
2. Red-first screen-level probe (new
   `Tests/Watchlists/test_watchlists_layout_hysteresis_probe.py`): count
   RIGHT_RAIL factory invocations and region-body mount presence across a
   +/-1-cell `on_resize` oscillation at the management boundary with the
   width source stubbed; today each up-crossing re-mounts the pane (factory
   churn), after the fix the count stays at zero once collapsed.
3. Implement: `LAYOUT_HYSTERESIS_WIDTH = 4` and the `previous` parameter in
   `UI/Watchlists_Modules/region_layout.py` — after the existing greedy
   collapse loop, a pane that is open in the bare result but collapsed in
   `previous` is kept collapsed unless `required_width +
   LAYOUT_HYSTERESIS_WIDTH <= width`, iterating in the same priority order
   and re-deducting each suppressed pane's minimum width so adjacent
   boundaries compose. `article_focus` early-return unchanged.
4. Thread state in the screen: `_recompute_effective_layout` gains a
   `previous` parameter forwarded to the effective resolve only; `on_resize`
   passes `self._effective_region_layout`. All other call sites (mount,
   gestures, section switches, article focus, rollback) keep `previous=None`
   so manual gestures and first resolves behave exactly as today.
   Scrollbar-flap aggravator: the hysteresis IS the code-level guard — the
   default vertical scrollbar is 2 cells < 4, so a scrollbar-induced width
   delta at a boundary is absorbed on every resize-driven resolve (a
   scrollbar toggle alone posts no Resize to the Screen, so it cannot
   trigger a recompute by itself; the width it perturbs is only read at the
   next resolve, where the guard applies).
5. Verify persistence safety: only preferred layouts
   (`self.region_layout`) reach `_schedule_layout_persist`; the hysteresis
   only touches `_effective_region_layout`.
6. Targeted suites (region layout, workbench, screen, cold-open, grips,
   store) + `--collect-only` sweep, tee'd; `./scripts/preflight.sh`.
7. Mutation test: remove the previous-state threading in `on_resize` (and
   separately neuter the resolver's hysteresis block) → oscillation tests
   must go red; restore via Edit.

## Implementation Notes

Matched the Library reader precedent exactly in shape and value:
`LAYOUT_HYSTERESIS_WIDTH = 4` in `UI/Watchlists_Modules/region_layout.py`,
and `resolve_effective_layout` gained a keyword-only
`previous: RegionLayout | None = None`. After the existing greedy collapse
pass, panes still open but collapsed in `previous` are re-collapsed unless
`required_width + LAYOUT_HYSTERESIS_WIDTH <= width`, iterating in the same
priority order (priority target last) and deducting each suppressed pane's
minimum width before judging the next — so two nearby boundaries compose
per region, not through one global flag. Hysteresis therefore only ever
*suppresses expansion*: collapse still fires exactly at the bare
threshold, the resolved collapsed set is provably a superset of the bare
resolution's (parametrized sweep over widths 0–200 x 4 previous states x
both modes), and no pane can be held open at a width where it cannot fit —
no cap logic is needed because the overflow case is unreachable by
construction.

Threading: only `WatchlistsCollectionsScreen.on_resize` passes
`previous=self._effective_region_layout` (via a new `previous` parameter
on `_recompute_effective_layout`). First-ever resolves, manual gestures
(`_toggle_preferred_region`, Article Focus), section switches, and
rollback keep `previous=None` and byte-identical pre-change behavior —
this also keeps a manual expand instantly honored inside the hysteresis
band, and the resize path still applies hysteresis to the priority target
so the just-expanded pane cannot flap during a drag.

Scrollbar aggravator (AC#2, second disjunct): resolved with a code-level
guard, not a width-source swap. `workbench.size.width` in Textual 8.2.8 is
`content_region` (region minus border/padding, NOT minus own scrollbars),
but it shrinks by 2 when the Screen (default `overflow-y: auto`) grows a
vertical scrollbar. A scrollbar toggle posts Resize only to the resized
widget (no bubbling), so it cannot trigger a recompute by itself; when the
perturbed width IS next read, the resize path's hysteresis (4 > 2)
absorbs the delta. Documented in `_available_layout_width`'s docstring and
in the constant's comment (the 4 > scrollbar-width invariant).

Persistence audited: `_schedule_layout_persist`/`save_region_layout` only
ever receive *preferred* layouts (`_apply_layout`,
`_toggle_preferred_region`, rollback); hysteresis touches only the
transient `_effective_region_layout`, so saved preferences cannot be
corrupted. Teardown: `on_resize` during unmount resolves at the 10_000
sentinel where hysteresis is a no-op, unchanged from before.

Evidence: red-first tests (pure suite failed on the missing parameter and,
for the probe, on real churn — 5 full RIGHT_RAIL pane rebuilds across 5
+/-1-cell oscillation cycles at the Read boundary; 0 after). New tests: 10
pure hysteresis tests in `Tests/Watchlists/test_watchlists_responsive_layout.py`
(boundary stability both directions in both modes, expand-boundary
clearance and its own stability, >= hysteresis crossings still flip,
two-boundary composition, priority-target hysteresis, article focus,
fixed point, superset sweep, previous=None equivalence) + screen probe
`Tests/Watchlists/test_watchlists_layout_hysteresis_probe.py` (zero
region-body rebuilds during oscillation, re-expand at boundary+4 works).
Suites: `Tests/Watchlists/` 802 passed; the seven `Tests/UI/`
watchlists files 262 passed + 1 pre-existing timing flake
(`test_loader_results_landing_before_textual_flips_is_mounted_still_paint`,
a 3s poll-budget precondition unrelated to layout: fails/passes on the
same tree, passes at dev baseline and with this change on rerun, and no
Resize fires in its scenario). Mutation-tested both halves: dropping the
`on_resize` threading reds the probe (5 rebuilds return); neutering the
resolver block reds 5 pure tests. `./scripts/preflight.sh` all green.

Files: `tldw_chatbook/UI/Watchlists_Modules/region_layout.py`,
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py`,
`Tests/Watchlists/test_watchlists_responsive_layout.py`,
`Tests/Watchlists/test_watchlists_layout_hysteresis_probe.py`.
