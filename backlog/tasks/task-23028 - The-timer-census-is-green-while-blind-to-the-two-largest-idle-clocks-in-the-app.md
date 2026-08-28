---
id: TASK-23028
title: >-
  The timer census is green while blind to the two largest idle clocks in the app
status: Done
assignee:
  - '@claude'
created_date: '2026-08-27'
labels:
  - testing
  - observability
  - performance
priority: high
---

## Description

`Tests/Architecture/test_timer_path_static_update_inventory.py` exists to find repeating clocks. It
is **green on current dev while missing both of the largest ones**, for two independent reasons:

1. **It matches `set_interval` as an exact callee name.** `UI/Console_Modules/realtime.py:345` now
   spells a **10 Hz** clock `self._set_interval(...)` through a constructor-injected callable, so it
   left the census silently. The root count did **not** move (35 -> 35) because another root arrived
   in the same window and the two cancelled - so nothing looked wrong.
2. **It parses only `tldw_chatbook/**.py`**, and no package file assigns `auto_refresh`. The 15 Hz
   `ProgressBar` clocks (TASK-23022) are armed inside `textual/dom.py` and are structurally
   invisible.

Two further known gaps remain from the prior review: two roots resolve to nothing and nothing
notices, and the call graph cannot cross constructor-injected callables at all - which is how the
whole `UI/Console_Modules` family is wired.

## Acceptance Criteria

- [x] A clock reached through a renamed or injected callable is either censused or **fails** the census loudly - silence is the defect
- [x] Framework-armed clocks (`auto_refresh`, indeterminate progress) are covered, or their absence is asserted explicitly rather than implied
- [x] An unresolvable root fails rather than being skipped
- [x] The census is verified against the two clocks it currently misses, as regression cases
- [x] Root-count stability is not treated as evidence of no change - this window had a net-zero count with two real changes underneath

## Implementation Plan

1. Widen the clock-root matcher from the exact callee `set_interval` to the wrapper family
   `^_?(create|set)_interval$`, so `self._set_interval(...)` (realtime.py:345, 10 Hz) and
   `self._create_interval(...)` (fleet.py:309, 1 Hz survivor tick) are censused.
2. Make unresolvable roots loud: every collected clock root must resolve at least one callback into
   the call graph, be a recognized pass-through wrapper whose exposed name itself matches the clock
   pattern (the `wiring.py` lambda shape), or carry a `CLASSIFIED_ROOTS` row -- otherwise a new
   `test_clock_roots_all_resolve_loudly` fails naming the site. Teach `_callback_names` to see
   through deferral shims (`call_later(cb)` inside a lambda), which resolves the
   `db_status_manager` root to its real callback.
3. Pin the root *set*, not the count: `EXPECTED_CLOCK_ROOTS` equality with directional failure
   messages, so a net-zero add/remove window can never look like "no change" again.
4. Fix the receiver-typing false positive without weakening the guard: an AST inference that
   auto-classifies a `.update(` receiver only when EVERY binding of that name (local scope, or
   `self.<attr>` across the class) is provably a dict/set constructor; anything unprovable still
   needs a `CLASSIFIED_SITES` row. Retire the prior NOT-A-WIDGET rows the inference now covers
   (they double as ground-truth validation).
5. Cover framework-armed clocks in a new, clearly-owned file
   (`Tests/Architecture/test_framework_armed_clock_inventory.py`, complementing -- not colliding
   with -- TASK-23022's in-flight instance fixes): census indeterminate `ProgressBar`
   constructions (`total` omitted / `total=None`), every `LoadingIndicator` construction, and
   `auto_refresh` armings (currently zero in package code -- asserted explicitly, not implied).
6. Mutation-prove all four detectors with fixture modules that reintroduce each blind spot and
   must go RED; pin the two originally-missed clocks as named regression asserts; triage every
   new finding the strengthened census raises on the live tree (real clocks reported, not fixed
   here; classification rows carry reviewed reasons).

## Evidence

Census run against three trees: 08-22 pin 35 roots / 3 unresolved; 08-24 pin 35 / 2; tip 35 / 2. The
tip diff is exactly two entries that cancel.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.

## Implementation Notes

All four defects fixed in `Tests/Architecture/test_timer_path_static_update_inventory.py` plus a
new, self-contained `Tests/Architecture/test_framework_armed_clock_inventory.py`. Verified against
the pristine base (`c4e52794e2`): its census sees 35 roots with realtime/fleet/db_status all
invisible and exactly two roots silently resolving to nothing (`wiring:callback`,
`db_status_manager:call_later`); the reworked census sees 36 distinct roots, 0 problems, 0
unclassified sites.

1. **Renamed/injected clock spellings** -- the root matcher is now the wrapper family
   `^_?(create|set)_interval$` (`CLOCK_CALLEE_RE`), which is the census's naming contract. Finds
   the 10 Hz `realtime.py` `self._set_interval(...)` clock AND a second previously-invisible one:
   the 1 Hz `fleet.py` `self._create_interval(...)` survivor tick. Both regression-pinned by name.
2. **Loud unresolvable roots** -- `_collect_clock_roots` returns `(roots, problems)`;
   `test_clock_roots_all_resolve_loudly` fails on any problem not in `CLASSIFIED_ROOTS` (empty on
   the current tree, deliberately). Pass-through wrappers (the wiring lambdas) are excused ONLY
   when exposed under a clock-family name; `_callback_names` resolves deferral shims
   (`lambda: app.call_later(cb)` -> `cb`), which recovered the db_status root. A param-shadowing
   guard stops a lambda parameter from spuriously resolving via the unique-global-method fallback
   (found live: a package method literally named `callback`).
3. **Root SET pinned** -- `EXPECTED_CLOCK_ROOTS` equality (36 entries) with directional messages;
   a net-zero add/remove window can no longer look like "no change".
4. **Receiver typing** -- `_receiver_is_provably_collection`: auto-classifies a `.update(`
   receiver only when EVERY binding in scope (local) or of `self.<attr>` (class-wide) is provably
   a dict/set constructor; subscript stores are mutations, not rebindings (Store-ctx check).
   Fixes the three-site dev red (console_chat_store `create_kwargs`, console_transcript
   `current_thinking_blocks` / `self._expanded_tool_output_ids`) and retired 4 of the 6 prior
   NOT-A-WIDGET rows as ground-truth validation. Can only err toward red; discrimination pinned by
   fixtures (query_one receiver, mixed binding, rebound self-attr).
5. **Framework-armed clocks** (new file, complements TASK-23022/PR #2156's
   `test_progress_widget_clock_guard.py` -- different file, construction-shaped census vs their
   mechanism guard): indeterminate `ProgressBar` constructions and every `LoadingIndicator`
   construction must carry a `FRAMEWORK_ARMED_CLOCK_ROWS` row; `auto_refresh` armings in package
   code are censused and currently asserted ZERO (absence explicit, not implied). 12 keys triaged:
   3 HIDDEN-WHILE-MOUNTED acknowledgements naming TASK-23022 as owner (Lab `install_progress`
   15 Hz, Personas CCP overlay 16 Hz, Console inspector next-send indicator 16 Hz -- the last one
   newly found by this census), 4 BOUNDED, 5 UNREACHABLE. Stale rows red with a retire-with-23022
   message -- the intended hand-off.

Mutation evidence (all run, all red): reverting `CLOCK_CALLEE_RE` to exact `set_interval` kills 3
tests; dropping the shim recursion kills 3 (incl. the live-tree loud-roots test); neutering
problem recording kills the unresolvable-root fixture; silently excusing unregistered wrappers
kills the wrapper-refusal fixture; making the receiver inference return True kills all 3
discrimination fixtures. Census wall time: pristine module 24.1-35.2s; reworked module + new
framework module together 32.6-39.2s (same parse, cached collection).
