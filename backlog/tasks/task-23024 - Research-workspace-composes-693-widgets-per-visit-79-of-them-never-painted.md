---
id: TASK-23024
title: >-
  Research workspace composes 693 widgets per visit, 79% of them never painted
status: Done
assignee:
  - '@claude'
created_date: '2026-08-27'
labels:
  - performance
  - ui
  - screens
priority: high
---

## Description

`ResearchWorkspaceScreen` (new 2026-08-24, reached by F10) is now the most expensive screen in the
app: **693 widgets constructed per visit**, 4.4x Library and 1.3x Console. **544 of the 691 mounted
(79%) sit inside `display=False` subtrees** - constructed, mounted and CSS-matched on every visit,
never painted. One whole-screen recompose: **1.73 s**.

Cause is eager slot pools: `Research_Workspace_Modules/source_list.py:196` composes
`MAX_VISIBLE_SOURCE_ROWS = 25` slots, each yielding ~13 widgets (`:41-88`), **on an empty profile** -
325 widgets before any data exists - plus 20 receipt slots (`source_receipt.py:91`).

The slot-pool pattern is defensible; allocating the full pool at compose is what makes the empty case
pay the maximum case, on every visit. Screens are never cached, so this is paid every time.

## Acceptance Criteria

- [x] Widgets composed on an empty profile scale with content, not with the maximum
- [x] Widget count per visit and recompose wall time measured before and after, interleaved
- [x] Scrolling and row recycling still work at the maximum row count - the pool exists to avoid mount/unmount churn and that benefit must survive
- [x] The `display=False` proportion is reported after the change

## Implementation Plan

1. Verify the finding's numbers on base c4e52794e2 with a config-isolated probe
   (real `ResearchWorkspaceScreen` under the bundled production stylesheet):
   widgets constructed per visit, mounted total, display=False proportion,
   whole-screen recompose wall time.
2. Convert both eager slot pools to demand-grown pools with a zero floor:
   - `source_list.py`: `ResearchSourceList` composes no slots; `sync_page`
     grows the pool to `min(len(rows), MAX_VISIBLE_SOURCE_ROWS)` and never
     shrinks it (recycling above the floor is unchanged — surplus slots go
     `display=False`).
   - `source_receipt.py`: same shape; receipt slots mount before the bound
     disclosure `Static` so child order is preserved.
   - Slots keep direct references to their child widgets (built at
     construction) so `sync_source`/`sync_operation` stay synchronous even
     while the slot's subtree is still mounting; this also deletes 13
     `query_one` calls per row sync.
3. Update the three existing UI test files where they pin the eager pool
   (the `== 25` inventory assertion) or query slot internals in the same
   message-loop turn as the first growth (add a `pilot.pause()` — a frame has
   to paint before a user could interact anyway).
4. New regression tests, each proven to fail against a deliberately broken
   implementation (eager pool restored; growth capped below demand; receipts
   mounted after the bound Static; pool shrunk on smaller pages): empty case
   composes zero slots; growth tracks demand exactly; slot identity is stable
   across page swaps at the max (no mount/unmount churn); shrink recycles via
   display=False; DOM order matches index order; synchronous
   `visible_owner_ids`/`selected_source_ids` contract; unmount/quit walk
   mid-growth.
5. Interleaved A/B measurement (base modules injected from the pinned commit
   vs the fixed tree, alternating arms + A/A control), churn counts during
   page swaps at 25 rows before vs after, and the display=False proportion
   after the change.
6. Preflight, targeted suites, task bookkeeping, commit.

## Evidence

693 constructed / 691 mounted per visit, identical on every lap (screens are never cached).
Composition measured: 265 Buttons, 201 Statics, 91 Horizontals. Recompose 1735/1731/1793 ms over 3
trials.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.

## Implementation Notes

Both eager slot pools now start empty and grow on demand:
`ResearchSourceList` composes zero slots and `sync_page` grows the pool to
`min(len(rows), MAX_VISIBLE_SOURCE_ROWS)`; `ResearchSourceReceiptList` does
the same up to `MAX_VISIBLE_SOURCE_RECEIPTS`, mounting new slots before the
bounded-result disclosure so child order stays heading, receipts…, bound.
Pools never shrink — surplus slots recycle via `display = False` exactly as
before, so paging at the maximum still mounts/unmounts nothing. Slots keep
direct references to their child widgets (built at construction), which
keeps `sync_source`/`sync_operation` synchronous while a fresh slot's
subtree is still mounting, preserves the region's same-turn
`selected_source_ids()`/`visible_owner_ids()` contract, and deletes 13
`query_one` calls per row sync.

Measured on the real screen under the production bundled stylesheet
(config-isolated subprocess probe; base arm = the two pinned-commit modules
injected into `sys.modules`, arms interleaved in both orders plus an A/A
control). Deterministic axes, byte-stable across every run:

- Widgets constructed per visit (empty profile): **694 → 199 (−71%)**;
  mounted under the screen 691 → 196.
- `display=False` subtree share on an empty visit: **578/691 (83.6%) →
  83/196 (42.3%)**; at 25 rows + 20 receipts both builds converge to the
  identical 691-widget DOM (43/691 = 6.2% hidden).
- Churn during 10 page+receipt swaps at the maximum with scrolling:
  0 constructions, 0 unregistrations, widget identity set unchanged — in
  BOTH arms.

Whole-screen recompose (wall time; A/A noise floor on this machine was
1166–2154 ms across consecutive identical base runs, so the ratio is the
honest number): base 986–2154 ms vs fixed 241–667 ms in the quiet windows,
3275–4709 ms vs 830–1274 ms in a loaded window — **~3–4x faster in every
interleaved pairing, both orders**. The one-time first fill (0→25 rows +
20 receipts) pays the growth the eager pool used to pay per visit
(~380–435 ms call vs ~93–161 ms, quiet window); steady-state swap syncs
overlap between arms (14–36 ms means).

Every new test in `Tests/UI/test_research_slot_pool_growth.py` was proven
to fail against at least one deliberately broken implementation (eager
pools restored; growth capped below demand; receipts mounted after the
bound Static; shrink unmounting the pool; slot state dropped for unmounted
slots; the naive query_one-based sync, which crashes mid-growth). Existing
tests updated only where they pinned the eager pool (`== 25`) or queried a
grown slot's children in the same message-loop turn as the first growth.

Modified: `tldw_chatbook/UI/Research_Workspace_Modules/source_list.py`,
`source_receipt.py`; `Tests/UI/test_research_sources_region.py`,
`test_research_source_receipt.py`; `Docs/User_Guide/research_workspace.md`
(stamp). Added: `Tests/UI/test_research_slot_pool_growth.py` (8 tests).
47/47 research-UI tests pass; preflight green.
