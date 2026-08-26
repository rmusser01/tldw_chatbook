---
id: TASK-22202
title: Equality-guard ConsoleWorkspaceTree.sync_projection and scope its invalidation
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-26 04:38'
labels:
  - performance
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22202).

`Widgets/Console/console_workspace_tree.py:313-467` (new in PR #2034): `sync_projection`
is called on every workspace-context push with no projection-level equality guard. Each
call rebuilds a `WorkspaceTreeNodeData` and a rich `Text` label for every workspace and
conversation, materializes the full node set for `can_focus` (`:443-450`), and calls
`get_node_at_line` twice plus `_update_tooltip()`. Any node change calls Textual's
`Tree._invalidate()`, whose `self._updates += 1` is part of the line-render cache key — one
changed node invalidates every cached tree line. During a run this fires at 5 Hz (the
projection embeds `selected`/`run_marker`/`loading`, so it genuinely changes).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An unchanged projection results in zero node writes and zero tree invalidations (counted by a probe on `Tree._invalidate`)
- [x] #2 A single-conversation change invalidates a bounded set of nodes, not the whole line cache, or the whole-cache cost is measured and accepted in the notes with numbers
- [x] #3 `can_focus` derivation does not materialize the full node set per pass
- [x] #4 Per-tick tree cost during streaming measured before/after
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red-first probes: (a) count Tree._invalidate + node-data/label allocations + wall time for a value-equal (unchanged) projection push at 50/200 rows; (b) single-conversation marker change: count Tree._invalidate and per-node _updates to establish whether invalidation is already bounded (set_label is node-scoped in Textual 8.2.8; only structural add/remove/expand/move call Tree._invalidate); (c) measure a structural full-invalidation repaint at 50/200 rows for the accepted-with-numbers leg.
2. Implement a projection-level equality fast path in ConsoleWorkspaceTree.sync_projection: memo (projection tuple, frozenset(expanded), search_active) stored after a completed full pass, cleared at pass start (fail-safe), on expand/collapse events, on search-state flips, and on unmount; is-compare first (identity-stable within one 22201 tick scope), == value-compare across ticks.
3. Replace can_focus derivation's full tuple materialization with a short-circuiting itertools.chain generator.
4. Permanent gates: unchanged push performs zero node-data/label builds, zero get_node_at_line, zero _update_tooltip work, zero invalidations; gesture-collapsed workspace is still re-expanded by a stale-expanded push (memo does not freeze); marker-change path still writes exactly one node.
5. Measure after: per-push wall unchanged/marker cases; document bounded-vs-accepted decision with numbers.
6. Targeted suites (tree, cursor-layout 22203 gate, run-tick 22201 gate, rail reconciliation, bounded section, performance) + collect-only sweep, tee everything; preflight; mutation tests (break fast path -> probe reds; break changed-node path -> marker test reds); teardown walk (push after unmount, stale-tree push).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a projection-level equality fast path to `ConsoleWorkspaceTree.sync_projection`, so the 5 Hz run-tick push costs nothing when nothing changed, and replaced the `can_focus` full-node-set materialization with a short-circuiting `itertools.chain`.

## The memo

`_projection_memo` holds `(projection, frozenset(expanded_ids), search_active)` from the last COMPLETED pass. A push matching all three returns before any per-row work — no `WorkspaceTreeNodeData`, no rich `Text` label, no `get_node_at_line`, no `_update_tooltip`, no node writes. Identity is compared first (TASK-22201's tick scope serves ONE shared state object to every leg within a tick, so `is` hits there) then value (fresh but equal tuples of frozen dataclasses across ticks).

Correctness comes from clearing the memo at every seam that changes what a pass would do with identical inputs: at the START of a real pass (a raising pass leaves it empty, so a half-applied tree is never skippable), on `on_tree_node_expanded`/`on_tree_node_collapsed` (a user disclosure gesture must not be frozen — the pre-change reconcile re-applied `expanded_workspace_ids` on every push, and that behavior is preserved), on both `set_search_active` transitions, and on `on_unmount`. The memo is only STORED after the pass fully completes.

## AC #2 — the review's premise is wrong; corrected with evidence

Finding 22202 states "one changed node blows EVERY cached tree line." That is **false for the case the tick actually produces**. Textual 8.2.8 keys `Tree._line_cache` on `(y, is_hover, width, self._updates, pseudo_classes, tuple(node._updates for node in line.path))` (`_tree.py:1325-1333`). `TreeNode.set_label` bumps only the NODE's `_updates` (`:348-354`); only structural edits (add/remove/move/expand/collapse) reach `Tree._invalidate` and the tree-wide `self._updates`.

Measured (200 alternating single-conversation run-marker toggles, mounted harness): **`Tree._invalidate` 0, tree `_updates` delta 0, exactly one node's `_updates` moving** — at both 50 and 200 rows. So marker/selection/title updates were ALREADY bounded and needed no change; this is now frozen as `test_single_conversation_change_invalidates_no_tree_wide_cache`.

For the genuinely structural case (a conversation reordering), invalidation IS tree-wide and cannot be bounded without upstream Textual changes — **accepted with numbers**: the line cache is viewport-bounded, not row-bounded, because Textual only renders visible lines. Post-invalidation cold repaint measured 0.348 ms (50 rows) and 0.351 ms (200 rows) vs 0.188/0.189 ms warm at a 34-row viewport — i.e. ~0.16 ms extra, and **identical at 4× the row count**, which is the proof it is O(viewport), not O(rows). At the 0.2 s tick that is under 0.1% of a frame budget. Not worth a private-API assault on Textual's cache key.

## Measured (mounted Textual harness, 200 iterations, Python 3.12.11 / Textual 8.2.8)

Unchanged push (fresh value-equal tuple — the real tick shape):
* 50 rows: **0.154 → 0.006 ms** median (p95 0.192 → 0.007); per push 60 → **0** node-data builds, 60 → **0** label builds, 1 → **0** `get_node_at_line`, 1 → **0** `_update_tooltip`
* 200 rows: **0.517 → 0.020 ms** median (p95 0.589 → 0.025); per push 210 → **0**, 210 → **0**, 1 → **0**, 1 → **0**

Identity push (the within-tick shared build): 0.151 → **0.000** ms (50 rows), 0.500 → **0.000** ms (200 rows).

Changed-projection pushes are unaffected by design (marker toggle 0.162/0.524 ms before, 0.162/0.531 ms after) — the fast path adds one tuple compare to a pass that was already doing O(rows) work.

## Verification

* `Tests/UI/test_console_workspace_tree.py` 54 passed (47 existing + 7 new)
* Both prior gate files green and unchanged: `test_console_workspace_tree_cursor_layout.py` (22203) + `test_console_run_tick_workspace_reads.py` (22201) — 20 passed
* `test_console_workspace_tree_performance.py`, `test_console_bounded_section.py`, `test_console_rail_reconciliation.py`, `test_console_workspace_context_rail.py` — 152 passed
* Collect-only sweep: 59,603 collected; the 28 collection errors are absent optional deps in this venv (numpy ×21, playwright ×3, plus cascading ImportErrors from those same modules) — none touch Console code, all pre-existing
* `./scripts/preflight.sh`: all derived-artifact checks passed
* Mutation-tested: disabling the fast path reds exactly the AC1 zero-work gate (1 failed); making it skip unconditionally reds 50 of 54; breaking the changed-node label write reds the marker-update test plus the post-skip reconcile gate (5 failed); removing the collapse memo-clear reds exactly the disclosure-gesture gate (1 failed)
* Teardown walk: a push racing `remove()`, a late push to an unmounted tree, and a stale tree object receiving the live tree's projection are all safe (stale tree rebuilds its own detached maps; live tree untouched)

## Files

* `tldw_chatbook/Widgets/Console/console_workspace_tree.py` — memo + clears, `can_focus` chain
* `Tests/UI/test_console_workspace_tree.py` — 7 new gates
<!-- SECTION:NOTES:END -->
