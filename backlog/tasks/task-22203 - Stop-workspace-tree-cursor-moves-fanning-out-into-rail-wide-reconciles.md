---
id: TASK-22203
title: Stop workspace-tree cursor moves fanning out into rail-wide reconciles
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25 16:07'
labels:
  - performance
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22203).

New with PR #2034. `console_workspace_tree.py:945-948`: every cursor move (arrow key,
click, sync) runs `_update_tooltip()` (per-move `get_node_at_line` + `cell_len` +
`scrollable_content_region`) and posts `WorkspaceTreeContextChanged`. The rail handler
(`UI/Console_Modules/left_rail.py:1992-2005`) calls
`sync_workspace_tree_context` (`Widgets/Console/console_workspace_context.py:1359-1434`),
which does an unguarded `context.update(copy)` — `Static.update` defaults to `layout=True`
in Textual 8.2.8, so one screen layout pass is armed per cursor keypress — plus an
unguarded tooltip assignment. When the cursor crosses a workspace<->conversation boundary
(constant while arrowing), the action-row `display` flip triggers `styles.height = "auto"`
+ `refresh(layout=True)` + two deferred frames ending in
`_reconcile_workspace_action_owners` (`:1448-1462`), which requests the full 7-section,
~45-`query_one` rail allocation pipeline (`left_rail.py:916-1035`). One arrow key = up to
2 extra frames + a full rail measure + >=2 layout passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An arrow-key move that does not cross the workspace/conversation boundary arms zero screen layout passes beyond the Tree's own repaint (probe on `Screen._refresh_layout`)
- [x] #2 Context tray updates are equality-guarded (content and tooltip) before any `Static.update`/tooltip write
- [x] #3 A boundary crossing performs at most one scoped reconcile, not the full rail allocation pipeline, or the full pipeline's per-press cost is measured and justified in the notes
- [x] #4 Per-press layout-pass count measured before/after
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red-first probes in a mounted-console harness (TASK-21117/22201 probe recipes): count Screen._refresh_layout calls, ConsoleLeftRail._run_allocation_reconcile invocations, and rail-side query_one calls per arrow-key press for (a) a conversation->conversation move inside one workspace (non-boundary) and (b) a workspace<->conversation move (boundary). Expect (a) >0 layout passes and (b) the full allocation pipeline today.\n2. Equality-guard the context tray's selection-copy update (_update_workspace_tree_selection_context): skip Static.update when context.content already equals the new copy; when it differs, pass layout=False (the slot is pinned to one nowrap/ellipsis row at compose time - hold that geometry to account in a test); equality-guard the tooltip write by comparing plain text before assignment.\n3. Memoize ConsoleWorkspaceTree._update_tooltip by (node key, tree content width, node label, expansion) so a call with an unchanged target and width skips the cell_len + region measurement and the tooltip write entirely (sync_projection's trailing call - the 5 Hz tick path - becomes a memo hit when nothing changed).\n4. Scope the boundary-crossing reconcile: _reconcile_workspace_action_owners stops requesting the rail-wide request_allocation_reconcile and instead requests a SCOPED bounded-section reconcile. Add ConsoleBoundedSection.request_scoped_reconcile() - same coalesced pass, but _ContextBoundedSection._run_scheduled_reconcile skips the demand-change escalation to the rail allocator for a purely-scoped pass (a plain request arriving while scheduled demotes it back to escalating, so genuine content changes keep full allocation correctness).\n5. Measure per-press layout passes + wall time before/after for both move classes (20 presses each); record numbers in the notes.\n6. Targeted suites: tree / left-rail / context-tray / action-row-geometry / rail-reconciliation + 22201 gate file + new probes; --collect-only sweep; ./scripts/preflight.sh; tee everything.\n7. Mutation tests: remove the equality guard -> non-boundary probe reds; remove the scoping -> boundary probe reds.\n8. Failure/teardown sweep: cursor move during screen teardown, tooltip on a node removed mid-move, boundary flip while an allocation reconcile is already in flight (the re-arm guard at left_rail.py:1015-1016).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cursor moves on the Console workspace Tree no longer arm screen layout
passes or the rail-wide allocation pipeline. Four changes:

1. **Equality-guarded, layout-free context tray writes**
   (`Widgets/Console/console_workspace_context.py`,
   `_update_workspace_tree_selection_context`): the selection copy repaints
   only when it changed, with `layout=False` (the slot is CSS-pinned to one
   `nowrap`/`ellipsis` row — the gate holds `context.region` to account),
   and the tooltip is written only when its plain text changed.
2. **Tooltip memo on the Tree**
   (`Widgets/Console/console_workspace_tree.py`, `_update_tooltip`):
   memoized by (node key, raw label, visible label, expansion, content
   width); unchanged target+width skips `get_node_at_line`-downstream
   measurement (`cell_len` + `_available_label_cells`) and the tooltip
   write. The memo key is a CLASS attribute default because
   `watch_cursor_line` fires inside `Tree.__init__` before subclass
   instance attributes exist (found by a red suite run).
3. **Scoped boundary reconcile**
   (`console_workspace_context._reconcile_workspace_action_owners` +
   `console_bounded_section.request_scoped_reconcile` +
   `left_rail._ContextBoundedSection._run_scheduled_reconcile`): the
   action-row display flip now requests a SCOPED bounded-section reconcile
   (same coalesced pass; the demand-change escalation to
   `request_allocation_reconcile` is skipped) and no longer requests the
   rail-wide allocation reconcile directly. A plain request coalescing into
   a pending scoped pass demotes it back to escalating, so state pushes,
   resizes, and section toggles keep full allocation correctness
   (gated by `test_projection_content_changes_still_reach_the_allocator`).
4. **Gates** in `Tests/UI/test_console_workspace_tree_cursor_layout.py`
   (11 tests): both press-class probes, the equality guards, the memo, the
   scoped/plain mechanism contract, teardown/removed-node/in-flight cases.

**Boundary-crossing decision (AC #3): scoped, not justified-full.** The flip
needs only the workspace section to re-fit its fixed chrome against its
existing allocation. Known bounded trade, documented in the code: while the
rail has spare space, the section keeps its current allocation until the
next genuine allocation trigger instead of growing by the flipped row. In
current production geometry the trade is actually ZERO: the workspaces tray
sits at its 12-row `max_height` cap (`overflow_y: hidden`), so the flip
cannot change section demand at all — measured `fixed=12` before/after the
flip at 160x44, 60x30, and a widened 90-col rail. The direct
`request_allocation_reconcile()` call was the entire live fan-out; the
scoped machinery future-proofs the uncapped case and is gated at the
mechanism level (stubbed `_measure_content_lines` seam) because no mounted
geometry can reach the demand-escalation leg — see the TASK-22203 entry in
`lessons-testing-evidence.md`.

**Measured, mounted console 160x44, 20 presses per class**
(`Tests/UI/test_console_workspace_tree_cursor_layout.py` printed metrics;
tees under the session scratchpad):

| metric (20 presses) | non-boundary before | after | boundary before | after |
| --- | --- | --- | --- | --- |
| `Screen._refresh_layout` | 20 (1/press) | **0** | 40 (2/press) | 40 (2/press, tray refit; tripwired at 3/press) |
| `_run_allocation_reconcile` | 0 | 0 | 20 (1/press) | **0** |
| `_prepare_allocation_reconcile` | 0 | 0 | 20 | **0** |
| rail `query_one` | 20 | 20 | 900 (45/press) | **20 (1/press)** |
| bounded-section reconciles | 0 | 0 | 160 (8/press, all 7 sections) | **20 (1/press, workspace only)** |
| wall (settle-dominated) | 4160 ms | 3712 ms | 5322 ms | 4838 ms |

Wall time is dominated by the probe's per-press `pilot.pause` settles; the
per-press counters are the honest measure.

**Mutation results**: (A) equality guard removed → 3 gates red
(non-boundary layout-pass gate + both guard gates); (B) scoped skip removed
→ escaped BOTH mounted boundary gates (12-row cap masks the demand leg —
the lesson above) → caught red by the mechanism gate
`test_scoped_reconcile_swallows_a_demand_delta_plain_still_escalates`;
(C) tooltip memo removed → memo gate red. All restored from the WIP commit.

**Tests**: new file 11 passed; adjacent suites all green against this
change: tree/bounded/context-tray/action-row/keyboard/recompose-guard 157
passed, rail reconciliation 56 passed, run-tick (22201 gate) + keystroke
(21118 gate) + left-rail + rail-sections + section-layout + tree-perf 92
passed, controller/lifecycle/details/edge-geometry/width-budget 116 passed.
Pre-existing dev reds (verified red on pristine `fce939e00` production
files, standalone): `test_console_workspace_dead_rows.py::test_failed_
resume_marks_row_broken_with_honest_single_toast`,
`test_console_new_workspace.py::test_console_new_workspace_creates_and_
activates` + `::test_console_new_workspace_announces_creation`
("Workspace section did not receive a usable allocation"), and
`test_console_rail_search_debounce.py::test_the_debounced_pass_still_does_
that_work` (`registry_calls == 0`). `--collect-only` sweep: 59,377
collected; 28 errors, all missing-`numpy` optional-dep modules
(Audio/TTS/Transcription/RAG), none under `Tests/UI`.
`./scripts/preflight.sh` all green.

**Files**: `tldw_chatbook/Widgets/Console/console_workspace_tree.py`,
`tldw_chatbook/Widgets/Console/console_workspace_context.py`,
`tldw_chatbook/Widgets/Console/console_bounded_section.py`,
`tldw_chatbook/UI/Console_Modules/left_rail.py`,
`Tests/UI/test_console_workspace_tree_cursor_layout.py` (new),
`backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
