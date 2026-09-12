---
id: TASK-15453
title: 'Console transcript: skip move_child for rows already in position'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - perf
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified first-hand: `_reconcile_rows` (`Widgets/Console/console_transcript.py:2314-2318`) calls `move_child` for every already-mounted row on every pass, unconditionally. Each `move_child` performs several O(rows) NodeList scans plus `refresh(layout=True)` plus a DOM-version bump that invalidates arrangement/query caches — even when the row is already in place. At ~2 rows per message, a 500-message conversation is ~1,000 rows, and the pass repeats on every 0.2 s streaming tick and on every transcript click (selection triggers a full reconcile). This predates the July task-259 work (a blind spot, not a regression — the content-signature diffing is intact and load-bearing).

Fix direction: track the expected index and skip the move when the widget is already in position; real order changes only occur via prune/variant/branch operations. Stability constraint: the reconciler carries subtle lifecycle guards (the closing/pruning abandon paths and the phantom-mount backstop at `:2306-2313`) — preserve them, and pin ordering behavior with tests covering prune, variant swap, and branch navigation before optimizing. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A steady-state reconcile pass (no order change) issues zero move_child calls (evidence)
- [x] #2 Ordering still correct after prune, variant swap, and branch navigation (tests)
- [x] #3 Reconcile pass time on a 500+-message transcript measured before/after and recorded
<!-- AC:END -->

## Implementation Plan

1. Pin ordering behavior first (TDD, tests must exist/extend before the fix):
   - `Tests/UI/test_console_native_transcript.py` (TASK-259 section, next to
     `test_transcript_signature_cache_survives_reorder`): add a
     `move_child`-call-counting spy; add a NEW test asserting a steady-state
     second `refresh_messages()` pass (same messages re-set, no order/content
     change) issues ZERO `move_child` calls -- this test is expected to be
     RED against current code (every unchanged row currently forces a move).
     Add a companion test asserting an actual reorder (permuted message
     list, same ids) still issues >0 `move_child` calls AND produces the
     correct final child order.
   - Extend `test_transcript_signature_cache_survives_variant_switch` with an
     explicit child-order assertion (variant switch must not disturb row
     position).
   - `Tests/UI/test_console_transcript_pruning.py`: extend
     `test_pruning_drops_oldest_rows_over_high_watermark` with an explicit
     full expected-order assertion (not just first/last presence).
   - Add a session-switch test (wholly different message id set) and a
     branch-navigation-style test (shared prefix, replaced suffix ids) with
     explicit final-order assertions.
2. Fix: in `_reconcile_rows`, when a row's widget was not (re)mounted this
   pass, compare its actual current index in `self.children` (read fresh
   each iteration -- never a stale snapshot) against the walk's expected
   index before calling `move_child`; skip the call when already correct.
   Preserve the `_closing/_pruning/is_attached` abandon check and the
   phantom-mount backstop verbatim, unmodified, in their current order.
3. Evidence: run the new tests (RED before the fix, GREEN after) plus the
   full console_transcript-touching UI/streaming test surface. Write an
   isolated timing probe (500+ row transcript, steady-state reconcile pass)
   comparing before/after wall time, record honest numbers in Implementation
   Notes.

## Implementation Notes

**Approach.** `_reconcile_rows` walks the desired row list with a running
`previous_widget`; for a row whose widget was not (re)mounted this pass, it
unconditionally called `move_child(before=0)` / `move_child(after=
previous_widget)` even when the widget was already exactly there. By
induction, every row processed so far (`0..index-1`) is already at its
correct slot in `self.children` (mounts land there directly via
`before=0`/`after=previous_widget`; moves, when needed, restore the
invariant) -- so row `index` is correctly positioned iff `self.children[index]
is widget`, checked fresh every iteration (never a cached snapshot, since
earlier mounts/removals in the same pass shift indices). `self.children`
is Textual's `NodeList`, whose `__getitem__` is a plain O(1) list index
(`_node_list.py:226`), so the check is cheap; `move_child` itself does two
`list.index()` scans plus `refresh(layout=True)` plus a NodeList
`_updates` bump that walks to the DOM root (`widget.py:1610-1677`,
`_node_list.py:71-78`), so skipping it when the row hasn't moved is the
entire win. The `_closing/_pruning/is_attached` abandon check and the
phantom-mount backstop are untouched, in their original order, ahead of
the changed block.

**TDD.** Added `test_reconcile_rows_steady_state_issues_zero_move_child_calls`
(`Tests/UI/test_console_native_transcript.py`) with a `move_child`-call
spy; confirmed RED against the pre-fix code (17 calls for an 8-message /
17-row steady-state pass), GREEN after the one-line guard. Added
companion tests pinning ordering across the scenarios named in the task:
a genuine reorder (`test_reconcile_rows_reorder_moves_widgets_and_lands_
correct_order`, still >0 `move_child` calls + correct final order), a
session switch (`test_reconcile_rows_session_switch_lands_correct_order`,
disjoint id sets), and a branch-navigation-style shared-prefix/swapped-
suffix swipe (`test_reconcile_rows_branch_navigation_replaces_suffix_in_
order`, modelling `ConsoleChatStore.set_active_leaf`'s active-path
recompute at the widget level, since the transcript only ever sees the
resulting message list). Extended the existing variant-switch test with
an explicit order assertion, and extended
`test_pruning_drops_oldest_rows_over_high_watermark`
(`Tests/UI/test_console_transcript_pruning.py`) from first/last presence
to a full expected-order assertion.

**Evidence (AC3, isolated probe, not committed -- scratchpad script).**
`ConsoleTranscript` seeded with 500 messages (1002 rows, matching the
audit's own "~2 rows/message" framing), steady-state `refresh_messages()`
(same message objects re-set, no order/content change -- the 0.2s
streaming-tick / transcript-click shape), 30 iterations, config-isolated
(scratch HOME/XDG/`TLDW_CONFIG_PATH`, mirroring `Tests/conftest.py`'s
bootstrap) and `sys.path`-pinned to this worktree (the venv's editable
install otherwise resolves `tldw_chatbook` to a different worktree per
the audit's "Environment note" -- confirmed hitting exactly that trap on
the first run). "Before" was captured by temporarily reverting only the
position-check guard (Edit-based, not git) and restoring it immediately
after:

| | mean | median | min | max |
|---|---|---|---|---|
| before (unconditional move_child) | 20.820 ms | 20.646 ms | 17.117 ms | 23.361 ms |
| after (skip when already positioned) | 1.974 ms | 1.900 ms | 1.762 ms | 2.532 ms |

~10.5x faster for a steady-state pass at 1002 rows; a smaller 8-row/522-row
run showed the same shape (6.138ms -> 0.954ms, ~6.4x) with a widening gap
as row count grows, consistent with `move_child`'s O(rows) cost per call
being cut to zero calls in steady state.

**Tests run (all targeted, cwd = this worktree, worktree's own venv;
pass counts read from output, not inferred).** Ran the full
console_transcript-touching UI/streaming surface named in the task
brief, in batches:
- `test_console_native_transcript.py` + `test_console_transcript_pruning.py`: 104 passed.
- `test_console_transcript_markdown.py` / `_markdown_widget` / `_tail_follow` / `_selection_contract` / `_jump_pill` / `_region` / `_diff_row`: 56 passed.
- `test_console_background_effects.py`, `test_console_citation_sources.py`, `test_console_composer_collapse.py`, `test_console_edit_resend_wiring.py`, `test_console_hands_free_wiring.py`: 153 passed, 1 failed.
- `test_console_realtime_wiring.py`, `test_console_regenerate_feedback.py`, `test_console_setup_lock_polish.py`, `test_console_stream_scrollback.py`, `test_console_tick_gating.py`, `test_console_workbench_contract.py`, `Tests/Widgets/test_console_video_card_rows.py`, `Tests/Chat/test_change_turn_tracking.py`, `Tests/Chat/test_console_generation_card.py`, `Tests/Chat/test_tool_output_disclosure.py`: 212 passed, 1 failed.
- `test_console_keyboard_trust.py`, `test_console_mcp_approval.py`, `test_console_message_controller.py`, `test_console_native_chat_flow.py`, `test_console_parallel_runs.py`: 416 passed, 1 xfailed.

Total: **941 passed**, 2 failed, 1 xfailed -- all 3 pre-existing and
unrelated to this change (verified, not assumed):
- `test_console_citation_sources.py::test_zero_only_count_cache_does_not_
  refresh_unchanged_transcript` fails at `chat_screen.py:15239`
  (`transcript.set_presentation_context(...)` called on a
  `SimpleNamespace` test double that never mocks that method) before
  ever reaching `_reconcile_rows`; `git diff --stat` against dev HEAD
  (`7cfe8df4e`) shows zero changes to either `chat_screen.py` or that
  test file on this branch.
- `test_console_workbench_contract.py::test_console_registers_footer_
  workbench_shortcuts` -- matches "footer-shortcut label" on this task's
  own known-pre-existing-dev-failures list.
- `test_console_native_chat_flow.py::test_console_browser_selecting_
  global_persisted_row_preserves_active_workspace` is a pre-existing
  tracked `xfail(strict=True)` citing task-15120, unrelated to row
  ordering/move_child.

**Files changed.**
- `tldw_chatbook/Widgets/Console/console_transcript.py` -- the
  position-check guard in `_reconcile_rows`.
- `Tests/UI/test_console_native_transcript.py` -- `_spy_move_child` /
  `_rendered_message_ids` helpers; 4 new tests; 1 extended test.
- `Tests/UI/test_console_transcript_pruning.py` -- 1 extended test
  (explicit order assertion).
