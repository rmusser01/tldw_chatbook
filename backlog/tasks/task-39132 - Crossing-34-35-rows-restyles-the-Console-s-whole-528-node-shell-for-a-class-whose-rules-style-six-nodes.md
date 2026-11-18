---
id: TASK-39132
title: "Crossing 34/35 rows restyles the Console's whole 528-node shell for a class whose rules style six nodes"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: '2026-11-10 22:38'
labels:
- follow-up
- console
- performance
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/css/features/_console.tcss
- Tests/UI/test_console_projection_restyles.py
dependencies:
- TASK-38950
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. Line numbers are dev's. TASK-38950 (round-11 branch fu/r11-console) scopes the header breakpoint's restyle; this one is separate.

**The flip.**
- `ChatScreen._adapt_console_shell_to_height` (`chat_screen.py:21656-21657`) runs `shell.set_class(height < 35, "-console-compact")` (`CONSOLE_COMPACT_HEIGHT_ROWS = 35`, `:745`).
- In Textual 8 a class change restyles the node's whole subtree.
- The only rules keyed on the class (`css/features/_console.tcss:2323-2349`) style six nodes.

**Measured.** A probe in dev's `test_console_projection_restyles.py` harness at 160x40:

| Height change | Crosses 34/35? | Nodes restyled |
|---|---|---|
| 40 → 38 | no | 0 |
| 38 → 34 | yes | 528 |
| 34 → 30 | no | 0 |
| 30 → 36 | yes | 528 |

528 is every node in `#console-shell`, of 555 on the screen. That is the same shape as TASK-38950's breakpoint flip, paid on every resize across 34 rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A crossing in either direction restyles only the nodes the `-console-compact` rules style, plus any subtree their inherited properties need, counted in a test that is red on dev (528 of 528)
- [x] #2 Styles after each flip equal those of a full restyle
- [x] #3 Height changes that do not cross the line still restyle nothing, and readers of the class (for example `ChatApprovalCard`) still see it
<!-- AC:END -->

## Implementation Plan

1. Reuse TASK-38950's `apply_scoped_breakpoints` (`UI/Console_Modules/frame.py`) for the shell's `-console-compact` flip: it already takes any node; let a breakpoint with an empty class name mean "no class at or above this threshold", so the shell's table is `[(0, "-console-compact"), (CONSOLE_COMPACT_HEIGHT_ROWS, "")]`. `_adapt_console_shell_to_height` calls it in place of `set_class` (no new line in `chat_screen.py`).
2. In the same helper, restyle a target's subtree only when its rule sets a property descendants read from their ancestors. The compact rules set padding, margin and border, so only the six targets restyle. Move the ancestor-read property set from the test into `frame.py` as the one source the helper and the guards read.
3. Test in `Tests/UI/test_console_projection_restyles.py`: at 160x40, dispatch Resizes 40->38->34->30->36 to the screen and count restyles (crossings restyle exactly what the compact rules style; non-crossings nothing; the class toggles); compare styles with a forced full restyle after each. Red on base (528). Update the header-breakpoint test's expected set to the same subtree rule.

## Implementation Notes

Reused TASK-38950's `apply_scoped_breakpoints` (`UI/Console_Modules/frame.py`); no second mechanism.

- **Compact flip.** `_adapt_console_shell_to_height` now calls `apply_scoped_breakpoints(shell, [(0, "-console-compact"), (CONSOLE_COMPACT_HEIGHT_ROWS, "")], height)` in place of `shell.set_class(...)`. The helper already took any node; an empty class name now means "no class at or above this threshold". `chat_screen.py` stays at 24617 lines (two lines replaced two), so the size ratchet and `LIVE_RATCHET_ROW` are unchanged.
- **Subtree only where inheritance needs it.** The helper restyled each target's whole subtree. The compact rules set padding, margin and border; none is read by descendants, but `#console-native-transcript`'s subtree holds every transcript row. The helper now walks a target's subtree only when its rule sets a property in `ANCESTOR_READ_RULES` (the set the restyle guards already used, moved from `Tests/UI/test_console_projection_restyles.py` into `frame.py` as the one source; the test module re-exports it for `test_console_composer_state_restyle.py`). The header breakpoint rules (min-height, padding, display) shrink the same way, and `_breakpoint_styled` in the header test now applies the same subtree rule.
- **Test.** `test_a_compact_flip_restyles_only_what_its_rules_style` (160x40; Resizes 40->38->34->30->36 dispatched to the screen): a crossing restyles exactly the six nodes the compact rules style, a non-crossing restyles nothing, `has_class("-console-compact")` follows `height < 35`, and every node's rules equal a forced full restyle after each step. Red before the fix: `40->34: restyled 528 of 528 shell nodes; the compact rules style 6`. The file is a whole-file UI census entry.

**Runs (load 45-98).** Covering files (projection restyles, composer restyle guard, keyboard trust, shell regions, resize reflow, narrow layout, approval card, internals decomposition, compaction hold, provider persistence, screen size ratchet, review-selection boundary): 219 failed / 149 passed on head and the same 219 failed / 148 passed on base 8b762b52fa (RecoveryRequired in the plain files). The compact readers among them, `test_console_shell_drops_header_at_small_height` and `test_console_compact_height_keeps_header_controls_and_row_budget[size0,size1]`, passed on head under a temporary `@private_profile_test` marker (3 passed; marker removed). `test_height_resize_requests_one_coalesced_context_reconcile` fails under that marker with `2 == 1` reconcile runs both with and without this change (3 runs each), so it is pre-existing and in no lane.


**Review r12p-rev1 M5 (2026-11-08).** No test failed if the subtree branch broke. New: `test_a_breakpoint_rule_descendants_read_refreshes_their_styles`.
- It caches every shell descendant's `visual_style`, adds a breakpoint rule setting `text-style: italic` on `#console-shell`, flips the class with `apply_scoped_breakpoints`, and compares each descendant's style with a fresh computation.
- Green on head.
- Under the mutant that never fills `deep`: red, `507 descendants kept a stale style`.

**Red on dev after merge (2026-11-10).**
- What failed: `test_a_compact_flip_restyles_only_what_its_rules_style` failed with `40->34: restyled 28 of 540 shell nodes; the compact rules style 6`. It failed on the PR head c89193d850 and at every dev merge from #3484's (e78b62f0e7) through ae9f9e8d21, each run checked. Because that test is in the required check, every PR's required check was red. The original head's covering runs had failed the plain file under RecoveryRequired, which is why nobody saw it.
- Where the 22 extra restyles come from: the approval card, a reader of the class (AC #3). `ChatScreen` calls `_sync_control_layout` after the flip; it sets the card's own `approval-compact` class (which `_console_approvals.tcss` styles) and switches four action buttons to compact. Each of those class changes restyles its own subtree, 22 nodes in all.
- How AC #1 is read: it covers the shell's own flip. Readers' own class changes (AC #3) are counted separately.
- The fix, in the test only:
  - The expected set now also includes the subtree of each node other than the shell whose classes changed during the flip.
  - That set must be empty when the height does not cross the line.
  - It must lie inside the approval card, the one allowed reader; a new reader has to be added to the test on purpose.
  - The shell's own flip is never excused.
- Verification:
  - Green on the fix head: the file passes 18 of 18.
  - Plain `set_class` mutant: red, `restyled 540 of 540 shell nodes; the compact rules style 6 and readers' reactions 22`.
  - A mutant that also flips a class on the shell's largest child: red, `386 reaction nodes outside the approval card`. Without the card bound it restyled 409 of 540 nodes and passed.

Files: `tldw_chatbook/UI/Console_Modules/frame.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `Tests/UI/test_console_projection_restyles.py`, `backlog/docs/lessons-textual.md` (the 38950 entry's "same shape elsewhere" line).

**Hotfix after the round-12 merge (2026-11-10, base dev ae9f9e8d21).** PR #3484 merged without CI, and on dev `test_a_compact_flip_restyles_only_what_its_rules_style` failed every run: `40->34: restyled 28 of 540 shell nodes; the compact rules style 6`.
- **Cause.** Bisected over dev's five first-parent commits between the round-12 base b590a15afd and the restack base 3a6498ae14, with this task's two commits cherry-picked onto each: green on 7e4a7efa88, red on b498bc3aec (PR #3036, Console approval UX). The 22 extra nodes are the hidden `ChatApprovalCard`'s subtree. `_adapt_console_shell_to_height` calls each card's `_sync_control_layout`, which derives `approval-compact` from the shell's `-console-compact` and sets it with `set_class`, a whole-subtree restyle. Before #3036 the hidden card's width was 0, so `width < 55` kept it compact and the class never flipped; #3036's `stacked = width > 0 and ...` lets it follow the shell, so every crossing restyled 22 nodes of a card with no approval pending. The CSS split in the same window (`_console_approvals.tcss`) is not involved. The test was right and is unchanged.
- **Fix.** `_sync_control_layout` returns while the card is hidden (`not self.display`). Every path that shows it (`set_batch` sets `display = True`, then `_defer_decisions`' `ready()`) reflows it after mount. A visible card still flips at the crossing (`test_compact_approval_layout_journeys`).
- **Pin.** `_verify_height_only_resize_reflows_existing_controls` (`Tests/UI/test_console_approval_compact_layout.py`) now hides the card, crosses to 24 rows, shows a new round and asserts `approval-compact`. Several show-time paths reflow the card (`ready()`, `_restore_presentation`'s options sync, `on_resize`, the Select change handler), so removing any one leaves it green. With all four removed it is red on that assertion (`assert (True and False)`, card shown but not compact).
- **Evidence.** Red on dev before the fix (above). Green after: the restyle file, the compact layout file and `test_chat_approval_card.py` at `-n 4`, 55 passed (load 21). Approval and Console covering set (23 files) at `-n 4`: head 88 failed / 460 passed, dev 82 failed / 466 passed (load 21-37). The difference is flaky nodes red on dev too when re-run: `test_console_headless_approval.py` in 3 of 3 re-runs each side, and `test_reason_draft_survives_sync_and_is_cleared_for_a_new_round` 2 of 5 on head and 1 of 5 on dev. The rest are `RecoveryRequired` in plain files, on both sides. `test_timer_path_static_update_inventory.py` has the same 3 failures on dev. Preflight passes.
- Files: `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`, `Tests/UI/test_console_approval_compact_layout.py`, `backlog/docs/lessons-textual.md` (a sub-bullet under the 38950 entry).
