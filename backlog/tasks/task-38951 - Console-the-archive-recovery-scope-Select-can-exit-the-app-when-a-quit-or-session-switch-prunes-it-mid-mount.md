---
id: TASK-38951
title: "Console: the archive-recovery scope Select can exit the app when a quit or session switch prunes it mid-mount"
status: Done
assignee: ['@claude']
updated_date: '2026-11-06 06:50'
created_date: 2026-11-05 21:35
labels:
- follow-up
- console
- crash
- flaky
priority: medium
references:
- tldw_chatbook/Widgets/Chat_Widgets/archive_recovery_cards.py
- tldw_chatbook/UI/Console_Modules/archive_recovery.py
- tldw_chatbook/Widgets/prune_safe_select.py
- Tests/UI/test_console_archive_recovery_cards.py
- Tests/Performance/test_compact_setup_ownership.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-36511.2's notes record the crash. In round 4 of its AC#3 gate runs (`queued_body`, load 24.6), the child process exited when Textual's `Select._on_mount` raised `NoMatches: SelectOverlay` on `Select#recovery-scope`.

- Every Console sync that renders a new session runs `await surface.remove_children()` followed by `await surface.mount(ArchiveRecoveryConfirmCard())` (`tldw_chatbook/UI/Console_Modules/archive_recovery.py:152-161`, reached from `chat_screen.py:17897`). So every Console attach mounts the card, even with no confirmation pending.
- The card builds a stock `Select` (`tldw_chatbook/Widgets/Chat_Widgets/archive_recovery_cards.py:259-263`). Its `_on_mount` calls `query_one(SelectOverlay)` (textual 8.2.8, `widgets/_select.py:549`, `:626-629`). If the Select is pruned before its children mount, by a quit or by the next session switch's `remove_children`, that call raises and the app exits.
- The repo already has a workaround, `tldw_chatbook/Widgets/prune_safe_select.py` (`PruneSafeSelect`, TASK-1960), but only Watchlists and Speech use it. Its control test `test_stock_select_still_raises_with_no_children_at_all` (`Tests/Widgets/test_prune_safe_select.py`, 14 passed) shows the stock Select still raises in this Textual version.
- None of the open archive-recovery tasks (TASK-33020, TASK-35311, TASK-35585, TASK-38753) covers this. The other SelectOverlay tasks are about other screens.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test prunes the archive-recovery card before its scope control's children mount and nothing raises; on dev the same test fails with NoMatches on SelectOverlay
- [x] #2 When the card is not pruned, the scope control behaves as today: the existing cases in Tests/UI/test_console_archive_recovery_cards.py and Tests/UI/test_console_archive_recovery_flow.py pass (setting the value, the Changed event, Confirm gating)
- [x] #3 The notes list every stock Select that a Console attach or sync always mounts, each marked prune-safe or with the reason it cannot be pruned mid-mount
<!-- AC:END -->

## Implementation Plan

1. Reproduce the crash deterministically: a mounted test prunes `ArchiveRecoveryConfirmCard` as its scope `Select` composes (the window a quit or the next sync's `remove_children` hits), red on dev with `NoMatches` on `SelectOverlay`.
2. Build the scope control as `PruneSafeSelect` (TASK-1960), which only skips the child lookups while the widget is pruning or closing.
3. Run the existing card and flow tests unchanged (value, Changed, Confirm gating).
4. Census, from a mounted Console, every stock `Select` that attach or sync always mounts, and record each as prune-safe or why it cannot be pruned mid-mount.

## Implementation Notes

The scope control in `ArchiveRecoveryConfirmCard` is now a `PruneSafeSelect` (TASK-1960) instead of a stock `Select`. Every Console sync that renders a new chat runs `remove_children()` and then mounts a fresh card, so a quit or the next switch can prune the card before the scope control composes. Textual then skips the control's child mounts but still runs `Select._on_mount`, whose `query_one(SelectOverlay)` raised `NoMatches` and exited the app. `PruneSafeSelect` skips only those child lookups while the widget is `_pruning` or `_closing`. In every other state it is a stock `Select`: the card's `query(Select)`, `Select.NULL` and `Select.Changed` all still match it.

**Test.** `test_a_card_pruned_before_its_scope_control_composes_never_exits_the_app` is in `Tests/UI/test_console_archive_recovery_cards.py`, which runs in the UI PR gate. It wraps `Select.compose`, which `PruneSafeSelect` inherits, so that the card's `remove()` lands while the scope control composes. `remove()` sets `_pruning` on the whole subtree at once. That is the window the real prune hits. On dev (the swap reverted, i.e. e8eeae7db9) the test is red with `NoMatches: No nodes match 'SelectOverlay' on Select(id='recovery-scope')`, and `app.is_running` is False. After the fix it is green.

**AC#2.** `Tests/UI/test_console_archive_recovery_cards.py`, `Tests/UI/test_console_archive_recovery_flow.py` and `Tests/Widgets/test_prune_safe_select.py`, run `-n 4` together: 91 passed, 177.8 s, load ~30-50.

**AC#3: stock Selects a Console attach or sync always mounts.** I took this census from a mounted Console: the send harness at 235x52, after attach, after a whole sync, and after a new-tab switch. All three phases showed the same three `Select`s:
- `#recovery-scope` (`ArchiveRecoveryConfirmCard`). Every sync that renders a new chat remounts it, so it can be pruned mid-mount. It is now prune-safe.
- `#compact-api-provider` and `#compact-api-model` (`CompactModelBar` inside `ConsoleControlBar`, kept hidden as compatibility controls). These are composed with each ChatScreen's own compose; a sync never remounts the bar (`ConsoleControlBar` remounts only its action buttons). The `chat` route is `reusable` (TASK-31520), but that screen is not resident for the app's whole life, as the first version of these notes said. When the runtime identity flips (local <-> server), `app_navigation` drops the cached ChatScreen, uninstalls it and removes it, and the next Console attach composes a new one with new compact Selects. A quit can also land during a ChatScreen's first compose. Either is the same prune window the card had. So these two are now prune-safe too (review fold-in below).

The other stock `Select`s under `Widgets/Console`, `Widgets/Chat_Widgets` and `UI/Console_Modules` are mounted on demand, never by an attach or sync. They are in modals and pushed screens (the model popover, goal status and history, prompts, response rules, code review, terminal session, endpoint template, generate image, scope picker) and in per-request approval cards (`chat_approval_card.py`).

**Review fold-in (r10s-rev1).** `CompactModelBar` now builds `#compact-api-provider` and `#compact-api-model` as `PruneSafeSelect`. I added `test_a_bar_pruned_while_a_compact_select_composes_never_exits_the_app` to `Tests/UI/test_console_control_bar_recovery_height.py`, which runs in the UI PR gate. It is parametrized over both ids and mounts the production `ConsoleControlBar`. As each Select composes, it removes the bar, which sets `_pruning` on the whole subtree, as a ChatScreen's removal or a quit would. With the stock `Select` both cases were red with `NoMatches: No nodes match 'SelectOverlay' on Select(id='compact-api-…')`, and the app stopped running. After the swap both are green.

Covering run, 13 files that touch the compact Selects (`-n 4`):
- Head: 133 failed, 127 passed.
- Base worktree e8eeae7db9: 134 failed, 124 passed.
- Every head failure also fails at base. The one base-only failure is a `held_controller_body` parametrization. 128 of the head failures are the local `RecoveryRequired` trap (TASK-38111), 127 of them in `test_console_internals_decomposition.py`.
- Under a temporary `bootstrap_profile` marker, removed afterwards: `test_console_provider_persistence_regressions.py` passed 7/7, which includes the compact provider-mirror and no-arbitrary-first-provider cases.
- `test_console_internals_decomposition.py` under the same marker: 10 failed at head and 9 with the swap reverted. The extra failure is `test_console_composer_active_run_blocker_keeps_generic_placeholder`. It flakes at both: 3 of 5 failed at head and 2 of 5 with the swap reverted (a Send tooltip race, "Add API key…" vs "Wait for the active Console run…").

**Boot budget.** `Widgets.prune_safe_select` is now resident at `_ui_ready`, measured 1037 against the old limit of 1037. Base measured 1036. The bootbudget step measured 1038 when `DB.character_conversation_search` also loaded before ready, which is the documented +/-1 wobble. Under the owner size waiver (2026-10-31), `MAX_TLDW_MODULES_AT_UI_READY` goes 1037 -> 1038, with a dated comment and an ADR-097 ledger row. I refreshed `Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt` with `scripts/update_boot_budget_snapshots.py --only ui-ready` through run.sh; it gains only `prune_safe_select`. Re-run, bootbudget.sh reports 32 passed.

Files: `tldw_chatbook/Widgets/Chat_Widgets/archive_recovery_cards.py`, `tldw_chatbook/Widgets/compact_model_bar.py`, `Tests/UI/test_console_archive_recovery_cards.py`, `Tests/UI/test_console_control_bar_recovery_height.py`, `Tests/Performance/test_ui_ready_module_census.py`, `Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt`, `backlog/decisions/097-boot-budget-ratchets.md`. No user-visible copy changes, so no User Guide change.
