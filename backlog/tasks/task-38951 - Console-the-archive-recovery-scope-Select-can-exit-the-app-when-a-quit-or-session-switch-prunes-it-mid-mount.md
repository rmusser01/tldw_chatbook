---
id: TASK-38951
title: "Console: the archive-recovery scope Select can exit the app when a quit or session switch prunes it mid-mount"
status: To Do
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
- [ ] #1 A test prunes the archive-recovery card before its scope control's children mount and nothing raises; on dev the same test fails with NoMatches on SelectOverlay
- [ ] #2 When the card is not pruned, the scope control behaves as today: the existing cases in Tests/UI/test_console_archive_recovery_cards.py and Tests/UI/test_console_archive_recovery_flow.py pass (setting the value, the Changed event, Confirm gating)
- [ ] #3 The notes list every stock Select that a Console attach or sync always mounts, each marked prune-safe or with the reason it cannot be pruned mid-mount
<!-- AC:END -->
