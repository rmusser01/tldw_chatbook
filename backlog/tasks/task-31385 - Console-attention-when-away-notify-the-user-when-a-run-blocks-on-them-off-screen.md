---
id: TASK-31385
title: >-
  Console attention when away: notify the user when a run blocks on them
  off-screen
status: Done
assignee:
  - '@claude'
created_date: '2026-09-04 19:29'
updated_date: '2026-09-05 01:25'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When an agent run blocks on the user -- an approval, a skill or worktree confirm, or an ask_user question -- the only attention affordances are inside Console: the rail badge, the parked-round toast, and the wake badge. A user on another screen (Library, Settings) or in another terminal window learns nothing until they come back, while the run sits paused for as long as ADR-067's indefinite default allows. Sub-project D of the design spec (2026-08-19-console-user-interaction-design.md section 4): a terminal bell and/or OSC notification, and a cross-screen badge on the Console nav item, raised when a round mounts or parks while Console is not the visible screen, and cleared when it resolves. Both reference implementations (Claude Code, Codex) ring or notify on a blocking prompt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A blocking round raised while Console is not the visible screen produces a terminal notification (bell or OSC 9/777) governed by a [console] setting that defaults on
- [x] #2 The Console entry in the app navigation shows a pending-interrupt badge until the round resolves
- [x] #3 A round raised while Console is visible produces no bell
- [x] #4 Headless and test runs never emit terminal control sequences
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Host: run_round notifies a late-bound seam on_pending_rounds_changed(total, kind, raised) after the mount/park step and after the teardown pop.
2. Controller: marshal to the UI thread; nav badge = app-level pending count applied to the Console button on the content screen's MainNavigationBar; bell via App.bell() only when raised, the Console view seams are detached, the [console] interrupt_bell setting (default on) is set, and the app is not headless.
3. MainNavigationBar.on_mount applies the app-level count so a screen composed after the round shows the badge.
4. Config default + user-guide line; tests for host hook, controller gate matrix, nav badge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** One hook on the interrupt host, two effects in the controller, one badge helper in the navigation module.

- `InterruptRoundHost.run_round` calls the late-bound seam `on_pending_rounds_changed(total, kind, raised)` after a round mounts or parks (`raised=True`) and after its registry entry is popped in teardown (`raised=False`); `pending_total()` counts every kind. A controller or test double without the seam hears nothing.
- `ConsoleChatController.on_pending_rounds_changed` marshals to the UI thread: `set_console_attention(app, total)` always; `App.bell()` only when the round is being raised, `[console] interrupt_bell` (default on) is set, no Console view is attached (`_approval_view_is_detached()` -- `detach_view` clears every slot together, so the approval pair stands for all five kinds; a Console never opened this launch counts as away), and the app is not headless (explicit `is_headless` gate; Textual's own `bell` also no-ops headless).
- `UI/Navigation/main_navigation.py`: `set_console_attention` stores the count on the app and repaints the `#nav-console` button on every screen in the stack; `MainNavigationBar.on_mount` applies it, so a bar composed after the round armed (each screen composes its own) shows the `◆` badge too. The badge clears when the total returns to zero.
- Bell only, no OSC 9/777: AC #1 allows either, the bell is one Textual call with no terminal detection.

**Files.** `tldw_chatbook/Chat/console_interrupt_rounds.py`, `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/UI/Navigation/main_navigation.py`, `tldw_chatbook/config.py` (default), `Docs/User_Guide/console/agent-runs-and-tools.md`; tests `Tests/Chat/test_console_interrupt_attention.py` (gate matrix + host seam), `Tests/UI/test_nav_console_attention.py` (badge on a mounted bar, badge on a bar mounted later, clears at zero).
<!-- SECTION:NOTES:END -->
