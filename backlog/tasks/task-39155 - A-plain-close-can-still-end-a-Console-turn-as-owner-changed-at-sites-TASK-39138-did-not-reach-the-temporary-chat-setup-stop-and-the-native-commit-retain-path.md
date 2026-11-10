---
id: TASK-39155
title: "A plain close can still end a Console turn as owner changed at sites TASK-39138 did not reach: the temporary-chat setup stop and the native-commit retain path"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- console
- lifecycle
priority: low
references:
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_native_commit.py
- tldw_chatbook/Chat/console_pre_provider_stop.py
dependencies:
- TASK-39138
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39138 and its review, round-12 branch fu/r12-ctlr). Read on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08; line numbers re-read on dev 3a6498ae14, 2026-11-09. Found by reading; not reproduced, and no test covers it. Line numbers are dev 3a6498ae14's unless marked.

**The fixed site.** TASK-39138 fixed one site where a session the user closed in place read as an ownership change. It added `AcceptedStopAuthority.closed_in_place` (round 12, `console_pre_provider_stop.py`), and the stopped-stream settlement now returns the closed-session result ("Session closed.", run state `STOPPED`). Before, it returned `BLOCKED` / `OWNER_CHANGED` and logged "Console submit requires attention".

**What a close does.** `begin_session_close` (`console_chat_controller.py:14458`) calls `_signal_stop(session_id=..., reason="session_close")` (`:14569`, `:14689`). `_signal_stop` (`:14960`) sets the run's own cancel event and marks the session's ordinary native commits `explicit_stop` (`:15013`). A close in place also removes the session from the store.

**Sites with the same shape.**
- **Temporary chat, agent pre-provider setup** (`console_chat_controller.py:25463`; round-12 stack `:25484`). This branch handles a cancelled reply in `_replies_in_pre_provider_setup` whose recovery is `EPHEMERAL_ACCEPTED`. It checks `setup_authority.is_current(self)` and, when that fails, returns `ConsoleSubmitResult(True, False, terminal_status=BLOCKED, blocker=OWNER_CHANGED)` with no copy. A closed temporary chat fails `is_current`, because its session is gone from the store.
- **Native-commit retain.** `_controller_retain_original_native_commit_recovery` (`console_native_commit.py:387-427`) returns `BLOCKED` / `OWNER_CHANGED` with "Saved turn retained for recovery." or "Prepared turn owner changed; draft kept.". It is called right after `_controller_ordinary_native_commit_current` (`:374-384`) comes back False, and that check also fails when the session was closed. The call sites:
  - the controller (`console_chat_controller.py:12452`, `:12922`, `:13051`, `:13098`);
  - `console_native_commit.py:517`, `:802`, `:836`, `:848` and `:1020-1021`.

Whether a plain close reaches each window in practice was not traced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For each site, a test closes the chat inside that window (a temporary chat during agent pre-provider setup; an ordinary native commit in flight) and shows whether a plain close reaches it; a site no close can reach is named in the notes with the reason
- [ ] #2 Where a plain close reaches a site, the turn ends with the closed-session result TASK-39138 gives ("Session closed.", run state `STOPPED`, no "Console submit requires attention" log); the test is red before the change
- [ ] #3 A real ownership change (a replaced store, persistence or database, or a successor session with the same id) still gets the owner-changed result at each site
- [ ] #4 The tests run on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
