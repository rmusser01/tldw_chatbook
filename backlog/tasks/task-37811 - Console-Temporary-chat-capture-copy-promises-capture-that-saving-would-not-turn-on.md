---
id: TASK-37811
title: >-
  Console: Temporary-chat capture copy promises capture that saving would not
  turn on
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-29 17:30'
updated_date: '2026-10-30 04:56'
labels:
  - console
  - copy
dependencies: []
references:
  - >-
    backlog/tasks/task-33621.17 -
    Console-Temporary-chats-stop-every-send-on-an-error-framed-capture-gate-and-its-default-Save-Send-never-sends.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33621.17 gives a Temporary chat Capture Off until the chat is saved (owner decision 2026-10-27). Saving hands Capture back to whatever the chat would otherwise resolve to. Three pieces of copy say that saving turns capture on:
- the composer menu row 'Save chat (enables capture)';
- the menu hint 'records no traces until saved';
- the User Guide line that saving turns capture on.

That is only true when the saved chat resolves Capture On. With global capture Off, or with a stored conversation Capture Off, saving changes nothing and the copy promises traces that never come. The wording was owner-specified, so the owner decides whether the suffix should depend on what the chat will resolve to once saved.

**Owner decision (2026-10-29).** The menu row, its hint and the User Guide line must say what will actually happen. Show '(enables capture)' only when saving really turns capture on (global capture On and no stored conversation Capture Off); otherwise the row reads just 'Save chat'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's decision on the conditional wording is recorded on this task
- [x] #2 The menu row, its hint and the User Guide match what saving resolves to: global capture On, global capture Off, and a stored conversation Capture Off
- [x] #3 A test pins the menu copy for each of the three cases
- [x] #4 No other Temporary-chat copy (the Trace screen's Exchange Capture Policy temporary guidance, the chip tooltip constant, the tab-strip User Guide line) says that saving turns capture on when it would not
- [x] #5 A one-time Next send Capture choice that differs from what the saved chat resolves to is named in the Save chat hint, so the row never says traces stay off before a captured send or promises capture before an uncaptured one; a test pins both cases
- [x] #6 The Trace screen's Exchange Capture Policy temporary guidance says Capture On for This conversation pauses each send and On for Next send pauses only that send; a test pins it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure on this head: in a mounted Console, for a Temporary chat with (a) global capture On, (b) a stored conversation Capture Off, (c) global capture Off, read the menu row + hint and the effective capture after Save chat.
2. Tests first: one mounted test (stock ChatScreen/controller/store, real config write for global Off, real Trace Privacy call for conversation Off) that reads the menu modal's Save row and hint and promotes, asserting the copy matches the post-save capture for all three cases; unit pins for build_composer_menu_entries and the resolution helper. Red on the pre-fix code.
3. Implement: a pure helper resolving the Capture a saved Temporary chat follows (conversation scope over global, the controller's own resolution); composer_menu_state feeds it to the menu; the row reads 'Save chat (enables capture)' + 'records no traces until saved' only when it resolves On, else 'Save chat' + a hint that traces stay off.
4. Copy that cannot say what saving does (Trace Privacy temporary guidance, chip tooltip constant): reword to what is always true.
5. User Guide: semantic-trace-capture.md, context-and-rag.md, sessions-tabs-workspaces.md.
6. Rebase onto newest dev, re-run gates, live-verify the three cases in the real app.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner decision 2026-10-29 implemented: the composer Menu's Save row says '(enables capture)' only when saving a Temporary chat really turns capture on.

**Measured first (pre-fix head, mounted Console, real store/controller/config).** Global capture On: row 'Save chat (enables capture)', and the saved chat resolved Capture On. Stored conversation Capture Off: same row, saved chat Capture Off. Global capture Off: same row, saved chat Capture Off. Global Off with the chat's own Capture On: same row, saved chat Capture On. So the copy was wrong in the two Off cases only.

**Approach.** `console_exchange_capture.saved_chat_capture(conversation, global_enabled=...)` returns the Capture a saved chat follows: its stored choice over the global setting, which is the controller's own resolution once the chat is no longer temporary. The one-shot next-send override is left out because it is not the chat's setting. `composer_run_controls.composer_menu_state` reads it from `capture_policy_snapshot` (only in a Temporary chat) and passes `save_enables_capture` to `build_composer_menu_entries` / `ConsoleComposerMenuModal`. The row reads 'Save chat (enables capture)' with the hint 'Not saved locally; records no traces until saved' only when that is On. Otherwise it reads 'Save chat' with 'Not saved locally; traces stay off once saved'. The default is False, so a caller that does not know never shows the promise.

**Interpretation, flagged for the owner.** Taken literally, the decision's parenthetical ('global capture On and no stored conversation Capture Off') would drop the suffix for global Off with conversation Capture On. Its rule ('only when saving really turns capture on') keeps it, because that chat captures once saved. The implementation follows the rule, and the test and the live round both pin this case.

**AC#4: fixed copy that cannot know what saving resolves to.** This now says only what is always true:
- the Trace screen's Exchange Capture Policy temporary guidance ('no traces are recorded while it stays temporary');
- the chip tooltip constant (unrendered since TASK-35160), which drops '(enables capture)'.

User Guide: semantic-trace-capture.md, context-and-rag.md and sessions-tabs-workspaces.md now say a saved chat follows its own Capture choice or the global default, and that the row reads 'Save chat (enables capture)' only when that is Capture On.

**Tests.**
- New `Tests/UI/test_console_temporary_save_copy.py` (added to the UI PR gate census, floor 1145 -> 1146) covers all four cases in one mount: global On, stored conversation Off, global Off, and global Off with conversation On. It drives the stock ChatScreen, controller, store and file DB, the real `replace_conversation_trace_privacy` and `apply_global_capture_settings` calls, and the Menu modal's own row (label and tooltip). It then presses the row and reads the saved chat's resolved Capture. It was red on the pre-fix head for the conversation Off and global Off cases ('Save chat (enables capture)' with the saved chat resolving Off), and it fails on dev.
- Unit pins in Tests/UI/test_console_composer_menu.py cover the builder and the helper (red pre-fix: ImportError). Tests/Chat/test_console_ephemeral.py checks that fixed copy never says 'enables capture' (red pre-fix on the tooltip). Tests/UI/test_console_temporary_capture_admission.py pins the dialog copy (red pre-fix).

**Live (fixed head, 160x45, real Anthropic via the review proxy; evidence/bd5/U07/30-39).**
- Global On: the row read 'Save chat (enables capture)' with the hint '...records no traces until saved'. After saving, a send logged `capture_enabled=true` and got the reply 'OK 31'.
- Conversation Capture Off (Trace Privacy > This conversation > Off): the row read 'Save chat' with '...traces stay off once saved'. After saving, a send logged `capture_enabled=false` and got the reply 'OK 33'.
- Global Off (Trace Privacy > Global default > Off, which wrote `exchange_capture = false` to the profile config): the row read 'Save chat' with '...traces stay off once saved'. After saving, a send logged `capture_enabled=false` and got the reply 'OK 36'.
- Global Off with conversation On: the row read 'Save chat (enables capture)'. After saving, a send logged `capture_enabled=true` and got the reply 'OK 38'.

**Gates (branch rebased onto dev ed7481a30c; plain command unless noted; reds compared by test id against a detached dev worktree).**
- test_console_temporary_save_copy: 1 passed; on dev 1 failed.
- test_console_temporary_capture_flow: 5 passed.
- test_console_composer_menu: 39 passed, 1 failed (test_impersonate_payload_obeys_the_provider_contract, red on dev too). Under the bootstrap profile, head has 1 red and dev has 2.
- test_console_ephemeral: 19 passed. Tests/Docs: the same 4 reds as dev, all in README / media-and-conversations.
- test_console_temporary_capture_admission, test_console_capture_policy_dialog and test_buddy_entry_points all pass under the bootstrap profile. A plain run hits RecoveryRequired setup errors on both sides.
- test_console_command_composer, under the bootstrap profile: 1 red on head, a subset of dev's 4.
- test_console_workbench_contract: 1 red on head only, test_console_inspector_prioritizes_actionable_status_before_secondary_groups. Run in isolation it fails 1 of 3 times on head and 1 of 4 on dev, so it is a flake rather than a regression.
- These files have no new reds against dev: test_console_exchange_capture, test_console_library_controls_workflow, test_console_received_import_contract, test_console_unsent_turn, test_console_staged_evidence_dispatch_pin, test_console_roleplay_model_workspace (dev's new file, after the session.py merge), test_console_global_capture_settings, and test_console_chat_controller_exchanges (2 reds, both also red on dev, which has 6).
- Module-size ratchet: the same 8 pre-existing reds as dev. The rebase merged dev's session.py row (6990, TASK-37620) with this branch's two dated raises (6990 -> 6994 -> 7001, the measured count).
- UI-ready module census: 4 passed. Boot CSS budget: 1 passed. ./scripts/preflight.sh: exit 0.

**Review fix (2026-10-29, AC#5).** The pre-merge review found the copy wrong when a one-time Next send Capture choice was set: promotion keeps the one-shot and it outranks the chat's own setting for one manual send, so with global Off and Next send On the row said 'traces stay off once saved' and the first send after saving was captured (mirror: global On with Next send Off promised capture and the first send was not captured). Reproduced first: the extended mounted test was red on both new cases for that reason.

Fix: `composer_menu_state` now passes the snapshot's `next_capture_enabled` as `next_send_capture`. When it differs from what the saved chat resolves to, the hint names that send: 'Not saved locally; once saved, only your next send is traced' (label 'Save chat') or 'Not saved locally; once saved, traces start after your next send' (label 'Save chat (enables capture)', because later sends are captured). Also from the review: the Off hint now reads 'Not saved locally; saving keeps traces off' (the old 'traces stay off once saved' read as permanent, but a chat with no choice of its own follows the global default if it changes); an unreadable capture policy (any exception, not only KeyError) makes no capture claim ('Save chat' / 'Not saved locally') and the Menu still opens; ADR-097's amendment says saving hands Capture to the conversation's own choice, else the global setting; the User Guide says a Next send choice pauses only the next send and names the one-shot hints.

Tests: `Tests/UI/test_console_temporary_save_copy.py` adds global Off + Next send On and global On + Next send Off, and now reads both the next send's and the later sends' capture after saving (the later read spends the one-shot through the dialog's own `set_next_trace_privacy`). Unit pins cover every row variant and the unreadable-policy path. Mutations: dropping the one-shot (`return saved, None`) turned the mounted test red on both new cases; catching only KeyError turned the unreadable-policy test red with the RuntimeError.

Live (fix head, 160x45, real Anthropic claude-haiku-4-5 via the review proxy): global Off + Next send On -> row 'Save chat', hint '...once saved, only your next send is traced'; after saving, sends logged capture_enabled=true then false (replies 'OK 41', 'OK 42'). Global On + Next send Off -> 'Save chat (enables capture)', hint '...once saved, traces start after your next send'; after saving, capture_enabled=false then true ('OK 43', 'OK 44'). Global Off alone -> 'Save chat' / 'Not saved locally; saving keeps traces off'.

Follow-ups filed: TASK-37840 (the always-visible Temporary surfaces no longer say 'not saved'; owner call), TASK-37841 (the Impersonate payload test's stale controller double, red on dev).
<!-- SECTION:NOTES:END -->
