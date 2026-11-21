---
id: TASK-42141
title: "Decide whether Console setup guidance leaves at once when the first send completes"
status: Done
assignee:
- '@claude'
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 18:45
labels:
- follow-up
- console
- onboarding
- decision
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/Chat/console_received_dispatch.py
- tldw_chatbook/Chat/console_onboarding_state.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoletests stream (TASK-42015, round 18). `ChatScreen._console_first_send_completed` (`tldw_chatbook/UI/Screens/chat_screen.py:11840-11857`) reads `console.onboarding.first_send_completed` from `app.app_config` once and keeps it in `_console_first_send_completed_cached` for the screen's life. The live save after a first reply goes through `ConsoleRuntime._record_successful_first_send` (it writes the flag with `apply_settings_mutation_to_cli_config`, `Chat/console_received_dispatch.py:952`) and does not touch that memo. The memo's only other writer, `ChatScreen._record_console_first_send` (`:11859`), has no production callers (TASK-42015 deletes it).

So on the same ChatScreen, a new empty tab opened after the first reply still shows the setup guidance (`first_send_completed=self._console_first_send_completed()` at `:13856`); a fresh ChatScreen reads the saved flag and shows none. Product question: should guidance leave as soon as the first send completes?
<!-- SECTION:DESCRIPTION:END -->

## Decision

Owner, 2026-11-21, verbatim: "141 - yes". Console setup guidance leaves as soon as the first send completes, including on new tabs opened later in the same ChatScreen visit.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's decision is recorded: whether setup guidance leaves at once when the first send completes, or stays for the rest of that ChatScreen's life as today
- [x] #2 If at once: a new empty tab opened after the first reply on the same ChatScreen shows no setup guidance, and a test shows it, red before the change
- [x] #3 If today's behaviour stays: a comment at the memo records the decision, and a test pins that the guidance stays on that ChatScreen (not applicable: the owner chose "at once")
<!-- AC:END -->

## Implementation Plan

1. Record the owner's decision (2026-11-21, verbatim "141 - yes"): guidance leaves as soon as the first send completes, new tabs in the same ChatScreen visit included.
2. Pilot test, red first: send a first message on a mounted Console, wait for the reply and the in-memory flag, open a new tab (the strip's New tab button), and check the new empty tab shows the quiet empty state with no Get started card.
3. Fix: `ChatScreen._console_first_send_completed` memoizes only a True value, so it re-reads `app.app_config` (a few dict lookups) until `ConsoleRuntime._record_successful_first_send` has set the flag, then stops reading.
4. Update the Console User Guide's first-run section; run the covering tests and the screen-size ratchet.

## Implementation Notes

**Decision recorded** above (owner, 2026-11-21, "141 - yes").

**Change.** `ChatScreen._console_first_send_completed` now caches only a True read (`if not self._console_first_send_completed_cached:`). Until the flag is set it re-reads `app.app_config["console"]["onboarding"]["first_send_completed"]`, which `ConsoleRuntime._record_successful_first_send` sets in memory right after the first completed manual turn (before the config.toml save that waits for quiescence). Once True, no more reads. The cost while False is three dict lookups per guidance build, and both callers (`_sync_console_transcript_guidance`, the composer's setup-card check in `composer_run_controls.py`) already derive settings readiness, which costs far more; nothing new runs on a key or tick path. The memo's docstring records the decision.

**Test.** `Tests/UI/test_console_runtime_ownership.py::test_a_tab_opened_after_the_first_reply_shows_no_setup_guidance` (bootstrap_profile, in the admission-sensitive PR lane with its file): reads the flag as False before any send, sends a first message through the Send button, waits for the reply and the in-memory flag, opens a tab with the strip's New tab button, waits until the new tab's empty transcript shows either form, then asserts the card state is `quiet`, no "Get started" and "No messages yet." visible. The new tab gets the default provider (not ready in this harness), so before the fix it shows the full Get started card. Red at base: `assert 'card' == 'quiet'`; green after (1 passed, load ~12).

**Covering runs** (`-n 4`, load 12-18): the whole runtime-ownership file, `Tests/Backup_Recovery/test_console_guidance_readiness.py` and the first-send/setup-card/empty-state/guidance nodes of `test_console_native_chat_flow.py`: 99 passed, 7 failed. The same 7 fail identically with chat_screen.py at base: five plain (non-bootstrap) chat-flow nodes on the local profile traps (three `RecoveryRequired: raw_source_selection_changed`, two at `_build_console_send_test_app`'s `save_setting_to_cli_config` returning False), and two guidance-readiness nodes (`('native config reads in one derivation', 0)` and a derive-count list). Under a temporary `@private_profile_test` marker the four chat-flow first-send/setup-card nodes (`test_console_blocked_empty_transcript_shows_setup_card_steps`, `test_console_first_send_flag_switches_empty_state_to_quiet`, `test_console_accepted_send_records_first_send_flag`, `test_console_failed_send_does_not_record_first_send_flag`) passed with the fix; the marker was then removed. Screen-size ratchet: 76 passed.

**Docs.** `Docs/User_Guide/console.md` ("First run: the Get started card"): once the first send's reply arrives, guidance is gone for good, new tabs included.

Files: `tldw_chatbook/UI/Screens/chat_screen.py`, `Tests/UI/test_console_runtime_ownership.py`, `Docs/User_Guide/console.md`.
