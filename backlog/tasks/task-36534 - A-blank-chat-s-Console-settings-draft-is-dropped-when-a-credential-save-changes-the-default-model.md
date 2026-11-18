---
id: TASK-36534
title: A blank chat's Console settings draft is dropped when a credential save changes the default model
status: Done
assignee:
- '@claude'
updated_date: 2026-11-03 17:09
created_date: 2026-10-24 01:37
labels:
- follow-up
- console
- settings
- ux
priority: medium
references:
- Tests/UI/test_console_settings_unsaved_guard.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies:
- TASK-36385
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while repairing TASK-36385's CI (PR #3225, 2026-10-24). A new user is usually in a blank chat when they first configure a credential: the chat has no settings of its own. From there:
1. They open Chat settings and pick a model whose provider has no key.
2. They click Configure credential, save the key in Settings, and press Return.
3. The Settings save also makes that model the default.

The blank chat then converges to the new default (ADR-095 D1, `_maybe_refresh_stale_default_console_settings`), which bumps its settings revision. The return is refused with "Chat settings changed while credentials were open", and the user's draft is dropped.

The "change" came from the user's own save during the handoff, not from anything concurrent. The implementer saw the same refusal on dev 5b5920b8e2 with the real, unstubbed Settings save, so this behaviour predates TASK-36385. That step 3's save changes the default model also comes from that reproduction; confirm it first.

`test_credential_round_trip_keeps_the_restored_edit_unsaved` gives its chat settings of its own, so that it does not take this path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 From a blank chat, Configure credential, a save made from that handoff's own Settings screen that changes the default model, and Return reopen Chat settings with the user's draft intact
- [x] #2 A Return after a change made somewhere else during the handoff, such as the same chat's settings edited in another view, is still refused with its message
- [x] #3 A test pins both outcomes through the real router, with the real Settings save writing to a private profile
<!-- AC:END -->

## Implementation Plan

1. Reproduce through the real router on a private profile (red first): a blank chat opens Chat settings, picks a model of a keyless provider and edits Temperature, Configure credential, saves the key through the real Settings writer (confirm that save makes the picked model the default), Return. Second case: the same chat's settings are edited elsewhere while Settings is open, then Return.
2. Root cause: the Console return compares the chat's raw settings revision with the handoff's (`ConversationSettingsReturnIntent.settings_revision`). The saved-default convergence (ADR-095 D1, `_maybe_refresh_stale_default_console_settings`, the only automatic settings replacement) bumps that revision, so the user's own save reads as a concurrent edit.
3. Fix: the session controller records, per chat, the revision span that saved-default convergence alone produced. The return's rejection check and the reopen's pre-push guard accept the handoff's revision when it is current or when convergence alone moved the chat from it; any other change breaks the span and is still refused with its message.
4. Remove the TASK-36534 workaround in `test_credential_round_trip_keeps_the_restored_edit_unsaved` (the chat no longer needs settings of its own).
5. Size ratchets, preflight, User Guide page for the Configure credential return.

## Implementation Notes

Confirmed first: the handoff's Settings save writes `chat_defaults.model` (D5, `build_provider_setup_mutation`), so the picked model becomes the default. The blank chat takes it lazily: D1 convergence (`_maybe_refresh_stale_default_console_settings`) runs inside the reopen's `_open_console_settings`, after the return's rejection check passed, and bumps the settings revision, so the reopen's pre-push guard failed and the retried return was refused with "Chat settings changed while credentials were open".

Fix: the session controller records the run of settings revisions that convergence alone moved a chat through, per chat (`_default_convergences`; convergence is the store's only `mark_user_work=False` replacement). `settings_revision_held()` (`Console_Modules/session.py`) accepts the handoff's own revision (`ConversationSettingsReturnIntent.settings_revision`, carried back in the return target) when it is current or inside that run; the return's rejection check and the reopen's pre-push guard in `chat_screen.py` both use it. Any other change in between (a settings or Context view edit) breaks the run and is refused with the same message. No new channel: the handoff's existing revision identifies it.

Tests: `Tests/UI/test_settings_save_reach.py::test_a_blank_chats_settings_draft_survives_its_own_credential_save` (whole-file census entry; real router, real Settings writer, `@private_profile_test`) has three cases: own-save (draft back: model and the Temperature edit, chat converged), settings-elsewhere and context-elsewhere (refused with the message; the Context edit leaves the chat unused, so D1 still moves it). The chat already took one saved default before the handoff, as after first-run setup.

Red before the fix: own-save failed on `assert not was_refused()` (refusal shown); a first exact-span version of the fix also failed it once the earlier convergence was added, which is why the run is a range. The two elsewhere cases pass at base and head.

Covering runs (load 12-26 on 18 CPUs, `-n 4`): the return/convergence slice of `test_console_session_settings.py`/`test_console_native_chat_flow.py` plus six whole files, 50 failures at head, all also red at base (37 `RecoveryRequired`, the rest pre-existing); base adds only the two expected reds. Four return-path tests in `test_console_native_chat_flow.py` (incl. `terminal_rejection_consumes_once[stale_revision]`) pass under a temporary `@private_profile_test` marker, since removed. `Tests/Architecture` serial: 41 failures, the same 41 node ids and messages at base. Preflight passes.

Size: `chat_screen.py` 24590 -> 24619, `session.py` 6973 -> 7009 and `console_settings_modal.py` 6168 -> 6176, rows re-pinned with dated waiver comments. Two hand-built controller hosts (`test_workspace_default_session.py`, `test_console_readiness_config_projection.py`) gain `_default_convergences`. User Guide: `Docs/User_Guide/settings.md` (Return to Chat settings).

Review follow-ups (coordinator, same PR):
- `test_a_held_settings_revision_spans_only_convergence` (same file) drives the controller's real convergence recording and `settings_revision_held`: held at the run's start, inside it and at its end, not ahead of the chat; an edit made elsewhere between two convergences ends the run for good. Red with the start-of-run check removed (`[True, True, True, True, True] == [False, ...]`) and with the run extension removed (`[False, True, True, False] == [True, ...]`), both temporary edits reverted. It also has another unused chat converge meanwhile, which leaves the first chat's run held; that case is red against a single-slot record (`[False, False, True, False] == [True, True, True, False]`), which is why the record stayed per chat (round 3 reverted 7d4348fd27).
- Fields the user left alone follow the chat. A handoff's own save can move more than the pair: a URL provider's Endpoint fixed on the Settings page, say. The reopened draft kept the old value, showed it as unsaved, and Apply would have written it back. Now, as with a pick's rebase (TASK-33006.16), `followed_suspended_draft()` (`console_settings_unsaved.py`) moves a settings field the draft left alone (pair, Endpoint, generation fields) to the chat's current value, but only when the chat's own value changed since the capturing modal opened (`unsaved_committed` against the chat now; never without `unsaved_committed`) and only while the draft holds the chat's pair after the pair's own follow; edits stay. `_open_console_settings` runs it on one read of the chat taken after the context preparation (the same read `remember_session_pair` takes) and rebuilds the opening draft only when something followed; the modal reads the chat again at construction, and a reopen whose construction read differs from the follow's refuses the push so the return retries. In the AC#1 flow and an ordinary round trip nothing follows (traced); in the qwencloud flow only `base_url` does. The suspended draft's `unsaved_baseline` now holds only the fields the capturing modal had edited, and `unsaved_committed` the chat's committed values when the capturing modal opened. `reopened_unsaved_baseline()` counts an untouched field as unedited at what the reopened modal shows, and an edit against the first modal's baseline, or against the chat's value when that moved meanwhile, so an edit the chat now holds (the picked model) is not unsaved. This replaces the baseline-only `committed_unsaved_baseline` of the previous round, and with it the edge case it left (an untouched field the save changed reading unsaved).
- Pins: the AC#1 test asserts the reopened labels are exactly `("Temperature",)` (red before the first follow-up at `('Model', 'Temperature') == ('Temperature',)`). `test_a_field_left_alone_follows_the_handoffs_own_save` (qwencloud: the Settings save also moves `api_base_url`, which the unused chat takes) asserts the reopened Endpoint is the new URL and the labels are exactly `("Temperature",)` with Temperature still 0.9; red at 68196fea15 with the old URL shown and labels `('Endpoint', 'Temperature')`.
- A blank chat's Model edit stops counting once the save converges the chat to it, so `test_credential_round_trip_keeps_the_restored_edit_unsaved` keeps its chat with settings of its own (its `[model]` case pins that a Model edit survives the round trip); only its comment changed. `test_snapshot_carries_the_unsaved_baseline_and_refuses_a_malformed_one` covers `unsaved_committed` the same way.
- Re-run (load 8-15): the covering set, including the snapshot/unsaved users, 59 failures, all red at base too; base adds only the new tests' expected reds. The four `test_console_session_settings.py` reopen tests that drive `_open_console_settings` with a suspended draft fail locally at base and head alike, at the same lines, even under a temporary `@private_profile_test` marker; the four return-path tests in `test_console_native_chat_flow.py` pass under it. Markers removed. The Architecture files that scan the changed modules: 19 failures, all in the base-red set. Ratchets and preflight pass.
- Round 3 (review of fef5288a2d): the first draft follow ran on every reopen and compared the chat with the draft, so in a chat that did not move a pick's rebased fields followed back to the old model's values (Apply would have written model-b with gpt-5's sampling). Fixed as above. Pins: the unsaved-guard round trip asserts the reopened draft's sampling values equal the first modal's after the pick (red at fef5288a2d: temperature 0.6 became 0.7); `test_a_suspended_draft_follows_only_what_moved_under_its_own_pair` covers no move, no committed values, a chat moved to another pair (red with the moved-only rule alone) and a same-pair Endpoint that follows.
- Round 3: an invalid Context and memory edit was dropped from the carried baseline (its label is in no baseline), so the reopened modal let Esc discard it. The carried mapping is now built over the edited labels (None for that flag), and `reopened_unsaved_baseline()` counts a label the reopened modal does not show at the chat's value. `test_an_invalid_context_edit_still_counts_after_the_round_trip`: kept invalid, Chat settings still asks (red before: `() == ('Context and memory',)`); fixed back, nothing is unsaved (red before and with the label fix alone).
- `test_console_endpoint_discovery.py::test_shared_evidence_change_refreshes_the_console_once` fails with base (9bbd5c2515) sources and with this branch at the same assertion (`console._console_credential_snapshot[2] == provider_connection_evidence(harness).version`, `1 == 2`): pre-existing, not from this task.
- Round 3 re-run (load 30-38, `-n 4`): `test_settings_save_reach.py` and the whole guard file 64 passed; the covering set 59 failures, the same set as round 2 and all red at base; the four return-path tests under a temporary `@private_profile_test` marker 6 passed (marker removed); the Architecture files that scan the changed modules 19 failures, all base-red; ratchets and preflight pass.
- Round 4 (review of 1b0e4272c1): pairs are compared and followed as `provider_identity_key` / `normalize_console_model_value` (the same pair spelled another way follows; red before, the Endpoint stayed old). The "moved" gate has its own case: a same-pair field left alone that differs from a chat value that did not move stays (red with the moved check removed: 0.4 became 0.7). The AC#1 test's `[during-discovery]` case moves the chat again (a new `max_tokens` default) while its models are listed: the first reopen does not open and the retry opens with `max_tokens` followed and only Temperature unsaved. In production the owner key of `prepare_console_settings_context` (it includes the settings revision) already refuses that reopen; with that check removed the new construction-read check alone keeps the case green, and with both removed it is red (the first reopen opened).
- Round 4 re-run (load 16-24, `-n 4`): `test_settings_save_reach.py`, the guard file and `Tests/State/test_screen_state_store.py` 108 passed; the covering set 59 failures, the same set as before and all red at base; the four return-path tests under a temporary `@private_profile_test` marker 6 passed (removed); the Architecture files that scan the changed modules 19 failures, all base-red; ratchets and preflight pass.

