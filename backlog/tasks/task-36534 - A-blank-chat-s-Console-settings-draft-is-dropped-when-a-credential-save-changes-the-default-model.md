---
id: TASK-36534
title: A blank chat's Console settings draft is dropped when a credential save changes the default model
status: Done
assignee:
- '@claude'
updated_date: 2026-11-03 13:35
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

Fix: the session controller records, per chat, the run of settings revisions that convergence alone moved it through (`_default_convergences`; convergence is the store's only `mark_user_work=False` replacement). `settings_revision_held()` (`Console_Modules/session.py`) accepts the handoff's own revision (`ConversationSettingsReturnIntent.settings_revision`, carried back in the return target) when it is current or inside that run; the return's rejection check and the reopen's pre-push guard in `chat_screen.py` both use it. Any other change in between (a settings or Context view edit) breaks the run and is refused with the same message. No new channel: the handoff's existing revision identifies it.

Tests: `Tests/UI/test_settings_save_reach.py::test_a_blank_chats_settings_draft_survives_its_own_credential_save` (whole-file census entry; real router, real Settings writer, `@private_profile_test`) has three cases: own-save (draft back: model and the Temperature edit, chat converged), settings-elsewhere and context-elsewhere (refused with the message; the Context edit leaves the chat unused, so D1 still moves it). The chat already took one saved default before the handoff, as after first-run setup.

Red before the fix: own-save failed on `assert not was_refused()` (refusal shown); a first exact-span version of the fix also failed it once the earlier convergence was added, which is why the run is a range. The two elsewhere cases pass at base and head. The 36534 workaround in `test_credential_round_trip_keeps_the_restored_edit_unsaved` is gone (blank chat, asserts nothing is applied after Return); its `[model]` case fails at base at `assert returned is not None` and passes with the fix.

Covering runs (load 12-26 on 18 CPUs, `-n 4`): the return/convergence slice of `test_console_session_settings.py`/`test_console_native_chat_flow.py` plus six whole files, 50 failures at head, all also red at base (37 `RecoveryRequired`, the rest pre-existing); base adds only the two expected reds. Four return-path tests in `test_console_native_chat_flow.py` (incl. `terminal_rejection_consumes_once[stale_revision]`) pass under a temporary `@private_profile_test` marker, since removed. `Tests/Architecture` serial: 41 failures, the same 41 node ids and messages at base. Preflight passes.

Size: `chat_screen.py` 24590 -> 24598 and `session.py` 6973 -> 7009, rows re-pinned with dated waiver comments. Two hand-built controller hosts (`test_workspace_default_session.py`, `test_console_readiness_config_projection.py`) gain `_default_convergences`. User Guide: `Docs/User_Guide/settings.md` (Return to Chat settings).

