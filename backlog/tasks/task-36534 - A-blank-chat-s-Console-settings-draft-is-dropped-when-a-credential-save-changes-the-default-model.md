---
id: TASK-36534
title: A blank chat's Console settings draft is dropped when a credential save changes the default model
status: To Do
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
- [ ] #1 From a blank chat, Configure credential, a save made from that handoff's own Settings screen that changes the default model, and Return reopen Chat settings with the user's draft intact
- [ ] #2 A Return after a change made somewhere else during the handoff, such as the same chat's settings edited in another view, is still refused with its message
- [ ] #3 A test pins both outcomes through the real router, with the real Settings save writing to a private profile
<!-- AC:END -->
