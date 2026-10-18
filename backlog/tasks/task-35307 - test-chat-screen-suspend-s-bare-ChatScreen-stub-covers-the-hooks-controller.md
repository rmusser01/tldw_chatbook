---
id: TASK-35307
title: 'test_chat_screen_suspend''s bare ChatScreen stub covers the hooks controller'
status: To Do
assignee: []
created_date: '2026-10-17 09:11'
labels:
- follow-up
- tests
- console
dependencies: []
references:
- Tests/UI/test_chat_screen_suspend.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Console_Modules/wiring.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen while checking PR #3144 (Phase 8); red on dev too. Tests/UI/test_chat_screen_suspend.py builds `ChatScreen.__new__(ChatScreen)` and stubs each seam `on_screen_suspend` calls. Since 2bce8ffa0f (2026-09-28, Console hook review) `on_screen_suspend` also reads `self._hooks.review_open` and may call `self._hooks.cancel_pending()`. `_hooks` is bound per instance in UI/Console_Modules/wiring.py, so the bare stub has none and the hook raises AttributeError. Checked at origin/dev 111f8e48fc by reading: locally both tests stop earlier, at setup, with the known `RecoveryRequired: raw_source_selection_changed` noise. No PR lane runs this file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both tests in Tests/UI/test_chat_screen_suspend.py reach their assertions and pass on dev, in a run that is not stopped by `RecoveryRequired` at setup (a plain CI-style run, or a private profile)
- [ ] #2 The suspend-order test asserts that the hooks controller's pending review is cancelled where `on_screen_suspend` cancels it
- [ ] #3 Each test fails when the seam it pins is removed from `on_screen_suspend` (negative control recorded in the notes)
<!-- AC:END -->
