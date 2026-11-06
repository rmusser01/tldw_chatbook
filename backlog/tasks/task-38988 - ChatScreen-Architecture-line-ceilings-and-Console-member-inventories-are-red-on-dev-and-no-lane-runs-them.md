---
id: TASK-38988
title: "ChatScreen Architecture line ceilings and Console member inventories are red on dev and no lane runs them"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- tests
- architecture
- console
priority: medium
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- Tests/Architecture
- Tests/Architecture/test_screen_size_ratchet.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b on 2026-11-05: a serial `Tests/Architecture` run gave 41 failed and 4105 passed in 1044 s, at load 7-10.

No lane runs these nodes on dev:
- `derived-artifacts.yml` targets none of the failing files.
- `test.yml:133` runs only on pushes to main and on manual dispatch.
- `nightly-deep.yml` runs from main; its last five runs all failed at main b0dadf1941, the newest on 2026-09-27.

This task covers 11 of the 41 nodes, all about `chat_screen.py`, which is now 24,604 lines.

**Line ceilings (6 nodes):**
- realtime `…projection_ceiling` (at most 18076);
- review-selection `…without_ratchet_raise` (at most (17215, 532)); TASK-35306 (To Do) already owns this one;
- wave6 closeout `…remaining_absolute_deficit` (at most (17037, 564));
- wave6 `…matches_the_implementation_base` (pinned 8d806b71d9, at most 22172);
- `…first_chat_task_ratchet_is_earned` (at most 19922);
- `…fleet_task_ratchet_is_earned` (at most 20065).

TASK-18610 AC#6 records an older version of this drift (21,253 lines against 20,943) and says the ceilings need an owner decision: extract, or re-baseline.

**Member inventories (5 nodes):**
- bare-shells (2 nodes): production now reads `_session` while the map still has `_library_activity`. The second node raises `AttributeError: 'ChatScreen' object has no attribute '_session'`.
- fleet ownership: new `_console_screen_displayed` and `_console_wake_probe_composer`.
- wave6 character controller: extra `render_radio`, `speech_presentation` and `set_active_workspace_for_session`.
- wave6 closeout margin: `on_console_selection_note_requested` lost `on(ConsoleTraceCopyRequested)`.

One PR can fix all 11, including TASK-35306's node. Of the branches due to merge before this one, only the TASK-38747 branch touches `Tests/Architecture`, and it does not touch these files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 11 nodes pass on dev; each ceiling is either re-measured with the owner's approval or retired in favour of the screen-size ratchet row, with the reason recorded (red before: 11 of 11 fail at 29592ecc9b)
- [ ] #2 Each ceiling that is kept still fails when chat_screen.py grows past it, shown with a recorded negative control
- [ ] #3 Each member inventory matches the shipped members, and each inventory change is listed in the notes as reviewed
- [ ] #4 No row in the screen-size ratchet rises
<!-- AC:END -->
