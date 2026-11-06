---
id: TASK-38988
title: "ChatScreen Architecture line ceilings and Console member inventories are red on dev, and no lane runs them"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:42
updated_date: 2026-11-06 08:53
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

One PR can fix all 11. TASK-35306's node closes with it, so TASK-35306 should be closed by the same PR rather than keep a second owner. Of the branches due to merge before this one, only the TASK-38747 branch touches `Tests/Architecture`, and it does not touch these files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The 11 nodes pass on dev; each ceiling is either re-measured with the owner's approval or retired in favour of the screen-size ratchet row, with the reason recorded (red before: 11 of 11 fail at 29592ecc9b)
- [x] #2 Each ceiling that is kept still fails when chat_screen.py grows past it, shown with a recorded negative control
- [x] #3 Each member inventory matches the shipped members, and each inventory change is listed in the notes as reviewed
- [x] #4 No row in the screen-size ratchet rises
<!-- AC:END -->

## Implementation Plan

1. Re-confirm the 11 reds at base e8eeae7db9.
2. Re-pin each live-screen ceiling at the measured value under the owner size waiver, with a dated comment. Keep every frozen-revision constant as it is.
3. Bring each member inventory in line with the shipped members, review every change, and list it in these notes.
4. Negative controls: grow chat_screen.py by one line, and raise the ChatScreen ratchet row by one.

## Implementation Notes

**Ceilings: re-pinned under the owner size waiver, not retired.** Every comment is dated 2026-11-06 and names TASK-38988. The measurements were taken at dev e8eeae7db9 with each file's own counting helper, and the size did not change on this branch.

| Node | Old ceiling | New ceiling |
| --- | --- | --- |
| realtime `…projection_ceiling` | 18,076 / 577 | `LIVE_SCREEN_*` 24,604 / 741 |
| review-selection `…without_ratchet_raise` | (17,215, 532) | `LIVE_SCREEN_COUNTS` (24,604, 709) |
| wave6 closeout `…remaining_absolute_deficit` | (17,037, 564) | (24,604, 741) |
| wave6 `…matches_the_implementation_base` | 22,172 and 20,943 / 712 | 24,604 / 741 |
| wave6 `…first_chat_task_ratchet_is_earned` | 19,922 / 633 | 24,604 / 741 |
| wave6 `…fleet_task_ratchet_is_earned` | 20,065 / 640 | 24,604 / 741 |

- Each ceiling gets its own `LIVE_*` constant. The historical constants (`PROJECTED_*`, `FINAL_CLOSEOUT_COUNTS`, the wave-6 bases) still pin their frozen revisions and stay as they are.
- The review-selection and closeout tests also pin the screen-size ratchet row. Their pins move with the ceilings: review-selection to `== ("ChatScreen", 24_616, 741)`, closeout to `<= (24_616, 741)`. A raise of that row therefore still fails both.
- TASK-18610 AC#6 had asked whether to extract or re-baseline. The owner waiver in the round-10 brief answers it: re-baseline.

**Member inventories. Every change was reviewed.**
- **Bare shells (2 nodes).**
  - The store setter's hook build (`wiring.console_view_hooks`) reads `self._session`'s project-instruction projections (aad71de772). It no longer reads `_library_activity.build_provider`. `CONTROLLER_STUBS` is now `{_fleet, _session}`.
  - New helper `stub_session_controller` in `Tests/UI/console_controller_stubs.py`: a real `ConsoleSessionController` whose 76 keyword-only callables, read from the signature, are all raisers. It follows the same `NO_APP` rule as the other helpers.
  - The ratchet then named 9 store-setting fixtures, and each now calls the helper before its store assignment:
    - attachment_riders and citation_sources: the helper replaces their hand-built `SimpleNamespace` sessions. citation_sources wires `chat_store_accessor` so it behaves the same.
    - generation_actions, live_work_handoffs and rag_settings: these keep their bare `__new__` session for the test body, assigned after the store. live_work_handoffs and rag_settings now seed `_console_visible_draft_session_id` on that bare session directly, because the screen proxy's setter reads it first.
    - composer_menu ×2 and video_actions: the helper call is added beside their existing fleet and library stubs.
  - The rag round-trip (`test_source_scope_survives_a_screen_state_round_trip`) went from failing in setup with `'ChatScreen' object has no attribute '_session'` at base to passing. live_work_handoffs went from 11 to 6 failures; the 6 left fail with RecoveryRequired config mutations, the TASK-38111 trap, on both trees.
- **Fleet ownership.** `_console_screen_displayed` and `_console_wake_probe_composer` now live on `ConsoleFleetLifecycleController` and read through wired accessors; the DOM-call check still passes. The whole moved group is therefore owner-checked, and the "must stay on the screen" assertion is gone.
- **Wave-6 character controller.** Three new dependencies: `set_active_workspace_for_session` (e289f48355), and `speech_presentation` and `render_radio` (b333cebdfe, radio portraits). Those two have no-op defaults. The test now allows defaults for exactly those two and requires every other dependency to be wired. Proposed follow-up: make them required. They were left alone here because the radio-portrait stream (TASK-36549/36550) is in flight.
- **Wave-6 closeout margin.** The test reads chat_screen.py at the frozen `AMENDMENT_BASE` d20dd733b7, where the handler had only `on(ConsoleSelectionNoteRequested)`. 681bc7becb added the live trace-copy decorator and wrote it into this frozen expectation, which made the test red. The expectation is back to the frozen shape.

**Evidence.**
- **Red before.** All 11 nodes failed in the serial base run.
- **After.** The six files pass serially: 69 passed.
- **Negative controls (AC#2), each a temporary edit, reverted.**
  - One line appended to chat_screen.py turned all 6 ceiling nodes red (6 failed, 52 passed).
  - The ChatScreen ratchet row raised from 24616 to 24617 turned the review-selection and closeout nodes red.
- **AC#4.** No screen-size ratchet row changed.
- **Covering tests, -n 4.** 11 fixture and character files gave 15 failed on mine against 16 at base. The only failure seen only on mine is `test_character_shapes_and_controls_settle_inside_current_budget[source_size0-pixels]`, a timing-budget node that also flaked alone (2 of 4 failed); this task does not touch the character avatar.

TASK-35306's node, review-selection `…without_ratchet_raise`, closes with this task; TASK-35306 is set Done.

**Review r10a-rev1 fix: live ceilings follow the ratchet row.**
- The measured pin of 24,604 sat below the screen-size ratchet's ChatScreen row (24,616 / 741). A waived growth inside the row would have turned these four laneless files red where nobody sees them.
- All four now read their live ceiling from `Tests/Architecture/test_screen_size_ratchet.py::_BUDGETS`. The review-selection and closeout row pins, `("ChatScreen", 24_616, 741)` and `<= (24_616, 741)`, still fail on any raise of the row.
- Negative control: 13 temporary lines take chat_screen.py to 24,617, one past the row, and all six ceiling nodes go red (6 failed, 53 passed). Reverted.

