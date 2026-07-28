---
id: TASK-1051
title: TaskResumeState.from_dict drops pending_skill_script
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 14:31'
updated_date: '2026-07-27 23:12'
labels:
  - console
  - skills
  - state-restore
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`TaskResumeState.from_dict` (`tldw_chatbook/UI/Screens/chat_screen_state.py:86`) restores `pending_skill_install` from a persisted Console snapshot but hardcodes `pending_skill_script=None`, discarding whatever was saved for that field. This is a pre-existing asymmetry between the two skill-confirm payload types the dataclass otherwise treats identically (`has_pending_skill_install`/`has_pending_skill_script`, `to_dict` serializes both fields the same way).

The failure mode is silent and fails closed: a session that was snapshotted mid skill-script-confirm restores with no visible pending confirm at all rather than an error, so a user resuming that session simply never sees the card they were about to decide on. Whether this was ever a deliberate decision (e.g. a script-confirm round cannot legitimately survive a restart/restore for some reason) is not documented anywhere near the field.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `pending_skill_install` and `pending_skill_script` are restored symmetrically by `from_dict`, OR the asymmetry is documented as deliberate at the field with the dead `pending_skill_script` field removed from the dataclass/serialization entirely.
- [x] #2 A regression test covers `from_dict` round-tripping a snapshot that carries a `pending_skill_script` payload.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the from_dict/to_dict call chain: who serializes/deserializes TaskResumeState, and is the round-trip a tab-switch (in-process) or an app-restart (disk) boundary.
2. Determine whether a restored pending_skill_install/pending_skill_script payload can ever reach a still-live ConsoleChatController round, or whether it is dead-by-construction.
3. Choose the AC branch supported by that evidence and implement it without touching pending_skill_install's separately-scoped, already-tested contract.
4. Confirm AC #2's regression-test requirement against the existing test suite; add coverage only if a gap is found.
5. Run the specified pytest gate and reconcile any failures against the documented pre-existing baseline.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Call-chain investigation (evidence, not assumption):

1. `TaskResumeState.to_dict()`/`from_dict()` round-trip exclusively through
   `ChatScreen.save_state`/`restore_state` (chat_screen.py:10778-10793),
   which `app.py`'s `handle_screen_navigation` (app.py:6243-6291) calls on
   every TAB SWITCH -- never on app shutdown/restart. The backing store,
   `ScreenStateStore` (UI/Navigation/screen_state_store.py), documents
   itself as "memory-only ownership for cross-visit screen snapshots" and
   never touches disk, confirming this is in-process only.

2. Even in-process, the round is not reconnectable: `ChatScreen.
   _create_navigation_screen`'s own docstring (chat_screen.py:6119-6141)
   states screens are "never cached and re-mounted" -- every navigation
   builds a brand-new `ChatScreen` instance whose `_console_chat_controller`
   starts as `None` (chat_screen.py:2124) and is lazily rebuilt from scratch
   by `_ensure_console_chat_controller` (chat_screen.py:3611-3677) with
   fresh, empty `_pending_skill_install_rounds`/`_pending_skill_script_
   rounds` dicts (console_chat_controller.py:860,888). A skill-script (or
   skill-install) confirm round only exists as an entry in those dicts on
   the OLD controller instance, guarding a worker thread blocked on a
   `threading.Event` -- gone the moment the screen is recreated.

3. Consequence: if `from_dict` restored `pending_skill_script`, `ChatTaskCards
   .sync_state` (chat_task_cards.py:54-55) would mount a fully-interactive
   `SkillScriptConfirmCard` from it, but any decision on it reaches
   `ConsoleChatController.resolve_pending_skill_script`
   (console_chat_controller.py:3022+), which silently drops a resolve whose
   `request_id` doesn't match a currently-armed round (fail-closed by
   design, console_chat_controller.py:3054-3070). That is a WORSE failure
   mode than today's (nothing mounts) -- a real-looking card whose Allow/Deny
   buttons do nothing forever, with no error and no explanation. Restoring
   the payload symmetrically (AC option 1) would therefore make the UX
   strictly worse, not better.

4. `pending_skill_install` goes through the byte-identical architecture
   (`_pending_skill_install_rounds`, `resolve_pending_skill_install`'s
   identical strict `request_id` match, console_chat_controller.py:2809-
   2852) and is exposed to the exact same dead-card hazard -- but IS
   restored by `from_dict` today, and that round-trip is pinned by an
   existing, deliberately-authored test from TASK-910
   (Tests/UI/test_console_skill_install_confirm.py::
   test_task_resume_state_pending_skill_install_roundtrip). This is a
   genuine, pre-existing asymmetry, but not one TASK-1051's AC authorizes
   fixing: TASK-910 added install's round-keyed restore before this
   dead-UI hazard was identified for script (task-6/script-confirm work,
   which shipped `from_dict`'s script-drop WITH a regression test already
   proving the "no dead card" contract --
   Tests/UI/test_skill_script_confirm_card.py::
   test_restored_state_drops_the_pending_script_so_no_dead_card_appears,
   committed 2026-07-25, predates this task). Changing pending_skill_install
   to match would break that unrelated, currently-passing contract and is
   out of this task's scope -- flagged here as a follow-up candidate, not
   silently fixed alongside it.

Decision: AC option 2 (document the asymmetry as deliberate). The dataclass
field itself is NOT dead -- `pending_skill_script` is the live carrier
`ChatScreen._set_console_pending_skill_script` mutates directly whenever a
real, in-session round is armed (chat_screen.py:15940-15953), and
`ChatTaskCards.sync_state` reads it to render the card during that live
window. Only the FROM_DICT direction of the round-trip is dead (a restored
request_id can never match a live round), and that was already correctly
hardcoded to None with a regression test pinning it
(test_skill_script_confirm_card.py, predates this task). Interpreting the
AC's "field removed from the dataclass/serialization entirely" literally
(deleting the field) would delete real, live, load-bearing functionality --
not what "dead field" can mean once the live-carrier role is accounted for.
Implemented instead: (a) a field-level comment on `pending_skill_script`
pointing at `from_dict`'s docstring, (b) an expanded `from_dict` docstring
that lays out the full call-chain evidence above, explicitly documents why
`pending_skill_script` is dropped while `pending_skill_install` (same
hazard) is not, and cites both regression tests. No behavior change --
`to_dict`/`from_dict` outputs are byte-identical before and after.

AC #2: already satisfied by pre-existing coverage, not new work --
`test_skill_script_confirm_card.py::
test_restored_state_drops_the_pending_script_so_no_dead_card_appears`
round-trips a `TaskResumeState.to_dict()` snapshot carrying a populated
`pending_skill_script` payload through `from_dict` and asserts it comes
back `None`/absent while a sibling `pending_skill_install` and `summary`
survive. No new test added since no behavior changed; nothing to pin that
wasn't already pinned.

Verification: `.venv/bin/python -m pytest Tests/UI/test_console_workbench_
contract.py Tests/UI/test_skill_script_confirm_card.py Tests/UI/test_console_
skill_install_confirm.py Tests/ProductionApp/test_chat_root_state_removal.py
Tests/UI/test_console_mcp_approval.py` -> 133 passed, 4 failed. Two failures
match the task's documented pre-existing baseline (CSS-geometry batch-row
geometry, MCP cancellation execution-log). The other two
(test_console_empty_transcript_choose_model_opens_settings,
test_set_console_pending_skill_script_preserves_other_resume_fields) were
independently verified pre-existing on HEAD f77132c87 via `git stash` before
any edit in this task -- unrelated to chat_screen_state.py (an
InvalidSelectValueError in a Settings Select widget, and a stale
`screen.chat_state` attribute reference that does not exist anywhere in
chat_screen.py -- neither touches TaskResumeState).
<!-- SECTION:NOTES:END -->
