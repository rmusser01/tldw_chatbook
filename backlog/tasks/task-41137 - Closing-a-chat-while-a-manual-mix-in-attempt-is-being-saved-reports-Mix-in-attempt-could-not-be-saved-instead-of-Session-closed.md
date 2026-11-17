---
id: TASK-41137
title: "Closing a chat while a manual mix-in attempt is being saved reports 'Mix-in attempt could not be saved' instead of 'Session closed.'"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:20
updated_date: '2026-11-16 22:45'
labels:
- follow-up
- console
- copy
priority: low
references:
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_prompt_mixin_service.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-22690 gave every close path one outcome, `SESSION_CLOSED_COPY = "Session closed."`, and TASK-40312 fixed the native-commit path to match. A close that lands while a manual mix-in attempt is being accepted still reads as a save failure.

**Evidence.** `console_chat_controller.py:24614-24629`: any `MixinSnapshotUnavailableError` from `accept_attempt` blocks with "Mix-in attempt could not be saved; response was not sent.", including the owner-changed refusal a close produces (`console_prompt_mixin_service.py`, `mixin_attempt_owner_changed`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A close during the manual mix-in accept reports 'Session closed.' in the result and the run state
- [x] #2 A real save failure still reports the save copy
- [x] #3 Tests cover both, the first red before the change
<!-- AC:END -->

## Implementation Plan

1. Reproduce: a test holds a regenerate's mix-in attempt save, closes the chat (begin + finalize), releases, and checks the turn result and run state; show it red ("Mix-in attempt could not be saved").
2. In the controller's `accept_attempt` refusal branch, end the turn with `_session_closed_result` when the controller is disposed or the session has a committed close / is gone (the TASK-40312 / TASK-39159 checks); keep the save copy otherwise.
3. A second case makes the held save raise (a real save failure) and pins the save copy and a blocked run state.
4. Run the covering Chat files.

## Implementation Notes

The premise held at base cc6006fcf5: a close during a regenerate's mix-in attempt save (`accept_attempt` -> `_run_owned_save`) purges the session, `accept_attempt` raises `mixin_attempt_owner_changed`, and the controller blocked the turn with "Mix-in attempt could not be saved; response was not sent.".

The controller's refusal branch (`_stream_assistant_response`, `console_chat_controller.py`) now ends the turn with `_session_closed_result` (STOPPED, `SESSION_CLOSED_COPY`, no blocker) when the controller is disposed, the session has a committed close (`_session_close_generations`, the TASK-40312 check) or the session is gone (the wake path's TASK-39159 check). Any other refusal, including a save that raises (`mixin_attempt_save_failed`), keeps the save copy and the blocked state.

Test: `Tests/Chat/test_automatic_wake_recovery.py::test_a_close_during_a_manual_mixin_save_reports_session_closed[close|save_failed]` holds the attempt save, closes (begin + finalize) or makes the save raise, and pins the turn result and run state. Red before the fix: `[close]` failed `assert 'Mix-in attem...was not sent.' == 'Session closed.'`; `[save_failed]` passed (the copy it pins was already right). Green after. The file is on the derived-artifacts lane already.

Covering runs (`-n 4`, load 30-37): the wake-recovery, self-hosted retry accounting and durable-commit files 69 passed; the close (`test_console_close_*`, `test_saved_progress_close.py`) and prompt mix-in (`test_console_prompt_mixin_*`) files 242 passed. The controller stays under its module-size row (28745 of 28746).

No User Guide page quotes the copy. Files: `tldw_chatbook/Chat/console_chat_controller.py`, `Tests/Chat/test_automatic_wake_recovery.py`.

**Review r15co-rev1 (2026-11-16): the guard's clauses are separated.** New case `[begun_close]`: the close is begun but not finalized, so the session still exists, and the held save then raises. Only the committed-close generation clause can tell that this refusal is a close; the test pins "Session closed." in the result and STOPPED in the run state. With `owner_id in self._session_close_generations` removed from the guard (temporary edit, reverted), `[begun_close]` failed (`'Mix-in attem...was not sent.' == 'Session closed.'`) while `[close]` and `[save_failed]` passed. With the clause in place all three pass (`-n 3`). Covering run (wake recovery, self-hosted retry accounting, durable commit; `-n 4`, load 77-80): 70 passed.
