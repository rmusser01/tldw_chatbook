# Plan: Resend a broken Console turn in place (TASK-33661)

Owner request 2026-10-01. One task, one PR, cut from dev 84247cb843. Task file: backlog/tasks/task-33661 - Resend-a-broken-Console-turn-in-place-from-the-user-message.md

## Global Constraints

- Work ONLY in this worktree: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-resend. Start EVERY shell command with `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/console-resend &&`. Never touch the main checkout at /Users/macbook-dev/Documents/GitHub/tldw_chatbook (another session's uncommitted work) or the model-config-p5 worktree (a live workflow).
- Python: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python (the worktree has no venv); run pytest FROM the worktree cwd. Tests/Architecture needs -p no:xdist. The venv's editable install points at the MAIN checkout: some tests can import the main checkout's code. Compare failure-NAME sets against a clean origin/dev worktree before calling anything a regression.
- NEVER write ~/.config/tldw_cli or ~/.local/share/tldw_cli. Never bypass test isolation (private_profile_test, TLDW_TEST_PRIVATE_PROFILE_NODE, HOME/XDG/TLDW_CONFIG_PATH). Live runs: scratch TLDW_CONFIG_PATH with a unique users_name, FULL SCREEN 211x44 (primary) and 235x52.
- ADR-126 RecoveryRequired in a clean worktree is environmental. To run gated UI tests, use a temp copy with `pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]` appended at the END, then delete it. Any new test that drives the real ChatScreen, controller or store must carry `bootstrap_profile` itself (lessons-testing-evidence.md: "A scratch plugin that marks every test bootstrap_profile hides missing markers"). The final gate is the plain `.venv/bin/python -m pytest <files>` with no extra -p plugins.
- Size ratchets never rise (ADR-097, Tests/Architecture/test_module_size_ratchet.py measures non-blank code lines). console_chat_controller.py, console_transcript.py and console_chat_store.py are AT their budgets, and the slack test forbids leaving budget unused. Put the resend logic in a NEW module (for example tldw_chatbook/Chat/console_turn_resend.py), keep delegation in those three files to net <= 0 lines (trim proven-dead code if needed), and leave their budget rows unchanged. chat_screen.py's screen-ratchet row is already red on dev; it must not grow.
- Design rules (owner decisions, binding):
  (1) Resend shows only for a broken LAST turn (see AC#1).
  (2) It is in place: never call edit_and_resend_message, never create_sibling, never fork.
  Reuse what exists: retry_message/prepare_message_retry for a failed reply; delete_message (subtree tombstone) to clear an empty stopped/discarded/restored reply and trailing transcript-only SYSTEM rows, then continue_from_message(user id); for a refused unpersisted echo, delete the echo subtree and re-dispatch its text and attachments through the NORMAL send path (chat_screen._dispatch_console_draft_send), with no duplicate left in the composer.
  A re-run must apply what a normal send of that turn applies: thinking-persistence preflight, pinned prefill, the session's agent or tool-calling mode, and the attachments and vision gate. continue_from_message skips some of these today, so check each one and record a Ruling for it.
  Clear stale failure SYSTEM rows so the new reply parents under the user message.
- Gate and route Resend exactly like Retry and Regenerate in UI/Console_Modules/message.py: send_refusal_copy first, then run_worker(..., exclusive=True, group=f"console-run-{sid}"). Hide it while a run is live or a dispatch recovery is unresolved.
- Broken-turn detection for the action row: compute it from the transcript's active-path rows and pass it into available_actions as a kwarg, following the fork_eligibility precedent (console_transcript.py _action_groups). On a broken USER row, swap the disabled ♻ for ("resend","Resend"), mirroring the failed-assistant Retry swap. Update the `r` key so it invokes resend or retry when that is the row's action.
- ADR-031: never bind Ctrl+C/V/X/S/D/Z/A/R/W; footer and help hints must match working bindings.
- TDD; real controller and store tests (in-memory SQLite) for every broken shape; rewrite pinned action-set tests on purpose and name them in the notes.
- Commit as `feat(console): <summary> (TASK-33661)` (fixes: `fix(console): …`) plus `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never push; NEVER merge origin/dev into the branch.
- Tick ACs, set Done, add Implementation Notes in the task file. Update Docs/User_Guide CONTENT for the Console; never add "Verified against" paragraphs (record verification in task notes).
- Lessons (backlog/docs/lessons-*.md): insert new entries MID-FILE near a related section, never at the end of the file.
- Evidence: commit captures (.txt/.ansi.txt) under qa/, but NEVER commit one-off driver or probe scripts; describe the capture procedure in a short qa README instead.
- Before DONE: covering tests + `PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python ./scripts/preflight.sh` (rc 0; never piped through tail).

## Task 1: Resend a broken Console turn in place from the user message (TASK-33661)

Task file: backlog/tasks/task-33661 - Resend-a-broken-Console-turn-in-place-from-the-user-message.md

### Why

Why: when a Console send fails or gets stuck, the user sees an error and has no direct way to try again. Today the user has to choose Edit, then Edit & resend. That forks a new sibling branch, and for some failure shapes it is the only path left. The owner asked (2026-10-01) for a one-click **Resend** on the user's own message that resumes the broken chat.

Owner decisions:
- Resend appears only when that turn failed or got no reply.
- It re-runs the same turn in place: no fork, no sibling branch, no second copy of the user message.

Lead defaults (the owner can override): Resend applies to the last turn of the active path only, since a failed reply higher up keeps its own Retry. An empty reply that the user stopped counts as "no reply", which makes stuck → Stop → Resend work. A partial reply does not count, because Continue covers it.

How the code models a broken turn today (research against dev 84247cb843, 2026-10-01):
- Provider or HTTP error: the user message and the assistant row are both persisted. The assistant is marked failed, and a transcript-only SYSTEM "Provider stream failed" row follows. The assistant row offers Retry, but nothing on the user row does.
- Empty stream: the assistant is marked failed with no SYSTEM row.
- Stop on a hung or stuck run: the assistant is marked stopped and a SYSTEM "Response stopped by user." row follows. It offers ♻ (which forks) and Continue, but no Retry.
- Readiness or other refusal before the send is accepted: the user echo is unpersisted and marked failed, a SYSTEM block row follows, and there is no assistant row. The draft and staged attachments stay in the composer. The user row shows Continue, which today parents the new reply under the SYSTEM row and leaves the echo failed (an untested hazard).
- After a restart: hydration rebuilds nodes as complete, keeping only assistant_generation_state. A failed reply then reads "Response failed." and loses Retry.
- Dispatch recovery after a restart (a reply left in flight) has its own card with Retry response and Discard. Discard settles the reply as discarded.

Existing machinery that resend can reuse without forking:
- The assistant in-place retry (retry_message / prepare_message_retry) for a failed reply.
- continue_from_message, which re-runs from a persisted user message and appends a reply at the active leaf.
- delete_message (subtree tombstone).
- The normal send path, for re-dispatching a refused echo.

Related open tasks, not absorbed here: TASK-370 (resume or retry for interrupted replies), TASK-33620.3 (a failed durable commit leaves a user message with no assistant row, stuck on Running), TASK-33621.22 (Stop during "Connecting tools…" wedges the next send).

### Acceptance criteria

- [ ] #1 A user message that is the last turn on the active path shows a Resend action only when its turn is broken. Broken means: its send was refused before acceptance; or it has no assistant reply; or its reply failed; or its reply is empty and stopped, discarded, or restored as "Response failed." after a restart. Resend never shows on a healthy turn, on a partial (non-empty) stopped reply, on a turn above the last one, or on an assistant row.
- [ ] #2 Resend re-runs that turn in place, with no new branch, no sibling and no second copy of the user message. The broken or empty reply and its trailing failure or stop SYSTEM rows are cleared, and the new reply appears directly under the same user message. A refused echo is re-sent with the same text and attachments as exactly one user message, and the composer is not left holding a duplicate of that text.
- [ ] #3 Resend is not offered while a run is live in that tab (Stop comes first) or while a dispatch-recovery card is unresolved (the card owns that case). Every normal send gate applies, including readiness, vision and skill refusal, and a refused Resend shows the same refusal copy as a normal send.
- [ ] #4 Resend works after an app restart for a persisted broken last turn: a restored failed or discarded reply, or a last user message with no reply.
- [ ] #5 Keyboard: with a broken user row selected, `r` runs Resend. With a failed assistant row selected, `r` runs Retry; today it does nothing there. The action's tooltip and the help or footer text describe Resend, and no hint names a key that does not work (ADR-031).
- [ ] #6 The selected action row still fits the reference terminals at 211x44 and 235x52. Tests that pin the user-row and assistant-row action sets are updated on purpose and named in the notes.
- [ ] #7 Tests cover each broken shape with a real controller and store: provider error, empty stream, refused echo, empty stopped reply, restored after restart, and discarded. Each proves the re-run is in place by message count, ids and parentage, with no sibling. There are negative tests for a healthy turn, a mid-path turn, a live run and an unresolved dispatch recovery. A pilot test clicks Resend in the real Console.
- [ ] #8 The Console User Guide documents Resend: when it appears and what it does.
- [ ] #9 No size ratchet rises (ADR-097). The resend logic lives in its own module. console_chat_controller.py, console_transcript.py and console_chat_store.py each net 0 or fewer lines, and preflight passes.

### Research map (dev 84247cb843; locate by symbol, line numbers drift)

- Action model: Chat/console_message_actions.py — ConsoleMessageAction, _COMPLETED_ACTIONS, _PRIMARY_ACTION_IDS, available_actions (failed-assistant retry swap), action_groups, dispatch (retry block), _action_enabled/_action_disabled_reason.
- Rendering: Widgets/Console/console_transcript.py — _action_groups (fork_eligibility kwarg), _action_row, _action_button (id console-message-action-<action>-<message>), _ACTION_TOOLTIPS, BINDINGS (`r` → invoke_selected_action('regenerate')), _press_selected_action_button.
- Routing: UI/Console_Modules/message.py handle_console_message_action (retry and regenerate branches) and button-id parsing.
- Controller: retry_message, continue_from_message, stop_active_run, retry_dispatch_recovery, discard_dispatch_recovery, send_refusal_copy, _append_failure_system_row, _mark_transient_echo_blocked, _block.
- Store: prepare_message_retry, mark_message_failed, mark_message_send_blocked, rollback_transient_send, delete_message.
- Hydration: Chat/console_conversation_hydration.py (status rebuilt as complete; assistant_generation_state carried); Chat/assistant_generation_state.py ("Response failed.").
- Pinned tests to update or extend: Tests/Chat/test_console_message_actions.py (test_failed_user_row_offers_no_retry_action, test_user_and_stopped_assistant_action_groups_are_exact, test_continue_action_remains_available_for_failed_user_message, the action_row_guide tests); Tests/UI/test_console_native_transcript.py (the width-budget tests, the stable-ids test, the user regenerate-disabled test); Tests/UI/test_console_native_chat_flow.py (failed-stream retry tests); Tests/Chat/test_console_chat_controller.py (retry, continue and echo tests).
