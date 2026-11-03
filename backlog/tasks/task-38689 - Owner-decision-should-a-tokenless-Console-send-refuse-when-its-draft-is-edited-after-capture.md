---
id: TASK-38689
title: "Owner decision: should a tokenless Console send refuse when its draft is edited after capture?"
status: Done
assignee:
- '@claude'
created_date: 2026-11-03 11:40
updated_date: 2026-11-03 21:13
labels:
- follow-up
- console
- needs-owner-decision
priority: medium
references:
- tldw_chatbook/UI/Console_Modules/wiring.py
- tldw_chatbook/UI/Console_Modules/hooks.py
- tldw_chatbook/UI/Screens/chat_screen.py
- Tests/UI/test_console_send_draft_snapshot.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-36520, PR #3376, 2026-11-03). Checked on dev 9bbd5c2515.

Two rules on the Console send path disagree about text typed after a send captures its draft.
- **hooks.py:430-440 (TASK-340).** After the hook snapshot read, "text typed after the capture belongs to the next draft ..., not a reason to refuse this send". hooks.py refuses an edited draft only when a hook review actually ran.
- **wiring.py:370-375.** `receive_console_visible_intent` arrived with owner commit 3e618d7119 ("wip: checkpoint reviewed Console preparation integration", 2026-10-15). When the send carries no captured inputs, it compares the live composer's generation and edit serial with the captured stash, and it refuses any difference with "Draft changed; Send again.".

**Which sends reach the wiring check.**
- Enter does not. It carries a pending-send token, and the token's received inputs go through as `_captured_inputs` (chat_screen.py:18320-18341, :18486-18489).
- Sends without a token do: a mouse or Workbench Send (chat_screen.py:18318, "Mouse/Workbench sends have no token"), and the voice command "Console, send.", which calls `handle_console_send_message` (chat_screen.py:10379).

When the stash is captured just before the check (composer_run_controls.py:1031-1039), no edit can land in between. When it is captured before an await, typing during that await is refused. `Tests/UI/test_console_send_draft_snapshot.py::test_console_armed_unknown_mouse_send_snapshots_before_skill_await` is such a case: its docstring says later typing is preserved and the send proceeds. On dev it fails with `['Draft changed; Send again.']` (2026-11-03, `-n 4`). PR #3376 isolates the test from the wiring check and leaves product code as it is.

No test asserts the "Draft changed; Send again." refusal.

**Question for the owner.** When a tokenless send's draft is edited after the send captured it, which should happen?
- (a) The send is refused with "Draft changed; Send again." (wiring.py today).
- (b) The captured text is sent, and the later typing stays as the next draft (hooks.py's TASK-340 rule).
- (c) Something else.

Owner decision 2026-11-03: TASK-340 wins (answer (b), recorded verbatim: "task-340"). Text the user types after a send captures its draft belongs to the next draft and is not a reason to refuse the send. Hook review's existing rule (refuse an edited draft only after a review ran) is unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's answer is recorded in this task
- [x] #2 Tokenless sends (button, Workbench and voice) follow that answer, and a test pins it, including the refusal copy if refusal stays
- [x] #3 A tokenless send (the spoken "Console, send." path) whose draft is edited after its capture sends the captured text to the provider, with no "Draft changed; Send again." refusal
- [x] #4 Text typed after that capture stays in the composer and in the chat's saved draft as the next draft
- [x] #5 Hook review's refusal of a draft edited after a review ran is unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace why `receive_console_visible_intent`'s live-vs-stash check exists (owner checkpoint 3e618d7119) and what else guards a stale stash or another chat.
2. Tests first, in `Tests/UI/test_console_send_draft_snapshot.py`: a mounted spoken "Console, send." test (capture, type during the skill await, send) on the stock send harness, counting provider calls; drop PR #3376's `stash=None` isolation from `test_console_armed_unknown_mouse_send_snapshots_before_skill_await` and fix its docstring. Show both red on dev.
3. Fix at the root in `wiring.py`: the tokenless received path sends the live composer, so when the composer is no longer the captured revision it declines (returns `None`) instead of refusing; the ordinary dispatch then sends the captured stash and commits only that revision (TASK-340). Session and chat-switch guards stay as they are.
4. Grep Tests/ for the old refusal copy and for tests pinning it; check the Console send and voice User Guide pages.
5. Gate the new test in a PR lane (its file is in none), then run the send/snapshot files with `-n 4`, the hook-admission tests, ratchets for touched modules, and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner decision 2026-11-03: TASK-340 wins. A tokenless Console send whose draft is edited after its capture now sends the captured text and keeps the edit as the next draft. It is no longer refused with "Draft changed; Send again.".

**Why the check existed (3e618d7119).** Without captured inputs, the received fast path sends the *live* composer. It publishes `draft` as the authored draft at the composer's current generation and edit serial, and after acceptance it reloads the composer from the store (`_project_console_received_input` gets no stash). So it cannot send an edited composer and keep the edit, and the live-vs-stash check stopped it. Refusing was stricter than needed. The ordinary dispatch (`_hooks.dispatch`, then the prompt queue) carries the stash, sends `stash.text`, and commits only the captured revision (TASK-340). The text-mismatch check just below already falls back to it. **Fix** (`wiring.py`): when the composer is no longer the captured revision, decline (`return None`) instead of notifying and returning `""`. Protection against a stale stash or another chat lives elsewhere and is unchanged:
- the session-id checks here and in the visible action;
- `_publish_console_authored_draft`'s switch-snapshot refusal (pinned by `test_switch_settlement_send_refuses_stale_draft_without_consumption`);
- hook review's chat re-check;
- `commit_captured_draft`'s generation check.

Declining sends nothing on this path, so it adds no exposure.

**Which sends are tokenless.** In production only `handle_console_send_message` is tokenless, called by the spoken "Console, send." (`_run_pending_console_voice_action`). Enter, the Send button and the Workbench "send" all go through `send_acknowledgement.request_visible_send` (TASK-33620.15) with a token and captured inputs, so they already followed TASK-340. The comment in `chat_screen.py` that said "Mouse/Workbench sends have no token" was stale; it is reworded, comment only. The window between a tokenless capture and the wiring check is the skill-context await of an armed unknown command.

**Tests** (`Tests/UI/test_console_send_draft_snapshot.py`):
- New `test_spoken_send_sends_its_capture_and_keeps_text_typed_after_it`: the stock send harness (`build()`, real runtime, a gateway that counts provider calls), hook review live, the edit landing as the held skill read returns.
  - Red on dev 335f984bd2: `['Not sent.'] == ['Sent.']`, notices `['Draft changed; Send again.']`.
  - Green with the fix: one provider call carrying `/nosuchcommand`, and the composer and saved draft both `x`. Passed in 3 repeat runs at load 35-46.
  - Its file is in no PR lane, so its node id is appended at the tail of `scripts/ui_pr_gate_census.txt`, with the floor raised to the measured count.
- `test_console_armed_unknown_mouse_send_snapshots_before_skill_await`: PR #3376's `stash=None` wiring isolation is removed and the docstring fixed. Red on dev with `['Draft changed; Send again.']`; green with the fix. Its older hook-review isolation is kept.
- No test pinned the old refusal. A grep of Tests/ for "Draft changed; Send again." finds only the new negative assertion.
- AC#5: `hooks.py` is untouched. `Tests/Chat/test_console_hook_admission.py::test_late_ready_cannot_send_a_cancelled_or_changed_draft[edit]` (refusal after a review) and `test_a_ready_send_rechecks_its_chat_but_not_text_typed_after_capture` pass.

**Runs** (all through run.sh, `-n 4`):
- Covering set (draft snapshot, send acknowledgement, resend guard, received intent feedback, hook admission, hook review send freeze): 117 passed, 11 failed. The same 11 fail at base with identical assertions (base files restored temporarily, same command), so they are pre-existing. None of these files is in a PR lane.
- Architecture size ratchets (serial): 205 passed.
- Hook pins plus the census CI tests: 124 passed.
- `check_markup_interpolation.py`: OK.
- `preflight.sh`: passed.

**Docs.** `Docs/User_Guide/console/chat-basics.md` now says a spoken "Console, send." follows Enter's rule. A lesson about the wrong-reason red is in `backlog/docs/lessons-testing-evidence.md`.

**Files:**
- `tldw_chatbook/UI/Console_Modules/wiring.py`
- `tldw_chatbook/UI/Screens/chat_screen.py` (comment)
- `Tests/UI/test_console_send_draft_snapshot.py`
- `scripts/ui_pr_gate_census.txt`
- `scripts/check_ui_pr_gate_census.py`
- `Docs/User_Guide/console/chat-basics.md`
- `backlog/docs/lessons-testing-evidence.md`
<!-- SECTION:NOTES:END -->
