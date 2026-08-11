---
id: TASK-15121
title: Console send-button state assertions fail on dev
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 05:20'
updated_date: '2026-08-11 13:57'
labels:
  - console
  - tests
  - dev-baseline
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two failures in `Tests/UI/test_console_native_chat_flow.py`, both about the send button's state classes:

- `test_console_composer_stop_is_subdued_when_idle` — expects `console-send-blocked`; the button carries `console-action-disabled console-send-inactive console-send-button console-action-subdued` instead.
- `test_console_duplicate_send_during_stream_does_not_break_stop_control` — expects `button.disabled is True`; it is `False`, with classes `console-send-ready … console-action-primary`.

**Proven pre-existing on dev, not introduced by task-14920's repair**: both were re-run against a pristine `git archive origin/dev | tar -x` tree and failed identically there. They were not part of the 20 that task documented — they appeared in the same file after dev moved 46 commits, so a send-button state refactor landed between the two runs.

Triage is the work: the class vocabulary may have been deliberately renamed (`console-send-blocked` → `console-action-disabled`/`console-send-inactive`), in which case the tests are stale pins; or the button genuinely no longer reaches the blocked/disabled state during a stream, which is a real control regression — the second test's name says the stop control must survive a duplicate send. Do not assume the rename reading: check what the button is meant to do mid-stream before rewriting either assertion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each failure is classified as a stale class-name pin or a real send/stop control regression, with the commit that changed the behaviour named
- [x] #2 A real regression is fixed in the product; a rename is followed in the test while preserving the original claim (that the control is genuinely unavailable, not merely styled differently)
- [x] #3 `Tests/UI/test_console_native_chat_flow.py` runs whole with a READ pass count and no unexpected failures
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both failures and read the exact assertion lines that fail (RED).
2. Find the commit that changed send-button state: `git log -S` over the class names and over the `send_blocked` computation.
3. Read that commit's intent (task/ADR/spec) and classify each failure: stale pin vs real regression.
4. Verify the send-block state machine has no hole (every run state vs the queue presentation) before accepting the contract-change reading.
5. Repair the tests to the shipped contract while preserving each test's original safety claim; mutation-check every new assertion.
6. Run the whole file for a READ pass count with the task-15120 xfail intact.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Verdict: neither failure is a class rename, and neither is a regression. Both are stale
pins of a contract that was deliberately replaced.** No product change was needed; the
repair is test-only, in `Tests/UI/test_console_native_chat_flow.py`.

**Cause (both failures): `14cc326e4` "feat(console): add visible prompt queue"**
(TASK-14808, ADR-046 §"Visible bounded Console prompt queue"). Found with
`git log -S 'send_blocked = not queue_presentation.send_enabled' -- tldw_chatbook/UI/Screens/chat_screen.py`
and `git log --diff-filter=A -- tldw_chatbook/UI/Console_Modules/prompt_queue.py` (same commit).
Intent, quoted from ADR-046: *"Queueing begins only after the active turn crosses the existing
accepted-send boundary. During provider and skill validation the action remains unavailable and
reads `Preparing...`. Once accepted, the normal `Send` action becomes `Queue`; Enter and the
button both enqueue the exact canonical text draft."* That commit rewrote the same assertions in
the sibling file `Tests/UI/test_console_send_disabled_state.py` (including renaming
`test_enter_hotkey_while_run_blocked_still_shows_feedback` →
`test_enter_hotkey_queues_draft_behind_accepted_run`) but missed these two.

**Per-failure classification**

1. `test_console_composer_stop_is_subdued_when_idle` — stale pin, control still unavailable.
   Mid-stream with the draft consumed, `send_button.disabled is True` still passes; the button
   fails the *empty-draft* gate (`console-send-inactive`), not the run gate. `console-send-blocked`
   is simply unreachable in an accepted-turn state now. Repaired to keep the original claim
   (`disabled is True`, `console-action-disabled`, not `console-action-primary`) and pin the
   reason honestly (`console-send-inactive`, not `console-send-blocked`, label `"Queue"`).
2. `test_console_duplicate_send_during_stream_does_not_break_stop_control` — pinned behaviour
   deliberately REMOVED. With a draft loaded mid-stream the button is legitimately enabled as
   `Queue`; a second send is admitted to the bounded FIFO queue instead of refused. The single
   `composer.load_draft("second send")` is the only difference from failure 1 — same symptom,
   opposite cause. The test's named claim is preserved and asserted harder: the duplicate send
   must not start a second run (`run_state.status == "streaming"`), must land in the queue
   (`prompt_queue_registry.snapshot(...).total_count` 0 → 1), must not be eaten
   (`composer.draft_text() == ""`, since admission is what clears the draft), and Stop must still
   work.

**New contract and where it is enforced instead of `console-send-blocked`:**
`derive_prompt_queue_presentation` in `tldw_chatbook/UI/Console_Modules/prompt_queue.py` is the
sole decider of `send_enabled`/`send_label`; `chat_screen.py::_sync_console_composer_action_state`
replaces the run-state-derived `send_blocked` with `not queue_presentation.send_enabled`, then
re-ORs the setup/attachment blocks. Checked for a hole before accepting the reading: every
non-`is_send_allowed` run status maps to either `occupies_slot and not queue_owned` →
`"Preparing..."`/disabled (VALIDATING, RETRYING, and STREAMING before acceptance) or
`accepted_live_turn` → `"Queue"`/enabled-but-queued. There is no state where a send is admitted
as a second concurrent run. `console-send-blocked` remains live and tested for Preparing/Queue
full/setup blocks — `test_console_streaming_chunks_render_after_slow_provider_validation` in this
same file still asserts it during VALIDATING and passes untouched.

**Evidence.** RED before: both tests failed at lines 2773 and 2809. GREEN after: both pass.
Whole file: **308 passed, 1 xfailed in 270.76s**, with task-15120's `xfail(strict=True)` still
XFAIL (not XPASS). `Tests/UI/test_console_send_disabled_state.py`: 9 passed. Mutation-checked
every new assertion against the product (each restored byte-identical afterwards):
dropping `set_class(not has_draft, "console-send-inactive")` kills the class assertion;
forcing `send_label = "Send"` in the queue-owned branch kills both `label.plain == "Queue"`
assertions; forcing `send_enabled = False` there kills `disabled is False` and the
`not console-send-blocked` assertion; disabling the `accepted_live_turn` admission branch kills
`total_count == 1`.

**Files changed:** `Tests/UI/test_console_native_chat_flow.py` (two assertion blocks),
`backlog/docs/lessons-testing-evidence.md` (new lesson: two tests failing on the same missing
class can have opposite causes; the causing commit's own *test* diff is the authority on intent).
<!-- SECTION:NOTES:END -->
