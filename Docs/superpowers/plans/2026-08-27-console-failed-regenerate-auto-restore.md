# Console Failed-Regenerate Auto-Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the previous good assistant answer on the active branch and in the next provider request whenever a regenerate attempt fails or produces no content, while retaining the failed sibling for inspection and retry.

**Architecture:** Keep the existing sibling-per-regenerate tree model and generic streaming failure handling. After `regenerate_message` reloads the replacement sibling, treat `status == "failed"` as a controller postcondition and point the session's active leaf back to the original assistant anchor with the existing `ConsoleChatStore.set_active_leaf` contract. Successful and stopped siblings remain active, and the failed sibling remains stored as a navigable sibling.

**Tech Stack:** Python 3.11+, Textual 8, pytest/pytest-asyncio, existing `ConsoleChatController` and `ConsoleChatStore` APIs.

---

## Scope and decision record

- Design: `Docs/superpowers/specs/2026-08-27-console-failed-regenerate-auto-restore-design.md`
- Backlog item: `backlog/tasks/task-571 - Console-branching-a-failed-regenerate-drops-the-prior-good-answer-from-provider-context-until-swipe-back.md`
- ADR required: no
- ADR path: N/A
- Reason: this is a routine recovery correction within the existing active-leaf and sibling-branch contracts. It changes no schema, persistence boundary, service contract, security policy, dependency, or long-lived application structure.

## Task 1: Pin transport-failure recovery at the controller boundary

**Files:**

- Modify: `Tests/Chat/test_console_regenerate_branching.py:143-176`
- Reference: `tldw_chatbook/Chat/console_chat_controller.py:11111-11284`

- [ ] Rename `test_regenerate_stream_failure_marks_new_sibling_failed_not_a1` to describe the complete contract: a failed replacement is retained but the original anchor is restored.
- [ ] Keep the existing assertions that the result reports the provider failure and that the anchor content/status remain unchanged.
- [ ] Identify the failed assistant sibling through `store.siblings_at(a1.id)` rather than only the active-path transcript, because the restored branch intentionally hides the failed sibling from `messages_for_session`.
- [ ] Add assertions that the failed sibling still exists with `status == "failed"`, the active leaf is `a1.id`, and the active path is the original user/assistant path.
- [ ] Call `controller._provider_messages_for_session(session.id)` and assert it contains the prior good `{"role": "assistant", "content": "seed"}` turn. This pins the user-visible bug at its provider-context boundary.
- [ ] Add `test_regenerate_mid_conversation_failure_restores_selected_anchor_not_former_tail`: seed `u1 -> a1 -> u2 -> a2`, fail a regenerate of `a1`, and assert the active path becomes exactly `[u1.id, a1.id]`. Assert `u2` and `a2` remain stored but off-path, the failed sibling remains alongside `a1`, and provider context ends at the restored `a1`. This pins the approved boundary: restore the selected anchor, not the entire pre-operation tail.
- [ ] Run both transport-failure tests before production changes:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/Chat/test_console_regenerate_branching.py::test_regenerate_stream_failure_retains_failed_sibling_and_restores_anchor \
    Tests/Chat/test_console_regenerate_branching.py::test_regenerate_mid_conversation_failure_restores_selected_anchor_not_former_tail
  ```

  Expected: 2 failed because the current active leaf is the failed branch's system row, the provider context omits the anchors, and the mid-conversation branch does not stop at `a1`.

## Task 2: Pin zero-chunk recovery without changing stopped-output behavior

**Files:**

- Modify: `Tests/Chat/test_console_variant_stream.py:366-413`
- Verify unchanged: `Tests/Chat/test_console_variant_stream.py:417-489`

- [ ] Rename the empty-stream test to describe automatic anchor restoration.
- [ ] Remove the manual `store.set_active_leaf(session.id, mid)` recovery step from the test.
- [ ] Assert that the empty replacement remains a failed sibling, `store.active_leaf(session.id) == mid`, the original message is on the active path, and the next provider context already contains `original`.
- [ ] Run the empty-stream test before production changes:

  ```bash
  ../../.venv/bin/python -m pytest -q Tests/Chat/test_console_variant_stream.py::test_regenerate_empty_stream_retains_failed_sibling_and_restores_anchor
  ```

  Expected: FAIL because the current implementation leaves the failed sibling active and excludes `original` from provider context.
- [ ] Strengthen the existing stopped-stream test before implementing the fix: assert the stopped sibling remains on the active path and the original anchor remains off-path. Do not require the stopped sibling to be the literal leaf because the existing `Response stopped by user.` system row may follow it.
- [ ] Run the strengthened stopped-stream test as a pre-change control:

  ```bash
  ../../.venv/bin/python -m pytest -q Tests/Chat/test_console_variant_stream.py::test_regenerate_stop_mid_stream_leaves_anchor_untouched_new_sibling_stopped
  ```

  Expected: PASS before the production change.

## Task 3: Add the minimal failed-regenerate postcondition

**Files:**

- Modify: `tldw_chatbook/Chat/console_chat_controller.py:11117-11144`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:11245-11265`

- [ ] Update the `regenerate_message` docstring so it no longer claims that a failed sibling stays on the active path. Document that the failed sibling remains stored/retryable while the original anchor becomes active again.
- [ ] Immediately after reloading `persisted_sibling`, restore the anchor only for the terminal failed state:

  ```python
  if persisted_sibling is not None and persisted_sibling.status == "failed":
      self.store.set_active_leaf(session_id, message_id)
  ```

- [ ] Do not add a store helper, schema field, preference, or UI state. Do not restore on `complete` or `stopped`, and do not attempt recovery when the replacement has disappeared.
- [ ] Keep trace construction unchanged so it still records the failed replacement's persisted ID and `failed` status after the active-leaf correction.
- [ ] Run the three new regression tests together:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/Chat/test_console_regenerate_branching.py::test_regenerate_stream_failure_retains_failed_sibling_and_restores_anchor \
    Tests/Chat/test_console_regenerate_branching.py::test_regenerate_mid_conversation_failure_restores_selected_anchor_not_former_tail \
    Tests/Chat/test_console_variant_stream.py::test_regenerate_empty_stream_retains_failed_sibling_and_restores_anchor
  ```

  Expected: 3 passed.
- [ ] Run success and stop controls:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/Chat/test_console_regenerate_branching.py::test_regenerate_creates_sibling_and_streams_into_new_active_leaf \
    Tests/Chat/test_console_variant_stream.py::test_regenerate_stop_mid_stream_leaves_anchor_untouched_new_sibling_stopped
  ```

  Expected: 2 passed; the successful/stopped replacement remains active.
- [ ] Commit the green controller slice:

  ```bash
  git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_regenerate_branching.py Tests/Chat/test_console_variant_stream.py
  git commit -m "fix(console): restore anchor after failed regenerate"
  ```

## Task 4: Verify the behavior through the mounted Console action

**Files:**

- Modify: `Tests/UI/test_console_regenerate_feedback.py:1-78`
- Reference: `Tests/UI/test_console_native_chat_flow.py:6864-6930`

- [ ] Add a small `FailingRegenerateGateway` beside `GatedGateway`; it resolves as ready and its async-generator `stream_chat` raises `RuntimeError("forced regenerate failure")` before content. Include an unreachable `yield ""` after the raise so the test exercises the intended provider exception through a valid async iterator.
- [ ] Add a mounted Textual test using the existing `ConsoleHarness`, `_seed_selected_assistant_message`, and real `#console-message-action-regenerate-{id}` click path.
- [ ] After clicking, wait for the existing `Provider stream failed:` feedback and separately poll until `store.active_leaf(session.id) == source.id`; the pre-existing source text alone is not completion evidence.
- [ ] Call `await console._sync_native_console_chat_ui()` after restoration, assert the source row is present in the mounted transcript, and assert the source answer is present in `_provider_messages_for_session` without any swipe or retry action.
- [ ] Assert the failed replacement still appears in `store.siblings_at(source.id)` with `status == "failed"`. This proves recovery does not delete diagnostic/retry state.
- [ ] Run the mounted test:

  ```bash
  ../../.venv/bin/python -m pytest -q Tests/UI/test_console_regenerate_feedback.py::test_console_failed_regenerate_auto_restores_previous_answer
  ```

  Expected: 1 passed. This is the live TUI acceptance check: a real mounted Console message action forces a post-validation provider failure and the known-good answer is restored automatically.
- [ ] Commit the mounted-flow coverage:

  ```bash
  git add Tests/UI/test_console_regenerate_feedback.py
  git commit -m "test(console): cover failed regenerate recovery in mounted UI"
  ```

## Task 5: Update the user-facing recovery documentation

**Files:**

- Modify: `Docs/User_Guide/console/branching-and-rewind.md:193-200`

- [ ] Replace the task-571 troubleshooting note with the shipped behavior: failed or empty regenerates automatically return to the previous good answer; the failed attempt remains available as a sibling for inspection or retry.
- [ ] Keep the note explicit that a stopped partial regenerate remains selected, because stopping is intentional rather than a provider failure.
- [ ] Check that the wording matches the approved design and mounted test; do not add a new setting or manual-recovery instruction.
- [ ] Commit the documentation update:

  ```bash
  git add Docs/User_Guide/console/branching-and-rewind.md
  git commit -m "docs(console): explain failed regenerate recovery"
  ```

## Task 6: Focused verification, review, and Backlog closeout

**Files:**

- Modify: `backlog/tasks/task-571 - Console-branching-a-failed-regenerate-drops-the-prior-good-answer-from-provider-context-until-swipe-back.md`
- Review: all files changed against `origin/dev`

- [ ] Run the focused controller modules and mounted regenerate-flow module:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/Chat/test_console_regenerate_branching.py \
    Tests/Chat/test_console_variant_stream.py \
    Tests/UI/test_console_regenerate_feedback.py
  ```

  Expected: all tests pass. Pre-existing dependency and macOS pytest temp-cleanup warnings may remain, but no test failures or new warnings should be introduced.
- [ ] Run static checks only on the touched Python files:

  ```bash
  ../../.venv/bin/python -m ruff check \
    tldw_chatbook/Chat/console_chat_controller.py \
    Tests/Chat/test_console_regenerate_branching.py \
    Tests/Chat/test_console_variant_stream.py \
    Tests/UI/test_console_regenerate_feedback.py
  git diff --check origin/dev...HEAD
  ```

  Expected: both commands exit 0.
- [ ] Review `git diff --stat origin/dev...HEAD`, `git diff origin/dev...HEAD`, and `git status --short`; confirm the change is limited to the approved recovery behavior, tests, docs, and task metadata.
- [ ] Update the Backlog task: check all acceptance criteria, add concise Implementation Notes listing the controller postcondition, retained failed sibling, provider-context coverage, mounted TUI verification, documentation update, and `ADR required: no` rationale.
- [ ] Set TASK-571 to Done with Backlog CLI only after the checks above pass:

  ```bash
  backlog task edit 571 -s Done
  ```

- [ ] Commit task closeout and verify the worktree is clean:

  ```bash
  git add "backlog/tasks/task-571 - Console-branching-a-failed-regenerate-drops-the-prior-good-answer-from-provider-context-until-swipe-back.md"
  git commit -m "chore(backlog): close task 571"
  git status --short
  ```

  Expected: `git status --short` prints nothing.
