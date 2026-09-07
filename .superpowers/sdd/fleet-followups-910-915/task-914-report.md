# task-914 — Remove or wire the dead single-approval card buttons

## Verdict: REMOVE (not wire)

Reachability sweep confirms the single-approval body (`#approval-single-body`, "Allow
once" `#approval-allow-once`, "Deny" `#approval-deny`) and its `set_approval()` method
were fully dead in production. Removed rather than wired.

## Reachability sweep — usage-site table

| Site | File | Role | Verdict |
|---|---|---|---|
| Mount | `Widgets/Chat_Widgets/chat_task_cards.py` (`ChatTaskCards.compose`) | Only place `ChatApprovalCard(id="chat-approval-card")` is instantiated in production | LIVE — sole mount site |
| Driver | `ChatTaskCards.sync_state` | Sole caller of the card's mutator methods; pre-fix branched on `"calls" in pending_approval` — `set_batch` for batch shape, **`set_approval` for anything else** | Was the single-approval API's only production caller; its `else` branch was never reachable in practice (see next row) |
| Producer | `Chat/console_chat_controller.py` (`request_mcp_approvals` → `_marshal_pending_approval`, plus `_parked_approval_payloads` reads at `create_session`/`switch_session`/`close_session`) | Sole writer of `TaskResumeState.pending_approval` (via `ChatScreen._set_console_pending_approval`) | Every payload it ever builds is the batch dict literal with a `"calls"` key (`console_chat_controller.py:1944-1960`), or `None`. **No code path anywhere constructs the legacy `{"summary","details",...}` shape.** |
| Consumer | `UI/Screens/chat_screen.py` (`@on(ChatApprovalCard.ApprovalDecided)`, `handle_console_inspector_review_approval`) | Forwards decisions to the controller; inspector "Review approval" seam focused the card's action button via a `batch_visible` ternary that fell back to `#approval-allow-once` | The `else` (non-batch) branch of the ternary was dead for the same reason — `set_batch` is the only thing that ever displays the card, so `#approval-batch-body` is always the visible body |
| Consumer | `Widgets/Console/console_status_chips.py` (`ConsoleApprovalsChip`/`_focus_pending_approval_card`) | Same pattern, duplicated: batch-visible ternary falling back to `#approval-allow-once` | Same — dead fallback |
| Legacy chat surface | `UI/Chat_Window_Enhanced.py` | Brief's flagged "legacy chat surfaces" caller of the single-approval API | **Does not exist** — fully retired by commit `94b2c558f` ("refactor(chat): retire dormant legacy composition (task-649)"), which deleted this file, its callers, and its dedicated pinning suite `Tests/UI/test_chat_approvals_and_resume.py` in one shot, without also removing the now-orphaned widget code in `chat_approval_card.py`. That gap is exactly what task-914 closes. |
| Comment-only | `Chat/console_chat_controller.py`, `Widgets/Chat_Widgets/skill_install_confirm_card.py` | Docstring/comment references to `ChatApprovalCard`/`set_batch` | Not code paths |
| Tests | `Tests/UI/test_console_mcp_approval.py`, `test_chat_approval_card.py`, `test_console_parallel_runs.py`, `test_console_workbench_contract.py` | Mount/drive the card via `set_batch` (or, pre-fix, one test drove the legacy shape directly through `set_task_resume_state`) | See below |

**Conclusion:** No production path ever populates the single-approval body. The only
thing keeping it alive was one test (`test_console_workbench_contract.py`) that
synthesized the legacy `{"summary","details"}` shape directly, and two duplicated
UI-thread fallbacks (`chat_screen.py`, `console_status_chips.py`) referencing
`#approval-allow-once` as a focus target that could never be reached.

## What was removed / changed

- `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`:
  - Removed the `#approval-single-body` `Container` and its three buttons
    (`#approval-allow-once`, `#approval-deny`, `#approval-details`) from `compose()`.
  - Removed the `set_approval()` method entirely.
  - Removed the now-dangling `self.query_one("#approval-single-body").display = False`
    line inside `set_batch()`.
  - Updated module/class docstrings that referenced the legacy API or the deleted
    `test_chat_approvals_and_resume.py` file, and recorded the task-649 provenance so a
    future reader doesn't have to re-derive it.
- `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py`:
  - `ChatTaskCards.sync_state` no longer branches on `"calls" in approval`; it always
    calls `approval_card.set_batch(approval.get("calls") or [], ...)`, which already
    treats an empty/absent list as "clear" — collapsing what used to be two code paths
    (batch vs. legacy) into one, since the legacy path was unreachable.
- `tldw_chatbook/UI/Screens/chat_screen.py` (`handle_console_inspector_review_approval`)
  and `tldw_chatbook/Widgets/Console/console_status_chips.py`
  (`_focus_pending_approval_card`): both dropped the `batch_visible` ternary and now
  focus `#approval-submit` unconditionally — that button is the card's only possible
  action target now that `set_batch` is its sole production entry point.

## Tests

- **Removed** `test_console_approvals_chip_activation_focuses_pending_approval_card`
  (`Tests/UI/test_console_workbench_contract.py`) — this was the one place that drove
  the legacy `pending_approval={"summary": ..., "details": ...}` shape and asserted
  focus landed on `#approval-allow-once`. It pinned dead behavior with no production
  path to reach it, so it was deleted rather than adapted.
- **Updated** `test_console_approvals_chip_activation_without_pending_approval_notifies`
  in the same file: its `!= "approval-allow-once"` assertion (an id that no longer
  exists) was changed to `!= "approval-submit"`, preserving the same intent — pressing
  the chip with nothing pending must not move focus onto the card's action button.
- **Added** (TDD, `Tests/UI/test_console_mcp_approval.py`):
  - `test_legacy_single_approval_api_was_removed` — pins that `set_approval` no longer
    exists on the class, so it can't quietly come back.
  - `test_card_never_renders_the_retired_single_approval_buttons` — asserts
    `#approval-single-body`/`#approval-allow-once`/`#approval-deny` are absent from the
    DOM in the card's default state, after `set_batch(<calls>, ...)`, and after
    `set_batch([], ...)` (cleared) — i.e. every state `set_batch` (the sole production
    entry point) can put the card in.
- Updated stale docstrings in `test_console_mcp_approval.py` and
  `test_console_parallel_runs.py` that referenced the deleted
  `test_chat_approvals_and_resume.py` file / `set_approval`.

## Test evidence

```
.venv/bin/python -c "import tldw_chatbook; print(tldw_chatbook.__file__)"
# -> /private/tmp/tldw-followups/tldw_chatbook/__init__.py   (confirmed worktree, not main checkout)

pytest Tests/UI/test_console_mcp_approval.py Tests/UI/test_chat_approval_card.py -q
# 2 failed, 41 passed  (both new task-914 tests pass; the 2 failures are pre-existing,
# unrelated — reproduced byte-identical on unmodified HEAD c171ae56a via `git stash`:
#   - test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css
#   - test_request_mcp_approvals_cancellation_records_denied_decision_to_execution_log

pytest Tests/UI/test_console_workbench_contract.py -q
# 36 failed, 22 passed — ALL 36 failures share one root cause, confirmed pre-existing
# (see "Pre-existing environment issue" below), including the two approval-chip tests
# this task touched.

pytest Tests/UI/test_console_parallel_runs.py -q
# 12 failed (all 12) — same pre-existing root cause, confirmed byte-identical on
# unmodified HEAD via `git stash`.

ruff check <every file touched> -> clean (one pre-existing E702 at
  test_console_workbench_contract.py:1467, untouched by this task, not introduced here)
```

## Pre-existing environment issue (NOT caused by this task — flagging for the fleet)

`Tests/UI/test_screen_navigation.py:800`'s shared `fake_runtime_policy()` helper (used
by `_build_test_app()`/`ConsoleHarness`, which most Console UI async tests depend on)
does `app.current_runtime_backend = "local"`. `TldwCli.current_runtime_backend` was
converted to a read-only `@property` (derived from
`_runtime_policy_projection_snapshot`) by commit `1df0c4cb4` ("fix: reconcile privacy
lifecycle eval and packaging hardening", 2026-07-27 09:06:13 -0700 — i.e. today, and an
ancestor of this worktree's HEAD `c171ae56a`), which also deleted the app-side
`self.current_runtime_backend = normalized_backend` assignment but never updated this
test helper to match. Every test that calls `_build_test_app()` now raises
`AttributeError: property 'current_runtime_backend' of 'TldwCli' object has no setter`
before its body ever runs.

Confirmed via `git stash` (reverting to the literal committed worktree HEAD, no task-914
changes applied): identical failures, same count, same message. This is **not a
regression from this task** — it blocks `Tests/UI/test_console_parallel_runs.py`
entirely (12/12) and most of `Tests/UI/test_console_workbench_contract.py` (36/58), and
will block verification for any of the other 5 fleet follow-up tasks that touch Console
UI async tests too. The likely one-line fix (untested, out of scope here): change line
800 to update `app._runtime_policy_projection_snapshot` (or call
`app._publish_runtime_policy_projection(context.state)`) instead of assigning the
read-only property directly.

## Backlog

`backlog/tasks/task-914 - Remove-or-wire-dead-single-approval-buttons.md`: both ACs
checked, status set to Done, Implementation Notes added.
