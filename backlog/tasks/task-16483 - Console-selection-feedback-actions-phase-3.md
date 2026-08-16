---
id: TASK-16483
title: Console selection feedback actions phase 3
status: In Progress
assignee: []
created_date: '2026-08-16 05:27'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review feedback actions (Request changes / LGTM / Comment) on selections in agent output, routed as the next user message via the prompt queue
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Diff rows support line-granularity selection (unified-diff projection, snap to whole diff lines, reverse-video strip)
- [x] #2 Menu offers Request changes / LGTM / Comment only when the selection sits in agent output (tool/diff rows)
- [x] #3 Without an active run, Request changes + LGTM render disabled with a visible hint; Comment stays enabled
- [x] #4 Comment modal collects an optional comment (empty submit sends without a comment; cancel/escape abandons)
- [x] #5 Feedback composes header + quoted selection + optional comment and routes via the prompt queue as the next user message (queues behind an active run; composer draft untouched)
- [x] #6 All selection/feedback/dismissal/transcript suites green; no new failures vs pre-existing baselines
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Per Docs/superpowers/plans/2026-08-15-console-selection-feedback-phase3.md (tasks 1-6: diff-row selection protocol → menu feedback entries + run gating → transcript wiring → comment modal → screen handler + prompt-queue dispatch → wrap-up)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR: backlog/decisions/068-console-text-selection-and-annotations.md (phase-3 consequence line added). Plan: Docs/superpowers/plans/2026-08-15-console-selection-feedback-phase3.md. Spec: Docs/superpowers/specs/2026-08-14-console-selection-annotations-design.md §3 (Feedback and annotations; Comment persistence deferred to phase 4 per §7 phasing).

Phase 3 implemented across commits 0ba93f948..8a7b11315 (+ this wrap-up) on feat/console-selection-feedback:

- **Diff-row selection protocol** (`tldw_chatbook/Widgets/Console/console_transcript.py`): `ConsoleToolDiffRow` implements the 4-method selection protocol; display/selection domain is the deterministic `difflib.unified_diff(old, new, path)` projection, line-granularity (offsets snap outward to whole diff lines via `_diff_cell_to_offset`, reusing the markdown row's even-distribution + nearest-clamp mapping), highlight is a reverse-video strip below the DiffView (DiffView internals never restyled). Diff content is immutable so there is no streaming clamp. Transcript row unions (`_selection_row_for`, `_active_selection_row`, `on_mouse_move` clear-others, `_selection_offset_for`) widened to accept the row.
- **Menu feedback entries + run gating** (`console_selection_menu.py`): ctor takes `feedback_available` / `run_active`; `RequestChanges`/`Lgm`/`Comment` messages post to the owner; when run-gated, Request changes + LGTM render `disabled` with the dim hint line `No active run — start a run to send review feedback` (also carried on tooltips); key-nav up/down and initial focus skip disabled buttons (focusing a disabled widget drops focus and strands navigation).
- **Transcript wiring** (`console_transcript.py`): `_text_selected` derives `feedback_available` from the origin row (TOOL role or diff row) and `run_active` via `getattr(self.screen, "_current_console_run_status_value")` in the active-run set; the three handlers post `ConsoleSelectionFeedbackRequested(action, quote=cap_quote(...))` with the same cleanup as Add to chat (selection cleared, menu removed). Empty-quote requests no-op silently.
- **Comment modal** (`console_feedback_comment_modal.py`, new): `ConsoleFeedbackCommentModal(SafeModalDismissMixin, ModalScreen[str | None])` — read-only capped quote preview, one optional single-line Input, Cancel/Submit. Contract: dismiss `""` = submit WITHOUT a comment (comment is optional; feedback still flows), `None` = Cancel/Escape/backdrop abandons the feedback entirely. Escape/backdrop ride `SafeModalDismissMixin` (task-16211). Added to the modal-dismissal inventory (reachable count 38 → 39).
- **Screen handler + dispatch** (`tldw_chatbook/UI/Screens/chat_screen.py`): `on_console_selection_feedback_requested` stops the event, guards blank quotes, captures action+quote and `run_worker`s the async flow; the worker awaits the modal via `self.app.push_screen_wait`, composes `[Request changes]|[LGTM]|[Comment]` header + `> `-quoted lines (blank lines a bare `>`, mirroring `insert_quote`) + verbatim optional comment, then dispatches via `await self._prompt_queue.dispatch(text)`.

Decisions:

- **Queue-dispatch seam (load-bearing).** `submit_draft` refuses during active runs (exactly when review feedback is sent) and Send-synthesis would clobber a mid-typed composer draft. `_prompt_queue.dispatch` is the ONLY send seam: it queues behind an active run, sends immediately otherwise, and owns every refusal/block toast. The composer draft is never read or written (asserted by the routing tests). Locked by T5 tests.
- **Worker-context push.** `push_screen_wait` raises `NoActiveWorker` when awaited directly inside a message handler, so the sync `@on` handler captures action+quote and runs the async flow as a worker in the non-exclusive group `console-selection-feedback` (cancelling a dialog-waiting worker would strand a live modal — the `EvalsScreen._on_delete_bench_pressed` pattern).
- **`""`/`None` modal contract.** Submit with empty input dismisses `""` (send header+quote, no comment block); Cancel/Escape/backdrop dismiss `None` (send nothing). The T4 modal originally conflated these (`comment.strip() or None`), making an optional comment effectively required — fixed in 8a7b11315.
- **Dead-zone dismissal vs disabled-button clicks.** A click on menu chrome that is not an action button (border/padding) is a popover dismissal (`Dismissed` posted, menu removed — verified empirically in the wrap-up). A click on a run-gated (disabled) Request changes/LGTM button is a silent no-op: the menu stays mounted and nothing posts (Textual blocks `Button.Pressed` on disabled widgets and the menu's dead-zone walk stops at any `Button`, disabled included). Verified empirically in the wrap-up; the earlier task-2 report line claiming a disabled click dismisses via the menu container was wrong — the disabled button itself is the dead-zone exception.

**Tests** (all green): selection suites `Tests/UI/test_console_selection_{core,rows,transcript,menu,end_to_end,app_smoke}.py` + `Tests/UI/test_console_feedback_comment_modal.py` + `Tests/UI/test_console_modal_dismissal.py` (inventory equality, count 39) + `Tests/UI/test_console_side_chat_modal.py` + `Tests/Chat/test_console_message_actions.py` — 348 passed. Baselines: `test_console_native_transcript.py` + `test_console_transcript_region.py` 108 passed (green); `test_console_native_chat_flow.py` (2 failed) + `test_console_transcript_markdown_widget.py` (4 failed) — exactly the pre-existing branch baseline, counts unchanged.

**Ruff**: `uvx ruff check` on all 9 files phase 3 touched — 216 findings, ZERO branch-owned (verified two ways: findings intersected against phase-3 changed line ranges = 0 hits; pre-phase-3 versions of the files carry 219, so the total went down).

**Live-terminal-only verification outstanding** (same as phases 1-2, per `backlog/docs/lessons-live-verification.md`): drag-select on a diff row, gated-button look/feel, and modal→queue round-trip against a real provider in a real terminal. Task stays In Progress pending that live spike.
<!-- SECTION:NOTES:END -->
