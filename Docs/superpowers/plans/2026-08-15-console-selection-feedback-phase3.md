# Console Selection Feedback Actions — Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Request changes | LGTM | Comment` entries in the selection menu when the selection sits in agent output (tool/diff rows), each composing a structured feedback message (action header + quoted selection + optional comment) routed as the NEXT user message to the agent session — queued behind an active run, sent immediately otherwise. No-run fallback: Request changes/LGTM disabled with a visible hint; Comment always available (routes the same way; persistence is phase 4).

**Architecture:** `ConsoleToolDiffRow` gains the same 4-method selection protocol as the other rows (domain = deterministic unified-diff projection, line granularity, reverse-video strip highlight). The menu grows three gated buttons posting new messages to the owning transcript, which posts `ConsoleSelectionFeedbackRequested(action, quote)` to the screen. The screen pushes a small comment modal (clone of the rename modal), composes the structured text, and dispatches through the existing **prompt-queue seam** (`_prompt_queue.dispatch`) — never touching the live composer draft and never calling `submit_draft` directly (which refuses during active runs).

**Tech Stack:** Textual 8.2.8 (SafeModalDismissMixin modal, screen-mounted menu), existing `ConsolePromptQueueUIController.dispatch`, pytest.

**Spec:** `Docs/superpowers/specs/2026-08-14-console-selection-annotations-design.md` §3 (Feedback and annotations), §4 (error handling), §5 (testing), §7 phase 3. ADR: `backlog/decisions/068-console-text-selection-and-annotations.md`.

```text
ADR required: no
ADR path: N/A — direct implementation of spec §3 under ADR-068; no schema change (phase 4), no new cross-module boundary (prompt-queue seam is existing).
Reason: routing-only feature reusing established seams.
```

## Global Constraints

- Feedback routes through `await self._prompt_queue.dispatch(text)` — the ONLY send seam (queues behind an active run, sends immediately otherwise, rides every refusal/block toast). NEVER `submit_draft` directly (it rejects during runs), NEVER the composer draft (would clobber in-progress typing), NEVER synthesizing Send-button presses.
- Structured message shape (all three actions): a header line, the `> `-quoted capped selection, and the optional comment block — exact template in Task 5.
- Quote capped by `cap_quote` (4000) before leaving the transcript; empty-quote requests no-op silently (mirrors Add-to-chat guard).
- Active-run test: `_current_console_run_status_value()` ∈ {validating, streaming, checking_citations, retrying}. When false: Request changes and LGTM render **disabled with a dim visible hint** ("No active run — start a run to send review feedback"); Comment stays enabled.
- The menu's up/down keyboard cycle must skip disabled buttons (currently filters `display` only).
- Diff-row selection domain: `difflib.unified_diff(old.splitlines(), new.splitlines(), path)` projection (keepends), line granularity (snap to whole diff lines), highlight via the reverse-video strip pattern (never restyle DiffView internals). Diff content is immutable — no streaming clamp; row removal rides the existing reconciliation guard.
- `_selection_row_for` returns the widened row union; `_active_selection_row`, `on_mouse_move` clear-others, and `_selection_offset_for` (new diff-line mapper) all handle the new type.
- Baselines: no new failures in any existing suite (`test_console_native_transcript` 3 / markdown-widget 4 / chat-flow 1 pre-existing).

---

### Task 1: Diff-row selection protocol

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (`ConsoleToolDiffRow` ~:1687-1731; `_selection_row_for` ~:3299; `_selection_offset_for` ~:3327; `_active_selection_row` ~:3608; `on_mouse_move` clear-others ~:3458)
- Test: `Tests/UI/test_console_selection_rows.py` (extend)

**Interfaces:**
- Consumes: `cap_quote`; the strip pattern from `ConsoleMarkdownMessage` (~:1231, :1310).
- Produces: `ConsoleToolDiffRow` implements `get_display_text() -> str` (unified-diff projection of `self._diff`, deterministic), `get_selection_text() -> str` (line-snapped, cap_quote'd), `set_selection_range(start, end)` (snap outward to whole diff lines; strip below the DiffView), `clear_selection()`; transcript accepts diff rows everywhere the row union is used.

- [ ] **Step 1: Failing tests** — protocol unit tests on a mounted diff row (build via a real `ConsoleChatMessage` with `role=TOOL`, `tool_diff=(path, old, new)`): display text is the unified diff; partial offsets snap to whole lines; strip shows/hides; cap applies; transcript `_selection_row_for` resolves a press on the DiffView to the diff row; a drag over it arms and extends (line granularity).
- [ ] **Step 2: Verify fail** (AttributeError: no protocol on the row).
- [ ] **Step 3: Implement** — row methods + `_diff_cell_to_offset(text, height, cell_x, cell_y)` mapper (reuse `_markdown_cell_to_offset` semantics — line distribution + nearest clamp); widen the isinstance unions at the four transcript sites; strip composed once, display-toggled.
- [ ] **Step 4: Verify pass** + full rows/transcript suites green.
- [ ] **Step 5: Commit** — `feat(console): line-level selection on tool diff rows`

### Task 2: Menu feedback entries + gating

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_selection_menu.py`
- Test: `Tests/UI/test_console_selection_menu.py` (extend)

**Interfaces:**
- Consumes: Task 1 (row union — menu gating derives from the origin row, passed by the transcript).
- Produces: ctor `feedback_available: bool = False, run_active: bool = False`; nested Messages `RequestChanges` / `Lgm` / `Comment`; buttons `#console-selection-request-changes`, `#console-selection-lgm`, `#console-selection-comment` (rendered after Ask in Side Chat, only when `feedback_available`); disabled Request/LGTM + hint Static when `not run_active`; key-nav skips disabled buttons.

- [ ] **Step 1: Failing tests** — three new buttons present/absent by `feedback_available`; disabled state by `run_active`; hint line visible only when gated; up/down cycle skips disabled; each enabled button posts its message (owner posting when owner passed).
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** per interfaces (hint as a dim `Static` inside the menu; disabled buttons use `Button(disabled=True)` + tooltip carrying the hint text).
- [ ] **Step 4: Verify pass** + existing menu suite green.
- [ ] **Step 5: Commit** — `feat(console): selection menu feedback actions with run gating`

### Task 3: Transcript feedback wiring

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/Widgets/Console/console_selection_menu.py` (message class home: `ConsoleSelectionFeedbackRequested`)
- Test: `Tests/UI/test_console_selection_end_to_end.py` (extend)

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: `ConsoleSelectionFeedbackRequested(Message)` with `action: str` (`"request-changes" | "lgm" | "comment"`) and `quote: str`; `_text_selected` derives `feedback_available` (origin row is TOOL role or a diff row) and `run_active` (via `getattr(self.screen, "_current_console_run_status_value", None)` in the active set); three `@on` handlers mirroring `_selection_add_to_chat` cleanup.

- [ ] **Step 1: Failing tests** — drag on a TOOL plain row → menu shows feedback entries; drag on a normal user row → not shown; `run_active` False path mounts gated buttons; pressing Comment (enabled) posts `ConsoleSelectionFeedbackRequested(action="comment", quote=capped)` to the app (app-level capture harness); cleanup (selection cleared, menu removed).
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** per interfaces.
- [ ] **Step 4: Verify pass** + suites green.
- [ ] **Step 5: Commit** — `feat(console): transcript posts feedback requests from agent-output selections`

### Task 4: Comment modal

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_feedback_comment_modal.py`
- Modify: `Tests/UI/test_console_modal_dismissal.py` (inventory + contract rows — the AST test enforces it)
- Test: `Tests/UI/test_console_feedback_comment_modal.py` (new)

**Interfaces:**
- Consumes: `SafeModalDismissMixin` (`Widgets/modal_dismissal.py`); skeleton `ConsoleRenameSessionModal` (`console_rename_session_modal.py:22-130`).
- Produces: `ConsoleFeedbackCommentModal(SafeModalDismissMixin, ModalScreen[str | None])` — `__init__(*, action: str, quote: str)`; a read-only quote preview, one `Input` (single-line comment; empty allowed = comment omitted), Cancel/Submit; `dismiss(comment or None)`; Escape/backdrop ≡ Cancel.

- [ ] **Step 1: Failing tests** — dismiss(None) on escape/backdrop/cancel; dismiss(text) on submit + Enter; empty input dismisses None; quote preview shows the (capped) quote; dismissal-inventory rows added.
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** (~130-line clone with the preview added).
- [ ] **Step 4: Verify pass** + full dismissal file green.
- [ ] **Step 5: Commit** — `feat(console): feedback comment modal`

### Task 5: Screen handler + dispatch

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (handler beside `_console_selection_quote_requested` ~:19792)
- Test: `Tests/UI/test_console_selection_end_to_end.py` (extend)

**Interfaces:**
- Consumes: `ConsoleSelectionFeedbackRequested`; `ConsoleFeedbackCommentModal`; `await self._prompt_queue.dispatch(text)` (prompt_queue.py:655-726).
- Produces: `@on(ConsoleSelectionFeedbackRequested)` — event.stop; empty-quote guard; `await self.push_screen_wait(ConsoleFeedbackCommentModal(...))`; compose and dispatch:

```text
[Request changes]  /  [LGTM]  /  [Comment]   # action header line
> <quoted selection lines>                   # "> "-prefixed, capped
<optional comment, only when provided>       # appended verbatim
```

- [ ] **Step 1: Failing tests** — handler with a stubbed `_prompt_queue` (record dispatch calls): composed text matches the template for each action with/without a comment; empty quote dispatches nothing; modal escape dispatches nothing; the composer draft is UNTOUCHED (assert draft_text unchanged).
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** per interfaces (dispatch failure toasts ride the queue controller's own paths — no extra handling).
- [ ] **Step 4: Verify pass** + full end-to-end + smoke suites.
- [ ] **Step 5: Commit** — `feat(console): route selection feedback as next user message via prompt queue`

### Task 6: Wrap-up

- [ ] **Step 1:** Full selection + feedback + dismissal + transcript suites green; `uvx ruff check` on touched files (branch-owned only); baselines 3/1/4 unchanged.
- [ ] **Step 2:** Backlog task (next free id) with honest ACs + Implementation Notes linking ADR-068/spec/plan; ADR-068 one-line consequence note (phase 3 landed; routing seam = prompt queue; phase 4 adds persistence). Do NOT mark Done (live spike pending, like prior phases).
- [ ] **Step 3:** Commit — `feat(console): feedback actions wrap-up docs`

## Self-Review Notes

- Spec §3 coverage: entries+gating (T2/T3), structured message + comment modal (T4/T5), routing via composer/queue seam (T5), no-run fallback (T2), Comment phase-3 scope = routing only (persistence deferred to phase 4 per spec phasing).
- The queue seam decision (dispatch over submit_draft/Send-synthesis) is load-bearing: submit_draft refuses during runs; Send-synthesis clobbers live drafts. Locked by T5 tests.
- Keyboard-only users: disabled buttons skipped by nav (T2); Comment reachable without a run.
