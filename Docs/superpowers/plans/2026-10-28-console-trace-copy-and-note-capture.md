# Console Trace Copy and Note Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Copy one complete response or the current chat's entire active branch as readable Markdown, and preserve full selected message text when creating a note or creating and opening it.

**Architecture:** Add one pure response trace formatter and compose it for whole-chat copying. Route selection events through the existing Console review/selection controller and selected-response actions through the existing message controller, with explicit callbacks wired in `UI/Console_Modules/wiring.py`. Reuse Notes persistence, note-id navigation, selection state, clipboard transport, and menu lifecycle.

**Tech Stack:** Python >=3.12, pinned Textual 8.2.8, Pydantic, SQLite, pytest/pytest-asyncio, existing Ruff tooling. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-10-28-console-copy-all-response-trace-design.md`

**Backlog:** `backlog/tasks/task-37300 - Console-trace-copying-and-full-message-note-capture.md`.

**Approval state:** The user approved this plan for direct implementation, including Select full message and removing the 4000-character cap from both note actions, on 2026-10-28.

## Global Constraints

- Exact labels: **Copy all**, **Copy all messages**, **Create Note**, **Create+Go to Note**, **Select full message**.
- Copy all targets one enclosing assistant response. Copy all messages includes user messages and every assistant trace on the current chat's active branch, including history outside the mounted window.
- All copied trace sections are expanded Markdown. Include retained displayable thinking, tool arguments/status/results, diffs, and answer text. Never reconstruct missing data or expose provider continuation/raw captures/system prompts/binary payloads.
- No new quote/preview cap on copy or note capture. Keep `SELECTION_QUOTE_CAP = 4000` for prompt-sending actions and bounded highlight previews. Preserve exact selected whitespace and Unicode.
- Notes keep the existing first-line title limit of 48 characters and the conversation/date provenance footer. Create Note stays in Console; Create+Go to Note opens the returned note ID after successful creation.
- No schema, retention, provider, settings, new clipboard backend, or existing whole-conversation export changes. Existing ADRs apply; see ADR check below.
- Use existing `$ds-*` tokens and menu focus/scroll behavior. Do not edit generated CSS by hand. No terminal-convention or global shortcut changes.
- Keep new orchestration in `UI/Console_Modules/`; screen handlers remain delegates. Do not add another controller solely for this feature.
- Workers handle expensive rendering and note I/O. Pin the source session before awaiting. Never cancel an issued note write merely because the user activates an action again.
- Targeted tests only. Preserve private test bootstrap/config ownership; compare inherited failures with the same targeted cases on the base revision.

## Review Focus

1. An old menu survives a tab switch, branch change, or message deletion: it must never copy or save content from a different target.
2. Multiple thinking blocks share a model round: every retained block must appear once, with no loss from round-based deduplication.
3. Tool output contains Markdown fences, HTML-looking text, Unicode, or trailing spaces: copied structure stays readable and content stays exact.
4. A note write succeeds but opening fails, or the action is pressed twice: recovery must not create a duplicate note or lose the saved ID.
5. A long selection's visual preview is abbreviated: copying and both note actions must still receive the complete selection, while prompt quotes stay capped.

## Files and Responsibilities

| File | Responsibility |
|---|---|
| Create `tldw_chatbook/Chat/console_response_trace.py` | Pure response lookup and Markdown rendering for one response or a chat |
| Modify `tldw_chatbook/Chat/console_turn_grouping.py` only if needed | Preserve all same-round thinking blocks in the existing activity-order projection |
| Modify `tldw_chatbook/Chat/console_message_actions.py` | Advertise Copy all in assistant-response overflow actions |
| Modify `tldw_chatbook/Widgets/Console/console_selection_menu.py` | New labels, menu events, capability flags and captured target identity |
| Modify `tldw_chatbook/Widgets/Console/console_transcript.py` | Resolve selection owners, select full body text, emit requests, preserve dismissal/focus |
| Modify `tldw_chatbook/UI/Console_Modules/review_selection.py` | Copy worker, uncapped note capture, source pinning, policy and duplicate-activation guards |
| Modify `tldw_chatbook/UI/Console_Modules/message.py` | Route selected-response Copy all and reuse canonical note write/navigation services |
| Modify `tldw_chatbook/UI/Console_Modules/wiring.py` | Late-bound clipboard, note-write and note-open callbacks |
| Modify `tldw_chatbook/UI/Screens/chat_screen.py` | Thin event delegates only; preserve screen-size ratchet |
| Modify generated `tldw_chatbook/css/widget_defaults_self.tcss` and `widget_defaults_scoped.tcss` if regenerated | Derived menu styles; stage only reproducible generator output |
| Update `Docs/User_Guide/console/text-selection-and-feedback.md` and `chat-basics.md` | Scope, labels, full-message capture and action-specific limits |
| Update `Docs/Architecture/chat-pipeline.md` | Formatter and action ownership entry points |

The tests named below are the executable contracts. New production helpers use public type hints and Google-style docstrings. Do not refactor unrelated message, selection, or note code.

### Task 1: Render a Complete Response and Compose a Chat Trace

**Files:** Create `Chat/console_response_trace.py` and `Tests/Chat/test_console_response_trace.py`; use existing `Chat/console_turn_grouping.py`, `console_chat_models.py`, and `thinking_blocks.py`.

**Interfaces:**

- `find_response_turn(messages: Sequence[ConsoleChatMessage], message_id: str) -> ConsoleAssistantTurn | None`
- `render_response_trace_markdown(turn: ConsoleAssistantTurn) -> str`
- `render_chat_trace_markdown(messages: Sequence[ConsoleChatMessage], *, title: str) -> str | None`
- Lookup consumes assistant/tool message IDs; the transcript resolves thinking-disclosure IDs to their assistant owner before calling it. A missing or standalone tool target returns None.

- [ ] **Write failing formatter tests.** Pin `test_response_trace_preserves_all_sections`, `test_chat_trace_contains_only_supplied_active_branch`, and `test_same_round_thinking_blocks_appear_once`. Assert full strings or exact fenced payload spans, not just heading presence. Include two same-name tool calls with different arguments/results, two thinking blocks in one round, an off-branch sentinel, and a full result longer than its preview.
- [ ] **Verify RED.** Run `python -m pytest Tests/Chat/test_console_response_trace.py -q`; confirm missing formatter behavior is the cause.
- [ ] **Implement the three interfaces.** Reuse assistant-turn grouping and its activity ordering. Group headers use escaped metadata; body/result text remains verbatim. Pick a fence longer than any backtick run in the fenced text. Prefer `tool_output_full`, otherwise available result text; retain known omission markers. Render each retained diff once from its stored before/after text. Use response status to distinguish partial/stopped/failed from completed output; do not label an unfinished body Final answer.
- [ ] **Cover the edge inputs.** Add `test_embedded_fences_unicode_and_whitespace_survive`, `test_unavailable_data_is_labelled_without_private_payloads`, and `test_chat_trace_does_not_duplicate_owned_or_standalone_tools`. Include proprietary/unreadable thinking, missing historical arguments, active partial tools, media labels, quarantined-generation snapshots, and no visible messages. Check private provider fields and system-prompt sentinels never appear.
- [ ] **Fix the existing ordering helper only if the same-round test demonstrates loss.** Keep the change narrow: emit all block refs in envelope order for a round before marking that round emitted. Preserve existing round ownership and unknown-round rules; add the regression to `Tests/Chat/test_console_turn_grouping.py`.
- [ ] **Verify GREEN.** Run `python -m pytest Tests/Chat/test_console_response_trace.py Tests/Chat/test_console_turn_grouping.py -q`.
- [ ] **Commit this tested unit** with message `feat: format complete Console response traces`.

### Task 2: Pin Copy Targets and Copy Through the Existing Clipboard Path

**Files:** Modify `UI/Console_Modules/review_selection.py`, `wiring.py`, and `message.py`; test `Tests/UI/test_console_review_selection_controller.py` and create `Tests/UI/test_console_trace_copy.py`.

**Interfaces:**

- `ConsoleReviewSelectionController.request_trace_copy(self, session_id: str, *, message_id: str | None = None) -> None`; None means the entire source chat.
- Inject `copy_to_clipboard: Callable[[str], None]` through the controller constructor and wiring. Clipboard publication and notifications happen on the UI thread.
- Use `ConsoleChatStore.read_only_messages_for_session(session_id)` for detached active-path snapshots including buffered text. Freeze the tuple once; the formatter must never reread a live store or mounted widgets.

- [ ] **Write failing controller tests.** `test_copy_request_keeps_original_session_after_tab_switch`, `test_copy_missing_response_preserves_clipboard`, and `test_chat_copy_includes_unmounted_history` assert exact source/clipboard content and exclusion of another chat's sentinel. Mutate source data after snapshot capture and verify the formatting job retains the captured values.
- [ ] **Verify RED** with `python -m pytest Tests/UI/test_console_trace_copy.py -q`.
- [ ] **Implement copy orchestration.** Validate the explicit session/target and capture detached snapshots at admission. Run large Markdown rendering with `asyncio.to_thread` under an exclusive `console-trace-copy` worker. Publish only the latest accepted copy request; stale worker results must not overwrite a later clipboard request. Use the original chat title and count only visible conversational messages, with activities counted separately if reported. Missing targets/rendering errors leave the clipboard unchanged.
- [ ] **Route selected-response actions** before the message action service's text-only clipboard rewriting. The More menu's captured message ID remains authoritative; validate that it still belongs to the captured session. Do not run Copy all through the existing `copy` dispatch branch.
- [ ] **Add size and lifecycle tests.** Assert a payload over 100,000 characters is complete, a UI heartbeat runs while formatting is held, and a late earlier worker cannot overwrite the later request. Include snapshot-acquisition cost in the responsiveness check, rather than measuring only formatting; if snapshot copying exceeds the existing 100 ms worker rule, pin the source branch identity before moving that read off-thread, validate it at capture, and refuse a changed branch instead of retargeting. Notifications say the response or message count/UTF-8 size was sent to the clipboard; they must not claim terminal acceptance.
- [ ] **Verify GREEN** with the two controller/UI test files and `Tests/Architecture/test_console_review_selection_controller_boundary.py`, updating its dependency inventory for the new explicit callback.
- [ ] **Commit this tested unit** with message `feat: copy captured Console response and chat traces`.

### Task 3: Expose Copy Actions and Select Full Message

**Files:** Modify `Widgets/Console/console_selection_menu.py`, `console_transcript.py`, `Chat/console_message_actions.py`, and the screen's thin delegates. Test `Tests/UI/test_console_selection_menu.py`, `test_console_trace_copy.py`, `test_console_keyboard_selection.py`, and `Tests/Chat/test_console_message_actions.py`.

**Interfaces:**

- Add `ConsoleTraceCopyRequested(session_id: str, *, message_id: str | None = None)` in the existing selection-message module. The request carries captured session identity, not a later lookup of the active tab.
- Menu capability arguments: `has_copy_all: bool = False`, `has_copy_all_messages: bool = False`, `has_select_full_message: bool = False`; menu messages: `CopyAll`, `CopyAllMessages`, and `SelectFullMessage`.
- `ConsoleTranscript.select_full_message_text(self) -> bool` operates only on the current selectable USER/ASSISTANT body and returns False for empty/stale/ineligible rows.
- Selected assistant More action ID: `copy-all`. Keep the existing `copy` and its `c` binding unchanged.

- [ ] **Write failing mounted journeys.** Assert the relevant labels/ordering and that Copy all from assistant prose, an owned tool, or a thinking disclosure resolves the same owner. Assert Copy all messages is available from user-message selection too. Cover pending/streaming, stopped and failed response snapshots: copying is a read-only action and must not inherit mutation/run gating. Stale menus must not follow a replacement selection.
- [ ] **Verify RED** with `python -m pytest Tests/UI/test_console_trace_copy.py Tests/UI/test_console_selection_menu.py -q`.
- [ ] **Add the menu entries.** Keep Copy selection first, followed by Copy all where eligible, then Copy all messages, then Select full message where eligible. Remaining quote/side-chat/note/review actions keep their relative order. Add tooltips distinguishing response trace, current chat, and message body. For Copy all also use the selected assistant's existing More menu; do not widen primary action rows.
- [ ] **Implement full-body selection.** Capture `[0, len(row.get_display_text()))` through the existing SelectionManager and row protocol. Keep the menu open and focus Copy selection so users can immediately copy or choose a note action. Preserve the original row target; no cross-row selection, new shortcut, or dependence on viewport height. Keep the visual preview bounded and display the complete selected-character count when abbreviated.
- [ ] **Add interaction regressions.** Cover collapsed/expanded equality, full-message selection from a small initial span, a 100,000-character body with offscreen text, plain/Markdown rows, empty/stale rows, and refusal on tool/diff-only rows. Check no duplicate menu IDs or drift in composer/selection focus.
- [ ] **Check small terminals and resizing.** Exercise all actions in a transcript shorter than the expanded action count, shrink/grow cycles, ANSI mode, and keyboard navigation. Add new button IDs to the menu's compact-state selectors. Regenerate styles with `python tldw_chatbook/css/build_css.py` when needed.
- [ ] **Verify GREEN** with the affected mounted suites plus `Tests/UI/test_console_selection_copy.py` and the message-action tests.
- [ ] **Commit this tested unit** with message `feat: expose full-trace copying and full-message selection`.

### Task 4: Preserve Complete Selections in Both Note Actions

**Files:** Modify `Widgets/Console/console_selection_menu.py`, `console_transcript.py`, `UI/Console_Modules/review_selection.py`, `message.py`, `wiring.py`, and the screen delegate. Test `Tests/UI/test_console_selection_end_to_end.py`, `test_console_review_selection_controller.py`, and create `Tests/UI/test_console_selection_notes.py`.

**Interfaces:**

- Extend `ConsoleSelectionNoteRequested(quote: str, *, source_session_id: str | None = None, open_note: bool = False)`; keep positional quote compatibility for existing consumers.
- Extend `request_selection_note(self, quote: str, *, source_session_id: str | None = None, open_note: bool = False) -> None`; resolve omitted identity at admission for legacy callers, never after an await.
- Inject `save_selection_note: Callable[[str, str], Awaitable[str | None]]` (title, content) and `open_selection_note: Callable[[str], bool]` (saved note ID, reports immediate handoff success).
- Reuse `NotesScopeService.save_note(scope=ScopeType.LOCAL_NOTE, title=..., content=..., user_id=..., keywords=["console"])` through the existing Console Notes seam. It returns a note row with ID when keywords are supplied. Preserve configured Notes identity and authorization; do not use the old direct `database.add_note` bypass.

- [ ] **Write failing note tests.** Parameterize lengths 3999, 4000, 4001, 10,000, and 100,000 across both buttons. Assert the exact full selected text plus provenance, title <=48 characters, one write, and correct returned-ID navigation only for Create+Go to Note. Include leading/trailing whitespace, non-ASCII text, and code containing HTML-looking text.
- [ ] **Verify RED** with `python -m pytest Tests/UI/test_console_selection_notes.py -q`; confirm current truncation/missing action is the cause.
- [ ] **Add the labels and requests.** Rename the existing button to Create Note and add Create+Go to Note with the explanatory tooltip. Read `get_selection_text(capped=False)` for notes and remove producer-side `cap_quote`. Preserve existing cleanup and retain the source session/title/date before scheduling work.
- [ ] **Implement strict full-selection capture.** Validate strict nonblank text without a new arbitrary character ceiling; use the common text validator with an explicit allowance for the captured source length. Never strip the stored payload. Remove the note-only `SELECTION_QUOTE_CAP + 64` check. Keep all feedback/side-chat/Add to chat caps and validation unchanged.
- [ ] **Use one save path for both actions.** Reuse the canonical local Notes service and existing `save-as-note` temporary-chat gate. Supply late-bound callbacks from wiring, update controller boundary fixtures, and perform actual blocking local note I/O off the UI loop through the existing owned-worker/DB lifetime conventions. The async save facade contains synchronous local work, so merely awaiting it is not sufficient evidence that it is nonblocking. Do not move notifications or navigation into the worker thread.
- [ ] **Guard duplicate activation and failures.** Track in-flight note creation per source session before scheduling, ignore duplicate activation while that write is outstanding, and release the guard only when issued work settles. A failed save never navigates. Pin the saved ID before handoff; immediate navigation failure reports that the note was saved and offers the existing Open note receipt/retry path with that ID, never re-runs creation. `NavigateToScreen` posting is not an acknowledgement of destination arrival; verify arrival in the integration test instead of inventing an acknowledgement protocol. If the user left the source chat while saving, leave a saved-note receipt rather than stealing their new context.
- [ ] **Verify real persistence and responsiveness.** Use a real temporary SQLite Notes backend through the service for at least one long-selection case. Hold its native write while a UI heartbeat runs; test temporary-session refusal, two rapid activations, save failure, successful save plus failed handoff, source-tab switch, and failure logs with payload sentinels absent.
- [ ] **Verify GREEN** with the new note suite, existing selection end-to-end/controller suites, and `Tests/Architecture/test_console_review_selection_controller_boundary.py`.
- [ ] **Commit this tested unit** with message `feat: save full Console selections and open created notes`.

### Task 5: Targeted Verification, Documentation, and Review

**Files:** User guide and architecture files listed above; new Backlog task; generated styles if changed. Test only affected functionality and governance.

- [ ] **Update documentation.** Explain three copy scopes (highlight, response trace, current active branch), Select full message versus Copy all, exact note labels, uncapped copy/note capture, retained 4000-character prompt quote limit, missing historical data, partial snapshots, and OSC 52 delivery limits. Update the architecture entry for the formatter/controller seams.
- [ ] **Run the targeted functional slice:**

```powershell
python -m pytest Tests/Chat/test_console_response_trace.py Tests/Chat/test_console_turn_grouping.py Tests/Chat/test_console_message_actions.py Tests/UI/test_console_trace_copy.py Tests/UI/test_console_selection_notes.py Tests/UI/test_console_selection_copy.py Tests/UI/test_console_selection_menu.py Tests/UI/test_console_selection_end_to_end.py Tests/UI/test_console_review_selection_controller.py Tests/UI/test_console_keyboard_selection.py -q
```

- [ ] **Run affected architecture and visual governance checks:**

```powershell
python -m pytest Tests/Architecture/test_console_review_selection_controller_boundary.py Tests/Architecture/test_screen_size_ratchet.py Tests/UI/test_design_token_governance.py Tests/UI/test_component_pattern_governance.py -q
python scripts/check_backlog_task_ids.py
git diff --check
```

- [ ] **Check CSS reproduction and scoped Ruff.** Run `python tldw_chatbook/css/check_bundle_sync.py` and use the repository's existing Ruff configuration on changed Python files. Inspect `python -m ruff check <changed Python files>` and `python -m ruff format --check <changed Python files>`; expected result is no new findings/format drift. Compare pre-existing diagnostics against base and do not reformat entire large legacy modules. Do not claim unchanged baseline debt passed.
- [ ] **Perform a disposable-state live Console check.** Generate one response with at least two tool calls, thinking, and a long result using an existing local test server/profile. Copy with activities collapsed and expanded, paste into a disposable text field, and compare payloads. Copy the chat containing an offscreen earlier message. Select a full long body and exercise both note buttons, then verify the exact saved note and the Notes destination. Preserve Console draft/run and return to the same chat. Native terminal/clipboard delivery limits must be reported separately from headless clipboard assertions; on Windows avoid an automation-only `TERM=dumb` environment.
- [ ] **Self-review against all five Review Focus items.** Review the diff for response/branch isolation, source pinning, complete payloads, duplicate writes, finite workers, logging, and no startup-heavy imports. Run additional tests only when a changed path or observed failure justifies them.
- [ ] **Record evidence and task status.** Add implementation notes only after implementation, check ACs only with evidence, and mark Done via Backlog CLI only after the repository DoD is satisfied. If the CLI is unavailable, do not claim CLI completion. Record a lesson only for a real new trap, with the incident that demonstrated it.
- [ ] **Commit the verified documentation and generated artifacts** with message `docs: explain Console trace copying and full note capture`.

## ADR Check

ADR required: no

ADR path: `backlog/decisions/068-console-text-selection-and-annotations.md`; `backlog/decisions/195-console-live-tool-call-presentation.md`; `backlog/decisions/210-console-region-ownership.md`

Reason: implements small additive actions using existing ownership, display data, local Notes authority, navigation and clipboard contracts. Uncapped note capture is requested behavior within the existing selection model; prompt quote policy remains unchanged. No schema, data ownership, capture retention, dependency, provider/runtime, or new navigation protocol is introduced. ADR-031 and ADR-150/161 continue to govern keys and UI.

## Execution Handoff

The user chose direct execution in this chat. The tasks share selection, event,
and controller interfaces; the execution ledger records combined commits and
environment-specific verification rulings. See TASK-37300 for implementation
evidence and any remaining closeout limitations.


## Execution Record

Executed directly in the original checkout on `codex/console-trace-copy`.
The user approved all proposed selection/note behavior before implementation.
Tasks 2–4 share menu payloads, callbacks and test fixtures, so their implementation
is committed together rather than leaving intermediate commits without their
collaborators. Canonical `ThinkingEnvelope` rejects duplicate round ordinals;
its existing contract and valid block ordering were retained.

A real-screen note handoff exposed an initial Library detail-load deadline
starting during mounting. `LibraryNotesController.defer_initial_note_load`
starts that existing deadline after first paint and refuses stale note/source/
view identities. This small additional fix makes the approved Create+Go to Note
journey reach the exact saved note without growing either screen budget.

One fresh whole-implementation review found stale same-session branch copying
and omitted raw-command metadata. Both have failing-then-passing regression
evidence, including preservation of separate raw-command JSON arguments.

Verification uses targeted checks only. Known inherited/platform failures are
reported rather than changing frozen assertions, unrelated AppleDouble files,
or UI ratchet values. Native terminal-driver journeys used retained synthetic
snapshots; OS clipboard acceptance and a full live-model/full-app startup walk
are not claimed. Backlog CLI closeout remains pending because the CLI is absent.
See the [verification record](../../Development/evidence/2026-10-28-console-trace-copy-verification.md)
and TASK-37300 implementation notes for commands, evidence and rulings.

## Pre-PR Review Follow-up

The user requested a fresh review, fixes and a PR against `dev`. Rebased the
feature cleanly onto `3efe8476cebf70eceafa605d1abed76a4aad4e0e` before review.
The five confirmed review findings extend existing acceptance criteria:

1. Invalidate pending trace publication after any later in-app clipboard write,
   including identical clipboard text.
2. Preserve canonical body Markdown in full-message selection with rendering
   disabled, while retaining ordinary drag selection's rendered text domain.
3. Require foreground Console ownership for post-save navigation and failed
   handoff receipts, even when the source session ID is unchanged. Recheck
   scheduled receipts at execution after possible screen/chat changes.
4. Pin deferred Library note loading to its complete captured route and Notes
   rail, so a Search route cannot start the former note load.
5. Bound and scroll the response More menu, preserving keyboard/mouse reachability
   across short-terminal shrink/grow cycles.

Each finding has failing-then-passing regression evidence. The real-screen
handoff test reads the actual saved note through the canonical Notes port before
mounting Library, then reuses that typed snapshot during its UI load deadline.
This isolates exact-note navigation/projection from cold SQLite/worker latency;
it does not establish cold-start latency. The production deadline is unchanged.
No ADR or dependency change is required. Publish the verified branch as a PR
against `dev`; keep formal Backlog CLI closeout pending while its CLI is absent.
