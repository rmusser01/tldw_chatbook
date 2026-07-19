# Console Screen Feedback — Design Spec

**Date:** 2026-07-19  
**Status:** Approved for implementation  
**Scope:** Native console screen (`ChatScreen` / `ConsoleTranscript`) only. Legacy chat window is deprecated and out of scope.

---

## 1. Context

Beta testers reported three friction points on the console screen:

1. **Selection cannot be cleared by clicking negative space.** Once a chat message is selected, the only way to dismiss the action row is the existing `escape` keybinding.
2. **Turn control is broken when using agents for turns.** When the user asks the model to explain available tools, the assistant errors out with a control-module-related message instead of completing its multi-turn reply.
3. **No way to inspect the full context.** Users cannot see the whole current conversation context, nor can they see the assembled provider payload (including hidden prompts, tool schemas, substitutions, and RAG context) that will be sent on the next message.

This spec addresses all three items with minimal, targeted changes.

---

## 2. Goals

- Allow users to clear the selected console message by clicking empty/negative space in the transcript.
- Ensure the local console agent mode can complete multi-turn replies (model → tool call → tool result → model again, etc.) without the controller forcing a user turn.
- Provide a read-only, two-tab context viewer modal that shows (a) the stored transcript context and (b) the exact next-send provider payload.

## 3. Non-Goals

- Do not change the legacy chat window (`ChatWindowEnhanced`).
- Do not add persistent side panels or a full debug console.
- Do not expose unredacted secrets by default.
- Do not auto-update the context viewer while it is open.

---

## 4. Design

### 4.1 Selection Clearing

**File:** `tldw_chatbook/Widgets/Console/console_transcript.py`

`ConsoleTranscript` adds a guarded `on_click` handler on the scroll container. The handler clears the current selection only when the click originated on the scroll background or another defined negative-space widget. Clicks that bubble up from message rows, action buttons, or the action-row background are ignored.

- Single click on a message row still selects that message.
- Clicks on the scrollbar are ignored.
- The existing `escape` keybinding continues to work.
- No double-click behavior is added, to avoid conflicting with text selection.

### 4.2 Agent Turn-Control Fix

**Files:**
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_agent_bridge.py`

The agent runtime (`AgentService.run_turn()` / `run_agent_loop()`) already supports multiple model-turns. The bug is in how the console controller consumes the result. We will harden the return path:

1. **Run the agent loop on the worker thread**, capture `(outcome, final_content, run_summary)` from `ConsoleAgentBridge.run_reply()`.
2. **Commit the result on the main Textual thread** via `call_from_thread`.
3. **Normalize all outcomes:**
   - `final_answer` with content → update placeholder content, mark `complete`.
   - `final_answer` with empty content → update placeholder with a "No response was generated" note, mark `complete`.
   - `error` / `cancelled` / `max_steps` → mark placeholder `failed`, set `error_detail`.
   - Unknown outcome → mark `failed`, log the full outcome.
4. **Placeholder-missing fallback:** if the placeholder is missing when the run completes, check whether the runtime already wrote an assistant message. If so, update it; otherwise append a new assistant message.
5. **Stop/cancel mid-run:** respect the existing `_stop_requested` / `_active_cancel_event` via the `should_cancel` closure. Finalize the placeholder with whatever partial result exists or a `stopped` status.
6. **Diagnostics:** add structured `loguru` events at start, each model turn, each tool call, and final commit.

### 4.3 Context Viewer Modal

**New file:** `tldw_chatbook/Widgets/Console/console_context_modal.py`

A new modal, opened from:
- Command palette entry: **View prompt context**.
- Keybinding: `f10` (configurable; avoids `ctrl+shift+c` terminal-copy conflict).

The modal has two tabs.

#### Tab 1: Current Context

Shows the session transcript as stored in `ConsoleChatStore`:
- Role, author, content, status, and variant info per message.
- Empty-state message when no conversation is active.

#### Tab 2: Next-Send Context

Shows the exact assembled provider payload that will be sent on the next message, computed by a new `ConsoleChatController.build_context_snapshot()` method:
- Leading system prompt (session prompt + agent operating prompt when agent mode is on).
- Provider-formatted messages, with image data replaced by placeholders.
- Skill substitution result applied to the final user message.
- Chat dictionary substitutions applied to the final user message.
- Agent tool schemas / MCP tool descriptions when agent mode is enabled.
- RAG/source context attachments when staged.
- Token estimate in the header.

**UI details:**
- Content is computed in a Textual worker with a loading indicator.
- Rendered view uses collapsible sections.
- Toggle for raw JSON view.
- Copy-to-clipboard and save-to-file actions.
- Secrets are lightly redacted in the rendered view.
- Refresh button to rebuild the snapshot.
- Warning and disabled refresh while a response is in progress.
- "Save to file" fallback if the rendered content exceeds a configurable size threshold.

---

## 5. Architecture & Data Flow

```
User Interaction
       │
       ├──► ConsoleTranscript.on_click (scroll background) ──► clear_selection()
       │
       ├──► ChatScreen keybinding/command ──► ConsoleContextModal
       │                                         │
       │                                         ▼
       │                              worker: build_context_snapshot()
       │                                         │
       │                                         ▼
       │                              render Current / Next-Send tabs
       │
       └──► ConsoleChatController.submit_draft()
                            │
                            ▼
                 _stream_assistant_response()
                            │
              agent mode? ──┬── yes ──► _run_agent_reply()
                            │              │
                            │              ▼
                            │     ConsoleAgentBridge.run_reply()
                            │              │
                            │              ▼
                            │        AgentService.run_turn()
                            │              │
                            │              ▼
                            │     run_agent_loop() (multi-turn capable)
                            │              │
                            │              ▼
                            │     return RunOutcome to bridge/controller
                            │              │
                            │              ▼
                            │     normalize & commit on main thread
                            │
                            └── no ──► provider_gateway.stream_chat()
```

**New/changed surfaces:**
- `ConsoleChatController.build_context_snapshot()` — public method returning current transcript + next-send payload.
- `ChatScreen.BINDINGS` + `ConsoleCommandProvider` — new context-viewer entry.
- `ConsoleTranscript` — internal click handler using existing selection API.

---

## 6. Error Handling & Edge Cases

### Selection Clearing
- No selection → handler is a no-op.
- Click on scrollbar → ignored.
- Click on message row/action button/action-row background → does not clear.
- Streaming/pending messages → selection of earlier messages is unaffected.
- Composer focus does **not** clear transcript selection (keeps scope minimal).

### Agent Turn-Control
- Unhandled exception in worker thread → caught, logged, placeholder marked `failed`.
- Placeholder missing at completion → check if runtime already wrote an assistant message; update it or append a new one.
- Tool call outside allow-list → runtime handles refusal; controller commits final runtime result.
- Stop/cancel during run → finalize placeholder with partial result or `stopped` status.

### Context Viewer
- Context build failure → show error message instead of crashing.
- No messages to send → show "No messages to send" empty state.
- No active conversation → show empty-state message in both tabs.
- Response in progress → warn that snapshot may change; disable refresh.
- Content exceeds size threshold → offer "Save to file" instead of rendering inline.
- Image data → shown as `[image: <filename>, <size> bytes]` placeholders.
- Secrets → redacted in rendered view; raw JSON copy/save may require confirmation.

---

## 7. Testing

### 7.1 Selection Clearing
- Mount `ConsoleTranscript` with sample messages.
- Select a message; simulate click on scroll background; assert `selected_message_id` is `None` and action row is removed.
- Click a message row; assert selection is set.
- Click an action button; assert selection is **not** cleared and action fires.

### 7.2 Agent Turn-Control
- **Bridge level:** mock `AgentService.run_turn()` and verify `ConsoleAgentBridge.run_reply()` returns normalized results for multi-turn success, empty content, error, and cancellation.
- **Controller level:** use a real `ConsoleChatStore` with a mocked agent bridge and verify placeholder lifecycle:
  - Multi-turn success → placeholder `complete` with content.
  - Empty content → placeholder `complete` with fallback note.
  - Error/cancel → placeholder `failed` with detail.
  - Exception in worker → caught, logged, placeholder `failed`.

### 7.3 Context Viewer
- Test `ConsoleChatController.build_context_snapshot()` returns correct current transcript and next-send payload.
- Test immutability: snapshot does not mutate store or draft.
- Test modal opens from command palette and keybinding.
- Test worker completion and rendering of both tabs.
- Test image placeholders and secrets redaction.
- Test empty-state and in-progress warnings.

### 7.4 Manual Smoke Tests
- Select a console message, click negative space, verify action row hides.
- Enable agent mode, ask "explain all tools available," verify response completes without control error.
- Open context viewer with a staged source and verify RAG context is visible.

---

## 8. Open Questions

None at time of approval. All clarifying questions were resolved during design review.

---

## 9. Decisions Log

| Decision | Rationale |
|---|---|
| Target native console only | Legacy chat window is deprecated. |
| Click on negative space only; no double-click | Avoids conflict with text selection. |
| `f10` keybinding for context viewer | Avoids `ctrl+shift+c` terminal-copy conflict. |
| Two-tab modal (current + next-send) | Directly matches beta-tester request. |
| Compute context in a worker | Prevents UI blocking on long conversations. |
| Add diagnostics/logging rather than broad defensive fix | Root cause of turn-control bug was unclear; logs enable confirmation and future diagnosis. |
