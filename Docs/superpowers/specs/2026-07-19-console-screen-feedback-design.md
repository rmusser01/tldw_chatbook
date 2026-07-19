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

`ConsoleTranscript` adds a guarded `on_click` handler on the scroll container. The handler inspects `event.control` and clears the current selection only when the click originated on the scroll container itself or on explicitly defined negative-space widgets (e.g., spacer rules). Clicks that bubble up from message rows, action buttons, action-row backgrounds, action-help rows, rule separators, empty-state panels, or scrollbar widgets are ignored.

- Single click on a message row still selects that message.
- Clicks on the scrollbar are ignored (verified during implementation; if they bubble, exclude by scrollbar CSS class).
- The existing `escape` keybinding continues to work.
- No double-click behavior is added, to avoid conflicting with text selection.

### 4.2 Agent Turn-Control Fix

**Files:**
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_agent_bridge.py`

The agent runtime (`AgentService.run_turn()` / `run_agent_loop()`) already supports multiple model-turns. The bug is in how the console controller consumes the result. We will harden the return path:

1. **Run the agent loop on the worker thread.** `ConsoleAgentBridge.run_reply()` continues to return a `RunOutcome` dataclass (`tldw_chatbook/Agents/agent_models.py`).
2. **Return the result to the main asyncio/Textual loop.** All store mutations happen on the main thread; UI mutations from inside the bridge continue to use `call_from_thread` where needed.
3. **Normalize all `RunOutcome` statuses:**
   - `RUN_DONE` with `final_text` → update placeholder content, mark `complete`.
   - `RUN_DONE` with empty `final_text` → update placeholder with a "No response was generated" note, mark `complete`. This fallback text is persisted as the assistant message content so retries/regenerations operate on a non-empty message.
   - `RUN_ERROR` / `RUN_STUCK` (covers errors and budget/loop/max-steps failures) → mark placeholder `failed`, set `error_detail` from `RunOutcome`.
   - `RUN_CANCELLED` → mark placeholder `failed` with a "stopped/cancelled" detail.
   - Unknown status → mark `failed`, log the full outcome.
4. **Placeholder-missing fallback:** if the placeholder is missing when the run completes, check whether the runtime already wrote an assistant message. If so, update that message; otherwise log an error and append a new assistant message at the end using the next available turn ID.
5. **Stop/cancel mid-run:** respect the existing `_stop_requested` / `_active_cancel_event` via the `should_cancel` closure. Finalize the placeholder with whatever partial result exists or a `stopped` status.
6. **Diagnostics:** add structured `loguru` events at start, each model turn, each tool call, and final commit.

### 4.3 Context Viewer Modal

**New file:** `tldw_chatbook/Widgets/Console/console_context_modal.py`

A new modal, opened from:
- Command palette entry: **View chat context**.
- Keybinding: `ctrl+shift+p` (avoids `ctrl+shift+c` terminal-copy and `f10` menu conflicts).

The modal has two tabs.

#### Tab 1: Current Context

Shows the session transcript as stored in `ConsoleChatStore`:
- Role, content, status, and variant info per message.
- Empty-state message when no conversation is active.

#### Tab 2: Next-Send Context

Shows the assembled provider payload that would be sent if the user submitted a given draft, computed by a new async method on `ConsoleChatController`:

```python
@dataclass
class ConsoleContextSnapshot:
    current_messages: list[ConsoleChatMessage]
    next_send_payload: dict[str, Any]

async def build_context_snapshot(
    self,
    draft: str,
    attachments: Iterable[MessageAttachment] | None = None,
    staged_sources: Iterable[StagedSource] | None = None,
) -> ConsoleContextSnapshot:
    ...
```

The snapshot includes:
- Leading system prompt (session prompt + agent operating prompt when agent mode is on).
- Provider-formatted messages, with image data replaced by placeholders.
- Skill substitution preview for the final user message. Skills with side effects are **not** executed for the preview; the viewer shows the raw skill command and a note that the resolved substitution is visible only at send time.
- Chat dictionary substitutions applied to the final user message.
- Native tool schemas. Live MCP tool catalog composition is out of scope for this spec; if MCP is enabled, the viewer shows a note that MCP tools are configured.
- RAG/source context attachments when staged.
- Approximate token estimate in the header (word count × 1.3, replaced by tiktoken if available as an optional dependency).

**UI details:**
- Content is computed in a Textual worker with a loading indicator.
- Rendered view uses collapsible sections.
- Toggle for raw JSON view.
- Copy-to-clipboard and save-to-file actions.
- Secrets redaction: values for keys matching `api_key`, `apikey`, `token`, `password`, `secret`, `bearer` are replaced with `"[redacted]"` in the rendered view. Raw JSON copy/save is allowed without extra confirmation because the snapshot is local-only.
- Refresh button to rebuild the snapshot.
- Warning and disabled refresh while a response is in progress.
- "Save to file" fallback if the rendered content exceeds 1 MiB.

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
- `ConsoleChatController.build_context_snapshot(draft, attachments, staged_sources)` — async public method returning `ConsoleContextSnapshot`.
- `ConsoleContextSnapshot` dataclass — `current_messages` + `next_send_payload`.
- `ChatScreen.BINDINGS` + `ConsoleCommandProvider` — new **View chat context** entry.
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
- Placeholder missing at completion → check if runtime already wrote an assistant message; update it, or append a new assistant message at the end using the next available turn ID.
- Tool call outside allow-list → runtime handles refusal; controller commits final runtime result.
- Stop/cancel during run → finalize placeholder with partial result or a "stopped/cancelled" status.
- `RUN_DONE` with empty `final_text` → persisted fallback note "No response was generated" so retries/regenerations see a non-empty message.

### Context Viewer
- Context build failure → show error message instead of crashing.
- No messages to send → show "No messages to send" empty state.
- No active conversation → show empty-state message in both tabs.
- Response in progress → warn that snapshot may change; disable refresh.
- Content exceeds 1 MiB → offer "Save to file" instead of rendering inline.
- Image data → shown as `[image: <filename>, <size> bytes]` placeholders.
- Secrets → redacted in rendered view for keys matching `api_key`, `apikey`, `token`, `password`, `secret`, `bearer`.

---

## 7. Testing

### 7.1 Selection Clearing
- Mount `ConsoleTranscript` with sample messages.
- Select a message; simulate click on scroll background; assert `selected_message_id` is `None` and action row is removed.
- Click a message row; assert selection is set.
- Click an action button; assert selection is **not** cleared and action fires.
- Click the action-row background (not a button); assert selection is **not** cleared.

### 7.2 Agent Turn-Control
- **Bridge level:** mock `AgentService.run_turn()` to return `RunOutcome` objects with statuses `RUN_DONE`, `RUN_ERROR`, `RUN_STUCK`, `RUN_CANCELLED`. Verify `ConsoleAgentBridge.run_reply()` returns the `RunOutcome` and that `final_text` is correctly populated.
- **Controller level:** use a real `ConsoleChatStore` with a mocked agent bridge and verify placeholder lifecycle:
  - `RUN_DONE` with `final_text` → placeholder `complete` with content.
  - `RUN_DONE` with empty `final_text` → placeholder `complete` with fallback note.
  - `RUN_ERROR` / `RUN_STUCK` → placeholder `failed` with detail.
  - `RUN_CANCELLED` → placeholder `failed` with stopped/cancelled detail.
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

The following questions were resolved during the spec review and are recorded here for traceability:

1. **RunOutcome shape:** Confirmed that `RunOutcome` uses statuses `RUN_DONE`, `RUN_ERROR`, `RUN_STUCK`, `RUN_CANCELLED` and exposes `final_text`.
2. **Agent result thread boundary:** Store mutations happen on the main asyncio/Textual loop; UI mutations from inside the bridge use `call_from_thread` where needed.
3. **Empty-content fallback persistence:** The fallback note is persisted as the assistant message content so retries/regenerations operate on a non-empty message.
4. **`build_context_snapshot()` signature:** Async method taking `draft`, `attachments`, and `staged_sources`, returning a `ConsoleContextSnapshot` dataclass.
5. **Skill substitution in preview:** Skills with side effects are not executed for the preview; the raw command is shown with a note.
6. **MCP tool visibility:** Live MCP catalog composition is out of scope; only native tool schemas are shown, with a note when MCP is configured.
7. **Keybinding:** `ctrl+shift+p` chosen to avoid terminal conflicts.
8. **Token estimate:** Approximate word count × 1.3, replaced by tiktoken if available.
9. **Secrets redaction:** Values for keys matching `api_key`, `apikey`, `token`, `password`, `secret`, `bearer` are redacted in the rendered view.

---

## 9. Decisions Log

| Decision | Rationale |
|---|---|
| Target native console only | Legacy chat window is deprecated. |
| Click on negative space only; no double-click | Avoids conflict with text selection. |
| `ctrl+shift+p` keybinding for context viewer | Avoids `ctrl+shift+c` terminal-copy and `f10` menu conflicts. |
| Two-tab modal (current + next-send) | Directly matches beta-tester request. |
| Compute context in a worker | Prevents UI blocking on long conversations. |
| Add diagnostics/logging rather than broad defensive fix | Root cause of turn-control bug was unclear; logs enable confirmation and future diagnosis. |
| Skills with side effects not executed in context preview | Prevents unwanted external calls when the user only wants to inspect context. |
| Live MCP catalog composition out of scope | Keeps the initial implementation bounded; can be added later. |
