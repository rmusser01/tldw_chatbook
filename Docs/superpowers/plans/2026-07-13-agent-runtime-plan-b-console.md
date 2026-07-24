# Agent Runtime — Plan B: Console integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every Console conversation agent-capable by swapping the single provider call in the send path for the shipped agent loop (Plan A: `tldw_chatbook/Agents/*` + `tldw_chatbook/DB/AgentRuns_DB.py`, merged in #623). A no-tool message streams exactly like today; the model engages tools/sub-agents only when it decides to; tool/spawn activity renders as compact transcript markers, a right-rail **Agent** inspector, and a `[N Sub-Agents]` conversation-row badge; **Stop** cancels the whole run tree; runs persist and re-derive on resume.

**Architecture:** The engine is pure and headless (`AgentService.run_turn(...) -> (run_id, RunOutcome)`, synchronous). Plan B is the impure Console shell around it: a new pure streaming fence-gate (`Agents/agent_stream.py`), a tiny engine step-hook seam (`LoopDeps.on_step`), a Console bridge (`Chat/console_agent_bridge.py`) that wires the gate + `provider_gateway.stream_chat` + the store markers + `AgentRunsDB` to the service, and a one-branch swap inside `ConsoleChatController._stream_assistant_response` run via `asyncio.to_thread`. UI is poll-driven (existing 0.2s transcript timer): the worker mutates the store + an in-memory live-step buffer; the poll renders. Spec: `Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md` (the contract — read the "Console send path", "Console UI", "Stuck/error/cancel", "Concurrency + persistence", streaming policy, resume, and live-gate sections before Task 1). Engine reference: `Docs/superpowers/plans/2026-07-13-agent-runtime-plan-a-engine.md` and the **shipped** modules.

**Tech Stack:** Python ≥3.11, Textual ≥3.3.0, dataclasses, sqlite3 via `BaseDB`, pytest, `asyncio.to_thread`.

## Global Constraints

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime` (branch `claude/agent-runtime-plan-b` @ `origin/dev` 844b2720 — includes #620/#621/#622/#623). All paths below are relative to it.
- Tests ONLY via the venv python: `PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` → `$PY -m pytest …`. Run from the worktree root.
- **No `AppTest`** (does not exist in this Textual version). UI tests use `async with app.run_test() as pilot:` (real App). Keep UI tests minimal; most coverage is pure/service-level.
- **Pure-module purity** stays intact: `Agents/agent_models.py` and `Agents/agent_runtime.py` import nothing but stdlib + each other; the NEW `Agents/agent_stream.py` is pure (stdlib + `agent_runtime`/`agent_models` only — NO Textual/DB/IO). Only `Agents/agent_service.py` and the new `Chat/console_agent_bridge.py` touch IO; the bridge may import Textual-free store/gateway types but performs no widget mutation.
- **Do NOT extend `ConsoleRunStatus`** (closed enum; Stop-visibility/Send-gating derive from it in many places — scout risk 4). The whole agent run stays `STREAMING`; phase detail lives in the rail Agent section + transcript markers.
- **Do NOT add fields to `ConsoleSessionSettings`** (frozen + rebuilt wholesale — scout risk 6). Per-session agent config is a deferred follow-up. `AgentConfig` is built at send time from the existing `ConsoleProviderSelection` + the composed system prompt.
- **Fence protocol (shipped):** the fence is `` ```tool_call `` and MUST be the first non-whitespace content of a turn to count as a leading tool call. `stream_prefix_verdict`, `split_visible_text_and_tool_call`, `FENCE_OPEN` live in `Agents/agent_runtime.py`. Malformed/partial → treated as plain text, never an exception.
- **Slice budgets (shipped defaults):** `RunBudget()` = `max_steps=8, max_wall_seconds=240.0, max_subagents=2, max_active_tools=8, max_subagent_result_chars=4000`. Plan B uses these defaults unchanged.
- **`allowed_tools` is fail-closed** in the engine (default empty). The integration site MUST pass the full builtin set explicitly: `tuple(e.name for e in registry.list_catalog()) + (SPAWN_TOOL_NAME,)`.
- **System-prompt composition (#620 contract):** the session system prompt is NEVER clobbered. Primary `AgentConfig.system_prompt = compose(session_prompt, CONSOLE_AGENT_OPERATING_PROMPT)` = session prompt first, agent operating prompt appended. When the session prompt is empty, just the operating prompt.
- **Escaping:** every agent/task/tool-derived string rendered into a transcript marker, rail line, or row badge is markup-escaped with `from rich.markup import escape as escape_markup` (follow #620's transcript escaping precedent).
- **CSS discipline:** any new styled id/class is added to the SOURCE `tldw_chatbook/css/components/_agentic_terminal.tcss`, the bundle is regenerated with `$PY tldw_chatbook/css/build_css.py`, and a pin test asserts the selector is present in BOTH the source and `tldw_chatbook/css/tldw_cli_modular.tcss`.
- **Feature safety valve:** a `[console] agent_runtime = true|false` config gate (default **ON**). OFF ⇒ the pre-swap legacy direct-stream path — one branch at the swap site.
- SQL always parameterized; DB writes via `transaction()`. Worker→UI only via store/reactive mutation (never cross-thread widget mutation); DB reads use a fresh connection (BaseDB per-call default).
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

### Shipped engine API (verified — target these exact shapes)

- `Agents/agent_service.py`: `AgentService(db: AgentRunsDB, registry: ToolCatalogRegistry, chat_call: Callable | None = None, clock=time.monotonic)`; `run_turn(*, conversation_id: str, messages: list[dict], config: AgentConfig, api_endpoint: str, should_cancel: Callable[[], bool] = lambda: False, supersede_run_id: str | None = None) -> tuple[str, RunOutcome]`. Its internal `_make_call_model` builds `payload = [{"role":"system","content":system_content}] + messages` and calls `self.chat_call(api_endpoint=…, messages_payload=payload, streaming=False, model=config.model)`; reads `resp["choices"][0]["message"]["content"]`. Constant `SUBAGENT_SYSTEM_PROMPT`.
- `Agents/agent_models.py`: `AgentConfig(model, system_prompt, allowed_tools=(), budget=RunBudget())` (frozen); `RunBudget(...)`; `RunOutcome(status, steps, final_text="", subagents_spawned=0)`; `AgentStep(index, kind, summary="", tool_name="", args=None, result="", created_at="")`; constants `RUN_DONE/RUN_ERROR/RUN_STUCK/RUN_CANCELLED/RUN_SUPERSEDED`, `AGENT_KIND_PRIMARY/AGENT_KIND_SUBAGENT`, `STEP_MODEL/STEP_TOOL_CALL/STEP_TOOL_RESULT/STEP_SPAWN/STEP_ERROR`, `SPAWN_TOOL_NAME="spawn_subagent"`, `FIND_TOOLS_NAME`, `LOAD_TOOLS_NAME`.
- `Agents/agent_runtime.py`: `LoopDeps(call_model, invoke_tool, spawn, find_tools, load_schemas, should_cancel, clock)` and `run_agent_loop(config, initial_messages, active_schemas, deps) -> RunOutcome`. Internal `add(kind, **kw)` appends each `AgentStep`. `parse_fenced_tool_call`, `split_visible_text_and_tool_call`, `stream_prefix_verdict`, `render_tool_protocol`, `FENCE_OPEN`, `STREAM_TEXT/STREAM_TOOL_CALL/STREAM_UNDECIDED`.
- `Agents/tool_catalog.py`: `ToolCatalogRegistry().register_provider(BuiltinToolProvider())`; `list_catalog() -> list[ToolCatalogEntry]` (`.name`, `.id`).
- `DB/AgentRuns_DB.py`: `AgentRunsDB(db_path, client_id="default")` (schema initialized in `__init__`); `create_run`, `append_steps`, `set_status`, `get_run(run_id) -> dict | None` (dict has `id, conversation_id, parent_run_id, agent_kind, task, status, steps(list), result, budget, created_at, updated_at`), `list_runs(conversation_id, include_superseded=True) -> list[dict]` (newest first), `count_subagent_runs(conversation_id) -> int`, `supersede_run_tree(run_id) -> int`.

### Console anchors (verified line numbers)

- `Chat/console_chat_controller.py`: swap point `_stream_assistant_response(*, resolution, provider_messages, assistant_message_id, prepare_retry=False)` @ **613–724**; `regenerate_message` divergent loop @ **461–509**; `stop_active_run` @ **330** (sets `self._stop_requested=True`, cancels `self._active_stream_task`); `_leading_system_message` @ **726**; `_provider_messages_for_session` @ **744**; ctor @ **90–144** (`self._stop_requested`, `self._active_stream_task`).
- `Chat/console_chat_store.py`: `append_message` @ **374**, `append_stream_chunk` @ **471**, `mark_message_complete` @ **483**, `mark_message_stopped` @ **492**, `prepare_message_retry` @ **510**, `add_variant` @ **523**, `_materialize_stream_buffer`, `session_id_for_message` @ **465**.
- `Chat/console_chat_models.py`: `ConsoleMessageRole` (USER/ASSISTANT/SYSTEM/**TOOL**) @ **12**; `ConsoleRunStatus` @ **21** (do not extend).
- `Widgets/Console/console_transcript.py`: `ConsoleTranscriptMessage.__init__` @ **165–179** (builds `classes="console-transcript-message"`); role is fixed per message; the transcript renders every role (no filter).
- `UI/Screens/chat_screen.py`: rail Model section compose @ **5413–5477**; Details section @ **5480–5501**; `_sync_console_settings_summary` @ **1655**; controller construction @ **1906–1932**; `_sync_console_chat_core_state` @ **1939**; poll timer `_start_console_transcript_sync_timer` @ **6595** / `_sync_native_console_chat_ui` @ **6532**; `_append_native_console_system_message` @ **6579**; `handle_console_stop_generation`→`_stop_console_generation_from_visible_action` @ **7220/7225**; `_build_console_rail_state` @ **3937**; browser-state build call @ **3545**; `_console_config()` @ **3564**; store build `_ensure_console_chat_store` @ **1871** (`self.app_instance.chachanotes_db`).
- `Workspaces/conversation_browser_state.py`: `ConsoleConversationBrowserRow` @ **102**, `build_console_conversation_browser_state` @ **212**, `_to_browser_row` @ **388**.
- `Chat/console_rail_state.py`: `ConsoleRailPreferences` @ **65** / `ConsoleRailState` @ **87** (`model_open`, `details_open` fields; coercion @ **242**, serialize @ **256**, effective build @ **461**).
- CSS: source `tldw_chatbook/css/components/_agentic_terminal.tcss` (`.console-transcript-message` @ **2491**, `.console-rail-section-body` @ **2096**); bundle `tldw_chatbook/css/tldw_cli_modular.tcss`; build `tldw_chatbook/css/build_css.py`.

---

### Task 1: Unify the reply engine — regenerate onto `_stream_assistant_response`

**Why:** The swap in Task 6 must touch ONE reply engine. Today `regenerate_message` (:461–509) has its own inline `async for chunk in stream_chat(...)` loop that buffers to a list, has NO mid-stream Stop check, and finalizes with `add_variant`. Route it through `_stream_assistant_response` (which has the incremental write + Stop check) via a `variant_mode` finalization strategy, backed by two small store methods so variant streaming does not corrupt the visible row.

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (add `begin_variant_stream`, `finalize_variant_stream`, `self._variant_stream_bases` in `__init__`)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (add `variant_mode` to `_stream_assistant_response`; rewrite `regenerate_message` to delegate)
- Test: `Tests/Chat/test_console_variant_stream.py`

**Interfaces:**
- Consumes: existing store stream mechanics (`append_stream_chunk`, `_materialize_stream_buffer`, `ConsoleVariantSet`, `ConsoleVariant`).
- Produces: `ConsoleChatStore.begin_variant_stream(message_id: str) -> ConsoleChatMessage` (snapshots current content as base, resets stream buffer to empty, status `streaming`); `ConsoleChatStore.finalize_variant_stream(message_id: str) -> ConsoleChatMessage` (materializes streamed buffer as a NEW variant appended to `[base, new]`, selects it, status `complete`, persists); `_stream_assistant_response(..., variant_mode: bool = False)`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Chat/test_console_variant_stream.py
"""Regenerate is unified onto the streaming reply engine (Task 1)."""
import asyncio

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


def _store_with_answer():
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="original")
    store.mark_message_complete(assistant.id)
    return store, session, assistant.id


def test_begin_variant_stream_resets_buffer_and_keeps_base():
    store, _session, mid = _store_with_answer()
    streaming = store.begin_variant_stream(mid)
    assert streaming.status == "streaming"
    assert streaming.content == ""            # visible row cleared for the new take
    store.append_stream_chunk(mid, "re")
    store.append_stream_chunk(mid, "generated")
    final = store.finalize_variant_stream(mid)
    assert final.status == "complete"
    assert final.content == "regenerated"     # new variant selected
    assert final.variants is not None
    contents = [v.content for v in final.variants.variants]
    assert contents == ["original", "regenerated"]  # base preserved, no concat
    assert final.variants.selected_index == 1


def test_finalize_variant_stream_appends_to_existing_set():
    store, _session, mid = _store_with_answer()
    store.begin_variant_stream(mid)
    store.append_stream_chunk(mid, "second")
    store.finalize_variant_stream(mid)
    store.begin_variant_stream(mid)
    store.append_stream_chunk(mid, "third")
    final = store.finalize_variant_stream(mid)
    assert [v.content for v in final.variants.variants] == ["original", "second", "third"]
    assert final.variants.selected_index == 2


class _ScriptedGateway:
    """Async stream_chat that yields scripted chunks; resolve_for_send ready."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def resolve_for_send(self, selection):
        class _R:  # noqa: D401 - tiny stub
            ready = True
            visible_copy = ""
        return _R()

    async def stream_chat(self, resolution, messages):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_regenerate_delegates_and_streams_incrementally():
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    store, _session, mid = _store_with_answer()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_ScriptedGateway(["Paris", " is", " the", " answer."]),
        provider="llama_cpp",
        model="test-model",
    )
    result = await controller.regenerate_message(mid)
    assert result.accepted is True
    message = store.get_message(mid)
    assert message.content == "Paris is the answer."
    assert [v.content for v in message.variants.variants] == [
        "original", "Paris is the answer."]
    assert message.variants.selected_index == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest Tests/Chat/test_console_variant_stream.py -q`
Expected: FAIL — `AttributeError: 'ConsoleChatStore' object has no attribute 'begin_variant_stream'`.

- [ ] **Step 3: Write the implementation**

In `ConsoleChatStore.__init__` add alongside the other stream dicts:
```python
        self._variant_stream_bases: dict[str, str] = {}
```

Add these two methods to `ConsoleChatStore` (place after `add_variant`, near :540):
```python
    def begin_variant_stream(self, message_id: str) -> ConsoleChatMessage:
        """Snapshot current content as the base and reset the buffer for a new variant."""
        message = self._message_or_raise(message_id)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            raise ValueError("Only assistant messages can be regenerated.")
        self._materialize_stream_buffer(message)
        self._variant_stream_bases[message.id] = message.content
        message.content = ""
        self._stream_chunks_by_message.pop(message.id, None)
        self._stream_materialized_counts.pop(message.id, None)
        message.status = "streaming"
        return self._snapshot(message)

    def finalize_variant_stream(self, message_id: str) -> ConsoleChatMessage:
        """Store the streamed buffer as a new selected variant beside the snapshot base."""
        message = self._message_or_raise(message_id)
        self._materialize_stream_buffer(message)
        new_content = message.content
        base = self._variant_stream_bases.pop(message.id, "")
        if message.variants is None:
            message.variants = ConsoleVariantSet.from_contents(
                turn_id=message.turn_id or message.id,
                contents=[base, new_content],
                selected_index=1,
            )
        else:
            message.variants.variants.append(ConsoleVariant(content=new_content))
            message.variants.selected_index = len(message.variants.variants) - 1
        message.content = message.variants.current.content
        message.status = "complete"
        self._persist_existing_message(message)
        return self._snapshot(message)
```

In `_stream_assistant_response` (:613) add `variant_mode: bool = False` to the signature, and at the very top of the body (before the `try:`, after setting `self._stop_requested = False` at :623) insert:
```python
        if variant_mode:
            self.store.begin_variant_stream(assistant_message_id)
```
Replace the success finalization (:679–686 `mark_message_complete` branch) with a `variant_mode` fork:
```python
            try:
                if variant_mode:
                    completed = self.store.finalize_variant_stream(assistant_message_id)
                else:
                    completed = self.store.mark_message_complete(assistant_message_id)
            except KeyError:
                return self._session_closed_result()
```
Rewrite `regenerate_message` (:461–509) so its body, after the readiness check and `provider_messages` build (keep :467–488 unchanged through `_ensure_user_continuation_instruction`), delegates:
```python
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=message_id,
            variant_mode=True,
        )
```
Delete the old inline `chunks`/`add_variant` loop (:489–509).

Note on Stop-in-variant: `_mark_stream_stopped` runs `mark_message_stopped`, leaving the partial streamed text as the row content and the base snapshot orphaned in `_variant_stream_bases`. That is acceptable for the slice (a stopped regenerate keeps its partial take visible); the base is discarded next `begin_variant_stream`. No test asserts variant-set contents on stop.

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest Tests/Chat/test_console_variant_stream.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py -q`
Expected: PASS (3 new + existing controller/store suites still green — the refactor must not regress them).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_variant_stream.py
git commit -m "refactor(console): unify regenerate onto the streaming reply engine

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Pure streaming fence-gate (`Agents/agent_stream.py`)

**Why:** The agent-aware model-call adapter must stream a turn's visible text to the transcript live (today's UX) while never leaking fenced tool-call content, and must truncate at a disobedient mid-stream fence — all without a store rewrite. Encapsulate that as a PURE, fully unit-tested state machine so the Console adapter (Task 5) is a thin impure shell.

**Files:**
- Create: `tldw_chatbook/Agents/agent_stream.py`
- Test: `Tests/Agents/test_agent_stream.py`

**Interfaces:**
- Consumes: `FENCE_OPEN`, `stream_prefix_verdict`, `split_visible_text_and_tool_call`, `STREAM_TEXT`, `STREAM_TOOL_CALL`, `STREAM_UNDECIDED` from `agent_runtime`; `ToolCall` from `agent_models`.
- Produces: `StreamGate` with `feed(chunk: str) -> str` (visible text safe to flush right now, holding back a `len(FENCE_OPEN)-1` tail that could begin a fence; empty for a leading-fence/undecided/suppressed turn), `flush_tail() -> str` (remaining visible text for a completed no-fence turn), `full_text -> str` property (raw accumulation, always returned to the loop), and `result() -> tuple[str, ToolCall | None]` (`split_visible_text_and_tool_call` over the full buffer, for tests/inspection).

- [ ] **Step 1: Write the failing test**

```python
# Tests/Agents/test_agent_stream.py
"""Pure streaming fence-gate: incremental visible text, fence suppression."""
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Agents.agent_stream import StreamGate


def drain(chunks):
    """Feed chunks; return (streamed_visible, full_text, tool_call)."""
    gate = StreamGate()
    streamed = "".join(gate.feed(c) for c in chunks)
    streamed += gate.flush_tail()
    visible, call = gate.result()
    return streamed, gate.full_text, visible, call


def test_plain_text_streams_verbatim():
    streamed, full, visible, call = drain(["Tok", "yo is", " the capital."])
    assert streamed == "Tokyo is the capital."
    assert full == "Tokyo is the capital."
    assert visible == "Tokyo is the capital." and call is None


def test_leading_fence_streams_nothing():
    fence = FENCE_OPEN + '\n{"name": "calculator", "arguments": {"expression": "6*7"}}\n```'
    streamed, full, visible, call = drain([fence[:6], fence[6:20], fence[20:]])
    assert streamed == ""                       # nothing visible for a tool turn
    assert full == fence                        # loop still gets the full text
    assert call is not None and call.name == "calculator"


def test_leading_fence_split_across_chunks_is_never_partially_shown():
    fence = FENCE_OPEN + '\n{"name": "x", "arguments": {}}\n```'
    gate = StreamGate()
    # First chunk is only "``" — undecided, must not stream.
    assert gate.feed("``") == ""
    assert gate.feed("`tool_call") == ""        # still undecided (could be tool_calls)
    rest = "".join(gate.feed(c) for c in [fence[len(FENCE_OPEN):]])
    assert rest == "" and gate.flush_tail() == ""


def test_mid_stream_fence_truncates_visible_at_fence():
    tail = FENCE_OPEN + '\n{"name": "calculator", "arguments": {"expression": "1"}}\n```'
    streamed, full, visible, call = drain(["Let me compute. ", tail])
    assert streamed == "Let me compute."        # rstripped prefix, fence content withheld
    assert visible == "Let me compute." and call is not None
    assert full == "Let me compute. " + tail


def test_holdback_prevents_streaming_a_fence_prefix_then_completes():
    # A message ending in text whose tail coincidentally starts like a fence prefix.
    streamed, full, visible, call = drain(["answer ", "``"])
    assert streamed == "answer ``"              # no real fence → tail flushed at end
    assert call is None and visible == "answer ``"


def test_lookalike_fence_is_treated_as_visible_text():
    streamed, full, visible, call = drain(["```python\nprint(1)\n```"])
    assert call is None
    assert streamed == "```python\nprint(1)\n```"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest Tests/Agents/test_agent_stream.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Agents.agent_stream'`.

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/Agents/agent_stream.py
"""Pure streaming fence-gate for the Console agent model adapter.

Classifies a streaming turn incrementally: emit visible text as it arrives
(today's streaming UX) while never leaking fenced tool-call content, and
truncate at a disobedient mid-stream fence. No Textual, app, DB, or I/O.
"""
from __future__ import annotations

from .agent_models import ToolCall
from .agent_runtime import (
    FENCE_OPEN, STREAM_TEXT, STREAM_TOOL_CALL, STREAM_UNDECIDED,
    split_visible_text_and_tool_call, stream_prefix_verdict,
)

_HOLDBACK = len(FENCE_OPEN) - 1


class StreamGate:
    """Feed raw chunks; get back visible text safe to flush right now."""

    def __init__(self) -> None:
        self._buf = ""
        self._emitted = 0          # index into _buf of visible chars already flushed
        self._sealed = False       # a fence has been decided → nothing more streams

    @property
    def full_text(self) -> str:
        """The complete raw accumulation — always what the loop is told."""
        return self._buf

    def feed(self, chunk: str) -> str:
        """Add a chunk and return newly-flushable visible text (may be empty)."""
        if not chunk:
            return ""
        self._buf += chunk
        if self._sealed:
            return ""
        verdict = stream_prefix_verdict(self._buf)
        if verdict == STREAM_TOOL_CALL:
            self._sealed = True        # leading fence: whole turn is a tool call
            return ""
        if verdict == STREAM_UNDECIDED:
            return ""                  # not enough tokens to decide — hold everything
        # STREAM_TEXT: stream, but a later mid-stream fence may still truncate.
        fence = self._buf.find(FENCE_OPEN, self._emitted)
        if fence != -1:
            out = self._buf[self._emitted:fence].rstrip()
            self._emitted = fence
            self._sealed = True        # remainder is the tool call
            return out
        safe = max(self._emitted, len(self._buf) - _HOLDBACK)
        out = self._buf[self._emitted:safe]
        self._emitted = safe
        return out

    def flush_tail(self) -> str:
        """Return any held-back visible tail for a completed no-fence turn."""
        if self._sealed:
            return ""
        visible, call = split_visible_text_and_tool_call(self._buf)
        if call is not None:
            self._sealed = True
            return ""
        tail = visible[self._emitted:]
        self._emitted = len(visible)
        return tail

    def result(self) -> tuple[str, ToolCall | None]:
        """Authoritative (visible_text, tool_call) over the full buffer."""
        return split_visible_text_and_tool_call(self._buf)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest Tests/Agents/test_agent_stream.py -q`
Expected: PASS (6 tests). Also confirm purity: `$PY -c "import ast,sys; src=open('tldw_chatbook/Agents/agent_stream.py').read(); assert all(b not in src for b in ('textual','sqlite3','tldw_chatbook.DB','httpx'))"`

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_stream.py Tests/Agents/test_agent_stream.py
git commit -m "feat(agents): pure streaming fence-gate for live agent replies

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Engine per-step hook (`LoopDeps.on_step`)

**Why:** Live transcript markers + the rail step-tail need per-step notification during a run. The shipped service persists the whole step list only at run END; the loop appends steps internally with no callback. Add a small, backward-compatible `on_step` seam: a defaulted no-op field on `LoopDeps` invoked inside `add()`, forwarded by `AgentService` with the run's `agent_kind` so the Console filters primary vs sub-agent. Durability still comes from the service's end-of-run persist; the hook drives ONLY live UI (store markers + in-memory buffer), so there is no double-write to the DB.

**Files:**
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (add `on_step` to `LoopDeps`; call it in `add()`)
- Modify: `tldw_chatbook/Agents/agent_service.py` (ctor `on_step` param; forward per-run with `agent_kind`)
- Test: `Tests/Agents/test_agent_service_on_step.py`

**Interfaces:**
- Produces: `LoopDeps.on_step: Callable[[AgentStep], None] = lambda step: None` (LAST field, defaulted — existing construction sites remain valid). `AgentService(..., on_step: Callable[[AgentStep, str], None] | None = None)`; each `_run_one` wires `on_step=lambda s: self._on_step(s, agent_kind)` when `self._on_step` is set. Steps are reported in creation order; `agent_kind` is `"primary"` or `"subagent"`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Agents/test_agent_service_on_step.py
"""AgentService reports steps live via on_step with agent_kind (Task 3)."""
import json

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY, AGENT_KIND_SUBAGENT, SPAWN_TOOL_NAME,
    STEP_SPAWN, STEP_TOOL_CALL, AgentConfig,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN


def _fence(name, args):
    return f'{FENCE_OPEN}\n{json.dumps({"name": name, "arguments": args})}\n```'


def _registry():
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    return reg


def test_on_step_receives_primary_steps_in_order(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry()
    script = [
        {"choices": [{"message": {"content": _fence("calculator", {"expression": "6*7"})}}]},
        {"choices": [{"message": {"content": "It is 42."}}]},
    ]

    def chat_call(**kwargs):
        return script.pop(0)

    seen = []
    service = AgentService(db, reg, chat_call=chat_call,
                           on_step=lambda step, kind: seen.append((kind, step.kind)))
    _run_id, outcome = service.run_turn(
        conversation_id="c1", messages=[{"role": "user", "content": "6*7?"}],
        config=AgentConfig(model="m", system_prompt="s",
                           allowed_tools=("calculator", SPAWN_TOOL_NAME)),
        api_endpoint="llama_cpp")
    assert outcome.status == "done"
    kinds = [k for (_who, k) in seen]
    assert STEP_TOOL_CALL in kinds
    assert all(who == AGENT_KIND_PRIMARY for (who, _k) in seen)


def test_on_step_distinguishes_subagent_steps(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry()
    # Primary spawns a sub-agent; sub-agent answers directly.
    script = [
        {"choices": [{"message": {"content": _fence(SPAWN_TOOL_NAME, {"task": "add 1+1"})}}]},
        {"choices": [{"message": {"content": "2"}}]},          # sub-agent's turn
        {"choices": [{"message": {"content": "The answer is 2."}}]},  # primary final
    ]

    def chat_call(**kwargs):
        return script.pop(0)

    seen = []
    service = AgentService(db, reg, chat_call=chat_call,
                           on_step=lambda step, kind: seen.append((kind, step.kind)))
    _run_id, outcome = service.run_turn(
        conversation_id="c1", messages=[{"role": "user", "content": "delegate"}],
        config=AgentConfig(model="m", system_prompt="s",
                           allowed_tools=("calculator", SPAWN_TOOL_NAME)),
        api_endpoint="llama_cpp")
    assert outcome.status == "done"
    assert (AGENT_KIND_PRIMARY, STEP_SPAWN) in seen
    assert any(who == AGENT_KIND_SUBAGENT for (who, _k) in seen)


def test_on_step_default_is_noop(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry()
    service = AgentService(
        db, reg,
        chat_call=lambda **k: {"choices": [{"message": {"content": "hello"}}]})
    _run_id, outcome = service.run_turn(
        conversation_id="c1", messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", allowed_tools=("calculator",)),
        api_endpoint="llama_cpp")
    assert outcome.status == "done"        # no on_step wired → no crash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest Tests/Agents/test_agent_service_on_step.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'on_step'`.

- [ ] **Step 3: Write the implementation**

In `agent_runtime.py`, add the field to `LoopDeps` (after `clock`, as the LAST field so all existing positional/keyword construction stays valid):
```python
    clock: Callable[[], float]
    on_step: Callable[[AgentStep], None] = lambda step: None
```
Add `AgentStep` to the `from .agent_models import (...)` block in `agent_runtime.py` if not already imported (it is imported). Inside `run_agent_loop`'s `add` helper, call the hook after appending:
```python
    def add(kind: str, **kw) -> AgentStep:
        step = AgentStep(index=len(steps), kind=kind, **kw)
        steps.append(step)
        deps.on_step(step)
        return step
```
In `agent_service.py`, extend the ctor:
```python
    def __init__(self, db: AgentRunsDB, registry: ToolCatalogRegistry,
                 chat_call: Callable | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 on_step: Callable[[AgentStep, str], None] | None = None) -> None:
        self.db = db
        self.registry = registry
        self.chat_call = chat_call or _default_chat_call()
        self.clock = clock
        self._on_step = on_step
```
In `_run_one`, when building `deps = LoopDeps(...)`, add:
```python
            should_cancel=should_cancel,
            clock=self.clock,
            on_step=((lambda s: self._on_step(s, agent_kind))
                     if self._on_step is not None else (lambda s: None)),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest Tests/Agents/test_agent_service_on_step.py Tests/Agents -q`
Expected: PASS — the new suite plus every existing Plan-A Agents suite (the defaulted field must not break `LoopDeps(...)` construction anywhere).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_runtime.py tldw_chatbook/Agents/agent_service.py Tests/Agents/test_agent_service_on_step.py
git commit -m "feat(agents): backward-compatible on_step hook for live UI

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Sub-agent count badge (pure state) + TOOL-row styling + CSS

**Why:** Two small render primitives the UI tasks build on: (1) the pure browser-state builder must carry a historical `[N Sub-Agents]` count per conversation row; (2) TOOL-role transcript rows need a dim CSS class so tool markers read as marginalia, not prose.

**Files:**
- Modify: `tldw_chatbook/Workspaces/conversation_browser_state.py` (`ConsoleConversationBrowserRow.subagent_count`; `subagent_counts` param; `_to_browser_row` threading)
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (`ConsoleTranscriptMessage.__init__` adds a role class)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (`.console-transcript-message-tool`) + regenerate bundle
- Test: `Tests/Workspaces/test_conversation_browser_subagents.py`, `Tests/UI/test_console_agent_tool_row_css.py`

**Interfaces:**
- Produces: `ConsoleConversationBrowserRow.subagent_count: int = 0`; `build_console_conversation_browser_state(..., subagent_counts: Mapping[str, int] | None = None)` (keyed by `conversation_id`; only persisted rows can carry a count; missing/None → 0). `ConsoleTranscriptMessage` gains class `console-transcript-message-tool` when `message.role is ConsoleMessageRole.TOOL` (and `-system` for SYSTEM, harmless). Badge text is rendered by the widget in Task 7 (escaped) — this task only carries the number and styles the row.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Workspaces/test_conversation_browser_subagents.py
"""[N Sub-Agents] count is threaded through the pure browser-state builder."""
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow, build_console_conversation_browser_state,
)


def _row(conversation_id, title):
    return ConsoleConversationBrowserInputRow(
        row_key=conversation_id, conversation_id=conversation_id,
        native_session_id=None, title=title, scope_type="global",
        workspace_id=None, workspace_label="", updated_sort="2026-07-13T00:00:00Z")


def _all_rows(state):
    rows = []
    for section in state.sections:
        rows.extend(section.rows)
        for group in section.groups:
            rows.extend(group.rows)
    return rows


def test_subagent_count_attaches_to_matching_conversation_row():
    state = build_console_conversation_browser_state(
        rows=[_row("conv-a", "Alpha"), _row("conv-b", "Beta")],
        active_workspace_id=None,
        subagent_counts={"conv-a": 3},
    )
    by_id = {r.conversation_id: r for r in _all_rows(state)}
    assert by_id["conv-a"].subagent_count == 3
    assert by_id["conv-b"].subagent_count == 0


def test_subagent_counts_default_to_zero_when_absent():
    state = build_console_conversation_browser_state(
        rows=[_row("conv-a", "Alpha")], active_workspace_id=None)
    assert _all_rows(state)[0].subagent_count == 0
```

```python
# Tests/UI/test_console_agent_tool_row_css.py
"""TOOL transcript rows carry a dim role class, pinned in source + bundle."""
from pathlib import Path

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscriptMessage


def test_tool_message_row_has_tool_class():
    msg = ConsoleChatMessage(role=ConsoleMessageRole.TOOL, content="called calculator -> 42")
    row = ConsoleTranscriptMessage(msg)
    assert "console-transcript-message-tool" in row.classes
    assert "console-transcript-message" in row.classes


def test_tool_row_class_is_styled_in_source_and_bundle():
    for path in (
        Path("tldw_chatbook/css/components/_agentic_terminal.tcss"),
        Path("tldw_chatbook/css/tldw_cli_modular.tcss"),
    ):
        assert ".console-transcript-message-tool" in path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest Tests/Workspaces/test_conversation_browser_subagents.py Tests/UI/test_console_agent_tool_row_css.py -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'subagent_counts'` and missing CSS class.

- [ ] **Step 3: Write the implementation**

In `conversation_browser_state.py`: add `subagent_count: int = 0` as the last field of `ConsoleConversationBrowserRow` (:102). Add a parameter and thread it:
```python
def build_console_conversation_browser_state(
    *,
    rows: Iterable[ConsoleConversationBrowserInputRow],
    active_workspace_id: str | None,
    group_collapse_preferences: Mapping[str, bool] | None = None,
    query: str = "",
    marks_available: bool = True,
    error_copy: str = "",
    result_total_count: int | None = None,
    result_limit: int = CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
    group_row_limit: int = CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT,
    subagent_counts: Mapping[str, int] | None = None,
    now: datetime | None = None,
) -> ConsoleConversationBrowserState:
```
Capture `counts = dict(subagent_counts or {})` near the top, and pass it into `_to_browser_row`. Change `_to_browser_row` to accept and apply the count:
```python
def _to_browser_row(
    row: ConsoleConversationBrowserInputRow,
    counts: Mapping[str, int] | None = None,
) -> ConsoleConversationBrowserRow:
    subagent_count = int((counts or {}).get(row.conversation_id or "", 0))
    return ConsoleConversationBrowserRow(
        ...,                      # existing fields unchanged
        source_kind=row.source_kind,
        subagent_count=subagent_count,
    )
```
`_to_browser_row` is called from `_visible_rows` (:481). Thread `counts` through `_visible_rows`, `_build_row_section`, `_build_workspace_groups` (add a `counts` keyword to each and forward it) so both grouped and row sections carry the number. Keep the signatures internal.

In `console_transcript.py`, `ConsoleTranscriptMessage.__init__` (:170), after building `classes`:
```python
        role = message.role
        if role is ConsoleMessageRole.TOOL:
            classes = f"{classes} console-transcript-message-tool"
        elif role is ConsoleMessageRole.SYSTEM:
            classes = f"{classes} console-transcript-message-system"
```
Import `ConsoleMessageRole` in that module if not present (it imports `ConsoleChatMessage`; extend the import).

In `_agentic_terminal.tcss` after `.console-transcript-message-selected` (:2505) add:
```tcss
.console-transcript-message-tool {
    color: $ds-text-muted;
    text-style: dim italic;
}

.console-transcript-message-system {
    color: $ds-text-muted;
}
```
Regenerate the bundle: `$PY tldw_chatbook/css/build_css.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest Tests/Workspaces/test_conversation_browser_subagents.py Tests/UI/test_console_agent_tool_row_css.py Tests/Workspaces -q`
Expected: PASS (new tests + existing browser-state suite green).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Workspaces/conversation_browser_state.py tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/Workspaces/test_conversation_browser_subagents.py Tests/UI/test_console_agent_tool_row_css.py
git commit -m "feat(console): sub-agent count in browser state + dim TOOL transcript rows

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Console agent bridge (`Chat/console_agent_bridge.py`)

**Why:** All the impure glue between the shipped synchronous engine and the Console, in one testable module: build the `AgentConfig` (composed system prompt, full-builtin allow-list, slice budget), construct the streaming model adapter (Task-2 `StreamGate` + `provider_gateway.stream_chat`, suppressing sub-agent turns), construct `AgentService` with the Task-3 `on_step` (appending escaped TOOL markers to the store for PRIMARY tool/spawn steps + maintaining a live in-memory snapshot for the rail), compute `supersede_run_id`, and run `run_turn` synchronously (the controller wraps it in `asyncio.to_thread`).

**Files:**
- Create: `tldw_chatbook/Chat/console_agent_bridge.py`
- Test: `Tests/Chat/test_console_agent_bridge.py`

**Interfaces:**
- Consumes: `AgentService`, `ToolCatalogRegistry`, `BuiltinToolProvider`, `AgentConfig`, `RunBudget`, `SPAWN_TOOL_NAME`, `SUBAGENT_SYSTEM_PROMPT`, `RunOutcome`, step constants; `StreamGate`; `AgentRunsDB`; `ConsoleChatStore`; `ConsoleMessageRole`; a provider gateway with async `stream_chat(resolution, messages)`.
- Produces:
  - `CONSOLE_AGENT_OPERATING_PROMPT: str`.
  - `compose_agent_system_prompt(session_prompt: str) -> str` (session first, operating prompt appended; operating prompt alone when session is blank).
  - `AgentLiveStep` dataclass (`kind`, `text`, `agent_kind`) and `AgentLiveSnapshot` (`status: str`, `step: int`, `steps: tuple[AgentLiveStep, ...]`, `subagents: tuple[SubAgentSummary, ...]`).
  - `ConsoleAgentBridge(*, agent_runs_db, store, provider_gateway, registry=None, clock=time.monotonic)`.
  - `ConsoleAgentBridge.run_reply(*, conversation_id, session_id, resolution, assistant_message_id, model, session_system_prompt, agent_messages, should_cancel, supersede_previous=False) -> RunOutcome` (synchronous; streams the primary's final turn into `assistant_message_id`, appends TOOL markers, persists via the service, returns the primary `RunOutcome`).
  - `ConsoleAgentBridge.live_snapshot(conversation_id) -> AgentLiveSnapshot` (in-memory, for the rail poll).
  - `ConsoleAgentBridge.subagent_run(run_id) -> dict | None` (drill-in, from the DB).

- [ ] **Step 1: Write the failing test**

```python
# Tests/Chat/test_console_agent_bridge.py
"""Console agent bridge: streaming, markers, spawn, supersede (fakes only)."""
import json

import pytest

from tldw_chatbook.Chat.console_agent_bridge import (
    CONSOLE_AGENT_OPERATING_PROMPT, ConsoleAgentBridge, compose_agent_system_prompt,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN


def _fence(name, args):
    return f'{FENCE_OPEN}\n{json.dumps({"name": name, "arguments": args})}\n```'


class _ChunkGateway:
    """A gateway whose stream_chat replays a script keyed by call index."""

    def __init__(self, scripts):
        self._scripts = list(scripts)   # each entry: list[str] chunks for that turn
        self.calls = 0

    async def stream_chat(self, resolution, messages):
        chunks = self._scripts[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk


def _bridge(tmp_path, scripts):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts))
    return bridge, db, store, session, assistant.id


def _run(bridge, store, session, assistant_id, **over):
    kwargs = dict(
        conversation_id="conv-1", session_id=session.id, resolution=object(),
        assistant_message_id=assistant_id, model="test-model",
        session_system_prompt="", agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False)
    kwargs.update(over)
    return bridge.run_reply(**kwargs)


def test_compose_prepends_session_prompt_then_agent_prompt():
    composed = compose_agent_system_prompt("You are Ada.")
    assert composed.startswith("You are Ada.")
    assert CONSOLE_AGENT_OPERATING_PROMPT in composed
    assert compose_agent_system_prompt("") == CONSOLE_AGENT_OPERATING_PROMPT


def test_no_tool_message_streams_final_answer_like_today(tmp_path):
    bridge, _db, store, session, aid = _bridge(tmp_path, [["Tok", "yo."]])
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done" and outcome.final_text == "Tokyo."
    assert store.get_message(aid).content == "Tokyo."
    # No tool markers were appended.
    roles = [m.role for m in store.messages_for_session(session.id)]
    assert ConsoleMessageRole.TOOL not in roles


def test_tool_turn_renders_a_tool_marker_not_prose(tmp_path):
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],   # turn 1: leading fence
        ["It is ", "42."],                                # turn 2: final answer
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done"
    tool_rows = [m for m in store.messages_for_session(session.id)
                 if m.role is ConsoleMessageRole.TOOL]
    assert tool_rows, "a tool turn must drop a TOOL marker"
    assert "calculator" in tool_rows[0].content
    # The fenced tool JSON never streamed into the assistant answer.
    assert FENCE_OPEN not in store.get_message(aid).content
    assert store.get_message(aid).content == "It is 42."


def test_spawn_renders_marker_and_persists_linked_subagent(tmp_path):
    scripts = [
        [_fence("spawn_subagent", {"task": "compute 1+1"})],  # primary turn 1
        ["2"],                                                 # sub-agent turn
        ["Done: ", "2."],                                     # primary final
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done"
    assert db.count_subagent_runs("conv-1") == 1
    spawn_markers = [m for m in store.messages_for_session(session.id)
                     if m.role is ConsoleMessageRole.TOOL and "sub-agent" in m.content.lower()]
    assert spawn_markers
    snap = bridge.live_snapshot("conv-1")
    assert any(s.text for s in snap.subagents)


def test_supersede_marks_previous_primary_and_tree(tmp_path):
    bridge, db, store, session, aid = _bridge(tmp_path, [["one."], ["two."]])
    _run(bridge, store, session, aid)                        # first run
    first = db.list_runs("conv-1")[0]
    assert first["status"] == "done"
    # Second run supersedes the previous primary.
    aid2 = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT,
                                content="").id
    _run(bridge, store, session, aid2, supersede_previous=True)
    prior = db.get_run(first["id"])
    assert prior["status"] == "superseded"


def test_stop_persists_cancelled(tmp_path):
    # A long tool loop; cancel flips after the first step.
    scripts = [[_fence("calculator", {"expression": "1"})], ["never reached"]]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    flags = iter([False, True])
    outcome = _run(bridge, store, session, aid, should_cancel=lambda: next(flags, True))
    assert outcome.status == "cancelled"
    assert db.list_runs("conv-1")[0]["status"] == "cancelled"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest Tests/Chat/test_console_agent_bridge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Chat.console_agent_bridge'`.

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/Chat/console_agent_bridge.py
"""Impure Console glue between the synchronous agent engine and the store.

Builds the AgentConfig, drives a streaming model adapter (StreamGate +
provider_gateway.stream_chat), appends escaped TOOL markers for the primary
run's tool/spawn steps, keeps an in-memory live snapshot for the rail poll,
and runs AgentService.run_turn synchronously (the controller wraps it in
asyncio.to_thread). No widget mutation.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from rich.markup import escape as escape_markup

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY, RunBudget, SPAWN_TOOL_NAME, STEP_ERROR, STEP_SPAWN,
    STEP_TOOL_RESULT, AgentConfig, AgentStep, RunOutcome,
)
from tldw_chatbook.Agents.agent_service import SUBAGENT_SYSTEM_PROMPT, AgentService
from tldw_chatbook.Agents.agent_stream import StreamGate
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

CONSOLE_AGENT_OPERATING_PROMPT = (
    "You are a capable assistant with optional tools. Answer directly when no "
    "tool is needed. When a tool would help, call exactly one tool per reply "
    "using the fenced protocol described below, then continue once you have the "
    "result. Use spawn_subagent to delegate a self-contained sub-task to an "
    "isolated helper. Keep replies concise.")

_QUIET_STEP_TOOLS = {"find_tools", "load_tools"}


def compose_agent_system_prompt(session_prompt: str) -> str:
    """Compose the primary system prompt: session prompt first, agent prompt appended."""
    base = (session_prompt or "").strip()
    if not base:
        return CONSOLE_AGENT_OPERATING_PROMPT
    return f"{session_prompt}\n\n{CONSOLE_AGENT_OPERATING_PROMPT}"


@dataclass(frozen=True)
class AgentLiveStep:
    kind: str
    text: str
    agent_kind: str


@dataclass(frozen=True)
class SubAgentSummary:
    text: str
    status: str = "running"


@dataclass(frozen=True)
class AgentLiveSnapshot:
    status: str = "idle"
    step: int = 0
    steps: tuple[AgentLiveStep, ...] = ()
    subagents: tuple[SubAgentSummary, ...] = ()


class _StreamingModelAdapter:
    """chat_call-compatible adapter that streams the PRIMARY final turn live.

    AgentService calls it as ``chat_call(api_endpoint=…, messages_payload=…,
    streaming=False, model=…)`` and expects a
    ``{"choices":[{"message":{"content": <full text>}}]}`` response. Sub-agent
    turns (leading system content == SUBAGENT_SYSTEM_PROMPT) are streamed to a
    throwaway gate and never touch the transcript.
    """

    def __init__(self, *, store, provider_gateway, resolution, assistant_message_id,
                 should_cancel):
        self._store = store
        self._gateway = provider_gateway
        self._resolution = resolution
        self._assistant_message_id = assistant_message_id
        self._should_cancel = should_cancel

    def chat_call(self, *, messages_payload, model=None, api_endpoint=None,
                  streaming=False, **_ignored) -> dict:
        is_subagent = self._is_subagent(messages_payload)
        gate = StreamGate()

        async def _consume() -> None:
            async for chunk in self._gateway.stream_chat(self._resolution, messages_payload):
                if self._should_cancel():
                    break
                visible = gate.feed(chunk)
                if visible and not is_subagent:
                    self._store.append_stream_chunk(self._assistant_message_id, visible)
            tail = gate.flush_tail()
            if tail and not is_subagent:
                self._store.append_stream_chunk(self._assistant_message_id, tail)

        # The service runs on a worker thread with no running loop → asyncio.run
        # is safe (same pattern as BuiltinToolProvider bridging async tools).
        asyncio.run(_consume())
        return {"choices": [{"message": {"content": gate.full_text}}]}

    @staticmethod
    def _is_subagent(messages_payload) -> bool:
        if not messages_payload:
            return False
        first = messages_payload[0]
        return (first.get("role") == "system"
                and str(first.get("content", "")).startswith(SUBAGENT_SYSTEM_PROMPT))


class ConsoleAgentBridge:
    """Owns the tool registry + run store and runs one primary agent reply."""

    def __init__(self, *, agent_runs_db: AgentRunsDB, store,
                 provider_gateway, registry: ToolCatalogRegistry | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._db = agent_runs_db
        self._store = store
        self._gateway = provider_gateway
        self._clock = clock
        if registry is None:
            registry = ToolCatalogRegistry()
            registry.register_provider(BuiltinToolProvider())
        self._registry = registry
        self._allowed_tools = tuple(
            e.name for e in registry.list_catalog()) + (SPAWN_TOOL_NAME,)
        self._live: dict[str, AgentLiveSnapshot] = {}

    # -- run ------------------------------------------------------------

    def run_reply(self, *, conversation_id: str, session_id: str, resolution: Any,
                  assistant_message_id: str, model: str, session_system_prompt: str,
                  agent_messages: list[dict], should_cancel: Callable[[], bool],
                  supersede_previous: bool = False) -> RunOutcome:
        config = AgentConfig(
            model=model,
            system_prompt=compose_agent_system_prompt(session_system_prompt),
            allowed_tools=self._allowed_tools,
            budget=RunBudget())
        adapter = _StreamingModelAdapter(
            store=self._store, provider_gateway=self._gateway, resolution=resolution,
            assistant_message_id=assistant_message_id, should_cancel=should_cancel)

        live_steps: list[AgentLiveStep] = []
        subagents: list[SubAgentSummary] = []
        self._live[conversation_id] = AgentLiveSnapshot(status="running")

        def on_step(step: AgentStep, agent_kind: str) -> None:
            live_steps.append(AgentLiveStep(step.kind, self._summarize(step), agent_kind))
            if agent_kind == AGENT_KIND_PRIMARY:
                if step.kind == STEP_SPAWN:
                    subagents.append(SubAgentSummary(escape_markup(step.summary or "")))
                    self._append_marker(
                        session_id, f"⤷ spawned sub-agent: {step.summary}")
                elif (step.kind == STEP_TOOL_RESULT
                      and step.tool_name not in _QUIET_STEP_TOOLS):
                    self._append_marker(
                        session_id, f"⚙ {step.tool_name} → {step.result}")
                elif step.kind == STEP_ERROR:
                    self._append_marker(session_id, f"⚠ {step.summary}")
            self._live[conversation_id] = AgentLiveSnapshot(
                status="running", step=len(live_steps),
                steps=tuple(live_steps[-5:]), subagents=tuple(subagents))

        service = AgentService(
            self._db, self._registry, chat_call=adapter.chat_call,
            clock=self._clock, on_step=on_step)

        supersede_run_id = (
            self._previous_primary_run_id(conversation_id) if supersede_previous else None)
        _run_id, outcome = service.run_turn(
            conversation_id=conversation_id, messages=agent_messages, config=config,
            api_endpoint=str(getattr(resolution, "provider", "") or "agent"),
            should_cancel=should_cancel, supersede_run_id=supersede_run_id)
        self._live[conversation_id] = AgentLiveSnapshot(
            status=outcome.status, step=len(live_steps),
            steps=tuple(live_steps[-5:]), subagents=tuple(subagents))
        return outcome

    # -- rail reads -----------------------------------------------------

    def live_snapshot(self, conversation_id: str) -> AgentLiveSnapshot:
        return self._live.get(conversation_id, AgentLiveSnapshot())

    def subagent_runs(self, conversation_id: str) -> list[dict]:
        return [r for r in self._db.list_runs(conversation_id)
                if r["agent_kind"] == "subagent"]

    def subagent_run(self, run_id: str) -> dict | None:
        return self._db.get_run(run_id)

    def subagent_count(self, conversation_id: str) -> int:
        return self._db.count_subagent_runs(conversation_id)

    # -- internals ------------------------------------------------------

    def _append_marker(self, session_id: str, text: str) -> None:
        try:
            self._store.append_message(
                session_id, role=ConsoleMessageRole.TOOL, content=escape_markup(text))
        except KeyError:
            pass   # session vanished mid-run; the rail still has the live snapshot

    @staticmethod
    def _summarize(step: AgentStep) -> str:
        raw = step.summary or step.result or step.tool_name or step.kind
        return escape_markup(str(raw)[:200])

    def _previous_primary_run_id(self, conversation_id: str) -> str | None:
        for record in self._db.list_runs(conversation_id, include_superseded=False):
            if record["agent_kind"] == "primary":
                return record["id"]
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest Tests/Chat/test_console_agent_bridge.py -q`
Expected: PASS (7 tests). If the spawn test's marker text differs, keep the assertion on substring `"sub-agent"` (the marker copy is `⤷ spawned sub-agent: …`).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_agent_bridge.py
git commit -m "feat(console): agent bridge — streaming adapter, markers, supersede, live snapshot

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Controller swap + `[console] agent_runtime` gate

**Why:** Turn the reply engine into the agent loop at the single swap site, run synchronously via `asyncio.to_thread` with a `should_cancel` closure reading `self._stop_requested` (tree-wide cancellation), and gate it behind config (OFF ⇒ the legacy direct-stream path unchanged).

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (ctor deps + branch in `_stream_assistant_response` + `_run_agent_reply`)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (build `ConsoleAgentBridge`, read the gate, pass into the controller ctor)
- Test: `Tests/Chat/test_console_agent_swap.py`

**Interfaces:**
- Consumes: `ConsoleAgentBridge` (Task 5); the shipped `self._stop_requested` / `self._active_stream_task` machinery.
- Produces: `ConsoleChatController(..., agent_bridge: ConsoleAgentBridge | None = None, agent_runtime_enabled: bool = True)`. A private `async _run_agent_reply(*, resolution, provider_messages, assistant_message_id, prepare_retry, variant_mode) -> ConsoleSubmitResult`. `_stream_assistant_response` branches to it when `self._agent_runtime_enabled and self._agent_bridge is not None`, else runs the legacy body unchanged.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Chat/test_console_agent_swap.py
"""The controller send path runs the agent loop when the bridge is wired."""
import json

import pytest

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN


def _fence(name, args):
    return f'{FENCE_OPEN}\n{json.dumps({"name": name, "arguments": args})}\n```'


class _Gateway:
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = 0

    async def resolve_for_send(self, selection):
        class _R:
            ready = True
            provider = "llama_cpp"
            visible_copy = ""
        return _R()

    async def stream_chat(self, resolution, messages):
        chunks = self._scripts[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk


def _controller(tmp_path, scripts, *, enabled=True):
    gateway = _Gateway(scripts)
    store = ConsoleChatStore()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, provider="llama_cpp", model="test-model",
        agent_bridge=bridge, agent_runtime_enabled=enabled)
    return controller, store, db


@pytest.mark.asyncio
async def test_agent_send_no_tools_streams_like_today(tmp_path):
    controller, store, _db = _controller(tmp_path, [["Tok", "yo."]])
    result = await controller.submit_draft("capital of Japan?")
    assert result.accepted is True
    messages = store.messages_for_session(store.active_session_id)
    assert messages[-1].role is ConsoleMessageRole.ASSISTANT
    assert messages[-1].content == "Tokyo."


@pytest.mark.asyncio
async def test_agent_tool_turn_renders_marker_not_prose(tmp_path):
    controller, store, _db = _controller(
        tmp_path, [[_fence("calculator", {"expression": "6*7"})], ["It is ", "42."]])
    await controller.submit_draft("what is 6*7?")
    messages = store.messages_for_session(store.active_session_id)
    assert any(m.role is ConsoleMessageRole.TOOL for m in messages)
    assert all(FENCE_OPEN not in m.content for m in messages
               if m.role is ConsoleMessageRole.ASSISTANT)


@pytest.mark.asyncio
async def test_stop_cancels_tree_and_persists_cancelled(tmp_path):
    controller, store, db = _controller(
        tmp_path, [[_fence("calculator", {"expression": "1"})], ["late"]])

    original = controller._agent_bridge.run_reply

    def cancel_after_first(*args, **kwargs):
        controller._stop_requested = True         # simulate Stop during the run
        return original(*args, **kwargs)

    controller._agent_bridge.run_reply = cancel_after_first
    await controller.submit_draft("loop please")
    assert db.list_runs("conv" if False else store.active_session_id) or True
    runs = db.list_runs(next(iter([r["conversation_id"]
                                   for r in db.list_runs.__self__.list_runs.__wrapped__(  # noqa: E501
                                       db, store.active_session_id)]), store.active_session_id))
    # Simpler assertion: the primary run persisted as cancelled.
    primary = [r for r in _all_runs(db) if r["agent_kind"] == "primary"]
    assert primary and primary[0]["status"] == "cancelled"


def _all_runs(db):
    # AgentRunsDB has no list-all; read via the conversation the bridge used.
    return db.list_runs.__self__.list_runs(db.__dict__.get("_scope", "")) if False else _scan(db)


def _scan(db):
    import sqlite3
    with db.connection() as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM agent_runs").fetchall()]


@pytest.mark.asyncio
async def test_config_gate_off_uses_legacy_path(tmp_path):
    controller, store, db = _controller(tmp_path, [["legacy answer."]], enabled=False)
    await controller.submit_draft("hi")
    messages = store.messages_for_session(store.active_session_id)
    assert messages[-1].content == "legacy answer."
    # Legacy path never touches AgentRunsDB.
    assert _scan(db) == []
```

> Simplify the cancel test in implementation review if the reflection helpers read awkwardly; the load-bearing assertions are: a TOOL marker exists, the assistant content has no fence, and (cancel) `_scan(db)` shows a primary row with status `cancelled`. Rewrite `_scan`-based reads plainly against `db.connection()` — do not ship the `__wrapped__`/`__self__` gymnastics.

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest Tests/Chat/test_console_agent_swap.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'agent_bridge'`.

- [ ] **Step 3: Write the implementation**

In `console_chat_controller.py` ctor (:114–144) add two params and store them:
```python
        system_prompt: str | None = None,
        agent_bridge: "ConsoleAgentBridge | None" = None,
        agent_runtime_enabled: bool = True,
    ) -> None:
        ...
        self._agent_bridge = agent_bridge
        self._agent_runtime_enabled = agent_runtime_enabled
```
(Use a string annotation / `TYPE_CHECKING` import to avoid a hard import cycle.)

At the TOP of `_stream_assistant_response` body (:621, before `self._active_assistant_message_id = ...`) branch:
```python
        if self._agent_runtime_enabled and self._agent_bridge is not None:
            return await self._run_agent_reply(
                resolution=resolution,
                provider_messages=provider_messages,
                assistant_message_id=assistant_message_id,
                prepare_retry=prepare_retry,
                variant_mode=variant_mode,
            )
```
(Requires threading `variant_mode` from Task 1's signature — pass it through.)

Add the agent orchestration method (place after `_stream_assistant_response`):
```python
    async def _run_agent_reply(
        self,
        *,
        resolution: Any,
        provider_messages: list[dict[str, Any]],
        assistant_message_id: str,
        prepare_retry: bool,
        variant_mode: bool,
    ) -> ConsoleSubmitResult:
        """Run the agent loop as the reply engine, streaming into the target row."""
        self._active_assistant_message_id = assistant_message_id
        self._active_stream_task = asyncio.current_task()
        self._stop_requested = False
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.STREAMING, "Agent running."))
        try:
            session_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            return self._session_closed_result()
        if variant_mode:
            self.store.begin_variant_stream(assistant_message_id)
        elif prepare_retry:
            self.store.prepare_message_retry(assistant_message_id)

        # Split the #620 leading session system message off the payload; the
        # agent config carries it (composed with the operating prompt).
        session_system_prompt = ""
        agent_messages = list(provider_messages)
        if agent_messages and agent_messages[0].get("role") == ConsoleMessageRole.SYSTEM.value:
            session_system_prompt = str(agent_messages[0].get("content", ""))
            agent_messages = agent_messages[1:]

        conversation_id = self._agent_conversation_id(session_id)
        should_cancel = lambda: self._stop_requested  # noqa: E731 — tiny closure

        try:
            outcome = await asyncio.to_thread(
                self._agent_bridge.run_reply,
                conversation_id=conversation_id,
                session_id=session_id,
                resolution=resolution,
                assistant_message_id=assistant_message_id,
                model=self.model or self.configured_model or "",
                session_system_prompt=session_system_prompt,
                agent_messages=agent_messages,
                should_cancel=should_cancel,
                supersede_previous=bool(prepare_retry or variant_mode),
            )
        except asyncio.CancelledError:
            if self._stop_requested:
                try:
                    stopped = self._mark_stream_stopped(
                        assistant_message_id, visible_copy="Response stopped.")
                except KeyError:
                    return self._session_closed_result()
                return ConsoleSubmitResult(True, True, stopped.content)
            raise
        finally:
            if self._active_stream_task is asyncio.current_task():
                self._active_assistant_message_id = None
                self._active_stream_task = None
                self._stop_requested = False

        return self._finalize_agent_reply(
            assistant_message_id, session_id, outcome, variant_mode=variant_mode)

    def _agent_conversation_id(self, session_id: str) -> str:
        """Return the durable id the run store is keyed by (persisted id when set)."""
        try:
            session = self.store.session_for_id(session_id)
            persisted = getattr(session, "persisted_conversation_id", None)
        except Exception:
            persisted = None
        return persisted or session_id

    def _finalize_agent_reply(
        self, assistant_message_id: str, session_id: str, outcome: Any,
        *, variant_mode: bool,
    ) -> ConsoleSubmitResult:
        from tldw_chatbook.Agents.agent_models import (
            RUN_CANCELLED, RUN_DONE)
        if outcome.status == RUN_CANCELLED:
            try:
                stopped = self._mark_stream_stopped(
                    assistant_message_id, visible_copy="Response stopped.")
            except KeyError:
                return self._session_closed_result()
            return ConsoleSubmitResult(True, True, stopped.content)
        try:
            message = self.store.get_message(assistant_message_id)
            if not message.content.strip() and outcome.status != RUN_DONE:
                # Tool-only/stuck/error ended with no visible answer; surface status.
                self.store.append_stream_chunk(
                    assistant_message_id, f"[agent {outcome.status}]")
            if variant_mode:
                completed = self.store.finalize_variant_stream(assistant_message_id)
            else:
                completed = self.store.mark_message_complete(assistant_message_id)
        except KeyError:
            return self._session_closed_result()
        status = (ConsoleRunStatus.COMPLETED if outcome.status == RUN_DONE
                  else ConsoleRunStatus.FAILED)
        copy = ("Response complete." if outcome.status == RUN_DONE
                else f"Agent run {outcome.status}.")
        self._set_run_state(ConsoleRunState(status, copy))
        return ConsoleSubmitResult(True, True, completed.content)
```

> `session_for_id` may not exist on the store; if not, resolve the session by scanning `self.store.sessions()` for `.id == session_id`. Verify against `console_chat_store.py` during implementation and adjust `_agent_conversation_id` accordingly (a plain scan is fine).

Wire the bridge in `chat_screen.py`. Add a gate reader near `_console_config()` (:3564):
```python
    def _console_agent_runtime_enabled(self) -> bool:
        value = self._console_config().get("agent_runtime", True)
        return bool(value) if isinstance(value, (bool, int)) else True
```
Add a lazy bridge builder near `_ensure_console_chat_store` (:1871):
```python
    def _ensure_console_agent_bridge(self):
        if getattr(self, "_console_agent_bridge", None) is not None:
            return self._console_agent_bridge
        db = getattr(self.app_instance, "chachanotes_db", None)
        db_path = getattr(db, "db_path", None) if db is not None else None
        if not db_path or str(db_path) == ":memory:":
            self._console_agent_bridge = None
            return None
        from pathlib import Path
        from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        runs_db = AgentRunsDB(Path(db_path).parent / "agent_runs.db")
        self._console_agent_bridge = ConsoleAgentBridge(
            agent_runs_db=runs_db,
            store=self._ensure_console_chat_store(),
            provider_gateway=self._ensure_console_provider_gateway(),
        )
        return self._console_agent_bridge
```
Initialize `self._console_agent_bridge = None` where the other console lazy attrs are declared. In the controller construction (:1910) pass:
```python
                system_prompt=selection.system_prompt,
                agent_bridge=self._ensure_console_agent_bridge(),
                agent_runtime_enabled=self._console_agent_runtime_enabled(),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest Tests/Chat/test_console_agent_swap.py Tests/Chat/test_console_chat_controller.py -q`
Expected: PASS — the four swap tests plus the existing controller suite (legacy path must remain byte-for-byte behavior when the bridge is absent).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_agent_swap.py
git commit -m "feat(console): swap the reply engine for the agent loop behind a config gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Rail Agent section + `[N Sub-Agents]` badge + resume

**Why:** The watch-and-drill surface. Add a fifth rail section "Agent" (status line, step tail, sub-agent list with status glyphs, click-through drill-in with a Back affordance) synced through the existing poll fan-out; render the historical `[N Sub-Agents]` badge on conversation rows from `AgentRunsDB.count_subagent_runs`; on conversation load, re-derive markers/rail from the run store (session-scoped TOOL markers are ephemeral — the DB is the durable source).

**Files:**
- Modify: `tldw_chatbook/Chat/console_rail_state.py` (`agent_open` on `ConsoleRailPreferences` + `ConsoleRailState`; coercion/serialize/effective-build)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (compose the Agent section; `_sync_console_agent_section`; feed it from the poll via `_sync_console_settings_summary`; supply `subagent_counts` to the browser-state build; resume re-derivation)
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py` (append the escaped `[N Sub-Agents]` badge to the conversation-row label)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (Agent-section + badge classes) + regenerate bundle
- Test: `Tests/UI/test_console_agent_rail.py`, `Tests/Chat/test_console_rail_state_agent.py`

**Interfaces:**
- Produces: rail-state `agent_open: bool = False`; screen method `_sync_console_agent_section()` reading `self._ensure_console_agent_bridge().live_snapshot(conversation_id)` + `subagent_runs(...)` and updating `#console-agent-section-status`, `#console-agent-section-steps`, `#console-agent-section-subagents` Statics; a rail "Agent" `ConsoleRailSectionHeader(section_id="agent")` + body mirroring the Model section; drill-in state `self._console_agent_drilldown_run_id` toggled by clicking a sub-agent row (renders that run's steps + a Back button `#console-agent-drilldown-back`); conversation-row badge ` [N Sub-Agents]` appended (dim, escaped) when N>0.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_console_rail_state_agent.py
"""agent_open threads through rail preferences + effective state."""
from tldw_chatbook.Chat.console_rail_state import (
    ConsoleRailPreferences, serialize_console_rail_preferences,
    deserialize_console_rail_preferences,
)


def test_agent_open_defaults_false_and_round_trips():
    prefs = ConsoleRailPreferences()
    assert prefs.agent_open is False
    blob = serialize_console_rail_preferences(prefs)
    assert blob["agent_open"] is False
    restored = deserialize_console_rail_preferences({**blob, "agent_open": True})
    assert restored.agent_open is True
```

> Match the real serialize/deserialize function names in `console_rail_state.py` (grep `def .*serialize`/`_coerce_bool` at :242/:256/:461) — this test names them as they exist; rename in the test to the actual symbols if they differ.

```python
# Tests/UI/test_console_agent_rail.py
"""Agent rail section + [N Sub-Agents] badge render via a real App (run_test)."""
import pytest

from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow, build_console_conversation_browser_state,
)


def _all_rows(state):
    rows = []
    for section in state.sections:
        rows.extend(section.rows)
        for group in section.groups:
            rows.extend(group.rows)
    return rows


def test_conversation_row_carries_badge_count_for_render():
    row = ConsoleConversationBrowserInputRow(
        row_key="c1", conversation_id="c1", native_session_id=None, title="Alpha",
        scope_type="global", workspace_id=None, workspace_label="",
        updated_sort="2026-07-13T00:00:00Z")
    state = build_console_conversation_browser_state(
        rows=[row], active_workspace_id=None, subagent_counts={"c1": 2})
    assert _all_rows(state)[0].subagent_count == 2


def test_conversation_row_badge_label_is_escaped_and_present():
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        format_console_conversation_row_label,
    )
    label = format_console_conversation_row_label("Beta [x]", subagent_count=3)
    assert "3 Sub-Agents" in label
    assert "\\[x]" in label or "[x]" not in label.replace("Sub-Agents", "")
```

> `format_console_conversation_row_label(title, *, subagent_count)` is a NEW small pure helper you add to `console_workspace_context.py` so the row-label composition (title + escaped ` [N Sub-Agents]`) is unit-testable without mounting. `_conversation_button` (:354) calls it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest Tests/Chat/test_console_rail_state_agent.py Tests/UI/test_console_agent_rail.py -q`
Expected: FAIL — missing `agent_open` and missing `format_console_conversation_row_label`.

- [ ] **Step 3: Write the implementation**

**Rail state** (`console_rail_state.py`): add `agent_open: bool = False` to `ConsoleRailPreferences` (:65) and `ConsoleRailState` (:87); add coercion at :242 (`agent_open=_coerce_bool(raw.get("agent_open"), defaults.agent_open)`), serialize at :256 (`"agent_open": bool(preferences.agent_open)`), and effective build at :461 (`agent_open=preferences.agent_open`).

**Row-label helper** (`console_workspace_context.py`): add
```python
from rich.markup import escape as _escape_markup

def format_console_conversation_row_label(title: str, *, subagent_count: int = 0) -> str:
    base = _escape_markup(str(title))
    if subagent_count > 0:
        return f"{base}  [dim]\\[{subagent_count} Sub-Agents][/dim]"
    return base
```
and call it from `_conversation_button` (:354) when constructing the button label (the browser-row render path passes `subagent_count=row.subagent_count`). Because `_conversation_button` currently wraps text in `Text(str(text))` (no markup), render the label via a markup-enabled `Static`/`Label` or `Text.from_markup(...)` for the badge; keep the base title escaped.

**Agent rail section** (`chat_screen.py`, in `compose_content` after the Model section, before Details @ :5480): add a fifth header + body mirroring the Model section:
```python
                        # Section 5: Agent (run inspector).
                        yield ConsoleRailSectionHeader(
                            "Agent",
                            section_id="agent",
                            open=rail_state.agent_open,
                            id="console-rail-section-header-agent",
                        )
                        agent_body = Vertical(
                            id="console-rail-section-body-agent",
                            classes="console-rail-section-body console-agent-section",
                        )
                        agent_body.styles.height = "auto"
                        if not rail_state.agent_open:
                            agent_body.styles.display = "none"
                        with agent_body:
                            status_line, steps_text, subagents_text = (
                                self._console_agent_section_lines())
                            yield Static(status_line, id="console-agent-section-status",
                                         classes="console-agent-section-line", markup=False)
                            yield Static(steps_text, id="console-agent-section-steps",
                                         classes="console-agent-section-steps", markup=False)
                            yield Static(subagents_text, id="console-agent-section-subagents",
                                         classes="console-agent-section-subagents", markup=False)
```
Add the section-lines builder + the sync method + the drill-in reader:
```python
    def _console_agent_section_lines(self) -> tuple[str, str, str]:
        bridge = self._ensure_console_agent_bridge()
        conversation_id = self._current_console_rail_conversation_id() or ""
        if bridge is None:
            return ("Agent: unavailable", "", "")
        drill = getattr(self, "_console_agent_drilldown_run_id", None)
        if drill:
            record = bridge.subagent_run(drill)
            if record is not None:
                steps = "\n".join(
                    f"{s.get('kind')}: {escape_markup(str(s.get('summary') or s.get('result') or ''))[:80]}"
                    for s in record.get("steps", []))
                return (f"Sub-agent · {record.get('status')} (Back)",
                        steps, escape_markup(str(record.get('task') or '')))
        snapshot = bridge.live_snapshot(conversation_id)
        status = f"Agent: {snapshot.status}"
        if snapshot.status == "running":
            status = f"Agent: running · step {snapshot.step}"
        steps = "\n".join(f"· {escape_markup(s.text)[:80]}" for s in snapshot.steps)
        glyphs = {"done": "✓", "running": "●", "stuck": "⚠", "error": "✗", "cancelled": "✗"}
        subagents = "\n".join(
            f"{glyphs.get(s.status, '●')} {escape_markup(s.text)[:60]}"
            for s in snapshot.subagents)
        return (status, steps, subagents)

    def _sync_console_agent_section(self) -> None:
        try:
            status_line, steps_text, subagents_text = self._console_agent_section_lines()
            self.query_one("#console-agent-section-status", Static).update(status_line)
            self.query_one("#console-agent-section-steps", Static).update(steps_text)
            self.query_one("#console-agent-section-subagents", Static).update(subagents_text)
        except (NoMatches, QueryError):
            pass
```
Call `self._sync_console_agent_section()` from the end of `_sync_console_settings_summary` (:1670) so the 0.2s poll refreshes the Agent section every tick (the poll already calls `_sync_console_settings_summary` via `_sync_native_console_chat_ui`). Add `self._console_agent_drilldown_run_id: str | None = None` to the console attrs; toggle it in the rail section-toggle / sub-agent click handler and add a Back affordance clearing it, then `run_worker(self._sync_native_console_chat_ui())`.

**Badge count source** (`chat_screen.py`, browser-state build @ :3545): pass counts derived from the bridge:
```python
        bridge = self._ensure_console_agent_bridge()
        subagent_counts = {}
        if bridge is not None:
            for input_row in rows:
                cid = getattr(input_row, "conversation_id", None)
                if cid:
                    subagent_counts[cid] = bridge.subagent_count(cid)
        browser = build_console_conversation_browser_state(
            rows=rows,
            active_workspace_id=self._current_console_workspace_context().active_workspace_id,
            group_collapse_preferences=(
                self._console_conversation_browser_collapse_preferences()),
            query=query,
            marks_available=marks_service is not None,
            error_copy=error_copy or self._console_conversation_browser_error,
            result_total_count=total,
            result_limit=CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
            subagent_counts=subagent_counts,
        )
```

**Resume:** because TOOL store markers are session-scoped and the DB is durable, on conversation load the rail Agent section (which reads the bridge's `live_snapshot`/DB) already reflects historical runs; the count badge derives from `count_subagent_runs`. No new persistence — the resume test asserts that after a run, re-reading `subagent_count` + `subagent_runs` from a fresh bridge over the same DB reproduces the markers/rail data.

**CSS** (`_agentic_terminal.tcss`, after the Model section block near :2111) add:
```tcss
.console-agent-section-line {
    height: 1;
    color: $ds-text-primary;
}

.console-agent-section-steps,
.console-agent-section-subagents {
    height: auto;
    color: $ds-text-muted;
    text-style: dim;
}
```
Regenerate: `$PY tldw_chatbook/css/build_css.py`. Add a pin test asserting `.console-agent-section-steps` is present in both the source and the bundle (extend `Tests/UI/test_console_agent_tool_row_css.py`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest Tests/Chat/test_console_rail_state_agent.py Tests/UI/test_console_agent_rail.py Tests/UI/test_console_agent_tool_row_css.py Tests/Chat/test_console_rail_state.py -q`
Expected: PASS. Then a real-App smoke check:
```python
# add to Tests/UI/test_console_agent_rail.py
@pytest.mark.asyncio
async def test_agent_rail_section_mounts():
    from tldw_chatbook.app import TldwCli
    app = TldwCli()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Navigating to Console mounts the rail; the Agent header exists.
        assert app.query("#console-rail-section-header-agent") is not None
```
(If the Console tab is not the default screen, drive to it with the app's tab-switch binding before asserting; keep this smoke test tolerant — its job is "the section composes without error".)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_rail_state.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_workspace_context.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_agent_rail.py Tests/Chat/test_console_rail_state_agent.py Tests/UI/test_console_agent_tool_row_css.py
git commit -m "feat(console): rail Agent inspector, [N Sub-Agents] badge, drill-in, resume

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Live gate — served captures on llama.cpp + Console suite sweep + STOP

**Why:** Prove the vertical slice end-to-end on the real local model, then STOP for user approval before any PR (the program's gate convention).

**Files:** none shipped (captures are artifacts). Use the established served-capture recipe (`textual-serve` at 2050×1240; the `cap.py` https-only route-abort + body-first-byte recipe from the core-loop UAT memory).

- [ ] **Step 1: Broad Console suite sweep**

Run:
```
$PY -m pytest Tests/Agents Tests/DB/test_agent_runs_db.py Tests/Chat/test_console_variant_stream.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_agent_swap.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_rail_state.py Tests/Chat/test_console_rail_state_agent.py Tests/Workspaces Tests/UI/test_console_agent_rail.py Tests/UI/test_console_agent_tool_row_css.py Tests/UI/test_console_workspace_context_rail.py -q
```
Expected: ALL PASS. Then the full Console-touching regression: `$PY -m pytest Tests/UI Tests/Chat Tests/Workspaces -q` (baseline: ~33 pre-existing UI failures per the dev-environment memory — confirm no NEW failures attributable to this branch).

- [ ] **Step 2: Live served captures on llama.cpp (2050×1240)**

Start the app under `textual-serve`; point the Console at a running llama.cpp server + a tool-capable local model (e.g. the 27B). With `[console] agent_runtime = true` (default), drive one conversation that:
1. asks a math question → the primary calls `calculator` → capture: a TOOL marker in the transcript (not fenced prose) + the rail Agent section showing `Agent: running · step N` then `done`, with the tool step in the tail;
2. asks something that makes the primary `spawn_subagent` → capture: a spawn TOOL marker in the transcript, the rail sub-agent list showing the child with a status glyph, click-through drill-in rendering the child's step log with Back, and the conversation-row `[1 Sub-Agents]` badge;
3. starts a long run and presses **Stop** → capture: "stopping…"/stopped copy, the run halting at a step boundary, and (verify via the run store) the primary run persisted `cancelled`;
4. reload the conversation → capture: the `[N Sub-Agents]` badge + rail re-derived from the run store (markers survive without being conversation messages).

Save captures under `Docs/superpowers/qa/agent-runtime-console-live-<date>/`.

- [ ] **Step 3: STOP for user approval**

Do NOT open a PR. Present the captures + the suite results and request explicit approval (Notes-redesign approval-gate convention: every screen change needs explicit sign-off before merge). Note any live-model degradations (e.g. the model ignoring the fence-first rule → graceful plain-answer fallback per spec) for the follow-up list.

---

## Plan-B exit criteria

- **Suites green** under the venv python: the new pure suites (`test_agent_stream`, `test_agent_service_on_step`, `test_console_variant_stream`, `test_console_agent_bridge`, `test_conversation_browser_subagents`, `test_console_rail_state_agent`), the swap suite (`test_console_agent_swap`), the UI pins (`test_console_agent_tool_row_css`, `test_console_agent_rail`), and NO new failures in the broad `Tests/UI Tests/Chat Tests/Workspaces` sweep (against the ~33-failure baseline).
- **Spec coverage** (Plan-B sections): no-tool send streams like today ✔ (Task 6); tool turn renders a TOOL marker not prose ✔ (Tasks 5/6); spawn renders marker + badge increment + rail lists it ✔ (Tasks 5/7); Stop cancels the tree, `cancelled` persisted ✔ (Tasks 5/6); resume re-derives markers/rail from the run store ✔ (Task 7); retry/regenerate supersede the prior run tree ✔ (Tasks 5/6); config gate OFF = legacy path ✔ (Task 6). `ConsoleRunStatus` unchanged; no new `ConsoleSessionSettings` fields; #620 system prompt composed, never clobbered; every rendered agent/tool string escaped; new styled classes pinned in source + bundle.
- **Live gate** captured on real llama.cpp (calculator call + sub-agent spawn in transcript + rail + badge; Stop cancels the tree) at 2050×1240 per the established recipe.
- **STOP for approval** before PR (program gate convention). Deferred, documented follow-ups: per-session agent config (`ConsoleSessionSettings` fields); strict transcript step/answer interleave ordering (markers currently append after the pre-created answer row in a tool turn — the rail is the chronological source); clean assistant-row reset when streaming continues after a disobedient mid-stream fence; live intra-sub-agent step streaming in the rail (currently a sub-agent's steps land when it completes, per the tight depth-1 budgets).
