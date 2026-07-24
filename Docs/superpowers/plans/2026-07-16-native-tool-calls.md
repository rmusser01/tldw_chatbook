# Native Provider Tool-Calls with Fence-Protocol Fallback — Implementation Plan (task-243)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agent runs on tool-capable providers use native `tools=` function-calling (multi-call batches, no per-turn fence-protocol prompt overhead) while every other provider — llama.cpp included — keeps the fence protocol byte-identically.

**Architecture:** A new pure module (`Agents/native_tools.py`) owns the handler-verified capability set, ToolSchema→OpenAI conversion, and response parsing. `agent_service._make_call_model` branches on capability: native mode passes `tools=`, suppresses `render_tool_protocol`, and populates `ModelTurn.tool_calls` (+ a provider-shaped `assistant_message` echo). The engine appends the echo verbatim and answers native calls with `role="tool"` messages keyed on `call_id` — the fence path's text convention is untouched. The Console path flows through `ConsoleProviderGateway.stream_chat`, which grows a `tools=` passthrough and accumulates streamed `delta.tool_calls` fragments into one `ProviderToolCalls` sentinel yielded at stream end (only when tools were requested — plain Console sends never see one); the bridge's `_StreamingModelAdapter` captures the sentinel and returns it as `message.tool_calls`.

**Tech Stack:** Python 3.11, existing agent runtime (`tldw_chatbook/Agents/*`), Console gateway/bridge (`tldw_chatbook/Chat/*`), pytest.

## Global Constraints

- Backlog ACs (task-243): (1) per-provider capability check w/ fence fallback — llama_cpp and other non-`tools` local backends keep working via fallback; (2) `ModelTurn.tool_calls` populated from a real native response for ≥1 cloud provider end-to-end; (3) a native multi-tool-call reply dispatched as multiple `ToolCall` entries in one `run_agent_loop` turn without changes to the loop's dispatch iteration; (4) existing fence-protocol tests and Console agent-reply integration tests pass unchanged for tool-incapable models.
- Vertical-slice spec (`Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md` §protocol): native tool-calls where the provider advertises them; fence-first text protocol as the fallback. Plan-B contract: structured deltas never hit the transcript.
- A provider enters the native set ONLY when all three hold: `PROVIDER_PARAM_MAP` forwards `tools`; its handler returns the RAW OpenAI-compatible response dict (verified — anthropic/google/cohere normalize and DROP `tool_use`, they stay fence-only); it accepts OpenAI-shape `role="tool"` history messages.
- The fence path must stay byte-identical: no new keys in fence-mode history messages, no `tools=` kwarg sent for fence providers, `render_tool_protocol` output unchanged.
- Kill-switch: `[console] native_tool_calls` (default ON), read per-run, mirroring `[console] agent_runtime`'s read shape (`chat_screen.py:2188-2191` `self._console_config().get(...)` — NOT `get_cli_setting` with a dotted section, which is silently broken).
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime`, branch `claude/native-tool-calls`. venv: `source .venv/bin/activate` at repo root (worktrees share it via `PYTHONPATH`/editable install — run pytest from the worktree root with the repo venv's python). `timeout` command unavailable. Run pytest FOREGROUND.
- Code style per CLAUDE.md: Google docstrings on public APIs, early returns, constants for magic values.

## File Structure

- **Create** `tldw_chatbook/Agents/native_tools.py` — capability set + converters (pure, no I/O).
- **Create** `Tests/Agents/test_native_tools.py`.
- **Modify** `tldw_chatbook/Chat/Chat_Functions.py` — `PROVIDER_PARAM_MAP` `tools`/`tool_choice` entries for `groq` and `deepseek` (handlers already accept them; the map silently drops them today).
- **Modify** `tldw_chatbook/Agents/agent_models.py` — `ModelTurn.assistant_message`, `AgentConfig.native_tools`.
- **Modify** `tldw_chatbook/Agents/agent_runtime.py` — history appends only (assistant echo; `role="tool"` results). The dispatch `for call in calls:` iteration is NOT touched (AC #3).
- **Modify** `tldw_chatbook/Agents/agent_service.py` — native branch in `_make_call_model`; spawn propagates `native_tools`.
- **Modify** `tldw_chatbook/Chat/console_provider_gateway.py` — `tools=` passthrough, `_ToolCallAccumulator`, `ProviderToolCalls` sentinel, tool-call-aware content extraction.
- **Modify** `tldw_chatbook/Chat/console_agent_bridge.py` — adapter `tools` forwarding + sentinel capture; `execution_key`-first `api_endpoint`; `native_tools_enabled` seam.
- **Modify** `tldw_chatbook/UI/Screens/chat_screen.py` — `[console] native_tool_calls` read wired into the bridge ctor.
- **Modify/Test** `Tests/Agents/test_agent_runtime.py`, `Tests/Agents/test_agent_service.py`, `Tests/Chat/test_console_provider_gateway.py`, `Tests/Chat/test_console_agent_bridge.py`.

---

### Task 1: `native_tools` module + provider-map gap fixes

**Files:**
- Create: `tldw_chatbook/Agents/native_tools.py`
- Modify: `tldw_chatbook/Chat/Chat_Functions.py` (the `groq` and `deepseek` entries of `PROVIDER_PARAM_MAP`, dict starts ~line 122)
- Test: `Tests/Agents/test_native_tools.py`

**Interfaces:**
- Consumes: `ToolCall`, `ToolSchema` from `tldw_chatbook/Agents/agent_models.py` (frozen dataclasses; `ToolSchema(id, name, description, parameters: dict)`, `ToolCall(name, args: dict, call_id: str = "")`).
- Produces (used verbatim by Tasks 3): `NATIVE_TOOLS_PROVIDERS: frozenset[str]`, `provider_supports_native_tools(api_endpoint: str) -> bool`, `schemas_to_openai_tools(schemas: list) -> list[dict]`, `parse_native_tool_calls(message: dict) -> tuple[ToolCall, ...]`.

Verified ground truth you must NOT re-derive: `chat_with_openai`/`chat_with_groq`/`chat_with_deepseek`/`chat_with_openrouter`/`chat_with_mistral`/`chat_with_moonshot` in `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` all return the raw OpenAI-compatible response dict for non-streaming calls (`return response_data` / `return result` of `response.json()`), so `choices[0].message.tool_calls` survives verbatim. `chat_with_anthropic` normalizes and DROPS `tool_use` content blocks (only `type=="text"` parts extracted, ~line 890); google/cohere likewise normalize — those stay OUT of the set. `chat_with_groq` (def ~line 1830) and `chat_with_deepseek` (def ~line 1367) already accept `tools`/`tool_choice` parameters, but their `PROVIDER_PARAM_MAP` entries never map them, so `chat_api_call` silently drops `tools=` before dispatch — that is the map fix. Do verify (read, don't assume) that both handler signatures actually name the params `tools` and `tool_choice` before mapping them.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Agents/test_native_tools.py`:

```python
# Tests/Agents/test_native_tools.py
"""native_tools: capability set, OpenAI conversion, response parsing."""
import json

from tldw_chatbook.Agents.agent_models import ToolSchema
from tldw_chatbook.Agents.native_tools import (
    NATIVE_TOOLS_PROVIDERS, parse_native_tool_calls,
    provider_supports_native_tools, schemas_to_openai_tools,
)
from tldw_chatbook.Chat.Chat_Functions import PROVIDER_PARAM_MAP


def test_capability_set_membership():
    assert provider_supports_native_tools("openai")
    assert provider_supports_native_tools("groq")
    assert provider_supports_native_tools("OpenAI")   # case-insensitive
    assert not provider_supports_native_tools("llama_cpp")
    assert not provider_supports_native_tools("local_llamacpp")
    assert not provider_supports_native_tools("anthropic")  # normalizer drops tool_use
    assert not provider_supports_native_tools("")
    assert not provider_supports_native_tools(None)


def test_every_native_provider_forwards_tools_in_param_map():
    for provider in NATIVE_TOOLS_PROVIDERS:
        mapping = PROVIDER_PARAM_MAP.get(provider)
        assert mapping is not None, provider
        assert mapping.get("tools") == "tools", provider


def test_schemas_to_openai_tools_shape_and_empty_parameters_default():
    schema = ToolSchema(id="b:calc", name="calculator",
                        description="Evaluate math.",
                        parameters={"type": "object",
                                    "properties": {"expression": {"type": "string"}},
                                    "required": ["expression"]})
    bare = ToolSchema(id="b:ping", name="ping", description="Ping.", parameters={})
    tools = schemas_to_openai_tools([schema, bare])
    assert tools[0] == {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate math.",
        "parameters": schema.parameters}}
    assert tools[1]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert schemas_to_openai_tools([]) == []


def _raw_call(name, args, call_id="c1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_parse_native_tool_calls_happy_path_and_order():
    message = {"content": None, "tool_calls": [
        _raw_call("calculator", {"expression": "2+2"}, "a"),
        _raw_call("get_current_datetime", {}, "b")]}
    calls = parse_native_tool_calls(message)
    assert [(c.name, c.args, c.call_id) for c in calls] == [
        ("calculator", {"expression": "2+2"}, "a"),
        ("get_current_datetime", {}, "b")]


def test_parse_native_tool_calls_malformed_and_junk():
    message = {"tool_calls": [
        {"id": "x", "type": "function",
         "function": {"name": "calculator", "arguments": "{not json"}},
        {"id": "y", "type": "function",
         "function": {"name": "calculator", "arguments": {"expression": "1"}}},
        {"id": "z", "type": "function", "function": {"name": ""}},
        "junk", {"function": "junk"}]}
    calls = parse_native_tool_calls(message)
    # Malformed arguments -> args={} (the tool's own validation error is
    # echoed back so the model can retry); dict arguments accepted as-is;
    # nameless/junk entries dropped.
    assert [(c.name, c.args) for c in calls] == [
        ("calculator", {}), ("calculator", {"expression": "1"})]
    assert parse_native_tool_calls({}) == ()
    assert parse_native_tool_calls({"tool_calls": None}) == ()
    assert parse_native_tool_calls(None) == ()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Agents/test_native_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: ... native_tools`.

- [ ] **Step 3: Implement**

Create `tldw_chatbook/Agents/native_tools.py`:

```python
# tldw_chatbook/Agents/native_tools.py
"""Native provider tool-calls: capability check, conversion, parsing.

The fence-first text protocol (``agent_runtime.render_tool_protocol``)
remains the fallback for every provider not listed here — see the
vertical-slice spec and the task-231 tool-call flow review (opportunity 1).

A provider earns a place in ``NATIVE_TOOLS_PROVIDERS`` only when ALL of:

1. ``PROVIDER_PARAM_MAP`` forwards ``tools`` (and the handler accepts it),
2. the handler returns the RAW OpenAI-compatible response dict, so
   ``choices[0].message.tool_calls`` survives verbatim (the anthropic /
   google / cohere handlers normalize and currently DROP tool-use blocks —
   they stay fence-only until their normalizers build
   ``message.tool_calls``; follow-up filed as task-246), and
3. the provider accepts OpenAI-shape ``role: "tool"`` history messages.

Pure module: no I/O, no provider imports.
"""
from __future__ import annotations

import json

from .agent_models import ToolCall, ToolSchema

NATIVE_TOOLS_PROVIDERS = frozenset({
    "openai", "groq", "openrouter", "mistral", "deepseek", "moonshot",
    "custom-openai-api", "custom-openai-api-2",
})

_EMPTY_PARAMETERS = {"type": "object", "properties": {}}


def provider_supports_native_tools(api_endpoint: str | None) -> bool:
    """Return whether ``api_endpoint`` supports native tool-calls end-to-end.

    Args:
        api_endpoint: The ``chat_api_call`` provider key. The Console passes
            ``ConsoleProviderResolution.execution_key`` — the key
            ``PROVIDER_PARAM_MAP`` is indexed by.

    Returns:
        True when the provider forwards ``tools=`` AND returns the raw
        OpenAI-compatible response shape (see module docstring).
    """
    return str(api_endpoint or "").strip().lower() in NATIVE_TOOLS_PROVIDERS


def schemas_to_openai_tools(schemas: list[ToolSchema]) -> list[dict]:
    """Convert ``ToolSchema`` entries to the OpenAI ``tools=`` wire format.

    Args:
        schemas: Disclosed tool schemas (runtime + active), in order.

    Returns:
        One ``{"type": "function", "function": {...}}`` entry per schema;
        an empty ``parameters`` dict is replaced with the minimal valid
        object schema (providers reject ``{}``).
    """
    tools = []
    for schema in schemas:
        tools.append({
            "type": "function",
            "function": {
                "name": schema.name,
                "description": schema.description,
                "parameters": schema.parameters or dict(_EMPTY_PARAMETERS),
            },
        })
    return tools


def parse_native_tool_calls(message: dict | None) -> tuple[ToolCall, ...]:
    """Parse OpenAI-shape ``message.tool_calls`` into ``ToolCall`` entries.

    Malformed ``arguments`` JSON yields ``args={}`` rather than dropping
    the call: the downstream tool's own validation error is echoed back to
    the model as a normal tool result, so it can retry with corrected
    arguments. Entries without a function name are dropped.

    Args:
        message: The ``choices[0].message`` dict from a provider response
            (or anything — junk yields no calls).

    Returns:
        Parsed calls in provider order, each carrying its ``call_id``.
    """
    if not isinstance(message, dict):
        return ()
    calls = []
    for raw in message.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        raw_args = function.get("arguments")
        args: dict = {}
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                args = parsed
        calls.append(ToolCall(name=name, args=args,
                              call_id=str(raw.get("id") or "")))
    return tuple(calls)
```

In `tldw_chatbook/Chat/Chat_Functions.py`, add to the **`groq`** entry of `PROVIDER_PARAM_MAP` (after its `'model':'model',` line) and to the **`deepseek`** entry (after its `'model':'model',` line) — first read both handler signatures in `LLM_Calls/LLM_API_Calls.py` (`chat_with_groq` ~1830, `chat_with_deepseek` ~1367) and confirm they take `tools` and `tool_choice` params; then add:

```python
        'tools': 'tools',
        'tool_choice': 'tool_choice',
```

Do NOT touch any other provider entry.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Agents/test_native_tools.py -v`
Expected: PASS (all).

- [ ] **Step 5: Regression + commit**

Run: `python -m pytest Tests/Agents/ Tests/Chat/test_chat_functions.py -q` (if that file doesn't exist, run `python -m pytest Tests/Agents/ -q` plus `python -m pytest Tests/Chat/ -q -k "chat_api or provider_param or Chat_Functions"`).
Expected: PASS.

```bash
git add tldw_chatbook/Agents/native_tools.py Tests/Agents/test_native_tools.py tldw_chatbook/Chat/Chat_Functions.py
git commit -m "feat(agents): native-tools capability set + OpenAI converters; map groq/deepseek tools passthrough (task-243)"
```

---

### Task 2: Engine — native history convention (assistant echo + role="tool" results)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (the `ModelTurn` dataclass, ~line 72)
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (assistant append ~line 274; tool-result append ~lines 382-384; `run_agent_loop` docstring's message-convention paragraph)
- Test: `Tests/Agents/test_agent_runtime.py` (append new tests; follow the file's existing fake-`LoopDeps` style)

**Interfaces:**
- Consumes: `ModelTurn`, `ToolCall` (Task 1 unchanged fields).
- Produces (relied on by Task 3): `ModelTurn` gains `assistant_message: dict | None = None`; the loop appends `turn.assistant_message` verbatim when present, else the existing `{"role": "assistant", "content": turn.text}`; a call with non-empty `call_id` gets its result appended as `{"role": "tool", "tool_call_id": call.call_id, "content": content}`, else the existing user-role `Tool result for {name}: {content}` line.

The dispatch iteration `for call in calls:` (agent_runtime.py ~line 276) already handles multi-call batches — DO NOT modify it (AC #3: native multi-call dispatch works "without engine changes" to the iteration). Only the two history-append sites change, both gated so the fence path is byte-identical (fence turns have `assistant_message=None` and `call_id=""`).

- [ ] **Step 1: Write the failing tests**

Read `Tests/Agents/test_agent_runtime.py` first and reuse its existing deps/fake helpers. Add (adapting helper names to the file's actual fixtures — the asserts below are the contract):

```python
def _native_turn(calls, text=""):
    raw = [{"id": c.call_id, "type": "function",
            "function": {"name": c.name, "arguments": json.dumps(c.args)}}
           for c in calls]
    return ModelTurn(text=text, tool_calls=tuple(calls),
                     assistant_message={"role": "assistant", "content": text,
                                        "tool_calls": raw})


def test_native_multi_call_batch_dispatches_both_in_one_turn():
    """AC #3: two native calls in one reply -> two tool invocations before
    the next model turn, results paired to call ids as role='tool'."""
    calls = [ToolCall("echo", {"v": "1"}, call_id="idA"),
             ToolCall("echo", {"v": "2"}, call_id="idB")]
    seen_messages = []
    turns = iter([_native_turn(calls), ModelTurn(text="done")])

    def call_model(messages, active):
        seen_messages.append([dict(m) for m in messages])
        return next(turns)

    invoked = []

    def invoke_tool(call):
        invoked.append(call)
        return ToolResult(ok=True, content=f"ok:{call.args['v']}")

    outcome = run_agent_loop(CFG, [{"role": "user", "content": "go"}], [],
                             deps_with(call_model=call_model,
                                       invoke_tool=invoke_tool))
    assert outcome.status == RUN_DONE and outcome.final_text == "done"
    assert [c.call_id for c in invoked] == ["idA", "idB"]  # one turn, both dispatched
    second_turn_history = seen_messages[1]
    assistant = second_turn_history[1]
    assert assistant["tool_calls"][0]["id"] == "idA"      # provider echo verbatim
    tool_msgs = [m for m in second_turn_history if m.get("role") == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_msgs] == [
        ("idA", "ok:1"), ("idB", "ok:2")]
    assert not any(m.get("role") == "user" and
                   str(m.get("content", "")).startswith("Tool result for")
                   for m in second_turn_history[1:])


def test_fence_history_convention_unchanged():
    """A fence-parsed call (call_id='') keeps the plain-text convention:
    assistant text verbatim, user-role 'Tool result for ...' line, and NO
    new keys leak into fence-mode history messages."""
    # Script: fence tool-call turn (via turn.text) then final answer, using
    # the file's existing fence() helper; capture second-turn messages.
    # Assert: history[1] == {"role": "assistant", "content": <fence text>}
    # exactly (keys == {"role", "content"}); history[2]["role"] == "user"
    # and history[2]["content"].startswith("Tool result for").
```

Write `test_fence_history_convention_unchanged` fully, modeled on the file's existing fence-round-trip test, with the exact key-set assertion `set(history[1].keys()) == {"role", "content"}`.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Tests/Agents/test_agent_runtime.py -v -k "native_multi or fence_history"`
Expected: the native test FAILS (`assistant_message` unexpected kwarg / role-tool messages absent); the fence test may already pass — that is fine, it is the pinned regression.

- [ ] **Step 3: Implement**

`agent_models.py` — replace the `ModelTurn` block:

```python
@dataclass(frozen=True)
class ModelTurn:
    """One provider response: raw text plus any native tool calls.

    ``assistant_message`` carries the provider-shaped assistant message for
    native tool-call turns (content plus the raw ``tool_calls`` array,
    echoed verbatim into history so the follow-up ``role="tool"`` results
    pair with their calls by id). ``None`` for fence-protocol turns, whose
    history keeps the plain-text convention.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    assistant_message: dict | None = None
```

`agent_runtime.py` — the assistant append (~line 274) becomes:

```python
        messages.append(turn.assistant_message
                        or {"role": "assistant", "content": turn.text})
```

The tool-result append (~lines 382-384) becomes:

```python
            if call.call_id:
                # Native protocol: providers require each tool_call_id to be
                # answered by a role="tool" message paired to the assistant
                # turn's tool_calls entry.
                messages.append({
                    "role": "tool", "tool_call_id": call.call_id,
                    "content": content})
            else:
                messages.append({
                    "role": "user",
                    "content": f"Tool result for {call.name}: {content}"})
```

Update the `run_agent_loop` docstring's "Message convention" sentence to cover both conventions (fence: verbatim assistant text + user-role result lines; native: provider-shaped assistant echo + `role="tool"` results).

- [ ] **Step 4: Run the full engine suite**

Run: `python -m pytest Tests/Agents/test_agent_runtime.py -v`
Expected: PASS (new tests + every existing fence test unchanged).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/agent_runtime.py Tests/Agents/test_agent_runtime.py
git commit -m "feat(agents): native history convention — assistant echo + role=tool results keyed on call_id (task-243)"
```

---

### Task 3: Service — native branch in `_make_call_model` + `native_tools` config

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (`AgentConfig`, ~line 100)
- Modify: `tldw_chatbook/Agents/agent_service.py` (`_make_call_model` lines 107-121; the `spawn` closure's child `AgentConfig` in `_run_one`; `_response_text` gets a `_response_message` sibling)
- Test: `Tests/Agents/test_agent_service.py`

**Interfaces:**
- Consumes: Task 1's `provider_supports_native_tools` / `schemas_to_openai_tools` / `parse_native_tool_calls`; Task 2's `ModelTurn.assistant_message`.
- Produces (relied on by Task 5): `AgentConfig` gains `native_tools: bool = True`; `_make_call_model` passes `tools=<openai list>` (and NO fence protocol) iff `config.native_tools and provider_supports_native_tools(api_endpoint)`; the extra kwarg reaches `self.chat_call(...)` as `tools=` ONLY in native mode (fence-mode call kwargs are byte-identical to today).

- [ ] **Step 1: Write the failing tests**

In `Tests/Agents/test_agent_service.py`, extend the fakes minimally (keep every existing test unchanged):

```python
def provider_reply(item):
    """str -> plain content reply; dict -> used as the full message."""
    if isinstance(item, dict):
        return {"choices": [{"message": item}]}
    return {"choices": [{"message": {"content": item}}]}


def native_call(name, args, call_id="c1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}
```

(`ScriptedChat` needs no change — `provider_reply` already runs inside it.) Add tests:

```python
NATIVE_CFG = dataclasses.replace(CFG)  # native_tools defaults True


def test_native_endpoint_sends_tools_and_suppresses_fence_protocol(db):
    service, chat = make_service(db, [
        {"content": None,
         "tool_calls": [native_call("calculator", {"expression": "2+2"})]},
        "4."])
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=CFG, api_endpoint="groq", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    first = chat.calls[0]
    names = [t["function"]["name"] for t in first["tools"]]
    assert "calculator" in names and "spawn_subagent" in names
    assert "tool_call" not in first["messages_payload"][0]["content"]  # no fence protocol
    # Second call's history carries the native pairing:
    second_payload = chat.calls[1]["messages_payload"]
    assistant = [m for m in second_payload if m["role"] == "assistant"][0]
    assert assistant["tool_calls"][0]["function"]["name"] == "calculator"
    tool_msg = [m for m in second_payload if m.get("role") == "tool"][0]
    assert tool_msg["tool_call_id"] == "c1" and "4" in tool_msg["content"]


def test_native_multi_call_reply_dispatches_both_tools_in_one_turn(db):
    service, chat = make_service(db, [
        {"content": None, "tool_calls": [
            native_call("calculator", {"expression": "2+2"}, "a"),
            native_call("get_current_datetime", {}, "b")]},
        "done"])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "go"}],
        config=CFG, api_endpoint="openai", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE
    tool_results = [s for s in outcome.steps if s.kind == "tool_result"]
    assert [s.tool_name for s in tool_results] == [
        "calculator", "get_current_datetime"]
    assert len(chat.calls) == 2  # one batch turn + one final turn


def test_fence_fallback_unchanged_for_llama_cpp(db):
    service, chat = make_service(db, [fence("calculator",
                                            {"expression": "2+2"}), "4."])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=CFG, api_endpoint="llama_cpp", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE
    assert "tools" not in chat.calls[0]                     # no tools= kwarg at all
    assert "tool_call" in chat.calls[0]["messages_payload"][0]["content"]


def test_native_kill_switch_forces_fence(db):
    cfg = dataclasses.replace(CFG, native_tools=False)
    service, chat = make_service(db, [fence("calculator",
                                            {"expression": "2+2"}), "4."])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=cfg, api_endpoint="groq", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE and "tools" not in chat.calls[0]


def test_native_subagent_turns_also_carry_tools(db):
    service, chat = make_service(db, [
        {"content": None,
         "tool_calls": [native_call("spawn_subagent",
                                    {"task": "say hi"}, "s1")]},
        "hi from child",   # child's (native-mode) only turn
        "done"])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "go"}],
        config=CFG, api_endpoint="groq", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE
    child_call = chat.calls[1]
    assert child_call["messages_payload"][0]["content"].startswith(
        SUBAGENT_SYSTEM_PROMPT)
    assert "tools" in child_call         # native_tools propagated to the child


def test_malformed_native_arguments_error_is_echoed_and_recoverable(db):
    bad = {"id": "m1", "type": "function",
           "function": {"name": "calculator", "arguments": "{broken"}}
    service, chat = make_service(db, [
        {"content": None, "tool_calls": [bad]},
        {"content": None,
         "tool_calls": [native_call("calculator", {"expression": "2+2"},
                                    "m2")]},
        "4."])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=CFG, api_endpoint="groq", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    retry_payload = chat.calls[1]["messages_payload"]
    tool_msgs = [m for m in retry_payload if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "m1"
    assert "ERROR" in tool_msgs[0]["content"]  # empty-args invoke fails, echoed
```

Notes: `import dataclasses` at top; `CFG` is the file's existing config constant. If `calculator` invoked with `{}` returns ok instead of an error, assert on whatever the real `ToolResult` carries — the contract is "the call is dispatched with `args={}` and its result (success or error) is echoed as the `role='tool'` message", not the specific error text.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Tests/Agents/test_agent_service.py -v -k "native or kill_switch or llama_cpp"`
Expected: FAIL (`AgentConfig` has no `native_tools`; no `tools` kwarg recorded).

- [ ] **Step 3: Implement**

`agent_models.py` — `AgentConfig` gains the flag:

```python
@dataclass(frozen=True)
class AgentConfig:
    model: str
    system_prompt: str
    allowed_tools: tuple[str, ...] = ()
    budget: RunBudget = field(default_factory=RunBudget)
    native_tools: bool = True
```

`agent_service.py` — imports:

```python
from .native_tools import (
    parse_native_tool_calls, provider_supports_native_tools,
    schemas_to_openai_tools,
)
```

Add beside `_response_text`:

```python
def _response_message(resp) -> dict:
    try:
        message = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {}
    return message if isinstance(message, dict) else {}
```

Replace `_make_call_model`:

```python
    def _make_call_model(self, config: AgentConfig, api_endpoint: str,
                         runtime_schemas: list):
        native = (config.native_tools
                  and provider_supports_native_tools(api_endpoint))

        def call_model(messages: list[dict], active_schemas: tuple) -> ModelTurn:
            schemas = runtime_schemas + list(active_schemas)
            system_content = config.system_prompt
            call_kwargs: dict = {}
            if native:
                # Native mode: the provider carries the tool catalog in
                # tools= — no fence-protocol section in the system prompt.
                tools = schemas_to_openai_tools(schemas)
                if tools:
                    call_kwargs["tools"] = tools
            else:
                protocol = render_tool_protocol(schemas)
                if protocol:
                    system_content = f"{config.system_prompt}\n\n{protocol}"
            payload = [{"role": "system", "content": system_content}]
            payload.extend(messages)
            resp = self.chat_call(
                api_endpoint=api_endpoint, messages_payload=payload,
                streaming=False, model=config.model, **call_kwargs)
            text = _response_text(resp)
            if not native:
                return ModelTurn(text=text)
            message = _response_message(resp)
            tool_calls = parse_native_tool_calls(message)
            assistant_message = None
            if tool_calls:
                assistant_message = {
                    "role": "assistant", "content": text,
                    "tool_calls": message.get("tool_calls")}
            return ModelTurn(text=text, tool_calls=tool_calls,
                             assistant_message=assistant_message)
        return call_model
```

In `_run_one`'s `spawn` closure, the child `AgentConfig(...)` gains `native_tools=config.native_tools,` (keep every other field as-is).

- [ ] **Step 4: Run the full service suite**

Run: `python -m pytest Tests/Agents/ -v`
Expected: PASS — every pre-existing test unchanged (fence-mode call kwargs identical: no `tools` key ever recorded for non-native endpoints).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/agent_service.py Tests/Agents/test_agent_service.py
git commit -m "feat(agents): native tools= call path in AgentService with fence fallback (task-243)"
```

---

### Task 4: Gateway — `tools=` passthrough + streamed tool-call accumulation

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Test: `Tests/Chat/test_console_provider_gateway.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-3 (Chat-layer only; no Agents imports).
- Produces (relied on by Task 5): `ProviderToolCalls` frozen dataclass (`tool_calls: tuple[dict, ...]`, OpenAI shape, fragments merged) exported from `console_provider_gateway`; `stream_chat(resolution, messages, tools: list | None = None)` yields `str` chunks and — ONLY when `tools` was passed and calls were seen — a single final `ProviderToolCalls`. With `tools=None` the generator's behavior and types are byte-identical to today.

Anchors in the file: `stream_chat` (~line 748; llama_cpp branch ignores `tools` — native never routes there), `_stream_generic_chat` (~796, the `worker()` closure), `_chat_api_kwargs` (~887, ends with a `if value is not None` filter — so a `"tools": tools` entry is auto-dropped when None), `_QueueItem` (~217), `_content_from_provider_mapping` (~1100), `_content_from_sse_data` (~1085), copy constants `NO_PROVIDER_CONTENT_COPY` / `UNSUPPORTED_PROVIDER_RESPONSE_COPY`.

- [ ] **Step 1: Write the failing tests**

Read the existing `Tests/Chat/test_console_provider_gateway.py` fixtures first (fake `_chat_api_call_fn` injection, resolution builder) and reuse them. Add:

```python
def _sse(payload):
    return "data: " + json.dumps(payload)


def _delta_fragment(index, call_id=None, name=None, arguments=None):
    frag = {"index": index, "function": {}}
    if call_id is not None:
        frag["id"] = call_id
        frag["type"] = "function"
    if name is not None:
        frag["function"]["name"] = name
    if arguments is not None:
        frag["function"]["arguments"] = arguments
    return {"choices": [{"delta": {"tool_calls": [frag]}}]}


TOOLS = [{"type": "function", "function": {
    "name": "calculator", "description": "d",
    "parameters": {"type": "object", "properties": {}}}}]


async def _collect(gateway, resolution, tools=None):
    items = []
    async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "q"}], tools=tools):
        items.append(chunk)
    return items


def test_stream_accumulates_sse_tool_call_fragments(...):
    """OpenAI streaming: id/name on the first fragment, arguments split
    across fragments -> ONE merged ProviderToolCalls yielded last."""
    script = iter([
        _sse(_delta_fragment(0, call_id="c9", name="calculator")),
        _sse(_delta_fragment(0, arguments='{"expres')),
        _sse(_delta_fragment(0, arguments='sion": "2+2"}')),
        "data: [DONE]",
    ])
    # fake chat_api_call_fn returns the iterator; resolution: a generic
    # ready provider with execution_key="groq", streaming=True.
    items = asyncio.run(_collect(gateway, resolution, tools=TOOLS))
    calls = [i for i in items if isinstance(i, ProviderToolCalls)]
    assert len(calls) == 1 and items[-1] is calls[0]
    (call,) = calls[0].tool_calls
    assert call == {"id": "c9", "type": "function", "function": {
        "name": "calculator", "arguments": '{"expression": "2+2"}'}}
    assert not any(isinstance(i, str) and i.strip() for i in items[:-1])  # no copy leaked


def test_non_streaming_message_tool_calls_surface(...):
    """resolution.streaming False: chat_api_call returns the full dict;
    message.tool_calls surfaces as ProviderToolCalls, content as text."""
    response = {"choices": [{"message": {
        "content": "Checking.",
        "tool_calls": [{"id": "n1", "type": "function", "function": {
            "name": "calculator", "arguments": "{}"}}]}}]}
    items = asyncio.run(_collect(gateway, resolution, tools=TOOLS))
    assert "Checking." in [i for i in items if isinstance(i, str)]
    (ptc,) = [i for i in items if isinstance(i, ProviderToolCalls)]
    assert ptc.tool_calls[0]["id"] == "n1"


def test_no_tools_requested_is_byte_identical(...):
    """Same fragment script WITHOUT tools=: no ProviderToolCalls, no new
    strings — the delta-only chunks stay silently dropped as today."""
    items = asyncio.run(_collect(gateway, resolution, tools=None))
    assert all(isinstance(i, str) for i in items)
    assert UNSUPPORTED_PROVIDER_RESPONSE_COPY not in items


def test_tool_call_only_stream_yields_no_fallback_copy(...):
    """A tools= run whose stream carries ONLY tool-call fragments must not
    inject NO_PROVIDER_CONTENT_COPY / UNSUPPORTED copy into the text
    stream (that copy would be echoed into agent history)."""
    items = asyncio.run(_collect(gateway, resolution, tools=TOOLS))
    texts = [i for i in items if isinstance(i, str)]
    assert NO_PROVIDER_CONTENT_COPY not in texts
    assert UNSUPPORTED_PROVIDER_RESPONSE_COPY not in texts
```

Fill the `...` fixture plumbing from the file's existing tests (it already constructs a gateway with an injected `_chat_api_call_fn` and a ready resolution — mirror that exactly; use `execution_key="groq"`, `streaming=True` except the non-streaming test).

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Tests/Chat/test_console_provider_gateway.py -v -k "tool_call or byte_identical or fallback_copy"`
Expected: FAIL (`ImportError: ProviderToolCalls` / unexpected `tools` kwarg).

- [ ] **Step 3: Implement**

In `console_provider_gateway.py`:

(a) Sentinel + accumulator (module level, near `_QueueItem`):

```python
@dataclass(frozen=True)
class ProviderToolCalls:
    """Accumulated native tool-calls, yielded as ``stream_chat``'s FINAL
    item — and only when the caller passed ``tools=``. Plain Console sends
    never receive one. ``tool_calls`` entries are OpenAI-shape dicts with
    streaming fragments already merged."""

    tool_calls: tuple[dict, ...]


class _ToolCallAccumulator:
    """Merges OpenAI streaming ``delta.tool_calls`` fragments (and
    non-streaming ``message.tool_calls`` entries) into complete calls."""

    def __init__(self) -> None:
        self._by_index: dict[int, dict] = {}
        self._order: list[int] = []

    def feed_payload(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            return
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        first = choices[0]
        if not isinstance(first, Mapping):
            return
        message = first.get("message")
        if isinstance(message, Mapping):
            for i, raw in enumerate(message.get("tool_calls") or []):
                if isinstance(raw, Mapping):
                    self._merge(i, raw)
        delta = first.get("delta")
        if isinstance(delta, Mapping):
            for raw in delta.get("tool_calls") or []:
                if isinstance(raw, Mapping):
                    try:
                        index = int(raw.get("index", 0))
                    except (TypeError, ValueError):
                        index = 0
                    self._merge(index, raw)

    def _merge(self, index: int, fragment: Mapping[str, Any]) -> None:
        if index not in self._by_index:
            self._by_index[index] = {"id": "", "type": "function",
                                     "function": {"name": "", "arguments": ""}}
            self._order.append(index)
        entry = self._by_index[index]
        if fragment.get("id"):
            entry["id"] = str(fragment["id"])
        if fragment.get("type"):
            entry["type"] = str(fragment["type"])
        function = fragment.get("function")
        if isinstance(function, Mapping):
            if function.get("name"):
                entry["function"]["name"] = str(function["name"])
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                entry["function"]["arguments"] += arguments
            elif isinstance(arguments, Mapping):
                entry["function"]["arguments"] = json.dumps(arguments)

    def calls(self) -> tuple[dict, ...]:
        return tuple(self._by_index[i] for i in self._order
                     if self._by_index[i]["function"]["name"])


def _decode_stream_item(item: Any) -> Any:
    """Best-effort payload decode for accumulator teeing: mappings pass
    through; SSE ``data: {...}`` strings/bytes are JSON-decoded; anything
    else (comments, [DONE], junk) yields None."""
    if isinstance(item, Mapping):
        return item
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(item, str):
        return None
    data = item.strip()
    if data.startswith("data:"):
        data = data[len("data:"):].strip()
    if not data or data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _tee_tool_calls(response: Any, accumulator: _ToolCallAccumulator) -> Any:
    """Feed every provider item through ``accumulator``, unchanged, for the
    three shapes ``chat_api_call`` returns: a full mapping (non-streaming),
    an iterator of mappings, or an iterator of SSE strings."""
    if isinstance(response, Mapping):
        accumulator.feed_payload(response)
        return response
    if not _is_iterable_response(response):
        return response

    def generator():
        for item in response:
            accumulator.feed_payload(_decode_stream_item(item))
            yield item
    return generator()
```

Compare `_decode_stream_item` against the existing `_content_from_sse_data` prefix handling and match it exactly (e.g. if that helper uses `removeprefix("data:")` semantics, use the same).

(b) `_QueueItem` grows a payload-carrying kind:

```python
class _QueueItem:
    kind: str
    text: str = ""
    payload: Any = None

    ...existing classmethods unchanged...

    @classmethod
    def native_tool_calls(cls, calls: tuple) -> "_QueueItem":
        return cls("tool_calls", payload=calls)
```

(If `_QueueItem` is a plain `@dataclass`, add the `payload` field with default None — existing constructions are positional `cls("content", text)` and stay valid.)

(c) `stream_chat` gains `tools: list | None = None` (docstring updated: "yields str chunks; when ``tools`` is passed and the provider returned native tool-calls, the final item is a ``ProviderToolCalls``"); the llama_cpp branch ignores it; the generic branch forwards: `self._stream_generic_chat(resolution, messages, tools=tools)`.

(d) `_stream_generic_chat` gains `tools: list | None = None`; its `worker()` becomes:

```python
        def worker() -> None:
            try:
                kwargs = self._chat_api_kwargs(resolution, messages, tools=tools)
                response = self._chat_api_call(**kwargs)
                accumulator = _ToolCallAccumulator() if tools else None
                if accumulator is not None:
                    response = _tee_tool_calls(response, accumulator)
                for text in self.normalize_provider_response(response):
                    if stop_event.is_set():
                        break
                    if accumulator is not None and text in (
                            NO_PROVIDER_CONTENT_COPY,
                            UNSUPPORTED_PROVIDER_RESPONSE_COPY):
                        # tools= runs: fallback UI copy must never leak into
                        # agent history — a tool-call-only turn legitimately
                        # has no visible content.
                        continue
                    enqueue(_QueueItem.content(text))
                if accumulator is not None:
                    calls = accumulator.calls()
                    if calls:
                        enqueue(_QueueItem.native_tool_calls(calls))
            except BaseException as exc:
                enqueue(_QueueItem.error(self._safe_error_copy(resolution.provider, exc)))
            finally:
                enqueue(_QueueItem.done())
```

The consumer loop adds, before the `if item.text:` line:

```python
                if item.kind == "tool_calls":
                    yield ProviderToolCalls(tuple(item.payload))
                    continue
```

(e) `_chat_api_kwargs` gains the `tools` parameter and one dict entry `"tools": tools,` (the existing `if value is not None` filter drops it for plain sends).

(f) `_content_from_provider_mapping` — two tool-call-aware guards so per-chunk UNSUPPORTED copy can't fire on tool-call-only payloads (the mapping-item case; SSE strings already fall to `_EMPTY_RESPONSE`): in the `delta` branch, after the content check, add

```python
            if isinstance(delta, Mapping) and delta.get("tool_calls"):
                return ""
```

and in the `message` branch, after the content check, add

```python
            if isinstance(message, Mapping) and message.get("tool_calls"):
                return ""
```

(An empty string is existing-vocabulary "no visible content"; `normalize_provider_response`'s empty-string fallback copy is already filtered for tools= runs by (d).)

- [ ] **Step 4: Run the gateway suite**

Run: `python -m pytest Tests/Chat/test_console_provider_gateway.py -v`
Expected: PASS — all new tests plus every existing streaming/normalization test unchanged.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
git commit -m "feat(console): gateway tools= passthrough + native tool-call accumulation sentinel (task-243)"
```

---

### Task 5: Bridge adapter native mode, execution-key endpoint, and the `[console] native_tool_calls` kill-switch

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`_StreamingModelAdapter.chat_call` ~lines 297-334; `ConsoleAgentBridge.__init__` ~520s; `run_reply` config + `api_endpoint` ~560/616)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (bridge construction ~line 2180; new `_console_native_tool_calls_enabled` beside `_console_agent_runtime_enabled` at 2188)
- Test: `Tests/Chat/test_console_agent_bridge.py`

**Interfaces:**
- Consumes: Task 4's `ProviderToolCalls`; Task 3's `AgentConfig.native_tools` + service native branch; Task 1's capability set (via the service — the bridge itself never imports it).
- Produces: `ConsoleAgentBridge(..., native_tools_enabled: Callable[[], bool] | None = None)`; `run_reply` builds `AgentConfig(..., native_tools=<flag>)` and passes `api_endpoint=execution_key-first`; the adapter forwards `tools=` to `stream_chat` and returns `{"choices": [{"message": {"content": ..., "tool_calls": [...]}}]}` when native calls arrived.

Why `execution_key`-first: the service's capability check keys off `api_endpoint`, and `ConsoleProviderResolution.execution_key` is by definition "Provider key passed to `chat_api_call`" (gateway docstring ~line 180) — exactly the `PROVIDER_PARAM_MAP` key space. `resolution.provider` is the display key. Today `run_reply` passes `provider`; the adapter ignores `api_endpoint` entirely, so the swap only affects the capability check (and reads more truthfully). Fake resolutions built as `object()` yield `""` for both and keep the `"agent"` fallback — every existing bridge test stays on the fence path untouched.

- [ ] **Step 1: Write the failing tests**

In `Tests/Chat/test_console_agent_bridge.py`, extend `_ChunkGateway` in place (all existing scripts are plain `list[str]` and behave identically):

```python
class _ChunkGateway:
    """A gateway whose stream_chat replays a script keyed by call index."""

    def __init__(self, scripts):
        self._scripts = list(scripts)   # each entry: list of str chunks and/or ProviderToolCalls
        self.calls = 0
        self.tools_seen = []

    async def stream_chat(self, resolution, messages, tools=None):
        self.tools_seen.append(tools)
        chunks = self._scripts[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk
```

Add a native-capable fake resolution + tests:

```python
class _NativeResolution:
    provider = "Groq"
    execution_key = "groq"


def _native_calls(name, args, call_id="c1"):
    return ProviderToolCalls(tool_calls=(
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}},))


def test_native_tool_call_round_trip_streams_final_answer(tmp_path):
    bridge, db, store, session, aid = _bridge(tmp_path, [
        [_native_calls("get_current_datetime", {})],
        ["It is ", "now."]])
    outcome = _run(bridge, store, session, aid,
                   resolution=_NativeResolution())
    assert outcome.status == "done"
    assert store.get_message(session.id, aid).content == "It is now."
    gateway = bridge._gateway
    assert gateway.tools_seen[0] is not None          # tools= sent on turn 1
    names = [t["function"]["name"] for t in gateway.tools_seen[0]]
    assert "get_current_datetime" in names
    kinds = [s.kind for s in db.get_steps(...)]       # use the file's existing step-read helper
    assert "tool_call" in kinds and "tool_result" in kinds
    # TOOL marker appended to the transcript (reuse the file's existing
    # marker assertions from the fence round-trip test).


def test_native_leaked_prose_is_reset_before_final_answer(tmp_path):
    """Prose streamed before the ProviderToolCalls arrives must not survive
    (Finding-A parity with the fence path)."""
    bridge, db, store, session, aid = _bridge(tmp_path, [
        ["Let me check. ", _native_calls("get_current_datetime", {})],
        ["Done."]])
    outcome = _run(bridge, store, session, aid, resolution=_NativeResolution())
    assert outcome.status == "done"
    assert store.get_message(session.id, aid).content == "Done."


def test_native_kill_switch_off_stays_on_fence_path(tmp_path):
    bridge, db, store, session, aid = _bridge(tmp_path, [
        [_fence("get_current_datetime", {})], ["Done."]])
    bridge._native_tools_enabled = lambda: False       # or ctor kwarg via _bridge(...)
    outcome = _run(bridge, store, session, aid, resolution=_NativeResolution())
    assert outcome.status == "done"
    assert bridge._gateway.tools_seen[0] is None       # no tools= despite groq
```

Prefer threading `native_tools_enabled` through the `_bridge` helper as a ctor kwarg over attribute poking; match the file's `store.get_message`/step-read helper names to what actually exists (read the file's existing fence round-trip test and mirror its assertions exactly).

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Tests/Chat/test_console_agent_bridge.py -v -k "native"`
Expected: FAIL (fake gateway lacks `tools`; adapter never returns `tool_calls`; no `ProviderToolCalls` import).

- [ ] **Step 3: Implement**

`console_agent_bridge.py`:

(a) Import `ProviderToolCalls` from `.console_provider_gateway`.

(b) Adapter — signature gains `tools=None`; `_consume` skips sentinels out of the gate; the return carries them:

```python
    def chat_call(self, *, messages_payload, model=None, api_endpoint=None,
                  streaming=False, tools=None, **_ignored) -> dict:
        is_subagent = self._is_subagent(messages_payload)
        gate = StreamGate()
        any_streamed = False
        native_calls: list[dict] = []

        async def _consume() -> None:
            nonlocal any_streamed
            async for chunk in self._gateway.stream_chat(
                    self._resolution, messages_payload, tools=tools):
                if isinstance(chunk, ProviderToolCalls):
                    # Plan-B contract: structured deltas never hit the
                    # transcript — captured here, surfaced only through the
                    # returned message dict.
                    native_calls.extend(chunk.tool_calls)
                    continue
                visible = gate.feed(chunk)
                ...(existing body unchanged)...
```

After `run_until_complete`, extend the Finding-A reset so a native tool-call turn also discards leaked prose, and return the calls:

```python
        self._loop.run_until_complete(_consume())
        if any_streamed and not is_subagent:
            _visible, tool_call = gate.result()
            if tool_call is not None or native_calls:
                self._store.reset_stream_content(self._assistant_message_id)
        message: dict = {"content": gate.full_text}
        if native_calls:
            message["tool_calls"] = native_calls
        return {"choices": [{"message": message}]}
```

(Keep the existing Finding-A comment; extend it with one line noting the native case.)

(c) `ConsoleAgentBridge.__init__` gains `native_tools_enabled: Callable[[], bool] | None = None` stored as `self._native_tools_enabled`; `run_reply`'s config becomes:

```python
        native_tools = (True if self._native_tools_enabled is None
                        else bool(self._native_tools_enabled()))
        config = AgentConfig(
            model=model,
            system_prompt=compose_agent_system_prompt(session_system_prompt),
            allowed_tools=allowed_tools,
            budget=CONSOLE_RUN_BUDGET,
            native_tools=native_tools)
```

and the `run_turn` call's endpoint becomes:

```python
                api_endpoint=str(
                    getattr(resolution, "execution_key", "")
                    or getattr(resolution, "provider", "") or "agent"),
```

(d) `chat_screen.py` — beside `_console_agent_runtime_enabled` (line 2188) add:

```python
    def _console_native_tool_calls_enabled(self) -> bool:
        """Return whether ``[console] native_tool_calls`` allows native provider tool-calls (default on)."""
        value = self._console_config().get("native_tool_calls", True)
        return bool(value) if isinstance(value, (bool, int)) else True
```

and the `ConsoleAgentBridge(...)` construction (line 2180) gains
`native_tools_enabled=self._console_native_tool_calls_enabled,`.

- [ ] **Step 4: Run the Console suites**

Run: `python -m pytest Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_native_chat_flow.py Tests/Chat/test_console_agent_swap.py Tests/UI/test_console_agent_rail.py -v` (drop any of those paths that don't exist; add the file that covers `chat_screen` console wiring if one names it).
Expected: PASS — AC #4's fence/Console integration tests unchanged (fakes with `resolution=object()` resolve to the `"agent"` endpoint → fence path).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_agent_bridge.py
git commit -m "feat(console): native tool-calls through the streaming bridge + [console] native_tool_calls kill-switch (task-243)"
```

---

### Task 6: Live gate (coordinator-run — NOT a subagent task) + close-out

**Files:**
- Create: `Docs/superpowers/qa/native-tool-calls-2026-07/README.md` (+ captures)
- Modify: `backlog/tasks/task-243 - Wire-native-provider-tool-calls-with-fence-protocol-fallback.md`
- Create: one follow-up backlog task (next free id after remote-max check; 246 as of planning) — "Native tool-calls for normalizing providers (Anthropic/Google/Cohere)": their handlers must build `message.tool_calls` from provider-native blocks (Anthropic `tool_use` content blocks + streaming `input_json_delta` reassembly; `role="tool"` → `tool_result` payload conversion) before they can join `NATIVE_TOOLS_PROVIDERS`.

Steps (established QA recipe — scratchpad `serve_qa.py` + `cap.py`, 2050×1240, `PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring` in both seed and serve env):

1. Discover a configured cloud provider key in `~/.config/tldw_cli/config.toml` (Groq precedent; never echo/log the key). Copy it into an isolated QA profile wired to that provider.
2. **AC #2 evidence**: send a Console message that forces a tool round-trip (e.g. "What is 234*77? Use the calculator tool."). Capture: TOOL markers + final answer; verify via `AgentRunsDB` steps (`tool_call`/`tool_result`) AND that the run went native (no fence protocol in the payload — assert by instrumented log line or by the absence of the fence instructions in the recorded first request; a `loguru` debug line in `_make_call_model` gated on native is acceptable if added in Task 3).
3. Multi-call attempt (AC #3 live flavor, best-effort): a prompt inviting two independent lookups in one turn ("Use tools to get the current date AND compute 91*7 — you may call both at once").
4. **Fence regression**: same flow against local llama.cpp @127.0.0.1:9099 — fence markers, tools_seen None, run completes (AC #1/#4 live).
5. Write the QA README (scope header, capture-per-bullet, honest notes on wait times); commit evidence.
6. Backlog close-out: mark task-243 ACs `[x]` with Implementation Notes; file the follow-up task (re-verify remote max id at filing time per [[backlog-ids-assign-against-origin-dev]]); update the SDD ledger.

---

## Self-Review

- **Spec coverage**: AC #1 → Tasks 1+3 (capability check + fallback; llama_cpp pinned in tests) and Task 5 kill-switch; AC #2 → Task 6 live gate (real cloud provider end-to-end through the Console); AC #3 → Task 2 engine test + Task 3 service test (multi-call batch, dispatch iteration untouched); AC #4 → Tasks 2/3/4/5 each pin fence-path byte-identity and run the pre-existing suites.
- **Known limitation (documented, accepted)**: capability is provider-level, not per-model. A tool-incapable model on a capable provider surfaces the provider's own error as an honest RUN_ERROR; `[console] native_tool_calls = false` is the escape hatch. Noted for the task's Implementation Notes.
- **Cross-task type consistency**: `ProviderToolCalls.tool_calls: tuple[dict, ...]` (Task 4) → adapter `native_calls: list[dict]` → `message["tool_calls"]` (Task 5) → `parse_native_tool_calls(message)` (Task 1, consumed in Task 3) → `ToolCall.call_id` → engine `role="tool"` pairing (Task 2). `AgentConfig.native_tools` named identically in Tasks 3 and 5.
- **Placeholder scan**: Task 2 Step 1's second test and Task 4/5 fixture plumbing intentionally direct the implementer to mirror named existing tests in the same file (exact helper names vary per file and are verified there); all product code is fully specified.
