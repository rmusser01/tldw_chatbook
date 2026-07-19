# Anthropic Native Tool-Calls — Implementation Plan (task-263, Anthropic-only scope)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `chat_with_anthropic` speaks OpenAI shapes on both sides — accepts OpenAI-format `tools=`, converts OpenAI tool history (`assistant.tool_calls`, `role="tool"`) to Anthropic blocks, and surfaces Anthropic tool_use responses (non-streaming AND streaming) as OpenAI `message.tool_calls` / `delta.tool_calls` — so flipping `"anthropic"` into `NATIVE_TOOLS_PROVIDERS` (gated on a live round-trip, AC #3) needs zero changes anywhere else.

**Architecture:** All conversion lives inside `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (`chat_with_anthropic`, def line ~681) plus small module-level pure helpers beside it. Request side: OpenAI→Anthropic (tools with passthrough detection; tool history with consecutive-tool_result coalescing into one user turn). Response side: tool_use blocks → OpenAI `tool_calls` (non-streaming), and `content_block_start`/`input_json_delta` events → OpenAI-style incremental `delta.tool_calls` fragments keyed by a stable position index — **exactly** the fragment shape the Console gateway's `_ToolCallAccumulator` (task-243) already merges, so the streaming pipeline needs no gateway change. The set flip + live gate + service-level native test are the LAST step (coordinator-run), per AC #3/#4: no partial states ship.

**Tech Stack:** Python 3.11, `requests` (the handler's existing HTTP stack), pytest with `requests.Session.post` mocking per `Tests/Chat/test_chat_mocked_apis.py` precedent.

## Global Constraints

- Backlog ACs (task-263): (1) `chat_with_anthropic` builds OpenAI-shape `message.tool_calls` from `tool_use` content blocks (non-streaming), and its streaming path reassembles `input_json_delta` fragments into the same shape; (2) OpenAI-shape `role="tool"` history messages are converted to Anthropic `tool_result` content blocks before dispatch; (3) each converted provider is added to `NATIVE_TOOLS_PROVIDERS` **only after an end-to-end native round-trip against the real API** (coordinator task — an Anthropic key is available for this gate; NEVER echo/log/commit it: it lives at the repo root `anthropic-api-key.txt`, git-excluded); (4) fence fallback stays intact until conversion lands — `"anthropic"` is NOT added to the set in Tasks 1-3, and google/cohere are untouched entirely (follow-up tasks).
- Non-tool behavior byte-identical: plain-text chats (no `tools=`, no tool messages) must produce byte-identical request payloads and normalized responses to today. Existing anthropic tests (`Tests/Chat/test_chat_mocked_apis.py::test_anthropic_chat_mocked`, `test_chat_unit_mocked_APIs.py`) pass unchanged.
- Preserve the handler's historical `tools` contract: entries already in Anthropic shape (carrying `input_schema`) pass through untouched; only OpenAI `{"type": "function", "function": {...}}` entries are converted.
- `tool_choice` stays out of scope (absent from the handler signature and commented out in `PROVIDER_PARAM_MAP` — Anthropic defaults to auto when `tools` present; the agent runtime never sends it).
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime`, branch `claude/anthropic-native-263` off latest origin/dev. Tests FOREGROUND via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`; `timeout` shell command unavailable.

## File Structure

- Modify `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` — all three tasks (helpers + three regions of `chat_with_anthropic`).
- Create `Tests/Chat/test_anthropic_native_tools.py` — one new test file for all conversion coverage (mock `requests.Session.post`; fixtures local to the file).
- Task 4 (coordinator): `tldw_chatbook/Agents/native_tools.py` (set flip + docstring), `Tests/Agents/test_agent_service.py` (anthropic native service test), QA evidence dir.

---

### Task 1: Request-side conversion (AC #2 + tools format)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (message loop ~728-755; `data["tools"]` line ~790; new helpers above `chat_with_anthropic`)
- Test: `Tests/Chat/test_anthropic_native_tools.py` (create)

**Interfaces:**
- Produces (consumed by Tasks 2-4): module-level `_anthropic_tools_payload(tools: list) -> list` and the in-loop conversions. No signature changes to `chat_with_anthropic`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_anthropic_native_tools.py`. Mirror `Tests/Chat/test_chat_mocked_apis.py::test_anthropic_chat_mocked`'s mocking pattern exactly (patch `requests.Session.post`, drive through `chat_api_call(api_endpoint="anthropic", ...)`, inspect `mock_post.call_args[1]["json"]`). A minimal non-streaming Anthropic response fixture keeps the handler happy:

```python
def _anthropic_text_response(text="ok"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}


OPENAI_TOOLS = [{"type": "function", "function": {
    "name": "calculator", "description": "Evaluate math.",
    "parameters": {"type": "object",
                   "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]}}}]


def test_openai_tools_convert_to_anthropic_input_schema(...):
    # chat_api_call("anthropic", messages_payload=[user msg], tools=OPENAI_TOOLS, streaming=False)
    sent = mock_post.call_args[1]["json"]
    assert sent["tools"] == [{"name": "calculator", "description": "Evaluate math.",
                              "input_schema": OPENAI_TOOLS[0]["function"]["parameters"]}]


def test_anthropic_shaped_tools_pass_through_untouched(...):
    native = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    # tools=native -> sent["tools"] == native (historical contract preserved)


def test_openai_tool_history_converts_to_anthropic_blocks(...):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "toolu_A", "type": "function",
                         "function": {"name": "calculator",
                                      "arguments": "{\"expression\": \"2+2\"}"}},
                        {"id": "toolu_B", "type": "function",
                         "function": {"name": "calculator",
                                      "arguments": "{\"expression\": \"3+3\"}"}}],
        },
        {"role": "tool", "tool_call_id": "toolu_A", "content": "4"},
        {"role": "tool", "tool_call_id": "toolu_B", "content": "6"},
    ]
    sent = mock_post.call_args[1]["json"]["messages"]
    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"] == [
        {"type": "tool_use", "id": "toolu_A", "name": "calculator",
         "input": {"expression": "2+2"}},
        {"type": "tool_use", "id": "toolu_B", "name": "calculator",
         "input": {"expression": "3+3"}}]
    # BOTH tool results coalesce into ONE user turn (Anthropic alternation):
    assert sent[2]["role"] == "user"
    assert sent[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_A", "content": "4"},
        {"type": "tool_result", "tool_use_id": "toolu_B", "content": "6"}]
    assert len(sent) == 3


def test_assistant_text_plus_tool_calls_keeps_text_block_first(...):
    # assistant content "Let me check." + one tool_call ->
    # [{"type":"text","text":"Let me check."}, {"type":"tool_use", ...}]


def test_malformed_tool_call_arguments_become_empty_input(...):
    # arguments "{broken" -> input {}


def test_plain_chat_payload_unchanged(...):
    # No tools/tool messages: sent payload has no "tools" key; messages are
    # the same plain role/content dicts as before this branch (pin exact).
```

Write every test fully (the `...` are the shared mock plumbing you extract into a small local helper).

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_anthropic_native_tools.py -v`
Expected: conversion tests FAIL (tools sent raw OpenAI-shape; `role="tool"` messages dropped with a warning; `tool_calls` ignored). `test_plain_chat_payload_unchanged` PASSES (pinned baseline).

- [ ] **Step 3: Implement**

Module-level helper above `chat_with_anthropic`:

```python
def _anthropic_tools_payload(tools: list) -> list:
    """Convert OpenAI function-format tool entries to Anthropic's format.

    Entries already in Anthropic shape (carrying ``input_schema``) pass
    through untouched — the handler's historical contract. Non-dict junk is
    dropped.

    Args:
        tools: The ``tools`` list as received (OpenAI or Anthropic shaped).

    Returns:
        Anthropic-format entries: ``{"name", "description", "input_schema"}``.
    """
    converted = []
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if entry.get("type") == "function" and isinstance(function, dict):
            converted.append({
                "name": str(function.get("name") or ""),
                "description": str(function.get("description") or ""),
                "input_schema": function.get("parameters")
                or {"type": "object", "properties": {}},
            })
        else:
            converted.append(entry)
    return converted
```

Line ~790 becomes:

```python
    if tools is not None: data["tools"] = _anthropic_tools_payload(tools)
```

In the message loop (~728-755), BEFORE the `if role not in ["user", "assistant"]` filter, add the two conversions (order: tool-role first, then assistant-with-tool_calls):

```python
        if role == "tool":
            # OpenAI tool-result convention -> Anthropic tool_result block.
            # Consecutive tool results coalesce into ONE user turn: they all
            # answer the same assistant tool_use turn, and Anthropic requires
            # alternating roles (task-263 AC#2).
            block = {"type": "tool_result",
                     "tool_use_id": str(msg.get("tool_call_id") or ""),
                     "content": str(content or "")}
            last = anthropic_messages[-1] if anthropic_messages else None
            if (last is not None and last.get("role") == "user"
                    and isinstance(last.get("content"), list)
                    and any(isinstance(b, dict) and b.get("type") == "tool_result"
                            for b in last["content"])):
                last["content"].append(block)
            else:
                anthropic_messages.append({"role": "user", "content": [block]})
            continue
        if role == "assistant" and msg.get("tool_calls"):
            # OpenAI assistant tool_calls echo -> Anthropic tool_use blocks
            # (text block first when the turn also carried visible content).
            blocks = []
            if isinstance(content, str) and content.strip():
                blocks.append({"type": "text", "text": content})
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                raw_args = function.get("arguments")
                tool_input = raw_args if isinstance(raw_args, dict) else {}
                if isinstance(raw_args, str) and raw_args.strip():
                    try:
                        parsed = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        tool_input = parsed
                blocks.append({"type": "tool_use",
                               "id": str(call.get("id") or ""),
                               "name": str(function.get("name") or ""),
                               "input": tool_input})
            anthropic_messages.append({"role": "assistant", "content": blocks})
            continue
```

(Keep the existing loop body for everything else byte-identical, including the unsupported-role warning for genuinely unknown roles.)

- [ ] **Step 4: Run**

Run: `pytest Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_chat_mocked_apis.py Tests/Chat/test_chat_unit_mocked_APIs.py -v`
Expected: PASS (new + both pre-existing anthropic tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_anthropic_native_tools.py
git commit -m "feat(llm): anthropic request-side native tools — OpenAI tools/tool-history converted to Anthropic blocks (task-263)"
```

---

### Task 2: Non-streaming response — `tool_use` blocks → `message.tool_calls` (AC #1a)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (normalization block ~887-925)
- Test: `Tests/Chat/test_anthropic_native_tools.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
def _anthropic_tool_use_response():
    return {"id": "msg_2", "type": "message", "role": "assistant", "model": "claude-x",
            "content": [
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_X", "name": "calculator",
                 "input": {"expression": "2+2"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 9}}


def test_tool_use_response_normalizes_to_openai_tool_calls(...):
    result = chat_api_call(api_endpoint="anthropic", ..., streaming=False)
    message = result["choices"][0]["message"]
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] == "Checking."
    assert message["tool_calls"] == [{
        "id": "toolu_X", "type": "function",
        "function": {"name": "calculator",
                     "arguments": json.dumps({"expression": "2+2"})}}]


def test_text_only_response_has_no_tool_calls_key(...):
    # _anthropic_text_response() -> "tool_calls" not in message (byte-compat)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_anthropic_native_tools.py -v -k "normalizes_to_openai or no_tool_calls_key"`
Expected: the tool_use test FAILS (`tool_calls` missing); the text-only test PASSES.

- [ ] **Step 3: Implement**

In the normalization block, after the text-parts extraction, build the entries and attach only when present:

```python
            tool_call_entries = []
            for part in response_data.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "tool_use":
                    tool_call_entries.append({
                        "id": str(part.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str(part.get("name") or ""),
                            "arguments": json.dumps(part.get("input") or {}),
                        },
                    })
            ...
            message_payload = {"role": "assistant",
                               "content": full_assistant_content}
            if tool_call_entries:
                message_payload["tool_calls"] = tool_call_entries
```

and use `message_payload` in `normalized_response["choices"][0]["message"]` (replacing the inline dict; everything else in the block byte-identical).

- [ ] **Step 4: Run** — `pytest Tests/Chat/test_anthropic_native_tools.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_anthropic_native_tools.py
git commit -m "feat(llm): anthropic non-streaming tool_use blocks normalize to OpenAI message.tool_calls (task-263)"
```

---

### Task 3: Streaming — `content_block_start`/`input_json_delta` → OpenAI `delta.tool_calls` fragments (AC #1b)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (SSE event loop ~810-884; the dead `tool_calls_delta` placeholder at ~836/859-861 goes live)
- Test: `Tests/Chat/test_anthropic_native_tools.py` (append)

The emitted fragment shape is BINDING — it must match what the gateway's `_ToolCallAccumulator` (console_provider_gateway.py, task-243) merges: first fragment carries `{"index": <position>, "id", "type": "function", "function": {"name", "arguments": ""}}`; continuation fragments carry `{"index": <position>, "function": {"arguments": "<partial_json fragment>"}}`. `<position>` is a 0-based counter over tool_use blocks in THIS response (NOT Anthropic's content-block index, which counts text blocks too).

- [ ] **Step 1: Write the failing test**

```python
def _anthropic_sse_lines():
    events = [
        {"type": "message_start", "message": {"id": "msg_3"}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Checking."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "toolu_S",
                           "name": "calculator", "input": {}}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"expres'}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": 'sion": "2+2"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]
    return [f"data: {json.dumps(e)}".encode() for e in events]


def test_streaming_tool_use_emits_openai_delta_fragments(...):
    # mock response.iter_lines() -> _anthropic_sse_lines(); streaming=True;
    # consume the generator; json-parse every yielded "data: {...}" line.
    fragments = [c["choices"][0]["delta"]["tool_calls"]
                 for c in chunks if "tool_calls" in c["choices"][0].get("delta", {})]
    assert fragments[0] == [{"index": 0, "id": "toolu_S", "type": "function",
                             "function": {"name": "calculator", "arguments": ""}}]
    assert fragments[1] == [{"index": 0,
                             "function": {"arguments": '{"expres'}}]
    assert fragments[2] == [{"index": 0,
                             "function": {"arguments": 'sion": "2+2"}'}}]
    # text still streams, finish_reason still maps:
    texts = [c["choices"][0]["delta"].get("content") for c in chunks]
    assert "Checking." in texts
    finishes = [c["choices"][0].get("finish_reason") for c in chunks]
    assert "tool_calls" in finishes


def test_streaming_fragments_reassemble_via_gateway_accumulator(...):
    """Cross-layer pin: feed this handler's yielded SSE strings through the
    REAL gateway accumulator path (_decode_stream_item + _ToolCallAccumulator
    from tldw_chatbook.Chat.console_provider_gateway) and assert one merged
    call: id toolu_S, name calculator, arguments '{"expression": "2+2"}'."""
```

Write both fully (the second imports the private gateway pieces directly — acceptable for a cross-layer contract pin; mark with a comment).

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_anthropic_native_tools.py -v -k streaming`
Expected: FAIL — no `tool_calls` fragments emitted today.

- [ ] **Step 3: Implement**

In the streaming generator, add position tracking before the event loop:

```python
                # task-263: map Anthropic tool_use content-block indexes to
                # 0-based OpenAI tool_calls positions (Anthropic's index also
                # counts text blocks; OpenAI consumers key fragments by
                # tool-call position — see the gateway's _ToolCallAccumulator).
                tool_call_positions = {}
                next_tool_position = 0
```

New/extended event branches (`tool_calls_delta` is the existing per-event local, currently always None):

```python
                        if anthropic_event.get("type") == "content_block_start":
                            block = anthropic_event.get("content_block") or {}
                            if block.get("type") == "tool_use":
                                index = int(anthropic_event.get("index", 0))
                                position = next_tool_position
                                next_tool_position += 1
                                tool_call_positions[index] = position
                                tool_calls_delta = [{
                                    "index": position,
                                    "id": str(block.get("id") or ""),
                                    "type": "function",
                                    "function": {
                                        "name": str(block.get("name") or ""),
                                        "arguments": ""},
                                }]
                        elif anthropic_event.get("type") == "content_block_delta":
                            delta = anthropic_event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                delta_content = delta.get("text")
                            elif delta.get("type") == "input_json_delta":
                                index = int(anthropic_event.get("index", 0))
                                if index in tool_call_positions:
                                    tool_calls_delta = [{
                                        "index": tool_call_positions[index],
                                        "function": {"arguments":
                                                     delta.get("partial_json", "")},
                                    }]
```

The existing placeholder attach (859-861) now fires. CRITICAL: verify the chunk-emission gate — find the condition that decides whether an SSE chunk is yielded for an event (today it likely emits only when `delta_content` is set or `finish_reason` arrived) and extend it so a `tool_calls_delta`-only event ALSO yields a chunk. Keep `[DONE]`/error handling byte-identical.

- [ ] **Step 4: Run**

Run: `pytest Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_chat_mocked_apis.py Tests/Chat/test_console_provider_gateway.py -q`
Expected: PASS (incl. the cross-layer accumulator pin and the untouched gateway suite).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_anthropic_native_tools.py
git commit -m "feat(llm): anthropic streaming input_json_delta reassembly as OpenAI delta.tool_calls fragments (task-263)"
```

---

### Task 4 (coordinator-run — NOT a subagent task): live gate → set flip → close-out (AC #3, #4)

1. Live gate FIRST, against the real API with the user's key (repo-root `anthropic-api-key.txt`, git-excluded; read at runtime, never echoed/committed): extend the scratchpad `native_gate.py` with an anthropic case — `ConsoleProviderResolution(provider="anthropic", execution_key="anthropic", api_key=<from file>, model="claude-haiku-4-5-20251001"(cheap), streaming=True, max_tokens=1024)`. NOTE the flip hasn't landed yet, so run the gate with a temporary in-process override (`native_tools.NATIVE_TOOLS_PROVIDERS` monkeypatched to include "anthropic" inside the harness) — evidence that the conversion works end-to-end BEFORE committing the flip. Cases: single calculator round-trip (streaming exercises Task 3 live) + multi-call attempt (Anthropic supports parallel tool use) + verify role="tool"→tool_result second-turn acceptance (no 400).
2. If (and only if) the gate passes: add `"anthropic"` to `NATIVE_TOOLS_PROVIDERS`, update the `native_tools.py` module docstring (anthropic converted; google/cohere still fence-only), add a service-level native test for `api_endpoint="anthropic"` in `Tests/Agents/test_agent_service.py` (mirror the groq one), and commit with the QA evidence (`Docs/superpowers/qa/anthropic-native-2026-07/`).
3. Backlog close-out: ACs (AC on google/cohere: note they remain fence-only by design — the task's own AC #4), Implementation Notes, Done. File follow-up tasks for google/cohere conversions (check remote max ID first — the parallel-audit collisions are live). Final whole-branch review (opus), PR.

## Self-Review

- AC #1 → Tasks 2 (non-streaming) + 3 (streaming, incl. the cross-layer accumulator pin). AC #2 → Task 1 (coalesced tool_result turns; alternation preserved). AC #3 → Task 4 (flip strictly after live round-trip). AC #4 → the flip being last + google/cohere untouched; until Task 4 lands, anthropic keeps fence (set unchanged) even though the handler understands tools — no partial state.
- Byte-compat pins: plain-payload test (Task 1), no-tool_calls-key test (Task 2), gateway suite re-run (Task 3).
- Fragment-shape consistency: Task 3's binding shape matches `_ToolCallAccumulator._merge` (id/name on first fragment, string-concatenated `arguments`, `index`-keyed) and is pinned by the cross-layer test.
