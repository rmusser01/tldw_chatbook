# Google/Gemini Native Tool-Calls — Implementation Plan (task-266)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `chat_with_google` speaks OpenAI shapes on both sides — OpenAI `tools=` wrapped as `functionDeclarations`, OpenAI tool history converted to Gemini `functionCall`/`functionResponse` parts, and streamed `functionCall` parts emitted as OpenAI `delta.tool_calls` fragments — so flipping `"google"` into `NATIVE_TOOLS_PROVIDERS` (gated on a live round-trip, AC #2) needs zero changes elsewhere.

**Architecture:** Same containment pattern as task-263 (Anthropic): everything lives in `chat_with_google` (`tldw_chatbook/LLM_Calls/LLM_API_Calls.py:1710`) plus small pure helpers beside it. Scout ground truth: the non-streaming response path ALREADY parses `functionCall` parts into OpenAI `tool_calls` with synthesized ids (lines ~1868-1884) — it needs tests, not code. The gaps: (a) request side drops assistant tool-call turns wholesale (empty `gemini_parts` → turn skipped) and drops `role="tool"` messages; (b) `tools` is raw passthrough (no `functionDeclarations` wrapping); (c) streaming ignores `functionCall` parts entirely. Gemini keys results by function NAME (no ids) with positional pairing for parallel same-name calls — the converter builds an id→name map from preceding assistant `tool_calls` echoes (always present for engine traffic) with a positional fallback against the preceding model turn.

**Tech Stack:** Python 3.11, `requests`, pytest with `requests.Session.post` mocking exactly per `Tests/Chat/test_anthropic_native_tools.py` (the task-263 test file is the template — same repo, same shapes).

## Global Constraints

- Backlog ACs (task-266): (1) `chat_with_google` accepts OpenAI-format tools and converts request/response/streaming shapes end-to-end; (2) google joins `NATIVE_TOOLS_PROVIDERS` only after a live round-trip against the real API (coordinator task; a Google key will be provided the same way as task-263's — repo-root file, git-excluded, NEVER echoed/logged/committed); (3) fence fallback intact until the flip — `"google"` is NOT added to the set in Tasks 1-2.
- Non-tool behavior byte-identical: plain-text chats produce identical request payloads and normalized responses; the existing non-streaming `functionCall` parsing and `finishReason` mapping stay as-is (including the likely-dead `"FUNCTION_CALL"` entry — do not remove; out of scope).
- Preserve raw-passthrough compat for callers already sending Gemini-shaped tools: if the `tools` list already looks Gemini-shaped (any entry carrying `functionDeclarations` or `function_declarations` or `googleSearch`-style keys), pass it through untouched; only convert when entries are OpenAI `{"type": "function", "function": {...}}` shaped.
- Junk guards mirror task-263's reviewed pattern: blank-name tool defs dropped locally (never a provider 400); an assistant message whose `tool_calls` are all junk falls through to plain content handling (no empty-parts turn).
- `tool_choice`/`toolConfig` stays out of scope (commented out in the map; the runtime never sends it).
- Observed pre-existing, out of scope (note in PR, don't fix): the google map's `'temperature': 'temp'` entry keys a nonexistent generic name (generic is `temp`), so temperature is silently dropped for google today; `system_instruction` snake_case casing is unverified against the live API (protobuf JSON accepts both casings — the live gate will confirm).
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime`, branch `claude/google-native-266` (off origin/dev df5fba20). Tests FOREGROUND via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`; `timeout` shell command unavailable. NEVER read/echo/reference any `*api-key*.txt` file.

## File Structure

- Modify `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` — helpers + two regions of `chat_with_google` (request build; streaming loop).
- Create `Tests/Chat/test_google_native_tools.py` — all coverage (request conversion, non-streaming pins for the EXISTING parsing, streaming emission, cross-layer accumulator pin).
- Task 3 (coordinator): `tldw_chatbook/Agents/native_tools.py` flip + docstring, service-level google test, QA evidence dir.

---

### Task 1: Request-side conversion (tools wrapping + tool-history parts)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (message loop ~1741-1761; `payload["tools"]` line ~1776; new helpers above `chat_with_google`)
- Test: `Tests/Chat/test_google_native_tools.py` (create)

**Interfaces:**
- Produces: module-level `_google_tools_payload(tools: list) -> list` and `_google_function_response(name: str, content) -> dict`. No signature changes to `chat_with_google`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_google_native_tools.py` mirroring `Tests/Chat/test_anthropic_native_tools.py`'s structure (module docstring, `_call_google`/`_call_google_get_result` helpers patching `requests.Session.post`, minimal Gemini response fixture). Gemini text fixture:

```python
def _gemini_text_response(text="ok"):
    return {"candidates": [{"content": {"parts": [{"text": text}],
                                        "role": "model"},
                            "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}}


OPENAI_TOOLS = [{"type": "function", "function": {
    "name": "calculator", "description": "Evaluate math.",
    "parameters": {"type": "object",
                   "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]}}}]
```

Tests (write each fully):

```python
def test_openai_tools_wrap_as_function_declarations(...):
    sent = ...  # chat_api_call("google", ..., tools=OPENAI_TOOLS)
    assert sent["tools"] == [{"functionDeclarations": [{
        "name": "calculator", "description": "Evaluate math.",
        "parameters": OPENAI_TOOLS[0]["function"]["parameters"]}]}]


def test_gemini_shaped_tools_pass_through_untouched(...):
    native = [{"functionDeclarations": [{"name": "t", "parameters": {}}]}]
    # tools=native -> sent["tools"] == native


def test_blank_name_openai_tool_dropped_locally(...):
    # [{"type":"function","function":{"name":"  "}}] + OPENAI_TOOLS[0]
    # -> only calculator forwarded (task-263 review precedent)


def test_openai_tool_history_converts_to_gemini_parts(...):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "",
         "tool_calls": [
             {"id": "call_A", "type": "function",
              "function": {"name": "calculator",
                           "arguments": "{\"expression\": \"2+2\"}"}},
             {"id": "call_B", "type": "function",
              "function": {"name": "calculator",
                           "arguments": "{\"expression\": \"3+3\"}"}}]},
        {"role": "tool", "tool_call_id": "call_A", "content": "4"},
        {"role": "tool", "tool_call_id": "call_B", "content": "6"},
    ]
    contents = sent["contents"]
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"] == [
        {"functionCall": {"name": "calculator", "args": {"expression": "2+2"}}},
        {"functionCall": {"name": "calculator", "args": {"expression": "3+3"}}}]
    # BOTH results coalesce into ONE user turn, positionally ordered
    # (Gemini pairs same-name parallel calls by order):
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"] == [
        {"functionResponse": {"name": "calculator", "response": {"result": "4"}}},
        {"functionResponse": {"name": "calculator", "response": {"result": "6"}}}]
    assert len(contents) == 3


def test_tool_result_with_json_object_content_passes_object(...):
    # role=tool content '{"answer": 42}' -> response {"answer": 42}
    # (dict-parseable content used directly; non-dict JSON like "[1,2]"
    # or plain text wraps as {"result": <string>})


def test_assistant_text_plus_tool_calls_keeps_text_part_first(...):
    # content "Let me check." -> parts[0] == {"text": "Let me check."}


def test_all_junk_tool_calls_fall_back_to_plain_content(...):
    # tool_calls ["junk", {"function": "junk"}, {"function": {"name": ""}}]
    # with content "hello" -> plain text turn (task-263 precedent)


def test_unknown_tool_call_id_uses_positional_fallback(...):
    # role=tool with tool_call_id "mystery" directly after a model turn
    # whose first functionCall is calculator -> functionResponse named
    # "calculator" (positional fallback when the id->name map misses)


def test_plain_chat_payload_unchanged(...):
    # no tools/tool messages: payload has no "tools" key; contents identical
    # to today's conversion (pin exact roles/parts for a 2-message chat)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_google_native_tools.py -v`
Expected: conversion tests FAIL (tools raw; tool turns dropped); `test_plain_chat_payload_unchanged` PASSES.

- [ ] **Step 3: Implement**

Helpers above `chat_with_google`:

```python
def _google_tools_payload(tools: list) -> list:
    """Wrap OpenAI function-format tool entries as Gemini functionDeclarations.

    Entries already Gemini-shaped (carrying ``functionDeclarations`` /
    ``function_declarations`` or other non-OpenAI keys) pass through
    untouched. OpenAI entries with a blank name are dropped locally —
    Gemini rejects empty tool names (task-263 review precedent).

    Args:
        tools: The ``tools`` list as received (OpenAI or Gemini shaped).

    Returns:
        A Gemini ``tools`` list; OpenAI entries collapse into ONE
        ``{"functionDeclarations": [...]}`` entry, passthrough entries keep
        their positions.
    """
    declarations = []
    passthrough = []
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if entry.get("type") == "function" and isinstance(function, dict):
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            parameters = function.get("parameters")
            if not isinstance(parameters, dict) or not parameters:
                parameters = {"type": "object", "properties": {}}
            declarations.append({
                "name": name,
                "description": str(function.get("description") or ""),
                "parameters": parameters,
            })
        else:
            passthrough.append(entry)
    result = list(passthrough)
    if declarations:
        result.append({"functionDeclarations": declarations})
    return result


def _google_function_response(name: str, content) -> dict:
    """Build a Gemini functionResponse part from an OpenAI tool result.

    Gemini requires ``response`` to be a JSON OBJECT: dict-parseable string
    content is used directly; anything else wraps as ``{"result": <str>}``.

    Args:
        name: The function name this result answers (Gemini pairs by name
            plus position — it has no call ids).
        content: The tool result content (string, typically).

    Returns:
        ``{"functionResponse": {"name": ..., "response": {...}}}``.
    """
    response = None
    if isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            response = parsed
    if response is None:
        response = {"result": str(content or "")}
    return {"functionResponse": {"name": name, "response": response}}
```

In the message loop (~1741-1761), BEFORE the existing role mapping, add the conversions. Maintain two loop-scoped state vars initialized before the loop: `tool_call_names = {}` (id→name accumulated from every converted assistant tool-call turn) and remember the last model turn's ordered functionCall names for the positional fallback:

```python
        if role == "tool":
            name = tool_call_names.get(str(msg.get("tool_call_id") or ""))
            if name is None:
                # Positional fallback: pair the nth consecutive result with
                # the nth functionCall of the preceding model turn (Gemini
                # pairs by name + order; it has no call ids).
                name = (last_function_call_names[consecutive_tool_results]
                        if consecutive_tool_results < len(last_function_call_names)
                        else "")
            part = _google_function_response(name, content)
            consecutive_tool_results += 1
            last = gemini_contents[-1] if gemini_contents else None
            if (last is not None and last.get("role") == "user"
                    and isinstance(last.get("parts"), list)
                    and any("functionResponse" in p for p in last["parts"]
                            if isinstance(p, dict))):
                last["parts"].append(part)
            else:
                gemini_contents.append({"role": "user", "parts": [part]})
            continue
        consecutive_tool_results = 0
        if role == "assistant" and msg.get("tool_calls"):
            parts = []
            if isinstance(content, str) and content.strip():
                parts.append({"text": content})
            call_names = []
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name") or "").strip()
                if not name:
                    continue
                raw_args = function.get("arguments")
                args = raw_args if isinstance(raw_args, dict) else {}
                if isinstance(raw_args, str) and raw_args.strip():
                    try:
                        parsed = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        args = parsed
                tool_call_names[str(call.get("id") or "")] = name
                call_names.append(name)
                parts.append({"functionCall": {"name": name, "args": args}})
            if call_names:
                last_function_call_names = call_names
                gemini_contents.append({"role": "model", "parts": parts})
                continue
            # All-junk tool_calls: fall through to plain content handling.
```

(Adapt state-variable names/placement to the real loop's structure; initialize `tool_call_names = {}`, `last_function_call_names = []`, `consecutive_tool_results = 0` before the loop. Everything else in the loop stays byte-identical.)

Line ~1776 becomes: `if tools: payload["tools"] = _google_tools_payload(tools)`.

- [ ] **Step 4: Run**

Run: `pytest Tests/Chat/test_google_native_tools.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_google_native_tools.py
git commit -m "feat(llm): google request-side native tools — functionDeclarations wrapping + functionCall/functionResponse history (task-266)"
```

---

### Task 2: Response pins (non-streaming, existing) + streaming emission

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (streaming chunk loop ~1807-1859 only)
- Test: `Tests/Chat/test_google_native_tools.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
def _gemini_function_call_response():
    return {"candidates": [{"content": {"parts": [
                {"text": "Checking."},
                {"functionCall": {"name": "calculator",
                                  "args": {"expression": "2+2"}}}],
                "role": "model"},
            "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 9}}


def test_non_streaming_function_call_normalizes_to_tool_calls(...):
    """PIN of EXISTING behavior (scout: lines ~1868-1884): functionCall
    parts already normalize to OpenAI tool_calls with synthesized ids."""
    message = result["choices"][0]["message"]
    (entry,) = message["tool_calls"]
    assert entry["type"] == "function"
    assert entry["function"]["name"] == "calculator"
    assert json.loads(entry["function"]["arguments"]) == {"expression": "2+2"}
    assert entry["id"]           # synthesized, non-empty
    # round-trips through the runtime parser:
    from tldw_chatbook.Agents.native_tools import parse_native_tool_calls
    parsed = parse_native_tool_calls(message)
    assert parsed[0].name == "calculator"


def test_streaming_function_call_chunk_emits_whole_openai_fragment(...):
    # SSE lines: a text chunk, then a chunk whose parts carry a complete
    # functionCall, then a STOP finish. Assert exactly one delta.tool_calls
    # fragment: [{"index": 0, "id": <non-empty>, "type": "function",
    #   "function": {"name": "calculator",
    #                "arguments": json.dumps({"expression": "2+2"})}}]
    # (Gemini streams functionCall parts WHOLE — one complete fragment is
    # accumulator-compatible: first fragment carries everything.)


def test_streaming_two_function_calls_get_distinct_indexes(...):
    # one chunk carrying two functionCall parts -> fragments with index 0
    # and 1, distinct synthesized ids


def test_streaming_fragments_reassemble_via_gateway_accumulator(...):
    # cross-layer pin exactly like test_anthropic_native_tools.py's:
    # feed yielded SSE strings through the REAL _decode_stream_item +
    # _ToolCallAccumulator -> one merged call, name calculator,
    # arguments '{"expression": "2+2"}'
```

Streaming mock: `mock_response.iter_lines.return_value` yielding `data: {...}`-encoded Gemini stream chunks (same pattern as the anthropic file's streaming test).

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_google_native_tools.py -v -k "streaming or non_streaming"`
Expected: the non-streaming pin PASSES (existing behavior); all streaming tests FAIL (functionCall parts ignored).

- [ ] **Step 3: Implement**

In the streaming chunk loop (~1823-1826 region), alongside the existing text extraction, collect functionCall parts and emit them as whole OpenAI fragments. Track `next_tool_position = 0` initialized before the loop:

```python
                        chunk_tool_calls = []
                        for part in parts:
                            if 'text' in part:
                                chunk_text += part.get('text', '')
                            if isinstance(part, dict) and 'functionCall' in part:
                                fc = part.get('functionCall') or {}
                                name = str(fc.get('name') or '').strip()
                                if not name:
                                    continue
                                chunk_tool_calls.append({
                                    "index": next_tool_position,
                                    "id": f"call_gemini_{time.time_ns()}_{next_tool_position}",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(fc.get('args') or {}),
                                    },
                                })
                                next_tool_position += 1
```

and attach `chunk_tool_calls` to the emitted OpenAI chunk's `delta` (`delta["tool_calls"] = chunk_tool_calls` when non-empty), making sure a tool-calls-only chunk (no text) still YIELDS (verify the existing emission gate, exactly as task-263's T3 did for anthropic). Everything else (text streaming, finishReason, `[DONE]`, error handling) byte-identical.

- [ ] **Step 4: Run**

Run: `pytest Tests/Chat/test_google_native_tools.py Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_console_provider_gateway.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_google_native_tools.py
git commit -m "feat(llm): google streaming functionCall parts emit OpenAI delta.tool_calls fragments (task-266)"
```

---

### Task 3 (coordinator-run — NOT a subagent task): live gate → flip → close-out

1. Obtain the Google/Gemini API key from the user (repo-root `google-api-key.txt`, add to `.git/info/exclude` BEFORE anything else; never echo/log/commit).
2. Live gate BEFORE the flip (in-harness `NATIVE_TOOLS_PROVIDERS` override, task-263 pattern): adapt the gate harness for `execution_key="google"`, a cheap Gemini model (e.g. `gemini-2.0-flash` or current cheap tier — check `/v1beta/models` first), streaming on. Cases: single calculator round-trip (verifies functionDeclarations acceptance, streamed functionCall emission, functionResponse turn acceptance — this also settles the `system_instruction` casing question live); parallel two-tool attempt.
3. If the gate passes: flip `"google"` into `NATIVE_TOOLS_PROVIDERS` + docstring update (cohere remains the last fence-only normalizer), update the capability test, service-level google native test, QA evidence dir `Docs/superpowers/qa/google-native-2026-07/`.
4. Backlog close-out (ACs, notes, Done), ledger, final whole-branch review (opus), PR.

## Self-Review

- AC #1 → Tasks 1-2 (request/response/streaming; response already existed — pinned). AC #2 → Task 3 (flip strictly after live gate). AC #3 → flip last; nothing partial ships.
- Name-vs-id pairing: id→name map from echo turns (guaranteed hit for engine traffic since the engine always echoes `tool_calls` before results) + positional fallback pinned by a test; results coalesce into one user turn in call order = Gemini's positional convention for same-name parallel calls.
- Byte-compat pins: plain-payload test (Task 1), non-streaming pin (Task 2), gateway suite re-run (Task 2).
- Fragment shape: whole-call first fragments are accumulator-compatible (id/name/arguments all on fragment one; nothing to concatenate) — cross-layer pinned.
