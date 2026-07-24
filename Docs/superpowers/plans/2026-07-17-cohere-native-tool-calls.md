# Cohere Native Tool-Calls (task-267) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `chat_with_cohere` accepts OpenAI-format tools and role="tool" history, emits OpenAI-shape `tool_calls` on both response paths, and `cohere` joins `NATIVE_TOOLS_PROVIDERS` — flip gated on a live API round-trip.

**Architecture / Decision (user-approved 2026-07-17):** **Migrate the handler from Cohere v1 `/chat` to v2 `/chat` FIRST, then map tools.** v1's shapes are structurally hostile (flat `parameter_definitions` cannot express nested JSON Schema → MCP tools inexpressible; `tool_results` is a separate request field outside the history model; no call ids). v2 is OpenAI-shaped end-to-end: `messages` array incl. `role:"tool"`, JSON-Schema `tools`, `tool_calls` with ids, incremental streaming deltas. The migration changes payload/parsing for ALL Cohere chats — text-only behavior is pinned by byte-compat tests before tools land.

**Tech Stack:** Python 3.11, `requests` (the handler's existing HTTP client), pytest.

## Global Constraints

- Fence fallback stays intact until the flip: `NATIVE_TOOLS_PROVIDERS` gains `"cohere"` ONLY in the final flip commit, created ONLY after the live gate passes (backlog AC; same ordering as tasks 263/266).
- OpenAI-shape byte-compat: text-only responses must keep the exact normalized shape callers see today (`choices[0].message.content`, finish_reason mapping); `tool_calls` attached ONLY when present (sibling pin precedent).
- Streaming `delta.tool_calls` fragments MUST match `_ToolCallAccumulator`'s contract (`tldw_chatbook/Chat/console_provider_gateway.py:255-319`): first fragment per position carries `id`/`type`/`function.name`; `function.arguments` is a STRING that the accumulator concatenates across fragments.
- Provider-specific fields that must round-trip (Cohere v2's `tool_plan`) ride the OpenAI shapes as an allow-listed extra — mirror `google_thought_signature` through `_PRESERVED_FRAGMENT_EXTRAS` (`console_provider_gateway.py:249-252`).
- NEVER read/echo/log/commit `cohere-api-key.txt` contents (repo root, git-excluded). Runtime read only: `Path(...).read_text().strip()`.
- Tests FOREGROUND via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` from the worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime`; the `timeout` shell command does not exist.
- Branch `claude/cohere-native-267` off origin/dev. Commit per task with the messages given.
- The sibling conversions ARE the pattern library — read them before writing: `_google_tools_payload`/google request-history/streaming in `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (task-266, closest template), `_anthropic_tools_payload` (task-263), tests `Tests/Chat/test_google_native_tools.py` and `test_anthropic_native_tools.py`.

## Cohere v2 shapes (verify against https://docs.cohere.com/reference/chat during Task 1; scout knowledge, not gospel)

- Request: `POST {base}/v2/chat`, `{"model": ..., "messages": [{"role": "system"|"user"|"assistant"|"tool", ...}], "tools": [{"type":"function","function":{"name","description","parameters":<JSON Schema>}}], "stream": bool, ...params}`. Assistant tool-call turns echo as `{"role":"assistant","tool_plan":<str>,"tool_calls":[...]}`; tool results as `{"role":"tool","tool_call_id":<id>,"content":[{"type":"document","document":{"data":<str>}}]}` (a plain string `content` may also be accepted — implementer verifies; prefer the documented shape).
- Non-streaming response: `{"id", "message": {"role":"assistant", "content":[{"type":"text","text":...}], "tool_calls":[{"id","type":"function","function":{"name","arguments":<json str>}}]?, "tool_plan":<str>?}, "finish_reason": "COMPLETE"|"TOOL_CALL"|"MAX_TOKENS"|"STOP_SEQUENCE"|..., "usage": {...}}`.
- Streaming SSE `data:` events discriminated by `"type"`: `message-start`, `content-start`, `content-delta` (`delta.message.content.text`), `content-end`, `tool-plan-delta` (`delta.message.tool_plan`), `tool-call-start` (`delta.message.tool_calls` w/ id+name, index), `tool-call-delta` (incremental `function.arguments` string), `tool-call-end`, `message-end` (`delta.finish_reason`, usage).
- v1→v2 param renames: `p`→`p` stays? NO — v2 uses `temperature`, `p`, `k`, `max_tokens`, `stop_sequences`, `seed`, `frequency_penalty`, `presence_penalty` (implementer verifies the exact v2 param names; drop v1-only `preamble`/`chat_history`/`message`/`connectors`/`num_generations`).

---

### Task 1: v2 endpoint migration, text paths only (byte-compat pinned)

**Files:** Modify `tldw_chatbook/LLM_Calls/LLM_API_Calls.py:1158-1543` (`chat_with_cohere`). Test: create `Tests/Chat/test_cohere_native_tools.py` (text-path pins first; tools tests join in Tasks 2-4).

- [ ] Write failing tests first (mock `requests.post` / SSE lines, mirroring the sibling test files' fixtures): (a) non-streaming text-only request builds `{base}/v2/chat` with a `messages` array — leading system message (or `system_prompt` param) becomes `{"role":"system","content":...}`, user/assistant history passes through with lowercase roles, NO `preamble`/`chat_history`/`message` keys, NO `Cohere-Version` header requirement (harmless if kept — implementer decides, but assert the endpoint); response `message.content[0].text` → `choices[0].message.content`, finish_reason `COMPLETE`→`stop`, `MAX_TOKENS`→`length`, NO `tool_calls` key on text-only responses; (b) streaming `content-delta` events → OpenAI `delta.content` chunks, `message-end` → finish chunk; (c) params map to their v2 names; `num_generations` (v1-only) dropped with a debug log.
- [ ] Implement: rewrite request build (drop the v1 `message`/`chat_history`/`preamble` split — v2 takes the history as-is with a system message inline), point at `/v2/chat`, rewrite BOTH response paths for text. Keep the handler's signature and its OpenAI-shape return contract byte-identical for text chats. Align the inline default model with config's (`command-a-03-2025`, config.py:1645-1656 — the handler's stale `command-r` fallback at 1186 is a known discrepancy; fix it here and note it).
- [ ] Run the new tests + `Tests/Chat/test_chat_unit_mocked_APIs.py -q` (the PROVIDER_PARAM_MAP invariants must keep passing — `chat_with_cohere`'s signature keeps every mapped target keyword-passable).
- [ ] Commit: `feat(llm): migrate chat_with_cohere to the v2 /chat API — text paths, byte-compat pinned (task-267)`

### Task 2: request-side tools + tool history

**Files:** Modify `chat_with_cohere` request build. Test: extend `Tests/Chat/test_cohere_native_tools.py`.

- [ ] Failing tests: (a) OpenAI-format `tools` passthrough into the v2 payload (v2 IS OpenAI-shaped — passthrough with a light validity filter: entries missing `function.name` dropped with a warning, mirroring `_google_tools_payload`'s blank-name guard); (b) assistant history message with `tool_calls` → v2 assistant turn carrying `tool_calls` (arguments as JSON string; malformed/dict arguments normalized — dict → `json.dumps`, unparseable string → passthrough as-is since v2 takes strings) and `tool_plan` re-attached from the message's `cohere_tool_plan` extra when present (content used otherwise); (c) `role:"tool"` history message → `{"role":"tool","tool_call_id":...,"content":[{"type":"document","document":{"data":<content str>}}]}`; a tool message missing `tool_call_id` falls back to the most recent assistant tool_call id (positional pairing, mirroring google's fallback).
- [ ] Implement in the request build; guards never raise on malformed entries (skip + log).
- [ ] Run the file. Commit: `feat(llm): cohere v2 request-side native tools + tool-role history (task-267)`

### Task 3: non-streaming response tool_calls

- [ ] Failing tests: v2 `message.tool_calls` → OpenAI `choices[0].message.tool_calls` (id/type/function passthrough; arguments guaranteed a string), attached ONLY when non-empty; `message.tool_plan` preserved onto the assistant message as `cohere_tool_plan`; finish_reason `TOOL_CALL`→`tool_calls`; text+tool_calls both present → content AND tool_calls both populated.
- [ ] Implement. Run. Commit: `feat(llm): cohere v2 non-streaming tool_calls parsing (task-267)`

### Task 4: streaming tool events + gateway extra

**Files:** Modify `chat_with_cohere` streaming loop; modify `tldw_chatbook/Chat/console_provider_gateway.py:249-252` (`_PRESERVED_FRAGMENT_EXTRAS` gains `"cohere_tool_plan"`). Test: extend the test file with a cross-layer pin importing the REAL `_ToolCallAccumulator`.

- [ ] Failing tests: `tool-call-start` → first fragment `{index, id, type, function:{name, arguments:""}}`; `tool-call-delta` → `{index, function:{arguments:<substring>}}` (accumulator concatenates); `tool-plan-delta` text accumulated and emitted as `cohere_tool_plan` on the first tool-call fragment (so the accumulator's extras allow-list preserves it); `message-end` after tool calls → finish_reason `tool_calls`. Cross-layer: feed the emitted fragments through the real accumulator, assert reassembled OpenAI tool_calls + preserved extra.
- [ ] Implement (position = Cohere's own `index` field if present, else a 0-based tool-event counter — verify which the API provides). Run the file + `Tests/Chat/test_console_provider_gateway.py -q` (accumulator suite must stay green).
- [ ] Commit: `feat(llm): cohere v2 streaming tool-call fragments + tool_plan extra (task-267)`

### Task 5: service-level pin + registry test prep

**Files:** `Tests/Agents/test_agent_service.py` (add `test_native_endpoint_cohere_sends_tools_and_suppresses_fence` mirroring :675-709 siblings — NOTE: it asserts against the OVERRIDDEN registry via monkeypatch, since the flip hasn't happened), `Tests/Agents/test_native_tools.py` (leave the `assert not provider_supports_native_tools("cohere")` as-is — it flips in Task 6).

- [ ] Write + run both files. Commit: `test(agents): cohere native-path service pin behind registry override (task-267)`

### Task 6 (coordinator): live gate → flip → close-out

- Pre-check model tool support (`command-a-03-2025`; fall back per live behavior — google's `gemini-2.5-flash` 404 precedent).
- `Docs/superpowers/qa/cohere-native-2026-07/cohere_gate.py`: copy `google_gate.py`'s structure — in-process registry override BEFORE bridge imports, real bridge→gateway→handler→HTTPS stack, RecordingGateway asserting tools sent + no fence leak. Case A: single calculator round-trip streaming, asserts `tool_result` step + second provider turn + `status=="done"`. Case B: two tools in one reply, ≥2 `tool_call` steps.
- On PASS: flip `native_tools.py` (set + docstring rewrite — cohere was the last fence-only named provider), flip `Tests/Agents/test_native_tools.py:21` assertion, drop the Task-5 monkeypatch override (assert against the real registry), commit evidence (README + raw gate output + script) with the flip.
- Backlog: split the concatenated AC into 3 checkboxes, check all, Implementation Notes, Done. Final whole-branch review (opus). PR.

## Self-Review

- AC1 (request/response/streaming end-to-end) → Tasks 1-4. AC2 (flip only after live round-trip) → Task 6 ordering. AC3 (fence intact until flip) → registry untouched through Task 5; service test uses an explicit override.
- v1→v2 blast radius pinned by Task 1's byte-compat tests before any tool code lands.
- The `tool_plan` round-trip mirrors the shipped `google_thought_signature` mechanism — same allow-list, same three touchpoints (parse, emit, re-attach).
