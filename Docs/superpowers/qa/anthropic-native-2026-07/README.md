# Anthropic native tool-calls — live gate (task-263)

**Date:** 2026-07-17 · **Branch:** `claude/anthropic-native-263` · **Model:** claude-haiku-4-5 (real api.anthropic.com, streaming ON)
**Harness:** `anthropic_gate.py` (this directory) — the real Console reply engine end-to-end (`ConsoleAgentBridge.run_reply` → `_StreamingModelAdapter` → real `ConsoleProviderGateway` → real `chat_api_call` → `chat_with_anthropic` → real HTTPS). Only instrumentation: a recording passthrough around `stream_chat` logging whether `tools=` was sent and whether the fence protocol appeared in the system prompt. Run BEFORE the set flip via an in-process `NATIVE_TOOLS_PROVIDERS` override, per the plan's AC #3 ordering — the flip commit exists only because this gate passed. API key read at runtime from a git-excluded local file; never logged, echoed, or committed.

## GATE: PASS — both cases (`gate-A-B-2026-07-17.txt`)

### Case A — single tool round-trip: PASS (1.5s)
- `tools=` on both turns (converted to Anthropic `input_schema` format by the handler — accepted, no 400), `fence_in_system=False`.
- Calculator round-trip: `tool_use` streamed via `input_json_delta`, reassembled by the gateway accumulator into `ModelTurn.tool_calls`, answered as `role="tool"` → converted to a `tool_result` block — **the second turn was accepted by the real API**, proving the request-side history conversion (AC #2) live.
- Final streamed answer `"18018"` (correct), status `done`.

### Case B — parallel multi-tool in ONE turn: PASS (2.0s)
- The model called **both** tools (`get_current_datetime` + `calculator`) **in a single reply** — 2 tool_calls, 2 model turns total. This is the genuine parallel-batch shape (the OpenAI-compatible gate at task-243 only ever saw serialized calls), dispatched as one batch by `run_agent_loop` with both `role="tool"` results paired by id and accepted on the follow-up turn.
- It also live-exercises the one path the unit suite left to trace-only verification: **two interleaved `tool_use` blocks in one stream** (positions 0/1, `input_json_delta` fragments reassembled per block index) — the T3 review advisory, now covered by real wire evidence.
- Correct combined final answer (date 2026-07-17 + 91×7=637), status `done`.

## What this proves per AC
- **AC #1** (non-streaming + streaming normalization): streaming path proven live (both cases ran `streaming=True`); non-streaming pinned by unit tests.
- **AC #2** (role="tool" → `tool_result` blocks): proven live — the API accepted both follow-up turns, including the coalesced two-result turn in case B.
- **AC #3** (set flip only after a real round-trip): satisfied — the flip commit follows this gate.
- **AC #4** (no partial states): google/cohere untouched and still fence-only; anthropic was fence-only until the flip landed atomically with its passing gate.

## Suites at gate HEAD
`Tests/Agents/` 136 passed (incl. the new anthropic service-level native test); `Tests/Chat/test_anthropic_native_tools.py` 14 passed; gateway suite unchanged.
