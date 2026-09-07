# Cohere Native Tool-Calls — Live Gate (task-267)

- **Date:** 2026-07-17
- **Branch:** `claude/cohere-native-267` (v2 /chat migration + native tools, commits f3dbd900..359f91f7 pre-flip)
- **Model:** `command-a-03-2025` (config default; handler fallback aligned in Task 1)
- **Recipe:** `cohere_gate.py` — real `ConsoleAgentBridge` → real `ConsoleProviderGateway` → real
  `chat_api_call` → `chat_with_cohere` → `api.cohere.com/v2/chat` over HTTPS, streaming ON
  (exercises the tool-call-start/delta fragment emission live). `NATIVE_TOOLS_PROVIDERS` overridden
  in-process (flip not yet landed — AC ordering, 263/266 precedent). Key from the git-excluded
  repo-root `cohere-api-key.txt`, never printed.

## Verdict: GATE PASS (both cases)

| Case | What | Result |
|---|---|---|
| A (`gate-A-B-2026-07-17.txt`) | Single calculator round-trip, streaming | ✅ native tools sent, no fence in system prompt, `tool_call`→`tool_result`→second model turn accepted, final answer `18018` (2.2s) |
| B first run (`gate-A-B-2026-07-17.txt`) | Parallel two-tool reply | ❌ 400 on the echo turn — found a REAL bug (below) |
| B rerun (`gate-A-B-rerun-2026-07-17.txt`) | Parallel two-tool reply | ✅ two `tool_call` steps in ONE reply (`get_current_datetime` + `calculator`), both results accepted, final answer correct (date + 637), `status=done` |

## Live-gate finding (fixed in 359f91f7)

Cohere rejected the case-B echo turn with
`invalid tool call provided in messages[2].tool_calls[0]: tool arguments must be a stringified JSON object`.
Root cause: a **no-arg** streamed tool call accumulates `arguments=""` — `tool-call-start` seeds the
empty string and no `tool-call-delta` ever follows — and the echo sent it back verbatim. Case A
passed only because the calculator call had real arguments. Fix at the request boundary:
`_cohere_request_tool_calls` normalizes empty/whitespace string arguments to `"{}"`; pinned by
`test_empty_streamed_arguments_echo_as_empty_json_object`.

This is the same class of finding the sibling gates produced (google: Gemini 3 `thoughtSignature`
echo; anthropic: none) — the reason the flip is gated on live evidence rather than mocks.

## Scout-shape verifications the gate settled

- v2 non-streaming `message.content` is a parts array; `tool_calls` carry real ids
  (`<name>_<suffix>`), `arguments` as JSON strings; `tool_plan` present on tool-call replies.
- The echo shape `{"role":"assistant","tool_plan":...,"tool_calls":[...]}` +
  `{"role":"tool","tool_call_id":...,"content":[{"type":"document","document":{"data":...}}]}` is
  accepted (200) including parallel calls.
- Streaming event stream drives the handler's `tool-call-start`/`tool-call-delta`/`message-end`
  branches end-to-end through the real gateway accumulator (case B: two calls in one stream).
