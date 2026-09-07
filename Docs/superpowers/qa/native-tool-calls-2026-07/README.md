# Native provider tool-calls — live gate (task-243)

**Date:** 2026-07-16 · **Branch:** `claude/native-tool-calls` (gate run at `bbf29ec8`)
**Provider under test:** `custom-openai-api` (member of `NATIVE_TOOLS_PROVIDERS`) → real llama.cpp server @ `127.0.0.1:9099` (Qwen3.6-27B, Q8). Fence-regression case: the `llama_cpp` provider branch (must stay fence-only).
**Harness:** `native_gate.py` (this directory) — the REAL Console reply engine end-to-end: `ConsoleAgentBridge.run_reply` → `_StreamingModelAdapter` → real `ConsoleProviderGateway` → real `chat_api_call` → real HTTP. The only instrumentation is a recording passthrough around `gateway.stream_chat` that logs whether `tools=` was sent and whether the fence protocol appeared in the system prompt, then delegates unchanged. Isolated `$HOME` config (`[api_settings.custom] api_url`), `PYTHON_KEYRING_BACKEND=fail`.

## Why custom-openai-api (and not a cloud endpoint)

No cloud provider credential was available in this environment (config holds placeholders; provider env vars unset). The local llama.cpp server natively honors OpenAI `tools=` (verified by direct curl: `finish_reason: "tool_calls"`, id-carrying `tool_calls` array), so `custom-openai-api` — which shares the exact request/response/streaming-delta shape and code path with every OpenAI-compatible cloud provider in the native set — was used as the real end-to-end provider. AC #2's literal "cloud provider" wording is flagged for the user: wire a real cloud key and re-run `native_gate.py A`, or accept this evidence.

## Blocker found and fixed on the way in

`chat_with_custom_openai` (and `chat_with_local_llm`) crashed on EVERY call — `provider_name=cfg.capitalize()` where `cfg` is the settings **dict** — meaning the custom-openai provider has been unusable end-to-end on dev. Pre-existing, surfaced by this gate; fixed in `858d9ce9` with 2 regression tests (`Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py`).

## Case A — native single tool-call round-trip: **PASS** (46.5s)

Evidence: `case-A-native-single-2026-07-16.txt` + run DB `A-native-single.db` (scratch).

- Prompt: *"What is 234\*77? Use the calculator tool, then answer with just the number."*
- Both provider turns carried `tools=['spawn_subagent', 'calculator', 'get_current_datetime']`; `fence_in_system=False` on both — the fence protocol was fully suppressed (AC #1 native selection).
- Run steps: `model → tool_call calculator → tool_result {"expression": "234 * 77", "result": 18018} → model "18018"`; status `done`; final streamed answer `"18018"` (correct).
- Transcript carries the real `⚙ calculator → …` TOOL marker — native calls render identically to fence calls.
- This is `ModelTurn.tool_calls` populated from a real native provider response, dispatched through `run_agent_loop`, answered as a `role="tool"` history message, end-to-end through the Console reply engine (AC #2 substance; provider caveat above).

## Case B — native multi-call attempt: **INCOMPLETE** (best-effort) → **PASS on rerun**

First run (2026-07-16, Qwen3.6-27B): the multi-call prompt hit the 300s provider read-timeout mid-generation; the run ended honestly as `error` ("Provider error from custom") with no wedge — and the llama.cpp server itself died around this point. AC #3 (multi-call batch dispatched in one turn) is demonstrated end-to-end by tests at both the engine level (`test_native_multi_call_batch_dispatches_both_in_one_turn`) and the service level (`test_native_multi_call_reply_dispatches_both_tools_in_one_turn`) against the exact OpenAI response shape case A proved real. See the rerun below.

## Case C — fence regression on llama_cpp: **PARTIAL** (server died) → **PASS on rerun**

First run: before the connection failed, the recorder captured the half that this branch changed: `tools=None` on the wire and `fence_in_system=True` — the `llama_cpp` provider correctly stays on the fence path (AC #1 fallback, AC #4 routing). The round-trip itself could not complete because the server was already down (`All connection attempts failed`, 0.0s — infrastructure, not code). See the rerun below.

## Rerun 2026-07-17 (post-merge, PR #648 on dev): **GATE PASS — A, B, C all green**

Evidence: `rerun-A-B-C-2026-07-17.txt`. Server back up with a different model (gemma-4-26B-A4B, Q4; the harness now reads the served model from `/v1/models`), which also natively honors `tools=` — a second real model exercising the same native path.

- **A (native round-trip): PASS**, 5.4s — `tools=` on both turns, `fence_in_system=False`, calculator round-trip, final answer `18018`.
- **B (native multi-tool): PASS**, 5.6s — both tools genuinely round-tripped natively (`get_current_datetime` then `calculator`), correct combined final answer ("The current date is 2026-07-17 and 91 * 7 is 637"), 2 ⚙ TOOL markers. The model serialized the calls across two turns rather than batching them in one reply — parallel single-turn batching remains pinned by the engine/service tests above (whether a model emits it is model-dependent).
- **C (fence regression): PASS**, 3.8s — the previously missing half completed: fence protocol in the system prompt, `tools=None` on the wire, the reply arrived as a real ` ```tool_call ` fence, parsed and round-tripped to the same `18018`. The `llama_cpp` provider path is confirmed fence-only end-to-end on the merged code.

## Suites at gate HEAD

`Tests/Agents/ + test_console_agent_bridge.py + test_console_provider_gateway.py + test_console_agent_swap.py + Tests/LLM_Provider_Catalog/` → **337 passed / 0 failed** (350 after the PR-review additions).

## Residuals

- Optional: re-run `native_gate.py A` against a real cloud key if one is ever wired (AC #2 was accepted on the native-capable-provider evidence per the 2026-07-16 user decision).
- task-263 filed: Anthropic/Google/Cohere handlers normalize away tool-use blocks; they stay fence-only until their normalizers build `message.tool_calls`.
