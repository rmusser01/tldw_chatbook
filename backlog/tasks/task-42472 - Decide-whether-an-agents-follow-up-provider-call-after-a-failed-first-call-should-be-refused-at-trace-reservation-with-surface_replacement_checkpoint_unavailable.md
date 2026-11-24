---
id: TASK-42472
title: "Decide whether an agent's follow-up provider call after a failed first call should be refused at trace reservation with surface_replacement_checkpoint_unavailable"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 05:28
labels:
- follow-up
- console
- agents
- trace
- investigation
priority: low
references:
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_trace_runtime.py
- tldw_chatbook/Agents/activation.py
- Tests/Performance/test_console_managed_received_acceptance.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42412. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** In TASK-42412's local reproductions (a natural one in 1 of 18 standalone runs, and a forced one), the agent's first provider call (AGENT_FIRST) reserved its trace, and the TOOL_LOOP call that followed (`first_call=False`) was refused at reservation. The refusal was `ValueError("surface_replacement_checkpoint_unavailable")` from `prepare_current_surface_delta` (`Chat/console_trace_service.py:2927`, the branch `if replacement_range is not None and bootstrap`). It was reached through `ConsoleTraceBoundaryFactory.__call__` (`Chat/console_trace_runtime.py:816`), re-raised as `TraceReservationRefused` by `_rolled_back_refusal` (line 84), and left the reservation `not_established`. There the first call had failed because the test's own recorder (its stub provider path) raised, not a real provider. Whether a real provider error on an agent's first call leads to the same refusal on its next call, so that a user's agent run stops at trace reservation instead of reporting the provider error, is not known.

**AC amendment (2026-11-23, round-24 review).** Three criteria were added, for these reasons:
- AC#4: review found that the first close-out covered only in-place retries. The agent's other follow-up call after a failure, the ADR-110 provider fallback, goes to a different provider or model, and the retry check refuses that.
- AC#5 and AC#6: the coordinator chose to fix, in this task, the two defects the investigation found, rather than file them. These are the managed received test's `**kwargs` stand-in, and llama.cpp agent runs never retrying.
- AC#7 (second review): a retry after a stream that broke mid-reply wrote the retry's text after the failed call's partial text in the same assistant message ("The capital of FrThe capital of France is Paris."). AC#6 made this reachable for llama.cpp, and it already happened on dev for hosted providers whose broken stream surfaces as `ChatProviderError(502)` (OpenAI, Groq, Mistral, OpenRouter, Moonshot, Fireworks, Together, Hugging Face).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes record whether a real provider error on an agent's first call (not a test double) can be followed by a TOOL_LOOP call, and if so whether that call is refused with `surface_replacement_checkpoint_unavailable`
- [x] #2 If it can, the intended behaviour is recorded (refusal with a reason the user sees, or a reservation against the failed call's checkpoint) and the code matches it, with a test shown red first for any change
- [x] #3 If it cannot, the task closes with the code path that rules it out
- [x] #4 The notes record the provider-fallback path after a failed call and whether a Capture-On run refuses it, from a probe on a real Console agent send; if it does, a follow-up task records the evidence and the options
- [x] #5 `Tests/Performance/test_console_managed_received_acceptance.py`'s provider stand-in keeps `chat_api_call`'s signature, and the test asserts its Capture-On trace has no omission node; red before the fix, green after
- [x] #6 A llama.cpp (direct HTTP) agent run retries a transient failure in place (HTTP 408, 425, 429, 500, 502, 503, 504 or 529; a timeout; a refused connection; a protocol error), while a 400/401/404 still ends the run at once; a test drives such a run, red before and green after
- [x] #7 When an agent run retries a model call in place, or falls back, after a call failed mid-stream, the reply starts again from what the message held before the failed call: no partial text or thinking from the failed call survives. If the retries run out, the failed message keeps only the last attempt's partial, as a run that fails without retrying does. Tests drive a llama.cpp and a hosted (OpenAI) streaming run through a mid-stream failure, red before the fix
<!-- AC:END -->

## Implementation Plan

1. Trace the code path from a failed AGENT_FIRST provider call to the next reservation: the agent loop's failure handling, the route choice, and the factory's failed-call retry rule.
2. Probe a real Console agent send with Capture On and the real `ConsoleTraceBoundaryFactory`, faking only the provider (HTTP or handler level), so the first call fails with a real provider error. Record whether a TOOL_LOOP call follows and whether its reservation is refused.
3. Re-run TASK-42412's forced reproduction with temporary instrumentation of the surface admission, to see what the refused TOOL_LOOP was compared against.
4. Change code only if a real path is refused; otherwise record why the behaviour is right and close.
5. (Review, AC#4.) Probe the ADR-110 provider-fallback path the same way, with a fallback chain configured inside the test only, and Capture On and Off. If Capture On refuses it, file TASK-42525 with the evidence and the options.
6. (AC#5.) Assert that the managed received test's Capture-On trace has no omission node (red with the bare stand-in), then give the stand-in `chat_api_call`'s signature (green).
7. (AC#6.) Drive a llama.cpp agent run through each transient and non-transient failure (red at base), then teach `Agents/model_retry.is_transient_model_error`, the one classifier both of `_recover_model_failure`'s branches use, the raw httpx shapes.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **In-place retries are fine.** A real provider error on an agent's first call is followed by an in-place TOOL_LOOP retry, and the trace admits that retry. TASK-42412's `surface_replacement_checkpoint_unavailable` came from that test's `def reply(**kwargs)` stand-in for `chat_api_call`, the trap TASK-41310 recorded.
- **A provider fallback is refused.** The other follow-up call after a failure, the ADR-110 provider fallback, is refused at dispatch whenever Capture is On. That is filed as TASK-42525 for a decision, not fixed here.
- **Two fixes in this task.** The managed received test's stand-in now keeps the real signature, and llama.cpp agent runs now retry transient failures in place.

**AC#1: in-place retries.** A TOOL_LOOP call follows a failed AGENT_FIRST for providers dispatched through `chat_api_call`, and it is admitted.
- `Agents/agent_runtime.py::_recover_model_failure` retries a transient error in place and returns `NEXT_TURN`. The Console default is `agent_max_model_retries = 2`.
- `Agents/model_retry.py::is_transient_model_error` counts as transient:
  - `ChatRateLimitError`;
  - `ChatProviderError` and `ChatAPIError` with status 408, 425, 429, 500, 502, 503, 504 or 529 (`ChatAuthenticationError`, `ChatBadRequestError` and `ChatConfigurationError` never);
  - the builtin `ConnectionError` and `TimeoutError`;
  - since AC#6, `httpx.HTTPStatusError` with one of those statuses, `httpx.TimeoutException`, `httpx.ConnectError` and `httpx.RemoteProtocolError`.
- `Chat/console_streaming_request.py` routes every call after the run's first (`request_count >= 1`) as `TOOL_LOOP`.
- The factory (`Chat/console_trace_runtime.py`, `failed_call_retry`) and `ConsoleTraceService.persist_request` admit that call against the failed call's saved surface only if its target, settings and surface are unchanged. In the words of the docstring of `Tests/Chat/test_console_failed_call_retry_ownership.py`, which pins the rule: "A failed call authorizes an unchanged retry, never arbitrary tool history."
- A temporary probe, deleted after use, ran a real `ConsoleChatController` + `ConsoleAgentBridge` + `ConsoleProviderGateway` agent send with Capture On and the real `ConsoleTraceBoundaryFactory`. It faked only the provider handler (`chat_api_call_fn`), so `Chat_Functions.chat_api_call` kept its real signature.
  - Cases: `ChatRateLimitError` (429), `ChatProviderError` (503), and an unexpected exception the gateway surfaced as `ChatProviderError` (502); complete and streaming; with no context, a session system prompt, project instructions (`AGENTS.md`), and both.
  - Result: 14/14 runs completed, with lineage `agent_first=error` then `tool_loop=complete` and no reservation error.

**Why TASK-42412's TOOL_LOOP was refused.**
- *Instrumented reproduction.* I re-ran TASK-42412's forced reproduction (a temporary 8 s sleep in `_read_default_stop`, the stop-path wait removed) with a temporary log of the surface admission. All were reverted.
  - AGENT_FIRST's saved surface was a single `('provider_overlay', 'omission')` node, not the user's saved revision.
  - The retry therefore planned a one-item replacement. `current_surface_checkpoint` matches only the route a checkpoint was published under, so no `tool_loop` checkpoint existed, and the replacement was refused with `surface_replacement_checkpoint_unavailable`.
- *Why the omission.* The trace verifier binds the final values through `inspect.signature(chat_api_call)`, and a bare `**kwargs` stand-in binds nothing (TASK-41310).
- *Confirmation.* The same forced run with `@functools.wraps(Chat_Functions.chat_api_call)` on the stand-in stored `('message', 'revision')`, and both in-place retries were admitted.

**AC#2: intended behaviour.** An unchanged retry after a failed call is reserved against that call's saved surface, and a changed one is refused. `test_failed_call_retry_requires_exact_durable_chain` pins this: `valid`, `cold`, `system_valid` and `updated_system` are admitted; the `changed_*` and unfinished cases are refused. The code matches this; no change. **AC#3** does not apply: its condition, that no TOOL_LOOP can follow, is false.

**AC#4: provider fallback. Refused with Capture On; filed as TASK-42525.**
- *The path.* After the in-place retries are spent (a transient error), or at once (a 402/403 credit-terminal error, or `ChatModelUnavailableError` for a preset target), `_recover_model_failure` switches to the next ready candidate. Users set the chain in `[console] agent_fallback_providers` (`Chat/console_agent_bridge.py::console_fallback_providers`, wired into `AgentConfig.fallback_providers`) or with preset fallback targets. The switched call goes out through `self.chat_call(api_endpoint=<candidate>)` (`Agents/agent_request_execution.py`) as the run's next call, a TOOL_LOOP call with a different provider, model and endpoint.
- *The refusal.* The previous call is in `error`, so `ConsoleTraceService.persist_request` compares the two request headers and raises `ValueError("trace_tool_chain_unavailable")`. The `changed_provider`, `changed_model` and `changed_endpoint` cases of the same test pin that.
- *The probe* used the same harness, with `agent_fallback_providers = ["groq"]` injected through a test-local `get_cli_setting` wrapper, a fixture `GROQ_API_KEY` set in the test process only, and a deepseek primary that always failed. 8 runs, all reproduced:
  - Capture On, `credit402` and `rate429_exhausted`, complete and streaming: `persist_request` refused the groq call with `trace_tool_chain_unavailable`. The run ended `blocked` (blocker `trace_call`, "Trace capture could not start. Retry, Send without capture, or Cancel."). Groq was never called, and the refused call was left in `reserved`. With `rate429_exhausted`, the two in-place retries were admitted before the fallback was refused.
  - Capture Off, the same four: all completed, and groq answered.
- *TASK-42525* records this evidence and the two options: a fallback starts a new trace chain, or the retry check admits a target change marked as a fallback. I recommend the new chain, because a fallback also rewrites the history into the new provider's protocol (`state.messages[:] = projected`). Not implemented here; the decision is the coordinator's.

**AC#5: the managed received test.** After its `len(provider_calls) == 1` check, `Tests/Performance/test_console_managed_received_acceptance.py` now counts `console_trace_surface_nodes` by `reference_kind`. It asserts the count is not empty and has no `omission` row. Red with the bare stand-in: `managed_received_trace_omissions:[('omission', 1)]`. Green once `reply` carries `@functools.wraps(Chat_Functions.chat_api_call)`. `import functools` is added to the child script.

**AC#6: llama.cpp in-place retry.**
- *The fix.* The llama.cpp direct HTTP path raises raw httpx errors, which `is_transient_model_error` did not count, so a 429/503, a refused connection or a read timeout ended the run at once. The classifier now counts the four httpx shapes listed under AC#1. It is the only classifier, and both branches of `_recover_model_failure` use it: the in-place retry and the fallback-eligibility test. A llama.cpp transient failure therefore also becomes eligible for a configured fallback once its retries are spent, like a hosted one. Non-retryable statuses (400/401/404) still end the run.
- *The test.* `Tests/Chat/test_console_project_instruction_traces.py::test_llamacpp_agent_retries_a_transient_failure_in_place` drives the file's real llama.cpp Console agent harness (httpx `MockTransport`, Capture On). The harness gained an optional `first_failure`.
  - Cases: `http429`, `http503`, `connect`, `timeout` and `remote_protocol` retry once; the run completes with lineage `agent_first=error` then `tool_loop=complete`. `http400`, `http401` and `http404` end the run after one attempt.
  - Red at base: the 5 transient cases (`assert 'failed' == 'completed'`); the 3 terminal ones pass at base too. Green after: 8 passed.
  - The file already runs in "Run the admission-sensitive Console P0 regression tests".
- *User Guide.* No page documents the agent's in-place model retry (`agent_max_model_retries` appears nowhere under `Docs/User_Guide`), so none changed.

**AC#7: a retry after a mid-stream failure no longer garbles the reply.**
- *Cause.* The Console streaming adapter (`Chat/console_streaming_request.py`) writes each visible chunk into the assistant message as it arrives (`append_stream_chunk`), and discards streamed prose only when a call succeeds and turns out to be a tool call. When a stream broke mid-reply, the partial stayed. The in-place retry, or a fallback, then streamed into the same message, so the reply read `<partial><retry text>`.
- *Where it happens.* For llama.cpp, since c24acf71683aa2f3f6a0a9c459f7194b284a7f3b (`RemoteProtocolError`, `ReadTimeout`). On dev already, for the hosted providers whose broken stream surfaces as a retryable `ChatProviderError(502)`. A temporary probe (deleted) checked 14 providers through the real gateway and `chat_api_call`, with a `requests` transport that broke mid-body:
  - OpenAI, Groq, Mistral, OpenRouter, Moonshot, Fireworks, Together, Hugging Face: content, then 502, so the call is retried.
  - DeepSeek: status `None`. Its hosted engine turns the broken read into `HostedChatProtocolError`, so the call is not retried.
  - Anthropic, Google, xAI, custom: 502 before any content.
  - Cohere: no failure surfaced.
- *Fix,* where the partial was written.
  - Before each primary (non-subagent) model call, the adapter records a rollback point: the message's text and `ThinkingCapture.checkpoint()`. It clears the point only when the call returns.
  - If the call raised, the next primary call on that message restores the point first. Only an in-place retry or a fallback follows a failed call in a run.
  - Restoring means: `reset_stream_content` (clears the content and the store's per-message chunk buffers); re-append the text the message held before the failed call (a prefill, or a redirect-cut turn); and `ThinkingCapture.restore()`, which drops the failed call's reasoning and republishes it through `replace_message_thinking` when it changed. Block ids keep counting, so a discarded block's id is never reused.
  - A message missing from the store arms nothing.
  - A run that ends after the failure keeps the last attempt's partial and its failure row, unchanged from a run that fails without retrying. An earlier attempt's text never reaches the database.
  - Not covered: character emote capture. It is armed only on the direct (non-agent) send path.
- *Tests.* `test_a_retry_after_a_mid_stream_failure_replaces_the_partial_reply`, in `Tests/Chat/test_console_project_instruction_traces.py`, covers llama.cpp and OpenAI, each retried and exhausted. For llama.cpp, a reasoning event and a content event come first, then `RemoteProtocolError`. For OpenAI, the real `chat_api_call` runs over a faked `requests.Session.post` whose stream breaks with `ChunkedEncodingError`. The harness gained `failures`, `hosted` and `structured_thinking`.
  - Red before the fix: retried read `'The caFixture reply'` for both providers; exhausted read `'The caThe caThe ca'` for both. (The stream gate holds back a short tail, so the visible partial is `The ca`.)
  - With the text reset but no thinking rollback: llama.cpp retried kept the failed call's reasoning block (`['Reasoning t...'] == []`), and exhausted repeated it.
  - Green: 4 passed. The saved database row equals the store's message in every case.
- *Fallback evidence* (temporary probe, deleted). Capture Off; a deepseek stream gives a partial, then 429, three times; then it falls back to groq. Before the fix the reply read `'The capital of FThe capital of FThe capital of Frecovered'`; after, `'recovered'`.
- *User Guide.* No page documents the agent's in-place retry or fallback, so none changed.

**Runs.**
- *Soak, 20 times each:*
  - The new llama.cpp test at `-n 4`: 20/20 runs, 160/160 cases, load 41-95.
  - The managed received test, serial: 20/20 passed in 26-71 s, load 41-106.
- *Timing.* The new llama.cpp test, serial with the Console P0 step's flags (`-p no:xdist --timeout=180`): 8 passed in 25.5 s (2.2-3.8 s per case) at load 43.
- *Covering files* (`-n 4`): `Tests/Agents/test_model_retry*.py`, `test_preset_fallback.py`, `test_fallback_chain.py`, `Tests/LLM_Calls/test_deepseek_protocol_failures.py`, `test_hosted_provider_engine_policy.py`, and the Chat project-instruction, failure-copy, provider-gateway, run-budget, self-hosted-retry, failed-call-retry and retry-recovery files.
  - Result: 956 passed, 4 failed.
  - The 4 fail the same way with base `model_retry.py` and `console_send_diagnostics.py` swapped in, so they are pre-existing locally: `test_console_send_leaves_base_url_unpinned_in_default_case[deepseek]`, `test_all_qwencloud_kwargs_paths_forward_pinned_mode_and_base`, `TestGatewayExchangeCapture::test_openai_stop_closes_transport_through_gateway_boundary`, `test_stream_chat_provider_400_reports_one_consistent_status`.
- *Second review (AC#7 and the scan pins).*
  - Soak at `-n 4`, 20 rounds: the whole `Tests/Chat/test_console_project_instruction_traces.py` (with the new mid-stream tests), the trace-diagnostic files and `Tests/Chat/test_console_thinking_capture.py`. 20/20 rounds, 2320/2320 tests, load 8-16.
  - Covering, `-n 4`: the 115 non-UI test files that drive the streaming adapter. 144 failed and 18 errored locally; an A/B on those 145 ids (`-n 4`, base `b31e3e9241` source for the four changed modules, then current) gave the same 143 failures on both trees, and none failing now that passes at base. Nearly all are the local `RecoveryRequired: raw_source_selection_changed` trap or `OriginalSourceChanged`.
  - The 30 UI/Performance files that drive the adapter, `-n 4`: 224 failed, 441 passed. Of those, 211 hit the `RecoveryRequired` trap. A/B on the 224 ids: 223 fail at base too. The one difference, `test_console_agent_rail.py::test_full_log_probe_is_cached_per_run_id_while_open`, failed once with a `TypeError` from the test's own lambda stub (`available_columns`), raised in `chat_screen.py`'s log probe, which this change does not touch; it passed 3/3 alone on the current tree.
  - `Tests/CI` 1823 passed, 2 skipped; `Tests/Architecture` serial 1492 passed, 5 skipped; preflight passed; ruff F clean; ruff gate OK.
  - "Run fast PR contract" 4739 passed, 4 skipped; "Run the admission-sensitive Console P0 regression tests" 512 passed (508 plus the 4 new mid-stream cases).
- *Lane steps,* serial with their flags, through the fixed `r19-step.sh`:
  - "Run fast PR contract": 4739 passed, 4 skipped.
  - "Run the admission-sensitive Console P0 regression tests": 508 passed.
  - "Run the recovered suites": 2006 passed, 1 skipped, 1 deselected.

**Files.** `tldw_chatbook/Agents/model_retry.py`; `tldw_chatbook/Chat/console_streaming_request.py` and `tldw_chatbook/Chat/console_thinking_capture.py` (AC#7); `Tests/Chat/test_console_project_instruction_traces.py`; `Tests/Performance/test_console_managed_received_acceptance.py`; `backlog/tasks/task-42525 - A-Console-agents-provider-fallback-is-refused-at-dispatch-when-Capture-is-On.md` (new); one sentence on TASK-42412's entry in `backlog/docs/lessons-testing-evidence.md` (first close-out).
<!-- SECTION:NOTES:END -->
