---
id: TASK-42472
title: "Decide whether an agent's follow-up provider call after a failed first call should be refused at trace reservation with surface_replacement_checkpoint_unavailable"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 10:23
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

**AC amendment (2026-11-24, round-24 stack).** Two criteria were added after the coordinator decided to lift the stacked #3517 port's Capture-On retry cut-off (`captured_run_budget`, TASK-34100.2 AC#18):
- AC#8: with the cut-off lifted, a probe found that a Capture-On llama.cpp streaming run (`streaming = true` ships on) whose stream and its non-streaming fallback both fail with a transient error ended on 'Trace capture could not start' instead of the provider's error. The trace factory compared the in-place retry with the segment's latest call, the failed `LLAMA_FALLBACK` call, not with the streamed call it repeats (`trace_tool_chain_unavailable`). AC#6 made this path reachable; the cut-off hid it under Capture On.
- AC#9: the cut-off's premise (a 2026-10-03 pass-1 log) predates dev's failed-call retry rule (164abdc1c9, 2026-10-04), and AC#1's probe shows captured in-place retries are admitted.
- AC#8 amended (2026-11-24, review of the first AC#8 fix): a run that recovered this way still blocked the user's *next* message (`completed_tool_turn_lineage`), because the completed-turn check counted the failed resends as foreign calls. The criterion now covers the next send.
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
- [x] #8 A Capture-On llama.cpp streaming agent run whose stream and its non-streaming fallback both fail with a transient error retries in place. The trace admits each retry, and the run ends on the provider's own error, never the trace card, when retries run out. After such a run recovers, the user's next message is sent
- [x] #9 Capture On keeps the configured model retries (the #3517 port's `captured_run_budget` cut-off is removed); a captured provider fallback stays refused (TASK-42525)
<!-- AC:END -->

## Implementation Plan

1. Trace the code path from a failed AGENT_FIRST provider call to the next reservation: the agent loop's failure handling, the route choice, and the factory's failed-call retry rule.
2. Probe a real Console agent send with Capture On and the real `ConsoleTraceBoundaryFactory`, faking only the provider (HTTP or handler level), so the first call fails with a real provider error. Record whether a TOOL_LOOP call follows and whether its reservation is refused.
3. Re-run TASK-42412's forced reproduction with temporary instrumentation of the surface admission, to see what the refused TOOL_LOOP was compared against.
4. Change code only if a real path is refused; otherwise record why the behaviour is right and close.
5. (Review, AC#4.) Probe the ADR-110 provider-fallback path the same way, with a fallback chain configured inside the test only, and Capture On and Off. If Capture On refuses it, file TASK-42525 with the evidence and the options.
6. (AC#5.) Assert that the managed received test's Capture-On trace has no omission node (red with the bare stand-in), then give the stand-in `chat_api_call`'s signature (green).
7. (AC#6.) Drive a llama.cpp agent run through each transient and non-transient failure (red at base), then teach `Agents/model_retry.is_transient_model_error`, the one classifier both of `_recover_model_failure`'s branches use, the raw httpx shapes.
8. (AC#8.) Drive the real llama.cpp Console harness with streaming, Capture On and each of the 6 transient failure kinds before any content, recovering after two rounds and exhausted. Show it red with the cut-off removed and no trace fix. Then, in the factory's tool-loop check (`Chat/console_trace_runtime.py`), treat an ERRORed `LLAMA_FALLBACK` call that directly follows an ERRORed `previous`, with the same owner, segment, turn, policy and surface, as part of `previous`, so the unchanged-retry check runs against the streamed call. Add trace-rule negatives: a COMPLETE fallback, and another chain's fallback, still refuse; changed model, settings or surface still refuse.
9. (AC#9.) Delete `captured_run_budget` and its call site; reword `captured_fallback_providers`' docstring; turn the AC#18 pins in `Tests/Chat/test_rate_limit_reads_as_rate_limit.py` into a captured 429 that retries through the real trace factory and ends on the rate-limit copy. Fix the User Guide, the lesson, TASK-34100.2 and TASK-34100.2.1. Lane dev's captured-retry tests (`Tests/Chat/test_console_trace_model_retries.py`, `Tests/UI/test_console_captured_model_retry.py`) and the trace-rule file. Check a captured OpenAI streaming 429's copy with a real `requests.Response` (report, do not fix).

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
- *Cause.* The Console streaming adapter (`Chat/console_streaming_request.py`) writes each visible chunk into the assistant message as it arrives (`append_stream_chunk`), and discards streamed prose only when a call succeeds and turns out to be a tool call. When a stream broke mid-reply, the partial stayed. The in-place retry, a fallback or the budget wrap-up then streamed into the same message, so the reply read `<partial><retry text>`.
- *Where it happens.* For llama.cpp, since c24acf71683aa2f3f6a0a9c459f7194b284a7f3b (`RemoteProtocolError`, `ReadTimeout`). On dev already, for the hosted providers whose broken stream surfaces as a retryable `ChatProviderError(502)`. A temporary probe (deleted) checked 14 providers through the real gateway and `chat_api_call`, with a `requests` transport that broke mid-body:
  - OpenAI, Groq, Mistral, OpenRouter, Moonshot, Fireworks, Together, Hugging Face: content, then 502, so the call is retried.
  - DeepSeek: status `None`. Its hosted engine turns the broken read into `HostedChatProtocolError`, so the call is not retried.
  - Anthropic, Google, xAI, custom: 502 before any content.
  - Cohere: no failure surfaced.
- *Fix,* where the partial was written.
  - Before each primary (non-subagent) model call, the adapter records a rollback point: the message's text and `ThinkingCapture.checkpoint()`. It clears the point only when the call returns.
  - If the call raised, the next primary call on that message restores the point first. After a failed call, a run asks again only with an in-place retry, a provider fallback, or the budget wrap-up (`_phase_turn_start` calls `active_call_model` when the budget is spent; `Agents/agent_runtime.py`).
  - Restoring means: `reset_stream_content` (clears the content and the store's per-message chunk buffers); re-append the text the message held before the failed call (a prefill, or a redirect-cut turn); and `ThinkingCapture.restore()`, which drops the failed call's reasoning and republishes it through `replace_message_thinking` when it changed. Block ids keep counting, so a discarded block's id is never reused.
  - A message missing from the store arms nothing.
  - A run that ends after the failure keeps the last attempt's partial and its failure row, unchanged from a run that fails without retrying. An earlier attempt's text never reaches the database.
  - Not covered: character emote capture. It is armed only on the direct (non-agent) send path.
- *Tests.* `test_a_retry_after_a_mid_stream_failure_replaces_the_partial_reply`, in `Tests/Chat/test_console_project_instruction_traces.py`, covers llama.cpp and OpenAI, each retried and exhausted. For llama.cpp, a reasoning event and a content event come first, then `RemoteProtocolError`. For OpenAI, the real `chat_api_call` runs over a faked `requests.Session.post` whose stream breaks with `ChunkedEncodingError`. The harness gained `failures`, `hosted` and `structured_thinking`.
  - Red before the fix: retried read `'The caFixture reply'` for both providers; exhausted read `'The caThe caThe ca'` for both. (The stream gate holds back a short tail, so the visible partial is `The ca`.)
  - With the text reset but no thinking rollback: llama.cpp retried kept the failed call's reasoning block (`['Reasoning t...'] == []`), and exhausted repeated it.
  - Green: 4 passed. The saved database row equals the store's message in every case.
  - Held text (third review). `test_a_retry_keeps_the_text_and_thinking_held_before_the_failed_call` drives the streaming adapter directly through three llama.cpp calls:
    - a turn cut by a redirect, which leaves its text, a separator and its reasoning in the message;
    - a call that breaks mid-stream;
    - the retry.

    The final message is the cut text plus the retry's reply, and its thinking is the cut turn's block alone. Each temporary edit below was reverted, and each is red:
    - Deleting the re-append of the held text: `assert 'Fixture reply' == 'Looking at t...Fixture reply'`.
    - Skipping the thinking rollback: `"Cut turn reasoning and the failed call's" != 'Cut turn reasoning'`.
  - `ThinkingCapture.checkpoint()`/`restore()` unit tests, in `Tests/Chat/test_console_thinking_capture.py`:
    - A block made after a restore gets a new id. It is red, with a colliding id, if `restore` resets the block sequence.
    - A restore brings back the shorter text of a block a delta had grown, and the next delta appends to that text.
    - `restore` after `settle` raises.
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

**Round-24 stack (2026-11-24): AC#8 and AC#9.** The coordinator lifted the #3517 port's Capture-On retry cut-off; the probe that checked its premise found one refused retry path, fixed here.

**AC#8: a failed llama.cpp stream fallback no longer refuses the retry.**
- *Cause.* A llama.cpp stream that fails before any content is sent once more without streaming, on its own `LLAMA_FALLBACK` call (task-19324; no chain, so its own run id). When that failed too, the agent's in-place retry had the streamed call as `previous`, but the segment's latest call boundary was the fallback. `ConsoleTraceBoundaryFactory.__call__`'s tool-loop check (`latest_call.call_id != previous.call_id`) refused it with `trace_tool_chain_unavailable`, and the run ended `blocked` on 'Trace capture could not start'.
  - Probe (temporary, deleted): with the cut-off removed, 12 of 12 Capture-On llama.cpp streaming cases (429, 500, 503, refused connection, read timeout, protocol error; PII on and off) ended that way. The traces held no omission node, so this was not the TASK-41310 trap. Capture Off, llama.cpp complete mode and OpenAI (complete and stream) were not affected.
  - `llama_cpp` ships `streaming = true`, so this is the default llama.cpp send. AC#6 made the path reachable; the cut-off hid it under Capture On.
- *Fix.* One shared helper in `Chat/console_trace_continuation.py`. `is_failed_stream_fallback(streamed, fallback)` holds only when all of these are true:
  - `fallback` is a `llama_fallback` call in `error` (outcome `error`, settled);
  - `streamed` is in `error`;
  - owner, segment, turn, policy and surface node match.

  `failed_stream_fallback_parent(repository, cursor, call_id)` returns the call whose call boundary directly precedes the fallback's, when that pair holds. Two checks use it:
  - *The retry.* `ConsoleTraceBoundaryFactory.__call__`'s tool-loop check sees past the latest call only when that call's parent is `previous`. The unchanged-retry checks then run against `previous`, the streamed call, as before.
  - *The next send (review, 2026-11-24).* The first fix tested only the run. After a recovered run, the user's next message was refused with `completed_tool_turn_lineage`: `validate_completed_tool_turn` refuses any call boundary in the finished run's span whose run id differs, and each failed resend has its own. That foreign-call query now excludes a boundary only when its parent is a call of the run being validated (`chain_terminal.run_id`); any other foreign call still refuses, and more than 256 refuse unread. `ConsoleTraceService.persist_request` and `prepare_current_surface_delta` both reach that validation for the next turn, so the helper covers them too.

  `persist_request`'s own failed-call header comparison reads the logical `previous` (sequence - 1), never the latest boundary, so it needed no change.
- *The exhausted case is a separate cause, filed as TASK-42534.* After a run that ends `failed`, the next send is refused with `completed_tool_turn_unavailable`. `validate_completed_tool_turn` refuses a closed turn whose terminal call never started a response (`closed and not explicit_discard and terminal.response_started_at is None`). The helper does not change that. The block reproduces on dev af4b13ea75, in a scratch worktree since removed:
  - for a deepseek run whose retries ran out;
  - for a llama.cpp stream whose resend also failed (dev does not retry llama.cpp), with lineage `agent_first`, `llama_fallback`, both `error`;
  - on this branch, also for OpenAI's real handler.
- *Tests.*
  - `Tests/Chat/test_console_project_instruction_traces.py::test_a_captured_llamacpp_stream_retries_after_its_fallback_fails`: the real llama.cpp Console harness, streaming, Capture On, each of the 6 failure kinds before any content.
    - `recovered`: two failed rounds (stream and fallback, twice), then a reply. Lineage is `agent_first, llama_fallback, tool_loop, llama_fallback` in `error`, then `tool_loop` complete. Then a second message is sent and completes with no reservation error.
    - `exhausted`: 6 requests, and the run ends `failed` on the provider's own copy ("Agent run failed: provider returned HTTP 503 (fixture)." and so on), with no recovery card.
    - Red with the cut-off removed and no trace fix: 12 failed (`assert ['trace_tool_..._unavailable'] == []`).
    - Red before the next-send fix: the 6 `recovered` cases (`assert ['completed_t...turn_lineage'] == []`, at the second message).
    - Green after: 12 passed.
  - `Tests/Chat/test_console_failed_call_retry_ownership.py::test_a_retry_after_a_failed_llama_stream_fallback_is_checked_against_the_stream`, gateway level with the real factory.
    - Admitted: `valid` and `second_round` (a retry after the retry's own failed fallback).
    - Refused: `fallback_completed` (a COMPLETE fallback, then a TOOL_LOOP claiming retry), `other_chain_fallback` (another run's failed stream and fallback in between), `other_chain_failure` (review: non-streaming, so another run's failed AGENT_FIRST directly follows this run's failed call), `changed_model`, `changed_temperature` and `changed_surface`.
  - `test_only_an_errored_resend_of_an_errored_call_is_its_failed_stream_fallback` (same file): the predicate on hand-built `TraceCallRecord`s. The valid pair holds, plus one refused case per condition:
    - route;
    - `streamed.state`;
    - `fallback.state`;
    - `fallback.outcome`;
    - `fallback.settled_at`;
    - each of the five identity fields.
  - Red evidence, each from a temporary edit that was reverted:
    - *The predicate.* Removing each condition reds exactly its own unit case. Removing the route condition also reds `other_chain_failure` (it is admitted); before the review, all 7 gateway cases stayed green under that mutation.
    - *The factory.* With no helper (latest must be `previous`), `valid`, `second_round` and all 12 harness cases are red. Accepting any failed fallback, whatever call it follows, reds `other_chain_fallback`.
    - *The completed-turn check.* With the old "any foreign boundary refuses" check, the 6 `recovered` cases are red at the next send.
    - *Not pinned: the completed-turn check's run-id comparison.* Excluding a failed fallback of *any* run leaves every test green. Through the product that case cannot arise: for another run's call to sit inside this run's span, this run would have to continue after it, and the factory refuses that (`other_chain_fallback`, `other_chain_failure`). The comparison stays as defence in depth.

**AC#9: the Capture-On cut-off is removed.**
- `captured_run_budget` and its call site in `Chat/console_agent_preparation.py` are deleted. `captured_fallback_providers` stays, with a docstring that gives its own reason: a fallback goes to another provider, model or endpoint, which the unchanged-retry rule refuses (TASK-42525).
- `Tests/Chat/test_rate_limit_reads_as_rate_limit.py`:
  - module docstring corrected;
  - `test_a_captured_agent_run_never_retries_a_failed_model_request` (a unit test of the deleted helper) is replaced by `test_a_captured_rate_limit_is_retried_and_ends_on_the_rate_limit`: the real controller, gateway and trace factory with Capture On, and OpenAI's real handler, with only `requests.Session.post` faked, returning a real 429 `requests.Response`. Three posts, both retries admitted (`agent_first` then two `tool_loop`, all `error`), and the run ends on "Provider error from OpenAI: rate limit exceeded. Status: 429. OpenAI says: ..." with no recovery card. It is red if the factory refuses a retry after a failed call (`assert 1 == (1 + 2)`) and red with the cut-off restored (the same);
  - `test_run_reply_plans_a_captured_run_without_model_retries` is now `test_run_reply_plans_a_captured_run_with_its_model_retries`: both modes plan the admitted 2 retries. `[CAPTURE_ON]` is red with the cut-off restored;
  - the fallback test's docstring cites TASK-42525.
- The 9 round-24 diag tests that the cut-off failed, and dev's 6 captured-retry tests (`Tests/Chat/test_console_trace_model_retries.py`, `Tests/UI/test_console_captured_model_retry.py`), went from red to green.
- No change was needed in `Tests/Agents/test_active_turn_redirect.py`, `test_fallback_chain.py`, `test_model_retry_loop.py`, `test_preset_fallback.py` or `Tests/Chat/test_openai_luna_native_tools.py`. Their `max_model_retries=0` builds their own `AgentConfig` budget for a pure agent-loop test and never touches capture mode.
- Docs: `Docs/User_Guide/console.md` (a run retries a 429 or 5xx with capture on or off; only the provider fallback needs capture off); the existing lesson in `backlog/docs/lessons-console-wiring.md` gained a dated correction; TASK-34100.2 and TASK-34100.2.1 have dated notes.
- *OpenAI streaming check (not fixed; reported).* With a real `requests.Response`, a *streamed* OpenAI 429 reads "Provider error from OpenAI: unexpected provider error." with capture on or off. `chat_with_openai`'s stream generator catches the `HTTPError` from `raise_for_status` as a `RequestException` and yields an SSE error line (`openai_stream_error`, "Stream connection error: 429 ..."). It has been on dev since ee46503f9c. A non-streaming 429 reads as a rate limit.

**Lanes (round 24).** Dev's two captured-retry files were red on the stack and on no lane.
- "Run the admission-sensitive Console P0 regression tests", mid-list after `Tests/Chat/test_console_project_instruction_traces.py`, with `Tests/CI/test_response_rules_critical_gate.py` updated in the same order, and a dated comment above the step: `Tests/Chat/test_console_trace_model_retries.py`. It stays in this profile because it is `bootstrap_profile`: its harness reads the tool gates through config, and without the marker it fails 4 of 4. 4 passed in 10.2 s serially with the step's flags (`-p no:xdist --timeout=300`), at load 5.
- "Run the Dreams tracking tests and unlaned riders" (second review, 2026-11-24): `Tests/Chat/test_console_failed_call_retry_ownership.py`, the failed-call retry rule, also on no lane before. It sits after `Tests/Tool_Packs` and before the character cadence file, has a dated comment above the step, and `DREAMS_TRACKING_RIDERS` is updated in the same order. 59 passed in 26.9 s serially with that step's flags (`--timeout=180 --tb=short`) at load 6.
  - *Why it moved twice.* The first review round put it in the sandboxed Console P0 step with its `bootstrap_profile` marker dropped, on the claim that nothing in it reads config, because 59 passed without the marker. That claim was false. Its gateway reads `[model_catalog] use_models_dev` (`stream_chat` -> `prepare_chat_request` -> `get_model_capabilities` -> `models_dev_catalog._enabled`). Without the marker, each read raised `RecoveryRequired('raw_source_selection_changed')`, which `_enabled()` swallows (`except Exception: return False`). The tests passed only because that fallback equals the default (the TASK-32628 swallowed-admission trap). A temporary recorder on `get_cli_setting` (not committed; serial) counted, with the marker, 5 `model_catalog` reads and 0 `RecoveryRequired`; without it, 5 reads and 5 `RecoveryRequired`.
  - *Where it is now.* The marker is back, with a comment that gives the reason. The file cannot return to the admission-sensitive Console P0 profile, which has no CI headroom (TASK-42533). It runs in the Dreams riders after Tool_Packs, because a `bootstrap_profile` file must never precede a sandboxed target there. Lesson: "Green without `bootstrap_profile` does not prove a file reads no config" in `backlog/docs/lessons-testing-evidence.md`.
- UI census tail: `Tests/UI/test_console_captured_model_retry.py`, whole. 22.6 s serially with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`), at load 15-29. The floor is the measured entry count.

**Size rows.** `console_agent_bridge.py` 10,170 -> 10,142, exact. `console_trace_runtime.py` and `console_trace_continuation.py` have no row; both are under 4,000 lines.

**Runs (AC#8/AC#9, round-24 stack).**
- *Soak,* 20 rounds at `-n 4`, load 8-15: the new llama.cpp stream test, diag's 9 formerly red tests (with the rest of their functions), the new trace-rule test, the three changed rate-limit tests and dev's two newly laned files. 20/20 rounds, 840/840 tests.
- *Covering,* `-n 4`: every `Tests/Chat/test_console_trace_*.py` file, the failed-attempt, retry-JSON, self-hosted-retry, run-budget and failure-copy files, the touched files, the Agents retry and fallback files and the luna native-tools file. 1,974 tests, 36 failed. The same 36 ids fail with HEAD's product files swapped in and on a detached worktree at the parent tip d0a7e3badc (36 of 36 each), so they predate this change. Most are `Hooks unavailable; review or disable hook` submit refusals in the sidecar replay and redaction files.
- `Tests/CI` `-n 4`: 1,948 passed, 2 skipped. `Tests/Architecture`, serial: 1,507 passed, 5 skipped. `preflight.sh`: all checks passed (UI census floor 2,766). `ruff check tldw_chatbook --select F`: clean. `Tests/lint/ruff_gate.py`: ratchet OK.
- *Lane steps,* via `r19-step.sh` with each step's flags, run beside one another at load 5-15:
  - "Run the admission-sensitive Console P0 regression tests": 576 passed in 1,654 s. That run used the brief's `-p no:xdist --timeout=180`; the step itself uses `--timeout=300`, which the review round below used;
  - "Run admission-sensitive suites 3" (the rate-limit file; `--timeout=300`): 435 passed in 957 s;
  - "Run fast PR contract" (the size ratchet; `--timeout=180`): 4,873 passed, 4 skipped in 1,148 s.

**Runs (review round, 2026-11-24, rebased onto fu/r24-p3517 f8880bc62c).** The rebase was clean and the replayed diff equals `d0a7e3badc..0306a7f634`. Load was 4-13.
- *Soak,* 20 rounds at `-n 4`: the llama.cpp stream harness test (with its next send), the whole failed-call retry rule file (without its marker at the time; it is back, see Lanes), the two changed rate-limit tests, and dev's two newly laned files. 20/20 rounds, 1,600/1,600 tests.
- *Covering,* `-n 4`, the same file set as the first round: 1,950 passed, 36 failed. These are the same 36 ids that fail at the parent tip; the rebase changed only CI files.
- `Tests/CI` `-n 4`: 1,948 passed, 2 skipped. `Tests/Architecture`, serial: 1,507 passed, 5 skipped. `preflight.sh`: all checks passed. `ruff check tldw_chatbook --select F`: clean. `Tests/lint/ruff_gate.py`: ratchet OK.
- *Lane steps,* via `r19-step.sh` with each step's own flags:
  - "Run the sandboxed Console P0 regression tests" (`-p no:xdist --timeout=180`): 707 passed in 342 s;
  - "Run the admission-sensitive Console P0 regression tests" (`-p no:xdist --timeout=300`): 529 passed in 1,394 s. That is 47 fewer than before, now that the rule file is out of this step; it went to the sandboxed step first, then to the Dreams riders in the second review round.

**Commits.**
- First round, rebased: `5e7f3e36b4b26f0fc4714a834f37e2bbe2167b25` (the trace fix, the cut-off removal, their tests and the size row), `1bd14c64039a622b83bfaf92fdb9c5957d0b94a5` (lanes) and `a0b3805a77116d96e38ebde09ef1592b3a6dcd89` (docs).
- Review round: `bfc8084e1a52113506fe3915ab4fa6149dd8754e` (the next-send fix and the shared helper), `1ab1a7333ac638d18b20f674fc8c6c6450755eb7` (the lane move), then this docs commit.

**Second review round (2026-11-24).**
- *What changed.* The rule file's `bootstrap_profile` marker is restored and the file moved to the Dreams tracking riders (see Lanes). TASK-42534 now inlines its probe and records the confirmed cause.
- *Runs,* at load 6-14:
  - the rule file, 20 times at `-n 4` with the marker: 20/20 rounds, 1,180/1,180;
  - `Tests/CI` `-n 4`: 1,948 passed, 2 skipped;
  - `Tests/Architecture`, serial: 1,507 passed, 5 skipped;
  - `preflight.sh`: all checks passed; ruff F clean; `ruff_gate` OK;
  - "Run the sandboxed Console P0 regression tests" (`-p no:xdist --timeout=180`): 648 passed in 260 s, 59 fewer now that the rule file is out;
  - "Run the Dreams tracking tests and unlaned riders" (`-p no:xdist --timeout=180 --tb=short`): 1,323 passed, 6 skipped, in 693 s in total.
- *Commit:* `fc88ba94306392a749f0fe8d281ab7e656fe2274` (the marker and the lane move), then this docs commit.

**Filed follow-ups (review round).** IDs were checked unused across all refs, worktrees, history names and ref-tip blobs; 42528 and 42531 were taken.
- TASK-42529: a streamed OpenAI 429 reads "unexpected provider error" (AC#18's streaming gap); Anthropic and Gemini are to be checked.
- TASK-42530: a llama.cpp tool call that arrives on the non-streaming fallback blocks the next call. It is on the parent tip too.
- TASK-42532: a streamed retry after a failed non-streaming resend in the same chain (TASK-37421's resend; llama.cpp with tools). Found by code reading and not probed.
- TASK-42533: the admission-sensitive Console P0 job's CI headroom.
- TASK-42534: after a captured run fails before any response, the next send is blocked. It is on dev af4b13ea75 too.
<!-- SECTION:NOTES:END -->
