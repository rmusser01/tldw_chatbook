# LLM — tldw_chatbook/LLM_Calls/ (19 files), 22,561 lines

Worktree: `/Users/macbook-dev/Documents/GitHub/tldw-review` (detached at origin/dev d8fb4053f9). Every command below ran as `cd $WT && source $SCRATCH/env.sh && PYTHONPATH=$WT $PY …`; repro scripts are saved in `$SCRATCH/llm_repro_generatorexit.py`, `$SCRATCH/llm_repro_summarization.py`, `$SCRATCH/llm_timing_probe.py`. No network calls were made; every provider "call" is `unittest.mock.patch("requests.Session.post")` / `requests.post`. No file in the worktree or main checkout was modified.

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| tldw_chatbook/LLM_Calls/LLM_API_Calls.py | 5639 | read in full (1-5639, four Read chunks) |
| tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py | 2504 | read in full (three Read chunks) |
| tldw_chatbook/LLM_Calls/Summarization_General_Lib.py | 2933 | read in full (three Read chunks) |
| tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py | 2340 | read in full (two Read chunks) |
| tldw_chatbook/LLM_Calls/recovery_review.py | 744 | read in full |
| tldw_chatbook/LLM_Calls/hosted_chat.py | 1006 | read in full |
| tldw_chatbook/LLM_Calls/hosted_chat_streaming.py | 227 | read in full |
| tldw_chatbook/LLM_Calls/qwencloud.py | 1296 | read in full |
| tldw_chatbook/LLM_Calls/qwencloud_streaming.py | 1058 | read in full |
| tldw_chatbook/LLM_Calls/qwencloud_url.py | 150 | read in full |
| tldw_chatbook/LLM_Calls/moonshot.py | 1060 | read in full |
| tldw_chatbook/LLM_Calls/zai.py | 941 | read in full |
| tldw_chatbook/LLM_Calls/anthropic_subscription.py | 402 | read in full |
| tldw_chatbook/LLM_Calls/pricing_catalog.py | 870 | read in full |
| tldw_chatbook/LLM_Calls/realtime/openai_session.py | 951 | read in full |
| tldw_chatbook/LLM_Calls/realtime/transport.py | 228 | read in full |
| tldw_chatbook/LLM_Calls/realtime/protocol.py | 190 | read in full |
| tldw_chatbook/LLM_Calls/realtime/__init__.py | 22 | read in full |
| tldw_chatbook/LLM_Calls/__init__.py | 0 | empty |
| (cross-slice, read for evidence only) `Chat/console_provider_gateway.py` 5245-5275, 5805-5975, 6180-6420, 7060-7150; `Chat/Chat_Functions.py` 937-980; `Chat/provider_continuation.py` 180-315; `Chat/thinking_blocks.py` 195-265; `Utils/input_validation.py` 355-530; `Utils/egress.py` 907-1046; `config.py` 1371-1420, 1540-1625, 2165-2195; ADR 012, ADR 062 (grep) | — | sampled |

Every one of the 19 files was read in full; nothing in the slice was sampled or scanned mechanically.

## Findings

### P1 [D1] — Seven of eight streaming handlers in `LLM_API_Calls.py` still `yield "data: [DONE]"` inside `finally`; a consumer Stop (`gen.close()`) raises `RuntimeError: generator ignored GeneratorExit` and the HTTP response is never closed. Only the OpenAI handler was fixed (and pinned).
- Where (yield-in-finally, ALL copies): anthropic `tldw_chatbook/LLM_Calls/LLM_API_Calls.py:2019-2022`, cohere `:2888-2901` (guarded by `not stream_properly_closed`, which is exactly the Stop case), deepseek `:3310-3313`, google `:3918-3921`, groq `:4370-4373`, mistral `:5135-5138`, openrouter `:5389-5392`. Fixed shape: openai `:916-928` (sentinel yielded AFTER the `finally`, with the comment explaining why). Clean: huggingface `:4800-4803`. Variant with no leak but the same RuntimeError: `LLM_API_Calls_Local.py:364-369` (`response.close()` runs BEFORE the sentinel yield; every local provider — llama.cpp/vllm/ollama/mlx/ooba/tabby/aphrodite/custom-openai 1+2 — routes through it).
- Evidence: `PYTHONPATH=$WT $PY $SCRATCH/llm_repro_generatorexit.py` (patch `requests.Session.post` with a 2-line mock stream, `chat_api_call(provider, streaming=True)`, `next()`, `.close()`) →
  ```
  openai      close() OK                                    response.close() called=True
  anthropic   RuntimeError: generator ignored GeneratorExit response.close() called=False
  deepseek    RuntimeError: generator ignored GeneratorExit response.close() called=False
  groq        RuntimeError: generator ignored GeneratorExit response.close() called=False
  mistral     RuntimeError: generator ignored GeneratorExit response.close() called=False
  openrouter  RuntimeError: generator ignored GeneratorExit response.close() called=False
  google      RuntimeError: generator ignored GeneratorExit response.close() called=False
  cohere      RuntimeError: generator ignored GeneratorExit response.close() called=False
  ```
  Consumer path: every stream is wrapped in `recovery_review._OpenAIStream` (`:485-539`) whose `close()` forwards to the inner generator; `Chat/console_provider_gateway.py:5904-5906 call_response_close` wraps that in `contextlib.suppress(Exception)`, so the RuntimeError is swallowed silently and the `response.close()` statement after the yield never runs.
- Why it matters: Console Stop on any non-OpenAI hosted stream leaves the streaming socket open (server keeps generating into a dead buffer up to the 180 s read timeout) until the generator frame is garbage-collected; the local-server variant turns every Stop into a swallowed RuntimeError.
- Recommended correction: same shape as the OpenAI fix — move the sentinel out of `finally` (after the try/except/finally, or a `stopped` flag set in an `except GeneratorExit`); Cohere keeps its `stream_properly_closed` sentinel but emits after `finally`. One mechanical PR across 8 sites plus the OpenAI pinning test's shape per provider.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Chat/test_openai_streaming_usage.py::test_stopping_stream_closes_transport_without_yielding_after_generator_exit` — states the requirement for OpenAI ONLY; `rg -n "GeneratorExit" Tests/Chat/` finds no sibling test.
- Already covered: none (`rg -il "GeneratorExit" backlog/tasks/` → only task-18300, an unrelated Console inspector review).

### P1 [D1] — Google and Cohere streaming handlers drop the provider's usage block; the gateway records usage only from SSE chunks, so streamed Gemini/Cohere turns carry no token counts or cost
- Where: google `tldw_chatbook/LLM_Calls/LLM_API_Calls.py:3798-3921` (the SSE→OpenAI translator reads `candidates` only; `usageMetadata`, present on every Gemini stream chunk, is never emitted); cohere `:2832-2846` (`message-end` carries `delta.finish_reason` AND `usage`; only `finish_reason` is forwarded). Contrast anthropic `:1846-2006` (accumulates `usage` and emits a trailing usage chunk) and openai `:714-715` (`stream_options.include_usage`).
- Evidence: `PYTHONPATH=$WT $PY -` (see the `run("google"…)`/`run("cohere"…)` snippet in this report's shell history: mocked stream with `usageMetadata` / `message-end.usage`, `chat_api_call(streaming=True)`) → `[google stream] yielded=1 chunks; any 'usage' key forwarded: False` / `[cohere stream] yielded=2 chunks; any 'usage' key forwarded: False`. Consumer: `Chat/console_provider_gateway.py:7067-7076 _maybe_record_usage` reads `payload.get("usage")` per chunk; `rg -n "estimate_usage|usage is None" Chat/console_provider_gateway.py` → no fallback estimate.
- Why it matters: the cost ticker / usage ledger (ADR 156 live per-run usage attribution) gets nothing for streamed Gemini and Cohere turns even though the provider sent the numbers; non-streaming turns on the same providers DO report usage (`:4018-4027`, `:2975-2996`), so the ledger is inconsistent per streaming toggle.
- Recommended correction: google — when a chunk carries `usageMetadata`, emit the same `{"choices": [], "usage": {prompt_tokens, completion_tokens, total_tokens}}` trailing chunk the anthropic branch emits (cumulative, so emit once at finish); cohere — on `message-end`, map `usage.billed_units|tokens` exactly as the non-streaming branch (`:2976-2986`) and emit it. S each.
- Size: S · ADR: no · Confidence: verified (handler drop) — consumer effect read from `_maybe_record_usage`, not run through the gateway
- Pinning test: none for streamed usage on either provider (`rg -ln usageMetadata Tests/Chat/` hits `test_google_native_tools.py` and the gateway tests, neither asserts stream usage).
- Already covered: none. Related open question (Left UNVERIFIED): deepseek/groq/mistral/openrouter streams never request `stream_options.include_usage` (`rg -n include_usage LLM_Calls/` → openai, moonshot, qwencloud only), so whether THEIR streamed usage arrives depends on each provider's default.

### P1 [D1] — `summarize_with_kobold` and `summarize_with_tabbyapi` are generator functions; every `analyze("koboldcpp"|"tabbyapi")` call returns `"Error: Unexpected result type <class '…_OpenAIStream'>"` and never contacts the server (ALREADY FILED: task-17387, To Do, priority high)
- Where: `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py:697,712,718,1190,1208,1214,1285` (bare `yield` in the function body, not in a nested generator); re-exported and dispatched via `Summarization_General_Lib.py:36-46`, `:298-308` (`_CHAT_DISPATCH_NAME_ALIASES` maps `koboldcpp`→`kobold`), `:431-453`.
- Evidence: `inspect.isgeneratorfunction(inspect.unwrap(L.summarize_with_kobold))` → True (also tabbyapi, and both re-exports); end-to-end with a mocked 200 `{"results":[{"text":"THE SUMMARY"}]}` and `load_settings` patched: `S.analyze("koboldcpp", "some text", "summarize it", api_key="k", streaming=False)` → `"Error: Unexpected result type <class 'tldw_chatbook.LLM_Calls.recovery_review._OpenAIStream'>"`; same for tabbyapi. New detail beyond the task text: `recovery_review.unqualified` (`:127-131`) sees the generator, wraps it in `_OpenAIStream`, and `analyze()`'s `consume_generator` (`:585-606`, `inspect.isgenerator` only) passes the wrapper through untouched — so the failure is a visible error string, not the "truthy generator stored as evidence" the task describes.
- Why it matters: Library ingest analysis with `[analysis_defaults] provider = koboldcpp|tabbyapi` can never produce a summary.
- Recommended correction: per task-17387 (nest the streaming bodies; re-key the diagnostic ledger). Note the ledger tests already consume these as generators (`Tests/LLM_Calls/test_summarization_diagnostic_privacy.py:506,2789,2818` — `_consume_generator(summarize_with_kobold(...))`), i.e. the tests pin the DEFECT; the task text acknowledges that.
- Size: M (governed diagnostic ledger) · ADR: no · Confidence: verified
- Pinning test: `test_summarization_diagnostic_privacy.py::_invoke_local_credential` et al. — they assert the current (broken) generator contract.
- Already covered: task-17387 (and task-17383 In Progress for the config-section half).

### P2 [D1] — `summarize_with_vllm` fails on every call that passes an explicit `api_key` (`loaded_config_data` is only bound on the no-key branch)
- Where: `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py:1301-1305` (binds `loaded_config_data` only when `api_key` is blank) vs `:1412-1413`, `:1463-1464` (reads it unconditionally). `analyze()` callers pass `api_key` through (`Local_Ingestion/Book_Ingestion_Lib.py:1354`, `PDF_Processing_Lib.py:862`, …).
- Evidence: `L.summarize_with_vllm("k", "text", "prompt")` with `load_settings` patched to a full `vllm_api` table and a mocked 200 → `"vLLM Summarization: Unexpected error occurred: cannot access local variable 'loaded_config_data' where it is not associated with a value"`; `L.summarize_with_vllm(None, …)` → `'THE SUMMARY'`.
- Why it matters: any ingest path that resolves a vLLM credential before calling `analyze()` (the documented, gated path) gets an error string; only the key-less fallback works.
- Recommended correction: hoist `loaded_config_data = load_settings()` above the key check (S).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (`rg -n "summarize_with_vllm" Tests/` → diagnostic-privacy canaries only, which call it with `api_key=None`).
- Already covered: none found (`rg -il "summarize_with_vllm" backlog/tasks/` → none).

### P2 [D1] — `summarize_with_anthropic` mounts a Retry adapter on a session it never uses; the request goes through bare `requests.post`, so configured `api_retries` never apply and a 429 returns `None` on the first attempt
- Where: `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py:1109-1131` (session + adapter built, then `requests.post(...)` at `:1131`), `:1253-1258` (non-200/non-500 → `return None`), `:1106` (manual loop only retries 500 / RequestException). The unused session is also never closed — one leaked `Session` per attempt.
- Evidence: `PYTHONPATH=$WT $PY $SCRATCH/llm_repro_summarization.py` (patch `requests.post` → 429, spy on `requests.Session.post`) → `[anthropic 429] result=None requests.post calls=1 Session.post calls=0`
- Why it matters: a rate-limited Anthropic summarization (Library ingest analysis) yields `None` → `analyze()` reports "Error: Summarization failed unexpectedly." with no retry despite `[anthropic_api] api_retries`; every sibling posts through the session and does get the policy.
- Recommended correction: `session.post(...)` at `:1131`; close the session in a `finally`. The diagnostic manifest (`test_summarization_diagnostic_privacy.py`) freezes this module's log call sites, so the fix must not add a log line.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none for the retry path
- Already covered: none

### P2 [D1] — Three summarization stream generators yield the accumulated full text AGAIN after the deltas, doubling the summary for any consumer that joins chunks
- Where: `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py:2290` (deepseek), `Local_Summarization_Lib.py:1993` (custom_openai), `:2252` (custom_openai_2). Siblings anthropic `:1217-1218`, groq `:1678-1679`, mistral `:2491-2492` carry the same line COMMENTED OUT.
- Evidence: `$SCRATCH/llm_repro_summarization.py` → `[deepseek stream] chunks=['Hello ', 'world', 'Hello world'] joined='Hello worldHello world'`
- Why it matters: doubled output whenever `analyze(streaming=True)` is used with these providers. Reachability today: every `analyze(` caller I could locate passes `streaming=False` or omits it (`Web_Scraping/WebSearch_APIs.py:1414`, `Local_Ingestion/Book_Ingestion_Lib.py:1354,1398,2029`, others default) — so LATENT, a P1 the moment a streaming caller appears.
- Recommended correction: delete the three trailing yields (S).
- Size: S · ADR: no · Confidence: verified (bug); reachability verified-latent
- Pinning test: none
- Already covered: none

### P2 [D1] — `hosted_chat.owned_json_post` and `qwencloud.chat_with_qwencloud` honour a provider `Retry-After` header with no upper bound and sleep it on the worker thread
- Where: `tldw_chatbook/LLM_Calls/hosted_chat.py:867-884` (`_retry_delay` returns `float(int(raw))` uncapped) → `:581 time.sleep(delay)`; consumers moonshot (`hosted_chat_request`, `moonshot.py:237`) and zai (`owned_json_post`, `zai.py:424`). Second copy: `qwencloud.py:130-156 _advance_retry_policy` (`retry_policy.get_retry_after(...)` → urllib3 returns the raw seconds) → `:1291 time.sleep(retry_sleep)`.
- Evidence: `_retry_delay(Mock(headers={"Retry-After":"99999999"}), attempt=0, retry_delay=1.0)` → `99999999.0 seconds (no cap)`. qwencloud copy: read only.
- Why it matters: `api_base_url` is user-configurable for all three providers; a misbehaving/hostile endpoint pins the gateway worker (`console_provider_gateway.py:6337 asyncio.to_thread(worker)`) in `time.sleep` for an arbitrary time — Stop cancels the task but cannot interrupt the thread.
- Recommended correction: clamp to a small cap (e.g. `min(delay, 60.0)` or `config.timeout`) in both copies; treat larger values as "fail now" (S).
- Size: S · ADR: no (ADR 062/045 cover the boundary, not the retry cap) · Confidence: verified (function) / inferred (thread pin)
- Pinning test: `Tests/Chat/test_dispatcher_status_mapping.py:66,75` exercises Retry-After parsing at the dispatcher, not this cap.
- Already covered: none

### P2 [D4b→D1] — Strict-JSON parsing exists in three drifted families; the wire family accepts duplicate keys that the storage family rejects, so a tool-call argument string accepted from a provider is refused when its continuation checkpoint is built
- Where (ALL copies):
  - Family A (depth+node cap, finite floats, str keys, NO duplicate-key rejection): `LLM_Calls/hosted_chat.py:633-678` (`_reject_json_constant`, `_json_shape_is_safe`, `_strict_json_loads`; depth 128 / 1,000,000 nodes / exact `dict`/`list` types) ≡ `LLM_Calls/qwencloud_streaming.py:61-109` (byte-identical modulo annotations); `LLM_Calls/moonshot.py:920-948 _json_shape_is_bounded` ≡ `LLM_Calls/zai.py:913-941` (depth 64 / 50,000 nodes / `Mapping`/`Sequence` / 16 MiB string cap — the excerpt's dup_verbatim row, confirmed).
  - Family B (duplicate-key rejection via `object_pairs_hook`, non-finite rejection, NO depth/node cap — bounded only by Python's recursion limit, which both modules catch at their public boundaries: `thinking_blocks.py:328,363,380,405`, `provider_continuation.py:450,573,584,601,689`): `Chat/thinking_blocks.py:221-235`, `Chat/provider_continuation.py:204-219`.
  - Family C (parse_constant-only, no shape check): `qwencloud.py:301-302,498,864`; `Tools/workspace_tool_protocol.py:394`; `Chat/library_activity.py:498`; `Chat/console_dispatch_checkpoint.py:288`; `Chat/library_preparation.py:187`; `Chat/console_generation_settings_metadata.py:381,387`; `Actor_Packs/export.py:360,371,718`; `Workflows/document_service.py:212`.
  - `Utils/input_validation.py:483 _validate_strict_json_value` (recursive, cycle-detect, finite floats, str keys, no dup-key since it validates decoded objects): private, 1 importer via `validate_tool_arguments` ← `MCP/hub_test_execution.py:22`.
- Which behaviour reaches the wire/storage: A guards provider→app; B guards the persisted continuation/thinking JSON. Drift consequence, verified: `hosted_chat._strict_json_loads('{"a":1,"a":2}')` → `{'a': 2}` (accepted, last-wins); `hosted_chat._normalize_tool_calls([...arguments='{"a":1,"a":2}'])` → accepted; `moonshot._moonshot_continuation_candidate(turn_with_that_call, …)` → `ContinuationValidationError: Invalid continuation data.` (raised by the `parse(dump(candidate))` round-trip at `moonshot.py:403`; `provider_continuation._strict_json_loads` rejects dup keys with `_InvalidContinuation`). `ContinuationValidationError` is not a `HostedChatProtocolError`, so `chat_with_moonshot:250` / `chat_with_zai:448` do not catch it — it escapes as an unexpected exception from the handler (non-streaming) or from `MoonshotStream.provider_continuation` (streaming).
- Why it matters: a provider emitting a duplicate key in tool arguments (unusual but legal JSON) turns a successful tool-call turn into an uncaught `ValueError` subclass at the Console boundary; more generally three definitions of "strict JSON" with different caps means the same payload can pass A and fail B.
- Recommended correction: one public `strict_json_loads(text, *, max_depth, max_nodes, reject_duplicate_keys=True)` in `Utils/input_validation.py` (the canonical home the brief names; today's private `_validate_strict_json_value` covers only the decoded-object half) combining A's caps with B's dup-key hook; A and B become thin wrappers; either make A reject dup keys (then the checkpoint round-trip cannot disagree) or make B tolerate them — pick one and pin it. ADR check: 062 (hosted boundary), 045 (qwencloud), 063 (durable tool continuation) define the boundaries but none prescribes the JSON acceptance rule — new decision.
- Size: M (one helper + wrappers) · ADR: new (acceptance rule shared by wire and storage) · Confidence: verified
- Pinning test: none for dup-key handling on either side (`rg -n '"a":1,"a"' Tests/` → none)
- Already covered: none

### P2 [D2] — `get_cli_setting` costs 12.6 ms per call because the cached read sits behind a per-call storage-admission handshake (~24 `posix.open` per read); this slice pays 3-6 reads per provider/summarization call
- Where (in-slice payers): `Summarization_General_Lib.py:841,851,905,907,919,929` (openai: 6 reads/call), `:1018,1072,1112,1114,1147` (anthropic: 5 reads, the last three INSIDE the retry loop — the excerpt's get_cli_setting_hot rows), and the same shape in every other `summarize_with_*`; `LLM_API_Calls.py:1139 _anthropic_caching_enabled` + `:1173 _cache_control_marker` (called 2-3× per Anthropic send), `:554 _openai_cache_key_enabled`. Cause (outside this slice): `config.py:8447 get_cli_setting` → `config.py:6253 load_cli_config_and_ensure_existence` → `Backup_Recovery/config_participants.py:400 wrapped` → `raw_participants.py:451 _scope` → `storage_admission.py:854 _acquire_storage` (0.279 s of 0.354 s for 20 calls) + `bootstrap.py:33 pinned_directory` / `Utils/private_paths.py:340 _open_verified_parent` (9,640 `posix.open` for 20 calls).
- Evidence: `$SCRATCH/llm_timing_probe.py` → `get_cli_setting(anthropic_api.api_retries): median=12600.4us max=26616.4us (3 reads/retry-attempt => ~37801us/attempt)`; cProfile of 20 calls (isolated env, warm cache) → top-of-stack lines listed above. Comparison: `recovery_review._ordinary_operation()+close` (the per-provider-call admission) = 5.8 ms median.
- Why it matters: the "config reads are cache-backed" known-deliberate holds for the DICT, not the cost — chunked summarization pays ~65-75 ms of config reads per chunk before any network I/O; an Anthropic Console send pays ~25-38 ms. The retry-loop placement (`:1112-1147`) itself is moot in-slice (the loop only re-enters after a 5 s sleep), so the three excerpt rows are P3 as placed; the cost is the wrapper.
- Recommended correction: outside this slice (ENTRY-config / Backup_Recovery reviewer): admit once per `load_cli_config_and_ensure_existence` cache generation, not per `get_cli_setting`. In-slice, hoisting the reads above the loop (S) is cosmetic until that lands.
- Size: S in-slice / M for the cause · ADR: ADR-126 governs startup admission; whether it intends per-read admission is the config reviewer's call · Confidence: verified (measurement, isolated HOME; production HOME not measured)
- Pinning test: none
- Already covered: not in the "already handled" list; cross-slice hand-off

### P3 [D4b] — `summarize_with_*` re-rolls the same "session + Retry adapter + post + stream_generator" block 16 times across two modules with behavioural drift (some close the response on abandon, most do not; some `raise_for_status`, some check status by hand; one hardcodes its URL)
- Where: `Summarization_General_Lib.py` openai `:904-967`, anthropic `:1109-1220`, cohere `:1381-1474`, groq `:1631-1681`, openrouter `:1813-1891`, huggingface `:2046-2097`, deepseek `:2236-2292`, mistral `:2424-2494`, google `:2646-2699`; `Local_Summarization_Lib.py` local_llm `:88-132`, llama `:400-452`, kobold `:645-718`, ooba `:917-971`, tabby `:1150-1214`, vllm `:1409-1456`, ollama `:1673-1752`, custom `:1945-1995`, custom_2 `:2204-2254`. Drift: `response.close()` in a `finally` only in openai `:964-965` and cohere `:1422-1427` ("sibling parity" claimed in the cohere comment, never propagated); `summarize_with_ollama:1756-1775` builds a second session+adapter it never uses; `summarize_with_local_llm:90` hardcodes `http://127.0.0.1:8080/v1/chat/completions` (the chat sibling reads `api_settings.local-llm.api_url`); `summarize_with_google:2667,2723` hardcodes `https://generativelanguage.googleapis.com/v1beta/openai/` (pinned by `test_summarization_diagnostic_privacy.py:5883`; see Left UNVERIFIED for whether that URL is even the chat-completions endpoint); `summarize_with_huggingface:2027` posts to the legacy `api-inference.huggingface.co/models/…` `inputs` API while `chat_with_huggingface` uses the router.
- Evidence: read only (line refs above); the two closing copies vs 14 non-closing verified by `rg -n "response.close()" LLM_Calls/Summarization_General_Lib.py LLM_Calls/Local_Summarization_Lib.py`.
- Why it matters: abandoned streaming consumers leak the response in 14 of 16 copies; each fix (retry policy, timeout, close) has to be applied 16 times and demonstrably was not (P2 finding above).
- Recommended correction: one `_post_with_retry(section, url, payload, *, streaming)` helper in `Summarization_General_Lib.py` (or `Utils/egress.py` if the chat handlers also adopt it) returning either the parsed body or a closing generator; L only if the diagnostic ledger re-attribution is in scope (task-17387 explains why).
- Size: M · ADR: no · Confidence: verified (structure), inferred (leak-on-abandon not exercised)
- Pinning test: `test_summarization_diagnostic_privacy.py` pins log call sites per function — any consolidation re-keys the ledger.
- Already covered: task-17387 notes the ledger constraint; no task covers the block itself.

### P3 [D4b] — `moonshot.py` and `zai.py` duplicate ~14 pure helpers verbatim (only the provider label in error strings differs)
- Where: `_resolve_api_key` (`moonshot.py:601-627` ≡ `zai.py:569-591`), `_resolve_base_url` (`:643-659` vs `:594-603`, moonshot adds `api_region`), `_positive_number`/`_nonnegative_number`/`_nonnegative_integer` (`:662-692` ≡ `:606-636`), `_normalize_messages` (`:695-753` ≡ `:646-709`; moonshot `deepcopy(safe)` at `:750`, zai none at `:706`), `_normalize_call_batch` (`:756-792` ≡ `:712-748`), `_normalize_tools` (`:795-826` ≡ `:751-781`), `_normalize_stop` (`:890-900` ≡ `:872-882`), `_normalize_response_format` (`:903-917` ≡ `:885-899`), `_json_shape_is_bounded` (`:920-948` ≡ `:913-941`), `_find_round_owner`/`_find_owner` (`:1040-1060` ≡ `:833-851`), `_apply_continuations` (`:951-1037` vs `:795-830`, moonshot has the k3 reasoning-replay branch). Policy that legitimately differs: `_DEFAULT_RETRY_DELAY` 1.0 vs 5.0, `_normalize_tool_choice` (moonshot auto/none/required/named vs zai "auto" only), finish policies, `thinking` payload.
- Evidence: read side by side; excerpt dup_verbatim row for `_json_shape_is_bounded` confirmed.
- Why it matters: ADR 062 rejected "Keep independent Moonshot and Z.ai implementations" precisely because it "continues duplicated lifecycle, retry, streaming, usage, and privacy defects" and moved transport/SSE/retry into `hosted_chat.py`; the validators listed here are mechanics, not the "provider-specific builders" the ADR keeps separate, so they drifted (`deepcopy`, `api_region`) without a test noticing.
- Recommended correction: move the pure validators and `_json_shape_is_bounded` into `hosted_chat.py` parameterised by provider label (canonical home per ADR 062); keep the builders per provider.
- Size: M · ADR: yes (062-hosted-chat-completions-provider-boundary.md — consistent with it, not against it) · Confidence: verified
- Pinning test: `Tests/LLM_Calls/test_moonshot*.py` / `test_zai*.py` exercise each copy independently (not inspected for names; targeted collect not run).
- Already covered: none

### P3 [D1] — Malformed-SSE warnings log the raw provider line regardless of `is_sensitive_llm_request()`
- Where: `LLM_API_Calls.py:2001-2003` (anthropic `f"...Could not decode JSON: {event_data_str}"`), `:2761-2763` (cohere), `:2750-2752` (cohere "Unexpected line format"), `:2855-2857` (cohere unknown event dumps `cohere_event`), `:3905-3907` (google), `:4789-4791` (huggingface). Contrast the request-side allowlisted summaries at `:818-831` etc. and the `task-2116` gating comments.
- Evidence: read only.
- Why it matters: a provider fragment (which can echo user/system text in error bodies) reaches the log on the sensitive/auxiliary paths that the request side deliberately protects.
- Recommended correction: log `len(line)` or `safe_llm_error_detail(line)` (`Utils/sensitive_llm_logging.py`, already imported) — S.
- Size: S · ADR: no · Confidence: verified (reading) — not exercised
- Pinning test: `Tests/Chat/test_sensitive_llm_logging.py` covers request payload logging, not these lines (not inspected in full).
- Already covered: none

### P3 [D3] — Assorted confirmed smells (one line each, all read-verified)
- `Summarization_General_Lib.py:78` — module-level `api_key = get_cli_setting("openai_api", "api_key", "")` at import: reads a secret into a module global that nothing reads (every use is a parameter/local; `rg` for importers of that name → none). Import side effect + one admission handshake at import.
- `Summarization_General_Lib.py:60-66` — `try/except ImportError` around the INTERNAL `Chunking.Chunk_Lib` import; `PYTHONPATH=$WT $PY -c "import tldw_chatbook.Chunking.Chunk_Lib"` → OK, so the guard only hides a future real ImportError as "chunking unavailable" and silently degrades recursive/chunked summarization to direct.
- `Summarization_General_Lib.py:29,47` — loguru `logger` (used once, `:65`) and stdlib `logging` (via `Logging_Config`) in the same module; `Local_Summarization_Lib.py:33` and `LLM_API_Calls_Local.py:26` import stdlib `logging` re-exported through `Utils/Utils.py:35` (private re-export across packages).
- `Summarization_General_Lib.py:1744-1745` — function-body `import requests, json` already imported at module scope; `:1771` and `:1999` raise `"No valid Anthropic API key available"` from the OpenRouter and HuggingFace summarizers (copy-paste); `:1773` `logging.error("OpenRouter: Error in processing: {str(e)}")` missing `f` (logs the literal template — the task-2116 bug class); `:1810-1891` openrouter "streaming" consumes the whole stream and returns a `str`.
- `Summarization_General_Lib.py:2834-2886 summarize_chunk` — passes `analyze(text, custom_prompt_input, api_name, …)` positionally into `analyze(api_name, input_data, custom_prompt_arg, …)`: verified `_dispatch_to_api` receives `api_name='the input text'`; zero callers (`rg -n "summarize_chunk\b" tldw_chatbook --include='*.py'` → only its own lines). Dead AND broken — delete.
- `LLM_API_Calls_Local.py:2037-2041, 2057-2061` — `chat_with_custom_openai` raises `provider="ollama"` / "Ollama API URL … required" for the Custom OpenAI provider (user-visible wrong-provider copy); `:2337` `provider_name=cfg_section.capitalize()` → "Custom_openai_api_2" in every log/error line.
- `LLM_API_Calls.py:4735-4742` — huggingface streaming uses bare `requests.post` (explicit `timeout` + `verify=requests_verify()`, so no D1) instead of `create_default_session()` like the other 12 call sites; `Summarization_General_Lib.py:1131` is the other bare `requests.post` (covered by the P2 above).
- `LLM_API_Calls.py:1440, 3543` — per-call `from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY` inside the two hottest handlers; `recovery_review.py:462-469 _identity` does `import asyncio` per provider call (cost is a `sys.modules` hit; the excerpt's seed_name__identity row). `qwencloud.py:1198` lazy-imports `qwencloud_streaming` per streaming call (no cycle: the reverse import is `TYPE_CHECKING`-only).
- Five re-rolls of the 4-line "best-effort close" helper: `hosted_chat.py:824-828`, `hosted_chat.py:204-209`, `hosted_chat_streaming.py:223-227`, `qwencloud.py:77-82`, `qwencloud_streaming.py:53-58` (all the excerpt's `except_exception_pass` rows) — one helper in `hosted_chat.py` would do.
- `LLM_API_Calls_Local.py`, `Summarization_General_Lib.py`, every hosted handler: a new `requests.Session` + `HTTPAdapter` per call (no pool reuse; N TLS handshakes for N chunks). Not measured — see Left UNVERIFIED.

## Credential-precedence table (brief item 1; ADR 012 / CLAUDE.md "~9 bridged providers")
| handler | precedence actually implemented | placeholder check (`resolve_provider_api_key`) | key can reach a log? |
|---|---|---|---|
| `chat_with_openai` `:624-649` | explicit → `api_settings.openai.{api_key,api_base_url}` overlaid on legacy `openai_api` (loader-bridged) → legacy resolved value; recovered-settings path when `recovery_review` history exists | not in handler; relies on loader bridge (`config.py:1573`). An `api_settings.openai.api_key` placeholder typed by a user is overlaid at `:640-642` unchecked — inferred edge | no (`safe_llm_request_payload_summary`; `Chat_Functions` logs `'api_key': 'key_hidden'`) |
| `chat_with_anthropic` `:1376-1407` | explicit → legacy `anthropic_api.api_key` (bridged) ; `auth_source=claude_subscription` from `api_settings.anthropic` replaces it with the borrowed OAuth token | loader only | no; `SubscriptionCredential.__repr__` masks the token |
| `chat_with_cohere` `:2460-2470`, `deepseek` `:3173-3181`, `groq` `:4230-4236`, `mistral` `:5006-5014`, `openrouter` `:5257-5266` | explicit → `api_settings.<p>.api_key` from `get_runtime_config_snapshot()` (bridged) | loader only | no |
| `chat_with_google` `:3493-3522` | explicit → `resolve_provider_api_key(api_settings.google.api_key)`; NO env fallback (the known open case) | yes, in handler | no; 3xx `Location` is logged (`:3762-3767`, reviewed per its comment) |
| `chat_with_huggingface` `:4498-4513` | explicit → legacy `huggingface_api.api_key` or `[API].huggingface`; key optional | loader only | no (`redacted_headers` `:4718-4722`) |
| `chat_with_moonshot` (`moonshot.py:601-627`) / `chat_with_zai` (`zai.py:569-591`) | explicit(resolve) → `api_settings.<p>.api_key` (resolve; a placeholder is a HARD `ChatConfigurationError`, no fall-through) → env `api_key_env_var` → default env. Not in the 9-provider `[API]` bridge (no legacy key) | yes, at every tier | no (`api_key` fields are `repr=False`) |
| `chat_with_qwencloud` (`qwencloud.py:359-402`) | explicit(resolve) → settings.api_key (resolve; placeholder falls through) → env | yes | no |
| local providers (`LLM_API_Calls_Local.py`) | explicit / `api_key_resolved` → `cfg.api_key`; `chat_with_custom_openai` uses `provider_readiness.resolve_provider_credential` | no (placeholder would go on the wire as Bearer to a local server) | no |
| `summarize_with_*` | explicit → legacy `<p>_api.api_key` via `get_cli_setting` (bridged for the 9); locals: modern table → legacy → env-var-by-name (`_resolve_provider_credential`) | no in-handler check | no ("Credential configured" lines carry no value) |
Net: precedence differs across three generations of handler (legacy-dict, snapshot-table, strict-adapter) but the loader bridge makes the 9 bridged providers consistent for CONFIG-sourced keys; only google/moonshot/zai/qwencloud apply the placeholder rule in-handler; no handler in this slice writes a key to a log line that I could find.

## Streaming inventory (brief item 5)
Every provider offers a non-streaming branch: openai `:932`, anthropic `:2025`, cohere `:2904`, deepseek `:3316`, google `:3924`, groq `:4376`, huggingface `:4806`, mistral `:5141`, openrouter `:5395`, moonshot/zai via `hosted_chat_request(streaming=False)`, qwencloud `:1210-1215`, every local handler via `_chat_with_openai_compatible_local_server:372`, kobold forces non-streaming (`:1007-1012`). Silent chunk drops: `_responses_stream_to_chat_sse:309-312` (`except JSONDecodeError: continue`, no log) — P3; the usage drops are the P1 above.

## Candidate dispositions
| candidate (file:line pattern) | confirmed / retired (why) / unverified (check) |
|---|---|
| dup_shape `terminal_turn@hosted_chat.py:193` (+6 shapes in Chat/UI) | retired — a 3-line "guard then return/raise" property; the six siblings are unrelated guards that match on shape only, no shared behaviour to extract |
| dup_verbatim `_json_shape_is_bounded` moonshot:920 ≡ zai:913 | confirmed — part of the strict-JSON family finding (P2) and the moonshot/zai helper-duplication finding (P3) |
| except_exception_pass hosted_chat:208, :827, hosted_chat_streaming:226, qwencloud:81, qwencloud_streaming:57 | retired as D1 (all five swallow only a `close()` failure, not a data-path error); confirmed as a 5-copy D4b helper (P3 list) |
| except_exception_return Summarization_General_Lib (1) | retired — `:275` `json.dumps` fallback to `str()` on a dict, no data lost |
| except_exception_return pricing_catalog (1) | retired — `:423-430` documented models.dev gap-fill returning None ("never breaks a lookup") |
| except_exception_return qwencloud_streaming (4) | retired — `:984-998, :1006` convert to sentinels that the caller turns into `ChatProviderError("malformed streaming data")`; nothing swallowed |
| function_body_import LLM_API_Calls (3) | confirmed P3 — `:1366` anthropic_subscription (lazy, fine), `:1440`/`:3543` per-call `EPHEMERAL_ORIGIN_KEY` import in the two hottest handlers |
| function_body_import LLM_API_Calls_Local (2) | retired — `:808` documented deferral to keep gateway deps off startup; `:2047` readiness helper, same reason |
| function_body_import Summarization_General_Lib (2) | confirmed P3 — `:1744-1745` redundant `import requests, json` |
| function_body_import pricing_catalog (1) | retired — `:584` lazy config import inside a once-per-process constructor |
| function_body_import recovery_review (4) | confirmed P3 for `:463 import asyncio` in per-call `_identity`; the other three are one-shot catalog/discovery paths |
| get_cli_setting_hot Summarization_General_Lib:1112, :1114, :1147 (summarize_with_anthropic loop) | confirmed but re-scoped: 3 × 12.6 ms = 38 ms per attempt (measured); in-slice P3 (loop re-enters only after a 5 s sleep); the cost is the admission wrapper — P2 cross-slice finding above |
| inline_truncate Summarization_General_Lib:2791 | retired — test-only mock summarizer |
| legacy_markers_per_file (9 files) | not examined — comment markers, no code claim attached |
| raw_1024x1024 (11 sites) | retired — named module constants (`_MAX_*`) for stream/JSON ceilings, not in-body magic numbers |
| re_compile_in_def pricing_catalog:644 `_compile_patterns` | retired — runs once per `PricingCatalog` instance; the only constructor is the lazy `get_pricing_catalog()` global (`rg -n "PricingCatalog(" tldw_chatbook` → none outside the module); `get_pricing` (per token count) uses the precompiled list |
| seed_name__identity recovery_review:462 | confirmed P3 — per-call `import asyncio` |
| seed_name__reject_json_constant qwencloud_streaming:61, hosted_chat:633 | confirmed — strict-JSON family (P2) |
| seed_name__strict_json_loads qwencloud_streaming:65, hosted_chat:673 | confirmed — strict-JSON family (P2) |
| strftime Summarization_General_Lib:2804 | retired — mock summarizer timestamp string, not a DB/SQL format |
| try_import_guard Local_Summarization_Lib:243 (summarize_with_llama, handlers=Exception) | retired — mis-detected: `:243` is the function's outer `try:`, no import inside it |
| try_import_guard Summarization_General_Lib:60 (module, ImportError) | confirmed P3 — internal import guard; `Chunk_Lib` imports cleanly in the isolated env |
| try_import_guard pricing_catalog:423 (`_models_dev_pricing`) | retired — documented optional gap-fill; failure → None by design |

## Verified-fine
- `Utils/egress.create_default_session()` (`:1012-1046`) returns a `DefaultTimeoutSession` + TLS trust; the per-handler `session.mount("https://", HTTPAdapter(Retry…))` calls (13 sites) replace only the default transport adapter — no SSRF/guard adapter is displaced (egress has none on this path). Every hosted/local call in the slice except `LLM_API_Calls.py:4735` and `Summarization_General_Lib.py:1131` goes through it; no `httpx` in `LLM_Calls/` (`rg -n 'httpx\.' LLM_Calls` → none; realtime uses `websockets` via `optional_deps.require_dependency`).
- Timeouts: every `session.post` either passes `timeout=` or inherits `DefaultTimeoutSession`'s config default — no infinite-wait call found (`rg -n "requests.post\(" LLM_Calls` → the two bare sites both pass `timeout=`).
- `time.sleep` sites: `hosted_chat.py:581,612`, `qwencloud.py:1291`, `Summarization_General_Lib.py:1252,1269` (retry backoff) and `:2808` (mock) — all reached from `chat_api_call`/`analyze`, which the Console runs under `asyncio.to_thread` (`console_provider_gateway.py:6337`) and ingestion runs on workers; no event-loop-path sleep found in this slice (callers outside the slice not audited).
- `recovery_review.unqualified` per-call admission: 5.8 ms median (measured) — small next to a network round trip; noted only because it multiplies with the config-read cost above.
- `realtime/` package: lazy `websockets` via `optional_deps`, `_safe_invoke` isolates callbacks, `close()` bounded at 2 s, `_enqueue` thread-safe via `call_soon_threadsafe`; no issues found.
- `pricing_catalog.py`: patterns compiled once; `cost_for_usage` arithmetic and the cache-attribution ordering are documented; `_lower_keys` normalisation applied to both seed and config.
- `anthropic_subscription.py`: Keychain memo (5 s TTL, lock), read-only, token masked in repr; `read_claude_code_credential` never raises outward.
- `hosted_chat.py` / `qwencloud*.py` / `moonshot.py` / `zai.py`: fail-closed validators, bounded SSE decoder (`SSERecordDecoder` caps), owned response/session lifetime (`OwnedSSEStream.close`, `QwenCloudStream.close`), no key in any log line, `Retry` transport disabled in favour of an explicit loop (the uncapped `Retry-After` is the one gap, P2 above).
- `LLM_API_Calls_Local.py:2219-2221` reading top-level `custom_openai_api_2`: the loader DOES build that section (`config.py:2616`), so it is not the task-625 "section that can never exist" class — retired as a candidate I raised.
- `chat_with_anthropic` / `chat_with_google` refuse 3xx with credentials in custom headers (`:1758,1802-1822`, `:3759-3776`) — correct and pinned per their comments.

## Retired
- "`get_cli_setting` re-read inside the retry loop is a hot-path cost" (excerpt rows 1112/1114/1147): symptom real (12.6 ms/read measured), cause wrong — the placement is not the problem (the loop body runs once unless a 500/RequestException already cost a 5 s sleep); the cost is the per-read storage-admission wrapper in `config.py`/`Backup_Recovery`, reported as a cross-slice P2 instead.
- "`pricing_catalog._compile_patterns` compiles per call" — retired with evidence (single lazy constructor; precompiled list used per lookup).
- "`chat_with_custom_openai_2` reads a config section the loader never builds" (my own candidate from the task-625 comment): retired — `config.py:2616` builds `custom_openai_api_2`.
- "`except Exception: pass` in the hosted/qwen streams drops chunks" — retired: all five are `close()` guards; malformed data raises `ChatProviderError`/`HostedChatProtocolError` instead.
- "`summarize_chunk` argument-order bug is user-reachable" — the bug is verified but no caller exists; downgraded to the dead-helper P3 list.
- "deepseek double-yield is a live P1" — verified bug, but every `analyze(` caller found passes/defaults `streaming=False`; kept at P2 as latent.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| `summarize_with_google` posts to `https://generativelanguage.googleapis.com/v1beta/openai/` (base path, no `/chat/completions`) and therefore 404s on the live API; the diagnostic-privacy test pins that exact URL (`Tests/LLM_Calls/test_summarization_diagnostic_privacy.py:5883`, inside `test_google_success_hides_credential_input_prompt_and_response`) | needs a network call (forbidden); the pinning test asserts the URL as current behaviour, so a change is a decision, not a fix | `cd $WT && source $SCRATCH/env.sh && GOOGLE_API_KEY=… $PY -c "import requests; print(requests.post('https://generativelanguage.googleapis.com/v1beta/openai/', json={}, timeout=10).status_code)"` (expect 404 vs 4xx-with-body on `/chat/completions`) |
| `summarize_with_huggingface` targets the retired `api-inference.huggingface.co/models/<id>` `inputs` API and cannot succeed against current HF | network | same shape as above against `https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2` |
| deepseek/groq/mistral/openrouter streams omit usage because `stream_options.include_usage` is never requested (`rg -n include_usage LLM_Calls/` → openai, moonshot, qwencloud only) | provider default behaviour needs a live stream | `rg -n "include_usage" LLM_Calls/LLM_API_Calls.py` (shows the gap) + one live streamed request per provider inspecting the final chunk for `usage` |
| Per-call `requests.Session`/`HTTPAdapter` construction (no pool reuse) is a measurable cost for N-chunk summarization | not timed | `$PY - <<EOF` timing 50 × `create_default_session(); session.mount(...); session.close()` vs one reused session against a local `http.server` |
| Leak-on-abandon in the 14 summarization stream generators lacking `response.close()` | not exercised | mock `iter_lines`, `next(gen)`, `del gen`, `gc.collect()`, assert `response.close.called` per generator |
| The `_maybe_record_usage` consumer produces an EMPTY ledger row (vs an estimate) for a streamed Gemini turn in the real Console | gateway not run | `cd $WT && source $SCRATCH/env.sh && $PY -m pytest Tests/Chat/test_console_provider_gateway.py -q` after adding a case that streams a `usageMetadata` chunk through `_stream_generic_chat` and asserts `signals.usage_snapshot()` |
| `get_cli_setting` costs 12.6 ms against a REAL profile (measured only in the isolated scratch HOME) | brief forbids the real profile (ADR-126 recovery gate) | rerun `$SCRATCH/llm_timing_probe.py` without `env.sh` on a machine whose profile is not in recovery |
| `chat_with_openai` overlays an unchecked `api_settings.openai.api_key` placeholder (`:640-642`) onto the wire | needs a config with `api_settings.openai.api_key = "<API_KEY_HERE>"` and a mocked post | `$PY -` with `load_settings` patched to that table, patch `requests.Session.post`, call `chat_with_openai(...)`, inspect `headers["Authorization"]` |
