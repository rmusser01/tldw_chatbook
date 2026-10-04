# PR #3017 cross-provider response audit

Date: 2026-10-04. Reviewed head: `16e5f0270e1ee02ed807abde940b25843a5b4c1d`.

## Result

The DeepSeek parser failure is also present in five other providers. Offline fixtures passed through `Chat_Functions.chat_api_call`, real provider adapters/generic handlers, HTTP boundary and strict shared parser: **10 expected-success cases failed** (five providers × complete/SSE); **six controls passed** (Moonshot, Mistral and ZAI × complete/SSE). HTTP alone was mocked. These are contract reproductions, not live calls or comprehensive provider qualifications.

| Provider | Exact probe additions | Result in both modes | Follow-up |
| --- | --- | --- | --- |
| Groq | envelope `x_groq: {id: "fixture"}`, choice `logprobs: null` | malformed response/event | TASK-34367.1 |
| OpenRouter | choice `native_finish_reason: "stop"` | malformed choice | TASK-34367.2 |
| Together | choice `logprobs: null` | malformed choice | TASK-34367.3 |
| Fireworks | choice `logprobs: null` | malformed choice | TASK-34367.4 |
| Cerebras | envelope `time_info: {total_time: 0.1}`, choice `logprobs: null` | malformed response/event | TASK-34367.5 |

All fixtures supply a real-shaped assistant reply, index 0, terminal stop and token usage. Complete failures are exposed as `ChatProviderError`; streamed failures originate as `HostedChatProtocolError` during iteration. Existing Console retry settlement is shared across providers, so PR #3017 repairs the retry deadlock for their known no-response failures too. It does not repair their response contracts.

## Official evidence and boundaries

- [Groq API reference](https://console.groq.com/docs/api-reference) includes `x_groq` and null `logprobs` in its ordinary response example. [Official stream SDK schema](https://raw.githubusercontent.com/groq/groq-python/main/src/groq/types/chat/chat_completion_chunk.py) defines nested `x_groq.usage` and `x_groq.error`. Ignoring the entire field can lose accounting or mask terminal errors; nonstream `x_groq.usage` has different hardware-cache meaning. Provider-specific normalization requires dedicated regressions.
- [OpenRouter overview](https://openrouter.ai/docs/api_reference/overview) explicitly specifies `native_finish_reason` for both body and stream choices. Its [current endpoint reference](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion) also documents optional `openrouter_metadata` and `service_tier`. Audit these alongside provider/logprobs/reasoning annotations rather than accepting arbitrary fields.
- [Together reference](https://docs.together.ai/reference/chat-completions) specifies additional choice annotations including logprobs and seed, message reasoning and envelope prompt/warnings.
- [Fireworks reference](https://docs.fireworks.ai/api-reference/post-chatcompletions) specifies logprobs; streaming metrics documentation also describes terminal perf_metrics.
- [Cerebras reference](https://inference-docs.cerebras.ai/api-reference/chat-completions) includes time_info and logprobs in its ordinary example, with further service-tier/reasoning fields in the schema.
- [Moonshot reference](https://platform.kimi.ai/docs/api/chat), [Mistral reference](https://docs.mistral.ai/api/endpoint/chat), and [ZAI reference](https://docs.z.ai/api-reference/llm/chat-completion) baseline text/usage fixtures pass. Moonshot documents optional logprobs, but its sample does not show them emitted by default: optional response-shape reconciliation remains unqualified.

Generic presets already have level-specific allowance support under ADR-179. Together/Fireworks/Cerebras currently intentionally pin empty provisional sets in `test_inference_cloud_presets.py`; those tests do not prove live compatibility. Registry inspection found additional presets with empty choice allowances. No provider-wide compatibility claim is made without an official/captured envelope and actual-adapter replay. OpenAI/Anthropic/Google and local adapters use other parser paths; the specific hosted-parser defect is not demonstrated there.

## Reproduction

`adapter_probe.py.txt` preserves the exact temporary pytest module used. In an isolated checkout, copy it to `Tests/LLM_Calls/test_pr3017_provider_audit_probe.py`, run that file with Python 3.12 pytest and the repository bootstrap-profile fixture, then remove the temporary module. It is intentionally archived outside normal test collection because it exposes unresolved follow-up bugs. Command used:

```sh
python -m pytest Tests/LLM_Calls/test_pr3017_provider_audit_probe.py -q --tb=short --show-capture=no --basetemp=/private/tmp/pr3017-provider-probes
```

Recorded outcome: `10 failed, 6 passed in 0.86s`. No credentials or live paid requests were used.

## Merge review

Qodo's changed-target retry finding was reproduced in 14 cases across both wire formats, then repaired at the final durable-header transaction. Provider/model/endpoint/generation/response/reasoning now match the preceding failed call; tool schemas and literal envelope components also match. Route AGENT_FIRST → TOOL_LOOP remains valid. All 40 ownership cases pass. Import grouping finding fixed. Independent reviewer found no remaining Critical/Important issue. Six affected PR modules pass **58 tests**; no full-suite claim.

Broader targeted trace runtime/service/system-prompt run: **160 passed, two failed**. Both failing cases are `test_failed_rollback_releases_observers_and_reports_ambiguous_outcome[False/True]`: the unchanged test constructs a transaction manager and exits without entering, so `_maintenance_context` is absent and masks its intended rollback/commit error. The same two failures reproduced after temporarily restoring trace_service from origin/dev; the DB and test files are byte-identical to origin/dev. The repaired service was restored afterward. These pre-existing test-fixture failures were not silently excluded or repaired in this provider PR.
