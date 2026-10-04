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

## TASK-34367.1-.5 reconciliation (2026-10-04)

The five known failures are repaired in the combined Console-fixes worktree.
The preserved audit fixture became the permanent
`Tests/LLM_Calls/test_documented_provider_response_contracts.py`: original RED
was **10 failed, 6 passed**, and the final affected-provider run is **409 passed**
across nine modules. Only `requests.Session.post` is replaced; dispatch,
provider adapters/generic handlers, transport ownership, SSE framing, strict
normalization, public answer and terminal accounting are real. This remains
an offline primary-schema qualification, not a paid/live capture receipt.

Current primary references were re-read on 2026-10-04. Provider-scoped changes:

- **Groq:** nullable/object choice logprobs and ordinary x_groq/service_tier
  metadata; complete x_groq requires its request ID and cache usage remains
  distinct from top-level accounting. SSE x_groq.usage (including timings and
  prompt/completion token details) is promoted and retained. Matching duplicated
  envelope usage is accepted; conflicting/malformed usage and nested error
  strings fail closed with redacted errors. Stream-only obfuscation padding is
  admitted as an annotation. Sources: [API reference](https://console.groq.com/docs/api-reference),
  [complete schema](https://raw.githubusercontent.com/groq/groq-python/main/src/groq/types/chat/chat_completion.py),
  [chunk schema](https://raw.githubusercontent.com/groq/groq-python/main/src/groq/types/chat/chat_completion_chunk.py),
  [usage schema](https://raw.githubusercontent.com/groq/groq-python/main/src/groq/types/completion_usage.py).
- **OpenRouter:** native_finish_reason/logprobs, provider/service_tier and
  openrouter_metadata are scoped annotations. A documented content-free final
  usage choice repeating the already accepted finish is converted to the
  neutral accounting frame. Changed finish/native finish/index/role, new
  text/tool/reasoning deltas, unknown fields, duplicate accounting and error
  frames are rejected. Native finish strings remain upstream-defined.
  Sources: [overview](https://openrouter.ai/docs/api_reference/overview),
  [endpoint](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion),
  [streaming](https://openrouter.ai/docs/api_reference/streaming),
  [errors](https://openrouter.ai/docs/api_reference/errors-and-debugging).
- **Together:** choice logprobs/seed/top_logprobs/text, message reasoning and
  envelope prompt/warnings use existing level-scoped allowances. Prompt is a
  documented array, correcting the older provisional comment's string guess.
  Source: [chat schema](https://docs.together.ai/reference/chat-completions).
- **Fireworks:** choice logprobs/raw_output and envelope perf_metrics/
  prompt_token_ids are scoped. The SSE fixture puts perf_metrics on the terminal
  chunk with usage; metrics do not authorize a missing finish or error success.
  Source: [chat schema and streaming metric contract](https://docs.fireworks.ai/api-reference/post-chatcompletions).
- **Cerebras:** time_info/service_tier/service_tier_used, choice logprobs/
  reasoning_logprobs and message reasoning are scoped.
  Source: [chat schema](https://inference-docs.cerebras.ai/api-reference/chat-completions).

Existing allowance filtering remains annotation-only; answer/usage/tool/finish
fields are not dropped. Existing ignored reasoning stays ignored. OpenRouter's
nonempty structured reasoning_details requires a separately governed durable
block contract and remains fail-closed (null/empty annotation accepted); it is
not silently discarded or claimed supported. Likewise opt-in nonempty
Fireworks choice token_ids remain unsupported by the existing level-value
contract. Neither vendor-hosted execution nor arbitrary annotation arrays are
newly enabled. [OpenRouter reasoning shapes](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
were audited to establish that boundary.

A narrow optional call-owned wire normalizer, governed by the pre-code
[ADR-179 amendment](../../../../backlog/decisions/179-generic-hosted-provider-engine-and-preset-registry.md),
handles Groq/OpenRouter semantics between bounded JSON loading and unchanged
strict parsing. Other providers supply no normalizer. Eighty both-mode
unknown/required-field negatives, provider-scope controls, baseline
Moonshot/Mistral/ZAI contracts and existing DeepSeek null-logprobs checks pass.
Provisional empty-set assertions for the three generic presets were removed
in favor of actual-handler behavioral fixtures.

Verification: Python 3.12 on Windows, private Tests bootstrap profile and
private TEMP/TMP; nine affected modules only, **409 passed, one existing
Pydantic deprecation warning**. Ruff check passes with E721 ignored for the
existing deliberate exact-type checks that reject booleans; adapter/test Ruff
format checks and scoped git diff whitespace checks pass. No full suite,
credentials, paid requests, main-checkout edits or child commits were used.
All five tasks remain In Progress pending the combined independent review/PR.

### Merged dev qualification (2026-10-04)

The provider fixes in `b4ed4f91df` were qualified after merging current
`origin/dev` (`49206beea9`) as `2a3d59baa2`. The automatic merge duplicated
Together's `choice_allowances` keyword and failed compilation. Removing only
the narrower duplicate retained the documented allowance superset and the
live-proven `stream_include_usage=True`, with `stream_usage_optional=False`.
The target's [Together capture](../../../../Tests/fixtures/cloud_live/together.json)
(captured 2026-10-04T17:56:04Z), null streamed logprobs, trailing accounting,
and bare-array model listing were preserved. No new live request was made.

The same nine provider modules plus `test_doc_derived_presets.py`,
`test_live_capture_replay.py`, and
`test_openai_compatible_model_discovery.py` passed: **644 passed, one existing
Pydantic warning in 44.03 seconds**. This includes the retained live capture
replay/listing and oversized model-template regression controls. Scoped Ruff
checks, four adapter/test formatting checks, and scoped diff whitespace
checks passed. Evidence log:
`C:/Users/GDesktop-1/.config/tldw-console-pause-34402/provider-merged-qualification.txt`.
The qualified `provider_registry.py` SHA256 is
`5C62B6DD78D3B299180B84E36BBE11A7A1C7711D8864A9B8240976A9C6137D80`;
Groq, OpenRouter, shared parser, and documented-contract test source are
unchanged from the preceding qualification.
