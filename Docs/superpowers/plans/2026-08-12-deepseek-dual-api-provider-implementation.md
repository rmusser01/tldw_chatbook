# DeepSeek Dual-API Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep DeepSeek as one ordinary provider while adding an explicit `api_mode` that defaults to Chat Completions and optionally uses the new stateless Responses API, with strict native tools, durable later-turn reasoning replay, Settings, discovery, cancellation, and usage.

**Architecture:** Replace the legacy DeepSeek function body with a dedicated adapter that resolves one frozen provider/model/base/key/mode/retry snapshot. Both modes use TASK-15676's route-neutral owned HTTP lifecycle; Chat mode additionally uses its generic Chat normalizer, while Responses mode wraps the same owned response/SSE stream with a DeepSeek-specific input translator and semantic normalizer keyed by SSE `event:` plus JSON `type`. Both modes emit the same typed model-turn/usage/checkpoint contract into the existing gateway/runtime and TASK-15675 message owner.

**Tech Stack:** Python 3.11+, existing hosted Chat `requests` transport, strict incremental SSE/JSON parsing, Textual 8.x Settings, Console gateway/native-agent runtime, TASK-15675 continuation persistence/sync/export, ADR-020 model catalog, pytest/pytest-asyncio, and optional isolated live tests.

---

## Design Sources And ADR Check

- Approved design: `Docs/superpowers/specs/2026-08-12-deepseek-dual-api-provider-design.md`
- Canonical decisions: `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md` and `backlog/decisions/064-deepseek-dual-api-provider-boundary.md`
- Dependencies: merged TASK-15675 and TASK-15676 implementations.
- Backlog source of truth: `backlog/tasks/task-15677 - Add-DeepSeek-dual-API-provider-support.md`
- Official sources linked from the approved design: DeepSeek Responses, create-response, Chat Completion, thinking, tools, models/pricing.

ADR required: yes

ADR paths: `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md`, `backlog/decisions/064-deepseek-dual-api-provider-boundary.md`

Reason: ADR-063 owns the durable private-history boundary and ADR-064 owns the one-provider/two-wire-mode decision and semantic Responses contract. No new ADR is needed.

## Scope Guardrails

- Preserve one stable `deepseek` identity, credential owner, model catalog, dispatcher key, agent loop, and tool executor.
- Default missing `api_mode` to `chat_completions`; accept only exact `chat_completions` and `responses` values. Do not change QwenCloud's Responses default.
- Support existing Chatbook function tools only. Reject DeepSeek web search, custom `apply_patch`, provider file/code/computer tools, and non-function output items.
- Responses remains stateless and explicit-history: no `previous_response_id`, `conversation`, `store`, background, prompt templates, or provider continuation IDs.
- Do not fork or weaken TASK-15676's route-neutral HTTP lifecycle. DeepSeek-specific Chat and Responses semantics stay in `deepseek.py`/`deepseek_streaming.py`.
- Do not add another schema, checkpoint format, approval path, or context budgeter.
- Preserve explicit historical models. Fresh/missing default remains `deepseek-v4-flash`; `deepseek-v4-pro` remains available.
- Canonical Settings only; no legacy Settings surfaces. Default tests make no paid request.

## Branch And Baseline Discipline

- [ ] Begin only after TASK-15675 and TASK-15676 are merged. Create a fresh `codex/deepseek-dual-api` branch/worktree from current `origin/dev`.
- [ ] Put TASK-15677 In Progress before production changes, then add a structured Implementation Plan section to its task file that links this document and ADR-063/064; do not replace it with a one-line CLI plan:

  ```bash
  backlog task edit 15677 -s "In Progress"
  # Use apply_patch to add the ordered plan and ADR required/path/reason block.
  backlog task 15677 --plain
  ```

- [ ] Record clean-base results for DeepSeek mocked calls, hosted Chat/Qwen/Kimi/Z.ai, gateway/native tools, continuation, Settings, and catalog. Rerun localhost suites outside the socket sandbox when required.
- [ ] Every new behavior starts RED at the actual boundary. Use verbatim official-shaped JSON/SSE fixtures and a request-validating loopback server; do not substitute an OpenAI/Qwen adapter or patch away gateway/runtime seams.

## Provider Interfaces To Implement

```python
# tldw_chatbook/LLM_Calls/deepseek.py
DeepSeekAPIMode = Literal["chat_completions", "responses"]
DeepSeekReasoningEffort = Literal["provider_default", "low", "high", "max"]

def normalize_deepseek_api_mode(
    value: object,
    *,
    provider_settings: Mapping[str, Any] | None = None,
) -> DeepSeekAPIMode: ...

def resolve_deepseek_request(
    *,
    explicit_api_key: object = None,
    explicit_base_url: object = None,
    explicit_model: object = None,
    explicit_api_mode: object = None,
    app_config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> DeepSeekResolution: ...

def build_deepseek_payload(
    *,
    resolution: DeepSeekResolution,
    messages_payload: Sequence[Mapping[str, Any]],
    system_message: object = None,
    streaming: object = False,
    tools: object = None,
    tool_choice: object = None,
    reasoning_effort: object = None,
    provider_continuations: Sequence[ProviderContinuationCheckpoint] = (),
    **generic: object,
) -> dict[str, Any]: ...

def chat_with_deepseek(...) -> dict[str, Any] | Iterator[dict[str, Any]]: ...
```

```python
# tldw_chatbook/LLM_Calls/deepseek_streaming.py
class DeepSeekResponsesTranslator:
    def feed(self, *, event: str, data: Mapping[str, Any]) -> tuple[dict[str, Any], ...]: ...
    def finish(self) -> tuple[dict[str, Any], ...]: ...
```

### Task 1: Resolve One DeepSeek Provider, Mode, Base, Key, And Policy

**Files:**

- Create: `tldw_chatbook/LLM_Calls/deepseek.py`
- Create: `Tests/LLM_Calls/test_deepseek.py`
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`
- Modify: `Tests/Chat/test_chat_unit_mocked_APIs.py`

- [ ] **Cycle 1A — mode RED:** add exact default/config/explicit tests: absent→`chat_completions`; explicit beats canonical; valid higher priority avoids reading malformed lower priority; empty/non-string/unknown present values raise provider-labelled configuration errors. Assert Qwen/unrelated settings cannot influence DeepSeek.
- [ ] **Cycle 1B — resolution RED:** exact canonical table, configured key, configured env name, default `DEEPSEEK_API_KEY`, model, normalized base, timeout, retries, retry delay, and streaming precedence. Malformed exact table, placeholders, blank model/key, invalid numeric fields, and unsafe base fail before network and before unnecessary lower-priority reads. Input mappings and environment remain unchanged.
- [ ] Reuse TASK-15676's pure structural base normalizer with a small option/set for the exact two allowed terminal suffixes, or add a provider-local wrapper if changing the shared helper would affect other callers. Strip one lowercase `/chat/completions` or `/responses`, with or without one trailing slash; reject stacked/repeated/case/lookalike/encoded structural tails.
- [ ] **Cycle 1C — public compatibility RED:** preserve the existing `chat_with_deepseek` signature and dispatcher import while adding `api_mode`. Replace the legacy body in `LLM_API_Calls.py` with a compatibility import/wrapper; raw requests/urllib3 exceptions and response bodies may no longer escape.
- [ ] Existing model-default characterization (`deepseek-v4-flash`, `deepseek-v4-pro`) may already be GREEN; keep it as a no-drift guard rather than manufacturing a false RED.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_deepseek.py -k "mode or resolution or public or url"
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_chat_unit_mocked_APIs.py -k deepseek
  ```

- [ ] Commit:

  ```bash
  git add tldw_chatbook/LLM_Calls/deepseek.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/LLM_Calls/test_deepseek.py Tests/Chat/test_chat_unit_mocked_APIs.py
  git commit -m "feat(deepseek): add frozen dual-mode resolution"
  ```

### Task 2: Implement Exact Chat And Responses Request Builders

**Files:**

- Modify: `tldw_chatbook/LLM_Calls/deepseek.py`
- Modify: `Tests/LLM_Calls/test_deepseek.py`

- [ ] **Cycle 2A — canonical history RED:** deep-copy/validate system, user, assistant, and tool rows; exact system ownership; bounded string content; complete assistant reasoning/calls; call IDs unique across outbound history; one paired result before another conversational turn; canonical call order even when internal result rows arrive out of order. Orphans, duplicates, missing/cross-batch results, malformed arguments, and mode/provider/model/base mismatch fail before I/O.
- [ ] **Cycle 2B — function tools RED:** accept exact top-level `{type,function}` only, shared safe names/descriptions/bounded object schema, and unique names. Reject private metadata and every non-function/vendor tool. In thinking/tool mode accept generic choice absent/default or `auto` only but omit wire `tool_choice`; reject none/required/forced objects. Omit `parallel_tool_calls` and `max_tool_calls`.
- [ ] **Cycle 2C — thinking RED:** expose `provider_default|low|high|max`; reject `minimal|medium|xhigh` and nonstrings. Provider default omits effort; others enable thinking with exact effort. Thinking-mode requests validate but omit temperature/top-p/presence/frequency fields. Response format supports exact text/JSON-object forms and rejects schemas/unknown forms.
- [ ] **Cycle 2D — Chat allowlist RED:** exact wire keys: `model`, `messages`, `stream`, `max_tokens`, `stop`, `response_format`, `tools`, `thinking`, `reasoning_effort`, adapter-owned streaming `stream_options`. Validate model/scalars/ranges/finite values and copied stop≤16. Omit logprobs/top-logprobs/user/prefix/FIM/beta flags/penalties/unknown kwargs.
- [ ] **Cycle 2E — Responses allowlist RED:** exact wire keys: `model`, `input`, `instructions`, `stream`, `max_output_tokens`, `tools`, `reasoning`, `text`. Convert system owner to instructions; function call immediately followed by matching `function_call_output`; reasoning adjacent to owning assistant/call round; flat function tools; no summary/encrypted content. Assert every forbidden stateful/background/tool-choice/sampler/cache field is absent.
- [ ] Responses `store` is not transmitted at all per DeepSeek's stateless contract. Do not copy Qwen's `store=false` rule into this adapter.
- [ ] **Cycle 2F — later-turn replay RED:** for both modes, every retained DeepSeek tool-bearing checkpoint round expands in every later same-provider request while owner remains in context. Tool-free reasoning is absent. Provider/model/mode/base switches block active restore rather than translating it. Atomic budget eviction tests from TASK-15675 stay green.
- [ ] Add mutation guards for call/output adjacency, reasoning omission, and input immutability; each mutation must fail the relevant focused test.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_deepseek.py -k "history or tools or thinking or allowlist or replay"
  ```

- [ ] Commit:

  ```bash
  git add tldw_chatbook/LLM_Calls/deepseek.py Tests/LLM_Calls/test_deepseek.py
  git commit -m "feat(deepseek): translate strict dual-mode requests"
  ```

### Task 3: Normalize Chat Through The Hosted Boundary

**Files:**

- Modify: `tldw_chatbook/LLM_Calls/deepseek.py`
- Modify: `tldw_chatbook/LLM_Calls/hosted_chat.py` only if a truly generic defect is exposed
- Modify: `tldw_chatbook/LLM_Calls/hosted_chat_streaming.py` only if a truly generic defect is exposed
- Modify: `Tests/LLM_Calls/test_deepseek.py`
- Modify: `Tests/LLM_Calls/test_hosted_chat.py`
- Modify: `Tests/LLM_Calls/test_hosted_chat_streaming.py`

- [ ] **Cycle 3A — nonstream RED:** official-shaped text/tool/mixed responses normalize with exact `reasoning_content`, one primary choice, valid finish, complete calls, and strict usage. Chat finishes: `stop`, `length`, `tool_calls`; map `content_filter` to refusal and `insufficient_system_resource` to transient provider error. Reject blank/unknown/contradictory finish, empty success, partial calls, malformed usage.
- [ ] **Cycle 3B — stream RED:** use hosted SSE framing with data-only DeepSeek chunks. Nullable content/reasoning/tool_calls/usage are absent, not coerced. Require stable choice/tool indexes, identity/name/argument fragment types, valid terminal finish, terminal usage chunk after requested completion, then `[DONE]`. Reject early `[DONE]`, EOF, post-terminal data, and incomplete calls.
- [ ] **Cycle 3C — transport RED:** call TASK-15676's `owned_json_post` with route `chat/completions`; assert exact URL/headers, frozen timeout, retry budget, Retry-After, sensitive zero retry, no retry after body byte/2xx, typed redacted errors, cancellation races, and exactly-once response/session close. Add no DeepSeek-only transport fork.
- [ ] Run hosted Qwen/Kimi/Z.ai compatibility after any shared edit:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_deepseek.py -k chat
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_hosted_chat_streaming.py Tests/LLM_Calls/test_qwencloud.py Tests/LLM_Calls/test_qwencloud_streaming.py Tests/LLM_Calls/test_moonshot.py Tests/LLM_Calls/test_zai.py
  ```

- [ ] Commit:

  ```bash
  git add tldw_chatbook/LLM_Calls/deepseek.py tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/hosted_chat_streaming.py Tests/LLM_Calls/test_deepseek.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_hosted_chat_streaming.py
  git commit -m "feat(deepseek): harden chat completions mode"
  ```

### Task 4: Implement Strict Responses Nonstream And Semantic SSE

**Files:**

- Create: `tldw_chatbook/LLM_Calls/deepseek_streaming.py`
- Create: `Tests/LLM_Calls/test_deepseek_streaming.py`
- Modify: `tldw_chatbook/LLM_Calls/deepseek.py`
- Modify: `Tests/LLM_Calls/test_deepseek.py`

- [ ] **Cycle 4A — discriminator/sequence RED:** consume TASK-15676 `SSERecord.event` plus its exact joined `data`; require the preserved SSE `event:` label to equal JSON `type` and require strict nonnegative integer `sequence_number`. Increasing skips are valid; duplicate/decreasing/missing/bool sequence values fail. JSON `event` is ignored/not required. `[DONE]` is invalid. A shared-regression test proves the neutral decoder preserves the label while `HostedChatStream` continues ignoring it.
- [ ] Accepted event set is exact: `response.created`, `response.in_progress`, `response.output_item.added`, `response.output_item.done`, `response.content_part.added`, `response.content_part.done`, `response.reasoning_text.delta`, `response.reasoning_text.done`, `response.output_text.delta`, `response.output_text.done`, `response.function_call_arguments.delta`, `response.function_call_arguments.done`, `response.completed`, `response.incomplete`, and `response.failed`. Unknown/web/custom events fail typed before execution.
- [ ] **Cycle 4B — items RED:** validate reasoning, message text, and function-call output items; stable output/call IDs and indexes; item status; required function `call_id` (never substitute transport item `id`); complete name/arguments; multiple interleaved indexes; content/tool coexistence. Provider-controlled state is capped and released on completion.
- [ ] **Cycle 4C — exactly-once recovery RED:** reconcile delta/done/full terminal representations without duplicate text/reasoning/arguments. Values and statuses must agree. Exact replay digest is accepted only where specified; conflicts/unseen post-terminal events fail. Use type-sensitive JSON equality and list/segment accumulation.
- [ ] **Cycle 4D — terminal RED:** completed returns text/tools; incomplete only with `max_output_tokens` maps length; other incomplete, failed, cancelled, malformed, missing, or contradictory terminal maps typed safe errors. Terminal event owns full usage. No terminal success with empty output and no complete calls.
- [ ] **Cycle 4E — owned transport/nonstream RED:** call TASK-15676's `owned_json_post` with route `responses`. Parse the same allowed output items/statuses/usage under strict depth/node/byte bounds and normalize identically to streaming. A 2xx invalid JSON/content encoding/body/shape error is closed and never retried. Loopback tests prove exact `/responses`, headers, frozen timeout/retry policy, and no duplicate session/response owner.
- [ ] **Cycle 4F — owned streaming lifecycle/privacy RED:** wrap the returned `OwnedSSEStream` in `DeepSeekResponsesTranslator`; truncated UTF-8/SSE, body read failures before/after event, cancellation before retention/during reasoning/text/partial call, explicit/repeated close, reentrant/raising cleanup, and deep RecursionError injection all produce context-free redacted DeepSeek errors or cancellation and close the underlying response/session once with no replay after body byte.
- [ ] Mutation guards: change label/type equality, permit duplicate sequence, use item `id` as `call_id`, or emit `[DONE]`; each must fail before restoration.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_deepseek_streaming.py Tests/LLM_Calls/test_deepseek.py -k responses
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/deepseek.py tldw_chatbook/LLM_Calls/deepseek_streaming.py Tests/LLM_Calls/test_deepseek.py Tests/LLM_Calls/test_deepseek_streaming.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy tldw_chatbook/LLM_Calls/deepseek.py tldw_chatbook/LLM_Calls/deepseek_streaming.py
  ```

- [ ] Commit:

  ```bash
  git add tldw_chatbook/LLM_Calls/deepseek.py tldw_chatbook/LLM_Calls/deepseek_streaming.py Tests/LLM_Calls/test_deepseek.py Tests/LLM_Calls/test_deepseek_streaming.py
  git commit -m "feat(deepseek): add strict Responses translation"
  ```

### Task 5: Dispatch, Freeze Mode, Carry Usage, And Prove Native Tools

**Files:**

- Modify: `tldw_chatbook/Chat/Chat_Functions.py`
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`
- Modify: `Tests/Agents/test_native_tools.py`
- Create: `Tests/Chat/test_deepseek_native_tools.py`

- [ ] **Cycle 5A — dispatcher RED:** map `api_mode` only to QwenCloud and DeepSeek. Remove the old DeepSeek text-only/history special branch from `Chat_Functions.py`; all translation now belongs to the adapter. Assert OpenAI/Kimi/Z.ai/other handler kwargs are unchanged.
- [ ] **Cycle 5B — frozen resolution RED:** Console resolves and freezes mode/base/key/model/timeout/retries/delay once; mutate Settings/config/environment between primary, second turn, auxiliary, and subagent calls and assert the snapshot remains. Qwen continues defaulting Responses; DeepSeek defaults Chat.
- [ ] **Cycle 5C — usage RED:** Chat and Responses terminal raw usage reaches call-scoped signals. Map Chat prompt/completion/cache-hit/cache-miss/reasoning and Responses input/output/cached-input/reasoning-output under strict integer validation. Malformed counts invoke the deterministic estimator; concurrent calls stay isolated.
- [ ] **Cycle 5D — joined HTTP RED:** parameterize both modes through real bridge→runtime→gateway→dispatcher→adapter→loopback HTTP. Server validates exact first/second requests before advancing. Prove complete multi-call batch, canonical result pairing/Responses adjacency, no synthetic user, reasoning replay on final and later same-provider turns, one real Calculator-influenced answer, terminal usage budget, and source immutability.
- [ ] DeepSeek is already in `NATIVE_TOOLS_PROVIDERS`; do not add another registry or pretend registry membership is the RED. The real RED is the missing dual-mode end-to-end contract. Keep the membership only after both unpatched joined cases pass.
- [ ] **Cycle 5E — crash/restore/cancel RED:** use TASK-15675 crash fixtures for pre-execution batch, result-before-continuation, restart/sync/JSON import/branch/regenerate/later turn. Completed/failed never re-run; executing blocks; pending explicit Resume plus approval. For partial streaming, cancel only after a parser/gateway/store-visible text marker and independently prove incomplete call/reasoning state was already observed; incomplete fragments create no checkpoint/owner row. Assert one live response close and zero execution/output.
- [ ] Mutation guards remove reasoning replay, call/output adjacency, checkpoint-before-execution, usage, or close forwarding; each must fail.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_deepseek_native_tools.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_agent_bridge.py Tests/Agents/test_native_tools.py Tests/Agents/test_agent_service.py Tests/Agents/test_agent_runtime.py
  ```

- [ ] Commit:

  ```bash
  git add tldw_chatbook/Chat/Chat_Functions.py tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Agents/agent_service.py Tests/Chat/test_deepseek_native_tools.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_agent_bridge.py Tests/Agents/test_native_tools.py
  git commit -m "feat(deepseek): connect both modes to native tools"
  ```

### Task 6: Add DeepSeek Mode To Readiness, Settings, And Discovery

**Files:**

- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/Chat/provider_readiness.py`
- Modify: `tldw_chatbook/Chat/console_provider_endpoints.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/model_catalog_settings.py`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_provider_identity.py`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py`
- Modify: `Tests/test_config_model_catalog_defaults.py`
- Modify: `Tests/Chat/test_provider_readiness.py`
- Create: `Tests/UI/test_settings_deepseek_api_mode.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `Tests/UI/test_settings_save_commit_models.py`
- Modify: `Tests/LLM_Provider_Catalog/test_model_catalog_settings.py`
- Modify: `Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py`
- Modify: `Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py`
- Modify: `Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py`
- Modify: `Tests/Provider/test_provider_model_resolution.py`
- Modify: `Tests/UI/test_provider_model_resolution.py`

- [ ] **Cycle 6A — config/readiness RED:** add canonical `api_base_url=https://api.deepseek.com` and `api_mode=chat_completions`; keep current model defaults. Readiness blocks malformed mode/table/base/key/model/retry before network with provider-specific recovery.
- [ ] **Cycle 6B — selector capability RED:** generalize the existing Qwen-only API Mode field to a small provider capability mapping:

  ```python
  API_MODE_OPTIONS = {
      "qwencloud": ("responses", "chat_completions"),
      "deepseek": ("chat_completions", "responses"),
  }
  ```

  Reuse namespaced drafts, canonical owner migration, atomic provider batch saves, Select.NULL invalid recovery, field search, revert, provider switching, and second-save no-op. Do not change Qwen defaults/options/copy. Non-mode providers keep the selector hidden/disabled.
- [ ] **Cycle 6C — reasoning/settings RED:** DeepSeek's existing reasoning control shows provider default/low/high/max only; rejects compatibility aliases; explains sampler/tool-choice omission, stateless explicit Responses, private later-turn tool reasoning, existing tools, and built-in exclusions. Active checkpoints pin mode/base/model and offer Resume/Discard rather than silent save-over.
- [ ] **Cycle 6D — discovery RED:** add DeepSeek to the existing auto-refresh/handler inventory if absent. Authenticated `{normalized_base}/models` uses the same base/key; mode never creates a second cache/identity. Preserve old cache on failure, selector cap/full search, IDs/timestamps-only disk, no capability inference, no active-model mutation.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_settings_deepseek_api_mode.py Tests/UI/test_settings_qwencloud_api_mode.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py Tests/Chat/test_provider_readiness.py Tests/Chat/test_console_provider_gateway.py Tests/LLM_Provider_Catalog/test_model_catalog_settings.py Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py Tests/Provider/test_provider_model_resolution.py Tests/UI/test_provider_model_resolution.py Tests/test_config_model_catalog_defaults.py
  ```

- [ ] Commit:

  ```bash
  git add tldw_chatbook/config.py tldw_chatbook/Chat/provider_readiness.py tldw_chatbook/Chat/console_provider_endpoints.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/LLM_Provider_Catalog/model_catalog_settings.py tldw_chatbook/LLM_Provider_Catalog/model_discovery_provider_identity.py tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py Tests/test_config_model_catalog_defaults.py Tests/Chat/test_provider_readiness.py Tests/UI/test_settings_deepseek_api_mode.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py Tests/LLM_Provider_Catalog/test_model_catalog_settings.py Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py Tests/Provider/test_provider_model_resolution.py Tests/UI/test_provider_model_resolution.py
  git commit -m "feat(settings): add DeepSeek API mode"
  ```

### Task 7: Document, Optionally Verify Live, And Close TASK-15677

**Files:**

- Modify: `README.md`
- Modify: `Docs/User_Guide/settings.md`
- Modify: `Docs/User_Guide/console.md`
- Create: `Tests/Chat/test_live_deepseek_api.py`
- Modify: `backlog/tasks/task-15677 - Add-DeepSeek-dual-API-provider-support.md`
- Modify: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md` only for a genuinely new incident-backed lesson.

- [ ] Document stable identity/key/current models/base; exact mode values and Chat default; exact per-mode allowlists/omissions; provider-default/low/high/max thinking; stateless explicit Responses history; tool-choice/sampler rules; existing function tools and web/apply_patch exclusions; durable later-turn reasoning and explicit recovery; stream terminal/usage differences; discovery/cache/unknown pricing; invalid recovery and live gates.
- [ ] Add two default-skipped live cases requiring both `TLDW_LIVE_DEEPSEEK=1` and nonblank `DEEPSEEK_API_KEY`. Each mode runs in a fresh isolated subprocess with environment/data/config set before imports, no log sinks, discarded stdout/stderr, randomized prompt/arithmetic, one exact Calculator call/result, and final marker derived only after the result. Never expose key/prompt/response/tool result in failures.
- [ ] Prove the default live file skips with zero request:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_live_deepseek_api.py
  ```

- [ ] Run the complete dual-mode/provider/shared surfaces and full suite:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_deepseek.py Tests/LLM_Calls/test_deepseek_streaming.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_hosted_chat_streaming.py Tests/LLM_Calls/test_qwencloud.py Tests/LLM_Calls/test_qwencloud_streaming.py Tests/LLM_Calls/test_moonshot.py Tests/LLM_Calls/test_zai.py Tests/Chat/test_deepseek_native_tools.py Tests/Chat/test_live_deepseek_api.py Tests/Agents/test_native_tools.py Tests/Agents/test_agent_service.py Tests/Agents/test_agent_runtime.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_provider_readiness.py Tests/Chat/test_provider_continuation.py Tests/UI/test_settings_deepseek_api_mode.py Tests/UI/test_settings_qwencloud_api_mode.py Tests/LLM_Provider_Catalog
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q
  ```

- [ ] Run static/security gates:

  ```bash
  git diff --check origin/dev...HEAD
  git diff --name-only -z --diff-filter=ACM origin/dev...HEAD -- '*.py' | xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check
  git diff --name-only -z --diff-filter=ACM origin/dev...HEAD -- '*.py' | xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy tldw_chatbook/LLM_Calls/deepseek.py tldw_chatbook/LLM_Calls/deepseek_streaming.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook/LLM_Calls/deepseek.py tldw_chatbook/LLM_Calls/deepseek_streaming.py
  ```

- [ ] Self-review every AC and ADR-063/064 invariant. Confirm no second provider/cache/loop/schema, no provider built-ins/stateful Responses fields, no Qwen/Kimi/Z.ai drift, no raw secret/body leakage, no hidden replay, and no paid default test.
- [ ] Check every AC individually, add observed Implementation Notes/evidence/deviations to the task file, verify the rendered task, and set Done only when complete:

  ```bash
  backlog task edit 15677 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 --check-ac 7 --check-ac 8 --check-ac 9 --check-ac 10
  # Use apply_patch to add Implementation Notes from the observed results.
  backlog task 15677 --plain
  backlog task edit 15677 -s Done
  ```

- [ ] Commit closeout:

  ```bash
  git add README.md Docs/User_Guide/settings.md Docs/User_Guide/console.md Docs/superpowers/plans/2026-08-12-deepseek-dual-api-provider-implementation.md Tests/Chat/test_live_deepseek_api.py "backlog/tasks/task-15677 - Add-DeepSeek-dual-API-provider-support.md"
  git commit -m "docs(deepseek): document dual API support"
  ```

## PR Boundary And Delivery

- This is PR 3 of 3 and opens against `dev` after TASK-15675/15676 merge.
- PR description links both ADRs, all three tasks/specs/plans, shared Chat regression evidence, semantic Responses mutation tests, durable restart/sync/import evidence, joined native-tool/cancellation evidence, optional-live status, and precise baseline-only failures.
- Once review comments and CI are green, rebase onto latest `dev`, rerun the affected and full gates, resolve review threads with evidence, then merge under the repository's normal PR policy.
