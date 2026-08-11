# QwenCloud Dual-API Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add QwenCloud as an ordinary hosted API provider with a pinned `api_mode` setting that defaults to Responses, supports Chat Completions as an alternative, and carries existing Chatbook function tools through both modes.

**Architecture:** Register one `qwencloud` provider behind the existing dispatcher and Console gateway. Keep all mode-specific endpoint, payload, tool-history, response, and SSE translation inside a dedicated adapter; the shared runtime receives only normalized OpenAI-shaped text/tool/usage events. Pin the validated mode and effective base URL in `ConsoleProviderResolution`, then prove the unchanged native-tool runtime can execute and continue multiple calls in both modes.

**Tech Stack:** Python 3.11+, `requests`/`urllib3` transport, Textual 8.x Settings UI, pytest/pytest-asyncio, existing Chatbook provider dispatcher, Console gateway, native-agent runtime, and model-catalog service.

---

## Design Sources And ADR Check

- Approved design: `Docs/superpowers/specs/2026-08-02-qwencloud-dual-api-provider-design.md`
- Existing decision: `backlog/decisions/045-qwencloud-dual-api-provider-boundary.md`
- Related decisions: ADR-006, ADR-012, ADR-020, and ADR-026.
- Backlog source of truth: `backlog/tasks/task-3771 - Add-QwenCloud-dual-API-provider-support.md`
- Official provider references: `https://www.qwencloud.com/skills.md` and `https://www.qwencloud.com/models/qwen3.8-max#api-reference`

ADR required: yes

ADR path: `backlog/decisions/045-qwencloud-dual-api-provider-boundary.md`

Reason: This feature changes a provider/runtime boundary and wire contract. ADR-045 already records the approved one-provider/two-mode decision, the pinned handoff, and adapter ownership, so no new ADR is needed.

## Scope Guardrails

- Implement existing Chatbook function tools only. QwenCloud built-in tools remain a separate feature.
- Do not add a second provider identity for Chat Completions.
- Do not add Responses server-side state (`previous_response_id`, `conversation`) and always send `store=false`.
- Do not add QwenCloud pricing without a verified official price source; the existing unknown-pricing path must continue to show token counts.
- Do not change legacy Settings surfaces. The only UI change belongs in `tldw_chatbook/UI/Screens/settings_screen.py`.
- Do not add a database schema or migration.
- Do not make paid calls in the default test suite. Optional live verification is explicitly gated.

## Baseline Evidence

Run from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/qwencloud-provider` with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`.

Frozen baseline commit: `97a75fb8b` (`origin/dev` when this plan was written).

- Provider/runtime/catalog baseline: 272 passed. Two first-run errors were only the managed sandbox blocking localhost socket binds; the identical command passed outside that restriction.
- Canonical Settings baseline: 323 passed, 4 failed on untouched `origin/dev`:
  - `test_settings_ownership_records_cover_categories_and_runtime_boundaries`
  - `test_settings_console_behavior_saves_display_name_exactly`
  - `test_settings_provider_category_saves_provider_defaults_without_sampling`
  - `test_settings_provider_switch_does_not_save_stale_endpoint`
- During implementation, compare any failure from an identical command against this baseline. A baseline failure is not evidence that a new QwenCloud test passed; every new test must first be shown red for the intended missing behavior and then green after the smallest implementation.
- Do not mark TASK-3771 Done if the feature adds a failure beyond the recorded baseline. If `origin/dev` moves, rerun both baseline and feature arms at the same base commit.

Reproduce the 272-pass baseline with:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_provider_gateway.py Tests/Agents/test_native_tools.py Tests/Chat/test_provider_readiness.py Tests/Chat/test_console_provider_endpoints.py Tests/Chat/test_console_provider_support.py Tests/LLM_Provider_Catalog/test_model_catalog_settings.py Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py Tests/test_config_model_catalog_defaults.py
```

Reproduce the Settings baseline with:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py
```

## TDD And Commit Discipline

- For every behavior below: add one focused test, run it and inspect the expected failure, implement only enough production code to satisfy it, rerun the focused test, then run the task's regression set.
- Use verbatim QwenCloud JSON/SSE fixtures. Network fakes may intercept I/O but must not replace production function signatures or the real dispatcher/gateway/native-runtime boundaries.
- Place a mutation guard around request inputs and assert they are unchanged after translation.
- Keep sensitive request bodies, message/tool content, keys, and raw provider error bodies out of logs.
- Commit after each task only when its focused tests pass. Suggested commit messages are included; do not commit a red intermediate state.

### Task 1: Establish provider configuration, identity, readiness, and endpoint contracts

**Files:**

- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/Chat/provider_catalog.py`
- Modify: `tldw_chatbook/Chat/console_provider_support.py`
- Modify: `tldw_chatbook/Chat/provider_readiness.py`
- Modify: `tldw_chatbook/Chat/console_provider_endpoints.py`
- Modify: `Tests/test_config_model_catalog_defaults.py`
- Modify: `Tests/Chat/test_console_provider_support.py`
- Modify: `Tests/Chat/test_provider_readiness.py`
- Modify: `Tests/Chat/test_console_provider_endpoints.py`
- Create: `Tests/Chat/test_qwencloud_provider_contract.py`

- [ ] **Cycle 1A — defaults red:** add `test_qwencloud_embedded_config_defaults` asserting `[providers].QwenCloud == ["qwen3.8-max"]`, `[api_settings.qwencloud].api_mode == "responses"`, `api_key_env_var == "DASHSCOPE_API_KEY"`, and the international compatible-mode base URL and retry/stream defaults from the design.
- [ ] Run the one new node:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_qwencloud_provider_contract.py::test_qwencloud_embedded_config_defaults
  ```

  Expected: fail because QwenCloud is absent from embedded config.

- [ ] Implement only the `[providers]` and `[api_settings.qwencloud]` defaults in `CONFIG_TOML_CONTENT` plus `QwenCloud` in `_cloud_provider_keys`; rerun the node and expect pass.
- [ ] **Cycle 1B — identity red:** add `test_qwencloud_uses_one_supported_console_identity` and extend the existing provider inventory assertions for display label `QwenCloud`, normalized readiness/execution key `qwencloud`, and ordinary Console sendability.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_qwencloud_provider_contract.py::test_qwencloud_uses_one_supported_console_identity Tests/Chat/test_console_provider_support.py
  ```

  Expected: fail because the catalog/support inventory has no QwenCloud identity.

- [ ] Add `"qwencloud": "QwenCloud"` to the shared provider display-name catalog and only the minimum support inventory needed for the derived Console catalog; rerun and expect pass.
- [ ] **Cycle 1C — readiness red:** add `test_qwencloud_readiness_uses_modern_config_before_its_env` and `test_qwencloud_readiness_never_borrows_another_provider`. Cover credential requirement, default `DASHSCOPE_API_KEY`, configured env-name override, modern-config-over-environment precedence, and isolation from OpenAI/DeepSeek/Custom OpenAI keys.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_qwencloud_provider_contract.py::test_qwencloud_readiness_uses_modern_config_before_its_env Tests/Chat/test_qwencloud_provider_contract.py::test_qwencloud_readiness_never_borrows_another_provider
  ```

  Expected: fail because readiness does not require or resolve QwenCloud credentials.

- [ ] Add `qwencloud` to `PROVIDERS_REQUIRING_API_KEY_KEYS` and map it to `DASHSCOPE_API_KEY` in `_DEFAULT_API_KEY_ENV_VAR_ALIASES`. Keep precedence in the shared readiness/config boundary and add no legacy bridge unless one already exists (none exists at plan time); rerun and expect pass.
- [ ] **Cycle 1D — endpoint registration red:** add `test_qwencloud_builtin_endpoint_is_international_compatible_base`, asserting the absent-config effective endpoint is `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` and QwenCloud is treated as an endpoint-using hosted provider.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_qwencloud_provider_contract.py::test_qwencloud_builtin_endpoint_is_international_compatible_base`, expect failure, then add the built-in endpoint/URL-provider registration and rerun green. Full path validation and pasted endpoint normalization deliberately land in Task 2's single pure normalizer, not in this registry step.
- [ ] Run regression tests:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_provider_support.py Tests/Chat/test_provider_readiness.py Tests/Chat/test_console_provider_endpoints.py Tests/test_config_model_catalog_defaults.py Tests/Chat/test_qwencloud_provider_contract.py
  ```

  Expected: pass with no OpenAI/DeepSeek/Custom OpenAI behavior changes.

- [ ] Commit:

  ```bash
  git add tldw_chatbook/config.py tldw_chatbook/Chat/provider_catalog.py tldw_chatbook/Chat/console_provider_support.py tldw_chatbook/Chat/provider_readiness.py tldw_chatbook/Chat/console_provider_endpoints.py Tests/test_config_model_catalog_defaults.py Tests/Chat/test_console_provider_support.py Tests/Chat/test_provider_readiness.py Tests/Chat/test_console_provider_endpoints.py Tests/Chat/test_qwencloud_provider_contract.py
  git commit -m "feat(qwencloud): add provider configuration and readiness"
  ```

### Task 2: Build fail-closed mode, request, history, and function-tool translation

**Files:**

- Create: `tldw_chatbook/LLM_Calls/qwencloud.py`
- Create: `Tests/LLM_Calls/test_qwencloud.py`

- [ ] Create these concrete pure interfaces before transport:

  ```python
  QwenCloudAPIMode = Literal["responses", "chat_completions"]

  def normalize_qwencloud_api_mode(
      api_mode: str | None,
      *,
      provider_settings: Mapping[str, Any] | None = None,
  ) -> QwenCloudAPIMode: ...

  def normalize_qwencloud_base_url(api_base_url: str | None) -> str: ...

  def resolve_qwencloud_api_key(
      explicit_api_key: str | None,
      *,
      provider_settings: Mapping[str, Any] | None = None,
      environ: Mapping[str, str] | None = None,
  ) -> str: ...

  def build_qwencloud_payload(
      *,
      api_mode: QwenCloudAPIMode,
      model: str,
      system_message: str | None,
      messages_payload: Sequence[Mapping[str, Any]],
      streaming: bool,
      tools: Sequence[Mapping[str, Any]] | None = None,
      tool_choice: str | None = None,
      **sampling: Any,
  ) -> dict[str, Any]: ...
  ```

  `build_qwencloud_payload` accepts only the dispatcher parameters explicitly mapped in Task 3; it must not forward arbitrary `sampling` keys. The loose annotation above reflects the existing dispatch surface, while the implementation selects named allowed fields one by one.

- [ ] **Cycle 2A — resolution helpers red:** add `test_api_mode_config_then_default_and_exact_values`, `test_base_url_normalizes_base_and_pasted_endpoints`, `test_base_url_rejects_unsafe_or_malformed_values`, and `test_api_key_precedence_is_provider_isolated`.
- [ ] Positive base cases must normalize all three to the same base: `/compatible-mode/v1`, `/compatible-mode/v1/responses`, and `/compatible-mode/v1/chat/completions` (with whitespace/trailing slash). Remove exactly one recognized terminal request suffix. Reject `/models`, embedded credentials, query/fragment, non-HTTP(S), missing host, double/unknown terminal paths, and malformed paths before a network trap is touched.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud.py::test_api_mode_config_then_default_and_exact_values Tests/LLM_Calls/test_qwencloud.py::test_base_url_normalizes_base_and_pasted_endpoints Tests/LLM_Calls/test_qwencloud.py::test_base_url_rejects_unsafe_or_malformed_values Tests/LLM_Calls/test_qwencloud.py::test_api_key_precedence_is_provider_isolated
  ```

  Expected: collection/import failure because the helpers do not exist.

- [ ] Implement constants and the three small resolution helpers. Mode aliases/unknowns raise `ChatConfigurationError`; endpoint/key errors use existing typed configuration/auth errors without including raw secrets. Rerun the four nodes and expect pass.
- [ ] **Cycle 2B — Responses request red:** add `test_responses_payload_has_exact_allowlist_and_stateless_invariants`, `test_responses_system_message_maps_to_instructions`, and `test_responses_reasoning_effort_enum_is_exact`. Cover the exact Responses allowlist:

  ```text
  model,input,instructions,stream,store,temperature,top_p,max_output_tokens,tools,tool_choice,reasoning
  ```

  Assert `store` is exactly `false`, stateful IDs are absent, `max_tokens` maps to `max_output_tokens` with minimum 16, `system_message` maps to `instructions`, equal duplicate leading system text is de-duplicated, a conflicting leading system instruction fails, and reasoning accepts exactly `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` under `reasoning={"effort": ...}`.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud.py::test_responses_payload_has_exact_allowlist_and_stateless_invariants Tests/LLM_Calls/test_qwencloud.py::test_responses_system_message_maps_to_instructions Tests/LLM_Calls/test_qwencloud.py::test_responses_reasoning_effort_enum_is_exact
  ```

  Expected: assertion failures from the absent Responses builder. Implement only the Responses scalar/system mapping and exact allowlist; rerun green.
- [ ] **Cycle 2C — Chat request red:** add `test_chat_payload_has_exact_allowlist_and_thinking_invariant` for the exact Chat Completions allowlist:

  ```text
  model,messages,stream,temperature,top_p,top_k,max_completion_tokens,seed,presence_penalty,stop,response_format,n,logprobs,top_logprobs,tools,tool_choice,reasoning_effort,preserve_thinking,stream_options
  ```

  Assert `preserve_thinking` is always `false`, streaming sets `stream_options.include_usage=true`, tool requests force `n=1`, and only `text`/`json_object` response formats pass. Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud.py::test_chat_payload_has_exact_allowlist_and_thinking_invariant` red, implement the Chat scalar mapping, then rerun green.
- [ ] **Cycle 2D — tool schema red:** add `test_function_tools_translate_by_mode` and `test_invalid_or_builtin_tools_fail_before_network`. Chat keeps the nested OpenAI function schema; Responses flattens it. Require `type="function"`, non-empty unique names, and object-shaped `parameters`; accept absent/`auto`/`none` tool choice; reject forced choices, duplicate/empty names, non-object parameters, and QwenCloud built-in tool types.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud.py::test_function_tools_translate_by_mode Tests/LLM_Calls/test_qwencloud.py::test_invalid_or_builtin_tools_fail_before_network` red, implement `_validate_and_translate_qwencloud_tools(...)`, wire it into the builder, and rerun green.
- [ ] **Cycle 2E — message/history red:** add `test_message_content_translation_is_role_safe_and_immutable`, `test_responses_assistant_text_is_id_free_easy_input_message`, `test_responses_pairs_out_of_order_results_by_call_id`, and `test_responses_rejects_unpairable_tool_batches_before_network`.
- [ ] Cover ordinary string text, text-array collapse, mixed user `input_text`/`input_image`, user-only images, supported roles, role/content type rejection, non-user image rejection, unknown part rejection, empty assistant content with tool calls, and deep input immutability.
- [ ] Assert prior assistant text is exactly `{"role":"assistant","content":[{"type":"output_text","text":"..."}]}` with no `id`, `status`, or top-level `type`.
- [ ] Use assistant text plus calls A/B followed by result rows B/A as the positive Responses fixture. Pair by `call_id` and emit assistant text, call A/output A, call B/output B. Keep missing, duplicate, orphaned, extra, reused, and cross-batch results as negative fixtures. Never synthesize a user message.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud.py::test_message_content_translation_is_role_safe_and_immutable Tests/LLM_Calls/test_qwencloud.py::test_responses_assistant_text_is_id_free_easy_input_message Tests/LLM_Calls/test_qwencloud.py::test_responses_pairs_out_of_order_results_by_call_id Tests/LLM_Calls/test_qwencloud.py::test_responses_rejects_unpairable_tool_batches_before_network` red, implement one validated canonical-history pass plus two mode renderers, then rerun green.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud.py
  ```

  Expected: pass.

- [ ] Run mutation/parameter regressions together with representative dispatcher tests:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud.py Tests/Chat/test_chat_unit_mocked_APIs.py -k "qwencloud or openai or deepseek"
  ```

  Expected: QwenCloud tests pass and existing representative providers do not regress.

- [ ] Commit:

  ```bash
  git add tldw_chatbook/LLM_Calls/qwencloud.py Tests/LLM_Calls/test_qwencloud.py
  git commit -m "feat(qwencloud): translate dual API requests and tools"
  ```

### Task 3: Add non-streaming transport, normalization, retries, dispatcher registration, and safe errors

**Files:**

- Modify: `tldw_chatbook/LLM_Calls/qwencloud.py`
- Modify: `tldw_chatbook/Chat/Chat_Functions.py`
- Modify: `Tests/LLM_Calls/test_qwencloud.py`
- Modify: `Tests/Chat/test_chat_unit_mocked_APIs.py`
- Modify: `Tests/Chat/test_sensitive_llm_logging.py`
- Modify: `Tests/Chat/test_qwencloud_provider_contract.py`

- [ ] Add these concrete adapter/dispatcher seams:

  ```python
  def normalize_qwencloud_response(
      payload: Mapping[str, Any], *, api_mode: QwenCloudAPIMode
  ) -> dict[str, Any]: ...

  def chat_with_qwencloud(
      input_data: list[dict[str, Any]],
      model: str | None = None,
      api_key: str | None = None,
      system_message: str | None = None,
      temp: float | None = None,
      streaming: bool | None = False,
      topp: float | None = None,
      topk: int | None = None,
      max_tokens: int | None = None,
      seed: int | None = None,
      stop: str | list[str] | None = None,
      logprobs: bool | None = None,
      top_logprobs: int | None = None,
      presence_penalty: float | None = None,
      response_format: dict[str, str] | None = None,
      n: int | None = None,
      tools: list[dict[str, Any]] | None = None,
      tool_choice: str | dict[str, Any] | None = None,
      reasoning_effort: str | None = None,
      api_base_url: str | None = None,
      api_mode: str | None = None,
  ) -> dict[str, Any] | Iterator[dict[str, Any]]: ...
  ```

  This matches the existing provider-handler convention: `PROVIDER_PARAM_MAP` renames generic `messages_payload` to `input_data`. It deliberately omits generic fields excluded by the approved allowlists. Transport `timeout`, `retries`, and `retry_delay` resolve inside the adapter from `[api_settings.qwencloud]`; this plan does **not** add those three to the shared dispatcher signature.

- [ ] **Cycle 3A — dispatcher red:** add `test_chat_api_call_forwards_qwencloud_mode_base_and_tools` and `test_qwencloud_is_sensitive_auxiliary_audited`. Change the public dispatcher contract to `chat_api_call(..., api_mode: str | None = None)`; update its docstring, let `_CHAT_API_GENERIC_PARAMS` derive the new name from the signature, register `API_CALL_HANDLERS["qwencloud"]`, and define `PROVIDER_PARAM_MAP["qwencloud"]` with the exact supported generic fields including `api_mode`, `api_base_url`, tools, sampling, model, key, messages, and streaming.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_qwencloud_provider_contract.py::test_chat_api_call_forwards_qwencloud_mode_base_and_tools Tests/Chat/test_qwencloud_provider_contract.py::test_qwencloud_is_sensitive_auxiliary_audited
  ```

  Expected: fail because the handler/signature/map/audit membership is absent. Add only that neutral plumbing and rerun green. Assert a representative OpenAI/DeepSeek call receives no `api_mode` kwarg.
- [ ] **Cycle 3B — transport/config red:** add `test_nonstream_transport_uses_exact_mode_url_headers_and_timeout` and `test_direct_adapter_loads_only_qwencloud_config_when_arguments_are_none`. Parameterize Responses/Chat and assert exact POST suffix, bearer header, JSON content type, config-resolved timeout, and Task 2 payload. The direct test calls `chat_with_qwencloud(input_data=[...], api_mode=None, api_base_url=None, api_key=None)` against an intercepted transport and proves it loads `[api_settings.qwencloud]` mode/base/key with the same precedence, defaulting behavior, and provider isolation as the pure helpers.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud.py::test_nonstream_transport_uses_exact_mode_url_headers_and_timeout Tests/LLM_Calls/test_qwencloud.py::test_direct_adapter_loads_only_qwencloud_config_when_arguments_are_none` red, implement adapter config loading plus session/request construction and endpoint suffix append, then rerun green. The intercepted request must never contain another provider's base or key.
- [ ] **Cycle 3C — normalization red:** add `test_nonstream_normalizes_text_tools_finish_and_usage` and `test_nonstream_rejects_empty_success_and_malformed_shapes`. Cover text-only, tool-only, mixed text/tool, multi-call, refusal/failure, incomplete calls, empty 2xx, and malformed envelopes. Normalize both modes to `choices[0].message`, standard `tool_calls`, `finish_reason`, and top-level usage; preserve Responses `input_tokens`, `output_tokens`, and `total_tokens`.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud.py::test_nonstream_normalizes_text_tools_finish_and_usage Tests/LLM_Calls/test_qwencloud.py::test_nonstream_rejects_empty_success_and_malformed_shapes` red, implement `normalize_qwencloud_response`, then rerun green. A 2xx with neither usable text nor complete tools must raise `ChatProviderError`, never become an empty successful answer.
- [ ] **Cycle 3D — retry/error red:** add `test_retry_policy_counts_status_connection_and_timeout_attempts`, `test_sensitive_request_forces_zero_retries`, `test_nontransient_4xx_and_mode_model_mismatch_are_not_retried`, and `test_qwencloud_errors_and_logs_redact_private_values`.
- [ ] Use a local scripted server plus patched connection/timeout exceptions. `retries` means additional attempts, negative values clamp to zero, and `llm_retry_count()` can force zero in sensitive context. Retry POST for 429/500/502/503/504, connection-establishment failures, and timeouts; honor integer/date `Retry-After`; use exponential `retry_delay`; attempt every other 4xx once.
- [ ] For a fixture representing model/mode incompatibility, assert one attempt and an actionable typed recovery message that recommends a compatible model or switching `api_mode`; do not expose the raw provider body. Assert authorization, messages, tools, arguments, results, and raw bodies never enter logs or sensitive-mode wrapped exceptions.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud.py::test_retry_policy_counts_status_connection_and_timeout_attempts Tests/LLM_Calls/test_qwencloud.py::test_sensitive_request_forces_zero_retries Tests/LLM_Calls/test_qwencloud.py::test_nontransient_4xx_and_mode_model_mismatch_are_not_retried Tests/LLM_Calls/test_qwencloud.py::test_qwencloud_errors_and_logs_redact_private_values` red, implement the `HTTPAdapter`/`Retry` policy plus typed safe errors, then rerun green.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud.py Tests/Chat/test_chat_unit_mocked_APIs.py -k "qwencloud or dispatch"
  ```

  Expected: pass.

- [ ] Run privacy and registration regressions:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_sensitive_llm_logging.py Tests/Chat/test_qwencloud_provider_contract.py
  ```

  Expected: pass; no QwenCloud secrets or payload fragments appear in captured logs.

- [ ] Commit:

  ```bash
  git add tldw_chatbook/LLM_Calls/qwencloud.py tldw_chatbook/Chat/Chat_Functions.py Tests/LLM_Calls/test_qwencloud.py Tests/Chat/test_chat_unit_mocked_APIs.py Tests/Chat/test_sensitive_llm_logging.py Tests/Chat/test_qwencloud_provider_contract.py
  git commit -m "feat(qwencloud): add transport dispatcher and retries"
  ```

### Task 4: Implement record-aware streaming and deterministic resource closure

**Files:**

- Create: `tldw_chatbook/LLM_Calls/qwencloud_streaming.py`
- Modify: `tldw_chatbook/LLM_Calls/qwencloud.py`
- Create: `Tests/LLM_Calls/test_qwencloud_streaming.py`

- [ ] Implement toward these concrete interfaces:

  ```python
  def iter_sse_data_records(chunks: Iterable[bytes]) -> Iterator[str]: ...

  class QwenResponsesStreamTranslator:
      def feed(self, event: Mapping[str, Any]) -> tuple[dict[str, Any], ...]: ...
      def finish(self) -> tuple[dict[str, Any], ...]: ...

  class QwenCloudStream(Iterator[dict[str, Any]]):
      def __init__(self, *, response: requests.Response, session: requests.Session,
                   api_mode: QwenCloudAPIMode) -> None: ...
      def __next__(self) -> dict[str, Any]: ...
      def close(self) -> None: ...
  ```

  `iter_sse_data_records` owns incremental UTF-8/newline/blank-record framing; the translator owns Responses event state; `QwenCloudStream.close()` is idempotent and owns response/session closure.

- [ ] **Cycle 4A — framing red:** add `test_sse_records_survive_adversarial_byte_boundaries` and `test_sse_comments_and_multiline_data_frame_without_decoding`. Use verbatim bytes split inside UTF-8 code points, `data:` lines, and CRLF boundaries. Collect multiline data through the blank terminator and ignore comments/heartbeats. This layer returns record strings and deliberately does not parse JSON.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud_streaming.py::test_sse_records_survive_adversarial_byte_boundaries Tests/LLM_Calls/test_qwencloud_streaming.py::test_sse_comments_and_multiline_data_frame_without_decoding` red, implement only `iter_sse_data_records`, and rerun green.
- [ ] **Cycle 4B — text/state red:** add `test_responses_text_delta_done_recovery_is_exactly_once`, `test_responses_sequence_duplicate_conflict_and_decrease`, and `test_responses_terminal_usage_finish_and_empty_delta`. Cover output/content part added, text delta/done, content/output item done, completed, exact duplicate replay, conflicting/decreasing sequence, distinct output indexes, post-terminal events, failed/incomplete status, malformed/missing terminal, nested usage, and terminal `delta.content == ""`.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud_streaming.py::test_responses_text_delta_done_recovery_is_exactly_once Tests/LLM_Calls/test_qwencloud_streaming.py::test_responses_sequence_duplicate_conflict_and_decrease Tests/LLM_Calls/test_qwencloud_streaming.py::test_responses_terminal_usage_finish_and_empty_delta` red, implement text/sequence/terminal state in `QwenResponsesStreamTranslator`, and rerun green.
- [ ] **Cycle 4C — function-call red:** add `test_responses_function_call_fragments_recover_without_duplication` and `test_responses_partial_or_mismatched_call_never_surfaces`. Item-added establishes index/call ID/name; deltas emit standard indexed `delta.tool_calls`; done/output-item/completed validates or recovers arguments once. Invalid JSON, incomplete identity, or mismatched terminal arguments raise before a complete call reaches the accumulator.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud_streaming.py::test_responses_function_call_fragments_recover_without_duplication Tests/LLM_Calls/test_qwencloud_streaming.py::test_responses_partial_or_mismatched_call_never_surfaces` red, add function-call state to the translator, and rerun green.
- [ ] **Cycle 4D — wrapper/lifecycle/decoding red:** add `test_chat_stream_preserves_openai_deltas_and_usage`, `test_stream_retries_only_before_first_consumed_byte`, `test_stream_close_is_idempotent_and_closes_response_and_session`, and `test_stream_malformed_json_and_error_event_are_typed_closed_and_not_retried`. Cover `[DONE]`, Chat text/tool fragments, finish/usage, pre-consumption status retry, no retry after any response byte/event, normal exhaustion, error, and caller `.close()`. Malformed JSON and provider `error` events are decoded in `QwenCloudStream`, raise typed `ChatProviderError(provider="qwencloud")`, close resources, and never retry because bytes were consumed.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud_streaming.py::test_chat_stream_preserves_openai_deltas_and_usage Tests/LLM_Calls/test_qwencloud_streaming.py::test_stream_retries_only_before_first_consumed_byte Tests/LLM_Calls/test_qwencloud_streaming.py::test_stream_close_is_idempotent_and_closes_response_and_session Tests/LLM_Calls/test_qwencloud_streaming.py::test_stream_malformed_json_and_error_event_are_typed_closed_and_not_retried` red, implement record decoding plus `QwenCloudStream` and return it from `chat_with_qwencloud`, then rerun green. Its `finally` path closes response/session exactly once; it never yields a partial complete call.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud_streaming.py Tests/LLM_Calls/test_qwencloud.py
  ```

  Expected: pass.

- [ ] Run a mutation check by temporarily changing one fixture sequence number/done value and confirm the relevant test fails, then restore the fixture and rerun green.
- [ ] Commit:

  ```bash
  git add tldw_chatbook/LLM_Calls/qwencloud.py tldw_chatbook/LLM_Calls/qwencloud_streaming.py Tests/LLM_Calls/test_qwencloud.py Tests/LLM_Calls/test_qwencloud_streaming.py
  git commit -m "feat(qwencloud): parse and normalize streaming events"
  ```

### Task 5: Pin QwenCloud mode/base in Console and fix gateway cancellation/usage handoff

**Files:**

- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`
- Modify: `Tests/Chat/test_qwencloud_provider_contract.py`

- [ ] Extend the frozen handoff exactly as follows:

  ```python
  @dataclass(frozen=True)
  class ConsoleProviderResolution:
      # existing fields unchanged
      api_mode: str | None = None
  ```

  `resolve_for_send` calls Task 2's pure `normalize_qwencloud_api_mode` and `normalize_qwencloud_base_url` only for execution key `qwencloud`. `_chat_api_kwargs_from_prepared`, `_chat_api_kwargs`, and `_auxiliary_chat_api_kwargs` add `api_mode` plus the pinned `api_base_url` only for QwenCloud. No shared caller selects a wire endpoint.

- [ ] **Cycle 5A — pinned resolution red:** add `test_qwencloud_resolution_pins_normalized_mode_and_base`, `test_qwencloud_resolution_rejects_invalid_mode_before_dispatch`, and `test_non_qwen_resolutions_omit_api_mode`. Include base, pasted `/responses`, and pasted `/chat/completions` selection values; all must pin the normalized base.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_provider_gateway.py::test_qwencloud_resolution_pins_normalized_mode_and_base Tests/Chat/test_console_provider_gateway.py::test_qwencloud_resolution_rejects_invalid_mode_before_dispatch Tests/Chat/test_console_provider_gateway.py::test_non_qwen_resolutions_omit_api_mode` red, add the optional field and Qwen-only `resolve_for_send` validation/recovery copy, then rerun green. Invalid mode/endpoint recovery names QwenCloud and the setting but never a key.
- [ ] **Cycle 5B — handoff red:** add `test_all_qwencloud_kwargs_paths_forward_pinned_mode_and_base` and `test_qwencloud_run_ignores_midrun_config_mutation`. Resolve Responses/base A, mutate config to Chat/base B, then make two turns plus an auxiliary completion with the original resolution; every call must remain Responses/base A. Representative OpenAI, DeepSeek, and Anthropic kwargs remain unchanged and omit `api_mode`.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_provider_gateway.py::test_all_qwencloud_kwargs_paths_forward_pinned_mode_and_base Tests/Chat/test_console_provider_gateway.py::test_qwencloud_run_ignores_midrun_config_mutation` red, add the three Qwen-only kwargs branches, and rerun green.
- [ ] **Cycle 5C — usage red:** add `test_qwencloud_responses_terminal_usage_reaches_console_signals_without_copy` and `test_qwencloud_total_tokens_reaches_agent_usage_counter`. Feed a normalized terminal chunk through the existing gateway and usage consumers; assert input/output details reach `ProviderUsage`, total reaches `AgentService._usage_total_tokens`, and no fallback content is emitted.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_provider_gateway.py::test_qwencloud_responses_terminal_usage_reaches_console_signals_without_copy Tests/Chat/test_console_provider_gateway.py::test_qwencloud_total_tokens_reaches_agent_usage_counter` red, make only the smallest gateway normalization adjustment if Task 4's standard terminal chunk is not already sufficient, and rerun green.
- [ ] **Cycle 5D — cancellation red:** add `test_gateway_cancellation_closes_qwencloud_iterator` and `test_tee_tool_calls_closes_underlying_iterator_once`. Cover async consumer cancellation, normal exhaustion, and provider exception.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_provider_gateway.py::test_gateway_cancellation_closes_qwencloud_iterator Tests/Chat/test_console_provider_gateway.py::test_tee_tool_calls_closes_underlying_iterator_once` red, retain the response/iterator in `_stream_generic_chat` and close it in `finally`, and make `_tee_tool_calls` forward `.close()` idempotently. Rerun green.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_qwencloud_provider_contract.py -k "qwencloud or chat_api_kwargs or tee_tool_calls or cancel or usage"
  ```

  Expected: pass.

- [ ] Run the complete gateway regression suite outside the managed sandbox if its localhost fixtures cannot bind:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_provider_gateway.py
  ```

  Expected: pass.

- [ ] Commit:

  ```bash
  git add tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_qwencloud_provider_contract.py
  git commit -m "feat(qwencloud): pin Console mode and close streams"
  ```

### Task 6: Prove native function-tool continuation in both modes, then enable the provider

**Files:**

- Modify: `tldw_chatbook/Agents/native_tools.py`
- Modify: `Tests/Agents/test_native_tools.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`
- Create: `Tests/Chat/test_qwencloud_native_tools.py`

- [ ] Use the real direction and current class names:

  ```text
  ConsoleAgentBridge.run_reply
    -> AgentService / agent_runtime
    -> _StreamingModelAdapter.chat_call
    -> ConsoleProviderGateway.stream_chat
    -> chat_api_call
    -> chat_with_qwencloud
    -> scripted local HTTP boundary
  ```

  There is no `GatewayChatAdapter`. Do not introduce one. The bridge may be invoked through `ConsoleAgentBridge.run_reply`; narrower cases may instantiate the existing private `_StreamingModelAdapter` exactly as current bridge tests do.

- [ ] **Cycle 6A — native eligibility red:** add `test_native_provider_contract_requires_qwencloud_dispatch_and_history` to the shared invariant. Assert tools forward, normalized `message.tool_calls` return, and canonical continuation is accepted. Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Agents/test_native_tools.py::test_native_provider_contract_requires_qwencloud_dispatch_and_history` and expect failure while `qwencloud` is absent from `NATIVE_TOOLS_PROVIDERS`; do not add membership yet.
- [ ] **Cycle 6B — exact joined continuation red:** add `test_console_agent_bridge_runs_qwencloud_two_call_continuation` parameterized over both modes. Invoke `ConsoleAgentBridge.run_reply` with the existing catalog/executor and a scripted local HTTP server. First turn emits calls A/B, the real runtime executes them, second request contains exact canonical continuation/no synthetic user row, and final turn returns a unique text sentinel.
- [ ] For Cycles 6B/6C only, the test fixture must temporarily monkeypatch `tldw_chatbook.Agents.native_tools.NATIVE_TOOLS_PROVIDERS` to include `qwencloud`. This bypasses only the capability gate so the proof can exercise the unchanged runtime before permanent eligibility; it must not patch `provider_supports_native_tools`, `AgentService`, `_StreamingModelAdapter`, the gateway, dispatcher, adapter, or HTTP boundary.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_qwencloud_native_tools.py::test_console_agent_bridge_runs_qwencloud_two_call_continuation
  ```

  Expected: fail at the first missing integration contract. Make only seam-compatible fixes in the adapter/gateway, never in `agent_runtime`, until both parameters pass. Responses must emit call A/output A, call B/output B even if runtime result rows are B/A; Chat must preserve the assistant batch plus exact tool IDs and `preserve_thinking=false`.
- [ ] **Cycle 6C — errors/cancellation/budget red:** add `test_qwencloud_tool_error_continues_structurally`, `test_qwencloud_partial_call_cancellation_never_executes`, and `test_qwencloud_responses_usage_enforces_agent_budget`, each parameterized where applicable. Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_qwencloud_native_tools.py::test_qwencloud_tool_error_continues_structurally Tests/Chat/test_qwencloud_native_tools.py::test_qwencloud_partial_call_cancellation_never_executes Tests/Chat/test_qwencloud_native_tools.py::test_qwencloud_responses_usage_enforces_agent_budget` red, then fix only adapter/gateway normalization or closure. Assert structured tool error continuation, cancelled state, closed iterator/response, zero partial executions, no unpaired output, and existing token-budget stop.
- [ ] **Cycle 6D — enable provider:** when Cycles 6B/6C are green under only that temporary capability fixture, add `qwencloud` to `NATIVE_TOOLS_PROVIDERS`, remove the monkeypatch fixture from Cycles 6B/6C, and rerun Cycles 6A/6B/6C unpatched. All must pass through permanent production eligibility. This membership change is the last production change in the task.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_qwencloud_native_tools.py Tests/Chat/test_console_agent_bridge.py Tests/Agents/test_native_tools.py
  ```

  Expected: pass in both API modes.

- [ ] Run the native/gateway regression set:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Agents/test_native_tools.py Tests/Agents/test_agent_service.py Tests/Agents/test_agent_runtime.py Tests/Chat/test_console_provider_gateway.py
  ```

  If exact Agent test filenames differ, use `rg --files Tests/Agents | sort` to select the existing AgentService/runtime suites; do not silently omit them. Expected: pass outside any known baseline failures.

- [ ] Commit:

  ```bash
  git add tldw_chatbook/Agents/native_tools.py Tests/Agents/test_native_tools.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_qwencloud_native_tools.py
  git commit -m "feat(qwencloud): enable native function tools in both modes"
  ```

### Task 7: Add the canonical Settings `api_mode` control with provider-isolated drafts

**Files:**

- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `Tests/UI/test_settings_save_commit_models.py`
- Create: `Tests/UI/test_settings_qwencloud_api_mode.py`

- [ ] Keep the existing single category-level `SettingsDraft`; do not invent a second draft object. Add a namespaced key helper such as `_provider_api_mode_draft_key(provider) -> "provider_api_mode:<normalized-provider>"` and store its original/value in the Providers & Models draft. Add `_provider_api_mode_value(provider)` to read the namespaced draft first, then that provider's saved config, then `responses` only for QwenCloud.
- [ ] On selector change, stage `provider_api_mode:qwencloud`. On provider switch, snapshot the current Qwen widget first and let `_clear_provider_auxiliary_draft_keys()` clear its existing endpoint/credential/profile keys while deliberately retaining every `provider_api_mode:*` namespace. The newly selected provider's widget loads through `_provider_api_mode_value`.
- [ ] Saving QwenCloud validates/writes `api_settings.qwencloud.api_mode` and removes only `provider_api_mode:qwencloud` from `values` and `originals` after success. Saving another provider neither writes nor clears that namespaced Qwen draft; if the existing save path rebuilds/pops the category draft, preserve and restore the namespace around that operation. Category Revert clears the namespace with the rest of the category draft and reloads saved Qwen mode. Provider test/readiness overlays the selected Qwen namespaced value without persisting it.
- [ ] **Cycle 7A — render/load red:** add `test_qwencloud_api_mode_selector_visibility_options_and_default` and `test_qwencloud_api_mode_loads_saved_chat_completions`. Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_settings_qwencloud_api_mode.py::test_qwencloud_api_mode_selector_visibility_options_and_default Tests/UI/test_settings_qwencloud_api_mode.py::test_qwencloud_api_mode_loads_saved_chat_completions` red, add `Select(..., id="settings-provider-api-mode")` to the canonical provider detail form plus loaded/display values, and rerun green. The selector exists and is enabled only for normalized QwenCloud identity.
- [ ] **Cycle 7B — draft isolation red:** add `test_qwencloud_api_mode_draft_survives_provider_switch` and `test_saving_other_provider_never_mutates_qwencloud_mode`. Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_settings_qwencloud_api_mode.py::test_qwencloud_api_mode_draft_survives_provider_switch Tests/UI/test_settings_qwencloud_api_mode.py::test_saving_other_provider_never_mutates_qwencloud_mode` red, implement the namespaced `SettingsDraft` key and switch/save preservation rules above, then rerun green. Saving QwenCloud writes only its provider table plus already-owned category fields.
- [ ] **Cycle 7C — validation/save/revert red:** add `test_qwencloud_api_mode_save_and_revert_exact_values` and `test_invalid_persisted_qwencloud_mode_blocks_save_and_send`. Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_settings_qwencloud_api_mode.py::test_qwencloud_api_mode_save_and_revert_exact_values Tests/UI/test_settings_qwencloud_api_mode.py::test_invalid_persisted_qwencloud_mode_blocks_save_and_send` red, wire Task 2 validation into the existing category validation/save/revert flow, and rerun green.
- [ ] **Cycle 7D — ownership/help red:** add `test_qwencloud_api_mode_field_guide_describes_mode_contract`. Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_settings_qwencloud_api_mode.py::test_qwencloud_api_mode_field_guide_describes_mode_contract` red. Update the provider category ownership record with `api_settings.<provider>.api_mode` and explain Responses stateless `store=false`, Chat thinking replay disabled, existing function tools supported, and QwenCloud built-ins excluded; rerun green.
- [ ] Do not touch `UI/Tools_Settings_Window.py` or `Widgets/enhanced_settings_sidebar.py` in any cycle.
- [ ] Run the new focused suite:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_settings_qwencloud_api_mode.py
  ```

  Expected: pass.

- [ ] Run Settings save-model regressions:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_settings_save_commit_models.py Tests/UI/test_settings_configuration_hub.py -k "provider or ownership or qwencloud"
  ```

  Expected: all new QwenCloud cases pass. Compare any of the four recorded baseline failures with the identical baseline command; do not count them as feature failures unless behavior changed.

- [ ] Run the complete canonical Settings files, record the result, and confirm there are no failures beyond the four baseline names:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py Tests/UI/test_settings_qwencloud_api_mode.py
  ```

- [ ] Commit:

  ```bash
  git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py Tests/UI/test_settings_qwencloud_api_mode.py
  git commit -m "feat(settings): add QwenCloud API mode selector"
  ```

### Task 8: Join QwenCloud to the existing cached model-catalog pipeline

**Files:**

- Modify: `tldw_chatbook/LLM_Provider_Catalog/model_catalog_settings.py`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_provider_identity.py`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py`
- Modify: `Tests/LLM_Provider_Catalog/test_model_catalog_settings.py`
- Modify: `Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py`
- Modify: `Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py`
- Modify: `Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py`
- Modify: `Tests/Provider/test_provider_model_resolution.py`
- Modify: `Tests/UI/test_provider_model_resolution.py`

- [ ] **Cycle 8A — identity red:** add `test_qwencloud_is_seventh_auto_refresh_cloud_provider` and `test_qwencloud_model_discovery_identity_uses_qwencloud`. Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Provider_Catalog/test_model_catalog_settings.py::test_qwencloud_is_seventh_auto_refresh_cloud_provider Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py::test_qwencloud_model_discovery_identity_uses_qwencloud` red, add QwenCloud to `AUTO_REFRESH_PROVIDER_LIST_KEYS`, `_MODEL_DISCOVERY_PROVIDER_HANDLER_KEYS`, and `_BASE_URL_INFERABLE_PROVIDER_KEYS`, then rerun green.
- [ ] **Cycle 8B — URL/credential seams red:** add `test_qwencloud_models_url_normalizes_base_and_both_request_endpoints` to the pure discovery module and `test_qwencloud_discovery_uses_only_its_modern_or_environment_key` to `test_local_llm_provider_catalog_service.py`. The URL test asserts `/compatible-mode/v1`, `/compatible-mode/v1/responses`, and `/compatible-mode/v1/chat/completions` all become exactly `/compatible-mode/v1/models`. The service test supplies conflicting Qwen modern/env and other-provider keys, invokes the real service resolution, and asserts the discovery client receives only the correctly prioritized QwenCloud key.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py::test_qwencloud_models_url_normalizes_base_and_both_request_endpoints` red. Update `_models_path_for_endpoint_path()` and explicit-compatible-path recognition so the Qwen-compatible base and `/responses` suffix are recognized alongside `/chat/completions`; preserve existing hosts and strip query/fragment through the existing safe URL builder. Rerun green.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py::test_qwencloud_discovery_uses_only_its_modern_or_environment_key` red. Wire QwenCloud through the service's existing `_resolve_api_key`/readiness ownership without moving credential policy into URL construction; rerun green.
- [ ] **Cycle 8C — cache/consumer red:** add `test_qwencloud_catalog_normalization_cache_fallback_and_write_through`, `test_qwencloud_discovered_models_use_capped_selector_merge`, and `test_qwencloud_full_catalog_remains_searchable_in_model_popover`. Prove the 50-model selector cap and full Alt+M/search catalog use the ordinary shared merge. Include malformed/empty discovery and configured/cached fallback.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py::test_qwencloud_catalog_normalization_cache_fallback_and_write_through
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Provider/test_provider_model_resolution.py::test_qwencloud_discovered_models_use_capped_selector_merge
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_provider_model_resolution.py::test_qwencloud_full_catalog_remains_searchable_in_model_popover
  ```

  Expected: fail before QwenCloud flows through shared resolution. Add only standard registry/fixture handling; do not create a Qwen-specific cache, selector, refresh worker, or source. Rerun green.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Provider_Catalog/test_model_catalog_settings.py Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py Tests/Provider/test_provider_model_resolution.py Tests/UI/test_provider_model_resolution.py Tests/test_config_model_catalog_defaults.py
  ```

  Expected: pass.

- [ ] Commit:

  ```bash
  git add tldw_chatbook/LLM_Provider_Catalog/model_catalog_settings.py tldw_chatbook/LLM_Provider_Catalog/model_discovery_provider_identity.py tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py Tests/LLM_Provider_Catalog/test_model_catalog_settings.py Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py Tests/Provider/test_provider_model_resolution.py Tests/UI/test_provider_model_resolution.py
  git commit -m "feat(qwencloud): add cached model discovery"
  ```

### Task 9: Document, optionally verify live, and complete repository evidence

**Files:**

- Modify: `Docs/superpowers/specs/2026-08-02-qwencloud-dual-api-provider-design.md` only if implementation evidence requires a factual clarification
- Modify: `README.md`
- Modify: `Docs/User_Guide/settings.md`
- Modify: `Docs/User_Guide/console.md`
- Create: `Tests/Chat/test_live_qwencloud_api.py`
- Modify: `backlog/tasks/task-3771 - Add-QwenCloud-dual-API-provider-support.md`
- Modify: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md` only if this implementation produced a genuinely reusable incident-backed lesson

- [ ] Add provider documentation covering `DASHSCOPE_API_KEY`, the international default and regional endpoint guidance, model default, both exact `api_mode` values and default, stateless Responses/`store=false`, Chat reasoning replay safety, parameter limits, existing function-tool support, built-in-tool exclusion, model discovery, and unknown-pricing behavior.
- [ ] Add a default-skipped live smoke test requiring both `TLDW_LIVE_QWENCLOUD=1` and `DASHSCOPE_API_KEY`. Parameterize both modes, ask the model to reproduce a harmless unique sentinel such as `QWENCLOUD_LIVE_TEXT_<random>` and assert that identifying content, then run one harmless existing function tool whose result contains a second unique sentinel and assert that result influences the final answer. Isolate config/data directories so the test never reads or writes the user's real config.
- [ ] Prove the live test is collected but skipped by default:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_live_qwencloud_api.py
  ```

  Expected: skipped unless the explicit gate and key are present.

- [ ] If the user explicitly authorizes paid live verification and the gate/key are available, run with isolated temporary configuration. The test itself must identify the harmless sentinels rather than merely observe no exception; implementation notes record only pass/fail shape and usage, never private prompt/response content or keys. Otherwise document that live verification was not run; do not treat that as an automated-test failure.
- [ ] Run formatting/static checks scoped to changed Python files using the repository's configured tools. At minimum:

  ```bash
  git diff --check
  git diff --name-only -z --diff-filter=ACM 97a75fb8b -- '*.py' | xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check
  git diff --name-only -z --diff-filter=ACM 97a75fb8b -- '*.py' | xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy tldw_chatbook/LLM_Calls/qwencloud.py tldw_chatbook/LLM_Calls/qwencloud_streaming.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook/LLM_Calls/qwencloud.py tldw_chatbook/LLM_Calls/qwencloud_streaming.py
  ```

- [ ] Run the complete QwenCloud and touched-surface regression suite:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_qwencloud.py Tests/LLM_Calls/test_qwencloud_streaming.py Tests/Chat/test_qwencloud_provider_contract.py Tests/Chat/test_qwencloud_native_tools.py Tests/Chat/test_live_qwencloud_api.py Tests/Chat/test_chat_unit_mocked_APIs.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_agent_bridge.py Tests/Agents/test_native_tools.py Tests/Chat/test_provider_readiness.py Tests/Chat/test_console_provider_endpoints.py Tests/Chat/test_console_provider_support.py Tests/Chat/test_sensitive_llm_logging.py Tests/LLM_Provider_Catalog/test_model_catalog_settings.py Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py Tests/Provider/test_provider_model_resolution.py Tests/UI/test_provider_model_resolution.py Tests/test_config_model_catalog_defaults.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py Tests/UI/test_settings_qwencloud_api_mode.py
  ```

  Expected: all non-live tests pass and live tests skip by default. Run outside the managed sandbox if localhost fixtures need to bind.

- [ ] Run the whole repository suite with the absolute venv interpreter:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q
  ```

  Compare any failure with an identical clean-base command. Resolve every QwenCloud/touched-surface regression before closeout; record unrelated baseline failures precisely rather than claiming a blanket pass.

- [ ] Self-review the diff against every acceptance criterion and ADR-045. Confirm no built-in QwenCloud tools, no stateful Responses fields, no legacy Settings changes, no unverified pricing, no secret-bearing logs, and no partial tool execution on cancellation.
- [ ] Update TASK-3771 only after evidence is complete: check every acceptance criterion, add concise Implementation Notes with test evidence and the ADR-045 link, document deviations/baseline failures, and set status Done with the Backlog CLI. If any criterion is incomplete, leave the task In Progress.
- [ ] Commit final documentation/task evidence:

  ```bash
  git add README.md Docs/User_Guide/settings.md Docs/User_Guide/console.md Docs/superpowers/specs/2026-08-02-qwencloud-dual-api-provider-design.md Docs/superpowers/plans/2026-08-11-qwencloud-dual-api-provider-implementation.md backlog/decisions/045-qwencloud-dual-api-provider-boundary.md "backlog/tasks/task-3771 - Add-QwenCloud-dual-API-provider-support.md" Tests/Chat/test_live_qwencloud_api.py
  git commit -m "docs(qwencloud): document setup and verification"
  ```

## Final Acceptance Checklist

- [ ] One `qwencloud` provider behaves like the existing hosted providers across dispatcher, readiness, Console, Settings, errors, usage, and model discovery.
- [ ] `api_mode` is explicit, validated, defaults to `responses`, and remains pinned with the base URL throughout a run.
- [ ] Streaming and non-streaming text/tool/finish/usage work in Responses and Chat Completions.
- [ ] Existing function tools complete multiple-call continuation through the real runtime in both modes; QwenCloud built-ins remain unsupported.
- [ ] Responses history is stateless and exactly paired/adjacent; Chat history cannot replay thinking.
- [ ] Streaming is record-aware, de-duplicated, safely terminal, non-retrying after consumption, and closeable on cancellation.
- [ ] Canonical Settings persistence is provider-isolated and model discovery uses the shared cached catalog.
- [ ] Automated tests are green beyond documented identical-base failures, sensitive logging is clean, optional live calls remain gated, ADR-045 is linked, and TASK-3771 hygiene is complete.
