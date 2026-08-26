# Kimi And Z.ai Hosted Chat-Completions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the existing Moonshot/Kimi and Z.ai/GLM integrations into first-class hosted Chat-Completions providers with current defaults, strict streaming and non-streaming behavior, existing Chatbook function tools, durable private reasoning continuation, Settings, and cached model discovery.

**Architecture:** Preserve the stable `moonshot` and `zai` identities and public handler signatures. Extract only proven-common HTTP ownership/retry, SSE framing, and OpenAI-shaped Chat normalization into a small provider-neutral boundary. Moonshot and Z.ai consume the full boundary; later DeepSeek uses the route-neutral HTTP lifecycle for both modes and the Chat normalization for Chat Completions. QwenCloud reuses only framing/normalization where parity is proven—its existing transport remains unchanged. Request allowlists, model-family policy, thinking/reasoning, finishes, and canonical-checkpoint-to-wire translation stay in dedicated provider adapters.

**Tech Stack:** Python 3.11+, `requests`/`urllib3`, strict incremental SSE/JSON parsing, Textual 8.x Settings, existing Console gateway/agent runtime/model catalog, pytest/pytest-asyncio, and optional subprocess-isolated live tests.

---

## Design Sources And ADR Check

- Approved design: `Docs/superpowers/specs/2026-08-12-kimi-zai-hosted-chat-completions-design.md`
- Canonical decision: `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md`
- Foundation dependency: TASK-15675 and its schema/runtime/export implementation.
- Backlog source of truth: `backlog/tasks/task-15676 - Harden-Moonshot-Kimi-and-Z.ai-GLM-as-first-class-hosted-providers.md`
- Official sources linked from the approved design: Kimi docs/OpenAPI and Z.ai docs/OpenAPI.

ADR required: yes

ADR path: `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md`

Reason: This PR implements ADR-063's reusable hosted Chat wire boundary and provider-specific continuation policies. No additional decision is needed.

## Scope Guardrails

- Chat Completions only. Do not add `api_mode`, `/responses`, server conversation IDs, or provider-managed session state.
- Support existing Chatbook function tools only. Reject Kimi/Z.ai hosted web search, retrieval, code runner, dynamic tools, and private top-level tool metadata.
- Do not create a broad “provider core.” Share only URL validation, route-neutral HTTP/SSE ownership, generic OpenAI-shaped choices/tool fragments/usage validation, and retry/error mechanics with concrete consumers. DeepSeek consumes the same HTTP lifecycle in TASK-15677; QwenCloud transport is not migrated.
- Provider-specific request keys, model capabilities, reasoning, thinking, finish states, and history shaping stay in `moonshot.py` and `zai.py`.
- If QwenCloud parity needs provider flags with no second consumer, narrow the extraction rather than generalizing it.
- Preserve explicit historical model selections. Update only fresh/missing defaults to `kimi-k3` and `glm-5.2`.
- Keep legacy Settings screens untouched; only F9 `settings_screen.py` changes.
- No paid request in default tests; live tests require both provider-specific gate and nonblank key.

## Branch And Baseline Discipline

- [x] Begin only after TASK-15675 is merged. Create a fresh `codex/kimi-zai-hosted-chat` branch/worktree from current `origin/dev`; do not reuse the foundation worktree.
- [x] Put TASK-15676 In Progress before production edits, then add a structured Implementation Plan section to its task file that links this document and ADR-063; do not replace it with a one-line CLI plan:

  ```bash
  backlog task edit 15676 -s "In Progress"
  # Use apply_patch to add the ordered plan and ADR required/path/reason block.
  backlog task 15676 --plain
  ```

- [x] Record clean-base results for QwenCloud, Moonshot/Z.ai mocked calls, gateway/native tools, Settings, and model catalog. Reproduce any localhost-only failures outside the socket sandbox.
- [x] Every cycle must be observed RED at the named behavior before implementation and GREEN afterward. Use captured official-shaped fixtures and real loopback HTTP for transport/joined tests; do not mock away dispatcher/gateway/runtime boundaries.

## Shared Interfaces To Implement

Create only these neutral interfaces:

```python
# tldw_chatbook/LLM_Calls/hosted_chat.py
@dataclass(frozen=True)
class HostedHTTPTransportConfig:
    provider: str
    base_url: str
    api_key: str = field(repr=False, compare=False)
    timeout: float
    retries: int
    retry_delay: float

def normalize_hosted_chat_base_url(value: object, *, default: str) -> str: ...

def owned_json_post(
    *,
    config: HostedHTTPTransportConfig,
    route: Literal["chat/completions", "responses"],
    payload: Mapping[str, Any],
    streaming: bool,
) -> dict[str, Any] | "OwnedSSEStream": ...

class HostedChatFinishPolicy(Protocol):
    def validate_finish(
        self, *, finish_reason: object, has_text: bool, has_calls: bool
    ) -> str: ...

    def validate_reasoning_content(self, value: object) -> str | None: ...

@dataclass(frozen=True)
class HostedChatTurn:
    text: str
    tool_calls: tuple[dict[str, Any], ...]
    assistant_message: dict[str, Any] | None
    finish_reason: str
    reasoning_content: str | None = field(default=None, repr=False)
    usage: dict[str, Any] | None = field(default=None, repr=False)

def hosted_chat_request(
    *,
    config: HostedHTTPTransportConfig,
    payload: Mapping[str, Any],
    streaming: bool,
    finish_policy: HostedChatFinishPolicy,
) -> HostedChatTurn | "HostedChatStream": ...
```

```python
# tldw_chatbook/LLM_Calls/hosted_chat_streaming.py
@dataclass(frozen=True)
class SSERecord:
    event: str | None
    data: str

class SSERecordDecoder:
    def feed(self, chunk: bytes) -> tuple[SSERecord, ...]: ...
    def finish(self) -> tuple[SSERecord, ...]: ...

class OwnedSSEStream(Iterator[SSERecord]):
    def close(self) -> None: ...

class HostedChatStream(Iterator[dict[str, Any]]):
    @property
    def terminal_turn(self) -> HostedChatTurn: ...

    def close(self) -> None: ...
```

The decoder preserves the exact optional SSE `event:` label and joined `data:` text; it ignores comments plus `id`, `retry`, and unknown non-data fields. `HostedChatStream` ignores `SSERecord.event` for OpenAI-shaped Chat, but DeepSeek Responses consumes it in TASK-15677. The neutral layer receives an already-built payload and narrow finish/reasoning-shape policy. It does not import config, Console, Settings, catalog, native tools, provider adapters, or the canonical continuation module. `owned_json_post` alone owns one `requests.Session`, retry budget, response/body lifecycle, typed redaction, and ownership transfer for either allowed route. Chat normalization wraps that primitive and returns bounded typed reasoning content; each provider adapter alone converts that result plus its canonical owner group to/from `ProviderContinuationCheckpoint`. DeepSeek Responses later wraps the same `OwnedSSEStream` with its provider-specific semantic translator. The layer validates generic OpenAI Chat choice/delta/tool-fragment/usage shapes but never invents a provider request field. Add repr/exception/log canary tests proving the API key cannot escape `HostedHTTPTransportConfig`.

`HostedChatStream.terminal_turn` is available only after one valid terminal finish plus required terminal usage and clean `[DONE]`; before that—including cancellation, explicit close, EOF, or provider error—it raises a context-free typed incomplete-stream error and cannot produce provider metadata. A Moonshot/Z.ai provider-local stream wrapper delegates iteration/close, then converts this terminal turn to `ProviderTurnMetadata`. The gateway reads that provider-local terminal metadata only after normal exhaustion and emits one final `ProviderToolCalls` sentinel; cancellation before terminal closes once and emits neither sentinel nor checkpoint.

### Task 1: Extract Minimal Hosted Chat Framing And Transport With Qwen Parity

**Files:**

- Create: `tldw_chatbook/LLM_Calls/hosted_chat.py`
- Create: `tldw_chatbook/LLM_Calls/hosted_chat_streaming.py`
- Create: `Tests/LLM_Calls/test_hosted_chat.py`
- Create: `Tests/LLM_Calls/test_hosted_chat_streaming.py`
- Modify: `tldw_chatbook/LLM_Calls/qwencloud_streaming.py`
- Modify: `Tests/LLM_Calls/test_qwencloud_streaming.py`

- [x] **Cycle 1A — URL RED:** add exact tests for HTTP(S), authority, userinfo/query/fragment, controls/backslashes, empty/doubled segments, terminal lowercase `/chat/completions`, `/responses`, case/lookalikes, repeated/stacked tails, encoded separators/dot segments/reserved tails, safe encoded ordinary data, IPv6/port, and the 2,000-character pre-parse cap. Implement the pure structural helper and assert source immutability.
- [x] The helper strips at most one exact terminal lowercase `/chat/completions`; it rejects `/responses` and ambiguous request tails. Chat URL is `{base}/chat/completions`; discovery later uses `{base}/models`.
- [x] **Cycle 1B — SSE framing RED:** copy the existing QwenCloud captured framing corpus into provider-neutral record tests: split UTF-8, CR/LF/CRLF, comments, ignored `id`/`retry`/unknown non-data fields, exact optional `event:` preservation, multiline `data`, blank-record dispatch, truncated EOF, line/record/event/depth/node/byte caps, and linear segment accumulation. Implement `SSERecordDecoder` without provider finish semantics.
- [x] **Cycle 1C — generic Chat normalization RED:** add text-only, tool-only, mixed text/tool, multiple interleaved call indexes, nullable fields, usage-only terminal, exact replay, finish disagreement, malformed arguments, deep JSON, and post-terminal cases. Keep provider finish sets and `reasoning_content` admission behind narrow supplied policies.
- [x] **Cycle 1D — route-neutral transport/lifecycle RED:** real loopback tests assert exact route, POST URL/headers/body/timeout; one global `retries+1` budget; POST-only 429/500/502/503/504 plus connect/timeout; integer/date/malformed Retry-After; no retry after a body byte or any 2xx body read; typed 401/403/429/4xx/5xx/network/malformed errors; and response/session closure exactly once on normal, error, cancellation, explicit/repeated close, and cleanup failures. Exercise both allowed routes without adding Responses semantics here.
- [x] **Cycle 1E — Qwen compatibility gate:** route only QwenCloud Chat SSE framing/generic normalization through the extracted primitives. Keep `qwencloud.py` transport and QwenCloud Responses untouched. Run the entire existing adapter/streaming/native suite before and after. Add a shared captured-event parity test and mutation-test one tool index or terminal finish; the mutation must fail.
- [x] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_hosted_chat_streaming.py Tests/LLM_Calls/test_qwencloud.py Tests/LLM_Calls/test_qwencloud_streaming.py Tests/Chat/test_qwencloud_native_tools.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/hosted_chat_streaming.py tldw_chatbook/LLM_Calls/qwencloud_streaming.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/hosted_chat_streaming.py
  ```

- [x] Commit:

  ```bash
  git add tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/hosted_chat_streaming.py tldw_chatbook/LLM_Calls/qwencloud_streaming.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_hosted_chat_streaming.py Tests/LLM_Calls/test_qwencloud_streaming.py
  git commit -m "refactor(llm): extract hosted chat wire boundary"
  ```

### Task 2: Build The Moonshot/Kimi Adapter And Current Model Policies

**Files:**

- Create: `tldw_chatbook/LLM_Calls/moonshot.py`
- Create: `Tests/LLM_Calls/test_moonshot.py`
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`
- Modify: `tldw_chatbook/Chat/Chat_Functions.py`
- Modify: `Tests/Chat/test_chat_unit_mocked_APIs.py`

- [x] Define pure public helpers:

  ```python
  def resolve_moonshot_request(
      *,
      explicit_api_key: object = None,
      explicit_base_url: object = None,
      explicit_model: object = None,
      app_config: Mapping[str, Any] | None = None,
      environ: Mapping[str, str] | None = None,
  ) -> MoonshotResolution: ...

  def build_moonshot_chat_payload(
      *,
      resolution: MoonshotResolution,
      messages_payload: Sequence[Mapping[str, Any]],
      system_message: object = None,
      streaming: object = False,
      tools: object = None,
      tool_choice: object = None,
      reasoning_effort: object = None,
      provider_continuations: Sequence[ProviderContinuationCheckpoint] = (),
      **generic: object,
  ) -> dict[str, Any]: ...
  ```

- [x] **Cycle 2A — resolution RED:** canonical `api_settings.moonshot` beats its environment; explicit args beat config; configured env name falls back to `MOONSHOT_API_KEY`; malformed canonical table/blank/placeholder key/model/base/timeout/retry fields fail before lower-priority reads or network. `api_region` is fallback only when canonical base is absent. Remove the orphaned top-level `moonshot_api` lookup rather than creating a legacy owner.
- [x] **Cycle 2B — message/tool RED:** validate/deep-copy system/user/assistant/tool history, exact call/result pairing and ordering, unique IDs across outbound history, standard function schema, and Moonshot tool choices absent/`auto`/`none`/`required`/forced function. Reject provider-built-in tool types/private metadata and malformed/cross-batch/orphan results before I/O.
- [x] **Cycle 2C — K3 allowlist RED:** for `kimi-k3`, allow only `model`, `messages`, `stream`, `max_completion_tokens`, `stop`, `response_format`, `tools`, `tool_choice`, `reasoning_effort`, and adapter-owned `stream_options`. Accept effort `low|high|max`; map generic `max_tokens`; set `include_usage` only for streaming; omit valid unsupported generic sampler defaults but reject invalid supplied values.
- [x] **Cycle 2D — family policy RED:** cover curated `moonshot-v1-*` documented sampler ranges/relationships and conservative unknown-model subset. Do not infer capabilities from substrings or newly discovered IDs. Preserve explicit historical IDs.
- [x] **Cycle 2E — continuation RED:** expand TASK-15675 checkpoints so K3 replays every retained exact `reasoning_content`; active/restored supported Kimi tool runs replay only documented content. A tool-free K3 final response yields a complete reasoning-only round tied to the same visible owner. Assert no visible/log/error/usage/human-export leakage.
- [x] **Cycle 2F — response RED:** run nonstream and stream official-shaped fixtures through the neutral boundary. Enforce Moonshot finishes `stop|tool_calls|length`, mixed text/tools, exact terminal usage, malformed/empty success rejection, typed errors, cancellation, and strict budget usage.
- [x] Replace the large `LLM_API_Calls.py` function body with a compatibility import/wrapper preserving the public signature and metrics/provider labels. Keep dispatcher parameter mapping explicit.
- [x] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_moonshot.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_hosted_chat_streaming.py Tests/Chat/test_chat_unit_mocked_APIs.py -k "moonshot or hosted_chat"
  ```

- [x] Commit:

  ```bash
  git add tldw_chatbook/LLM_Calls/moonshot.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/Chat/Chat_Functions.py Tests/LLM_Calls/test_moonshot.py Tests/Chat/test_chat_unit_mocked_APIs.py
  git commit -m "feat(moonshot): add strict Kimi chat adapter"
  ```

### Task 3: Build The Z.ai/GLM Adapter And Enable Native Tools Last

**Files:**

- Create: `tldw_chatbook/LLM_Calls/zai.py`
- Create: `Tests/LLM_Calls/test_zai.py`
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`
- Modify: `tldw_chatbook/Chat/Chat_Functions.py`
- Modify: `tldw_chatbook/Agents/native_tools.py`
- Modify: `Tests/Agents/test_native_tools.py`
- Modify: `Tests/Chat/test_chat_unit_mocked_APIs.py`

- [x] Mirror Task 2's pure resolution/builder split, using exact canonical `api_settings.zai`, `ZAI_API_KEY`, general base `https://api.z.ai/api/paas/v4`, and default `glm-5.2`. Malformed exact tables block before environment/network and source mappings remain unchanged.
- [x] **Cycle 3A — allowlist RED:** allow only `model`, `messages`, `do_sample`, `stream`, `thinking`, `temperature`, `top_p`, `reasoning_effort`, `max_tokens`, `tools`, `tool_choice`, `stop`, `response_format`, `request_id`, and generic user identifier mapped to `user_id`. `glm-5.2` effort values are exactly `none|minimal|low|medium|high|xhigh|max`; other curated models own explicit policies. Unknown kwargs are omitted.
- [x] **Cycle 3B — thinking/tool RED:** ordinary/tool-free chat sends `clear_thinking=true`; only an active or explicitly restored function-tool run sends `thinking.clear_thinking=false` and exact checkpoint reasoning. Accept only function tools and tool choice absent/`auto`; reject forced/required/none/provider tool types. Keep `tool_stream` absent.
- [x] **Cycle 3C — response RED:** enforce usable finishes `stop|tool_calls|length`; map `sensitive`, `model_context_window_exceeded`, and `network_error` to safe terminal provider errors. Normalize bounded object arguments deterministically to a JSON string; reject other scalar/container shapes, incomplete calls, empty success, unknown/blank/conflicting finishes, and malformed usage.
- [x] Replace the legacy function body with a compatibility wrapper and keep Z.ai out of the native registry while its direct/gateway tests are developed.
- [x] **Cycle 3D — eligibility RED/GREEN:** add a registry test that is initially RED for missing `zai`. Run joined tool tests under a temporary capability fixture first; only after forwarding/history/cancellation/closure pass, add `zai` to `NATIVE_TOOLS_PROVIDERS`, remove the fixture, and rerun unpatched. The membership change is this task's last production change.
- [x] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_zai.py Tests/Agents/test_native_tools.py Tests/Chat/test_chat_unit_mocked_APIs.py -k "zai or hosted_chat or native_provider_contract"
  ```

- [x] Commit:

  ```bash
  git add tldw_chatbook/LLM_Calls/zai.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/Chat/Chat_Functions.py tldw_chatbook/Agents/native_tools.py Tests/LLM_Calls/test_zai.py Tests/Agents/test_native_tools.py Tests/Chat/test_chat_unit_mocked_APIs.py
  git commit -m "feat(zai): add strict GLM chat adapter"
  ```

### Task 4: Pin Provider Resolution And Carry Continuation/Usage Through Console

**Files:**

- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`
- Create: `Tests/Chat/test_kimi_zai_provider_contract.py`

- [x] **Cycle 4A — frozen handoff RED:** assert resolution freezes provider/model/base/key/timeout/retries/retry delay for the whole send and auxiliary calls. Mutate config and environment between model turns; exact pinned values remain. Do not add `api_mode` for either provider.
- [x] Extend the existing immutable Console resolution/kwargs paths with timeout/retry policy only if not already carried; do not create provider-specific Console resolution types. Provider-local frozen `MoonshotResolution`/`ZAIResolution` remain required at the adapter boundary. Sensitive keys remain repr/log safe and are never persisted in checkpoints.
- [x] **Cycle 4B — checkpoint handoff RED:** normalized provider responses carry only typed TASK-15675 candidates into `ModelTurn`. Tool batch/result/final reasoning checkpoints use the foundation hooks; hidden reasoning never appears in stream chunks or `AgentStep` summaries.
- [x] Extend the existing final native sentinel instead of adding a metadata bag:

  ```python
  @dataclass(frozen=True)
  class ProviderTurnMetadata:
      finish_reason: str
      provider_continuation: ProviderContinuationCheckpoint | None = field(
          default=None, repr=False
      )
      usage: Mapping[str, Any] | None = field(default=None, repr=False)

  @dataclass(frozen=True)
  class ProviderToolCalls:
      tool_calls: tuple[dict[str, Any], ...]
      metadata: ProviderTurnMetadata | None = field(default=None, repr=False)
  ```

  Nonstream adapters convert `HostedChatTurn` directly. Streaming adapters expose a provider-local wrapper whose metadata accessor converts `HostedChatStream.terminal_turn` only after clean exhaustion. `ConsoleProviderGateway.stream_chat` then yields the final `ProviderToolCalls` sentinel (including an empty call tuple when an agent turn has final continuation metadata); `_StreamingModelAdapter` validates it once and constructs `ModelTurn.provider_continuation`. Add direct seam tests for `HostedChatStream.terminal_turn -> provider wrapper -> stream_chat -> ProviderToolCalls -> _StreamingModelAdapter -> ModelTurn`, including cancellation-before-terminal/no-sentinel behavior, repr/log canaries, and concurrent-call isolation.
- [x] **Cycle 4C — usage RED:** terminal Moonshot/Z.ai raw usage reaches call-scoped Console signals; strict nonnegative integer prompt/completion/total/cached fields reach AgentService budget; bool/string/float/negative/inconsistent details use the existing estimator. Concurrent calls cannot exchange usage/checkpoint state.
- [x] **Cycle 4D — cancellation/tee RED:** cancellation before return, after retention, during hidden reasoning, visible text, or partial call closes iterator/response/session exactly once and executes no incomplete call. Close/reentrant-close/cleanup failures never mask the primary outcome.
- [x] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_kimi_zai_provider_contract.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_agent_bridge.py Tests/Agents/test_agent_service.py
  ```

- [x] Commit:

  ```bash
  git add tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Agents/agent_service.py Tests/Chat/test_kimi_zai_provider_contract.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_agent_bridge.py
  git commit -m "feat(console): carry Kimi and GLM provider state"
  ```

### Task 5: Prove Joined Function-Tool Continuation Through Real HTTP

**Files:**

- Create: `Tests/Chat/test_kimi_zai_native_tools.py`
- Modify: `Tests/Agents/test_native_tools.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`

- [x] Build one temporary scripted loopback server whose request validator rejects mismatched URL, headers, payload, complete assistant call batch, reasoning, or result ordering before releasing the next response. Traverse the real chain:

  ```text
  ConsoleAgentBridge -> AgentService/agent_runtime -> _StreamingModelAdapter
  -> ConsoleProviderGateway -> chat_api_call -> provider adapter
  -> HostedChatStream -> loopback HTTP
  ```

- [x] **Cycle 5A — Moonshot joined RED:** run two calculator calls with exact IDs and results, no synthetic user row, mixed text/tool support, K3 reasoning preserved, first batch persisted before execution, final reasoning-only round on the same assistant owner, later-turn replay, and authoritative terminal usage.
- [x] **Cycle 5B — Z.ai joined RED:** prove complete multi-call batch/tool result history, `clear_thinking=false` only during the active/restored loop, ordinary next chat returns to true, exact reasoning replay for the loop, and usage budget handoff.
- [x] **Cycle 5C — error/restore RED:** structured tool failures continue; completed/failed restored calls never execute; executing restore blocks; pending requires explicit Resume plus fresh approval. Duplicate IDs and out-of-order/cross-batch results fail before server script advancement.
- [x] **Cycle 5D — partial cancellation RED:** the server emits incomplete function fragments plus a visible downstream text marker while holding a nonterminal chunked response open. Set cancellation only after the real parser/gateway/store path observes that marker and independently assert the parser had already entered incomplete-call state. Because incomplete fragments never form a checkpoint, assert no assistant continuation owner/checkpoint row, one response close, one request, zero executions/tool results/unpaired outputs, and no server timeout. A text-only fixture mutation must fail the partial-call guard.
- [x] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_kimi_zai_native_tools.py Tests/Agents/test_native_tools.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_provider_gateway.py
  ```

- [x] Commit:

  ```bash
  git add Tests/Chat/test_kimi_zai_native_tools.py Tests/Agents/test_native_tools.py Tests/Chat/test_console_provider_gateway.py
  git commit -m "test(hosted): prove Kimi and GLM native tools"
  ```

### Task 6: Update Defaults, Readiness, Settings, And Model Discovery

**Files:**

- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/Chat/provider_readiness.py`
- Modify: `tldw_chatbook/Chat/console_provider_endpoints.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py`
- Modify: `Tests/test_config_model_catalog_defaults.py`
- Modify: `Tests/Chat/test_provider_readiness.py`
- Create: `Tests/UI/test_settings_kimi_zai.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `Tests/UI/test_settings_save_commit_models.py`
- Modify: `Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py`
- Modify: `Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py`
- Modify: `Tests/Provider/test_provider_model_resolution.py`
- Modify: `Tests/UI/test_provider_model_resolution.py`

- [x] **Cycle 6A — defaults RED:** assert fresh `[providers]` lists lead with `kimi-k3` and `glm-5.2`; `[api_settings]` defaults use those models, canonical env names, general bases, timeouts/retries, and streaming true. Explicit saved old IDs remain unchanged through load/save.
- [x] **Cycle 6B — readiness parity RED:** readiness/direct/chat/discovery share canonical settings, credential precedence, and the same URL helper. Malformed exact tables, placeholders, blank models, unsafe endpoints, bad numeric fields, and alias conflicts block safely before network.
- [x] **Cycle 6C — Settings RED:** real Pilot tests cover provider switch/draft isolation, model/endpoint/credential/reasoning controls, invalid recovery, atomic save/revert, second-save no-op, and field search/focus. No API-mode selector. K3 reasoning choices are exact `low|high|max`; `glm-5.2` choices exact `none|minimal|low|medium|high|xhigh|max`. Historical/unknown models retain user choice and conservative guidance.
- [x] Reuse the existing category draft/atomic save and generic reasoning control. Do not create provider-specific settings state or touch legacy Settings surfaces.
- [x] **Cycle 6D — discovery RED:** Moonshot authenticated `{base}/models`; Z.ai best-effort `{base}/models`. Use identical resolved base/key as chat, preserve prior cache on failure, keep IDs/timestamps only, cap selector at 50, keep uncapped picker searchable, and never infer capabilities from discovered names. Z.ai discovery failure must not block chat readiness.
- [x] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/test_config_model_catalog_defaults.py Tests/Chat/test_provider_readiness.py Tests/UI/test_settings_kimi_zai.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py Tests/Provider/test_provider_model_resolution.py Tests/UI/test_provider_model_resolution.py
  ```

- [x] Commit:

  ```bash
  git add tldw_chatbook/config.py tldw_chatbook/Chat/provider_readiness.py tldw_chatbook/Chat/console_provider_endpoints.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py Tests/test_config_model_catalog_defaults.py Tests/Chat/test_provider_readiness.py Tests/UI/test_settings_kimi_zai.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py Tests/Provider/test_provider_model_resolution.py Tests/UI/test_provider_model_resolution.py
  git commit -m "feat(settings): refresh Kimi and GLM provider UX"
  ```

### Task 7: Document, Optionally Verify Live, And Close TASK-15676

**Files:**

- Modify: `README.md`
- Modify: `Docs/User_Guide/settings.md`
- Modify: `Docs/User_Guide/console.md`
- Create: `Tests/Chat/test_live_moonshot_zai_api.py`
- Modify: `backlog/tasks/task-15676 - Harden-Moonshot-Kimi-and-Z.ai-GLM-as-first-class-hosted-providers.md`
- Modify: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md` only for a genuinely new incident-backed lesson.

- [x] Document stable provider names, `MOONSHOT_API_KEY`/`ZAI_API_KEY`, current defaults and retained historical models, general/China/custom endpoints, Chat-only scope, exact reasoning/tool-choice subsets, usage, existing function tools/built-in exclusions, checkpoint privacy/context cost, discovery/cache, unknown pricing, and recovery.
- [x] Add default-skipped live cases requiring both `TLDW_LIVE_MOONSHOT=1` + key or `TLDW_LIVE_ZAI=1` + key. Each provider runs in a fresh subprocess with isolated HOME/XDG/config/data before imports, muted logs/stdout/stderr, randomized text/arithmetic, exactly one Calculator call/result, and a final marker derived only after the tool result. Assertion text contains no key/prompt/response/tool result.
- [x] Prove default collection makes no paid call:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_live_moonshot_zai_api.py
  ```

  Expected: both provider cases skipped unless their exact gate and nonblank key are present.
- [x] Run the provider/Qwen/Console/Settings/catalog surface within the final
  user-authorized test scope. **Deviation:** on 2026-08-13 the user explicitly
  stopped the broad run and required only tests for touched files or related
  functionality. The full-repository command below was therefore not run;
  focused named modules and selected continuation seams replaced it, with exact
  evidence recorded in the Backlog task.

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_hosted_chat_streaming.py Tests/LLM_Calls/test_moonshot.py Tests/LLM_Calls/test_zai.py Tests/LLM_Calls/test_qwencloud.py Tests/LLM_Calls/test_qwencloud_streaming.py Tests/Chat/test_kimi_zai_provider_contract.py Tests/Chat/test_kimi_zai_native_tools.py Tests/Chat/test_live_moonshot_zai_api.py Tests/Agents/test_native_tools.py Tests/Agents/test_agent_service.py Tests/Agents/test_agent_runtime.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_provider_readiness.py Tests/UI/test_settings_kimi_zai.py Tests/LLM_Provider_Catalog
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q
  ```

- [x] Run static/security gates:

  ```bash
  git diff --check origin/dev...HEAD
  git diff --name-only -z --diff-filter=ACM origin/dev...HEAD -- '*.py' | xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check
  git diff --name-only -z --diff-filter=ACM origin/dev...HEAD -- '*.py' | xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/hosted_chat_streaming.py tldw_chatbook/LLM_Calls/moonshot.py tldw_chatbook/LLM_Calls/zai.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook/LLM_Calls
  ```

- [x] Self-review every AC and ADR-063 rule. Confirm Qwen Responses is untouched, Qwen Chat parity is green, no other provider migrated, no API mode/vendor built-in tools/server state/new schema, and default tests made no paid calls.
- [x] Check all ACs individually, add observed Implementation Notes/evidence/deviations to the task file, verify the rendered task, and set Done only after the gates pass:

  ```bash
  backlog task edit 15676 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 --check-ac 7 --check-ac 8 --check-ac 9 --check-ac 10
  # Use apply_patch to add Implementation Notes from the observed results.
  backlog task 15676 --plain
  backlog task edit 15676 -s Done
  ```

- [x] Commit closeout:

  ```bash
  git add README.md Docs/User_Guide/settings.md Docs/User_Guide/console.md Docs/superpowers/plans/2026-08-12-kimi-zai-hosted-chat-completions-implementation.md Tests/Chat/test_live_moonshot_zai_api.py "backlog/tasks/task-15676 - Harden-Moonshot-Kimi-and-Z.ai-GLM-as-first-class-hosted-providers.md"
  git commit -m "docs(hosted): document Kimi and GLM providers"
  ```

## PR Boundary And Handoff

- This is PR 2 of 3. Open against `dev` only after TASK-15675 is merged.
- Merge before starting TASK-15677 so DeepSeek consumes the landed neutral Chat boundary rather than duplicating it.
- PR description links ADR-063, TASK-15675/15676, both approved specs/plans, the Qwen parity evidence, joined tool/cancellation evidence, optional-live status, and exact baseline-only failures.
