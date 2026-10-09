"""Single source of truth for provider identity and hosted preset data.

Leaf module: stdlib imports ONLY, so ``config.py`` can consume it without
import cycles (ADR-179). Identity fields cover every provider; preset
fields (``engine_driven``) are consumed by
``LLM_Calls.hosted_provider_engine``. Behavior never lives here.

Transcription sources (values copied verbatim; each guarded by a parity
test in ``Tests/test_provider_registry.py`` unless noted):

- ``config_key``: the ``[providers]`` table spelling, which is exactly what
  ``config.py::_cloud_provider_keys`` lists for cloud providers.
- ``api_key_env_var`` / ``api_key_env_candidates``: the
  ``api_key_env_var`` value of the provider's ``[api_settings.*]`` table in
  ``config.py`` (absent there -> ``None`` / empty).
- ``default_base_url``: the table's ``api_base_url`` when the provider's
  ``[api_settings.*]`` table defines one (huggingface, moonshot, qwencloud,
  zai); otherwise the chat path's builtin fallback from
  ``Chat/console_provider_endpoints.py::_BUILTIN_PROVIDER_ENDPOINTS``.
  Local providers configure ``api_url`` (a FULL endpoint path, not a base
  URL), so their ``default_base_url`` stays ``None``.
- ``display_name``: transcribed from the retired
  ``Chat/console_provider_support.py::_PROVIDER_DISPLAY_NAMES`` map (and its
  title-case fallback), not parity-guarded. These values are engine
  error-copy prefixes only; every UI label reads
  ``Chat/provider_catalog.py::provider_display_name`` (TASK-33002.5).
- ``native_tools``: ``Agents/native_tools.py::NATIVE_TOOLS_PROVIDERS``.
- ``auto_refresh``: ``LLM_Provider_Catalog/model_catalog_settings.py::
  AUTO_REFRESH_PROVIDER_LIST_KEYS``.
- ``classification`` cloud keys: ``config.py::_cloud_provider_keys``.

``reasoning_effort`` is True only where the hosted chat path consumes a
``reasoning_effort`` parameter today (openai always; moonshot, zai and
qwencloud model-gated) -- not parity-gated.

xAI/Grok is deliberately absent (ADR-179: maintainer decision).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Mapping

_CLOUD = "cloud"
_LOCAL = "local"

#: Shared hosted-transport defaults (Qodo finding 7): one source for the
#: numeric transport policy every engine preset ships in
#: ``settings_defaults``. Provider-specific overrides (streaming, custom
#: timeouts, the custom family's 120/1/1.0 policy) stay explicit per
#: record; records spread this into their own dict so no two presets share
#: a mutable mapping.
_HOSTED_TRANSPORT_DEFAULTS = {"timeout": 90, "retries": 3, "retry_delay": 5.0}


@dataclass(frozen=True)
class ProviderRecord:
    """Identity (all providers) + preset data (engine-driven providers).

    Immutable transcription of one provider's identity and -- when
    ``engine_driven`` -- its hosted-preset behavior data. Behavior never
    lives here; consumers read these fields and act (ADR-179).

    Attributes:
        key: Canonical dispatch/execution key (e.g. ``"databricks"``); the
            engine's provider identity in errors, metrics, and checkpoints.
        config_key: The ``[providers]`` table spelling (e.g.
            ``"Databricks"``); parity-gated against ``config.py``.
        display_name: Human-readable label used to prefix user-facing
            engine error copy.
        classification: ``"cloud"`` or ``"local"``; selects the config-key
            list the record feeds.
        api_key_env_var: Conventional credential environment variable, or
            ``None`` when the provider has none.
        api_key_env_candidates: Full env-candidate chain (configured name
            first, then canonical) the engine's credential resolution walks.
        default_base_url: Shipped base URL fallback, or ``None`` when the
            host is per-account (Databricks) or per-entry (custom family).
        native_tools: Whether the provider's chat path advertises native
            tool calling.
        reasoning_effort: Whether the chat path consumes a
            ``reasoning_effort`` parameter for this provider.
        auto_refresh: Whether the model catalog auto-refreshes this
            provider's model list.
        settings_defaults: Shipped ``api_settings`` fallbacks (model,
            streaming, transport policy, env-var name) read by engine
            resolution after explicit kwargs and the settings table.
        pricing_seeds: Seed per-model pricing (USD input/output token
            pairs) for the pricing catalog; empty when pricing is deferred.
        engine_driven: Whether dispatch executes this provider through the
            strict hosted engine (``hosted_provider_engine``) instead of a
            hand-written ``LLM_Calls`` module.
        base_url_suffix: Path appended to a bare configured host (e.g.
            Databricks ``"/openai/v1"``); ``None`` when the default URL is
            already complete.
        finish_terminal: Finish reasons the finish policy accepts as
            terminal for a completed turn.
        finish_provider_errors: Finish reasons treated as provider-side
            failures (502 ``ChatProviderError``) rather than turn outcomes.
        payload_flags: Optional body fields this preset emits; a
            flag-off field with a caller-supplied value is a bad request.
        reasoning_effort_key: Payload key for ``reasoning_effort`` when the
            provider spells it differently; ``None`` keeps the standard key.
        reasoning_effort_values: For a preset that sends ``reasoning_effort``,
            the levels it accepts; any other level is refused locally (and
            left out of Settings). ``None`` forwards any level unchanged.
        reasoning_effort_map: Console levels the provider spells differently,
            mapped to the level actually sent (Fireworks has no ``minimal``, so
            it sends ``low``); applied before ``reasoning_effort_values``.
        thinking_toggle_models: For a preset that refuses
            ``reasoning_effort``, model globs (``fnmatch``, case-sensitive)
            whose chat template switches thinking with one boolean
            ``chat_template_kwargs`` key, mapped to that key. A matching
            model sends ``{key: effort != "none"}`` instead of a refusal.
        max_tokens_key: Payload key for ``max_tokens`` when the provider
            requires another spelling (Azure's ``max_completion_tokens``);
            ``None`` keeps ``max_tokens``.
        stream_annotation_key: A top-level key (also in
            ``response_allowances``) marking a provider annotation frame: a
            stream event with no choices and no usage that carries it is
            accepted and dropped (Azure's ``prompt_filter_results``).
        config_headers: Optional request headers read from the preset's
            ``api_settings`` table, as header name -> setting name; a header
            is sent only when its setting is a non-empty single-line string.
        extra_body_fields: Preset-authored extra body fields merged last,
            each validated bounded.
        response_allowances: Tolerated extra top-level response/stream
            event keys (validated then dropped, never passed through).
        choice_allowances: Tolerated extra choice-level keys (value rule:
            null, scalar, or shape-safe mapping).
        message_allowances: Tolerated extra message/delta-level keys
            (same value rule).
        tool_call_allowances: Tolerated extra keys on a non-streamed tool
            call object (same value rule).
        stream_include_usage: Send ``stream_options.include_usage`` on
            streaming requests (providers that only stream usage on request).
        stream_usage_optional: Accept a stream that ends without usage
            (providers whose chunk schema carries none); a finish reason is
            still required.
        status_envelope_key: Top-level provider status object checked on
            every body and stream event; a nonzero ``status_code`` raises a
            provider error, ``None`` disables the check.
        error_frame_key: Top-level key of a gateway's mid-stream error
            frame; an event carrying it raises a provider error. ``None``
            disables the check.
        tolerant_response_extras: Long-tail tolerant profile switch
            (custom family only, fixture-gated): shape-safe unknown
            top/event keys and null-valued unknown choice/message keys are
            dropped; tool-call objects may carry extra keys; a stream
            terminal without usage becomes a usage-None turn; the finish
            policy accepts stop/length with empty text and no calls.
        reasoning_disposition: How reasoning content is handled:
            ``"ignored"`` (dropped at the finish policy), ``"displayable"``
            (kept visible in stream deltas and the response message), or
            ``"proprietary"`` (private to the terminal turn).
        auth_scheme: Credential contract of engine resolution and
            transport: ``"bearer"`` hard-requires a key;
            ``"bearer_optional"`` lets keyless endpoints (ADR-146) execute
            with no Authorization header; ``"api_key_header"`` (Phase 3)
            hard-requires a key and sends it as an ``api-key`` header.
        continuation_protocol: Protocol for provider continuation
            checkpoints (``"chat_completions"``), or ``None`` when the
            preset builds no checkpoints.
        discovery_route: Route appended to the base URL for model
            discovery (e.g. ``"models"``), or ``None`` when the provider
            documents no models route (discovery is refused).
        defaults_settings_section: Legacy ``api_settings`` section the
            engine reads for per-call fallbacks under the legacy handler's
            exact key spellings, or ``None`` to read the ``key``-named
            table.
    """

    key: str
    config_key: str
    display_name: str
    classification: str
    api_key_env_var: str | None = None
    api_key_env_candidates: tuple[str, ...] = ()
    default_base_url: str | None = None
    native_tools: bool = False
    reasoning_effort: bool = False
    auto_refresh: bool = False
    settings_defaults: Mapping[str, object] = field(default_factory=dict)
    pricing_seeds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    # --- preset fields (engine-driven providers only) ---
    engine_driven: bool = False
    base_url_suffix: str | None = None
    finish_terminal: frozenset[str] = frozenset({"stop", "tool_calls", "length"})
    finish_provider_errors: frozenset[str] = frozenset()
    payload_flags: frozenset[str] = frozenset(
        {
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "response_format",
            "seed",
            "n",
            "user",
            "tool_choice",
        }
    )
    reasoning_effort_key: str | None = None
    reasoning_effort_values: frozenset[str] | None = None
    reasoning_effort_map: Mapping[str, str] = field(default_factory=dict)
    thinking_toggle_models: Mapping[str, str] = field(default_factory=dict)
    max_tokens_key: str | None = None
    stream_annotation_key: str | None = None
    config_headers: Mapping[str, str] = field(default_factory=dict)
    extra_body_fields: Mapping[str, object] = field(default_factory=dict)
    # Tolerated extra response/stream keys, LEVEL-KEYED (ADR-179 Phase 2):
    # ``response_allowances`` keeps its Phase 1 meaning (top-level response
    # and stream-event keys only); ``choice_allowances`` subtracts at the
    # choice level (body and stream choice); ``message_allowances`` at the
    # message/delta level. Level-allowlisted values follow the value rule
    # (null, scalar, or shape-safe mapping), validated then dropped -- never
    # passed through to the normalized turn. Existing strict presets
    # (moonshot/zai byte-identity) ship all three empty.
    response_allowances: frozenset[str] = frozenset()
    choice_allowances: frozenset[str] = frozenset()
    message_allowances: frozenset[str] = frozenset()
    # Extra keys on a non-streamed ``tool_calls[]`` object beyond id/type/
    # function, under the same value rule (TASK-34364).
    tool_call_allowances: frozenset[str] = frozenset()
    # Streamed-usage contract (TASK-33201). Strict records fail a stream that
    # ends without usage, and OpenAI-semantics providers only send streamed
    # usage when asked: ``stream_include_usage`` adds
    # ``stream_options: {"include_usage": true}`` to streaming payloads.
    # ``stream_usage_optional`` is for providers whose documented chunk schema
    # carries no usage at all -- the stream then ends as a usage-None turn
    # (finish reason still required). Both default off so existing presets
    # send byte-identical payloads.
    stream_include_usage: bool = False
    stream_usage_optional: bool = False
    # Provider status envelope (TASK-33201): the top-level object some
    # providers attach to every body/stream event (MiniMax ``base_resp``).
    # A nonzero ``status_code`` is a provider error even when ``choices``
    # look valid, so the engine checks it before normalizing anything.
    status_envelope_key: str | None = None
    # Mid-stream error frame (TASK-33350): some gateways (Tencent TokenHub)
    # report a failure after the 200 header as a bare ``{"error": {...}}``
    # event. With this key set, such a frame is a provider error instead of
    # a "malformed response" protocol error.
    error_frame_key: str | None = None
    # Long-tail tolerant profile (custom family only, ADR-179 Phase 2,
    # fixture-gated): shape-safe unknown top/event keys dropped; null-valued
    # unknown choice/message keys dropped (non-null ones still fail closed
    # unless level-allowlisted); tool-call objects may carry extra keys
    # (id/type/function stay mandatory); a stream terminal without usage
    # becomes a usage-None turn; the engine finish policy accepts stop/
    # length with empty text and no calls (legacy empty reply).
    tolerant_response_extras: bool = False
    reasoning_disposition: str = "ignored"
    # Auth contract of the engine's credential resolution and transport:
    # "bearer" hard-requires a key (a missing key is an actionable
    # configuration error); "bearer_optional" (Phase 2) lets keyless
    # endpoints (ADR-146 custom endpoints) execute with no Authorization
    # header while still using a resolved key when one exists;
    # "api_key_header" (Phase 3, TASK-33350) requires a key and sends it as
    # an ``api-key`` header instead of Authorization. See ADR-179 spec §3.
    auth_scheme: str = "bearer"
    continuation_protocol: str | None = "chat_completions"
    # ``None`` = the provider documents no models route: discovery is
    # refused (seeded-only preset) instead of probing an unlisted URL.
    discovery_route: str | None = "models"
    # Settings-table fallbacks (Phase 2 Task 6): when set, the engine's
    # resolution reads this ``api_settings`` section (instead of a table
    # keyed by ``key``) for per-call sampling/streaming/transport fallbacks
    # under the legacy handler's exact key spellings. Explicit request
    # kwargs still win; ``settings_defaults`` is the last resort.
    defaults_settings_section: str | None = None


# --- Databricks (first engine-driven preset; AI Gateway external models) ---
DATABRICKS = ProviderRecord(
    key="databricks",
    config_key="Databricks",
    display_name="Databricks",
    classification=_CLOUD,
    api_key_env_var="DATABRICKS_TOKEN",
    api_key_env_candidates=("DATABRICKS_TOKEN",),
    default_base_url=None,  # workspace host is per-account; user-configured
    native_tools=True,      # confirmed (or honestly disabled) at live gate (Task 14)
    reasoning_effort=False, # gateway model support varies; enable per-model later
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "DATABRICKS_TOKEN",
        # No "model" key: served models are workspace-configured and fill via
        # discovery/seeding. The resolver's unset path yields "" (payload-
        # gated); a shipped present-but-blank value would fail closed at
        # resolution, so the key stays absent (Task 12 review fix).
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},       # gateway pricing is workspace/model-config dependent
    engine_driven=True,
    base_url_suffix="/openai/v1",
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)

# --- Inference-cloud presets (ADR-179 Phase 2 Task 5) ---
# Together / Fireworks / Cerebras: strict engine presets whose entire
# implementation is this record plus one dispatch entry (no per-provider
# LLM_Calls module).
#
# TASK-34367.3-.5: scoped allowances reconciled with current official schemas
# (2026-10-04) and actual-adapter complete/SSE replays. These are offline
# contracts; live qualification remains TASK-33640. Sources:
# docs.together.ai/reference/chat-completions -- choice logprobs/seed/text/
# top_logprobs, message reasoning, envelope prompt (array)/warnings.
# docs.fireworks.ai/api-reference/post-chatcompletions -- choice logprobs/
# raw_output, envelope perf_metrics (terminal-only in SSE)/prompt_token_ids.
# inference-docs.cerebras.ai/api-reference/chat-completions -- choice
# logprobs/reasoning_logprobs, message reasoning, envelope time_info and
# service_tier/service_tier_used. No allowance applies to another provider.
# Opt-in nonempty choice token_ids and structured reasoning_details are not
# permitted by the existing level-value contract; do not silently drop them.
#
# Fireworks returns reasoning in ``message.reasoning_content`` (and stream
# deltas) and requires it replayed on interleaved tool turns
# (docs.fireworks.ai/guides/reasoning, read 2026-09-29), so its
# disposition is "proprietary" (private, replayed through continuations).
# Its guide says a request carrying both ``thinking`` and
# ``reasoning_effort`` fails validation; this record sends neither
# (reasoning_effort=False, no extra body), which
# test_no_engine_preset_can_send_thinking_with_reasoning_effort pins
# (TASK-33503). Together and Cerebras reason transparently but are not
# wired to a reasoning_effort parameter.
#
# Cerebras function tools are sent WITHOUT ``strict``. oh-my-pi's note that
# Cerebras needs ``strict: true`` on every tool is wrong: the API reference
# (inference-docs.cerebras.ai/api-reference/chat-completions, read
# 2026-09-29) documents ``strict`` as optional, default false; only
# kimi-k2.7-code requires the SAME value on every tool (omitting it
# everywhere satisfies that). Under strict, API version 2 (default since
# 2026-07-22) rejects schemas lacking ``additionalProperties: false`` or
# using pattern/format/minLength/oneOf -- most of Chatbook's tool schemas --
# so opting in would break tool turns (TASK-33500).
#
# Together, captured live 2026-10-04 (Tests/fixtures/cloud_live/together.json,
# TASK-33640): stream choices carry ``logprobs`` (null), and a stream sends
# usage only when ``stream_options.include_usage`` asks (a final chunk with no
# choices). Without both, every streamed reply failed (TASK-34362). /models
# answers with a bare array, which discovery accepts (TASK-34361).
TOGETHER = ProviderRecord(
    key="together",
    config_key="Together",
    display_name="Together",
    classification=_CLOUD,
    api_key_env_var="TOGETHER_API_KEY",
    api_key_env_candidates=("TOGETHER_API_KEY",),
    default_base_url="https://api.together.xyz/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "TOGETHER_API_KEY",
        # No "model" key: models fill via discovery/seeding; the unset key
        # resolves to the payload-gated "" (Phase 1 blank-model lesson --
        # a shipped present-but-blank value would fail closed).
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},       # per-model pricing lands with the catalog
    engine_driven=True,
    response_allowances=frozenset({"prompt", "warnings"}),
    choice_allowances=frozenset({"logprobs", "seed", "top_logprobs", "text"}),
    message_allowances=frozenset({"reasoning"}),
    base_url_suffix=None,   # the default URL is already complete
    reasoning_disposition="ignored",
    auth_scheme="bearer",
    stream_include_usage=True,
)
# Fireworks, captured live 2026-10-04 (Tests/fixtures/cloud_live/fireworks.json,
# TASK-33640): a non-streamed tool call carries ``index`` and ``name: null``
# beside id/type/function, which failed every non-streamed tool reply
# (TASK-34364). Streamed usage arrives without asking.
FIREWORKS = ProviderRecord(
    key="fireworks",
    config_key="Fireworks",
    display_name="Fireworks",
    classification=_CLOUD,
    api_key_env_var="FIREWORKS_API_KEY",
    api_key_env_candidates=("FIREWORKS_API_KEY",),
    default_base_url="https://api.fireworks.ai/inference/v1",
    native_tools=True,
    # docs.fireworks.ai/api-reference/post-chatcompletions (read 2026-09-29):
    # reasoning_effort takes none/low/medium/high/xhigh (plus max/adaptive,
    # which Console does not offer); there is no "minimal". Support varies by
    # model (some always think; DeepSeek V3.1 is off by default) -- not yet
    # live-verified (TASK-33640). Console's "minimal" is sent as the smallest
    # non-zero level, "low", so every level Console offers is really sent.
    reasoning_effort=True,
    reasoning_effort_values=frozenset({"none", "low", "medium", "high", "xhigh"}),
    reasoning_effort_map={"minimal": "low"},
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "FIREWORKS_API_KEY",
        # No "model" key (see TOGETHER): discovery/seeding fills models.
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    response_allowances=frozenset({"perf_metrics", "prompt_token_ids"}),
    choice_allowances=frozenset({"logprobs", "raw_output"}),
    base_url_suffix=None,
    reasoning_disposition="proprietary",  # reasoning_content, replayed on tool turns
    auth_scheme="bearer",
    tool_call_allowances=frozenset({"index", "name"}),
)
CEREBRAS = ProviderRecord(
    key="cerebras",
    config_key="Cerebras",
    display_name="Cerebras",
    classification=_CLOUD,
    api_key_env_var="CEREBRAS_API_KEY",
    api_key_env_candidates=("CEREBRAS_API_KEY",),
    default_base_url="https://api.cerebras.ai/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "CEREBRAS_API_KEY",
        # No "model" key (see TOGETHER): discovery/seeding fills models.
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    response_allowances=frozenset({"time_info", "service_tier", "service_tier_used"}),
    choice_allowances=frozenset({"logprobs", "reasoning_logprobs"}),
    message_allowances=frozenset({"reasoning"}),
    base_url_suffix=None,
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)

# --- Doc-derived inference-cloud presets (TASK-33201) ---
# Six more strict engine presets. Unlike Together/Fireworks/Cerebras, these
# records are derived from each provider's PUBLIC DOCUMENTATION (read
# 2026-09-27; no keys, no captured fixtures). Every allowance below is a
# field the provider documents in its Chat Completions response schema;
# anything a doc did not show stays strict (fails closed). The first live
# capture reconciles these sets -- amend, never silent.
#
# Excluded on purpose (both retired, confirmed on the providers' own pages
# 2026-09-27): GitHub Models (retired 2026-07-30, docs.github.com/en/
# github-models) and Hyperbolic serverless inference (hyperbolic.ai/docs/
# faq/inference-models).
#
# SambaNova Cloud -- OpenAPI spec github.com/sambanova/sambanova-inference-
# api-spec (openapi.documented.json): choice ``logprobs`` (nullable); the
# streaming delta alone carries ``reasoning``/``channel`` (gpt-oss) -- the
# non-streaming message has no reasoning field (reasoning is inline
# ``<think>`` text there), so reasoning is "ignored" rather than routed.
# ``stream_options.include_usage`` is a documented request field.
SAMBANOVA = ProviderRecord(
    key="sambanova",
    config_key="SambaNova",
    display_name="SambaNova",
    classification=_CLOUD,
    api_key_env_var="SAMBANOVA_API_KEY",
    api_key_env_candidates=("SAMBANOVA_API_KEY",),
    default_base_url="https://api.sambanova.ai/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "SAMBANOVA_API_KEY",
        # No "model" key (see TOGETHER): discovery/seeding fills models.
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    choice_allowances=frozenset({"logprobs"}),
    message_allowances=frozenset({"reasoning", "channel"}),
    stream_include_usage=True,
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)
# NVIDIA NIM (build.nvidia.com) -- docs.api.nvidia.com/nim/reference/
# llm-apis + per-model OpenAPI schemas: message ``reasoning_content``
# (model-specific), and the ChatCompletionChunk schema has NO ``usage`` and
# no ``stream_options`` request field, so streams may end without usage.
# ``GET /v1/models`` is not in the official docs but answers OpenAI-shaped
# (unauthenticated probe, 2026-09-27), so the catalog auto-refreshes.
# Qwen3.5 thinks by default; its schema (build.nvidia.com/qwen/qwen3.5-397b-
# a17b, read 2026-09-29) takes no ``reasoning_effort``, only
# ``chat_template_kwargs: {"enable_thinking": bool}`` -- so effort "none"
# turns thinking off and any other level leaves it on (TASK-33502). Note:
# the public /v1/models listing (no key, 2026-09-30) names 81 models and no
# Qwen at all, so whether Qwen3.5 is served at this endpoint needs a keyed
# capture (TASK-33640); the older qwen3-235b "thinking" kwarg is not added
# for the same reason.
NVIDIA = ProviderRecord(
    key="nvidia",
    config_key="NVIDIA",
    display_name="NVIDIA NIM",
    classification=_CLOUD,
    api_key_env_var="NVIDIA_API_KEY",
    api_key_env_candidates=("NVIDIA_API_KEY",),
    default_base_url="https://integrate.api.nvidia.com/v1",
    native_tools=True,
    reasoning_effort=False,  # enum varies per model page
    thinking_toggle_models={"qwen/qwen3.5-*": "enable_thinking"},
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "NVIDIA_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    stream_usage_optional=True,
    reasoning_disposition="proprietary",
    auth_scheme="bearer",
)
# DeepInfra -- docs.deepinfra.com/chat/overview: base ``/v1/openai``,
# top-level ``service_tier``, extra ``usage.estimated_cost`` (usage is only
# shape-checked, so no allowance needed). Streamed usage arrives on the
# finish chunk without any request option (docs.deepinfra.com/chat/
# streaming). Its reasoning field name is undocumented, so reasoning stays
# "ignored" and an unknown message field still fails closed.
DEEPINFRA = ProviderRecord(
    key="deepinfra",
    config_key="DeepInfra",
    display_name="DeepInfra",
    classification=_CLOUD,
    api_key_env_var="DEEPINFRA_API_KEY",
    api_key_env_candidates=("DEEPINFRA_API_KEY",),
    default_base_url="https://api.deepinfra.com/v1/openai",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "DEEPINFRA_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    response_allowances=frozenset({"service_tier"}),
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)
# Nebius Token Factory (renamed from Nebius AI Studio in 2026; the legacy
# api.studio.nebius.com host is being retired) -- docs.tokenfactory.
# nebius.com/api-reference/inference/create-chat-completion: top-level
# ``service_tier``, choice ``logprobs``, message ``reasoning_content``,
# finish reasons stop/length/tool_calls/content_filter, documented
# ``stream_options.include_usage``.
NEBIUS = ProviderRecord(
    key="nebius",
    config_key="Nebius",
    display_name="Nebius Token Factory",
    classification=_CLOUD,
    api_key_env_var="NEBIUS_API_KEY",
    api_key_env_candidates=("NEBIUS_API_KEY",),
    default_base_url="https://api.tokenfactory.nebius.com/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "NEBIUS_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    finish_provider_errors=frozenset({"content_filter"}),
    response_allowances=frozenset({"service_tier"}),
    choice_allowances=frozenset({"logprobs"}),
    stream_include_usage=True,
    reasoning_disposition="proprietary",
    auth_scheme="bearer",
)
# Novita AI -- docs.novita.ai/guides/llm-api: chat is served at
# ``/openai/v1/chat/completions`` (the old ``/v3/openai`` base is
# superseded). docs.novita.ai/guides/llm-reasoning: reasoning goes to
# message/delta ``reasoning_content`` only when ``separate_reasoning`` is
# set, so the record sends it. Documented ``stream_options.include_usage``.
NOVITA = ProviderRecord(
    key="novita",
    config_key="Novita",
    display_name="Novita AI",
    classification=_CLOUD,
    api_key_env_var="NOVITA_API_KEY",
    api_key_env_candidates=("NOVITA_API_KEY",),
    default_base_url="https://api.novita.ai/openai/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "NOVITA_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    extra_body_fields={"separate_reasoning": True},
    stream_include_usage=True,
    reasoning_disposition="proprietary",
    auth_scheme="bearer",
)
# MiniMax (international platform) -- platform.minimax.io/docs/
# api-reference/text-chat-openai: top-level ``base_resp`` plus the
# ``input_sensitive``/``output_sensitive`` (+``_type``) safety flags;
# message ``name`` and ``audio_content``; ``reasoning_split: true`` moves
# thinking out of ``<think>`` text into ``reasoning_content``; finish
# reasons add ``content_filter``; ``stream_options.include_usage`` defaults
# false. The OpenAI quickstart repoints ``OPENAI_API_KEY`` -- deliberately
# NOT a candidate here (it would send an OpenAI key to MiniMax). No
# ``/v1/models`` is documented (401 without a key), so models are seeded
# from the documented ``model`` enum and the catalog does not auto-refresh.
MINIMAX = ProviderRecord(
    key="minimax",
    config_key="MiniMax",
    display_name="MiniMax",
    classification=_CLOUD,
    api_key_env_var="MINIMAX_API_KEY",
    api_key_env_candidates=("MINIMAX_API_KEY",),
    default_base_url="https://api.minimax.io/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=False,
    settings_defaults={
        "api_key_env_var": "MINIMAX_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    finish_provider_errors=frozenset({"content_filter"}),
    extra_body_fields={"reasoning_split": True},
    response_allowances=frozenset(
        {
            "base_resp",
            "input_sensitive",
            "input_sensitive_type",
            "output_sensitive",
            "output_sensitive_type",
        }
    ),
    message_allowances=frozenset({"name", "audio_content"}),
    status_envelope_key="base_resp",
    discovery_route=None,  # seeded-only: no documented /models
    stream_include_usage=True,
    reasoning_disposition="proprietary",
    auth_scheme="bearer",
)

# --- Top-15 OpenRouter model makers (TASK-33350) ---
# OpenRouter's usage rankings (week of 2026-09-21) name the model makers
# people use; these four had no first-party provider here. Like the
# TASK-33201 presets they are built from public documentation (read
# 2026-09-28) plus unauthenticated probes of each host: every allowance is a
# documented field, everything else stays strict. Meta's Llama API was
# retired on 2026-07-06 (llama.developer.meta.com/docs/llama-api-deprecation),
# so StepFun takes its place; xAI stays excluded (ADR-179).
#
# Xiaomi MiMo -- mimo.mi.com/docs/en-US/api/chat/openai-api and
# .../quick-start/summary/first-api-call: the quickstart authenticates with
# an ``api-key:`` header, not Authorization (an invalid-key probe returns the
# same 401 either way, so Bearer is unproven) -> the Phase 3 api_key_header
# scheme. ``thinking`` defaults on and reasoning lands in
# ``reasoning_content``; the FAQ asks for it to be resent in tool loops, so it
# is kept private and round-tripped. Finish reasons add ``content_filter``
# (provider error) and ``repetition_truncation`` (a normal, truncated end).
# ``annotations`` (web-search citations, an array) is NOT allowed: it only
# appears with MiMo's built-in web search, which is never requested. Usage
# streaming is requested (OpenAI convention; exact key unconfirmed) and its
# absence tolerated. No models route is confirmed and discovery only speaks
# Bearer, so the list is seeded from the documented model IDs.
MIMO = ProviderRecord(
    key="mimo",
    config_key="MiMo",
    display_name="Xiaomi MiMo",
    classification=_CLOUD,
    api_key_env_var="MIMO_API_KEY",
    api_key_env_candidates=("MIMO_API_KEY",),
    default_base_url="https://api.xiaomimimo.com/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=False,
    settings_defaults={
        "api_key_env_var": "MIMO_API_KEY",
        # No "model" key (see TOGETHER): seeded [providers] list instead.
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
        # MiMo can take minutes to send its first token: oh-my-pi documents a
        # 5-minute stream-idle floor for it (docs/provider-quirks.md, Xiaomi).
        "timeout": 300,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    finish_terminal=frozenset({"stop", "tool_calls", "length", "repetition_truncation"}),
    finish_provider_errors=frozenset({"content_filter"}),
    discovery_route=None,  # seeded-only: no confirmed models route, discovery is Bearer-only
    stream_include_usage=True,
    stream_usage_optional=True,
    reasoning_disposition="proprietary",
    auth_scheme="api_key_header",
)
# ByteDance Seed via BytePlus ModelArk (international) -- ModelArk chat
# completions docs (docs.byteplus.com/en/docs/ModelArk) and the Volcengine
# twin (volcengine.com/docs/82379/1298454): Bearer ``ARK_API_KEY`` (probe
# confirms the header is read), ``model`` takes a plain model ID or the
# account's own ``ep-...`` endpoint ID. Keys are region-locked: a China
# (Volcengine) key fails here; China users point ``api_base_url`` at
# https://ark.cn-beijing.volces.com/api/v3. Message/delta carry
# ``reasoning_content`` (kept private) and ``encrypted_content``;
# ``service_tier`` is a documented request field whose response echo is
# unconfirmed but OpenAI-standard, so it is tolerated. ``content_filter``
# finishes are provider errors. Users report streamed usage missing even when
# requested, so it is requested AND its absence tolerated. The models route
# exists but its shape is undocumented: seeded from documented IDs.
BYTEPLUS = ProviderRecord(
    key="byteplus",
    config_key="BytePlus",
    display_name="ByteDance Seed (BytePlus)",
    classification=_CLOUD,
    api_key_env_var="ARK_API_KEY",
    api_key_env_candidates=("ARK_API_KEY",),
    default_base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=False,
    settings_defaults={
        "api_key_env_var": "ARK_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    finish_provider_errors=frozenset({"content_filter"}),
    response_allowances=frozenset({"service_tier"}),
    message_allowances=frozenset({"encrypted_content"}),
    discovery_route=None,  # seeded-only: models route shape undocumented
    stream_include_usage=True,
    stream_usage_optional=True,
    reasoning_disposition="proprietary",
    auth_scheme="bearer",
)
# StepFun (international) -- platform.stepfun.ai/docs/en/api-reference/chat/
# chat-completion-create: Bearer auth, ``GET /v1/models`` is
# OpenAI-shaped, ``stream_options.include_usage`` is supported. Reasoning
# models return ``reasoning`` (StepFun's own name, message and delta) --
# tolerated and dropped. Tool calling ships OFF: the tool-call reference shows
# ``finish_reason: "stop"`` alongside ``tool_calls``, which the strict finish
# policy rejects; enabling it needs a live check. China users point
# ``api_base_url`` at https://api.stepfun.com/v1.
# Env var: ``STEPFUN_API_KEY`` only. StepFun's own samples use
# ``STEP_API_KEY``; users who keep that name set ``api_key_env_var =
# "STEP_API_KEY"`` in the table. A second candidate here would be walked by
# the engine but not by Console readiness or discovery (Qodo, PR #2889).
STEPFUN = ProviderRecord(
    key="stepfun",
    config_key="StepFun",
    display_name="StepFun",
    classification=_CLOUD,
    api_key_env_var="STEPFUN_API_KEY",
    api_key_env_candidates=("STEPFUN_API_KEY",),
    default_base_url="https://api.stepfun.ai/v1",
    native_tools=False,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "STEPFUN_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    message_allowances=frozenset({"reasoning"}),
    stream_include_usage=True,
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)

# Tencent Cloud TokenHub (international) -- Tencent's gateway serving its
# Hy4 preview (the model that puts Tencent in OpenRouter's top 15; the
# direct Hunyuan API is China-only and stops at hunyuan-a13b) plus DeepSeek,
# GLM and Kimi models. tencentcloud.com/document/product/1300/78940 (API
# integration guide) and 78932 (model list): Bearer auth (probe confirms),
# ``$TOKENHUB_API_KEY`` in the models example, OpenAI-shaped ``GET /v1/models``,
# ``stream_options.include_usage`` default false (usage only on request).
# Documented extras: top-level ``search_info`` (null unless web search ran),
# choice ``logprobs``, message ``refusal``; ``reasoning_content`` is kept
# private and round-tripped (Preserved Thinking requires it back). NOT
# allowed: ``reasoning_details`` and delta ``search_results`` (arrays; the
# value rule rejects them -- they fail closed if a model sends them).
# Finish reasons add ``content_filter`` (provider error) and
# ``repetition_truncation`` (normal, truncated end). A failure after the 200
# header arrives as a bare ``{"error": {...}}`` frame -> provider error.
# Other hosts: https://tokenhub-us.tencentcloudmaas.com/v1 (US) and
# https://tokenhub.tencentcloudmaas.com/v1 (Chinese mainland).
TOKENHUB = ProviderRecord(
    key="tokenhub",
    config_key="TokenHub",
    display_name="Tencent TokenHub",
    classification=_CLOUD,
    api_key_env_var="TOKENHUB_API_KEY",
    api_key_env_candidates=("TOKENHUB_API_KEY",),
    default_base_url="https://tokenhub-intl.tencentcloudmaas.com/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "TOKENHUB_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    finish_terminal=frozenset({"stop", "tool_calls", "length", "repetition_truncation"}),
    finish_provider_errors=frozenset({"content_filter"}),
    response_allowances=frozenset({"search_info"}),
    choice_allowances=frozenset({"logprobs"}),
    message_allowances=frozenset({"refusal"}),
    error_frame_key="error",
    stream_include_usage=True,
    reasoning_disposition="proprietary",
    auth_scheme="bearer",
)

# --- Gateway and host presets from the Hermes / oh-my-pi comparison (TASK-33351) ---
# Thirteen OpenAI Chat Completions presets that Hermes (NousResearch/
# hermes-agent) and/or oh-my-pi (can1357/oh-my-pi) support and this registry
# did not. Like TASK-33201/33350, every value comes from the provider's own
# public docs (read 2026-09-28) plus unauthenticated probes of each host;
# every allowance is a documented field and everything else stays strict.
#
# Vercel AI Gateway -- vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-
# completions: Bearer ``AI_GATEWAY_API_KEY``; public OpenAI-shaped ``/v1/models``
# (probe: 390 models). Reasoning comes back as ``reasoning`` plus a
# ``reasoning_details`` ARRAY the strict value rule rejects, so every request
# sends the documented ``reasoning: {exclude: true}``; reasoning is not shown
# for gateway presets anyway. Streamed usage is undocumented: requested and
# its absence tolerated.
VERCEL = ProviderRecord(
    key="vercel",
    config_key="Vercel",
    display_name="Vercel AI Gateway",
    classification=_CLOUD,
    api_key_env_var="AI_GATEWAY_API_KEY",
    api_key_env_candidates=("AI_GATEWAY_API_KEY",),
    default_base_url="https://ai-gateway.vercel.sh/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "AI_GATEWAY_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    extra_body_fields={"reasoning": {"exclude": True}},
    message_allowances=frozenset({"reasoning"}),
    stream_include_usage=True,
    stream_usage_optional=True,
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)
# ZenMux -- zenmux.ai/docs/api/openai/create-chat-completion.html: Bearer
# ``ZENMUX_API_KEY``, public ``/api/v1/models``. Documented extras: top-level
# ``service_tier``; message ``refusal``/``reasoning`` and delta
# ``reasoning_content``; ``reasoning_details`` and ``annotations`` are arrays
# (rejected -- reasoning is excluded per request as for Vercel; annotations
# only appear with web search, never requested). ``content_filter`` finishes
# are provider errors; streamed usage only on request.
ZENMUX = ProviderRecord(
    key="zenmux",
    config_key="ZenMux",
    display_name="ZenMux",
    classification=_CLOUD,
    api_key_env_var="ZENMUX_API_KEY",
    api_key_env_candidates=("ZENMUX_API_KEY",),
    default_base_url="https://zenmux.ai/api/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "ZENMUX_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    finish_provider_errors=frozenset({"content_filter"}),
    extra_body_fields={"reasoning": {"exclude": True}},
    response_allowances=frozenset({"service_tier"}),
    message_allowances=frozenset({"refusal", "reasoning", "reasoning_content"}),
    stream_include_usage=True,
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)
# Kilo Gateway -- kilo.ai/docs/gateway/api-reference and /streaming: Bearer
# ``KILO_API_KEY``, public ``/api/gateway/models``; the gateway injects
# ``include_usage`` itself (trailing ``choices: []`` usage chunk). A failure
# after the 200 arrives as a frame carrying a top-level ``error`` object and
# ``finish_reason: "error"`` -- allowed at the top level so the finish policy
# can turn it into a provider error.
KILO = ProviderRecord(
    key="kilo",
    config_key="Kilo",
    display_name="Kilo Gateway",
    classification=_CLOUD,
    api_key_env_var="KILO_API_KEY",
    api_key_env_candidates=("KILO_API_KEY",),
    default_base_url="https://api.kilo.ai/api/gateway",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "KILO_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    finish_provider_errors=frozenset({"error"}),
    response_allowances=frozenset({"error"}),
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)
# SiliconFlow (international) -- docs.siliconflow.com/en/api-reference/chat-
# completions: Bearer (samples use a placeholder; ``SILICONFLOW_API_KEY`` is
# the integration convention), OpenAI-shaped ``/v1/models``. Reasoning in
# ``reasoning_content`` (kept private, sent back -- DeepSeek models need the
# replay in tool loops). Finish reasons add ``eos`` (a normal end). Streamed
# usage is undocumented: requested and its absence tolerated. China users
# point ``api_base_url`` at https://api.siliconflow.cn/v1.
SILICONFLOW = ProviderRecord(
    key="siliconflow",
    config_key="SiliconFlow",
    display_name="SiliconFlow",
    classification=_CLOUD,
    api_key_env_var="SILICONFLOW_API_KEY",
    api_key_env_candidates=("SILICONFLOW_API_KEY",),
    default_base_url="https://api.siliconflow.com/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "SILICONFLOW_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    finish_terminal=frozenset({"stop", "tool_calls", "length", "eos"}),
    stream_include_usage=True,
    stream_usage_optional=True,
    reasoning_disposition="proprietary",
    auth_scheme="bearer",
)
# Baseten Model APIs -- docs.baseten.co/reference/inference-api/chat-
# completions: Bearer ``BASETEN_API_KEY``, OpenAI-shaped ``/v1/models``,
# ``stream_options.include_usage`` documented. Choices carry ``stop_reason``
# (int/str/null) and ``logprobs``; reasoning in ``reasoning_content``. A
# mid-stream failure just ends the stream early, which the strict parser
# already treats as a failure.
BASETEN = ProviderRecord(
    key="baseten",
    config_key="Baseten",
    display_name="Baseten",
    classification=_CLOUD,
    api_key_env_var="BASETEN_API_KEY",
    api_key_env_candidates=("BASETEN_API_KEY",),
    default_base_url="https://inference.baseten.co/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "BASETEN_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    choice_allowances=frozenset({"stop_reason", "logprobs"}),
    stream_include_usage=True,
    reasoning_disposition="proprietary",
    auth_scheme="bearer",
)
# GMI Cloud -- docs.gmicloud.ai/inference-engine/api-reference/llm-api-
# reference: Bearer ``GMI_API_KEY`` (probe confirms the header is read),
# ``/v1/models``; "streaming responses include usage statistics in the final
# data chunk". Its docs document no extras, so it is fully strict.
GMI = ProviderRecord(
    key="gmi",
    config_key="GMI",
    display_name="GMI Cloud",
    classification=_CLOUD,
    api_key_env_var="GMI_API_KEY",
    api_key_env_candidates=("GMI_API_KEY",),
    default_base_url="https://api.gmi-serving.com/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "GMI_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)
# Ollama Cloud -- docs.ollama.com/api/openai-compatibility: ollama.com/v1 is
# OpenAI Chat Completions (not only the native /api/chat); Bearer
# ``OLLAMA_API_KEY`` (unused by the keyless local Ollama provider); public
# ``/v1/models``; ``include_usage`` supported. From Ollama's own server source
# (openai/openai.go): top-level ``timings``, message/delta ``reasoning``. A
# bare ``{"error": ...}`` written after the 200 fails closed as a protocol
# error. ``tool_choice``/``user``/``n`` are documented as unsupported.
OLLAMA_CLOUD = ProviderRecord(
    key="ollama_cloud",
    config_key="OllamaCloud",
    display_name="Ollama Cloud",
    classification=_CLOUD,
    api_key_env_var="OLLAMA_API_KEY",
    api_key_env_candidates=("OLLAMA_API_KEY",),
    default_base_url="https://ollama.com/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "OLLAMA_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    # Documented-unsupported ``n``/``user``/``tool_choice`` are flagged off:
    # a caller value fails closed instead of being sent (Qodo #2896).
    payload_flags=frozenset(
        {"temperature", "top_p", "max_tokens", "stop", "response_format", "seed"}
    ),
    response_allowances=frozenset({"timings"}),
    message_allowances=frozenset({"reasoning"}),
    stream_include_usage=True,
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)
# Upstage (Solar) -- console.upstage.ai/api/chat: Bearer ``UPSTAGE_API_KEY``.
# No models route is documented, so models are seeded from the API
# reference. Reasoning is ``reasoning`` (not ``reasoning_content``);
# ``logprobs`` is present but always null. Streamed usage is unconfirmed:
# requested and its absence tolerated.
UPSTAGE = ProviderRecord(
    key="upstage",
    config_key="Upstage",
    display_name="Upstage",
    classification=_CLOUD,
    api_key_env_var="UPSTAGE_API_KEY",
    api_key_env_candidates=("UPSTAGE_API_KEY",),
    default_base_url="https://api.upstage.ai/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=False,
    settings_defaults={
        "api_key_env_var": "UPSTAGE_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    choice_allowances=frozenset({"logprobs"}),
    message_allowances=frozenset({"reasoning"}),
    discovery_route=None,  # seeded-only: no documented models route
    stream_include_usage=True,
    stream_usage_optional=True,
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)
# Arcee AI -- docs.arcee.ai/api-reference/chat-completion and /models: Bearer
# ``ARCEE_API_KEY`` (``rcai-`` keys; probe confirms the header is read),
# ``GET /api/v1/models`` returns ``{"data": [...]}`` (no ``object`` wrapper --
# discovery only needs ``data``). Reasoning in ``reasoning_content``.
# Streamed usage is unconfirmed: requested and its absence tolerated.
ARCEE = ProviderRecord(
    key="arcee",
    config_key="Arcee",
    display_name="Arcee AI",
    classification=_CLOUD,
    api_key_env_var="ARCEE_API_KEY",
    api_key_env_candidates=("ARCEE_API_KEY",),
    default_base_url="https://api.arcee.ai/api/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "ARCEE_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    stream_include_usage=True,
    stream_usage_optional=True,
    reasoning_disposition="proprietary",
    auth_scheme="bearer",
)
# Baidu Qianfan (ERNIE, v2 OpenAI-compatible API) -- cloud.baidu.com/doc/
# qianfan-api/s/3m7of64lb: ``Authorization: Bearer bce-v3/ALTAK-...`` (the
# whole string is the key); no env var is named (``QIANFAN_API_KEY`` is the
# integration convention); no models route, so models are seeded. Extras:
# top-level ``search_results`` (web search only), choice ``flag``/``ban_round``
# (safety classification). ``content_filter`` finishes are provider errors.
# Requires Baidu Cloud real-name verification.
QIANFAN = ProviderRecord(
    key="qianfan",
    config_key="Qianfan",
    display_name="Baidu Qianfan",
    classification=_CLOUD,
    api_key_env_var="QIANFAN_API_KEY",
    api_key_env_candidates=("QIANFAN_API_KEY",),
    default_base_url="https://qianfan.baidubce.com/v2",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=False,
    settings_defaults={
        "api_key_env_var": "QIANFAN_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    finish_provider_errors=frozenset({"content_filter"}),
    response_allowances=frozenset({"search_results"}),
    choice_allowances=frozenset({"flag", "ban_round"}),
    discovery_route=None,  # seeded-only: no documented models route
    stream_include_usage=True,
    reasoning_disposition="proprietary",
    auth_scheme="bearer",
)
# Nous Research -- portal.nousresearch.com/api-docs (bot-walled; facts via
# NousResearch/hermes-agent#47950 quoting it): Chat Completions with a plain
# Bearer key, ``NOUS_API_KEY`` as used in its examples; public ``/v1/models``
# (probe: 421 models). The response schema could not be read, so the record
# is fully strict, tools stay off until a live check, and streamed usage is
# requested with its absence tolerated.
NOUS = ProviderRecord(
    key="nous",
    config_key="Nous",
    display_name="Nous Research",
    classification=_CLOUD,
    api_key_env_var="NOUS_API_KEY",
    api_key_env_candidates=("NOUS_API_KEY",),
    default_base_url="https://inference-api.nousresearch.com/v1",
    native_tools=False,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "NOUS_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    stream_include_usage=True,
    stream_usage_optional=True,
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)
# Venice -- docs.venice.ai/api-reference/endpoint/chat/completions: Bearer
# ``VENICE_API_KEY``, public ``/api/v1/models``, include_usage documented.
# Extras: top-level ``cost``, ``prompt_logprobs``, ``venice_parameters``;
# choice ``stop_reason``/``logprobs``; message ``refusal``,
# ``thought_signature``; reasoning in ``reasoning_content``. The documented
# ``reasoning_details`` ARRAY fails closed if a model sends it.
# ``content_filter`` finishes are provider errors.
VENICE = ProviderRecord(
    key="venice",
    config_key="Venice",
    display_name="Venice",
    classification=_CLOUD,
    api_key_env_var="VENICE_API_KEY",
    api_key_env_candidates=("VENICE_API_KEY",),
    default_base_url="https://api.venice.ai/api/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "VENICE_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    finish_provider_errors=frozenset({"content_filter"}),
    response_allowances=frozenset({"cost", "prompt_logprobs", "venice_parameters"}),
    choice_allowances=frozenset({"stop_reason", "logprobs"}),
    message_allowances=frozenset({"refusal", "thought_signature"}),
    stream_include_usage=True,
    reasoning_disposition="proprietary",
    auth_scheme="bearer",
)
# Meta Model API (Muse Spark; replaced the retired Llama API) -- dev.meta.ai/
# docs/protocols/chat-completions and /models: Chat Completions is supported
# ("Available on: Responses, Chat Completions, Messages"); Bearer key; Meta's
# SDKs read the generic ``MODEL_API_KEY`` -- this record reads
# ``META_API_KEY`` so an unrelated MODEL_API_KEY is never sent to Meta (set
# ``api_key_env_var = "MODEL_API_KEY"`` to use Meta's name). Only
# ``tool_choice: "auto"`` is accepted and ``reasoning_effort: "none"`` is a
# 400 (never sent: reasoning_effort is off). Reasoning is redacted for
# external callers. Extras: top-level ``service_tier``; choice ``logprobs``;
# message ``refusal``/``reasoning_content``. ``content_filter`` is a provider
# error; include_usage documented.
META = ProviderRecord(
    key="meta",
    config_key="Meta",
    display_name="Meta (Muse Spark)",
    classification=_CLOUD,
    api_key_env_var="META_API_KEY",
    api_key_env_candidates=("META_API_KEY",),
    default_base_url="https://api.meta.ai/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "META_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    finish_provider_errors=frozenset({"content_filter"}),
    response_allowances=frozenset({"service_tier"}),
    choice_allowances=frozenset({"logprobs"}),
    message_allowances=frozenset({"refusal"}),
    stream_include_usage=True,
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)

# --- Follow-up presets from the Hermes / oh-my-pi comparison (TASK-33505..33509) ---
# Deferred from TASK-33351 because each needed a small engine capability:
# a user-supplied host (Azure, Cloudflare -- the Databricks pattern), a
# ``max_completion_tokens`` spelling and content-filter annotation frames
# (Azure), or a config-sourced optional header (W&B, Cloudflare). Built from
# public documentation (read 2026-09-29) plus unauthenticated probes, like
# TASK-33201: every allowance is documented or captured in a public
# fixture, anything else fails closed, and the first live capture
# reconciles the sets (amend, never silent).
#
# Azure OpenAI (Foundry) v1 API -- learn.microsoft.com/azure/ai-foundry/
# openai/api-version-lifecycle: POST {resource}/openai/v1/chat/completions
# with no api-version; the documented REST key header is ``api-key``;
# ``model`` is the DEPLOYMENT name, and the models route lists base models,
# not deployments, so the list is user-seeded (discovery off). The resource
# host is per-account: the user sets api_base_url and /openai/v1 is
# appended. v1 deprecates ``max_tokens`` (o-series and gpt-5 reject it), so
# ``max_completion_tokens`` is sent. Content filtering annotates replies:
# top-level ``prompt_filter_results`` (a list; top-level allowances carry no
# value rule), choice ``content_filter_results`` (sometimes the singular
# ``content_filter_result``), message ``refusal`` and ``annotations: []``,
# stream-chunk ``obfuscation``/``service_tier``, the trailing usage chunk's
# ``latency_checkpoint``/``routing``, and a leading stream frame with no
# choices and no usage that carries only ``prompt_filter_results``
# (stream_annotation_key). A ``content_filter`` finish is a provider error.
# NOT supported: Asynchronous Filter mode (opt-in per deployment; it sends
# delta-less choices after the finish) and Entra ID bearer tokens.
AZURE = ProviderRecord(
    key="azure",
    config_key="Azure",
    display_name="Azure OpenAI",
    classification=_CLOUD,
    api_key_env_var="AZURE_OPENAI_API_KEY",
    api_key_env_candidates=("AZURE_OPENAI_API_KEY",),
    default_base_url=None,  # resource host is per-account; user-configured
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=False,
    settings_defaults={
        "api_key_env_var": "AZURE_OPENAI_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix="/openai/v1",
    finish_provider_errors=frozenset({"content_filter"}),
    max_tokens_key="max_completion_tokens",
    response_allowances=frozenset(
        {"prompt_filter_results", "obfuscation", "service_tier", "latency_checkpoint", "routing"}
    ),
    choice_allowances=frozenset({"content_filter_results", "content_filter_result", "logprobs"}),
    message_allowances=frozenset({"refusal", "annotations"}),
    stream_annotation_key="prompt_filter_results",
    stream_include_usage=True,
    discovery_route=None,  # the models route lists base models, not deployments
    reasoning_disposition="ignored",
    auth_scheme="api_key_header",
)
# W&B Inference by CoreWeave -- docs.wandb.ai/inference/api-reference;
# coreweave.com/products/serverless-inference is this API. Bearer W&B API
# key (probe: 401 "Missing bearer authentication in header"). The optional
# ``OpenAI-Project: <team>/<project>`` header picks the billing project
# (unset: the key's default entity, project "inference"), sent from
# ``[api_settings.wandb] project``. ``GET /v1/models`` is authenticated and
# OpenAI-shaped. Function tools only. Reasoning arrives in message
# ``reasoning`` (null for non-reasoning models); streamed usage is
# undocumented, so it is requested and its absence tolerated.
WANDB = ProviderRecord(
    key="wandb",
    config_key="WandB",
    display_name="W&B Inference (CoreWeave)",
    classification=_CLOUD,
    api_key_env_var="WANDB_API_KEY",
    api_key_env_candidates=("WANDB_API_KEY",),
    default_base_url="https://api.inference.wandb.ai/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "WANDB_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    message_allowances=frozenset({"reasoning"}),
    config_headers={"OpenAI-Project": "project"},
    stream_include_usage=True,
    stream_usage_optional=True,
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)
# Cloudflare Workers AI through the REST API -- developers.cloudflare.com/
# ai-gateway/usage/rest-api, which Cloudflare recommends for new
# integrations: POST https://api.cloudflare.com/client/v4/accounts/
# {account_id}/ai/v1/chat/completions with a Cloudflare API token (Bearer;
# needs Account > Workers AI > Read). The account id is part of the path, so
# the user sets the full ``.../ai/v1`` base URL (no suffix: the URL already
# has a path). There is no models route (GET answers 405), so Workers AI
# ``@cf/...`` models are seeded. An optional ``cf-aig-gateway-id`` header
# routes through a named AI Gateway, from ``[api_settings.cloudflare]
# gateway_id``. Documented extras: message ``refusal``, choice ``logprobs``;
# reasoning arrives as ``reasoning_content`` (private). Not used: the
# gateway.ai.cloudflare.com compat endpoint, which takes two credential
# headers (``cf-aig-authorization`` plus the upstream provider key).
CLOUDFLARE = ProviderRecord(
    key="cloudflare",
    config_key="Cloudflare",
    display_name="Cloudflare Workers AI",
    classification=_CLOUD,
    api_key_env_var="CLOUDFLARE_API_TOKEN",
    api_key_env_candidates=("CLOUDFLARE_API_TOKEN",),
    default_base_url=None,  # the account id is in the path; user-configured
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=False,
    settings_defaults={
        "api_key_env_var": "CLOUDFLARE_API_TOKEN",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    choice_allowances=frozenset({"logprobs"}),
    message_allowances=frozenset({"refusal"}),
    config_headers={"cf-aig-gateway-id": "gateway_id"},
    stream_include_usage=True,
    stream_usage_optional=True,
    discovery_route=None,  # no models route (405)
    reasoning_disposition="proprietary",
    auth_scheme="bearer",
)
# OpenCode Zen -- opencode.ai/docs/zen: a pay-per-request gateway at
# opencode.ai/zen/v1 with a Bearer key. Each model lives on ONE protocol and
# Zen does not translate (anomalyco/opencode zen/util/handler.ts: "Zen
# provider format must match request format"), so only its Chat Completions
# models are seeded; the public /models list has no protocol field, so
# discovery stays off. Same-protocol replies are the upstream's bytes plus a
# top-level ``cost`` string on non-streaming replies; the streamed cost frame
# arrives after [DONE] and is never read. Reasoning arrives as
# ``reasoning_content`` and must be replayed on thinking tool turns
# (proprietary). Upstream extras beyond these stay strict until a live
# capture. OpenCode Go is NOT offered: a subscription "designed for OpenCode
# and other coding agents" that requires a per-conversation
# ``x-opencode-session`` header, which record data cannot express.
OPENCODE_ZEN = ProviderRecord(
    key="opencode_zen",
    config_key="OpenCodeZen",
    display_name="OpenCode Zen",
    classification=_CLOUD,
    api_key_env_var="OPENCODE_API_KEY",
    api_key_env_candidates=("OPENCODE_API_KEY",),
    default_base_url="https://opencode.ai/zen/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=False,
    settings_defaults={
        "api_key_env_var": "OPENCODE_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    response_allowances=frozenset({"cost"}),
    stream_include_usage=True,
    stream_usage_optional=True,
    discovery_route=None,  # /models lists every protocol's models
    reasoning_disposition="proprietary",
    auth_scheme="bearer",
)
# Command Code Provider API -- commandcode.ai/docs/provider:
# api.commandcode.ai/provider/v1 with a Bearer key (Provider plan, or GOAT/
# Pro/Max/Team). "Every endpoint emits token usage at the end of every
# stream... No opt-in required", so ``stream_options`` is not sent and usage
# is required. Claude models are served only on /messages (400 here), so
# only Chat Completions models are seeded; the public /models rows would
# also list the Messages-only ones, so discovery stays off. The API follows
# the OpenAI schema: message ``refusal`` and ``annotations`` (an empty list)
# are tolerated; anything else stays strict until a live capture.
COMMANDCODE = ProviderRecord(
    key="commandcode",
    config_key="CommandCode",
    display_name="Command Code",
    classification=_CLOUD,
    api_key_env_var="COMMANDCODE_API_KEY",
    api_key_env_candidates=("COMMANDCODE_API_KEY",),
    default_base_url="https://api.commandcode.ai/provider/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=False,
    settings_defaults={
        "api_key_env_var": "COMMANDCODE_API_KEY",
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    message_allowances=frozenset({"refusal", "annotations"}),
    discovery_route=None,  # /models would list Messages-only Claude rows
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)

# --- Custom hosted family (ADR-179 Phase 2 Task 6) ---
# The engine-driven execution surface for the ADR-146 custom-endpoint
# ``openai_compatible`` family, swapped in at the Console gateway identity
# site when ``[console] custom_endpoints_use_engine`` is on. Identity,
# readiness, and saved sessions keep the ``custom``/``custom-ep:<slug>``
# spellings -- only the execution key changes, so this record deliberately
# ships NO env var (credentials are entry-resolved by the gateway;
# ``[api_settings.custom].api_key_env_var`` still flows through the
# settings section when the engine resolves a key on its own) and no
# default base URL (the per-entry URL is forwarded explicitly; a direct
# call without one fails with actionable copy). Fallbacks read the legacy
# ``[api_settings.custom]`` table via ``defaults_settings_section`` under
# ``chat_with_custom_openai``'s exact key spellings (``api_timeout`` et al;
# the shipped table's ``timeout`` spellings were never read by the legacy
# handler and stay unread). ADR-066 custom row: ``reasoning_effort`` is
# consumed verbatim; a thinking budget is accepted and dropped (strict
# OpenAI proxies may reject llama.cpp-specific fields).
CUSTOM_HOSTED = ProviderRecord(
    key="custom-hosted",
    config_key="Custom-hosted",
    display_name="Custom Hosted",
    classification=_LOCAL,
    api_key_env_var=None,
    api_key_env_candidates=(),
    default_base_url=None,
    native_tools=True,
    reasoning_effort=True,
    auto_refresh=False,
    settings_defaults={
        "streaming": False,
        "max_tokens": 4096,
        "timeout": 120,
        "retries": 1,
        "retry_delay": 1.0,
    },
    engine_driven=True,
    payload_flags=frozenset(
        {
            "temperature",
            "top_p",
            "min_p",
            "top_k",
            "max_tokens",
            "stop",
            "response_format",
            "seed",
            "n",
            "user",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "thinking_budget_tokens",
            "tool_choice",
        }
    ),
    # "logprobs": evidence-backed (non-null once the forwarded logprobs
    # param is set). "stop_reason": PROVISIONAL -- vLLM memory, not fixture
    # evidence (no CUDA host for capture; see Tests/fixtures/longtail/
    # CAPTURE.md). Reconcile when a vLLM capture lands.
    choice_allowances=frozenset({"logprobs", "stop_reason"}),
    tolerant_response_extras=True,
    reasoning_disposition="ignored",
    auth_scheme="bearer_optional",
    continuation_protocol="chat_completions",
    defaults_settings_section="custom",
)

# --- existing cloud providers (opaque identity records) ---
# api_key_env_var / default api_base_url transcribed EXACTLY from config.py's
# [api_settings.*] tables (lines ~4166-4325) and, where a table defines no
# api_base_url, from Chat/console_provider_endpoints.py::_BUILTIN_PROVIDER_
# ENDPOINTS (the chat path's fallback). config_key spellings are the
# _cloud_provider_keys list (config.py ~L9843).
OPENAI = ProviderRecord(
    key="openai", config_key="OpenAI", display_name="OpenAI", classification=_CLOUD,
    api_key_env_var="OPENAI_API_KEY",
    api_key_env_candidates=("OPENAI_API_KEY",),
    default_base_url="https://api.openai.com/v1",
    native_tools=True, reasoning_effort=True, auto_refresh=True,
)
ANTHROPIC = ProviderRecord(
    key="anthropic", config_key="Anthropic", display_name="Anthropic", classification=_CLOUD,
    api_key_env_var="ANTHROPIC_API_KEY",
    api_key_env_candidates=("ANTHROPIC_API_KEY",),
    default_base_url="https://api.anthropic.com/v1",
    native_tools=True, reasoning_effort=False, auto_refresh=True,
)
COHERE = ProviderRecord(
    key="cohere", config_key="Cohere", display_name="Cohere", classification=_CLOUD,
    api_key_env_var="COHERE_API_KEY",
    api_key_env_candidates=("COHERE_API_KEY",),
    default_base_url="https://api.cohere.com",
    native_tools=True, reasoning_effort=False, auto_refresh=False,
)
GROQ = ProviderRecord(
    key="groq", config_key="Groq", display_name="Groq", classification=_CLOUD,
    api_key_env_var="GROQ_API_KEY",
    api_key_env_candidates=("GROQ_API_KEY",),
    default_base_url="https://api.groq.com/openai/v1",
    native_tools=True, reasoning_effort=False, auto_refresh=False,
)
OPENROUTER = ProviderRecord(
    key="openrouter", config_key="OpenRouter", display_name="OpenRouter", classification=_CLOUD,
    api_key_env_var="OPENROUTER_API_KEY",
    api_key_env_candidates=("OPENROUTER_API_KEY",),
    default_base_url="https://openrouter.ai/api/v1",
    native_tools=True, reasoning_effort=False, auto_refresh=True,
)
DEEPSEEK = ProviderRecord(
    key="deepseek", config_key="DeepSeek", display_name="DeepSeek", classification=_CLOUD,
    api_key_env_var="DEEPSEEK_API_KEY",
    api_key_env_candidates=("DEEPSEEK_API_KEY",),
    default_base_url="https://api.deepseek.com",
    native_tools=True, reasoning_effort=False, auto_refresh=False,
)
MISTRAL = ProviderRecord(
    key="mistral", config_key="MistralAI", display_name="Mistral", classification=_CLOUD,
    api_key_env_var="MISTRAL_API_KEY",
    api_key_env_candidates=("MISTRAL_API_KEY",),
    default_base_url="https://api.mistral.ai/v1",
    native_tools=True, reasoning_effort=False, auto_refresh=True,
)
GOOGLE = ProviderRecord(
    key="google", config_key="Google", display_name="Google", classification=_CLOUD,
    api_key_env_var="GOOGLE_API_KEY",
    api_key_env_candidates=("GOOGLE_API_KEY",),
    default_base_url="https://generativelanguage.googleapis.com/v1beta",
    native_tools=True, reasoning_effort=False, auto_refresh=False,
)
HUGGINGFACE = ProviderRecord(
    key="huggingface", config_key="HuggingFace", display_name="Hugging Face", classification=_CLOUD,
    api_key_env_var="HUGGINGFACE_API_KEY",
    api_key_env_candidates=("HUGGINGFACE_API_KEY",),
    default_base_url="https://router.huggingface.co/v1",  # [api_settings.huggingface].api_base_url
    native_tools=False, reasoning_effort=False, auto_refresh=False,
)
MOONSHOT = ProviderRecord(
    key="moonshot", config_key="Moonshot", display_name="Moonshot", classification=_CLOUD,
    api_key_env_var="MOONSHOT_API_KEY",
    api_key_env_candidates=("MOONSHOT_API_KEY",),
    default_base_url="https://api.moonshot.ai/v1",  # [api_settings.moonshot].api_base_url
    native_tools=True, reasoning_effort=True, auto_refresh=True,
)
ZAI = ProviderRecord(
    key="zai", config_key="ZAI", display_name="Z.ai", classification=_CLOUD,
    api_key_env_var="ZAI_API_KEY",
    api_key_env_candidates=("ZAI_API_KEY",),
    default_base_url="https://api.z.ai/api/paas/v4",  # [api_settings.zai].api_base_url
    native_tools=True, reasoning_effort=True, auto_refresh=True,
)
QWENCLOUD = ProviderRecord(
    key="qwencloud", config_key="QwenCloud", display_name="QwenCloud", classification=_CLOUD,
    api_key_env_var="DASHSCOPE_API_KEY",  # [api_settings.qwencloud].api_key_env_var
    api_key_env_candidates=("DASHSCOPE_API_KEY",),
    default_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",  # [api_settings.qwencloud].api_base_url
    native_tools=True, reasoning_effort=True, auto_refresh=True,
)

# --- local providers (opaque identity records; classification=_LOCAL) ---
# config_key spellings transcribed from the [providers] table's local keys
# (the _cloud_provider_keys complement in config.py); env vars from the
# [api_settings.*] local tables where defined. Local tables configure
# api_url (a full endpoint), so default_base_url stays None. local_onnx /
# local_transformers have config tables but no dispatch handler and no
# audited endpoint -- they are not registry records.
LLAMA_CPP = ProviderRecord(
    key="llama_cpp", config_key="Llama_cpp", display_name="llama.cpp", classification=_LOCAL,
    api_key_env_var="LLAMA_CPP_API_KEY",
    api_key_env_candidates=("LLAMA_CPP_API_KEY",),
)
KOBOLDCPP = ProviderRecord(
    key="koboldcpp", config_key="koboldcpp", display_name="Koboldcpp", classification=_LOCAL,
)
OOABOOGA = ProviderRecord(
    key="oobabooga", config_key="Oobabooga", display_name="Oobabooga", classification=_LOCAL,
    api_key_env_var="OOBABOOGA_API_KEY",
    api_key_env_candidates=("OOBABOOGA_API_KEY",),
)
TABBYAPI = ProviderRecord(
    key="tabbyapi", config_key="TabbyAPI", display_name="Tabbyapi", classification=_LOCAL,
    api_key_env_var="TABBYAPI_API_KEY",
    api_key_env_candidates=("TABBYAPI_API_KEY",),
)
VLLM = ProviderRecord(
    key="vllm", config_key="vLLM", display_name="vLLM", classification=_LOCAL,
    api_key_env_var="VLLM_API_KEY",
    api_key_env_candidates=("VLLM_API_KEY",),
)
OLLAMA = ProviderRecord(
    key="ollama", config_key="Ollama", display_name="Ollama", classification=_LOCAL,
)
APHRODITE = ProviderRecord(
    key="aphrodite", config_key="Aphrodite", display_name="Aphrodite", classification=_LOCAL,
    api_key_env_var="APHRODITE_API_KEY",
    api_key_env_candidates=("APHRODITE_API_KEY",),
)
LOCAL_LLM = ProviderRecord(
    key="local-llm", config_key="local-llm", display_name="Local Llm", classification=_LOCAL,
)
# Execution keys custom-openai-api / custom-openai-api-2 read the
# [api_settings.custom] / [api_settings.custom_2] tables ("custom" /
# "custom_2" are their readiness spellings -- see
# Chat/console_provider_support.py::_READINESS_TO_EXECUTION_ALIASES).
CUSTOM_OPENAI_API = ProviderRecord(
    key="custom-openai-api", config_key="Custom", display_name="Custom OpenAI", classification=_LOCAL,
    api_key_env_var="CUSTOM_API_KEY",
    api_key_env_candidates=("CUSTOM_API_KEY",),
    native_tools=True,
)
CUSTOM_OPENAI_API_2 = ProviderRecord(
    key="custom-openai-api-2", config_key="Custom_2", display_name="Custom OpenAI 2", classification=_LOCAL,
    api_key_env_var="CUSTOM_2_API_KEY",
    api_key_env_candidates=("CUSTOM_2_API_KEY",),
    native_tools=True,
)
# MLX-LM server: the shipped config tables ([providers]/[api_settings]) spell
# it local_mlx_lm; chat_with_mlx_lm also reads api_settings.mlx_lm when no
# provider_name is passed. "local_mlx_lm" is the [providers] spelling.
MLX_LM = ProviderRecord(
    key="mlx_lm", config_key="local_mlx_lm", display_name="MLX LM", classification=_LOCAL,
)

ALL_RECORDS: tuple[ProviderRecord, ...] = (
    OPENAI, ANTHROPIC, COHERE, GROQ, OPENROUTER, DEEPSEEK, MISTRAL, GOOGLE,
    HUGGINGFACE, MOONSHOT, ZAI, QWENCLOUD, DATABRICKS,
    TOGETHER, FIREWORKS, CEREBRAS, CUSTOM_HOSTED,
    SAMBANOVA, NVIDIA, DEEPINFRA, NEBIUS, NOVITA, MINIMAX,
    MIMO, TOKENHUB, BYTEPLUS, STEPFUN,
    VERCEL, ZENMUX, KILO, SILICONFLOW, BASETEN, GMI, OLLAMA_CLOUD,
    UPSTAGE, ARCEE, QIANFAN, NOUS, VENICE, META,
    AZURE, WANDB, CLOUDFLARE, OPENCODE_ZEN, COMMANDCODE,
    LLAMA_CPP, KOBOLDCPP, OOABOOGA, TABBYAPI, VLLM, OLLAMA, APHRODITE,
    LOCAL_LLM, CUSTOM_OPENAI_API, CUSTOM_OPENAI_API_2, MLX_LM,
)

RECORDS_BY_KEY: dict[str, ProviderRecord] = {record.key: record for record in ALL_RECORDS}
#: Dispatch-key aliases (legacy spellings) mapped onto canonical records.
ALIASES: dict[str, str] = {
    "mistralai": "mistral",
    "local_llamacpp": "llama_cpp",
    "local_llamafile": "llama_cpp",
    "local_vllm": "vllm",
    "local_ollama": "ollama",
    "local_mlx_lm": "mlx_lm",
}
# Alias-aware lookup: resolve any dispatch spelling (canonical or legacy)
# to its canonical record, so every API_CALL_HANDLERS key is
# lookup-complete in RECORDS_BY_KEY (parity: test_records_unique_and_complete).
for _alias, _canonical in ALIASES.items():
    RECORDS_BY_KEY[_alias] = RECORDS_BY_KEY[_canonical]
del _alias, _canonical


def thinking_toggle_key(record: ProviderRecord, model: str | None) -> str | None:
    """Return the ``chat_template_kwargs`` key that toggles ``model``'s thinking.

    Args:
        record: Provider record whose ``thinking_toggle_models`` is consulted.
        model: Selected model identifier, if any.

    Returns:
        The toggle key for the first matching glob, or ``None`` when the
        record declares no toggle for this model.
    """
    if not model:
        return None
    for pattern, key in record.thinking_toggle_models.items():
        if fnmatchcase(model, pattern):
            return key
    return None

CLOUD_PROVIDER_CONFIG_KEYS: tuple[str, ...] = tuple(
    record.config_key for record in ALL_RECORDS if record.classification == _CLOUD
)
LOCAL_PROVIDER_CONFIG_KEYS: tuple[str, ...] = tuple(
    record.config_key for record in ALL_RECORDS if record.classification == _LOCAL
)
ENGINE_RECORDS: tuple[ProviderRecord, ...] = tuple(
    record for record in ALL_RECORDS if record.engine_driven
)
AUTO_REFRESH_KEYS: frozenset[str] = frozenset(
    record.key for record in ALL_RECORDS if record.auto_refresh
)
NATIVE_TOOLS_KEYS: frozenset[str] = frozenset(
    record.key for record in ALL_RECORDS if record.native_tools
)
#: Every dispatch key (canonical + aliases) — the sensitive-audit universe.
AUDITED_ENDPOINT_KEYS: frozenset[str] = frozenset(RECORDS_BY_KEY) | frozenset(ALIASES)
