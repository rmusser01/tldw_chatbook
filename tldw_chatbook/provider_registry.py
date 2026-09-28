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
- ``display_name``: ``Chat/console_provider_support.py::
  _PROVIDER_DISPLAY_NAMES``; keys absent from that map (directly or via a
  readiness alias) use the same module's title-case fallback.
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
        extra_body_fields: Preset-authored extra body fields merged last,
            each validated bounded.
        response_allowances: Tolerated extra top-level response/stream
            event keys (validated then dropped, never passed through).
        choice_allowances: Tolerated extra choice-level keys (value rule:
            null, scalar, or shape-safe mapping).
        message_allowances: Tolerated extra message/delta-level keys
            (same value rule).
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
            with no Authorization header; ``"api_key_header"`` is the Phase
            3 scheme.
        continuation_protocol: Protocol for provider continuation
            checkpoints (``"chat_completions"``), or ``None`` when the
            preset builds no checkpoints.
        discovery_route: Route appended to the base URL for model
            discovery (e.g. ``"models"``).
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
        {"temperature", "top_p", "max_tokens", "stop", "response_format", "seed", "n", "user"}
    )
    reasoning_effort_key: str | None = None
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
    # "api_key_header" is the Phase 3 scheme. See ADR-179 spec §3.
    auth_scheme: str = "bearer"
    continuation_protocol: str | None = "chat_completions"
    discovery_route: str = "models"
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
# ALLOWANCES ARE PROVISIONAL PENDING FIRST LIVE CAPTURE: this environment
# holds no provider keys, so Task 2 captured NO cloud fixtures, and all
# three records ship EMPTY response/choice/message allowance sets -- the
# strict parser unchanged. Task 7's live probes capture real envelopes and
# reconcile these sets (amend, never silent).
#
# Memory of expected extras (recorded here as memory, NOT as live
# allowances -- fixture-unproven): Together -- a top-level ``prompt``
# string (the prompt tokens actually shown) and choice-level ``logprobs``;
# Cerebras -- a top-level ``time_info`` object on some responses.
#
# Fireworks hides reasoning behind its own API surface (response_format
# modes), so its disposition is "proprietary"; Together and Cerebras
# reason transparently but are not wired to a reasoning_effort parameter.
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
    base_url_suffix=None,   # the default URL is already complete
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)
FIREWORKS = ProviderRecord(
    key="fireworks",
    config_key="Fireworks",
    display_name="Fireworks",
    classification=_CLOUD,
    api_key_env_var="FIREWORKS_API_KEY",
    api_key_env_candidates=("FIREWORKS_API_KEY",),
    default_base_url="https://api.fireworks.ai/inference/v1",
    native_tools=True,
    reasoning_effort=False,
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "FIREWORKS_API_KEY",
        # No "model" key (see TOGETHER): discovery/seeding fills models.
        "streaming": True,
        **_HOSTED_TRANSPORT_DEFAULTS,
    },
    pricing_seeds={},
    engine_driven=True,
    base_url_suffix=None,
    reasoning_disposition="proprietary",  # reasoning behind its own API surface
    auth_scheme="bearer",
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
    base_url_suffix=None,
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
