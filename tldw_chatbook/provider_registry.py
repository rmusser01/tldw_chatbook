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


@dataclass(frozen=True)
class ProviderRecord:
    """Identity (all providers) + preset data (engine-driven providers)."""

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
    response_allowances: frozenset[str] = frozenset()
    reasoning_disposition: str = "ignored"
    auth_scheme: str = "bearer"  # Phase 1: bearer only; see spec §3
    continuation_protocol: str | None = "chat_completions"
    discovery_route: str = "models"


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
        "model": "",
        "streaming": True,
        "timeout": 90,
        "retries": 3,
        "retry_delay": 5.0,
    },
    pricing_seeds={},       # gateway pricing is workspace/model-config dependent
    engine_driven=True,
    base_url_suffix="/openai/v1",
    reasoning_disposition="ignored",
    auth_scheme="bearer",
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
