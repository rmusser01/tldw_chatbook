"""Pure Console session settings contracts and helpers."""

from __future__ import annotations

import functools
import json
import math
import os
import re
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from enum import Enum
from ipaddress import ip_address
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Callable,
    Collection,
    Literal,
    Mapping,
    Sequence,
    get_args,
)
from urllib.parse import urlparse, urlunparse

from tldw_chatbook.Chat.console_provider_support import (
    DIRECT_CONSOLE_PROVIDER_KEYS,
    MODEL_FIELD_LABELS,
    ConsoleProviderCatalogEntry,
    build_local_thinking_payload_fields,
    resolve_console_provider_identity,
    supported_console_provider_catalog,
    supported_console_provider_readiness_keys,
    supported_generation_fields,
)
from tldw_chatbook.Chat.console_provider_endpoints import (
    DEFAULT_LLAMACPP_BASE_URL,  # noqa: F401  (re-exported; console_settings_modal imports it from here)
    INVALID_LLAMACPP_BASE_URL_COPY,
    URL_BASED_PROVIDER_KEYS,  # also re-exported; console_settings_modal imports it from here
    effective_provider_endpoint,
    first_configured_endpoint,
    generic_endpoint_differs,
    normalize_generic_endpoint_for_compare,
    provider_uses_endpoint,
    safe_endpoint_display,
    unsaved_endpoint_copy,
)
from tldw_chatbook.Chat.provider_catalog import (
    PROVIDER_CUSTOM_GROUP_KEYS,
    provider_display_name,
)
from tldw_chatbook.Provider_Contract.provider_endpoint_contract import (
    REMOTE_ENDPOINT_SCHEME_REQUIRED,
    canonical_connection_identity,
    read_schemeless_loopback,
    resolve_provider_endpoint,
)
from tldw_chatbook.Chat.provider_readiness import (
    KEYLESS_PROVIDER_KEYS,
    STILL_ENCRYPTED_REASON,
    ProviderReadiness,
    configured_provider_credential_source,
    custom_endpoint_family_readiness,
    get_provider_readiness,
    provider_config_key,
    safe_provider_label,
)
from tldw_chatbook.Chat.provider_test_evidence import (
    ConfigurationFacet,
    ConfigurationIssueCode,
    CredentialFacet,
    CredentialSource,
    EndpointFailureCategory,
    EndpointFacet,
    GenerationFailureCategory,
    GenerationFacet,
    LISTING_UNAVAILABLE_CATEGORIES,
    ModelFacet,
    PUBLIC_MODEL_LISTING_PROVIDER_KEYS,
    ProviderConnectionEvidence,
    ProviderDraftIdentity,
    ProviderReadinessSnapshot,
    ProviderTestEvidence,
    connection_credential_revision,
    provider_readiness_verdict,
)
from tldw_chatbook.provider_registry import CONSOLE_EXECUTION_PROVIDER_KEYS
from tldw_chatbook.Provider_Contract.sampling_params import (
    TEMPERATURE_RANGE,
    _is_blank_value,
    _parse_optional_int,
    _string_value,
    bool_setting_from_sources as _bool_setting_from_sources,
    float_setting_from_sources as _float_setting_from_sources,
    optional_float_setting_from_sources as _optional_float_setting_from_sources,
    optional_int_setting_from_sources as _optional_int_setting_from_sources,
    optional_string_setting_from_sources as _optional_string_setting_from_sources,
    temperature_range,
)
# The enum sets canonicalize in sampling_params.py (ADR-344's shared leaf)
# so presets, registry entries, and Console settings validate against ONE
# contract; the underscore aliases keep this module's call sites unchanged.
from tldw_chatbook.Provider_Contract.sampling_params import (
    REASONING_EFFORT_VALUES as _REASONING_EFFORT_VALUES,
    REASONING_SUMMARY_VALUES as _REASONING_SUMMARY_VALUES,
    THINKING_EFFORT_VALUES as _THINKING_EFFORT_VALUES,
    VERBOSITY_VALUES as _VERBOSITY_VALUES,
)
from tldw_chatbook.config import (
    DEFAULT_CONFIG_FROM_TOML,
    ProviderSettingsError,
    is_encrypted_config_value,
    PRE_MARKER_TEMPLATE_SAMPLES,
    holds_template_sample_identity,
    provider_settings_for_key,
    resolve_provider_api_key,
)
from tldw_chatbook.model_capabilities import anthropic_model_rejects_disabled_thinking
from tldw_chatbook.Utils.input_validation import (
    validate_env_var_reference,
    validate_url,
)
from tldw_chatbook.Utils.token_counter import count_tokens_messages
from tldw_chatbook.Utils.character_display_text import sanitize_character_display_label

if TYPE_CHECKING:
    # Imported lazily at call sites: custom_endpoint_registry imports this
    # module for URL normalization, so a module-level import would cycle.
    from tldw_chatbook.Chat.custom_endpoint_registry import CustomEndpointEntry
    from tldw_chatbook.Chat.console_request_capacity import ConsoleRequestCapacity
    from tldw_chatbook.Utils.token_counter import ContextWindowResolution


NATIVE_CONSOLE_PROVIDER_KEYS = DIRECT_CONSOLE_PROVIDER_KEYS
CONSOLE_SESSION_SETTINGS_SOURCES = frozenset({"derived", "user"})
#: Provider keys the Console settings modal offers and readiness accepts.
#: Includes the ADR-179 engine-driven cloud presets (databricks, together,
#: fireworks, cerebras -- TASK-32919): they dispatch through
#: ``API_CALL_HANDLERS`` like every legacy key, so they are identity keys
#: here too; only ``custom-hosted`` stays execution-only (Phase 2 Task 6
#: decision 1). Derived from the registry: the ``RECORDS_BY_KEY`` keys minus
#: ``custom-hosted`` (aliases such as ``mistralai`` are keys there too). Reconciled against the handler map by
#: ``test_settings_execution_provider_keys_match_chat_api_handlers``.
CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS = CONSOLE_EXECUTION_PROVIDER_KEYS


def settings_provider_catalog() -> tuple[ConsoleProviderCatalogEntry, ...]:
    """Return the providers the Settings picker and first-run setup offer.

    One set for both surfaces (TASK-33621.14, TASK-33002.13). The first-run
    wizard used to list the whole ``chat_api_call`` handler catalog, so it
    offered keys that setup cannot save, execution-only ``custom-hosted``
    among them, and highlighting one raised and blanked the step. Every key
    here is one ``provider_setup_persistence`` owns (engine presets through
    its registry derivation, TASK-33510); the setup-persistence tests sweep
    this set, so a key added without ownership fails there, not on a user's
    first run.

    Returns:
        Catalog entries for ``CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS``.
    """
    return supported_console_provider_catalog(CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS)


MODEL_OPTION_PLACEHOLDER_VALUES = frozenset({"none", "null"})
TokenCounter = Callable[[Sequence[Mapping[str, str]], str, str], int]
TokenLimitResolver = Callable[[str, str], int]
ConsoleOperability = Literal["ready_to_send", "not_ready"]
ConsoleSettingsBlockerCode = Literal[
    "provider_missing",
    "provider_unsupported",
    "provider_configuration_invalid",
    "endpoint_invalid",
    "endpoint_not_saved",
    "credential_missing",
    "credential_rejected",
    "model_missing",
    "endpoint_unreachable",
    "active_run",
    "readiness_unknown",
    # TASK-34100.2 AC#2 (new-cross-cutting-01): model-level blockers.
    "capacity_exhausted",
    "key_rejected_on_send",
    "model_unavailable",
]
ConsoleSettingsRecoveryAction = Literal[
    "select_provider",
    "select_supported_provider",
    "review_provider_settings",
    "configure_endpoint",
    "save_endpoint",
    "configure_credential",
    "select_model",
    "retry_connection",
    "wait_for_active_run",
    "adjust_model_limits",
]
_CONSOLE_OPERABILITY_VALUES = frozenset({"ready_to_send", "not_ready"})
_CONSOLE_SETTINGS_BLOCKER_VALUES = frozenset(
    {
        "provider_missing",
        "provider_unsupported",
        "provider_configuration_invalid",
        "endpoint_invalid",
        "endpoint_not_saved",
        "credential_missing",
        "credential_rejected",
        "model_missing",
        "endpoint_unreachable",
        "active_run",
        "readiness_unknown",
        "capacity_exhausted",
        "key_rejected_on_send",
        "model_unavailable",
    }
)
_CONSOLE_SETTINGS_RECOVERY_VALUES = frozenset(
    {
        "select_provider",
        "select_supported_provider",
        "review_provider_settings",
        "configure_endpoint",
        "save_endpoint",
        "configure_credential",
        "select_model",
        "retry_connection",
        "wait_for_active_run",
        "adjust_model_limits",
    }
)
_BLOCKER_RECOVERY_ACTION = {
    "provider_missing": "select_provider",
    "provider_unsupported": "select_supported_provider",
    "provider_configuration_invalid": "review_provider_settings",
    "endpoint_invalid": "configure_endpoint",
    "endpoint_not_saved": "save_endpoint",
    "credential_missing": "configure_credential",
    "credential_rejected": "configure_credential",
    "model_missing": "select_model",
    "active_run": "wait_for_active_run",
    "readiness_unknown": "review_provider_settings",
    "capacity_exhausted": "adjust_model_limits",
    "key_rejected_on_send": "configure_credential",
    "model_unavailable": "select_model",
}
#: Model-level blockers (TASK-34100.2): the provider is set up and keyed, but
#: this model cannot take a turn. The switcher groups such a provider with the
#: ready ones and judges each model row on its own.
MODEL_LEVEL_BLOCKERS = frozenset({"capacity_exhausted", "model_unavailable"})
#: Blockers a past send recorded (TASK-34100.2 review F2): they say why the
#: last send failed on every surface, but never refuse the next send -- that
#: send is the retry, and its reply clears them (``refuses_turn``).
SEND_OUTCOME_BLOCKERS = frozenset({"key_rejected_on_send", "model_unavailable"})
_BLOCKER_PRECEDENCE = {
    "provider_missing": 0,
    "provider_unsupported": 1,
    "endpoint_invalid": 2,
    "provider_configuration_invalid": 3,
    "endpoint_not_saved": 4,
    "credential_missing": 5,
    "model_missing": 6,
    "capacity_exhausted": 6.5,
    "credential_rejected": 7,
    "endpoint_unreachable": 7,
    "active_run": 8,
    "key_rejected_on_send": 8.5,
    "model_unavailable": 8.5,
}
_CONFIGURATION_ISSUE_BLOCKER = {
    "provider_missing": "provider_missing",
    "endpoint_missing": "endpoint_invalid",
    "invalid_settings": "provider_configuration_invalid",
    "credential_missing": "credential_missing",
}
_RETRYABLE_ENDPOINT_FAILURE_CATEGORIES = frozenset(
    {"timeout", "connection_refused", "connection_error"}
)
_BLOCKER_REQUIRED_FACETS: dict[str, Mapping[str, object]] = {
    "provider_missing": {
        "configuration": "incomplete",
        "configuration_issue": "provider_missing",
    },
    "provider_configuration_invalid": {
        "configuration": "incomplete",
        "configuration_issue": "invalid_settings",
    },
    "credential_missing": {
        "configuration": "incomplete",
        "configuration_issue": "credential_missing",
        "credential": "missing",
    },
    "model_missing": {"model": "missing"},
    "credential_rejected": {
        "endpoint": "unreachable",
        "endpoint_category": frozenset({"unauthorized", "forbidden"}),
    },
    "endpoint_unreachable": {"endpoint": "unreachable"},
}
_CONFIGURATION_VALUES = frozenset({"incomplete", "configured"})
_CONFIGURATION_ISSUE_VALUES = frozenset(
    {"provider_missing", "credential_missing", "endpoint_missing", "invalid_settings"}
)
_CREDENTIAL_VALUES = frozenset(get_args(CredentialFacet))
#: Credential states that were checked by the provider (TASK-33005.3).
_ACCEPTED_CREDENTIALS = frozenset({"listing_accepted", "authenticated"})
_CREDENTIAL_SOURCE_VALUES = frozenset({"none", "stored", "environment", "draft"})
_ENDPOINT_VALUES = frozenset(
    {
        "not_tested",
        "testing",
        "reachable",
        "unreachable",
        "model_listing_unavailable",
        "changed_since_test",
    }
)
_ENDPOINT_FAILURE_CATEGORY_VALUES = frozenset(
    {
        "timeout",
        "connection_refused",
        "unauthorized",
        "forbidden",
        "http_status",
        "invalid_payload",
        "connection_error",
    }
)
_MODEL_VALUES = frozenset({"missing", "confirmed", "unconfirmed"})
_GENERATION_VALUES = frozenset(
    {"not_tested", "testing", "succeeded", "failed", "changed_since_test"}
)
_GENERATION_FAILURE_CATEGORY_VALUES = frozenset(
    {
        "authentication",
        "rate_limit",
        "bad_request",
        "timeout",
        "connection_error",
        "provider_error",
    }
)
_LEGACY_CHAT_PROVIDER_ALIASES = {
    "openai_compatible": "openai",
}
# Generation-aware: dotted Qwen3.x generations consume effort levels;
# original Qwen3 is a thinking toggle only. The dotted-Qwen regex in
# reasoning_effort_hint_for_model enforces generation specificity; needle
# order within this table is immaterial (no needle contains another).
_REASONING_EFFORT_MODEL_HINTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("gpt-oss", frozenset({"low", "medium", "high"})),
    ("qwen3", frozenset({"none"})),
)
# "none" and "high" are live-verified on dotted Qwens: "none" is consumed
# via our enable_thinking=false mapping, and the template aliases "high" to
# "xhigh" — neither must warn as unconsumed.
_QWEN_DOTTED_EFFORT_VALUES = frozenset({"low", "medium", "high", "xhigh", "none"})
# local-llm sends compose llama.cpp-family wire fields (chat_template_kwargs
# reasoning/enable_thinking + reasoning_budget_tokens), so its users need the
# --jinja/b9982 requirements note too.
_LLAMA_CPP_FAMILY_PROVIDERS = frozenset(
    {"llama_cpp", "local_llamacpp", "local_llamafile", "local_llm"}
)
_LLAMACPP_THINKING_REQUIREMENTS_NOTE = (
    "Thinking controls on llama.cpp need llama-server started with --jinja; "
    "per-request reasoning_budget_tokens needs llama.cpp b9982 or newer."
)


def normalize_llamacpp_base_url(api_url: str | None) -> str:
    """Return the llama.cpp origin root used before appending OpenAI paths."""
    raw_url = str(api_url or "").strip()
    if not raw_url:
        return DEFAULT_LLAMACPP_BASE_URL

    # TASK-39151: only a loopback spelling typed without a scheme is read as
    # http://, the contract's rule; any other stays schemeless and invalid.
    candidate = read_schemeless_loopback(raw_url)
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return raw_url.rstrip("/")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return raw_url.rstrip("/")

    path = parsed.path.rstrip("/")
    normalized_endpoint_paths = {
        "/v1",
        "/v1/models",
        "/models",
        "/v1/chat/completions",
        "/chat/completions",
        "/completion",
        "/completions",
    }
    if path.lower() in normalized_endpoint_paths:
        path = ""
    normalized = urlunparse((parsed.scheme, parsed.netloc, path, "", "", "")).rstrip(
        "/"
    )
    return normalized or DEFAULT_LLAMACPP_BASE_URL


@dataclass(frozen=True)
class ConsoleSessionSettings:
    """User-editable Console chat session settings."""

    provider: str
    model: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    min_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    verbosity: str | None = None
    thinking_effort: str | None = None
    thinking_budget_tokens: int | None = None
    streaming: bool = True
    character_label: str = ""
    #: Optional per-session system prompt prepended as the first provider
    #: message on every native Console send (submit/retry/regenerate/continue).
    #: Defaults to ``None`` (native Console sends no system message unless a
    #: user explicitly sets one for this session) -- it is never seeded from
    #: ``[chat_defaults].system_prompt``.
    system_prompt: str | None = None
    #: Provenance of this snapshot: ``"derived"`` for config-derived defaults
    #: (refreshable when config changes while the session is unused) vs
    #: ``"user"`` for explicit user selections (never auto-replaced).
    source: str = "derived"
    #: Pinned response prefill applied to every submit/retry/regenerate;
    #: persisted per-conversation in conversations.metadata (one-shot
    #: prefill is transient store state, not settings).
    pinned_prefill: str | None = None
    #: Workspace assistant defaults (Task 9): memory mode of the persona a
    #: NEW session was seeded from ("read_only" | "read_write"). ``None``
    #: means the session has no workspace-default persona provenance.
    #: Snapshot semantics: stamped once at creation, never re-resolved.
    persona_memory_mode: str | None = None


def parse_persisted_console_session_settings(
    raw_metadata: object,
) -> ConsoleSessionSettings | None:
    """Decode one complete versioned settings snapshot without partial fallback."""

    try:
        metadata = json.loads(raw_metadata or "{}")
    except (TypeError, ValueError, RecursionError):
        return None
    if not isinstance(metadata, dict):
        return None
    payload = metadata.get("console_session_settings")
    field_names = {field.name for field in fields(ConsoleSessionSettings)}
    if (
        not isinstance(payload, dict)
        or set(payload) != {"version", *field_names}
        or type(payload.get("version")) is not int
        or payload["version"] != 1
    ):
        return None

    required_text = {"provider", "character_label", "source"}
    optional_text = {
        "model",
        "base_url",
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
        "system_prompt",
        "pinned_prefill",
        "persona_memory_mode",
    }
    required_float = {"temperature", "top_p"}
    optional_float = {"min_p", "presence_penalty", "frequency_penalty"}
    optional_int = {"top_k", "max_tokens", "seed", "thinking_budget_tokens"}
    values = {name: payload[name] for name in field_names}
    if any(type(values[name]) is not str for name in required_text):
        return None
    if not values["provider"]:
        return None
    if any(type(values[name]) not in {str, type(None)} for name in optional_text):
        return None
    if any(type(values[name]) not in {int, float} for name in required_float):
        return None
    if any(
        type(values[name]) not in {int, float, type(None)} for name in optional_float
    ):
        return None
    if any(type(values[name]) not in {int, type(None)} for name in optional_int):
        return None
    if type(values["streaming"]) is not bool:
        return None
    for name in required_float | optional_float:
        if values[name] is not None:
            try:
                normalized = float(values[name])
            except OverflowError:
                return None
            if not math.isfinite(normalized):
                return None
            values[name] = normalized
    try:
        settings = ConsoleSessionSettings(**values)
    except TypeError:
        return None
    if _console_session_settings_structural_errors(settings):
        return None
    return settings


@dataclass(frozen=True, slots=True)
class EffectiveChatConfiguration:
    """Canonical provider, model, and endpoint selected for a chat session."""

    provider: str
    model: str | None
    base_url: str | None
    model_source: str


@dataclass(frozen=True)
class ConsoleSettingsOption:
    """Selectable settings option for provider and model controls."""

    label: str
    value: str


@dataclass(frozen=True)
class ConsoleSettingsReadiness:
    """Console operability plus independent provider-test evidence.

    The first three fields retain their positional constructor contract for
    existing callers. New callers should consume the typed fields instead of
    inferring state from ``label`` or ``detail``.
    """

    label: str
    detail: str
    native_send_supported: bool
    operability: ConsoleOperability | None = None
    blocker: ConsoleSettingsBlockerCode | None = None
    recovery_action: ConsoleSettingsRecoveryAction | None = None
    provider_display_name: str = ""
    configuration: ConfigurationFacet = "incomplete"
    configuration_issue: ConfigurationIssueCode | None = None
    credential: CredentialFacet = "missing"
    credential_source: CredentialSource = "none"
    endpoint: EndpointFacet = "not_tested"
    endpoint_category: EndpointFailureCategory | None = None
    model: ModelFacet = "missing"
    generation: GenerationFacet = "not_tested"
    generation_category: GenerationFailureCategory | None = None
    subscription_status: Literal["pending", "ready", "expired", "missing"] | None = None
    #: TASK-33005.2: the connection whose shared test evidence this readiness
    #: read (the Console's one-action retry probes it) and when its endpoint
    #: and generation facts were observed. ``None`` when no evidence backs it.
    connection: ProviderDraftIdentity | None = None
    observed_at: datetime | None = None
    generation_observed_at: datetime | None = None
    #: TASK-33005.11: the evidence behind ``connection``, so one provider's
    #: readiness can word each of its models (``model_readiness_words``).
    evidence: ProviderTestEvidence | None = field(
        default=None, compare=False, repr=False
    )
    #: TASK-39151 review: why the endpoint the send would use is invalid, for
    #: the Endpoint line: "needs_scheme" (an address the contract refuses only
    #: for lacking http:// or https://) or "invalid"; ``None`` when it is not.
    invalid_endpoint: Literal["needs_scheme", "invalid"] | None = None

    @property
    def refuses_turn(self) -> bool:
        """Whether this verdict refuses a new turn (a send, or Apply in settings).

        False when ready, or when the only blocker is a past send's outcome
        (``SEND_OUTCOME_BLOCKERS``): the next send is the retry (TASK-34100.2
        review round 1, F2). ``console_display_state`` gates the composer on this.
        """
        return (
            self.operability != "ready_to_send"
            and self.blocker not in SEND_OUTCOME_BLOCKERS
        )

    def __post_init__(self) -> None:
        """Normalize legacy construction and reject contradictory typed states."""
        if type(self.native_send_supported) is not bool:
            raise ValueError("Console native-send support flag is invalid.")
        if self.operability is None:
            self._normalize_legacy_state()
        self._validate_structured_state()

    def _normalize_legacy_state(self) -> None:
        if self.native_send_supported:
            replacements = {
                "operability": "ready_to_send",
                "blocker": None,
                "recovery_action": None,
                "configuration": "configured",
                "configuration_issue": None,
                "credential": "not_required",
                "credential_source": "none",
                "endpoint": "not_tested",
                "endpoint_category": None,
                "model": "unconfirmed",
                "generation": "not_tested",
                "generation_category": None,
            }
        else:
            replacements = {
                "operability": "not_ready",
                "blocker": "readiness_unknown",
                "recovery_action": "review_provider_settings",
                "configuration": "incomplete",
                "configuration_issue": "invalid_settings",
                "credential": "missing",
                "credential_source": "none",
                "endpoint": "not_tested",
                "endpoint_category": None,
                "model": "missing",
                "generation": "not_tested",
                "generation_category": None,
            }
        for field_name, value in replacements.items():
            object.__setattr__(self, field_name, value)

    def _validate_structured_state(self) -> None:
        _validate_console_readiness_literals(self)
        if type(self.connection) not in {ProviderDraftIdentity, type(None)} or {
            type(self.observed_at),
            type(self.generation_observed_at),
        } - {datetime, type(None)}:
            raise ValueError("Console readiness evidence is invalid.")
        if self.subscription_status is not None and (
            (self.subscription_status == "ready")
            != (self.credential in {"present_unverified", *_ACCEPTED_CREDENTIALS})
        ):
            raise ValueError(
                "Console subscription state conflicts with its credential."
            )
        expected_operability = (
            "ready_to_send" if self.native_send_supported else "not_ready"
        )
        if self.operability != expected_operability:
            raise ValueError("Console operability contradicts native-send support.")
        if self.operability == "ready_to_send":
            if self.blocker is not None or self.recovery_action is not None:
                raise ValueError("Ready Console settings cannot include a blocker.")
            if (
                self.configuration != "configured"
                or self.credential == "missing"
                or self.model == "missing"
                or self.endpoint == "unreachable"
            ):
                raise ValueError("Ready Console settings contain blocking facets.")
        else:
            if self.blocker is None or self.recovery_action is None:
                raise ValueError("Blocked Console settings require recovery.")

        if self.configuration == "configured" and self.configuration_issue is not None:
            raise ValueError("Configured Console settings cannot include an issue.")
        if self.configuration_issue is not None and self.configuration != "incomplete":
            raise ValueError("Console configuration issue conflicts with its facet.")
        if self.credential in {"missing", "not_required"}:
            if self.credential_source != "none":
                raise ValueError("Console credential source conflicts with its facet.")
        elif self.credential_source == "none":
            raise ValueError("Present Console credential requires a source.")
        if self.configuration == "incomplete" and self.credential in _ACCEPTED_CREDENTIALS:
            raise ValueError("Incomplete Console settings cannot be authenticated.")

        if self.endpoint == "unreachable":
            pass
        elif self.endpoint == "model_listing_unavailable":
            if self.endpoint_category not in LISTING_UNAVAILABLE_CATEGORIES:
                raise ValueError("Console endpoint category conflicts with its facet.")
        elif self.endpoint_category is not None:
            raise ValueError("Console endpoint category conflicts with its facet.")
        if self.generation == "failed":
            if self.generation_category is None:
                raise ValueError("Failed Console generation requires a category.")
        elif self.generation_category is not None:
            raise ValueError("Console generation category conflicts with its facet.")

        _validate_console_blocker_contract(self)


def _validate_console_readiness_literals(readiness: ConsoleSettingsReadiness) -> None:
    for value, allowed, label in (
        (readiness.operability, _CONSOLE_OPERABILITY_VALUES, "operability"),
        (readiness.blocker, _CONSOLE_SETTINGS_BLOCKER_VALUES, "blocker"),
        (
            readiness.recovery_action,
            _CONSOLE_SETTINGS_RECOVERY_VALUES,
            "recovery action",
        ),
        (readiness.configuration, _CONFIGURATION_VALUES, "configuration"),
        (
            readiness.configuration_issue,
            _CONFIGURATION_ISSUE_VALUES,
            "configuration issue",
        ),
        (readiness.credential, _CREDENTIAL_VALUES, "credential"),
        (
            readiness.subscription_status,
            {"pending", "ready", "expired", "missing"},
            "subscription status",
        ),
        (
            readiness.credential_source,
            _CREDENTIAL_SOURCE_VALUES,
            "credential source",
        ),
        (readiness.endpoint, _ENDPOINT_VALUES, "endpoint"),
        (
            readiness.endpoint_category,
            _ENDPOINT_FAILURE_CATEGORY_VALUES,
            "endpoint category",
        ),
        (readiness.model, _MODEL_VALUES, "model"),
        (readiness.generation, _GENERATION_VALUES, "generation"),
        (
            readiness.generation_category,
            _GENERATION_FAILURE_CATEGORY_VALUES,
            "generation category",
        ),
    ):
        if value is not None and (type(value) is not str or value not in allowed):
            raise ValueError(f"Console {label} is invalid.")
    if not all(
        type(value) is str
        for value in (
            readiness.label,
            readiness.detail,
            readiness.provider_display_name,
        )
    ):
        raise ValueError("Console readiness display value is invalid.")


def _expected_console_recovery_action(
    readiness: ConsoleSettingsReadiness,
) -> ConsoleSettingsRecoveryAction:
    """Return the one recovery action allowed by a structured blocker."""

    if readiness.blocker == "endpoint_unreachable":
        if readiness.endpoint_category in _RETRYABLE_ENDPOINT_FAILURE_CATEGORIES:
            return "retry_connection"
        return "review_provider_settings"
    return _BLOCKER_RECOVERY_ACTION[readiness.blocker]


def _facet_indicated_blocker(
    readiness: ConsoleSettingsReadiness,
) -> ConsoleSettingsBlockerCode | None:
    """Return the highest-priority blocker encoded by readiness facets."""

    candidates: list[ConsoleSettingsBlockerCode] = []
    if readiness.configuration == "incomplete":
        if readiness.configuration_issue is None:
            candidates.append("provider_configuration_invalid")
        else:
            candidates.append(
                _CONFIGURATION_ISSUE_BLOCKER[readiness.configuration_issue]
            )
    if readiness.credential == "missing":
        candidates.append("credential_missing")
    if readiness.model == "missing":
        candidates.append("model_missing")
    if readiness.endpoint == "unreachable":
        endpoint_blocker: ConsoleSettingsBlockerCode = "endpoint_unreachable"
        if readiness.endpoint_category in {"unauthorized", "forbidden"}:
            endpoint_blocker = "credential_rejected"
        candidates.append(endpoint_blocker)
    if not candidates:
        return None
    return min(candidates, key=_BLOCKER_PRECEDENCE.__getitem__)


def _validate_console_blocker_contract(
    readiness: ConsoleSettingsReadiness,
) -> None:
    """Reject blocked snapshots that disagree with builder precedence."""

    if readiness.operability == "ready_to_send":
        return
    if readiness.blocker is None or readiness.recovery_action is None:
        return

    expected_recovery = _expected_console_recovery_action(readiness)
    if readiness.recovery_action != expected_recovery:
        raise ValueError("Console blocker and recovery action conflict.")

    if readiness.blocker == "readiness_unknown":
        legacy_facets = (
            readiness.configuration,
            readiness.configuration_issue,
            readiness.credential,
            readiness.credential_source,
            readiness.endpoint,
            readiness.endpoint_category,
            readiness.model,
            readiness.generation,
            readiness.generation_category,
        )
        if legacy_facets != (
            "incomplete",
            "invalid_settings",
            "missing",
            "none",
            "not_tested",
            None,
            "missing",
            "not_tested",
            None,
        ):
            raise ValueError("Unknown Console readiness must use legacy facets.")
        return

    required_facets = _BLOCKER_REQUIRED_FACETS.get(readiness.blocker, {})
    for field_name, expected in required_facets.items():
        actual = getattr(readiness, field_name)
        if isinstance(expected, frozenset):
            matches = actual in expected
        else:
            matches = actual == expected
        if not matches:
            raise ValueError("Console blocker conflicts with readiness facets.")

    indicated = _facet_indicated_blocker(readiness)
    if indicated is None:
        return
    blocker_priority = _BLOCKER_PRECEDENCE[readiness.blocker]
    indicated_priority = _BLOCKER_PRECEDENCE[indicated]
    if blocker_priority > indicated_priority or (
        blocker_priority == indicated_priority and readiness.blocker != indicated
    ):
        raise ValueError("Console blocker violates readiness precedence.")


@dataclass(frozen=True)
class ConsoleSettingsContextEstimate:
    """Estimated context usage for the current Console session."""

    used_tokens: int | None
    token_limit: int | None
    label: str
    staged_source_count: int = 0
    staged_context_summary: str = ""
    token_limit_verified: bool | None = None
    token_limit_source: str = ""


@dataclass(frozen=True)
class ConsoleSettingsSummaryState:
    """Compact Console settings summary rows for rail display."""

    model_row: str
    context_row: str
    sampling_row: str
    identity_row: str
    #: Structured sampling values for the left rail's Model section
    #: (TASK-32338): the rail used to regex-parse these back out of
    #: ``sampling_row``; empty string means "not set" (renders as an
    #: em-dash placeholder at the rail).
    temperature: str = ""
    max_tokens: str = ""
    #: "On"/"Off" for the rail's Streaming row (TASK-33004.7, task-338).
    streaming: str = ""
    readiness_label: str = ""
    provider_row: str = ""
    endpoint_row: str = ""
    credential_row: str = ""
    transport_row: str = ""
    action_label: str = "Configure"
    action_tooltip: str = "Configure Console settings"
    readiness: ConsoleSettingsReadiness | None = None


def _summary_row_value(row: str) -> str:
    text = str(row or "").strip()
    _label, separator, value = text.partition(":")
    return value.strip() if separator else text


CONSOLE_MODEL_SECTION_MODEL_MAX_CHARS = 24


def _truncate_model_section_value(
    value: str, limit: int = CONSOLE_MODEL_SECTION_MODEL_MAX_CHARS
) -> str:
    """Truncate a rail model label so the provider/model line stays on one row.

    Long local model names (e.g. ``Qwen3.6-27B-UD-Q4_K_XL.gguf``) word-wrap in
    the narrow left rail; with the rail line clipped to one row the whole model
    token silently disappeared, leaving ``"llama_cpp / "`` on screen.
    """
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 1)].rstrip() + "…"


def build_console_model_section_lines(
    summary: ConsoleSettingsSummaryState,
) -> tuple[str, str]:
    """Build the two compact Model rail-section lines from summary rows.

    Args:
        summary: Preformatted Console settings summary rows.

    Returns:
        Tuple of ``(provider/model line, sampling·context·streaming line)``.
    """
    provider = _summary_row_value(summary.provider_row) or "not selected"
    model = _summary_row_value(summary.model_row) or "no model"
    model = _truncate_model_section_value(model)
    sampling = _summary_row_value(summary.sampling_row).partition(",")[0].strip()
    context = _summary_row_value(summary.context_row).partition(";")[0].strip()
    transport = str(summary.transport_row or "").strip()
    detail_parts = [part for part in (sampling, context, transport) if part]
    return f"{provider} / {model}", " · ".join(detail_parts)


CONSOLE_RAIL_SYSTEM_PREVIEW_MAX_CHARS = 40
CONSOLE_RAIL_SYSTEM_NONE_LINE = "System: none"


def _collapse_system_prompt_preview_whitespace(text: str) -> str:
    """Collapse a (possibly multi-line) system prompt onto a single rail row."""
    return " ".join(text.split())


def build_console_rail_system_line(system_prompt: str | None) -> str:
    """Build the Model rail-section ``System: <preview>`` line.

    Mirrors ``build_console_model_section_lines``'s long-value handling
    (task-186): the rail line is clipped to one row, so a long or
    multi-line system prompt must be collapsed to a single line AND
    truncated in the text itself -- not left to CSS ``text-overflow:
    ellipsis`` alone -- or it silently word-wraps onto the hidden second
    row. An unset (``None``/blank) system prompt renders the dim
    ``"System: none"`` sentinel line instead.

    Args:
        system_prompt: The session's current system prompt text, or
            ``None``/blank when unset. This is a display-only preview --
            the value is collapsed/truncated here but never mutated in
            storage or in the provider payload.

    Returns:
        ``"System: none"`` when ``system_prompt`` is ``None`` or blank;
        otherwise ``"System: <preview>"`` with the prompt collapsed to a
        single line and truncated to
        ``CONSOLE_RAIL_SYSTEM_PREVIEW_MAX_CHARS`` characters.
    """
    normalized = str(system_prompt or "").strip()
    if not normalized:
        return CONSOLE_RAIL_SYSTEM_NONE_LINE
    preview = _truncate_model_section_value(
        _collapse_system_prompt_preview_whitespace(normalized),
        CONSOLE_RAIL_SYSTEM_PREVIEW_MAX_CHARS,
    )
    return f"System: {preview}"


_CUSTOM_AND_LEGACY_GROUP_KEYS = frozenset(
    provider_config_key(key) for key in PROVIDER_CUSTOM_GROUP_KEYS
)


def _provider_option_group_rank(
    provider_key: str,
    requires_api_key_by_provider: Mapping[str, bool],
) -> int:
    """Return the Settings/Wizard provider-group rank for an option key.

    Mirrors the F4 Settings and First-Run Wizard taxonomy (task-180): cloud
    providers first, then local ones, then custom slots and legacy aliases
    together. Off-catalog keys (e.g. configured WIP providers) sort with the
    custom-and-legacy group rather than interleaving into cloud/local.
    """
    if provider_key in _CUSTOM_AND_LEGACY_GROUP_KEYS:
        return 2
    if provider_key not in requires_api_key_by_provider:
        return 2
    return 0 if requires_api_key_by_provider[provider_key] else 1


def build_console_provider_options(
    providers_models: Mapping[str, Sequence[str]],
    app_config: Mapping[str, object] | None = None,
) -> list[ConsoleSettingsOption]:
    """Return grouped, display-ordered Console provider options.

    Options are ordered by the shared provider-group taxonomy (Cloud, Local,
    Custom & legacy) and then by display name, so consumers that render
    ``provider_display_name`` labels (the Console settings modal) present the
    same visible order the Settings screen and First-Run Wizard teach, instead
    of an alphabetical-by-config-key order that shuffles the display labels.
    Option values stay raw provider config keys (task-191).

    Args:
        providers_models: Configured provider -> model ids mapping feeding
            the WIP-marked configured-provider options.
        app_config: Full CLI config mapping. When given, registry entries
            (ADR-343) follow the sorted built-ins as a final run: value
            ``custom-ep:<slug>``, label ``display_name``, ordered by creation
            (config file order) then display name. None keeps the
            built-in-only result so existing callers are unchanged.

    Returns:
        The ordered provider options, values carrying raw provider config
        keys (or registry ids) and labels carrying display names.
    """
    supported_provider_keys = supported_console_provider_readiness_keys(
        CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS
    )
    catalog_entries = supported_console_provider_catalog(
        CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS
    )
    requires_api_key_by_provider = {
        entry.readiness_key: entry.requires_api_key for entry in catalog_entries
    }

    def _option_sort_key(provider_key: str) -> tuple[int, str, str]:
        group_rank = _provider_option_group_rank(
            provider_key, requires_api_key_by_provider
        )
        return (
            group_rank,
            provider_display_name(provider_key).casefold(),
            provider_key,
        )

    provider_keys = sorted(
        {
            *(
                key
                for key in (
                    provider_config_key(provider) for provider in providers_models
                )
                if key
            ),
            *(entry.readiness_key for entry in catalog_entries),
        },
        key=_option_sort_key,
    )
    options = [
        ConsoleSettingsOption(
            label=provider_key
            if provider_key in supported_provider_keys
            else f"{provider_key} (WIP)",
            value=provider_key,
        )
        for provider_key in provider_keys
    ]
    options.extend(_custom_endpoint_provider_options(app_config))
    return options


def _custom_endpoint_provider_options(
    app_config: Mapping[str, object] | None,
) -> list[ConsoleSettingsOption]:
    """Return registry entry options appended after the built-in providers.

    ADR-343: each valid ``[custom_endpoints.<slug>]`` entry renders as a
    first-class provider option -- value ``custom-ep:<slug>``, label
    ``display_name`` -- after the built-in Custom & legacy group, ordered by
    creation (config file order, which ``load_custom_endpoints`` preserves)
    then display name. ``app_config=None`` yields no entries so callers that
    predate the registry keep their built-in-only result.
    """
    if app_config is None:
        return []
    # Lazy import: custom_endpoint_registry imports this module for URL
    # normalization, so a module-level import would cycle.
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        CUSTOM_ENDPOINT_ID_PREFIX,
        load_custom_endpoints,
    )

    indexed_entries = sorted(
        enumerate(load_custom_endpoints(app_config).values()),
        key=lambda indexed: (indexed[0], indexed[1].display_name.casefold()),
    )
    return [
        ConsoleSettingsOption(
            label=entry.display_name,
            value=f"{CUSTOM_ENDPOINT_ID_PREFIX}{entry.slug}",
        )
        for _creation_index, entry in indexed_entries
    ]


def build_console_model_options(
    provider: str,
    providers_models: Mapping[str, Sequence[str]],
    current_model: str | None = None,
) -> list[ConsoleSettingsOption]:
    """Return model options for a provider, preserving the current model."""
    provider_key = provider_config_key(provider)
    model_values: list[str] = []

    current_model_value = normalize_console_model_value(current_model)
    if current_model_value and current_model_value not in model_values:
        model_values.append(current_model_value)

    for configured_provider, configured_models in providers_models.items():
        if provider_config_key(configured_provider) != provider_key:
            continue
        for configured_model in configured_models:
            configured_model_value = normalize_console_model_value(configured_model)
            if configured_model_value and configured_model_value not in model_values:
                model_values.append(configured_model_value)

    return [ConsoleSettingsOption(label=model, value=model) for model in model_values]


class ConsoleValueLayer(str, Enum):
    """One layer of the Console parameter stack (ADR-344), highest first."""

    EDITED_DRAFT = "edited_draft"
    THIS_CHAT = "this_chat"
    MODEL_DEFAULT = "model_default"
    CONSOLE_PROVIDER_DEFAULT = "console_provider_default"
    CUSTOM_ENDPOINT_PARAMS = "custom_endpoint_params"
    CHAT_DEFAULTS = "chat_defaults"
    PROVIDER_SCALARS = "provider_scalars"
    BUILT_IN = "built_in"


#: Spec §6 Source words. Every layer maps to exactly one; the three
#: provider-scoped config layers share "provider", because "Console Behavior"
#: names only the global fallbacks (chat_defaults) that Settings page edits.
CONSOLE_VALUE_SOURCE_WORDS: Mapping[ConsoleValueLayer, str] = MappingProxyType(
    {
        ConsoleValueLayer.EDITED_DRAFT: "edited *",
        ConsoleValueLayer.THIS_CHAT: "this chat",
        ConsoleValueLayer.MODEL_DEFAULT: "model default",
        ConsoleValueLayer.CONSOLE_PROVIDER_DEFAULT: "provider",
        ConsoleValueLayer.CUSTOM_ENDPOINT_PARAMS: "provider",
        ConsoleValueLayer.CHAT_DEFAULTS: "Console Behavior",
        ConsoleValueLayer.PROVIDER_SCALARS: "provider",
        ConsoleValueLayer.BUILT_IN: "built-in",
    }
)


def _console_default_layers(
    app_config: Mapping[str, object],
    provider: str | None,
    model: str | None,
    *,
    excluded_model_profile_fields: frozenset[str] = frozenset(),
    extra_sources: Sequence[Mapping[str, object]] = (),
) -> tuple[
    EffectiveChatConfiguration,
    tuple[tuple[ConsoleValueLayer, Mapping[str, object]], ...],
]:
    """The default chain's sources in precedence order, each with its layer.

    The one spelling of the order that both the default builder and the
    Source-word resolver walk.
    """
    chat_defaults = _chat_defaults_with_streaming_compat(
        _mapping_value(app_config, "chat_defaults")
    )
    effective = resolve_effective_chat_configuration(
        app_config,
        provider=provider,
        model=model,
    )
    provider_settings = console_provider_settings(app_config, effective.provider)
    model_profile = _model_default_profile(provider_settings, effective.model)
    if excluded_model_profile_fields:
        model_profile = {
            name: value
            for name, value in model_profile.items()
            if name not in excluded_model_profile_fields
        }
    # TASK-342: [console.provider_defaults.<provider>] holds ONLY values the
    # Console's Save-as-default wrote, so it outranks everything except a
    # model profile. chat_defaults stays ahead of raw [api_settings.*]
    # scalars (f14d22dc3, review feedback): factory provider templates carry
    # sampling values for every provider and must not shadow user-tuned
    # global defaults — which is precisely why saved defaults need their own
    # section instead of writing into api_settings.
    saved_defaults = _mapping_value(
        _mapping_value(_mapping_value(app_config, "console"), "provider_defaults"),
        effective.provider,
    )
    return effective, (
        (ConsoleValueLayer.MODEL_DEFAULT, model_profile),
        (ConsoleValueLayer.CONSOLE_PROVIDER_DEFAULT, saved_defaults),
        *((ConsoleValueLayer.CUSTOM_ENDPOINT_PARAMS, extra) for extra in extra_sources),
        (ConsoleValueLayer.CHAT_DEFAULTS, chat_defaults),
        (ConsoleValueLayer.PROVIDER_SCALARS, provider_settings),
    )


_CONSOLE_INT_FIELDS = frozenset({"top_k", "max_tokens", "seed", "thinking_budget_tokens"})
_CONSOLE_STRING_FIELDS = frozenset(
    {"reasoning_effort", "reasoning_summary", "verbosity", "thinking_effort"}
)


def _value_from_source(source: Mapping[str, object], name: str) -> object:
    """One source's usable value for ``name`` under the builder's own coercion."""
    if name == "streaming":
        return _bool_setting_from_sources((source,), name, None)
    if name in _CONSOLE_INT_FIELDS:
        return _optional_int_setting_from_sources((source,), name)
    if name in _CONSOLE_STRING_FIELDS:
        return _optional_string_setting_from_sources((source,), name)
    return _optional_float_setting_from_sources((source,), name)


def chat_defaults_value(app_config: Mapping[str, object], name: str) -> object:
    """Read one generation field from ``[chat_defaults]`` as a new chat does.

    That one layer, under the default chain's own reading (legacy
    ``enable_streaming`` bridged, the builder's coercion), so Settings shows
    the value a new chat would take from there (TASK-33007.7).

    Args:
        app_config: The configuration snapshot the default chain reads.
        name: A generation field, e.g. ``"temperature"``.

    Returns:
        The usable value, or ``None`` when ``[chat_defaults]`` holds none.
    """
    chat_defaults = _chat_defaults_with_streaming_compat(
        _mapping_value(app_config, "chat_defaults")
    )
    return _value_from_source(chat_defaults, name)


def model_default_value(profile: Mapping[str, object], name: str) -> object:
    """Read one generation field from a model's defaults profile as a new chat does.

    The ``chat_defaults_value`` twin for Settings ▸ Model defaults: the
    builder's own coercion, so a hand-edited ``streaming = "0"`` reads Off
    and a ``top_k = 2.5`` a new chat ignores reads as unset (TASK-33007.7).

    Args:
        profile: One ``[api_settings.<provider>.model_defaults.<model>]``
            table.
        name: A generation field, e.g. ``"temperature"``.

    Returns:
        The usable value, or ``None`` when the profile holds none.
    """
    return _value_from_source(profile, name)


def chat_defaults_held_fields(
    app_config: Mapping[str, object], names: Sequence[str]
) -> frozenset[str]:
    """Name the generation fields ``[chat_defaults]`` holds a usable value for.

    Settings says "Console Behavior" only for these (TASK-33007.7).

    Args:
        app_config: The configuration snapshot the default chain reads.
        names: Generation fields to check.

    Returns:
        The names ``chat_defaults_value`` reads a value for.
    """
    return frozenset(
        name for name in names if chat_defaults_value(app_config, name) is not None
    )


def resolve_console_value_layers(
    app_config: Mapping[str, object],
    provider: str | None,
    model: str | None,
    names: Sequence[str],
    *,
    edited: frozenset[str] = frozenset(),
    chat_settings: ConsoleSessionSettings | None = None,
    extra_sources: Sequence[Mapping[str, object]] = (),
    excluded_model_profile_fields: frozenset[str] = frozenset(),
) -> dict[str, ConsoleValueLayer]:
    """Name the parameter-stack layer each shown value comes from (spec §6).

    Map a result through ``CONSOLE_VALUE_SOURCE_WORDS`` for the Source word.

    Args:
        app_config: The configuration snapshot the default chain reads.
        provider: The shown pair's provider.
        model: The shown pair's model.
        names: Generation fields to resolve.
        edited: Fields the user edited in the open draft.
        chat_settings: This chat's committed settings when the shown pair is
            its pair; a value that differs from the default chain is the chat's.
        extra_sources: ADR-344 registry params, as the builder takes them.
        excluded_model_profile_fields: Model-default fields to skip, so a
            blank Settings model default names the layer it inherits from
            (TASK-33007.5), as the builder takes them. The chain the chat is
            compared with skips them too.

    Returns:
        ``{name: layer}`` for every name.
    """
    _effective, layers = _console_default_layers(
        app_config,
        provider,
        model,
        excluded_model_profile_fields=excluded_model_profile_fields,
        extra_sources=extra_sources,
    )
    defaults = (
        build_default_console_session_settings(
            app_config,
            provider,
            model,
            excluded_model_profile_fields=excluded_model_profile_fields,
            extra_sources=extra_sources,
        )
        if chat_settings is not None
        else None
    )
    resolved: dict[str, ConsoleValueLayer] = {}
    for name in names:
        if name in edited:
            resolved[name] = ConsoleValueLayer.EDITED_DRAFT
        elif defaults is not None and getattr(chat_settings, name) != getattr(
            defaults, name
        ):
            resolved[name] = ConsoleValueLayer.THIS_CHAT
        else:
            resolved[name] = next(
                (
                    layer
                    for layer, source in layers
                    if _value_from_source(source, name) is not None
                ),
                ConsoleValueLayer.BUILT_IN,
            )
    return resolved


def build_default_console_session_settings(
    app_config: Mapping[str, object],
    provider: str | None = None,
    model: str | None = None,
    *,
    excluded_model_profile_fields: frozenset[str] = frozenset(),
    extra_sources: Sequence[Mapping[str, object]] = (),
) -> ConsoleSessionSettings:
    """Build default Console settings from chat defaults and provider config.

    Args:
        app_config: The live app configuration snapshot.
        provider: An explicit provider override, or ``None``.
        model: An explicit model override, or ``None``.
        excluded_model_profile_fields: Exact model-profile fields to skip so
            inherited controls can re-resolve lower-precedence defaults.
        extra_sources: Optional param-name → value mappings slotted between
            the saved console provider defaults and ``chat_defaults`` in the
            precedence walk (ADR-344: registry entry params ride this seam).
            The default ``()`` keeps the source order unchanged.
    """
    effective, layers = _console_default_layers(
        app_config,
        provider,
        model,
        excluded_model_profile_fields=excluded_model_profile_fields,
        extra_sources=extra_sources,
    )
    default_sources = tuple(source for _layer, source in layers)

    return ConsoleSessionSettings(
        provider=effective.provider,
        model=effective.model,
        base_url=effective.base_url,
        temperature=_float_setting_from_sources(default_sources, "temperature", 0.7),
        top_p=_float_setting_from_sources(default_sources, "top_p", 0.95),
        min_p=_optional_float_setting_from_sources(default_sources, "min_p"),
        top_k=_optional_int_setting_from_sources(default_sources, "top_k"),
        max_tokens=_optional_int_setting_from_sources(default_sources, "max_tokens"),
        seed=_optional_int_setting_from_sources(default_sources, "seed"),
        presence_penalty=_optional_float_setting_from_sources(
            default_sources, "presence_penalty"
        ),
        frequency_penalty=_optional_float_setting_from_sources(
            default_sources, "frequency_penalty"
        ),
        reasoning_effort=_optional_string_setting_from_sources(
            default_sources, "reasoning_effort"
        ),
        reasoning_summary=_optional_string_setting_from_sources(
            default_sources, "reasoning_summary"
        ),
        verbosity=_optional_string_setting_from_sources(default_sources, "verbosity"),
        thinking_effort=_optional_string_setting_from_sources(
            default_sources, "thinking_effort"
        ),
        thinking_budget_tokens=_optional_int_setting_from_sources(
            default_sources, "thinking_budget_tokens"
        ),
        streaming=_bool_setting_from_sources(default_sources, "streaming", True),
    )


def default_console_session_settings(
    app_config: Mapping[str, object],
    provider: str | None = None,
    model: str | None = None,
    *,
    excluded_model_profile_fields: frozenset[str] = frozenset(),
) -> ConsoleSessionSettings:
    """The default settings snapshot a NEW Console session starts from.

    `build_default_console_session_settings` plus the one rule that always
    accompanied it at the screen's call site: a llama.cpp session takes no
    `base_url` from configuration (the gateway normalizes the origin at
    send time; a stale configured URL here would pin it).

    Named here, and not left inline in
    `ConsoleSessionController._default_console_session_settings`, because
    task-15860's launch wake builds a session with no screen in existence
    and must start from the same defaults rather than a second spelling of
    them.

    Args:
        app_config: The live app configuration snapshot.
        provider: An explicit provider override (the Console control bar's
            selection when there is a view), or ``None``.
        model: An explicit model override, or ``None``.
        excluded_model_profile_fields: Exact model-profile fields to skip so
            inherited controls can re-resolve lower-precedence defaults.

    Returns:
        The default settings for a new session.
    """
    settings = build_default_console_session_settings(
        app_config,
        provider,
        model,
        excluded_model_profile_fields=excluded_model_profile_fields,
    )
    provider_key = provider_config_key(settings.provider)
    return replace(
        settings,
        base_url=(
            None
            if provider_key in {"llama_cpp", "local_llamacpp"}
            else settings.base_url
        ),
    )


def blank_console_session_settings(
    app_config: Mapping[str, object],
) -> ConsoleSessionSettings:
    """Return config-owned defaults for one eligible blank Console chat."""
    return default_console_session_settings(app_config)


def build_target_default_console_session_settings(
    app_config: Mapping[str, object],
    provider: str,
    model: str | None,
    *,
    excluded_model_profile_fields: frozenset[str] = frozenset(),
) -> ConsoleSessionSettings:
    """Return fresh effective defaults for one provider and literal model ID.

    This is the provider/model rebase entry point. It deliberately delegates to
    :func:`default_console_session_settings` so exact-model profiles keep the
    established precedence and provider endpoint normalization.

    Args:
        app_config: The live application configuration snapshot.
        provider: Provider selected by the settings transaction.
        model: Literal model ID selected by the settings transaction.
        excluded_model_profile_fields: Exact model-profile fields to skip so
            inherited controls can re-resolve lower-precedence defaults.

    Returns:
        A fresh settings value resolved for the exact target.
    """

    return replace(
        default_console_session_settings(
            app_config,
            provider,
            model,
            excluded_model_profile_fields=excluded_model_profile_fields,
        )
    )


def normalized_console_model_profile_overrides(
    app_config: Mapping[str, object],
    provider: str,
    model: str | None,
) -> dict[str, object]:
    """Return valid normalized overrides from one exact model profile.

    Blank and invalid entries are omitted so callers can distinguish inheritance
    from an explicit override. The conversions delegate to the same per-field
    helpers used by :func:`build_default_console_session_settings`; this function
    does not resolve any fallback precedence.
    """

    provider_settings = console_provider_settings(
        app_config,
        _canonical_chat_provider_id(provider),
    )
    profile = _model_default_profile(provider_settings, model)
    sources = (profile,)
    overrides: dict[str, object] = {}

    for name in (
        "temperature",
        "top_p",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
    ):
        value = _optional_float_setting_from_sources(sources, name)
        if value is not None:
            overrides[name] = value
    for name in (
        "top_k",
        "max_tokens",
        "seed",
        "thinking_budget_tokens",
    ):
        value = _optional_int_setting_from_sources(sources, name)
        if value is not None:
            overrides[name] = value
    for name in (
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
    ):
        value = _optional_string_setting_from_sources(sources, name)
        if value is not None:
            overrides[name] = value
    streaming = _bool_setting_from_sources(sources, "streaming", None)
    if streaming is not None:
        overrides["streaming"] = streaming
    return overrides


def resolve_effective_chat_configuration(
    app_config: Mapping[str, object],
    *,
    provider: str | None = None,
    model: str | None = None,
) -> EffectiveChatConfiguration:
    """Resolve canonical chat defaults without mutating loaded configuration.

    ``chat_defaults.model`` belongs to ``chat_defaults.provider`` (ADR-006):
    an explicit provider that is not canonically that provider falls through
    to its own configured model, never the global default model.

    Args:
        app_config: The live app configuration snapshot. Read only; never
            mutated.
        provider: An explicit provider override (e.g. the session's current
            selection), or ``None`` to use ``chat_defaults.provider``.
        model: An explicit model override, or ``None`` to fall back to
            ``chat_defaults.model`` (only when the provider owns it) and then
            the provider's own configured model.

    Returns:
        The effective configuration: the canonical provider id, the resolved
        model (``None`` when nothing is configured), the provider's base URL,
        and ``model_source`` naming which layer supplied the model.
    """
    chat_defaults = _chat_defaults_with_streaming_compat(
        _mapping_value(app_config, "chat_defaults")
    )
    provider_id = _canonical_chat_provider_id(
        _string_value(provider) or _string_setting(chat_defaults, "provider"),
        app_config,
    )
    owns_defaults_model = owns_chat_defaults_model(
        provider_id, chat_defaults, app_config
    )
    provider_settings = console_provider_settings(app_config, provider_id)
    candidates = (
        ("session", model),
        ("chat_defaults", chat_defaults.get("model") if owns_defaults_model else None),
        ("provider_fallback", configured_provider_model(provider_settings)),
    )
    model_source = "none"
    resolved_model = None
    for candidate_source, candidate_model in candidates:
        resolved_model = _string_value(candidate_model)
        if resolved_model is not None:
            model_source = candidate_source
            break

    return EffectiveChatConfiguration(
        provider=provider_id,
        model=resolved_model,
        base_url=_default_base_url(provider_id, provider_settings),
        model_source=model_source,
    )


def owns_chat_defaults_model(
    provider: object,
    chat_defaults: Mapping[str, object],
    app_config: Mapping[str, object] | None = None,
) -> bool:
    """Whether ``chat_defaults.model`` belongs to ``provider`` (ADR-006).

    It belongs only to ``chat_defaults.provider``, compared as canonical
    provider identities (case, legacy aliases, registry-id spelling).

    Args:
        provider: The provider a model is being resolved for.
        chat_defaults: The persisted ``chat_defaults`` mapping.
        app_config: The app config, so a registry id resolves as itself.

    Returns:
        True when ``provider`` canonically is ``chat_defaults.provider``.
    """
    # Lazy import: custom_endpoint_registry imports this module (see
    # _canonical_chat_provider_id).
    from tldw_chatbook.Chat.custom_endpoint_registry import provider_identity_key

    def identity(value: object) -> str:
        # Identity spelling: a registry id may arrive as ``custom_ep:<slug>``.
        canonical = _canonical_chat_provider_id(_string_value(value) or "", app_config)
        return provider_identity_key(canonical)

    return identity(provider) == identity(chat_defaults.get("provider"))


def build_canonical_chat_defaults_mutation(
    effective: EffectiveChatConfiguration,
    app_config: Mapping[str, object] | None = None,
) -> dict[str, dict[str, str]]:
    """Build the canonical provider/model fragment for an explicit save.

    ``app_config`` (when provided) keeps a resolvable custom-ep provider id
    intact instead of collapsing it onto the generic ``custom`` slot, so a
    saved default reboots onto the registry entry it selected.
    """
    chat_defaults: dict[str, str] = {}
    provider_id = _canonical_chat_provider_id(effective.provider, app_config)
    model = _string_value(effective.model)
    if provider_id:
        chat_defaults["provider"] = provider_id
    if model:
        chat_defaults["model"] = model
    return {"chat_defaults": chat_defaults}


def reasoning_effort_hint_for_model(model: str | None) -> frozenset[str] | None:
    """Return the effort values this model family's template consumes.

    Args:
        model: Model identifier as selected in the Console (e.g.
            ``"Qwen3.8-27B"``). Case-insensitive; ``None``/blank allowed.

    Returns:
        The set of ``reasoning_effort`` values the model family's chat
        template consumes, or ``None`` when the family is unknown and no
        hint should be shown.
    """
    lowered = str(model or "").strip().lower()
    if not lowered:
        return None
    if re.search(r"qwen3\.\d", lowered):
        return _QWEN_DOTTED_EFFORT_VALUES
    for needle, values in _REASONING_EFFORT_MODEL_HINTS:
        if needle in lowered:
            return values
    return None


def console_settings_warnings(settings: ConsoleSessionSettings) -> list[str]:
    """Return non-blocking warnings for the Console settings modal.

    Warnings never block a save or send (ADR-311); blocking validation
    lives in :func:`validate_console_session_settings`.

    Args:
        settings: The freshly parsed Console session settings.

    Returns:
        Zero or more user-facing warning strings: an effort value the
        selected model family does not consume; for llama.cpp-family
        providers with a thinking value set — the server requirements note
        (``--jinja``; per-request budget needs llama.cpp b9982+); and for
        thinking effort ``off`` on an always-on-thinking Anthropic model
        (Fable 5 / Mythos 5) — that thinking cannot actually be turned off
        there (TASK-18800: the API rejects the explicit disabled config, so
        the request omits the parameter and adaptive thinking still runs).
    """
    warnings: list[str] = []
    effort = str(settings.reasoning_effort or "").strip().lower()
    has_thinking_value = bool(effort) or settings.thinking_budget_tokens is not None
    if effort:
        hint = reasoning_effort_hint_for_model(settings.model)
        if hint is not None and effort not in hint:
            warnings.append(
                f"Reasoning effort '{effort}' is not consumed by this model "
                f"family; expected one of: {', '.join(sorted(hint))}."
            )
    if has_thinking_value and settings.provider in _LLAMA_CPP_FAMILY_PROVIDERS:
        warnings.append(_LLAMACPP_THINKING_REQUIREMENTS_NOTE)
    thinking_effort = str(settings.thinking_effort or "").strip().lower()
    if thinking_effort == "off" and anthropic_model_rejects_disabled_thinking(
        settings.model
    ):
        warnings.append(
            f"{settings.model} always thinks: the API rejects an explicit "
            "thinking-off setting, so 'off' sends no thinking parameter and "
            "adaptive thinking still runs (and is billed)."
        )
    return warnings


def _console_session_settings_structural_errors(
    settings: ConsoleSessionSettings,
    *,
    blank_ok: Collection[str] = (),
    temperature_bounds: tuple[float, float] = TEMPERATURE_RANGE,
) -> list[str]:
    """Return pure shape/range errors shared by live and persisted settings.

    Args:
        settings: The settings to check.
        blank_ok: Required samplers that may be blank: the ones the provider
            does not accept, which the draft rebase commits blank.
        temperature_bounds: The Temperature range to hold it to; the
            provider's own range when the caller knows the provider.

    Returns:
        User-facing errors, empty when the settings are well formed.
    """
    errors: list[str] = []
    if (
        type(settings.provider) is not str
        or not settings.provider.strip()
        or settings.provider != settings.provider.strip()
    ):
        errors.append("Provider is required.")
    if settings.source not in CONSOLE_SESSION_SETTINGS_SOURCES:
        errors.append("Settings source must be derived or user.")

    low, high = temperature_bounds
    if not (
        settings.temperature is None and "temperature" in blank_ok
    ) and not _float_in_range(settings.temperature, low, high):
        errors.append(f"Temperature must be between {low:g} and {high:g}.")
    if not (settings.top_p is None and "top_p" in blank_ok) and not _float_in_range(
        settings.top_p, 0.0, 1.0
    ):
        errors.append("Top P must be between 0 and 1.")
    if not _is_blank_value(settings.min_p) and not _float_in_range(
        settings.min_p, 0.0, 1.0
    ):
        errors.append("Min P must be between 0 and 1.")
    if not _is_blank_value(settings.top_k) and not _optional_int_at_least(
        settings.top_k, 0
    ):
        errors.append("Top K must be 0 or greater.")
    if not _is_blank_value(settings.max_tokens) and not _optional_int_at_least(
        settings.max_tokens, 1
    ):
        errors.append(f"{MODEL_FIELD_LABELS['max_tokens']} must be 1 or greater.")
    if not _is_blank_value(settings.seed) and not _optional_int_at_least(
        settings.seed, 0
    ):
        errors.append("Seed must be 0 or greater.")
    if not _is_blank_value(settings.presence_penalty) and not _float_in_range(
        settings.presence_penalty, -2.0, 2.0
    ):
        errors.append("Presence penalty must be between -2 and 2.")
    if not _is_blank_value(settings.frequency_penalty) and not _float_in_range(
        settings.frequency_penalty, -2.0, 2.0
    ):
        errors.append("Frequency penalty must be between -2 and 2.")
    if (
        not _is_blank_value(settings.reasoning_effort)
        and settings.reasoning_effort not in _REASONING_EFFORT_VALUES
    ):
        errors.append(
            "Reasoning effort must be one of none, minimal, low, medium, high, or xhigh."
        )
    if (
        not _is_blank_value(settings.reasoning_summary)
        and settings.reasoning_summary not in _REASONING_SUMMARY_VALUES
    ):
        errors.append(
            "Reasoning summary must be one of auto, concise, detailed, or none."
        )
    if (
        not _is_blank_value(settings.verbosity)
        and settings.verbosity not in _VERBOSITY_VALUES
    ):
        errors.append("Verbosity must be one of low, medium, or high.")
    if (
        not _is_blank_value(settings.thinking_effort)
        and settings.thinking_effort not in _THINKING_EFFORT_VALUES
    ):
        errors.append(
            f"{MODEL_FIELD_LABELS['thinking_effort']} must be one of off, low, "
            "medium, high, xhigh, or max."
        )
    if not _is_blank_value(
        settings.thinking_budget_tokens
    ) and not _optional_int_at_least(settings.thinking_budget_tokens, 1024):
        errors.append(
            f"{MODEL_FIELD_LABELS['thinking_budget_tokens']} must be at least 1024."
        )

    return errors


def validate_console_session_settings(
    settings: ConsoleSessionSettings,
    *,
    app_config: Mapping[str, object],
) -> list[str]:
    """Return user-facing validation errors for Console settings."""
    unaccepted = {"temperature", "top_p"} - supported_generation_fields(
        settings.provider, settings.model, app_config
    )
    provider_key = provider_config_key(settings.provider)
    # Capture finding 11: the provider's own Temperature range (Anthropic's
    # is 0 to 1), as Settings ▸ Providers & Models holds it.
    errors = _console_session_settings_structural_errors(
        settings,
        blank_ok=unaccepted,
        temperature_bounds=temperature_range(provider_key),
    )
    provider_settings = console_provider_settings(app_config, provider_key)

    if provider_key not in NATIVE_CONSOLE_PROVIDER_KEYS and not _string_value(
        settings.model
    ):
        errors.append("Model is required.")

    base_url = _string_value(settings.base_url)
    if (
        base_url
        and _is_url_based_provider(provider_key, provider_settings)
        and not _valid_base_url(provider_key, base_url)
    ):
        errors.append(f"{MODEL_FIELD_LABELS['endpoint']} must be a valid http(s) URL.")

    return errors


def build_console_settings_readiness(
    settings: ConsoleSessionSettings,
    *,
    app_config: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
    native_provider_keys: set[str] | None = None,
    evidence: ProviderTestEvidence | None = None,
    current_identity: ProviderDraftIdentity | None = None,
    active_run: bool = False,
    background_credentials: bool = True,
    connection_evidence: ProviderConnectionEvidence | None = None,
) -> ConsoleSettingsReadiness:
    """Project one deterministic Console blocker and independent evidence.

    ``connection_evidence`` (TASK-33005.2) is the app's shared owner: without
    explicit ``evidence``, the result settled for this exact connection on any
    surface applies, keyed from values resolved here (no extra config read).
    """
    if type(active_run) is not bool:
        raise ValueError("Active-run state must be boolean.")
    # ADR-343 (registry seam): a custom-ep provider resolves through its
    # registry entry -- the family it executes as, and the entry's persisted
    # endpoint when the session carries no explicit base URL. Lazy import:
    # custom_endpoint_registry imports this module for URL normalization, so
    # a module-level import would cycle.
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        entry_for,
        family_execution_key,
    )

    entry = entry_for(app_config, settings.provider)
    canonical_provider = _canonical_chat_provider_id(
        family_execution_key(entry.family) if entry is not None else settings.provider
    )
    identity = resolve_console_provider_identity(
        canonical_provider,
        handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    )
    provider_key = identity.readiness_key
    supported_keys = _supported_readiness_keys(native_provider_keys)
    send_capable_keys = _send_capable_readiness_keys(native_provider_keys)

    base_url = _string_value(settings.base_url)
    if entry is not None and not base_url:
        # ADR-343: the entry is the persisted endpoint carrier, so a blank
        # session base_url resolves from it instead of the family default.
        base_url = entry.base_url
    provider_settings, provider_configuration_invalid = (
        _provider_settings_with_validity(
            app_config,
            settings.provider if entry is not None else provider_key,
        )
    )
    # TASK-34100.4 review (F3): an entry rides its family's readiness with
    # the family slot's own stranded key lifted (never the entry's key).
    family_readiness = (
        custom_endpoint_family_readiness
        if entry is not None
        else get_provider_readiness
    )
    readiness = family_readiness(
        provider_key,
        app_config,
        environ=environ,
        background_credentials=background_credentials,
    )
    if entry is not None and readiness.ready:
        readiness = (
            custom_endpoint_missing_key_readiness(entry, provider_key, environ)
            or readiness
        )
    # H4 wording: an entry that declares a resolving credential rides the
    # keyless family readiness, whose "No API key is required" copy would
    # misdescribe a send that actually authenticates with that credential.
    declared_credential: str | None = None
    declared_credential_source: CredentialSource = "none"
    if entry is not None and readiness.ready:
        declared = _custom_endpoint_declared_credential(entry, environ)
        if declared is not None:
            declared_credential, declared_credential_source = declared
    if (
        connection_evidence is not None
        and connection_evidence.version  # Nothing tested yet: no key to build.
        and evidence is None
        and (not readiness.requires_api_key or readiness.ready)
    ):
        connection = _console_connection_identity(
            provider_key, base_url, provider_settings, readiness, entry, environ
        )
        evidence = connection_evidence.evidence_for(connection) if connection else None
        current_identity = connection if evidence is not None else current_identity
    exact_identity_evidence = bool(
        evidence is not None
        and current_identity is not None
        and evidence.identity == current_identity
        and current_identity.provider_key == provider_key
    )

    credential_source: CredentialSource = "none"
    if readiness.api_key_source:
        credential_source = (
            "environment" if readiness.api_key_source.startswith("env:") else "stored"
        )

    if (
        not readiness.requires_api_key
        and declared_credential is None
        # TASK-34100.4 review round 2: a keyless provider whose saved key is
        # still encrypted is blocked on that credential (readiness reports
        # credential_missing); "not_required" would leave the blocked
        # readiness without a blocker.
        and readiness.configuration_issue != "credential_missing"
    ):
        credential: CredentialFacet = "not_required"
        credential_source = "none"
    elif not readiness.ready:
        credential = "missing"
    else:
        if (
            exact_identity_evidence
            and evidence is not None
            and evidence.credential in _ACCEPTED_CREDENTIALS
        ):
            credential = evidence.credential
        else:
            credential = "present_unverified"
        if declared_credential is not None:
            # PR-2646 review: the resolved entry credential is what the
            # gateway actually sends, so the structured facet must present
            # it (naming its provenance) instead of "not_required" -- the
            # keyless-family facet only stands for entries that declare
            # nothing.
            credential_source = declared_credential_source

    evidence_is_current = bool(
        exact_identity_evidence and (not readiness.requires_api_key or readiness.ready)
    )
    snapshot = readiness.snapshot(
        selected_model=settings.model,
        evidence=evidence,
        current_identity=current_identity if evidence_is_current else None,
    )

    generation_at = None
    if evidence_is_current and evidence is not None:
        # Qodo #2958: a paid test proves only the model it sent.
        tested = evidence.for_model(normalize_console_model_value(settings.model))
        generation: GenerationFacet = tested.generation
        generation_category = tested.generation_category
        generation_at = tested.generation_observed_at
    elif evidence is not None and evidence.generation in {
        "succeeded",
        "failed",
        "changed_since_test",
    }:
        generation = "changed_since_test"
        generation_category = None
    else:
        generation = "not_tested"
        generation_category = None

    provider_supported = provider_key in supported_keys
    execution_supported = provider_key in send_capable_keys
    # TASK-39151: with no session endpoint, llama.cpp sends to its fallback
    # chain's endpoint (environment, Console override, config), so check it.
    checked_url = base_url or (
        _console_endpoint_restart_fallback(
            provider_key, provider_settings, app_config, environ
        )
        if entry is None and provider_key in {"llama_cpp", "local_llamacpp"}
        else None
    )
    endpoint_invalid = bool(
        checked_url
        and _is_url_based_provider(provider_key, provider_settings)
        and not _valid_base_url(provider_key, checked_url)
    )
    invalid_endpoint = (
        _invalid_endpoint_kind(provider_key, checked_url) if endpoint_invalid else None
    )
    endpoint_not_saved = bool(
        base_url
        and not identity.uses_direct_llama_path
        and _is_url_based_provider(provider_key, provider_settings)
        and _endpoint_differs_for_provider(provider_key, base_url, provider_settings)
    )
    model_missing = normalize_console_model_value(settings.model) is None
    send_outcome: ConsoleSendOutcome | None = None

    blocker: ConsoleSettingsBlockerCode | None
    recovery_action: ConsoleSettingsRecoveryAction | None
    if not provider_key:
        blocker, recovery_action = "provider_missing", "select_provider"
    elif not provider_supported or not execution_supported:
        blocker, recovery_action = (
            "provider_unsupported",
            "select_supported_provider",
        )
    elif endpoint_invalid:
        blocker, recovery_action = "endpoint_invalid", "configure_endpoint"
    elif readiness.configuration_issue == "endpoint_missing":
        # ADR-179 (databricks, first provider blocked on a missing endpoint
        # while credentialed): the structured contract maps
        # endpoint_missing to the endpoint_invalid blocker
        # (_CONFIGURATION_ISSUE_BLOCKER), which outranks credential_missing
        # in _BLOCKER_PRECEDENCE -- the builder must pick it here or
        # _validate_console_blocker_contract rejects the snapshot and the
        # modal raises instead of showing the workspace-URL remedy.
        blocker, recovery_action = "endpoint_invalid", "configure_endpoint"
    elif provider_configuration_invalid or (
        readiness.configuration_issue == "invalid_settings"
    ):
        blocker, recovery_action = (
            "provider_configuration_invalid",
            "review_provider_settings",
        )
    elif endpoint_not_saved:
        blocker, recovery_action = "endpoint_not_saved", "save_endpoint"
    elif credential == "missing":
        blocker, recovery_action = "credential_missing", "configure_credential"
    elif model_missing:
        blocker, recovery_action = "model_missing", "select_model"
    # TASK-34100.2 AC#2: what Console's send preflight would refuse, from the
    # same inputs, then how this model's last real send ended. Evaluated only
    # when nothing above blocks: the capacity check reads the model catalog,
    # which loads the config on its first use.
    elif console_reply_capacity_exhausted(
        provider_key,
        str(settings.model),
        _reply_max_tokens(settings, app_config, provider_key),
        _send_endpoint(provider_key, base_url, provider_settings),
    ):
        blocker, recovery_action = "capacity_exhausted", "adjust_model_limits"
    elif snapshot.endpoint == "unreachable":
        blocker, recovery_action = _endpoint_failure_blocker(snapshot.category)
    elif active_run:
        # Before the soft send outcomes: a retry in flight is a run, and the
        # run keeps Settings from mutating mid-turn (review round 1).
        blocker, recovery_action = "active_run", "wait_for_active_run"
    elif (
        send_outcome := CONSOLE_SEND_OUTCOMES.outcome_for(
            provider_key, str(settings.model), readiness.api_key
        )
    ) in {"unauthorized", "key_check_rejected"}:
        blocker, recovery_action = "key_rejected_on_send", "configure_credential"
    elif send_outcome == "not_found":
        blocker, recovery_action = "model_unavailable", "select_model"
    else:
        blocker, recovery_action = None, None

    native_send_supported = blocker is None
    display_name = provider_display_name(provider_key)
    if blocker == "endpoint_invalid":
        if not endpoint_invalid:
            # Missing-endpoint copy path (see the endpoint_missing branch
            # above): nothing the user typed is invalid, so the remedy is
            # the readiness copy naming the required endpoint setting
            # (e.g. Databricks's per-account workspace URL), not the
            # generic malformed-URL wording.
            detail = readiness.user_message
        else:
            detail = (
                INVALID_LLAMACPP_BASE_URL_COPY
                if provider_key in NATIVE_CONSOLE_PROVIDER_KEYS
                else "Provider blocked: invalid base URL. Use an http(s) URL."
            )
        label = "Invalid URL"
    elif blocker == "endpoint_not_saved":
        label = "Endpoint not saved"
        detail = unsaved_endpoint_copy(base_url, provider_settings)
    elif blocker in {"provider_missing", "provider_unsupported"}:
        label = "Unknown"
        detail = readiness.user_message
        if blocker == "provider_unsupported":
            detail = (
                f"Provider blocked: '{provider_key}' is not available in Console yet. "
                "Choose a supported provider."
            )
    elif blocker == "credential_missing":
        label = "Missing key"
        detail = readiness.user_message
    elif blocker == "model_missing":
        label = "Missing model"
        detail = readiness.user_message
    elif blocker in {"endpoint_unreachable", "credential_rejected"}:
        label = "Not ready"
        reason = str(snapshot.category or "error").replace("_", " ")
        detail = f"{display_name} failed its last connection test ({reason})."
    elif blocker in MODEL_LEVEL_BLOCKERS | {"key_rejected_on_send"}:
        label = "Not ready"
        detail = _model_level_blocker_detail(
            blocker,
            display_name,
            str(settings.model),
            _reply_max_tokens(settings, app_config, provider_key),
            provider_key, key_checked=send_outcome == "key_check_rejected",
            endpoint=_send_endpoint(provider_key, base_url, provider_settings),
        )
    elif blocker is not None:
        label = "Not ready"
        detail = readiness.user_message
    else:
        label = "Ready"
        detail = readiness.user_message
        if declared_credential is not None:
            detail = (
                f"{readiness.provider} is ready. "
                f"Credential: {declared_credential}."
            )

    configuration = snapshot.configuration
    configuration_issue = snapshot.configuration_issue
    if provider_configuration_invalid:
        configuration = "incomplete"
        configuration_issue = "invalid_settings"

    return ConsoleSettingsReadiness(
        label=label,
        detail=detail,
        native_send_supported=native_send_supported,
        operability="ready_to_send" if blocker is None else "not_ready",
        blocker=blocker,
        recovery_action=recovery_action,
        provider_display_name=display_name,
        configuration=configuration,
        configuration_issue=configuration_issue,
        credential=credential,
        credential_source=credential_source,
        endpoint=snapshot.endpoint,
        endpoint_category=snapshot.category,
        model=snapshot.model,
        generation=generation,
        generation_category=generation_category,
        subscription_status=readiness.subscription_status,
        connection=current_identity if evidence_is_current else None,
        observed_at=evidence.observed_at if evidence_is_current and evidence else None,
        generation_observed_at=generation_at,
        evidence=evidence if evidence_is_current else None,
        invalid_endpoint=invalid_endpoint,
    )


def _invalid_endpoint_kind(
    provider_key: str, value: str
) -> Literal["needs_scheme", "invalid"]:
    """Whether an invalid endpoint only lacks its scheme (TASK-39151 review).

    A host and port, or an IP literal, that the contract refuses only because
    it was typed without http:// or https://. A value without a port that is
    not an IP literal could be a pasted key, which must never be told to add
    a scheme (TASK-39042), so it is just "invalid".
    """
    if resolve_provider_endpoint(provider_key, value).errors != (
        REMOTE_ENDPOINT_SCHEME_REQUIRED,
    ):
        return "invalid"
    if safe_endpoint_display(value) != "invalid endpoint":
        return "needs_scheme"
    try:
        ip_address(urlparse(f"http://{value.strip()}").hostname or "")
    except ValueError:
        return "invalid"
    return "needs_scheme"


def _default_supported_readiness_keys() -> frozenset[str]:
    """Return the no-injection supported set, computed once.

    TASK-18909: ``build_console_settings_readiness`` runs ~400 times during
    one warm Console screen switch, and this set is a pure function of a
    module constant -- recomputing it resolved provider identity for all
    29 handler keys on every call (24k resolutions, the largest app-side
    cost of the switch). Cached at first use; ``_supported_readiness_keys``
    retains the test-injection seam uncached.
    """
    return supported_console_provider_readiness_keys(
        CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    )


_default_supported_readiness_keys = functools.cache(_default_supported_readiness_keys)


def _supported_readiness_keys(
    native_provider_keys: set[str] | None = None,
) -> frozenset[str]:
    """Return readiness keys accepted by Console readiness.

    ``native_provider_keys`` is retained for older tests/callers that injected a
    support set before generic Console provider support existed.
    """
    if native_provider_keys is None:
        return _default_supported_readiness_keys()
    supported_keys = _default_supported_readiness_keys()
    injected_keys = frozenset(
        resolve_console_provider_identity(
            provider,
            handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
        ).readiness_key
        for provider in native_provider_keys
    )
    return supported_keys | injected_keys


def _send_capable_readiness_keys(
    native_provider_keys: set[str] | None = None,
) -> frozenset[str]:
    """Return readiness keys that currently have a wired Console send path."""
    if native_provider_keys is None:
        # Same constant inputs as the supported set (see
        # `_default_supported_readiness_keys`); one shared cache serves both.
        return _default_supported_readiness_keys()
    send_capable_keys = _default_supported_readiness_keys()
    injected_keys = frozenset(
        resolve_console_provider_identity(
            provider,
            handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
        ).readiness_key
        for provider in native_provider_keys
    )
    return send_capable_keys | injected_keys


def build_console_settings_summary_state(
    settings: ConsoleSessionSettings,
    context_estimate: ConsoleSettingsContextEstimate,
    readiness: ConsoleSettingsReadiness,
) -> ConsoleSettingsSummaryState:
    """Build compact display rows for the Console settings summary widget."""
    provider_label = _string_value(readiness.provider_display_name) or "Provider"
    model_value = _string_value(settings.model)
    model_is_missing = not model_value and readiness.blocker == "model_missing"
    model_label = model_value or ("Missing" if model_is_missing else "Default")
    action_label = "Configure"
    action_tooltip = "Configure Console settings"
    if readiness.recovery_action == "select_model":
        action_label = "Choose Model"
        action_tooltip = "Choose a model for this Console session"

    sampling_parts = [
        f"T {_format_summary_float(settings.temperature)}",
        f"P {_format_summary_float(settings.top_p)}",
    ]
    if settings.min_p is not None:
        sampling_parts.append(f"min_p {_format_summary_float(settings.min_p)}")
    if settings.top_k is not None:
        sampling_parts.append(f"top_k {settings.top_k}")
    if settings.max_tokens is not None:
        sampling_parts.append(f"max_tokens {settings.max_tokens}")
    if settings.seed is not None:
        sampling_parts.append(f"seed {settings.seed}")
    if settings.reasoning_effort:
        sampling_parts.append(f"reasoning {settings.reasoning_effort}")
    elif settings.thinking_effort:
        sampling_parts.append(f"thinking {settings.thinking_effort}")
    if settings.thinking_budget_tokens is not None:
        sampling_parts.append(f"think budget {settings.thinking_budget_tokens}")
    if settings.reasoning_effort or settings.thinking_budget_tokens is not None:
        identity = resolve_console_provider_identity(settings.provider)
        wire_fields = build_local_thinking_payload_fields(
            identity.execution_key,
            settings.reasoning_effort,
            settings.thinking_budget_tokens,
        )
        if wire_fields:
            wire_parts = []
            template_kwargs = wire_fields.get("chat_template_kwargs")
            if template_kwargs:
                rendered = ", ".join(
                    f"{k}={v}" for k, v in sorted(template_kwargs.items())
                )
                wire_parts.append(f"chat_template_kwargs[{rendered}]")
            if "reasoning_budget_tokens" in wire_fields:
                wire_parts.append(
                    f"reasoning_budget_tokens={wire_fields['reasoning_budget_tokens']}"
                )
            if "reasoning_effort" in wire_fields:
                wire_parts.append(f"reasoning_effort={wire_fields['reasoning_effort']}")
            sampling_parts.append("wire: " + "; ".join(wire_parts))

    character_label = sanitize_character_display_label(
        settings.character_label,
        max_characters=180,
    )
    identity_row = (
        f"Character: {character_label}" if character_label else "Assistant: General"
    )

    return ConsoleSettingsSummaryState(
        model_row=f"Model: {model_label}",
        context_row=_format_context_summary_row(context_estimate.label),
        sampling_row=f"Sampling: {', '.join(sampling_parts)}",
        temperature=_format_summary_float(settings.temperature),
        max_tokens=(
            str(settings.max_tokens) if settings.max_tokens is not None else ""
        ),
        streaming="On" if settings.streaming else "Off",
        identity_row=identity_row,
        readiness_label=readiness_words(readiness),
        provider_row=f"Provider: {provider_label}",
        endpoint_row=_format_endpoint_summary_row(settings, readiness),
        credential_row=_format_credential_summary_row(readiness),
        transport_row=f"Streaming: {'on' if settings.streaming else 'off'}",
        action_label=action_label,
        action_tooltip=action_tooltip,
        readiness=readiness,
    )


def build_console_context_estimate(
    messages: Sequence[Mapping[str, str]],
    provider: str,
    model: str | None,
    staged_source_count: int = 0,
    staged_context_summary: str = "",
    max_tokens_response: int | None = None,
    system_prompt: str | None = None,
    *,
    staged_text: str = "",
    token_counter: TokenCounter | None = None,
    token_limit_resolver: TokenLimitResolver | None = None,
    context_window: ContextWindowResolution | None = None,
    history_used_tokens: int | None = None,
) -> ConsoleSettingsContextEstimate:
    """Estimate current context tokens for display in Console settings.

    Args:
        staged_text: Canonical formatted pre-authority evidence used for cost
            estimation (task-6). Passed in by the caller so this builder stays
            pure (no I/O, no bundle parsing here); blank/whitespace text
            contributes nothing. Authoritative send capture may shrink this
            context after rechecking local authority. Folded into `used_tokens`
            as one additional message; `staged_source_count` still drives only
            the label's "; N sources staged" suffix, unchanged.
        history_used_tokens: Already counted settled prefix, if `messages`
            contains only the incremental live draft. The shared chat framing
            base is counted once across both parts.
    """
    model_name = _string_value(model)
    if not model_name:
        return ConsoleSettingsContextEstimate(
            used_tokens=None,
            token_limit=None,
            label="Context: unavailable",
            staged_source_count=staged_source_count,
            staged_context_summary=staged_context_summary,
        )

    provider_key = provider_config_key(provider)
    estimate_messages: list[Mapping[str, str]] = []
    if system_prompt:
        estimate_messages.append({"role": "system", "content": system_prompt})
    estimate_messages.extend(messages)
    if staged_text and staged_text.strip():
        estimate_messages.append({"role": "user", "content": staged_text})

    try:
        counter = token_counter or _estimate_tokens_locally
        limit_resolver = token_limit_resolver or _resolve_token_limit_locally
        used_tokens = counter(list(estimate_messages), model_name, provider_key)
        if history_used_tokens is not None:
            # Chat framing has one base allowance for the whole request.
            # The settled prefix and the live draft are counted separately;
            # subtract the second base when both sides contain messages.
            if history_used_tokens and estimate_messages:
                base = 3 if model_name.startswith(("gpt-3.5", "gpt-4")) else 2
                used_tokens += history_used_tokens - base
            else:
                used_tokens += history_used_tokens
        if context_window is not None:
            token_limit = context_window.tokens
            token_limit_verified = context_window.verified
            token_limit_source = context_window.source
        elif token_limit_resolver is None:
            token_limit, token_limit_verified, token_limit_source = (
                _resolve_token_limit_locally_with_provenance(
                    model_name,
                    provider_key,
                )
            )
        else:
            token_limit = limit_resolver(model_name, provider_key)
            token_limit_verified = True
            token_limit_source = "provided resolver"
    except Exception:
        return ConsoleSettingsContextEstimate(
            used_tokens=None,
            token_limit=None,
            label="Context: unavailable",
            staged_source_count=staged_source_count,
            staged_context_summary=staged_context_summary,
        )

    label = f"{used_tokens:,} / {token_limit:,} tokens"
    if not token_limit_verified:  # TASK-33007 #12: Settings' words for a fallback
        label = f"{label} (assumed; window unknown)"
    if max_tokens_response is not None:
        label = f"{label}; {max_tokens_response:,} response tokens reserved"
    if staged_source_count:
        source_word = "source" if staged_source_count == 1 else "sources"
        label = f"{label}; {staged_source_count} {source_word} staged"

    return ConsoleSettingsContextEstimate(
        used_tokens=used_tokens,
        token_limit=token_limit,
        label=label,
        staged_source_count=staged_source_count,
        staged_context_summary=staged_context_summary,
        token_limit_verified=token_limit_verified,
        token_limit_source=token_limit_source,
    )


def _mapping_value(source: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = source.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _chat_defaults_with_streaming_compat(
    chat_defaults: Mapping[str, object],
) -> Mapping[str, object]:
    """Return chat defaults with the legacy streaming key bridged.

    `chat_defaults.streaming` is the canonical Console default. Older config can
    still provide `chat_defaults.enable_streaming`; it is only read when the
    canonical key is absent.
    """
    if "streaming" in chat_defaults or "enable_streaming" not in chat_defaults:
        return chat_defaults
    compatible_defaults = dict(chat_defaults)
    compatible_defaults["streaming"] = chat_defaults.get("enable_streaming")
    return compatible_defaults


def _canonical_chat_provider_id(
    provider: str | None,
    app_config: Mapping[str, object] | None = None,
) -> str:
    """Return the canonical provider id for a chat-defaults provider value.

    A ``custom-ep:<slug>`` id whose registry entry resolves in ``app_config``
    is returned unchanged (ADR-343: the entry -- not its family -- is the
    persisted default, and the config-key normalization below would mangle
    the prefix's hyphen into an underscore). Every other value keeps the
    legacy-alias and readiness-key fallback chain.
    """
    # Lazy import: custom_endpoint_registry imports this module for URL
    # normalization, so a module-level import would cycle.
    from tldw_chatbook.Chat.custom_endpoint_registry import entry_for

    if app_config is not None and entry_for(app_config, provider) is not None:
        return str(provider)
    normalized = provider_config_key(provider)
    normalized = _LEGACY_CHAT_PROVIDER_ALIASES.get(normalized, normalized)
    return resolve_console_provider_identity(
        normalized,
        handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    ).readiness_key


def console_provider_settings(
    app_config: Mapping[str, object], provider_key: str, *, strict: bool = False
) -> Mapping[str, object]:
    """Return one provider's settings: the Console's one lookup (TASK-33004.2).

    Args:
        app_config: The full CLI config mapping.
        provider_key: Provider id or config key.
        strict: Re-raise a malformed or ambiguous table (the gateway's policy)
            instead of treating it as empty.

    Returns:
        The registry entry's flattened view for a ``custom-ep:`` id (ADR-343),
        else the matching ``api_settings`` table, else an empty mapping.

    Raises:
        ProviderSettingsError: Only when ``strict`` and the table is invalid.
    """
    # Lazy import: custom_endpoint_registry imports this module for URL
    # normalization, so a module-level import would cycle.
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        custom_endpoint_provider_settings,
    )

    custom_endpoint_settings = custom_endpoint_provider_settings(
        app_config, provider_key
    )
    if custom_endpoint_settings is not None:
        return custom_endpoint_settings
    api_settings = _mapping_value(app_config, "api_settings")
    try:
        return provider_settings_for_key(api_settings, provider_key)
    except ProviderSettingsError:
        if strict:
            raise
        return {}


def configured_provider_model(provider_settings: Mapping[str, object]) -> str | None:
    """Return a provider table's own model: ``model``, ``api_model``, ``default_model``.

    TASK-33004.2: the one spelling of this fallback chain. Blank values and
    placeholder sentinels (``None``/``null``) are skipped.
    """
    for key in ("model", "api_model", "default_model"):
        model = normalize_console_model_value(provider_settings.get(key))
        if model is not None:
            return model
    return None


def _custom_endpoint_declared_credential(
    entry: "CustomEndpointEntry",
    environ: Mapping[str, str] | None,
) -> tuple[str, CredentialSource] | None:
    """Return the provenance descriptor and source for an entry credential.

    Companion to :func:`custom_endpoint_missing_key_readiness`: that helper
    blocks when a declared credential does not resolve; this one names the
    credential that DOES flow for a ready entry, so the readiness copy never
    claims "No API key is required" while a send authenticates with one.
    ADR-343 precedence applies (the env reference wins over the stored key).

    Args:
        entry: Resolved registry entry for the session's custom-ep provider.
        environ: Environment mapping, injectable for deterministic tests;
            ``None`` reads ``os.environ``.

    Returns:
        ``("env <VAR>", "environment")`` or ``("stored api_key", "stored")``
        when a declared credential resolves, else None (nothing declared, or
        nothing that resolves).
    """
    env = environ if environ is not None else os.environ
    if entry.api_key_env:
        if (
            resolve_provider_api_key(env.get(entry.api_key_env, "")) is not None
        ):
            return f"env {entry.api_key_env}", "environment"
    if resolve_provider_api_key(entry.api_key) is not None:
        return "stored api_key", "stored"
    return None


def _send_endpoint(
    provider_key: str, base_url: str | None, provider_settings: Mapping[str, object]
) -> str | None:
    """The endpoint a send for this chat goes to (``None`` when none resolves).

    The connection identity and the server-reported window (pass 2) both key
    on it, so it is one rule.
    """
    endpoint = effective_provider_endpoint(provider_key, base_url, provider_settings)
    if endpoint is None and provider_key in {"llama_cpp", "local_llamacpp"}:
        endpoint = DEFAULT_LLAMACPP_BASE_URL  # The origin a new chat sends to.
    return endpoint


def _console_connection_identity(
    provider_key: str,
    base_url: str | None,
    provider_settings: Mapping[str, object],
    readiness: ProviderReadiness,
    entry: CustomEndpointEntry | None,
    environ: Mapping[str, str] | None,
) -> ProviderDraftIdentity | None:
    """Return the connection a send uses, keyed as Chat settings keys it.

    TASK-33005.2: mirrors ``ConsoleSettingsModal._current_connection_probe_
    identity`` (endpoint the send uses, sent key's digest, credential source)
    from values the readiness build already resolved.

    Returns:
        The connection identity, or ``None`` when no endpoint resolves.
    """
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        CUSTOM_ENDPOINT_ID_PREFIX,
        resolve_entry_credential,
    )

    endpoint = _send_endpoint(provider_key, base_url, provider_settings)
    connection = canonical_connection_identity(provider_key, endpoint)
    if connection is None:
        return None
    if entry is not None:
        api_key, source = resolve_entry_credential(entry, environ)
        credential_source = None
    else:
        api_key, source = readiness.api_key, readiness.api_key_source
        credential_source = configured_provider_credential_source(provider_settings)
    if credential_source is None:
        credential_source = (
            "none" if not source else "environment" if source.startswith("env:") else "stored"
        )
    try:
        return ProviderDraftIdentity(
            provider_key=provider_key,
            connection_identity=connection,
            credential_source=credential_source,
            credential_revision=connection_credential_revision(api_key),
            draft_generation=0,
            custom_endpoint_id=(
                f"{CUSTOM_ENDPOINT_ID_PREFIX}{entry.slug}" if entry else None
            ),
        )
    except ValueError:
        return None


def console_send_connection(
    settings: ConsoleSessionSettings,
    *,
    app_config: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
) -> ProviderDraftIdentity | None:
    """Return the connection a send with ``settings`` uses (TASK-33005.5).

    Resolved exactly as :func:`build_console_settings_readiness` resolves it
    before its shared evidence lookup, so a result settled for this identity
    is the one readiness reads back. Blocking: reads config and credentials.

    Args:
        settings: The chat's (or a future chat's) effective settings.
        app_config: The live application configuration snapshot.
        environ: Environment mapping; ``None`` reads ``os.environ``.

    Returns:
        The connection identity, or ``None`` when no endpoint resolves.
    """
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        entry_for,
        family_execution_key,
    )

    entry = entry_for(app_config, settings.provider)
    provider_key = resolve_console_provider_identity(
        _canonical_chat_provider_id(
            family_execution_key(entry.family) if entry is not None else settings.provider
        ),
        handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    ).readiness_key
    base_url = _string_value(settings.base_url) or (entry.base_url if entry else None)
    provider_settings, _invalid = _provider_settings_with_validity(
        app_config, settings.provider if entry is not None else provider_key
    )
    readiness = get_provider_readiness(
        provider_key, app_config, environ=environ, background_credentials=True
    )
    return _console_connection_identity(
        provider_key, base_url, provider_settings, readiness, entry, environ
    )


def provider_left_at_shipped_default(
    app_config: Mapping[str, object], provider_key: str
) -> bool:
    """Whether the user never set this provider up (TASK-33005.6 AC#3 ruling).

    Every loaded profile carries the shipped template's ``[api_settings.*]``
    tables, so having a table proves nothing (``any_provider_configured``
    refuses template endpoints for the same reason); a table that still
    equals the template's does. A ``custom-ep:`` registry entry is always the
    user's own.

    Args:
        app_config: The live configuration snapshot.
        provider_key: A canonical provider key or ``custom-ep:`` id.

    Returns:
        Whether its saved settings are exactly the shipped template's.
    """
    # Lazy: custom_endpoint_registry imports this module (a module-level
    # import would cycle).
    from tldw_chatbook.Chat.custom_endpoint_registry import CUSTOM_ENDPOINT_ID_PREFIX

    if provider_key.startswith(CUSTOM_ENDPOINT_ID_PREFIX):
        return False
    try:
        return provider_settings_for_key(
            app_config.get("api_settings"), provider_key
        ) == provider_settings_for_key(
            DEFAULT_CONFIG_FROM_TOML.get("api_settings"), provider_key
        )
    except ProviderSettingsError:
        return False


def provider_is_template_sample(
    app_config: Mapping[str, object], provider_key: str
) -> bool:
    """Whether this provider still carries the template's sample values.

    TASK-34100.2 AC#9 (entry-exit-handoff-23): the shipped template marks its
    placeholder local sections (TabbyAPI's ``tabby-model``, vLLM's
    ``vllm-model-x``, MLX-LM, the Custom slots...) ``template_sample = true``,
    and saving the provider writes it ``false``. An explicit marker, never a
    comparison with the template's values.

    Args:
        app_config: The live configuration snapshot.
        provider_key: A canonical provider key.

    Returns:
        Whether the provider's table is still marked as a template sample.
    """
    try:
        table = provider_settings_for_key(app_config.get("api_settings"), provider_key)
    except ProviderSettingsError:
        return False
    return isinstance(table, Mapping) and table.get("template_sample") is True


def provider_holds_template_sample_values(
    app_config: Mapping[str, object], provider_key: str
) -> bool:
    """Whether this provider still points at its template sample's endpoint
    and model (TASK-34100.2 pass 3).

    The switcher asks this of a provider ``provider_is_template_sample``
    already marks, to tell a sample the user edited by hand (its own endpoint
    or model, whose refusal stays a NEEDS SETUP row) from one whose other keys
    merely drifted with the template (a profile written before the marker
    still holds Custom's old ``streaming = false``). It never decides the
    marker itself. The endpoint is the one a send would use, so a marked
    table sent to the user's own ``api_base_url`` is theirs (review round 1,
    R1-2), and a sample from an older template still matches its frozen
    values (R1-3).

    Args:
        app_config: The live configuration snapshot.
        provider_key: A canonical provider key.

    Returns:
        Whether the provider's endpoint and model equal the template's.
    """
    try:
        table = provider_settings_for_key(app_config.get("api_settings"), provider_key)
        template = provider_settings_for_key(
            DEFAULT_CONFIG_FROM_TOML.get("api_settings"), provider_key
        )
        pre_marker = provider_settings_for_key(PRE_MARKER_TEMPLATE_SAMPLES, provider_key)
    except ProviderSettingsError:
        return False
    return (
        isinstance(table, Mapping)
        and isinstance(template, Mapping)
        and holds_template_sample_identity(table, template, pre_marker or None)
    )


def custom_endpoint_missing_key_readiness(
    entry: "CustomEndpointEntry",
    provider_key: str,
    environ: Mapping[str, str] | None,
) -> ProviderReadiness | None:
    """Return missing-key readiness for an entry whose declared credential
    does not resolve, or None when it resolves or nothing is declared.

    ADR-343: entry credentials mirror provider-settings semantics — the env
    reference wins over the stored key, and an entry declaring neither rides
    the plain family readiness. An entry that declares a credential the
    environment cannot satisfy blocks the send instead of silently sending
    keyless through a family that is otherwise keyless-ready.

    Args:
        entry: Resolved registry entry for the session's custom-ep provider.
        provider_key: Family readiness key the entry executes as.
        environ: Environment mapping, injectable for deterministic tests;
            ``None`` reads ``os.environ``.

    Returns:
        A not-ready ``ProviderReadiness`` naming the recovery, or None when
        the family readiness should stand.
    """
    env = environ if environ is not None else os.environ
    stored_key = resolve_provider_api_key(entry.api_key)
    env_key = (
        resolve_provider_api_key(env.get(entry.api_key_env, ""))
        if entry.api_key_env
        else None
    )
    if env_key is not None or stored_key is not None:
        return None
    # TASK-34100.4 AC #10 (part 2): a stored key that is still `enc:`
    # ciphertext is declared but unreadable. Letting it ride the keyless
    # family read "Ready · not tested" and sent no key to a server that
    # evidently wants one (a bare HTTP 401).
    stranded = is_encrypted_config_value(entry.api_key)
    if not entry.api_key_env and not stranded:
        return None
    # Lazy import: custom_endpoint_registry imports this module.
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        CUSTOM_ENDPOINT_ID_PREFIX,
        ENV_VAR_NAME_RULE_COPY,
    )

    # TASK-33002.12: a name the record cannot carry reads as the entry's id.
    label = safe_provider_label(
        entry.display_name, f"{CUSTOM_ENDPOINT_ID_PREFIX}{entry.slug}"
    )
    if entry.api_key_env and not validate_env_var_reference(entry.api_key_env):
        # Qodo #2876: a hand-edited name the record rejects was an unhandled
        # ValueError. Name the problem, never the value (it may be a secret
        # pasted into the name field).
        return ProviderReadiness(
            provider=label,
            provider_key=provider_key,
            requires_api_key=True,
            ready=False,
            api_key=None,
            api_key_source=None,
            env_var=None,
            reason="Invalid provider settings",
            recovery=(
                f"The '{label}' endpoint's credential env var name is invalid. "
                f"{ENV_VAR_NAME_RULE_COPY} Fix it under Custom endpoints."
            ),
        )
    if stranded:
        reason = STILL_ENCRYPTED_REASON
        recovery = (
            f"Re-enter or clear the stored api_key for the '{label}' endpoint "
            "under Custom endpoints; it was encrypted with a master password "
            "this session does not have."
        )
    elif entry.api_key_env:
        reason = "Missing API key"
        recovery = (
            f"Set {entry.api_key_env} or update the stored api_key for the "
            f"'{label}' endpoint."
        )
    else:
        reason = "Missing API key"
        recovery = f"Update the stored api_key for the '{label}' endpoint."
    return ProviderReadiness(
        provider=label,
        provider_key=provider_key,
        requires_api_key=True,
        ready=False,
        api_key=None,
        api_key_source=None,
        env_var=entry.api_key_env,
        reason=reason,
        recovery=recovery,
    )



def _provider_settings_with_validity(
    app_config: Mapping[str, object], provider_key: str
) -> tuple[Mapping[str, object], bool]:
    """Return selected provider settings and whether their table is malformed."""

    # ADR-343: a registry entry is its own settings carrier, so an entry
    # provider is never "missing from api_settings" -- route it through the
    # registry-aware reader and skip the table-validity walk.
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        custom_endpoint_provider_settings,
    )

    custom = custom_endpoint_provider_settings(app_config, provider_key)
    if custom is not None:
        return custom, False
    raw_api_settings = app_config.get("api_settings", {})
    if not isinstance(raw_api_settings, Mapping):
        return {}, "api_settings" in app_config
    try:
        settings = provider_settings_for_key(raw_api_settings, provider_key)
    except ProviderSettingsError:
        return {}, True
    if provider_key in {"moonshot", "qwencloud", "zai"}:
        return settings, False
    for configured_provider, configured_settings in raw_api_settings.items():
        if provider_config_key(configured_provider) != provider_key:
            continue
        return settings, not isinstance(configured_settings, Mapping)
    return settings, False


def _endpoint_failure_blocker(
    category: EndpointFailureCategory | None,
) -> tuple[ConsoleSettingsBlockerCode, ConsoleSettingsRecoveryAction]:
    """Map bounded endpoint evidence to one actionable recovery."""

    if category in {"unauthorized", "forbidden"}:
        return "credential_rejected", "configure_credential"
    if category in {"timeout", "connection_refused", "connection_error"}:
        return "endpoint_unreachable", "retry_connection"
    return "endpoint_unreachable", "review_provider_settings"


# TASK-33005.3 (spec §5): the one readiness vocabulary. Every model surface --
# status chip, rail, setup card, switcher rows, Chat settings, Settings' test
# result -- reads its word from these maps; nothing else spells one.
READY_NOT_TESTED = "Ready · not tested"
#: "Not ready · <reason>" per blocker. Tests/Chat/test_readiness_words.py
#: fails for a blocker without one; the longest fits the switcher's column.
_NOT_READY_REASONS: Mapping[str, str] = MappingProxyType(
    {
        "provider_missing": "no provider",
        "provider_unsupported": "unsupported",
        "provider_configuration_invalid": "check settings",
        "endpoint_invalid": "invalid URL",
        "endpoint_not_saved": "endpoint unsaved",
        "credential_missing": "no key",
        "credential_rejected": "key rejected",
        "model_missing": "no model",
        "endpoint_unreachable": "unreachable",
        "active_run": "run active",
        "readiness_unknown": "check settings",
        # TASK-34100.2 AC#2: the reply reservation fills the model's window.
        "capacity_exhausted": "window too small",
        "key_rejected_on_send": "key rejected",
        "model_unavailable": "model not found",
    }
)
#: A known connection failure says what happened ("refused" gains the port).
_FAILURE_REASONS: Mapping[str, str] = MappingProxyType(
    {
        "timeout": "timed out",
        "connection_refused": "refused",
        "unauthorized": "key rejected",
        "forbidden": "key rejected",
        "http_status": "server error",
        "invalid_payload": "bad response",
        "connection_error": "unreachable",
        # TASK-34100.6 (review coverage-19, provider-15).
        "tls_untrusted": "untrusted cert",
        "tls_plain_http": "plain http",
    }
)
#: A Claude subscription has a login, not a key.
_SUBSCRIPTION_REASONS: Mapping[str, str] = MappingProxyType(
    {"pending": "checking login", "expired": "login expired", "missing": "no login"}
)
#: Each setup verdict reads as a blocker's "Not ready · <reason>", as Ready
#: qualified by its "evidence", or (None) as Ready with no current evidence.
_VERDICT_READINESS: Mapping[str, str | None] = MappingProxyType(
    {
        "incomplete": "provider_configuration_invalid",  # its issue refines it
        "model_missing": "model_missing",
        "connection_failed": "endpoint_unreachable",  # its category refines it
        "testing": "evidence",
        "not_tested": "evidence",
        "model_listing_unavailable": "evidence",
        "model_unconfirmed": "evidence",
        "verified": "evidence",
        "changed_since_test": None,
    }
)


def readiness_words(readiness: ConsoleSettingsReadiness) -> str:
    """Return the spec §5 readiness word for one Console readiness.

    Args:
        readiness: A built Console readiness (it carries the connection and
            observation time of current evidence, TASK-33005.2).

    Returns:
        'Ready · not tested', 'Ready · reachable HH:MM', 'Key verified HH:MM'
        (a listing accepted the key; TASK-34100.2), 'Ready · verified HH:MM'
        (a reply arrived) or 'Not ready · <reason>'.
    """
    if readiness.operability == "ready_to_send":
        return _ready_words(
            readiness.connection,
            readiness.endpoint,
            readiness.credential,
            readiness.generation,
            readiness.observed_at,
            readiness.generation_observed_at,
            readiness.model,
        )
    return _not_ready_words(
        readiness.blocker or "readiness_unknown",
        category=readiness.endpoint_category,
        issue=readiness.configuration_issue,
        subscription=readiness.subscription_status,
        connection=readiness.connection,
    )


def verdict_readiness_words(
    snapshot: ProviderReadinessSnapshot,
    evidence: object = None,
    *,
    subscription_status: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Return the same word for a setup verdict (Settings' test result).

    Args:
        snapshot: The draft's readiness facets.
        evidence: The ``ProviderTestEvidence`` behind ``snapshot`` (or a
            just-settled ``ProviderProbeResult``): its time, connection and
            key/generation facts.
        subscription_status: The Claude subscription state, if any.
        provider: The provider readiness key under test; with ``model``, a
            Ready verdict is judged for that model too (TASK-34100.2 AC#2:
            the same capacity blocker the Console and switcher read).
        model: The model under test.
        max_tokens: The reply maximum that model would be sent with.

    Returns:
        The readiness word.
    """
    word = _setup_verdict_words(snapshot, evidence, subscription_status)
    if provider and not word.startswith("Not ready"):
        return model_level_not_ready_words(provider, model, max_tokens) or word
    return word


def model_level_not_ready_words(
    provider: str, model: str | None, max_tokens: int | None
) -> str | None:
    """The model-level 'Not ready' word for one provider/model pair, or None.

    TASK-34100.2 AC#2: the reply reservation fills the model's window
    ('window too small'), or the model's last send got a 404 ('model not
    found'). One rule for the switcher rows and Settings' Readiness row.
    """
    model_id = normalize_console_model_value(model)
    if not model_id:
        return None
    if console_reply_capacity_exhausted(provider, model_id, max_tokens):
        return f"Not ready · {_NOT_READY_REASONS['capacity_exhausted']}"
    if CONSOLE_SEND_OUTCOMES.outcome_for(provider, model_id, None) == "not_found":
        return f"Not ready · {_NOT_READY_REASONS['model_unavailable']}"
    return None


def _setup_verdict_words(
    snapshot: ProviderReadinessSnapshot,
    evidence: object,
    subscription_status: str | None,
) -> str:
    """The provider-level word for a setup verdict (no model-level check)."""
    code = provider_readiness_verdict(snapshot).code
    # AC#8: rank blockers as the Console does (_BLOCKER_PRECEDENCE) --
    # configuration, then a missing model, then a failed test. The verdict
    # puts a stale, running or failed test first, which hid them.
    if snapshot.configuration == "incomplete":
        code = "incomplete"
    elif snapshot.model == "missing":
        code = "model_missing"
    reads = _VERDICT_READINESS[code]
    connection = getattr(evidence, "identity", None)
    if reads is None:
        return READY_NOT_TESTED
    if reads == "evidence":
        return _ready_words(
            connection,
            snapshot.endpoint,
            getattr(evidence, "credential", "not_required"),
            getattr(evidence, "generation", "not_tested"),
            getattr(evidence, "observed_at", None),
            getattr(evidence, "generation_observed_at", None),
            snapshot.model,
        )
    if reads == "provider_configuration_invalid":
        reads = _CONFIGURATION_ISSUE_BLOCKER.get(snapshot.configuration_issue, reads)
    return _not_ready_words(
        reads,
        category=snapshot.category,
        issue=snapshot.configuration_issue,
        subscription=subscription_status,
        connection=connection,
    )


def model_readiness_words(
    readiness: ConsoleSettingsReadiness, model: str | None
) -> str:
    """Return the readiness word for ``model`` on ``readiness``'s connection.

    TASK-33005.11: Switch model builds one readiness per provider and lists
    many of its models. A key check's listing verifies only the models it
    returned, and a paid test only the model it sent, so each row is worded
    from the evidence for its own model.

    Args:
        readiness: One provider's built readiness.
        model: The row's model id (``None`` reads ``readiness`` as built).

    Returns:
        The same word ``readiness_words`` gives a readiness built for
        ``model``.
    """
    evidence = readiness.evidence
    if readiness.operability != "ready_to_send" or evidence is None or not model:
        return readiness_words(readiness)
    tested = evidence.for_model(model)
    return _ready_words(
        readiness.connection,
        readiness.endpoint,
        readiness.credential,
        tested.generation,
        readiness.observed_at,
        tested.generation_observed_at,
        "confirmed"
        if readiness.endpoint == "reachable" and model in evidence.model_ids
        else "unconfirmed",
    )


_LOCAL_PROVIDER_KEYS = URL_BASED_PROVIDER_KEYS | KEYLESS_PROVIDER_KEYS


def _ready_words(
    connection: ProviderDraftIdentity | None,
    endpoint: str,
    credential: str,
    generation: str,
    endpoint_at: datetime | None,
    generation_at: datetime | None,
    model: str,
) -> str:
    """Qualify Ready by what was observed, and when (local HH:MM).

    'verified': a successful paid test (its time). 'Key verified' (TASK-34100.2
    AC#2): a cloud key its authenticated listing accepted (the listing's
    time) for a model that listing returned (``model`` facet 'confirmed',
    TASK-33005.11). Another model under that accepted key reads 'reachable':
    the server answered, but nothing says this model will. 'reachable': a local, URL or custom
    endpoint answered its listing -- even with a key, since a self-hosted
    server may ignore it. A public cloud listing (OpenRouter) proves
    nothing: 'not tested'.
    """
    if connection is None:
        return READY_NOT_TESTED
    if generation == "succeeded" and generation_at is not None:
        return f"Ready · verified {generation_at.astimezone():%H:%M}"
    if endpoint_at is None:
        return READY_NOT_TESTED
    local = (
        connection.custom_endpoint_id is not None
        or connection.provider_key in _LOCAL_PROVIDER_KEYS
    )
    when = endpoint_at.astimezone().strftime("%H:%M")
    accepted = credential == "listing_accepted" or (
        # Qodo #2958: a paid test (of another model) upgraded the listing's
        # acceptance; a public or failed listing accepted nothing.
        credential == "authenticated"
        and endpoint == "reachable"
        and connection.provider_key not in PUBLIC_MODEL_LISTING_PROVIDER_KEYS
    )
    if accepted and not local:
        # TASK-34100.2 AC#2: a listing proves the key, never a reply, so the
        # word names its scope -- on every surface (Settings, header,
        # switcher). TASK-33005.11: it vouches only for a model it listed;
        # another model under that accepted key reads 'reachable'.
        if model == "confirmed":
            return f"Key verified {when}"
        return f"Ready · reachable {when}"
    if endpoint == "reachable" and local:
        return f"Ready · reachable {when}"
    return READY_NOT_TESTED


def _not_ready_words(
    blocker: str,
    *,
    category: str | None,
    issue: str | None,
    subscription: str | None,
    connection: ProviderDraftIdentity | None,
) -> str:
    reason = _NOT_READY_REASONS.get(blocker, "check settings")
    if blocker in {"endpoint_unreachable", "credential_rejected"} and category:
        reason = _FAILURE_REASONS.get(category, reason)
        if category == "connection_refused" and connection is not None:
            try:  # Only the port: never the host, path or query.
                port = urlparse(connection.connection_identity[1]).port
            except ValueError:
                port = None
            reason += f" :{port}" if port else ""
    elif blocker == "credential_missing" and subscription in _SUBSCRIPTION_REASONS:
        reason = _SUBSCRIPTION_REASONS[subscription]
    elif blocker == "endpoint_invalid" and issue == "endpoint_missing":
        # The row to fill, as "no key" and "no model" do: "no URL" read as
        # something else beside Azure's "your resource host" (#3057).
        reason = "no endpoint"
    return f"Not ready · {reason}"


def _model_default_profile(
    provider_settings: Mapping[str, object],
    model: str | None,
) -> Mapping[str, object]:
    model_name = _string_value(model)
    if not model_name:
        return {}
    model_defaults = provider_settings.get("model_defaults", {})
    if not isinstance(model_defaults, Mapping):
        return {}
    profile = model_defaults.get(model_name, {})
    return profile if isinstance(profile, Mapping) else {}


def _has_provider_settings_key(
    app_config: Mapping[str, object], provider_key: str
) -> bool:
    api_settings = _mapping_value(app_config, "api_settings")
    return any(
        provider_config_key(configured_provider) == provider_key
        for configured_provider in api_settings
    )


def _default_base_url(
    provider_key: str, provider_settings: Mapping[str, object]
) -> str | None:
    base_url = _first_string(
        provider_settings.get("api_base_url"),
        provider_settings.get("api_base"),
        provider_settings.get("base_url"),
        provider_settings.get("api_url"),
    )
    if provider_key in {"llama_cpp", "local_llamacpp"}:
        return normalize_llamacpp_base_url(base_url or DEFAULT_LLAMACPP_BASE_URL)
    return base_url


def _is_url_based_provider(
    provider_key: str, provider_settings: Mapping[str, object]
) -> bool:
    return provider_uses_endpoint(provider_key, provider_settings)


def _endpoint_differs_for_provider(
    provider_key: str,
    base_url: str | None,
    provider_settings: Mapping[str, object],
) -> bool:
    """Return whether a selected endpoint differs from persisted provider settings."""
    if provider_key in {"llama_cpp", "local_llamacpp"}:
        configured_endpoint = first_configured_endpoint(provider_settings)
        if not configured_endpoint:
            selected = normalize_generic_endpoint_for_compare(
                normalize_llamacpp_base_url(base_url)
            )
            default = normalize_generic_endpoint_for_compare(DEFAULT_LLAMACPP_BASE_URL)
            return bool(selected) and selected != default
        selected = normalize_generic_endpoint_for_compare(
            normalize_llamacpp_base_url(base_url)
        )
        configured = normalize_generic_endpoint_for_compare(
            normalize_llamacpp_base_url(configured_endpoint)
        )
        return selected != configured
    return generic_endpoint_differs(base_url, provider_settings)


def _console_endpoint_restart_fallback(
    provider_key: str,
    provider_settings: Mapping[str, object],
    app_config: Mapping[str, object],
    environ: Mapping[str, str] | None,
) -> str | None:
    """Return the endpoint the next boot would derive for this provider.

    Mirrors the base-URL chain in the Console selection core
    (``console_chat_controller.resolve_console_selection_core``): llama.cpp
    resolves env override -> ``[console] llama_cpp_base_url_override`` -> the
    provider's configured endpoint -> the built-in default; other URL-based
    providers resolve only their configured endpoint.

    Args:
        provider_key: Normalized provider readiness key.
        provider_settings: The provider's persisted ``api_settings`` section.
        app_config: Application configuration mapping.
        environ: Environment override source; ``None`` reads ``os.environ``.

    Returns:
        The restart fallback endpoint, or ``None`` for URL-based providers
        with nothing configured.
    """
    if provider_key in {"llama_cpp", "local_llamacpp"}:
        env = environ if environ is not None else os.environ
        console_config = _mapping_value(app_config, "console")
        fallback = (
            env.get("TLDW_CONSOLE_LLAMA_CPP_BASE_URL")
            or _string_value(console_config.get("llama_cpp_base_url_override"))
            or first_configured_endpoint(provider_settings)
            or DEFAULT_LLAMACPP_BASE_URL
        )
        return normalize_llamacpp_base_url(fallback)
    return first_configured_endpoint(provider_settings)


def console_session_endpoint_survives_restart(
    settings: ConsoleSessionSettings,
    *,
    app_config: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the session endpoint is backed for the next boot.

    ``True`` when the provider uses no endpoint, the session carries no
    endpoint, the session rides a resolvable registry entry (ADR-343: the
    entry is the persisted endpoint carrier), or the session endpoint equals
    the restart fallback chain's value (so re-deriving defaults next boot
    reproduces it). ``False`` means the endpoint lives only in this session
    and is silently lost on restart -- the task-16473 persistence trap.

    Args:
        settings: Console session settings carrying the endpoint to check.
        app_config: Application configuration mapping.
        environ: Environment override source; ``None`` reads ``os.environ``.

    Returns:
        Whether re-deriving defaults on the next boot would reproduce the
        session's endpoint.
    """
    # Lazy import: custom_endpoint_registry imports this module for URL
    # normalization, so a module-level import would cycle.
    from tldw_chatbook.Chat.custom_endpoint_registry import entry_for

    if entry_for(app_config, settings.provider) is not None:
        return True
    provider_key = provider_config_key(settings.provider)
    provider_settings = console_provider_settings(app_config, provider_key)
    base_url = _string_value(settings.base_url)
    if not base_url or not _is_url_based_provider(provider_key, provider_settings):
        return True
    fallback = _console_endpoint_restart_fallback(
        provider_key, provider_settings, app_config, environ
    )
    if provider_key in {"llama_cpp", "local_llamacpp"}:
        selected = normalize_generic_endpoint_for_compare(
            normalize_llamacpp_base_url(base_url)
        )
        resolved = normalize_generic_endpoint_for_compare(
            normalize_llamacpp_base_url(fallback or DEFAULT_LLAMACPP_BASE_URL)
        )
        return selected == resolved
    if not fallback:
        return False
    return normalize_generic_endpoint_for_compare(
        base_url
    ) == normalize_generic_endpoint_for_compare(fallback)


def unsaved_console_endpoint_warning(
    settings: ConsoleSessionSettings,
    *,
    app_config: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Build the session-only endpoint warning copy, or ``None`` when backed.

    task-16473: llama.cpp readiness reports "Ready" for session-scoped
    endpoints (the direct llama path skips the endpoint-saved check), so
    nothing else tells the user their endpoint evaporates on restart. This is
    the warning that apply surfaces.

    Args:
        settings: Console session settings carrying the endpoint to describe.
        app_config: Application configuration mapping.
        environ: Environment override source; ``None`` reads ``os.environ``.

    Returns:
        User-facing warning copy, or ``None`` when the endpoint is backed for
        the next boot.
    """
    if console_session_endpoint_survives_restart(
        settings, app_config=app_config, environ=environ
    ):
        return None
    display = safe_endpoint_display(settings.base_url) or "the current endpoint"
    return (
        f"Endpoint {display} is saved for this session only and will not "
        "survive a restart. Use Save as default (or Settings) to keep it."
    )


def _valid_base_url(provider_key: str, base_url: str) -> bool:
    # TASK-39042: the endpoint contract's scheme rule, the one the send reads
    # (``effective_provider_endpoint``). Only a loopback spelling may omit
    # the scheme; llama.cpp's normalizer had added http:// to a remote host.
    base_url = read_schemeless_loopback(base_url)
    if "://" not in base_url:
        return False
    try:
        candidate = (
            normalize_llamacpp_base_url(base_url)
            if provider_key in NATIVE_CONSOLE_PROVIDER_KEYS
            else base_url
        )
    except ValueError:
        return False
    return validate_url(candidate) and _has_valid_url_port(candidate)


def _has_valid_url_port(url: str) -> bool:
    try:
        parsed = urlparse(url)
        parsed.port
    except ValueError:
        return False
    return parsed.port is None or 0 < parsed.port <= 65535


def _float_in_range(value: object, minimum: float, maximum: float) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return minimum <= number <= maximum


def _optional_int_at_least(value: object, minimum: int) -> bool:
    parsed = _parse_optional_int(value)
    return parsed is not None and parsed >= minimum


def _float_setting(
    primary: Mapping[str, object],
    fallback: Mapping[str, object],
    key: str,
    default: float,
) -> float:
    value = primary.get(key) if key in primary else fallback.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _setting_value_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
    default: object = None,
) -> object:
    for source in sources:
        if key in source:
            value = source.get(key)
            if not _is_blank_value(value):
                return value
    return default


def _optional_float_setting(
    primary: Mapping[str, object],
    fallback: Mapping[str, object],
    key: str,
) -> float | None:
    if key in primary:
        value = primary.get(key)
    else:
        value = fallback.get(key)
    if _is_blank_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int_setting(
    primary: Mapping[str, object],
    fallback: Mapping[str, object],
    key: str,
) -> int | None:
    if key in primary:
        value = primary.get(key)
    else:
        value = fallback.get(key)
    return _parse_optional_int(value)


def _estimate_tokens_locally(
    messages: Sequence[Mapping[str, str]],
    model: str,
    provider: str,
) -> int:
    """Estimate a message list's token count with the real counter.

    Delegates to :func:`count_tokens_messages` (custom tokenizer -> tiktoken
    -> conservative chars floor, never a whitespace word count -- see
    ``Utils/token_counter.py``), which both `model` and `provider` actually
    drive: `model` selects the tokenizer/encoding and the chat-format
    per-message framing convention below; `provider` selects the chars-floor
    ratio table when no tokenizer is installed.

    The previous placeholder's fake `len(messages) * 10` *chars* overhead is
    retired outright, not replaced: `count_tokens_messages` already supplies
    a real per-message allowance (OpenAI's documented chat-format framing --
    3 tokens/message + 3 base for `gpt-3.5`/`gpt-4` models, 2/2 for others),
    the same convention this codebase already uses for dispatch-time history
    budgeting (`console_history_budget.py`) and agent runs
    (`Agents/agent_service.py`) -- so no bespoke overhead is invented here.
    """
    return count_tokens_messages(list(messages), model, provider)


def _resolve_token_limit_locally(model: str, provider: str) -> int:
    """Resolve a local model-window estimate without exposing provenance."""
    limit, _verified, _source = _resolve_token_limit_locally_with_provenance(
        model,
        provider,
    )
    return limit


def _resolve_token_limit_locally_with_provenance(
    model: str,
    provider: str,
) -> tuple[int, bool, str]:
    """Resolve the shared model/API/system default without network I/O."""
    from tldw_chatbook.Utils.token_counter import resolve_context_window

    result = resolve_context_window(provider, model)
    return result.tokens, result.verified, result.source


def _bool_setting(
    primary: Mapping[str, object],
    fallback: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    value = primary.get(key) if key in primary else fallback.get(key, default)
    return value if isinstance(value, bool) else default


def _first_string(*values: object) -> str | None:
    for value in values:
        text = _string_value(value)
        if text:
            return text
    return None


def _string_setting(source: Mapping[str, object], key: str) -> str:
    return _string_value(source.get(key)) or ""


def normalize_console_model_value(value: object) -> str | None:
    """Return a model value unless it is blank or a placeholder sentinel."""
    text = _string_value(value)
    if text is None or text.lower() in MODEL_OPTION_PLACEHOLDER_VALUES:
        return None
    return text


def _format_summary_float(value: float | None) -> str:
    """Format a sampler for the summary; blank means the provider decides.

    Args:
        value: The sampler, None when the provider does not accept it.

    Returns:
        Two decimals, or the field rows' Source word for a blank row.
    """
    if value is None:
        return CONSOLE_VALUE_SOURCE_WORDS[ConsoleValueLayer.PROVIDER_SCALARS]
    return f"{float(value):.2f}"


def _format_context_summary_row(label: str) -> str:
    label_text = _string_value(label) or "unavailable"
    if label_text.lower() in {"unknown", "context: unknown"}:
        label_text = "Context: unavailable"
    return (
        label_text if label_text.startswith("Context: ") else f"Context: {label_text}"
    )


def _format_endpoint_summary_row(
    settings: ConsoleSessionSettings,
    readiness: ConsoleSettingsReadiness,
) -> str:
    if readiness.blocker == "endpoint_invalid" and readiness.invalid_endpoint:
        # TASK-39151: the session's or the config's value is invalid; say
        # what to change when that is safe. The value itself is never shown.
        return (
            "Endpoint: Invalid (add http:// or https://)"
            if readiness.invalid_endpoint == "needs_scheme"
            else "Endpoint: Invalid"
        )
    if (
        readiness.blocker
        in {
            "provider_missing",
            "provider_unsupported",
            "provider_configuration_invalid",
            "endpoint_invalid",
        }
        or readiness.configuration_issue == "endpoint_missing"
    ):
        return "Endpoint: Not configured"
    endpoint = safe_endpoint_display(settings.base_url)
    return f"Endpoint: {endpoint or 'Provider default'}"


def _format_credential_summary_row(readiness: ConsoleSettingsReadiness) -> str:
    if readiness.subscription_status is not None:
        state = {
            "pending": "checking",
            "ready": "not verified",
            "expired": "expired",
            "missing": "missing",
        }[readiness.subscription_status]
        return f"Credential: Claude subscription ({state})"
    if readiness.credential == "missing":
        return "Credential: missing"
    if readiness.credential == "not_required":
        return "Credential: not required"
    if readiness.credential == "authenticated":
        return "Credential: authenticated"
    if readiness.credential == "listing_accepted":
        return "Credential: accepted by model listing"
    source = {
        "stored": "local config",
        "environment": "environment variable",
        "draft": "unsaved draft",
    }.get(readiness.credential_source, "present")
    return f"Credential: {source} (not verified)"


@dataclass(frozen=True)
class ConsoleAssistantStartup:
    """Resolved settings and identity for one new conversation."""

    settings: ConsoleSessionSettings
    assistant_kind: str = "generic"
    assistant_id: str = "console"
    persona_memory_mode: str | None = None
    notice: str = ""


# ---------------------------------------------------------------------------
# TASK-34100.2 AC#2 (new-cross-cutting-01): "first chat works" readiness.
# The verdict covered credentials and connection only, so the header, Settings
# and the switcher said "Ready" above a model every send was refused on. Two
# facts close that gap: the reply capacity Console's send preflight computes
# (from the same resolvers), and how the current model's last real send ended.
# ---------------------------------------------------------------------------

#: Last send outcomes that take a model out of Ready until its inputs change.
#: ``key_check_rejected`` is a key check's 401 (setup, Settings 't'), recorded
#: by the shared catalog service (review round 2, F4).
ConsoleSendOutcome = Literal["unauthorized", "not_found", "key_check_rejected"]
#: How long a server-reported context window stands for readiness: the
#: gateway's ``console_context_window.METADATA_SUCCESS_TTL`` (not imported --
#: that module loads the HTTP client and the egress guard; a test pins the two
#: equal).
OBSERVED_WINDOW_TTL_SECONDS = 60
#: Only a 401 rejects a key (owner ruling 2026-10-02, PR #2958): a 403 is
#: often about one message or model (OpenRouter's moderation, a model
#: permission, a region limit), so it records nothing (review round 1, F1).
_SEND_OUTCOME_BY_STATUS: Mapping[int, ConsoleSendOutcome] = MappingProxyType(
    {401: "unauthorized", 404: "not_found"}
)


def _send_outcome_provider_key(provider: object) -> str:
    """The readiness key a provider spelling resolves to (one key per surface)."""
    return _readiness_key_for_spelling(str(provider or "").strip())


@functools.lru_cache(maxsize=512)
def _readiness_key_for_spelling(raw: str) -> str:
    """Memoized: identity resolution walks every handler key (TASK-18909)."""
    identity = resolve_console_provider_identity(
        _canonical_chat_provider_id(raw),
        handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    )
    return identity.readiness_key or provider_config_key(raw)


class ConsoleSendOutcomes:
    """Process-wide record of how real sends ended, per provider.

    A 401 rejects the KEY (kept per provider and credential revision, so a
    new key starts clean); a 404 loses the MODEL (kept per provider and
    model). Either stands only while that provider's own settings are the
    ones it was recorded under (``_provider_settings_scope``): saving its
    key, endpoint or any of its settings clears it, so a composer blocked on
    it is never a dead end, while an unrelated save -- the first reply's
    onboarding flag, the Library rail's state, another provider -- keeps it
    (pass-2 review round 1, F8: keyed to the GLOBAL generation, one click on
    Library put a 404'd model back to 'Ready · not tested'). A later
    successful reply clears it too. It also keeps the context window a
    serving endpoint reported for a model, so the offline capacity check sees
    the window the send preflight used -- but only under the provider
    settings it was seen with, and only as long as the gateway's own
    metadata cache would keep it (``OBSERVED_WINDOW_TTL_SECONDS``). A capacity
    blocker refuses at the composer, so no send can refresh a stale window;
    without the expiry a server restarted with a larger context stayed 'window
    too small' until the app restarted (review round 2, R2-3).
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        from threading import Lock
        from time import monotonic

        self._lock = Lock()
        self._clock = clock or monotonic
        #: Each value is the provider settings scope it was recorded under.
        self._unauthorized: dict[tuple[str, int], str] = {}
        self._not_found: dict[tuple[str, str], str] = {}
        #: Keys a key check rejected. Not tied to the config generation:
        #: setup's own commit publishes one right after its check. A new key
        #: is a new revision, so it starts clean.
        self._rejected_keys: set[tuple[str, int]] = set()
        #: (tokens, verified, provider settings scope, expiry on ``_clock``,
        #: the endpoint that reported it -- "" when unknown).
        self._windows: dict[
            tuple[str, str], tuple[int, bool, str, float, str]
        ] = {}

    @staticmethod
    def _generation() -> int:
        """The runtime config publication counter, read without importing.

        Readiness must stay config-import free (it runs before the config
        module is loaded in some callers); an unloaded config has published
        nothing, so the counter reads 0.
        """
        import sys

        config_module = sys.modules.get("tldw_chatbook.config")
        reader = getattr(config_module, "get_runtime_config_generation", None)
        return reader() if callable(reader) else 0

    @classmethod
    def _scope(cls, provider_key: str) -> str:
        """What a recorded outcome for ``provider_key`` depends on.

        A fingerprint of that provider's own published settings table, read
        without importing config; memoized per config generation, so a
        readiness build costs one dict lookup.
        """
        return _provider_settings_scope(provider_key, cls._generation())

    def record_status(
        self, provider: object, model: object, api_key: object, status_code: object
    ) -> None:
        """Record a failed send's HTTP status (only 401 and 404 are kept)."""
        outcome = (
            _SEND_OUTCOME_BY_STATUS.get(status_code) if type(status_code) is int else None
        )
        model_id = str(model or "").strip()
        if outcome is None or not model_id:
            return
        key = _send_outcome_provider_key(provider)
        scope = self._scope(key)
        with self._lock:
            if outcome == "unauthorized":
                self._unauthorized[(key, connection_credential_revision(api_key))] = (
                    scope
                )
            else:
                self._not_found[(key, model_id)] = scope

    def record_success(self, provider: object, model: object, api_key: object) -> None:
        """A reply arrived: this key and this model are not known to fail."""
        key = _send_outcome_provider_key(provider)
        revision = connection_credential_revision(api_key)
        with self._lock:
            self._unauthorized.pop((key, revision), None)
            self._rejected_keys.discard((key, revision))
            self._not_found.pop((key, str(model or "").strip()), None)

    def record_key_check(
        self, provider: object, api_key: object, *, rejected: bool
    ) -> None:
        """Record a key check's verdict on this key (review round 2, F4).

        Args:
            provider: The provider the key was checked for.
            api_key: The key that was checked (never stored; only its
                credential revision is).
            rejected: True for a 401 (the key is rejected), False for a
                check that proved the key (it clears any rejection).
        """
        if not api_key:
            return
        entry = (
            _send_outcome_provider_key(provider),
            connection_credential_revision(api_key),
        )
        with self._lock:
            if rejected:
                self._rejected_keys.add(entry)
            else:
                self._rejected_keys.discard(entry)
                self._unauthorized.pop(entry, None)

    def outcome_for(
        self, provider: object, model: object, api_key: object
    ) -> ConsoleSendOutcome | None:
        """The outcome that still stands for this provider, model and key."""
        key = _send_outcome_provider_key(provider)
        scope = self._scope(key)
        with self._lock:
            if (
                self._unauthorized.get((key, connection_credential_revision(api_key)))
                == scope
            ):
                return "unauthorized"
            if (key, connection_credential_revision(api_key)) in self._rejected_keys:
                return "key_check_rejected"
            if self._not_found.get((key, str(model or "").strip())) == scope:
                return "not_found"
        return None

    def record_window(
        self,
        provider: object,
        model: object,
        tokens: object,
        verified: object,
        *,
        endpoint: object = None,
    ) -> None:
        """Keep the window a serving endpoint reported (server metadata).

        Args:
            provider: The provider the send resolved to.
            model: The model the window belongs to.
            tokens: The reported context window.
            verified: Whether the window is the server's own figure.
            endpoint: The endpoint that reported it (pass 2): another server
                serving the same model id never inherits it.
        """
        if type(tokens) is not int or tokens <= 0 or not str(model or "").strip():
            return
        key = (_send_outcome_provider_key(provider), str(model).strip())
        record = (
            tokens,
            bool(verified),
            self._scope(key[0]),
            self._clock() + OBSERVED_WINDOW_TTL_SECONDS,
            _window_endpoint(provider, endpoint),
        )
        with self._lock:
            self._windows[key] = record

    def window_for(
        self, provider: object, model: object, endpoint: object = None
    ) -> tuple[int, bool] | None:
        """The reported window that still stands for this model, if any.

        Args:
            provider: The provider being judged.
            model: The model being judged.
            endpoint: The endpoint a send would use. A window reported by a
                different endpoint does not stand for it; a reader that does
                not know its endpoint (``None``) accepts the record.

        Returns:
            ``(tokens, verified)``, or ``None``.
        """
        key = (_send_outcome_provider_key(provider), str(model or "").strip())
        scope = self._scope(key[0])
        now = self._clock()
        wanted = _window_endpoint(provider, endpoint)
        with self._lock:
            record = self._windows.get(key)
            if record is None:
                return None
            tokens, verified, seen_scope, expires_at, reported_by = record
            if seen_scope != scope or now >= expires_at:
                del self._windows[key]
                return None
            if wanted and reported_by and wanted != reported_by:
                return None  # another server's window; kept for its own chats
            return tokens, verified

    def clear(self) -> None:
        """Forget everything (tests)."""
        _catalog_capacity_inputs_for.cache_clear()
        _provider_settings_scope.cache_clear()
        with self._lock:
            self._unauthorized.clear()
            self._not_found.clear()
            self._rejected_keys.clear()
            self._windows.clear()


@functools.lru_cache(maxsize=256)
def _provider_settings_scope(provider_key: str, generation: int) -> str:
    """Fingerprint one provider's published settings table (memoized).

    Read from the config module's published ``settings`` when it is loaded
    (never imported or reloaded here); the provider's own table only, through
    the Console's one lookup, so a custom endpoint is its registry entry.
    When the table cannot be read, or the lookup finds none (a provider with
    no settings table, such as 'mistral'), the fingerprint falls back to the
    generation, i.e. any save retires the outcome -- the pre-round-1 rule.
    A constant fingerprint would let only a reply or a restart clear it.

    Args:
        provider_key: The readiness key of the provider.
        generation: The config publication counter (the memo key only).

    Returns:
        A short hex digest; equal digests mean the settings are unchanged.
    """
    import hashlib
    import sys

    config_module = sys.modules.get("tldw_chatbook.config")
    published = getattr(config_module, "settings", None)
    if not isinstance(published, Mapping):
        return f"generation:{generation}"
    try:
        table = console_provider_settings(published, provider_key)
        if not table:
            return f"generation:{generation}"
        encoded = json.dumps(table, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 - a fingerprint must never break readiness
        return f"generation:{generation}"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def _window_endpoint(provider: object, endpoint: object) -> str:
    """One spelling per serving endpoint, so a window matches its own server.

    The provider's canonical connection endpoint when the contract can read
    it (the identity Chat settings keys a connection by), else the stripped
    URL without a trailing slash; ``""`` when unknown.
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        return ""
    url = endpoint.strip()
    connection = canonical_connection_identity(_send_outcome_provider_key(provider), url)
    if connection is not None:
        return connection[1]
    return url.rstrip("/").lower()


#: The one process-wide ledger the gateway writes and readiness reads.
CONSOLE_SEND_OUTCOMES = ConsoleSendOutcomes()


def _catalog_generation() -> tuple[int, int]:
    """What the catalog memo depends on, read without importing anything.

    Settings saves a model's context window and reloads the model catalog
    (``reload_capabilities``), and every config save publishes a new runtime
    generation; either one must reach readiness without a restart (review
    round 1, F3).
    """
    import sys

    capabilities = sys.modules.get("tldw_chatbook.model_capabilities")
    reloads = getattr(capabilities, "capabilities_reload_count", None)
    return (
        reloads() if callable(reloads) else 0,
        ConsoleSendOutcomes._generation(),
    )


def _catalog_capacity_inputs(provider: str, model: str) -> tuple[int, bool, int | None]:
    """The catalog window and output cap for one model, memoized per catalog."""
    return _catalog_capacity_inputs_for(provider, model, _catalog_generation())


@functools.lru_cache(maxsize=1024)
def _catalog_capacity_inputs_for(
    provider: str, model: str, _generation: tuple[int, int]
) -> tuple[int, bool, int | None]:
    """The catalog window and output cap for one model (memoized).

    Readiness is built hundreds of times per Console screen switch
    (TASK-18909); these catalog lookups are the only non-trivial part of the
    capacity check. ``_generation`` keys the memo to the catalog it read.
    """
    from tldw_chatbook.Utils.token_counter import resolve_context_window

    window = resolve_context_window(provider, model)
    output_cap = None
    try:
        from tldw_chatbook.model_capabilities import get_model_capabilities

        caps = get_model_capabilities().get_model_capabilities(provider, model)
        for name in ("max_output_tokens", "output_token_limit", "provider_output_cap"):
            value = caps.get(name)
            if type(value) is int and value > 0:
                output_cap = value
                break
    except Exception:  # noqa: BLE001 -- optional catalog never blocks readiness
        output_cap = None
    return window.tokens, window.verified, output_cap


def console_reply_capacity(
    provider: str, model: str, max_tokens: int | None, endpoint: str | None = None
) -> "ConsoleRequestCapacity":
    """Console's request capacity for one model, offline.

    The same inputs the send preflight uses -- the serving endpoint's
    reported window when one was seen, else ``resolve_context_window`` --
    through the same ``resolve_request_capacity`` (which caps the reply
    reservation for an estimated window, AC#1).

    Args:
        provider: Provider readiness key.
        model: The selected model id.
        max_tokens: The configured response maximum, or ``None`` for the
            Console default reservation.
        endpoint: The endpoint a send would use; a server-reported window
            counts only for the endpoint that reported it.

    Returns:
        The resolved capacity.
    """
    from tldw_chatbook.Chat.console_request_capacity import resolve_request_capacity

    tokens, verified, output_cap = _catalog_capacity_inputs(provider, model)
    observed = CONSOLE_SEND_OUTCOMES.window_for(provider, model, endpoint)
    if observed is not None:
        tokens, verified = observed
    requested = max_tokens if type(max_tokens) is int and max_tokens > 0 else None
    return resolve_request_capacity(
        context_window_tokens=tokens,
        context_window_verified=verified,
        provider_output_cap_tokens=output_cap,
        requested_response_tokens=requested,
    )


def _reply_max_tokens(
    settings: ConsoleSessionSettings,
    app_config: Mapping[str, object],
    provider_key: str,
) -> int | None:
    """The response maximum a send of these settings would reserve.

    A blank Max tokens means 'inherit': the applied chat takes the value the
    defaults resolve for this provider and model (a model profile, the
    provider table, ``[chat_defaults]``), so readiness judges that value,
    not the Console default reservation (review round 1, F6: a blank read
    'Ready' in Chat settings while the applied chat inherited 200,000 and
    read 'Not ready · window too small').
    """
    if settings.max_tokens is not None:
        return settings.max_tokens
    return default_reply_max_tokens(app_config, provider_key, settings.model)[0]


def default_reply_max_tokens(
    app_config: Mapping[str, object], provider_key: str, model: str | None
) -> tuple[int | None, bool]:
    """The response maximum a new chat of this pair reserves, and its source.

    The one default chain the Console builds a new chat with (a model
    profile, the Console's saved provider defaults, ``[chat_defaults]``, the
    provider table), so Settings' Test Provider judges the same value as the
    header and the switcher (TASK-34100.2 review round 1, F1: it read only the
    provider's ``max_tokens`` and said 'Key verified' for a model whose saved
    200,000-token model default the Console called 'window too small').

    Args:
        app_config: The configuration snapshot (Settings passes its draft).
        provider_key: The provider's canonical key.
        model: The model a new chat would use.

    Returns:
        ``(max_tokens, from_model_default)``: the value (None when the lookup
        fails or nothing sets one) and whether this model's own saved default
        (``model_defaults.<model>.max_tokens``) is what sets it.
    """
    try:
        value = build_target_default_console_session_settings(
            app_config, provider_key, model
        ).max_tokens
        layer = resolve_console_value_layers(
            app_config, provider_key, model, ("max_tokens",)
        )["max_tokens"]
    except Exception:  # noqa: BLE001 - readiness never fails on a lookup
        return None, False
    return value, layer is ConsoleValueLayer.MODEL_DEFAULT


def console_reply_capacity_exhausted(
    provider: str, model: str, max_tokens: int | None, endpoint: str | None = None
) -> bool:
    """Whether the reply reservation leaves this model no input at all.

    Exactly the send preflight's "Response reservation and safety margin
    leave no model input capacity" refusal (TASK-34100.2 AC#2).
    """
    try:
        capacity = console_reply_capacity(provider, model, max_tokens, endpoint)
    except (TypeError, ValueError):
        return False
    return (
        capacity.context_window_tokens is not None
        and capacity.effective_input_ceiling_tokens == 0
    )


def _model_level_blocker_detail(
    blocker: str,
    display_name: str,
    model: str,
    max_tokens: int | None,
    provider_key: str,
    *,
    key_checked: bool = False,
    endpoint: str | None = None,
) -> str:
    """Plain words for a model-level blocker, naming the fix."""
    if blocker == "capacity_exhausted":
        capacity = console_reply_capacity(provider_key, model, max_tokens, endpoint)
        return (
            f"Replies reserve {capacity.effective_response_tokens:,} of {model}'s "
            f"{capacity.context_window_tokens or 0:,}-token context window, so no "
            "room is left for your message. Lower the response maximum or "
            "choose another model."
        )
    if blocker == "key_rejected_on_send" and key_checked:
        return (
            f"{display_name} rejected this API key when it was checked. "
            "Update the key, then send again."
        )
    if blocker == "key_rejected_on_send":
        return (
            f"{display_name} rejected the API key on the last message to {model}. "
            "Update the key, then send again."
        )
    return (
        f"{display_name} could not find {model} on the last message (HTTP 404). "
        "Choose another model, or check the endpoint."
    )


def default_save_refusal(
    readiness: ConsoleSettingsReadiness,
    default_readiness: ConsoleSettingsReadiness,
) -> str:
    """Why Chat settings' default actions are unavailable, or ``""``.

    TASK-34100.2 pass 3 review round 1 (F3): 'Save as model default' and
    'Make default for new chats' also apply the pair to this chat, so a draft
    whose replies fill its model's window is refused as Apply and the
    switcher refuse it, in the capacity verdict's own words; otherwise a
    pair that cannot send natively is refused as before.

    Args:
        readiness: The open draft's verdict (its own Max tokens).
        default_readiness: The future-chat verdict for the draft's pair.

    Returns:
        The 'Unavailable: ...' copy, or an empty string when saving is fine.
    """
    if readiness.blocker == "capacity_exhausted":
        return f"Unavailable: {readiness.detail}"
    if not default_readiness.native_send_supported:
        return f"Unavailable: {default_readiness.detail}"
    return ""


def readiness_words_for_model(
    readiness: ConsoleSettingsReadiness,
    provider: str,
    model: str | None,
    *,
    max_tokens: int | None = None,
) -> str:
    """One switcher row's word: the provider's verdict, judged for ``model``.

    A provider blocked only at the model level (``MODEL_LEVEL_BLOCKERS``) is
    set up and keyed, so each of its models gets its own verdict: a row whose
    model cannot take a turn says why, every other row reads the provider's
    Ready word.

    Args:
        readiness: The provider's readiness (built for one representative
            model).
        provider: The row's provider readiness key.
        model: This row's model id.
        max_tokens: The response maximum this row would send with.

    Returns:
        The spec §5 readiness word for this provider/model pair.
    """
    if readiness.operability != "ready_to_send" and (
        readiness.blocker not in MODEL_LEVEL_BLOCKERS
    ):
        return readiness_words(readiness)
    blocked = model_level_not_ready_words(provider, model, max_tokens)
    if blocked is not None:
        return blocked
    return _ready_words(
        readiness.connection,
        readiness.endpoint,
        readiness.credential,
        readiness.generation,
        readiness.observed_at,
        readiness.generation_observed_at,
    )
