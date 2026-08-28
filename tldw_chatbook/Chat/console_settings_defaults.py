"""Atomic exact-model Console default persistence and recovery helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
from ipaddress import ip_address
import threading
from types import MappingProxyType
from urllib.parse import urlsplit

from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_provider_support import (
    build_local_thinking_payload_fields,
    resolve_console_provider_identity,
)
from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    build_console_settings_readiness,
    build_target_default_console_session_settings,
    validate_console_session_settings,
)
from tldw_chatbook.Chat.console_settings_apply import (
    FULL_MODEL_DEFAULT_FIELDS,
    QUICK_MODEL_DEFAULT_FIELDS,
    ConsoleSettingsAction,
)
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.model_capabilities import (
    anthropic_model_rejects_fixed_thinking_budget,
    moonshot_model_supports_reasoning_effort,
    zai_model_supports_reasoning_effort,
)


_PROVIDER_SPECIFIC_FIELDS = frozenset(
    {
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
        "thinking_budget_tokens",
    }
)
_DIRECT_PROVIDER_FIELDS = {
    "openai": frozenset({"reasoning_effort", "reasoning_summary", "verbosity"}),
    "qwencloud": frozenset({"reasoning_effort"}),
}
_LOCAL_THINKING_BUDGET_KEYS = frozenset(
    {"llama_cpp", "local_llamacpp", "local_llamafile", "local-llm"}
)
_ENDPOINT_KEYS = (
    "api_base_url",
    "api_base",
    "base_url",
    "api_url",
    "api_endpoint",
    "endpoint",
    "router_base_url",
    "huggingface_router_base_url",
)
_INTENT_GENERATION_LOCK = threading.RLock()
_LATEST_INTENT_GENERATION: int | None = None
_LATEST_INTENT_FINGERPRINT: str | None = None


class ConsoleDefaultSavePhase(str, Enum):
    """Durability phase in which a Console default save failed."""

    BEFORE_REPLACE = "before_replace"
    CACHE_PUBLICATION = "cache_publication"


@dataclass(frozen=True, slots=True)
class ConsoleEndpointPatch:
    """One explicitly opted-in endpoint edit bound to its provider owner."""

    value: str
    bound_provider_config_key: str
    dirty: bool
    checked: bool


@dataclass(frozen=True, slots=True)
class ConsoleDefaultMutationIntent:
    """Immutable exact-field mutation requested after a successful live Apply."""

    generation: int
    action: ConsoleSettingsAction
    provider_config_key: str
    literal_model_id: str
    field_mask: frozenset[str]
    values: Mapping[str, object | None]
    endpoint_patch: ConsoleEndpointPatch | None

    def __post_init__(self) -> None:
        """Detach retry state from caller-owned mutable containers."""

        object.__setattr__(self, "field_mask", frozenset(self.field_mask))
        object.__setattr__(
            self,
            "values",
            MappingProxyType(dict(self.values)),
        )


@dataclass(frozen=True, slots=True)
class ConsoleDefaultMutationOutcome:
    """Disk/runtime result for one immutable default intent generation."""

    intent_generation: int
    file_replaced: bool
    runtime_published: bool
    settings_view: Mapping[str, object] | None
    failure_phase: ConsoleDefaultSavePhase | None


@dataclass(frozen=True, slots=True)
class RuntimeConfigPublicationResult:
    """Cache-only publication outcome after a prior successful disk write."""

    published: bool
    settings_view: Mapping[str, object] | None
    failure_phase: str | None


class ConsoleDefaultRecoveryAction(str, Enum):
    """Explicit recovery actions shown for app-global default durability."""

    RETRY_SAVE = "retry_save"
    DISCARD_RETRY = "discard_retry"
    REFRESH_RUNNING_APP = "refresh_running_app"
    DISMISS_REFRESH = "dismiss_refresh"


@dataclass(frozen=True, slots=True)
class ConsoleDefaultRecoveryRequest:
    """Recovery command bound to one exact default-intent generation."""

    action: ConsoleDefaultRecoveryAction
    intent_generation: int


@dataclass(frozen=True, slots=True)
class ConsoleEndpointPreview:
    """Credential-free endpoint authority and conservative network class."""

    authority: str
    network_classification: str


def _supported_profile_fields(provider: str, model: str) -> frozenset[str]:
    """Mirror the controller's provider/model capability projection."""

    canonical = provider_config_key(provider)
    supported = set(FULL_MODEL_DEFAULT_FIELDS - _PROVIDER_SPECIFIC_FIELDS)
    if canonical == "moonshot" and moonshot_model_supports_reasoning_effort(model):
        supported.add("reasoning_effort")
    if canonical == "zai" and zai_model_supports_reasoning_effort(model):
        supported.add("reasoning_effort")
    supported.update(_DIRECT_PROVIDER_FIELDS.get(canonical, ()))
    if canonical == "anthropic":
        supported.add("thinking_effort")
        if not anthropic_model_rejects_fixed_thinking_budget(model):
            supported.add("thinking_budget_tokens")

    identity = resolve_console_provider_identity(
        canonical,
        handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    )
    if build_local_thinking_payload_fields(identity.execution_key, "low", None):
        supported.add("reasoning_effort")
    if identity.execution_key in _LOCAL_THINKING_BUDGET_KEYS:
        supported.add("thinking_budget_tokens")
    return frozenset(supported)


def _intent_fingerprint(intent: ConsoleDefaultMutationIntent) -> str:
    """Return a value-hiding identity for same-generation retry validation."""

    endpoint = intent.endpoint_patch
    material = repr(
        (
            intent.generation,
            intent.action.value if isinstance(intent.action, ConsoleSettingsAction) else "",
            intent.provider_config_key,
            intent.literal_model_id,
            tuple(sorted(intent.field_mask)),
            tuple(sorted((name, repr(value)) for name, value in intent.values.items())),
            None
            if endpoint is None
            else (
                endpoint.bound_provider_config_key,
                endpoint.dirty,
                endpoint.checked,
                repr(endpoint.value),
            ),
        )
    ).encode("utf-8", errors="replace")
    return hashlib.sha256(material).hexdigest()


def _reserve_intent_generation(intent: ConsoleDefaultMutationIntent) -> str | None:
    """Accept a new generation or the byte-equivalent retry of the newest one."""

    global _LATEST_INTENT_FINGERPRINT, _LATEST_INTENT_GENERATION

    fingerprint = _intent_fingerprint(intent)
    with _INTENT_GENERATION_LOCK:
        if (
            _LATEST_INTENT_GENERATION is None
            or intent.generation > _LATEST_INTENT_GENERATION
        ):
            _LATEST_INTENT_GENERATION = intent.generation
            _LATEST_INTENT_FINGERPRINT = fingerprint
            return fingerprint
        if (
            intent.generation == _LATEST_INTENT_GENERATION
            and fingerprint == _LATEST_INTENT_FINGERPRINT
        ):
            return fingerprint
        return None


def _intent_is_current(generation: int, fingerprint: str) -> bool:
    """Check the latest explicit intent while the config transaction is locked."""

    with _INTENT_GENERATION_LOCK:
        return (
            generation == _LATEST_INTENT_GENERATION
            and fingerprint == _LATEST_INTENT_FINGERPRINT
        )


def _raw_provider_section_name(
    raw_values: Mapping[str, object],
    canonical_provider: str,
) -> str:
    """Resolve one provider table name solely from authoritative user TOML."""

    api_settings = raw_values.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        raise TypeError("api_settings must be a mapping")
    matches = [
        str(name)
        for name in api_settings
        if type(name) is str and provider_config_key(name) == canonical_provider
    ]
    if canonical_provider in {"moonshot", "qwencloud", "zai"} and (
        canonical_provider in matches
    ):
        return canonical_provider
    if matches:
        return matches[0]
    return canonical_provider


def _raw_provider_settings(
    raw_values: Mapping[str, object],
    section_name: str,
) -> Mapping[str, object]:
    api_settings = raw_values.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        raise TypeError("api_settings must be a mapping")
    settings = api_settings.get(section_name, {})
    if not isinstance(settings, Mapping):
        raise TypeError("provider settings must be a mapping")
    return settings


def _configured_endpoint_key(provider_settings: Mapping[str, object]) -> str | None:
    """Return the first endpoint key already configured in authoritative TOML."""

    for key in _ENDPOINT_KEYS:
        value = provider_settings.get(key)
        if isinstance(value, str) and value.strip():
            return key
    return None


def _validate_intent(intent: ConsoleDefaultMutationIntent) -> tuple[str, str]:
    """Validate immutable intent identity without touching configuration."""

    if type(intent.generation) is not int or intent.generation < 0:
        raise ValueError("Intent generation must be a nonnegative integer")
    if not isinstance(intent.action, ConsoleSettingsAction) or intent.action not in {
        ConsoleSettingsAction.SAVE_MODEL_DEFAULT,
        ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
    }:
        raise ValueError("Intent action cannot mutate defaults")
    if type(intent.provider_config_key) is not str:
        raise TypeError("Provider must be a string")
    canonical_provider = provider_config_key(intent.provider_config_key)
    if not canonical_provider:
        raise ValueError("Provider is required")
    if type(intent.literal_model_id) is not str:
        raise TypeError("Literal model ID must be a string")
    literal_model = intent.literal_model_id
    if (
        not literal_model
        or literal_model != literal_model.strip()
        or len(literal_model) > 512
        or not literal_model.isprintable()
    ):
        raise ValueError("Literal model ID is invalid")
    if intent.field_mask not in {
        QUICK_MODEL_DEFAULT_FIELDS,
        FULL_MODEL_DEFAULT_FIELDS,
    }:
        raise ValueError("Default field mask is invalid")
    if not isinstance(intent.values, Mapping):
        raise TypeError("Default values must be a mapping")
    for name in intent.values:
        if type(name) is not str or not name:
            raise TypeError("Default field names must be non-empty strings")
    if intent.field_mask == QUICK_MODEL_DEFAULT_FIELDS and any(
        name not in intent.values or intent.values[name] is None
        for name in QUICK_MODEL_DEFAULT_FIELDS
    ):
        raise ValueError("Quick default fields must be materialized")
    return canonical_provider, literal_model


def _endpoint_patch_is_authorized(
    intent: ConsoleDefaultMutationIntent,
    canonical_provider: str,
) -> bool:
    patch = intent.endpoint_patch
    if patch is None:
        return True
    return (
        intent.action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT
        and intent.field_mask == FULL_MODEL_DEFAULT_FIELDS
        and patch.dirty is True
        and patch.checked is True
        and provider_config_key(patch.bound_provider_config_key)
        == canonical_provider
        and parse_console_endpoint_preview(patch.value) is not None
    )


def _build_locked_default_mutation(
    intent: ConsoleDefaultMutationIntent,
    canonical_provider: str,
    literal_model: str,
    snapshot: config_module.AtomicLiteralMutationSnapshot,
) -> config_module.LiteralSettingsMutation:
    """Build one exact mutation from locked authoritative raw/effective views."""

    raw_section_name = _raw_provider_section_name(
        snapshot.raw_values,
        canonical_provider,
    )
    raw_provider = _raw_provider_settings(snapshot.raw_values, raw_section_name)
    defaults = build_target_default_console_session_settings(
        snapshot.effective_values,
        canonical_provider,
        literal_model,
    )
    if intent.action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT:
        validation_errors = validate_console_session_settings(
            defaults,
            app_config=snapshot.effective_values,
        )
        readiness = build_console_settings_readiness(
            defaults,
            app_config=snapshot.effective_values,
        )
        if validation_errors or not readiness.native_send_supported:
            raise ValueError("Selected provider/model is not ready for new chats")

    supported = _supported_profile_fields(canonical_provider, literal_model)
    owned_fields = intent.field_mask & supported & frozenset(intent.values)
    profile_values: dict[str, object] = {}
    profile_deletes: list[str] = []
    for name in sorted(owned_fields):
        value = intent.values[name]
        if value is None:
            profile_deletes.append(name)
            continue
        if name == "streaming" and type(value) is not bool:
            raise TypeError("Streaming default must be a strict boolean or inherit")
        profile_values[name] = value

    profile_path = (
        "api_settings",
        raw_section_name,
        "model_defaults",
        literal_model,
    )
    section_values: dict[tuple[str, ...], Mapping[str, object]] = {
        profile_path: profile_values
    }
    delete_keys: dict[tuple[str, ...], tuple[str, ...]] = {
        profile_path: tuple(profile_deletes)
    }
    if intent.action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT:
        section_values[("chat_defaults",)] = {
            "provider": canonical_provider,
            "model": literal_model,
        }

    patch = intent.endpoint_patch
    if patch is not None:
        endpoint_key = _configured_endpoint_key(raw_provider)
        if endpoint_key is None:
            raise ValueError("Provider has no authoritative endpoint key to patch")
        section_values[("api_settings", raw_section_name)] = {
            endpoint_key: patch.value.strip()
        }

    return config_module.LiteralSettingsMutation(
        section_values=section_values,
        delete_keys=delete_keys,
    )


def apply_console_default_intent(
    intent: ConsoleDefaultMutationIntent,
) -> ConsoleDefaultMutationOutcome:
    """Atomically apply one exact-model default intent and publish runtime config."""

    try:
        canonical_provider, literal_model = _validate_intent(intent)
        if not _endpoint_patch_is_authorized(intent, canonical_provider):
            raise ValueError("Endpoint patch is not authorized for this action")
        fingerprint = _reserve_intent_generation(intent)
        if fingerprint is None:
            raise ValueError("Default intent has been superseded")
    except Exception:
        return ConsoleDefaultMutationOutcome(
            intent_generation=(
                intent.generation if type(intent.generation) is int else -1
            ),
            file_replaced=False,
            runtime_published=False,
            settings_view=None,
            failure_phase=ConsoleDefaultSavePhase.BEFORE_REPLACE,
        )

    result = config_module.apply_literal_settings_transaction_to_cli_config(
        lambda snapshot: _build_locked_default_mutation(
            intent,
            canonical_provider,
            literal_model,
            snapshot,
        ),
        mutation_precondition=lambda: _intent_is_current(
            intent.generation,
            fingerprint,
        ),
    )
    if result.caches_reloaded and result.settings_view is not None:
        return ConsoleDefaultMutationOutcome(
            intent_generation=intent.generation,
            file_replaced=result.file_replaced,
            runtime_published=True,
            settings_view=result.settings_view,
            failure_phase=None,
        )
    phase = (
        ConsoleDefaultSavePhase.CACHE_PUBLICATION
        if result.file_replaced
        else ConsoleDefaultSavePhase.BEFORE_REPLACE
    )
    return ConsoleDefaultMutationOutcome(
        intent_generation=intent.generation,
        file_replaced=result.file_replaced,
        runtime_published=False,
        settings_view=None,
        failure_phase=phase,
    )


def refresh_console_runtime_after_saved_default() -> RuntimeConfigPublicationResult:
    """Reread and republish config caches without repeating a saved mutation."""

    result = config_module.refresh_runtime_config_from_cli_config()
    return RuntimeConfigPublicationResult(
        published=result.caches_reloaded and result.settings_view is not None,
        settings_view=result.settings_view,
        failure_phase=result.failure_phase,
    )


def parse_console_endpoint_preview(value: str) -> ConsoleEndpointPreview | None:
    """Return only a sanitized authority and syntactic network classification."""

    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        parsed = urlsplit(candidate if "://" in candidate else f"//{candidate}")
        if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        hostname = parsed.hostname
        if not hostname:
            return None
        port = parsed.port
    except (TypeError, ValueError):
        return None

    normalized_host = hostname.rstrip(".").lower()
    if not normalized_host or any(character.isspace() for character in normalized_host):
        return None
    try:
        address = ip_address(normalized_host)
    except ValueError:
        address = None

    if normalized_host == "localhost" or (address is not None and address.is_loopback):
        classification = "Local"
    elif (
        normalized_host.endswith(".local")
        or (
            address is not None
            and (address.is_private or address.is_link_local)
        )
    ):
        classification = "LAN"
    elif address is not None and address.is_global:
        classification = "Remote"
    else:
        classification = "Remote/unknown"

    rendered_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    authority = rendered_host if port is None else f"{rendered_host}:{port}"
    return ConsoleEndpointPreview(authority, classification)


def format_console_endpoint_preview(value: str) -> str | None:
    """Format the safe endpoint preview used by Console default confirmation copy."""

    preview = parse_console_endpoint_preview(value)
    if preview is None:
        return None
    return f"{preview.authority} · {preview.network_classification}"
