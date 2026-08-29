"""Atomic exact-model Console default persistence and recovery helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
from ipaddress import ip_address, ip_network
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
    ConsoleEndpointDraft,
    ConsoleSettingsAction,
    ConsoleSettingsFieldDraft,
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
_INTENT_CALL_LOCK = threading.Lock()
_ACTIVE_INTENT_CALLS: set[tuple[int, str]] = set()
_LATEST_INTENT_GENERATION: int | None = None
_LATEST_INTENT_FINGERPRINT: str | None = None
_LATEST_INTENT_ACTION: ConsoleSettingsAction | None = None
_RFC1918_NETWORKS = (
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
)
_IPV6_UNIQUE_LOCAL_NETWORK = ip_network("fc00::/7")


class _IntentLifecycle(str, Enum):
    """Private state machine for the newest explicit default intent."""

    RESERVED = "reserved"
    IN_FLIGHT = "in_flight"
    RUNTIME_PUBLICATION_PENDING = "runtime_publication_pending"
    BEFORE_REPLACE_RETRYABLE = "before_replace_retryable"
    CACHE_PUBLICATION_RETRYABLE = "cache_publication_retryable"
    TERMINAL = "terminal"


_LATEST_INTENT_LIFECYCLE: _IntentLifecycle | None = None


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


def build_console_default_intent(
    *,
    generation: int,
    action: ConsoleSettingsAction,
    provider_config_key: str,
    literal_model_id: str,
    field_drafts: tuple[ConsoleSettingsFieldDraft, ...],
    field_mask: frozenset[str],
    endpoint: ConsoleEndpointDraft | None,
) -> ConsoleDefaultMutationIntent:
    """Build immutable persistence values from one submitted settings draft."""

    values: dict[str, object | None] = {}
    exposed_names: set[str] = set()
    for draft in field_drafts:
        if not isinstance(draft, ConsoleSettingsFieldDraft):
            raise TypeError("Default field drafts are invalid")
        if draft.name in exposed_names:
            raise ValueError("Default field drafts must have unique names")
        exposed_names.add(draft.name)
        if draft.name not in field_mask:
            continue
        values[draft.name] = (
            draft.effective_value
            if field_mask == QUICK_MODEL_DEFAULT_FIELDS
            else draft.profile_override
        )

    endpoint_patch = (
        None
        if action is not ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT
        or endpoint is None
        else ConsoleEndpointPatch(
            value=endpoint.value,
            bound_provider_config_key=endpoint.bound_provider_config_key,
            dirty=endpoint.dirty,
            checked=endpoint.checked,
        )
    )
    return ConsoleDefaultMutationIntent(
        generation=generation,
        action=action,
        provider_config_key=provider_config_key,
        literal_model_id=literal_model_id,
        field_mask=field_mask,
        values=values,
        endpoint_patch=endpoint_patch,
    )


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
class ConsoleDefaultDurabilityState:
    """Newest app-owned Console default intent and its recovery phase.

    This presentation-neutral value belongs to the application lifetime.  A
    settings surface receives a snapshot and emits generation-bound recovery
    requests; it never owns or mutates the durable operation itself.
    """

    newest_intent_generation: int = 0
    recovery_intent: ConsoleDefaultMutationIntent | None = None
    failure_phase: ConsoleDefaultSavePhase | None = None
    runtime_published_intent_generation: int | None = None

    def __post_init__(self) -> None:
        """Reject internally inconsistent recovery snapshots."""

        if type(self.newest_intent_generation) is not int:
            raise TypeError("Newest default intent generation must be an integer")
        if self.newest_intent_generation < 0:
            raise ValueError("Newest default intent generation cannot be negative")
        intent = self.recovery_intent
        if (intent is None) is not (self.failure_phase is None):
            raise ValueError("Default recovery intent and failure phase must agree")
        if intent is not None and intent.generation != self.newest_intent_generation:
            raise ValueError("Default recovery must target the newest intent")

    def accept_runtime_publication(
        self,
        intent_generation: int,
    ) -> tuple[ConsoleDefaultDurabilityState, bool]:
        """Acknowledge one current runtime publication idempotently.

        Args:
            intent_generation: Generation whose fresh settings mapping was
                assigned to the running application.

        Returns:
            ``(state, accepted)``.  ``accepted`` is true only for the first
            publication of the newest intent generation.
        """

        if (
            type(intent_generation) is not int
            or intent_generation != self.newest_intent_generation
            or self.runtime_published_intent_generation == intent_generation
        ):
            return self, False
        return (
            ConsoleDefaultDurabilityState(
                newest_intent_generation=self.newest_intent_generation,
                runtime_published_intent_generation=intent_generation,
            ),
            True,
        )


@dataclass(frozen=True, slots=True)
class ConsoleEndpointPreview:
    """Credential-free endpoint authority and conservative network class."""

    authority: str
    network_classification: str


@dataclass(frozen=True, slots=True)
class _PendingRetryState:
    """Private optimistic baseline for the newest retryable disk intent."""

    generation: int
    fingerprint: str
    owned_baseline: tuple[tuple[tuple[str, ...], str, str], ...]


_PENDING_RETRY_STATE: _PendingRetryState | None = None


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
    """Start a newer intent or one explicit retry of a retryable generation."""

    global _LATEST_INTENT_ACTION, _LATEST_INTENT_FINGERPRINT, _LATEST_INTENT_GENERATION
    global _LATEST_INTENT_LIFECYCLE, _PENDING_RETRY_STATE

    fingerprint = _intent_fingerprint(intent)
    with _INTENT_GENERATION_LOCK:
        if (
            _LATEST_INTENT_GENERATION is None
            or intent.generation > _LATEST_INTENT_GENERATION
        ):
            _LATEST_INTENT_GENERATION = intent.generation
            _LATEST_INTENT_FINGERPRINT = fingerprint
            _LATEST_INTENT_ACTION = intent.action
            _LATEST_INTENT_LIFECYCLE = _IntentLifecycle.IN_FLIGHT
            _PENDING_RETRY_STATE = None
            return fingerprint
        if (
            intent.generation == _LATEST_INTENT_GENERATION
            and fingerprint == _LATEST_INTENT_FINGERPRINT
            and _LATEST_INTENT_LIFECYCLE
            in {
                _IntentLifecycle.RESERVED,
                _IntentLifecycle.BEFORE_REPLACE_RETRYABLE,
            }
        ):
            _LATEST_INTENT_LIFECYCLE = _IntentLifecycle.IN_FLIGHT
            return fingerprint
        return None


def reserve_console_default_intent_generation(
    intent: ConsoleDefaultMutationIntent,
    *,
    pending_runtime_publisher: Callable[
        [int, ConsoleSettingsAction, Mapping[str, object]],
        bool,
    ]
    | None = None,
) -> bool:
    """Reserve a new application intent before its worker can be scheduled.

    This synchronous reservation is deliberately separate from disk mutation.
    The config transaction's locked precondition can therefore observe a newer
    user intent even while an older worker is waiting inside the config lock.

    Args:
        intent: Exact default mutation the caller is about to schedule.
        pending_runtime_publisher: Nonblocking application-view publisher for
            a prior successful intent. It runs while config and intent
            publication are fenced, before the newer intent is reserved.

    Returns:
        ``True`` when this exact intent owns the current reservation.

    Raises:
        TypeError: An argument has the wrong type.
        RuntimeError: A prior durable runtime view could not be published.
    """

    global _LATEST_INTENT_ACTION, _LATEST_INTENT_FINGERPRINT, _LATEST_INTENT_GENERATION
    global _LATEST_INTENT_LIFECYCLE, _PENDING_RETRY_STATE

    if not isinstance(intent, ConsoleDefaultMutationIntent):
        raise TypeError("intent must be ConsoleDefaultMutationIntent")
    if pending_runtime_publisher is not None and not callable(
        pending_runtime_publisher
    ):
        raise TypeError("pending_runtime_publisher must be callable")
    fingerprint = _intent_fingerprint(intent)

    def reserve_unlocked() -> bool:
        global _LATEST_INTENT_ACTION, _LATEST_INTENT_FINGERPRINT
        global _LATEST_INTENT_GENERATION, _LATEST_INTENT_LIFECYCLE
        global _PENDING_RETRY_STATE

        if (
            _LATEST_INTENT_GENERATION is None
            or intent.generation > _LATEST_INTENT_GENERATION
        ):
            _LATEST_INTENT_GENERATION = intent.generation
            _LATEST_INTENT_FINGERPRINT = fingerprint
            _LATEST_INTENT_ACTION = intent.action
            _LATEST_INTENT_LIFECYCLE = _IntentLifecycle.RESERVED
            _PENDING_RETRY_STATE = None
            return True
        return bool(
            intent.generation == _LATEST_INTENT_GENERATION
            and fingerprint == _LATEST_INTENT_FINGERPRINT
            and _LATEST_INTENT_LIFECYCLE is _IntentLifecycle.RESERVED
        )

    while True:
        with _INTENT_GENERATION_LOCK:
            if (
                _LATEST_INTENT_LIFECYCLE
                is not _IntentLifecycle.RUNTIME_PUBLICATION_PENDING
            ):
                return reserve_unlocked()
        if pending_runtime_publisher is None:
            return False

        # Read config before taking the intent lock, then validate that exact
        # generation again under the config lock. The callback and reservation
        # run in config -> intent order, matching the mutation transaction.
        # No settings mapping is retained in process-global intent state.
        snapshot = config_module.get_runtime_config_snapshot()
        reservation_accepted = False

        def publish_pending_and_reserve() -> bool:
            nonlocal reservation_accepted
            global _LATEST_INTENT_LIFECYCLE, _PENDING_RETRY_STATE

            with _INTENT_GENERATION_LOCK:
                if (
                    _LATEST_INTENT_LIFECYCLE
                    is _IntentLifecycle.RUNTIME_PUBLICATION_PENDING
                ):
                    generation = _LATEST_INTENT_GENERATION
                    action = _LATEST_INTENT_ACTION
                    if generation is None or action is None:
                        raise RuntimeError("Pending default publication is invalid")
                    if (
                        pending_runtime_publisher(
                            generation,
                            action,
                            snapshot.values,
                        )
                        is not True
                    ):
                        raise RuntimeError("Pending default publication was rejected")
                    _LATEST_INTENT_LIFECYCLE = _IntentLifecycle.TERMINAL
                    _PENDING_RETRY_STATE = None
                reservation_accepted = reserve_unlocked()
            return True

        if config_module.run_if_runtime_config_generation_current(
            snapshot.generation,
            publish_pending_and_reserve,
        ):
            return reservation_accepted


def next_console_default_intent_generation(after: int) -> int:
    """Return a process-monotonic candidate newer than app and service state."""

    if type(after) is not int or after < 0:
        raise ValueError("after must be a nonnegative integer")
    with _INTENT_GENERATION_LOCK:
        return max(after, _LATEST_INTENT_GENERATION or 0) + 1


def _retry_state_for_intent(
    generation: int,
    fingerprint: str,
) -> _PendingRetryState | None:
    """Return retry state only while it still belongs to this exact intent."""

    with _INTENT_GENERATION_LOCK:
        retry_state = _PENDING_RETRY_STATE
        if retry_state is None or (
            retry_state.generation != generation
            or retry_state.fingerprint != fingerprint
        ):
            return None
        return retry_state


def _finish_intent_success(generation: int, fingerprint: str) -> None:
    """Leave a still-current success pending application-view publication."""

    global _LATEST_INTENT_LIFECYCLE, _PENDING_RETRY_STATE
    with _INTENT_GENERATION_LOCK:
        if (
            generation != _LATEST_INTENT_GENERATION
            or fingerprint != _LATEST_INTENT_FINGERPRINT
        ):
            return
        _LATEST_INTENT_LIFECYCLE = _IntentLifecycle.RUNTIME_PUBLICATION_PENDING
        if _PENDING_RETRY_STATE is not None and (
            _PENDING_RETRY_STATE.generation == generation
            and _PENDING_RETRY_STATE.fingerprint == fingerprint
        ):
            _PENDING_RETRY_STATE = None


def publish_console_default_runtime_if_current(
    intent: ConsoleDefaultMutationIntent,
    outcome: ConsoleDefaultMutationOutcome,
    publisher: Callable[[Mapping[str, object]], bool],
) -> bool:
    """Publish one current runtime view while fencing newer reservations.

    Args:
        intent: Intent whose worker produced ``outcome``.
        outcome: Successful runtime publication result to install.
        publisher: Nonblocking application-view assignment callback.

    Returns:
        ``True`` only when this exact current intent was published once.

    Raises:
        TypeError: An argument has the wrong type.
    """

    if not isinstance(intent, ConsoleDefaultMutationIntent):
        raise TypeError("intent must be ConsoleDefaultMutationIntent")
    if not isinstance(outcome, ConsoleDefaultMutationOutcome):
        raise TypeError("outcome must be ConsoleDefaultMutationOutcome")
    if not callable(publisher):
        raise TypeError("publisher must be callable")
    fingerprint = _intent_fingerprint(intent)
    global _LATEST_INTENT_LIFECYCLE, _PENDING_RETRY_STATE
    with _INTENT_GENERATION_LOCK:
        if (
            outcome.intent_generation != intent.generation
            or not outcome.runtime_published
            or outcome.settings_view is None
            or outcome.failure_phase is not None
            or intent.generation != _LATEST_INTENT_GENERATION
            or fingerprint != _LATEST_INTENT_FINGERPRINT
            or _LATEST_INTENT_LIFECYCLE
            not in {
                _IntentLifecycle.RUNTIME_PUBLICATION_PENDING,
                _IntentLifecycle.CACHE_PUBLICATION_RETRYABLE,
                # Test doubles may supply the worker outcome directly after
                # reserving; production workers transition through IN_FLIGHT.
                _IntentLifecycle.RESERVED,
            }
        ):
            return False
        if publisher(outcome.settings_view) is not True:
            return False
        _LATEST_INTENT_LIFECYCLE = _IntentLifecycle.TERMINAL
        _PENDING_RETRY_STATE = None
        return True


def _finish_intent_failure(
    generation: int,
    fingerprint: str,
    phase: ConsoleDefaultSavePhase,
    *,
    captured_baseline: tuple[tuple[tuple[str, ...], str, str], ...] | None,
    was_retry: bool,
) -> None:
    """Publish retry lifecycle only if no newer reservation superseded it."""

    global _LATEST_INTENT_LIFECYCLE, _PENDING_RETRY_STATE
    with _INTENT_GENERATION_LOCK:
        if (
            generation != _LATEST_INTENT_GENERATION
            or fingerprint != _LATEST_INTENT_FINGERPRINT
        ):
            return
        if phase is ConsoleDefaultSavePhase.CACHE_PUBLICATION:
            _LATEST_INTENT_LIFECYCLE = (
                _IntentLifecycle.CACHE_PUBLICATION_RETRYABLE
            )
            if _PENDING_RETRY_STATE is not None and (
                _PENDING_RETRY_STATE.generation == generation
                and _PENDING_RETRY_STATE.fingerprint == fingerprint
            ):
                _PENDING_RETRY_STATE = None
            return
        _LATEST_INTENT_LIFECYCLE = _IntentLifecycle.BEFORE_REPLACE_RETRYABLE
        if captured_baseline is not None and not was_retry:
            _PENDING_RETRY_STATE = _PendingRetryState(
                generation=generation,
                fingerprint=fingerprint,
                owned_baseline=captured_baseline,
            )


def _acquire_current_intent_commit_fence(
    generation: int,
    fingerprint: str,
) -> bool:
    """Fence a current intent through file replacement and cache publication.

    The transaction invokes this while holding the config write lock, which
    establishes the only cross-lock order: config, then intent.  A successful
    check deliberately keeps the reentrant intent lock held until the caller
    has consumed the transaction result and published its lifecycle state.
    """

    _INTENT_GENERATION_LOCK.acquire()
    if (
        generation == _LATEST_INTENT_GENERATION
        and fingerprint == _LATEST_INTENT_FINGERPRINT
        and _LATEST_INTENT_LIFECYCLE is _IntentLifecycle.IN_FLIGHT
    ):
        return True
    _INTENT_GENERATION_LOCK.release()
    return False


def _register_active_intent_call(generation: int, fingerprint: str) -> bool:
    """Reject an exact duplicate while its first worker is still active."""

    key = (generation, fingerprint)
    with _INTENT_CALL_LOCK:
        if key in _ACTIVE_INTENT_CALLS:
            return False
        _ACTIVE_INTENT_CALLS.add(key)
        return True


def _unregister_active_intent_call(generation: int, fingerprint: str) -> None:
    """Release one exact worker-call reservation."""

    with _INTENT_CALL_LOCK:
        _ACTIVE_INTENT_CALLS.discard((generation, fingerprint))


def _owned_baseline_digest(*, present: bool, value: object = None) -> str:
    """Hash one raw owned value so private retry state retains no payload."""

    material = repr(
        (
            present,
            type(value).__module__ if present else "",
            type(value).__qualname__ if present else "",
            value if present else "",
        )
    ).encode("utf-8", errors="replace")
    return hashlib.sha256(material).hexdigest()


def _raw_owned_value_digest(
    raw_values: Mapping[str, object],
    path: tuple[str, ...],
    key: str,
) -> str:
    """Read and hash one exact raw path/key without creating mappings."""

    current: object = raw_values
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return _owned_baseline_digest(present=False)
        current = current[part]
    if not isinstance(current, Mapping) or key not in current:
        return _owned_baseline_digest(present=False)
    return _owned_baseline_digest(present=True, value=current[key])


def _capture_owned_baseline(
    raw_values: Mapping[str, object],
    mutation: config_module.LiteralSettingsMutation,
) -> tuple[tuple[tuple[str, ...], str, str], ...]:
    """Capture hashes for only the exact keys one immutable intent owns."""

    targets = {
        (path, key)
        for path, values in mutation.section_values.items()
        for key in values
    }
    targets.update(
        (path, key)
        for path, keys in mutation.delete_keys.items()
        for key in keys
    )
    return tuple(
        (path, key, _raw_owned_value_digest(raw_values, path, key))
        for path, key in sorted(targets)
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

    call_fingerprint = _intent_fingerprint(intent)
    if not _register_active_intent_call(intent.generation, call_fingerprint):
        return ConsoleDefaultMutationOutcome(
            intent_generation=intent.generation,
            file_replaced=False,
            runtime_published=False,
            settings_view=None,
            failure_phase=ConsoleDefaultSavePhase.BEFORE_REPLACE,
        )

    try:
        fingerprint = _reserve_intent_generation(intent)
        if fingerprint is None:
            return ConsoleDefaultMutationOutcome(
                intent_generation=intent.generation,
                file_replaced=False,
                runtime_published=False,
                settings_view=None,
                failure_phase=ConsoleDefaultSavePhase.BEFORE_REPLACE,
            )
        retry_state = _retry_state_for_intent(intent.generation, fingerprint)
        captured_baselines: list[
            tuple[tuple[tuple[str, ...], str, str], ...]
        ] = []

        def build_mutation(
            snapshot: config_module.AtomicLiteralMutationSnapshot,
        ) -> config_module.LiteralSettingsMutation:
            mutation = _build_locked_default_mutation(
                intent,
                canonical_provider,
                literal_model,
                snapshot,
            )
            baseline = _capture_owned_baseline(snapshot.raw_values, mutation)
            captured_baselines.append(baseline)
            if retry_state is not None and baseline != retry_state.owned_baseline:
                raise ValueError("Retry target changed after the failed save")
            return mutation

        commit_fence_acquired = False

        def acquire_commit_fence() -> bool:
            nonlocal commit_fence_acquired
            if commit_fence_acquired:
                return True
            commit_fence_acquired = _acquire_current_intent_commit_fence(
                intent.generation,
                fingerprint,
            )
            return commit_fence_acquired

        try:
            result = config_module.apply_literal_settings_transaction_to_cli_config(
                build_mutation,
                mutation_precondition=acquire_commit_fence,
            )
            if result.caches_reloaded and result.settings_view is not None:
                _finish_intent_success(intent.generation, fingerprint)
                return ConsoleDefaultMutationOutcome(
                    intent_generation=intent.generation,
                    file_replaced=result.file_replaced,
                    runtime_published=True,
                    settings_view=result.settings_view,
                    failure_phase=None,
                )
            phase = (
                ConsoleDefaultSavePhase.CACHE_PUBLICATION
                if result.failure_phase == "cache_reload"
                else ConsoleDefaultSavePhase.BEFORE_REPLACE
            )
            _finish_intent_failure(
                intent.generation,
                fingerprint,
                phase,
                captured_baseline=(captured_baselines[0] if captured_baselines else None),
                was_retry=retry_state is not None,
            )
            return ConsoleDefaultMutationOutcome(
                intent_generation=intent.generation,
                file_replaced=result.file_replaced,
                runtime_published=False,
                settings_view=None,
                failure_phase=phase,
            )
        finally:
            if commit_fence_acquired:
                _INTENT_GENERATION_LOCK.release()
    finally:
        _unregister_active_intent_call(intent.generation, call_fingerprint)


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
    if (
        not candidate
        or not candidate.isprintable()
        or any(character.isspace() for character in candidate)
        or "\\" in candidate
    ):
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
    if not normalized_host or "%" in normalized_host:
        return None
    try:
        address = ip_address(normalized_host)
    except ValueError:
        address = None
    if address is None and not _is_valid_ascii_hostname(normalized_host):
        return None

    if normalized_host == "localhost" or (address is not None and address.is_loopback):
        classification = "Local"
    elif (
        normalized_host.endswith(".local")
        or (
            address is not None
            and (
                address.is_link_local
                or (
                    address.version == 4
                    and any(address in network for network in _RFC1918_NETWORKS)
                )
                or (
                    address.version == 6
                    and address in _IPV6_UNIQUE_LOCAL_NETWORK
                )
            )
        )
    ):
        classification = "LAN"
    elif address is not None and address.is_global and not address.is_multicast:
        classification = "Remote"
    else:
        classification = "Remote/unknown"

    rendered_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    authority = rendered_host if port is None else f"{rendered_host}:{port}"
    return ConsoleEndpointPreview(authority, classification)


def _is_valid_ascii_hostname(hostname: str) -> bool:
    """Return whether a non-IP host is a conservative ASCII DNS name."""

    if not hostname.isascii() or len(hostname) > 253:
        return False
    labels = hostname.split(".")
    return all(
        1 <= len(label) <= 63
        and label[0] != "-"
        and label[-1] != "-"
        and all(character.isascii() and (character.isalnum() or character == "-") for character in label)
        for label in labels
    )


def format_console_endpoint_preview(value: str) -> str | None:
    """Format the safe endpoint preview used by Console default confirmation copy."""

    preview = parse_console_endpoint_preview(value)
    if preview is None:
        return None
    return f"{preview.authority} · {preview.network_classification}"
