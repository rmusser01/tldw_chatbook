"""Provider-owned setup persistence contracts and atomic mutation helpers."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Literal
from unicodedata import category as unicode_category
from urllib.parse import urlsplit, urlunsplit
from weakref import ReferenceType, ref

from ..config import (
    DEFAULT_CONFIG_FROM_TOML,
    AtomicConfigSnapshot,
    ConfigMutationResult,
    apply_settings_mutation_to_cli_config,
    is_valid_provider_api_key,
)
from .provider_endpoint_contract import (
    canonical_connection_identity,
    resolve_provider_endpoint,
)
from .provider_test_evidence import CredentialSource, ProviderDraftIdentity

_MAX_MODEL_CHARS = 120
_MAX_SECRET_CHARS = 8192
_MAX_CONFIG_PROVIDERS = 256
_MAX_IDENTITY_COUNTER = 2**63 - 1
_MAX_MUTATION_SECTIONS = 3
_MAX_MUTATION_KEYS = 16
_MAX_COMBINED_MUTATION_SECTIONS = 8
_MAX_COMBINED_MUTATION_KEYS = 32
_UNSAFE_TEXT_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_ENV_VAR_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_SECTION_NAME_PATTERN = re.compile(r"[A-Za-z0-9_. -]{1,192}")
_CONFIG_KEY_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_ENDPOINT_KEY_PRECEDENCE = (
    "api_base_url",
    "api_base",
    "base_url",
    "api_url",
    "endpoint",
)
_API_BASE_ENDPOINT_KEYS = frozenset({"api_base_url", "api_base", "base_url"})
_ROOT_ENDPOINT_PROVIDER_KEYS = frozenset({"llama_cpp", "local_llamacpp"})
_ROUTING_SETTING_KEYS = (
    *_ENDPOINT_KEY_PRECEDENCE,
    "api_endpoint",
    "router_base_url",
    "huggingface_router_base_url",
    "api_region",
    "use_router_url_format",
    "huggingface_use_router_url_format",
)
_ISSUED_MUTATION_LOCK = RLock()
_ISSUED_MUTATIONS: dict[int, ReferenceType[ProviderSetupMutation]] = {}
_WRITE_EXPECTATIONS: dict[int, _ProviderSetupWriteBinding] = {}
_CREDENTIAL_OBSERVATION_KEY = secrets.token_bytes(32)


@dataclass(frozen=True, slots=True)
class _ProviderOwnership:
    provider_key: str
    config_section: str
    endpoint_key: str
    model_key: str = "model"
    credential_keys: tuple[str, str] = ("api_key", "api_key_env_var")


_API_URL_PROVIDER_KEYS = frozenset(
    {
        "aphrodite",
        "custom",
        "custom_2",
        "koboldcpp",
        "llama_cpp",
        "local_llamacpp",
        "local_llamafile",
        "local_llm",
        "local_mlx_lm",
        "local_ollama",
        "local_onnx",
        "local_transformers",
        "local_vllm",
        "mlx_lm",
        "ollama",
        "oobabooga",
        "tabbyapi",
        "vllm",
    }
)
_CANONICAL_PROVIDER_KEYS = frozenset(
    {
        "anthropic",
        "aphrodite",
        "cohere",
        "custom",
        "custom_2",
        "deepseek",
        "google",
        "groq",
        "huggingface",
        "koboldcpp",
        "llama_cpp",
        "local_llm",
        "local_llamacpp",
        "local_llamafile",
        "local_mlx_lm",
        "local_ollama",
        "local_onnx",
        "local_transformers",
        "local_vllm",
        "mistral",
        "mistralai",
        "moonshot",
        "ollama",
        "oobabooga",
        "openai",
        "openrouter",
        "qwencloud",
        "tabbyapi",
        "vllm",
        "zai",
    }
)
_CONFIG_SECTION_OVERRIDES = {
    "local_llm": "local-llm",
}
_ALIASES = {
    "Anthropic": "anthropic",
    "Aphrodite": "aphrodite",
    "Cohere": "cohere",
    "Custom": "custom",
    "Custom OpenAI": "custom",
    "Custom OpenAI API": "custom",
    "custom-openai": "custom",
    "custom_openai": "custom",
    "custom-openai-api": "custom",
    "custom_openai_api": "custom",
    "Custom 2": "custom_2",
    "Custom-2": "custom_2",
    "Custom OpenAI 2": "custom_2",
    "Custom OpenAI API 2": "custom_2",
    "Custom OpenAI API-2": "custom_2",
    "custom-2": "custom_2",
    "custom-openai-2": "custom_2",
    "custom_openai_2": "custom_2",
    "custom-openai-api-2": "custom_2",
    "custom_openai_api_2": "custom_2",
    "DeepSeek": "deepseek",
    "Google": "google",
    "Groq": "groq",
    "HuggingFace": "huggingface",
    "Hugging Face": "huggingface",
    "Llama_cpp": "llama_cpp",
    "llama.cpp": "llama_cpp",
    "local llama.cpp": "local_llamacpp",
    "local-llamacpp": "local_llamacpp",
    "local-llm": "local_llm",
    "mlx_lm": "local_mlx_lm",
    "Mistral": "mistral",
    "MistralAI": "mistralai",
    "Moonshot": "moonshot",
    "Ollama": "ollama",
    "Oobabooga": "oobabooga",
    "OpenAI": "openai",
    "OpenRouter": "openrouter",
    "QwenCloud": "qwencloud",
    "TabbyAPI": "tabbyapi",
    "vLLM": "vllm",
    "ZAI": "zai",
}


def _ownership_for(provider: object) -> _ProviderOwnership:
    if type(provider) is not str or not provider or len(provider) > 128:
        raise ValueError("Provider is not supported.")
    provider_key = _ALIASES.get(provider, provider)
    if provider_key not in _CANONICAL_PROVIDER_KEYS:
        raise ValueError("Provider is not supported.")
    config_section = _CONFIG_SECTION_OVERRIDES.get(provider_key, provider_key)
    endpoint_key = (
        "api_url" if provider_key in _API_URL_PROVIDER_KEYS else "api_base_url"
    )
    return _ProviderOwnership(provider_key, config_section, endpoint_key)


def canonical_provider_key(provider: object) -> str:
    """Return the canonical config owner key for a supported provider alias."""

    return _ownership_for(provider).provider_key


def provider_endpoint_key(provider: object) -> str:
    """Return the exact endpoint key owned by an established provider."""

    return _ownership_for(provider).endpoint_key


def provider_model_key(provider: object) -> str:
    """Return the exact remembered-model key owned by a provider."""

    return _ownership_for(provider).model_key


def provider_credential_keys(provider: object) -> tuple[str, str]:
    """Return stored-key and environment-source keys for a provider."""

    return _ownership_for(provider).credential_keys


@dataclass(frozen=True, slots=True)
class ProviderSetupDraft:
    """Validated inputs needed to persist one provider setup."""

    provider: str
    model: str
    endpoint: str
    credential_source: CredentialSource
    credential_revision: int
    draft_generation: int
    credential_value: str | None = field(default=None, repr=False)
    credential_env_var: str | None = None

    def __post_init__(self) -> None:
        _ownership_for(self.provider)
        if type(self.model) is not str or (
            self.model.strip() and _safe_model(self.model) != self.model.strip()
        ):
            raise ValueError("Model is invalid.")
        if type(self.endpoint) is not str:
            raise ValueError("Endpoint is invalid.")
        if self.credential_source not in {"none", "stored", "environment", "draft"}:
            raise ValueError("Credential source is invalid.")
        if (
            type(self.credential_revision) is not int
            or not 0 <= self.credential_revision <= _MAX_IDENTITY_COUNTER
        ):
            raise ValueError("Credential revision is invalid.")
        if (
            type(self.draft_generation) is not int
            or not 0 <= self.draft_generation <= _MAX_IDENTITY_COUNTER
        ):
            raise ValueError("Draft generation is invalid.")
        if self.credential_value is not None:
            _validate_credential_value(self.credential_value)
        if (
            self.credential_env_var is not None
            and self.credential_env_var
            and _ENV_VAR_PATTERN.fullmatch(self.credential_env_var) is None
        ):
            raise ValueError("Credential environment variable is invalid.")
        credential_value = (
            self.credential_value.strip() if self.credential_value is not None else ""
        )
        credential_env_var = (
            self.credential_env_var.strip()
            if self.credential_env_var is not None
            else ""
        )
        if self.credential_source in {"draft", "stored"} and credential_env_var:
            raise ValueError("Credential setup is invalid.")
        if self.credential_source == "environment" and credential_value:
            raise ValueError("Credential setup is invalid.")
        if self.credential_source == "none" and (
            credential_value or credential_env_var
        ):
            raise ValueError("Credential setup is invalid.")


ModelSelectionProvenance = Literal["discovered", "manual"]


@dataclass(frozen=True, slots=True)
class ProviderSetupWriteIdentity:
    """Secret-free identity of one provider/model decision at save time."""

    provider_key: str
    connection_identity: tuple[str, str]
    credential_source: CredentialSource
    credential_revision: int
    model_id: str
    model_provenance: ModelSelectionProvenance

    def __post_init__(self) -> None:
        model = _safe_model(self.model_id)
        if model is None or model != self.model_id:
            raise ValueError("Provider setup write model is invalid.")
        if self.model_provenance not in {"discovered", "manual"}:
            raise ValueError("Provider setup write provenance is invalid.")
        try:
            identity = ProviderDraftIdentity(
                provider_key=self.provider_key,
                connection_identity=self.connection_identity,
                credential_source=self.credential_source,
                credential_revision=self.credential_revision,
                draft_generation=0,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Provider setup write identity is invalid.") from exc
        if identity.provider_key != canonical_provider_key(self.provider_key):
            raise ValueError("Provider setup write identity is invalid.")


@dataclass(frozen=True, slots=True)
class ProviderSetupWriteExpectation:
    """Generation-stamped compare-and-swap token for one atomic write."""

    identity: ProviderSetupWriteIdentity
    generation: int

    def __post_init__(self) -> None:
        if type(self.identity) is not ProviderSetupWriteIdentity:
            raise ValueError("Provider setup write expectation is invalid.")
        if (
            type(self.generation) is not int
            or not 0 <= self.generation <= _MAX_IDENTITY_COUNTER
        ):
            raise ValueError("Provider setup write generation is invalid.")


class ProviderSetupWriteGuard:
    """Synchronize identity changes with the provider's atomic config writer."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._generation = 0
        self._identity: ProviderSetupWriteIdentity | None = None

    def arm(
        self, identity: ProviderSetupWriteIdentity
    ) -> ProviderSetupWriteExpectation:
        if type(identity) is not ProviderSetupWriteIdentity:
            raise ValueError("Provider setup write identity is invalid.")
        with self._lock:
            self._advance()
            self._identity = identity
            return ProviderSetupWriteExpectation(identity, self._generation)

    def invalidate(self) -> None:
        with self._lock:
            self._advance()
            self._identity = None

    def _advance(self) -> None:
        if self._generation >= _MAX_IDENTITY_COUNTER:
            self._generation = 0
        else:
            self._generation += 1

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Hold identity changes until the atomic config writer returns."""

        with self._lock:
            yield

    def matches(
        self,
        expectation: ProviderSetupWriteExpectation,
    ) -> bool:
        """Evaluate one expectation while the caller owns the guard lease."""

        if type(expectation) is not ProviderSetupWriteExpectation:
            raise ValueError("Provider setup write expectation is invalid.")
        with self._lock:
            return bool(
                self._generation == expectation.generation
                and self._identity == expectation.identity
            )


class _ProviderCredentialObservation:
    """Private process-local equality token for one effective credential."""

    __slots__ = ("_tag", "source")

    def __init__(self, source: CredentialSource, payload: str) -> None:
        self.source = source
        self._tag = hmac.new(
            _CREDENTIAL_OBSERVATION_KEY,
            payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()

    def matches(self, other: object) -> bool:
        return bool(
            type(other) is _ProviderCredentialObservation
            and self.source == other.source
            and hmac.compare_digest(self._tag, other._tag)
        )

    def __repr__(self) -> str:
        return f"_ProviderCredentialObservation(source={self.source!r})"


class ProviderSetupConfigPrecondition:
    """Immutable secret-free provider state captured before model selection."""

    __slots__ = (
        "__credential_observation",
        "config_generation",
        "configured_connection_identity",
        "configured_model_state",
        "configured_routing_state",
        "provider_key",
    )

    def __init__(
        self,
        *,
        config_generation: int,
        provider_key: str,
        configured_connection_identity: tuple[str, str] | None,
        configured_routing_state: tuple[tuple[str, str], ...],
        configured_model_state: tuple[str | None, str | None, str | None],
        credential_observation: _ProviderCredentialObservation,
    ) -> None:
        if type(config_generation) is not int or config_generation < 0:
            raise ValueError("Provider setup expected generation is invalid.")
        canonical_key = canonical_provider_key(provider_key)
        if canonical_key != provider_key:
            raise ValueError("Provider setup expected provider is invalid.")
        if not (
            configured_connection_identity is None
            or (
                type(configured_connection_identity) is tuple
                and len(configured_connection_identity) == 2
                and all(type(item) is str for item in configured_connection_identity)
            )
        ):
            raise ValueError("Provider setup expected route is invalid.")
        if (
            type(configured_routing_state) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or any(type(value) is not str for value in item)
                for item in configured_routing_state
            )
            or type(configured_model_state) is not tuple
            or len(configured_model_state) != 3
            or any(
                item is not None and type(item) is not str
                for item in configured_model_state
            )
            or type(credential_observation) is not _ProviderCredentialObservation
        ):
            raise ValueError("Provider setup expected state is invalid.")
        object.__setattr__(self, "config_generation", config_generation)
        object.__setattr__(self, "provider_key", canonical_key)
        object.__setattr__(
            self,
            "configured_connection_identity",
            configured_connection_identity,
        )
        object.__setattr__(self, "configured_routing_state", configured_routing_state)
        object.__setattr__(self, "configured_model_state", configured_model_state)
        object.__setattr__(
            self,
            "_ProviderSetupConfigPrecondition__credential_observation",
            credential_observation,
        )

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Provider setup config precondition is immutable.")

    def __repr__(self) -> str:
        return (
            "ProviderSetupConfigPrecondition("
            f"config_generation={self.config_generation!r}, "
            f"provider_key={self.provider_key!r}, "
            "configured_connection_identity="
            f"{self.configured_connection_identity!r}, "
            f"configured_routing_state={self.configured_routing_state!r}, "
            f"configured_model_state={self.configured_model_state!r})"
        )

    def _credential(self) -> _ProviderCredentialObservation:
        return object.__getattribute__(
            self,
            "_ProviderSetupConfigPrecondition__credential_observation",
        )


class ExpectedProviderSetupState:
    """Immutable secret-free CAS state captured from authoritative config."""

    __slots__ = (
        "__credential_observation",
        "config_generation",
        "configured_connection_identity",
        "configured_model_state",
        "configured_routing_state",
        "identity",
    )

    def __init__(
        self,
        *,
        config_generation: int,
        identity: ProviderSetupWriteIdentity,
        configured_connection_identity: tuple[str, str] | None,
        configured_routing_state: tuple[tuple[str, str], ...],
        configured_model_state: tuple[str | None, str | None, str | None],
        credential_observation: _ProviderCredentialObservation,
    ) -> None:
        if type(config_generation) is not int or config_generation < 0:
            raise ValueError("Provider setup expected generation is invalid.")
        if type(identity) is not ProviderSetupWriteIdentity:
            raise ValueError("Provider setup expected identity is invalid.")
        if not (
            configured_connection_identity is None
            or (
                type(configured_connection_identity) is tuple
                and len(configured_connection_identity) == 2
                and all(type(item) is str for item in configured_connection_identity)
            )
        ):
            raise ValueError("Provider setup expected route is invalid.")
        if (
            type(configured_routing_state) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or any(type(value) is not str for value in item)
                for item in configured_routing_state
            )
            or type(configured_model_state) is not tuple
            or len(configured_model_state) != 3
            or any(
                item is not None and type(item) is not str
                for item in configured_model_state
            )
            or type(credential_observation) is not _ProviderCredentialObservation
        ):
            raise ValueError("Provider setup expected state is invalid.")
        object.__setattr__(self, "config_generation", config_generation)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(
            self,
            "configured_connection_identity",
            configured_connection_identity,
        )
        object.__setattr__(self, "configured_routing_state", configured_routing_state)
        object.__setattr__(self, "configured_model_state", configured_model_state)
        object.__setattr__(
            self,
            "_ExpectedProviderSetupState__credential_observation",
            credential_observation,
        )

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Provider setup expected state is immutable.")

    def __repr__(self) -> str:
        return (
            "ExpectedProviderSetupState("
            f"config_generation={self.config_generation!r}, "
            f"identity={self.identity!r}, "
            "configured_connection_identity="
            f"{self.configured_connection_identity!r}, "
            f"configured_routing_state={self.configured_routing_state!r}, "
            f"configured_model_state={self.configured_model_state!r})"
        )

    def _matches_snapshot(self, snapshot: AtomicConfigSnapshot) -> bool:
        try:
            current_route, current_routing, current_models, current_credential = (
                _provider_setup_observations(
                    snapshot.values, self.identity.provider_key
                )
            )
            expected_credential = object.__getattribute__(
                self,
                "_ExpectedProviderSetupState__credential_observation",
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return bool(
            current_route == self.configured_connection_identity
            and current_routing == self.configured_routing_state
            and current_models == self.configured_model_state
            and expected_credential.matches(current_credential)
        )


def capture_expected_provider_setup_state(
    snapshot: AtomicConfigSnapshot,
    *,
    identity: ProviderSetupWriteIdentity,
) -> ExpectedProviderSetupState:
    """Capture relevant provider state from one authoritative config snapshot."""

    if type(snapshot) is not AtomicConfigSnapshot:
        raise ValueError("Provider setup config snapshot is invalid.")
    if type(identity) is not ProviderSetupWriteIdentity:
        raise ValueError("Provider setup write identity is invalid.")
    precondition = capture_provider_setup_precondition(
        snapshot,
        provider=identity.provider_key,
    )
    return bind_provider_setup_precondition(precondition, identity=identity)


def provider_setup_expected_state_matches_snapshot(
    expected_state: ExpectedProviderSetupState,
    snapshot: AtomicConfigSnapshot,
) -> bool:
    """Compare one secret-free expected state with an authoritative snapshot."""

    if type(expected_state) is not ExpectedProviderSetupState:
        raise ValueError("Provider setup expected state is invalid.")
    if type(snapshot) is not AtomicConfigSnapshot:
        raise ValueError("Provider setup config snapshot is invalid.")
    return expected_state._matches_snapshot(snapshot)


def capture_provider_setup_precondition(
    snapshot: AtomicConfigSnapshot,
    *,
    provider: str,
) -> ProviderSetupConfigPrecondition:
    """Capture one provider's relevant locked config before model selection."""

    if type(snapshot) is not AtomicConfigSnapshot:
        raise ValueError("Provider setup config snapshot is invalid.")
    provider_key = canonical_provider_key(provider)
    route, routing, models, credential = _provider_setup_observations(
        snapshot.values,
        provider_key,
    )
    return ProviderSetupConfigPrecondition(
        config_generation=snapshot.generation,
        provider_key=provider_key,
        configured_connection_identity=route,
        configured_routing_state=routing,
        configured_model_state=models,
        credential_observation=credential,
    )


def bind_provider_setup_precondition(
    precondition: ProviderSetupConfigPrecondition,
    *,
    identity: ProviderSetupWriteIdentity,
) -> ExpectedProviderSetupState:
    """Bind model provenance to an earlier authoritative config observation."""

    if type(precondition) is not ProviderSetupConfigPrecondition:
        raise ValueError("Provider setup config precondition is invalid.")
    if type(identity) is not ProviderSetupWriteIdentity:
        raise ValueError("Provider setup write identity is invalid.")
    if precondition.provider_key != identity.provider_key:
        raise ValueError("Provider setup expected provider changed.")
    return ExpectedProviderSetupState(
        config_generation=precondition.config_generation,
        identity=identity,
        configured_connection_identity=precondition.configured_connection_identity,
        configured_routing_state=precondition.configured_routing_state,
        configured_model_state=precondition.configured_model_state,
        credential_observation=precondition._credential(),
    )


@dataclass(frozen=True, slots=True)
class _ProviderSetupWriteBinding:
    mutation_ref: ReferenceType[ProviderSetupMutation]
    guard: ProviderSetupWriteGuard
    expectation: ProviderSetupWriteExpectation
    expected_state: ExpectedProviderSetupState


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ProviderSetupMutation:
    """One sparse atomic provider/default/provenance configuration mutation."""

    section_values: Mapping[str, Mapping[str, object]]
    delete_keys: Mapping[str, tuple[str, ...]]
    semantic_identity: ProviderDraftIdentity | None

    def __post_init__(self) -> None:
        _validate_provider_setup_mutation(
            self,
            require_issued=False,
            validate_credentials=False,
        )

    def __repr__(self) -> str:
        set_keys = {
            section: tuple(values) for section, values in self.section_values.items()
        }
        return (
            "ProviderSetupMutation("
            f"section_keys={set_keys!r}, delete_keys={dict(self.delete_keys)!r}, "
            f"semantic_identity={self.semantic_identity!r})"
        )


def project_provider_setup_expected_state(
    snapshot: AtomicConfigSnapshot,
    *,
    mutation: ProviderSetupMutation,
    identity: ProviderSetupWriteIdentity,
) -> ExpectedProviderSetupState:
    """Project the exact validated mutation into a secret-free postcondition."""

    if type(snapshot) is not AtomicConfigSnapshot:
        raise ValueError("Provider setup config snapshot is invalid.")
    if type(mutation) is not ProviderSetupMutation:
        raise ValueError("Provider setup mutation is invalid.")
    if type(identity) is not ProviderSetupWriteIdentity:
        raise ValueError("Provider setup write identity is invalid.")
    _validate_provider_setup_mutation(
        mutation,
        require_issued=True,
        validate_credentials=True,
    )

    projected = deepcopy(dict(snapshot.values))
    for section, keys in mutation.delete_keys.items():
        current: object = projected
        default_current: object = DEFAULT_CONFIG_FROM_TOML
        for part in section.split("."):
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(part)
            default_current = (
                default_current.get(part)
                if isinstance(default_current, Mapping)
                else None
            )
        if isinstance(current, dict):
            for key in keys:
                if isinstance(default_current, Mapping) and key in default_current:
                    current[key] = deepcopy(default_current[key])
                else:
                    current.pop(key, None)
    for section, values in mutation.section_values.items():
        current = projected
        for part in section.split("."):
            child = current.get(part)
            if not isinstance(child, dict):
                child = {}
                current[part] = child
            current = child
        current.update(values)

    return capture_expected_provider_setup_state(
        AtomicConfigSnapshot(snapshot.generation, projected),
        identity=identity,
    )


def resolve_remembered_provider_model(
    app_config: object,
    provider: object,
) -> str | None:
    """Resolve a provider's remembered model without cross-provider borrowing."""

    ownership = _ownership_for(provider)
    if not isinstance(app_config, Mapping):
        return None
    chat_defaults = app_config.get("chat_defaults")
    if isinstance(chat_defaults, Mapping):
        try:
            defaults_ownership = _ownership_for(chat_defaults.get("provider"))
        except ValueError:
            defaults_ownership = None
        if (
            defaults_ownership is not None
            and defaults_ownership.provider_key == ownership.provider_key
        ):
            model = _safe_model(chat_defaults.get("model"))
            if model is not None:
                return model

    provider_settings = _provider_settings(app_config, ownership)
    return _safe_model(provider_settings.get(ownership.model_key))


def build_provider_setup_mutation(
    draft: ProviderSetupDraft,
    app_config: object,
) -> ProviderSetupMutation:
    """Build one provider-owned sparse mutation without performing I/O."""

    if type(draft) is not ProviderSetupDraft:
        raise ValueError("Provider setup draft is invalid.")
    if not isinstance(app_config, Mapping):
        app_config = {}
    ownership = _ownership_for(draft.provider)
    model = _safe_model(draft.model)
    if model is None:
        model = ""

    provider_section = f"api_settings.{_configured_section_key(app_config, ownership)}"
    provider_settings = _provider_settings(app_config, ownership)
    endpoint_key = _selected_endpoint_key(provider_settings, ownership)
    provider_values: dict[str, object] = {ownership.model_key: model}
    section_values: dict[str, dict[str, object]] = {
        provider_section: provider_values,
        "chat_defaults": {
            "provider": ownership.provider_key,
            "model": model,
        },
    }
    deletes: dict[str, list[str]] = {}
    semantic_identity: ProviderDraftIdentity | None = None

    endpoint = draft.endpoint.strip()
    if endpoint:
        resolution = resolve_provider_endpoint(ownership.provider_key, endpoint)
        persisted_endpoint = _persisted_endpoint_for_owner(
            ownership.provider_key,
            endpoint_key,
            resolution.persisted_endpoint,
            resolution.chat_url,
        )
        if persisted_endpoint is None:
            raise ValueError("Endpoint is invalid.")
        provider_values[endpoint_key] = persisted_endpoint
        section_values["provider_setup.confirmed"] = {ownership.provider_key: True}
        connection_identity = canonical_connection_identity(
            ownership.provider_key,
            persisted_endpoint,
        )
        if connection_identity is None:
            raise ValueError("Endpoint is invalid.")
    else:
        deletes[provider_section] = [endpoint_key]
        deletes["provider_setup.confirmed"] = [ownership.provider_key]

    stored_key, environment_key = ownership.credential_keys
    credential_source = draft.credential_source
    saved_credential_source: CredentialSource = credential_source
    credential_value = (
        draft.credential_value.strip() if draft.credential_value is not None else ""
    )
    credential_env_var = (
        draft.credential_env_var.strip() if draft.credential_env_var is not None else ""
    )
    if credential_source == "draft":
        if not credential_value:
            raise ValueError("Credential setup is invalid.")
        provider_values[stored_key] = credential_value
        deletes.setdefault(provider_section, []).append(environment_key)
        saved_credential_source = "stored"
    elif credential_source == "stored":
        stored_value = credential_value or _existing_credential_value(
            provider_settings.get(stored_key)
        )
        if not stored_value:
            raise ValueError("Credential setup is invalid.")
        provider_values[stored_key] = stored_value
        deletes.setdefault(provider_section, []).append(environment_key)
    elif credential_source == "environment":
        env_var = credential_env_var or _existing_environment_name(
            provider_settings.get(environment_key)
        )
        if not env_var:
            raise ValueError("Credential setup is invalid.")
        provider_values[environment_key] = env_var
        deletes.setdefault(provider_section, []).append(stored_key)
    else:
        deletes.setdefault(provider_section, []).extend((stored_key, environment_key))
    provider_values["credential_source"] = saved_credential_source

    if endpoint:
        semantic_identity = ProviderDraftIdentity(
            provider_key=ownership.provider_key,
            connection_identity=connection_identity,
            credential_source=saved_credential_source,
            credential_revision=draft.credential_revision,
            draft_generation=draft.draft_generation,
        )

    frozen_sections = MappingProxyType(
        {
            section: MappingProxyType(dict(values))
            for section, values in section_values.items()
            if values
        }
    )
    frozen_deletes = MappingProxyType(
        {
            section: tuple(dict.fromkeys(keys))
            for section, keys in deletes.items()
            if keys
        }
    )
    mutation = ProviderSetupMutation(
        section_values=frozen_sections,
        delete_keys=frozen_deletes,
        semantic_identity=semantic_identity,
    )
    _mark_provider_setup_mutation_issued(mutation)
    return mutation


def provider_setup_draft_identity(
    draft: ProviderSetupDraft,
    app_config: object,
) -> ProviderDraftIdentity | None:
    """Return the exact tested-draft identity before save-source rebasing."""

    mutation = build_provider_setup_mutation(draft, app_config)
    saved_identity = mutation.semantic_identity
    if saved_identity is None:
        return None
    return ProviderDraftIdentity(
        provider_key=saved_identity.provider_key,
        connection_identity=saved_identity.connection_identity,
        credential_source=draft.credential_source,
        credential_revision=draft.credential_revision,
        draft_generation=draft.draft_generation,
    )


def persist_provider_setup(mutation: ProviderSetupMutation) -> ConfigMutationResult:
    """Persist one setup through the shared atomic config mutation owner."""

    if type(mutation) is not ProviderSetupMutation:
        raise ValueError("Provider setup mutation is invalid.")
    _validate_provider_setup_mutation(
        mutation,
        require_issued=True,
        validate_credentials=True,
    )
    chat_defaults = mutation.section_values.get("chat_defaults", {})
    binding = _provider_setup_write_binding(mutation)

    def persist(
        mutation_precondition: Callable[[], bool] | None = None,
        locked_snapshot_precondition: Callable[[AtomicConfigSnapshot], bool]
        | None = None,
    ) -> ConfigMutationResult:
        return persist_provider_settings_atomic(
            mutation,
            provider=chat_defaults.get("provider"),
            model=chat_defaults.get("model"),
            section_values=mutation.section_values,
            delete_keys=mutation.delete_keys,
            mutation_precondition=mutation_precondition,
            locked_snapshot_precondition=locked_snapshot_precondition,
        )

    if binding is None:
        return persist()
    with binding.guard.hold():
        return persist(
            locked_snapshot_precondition=lambda snapshot: (
                binding.guard.matches(binding.expectation)
                and binding.expected_state.identity == binding.expectation.identity
                and binding.expected_state._matches_snapshot(snapshot)
            )
        )


def bind_provider_setup_write_expectation(
    mutation: ProviderSetupMutation,
    *,
    guard: ProviderSetupWriteGuard,
    expectation: ProviderSetupWriteExpectation,
    expected_state: ExpectedProviderSetupState,
) -> None:
    """Bind a first-run CAS expectation without changing the writer call API."""

    _validate_provider_setup_mutation(
        mutation,
        require_issued=True,
        validate_credentials=True,
    )
    if (
        type(guard) is not ProviderSetupWriteGuard
        or type(expectation) is not ProviderSetupWriteExpectation
        or type(expected_state) is not ExpectedProviderSetupState
        or expected_state.identity != expectation.identity
    ):
        raise ValueError("Provider setup write expectation is invalid.")
    mutation_id = id(mutation)
    with _ISSUED_MUTATION_LOCK:
        reference = _ISSUED_MUTATIONS.get(mutation_id)
        if reference is None or reference() is not mutation:
            raise ValueError("Provider setup mutation is invalid.")
        _WRITE_EXPECTATIONS[mutation_id] = _ProviderSetupWriteBinding(
            reference,
            guard,
            expectation,
            expected_state,
        )


def _provider_setup_write_binding(
    mutation: ProviderSetupMutation,
) -> _ProviderSetupWriteBinding | None:
    with _ISSUED_MUTATION_LOCK:
        binding = _WRITE_EXPECTATIONS.get(id(mutation))
        if binding is None or binding.mutation_ref() is not mutation:
            return None
        return binding


def persist_provider_settings_atomic(
    setup_mutation: ProviderSetupMutation | None,
    *,
    provider: object,
    model: object,
    section_values: Mapping[str, Mapping[str, object]],
    delete_keys: Mapping[str, tuple[str, ...]],
    mutation_precondition: Callable[[], bool] | None = None,
    locked_snapshot_precondition: Callable[[AtomicConfigSnapshot], bool] | None = None,
) -> ConfigMutationResult:
    """Validate and persist one combined provider Settings mutation."""

    ownership = _ownership_for(provider)
    if type(model) is not str or (model and _safe_model(model) != model):
        raise ValueError("Provider settings model is invalid.")
    _validate_combined_provider_settings_mutation(
        setup_mutation,
        ownership,
        model,
        section_values,
        delete_keys,
    )
    try:
        if mutation_precondition is None and locked_snapshot_precondition is None:
            result = apply_settings_mutation_to_cli_config(
                section_values,
                delete_keys=delete_keys,
            )
        elif locked_snapshot_precondition is None:
            result = apply_settings_mutation_to_cli_config(
                section_values,
                delete_keys=delete_keys,
                mutation_precondition=mutation_precondition,
            )
        elif mutation_precondition is None:
            result = apply_settings_mutation_to_cli_config(
                section_values,
                delete_keys=delete_keys,
                locked_snapshot_precondition=locked_snapshot_precondition,
            )
        else:
            result = apply_settings_mutation_to_cli_config(
                section_values,
                delete_keys=delete_keys,
                mutation_precondition=mutation_precondition,
                locked_snapshot_precondition=locked_snapshot_precondition,
            )
    except Exception:  # noqa: BLE001 - persistence must fail closed on writer errors.
        return ConfigMutationResult(False, False, "before_replace")
    if type(result) is not ConfigMutationResult:
        return ConfigMutationResult(False, False, "before_replace")
    return result


def provider_setup_is_explicitly_configured(
    app_config: object,
    provider: object,
) -> bool:
    """Read setup provenance, with a bounded legacy-config compatibility path."""

    ownership = _ownership_for(provider)
    if not isinstance(app_config, Mapping):
        return False
    missing = object()
    provider_setup = app_config.get("provider_setup", missing)
    if provider_setup is not missing:
        if not isinstance(provider_setup, Mapping):
            return False
        confirmed = provider_setup.get("confirmed", missing)
        if confirmed is missing or not isinstance(confirmed, Mapping):
            return False
        value = confirmed.get(ownership.provider_key, False)
        return value if type(value) is bool else False

    current_settings = _provider_settings(app_config, ownership)
    template_settings = _provider_settings(DEFAULT_CONFIG_FROM_TOML, ownership)
    owned_keys = (
        *_ENDPOINT_KEY_PRECEDENCE,
        ownership.model_key,
        *ownership.credential_keys,
    )
    for key in owned_keys:
        current_value = current_settings.get(key)
        if _safe_nonempty_scalar(
            current_value
        ) and current_value != template_settings.get(key):
            return True

    current_defaults = app_config.get("chat_defaults")
    template_defaults = DEFAULT_CONFIG_FROM_TOML.get("chat_defaults", {})
    if isinstance(current_defaults, Mapping):
        try:
            defaults_owner = _ownership_for(current_defaults.get("provider"))
        except ValueError:
            defaults_owner = None
        current_model = _safe_model(current_defaults.get("model"))
        if (
            defaults_owner is not None
            and defaults_owner.provider_key == ownership.provider_key
            and current_model is not None
            and (
                current_defaults.get("provider") != template_defaults.get("provider")
                or current_model != template_defaults.get("model")
            )
        ):
            return True
    return False


def _provider_settings(
    app_config: Mapping[object, object],
    ownership: _ProviderOwnership,
) -> Mapping[str, object]:
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return {}
    direct = api_settings.get(ownership.config_section)
    if isinstance(direct, Mapping):
        return direct
    for index, (configured_provider, settings) in enumerate(api_settings.items()):
        if index >= _MAX_CONFIG_PROVIDERS:
            break
        try:
            configured_ownership = _ownership_for(configured_provider)
        except ValueError:
            continue
        if configured_ownership.provider_key != ownership.provider_key:
            continue
        return settings if isinstance(settings, Mapping) else {}
    return {}


def _provider_setup_observations(
    app_config: Mapping[object, object],
    provider: object,
) -> tuple[
    tuple[str, str] | None,
    tuple[tuple[str, str], ...],
    tuple[str | None, str | None, str | None],
    _ProviderCredentialObservation,
]:
    """Derive the non-secret provider CAS state from one config snapshot."""

    from .console_provider_endpoints import effective_provider_discovery_endpoint
    from .provider_readiness import resolve_provider_credential

    if not isinstance(app_config, Mapping):
        raise TypeError("Provider setup config snapshot is invalid.")
    ownership = _ownership_for(provider)
    provider_settings = _provider_settings(app_config, ownership)
    endpoint = effective_provider_discovery_endpoint(
        ownership.provider_key,
        None,
        provider_settings,
    )
    route_identity = None
    if endpoint:
        route_identity = canonical_connection_identity(
            ownership.provider_key,
            endpoint,
        )
        if route_identity is None:
            raise ValueError("Provider setup configured route is invalid.")
    routing_state = tuple(
        (key, _safe_routing_setting(provider_settings.get(key)))
        for key in _ROUTING_SETTING_KEYS
        if key in provider_settings
    )

    provider_model = _safe_model(provider_settings.get(ownership.model_key))
    chat_provider = None
    chat_model = None
    chat_defaults = app_config.get("chat_defaults")
    if isinstance(chat_defaults, Mapping):
        try:
            chat_provider = _ownership_for(chat_defaults.get("provider")).provider_key
        except ValueError:
            chat_provider = None
        chat_model = _safe_model(chat_defaults.get("model"))

    credential_value, credential_source, env_var = resolve_provider_credential(
        ownership.provider_key,
        provider_settings,
        environ=os.environ,
    )
    configured_source_value = provider_settings.get("credential_source")
    configured_source = (
        configured_source_value.strip().lower()
        if type(configured_source_value) is str
        and configured_source_value.strip().lower()
        in {"none", "stored", "environment"}
        else "none"
    )
    _, environment_key = ownership.credential_keys
    configured_env_var = _existing_environment_name(
        provider_settings.get(environment_key)
    )
    credential_routing = (
        configured_source
        if configured_source == "stored"
        else f"{configured_source}\0{configured_env_var or ''}"
    )
    if (
        credential_value is not None
        and credential_source is not None
        and credential_source.startswith("config:")
    ):
        credential = _ProviderCredentialObservation(
            "stored",
            f"stored\0{credential_routing}\0{credential_value}",
        )
    elif (
        credential_value is not None
        and credential_source is not None
        and credential_source.startswith("env:")
    ):
        credential = _ProviderCredentialObservation(
            "environment",
            f"environment\0{credential_routing}\0{env_var or ''}\0{credential_value}",
        )
    else:
        credential = _ProviderCredentialObservation(
            "none",
            f"none\0{credential_routing}",
        )
    return (
        route_identity,
        routing_state,
        (provider_model, chat_provider, chat_model),
        credential,
    )


def _safe_routing_setting(value: object) -> str:
    """Normalize one bounded non-secret routing value for CAS comparison."""

    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is not str:
        raise ValueError("Provider setup routing setting is invalid.")
    normalized = value.strip()
    if (
        len(normalized) > 2048
        or not normalized.isprintable()
        or any(
            unicode_category(character) in _UNSAFE_TEXT_CATEGORIES
            for character in normalized
        )
    ):
        raise ValueError("Provider setup routing setting is invalid.")
    return normalized


def _selected_endpoint_key(
    provider_settings: Mapping[str, object],
    ownership: _ProviderOwnership,
) -> str:
    for key in _ENDPOINT_KEY_PRECEDENCE:
        if key in provider_settings:
            return key
    return ownership.endpoint_key


def _persisted_endpoint_for_owner(
    provider_key: str,
    endpoint_key: str,
    root_endpoint: str | None,
    chat_url: str | None,
) -> str | None:
    if provider_key in _ROOT_ENDPOINT_PROVIDER_KEYS:
        return root_endpoint
    if endpoint_key in _API_BASE_ENDPOINT_KEYS:
        return _api_base_from_chat_url(chat_url)
    return chat_url


def _api_base_from_chat_url(chat_url: str | None) -> str | None:
    if type(chat_url) is not str:
        return None
    try:
        parsed = urlsplit(chat_url)
    except ValueError:
        return None
    segments = parsed.path.split("/")
    if len(segments) < 3 or segments[-2:] != ["chat", "completions"]:
        return None
    base_path = "/".join(segments[:-2]) or "/"
    return urlunsplit((parsed.scheme, parsed.netloc, base_path, "", ""))


def _existing_credential_value(value: object) -> str | None:
    if type(value) is not str:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        _validate_credential_value(candidate)
    except ValueError:
        return None
    return candidate


def _existing_environment_name(value: object) -> str | None:
    if type(value) is not str:
        return None
    candidate = value.strip()
    if not candidate or _ENV_VAR_PATTERN.fullmatch(candidate) is None:
        return None
    return candidate


def _provider_draft_identity_is_valid(identity: object) -> bool:
    if type(identity) is not ProviderDraftIdentity:
        return False
    try:
        provider_key = identity.provider_key
        connection_identity = identity.connection_identity
        credential_source = identity.credential_source
        credential_revision = identity.credential_revision
        draft_generation = identity.draft_generation
        rebuilt = ProviderDraftIdentity(
            provider_key=provider_key,
            connection_identity=connection_identity,
            credential_source=credential_source,
            credential_revision=credential_revision,
            draft_generation=draft_generation,
        )
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        rebuilt == identity
        and type(credential_revision) is int
        and 0 <= credential_revision <= _MAX_IDENTITY_COUNTER
        and type(draft_generation) is int
        and 0 <= draft_generation <= _MAX_IDENTITY_COUNTER
    )


def _validate_combined_provider_settings_mutation(
    setup_mutation: ProviderSetupMutation | None,
    ownership: _ProviderOwnership,
    model: str,
    section_values: object,
    delete_keys: object,
) -> None:
    error = ValueError("Combined provider settings mutation is invalid.")
    connection_error = ValueError(
        "Combined provider settings connection mutation requires validated setup."
    )
    if not isinstance(section_values, Mapping) or not isinstance(delete_keys, Mapping):
        raise error
    if (
        not section_values
        and not delete_keys
        or len(section_values) > _MAX_COMBINED_MUTATION_SECTIONS
    ):
        raise error
    if len(delete_keys) > _MAX_COMBINED_MUTATION_SECTIONS:
        raise error

    total_keys = 0
    for section, values in section_values.items():
        if (
            type(section) is not str
            or _SECTION_NAME_PATTERN.fullmatch(section) is None
            or not isinstance(values, Mapping)
            or not values
        ):
            raise error
        total_keys += len(values)
        if section == "model_capabilities.models":
            invalid_keys = any(_safe_model(key) != key for key in values)
        else:
            invalid_keys = any(
                type(key) is not str or _CONFIG_KEY_PATTERN.fullmatch(key) is None
                for key in values
            )
        if invalid_keys:
            raise error
    for section, keys in delete_keys.items():
        if (
            type(section) is not str
            or _SECTION_NAME_PATTERN.fullmatch(section) is None
            or type(keys) is not tuple
            or not keys
            or len(keys) != len(set(keys))
        ):
            raise error
        total_keys += len(keys)
        if section == "model_capabilities.models":
            invalid_keys = any(_safe_model(key) != key for key in keys)
        else:
            invalid_keys = any(
                type(key) is not str or _CONFIG_KEY_PATTERN.fullmatch(key) is None
                for key in keys
            )
        if invalid_keys:
            raise error
        values = section_values.get(section, {})
        if any(key in values for key in keys):
            raise ValueError(
                "Combined provider settings mutation has overlapping keys."
            )
    if total_keys > _MAX_COMBINED_MUTATION_KEYS:
        raise error

    setup_set_keys: dict[str, frozenset[str]] = {}
    setup_delete_keys: dict[str, frozenset[str]] = {}
    if setup_mutation is not None:
        _validate_provider_setup_mutation(
            setup_mutation,
            require_issued=True,
            validate_credentials=True,
        )
        setup_defaults = setup_mutation.section_values["chat_defaults"]
        if (
            setup_defaults.get("provider") != ownership.provider_key
            or setup_defaults.get("model") != model
        ):
            raise error
        for section, values in setup_mutation.section_values.items():
            combined_values = section_values.get(section)
            if not isinstance(combined_values, Mapping) or any(
                key not in combined_values or combined_values[key] != value
                for key, value in values.items()
            ):
                raise error
            setup_set_keys[section] = frozenset(values)
        for section, keys in setup_mutation.delete_keys.items():
            combined_deletes = delete_keys.get(section, ())
            if any(key not in combined_deletes for key in keys):
                raise error
            setup_delete_keys[section] = frozenset(keys)

    for section, values in section_values.items():
        extra_keys = set(values) - setup_set_keys.get(section, frozenset())
        if section.startswith("api_settings."):
            try:
                section_owner = _ownership_for(section.removeprefix("api_settings."))
            except ValueError as exc:
                raise error from exc
            if section_owner.provider_key != ownership.provider_key:
                raise error
            allowed_extra_keys = {"model_defaults"}
            if ownership.provider_key == "qwencloud":
                allowed_extra_keys.add("api_mode")
            if not extra_keys.issubset(allowed_extra_keys):
                if setup_mutation is None:
                    raise connection_error
                raise error
            if "model_defaults" in extra_keys and not isinstance(
                values["model_defaults"], Mapping
            ):
                raise error
            if "api_mode" in extra_keys and values["api_mode"] not in {
                "responses",
                "chat_completions",
            }:
                raise error
        elif section == "model_capabilities.models":
            if (
                not model
                or extra_keys != {model}
                or not isinstance(values[model], Mapping)
            ):
                raise error
        elif extra_keys:
            if setup_mutation is None and section in {
                "chat_defaults",
                "provider_setup.confirmed",
            }:
                raise connection_error
            raise error

    for section, keys in delete_keys.items():
        extra_keys = set(keys) - setup_delete_keys.get(section, frozenset())
        if not extra_keys:
            continue
        if section.startswith("api_settings."):
            try:
                section_owner = _ownership_for(section.removeprefix("api_settings."))
            except ValueError as exc:
                raise error from exc
            if (
                section_owner.provider_key != ownership.provider_key
                or ownership.provider_key != "qwencloud"
                or extra_keys != {"api_mode"}
            ):
                if setup_mutation is None:
                    raise connection_error
                raise error
        elif section == "model_capabilities.models":
            if not model or extra_keys != {model}:
                raise error
        else:
            if setup_mutation is None:
                raise connection_error
            raise error


def _mark_provider_setup_mutation_issued(mutation: ProviderSetupMutation) -> None:
    mutation_id = id(mutation)

    def discard(reference: ReferenceType[ProviderSetupMutation]) -> None:
        with _ISSUED_MUTATION_LOCK:
            if _ISSUED_MUTATIONS.get(mutation_id) is reference:
                _ISSUED_MUTATIONS.pop(mutation_id, None)
                _WRITE_EXPECTATIONS.pop(mutation_id, None)

    reference = ref(mutation, discard)
    with _ISSUED_MUTATION_LOCK:
        _ISSUED_MUTATIONS[mutation_id] = reference


def _provider_setup_mutation_is_issued(mutation: ProviderSetupMutation) -> bool:
    with _ISSUED_MUTATION_LOCK:
        reference = _ISSUED_MUTATIONS.get(id(mutation))
        return reference is not None and reference() is mutation


def _validate_provider_setup_mutation(
    mutation: object,
    *,
    require_issued: bool,
    validate_credentials: bool,
) -> None:
    error = ValueError("Provider setup mutation is invalid.")
    overlap_error = ValueError("Provider setup mutation has overlapping keys.")
    ownership_error = ValueError("Provider setup mutation ownership is invalid.")
    identity_error = ValueError("Provider setup mutation identity is invalid.")
    if type(mutation) is not ProviderSetupMutation:
        raise error
    try:
        section_values = mutation.section_values
        delete_keys = mutation.delete_keys
        semantic_identity = mutation.semantic_identity
    except Exception as exc:
        raise error from exc
    if require_issued and not _provider_setup_mutation_is_issued(mutation):
        raise error
    if (
        type(section_values) is not _MAPPING_PROXY_TYPE
        or type(delete_keys) is not _MAPPING_PROXY_TYPE
    ):
        raise error
    if not 1 <= len(section_values) <= _MAX_MUTATION_SECTIONS:
        raise error
    if len(delete_keys) > _MAX_MUTATION_SECTIONS:
        raise error

    total_keys = 0
    for section, values in section_values.items():
        if (
            type(section) is not str
            or _SECTION_NAME_PATTERN.fullmatch(section) is None
            or type(values) is not _MAPPING_PROXY_TYPE
            or not values
        ):
            raise error
        total_keys += len(values)
        for key, value in values.items():
            if (
                type(key) is not str
                or _CONFIG_KEY_PATTERN.fullmatch(key) is None
                or type(value) not in {str, bool}
            ):
                raise error
    for section, keys in delete_keys.items():
        if (
            type(section) is not str
            or _SECTION_NAME_PATTERN.fullmatch(section) is None
            or type(keys) is not tuple
            or not keys
            or len(keys) != len(set(keys))
        ):
            raise error
        total_keys += len(keys)
        if any(
            type(key) is not str or _CONFIG_KEY_PATTERN.fullmatch(key) is None
            for key in keys
        ):
            raise error
        set_keys = section_values.get(section, {})
        if any(key in set_keys for key in keys):
            raise overlap_error
    if total_keys > _MAX_MUTATION_KEYS:
        raise error

    chat_defaults = section_values.get("chat_defaults")
    if type(chat_defaults) is not _MAPPING_PROXY_TYPE or set(chat_defaults) != {
        "provider",
        "model",
    }:
        raise error
    provider = chat_defaults.get("provider")
    model = chat_defaults.get("model")
    if type(provider) is not str or type(model) is not str:
        raise error
    try:
        ownership = _ownership_for(provider)
    except ValueError as exc:
        raise error from exc
    if model and _safe_model(model) != model:
        raise error

    provider_sections = {
        section
        for section in (*section_values, *delete_keys)
        if section.startswith("api_settings.")
    }
    if len(provider_sections) != 1:
        raise error
    provider_section = next(iter(provider_sections))
    try:
        section_ownership = _ownership_for(
            provider_section.removeprefix("api_settings.")
        )
    except ValueError as exc:
        raise error from exc
    if section_ownership.provider_key != ownership.provider_key:
        raise error

    allowed_sections = {
        provider_section,
        "chat_defaults",
        "provider_setup.confirmed",
    }
    if not set(section_values).issubset(allowed_sections) or not set(
        delete_keys
    ).issubset({provider_section, "provider_setup.confirmed"}):
        raise ownership_error
    provider_values = section_values.get(provider_section)
    if type(provider_values) is not _MAPPING_PROXY_TYPE:
        raise error
    allowed_provider_keys = {
        ownership.model_key,
        "credential_source",
        *ownership.credential_keys,
        *_ENDPOINT_KEY_PRECEDENCE,
    }
    if not set(provider_values).issubset(allowed_provider_keys):
        raise error
    if provider_values.get(ownership.model_key) != model:
        raise error
    configured_source = provider_values.get("credential_source")
    if configured_source not in {"none", "stored", "environment"}:
        raise error
    provider_deletes = delete_keys.get(provider_section, ())
    if not set(provider_deletes).issubset(
        {*ownership.credential_keys, *_ENDPOINT_KEY_PRECEDENCE}
    ):
        raise error

    set_endpoint_keys = [
        key for key in _ENDPOINT_KEY_PRECEDENCE if key in provider_values
    ]
    deleted_endpoint_keys = [
        key for key in _ENDPOINT_KEY_PRECEDENCE if key in provider_deletes
    ]
    if len(set_endpoint_keys) + len(deleted_endpoint_keys) != 1:
        raise error

    stored_key, environment_key = ownership.credential_keys
    stored_is_set = stored_key in provider_values
    environment_is_set = environment_key in provider_values
    stored_is_deleted = stored_key in provider_deletes
    environment_is_deleted = environment_key in provider_deletes
    if validate_credentials and stored_is_set:
        stored_value = provider_values.get(stored_key)
        if (
            type(stored_value) is not str
            or not stored_value
            or stored_value.strip() != stored_value
        ):
            raise error
        try:
            _validate_credential_value(stored_value)
        except ValueError as exc:
            raise error from exc
    if validate_credentials and environment_is_set:
        environment_name = provider_values.get(environment_key)
        if (
            type(environment_name) is not str
            or _existing_environment_name(environment_name) != environment_name
        ):
            raise error
    if stored_is_set and environment_is_deleted and not environment_is_set:
        desired_source: CredentialSource = "stored"
    elif environment_is_set and stored_is_deleted and not stored_is_set:
        desired_source = "environment"
    elif (
        stored_is_deleted
        and environment_is_deleted
        and not stored_is_set
        and not environment_is_set
    ):
        desired_source = "none"
    else:
        raise error
    if configured_source != desired_source:
        raise error

    if set_endpoint_keys:
        confirmation = section_values.get("provider_setup.confirmed")
        if (
            type(confirmation) is not _MAPPING_PROXY_TYPE
            or dict(confirmation) != {ownership.provider_key: True}
            or "provider_setup.confirmed" in delete_keys
        ):
            raise error
        if type(semantic_identity) is not ProviderDraftIdentity:
            raise identity_error
        if not _provider_draft_identity_is_valid(semantic_identity):
            raise identity_error
        endpoint_value = provider_values[set_endpoint_keys[0]]
        if type(endpoint_value) is not str:
            raise error
        connection_identity = canonical_connection_identity(
            ownership.provider_key, endpoint_value
        )
        if (
            connection_identity is None
            or semantic_identity.provider_key != ownership.provider_key
            or semantic_identity.connection_identity != connection_identity
            or semantic_identity.credential_source != desired_source
        ):
            raise identity_error
    else:
        if "provider_setup.confirmed" in section_values or delete_keys.get(
            "provider_setup.confirmed"
        ) != (ownership.provider_key,):
            raise error
        if semantic_identity is not None:
            raise identity_error


def _configured_section_key(
    app_config: Mapping[object, object],
    ownership: _ProviderOwnership,
) -> str:
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return ownership.config_section
    if ownership.config_section in api_settings:
        return ownership.config_section
    for index, configured_provider in enumerate(api_settings):
        if index >= _MAX_CONFIG_PROVIDERS:
            raise ValueError("Provider settings alias scan limit was exceeded.")
        try:
            configured_ownership = _ownership_for(configured_provider)
        except ValueError:
            continue
        if configured_ownership.provider_key == ownership.provider_key:
            return str(configured_provider)
    return ownership.config_section


def _safe_model(value: object) -> str | None:
    if type(value) is not str:
        return None
    model = value.strip()
    if (
        not model
        or len(model) > _MAX_MODEL_CHARS
        or not model.isprintable()
        or any(
            unicode_category(character) in _UNSAFE_TEXT_CATEGORIES
            for character in model
        )
    ):
        return None
    return model


def _validate_credential_value(value: str) -> None:
    stripped = value.strip()
    if (
        (stripped and not is_valid_provider_api_key(stripped))
        or len(value) > _MAX_SECRET_CHARS
        or any(
            unicode_category(character) in _UNSAFE_TEXT_CATEGORIES
            for character in value
        )
    ):
        raise ValueError("Credential value is invalid.")


def _safe_nonempty_scalar(value: object) -> bool:
    return type(value) in {str, int, float, bool} and bool(value)
