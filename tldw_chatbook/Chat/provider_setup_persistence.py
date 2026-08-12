"""Provider-owned setup persistence contracts and atomic mutation helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from unicodedata import category as unicode_category

from ..config import (
    DEFAULT_CONFIG_FROM_TOML,
    ConfigMutationResult,
    apply_settings_mutation_to_cli_config,
)
from .provider_endpoint_contract import (
    canonical_connection_identity,
    resolve_provider_endpoint,
)
from .provider_test_evidence import CredentialSource, ProviderDraftIdentity

_MAX_MODEL_CHARS = 120
_MAX_SECRET_CHARS = 8192
_MAX_CONFIG_PROVIDERS = 256
_UNSAFE_TEXT_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_ENV_VAR_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")


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
    "Mistral": "mistralai",
    "mistral": "mistralai",
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
        if type(self.credential_revision) is not int or self.credential_revision < 0:
            raise ValueError("Credential revision is invalid.")
        if type(self.draft_generation) is not int or self.draft_generation < 0:
            raise ValueError("Draft generation is invalid.")
        if self.credential_value is not None:
            _validate_credential_value(self.credential_value)
        if (
            self.credential_env_var is not None
            and self.credential_env_var
            and _ENV_VAR_PATTERN.fullmatch(self.credential_env_var) is None
        ):
            raise ValueError("Credential environment variable is invalid.")


@dataclass(frozen=True, slots=True)
class ProviderSetupMutation:
    """One sparse atomic provider/default/provenance configuration mutation."""

    section_values: Mapping[str, Mapping[str, object]]
    delete_keys: Mapping[str, tuple[str, ...]]
    semantic_identity: ProviderDraftIdentity | None

    def __repr__(self) -> str:
        set_keys = {
            section: tuple(values)
            for section, values in self.section_values.items()
        }
        return (
            "ProviderSetupMutation("
            f"section_keys={set_keys!r}, delete_keys={dict(self.delete_keys)!r}, "
            f"semantic_identity={self.semantic_identity!r})"
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

    provider_section = (
        f"api_settings.{_configured_section_key(app_config, ownership)}"
    )
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
        if resolution.persisted_endpoint is None:
            raise ValueError("Endpoint is invalid.")
        provider_values[ownership.endpoint_key] = resolution.persisted_endpoint
        section_values["provider_setup.confirmed"] = {
            ownership.provider_key: True
        }
        connection_identity = canonical_connection_identity(
            ownership.provider_key,
            resolution.persisted_endpoint,
        )
        if connection_identity is None:
            raise ValueError("Endpoint is invalid.")
        semantic_identity = ProviderDraftIdentity(
            provider_key=ownership.provider_key,
            connection_identity=connection_identity,
            credential_source=draft.credential_source,
            credential_revision=draft.credential_revision,
            draft_generation=draft.draft_generation,
        )
    else:
        deletes[provider_section] = [ownership.endpoint_key]
        deletes["provider_setup.confirmed"] = [ownership.provider_key]

    stored_key, environment_key = ownership.credential_keys
    if draft.credential_value is not None:
        credential_value = draft.credential_value.strip()
        if credential_value:
            provider_values[stored_key] = credential_value
        else:
            deletes.setdefault(provider_section, []).append(stored_key)
    if draft.credential_env_var is not None:
        env_var = draft.credential_env_var.strip()
        if env_var:
            provider_values[environment_key] = env_var
        else:
            deletes.setdefault(provider_section, []).append(environment_key)

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
    return ProviderSetupMutation(
        section_values=frozen_sections,
        delete_keys=frozen_deletes,
        semantic_identity=semantic_identity,
    )


def persist_provider_setup(mutation: ProviderSetupMutation) -> ConfigMutationResult:
    """Persist one setup through the shared atomic config mutation owner."""

    if type(mutation) is not ProviderSetupMutation:
        raise ValueError("Provider setup mutation is invalid.")
    return apply_settings_mutation_to_cli_config(
        mutation.section_values,
        delete_keys=mutation.delete_keys,
    )


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
        ownership.endpoint_key,
        ownership.model_key,
        *ownership.credential_keys,
    )
    for key in owned_keys:
        current_value = current_settings.get(key)
        if _safe_nonempty_scalar(current_value) and current_value != template_settings.get(
            key
        ):
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
            break
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
    if len(value) > _MAX_SECRET_CHARS or any(
        unicode_category(character) in _UNSAFE_TEXT_CATEGORIES for character in value
    ):
        raise ValueError("Credential value is invalid.")


def _safe_nonempty_scalar(value: object) -> bool:
    return type(value) in {str, int, float, bool} and bool(value)
