"""Side-effect-free provider readiness helpers for Chat."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

# "Valid provider API key" has exactly ONE definition in this codebase --
# PR-T2 Task 7 -- shared with `config.py`'s `_normalize_legacy_provider_
# api_key` (the credential bridge that keeps this module's readiness check
# and the actual spend path from disagreeing about the same config).
#
# The definition lives in `config.py`, NOT here, even though this module
# used it first: `config` is the dependency root nearly every other module
# in this app (including this one) already imports directly, so `config`
# importing FROM `Chat.provider_readiness` inverted the natural layering.
# A first attempt at sharing this check did exactly that (a function-local
# import inside `config.py`) and it reproduced a real cycle -- `config` ->
# `Chat/__init__` -> `server_chat_conversation_service` -> `runtime_policy.
# bootstrap` -> back into `config` -- that broke standalone collection of
# `Tests/RuntimePolicy/` (hidden in a full-suite run only by alphabetical
# import ordering happening to load `config` cleanly first). Importing
# `config` from here instead is the direction every other Chat submodule
# already takes, and cannot cycle back: `config.py`'s own top-level
# imports (`DB.*`, `Utils.*`) never reach into `Chat`.
from ..config import (
    is_valid_provider_api_key,
    normalize_provider_config_key,
    provider_settings_for_key,
    resolve_provider_api_key as _valid_api_key,
)


PROVIDERS_REQUIRING_API_KEY_KEYS = frozenset(
    {
        "anthropic",
        "cohere",
        "deepseek",
        "google",
        "groq",
        "huggingface",
        "mistral",
        "mistralai",
        "moonshot",
        "openai",
        "openrouter",
        "qwencloud",
        "zai",
    }
)
#: Providers that dispatch without a credential. Membership is what makes a
#: provider dispatchable at all through the readiness gate (`KNOWN_PROVIDER_
#: KEYS` below), so it must be kept in terms of what `Chat/Chat_Functions.
#: py`'s `API_CALL_HANDLERS` can actually dispatch -- normalized through
#: `provider_config_key`, since that is the only form this module ever sees.
#:
#: `custom_openai_api`, `custom_openai_api_2` and `mlx_lm` are here for
#: exactly that reason (PR-T2 review round 3, finding I2): the dispatch
#: table's own keys are `"custom-openai-api"`, `"custom-openai-api-2"` and
#: `"mlx_lm"` -- verbatim spellings a self-hoster puts in `default_api_
#: endpoint` and which DO dispatch -- but `provider_config_key` normalizes
#: the hyphens to `custom_openai_api`/`custom_openai_api_2`, neither of
#: which is the same key as the `custom`/`custom_2` entries above (those
#: are the `[api_settings.custom]` tables, a different provider row). Left
#: out, all three fell to the "Unknown provider" branch below and a
#: previously-working Run button went permanently disabled with copy that
#: named no remedy.
KEYLESS_PROVIDER_KEYS = frozenset(
    {
        "aphrodite",
        "custom",
        "custom_2",
        "custom_openai_api",
        "custom_openai_api_2",
        "koboldcpp",
        "mlx_lm",
        "llama_cpp",
        "local_llm",
        "local_llamacpp",
        "local_llamafile",
        "local_mlx_lm",
        "local_ollama",
        "local_onnx",
        "local_transformers",
        "local_vllm",
        "ollama",
        "oobabooga",
        "tabbyapi",
        "vllm",
    }
)
KNOWN_PROVIDER_KEYS = PROVIDERS_REQUIRING_API_KEY_KEYS | KEYLESS_PROVIDER_KEYS

_DEFAULT_API_KEY_ENV_VAR_ALIASES = {
    "mistralai": "MISTRAL_API_KEY",
    "qwencloud": "DASHSCOPE_API_KEY",
}


@dataclass(frozen=True)
class ProviderReadiness:
    """Current readiness state for the selected Chat provider."""

    provider: str
    provider_key: str
    requires_api_key: bool
    ready: bool
    api_key: Optional[str]
    api_key_source: Optional[str]
    env_var: Optional[str]
    reason: str
    recovery: Optional[str]

    @property
    def user_message(self) -> str:
        """User-facing readiness text that never includes secret values."""
        if self.ready:
            if self.requires_api_key:
                source = self.api_key_source or "configured credentials"
                return f"{self.provider} is ready. API key found via {source}."
            return f"{self.provider} is ready. No API key is required."

        if self.recovery:
            return f"{self.provider} is not ready: {self.reason}. {self.recovery}"
        return f"{self.provider} is not ready: {self.reason}."


def provider_config_key(provider: Optional[str]) -> str:
    """Return the normalized key used under ``api_settings``."""
    return normalize_provider_config_key(provider)


def _requires_api_key(provider_key: str) -> bool:
    """Return True unless the provider is known to work without credentials."""
    return provider_key not in KEYLESS_PROVIDER_KEYS


def default_api_key_env_var(provider_key: str) -> Optional[str]:
    """Return the conventional environment variable for known keyed providers.

    Single source of truth for the ``<PROVIDER>_API_KEY`` naming convention
    (plus known aliases such as ``mistralai`` -> ``MISTRAL_API_KEY``); also
    consumed by ``first_run_setup_state.read_provider_secret_presence`` so the
    wizard's "found in your environment" detection agrees with Chat's own
    readiness resolution even before ``api_key_env_var`` is explicitly
    persisted to config.

    Args:
        provider_key: The normalized provider key (e.g. via
            ``provider_config_key``), such as "openai" or "mistralai".

    Returns:
        The conventional environment variable name (e.g. "OPENAI_API_KEY"),
        or None when the provider is not one of the known keyed providers.
    """
    if provider_key not in PROVIDERS_REQUIRING_API_KEY_KEYS:
        return None
    return _DEFAULT_API_KEY_ENV_VAR_ALIASES.get(
        provider_key, f"{provider_key.upper()}_API_KEY"
    )


def get_provider_readiness(
    provider: Optional[str],
    app_config: Mapping[str, object],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> ProviderReadiness:
    """Resolve whether the selected provider has enough credentials to send.

    Args:
        provider: Display provider name from the Chat selector.
        app_config: Loaded app configuration.
        environ: Environment mapping, injectable for deterministic tests.

    Returns:
        Readiness state. If a key is found, it is returned for call wiring but
        never included in ``user_message``.
    """
    provider_name = (provider or "").strip()
    provider_key = provider_config_key(provider_name)
    env = environ if environ is not None else os.environ

    if not provider_name:
        return ProviderReadiness(
            provider="No provider",
            provider_key="",
            requires_api_key=False,
            ready=False,
            api_key=None,
            api_key_source=None,
            env_var=None,
            reason="Select a provider",
            recovery="Choose a provider and model before sending.",
        )

    api_settings = app_config.get("api_settings", {})
    provider_settings = provider_settings_for_key(api_settings, provider_key)

    requires_api_key = _requires_api_key(provider_key)
    configured_key = _valid_api_key(provider_settings.get("api_key"))
    if configured_key:
        return ProviderReadiness(
            provider=provider_name,
            provider_key=provider_key,
            requires_api_key=requires_api_key,
            ready=True,
            api_key=configured_key,
            api_key_source=f"config:api_settings.{provider_key}.api_key",
            env_var=None,
            reason="Ready",
            recovery=None,
        )

    env_var_value = provider_settings.get("api_key_env_var")
    env_var = (
        env_var_value.strip()
        if isinstance(env_var_value, str) and env_var_value.strip()
        else default_api_key_env_var(provider_key)
    )
    env_key = _valid_api_key(env.get(env_var, "")) if env_var else None
    if env_key:
        return ProviderReadiness(
            provider=provider_name,
            provider_key=provider_key,
            requires_api_key=requires_api_key,
            ready=True,
            api_key=env_key,
            api_key_source=f"env:{env_var}",
            env_var=env_var,
            reason="Ready",
            recovery=None,
        )

    if provider_key not in KNOWN_PROVIDER_KEYS and not provider_settings:
        return ProviderReadiness(
            provider=provider_name,
            provider_key=provider_key,
            requires_api_key=True,
            ready=False,
            api_key=None,
            api_key_source=None,
            env_var=env_var,
            reason="Unknown provider",
            recovery=f"Choose a supported provider or add api_key under [api_settings.{provider_key}].",
        )

    if not requires_api_key:
        return ProviderReadiness(
            provider=provider_name,
            provider_key=provider_key,
            requires_api_key=False,
            ready=True,
            api_key=None,
            api_key_source=None,
            env_var=env_var,
            reason="Ready",
            recovery=None,
        )

    recovery_target = f"api_key under [api_settings.{provider_key}]"
    if env_var:
        recovery = f"Set {env_var} or add {recovery_target}."
    else:
        recovery = f"Add {recovery_target}."

    return ProviderReadiness(
        provider=provider_name,
        provider_key=provider_key,
        requires_api_key=True,
        ready=False,
        api_key=None,
        api_key_source=None,
        env_var=env_var,
        reason="Missing API key",
        recovery=recovery,
    )


@dataclass(frozen=True)
class ChatApiKeyFieldState:
    """Render + persistence state for the inline Chat-Defaults API-key input."""

    value: str  # masked prefill value; "" when nothing should be shown
    disabled: bool  # True for keyless providers or a locked/encrypted config
    placeholder: str  # hint shown when the box is empty
    can_persist: bool  # whether a user-entered value should be written on save


def chat_api_key_field_state(
    readiness: ProviderReadiness,
    *,
    locked: bool,
) -> ChatApiKeyFieldState:
    """Map provider readiness to the inline API-key field's UI/persistence state.

    Args:
        readiness: Resolved readiness for the currently selected provider.
        locked: True when config encryption is enabled but no session password is
            available (stored values are ciphertext and must not be shown/saved).

    Returns:
        The field state to render and the flag for whether a typed value is savable.
    """
    if not readiness.requires_api_key:
        return ChatApiKeyFieldState(
            value="",
            disabled=True,
            placeholder="No API key needed for this provider.",
            can_persist=False,
        )
    if locked:
        return ChatApiKeyFieldState(
            value="",
            disabled=True,
            placeholder="Unlock config to edit keys.",
            can_persist=False,
        )
    source = readiness.api_key_source or ""
    if source.startswith("config:") and readiness.api_key:
        return ChatApiKeyFieldState(
            value=readiness.api_key,
            disabled=False,
            placeholder="Enter API key",
            can_persist=True,
        )
    if source.startswith("env:") and readiness.env_var:
        return ChatApiKeyFieldState(
            value="",
            disabled=False,
            placeholder=f"Detected from ${readiness.env_var} — leave blank to keep it",
            can_persist=True,
        )
    return ChatApiKeyFieldState(
        value="",
        disabled=False,
        placeholder="Enter your API key to start using this provider",
        can_persist=True,
    )


def chat_api_key_value_to_persist(
    new_value: object,
    field_state: ChatApiKeyFieldState,
) -> Optional[str]:
    """Return the API-key value to persist, or None to skip the write.

    Skips when the field is non-persistable, blank, a placeholder, or unchanged
    from the currently displayed value.

    Args:
        new_value: The raw value typed in the field.
        field_state: The ChatApiKeyFieldState for the selected provider.

    Returns:
        The stripped value to persist, or None to skip the write.
    """
    if not field_state.can_persist:
        return None
    candidate = new_value.strip() if isinstance(new_value, str) else ""
    if not is_valid_provider_api_key(candidate):
        return None
    if candidate == field_state.value:
        return None
    return candidate
