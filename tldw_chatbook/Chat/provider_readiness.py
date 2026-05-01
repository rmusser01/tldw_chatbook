"""Side-effect-free provider readiness helpers for Chat."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional


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
        "zai",
    }
)
KEYLESS_PROVIDER_KEYS = frozenset(
    {
        "aphrodite",
        "custom",
        "custom_2",
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
        "ollama",
        "oobabooga",
        "tabbyapi",
        "vllm",
    }
)
KNOWN_PROVIDER_KEYS = PROVIDERS_REQUIRING_API_KEY_KEYS | KEYLESS_PROVIDER_KEYS

_PLACEHOLDER_KEYS = frozenset(
    {
        "",
        "<API_KEY_HERE>",
        "YOUR_KEY",
        "your_key",
        "your-api-key",
    }
)


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
    return (provider or "").strip().lower().replace(" ", "_").replace("-", "_")


def _valid_api_key(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped in _PLACEHOLDER_KEYS:
        return None
    return stripped or None


def _requires_api_key(provider_key: str) -> bool:
    """Return True unless the provider is known to work without credentials."""
    return provider_key not in KEYLESS_PROVIDER_KEYS


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
    provider_settings = {}
    if isinstance(api_settings, Mapping):
        maybe_provider_settings = api_settings.get(provider_key, {})
        if isinstance(maybe_provider_settings, Mapping):
            provider_settings = maybe_provider_settings

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
    env_var = env_var_value.strip() if isinstance(env_var_value, str) else None
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
