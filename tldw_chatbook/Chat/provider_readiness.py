"""Side-effect-free provider readiness helpers for Chat."""

from __future__ import annotations

import os

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from unicodedata import category as unicode_category

from .Chat_Deps import ChatConfigurationError

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
    ProviderSettingsError,
    is_valid_provider_api_key,
    normalize_provider_config_key,
    provider_settings_for_key,
)
from ..config import (
    resolve_provider_api_key as _valid_api_key,
)
from .provider_test_evidence import (
    ConfigurationFacet,
    ConfigurationIssueCode,
    ProviderDraftIdentity,
    ProviderReadinessSnapshot,
    ProviderReadinessVerdict,
    ProviderTestEvidence,
    provider_readiness_verdict,
)

# TASK-26022: readiness label for the borrowed Claude Code credential. The
# token itself is NEVER carried on a readiness record (the call path re-reads
# it); the source string is the AC#5 "which credential is in use" signal.
SUBSCRIPTION_SOURCE = "subscription:claude_code"

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
_STRICT_HOSTED_PROVIDER_KEYS = frozenset({"moonshot", "zai"})
_PROVIDER_KEY_PATTERN = re.compile(r"[a-z0-9_.]+")
_ENV_VAR_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_UNSAFE_TEXT_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_MAX_PROVIDER_CHARS = 256
_MAX_PROVIDER_KEY_CHARS = 128
_MAX_SOURCE_CHARS = 256
_MAX_ENV_VAR_CHARS = 128
_MAX_REASON_CHARS = 128
_MAX_RECOVERY_CHARS = 1024
_CONFIGURATION_STATE_BY_REASON: dict[
    str, tuple[ConfigurationFacet, ConfigurationIssueCode | None]
] = {
    "Ready": ("configured", None),
    # TASK-26022: subscription-mode states (AC#5). Both blocked states map to
    # credential_missing -- the remedy is external (log in with Claude Code).
    "Ready (Claude subscription)": ("configured", None),
    "Claude subscription credential is expired": ("incomplete", "credential_missing"),
    "No Claude subscription credential found": ("incomplete", "credential_missing"),
    "Select a provider": ("incomplete", "provider_missing"),
    "Missing API key": ("incomplete", "credential_missing"),
    "Invalid provider settings": ("incomplete", "invalid_settings"),
    "Unknown provider": ("incomplete", "invalid_settings"),
}
_PERSISTED_CREDENTIAL_SOURCES = frozenset({"none", "stored", "environment"})


def _validate_safe_text(value: object, *, label: str, max_chars: int) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > max_chars
        or not value.isprintable()
        or any(
            unicode_category(character) in _UNSAFE_TEXT_CATEGORIES
            for character in value
        )
    ):
        raise ValueError(f"{label} is invalid.")


def _validate_provider_key(provider_key: object) -> None:
    if type(provider_key) is not str:
        raise ValueError("Provider key is invalid.")
    if provider_key and (
        len(provider_key) > _MAX_PROVIDER_KEY_CHARS
        or _PROVIDER_KEY_PATTERN.fullmatch(provider_key) is None
    ):
        raise ValueError("Provider key is invalid.")


def _validate_env_var(env_var: object) -> None:
    if (
        type(env_var) is not str
        or len(env_var) > _MAX_ENV_VAR_CHARS
        or _ENV_VAR_PATTERN.fullmatch(env_var) is None
        or not env_var.isprintable()
        or any(
            unicode_category(character) in _UNSAFE_TEXT_CATEGORIES
            for character in env_var
        )
    ):
        raise ValueError("Environment variable is invalid.")


@dataclass(frozen=True)
class ProviderReadiness:
    """Current readiness state for the selected Chat provider."""

    provider: str
    provider_key: str
    requires_api_key: bool
    ready: bool
    api_key: str | None = field(repr=False)
    api_key_source: str | None
    env_var: str | None
    reason: str
    recovery: str | None

    def __post_init__(self) -> None:
        """Reject legacy boolean states that contradict their structured truth."""
        if type(self.ready) is not bool or type(self.requires_api_key) is not bool:
            raise ValueError("Provider readiness flags are invalid.")
        _validate_safe_text(
            self.provider,
            label="Provider",
            max_chars=_MAX_PROVIDER_CHARS,
        )
        _validate_provider_key(self.provider_key)
        _validate_safe_text(
            self.reason,
            label="Provider reason",
            max_chars=_MAX_REASON_CHARS,
        )
        if self.recovery is not None:
            _validate_safe_text(
                self.recovery,
                label="Provider recovery",
                max_chars=_MAX_RECOVERY_CHARS,
            )
        if self.env_var is not None:
            _validate_env_var(self.env_var)

        structured_state = _CONFIGURATION_STATE_BY_REASON.get(self.reason)
        if structured_state is None:
            raise ValueError("Provider reason is not a supported readiness state.")
        configuration_facet, configuration_issue = structured_state
        if self.ready != (configuration_facet == "configured"):
            raise ValueError("Provider readiness state is inconsistent.")
        if self.ready and self.recovery is not None:
            raise ValueError("Ready provider state is inconsistent.")
        if not self.ready and self.recovery is None:
            raise ValueError("Blocked provider recovery is missing.")
        object.__setattr__(self, "_configuration_facet", configuration_facet)
        object.__setattr__(self, "_configuration_issue", configuration_issue)

        if self.reason == "Select a provider":
            if (
                self.provider != "No provider"
                or self.provider_key
                or self.requires_api_key
                or self.env_var is not None
            ):
                raise ValueError("Provider selection state is inconsistent.")
        elif not self.provider_key:
            raise ValueError("Provider key is required for the selected provider.")
        if self.reason == "Missing API key" and not self.requires_api_key:
            raise ValueError("Keyless provider cannot report a missing API key.")
        if self.reason == "Unknown provider" and not self.requires_api_key:
            raise ValueError("Unknown provider credential state is inconsistent.")

        has_key = self.api_key is not None
        has_source = self.api_key_source is not None
        subscription_source = self.api_key_source == SUBSCRIPTION_SOURCE
        # TASK-26022: the subscription source is deliberately key-less -- the
        # borrowed token never rides a readiness record (AC#3); the call path
        # reads the credential file itself.
        if has_key != has_source and not (subscription_source and not has_key):
            raise ValueError("Provider credential source is inconsistent.")
        if not self.ready and (has_key or has_source):
            raise ValueError("Blocked provider cannot retain a credential.")
        if self.ready and self.requires_api_key and not has_source:
            raise ValueError("Ready keyed provider requires a credential source.")
        if has_key:
            if not is_valid_provider_api_key(self.api_key):
                raise ValueError("Provider credential state is invalid.")
            source = self.api_key_source
            assert source is not None
            _validate_safe_text(
                source,
                label="Provider credential source",
                max_chars=_MAX_SOURCE_CHARS,
            )
            if source.startswith("env:"):
                if (
                    self.env_var is None
                    or source != f"env:{self.env_var}"
                ):
                    raise ValueError("Environment credential source is inconsistent.")
            elif source != f"config:api_settings.{self.provider_key}.api_key":
                raise ValueError("Provider credential source is invalid.")

    @property
    def configuration_facet(self) -> ConfigurationFacet:
        """Compatibility-safe structured view of configuration readiness."""
        return self._configuration_facet

    @property
    def configuration_issue(self) -> ConfigurationIssueCode | None:
        """Return a bounded explanation for an incomplete legacy state."""
        return self._configuration_issue

    def snapshot(
        self,
        *,
        selected_model: object = "",
        evidence: ProviderTestEvidence | None = None,
        current_identity: ProviderDraftIdentity | None = None,
    ) -> ProviderReadinessSnapshot:
        """Combine legacy configuration readiness with current test evidence."""
        model_id = selected_model.strip() if isinstance(selected_model, str) else ""
        evidence_is_current = bool(
            evidence is not None
            and current_identity is not None
            and evidence.identity == current_identity
            and current_identity.provider_key == self.provider_key
        )
        if evidence is not None and not evidence_is_current:
            endpoint = "changed_since_test"
        else:
            endpoint = evidence.endpoint if evidence is not None else "not_tested"
        if not model_id:
            model = "missing"
        elif (
            evidence_is_current
            and evidence.endpoint == "reachable"
            and model_id in evidence.model_ids
        ):
            model = "confirmed"
        else:
            model = "unconfirmed"
        return ProviderReadinessSnapshot(
            configuration=self.configuration_facet,
            endpoint=endpoint,
            model=model,
            category=evidence.category if evidence_is_current else None,
            configuration_issue=self.configuration_issue,
        )

    def verdict(
        self,
        *,
        selected_model: object = "",
        evidence: ProviderTestEvidence | None = None,
        current_identity: ProviderDraftIdentity | None = None,
    ) -> ProviderReadinessVerdict:
        """Return one structured verdict while preserving legacy properties."""
        return provider_readiness_verdict(
            self.snapshot(
                selected_model=selected_model,
                evidence=evidence,
                current_identity=current_identity,
            )
        )

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


def provider_config_key(provider: str | None) -> str:
    """Return the normalized key used under ``api_settings``."""
    return normalize_provider_config_key(provider)


def _requires_api_key(provider_key: str) -> bool:
    """Return True unless the provider is known to work without credentials."""
    return provider_key not in KEYLESS_PROVIDER_KEYS


def default_api_key_env_var(provider_key: str) -> str | None:
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


def _resolved_hosted_api_key(
    provider_key: str,
    app_config: Mapping[str, object],
    environ: Mapping[str, str],
) -> str | None:
    """Validate hosted readiness through the provider's send resolver."""
    if provider_key == "moonshot":
        from tldw_chatbook.LLM_Calls.moonshot import resolve_moonshot_request

        return resolve_moonshot_request(
            app_config=app_config,
            environ=environ,
        ).api_key
    if provider_key == "zai":
        from tldw_chatbook.LLM_Calls.zai import resolve_zai_request

        return resolve_zai_request(
            app_config=app_config,
            environ=environ,
        ).api_key
    return None


def _invalid_settings_readiness(
    provider_name: str,
    provider_key: str,
) -> ProviderReadiness:
    return ProviderReadiness(
        provider=provider_name,
        provider_key=provider_key,
        requires_api_key=_requires_api_key(provider_key),
        ready=False,
        api_key=None,
        api_key_source=None,
        env_var=None,
        reason="Invalid provider settings",
        recovery=(
            f"Replace api_settings.{provider_key} with one valid configuration "
            "table in Advanced Config or config.toml."
        ),
    )


def configured_provider_credential_source(
    provider_settings: Mapping[str, object],
) -> str | None:
    """Return one explicit persisted auth decision, or legacy mode when absent."""

    value = provider_settings.get("credential_source")
    if type(value) is not str:
        return None
    normalized = value.strip().lower()
    return normalized if normalized in _PERSISTED_CREDENTIAL_SOURCES else None


def resolve_provider_credential(
    provider_key: str,
    provider_settings: Mapping[str, object],
    *,
    environ: Mapping[str, str],
) -> tuple[str | None, str | None, str | None]:
    """Resolve the selected credential without overriding an explicit auth mode."""

    selected_source = configured_provider_credential_source(provider_settings)
    configured_key = _valid_api_key(provider_settings.get("api_key"))
    env_var_value = provider_settings.get("api_key_env_var")
    configured_env_var = (
        env_var_value.strip()
        if isinstance(env_var_value, str) and env_var_value.strip()
        else None
    )

    if selected_source == "none":
        return None, None, None
    if selected_source == "stored":
        if configured_key is None:
            return None, None, None
        return (
            configured_key,
            f"config:api_settings.{provider_key}.api_key",
            None,
        )

    env_var = configured_env_var
    if selected_source is None and env_var is None:
        env_var = default_api_key_env_var(provider_key)
    if selected_source is None and configured_key is not None:
        return (
            configured_key,
            f"config:api_settings.{provider_key}.api_key",
            None,
        )
    env_key = _valid_api_key(environ.get(env_var, "")) if env_var else None
    if env_key is None:
        return None, None, env_var
    return env_key, f"env:{env_var}", env_var

def get_provider_readiness(
    provider: str | None,
    app_config: Mapping[str, object],
    *,
    environ: Mapping[str, str] | None = None,
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
    try:
        provider_settings = provider_settings_for_key(api_settings, provider_key)
    except ProviderSettingsError:
        return _invalid_settings_readiness(provider_name, provider_key)

    requires_api_key = _requires_api_key(provider_key)

    if provider_key == "anthropic":
        from tldw_chatbook.LLM_Calls.anthropic_subscription import (
            anthropic_auth_source,
            read_claude_code_credential,
        )

    # TASK-26022 (AC#5): explicit subscription mode for Anthropic. Reported as
    # its own source so subscription vs API key is visible at a glance; the
    # token never rides the record. Missing/expired blocks with the
    # refresh-in-Claude-Code copy -- never a silent API-key fallback.
    if provider_key == "anthropic" and anthropic_auth_source(
        provider_settings
    ) == "claude_subscription":
        _sub = read_claude_code_credential()
        if _sub is not None and not _sub.expired:
            return ProviderReadiness(
                provider=provider_name,
                provider_key=provider_key,
                requires_api_key=requires_api_key,
                ready=True,
                api_key=None,
                api_key_source=SUBSCRIPTION_SOURCE,
                env_var=None,
                reason="Ready (Claude subscription)",
                recovery=None,
            )
        return ProviderReadiness(
            provider=provider_name,
            provider_key=provider_key,
            requires_api_key=requires_api_key,
            ready=False,
            api_key=None,
            api_key_source=None,
            env_var=None,
            reason=(
                "Claude subscription credential is expired"
                if _sub is not None
                else "No Claude subscription credential found"
            ),
            recovery=(
                "Refresh it in the tool that owns it (log in with Claude Code), "
                "or set [api_settings.anthropic] auth_source back to \"api_key\"."
            ),
        )

    configured_key, configured_source, env_var = resolve_provider_credential(
        provider_key,
        provider_settings,
        environ=env,
    )
    if configured_key and provider_key in _STRICT_HOSTED_PROVIDER_KEYS:
        try:
            configured_key = _resolved_hosted_api_key(provider_key, app_config, env)
        except ChatConfigurationError:
            return _invalid_settings_readiness(provider_name, provider_key)
    if configured_key:
        return ProviderReadiness(
            provider=provider_name,
            provider_key=provider_key,
            requires_api_key=requires_api_key,
            ready=True,
            api_key=configured_key,
            api_key_source=configured_source,
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
) -> str | None:
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
