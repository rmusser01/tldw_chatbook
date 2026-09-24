"""Generic strict hosted Chat-Completions engine for preset providers (ADR-179).

Phase 1, Task 4: the RESOLUTION layer only. ``resolve_hosted_request`` ports
``LLM_Calls/zai.py::resolve_zai_request`` (and its helpers), parameterized by
a ``provider_registry.ProviderRecord`` preset instead of hardcoded Z.ai
constants: error copy is ``record.display_name``-prefixed, provider identity
is ``record.key``, env candidates are ``record.api_key_env_candidates``, and
numeric/streaming defaults come from ``record.settings_defaults``.

Deliberate divergences from the zai template (spec: Generic hosted provider
engine and presets design, "Data flow" step 2):

- A record may ship no default base URL (Databricks workspace hosts are
  per-account): a missing URL raises an actionable "workspace base URL is
  required" error instead of falling back to a builtin URL.
- When the configured URL's path is empty or ``/`` and the record carries a
  ``base_url_suffix`` (e.g. Databricks ``/openai/v1``), the suffix is
  appended; any other path is validated as-is.
- A pasted terminal ``/chat/completions`` URL is REJECTED with actionable
  copy. (``normalize_hosted_chat_base_url`` accepts-and-strips that suffix --
  zai relies on the strip -- but the engine preset contract rejects the
  paste outright, so the check lives here, before the shared normalizer.)
- A preset may ship a blank default model (Databricks has none: gateway
  models are workspace-configured, and readiness requires key and URL, not a
  model). An unset model therefore resolves to ``""`` and is gated by the
  payload layer; a user-supplied blank model still fails closed here.
- Credentials and endpoint are resolved before the model so missing-key and
  missing-URL errors surface with their actionable copy first.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from tldw_chatbook.LLM_Calls.hosted_chat import (
    ProviderPayloadValidators,
    normalize_hosted_chat_base_url,
)
from tldw_chatbook.config import (
    ProviderSettingsError,
    get_runtime_config_snapshot,
    provider_settings_for_key,
    resolve_provider_api_key,
)
from tldw_chatbook.provider_registry import ProviderRecord


@dataclass(frozen=True)
class HostedProviderResolution:
    """Immutable resolved engine request identity and transport policy."""

    provider: str
    model: str
    api_key: str = field(repr=False, compare=False)
    base_url: str
    timeout: float
    retries: int
    retry_delay: float
    streaming: bool


def resolve_hosted_request(
    record: ProviderRecord,
    *,
    explicit_api_key: object = None,
    explicit_base_url: object = None,
    explicit_model: object = None,
    explicit_timeout: object = None,
    explicit_retries: object = None,
    explicit_retry_delay: object = None,
    app_config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> HostedProviderResolution:
    """Resolve one engine-driven hosted request from canonical immutable sources.

    Args:
        record: Registry preset driving every default, env candidate, and
            error message.
        explicit_api_key: Caller-supplied credential; wins over settings/env.
        explicit_base_url: Caller-supplied base URL; wins over settings.
        explicit_model: Caller-supplied model; wins over settings.
        explicit_timeout: Caller-supplied timeout override.
        explicit_retries: Caller-supplied retry override.
        explicit_retry_delay: Caller-supplied retry-delay override.
        app_config: Canonical config mapping; defaults to the runtime
            snapshot. Never mutated.
        environ: Environment mapping; defaults to ``os.environ``.

    Returns:
        The immutable resolved request identity and transport policy.

    Raises:
        ChatConfigurationError: On any missing or malformed credential,
            base URL, model, or transport setting. Credential and endpoint
            failures surface before model failures (see module docstring).
    """
    validators = _validators_for(record)
    config: object = (
        get_runtime_config_snapshot().values if app_config is None else app_config
    )
    if not isinstance(config, Mapping):
        raise validators.configuration_error(
            f"{record.display_name} application configuration is invalid."
        )
    api_settings = config.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        raise validators.configuration_error(
            f"{record.display_name} api_settings must be a configuration table."
        )
    try:
        settings = provider_settings_for_key(api_settings, record.key)
    except ProviderSettingsError:
        raise validators.configuration_error(
            f"{record.display_name} api_settings.{record.key} must be one "
            "unambiguous configuration table."
        ) from None
    defaults = record.settings_defaults
    transport_settings = dict(settings)
    if explicit_timeout is not None:
        transport_settings["timeout"] = explicit_timeout
    if explicit_retries is not None:
        transport_settings["retries"] = explicit_retries
    if explicit_retry_delay is not None:
        transport_settings["retry_delay"] = explicit_retry_delay
    environment = os.environ if environ is None else environ
    api_key = _resolve_api_key(
        record, explicit=explicit_api_key, settings=settings, environ=environment
    )
    base_url = _resolve_base_url(record, explicit=explicit_base_url, settings=settings)
    return HostedProviderResolution(
        provider=record.key,
        model=_resolve_string(
            record,
            explicit=explicit_model,
            settings=settings,
            name="model",
            default=_settings_default(defaults, "model", ""),
        ),
        api_key=api_key,
        base_url=base_url,
        timeout=validators.positive_number(
            transport_settings, "timeout", _settings_default(defaults, "timeout", 90.0)
        ),
        retries=validators.nonnegative_integer(
            transport_settings, "retries", _settings_default(defaults, "retries", 3)
        ),
        retry_delay=validators.nonnegative_number(
            transport_settings,
            "retry_delay",
            _settings_default(defaults, "retry_delay", 5.0),
        ),
        streaming=_resolve_streaming(
            record,
            settings=settings,
            default=_settings_default(defaults, "streaming", True),
        ),
    )


def _validators_for(record: ProviderRecord) -> ProviderPayloadValidators:
    """Build the shared error/validator family for one record."""
    return ProviderPayloadValidators(record.key, record.display_name)


def _settings_default(
    defaults: Mapping[str, object], name: str, fallback: object
) -> object:
    """Return a preset's shipped default, or the fallback when absent."""
    value = defaults.get(name, fallback)
    return fallback if value is None else value


def _resolve_string(
    record: ProviderRecord,
    *,
    explicit: object,
    settings: Mapping[str, object],
    name: str,
    default: object,
) -> str:
    validators = _validators_for(record)
    if explicit is not None:
        value: object = explicit
        supplied = True
    elif name in settings:
        value = settings.get(name)
        supplied = True
    else:
        value = default
        supplied = False
    if not isinstance(value, str):
        raise validators.configuration_error(
            f"{record.display_name} {name} is invalid."
        )
    stripped = value.strip()
    if not stripped:
        # A preset may ship a blank default (Databricks has no default
        # model); an unset value then passes through blank for the payload
        # layer to gate. A user-supplied blank is still an error.
        if supplied:
            raise validators.configuration_error(
                f"{record.display_name} {name} is invalid."
            )
        return ""
    return stripped


def _resolve_api_key(
    record: ProviderRecord,
    *,
    explicit: object,
    settings: Mapping[str, object],
    environ: Mapping[str, str],
) -> str:
    validators = _validators_for(record)
    if explicit is not None:
        resolved = resolve_provider_api_key(explicit)
        if resolved is None:
            raise validators.configuration_error(
                f"{record.display_name} explicit API key is invalid."
            )
        return resolved
    if "api_key" in settings:
        resolved = resolve_provider_api_key(settings.get("api_key"))
        if resolved is None:
            raise validators.configuration_error(
                f"{record.display_name} api_settings.{record.key}.api_key is invalid."
            )
        return resolved
    env_name = settings.get("api_key_env_var", record.api_key_env_var)
    if not isinstance(env_name, str) or not env_name.strip():
        raise validators.configuration_error(
            f"{record.display_name} api_settings.{record.key}.api_key_env_var "
            "is invalid."
        )
    for candidate in dict.fromkeys(
        (env_name.strip(), *record.api_key_env_candidates)
    ):
        resolved = resolve_provider_api_key(environ.get(candidate))
        if resolved is not None:
            return resolved
    raise validators.configuration_error(
        f"{record.display_name} API key is required."
    )


def _resolve_base_url(
    record: ProviderRecord,
    *,
    explicit: object,
    settings: Mapping[str, object],
) -> str:
    validators = _validators_for(record)
    candidate: object = (
        explicit if explicit is not None else settings.get("api_base_url")
    )
    if candidate is None and isinstance(record.default_base_url, str):
        candidate = record.default_base_url
    if candidate is None:
        raise validators.configuration_error(
            f"{record.display_name} workspace base URL is required."
        )
    if (
        isinstance(candidate, str)
        and isinstance(record.base_url_suffix, str)
        and _url_path(candidate) in ("", "/")
    ):
        candidate = f"{candidate.rstrip('/')}{record.base_url_suffix}"
    if isinstance(candidate, str) and _names_terminal_chat_completions(candidate):
        raise validators.configuration_error(
            f"{record.display_name} API base URL must not include a terminal "
            "/chat/completions endpoint path."
        )
    default_url = (
        record.default_base_url
        if isinstance(record.default_base_url, str)
        else ""
    )
    try:
        return normalize_hosted_chat_base_url(candidate, default=default_url)
    except ValueError:
        raise validators.configuration_error(
            f"{record.display_name} API base URL is invalid."
        ) from None


def _resolve_streaming(
    record: ProviderRecord,
    *,
    settings: Mapping[str, object],
    default: object,
) -> bool:
    validators = _validators_for(record)
    value = settings.get("streaming", default)
    if type(value) is not bool:
        raise validators.configuration_error(
            f"{record.display_name} streaming must be a boolean."
        )
    return value


def _url_path(candidate: str) -> str | None:
    """Return the URL path, or None when the candidate cannot be split."""
    try:
        return urlsplit(candidate).path
    except ValueError:
        return None


def _names_terminal_chat_completions(candidate: str) -> bool:
    """Check whether the URL path ends with a pasted ``/chat/completions``."""
    path = _url_path(candidate)
    if path is None:
        return False
    segments = tuple(
        segment.casefold() for segment in path.strip("/").split("/")
    )
    return len(segments) >= 2 and segments[-2:] == ("chat", "completions")
