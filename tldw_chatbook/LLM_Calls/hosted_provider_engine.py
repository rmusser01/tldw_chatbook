"""Generic strict hosted Chat-Completions engine for preset providers (ADR-179).

Phase 1, Tasks 4-5: the RESOLUTION layer (``resolve_hosted_request``) and
the PAYLOAD layer (``build_hosted_chat_payload``). Both port their
``LLM_Calls/zai.py`` counterparts, parameterized by a
``provider_registry.ProviderRecord`` preset instead of hardcoded Z.ai
constants: error copy is ``record.display_name``-prefixed, provider identity
is ``record.key``, env candidates are ``record.api_key_env_candidates``,
numeric/streaming defaults come from ``record.settings_defaults``, and the
payload surface is gated by ``record.payload_flags``.

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
- ``record.auth_scheme`` gates credential strictness (Phase 2, Task 3):
  ``"bearer"`` (the default) keeps the hard key requirement, while
  ``"bearer_optional"`` lets keyless endpoints (ADR-146 custom endpoints)
  execute -- an unresolved chain yields ``api_key == ""`` and the transport
  sends no Authorization header at all. ``"api_key_header"`` is Phase 3.

Payload-layer (Task 5) divergences from the zai template:

- A blank-default record may resolve ``model`` to ``""`` (Databricks ships
  none); the payload builder fails closed on an empty model instead of
  sending one.
- Optional body fields are gated by ``record.payload_flags``: a flag-off
  field with a caller-supplied value is a bad request (never silently
  dropped -- hidden caller intent is the failure the strict engine exists
  to surface), and ``reasoning_effort`` is gated by ``record.reasoning_effort``
  with the key from ``record.reasoning_effort_key``.
- ``record.extra_body_fields`` merge into the payload last, each validated
  bounded (preset authoring data, not caller input).
- No provider-invented fields: zai's ``thinking``/``request_id``/``user_id``
  quirks do not exist here -- the engine emits exactly the OpenAI body the
  record describes. Unknown keyword arguments are swallowed (``**_generic``)
  so one shared per-provider param map can drive every preset.

Response-layer (Task 6) divergences from the zai template (spec: Generic
hosted provider engine and presets design, "Data flow" step 4):

- Provider-terminal finish reasons are record data
  (``record.finish_provider_errors``) raising ``ChatProviderError`` with
  ``record.key`` identity and 502, and the allowed terminal set is
  ``record.finish_terminal`` -- not Z.ai's hardcoded sets.
- ``record.reasoning_disposition`` gates reasoning everywhere: ``"ignored"``
  drops it at the finish policy (terminal turns carry none);
  ``"displayable"`` keeps it visible in stream deltas; ``"proprietary"``
  keeps it private to the terminal turn. zai strips ``reasoning_content``
  from visible deltas unconditionally (its disposition is proprietary).
- ``record.response_allowances`` feed the shared hosted boundary's tolerated
  extra top-level response/stream keys (Task 3 seam).
- Phase 2 Task 4: ``record.choice_allowances``/``record.message_allowances``
  subtract at the choice and message/delta levels (value rule: null, scalar,
  or shape-safe mapping -- validated then dropped), and
  ``record.tolerant_response_extras`` (custom family only) switches the
  fixture-gated long-tail tolerant profile: shape-safe unknown top/event
  keys dropped; null-valued unknown choice/message keys dropped (non-null
  ones still fail closed unless level-allowlisted); tool-call objects may
  carry extra keys (id/type/function stay mandatory); a stream terminal
  without usage becomes a usage-None turn; and the finish policy accepts
  stop/length with empty text and no calls (legacy empty reply).
- The stream wrapper's continuation candidate is built against the record's
  key/protocol; the canonical-format parse round-trip admits the
  continuation ``_PAIRINGS`` providers, which Task 7 widened with
  ``("databricks", "chat_completions")`` (the registry's engine preset).

Handler layer (Task 7) notes:

- ``build_hosted_chat_handler(record)`` closes over the Task 4-6 layers to
  produce one provider's ``chat_with_*`` callable: the ``chat_with_zai``
  signature minus the provider-invented ``do_sample``/``request_id``
  parameters (the engine emits exactly the OpenAI body the record
  describes). The continuation candidate is skipped entirely for records
  whose ``continuation_protocol`` is None.
- The response message keeps ``reasoning_content`` only when the record's
  disposition is ``"displayable"`` (zai pops it unconditionally; its
  disposition is proprietary).
- The per-provider metric counters live inside the factory closure -- the
  exact ``log_counter``/``log_histogram`` call shapes of the
  ``chat_with_zai``/``chat_with_moonshot`` compatibility wrappers in
  ``LLM_API_Calls.py``, with ``record.key`` as the provider identity --
  once, for every engine provider, with no wrapper module.
- ``resolve_hosted_engine_request`` is the thin public alias of
  ``resolve_hosted_request`` consumed by the catalog service (Task 12).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import urlsplit

from loguru import logger

from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError, ChatProviderError
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRestoreTarget,
    ContinuationRound,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
    parse_provider_continuation_json,
    validate_continuation_restore,
)
from tldw_chatbook.Chat.provider_readiness import configured_workspace_base_url
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    HostedChatStream,
    HostedChatTurn,
    HostedHTTPTransportConfig,
    ProviderPayloadValidators,
    ReasoningDisposition,
    TOOL_FUNCTION_NAME,
    normalize_hosted_chat_base_url,
    normalize_hosted_chat_response,
    owned_json_post,
)
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
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
    # Section-backed sampler fallbacks (records with a
    # ``defaults_settings_section`` only, Task 6): values read from the
    # legacy ``api_settings`` section when the caller supplies none. Other
    # records leave these None -- their payloads stay caller-driven.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: object = None
    response_format: object = None


def resolve_hosted_request(
    record: ProviderRecord,
    *,
    explicit_api_key: object = None,
    explicit_base_url: object = None,
    explicit_model: object = None,
    explicit_timeout: object = None,
    explicit_retries: object = None,
    explicit_retry_delay: object = None,
    api_key_resolved: bool | None = None,
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
        explicit_retries: Caller-supplied retries override.
        explicit_retry_delay: Caller-supplied retry-delay override.
        api_key_resolved: Whether a trusted caller (the Console gateway)
            already made the final credential decision. When True the
            supplied ``explicit_api_key`` is the exact credential: an
            unusable/empty supply is the explicit keyless decision (no
            Authorization header) for ``bearer_optional`` records and still
            fails closed for strict ``bearer`` records -- the settings/env
            chain is never consulted as a fallback, so a globally configured
            key can never leak to a caller-controlled endpoint URL.
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
    settings_key = (
        record.defaults_settings_section
        if record.defaults_settings_section is not None
        else record.key
    )
    try:
        settings = provider_settings_for_key(api_settings, settings_key)
    except ProviderSettingsError:
        raise validators.configuration_error(
            f"{record.display_name} api_settings.{settings_key} must be one "
            "unambiguous configuration table."
        ) from None
    defaults = record.settings_defaults
    transport_settings = dict(settings)
    if record.defaults_settings_section is not None:
        # Legacy transport spellings (chat_with_custom_openai): the section
        # spells the api_* forms; the plain forms were never read there.
        for legacy_name, canonical_name in (
            ("api_timeout", "timeout"),
            ("api_retries", "retries"),
            ("api_retry_delay", "retry_delay"),
        ):
            if legacy_name in settings and canonical_name not in settings:
                transport_settings[canonical_name] = settings[legacy_name]
    if explicit_timeout is not None:
        transport_settings["timeout"] = explicit_timeout
    if explicit_retries is not None:
        transport_settings["retries"] = explicit_retries
    if explicit_retry_delay is not None:
        transport_settings["retry_delay"] = explicit_retry_delay
    environment = os.environ if environ is None else environ
    api_key = _resolve_api_key(
        record,
        explicit=explicit_api_key,
        settings=settings,
        environ=environment,
        resolved=api_key_resolved,
    )
    base_url = _resolve_base_url(record, explicit=explicit_base_url, settings=settings)
    section_sampling = (
        _resolve_section_sampling(record, settings)
        if record.defaults_settings_section is not None
        else None
    )
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
        **(section_sampling or {}),
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
    resolved: bool | None = None,
) -> str:
    """Resolve the request credential under the record's auth scheme.

    ``resolved=True`` is the trusted-caller credential decision (the Console
    gateway's ``api_key_resolved``): the supplied ``explicit`` value is the
    FINAL credential, used exactly as supplied with no settings/env
    fallback -- an empty or absent supply is the keyless decision for
    ``bearer_optional`` records (the transport then sends no Authorization
    header at all) and still fails closed for strict ``bearer`` records.

    ``"bearer"`` (the default) hard-requires a key: an unusable explicit or
    stored value, or an unresolved env chain, is an actionable configuration
    error. ``"bearer_optional"`` lets keyless endpoints (ADR-146) execute:
    an explicit blank string is "no key" (not an invalid one) and falls
    through to the settings/env chain, and an unresolved chain resolves to
    ``""`` -- the transport then sends no Authorization header at all.
    """
    validators = _validators_for(record)
    optional = record.auth_scheme == "bearer_optional"
    if resolved is True:
        supplied = resolve_provider_api_key(explicit)
        if supplied is not None:
            return supplied
        if optional:
            return ""
        raise validators.configuration_error(
            f"{record.display_name} explicit API key is invalid."
        )
    if explicit is not None:
        resolved = resolve_provider_api_key(explicit)
        if resolved is not None:
            return resolved
        if not (optional and isinstance(explicit, str)):
            raise validators.configuration_error(
                f"{record.display_name} explicit API key is invalid."
            )
        # bearer_optional: an explicit blank ("no key configured") defers to
        # the settings/env chain instead of winning with an unusable value.
    if "api_key" in settings:
        resolved = resolve_provider_api_key(settings.get("api_key"))
        if resolved is None:
            raise validators.configuration_error(
                f"{record.display_name} api_settings.{record.key}.api_key is invalid."
            )
        return resolved
    env_name = settings.get("api_key_env_var", record.api_key_env_var)
    names: tuple[str, ...] = ()
    if isinstance(env_name, str):
        if env_name.strip():
            names = (env_name.strip(),)
        # A blank name is "no name set", not an invalid one: a keyless
        # preset ships neither a name nor candidates, so the name-validity
        # check is skipped and the record's shipped candidates still run.
    elif env_name is not None:
        raise validators.configuration_error(
            f"{record.display_name} api_settings.{record.key}.api_key_env_var "
            "is invalid."
        )
    for candidate in dict.fromkeys((*names, *record.api_key_env_candidates)):
        resolved = resolve_provider_api_key(environ.get(candidate))
        if resolved is not None:
            return resolved
    if optional:
        return ""
    raise validators.configuration_error(
        f"{record.display_name} API key is required."
    )


def _resolve_base_url(
    record: ProviderRecord,
    *,
    explicit: object,
    settings: Mapping[str, object],
) -> str:
    """Resolve the request base URL.

    Precedence: the caller's explicit URL, then the settings table's first
    configured base-URL alias (the shared readiness alias list -- a URL
    spelled ``base_url``/``api_base``/``api_url``/``endpoint`` resolves the
    same as ``api_base_url``; Qodo finding 2), then the record's shipped
    default. A record with no default (Databricks) fails closed here.
    """
    validators = _validators_for(record)
    candidate: object = explicit
    if candidate is None:
        candidate = configured_workspace_base_url(settings)
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


def _resolve_section_sampling(
    record: ProviderRecord, settings: Mapping[str, object]
) -> dict[str, object]:
    """Read one legacy settings section's per-call sampler fallbacks.

    ``chat_with_custom_openai`` read, per call, ``temperature`` (fallback
    ``temp``), ``top_p`` (``maxp``), ``top_k`` (``topk``), ``min_p``
    (``minp``), ``max_tokens`` (default 4096), ``seed``, ``stop``, and
    ``response_format`` from its ``api_settings`` section. The engine reads
    the same spellings, strictly validated; absent keys resolve to the
    record's ``settings_defaults`` (or None when no default ships). The
    legacy string coercions (``streaming = "true"``) are NOT ported: the
    strict engine fails closed with actionable copy instead.
    """
    validators = _validators_for(record)
    defaults = record.settings_defaults
    resolved: dict[str, object] = {}
    samplers: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("temperature", ("temperature", "temp")),
        ("top_p", ("top_p", "maxp")),
        ("min_p", ("min_p", "minp")),
    )
    for name, spellings in samplers:
        value = _first_present(settings, spellings)
        if value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise validators.configuration_error(
                    f"{record.display_name} {name} is invalid."
                )
            resolved[name] = float(value)
    top_k = _first_present(settings, ("top_k", "topk"))
    if top_k is not None:
        # A count, not a probability: 0 is llama-family "off" (the shipped
        # table spells it) and flows through exactly as the legacy handler
        # passed it.
        if type(top_k) is not int or top_k < 0:
            raise validators.configuration_error(
                f"{record.display_name} top_k is invalid."
            )
        resolved["top_k"] = top_k
    max_tokens = settings.get("max_tokens")
    if max_tokens is not None:
        resolved["max_tokens"] = validators.positive_integer(
            "max_tokens", max_tokens
        )
    else:
        default_max_tokens = _settings_default(defaults, "max_tokens", None)
        if default_max_tokens is not None:
            resolved["max_tokens"] = validators.positive_integer(
                "max_tokens", default_max_tokens
            )
    seed = settings.get("seed")
    if seed is not None:
        if type(seed) is not int:
            raise validators.configuration_error(
                f"{record.display_name} seed is invalid."
            )
        resolved["seed"] = seed
    stop = settings.get("stop")
    if stop is not None:
        try:
            validators.normalize_stop(stop)
        except ChatBadRequestError:
            raise validators.configuration_error(
                f"{record.display_name} stop is invalid."
            ) from None
        resolved["stop"] = stop
    response_format = settings.get("response_format")
    if response_format is not None:
        try:
            validators.normalize_response_format(response_format)
        except ChatBadRequestError:
            raise validators.configuration_error(
                f"{record.display_name} response_format is invalid."
            ) from None
        resolved["response_format"] = response_format
    return resolved


def _first_present(
    settings: Mapping[str, object], names: tuple[str, ...]
) -> object:
    """Return the first present value among the legacy key spellings."""
    for name in names:
        if name in settings:
            return settings.get(name)
    return None


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


# --- payload layer (Task 5; ports zai.py's payload-side helpers) ---


def build_hosted_chat_payload(
    record: ProviderRecord,
    *,
    resolution: HostedProviderResolution,
    messages_payload: Sequence[Mapping[str, Any]],
    system_message: object = None,
    streaming: object = None,
    tools: object = None,
    tool_choice: object = None,
    reasoning_effort: object = None,
    provider_continuations: Sequence[ProviderContinuationCheckpoint] = (),
    temperature: object = None,
    top_p: object = None,
    max_tokens: object = None,
    stop: object = None,
    response_format: object = None,
    seed: object = None,
    n: object = None,
    user: object = None,
    min_p: object = None,
    top_k: object = None,
    presence_penalty: object = None,
    frequency_penalty: object = None,
    logit_bias: object = None,
    logprobs: object = None,
    top_logprobs: object = None,
    thinking_budget_tokens: object = None,
    **_generic: object,
) -> dict[str, Any]:
    """Build one validated engine request without mutating inputs.

    Args:
        record: Registry preset driving flags, error copy, and extra body.
        resolution: Resolved request identity from ``resolve_hosted_request``.
        messages_payload: Chat-Completions message history.
        system_message: Optional system prompt; errors when the history
            already carries a system message.
        streaming: Transport streaming override; defaults to the resolution.
        tools: OpenAI function-tool descriptors.
        tool_choice: Only ``"auto"`` (with tools) is supported.
        reasoning_effort: Reasoning-effort request; requires
            ``record.reasoning_effort``.
        provider_continuations: Durable continuation checkpoints to restore
            onto the message history.
        temperature: Sampler in [0, 1]; requires the ``temperature`` flag.
        top_p: Sampler in [0, 1]; requires the ``top_p`` flag.
        max_tokens: Positive integer; requires the ``max_tokens`` flag.
        stop: Stop sequence(s); requires the ``stop`` flag.
        response_format: ``text``/``json_object``/``json_schema`` mapping;
            requires the ``response_format`` flag.
        seed: Integer seed; requires the ``seed`` flag.
        n: Positive completion count; requires the ``n`` flag.
        user: Bounded caller identity; requires the ``user`` flag.
        min_p: Sampler in [0, 1]; requires the ``min_p`` flag (custom
            family, Task 6).
        top_k: Positive integer; requires the ``top_k`` flag (custom family).
        presence_penalty: Penalty in [-2, 2]; requires its flag.
        frequency_penalty: Penalty in [-2, 2]; requires its flag.
        logit_bias: Bounded token-id to bias mapping; requires its flag.
        logprobs: Boolean logprob request; requires its flag.
        top_logprobs: Count in [0, 20]; requires ``logprobs`` and its flag.
        thinking_budget_tokens: Positive thinking budget; requires its flag.
            Accepted-and-dropped per the ADR-066 custom row (the record's
            reasoning composition decides emission; the custom family drops
            it so strict OpenAI proxies never see llama.cpp-specific fields).
        **_generic: Unknown keywords are ignored (one shared per-provider
            param map drives every preset).

    Returns:
        The validated request body.

    Raises:
        ChatBadRequestError: On an empty model, a resolution/record
            mismatch, or any malformed, unsupported, or flag-off-supplied
            value.
    """
    validators = _validators_for(record)
    bad_request = validators.bad_request
    if (
        type(resolution) is not HostedProviderResolution
        or resolution.provider != record.key
    ):
        raise bad_request(f"{record.display_name} resolution is invalid.")
    if not resolution.model:
        # Task 4 contract: a blank-default record resolves model to "" and
        # the payload layer gates it (readiness requires key and URL, not a
        # model) instead of sending a modelless request.
        raise bad_request(f"{record.display_name} model is required.")
    # Section-backed sampler fallbacks (Task 6): a caller-supplied value
    # wins over the resolution's settings-section value (the resolution is
    # explicit > section > record-defaults in source order).
    if temperature is None:
        temperature = resolution.temperature
    if top_p is None:
        top_p = resolution.top_p
    if min_p is None:
        min_p = resolution.min_p
    if top_k is None:
        top_k = resolution.top_k
    if max_tokens is None:
        max_tokens = resolution.max_tokens
    if seed is None:
        seed = resolution.seed
    if stop is None:
        stop = resolution.stop
    if response_format is None:
        response_format = resolution.response_format
    stream = resolution.streaming if streaming is None else streaming
    if type(stream) is not bool:
        raise bad_request(f"{record.display_name} streaming must be a boolean.")
    _validate_sampler(record, "temperature", temperature)
    _validate_sampler(record, "top_p", top_p)
    _validate_sampler(record, "min_p", min_p)
    messages = _normalize_messages(
        record, messages_payload, system_message=system_message
    )
    validated_tools = _normalize_tools(record, tools)
    validated_choice = _normalize_tool_choice(record, tool_choice, validated_tools)
    _apply_continuations(record, messages, provider_continuations, resolution)

    payload: dict[str, Any] = {
        "model": resolution.model,
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if min_p is not None:
        payload["min_p"] = min_p
    if top_k is not None:
        _require_payload_flag(record, "top_k")
        if type(top_k) is not int or top_k < 0:
            # Non-negative: 0 is llama-family "off" and is forwarded as-is.
            raise bad_request(f"{record.display_name} top_k is invalid.")
        payload["top_k"] = top_k
    if max_tokens is not None:
        _require_payload_flag(record, "max_tokens")
        payload["max_tokens"] = validators.positive_integer("max_tokens", max_tokens)
    if stop is not None:
        _require_payload_flag(record, "stop")
        payload["stop"] = validators.normalize_stop(stop)
    if response_format is not None:
        _require_payload_flag(record, "response_format")
        payload["response_format"] = validators.normalize_response_format(
            response_format
        )
    if seed is not None:
        _require_payload_flag(record, "seed")
        if type(seed) is not int:
            raise bad_request(f"{record.display_name} seed is invalid.")
        payload["seed"] = seed
    if n is not None:
        _require_payload_flag(record, "n")
        payload["n"] = validators.positive_integer("n", n)
    if presence_penalty is not None:
        payload["presence_penalty"] = _validate_penalty(
            record, "presence_penalty", presence_penalty
        )
    if frequency_penalty is not None:
        payload["frequency_penalty"] = _validate_penalty(
            record, "frequency_penalty", frequency_penalty
        )
    if logit_bias is not None:
        payload["logit_bias"] = _validate_logit_bias(record, logit_bias)
    if logprobs is not None:
        _require_payload_flag(record, "logprobs")
        if type(logprobs) is not bool:
            raise bad_request(f"{record.display_name} logprobs is invalid.")
        payload["logprobs"] = logprobs
    if top_logprobs is not None:
        _require_payload_flag(record, "top_logprobs")
        if (
            type(top_logprobs) is not int
            or not 0 <= top_logprobs <= 20
            or logprobs is not True
        ):
            # The legacy handler silently ignored top_logprobs without
            # logprobs=True; the strict engine surfaces the hidden intent
            # instead of dropping it.
            raise bad_request(f"{record.display_name} top_logprobs is invalid.")
        payload["top_logprobs"] = top_logprobs
    if thinking_budget_tokens is not None:
        _require_payload_flag(record, "thinking_budget_tokens")
        validators.positive_integer(
            "thinking_budget_tokens", thinking_budget_tokens
        )
        # ADR-066 composition is record data: only a record whose
        # composition consumes a budget emits one. Today every engine
        # record follows the custom row (accepted, validated, dropped).
        logger.debug(
            "{} thinking budget is not consumable on this wire format; dropped",
            record.key,
        )
    if user is not None:
        _require_payload_flag(record, "user")
        payload["user"] = _bounded_identifier(record, "user", user)
    if validated_tools is not None:
        payload["tools"] = validated_tools
    if validated_choice is not None:
        payload["tool_choice"] = validated_choice
    if reasoning_effort is not None:
        if not record.reasoning_effort:
            raise bad_request(
                f"{record.display_name} reasoning effort is unsupported."
            )
        payload[record.reasoning_effort_key or "reasoning_effort"] = (
            _bounded_identifier(record, "reasoning effort", reasoning_effort)
        )
    for key, value in record.extra_body_fields.items():
        if not validators.json_shape_is_bounded(value):
            raise bad_request(f"{record.display_name} extra body field {key} is invalid.")
        payload[key] = value
    return payload


def _require_payload_flag(record: ProviderRecord, flag: str) -> None:
    """Fail closed when a preset lacks a payload flag but a value arrived."""
    if flag not in record.payload_flags:
        raise _validators_for(record).bad_request(
            f"{record.display_name} {flag} is unsupported."
        )


def _normalize_messages(
    record: ProviderRecord,
    value: object,
    *,
    system_message: object,
) -> list[dict[str, Any]]:
    validators = _validators_for(record)
    bad_request = validators.bad_request
    display = record.display_name
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise bad_request(f"{display} messages must be a sequence.")
    result: list[dict[str, Any]] = []
    has_system = any(
        isinstance(message, Mapping) and message.get("role") == "system"
        for message in value
    )
    if system_message is not None:
        if not isinstance(system_message, str) or has_system:
            raise bad_request(f"{display} system message ownership is invalid.")
        result.append({"role": "system", "content": system_message})

    call_ids: set[str] = set()
    pending_ids: list[str] = []
    for raw_message in value:
        if not isinstance(raw_message, Mapping):
            raise bad_request(f"{display} message is malformed.")
        role = raw_message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise bad_request(f"{display} message role is invalid.")
        if role == "tool":
            if not pending_ids or set(raw_message) != {
                "role",
                "tool_call_id",
                "content",
            }:
                raise bad_request(f"{display} tool result is malformed or orphaned.")
            call_id = raw_message.get("tool_call_id")
            content = raw_message.get("content")
            if call_id != pending_ids[0] or not isinstance(content, str):
                raise bad_request(f"{display} tool result ordering is invalid.")
            pending_ids.pop(0)
            result.append(deepcopy(dict(raw_message)))
            continue
        if pending_ids:
            raise bad_request(f"{display} tool call batch is incomplete.")
        allowed = {"role", "content"} | (
            {"tool_calls"} if role == "assistant" else set()
        )
        if set(raw_message) - allowed:
            raise bad_request(f"{display} message fields are unsupported.")
        content = raw_message.get("content")
        if role == "assistant":
            if content is not None and not isinstance(content, str):
                raise bad_request(f"{display} assistant content is invalid.")
        elif not isinstance(content, str):
            raise bad_request(f"{display} message content is invalid.")
        safe: dict[str, Any] = {
            "role": role,
            "content": "" if content is None else content,
        }
        if role == "assistant" and "tool_calls" in raw_message:
            calls = validators.normalize_call_batch(raw_message.get("tool_calls"), call_ids)
            safe["tool_calls"] = list(calls)
            pending_ids = [cast(str, call["id"]) for call in calls]
        result.append(safe)
    if pending_ids:
        raise bad_request(f"{display} tool call batch is incomplete.")
    return result


def _normalize_tools(
    record: ProviderRecord, value: object
) -> list[dict[str, Any]] | None:
    validators = _validators_for(record)
    bad_request = validators.bad_request
    display = record.display_name
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise bad_request(f"{display} tools are malformed.")
    names: set[str] = set()
    result: list[dict[str, Any]] = []
    for raw_tool in value:
        if not isinstance(raw_tool, Mapping) or set(raw_tool) != {"type", "function"}:
            raise bad_request(f"{display} supports function tools only.")
        function = raw_tool.get("function")
        if raw_tool.get("type") != "function" or not isinstance(function, Mapping):
            raise bad_request(f"{display} supports function tools only.")
        if set(function) != {"name", "description", "parameters"}:
            raise bad_request(f"{display} function tool is malformed.")
        name = function.get("name")
        parameters = function.get("parameters")
        if (
            not isinstance(name, str)
            or not TOOL_FUNCTION_NAME.fullmatch(name)
            or name in names
            or not isinstance(function.get("description"), str)
            or not cast(str, function.get("description")).strip()
            or not isinstance(parameters, Mapping)
            or parameters.get("type") != "object"
            or not validators.json_shape_is_bounded(parameters)
        ):
            raise bad_request(f"{display} function tool is malformed.")
        names.add(name)
        result.append(deepcopy(dict(raw_tool)))
    return result


def _normalize_tool_choice(
    record: ProviderRecord,
    value: object,
    tools: Sequence[Mapping[str, Any]] | None,
) -> str | None:
    if value is None:
        return None
    if value == "auto" and tools is not None:
        return "auto"
    raise _validators_for(record).bad_request(
        f"{record.display_name} tool choice is unsupported."
    )


def _apply_continuations(
    record: ProviderRecord,
    messages: list[dict[str, Any]],
    checkpoints: Sequence[ProviderContinuationCheckpoint],
    resolution: HostedProviderResolution,
) -> None:
    bad_request = _validators_for(record).bad_request
    display = record.display_name
    if not isinstance(checkpoints, Sequence):
        raise bad_request(f"{display} provider continuation is invalid.")
    cursor = 0
    for checkpoint in checkpoints:
        try:
            validate_continuation_restore(
                checkpoint,
                ContinuationRestoreTarget(
                    provider=record.key,
                    protocol=record.continuation_protocol or "chat_completions",
                    model=resolution.model,
                    api_base_url=resolution.base_url,
                ),
            )
        except Exception:
            raise bad_request(f"{display} provider continuation is invalid.") from None
        for round_ in checkpoint.rounds:
            call_ids = tuple(call.call_id for call in round_.calls)
            match_index = _find_owner(
                messages,
                assistant_content=round_.assistant_content,
                call_ids=call_ids,
                start=cursor,
            )
            if match_index is None:
                raise bad_request(f"{display} continuation owner is missing.")
            if round_.reasoning_blocks:
                messages[match_index]["reasoning_content"] = "".join(
                    round_.reasoning_blocks
                )
            cursor = match_index + 1


def _find_owner(
    messages: Sequence[Mapping[str, Any]],
    *,
    assistant_content: str,
    call_ids: tuple[str, ...],
    start: int,
) -> int | None:
    for index in range(start, len(messages)):
        message = messages[index]
        if (
            message.get("role") != "assistant"
            or message.get("content") != assistant_content
        ):
            continue
        raw_calls = message.get("tool_calls", ())
        ids = tuple(call.get("id") for call in raw_calls if isinstance(call, Mapping))
        if ids == call_ids:
            return index
    return None


def _validate_sampler(record: ProviderRecord, name: str, value: object) -> None:
    if value is None:
        return
    validators = _validators_for(record)
    if name not in record.payload_flags:
        raise validators.bad_request(f"{record.display_name} {name} is unsupported.")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise validators.bad_request(f"{record.display_name} {name} is invalid.")


def _validate_penalty(record: ProviderRecord, name: str, value: object) -> float:
    """Validate one OpenAI penalty in [-2, 2], flag-gated (Task 6)."""
    validators = _validators_for(record)
    _require_payload_flag(record, name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not -2 <= float(value) <= 2
    ):
        raise validators.bad_request(f"{record.display_name} {name} is invalid.")
    return float(value)


def _validate_logit_bias(
    record: ProviderRecord, value: object
) -> dict[str, float]:
    """Validate an OpenAI logit_bias mapping, flag-gated (Task 6)."""
    validators = _validators_for(record)
    _require_payload_flag(record, "logit_bias")
    if not isinstance(value, Mapping) or not validators.json_shape_is_bounded(value):
        raise validators.bad_request(f"{record.display_name} logit_bias is invalid.")
    normalized: dict[str, float] = {}
    for token, bias in value.items():
        if (
            not isinstance(token, str)
            or not token
            or isinstance(bias, bool)
            or not isinstance(bias, (int, float))
            or not math.isfinite(float(bias))
            or not -100 <= float(bias) <= 100
        ):
            raise validators.bad_request(
                f"{record.display_name} logit_bias is invalid."
            )
        normalized[token] = float(bias)
    return normalized


def _bounded_identifier(record: ProviderRecord, name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise _validators_for(record).bad_request(
            f"{record.display_name} {name} is invalid."
        )
    return value


# --- response layer (Task 6; ports zai.py's response-side wrappers) ---


class HostedPresetFinishPolicy:
    """Validate preset finishes and disposition-gated reasoning content.

    Ports ``ZAIFinishPolicy`` with Z.ai's literal sets replaced by the
    record's preset data: provider-terminal reasons raise a safe provider
    error (record identity, 502, reason text never in the message), and the
    allowed terminal set is ``record.finish_terminal``.
    """

    reasoning_disposition: ReasoningDisposition

    def __init__(self, record: ProviderRecord) -> None:
        self._record = record
        # Registry data is maintainer-authored and typed ``str``; cast to
        # the protocol's literal. An unknown value behaves like the
        # conservative "proprietary" path (kept private, never displayed).
        self.reasoning_disposition = cast(
            ReasoningDisposition, record.reasoning_disposition
        )

    def validate_finish(
        self,
        *,
        finish_reason: object,
        has_text: bool,
        has_calls: bool,
    ) -> str:
        record = self._record
        if finish_reason in record.finish_provider_errors:
            raise ChatProviderError(
                provider=record.key,
                message=(
                    f"{record.display_name} ended the request with a "
                    "provider terminal error."
                ),
                status_code=502,
            )
        if finish_reason not in record.finish_terminal:
            raise HostedChatProtocolError(
                f"{record.display_name} finish state is malformed."
            )
        if finish_reason == "tool_calls":
            if not has_calls:
                raise HostedChatProtocolError(
                    f"{record.display_name} finish state is inconsistent."
                )
        elif has_calls or (
            not has_text and not record.tolerant_response_extras
        ):
            # Tolerant profile (custom family, ADR-179 Phase 2): stop/length
            # with empty text and no calls is the legacy empty reply -- an
            # empty-text turn, not a failure. stop/length WITH calls and
            # tool_calls WITHOUT calls still fail closed above/here.
            raise HostedChatProtocolError(
                f"{record.display_name} finish state is inconsistent."
            )
        return cast(str, finish_reason)

    def validate_reasoning_content(self, value: object) -> str | None:
        if self.reasoning_disposition == "ignored":
            return None
        if value is None:
            return None
        if not isinstance(value, str):
            raise HostedChatProtocolError(
                f"{self._record.display_name} reasoning content is malformed."
            )
        return value


def normalize_hosted_provider_response(
    record: ProviderRecord, response: object
) -> HostedChatTurn:
    """Normalize one official-shaped preset response through the hosted boundary.

    Ports ``normalize_zai_response``: deep-copy, coerce dict tool arguments
    to deterministic JSON strings (some gateways return dict arguments),
    then delegate to ``normalize_hosted_chat_response`` with the record's
    finish policy and tolerated extra keys. Protocol failures map to a
    display-name provider error (502).

    Args:
        record: Registry preset driving finish policy, error copy, and the
            tolerated extra response keys.
        response: Raw provider response; never mutated.

    Returns:
        The normalized assistant turn.

    Raises:
        ChatProviderError: When the successful response is malformed, or the
            preset declares its finish reason a provider terminal error.
    """
    validators = _validators_for(record)
    safe = deepcopy(response)
    try:
        if isinstance(safe, Mapping):
            choices = safe.get("choices")
            if isinstance(choices, Sequence) and not isinstance(
                choices, (str, bytes)
            ):
                for choice in choices:
                    if not isinstance(choice, Mapping):
                        continue
                    message = choice.get("message")
                    if not isinstance(message, Mapping):
                        continue
                    calls = message.get("tool_calls")
                    if not isinstance(calls, Sequence) or isinstance(
                        calls, (str, bytes)
                    ):
                        continue
                    for call in calls:
                        if not isinstance(call, dict):
                            continue
                        function = call.get("function")
                        if not isinstance(function, dict):
                            continue
                        arguments = function.get("arguments")
                        if isinstance(arguments, Mapping):
                            if not validators.json_shape_is_bounded(arguments):
                                raise HostedChatProtocolError(
                                    f"{record.display_name} tool arguments "
                                    "are malformed."
                                )
                            function["arguments"] = json.dumps(
                                arguments,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            )
                        elif not isinstance(arguments, str):
                            raise HostedChatProtocolError(
                                f"{record.display_name} tool arguments "
                                "are malformed."
                            )
        return normalize_hosted_chat_response(
            safe,
            finish_policy=HostedPresetFinishPolicy(record),
            allowed_extra_keys=record.response_allowances,
            allowed_choice_keys=record.choice_allowances,
            allowed_message_keys=record.message_allowances,
            tolerant_top_level_extras=record.tolerant_response_extras,
        )
    except ChatProviderError:
        raise
    except HostedChatProtocolError:
        raise ChatProviderError(
            provider=record.key,
            message=(
                f"{record.display_name} returned a malformed successful "
                "response."
            ),
            status_code=502,
        ) from None


class HostedProviderStream(Iterator[dict[str, Any]]):
    """Expose visible preset chunks while retaining private terminal state.

    Ports ``ZAIStream``. ``reasoning_content`` is stripped from visible
    deltas only when the record's disposition is not ``"displayable"``
    (zai strips unconditionally); reasoning and control frames still get an
    explicit empty ``content`` so generic consumers do not render fallback
    diagnostics for them.
    """

    def __init__(
        self,
        stream: HostedChatStream,
        *,
        record: ProviderRecord,
        resolution: HostedProviderResolution | None = None,
        provider_continuations: Sequence[ProviderContinuationCheckpoint] = (),
    ) -> None:
        self._stream = stream
        self._record = record
        self._resolution = resolution
        self._provider_continuations = tuple(provider_continuations)
        self._reasoning_visible = record.reasoning_disposition == "displayable"

    def __iter__(self) -> HostedProviderStream:
        return self

    def __next__(self) -> dict[str, Any]:
        event = deepcopy(next(self._stream))
        for choice in event.get("choices", ()):
            if isinstance(choice, dict) and isinstance(choice.get("delta"), dict):
                delta = choice["delta"]
                if not self._reasoning_visible:
                    delta.pop("reasoning_content", None)
                # Reasoning and control frames have no visible content. Keep
                # that explicit so generic consumers do not render fallback
                # diagnostics for each private reasoning token.
                if delta.get("content") is None and not delta.get("tool_calls"):
                    delta["content"] = ""
        return event

    @property
    def terminal_turn(self) -> HostedChatTurn:
        """Return terminal state after clean stream exhaustion."""
        return self._stream.terminal_turn

    @property
    def provider_continuation(self) -> ProviderContinuationCheckpoint | None:
        """Return a canonical candidate only after clean stream exhaustion."""
        if self._resolution is None:
            raise HostedChatProtocolError(
                f"{self._record.display_name} stream metadata is incomplete."
            )
        return _hosted_continuation_candidate(
            self.terminal_turn,
            record=self._record,
            resolution=self._resolution,
            provider_continuations=self._provider_continuations,
        )

    def close(self) -> None:
        """Close the owned underlying stream."""
        self._stream.close()


class HostedProviderResponse(dict[str, Any]):
    """Public response mapping with terminal state kept out of the mapping."""

    def __init__(
        self,
        value: Mapping[str, Any],
        *,
        terminal_turn: HostedChatTurn,
        provider_continuation: ProviderContinuationCheckpoint | None,
    ) -> None:
        super().__init__(value)
        self._terminal_turn = terminal_turn
        self._provider_continuation = provider_continuation

    @property
    def terminal_turn(self) -> HostedChatTurn:
        return self._terminal_turn

    @property
    def provider_continuation(self) -> ProviderContinuationCheckpoint | None:
        return self._provider_continuation


def _hosted_continuation_candidate(
    turn: HostedChatTurn,
    *,
    record: ProviderRecord,
    resolution: HostedProviderResolution,
    provider_continuations: Sequence[ProviderContinuationCheckpoint],
) -> ProviderContinuationCheckpoint | None:
    """Build the next canonical continuation checkpoint for one turn.

    Ports ``_zai_continuation_candidate`` with Z.ai's identity strings
    replaced by the record's key and continuation protocol. A record that
    ships no continuation protocol gets no candidate at all (skip, not an
    error). The final canonical parse round-trip admits the continuation
    ``_PAIRINGS`` providers (moonshot/zai/deepseek/databricks).
    """
    if record.continuation_protocol is None:
        return None
    active = tuple(
        checkpoint
        for checkpoint in provider_continuations
        if checkpoint.state == "active"
    )
    if len(active) > 1:
        raise HostedChatProtocolError(
            f"{record.display_name} continuation state is ambiguous."
        )
    current = active[0] if active else None
    protocol = record.continuation_protocol or "chat_completions"
    if turn.tool_calls:
        round_ = ContinuationRound(
            assistant_content=turn.text,
            reasoning_blocks=(turn.reasoning_content,)
            if turn.reasoning_content is not None
            else (),
            calls=tuple(
                ContinuationCall(
                    call_id=cast(str, call["id"]),
                    name=cast(str, call["function"]["name"]),
                    arguments=cast(str, call["function"]["arguments"]),
                    state="pending",
                )
                for call in turn.tool_calls
            ),
        )
        candidate = ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=(current.checkpoint_revision + 1 if current else 1),
            provider=record.key,
            protocol=protocol,
            model=resolution.model,
            api_base_url=resolution.base_url,
            state="active",
            rounds=((*current.rounds, round_) if current else (round_,)),
        )
    elif current is not None:
        candidate = ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=current.checkpoint_revision + 1,
            provider=record.key,
            protocol=protocol,
            model=resolution.model,
            api_base_url=resolution.base_url,
            state="complete",
            rounds=current.rounds,
        )
    else:
        return None
    return parse_provider_continuation_json(
        dump_provider_continuation_json(candidate)
    )


# --- handler layer (Task 7; ports zai.py's chat_with_zai joining flow) ---


def resolve_hosted_engine_request(
    record: ProviderRecord,
    *,
    explicit_api_key: object = None,
    explicit_base_url: object = None,
    explicit_model: object = None,
    explicit_timeout: object = None,
    explicit_retries: object = None,
    explicit_retry_delay: object = None,
    api_key_resolved: bool | None = None,
    app_config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> HostedProviderResolution:
    """Resolve one engine request through the public engine seam.

    Thin public alias of :func:`resolve_hosted_request` for service callers
    (the model-catalog service, Task 12): they resolve one engine-driven
    provider's endpoint and credential exactly the chat path does, without
    depending on the private resolution symbol.

    Args:
        record: Registry preset driving every default, env candidate, and
            error message.
        explicit_api_key: Caller-supplied credential; wins over settings/env.
        explicit_base_url: Caller-supplied base URL; wins over settings.
        explicit_model: Caller-supplied model; wins over settings.
        explicit_timeout: Caller-supplied timeout override.
        explicit_retries: Caller-supplied retries override.
        explicit_retry_delay: Caller-supplied retry-delay override.
        api_key_resolved: Whether a trusted caller already made the final
            credential decision (see :func:`resolve_hosted_request`).
        app_config: Canonical config mapping; defaults to the runtime
            snapshot. Never mutated.
        environ: Environment mapping; defaults to ``os.environ``.

    Returns:
        The immutable resolved request identity and transport policy.

    Raises:
        ChatConfigurationError: On any missing or malformed credential,
            base URL, model, or transport setting.
    """
    return resolve_hosted_request(
        record,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
        explicit_model=explicit_model,
        explicit_timeout=explicit_timeout,
        explicit_retries=explicit_retries,
        explicit_retry_delay=explicit_retry_delay,
        api_key_resolved=api_key_resolved,
        app_config=app_config,
        environ=environ,
    )


def build_hosted_chat_handler(
    record: ProviderRecord,
) -> Callable[..., dict[str, Any] | HostedProviderStream]:
    """Build one provider's ``chat_with_*`` handler from its preset record.

    The returned callable mirrors ``chat_with_zai``'s signature minus the
    provider-invented ``do_sample`` and ``request_id`` parameters, joins the
    Task 4-6 layers (resolve -> payload -> owned transport -> response
    normalization), and emits the per-provider metric counters of the
    ``LLM_API_Calls`` compatibility wrappers with ``record.key`` identity.

    Args:
        record: Registry preset driving every layer of the request.

    Returns:
        The chat handler Task 8 registers in dispatch for this provider.
    """

    def chat_with_hosted_provider(
        input_data: list[dict[str, Any]],
        model: str | None = None,
        api_key: str | None = None,
        system_message: str | None = None,
        temp: float | None = None,
        maxp: float | None = None,
        minp: float | None = None,
        topk: int | None = None,
        streaming: bool | None = False,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        custom_prompt_arg: str | None = None,
        api_base_url: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: str | list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        user_identifier: str | None = None,
        seed: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        reasoning_effort: str | None = None,
        thinking_budget_tokens: int | None = None,
        provider_continuations: Sequence[ProviderContinuationCheckpoint] = (),
        request_timeout: float | None = None,
        request_retries: int | None = None,
        request_retry_delay: float | None = None,
        api_key_resolved: bool | None = None,
    ) -> dict[str, Any] | HostedProviderStream:
        del custom_prompt_arg
        # api_key_resolved: the Console gateway passes it after making the
        # credential decision itself (keyed OR explicitly keyless); it is
        # threaded into resolution so the supplied ``api_key`` is the exact
        # credential and the settings/env chain never back-fills a global
        # key onto a caller-controlled endpoint URL.
        started_at = time.time()
        labels = {"model": model or "configured", "streaming": str(bool(streaming))}
        log_counter(f"{record.key}_api_request", labels=labels)
        try:
            result = _send_hosted_chat_request(
                record,
                input_data=input_data,
                model=model,
                api_key=api_key,
                api_key_resolved=api_key_resolved,
                system_message=system_message,
                temp=temp,
                maxp=maxp,
                minp=minp,
                topk=topk,
                streaming=streaming,
                max_tokens=max_tokens,
                tools=tools,
                api_base_url=api_base_url,
                tool_choice=tool_choice,
                stop=stop,
                response_format=response_format,
                user=user if user is not None else user_identifier,
                reasoning_effort=reasoning_effort,
                thinking_budget_tokens=thinking_budget_tokens,
                seed=seed,
                n=n,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                provider_continuations=provider_continuations,
                request_timeout=request_timeout,
                request_retries=request_retries,
                request_retry_delay=request_retry_delay,
            )
        except Exception as exc:
            log_counter(
                f"{record.key}_api_error",
                labels={**labels, "error_type": type(exc).__name__},
            )
            log_histogram(
                f"{record.key}_api_error_response_time",
                time.time() - started_at,
                labels=labels,
            )
            raise
        log_counter(f"{record.key}_api_success", labels=labels)
        log_histogram(
            f"{record.key}_api_response_time",
            time.time() - started_at,
            labels=labels,
        )
        return result

    return chat_with_hosted_provider


def _send_hosted_chat_request(
    record: ProviderRecord,
    *,
    input_data: list[dict[str, Any]],
    model: str | None,
    api_key: str | None,
    api_key_resolved: bool | None,
    system_message: str | None,
    temp: float | None,
    maxp: float | None,
    minp: float | None,
    topk: int | None,
    streaming: bool | None,
    max_tokens: int | None,
    tools: list[dict[str, Any]] | None,
    api_base_url: str | None,
    tool_choice: str | dict[str, Any] | None,
    stop: str | list[str] | None,
    response_format: dict[str, Any] | None,
    user: str | None,
    reasoning_effort: str | None,
    thinking_budget_tokens: int | None,
    seed: int | None,
    n: int | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
    logit_bias: dict[str, float] | None,
    logprobs: bool | None,
    top_logprobs: int | None,
    provider_continuations: Sequence[ProviderContinuationCheckpoint],
    request_timeout: float | None,
    request_retries: int | None,
    request_retry_delay: float | None,
) -> dict[str, Any] | HostedProviderStream:
    """Run one engine request end to end (the ``chat_with_zai`` body)."""
    resolution = resolve_hosted_request(
        record,
        explicit_api_key=api_key,
        explicit_base_url=api_base_url,
        explicit_model=model,
        explicit_timeout=request_timeout,
        explicit_retries=request_retries,
        explicit_retry_delay=request_retry_delay,
        api_key_resolved=api_key_resolved,
    )
    payload = build_hosted_chat_payload(
        record,
        resolution=resolution,
        messages_payload=input_data,
        system_message=system_message,
        streaming=streaming,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
        provider_continuations=provider_continuations,
        temperature=temp,
        top_p=maxp,
        min_p=minp,
        top_k=topk,
        max_tokens=max_tokens,
        stop=stop,
        response_format=response_format,
        seed=seed,
        n=n,
        user=user,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        thinking_budget_tokens=thinking_budget_tokens,
    )
    try:
        raw = owned_json_post(
            config=HostedHTTPTransportConfig(
                provider=record.key,
                base_url=resolution.base_url,
                api_key=resolution.api_key,
                timeout=resolution.timeout,
                retries=resolution.retries,
                retry_delay=resolution.retry_delay,
                auth_scheme=record.auth_scheme,
            ),
            route="chat/completions",
            payload=payload,
            streaming=cast(bool, payload["stream"]),
        )
        if payload["stream"]:
            return HostedProviderStream(
                HostedChatStream(
                    cast(Iterator[Any], raw),
                    finish_policy=HostedPresetFinishPolicy(record),
                    allowed_extra_keys=record.response_allowances,
                    allowed_choice_keys=record.choice_allowances,
                    allowed_message_keys=record.message_allowances,
                    tolerant_top_level_extras=record.tolerant_response_extras,
                ),
                record=record,
                resolution=resolution,
                provider_continuations=provider_continuations,
            )
        turn = normalize_hosted_provider_response(record, raw)
    except ChatProviderError:
        raise
    except HostedChatProtocolError:
        raise ChatProviderError(
            provider=record.key,
            message=(
                f"{record.display_name} returned a malformed successful "
                "response."
            ),
            status_code=502,
        ) from None
    return _turn_response(
        turn,
        record=record,
        resolution=resolution,
        provider_continuations=provider_continuations,
    )


def _turn_response(
    turn: HostedChatTurn,
    *,
    record: ProviderRecord,
    resolution: HostedProviderResolution,
    provider_continuations: Sequence[ProviderContinuationCheckpoint],
) -> HostedProviderResponse:
    message = deepcopy(turn.assistant_message)
    if message is None:
        raise HostedChatProtocolError(
            f"{record.display_name} response message is incomplete."
        )
    if record.reasoning_disposition != "displayable":
        # zai pops unconditionally (its disposition is proprietary); the
        # engine keeps reasoning in the public response message only where
        # the record declares it displayable. "ignored" turns carry none at
        # all (the finish policy dropped them), so the pop is a no-op there.
        message.pop("reasoning_content", None)
    response: dict[str, Any] = {
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": turn.finish_reason,
            }
        ]
    }
    if turn.usage is not None:
        response["usage"] = deepcopy(turn.usage)
    return HostedProviderResponse(
        response,
        terminal_turn=turn,
        provider_continuation=_hosted_continuation_candidate(
            turn,
            record=record,
            resolution=resolution,
            provider_continuations=provider_continuations,
        ),
    )
