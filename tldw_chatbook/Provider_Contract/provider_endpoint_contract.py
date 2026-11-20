"""Pure provider endpoint interpretation for OpenAI-compatible APIs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from ipaddress import ip_address
from typing import Literal
from unicodedata import category
from urllib.parse import urlparse, urlsplit, urlunparse, urlunsplit

from tldw_chatbook.provider_registry import (
    ALL_RECORDS,
    CUSTOM_HOSTED,
    RECORDS_BY_KEY,
    ProviderRecord,
)

EndpointForm = Literal[
    "origin",
    "api_base",
    "chat_url",
    "models_url",
    "legacy_local",
]

_LLAMA_PROVIDER_KEYS = frozenset({"llama_cpp", "local_llamacpp"})
#: Providers whose existing chat endpoint contract exposes a bounded models
#: route suitable for an explicit, non-generating connection check.
URL_BASED_PROVIDER_KEYS = frozenset(
    {
        "aphrodite",
        "custom",
        "custom_2",
        "koboldcpp",
        "llama_cpp",
        "strata",
        "polystrata",
        "local_llamacpp",
        "local_llamafile",
        "local_ollama",
        "local_vllm",
        "ollama",
        "oobabooga",
        "qwencloud",
        "tabbyapi",
        "vllm",
    }
)
_PROVIDER_ALIASES = {
    "Custom": "custom",
    "Custom OpenAI": "custom",
    "Custom OpenAI API": "custom",
    "custom-openai": "custom",
    "custom_openai": "custom",
    "custom-openai-api": "custom",
    "custom_openai_api": "custom",
    "Custom-2": "custom_2",
    "Custom 2": "custom_2",
    "custom-2": "custom_2",
    "Custom OpenAI 2": "custom_2",
    "custom-openai-2": "custom_2",
    "custom_openai_2": "custom_2",
    "Custom OpenAI API-2": "custom_2",
    "Custom OpenAI API 2": "custom_2",
    "custom-openai-api-2": "custom_2",
    "custom_openai_api_2": "custom_2",
    # ADR-179 Phase 2 Task 6: the engine-driven custom-endpoint execution key
    # is hyphenated (the registry ``record.key`` that continuation
    # checkpoints and restore targets pin); accept it alongside the legacy
    # custom spellings so its connection identity resolves.
    "custom-hosted": "custom_hosted",
    "Custom Hosted": "custom_hosted",
    "llama.cpp": "llama_cpp",
    "local llama.cpp": "local_llamacpp",
    "OpenRouter": "openrouter",
    "local-llamacpp": "local_llamacpp",
}
#: Engine presets resolve around their own API base (TASK-33510);
#: ``custom-hosted`` is an execution key, never a configured provider.
_ENGINE_PRESETS = {
    record.key: record
    for record in ALL_RECORDS
    if record.engine_driven and record.key != "custom-hosted"
}
#: TASK-41700 (owner decision 2026-11-18, "Allow gateway paths"): a
#: first-party cloud adapter appends its route to its configured base, so a
#: base whose ``v1`` is not last (a Cloudflare AI Gateway
#: ``/v1/<account>/<gateway>/openai``, DeepInfra's ``/v1/openai``) is read
#: as its API base, as the custom-hosted engine reads one.
_CLOUD_BASE_KEYS = frozenset(
    key
    for key, record in RECORDS_BY_KEY.items()
    if record.classification == "cloud" and not record.engine_driven
)
#: TASK-41704: Cohere's adapter appends its whole versioned route
#: (``/v2/chat``), so a bare host is its base on any host, not ``<host>/v1``.
#: DeepSeek's bare host is its base only on its own host (review r17cu-rev1
#: M-1: an OpenAI-compatible proxy under its key may serve only ``/v1``).
_BARE_BASE_ANY_HOST_KEYS = frozenset({"cohere"})
#: TASK-42002 (owner decision 2026-11-19, "Migrate on load"): the paths an
#: earlier Settings saved with ``/v1`` wrongly appended to a first-party base.
#: On the provider's own host each reads as its documented path, so a save
#: writes that base (TASK-42019) and config load rewrites a saved one
#: (``config._migrate_v1_appended_provider_bases``). Google's ``/v1`` is a
#: real Gemini API version, so it is not here.
_V1_APPENDED_PATHS = {
    "google": ("v1beta", "v1"),
    "cohere": ("v1",),
    "groq": ("v1",),
    "openrouter": ("v1",),
    "qwencloud": ("v1",),
}


#: TASK-34100.6 (review provider-20 / task-33621.32): a preset whose saved base
#: already carries its API version owns its route suffix, so it resolves like
#: an engine preset (``<base>/chat/completions``). Z.ai's ``/api/paas/v4``
#: base previewed and saved ``…/v4/v1/chat/completions`` under the generic rule.
#: The setup facts' ``ROUTE_SUFFIXES`` is the one owner (review round 1, F11).
@lru_cache(maxsize=1)
def _route_owning_presets() -> dict[str, ProviderRecord]:
    """Load setup facts when interpreting an endpoint, keeping imports pure."""
    from tldw_chatbook.Provider_Contract.provider_setup_facts import ROUTE_SUFFIXES

    return {
        record.key: record for record in ALL_RECORDS if record.key in ROUTE_SUFFIXES
    }


#: Cloudflare Workers AI's OpenAI-compatible base is ``…/accounts/<id>/ai/v1``
#: (record ``CLOUDFLARE``, which carries no suffix). The account URL alone has
#: no chat route, and Workers AI has no listing to catch it before the first
#: chat 404s, so a URL that stops at the account gets the route (TASK-34100.6
#: pre-merge review 2026-10-29).
_CLOUDFLARE_ACCOUNT_ROUTE = ("ai", "v1")
#: ``htp:/x`` or ``http:/x``: a scheme-like word and a colon not followed by a
#: port, typed without "//" (review provider-15).
_SCHEME_TYPO = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:(?!\d)")
_SCHEME_HINT = (
    "Start the address with http:// or https:// (for example http://127.0.0.1:8080)."
)
_REMOTE_HTTP_WARNING = "Remote HTTP endpoints are not encrypted."
#: The refusal of a remote endpoint typed without a scheme; Console's Endpoint
#: line reads it to say what to add (TASK-39151).
REMOTE_ENDPOINT_SCHEME_REQUIRED = (
    "Remote endpoint URLs must include an HTTP or HTTPS scheme."
)
#: The refusal of a path that names a request route the adapter adds to its
#: base, such as ``/responses`` or Anthropic's ``/v1/messages`` (review
#: r16cu-rev3: it is not ambiguous, so it no longer says so).
ENDPOINT_ROUTE_REFUSAL = "Endpoint path names a request route; enter the API base."
_PROVIDER_KEY = re.compile(r"[a-z0-9_]+")
_UNSAFE_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
#: One part ``inet_aton`` reads as a number: decimal, octal (leading 0) or
#: hex (``0x``, digits optional).
_INET_ATON_PART = re.compile(r"0[xX][0-9a-fA-F]*|[0-9]+")
_UNRESERVED_ASCII = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
_MAX_PROVIDER_LENGTH = 128
_MAX_ENDPOINT_LENGTH = 4096
#: KoboldAI's native generate route, koboldcpp's template endpoint. It lists
#: nothing itself; koboldcpp serves its OpenAI-compatible ``/v1/models`` at
#: the same root (TASK-35010, owner ruling 2026-10-20).
_KOBOLD_NATIVE_GENERATE = ("api", "v1", "generate")
_KOBOLD_NATIVE_PATH = "/" + "/".join(_KOBOLD_NATIVE_GENERATE)
_SUFFIXES: tuple[tuple[tuple[str, ...], EndpointForm], ...] = (
    (("v1", "chat", "completions"), "chat_url"),
    (("v1", "models"), "models_url"),
    (("completion",), "legacy_local"),
    (("v1",), "api_base"),
)


@dataclass(frozen=True, slots=True)
class ProviderEndpointResolution:
    """Normalized endpoint URLs and safe display values for one provider."""

    provider_key: str
    normalized_input: str
    persisted_endpoint: str | None
    chat_url: str | None
    models_url: str | None
    persisted_display: str
    chat_display: str
    models_display: str
    form: EndpointForm | None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class ConnectionProbeAvailability(StrEnum):
    """Whether the provider draft has a meaningful non-generating probe."""

    MODELS_ROUTE = "models_route"
    UNAVAILABLE = "unavailable"


def connection_probe_availability(
    provider: str,
    endpoint: str | None,
) -> ConnectionProbeAvailability:
    """Return probe availability without reading config or performing I/O.

    Args:
        provider: Provider identifier for the draft connection.
        endpoint: Optional endpoint entered for that provider.

    Returns:
        Whether the provider exposes a valid non-generating models route.
    """

    provider_key = normalize_provider_key_for_contract(provider)
    if provider_key not in URL_BASED_PROVIDER_KEYS:
        return ConnectionProbeAvailability.UNAVAILABLE
    resolution = resolve_provider_endpoint(provider_key, endpoint)
    if resolution.errors or resolution.models_url is None:
        return ConnectionProbeAvailability.UNAVAILABLE
    return ConnectionProbeAvailability.MODELS_ROUTE


def resolve_provider_endpoint(
    provider: str, value: object
) -> ProviderEndpointResolution:
    """Interpret one endpoint value without reading config or performing I/O.

    TASK-33005.10: memoized. With shared evidence present, every Console
    readiness build keys its connection, which parsed the endpoint three more
    times: a warm build went from 10 to 34 us (llama.cpp) and 25 to 53 us
    (OpenAI), on a path TASK-18909 counted ~400 builds per warm switch. With
    the memo it is 15 and 31 us. The result is a frozen pure function of two
    strings. A value that could carry a secret (user info, a query, a
    fragment) is refused anyway and never kept as a cache key.
    """

    if (
        type(provider) is str
        and type(value) is str
        and "@" not in value
        and "?" not in value
        and "#" not in value
    ):
        return _memoized_resolution(provider, value)
    return _resolve_provider_endpoint(provider, value)


def _resolve_provider_endpoint(
    provider: str, value: object
) -> ProviderEndpointResolution:
    if not isinstance(provider, str) or len(provider) > _MAX_PROVIDER_LENGTH:
        return _invalid_resolution("", "Provider name is invalid or too long.")
    provider_key = normalize_provider_key_for_contract(provider)
    if not provider_key:
        return _invalid_resolution("", "Select a valid provider.")
    if not isinstance(value, str):
        return _invalid_resolution(provider_key, "Enter an endpoint URL.")
    if len(value) > _MAX_ENDPOINT_LENGTH:
        return _invalid_resolution(provider_key, "Endpoint URL is too long.")
    if _contains_unsafe_unicode(value):
        return _invalid_resolution(
            provider_key, "Endpoint URL contains invalid characters."
        )

    raw_value = value.strip()
    if not raw_value:
        return _invalid_resolution(provider_key, "Enter an endpoint URL.")
    if any(character.isspace() for character in raw_value):
        return _invalid_resolution(
            provider_key, "Endpoint URL contains invalid characters."
        )
    if "?" in raw_value:
        return _invalid_resolution(
            provider_key, "Endpoint URL must not include a query."
        )
    if "#" in raw_value:
        return _invalid_resolution(
            provider_key, "Endpoint URL must not include a fragment."
        )

    has_explicit_scheme = "://" in raw_value
    if not has_explicit_scheme and _SCHEME_TYPO.match(raw_value):
        return _invalid_resolution(provider_key, _SCHEME_HINT)
    candidate = raw_value if has_explicit_scheme else f"http://{raw_value}"
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return _invalid_resolution(provider_key, "Endpoint URL is malformed.")

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return _invalid_resolution(provider_key, _SCHEME_HINT)
    if (
        parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
    ):
        return _invalid_resolution(
            provider_key, "Endpoint URL must not include user information."
        )
    if parsed.netloc.endswith(":"):
        return _invalid_resolution(
            provider_key, "Endpoint URL must not include an empty port."
        )
    if parsed.query:
        return _invalid_resolution(
            provider_key, "Endpoint URL must not include a query."
        )
    if parsed.fragment:
        return _invalid_resolution(
            provider_key, "Endpoint URL must not include a fragment."
        )
    canonical_host = canonical_endpoint_host(hostname) if hostname else None
    if not canonical_host or port == 0:
        return _invalid_resolution(
            provider_key, "Endpoint URL must include a valid host and port."
        )
    if not has_explicit_scheme and not is_exact_schemeless_host(parsed.netloc):
        return _invalid_resolution(provider_key, REMOTE_ENDPOINT_SCHEME_REQUIRED)

    original_path = parsed.path
    if "//" in original_path or "\\" in original_path:
        return _invalid_resolution(provider_key, "Endpoint path is ambiguous.")
    canonical_path = _canonicalize_path(original_path)
    if canonical_path is None:
        return _invalid_resolution(
            provider_key, "Endpoint path contains invalid percent encoding."
        )

    path = canonical_path.removesuffix("/")
    segments = tuple(segment for segment in path.split("/") if segment)
    if any(segment in {".", ".."} for segment in segments):
        return _invalid_resolution(provider_key, "Endpoint path is ambiguous.")
    segments = _read_v1_appended_path(provider_key, canonical_host, segments)
    if provider_key == "qwencloud" and segments[-1:] == ("models",):
        # TASK-42021: QwenCloud's adapter refuses a /models base before it
        # strips /responses, so Console read Ready for a base it refused.
        return _invalid_resolution(provider_key, ENDPOINT_ROUTE_REFUSAL)
    if provider_key == "qwencloud" and segments[-1:] == ("responses",):
        # TASK-41722: QwenCloud's adapter strips its own /responses from a
        # pasted endpoint (normalize_qwencloud_base_url), so this reads the
        # base before it too; Console refused what the adapter sent.
        segments = segments[:-1]
    if _names_a_native_route(provider_key, segments):
        return _invalid_resolution(provider_key, ENDPOINT_ROUTE_REFUSAL)

    netloc = _normalized_netloc(canonical_host, port)
    preset = _ENGINE_PRESETS.get(provider_key) or _route_owning_presets().get(
        provider_key
    )
    if preset is not None:
        # TASK-42020: a route is not a preset's base either (Z.ai's
        # .../v4/responses was sent as .../responses/chat/completions).
        if _names_a_route(segments):
            return _invalid_resolution(provider_key, ENDPOINT_ROUTE_REFUSAL)
        return _engine_preset_resolution(
            preset, provider_key, scheme, netloc, canonical_host, segments
        )

    native = _is_kobold_native_generate(provider_key, segments)
    suffix = _terminal_suffix(
        segments[:-3] if native else segments,
        allow_legacy_local=provider_key in _LLAMA_PROVIDER_KEYS,
    )
    if native:
        # Only from a plain root: an API suffix before the route is ambiguous.
        suffix = (suffix[0], "chat_url") if suffix and suffix[1] == "origin" else None
    cloud = provider_key in _CLOUD_BASE_KEYS
    documented = _documented_own_path(provider_key, canonical_host) if cloud else None
    bare_base = provider_key in _BARE_BASE_ANY_HOST_KEYS or documented == ()
    if (suffix is None and (provider_key == "custom_hosted" or cloud)) or (
        # TASK-41704: nor is a cloud path with no v1 an origin to append /v1
        # to (Google's own .../v1beta was saved as .../v1beta/v1); and a
        # route there is refused (TASK-41727: .../responses was accepted).
        cloud
        and suffix is not None
        and suffix[1] == "origin"
        # Review r17cu-rev2: the provider's own bare host is completed to its
        # documented base (Groq's /openai/v1), not given /v1.
        and (segments or bare_base or documented is not None)
    ):
        # TASK-40300, owner decision 2026-11-13: an OpenAI-compatible custom
        # endpoint's path is its API base even when its v1 is not the last
        # segment (DeepInfra /v1/openai, Cloudflare /v1/<acct>/<gw>/compat),
        # read as its custom-hosted engine reads it. A route is not a base
        # (chat would post to .../completions/chat/completions) and stays
        # refused, with the route reason (review r16cu-rev3).
        # TASK-41700: so is a first-party cloud provider's gateway base.
        if _names_a_route(segments):
            return _invalid_resolution(provider_key, ENDPOINT_ROUTE_REFUSAL)
        return _engine_preset_resolution(
            CUSTOM_HOSTED,
            provider_key,
            scheme,
            netloc,
            canonical_host,
            segments,
            bare_host_base=bare_base,
            documented_path=documented,
        )
    if suffix is None:
        return _invalid_resolution(
            provider_key, "Endpoint path has ambiguous API suffixes."
        )
    prefix_segments, form = suffix

    prefix_path = f"/{'/'.join(prefix_segments)}" if prefix_segments else ""
    root = urlunsplit((scheme, netloc, prefix_path, "", ""))
    normalized_path = f"/{'/'.join(segments)}" if segments else ""
    normalized_input = urlunsplit((scheme, netloc, normalized_path, "", ""))
    # A koboldcpp native endpoint is the URL its send posts to: kept whole.
    chat_url = normalized_input if native else f"{root}/v1/chat/completions"
    models_url = f"{root}/v1/models"
    persisted_endpoint = root if provider_key in _LLAMA_PROVIDER_KEYS else chat_url
    warnings = (
        (_REMOTE_HTTP_WARNING,)
        if scheme == "http" and not _is_loopback_host(canonical_host)
        else ()
    )

    return ProviderEndpointResolution(
        provider_key=provider_key,
        normalized_input=normalized_input,
        persisted_endpoint=persisted_endpoint,
        chat_url=chat_url,
        models_url=models_url,
        persisted_display=persisted_endpoint,
        chat_display=chat_url,
        models_display=models_url,
        form=form,
        warnings=warnings,
    )


def endpoint_refusal(
    provider: object, value: object, *, custom_endpoint: bool = False
) -> str:
    """The contract's reason for refusing an endpoint, or "" when it accepts it.

    TASK-40300: Console's send, readiness, dispatch and custom endpoint
    registry, agent routing and ``chat_with_llama`` all decide with this, so
    none accepts a value the save refuses. TASK-41131: so do the shared local
    transport, KoboldCpp's send and the hosted engine, for every
    ``chat_api_call`` caller; TASK-41522: and the first-party cloud adapters
    and the strict hosted transport. The reason never names any part of the
    value.

    Args:
        provider: Provider key; one the contract cannot name (``local-llm``)
            reads as the generic ``custom``.
        value: The endpoint as typed, stored or resolved.
        custom_endpoint: True for a custom endpoint entry: an
            OpenAI-compatible one reads as its custom-hosted engine reads it
            (owner decision 2026-11-13).

    Returns:
        The first error the contract reports, or "".
    """
    key = normalize_provider_key_for_contract(provider) or "custom"
    if custom_endpoint and key == "custom":
        key = "custom_hosted"
    errors = resolve_provider_endpoint(key, value).errors
    return errors[0] if errors else ""


def _documented_own_path(provider_key: str, host: str) -> tuple[str, ...] | None:
    """A cloud provider's documented path segments when ``host`` is its own.

    Args:
        provider_key: A first-party cloud provider's key.
        host: The canonical host typed.

    Returns:
        The segments of the registry default's path (``()`` for a bare
        documented host), or None for any other host.
    """
    default = urlsplit(RECORDS_BY_KEY[provider_key].default_base_url or "")
    if default.hostname != host:
        return None
    return tuple(segment for segment in default.path.split("/") if segment)


def _read_v1_appended_path(
    provider_key: str, host: str, segments: tuple[str, ...]
) -> tuple[str, ...]:
    """The documented path for a ``_V1_APPENDED_PATHS`` shape on the provider's
    own host; any other path unchanged (TASK-42002).

    The shape may be followed by a tail the reading drops (a pasted
    ``/chat/completions`` or ``/models``, and QwenCloud's ``/responses``);
    the tail is kept after the documented path so the usual reading (or
    TASK-42021's QwenCloud ``/models`` refusal) still applies (review r18-rev1).
    """
    shape = _V1_APPENDED_PATHS.get(provider_key)
    if shape is None or segments[: len(shape)] != shape:
        return segments
    tail = segments[len(shape) :]
    if tail not in {(), ("chat", "completions"), ("models",)} and (
        provider_key != "qwencloud" or tail != ("responses",)
    ):
        return segments
    documented = _documented_own_path(provider_key, host)
    return segments if documented is None else documented + tail


def v1_appended_documented_base(provider: object, value: object) -> str | None:
    """The documented base a value saved with ``/v1`` wrongly appended reads as.

    TASK-42002: config load rewrites such a saved base with this, the rule a
    save resolves through (``_V1_APPENDED_PATHS``).

    Args:
        provider: Provider key.
        value: A saved ``api_base_url``.

    Returns:
        The base a save writes for the value (the documented base on its own
        scheme and host, any pasted tail dropped), or None for any other
        value, including one the contract refuses (user info, a query or a
        fragment never reach the result).
    """
    key = normalize_provider_key_for_contract(provider)
    if key not in _V1_APPENDED_PATHS or type(value) is not str:
        return None
    try:
        parsed = urlsplit(value.strip())
        host = canonical_endpoint_host(parsed.hostname or "")
    except ValueError:
        return None
    segments = tuple(segment for segment in parsed.path.split("/") if segment)
    if host is None or _read_v1_appended_path(key, host, segments) == segments:
        return None
    resolution = resolve_provider_endpoint(key, value)
    if resolution.errors or resolution.chat_url is None:
        return None
    # As the save writes an api_base_url (provider_setup_persistence's
    # _api_base_from_chat_url), so load and save agree on a pasted tail.
    return resolution.chat_url.removesuffix("/chat/completions")


def _names_a_native_route(provider_key: str, segments: tuple[str, ...]) -> bool:
    """Whether a path ends in a native cloud adapter's own route.

    TASK-41700 review r16cu-rev2: the adapter appends that route to its base,
    so a pasted one was sent twice (``.../v1/messages/messages``, a 404).
    Anthropic's ``/messages`` and legacy ``/complete``, Cohere's ``/chat``
    (``/v2/chat``), and any Google segment naming ``:generateContent`` or
    ``:streamGenerateContent``.
    """
    if provider_key == "anthropic":
        return segments[-1:] in {("messages",), ("complete",)}
    if provider_key == "cohere":
        return segments[-1:] == ("chat",)
    if provider_key == "google":
        return any(
            method in segment.casefold().replace("%3a", ":")
            for segment in segments
            for method in (":generatecontent", ":streamgeneratecontent")
        )
    return False


def _names_a_route(segments: tuple[str, ...]) -> bool:
    """Whether a path ends in, or runs past, an API route rather than a base.

    A legacy ``/completions``, ``/chat``, ``/embeddings``, ``/responses``, or
    ``/chat/completions`` with more after it (TASK-40300 review r14ep-rev3).
    """
    if segments[-1:] in {("chat",), ("embeddings",), ("responses",)}:
        return True
    if segments[-1:] == ("completions",) and segments[-2:-1] != ("chat",):
        return True
    return any(
        segments[index : index + 2] == ("chat", "completions")
        for index in range(len(segments) - 2)
    )


def _engine_preset_resolution(
    preset: ProviderRecord,
    provider_key: str,
    scheme: str,
    netloc: str,
    host: str,
    segments: tuple[str, ...],
    *,
    bare_host_base: bool = False,
    documented_path: tuple[str, ...] | None = None,
) -> ProviderEndpointResolution:
    """Resolve an engine preset's endpoint around its own API base.

    Engine presets' bases are not all ``.../v1`` (BytePlus ``/api/v3``, Kilo
    ``/api/gateway``, Qianfan ``/v2``, DeepInfra ``/v1/openai``), so the
    generic ``/v1`` suffix model would rewrite or reject them (TASK-33510).
    The entered path IS the base, minus a pasted ``/chat/completions`` or
    ``/models``; a bare host gets the record's suffix (Azure/Databricks
    ``/openai/v1``), the documented path when it is the preset's own host,
    or ``/v1``; a Cloudflare URL that stops at ``/accounts/<id>`` gets
    ``/ai/v1``.

    Args:
        preset: The provider's registry record.
        provider_key: Canonical provider key.
        scheme: Validated URL scheme.
        netloc: Normalized network location.
        host: Canonical host name.
        segments: Canonical, validated path segments.
        bare_host_base: A bare host is the base itself (TASK-41704: Cohere,
            DeepSeek on its own host).
        documented_path: The provider's documented path when ``host`` is
            its own; a prefix of it, the bare host included, reads as that
            path (reviews r17cu-rev1 I-1 and r17cu-rev2: Groq's
            ``https://api.groq.com/openai`` and ``https://api.groq.com`` are
            ``/openai/v1``).

    Returns:
        The resolution whose chat URL is ``<base>/chat/completions``.
    """
    form: EndpointForm = "api_base"
    if segments[-2:] == ("chat", "completions"):
        segments, form = segments[:-2], "chat_url"
    elif segments[-1:] == ("models",):
        segments, form = segments[:-1], "models_url"
    if provider_key == "cloudflare" and segments[-2:-1] == ("accounts",):
        segments = (*segments, *_CLOUDFLARE_ACCOUNT_ROUTE)
    if documented_path is not None and segments == documented_path[: len(segments)]:
        if not segments:
            form = "origin"
        segments = documented_path
    if not segments:
        form = "origin"
        default = urlsplit(preset.default_base_url or "")
        if bare_host_base:
            base_path = ""
        elif preset.base_url_suffix:
            base_path = preset.base_url_suffix
        elif default.hostname == host and default.path:
            base_path = default.path
        else:
            base_path = "/v1"
        segments = tuple(segment for segment in base_path.split("/") if segment)
    base = urlunsplit((scheme, netloc, "/".join(("", *segments)), "", ""))
    chat_url = f"{base}/chat/completions"
    models_url = f"{base}/models"
    return ProviderEndpointResolution(
        provider_key=provider_key,
        normalized_input=base,
        persisted_endpoint=chat_url,
        chat_url=chat_url,
        models_url=models_url,
        persisted_display=chat_url,
        chat_display=chat_url,
        models_display=models_url,
        form=form,
        warnings=(
            (_REMOTE_HTTP_WARNING,)
            if scheme == "http" and not _is_loopback_host(host)
            else ()
        ),
    )


#: ponytail: 256 entries bounds it; the strings are endpoints, short and few.
_memoized_resolution = lru_cache(maxsize=256)(_resolve_provider_endpoint)


def _is_kobold_native_generate(provider_key: str, segments: tuple[str, ...]) -> bool:
    return provider_key == "koboldcpp" and segments[-3:] == _KOBOLD_NATIVE_GENERATE


#: TASK-36516's refusal, first the wizard's (TASK-35010).
KOBOLDCPP_NATIVE_ROUTE_REFUSAL = (
    "KoboldCpp needs its native /api/v1/generate address. "
    "For an OpenAI-compatible server, choose Custom instead."
)


def koboldcpp_route_refusal(provider: object, endpoint: object) -> str:
    """Refuse a koboldcpp endpoint its native send cannot use.

    ``chat_with_kobold`` posts KoboldAI's native payload to the saved
    endpoint, and the contract reads a root or ``/v1`` as an OpenAI chat
    route, so only the native ``/api/v1/generate`` route keeps sending. The
    wizard, Settings, Chat settings' Make default and the model switcher's
    adoption all write a koboldcpp endpoint through this rule (TASK-38964).

    Args:
        provider: Provider key.
        endpoint: The endpoint, or the chat URL the contract resolved it to.

    Returns:
        ``KOBOLDCPP_NATIVE_ROUTE_REFUSAL`` for a koboldcpp endpoint the
        contract reads off the native route; "" for one on it, for one the
        contract refuses outright (its own error says why) and for any other
        provider.
    """
    if normalize_provider_key_for_contract(provider) != "koboldcpp":
        return ""
    chat_url = resolve_provider_endpoint("koboldcpp", endpoint).chat_url
    if chat_url is None or chat_url.endswith(_KOBOLD_NATIVE_PATH):
        return ""
    return KOBOLDCPP_NATIVE_ROUTE_REFUSAL


def koboldcpp_native_endpoint(endpoint: object) -> str | None:
    """The native ``/api/v1/generate`` route at a koboldcpp endpoint's root.

    Args:
        endpoint: A koboldcpp endpoint: a root, ``/v1``, a chat route or the
            native route itself.

    Returns:
        The native route, unchanged for an endpoint already on it, or None
        when the contract refuses the endpoint.
    """
    resolution = resolve_provider_endpoint("koboldcpp", endpoint)
    if resolution.models_url is None:
        return None
    if not koboldcpp_route_refusal("koboldcpp", endpoint):
        return resolution.chat_url
    root = urlsplit(resolution.models_url.removesuffix("/v1/models"))
    # TASK-38964 review: a root at KoboldAI's /api base (configured as /api or
    # /api/v1) takes the route's /v1/generate, not a second /api.
    path = root.path.removesuffix("/api") + _KOBOLD_NATIVE_PATH
    native = urlunsplit((root.scheme, root.netloc, path, "", ""))
    return resolve_provider_endpoint("koboldcpp", native).chat_url


def canonical_connection_identity(
    provider: str, value: object
) -> tuple[str, str] | None:
    """Return the provider and canonical persisted endpoint when valid."""

    resolution = resolve_provider_endpoint(provider, value)
    if resolution.persisted_endpoint is None:
        return None
    return (
        resolution.provider_key,
        _drop_default_port(resolution.persisted_endpoint),
    )


def normalize_provider_key_for_contract(provider: object) -> str:
    """Return the canonical provider key accepted by this endpoint contract.

    Args:
        provider: Provider config key or established display-name alias.

    Returns:
        The canonical provider key, or an empty string when unsupported.
    """
    if not isinstance(provider, str):
        return ""
    provider_key = provider.strip()
    alias = _PROVIDER_ALIASES.get(provider_key)
    if alias is not None:
        return alias
    if not _PROVIDER_KEY.fullmatch(provider_key):
        return ""
    return provider_key


def _invalid_resolution(provider_key: str, error: str) -> ProviderEndpointResolution:
    return ProviderEndpointResolution(
        provider_key=provider_key,
        normalized_input="",
        persisted_endpoint=None,
        chat_url=None,
        models_url=None,
        persisted_display="",
        chat_display="",
        models_display="",
        form=None,
        errors=(error,),
    )


def _terminal_suffix(
    segments: tuple[str, ...],
    *,
    allow_legacy_local: bool,
) -> tuple[tuple[str, ...], EndpointForm] | None:
    occurrences: list[tuple[int, tuple[str, ...], EndpointForm]] = []
    for index in range(len(segments)):
        for suffix, form in _SUFFIXES:
            if form == "legacy_local" and not allow_legacy_local:
                continue
            if segments[index : index + len(suffix)] == suffix:
                occurrences.append((index, suffix, form))
                break

    if not occurrences:
        return (segments, "origin")
    if len(occurrences) != 1:
        return None

    index, suffix, form = occurrences[0]
    if index + len(suffix) != len(segments):
        return None
    return (segments[:index], form)


def _contains_unsafe_unicode(value: str) -> bool:
    return any(category(character) in _UNSAFE_UNICODE_CATEGORIES for character in value)


def _canonicalize_path(path: str) -> str | None:
    canonical: list[str] = []
    index = 0
    while index < len(path):
        character = path[index]
        if character != "%":
            if character.isascii():
                canonical.append(character)
            else:
                canonical.extend(f"%{octet:02X}" for octet in character.encode("utf-8"))
            index += 1
            continue

        octets = bytearray()
        while index < len(path) and path[index] == "%":
            if (
                index + 2 >= len(path)
                or path[index + 1] not in _HEX_DIGITS
                or path[index + 2] not in _HEX_DIGITS
            ):
                return None
            octets.append(int(path[index + 1 : index + 3], 16))
            index += 3

        if _percent_octets_contain_unsafe_unicode(octets):
            return None
        for octet in octets:
            if octet in {0x2F, 0x5C} or octet < 0x20 or octet == 0x7F:
                return None
            decoded = chr(octet)
            canonical.append(
                decoded if decoded in _UNRESERVED_ASCII else f"%{octet:02X}"
            )
    return "".join(canonical)


def _percent_octets_contain_unsafe_unicode(octets: bytearray) -> bool:
    try:
        decoded = bytes(octets).decode("utf-8")
    except UnicodeDecodeError:
        return True
    return _contains_unsafe_unicode(decoded)


def canonical_endpoint_host(hostname: str) -> str | None:
    """The contract's host rule: the canonical host, or None when refused.

    TASK-38963: Discover models and Console's endpoint display and compare
    decide hosts with this rule, so none of them accepts a host the save
    refuses or refuses one it accepts.

    Args:
        hostname: A URL's host, without brackets or port.

    Returns:
        The lower-cased host (an IP in its compressed form), or None for a
        host that is not ASCII, holds "%", has only parts ``inet_aton``
        reads as numbers (``127.1``, ``0x7f.1``), is over 253 characters,
        or has a label ``valid_dns_label`` refuses.
    """
    if not hostname.isascii() or "%" in hostname:
        return None
    try:
        return ip_address(hostname).compressed.lower()
    except ValueError:
        pass

    if hostname.endswith(".."):
        return None
    host = hostname.removesuffix(".")
    if not host or len(host) > 253:
        return None
    labels = host.split(".")
    if all(_INET_ATON_PART.fullmatch(label) for label in labels):
        # ``127.1``, ``0x7f.1``, ``0x7f000001``: a resolver reads every part
        # as a number (``inet_aton``) and dials an address the dotted form
        # hides (TASK-41132).
        return None
    if not all(valid_dns_label(label) for label in labels):
        return None
    return host.lower()


def valid_dns_label(label: str) -> bool:
    """Whether one host label is valid; callers check the host is ASCII.

    TASK-36517: "_" as in Docker Compose names (kobold_cpp), which httpx and
    discovery accept; a hyphen still never starts or ends a label. Console's
    endpoint preview, display and compare helpers share this rule.

    Args:
        label: One host label, without dots.

    Returns:
        True when the label is 1-63 characters, does not start or end with a
        hyphen, and holds only letters, digits, hyphens and underscores.
    """
    return (
        1 <= len(label) <= 63
        and "-" not in (label[0], label[-1])
        and all(character.isalnum() or character in "-_" for character in label)
    )


def _normalized_netloc(hostname: str, port: int | None) -> str:
    host = hostname
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{port}" if port is not None else host


def is_exact_schemeless_host(netloc: str) -> bool:
    """Whether an endpoint typed without a scheme may be read as ``http://``.

    Only the loopback spellings may; a remote endpoint needs its scheme.
    Discover models applies the same rule (TASK-38963).

    Args:
        netloc: The endpoint's network location, as typed.

    Returns:
        True for localhost, 127.0.0.1 and [::1], with or without a port.
    """
    return any(
        netloc == host or netloc.startswith(f"{host}:")
        for host in ("localhost", "127.0.0.1", "[::1]")
    )


def read_schemeless_loopback(endpoint: str) -> str:
    """The endpoint with ``http://`` added when it is a loopback spelling typed
    without a scheme, as the contract reads it; anything else unchanged.

    TASK-39042: readiness, the Console send, the Chat settings draft check and
    QwenCloud's Discover all read a schemeless endpoint through this one rule.

    Args:
        endpoint: The endpoint as typed or stored.

    Returns:
        ``http://<endpoint>`` for localhost, 127.0.0.1 or [::1] (with or
        without a port) typed without a scheme, else ``endpoint`` itself.
    """
    raw = endpoint.strip()
    if not raw or "://" in raw:
        return endpoint
    try:
        netloc = urlsplit(f"http://{raw}").netloc
    except ValueError:
        return endpoint
    if "@" in netloc:  # user info: the host is after it (the contract refuses)
        return endpoint
    return f"http://{raw}" if is_exact_schemeless_host(netloc) else endpoint


def _drop_default_port(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    default_port = (parsed.scheme == "http" and parsed.port == 80) or (
        parsed.scheme == "https" and parsed.port == 443
    )
    if not default_port or parsed.hostname is None:
        return endpoint
    canonical_host = canonical_endpoint_host(parsed.hostname)
    if canonical_host is None:
        return endpoint
    netloc = _normalized_netloc(canonical_host, None)
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


_INVALID_ENDPOINT_DISPLAY = "invalid endpoint"


def safe_endpoint_display(url: str | None) -> str:
    """Return a credential-free endpoint label safe for user-visible UI.

    Args:
        url: Raw endpoint value from config or user input.

    Returns:
        A host/path endpoint label with user info, query strings, and fragments
        removed. Malformed endpoints return ``"invalid endpoint"`` instead of
        echoing raw input.
    """
    parsed_endpoint = _parse_http_endpoint(url)
    if parsed_endpoint is None:
        return "" if not str(url or "").strip() else _INVALID_ENDPOINT_DISPLAY
    has_scheme, _scheme, hostname, port, _path = parsed_endpoint
    if not (has_scheme or port is not None or _is_loopback_name(hostname)):
        # TASK-39042: typed without a scheme or a port, a remote value may be
        # a pasted key, such as ZAI's ``<32 hex>.<16>``.
        return _INVALID_ENDPOINT_DISPLAY
    resolution = _display_resolution(url)
    if resolution.persisted_endpoint is not None:
        return resolution.normalized_input
    legacy_display = _format_endpoint(parsed_endpoint, drop_default_port=False)
    legacy_resolution = _display_resolution(legacy_display)
    if legacy_resolution.persisted_endpoint is not None:
        return legacy_resolution.normalized_input if has_scheme else legacy_display

    if not has_scheme:
        explicit_legacy_resolution = _display_resolution(f"http://{legacy_display}")
        if explicit_legacy_resolution.persisted_endpoint is not None:
            return legacy_display
    return _INVALID_ENDPOINT_DISPLAY


def _is_loopback_name(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def _display_resolution(value: str):
    """Read an endpoint for display as the contract does.

    An OpenAI-compatible reading first, then koboldcpp's, so its native
    /api/v1/generate route displays as itself (TASK-38963: the adoption toast
    and the unsaved-endpoint copy read "invalid endpoint" for it).
    """
    resolution = resolve_provider_endpoint("custom", value)
    if resolution.persisted_endpoint is None:
        resolution = resolve_provider_endpoint("koboldcpp", value)
    if resolution.persisted_endpoint is None:
        # TASK-41700: a gateway base (``/v1/<account>/<gateway>/openai``)
        # displays as itself, not "invalid endpoint".
        resolution = resolve_provider_endpoint("custom_hosted", value)
    return resolution


def normalize_generic_endpoint_for_compare(url: str | None) -> str:
    """Normalize a generic provider endpoint for stable comparison.

    Args:
        url: Raw endpoint value from config or user input.

    Returns:
        Normalized endpoint identity. Empty input returns an empty string, while
        malformed input returns a non-secret invalid sentinel.
    """
    if not str(url or "").strip():
        return ""
    url = read_schemeless_loopback(str(url))  # TASK-39042: as the send reads it
    # TASK-38963 review: the contract's identity comes before the display
    # rule, so two different endpoints that display alike (``http://kobold``,
    # ``http://other``) never compare as the one sentinel.
    identity = canonical_connection_identity("custom", url)
    if identity is not None:
        return identity[1]
    parsed_endpoint = _parse_http_endpoint(url)
    if parsed_endpoint is None:
        return _INVALID_ENDPOINT_DISPLAY
    return _format_endpoint(parsed_endpoint, drop_default_port=True)


def _parse_http_endpoint(
    url: str | None,
) -> tuple[bool, str, str, int | None, str] | None:
    raw_value = str(url or "")
    raw_url = raw_value.strip()
    if not raw_url:
        return None
    has_unsafe_character = any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in raw_value
    )
    if has_unsafe_character:
        return None
    has_scheme = "://" in raw_url
    candidate = raw_url if has_scheme else f"http://{raw_url}"
    try:
        parsed = urlparse(candidate)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if scheme not in {"http", "https"} or not hostname:
        return None
    hostname = hostname.lower()
    path = parsed.path.rstrip("/")
    if not _is_allowed_endpoint_host(
        hostname, port=port, path=path if has_scheme else ""
    ):
        return None
    return (has_scheme, scheme, hostname, port, path)


def _is_allowed_endpoint_host(hostname: str, *, port: int | None, path: str) -> bool:
    """The endpoint contract's host rule, plus a port or path for a single label.

    TASK-38963: a host the contract accepts displays as itself. A single
    label other than localhost also needs a port, or a path after a typed
    scheme (review r10e-rev2: a key such as ``sk-abc/def`` has a "path"), so
    a key pasted into the endpoint field is never echoed. TASK-39041: an IPv6
    zone id, which the contract refuses, no longer compares as itself.

    Args:
        hostname: The lower-cased host.
        port: The typed port, or None.
        path: The typed path without its trailing "/"; "" when no scheme
            was typed.
    """
    host = canonical_endpoint_host(hostname)
    if host is None:
        return False
    try:
        ip_address(host)
        return True
    except ValueError:
        pass
    return (
        "." in host or host == "localhost" or port is not None or bool(path)
    )


def _format_endpoint(
    endpoint: tuple[bool, str, str, int | None, str],
    *,
    drop_default_port: bool,
) -> str:
    has_scheme, scheme, hostname, port, path = endpoint
    host = hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    netloc = (
        host
        if port is None or (drop_default_port and default_port)
        else f"{host}:{port}"
    )
    if has_scheme:
        return urlunparse((scheme, netloc, path, "", "", "")).rstrip("/")
    return f"{netloc}{path}".rstrip("/")
