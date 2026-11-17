"""Shared Console provider endpoint display and comparison helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

from tldw_chatbook.Provider_Contract import ChatConfigurationError
from tldw_chatbook.Provider_Contract.provider_endpoint_contract import (
    URL_BASED_PROVIDER_KEYS,
    endpoint_refusal,
    normalize_generic_endpoint_for_compare as normalize_generic_endpoint_for_compare,
    read_schemeless_loopback,
    resolve_provider_endpoint,
    safe_endpoint_display as safe_endpoint_display,
)

if TYPE_CHECKING:
    from tldw_chatbook.Agents.goal_models import GoalProviderRef
    from .console_provider_gateway import ConsoleProviderResolution

#: Chat settings' footer action that saves an unsaved endpoint.
SAVE_ENDPOINT_ACTION_LABEL = "Save endpoint & use model"
UNSAVED_ENDPOINT_COPY = (
    "Provider blocked: this endpoint is not saved. Save it with "
    f"{SAVE_ENDPOINT_ACTION_LABEL} in Chat settings, or in F4 Settings, before "
    "sending."
)
#: Single source of the llama.cpp origin Chatbook defaults to (and teaches in
#: its docs) when nothing is configured. The stock `llama-server` port 8080 is
#: covered by local discovery candidates, not by this default.
DEFAULT_LLAMACPP_BASE_URL = "http://127.0.0.1:9099"
#: A ``custom-ep:`` id with no configured entry (removed, or skipped at load);
#: the send and readiness both say it (TASK-40302).
CUSTOM_ENDPOINT_NOT_CONFIGURED_COPY = (
    "Provider blocked: this custom endpoint is no longer configured. "
    "Restore its entry or choose another provider before sending."
)
_ENDPOINT_SETTING_KEYS = (
    "api_base_url",
    "api_base",
    "base_url",
    "api_url",
    "api_endpoint",
    "endpoint",
    "router_base_url",
    "huggingface_router_base_url",
)
_URL_PROVIDER_SETTING_KEYS = (
    "api_base_url",
    "api_base",
    "base_url",
    "api_url",
    "router_base_url",
    "huggingface_router_base_url",
)
_BUILTIN_PROVIDER_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com/v1",
    "cohere": "https://api.cohere.com",
    "deepseek": "https://api.deepseek.com",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "groq": "https://api.groq.com/openai/v1",
    "huggingface": "https://api-inference.huggingface.co/v1",
    "mistral": "https://api.mistral.ai/v1",
    "mistralai": "https://api.mistral.ai/v1",
    "moonshot": "https://api.moonshot.ai/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "qwencloud": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "zai": "https://api.z.ai/api/paas/v4",
    # ADR-179: "databricks" deliberately has NO entry here. Every value in
    # this map is a live send-path fallback (``effective_provider_endpoint``
    # -> the gateway's effective_base_url), and Databricks has no shippable
    # default -- the workspace host is per-account. The documented per-
    # account shape is "<workspace-host>/openai/v1" (the engine appends the
    # /openai/v1 suffix from provider_registry.DATABRICKS.base_url_suffix),
    # and provider_readiness blocks with workspace-URL copy until
    # api_base_url is configured, so a placeholder must never ship as a
    # fallback value: ``builtin_provider_endpoint("databricks", ...) is
    # None`` is the contract.
}


def first_configured_endpoint(provider_settings: Mapping[str, object]) -> str | None:
    """Return the first configured provider endpoint from known config aliases.

    Args:
        provider_settings: Provider-specific configuration mapping.

    Returns:
        The first non-empty endpoint string, or ``None`` when no endpoint is
        configured.
    """
    for key in _ENDPOINT_SETTING_KEYS:
        value = provider_settings.get(key)
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def with_provider_endpoint(
    app_config: Mapping[str, object], provider_key: str, endpoint: str
) -> dict[str, object]:
    """Return a copy of ``app_config`` whose provider's one endpoint is ``endpoint``.

    In memory only: readiness and probes then read an endpoint the session
    uses (a detected server, a session-only endpoint) as the configured one.
    Every endpoint alias is dropped first, so no higher-ranked alias outranks
    it. The input is not changed.

    Args:
        app_config: The configuration snapshot.
        provider_key: The ``api_settings`` section to configure.
        endpoint: The endpoint the copy is configured with.

    Returns:
        The copied configuration with that section's ``api_url`` set.
    """
    api_settings = app_config.get("api_settings")
    api_settings = dict(api_settings) if isinstance(api_settings, Mapping) else {}
    section = api_settings.get(provider_key)
    settings = {
        key: value
        for key, value in (section.items() if isinstance(section, Mapping) else ())
        if key not in _ENDPOINT_SETTING_KEYS
    }
    settings["api_url"] = endpoint
    api_settings[provider_key] = settings
    return {**app_config, "api_settings": api_settings}


def effective_provider_endpoint(
    provider_key: str,
    selected_endpoint: str | None,
    provider_settings: Mapping[str, object],
    *,
    custom_endpoint: bool = False,
) -> str | None:
    """Resolve the exact endpoint a provider call would use at this moment.

    Explicit Console selection wins, followed by the provider's configured
    endpoint aliases and finally the adapter's built-in cloud default. Custom
    OpenAI-compatible endpoint forms resolve to the chat URL actually used.

    Args:
        provider_key: Normalized provider readiness key.
        selected_endpoint: Optional Console-selected endpoint.
        provider_settings: Provider-specific configuration mapping.
        custom_endpoint: True for a custom endpoint entry, which the contract
            reads as its engine does (TASK-40300 review r14ep-rev2 M-a: a
            relaxed base pasted with ``/models`` sent to
            ``.../models/chat/completions``).

    Returns:
        Effective endpoint to pin into the provider resolution, or ``None``
        when the provider has neither a configured nor built-in endpoint.
    """
    if custom_endpoint and provider_key == "custom":
        provider_key = "custom_hosted"
    if isinstance(selected_endpoint, str) and selected_endpoint.strip():
        return _effective_contract_endpoint(provider_key, selected_endpoint)
    if provider_key == "huggingface" and _huggingface_router_mode(provider_settings):
        for key in ("router_base_url", "huggingface_router_base_url"):
            router_endpoint = provider_settings.get(key)
            if isinstance(router_endpoint, str) and router_endpoint.strip():
                return router_endpoint.strip()
        return builtin_provider_endpoint(provider_key, provider_settings)
    configured_endpoint = first_configured_endpoint(provider_settings)
    if configured_endpoint:
        return _effective_contract_endpoint(provider_key, configured_endpoint)
    return builtin_provider_endpoint(provider_key, provider_settings)


def effective_provider_discovery_endpoint(
    provider_key: str,
    selected_endpoint: str | None,
    provider_settings: Mapping[str, object],
) -> str | None:
    """Resolve the settings-aware OpenAI-compatible model-listing base."""

    endpoint = effective_provider_endpoint(
        provider_key,
        selected_endpoint,
        provider_settings,
    )
    if endpoint is None or provider_key != "huggingface":
        return endpoint
    if not _huggingface_router_mode(provider_settings):
        return endpoint
    try:
        parsed = urlparse(endpoint)
    except ValueError:
        return endpoint
    if (parsed.hostname or "").lower() != "router.huggingface.co":
        return endpoint
    return urlunparse((parsed.scheme, parsed.netloc, "/v1", "", "", ""))


def builtin_provider_endpoint(
    provider_key: str,
    provider_settings: Mapping[str, object] | None = None,
) -> str | None:
    """Return the canonical adapter fallback endpoint for a provider."""

    settings = provider_settings or {}
    if (
        provider_key == "moonshot"
        and str(settings.get("api_region", "")).lower() == "china"
    ):
        return "https://api.moonshot.cn/v1"
    if provider_key == "huggingface" and _huggingface_router_mode(settings):
        return "https://router.huggingface.co/hf-inference"
    return _BUILTIN_PROVIDER_ENDPOINTS.get(provider_key)


#: TASK-2117: first-party cloud providers whose adapters accept an
#: ``api_base_url`` request pin but whose primary Console sends previously
#: left the kwarg unset (the auxiliary path already pins it). Shared by the
#: gateway kwargs builders and the trace verifier's independent
#: reconstruction, so the two can never drift.
CONDITIONAL_BASE_URL_EXECUTION_KEYS = frozenset(
    {
        "openai",
        "cohere",
        "deepseek",
        "google",
        "groq",
        "huggingface",
        "openrouter",
    }
)
#: TASK-39042 review r11g-rev2: local adapters that otherwise read their own
#: config raw, so a schemeless endpoint readiness reads as http:// went as
#: typed. Every primary send pins the resolved base for them; shared by the
#: gateway kwargs builders and the trace verifier's reconstruction (r11g-rev3).
#: MLX-LM joins them (TASK-40300 review r14ep-rev3 follow-up and rev4 C-1):
#: Console checks and pins the URL MLX-LM builds from its own ``host`` and
#: ``port`` (``http://127.0.0.1:8080/v1`` when unset), never the ``api_url``
#: in its table, which the adapter ignores.
CONFIG_READING_LOCAL_EXECUTION_KEYS = frozenset(
    {"aphrodite", "koboldcpp", "local_mlx_lm", "oobabooga", "tabbyapi"}
)
#: Where an adapter reads its endpoint when the request pins none, in its own
#: order, when that is not its ``api_settings`` table under the dispatch name
#: (TASK-40300 review r14ep-rev3 I-1/I-3: ``[api_settings.ooba_api] api_ip``
#: was sent to unchecked).
_ADAPTER_OWN_ENDPOINTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "openai": (("openai_api", "api_base_url"),),
    "anthropic": (("anthropic_api", "api_base_url"),),
    "huggingface": (("huggingface_api", "api_base_url"),),
    "local-llm": (
        ("api_settings", "local-llm", "api_url"),
        ("api_settings", "local-llm", "api_ip"),
    ),
    "custom-openai-api": (("api_settings", "custom", "api_url"),),
    "custom-openai-api-2": (
        ("api_settings", "custom_2", "api_url"),
        ("custom_openai_api_2", "api_ip"),
    ),
    "oobabooga": (("api_settings", "ooba_api", "api_ip"),),
    "aphrodite": (("api_settings", "aphrodite_api", "api_url"),),
    "tabbyapi": (("api_settings", "tabby_api", "api_url"),),
    "koboldcpp": (("api_settings", "koboldcpp", "api_url"),),
}


#: The code default of the legacy ``[API] custom_openai2_api_ip``
#: (``config.load_settings``), which is not an explicit endpoint.
LEGACY_CUSTOM_2_API_IP_DEFAULT = "http://127.0.0.1:5000/v1/chat/completions"


def keep_legacy_custom_2_endpoint(
    merged: dict, legacy: object, template: Mapping[str, object]
) -> None:
    """Keep an explicit legacy Custom OpenAI API 2 endpoint in the merged view.

    The adapter reads ``[api_settings.custom_2] api_url`` before the legacy
    ``[API] custom_openai2_api_ip`` (owner decision 2026-11-16). An explicit
    legacy endpoint must not be silently redirected (ruling 2026-11-17), so
    while the slot's ``api_url`` is absent, blank or still the shipped
    template's placeholder, and the legacy value is anything but its default,
    the view's ``api_url`` takes the legacy value. In memory only, like
    ``config._normalize_legacy_provider_api_key``: every reader (the adapter,
    ``adapter_own_endpoint``, Console, Settings) sees one value, and the
    user's file is never written.

    Args:
        merged: The user's config merged over the template; changed in place.
        legacy: The user's ``[API] custom_openai2_api_ip``.
        template: The shipped template (``DEFAULT_CONFIG_FROM_TOML``).
    """
    if not isinstance(legacy, str) or legacy.strip() in {"", LEGACY_CUSTOM_2_API_IP_DEFAULT}:
        return
    api_settings = merged.get("api_settings")
    slot = api_settings.get("custom_2") if isinstance(api_settings, dict) else None
    shipped = ((template.get("api_settings") or {}).get("custom_2") or {}).get("api_url")
    if isinstance(slot, dict) and str(slot.get("api_url") or "").strip() in {"", shipped}:
        # What the slot read before, so a save that leaves the Endpoint alone
        # writes that, not the legacy value (review r15ep-rev2 I-1).
        merged[LEGACY_CUSTOM_2_BRIDGE] = {
            "legacy": legacy.strip(),
            "before": slot.get("api_url") or "",
        }
        slot["api_url"] = legacy.strip()


#: Where ``keep_legacy_custom_2_endpoint`` records a bridge in the merged
#: view (a provenance key, like ``legacy_api_key_providers``; never written).
LEGACY_CUSTOM_2_BRIDGE = "legacy_custom_2_endpoint_bridge"


def reapply_legacy_custom_2_bridge(view: dict) -> None:
    """Re-run ``keep_legacy_custom_2_endpoint`` on an in-memory settings view.

    Settings patches the saved values into the live view; an unedited Custom 2
    save writes the slot's earlier value, so without this the view showed
    (and Test and Discover staged) the placeholder while sends used the
    legacy endpoint (review r15ep-rev3 I-1). One rule, the load-time one.

    Args:
        view: The live settings view (``load_settings`` shape); changed in place.
    """
    raw = view.get("COMPREHENSIVE_CONFIG_RAW")
    bridge = view.get(LEGACY_CUSTOM_2_BRIDGE) or (
        raw.get(LEGACY_CUSTOM_2_BRIDGE) if isinstance(raw, Mapping) else None
    )
    if isinstance(bridge, Mapping):
        from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

        keep_legacy_custom_2_endpoint(view, bridge.get("legacy"), DEFAULT_CONFIG_FROM_TOML)


def endpoint_to_save_unedited(provider_key: str, shown: str) -> str:
    """The endpoint a save writes when the Endpoint field was not edited.

    The field shows Custom 2's bridged legacy endpoint (``keep_legacy_custom_2
    _endpoint``); a save that did not touch it writes back what the slot held
    before the bridge, so the legacy key keeps being read (TASK-41145 ruling,
    review r15ep-rev2 I-1). Any other value is written as shown. The bridge is
    read from the file as it is now, not from the live view: once first-run or
    a save wrote a real ``api_url`` there is none (review r15ep-rev4).

    Args:
        provider_key: The provider the save is for.
        shown: The Endpoint field's value.

    Returns:
        The endpoint to write.
    """
    if provider_key != "custom_2":
        return shown
    from tldw_chatbook.config import get_atomic_config_snapshot

    try:
        bridge = get_atomic_config_snapshot().values.get(LEGACY_CUSTOM_2_BRIDGE)
    except ValueError:  # unreadable or undecryptable; the save's write refuses it too
        return shown
    if not isinstance(bridge, Mapping):
        return shown
    return str(bridge.get("before") or "") if shown == bridge.get("legacy") else shown


#: Adapters that always build their endpoint themselves (MLX-LM's host and
#: port) and ignore any URL in their ``api_settings`` table.
ADAPTER_ONLY_ENDPOINT_KEYS = frozenset({"mlx_lm", "local_mlx_lm"})


def mlx_endpoint(table: Mapping[str, object]) -> str:
    """The URL ``chat_with_mlx_lm`` builds from its table's host and port.

    Args:
        table: MLX-LM's ``api_settings`` table.

    Returns:
        ``http://<host>:<port>/v1``, 127.0.0.1 and 8080 where unset.
    """
    return f"http://{table.get('host', '127.0.0.1')}:{table.get('port', 8080)}/v1"


def configured_provider_endpoint(
    provider: object, provider_settings: Mapping[str, object]
) -> str | None:
    """The endpoint a provider's table holds, as setup screens show it.

    Args:
        provider: The provider key.
        provider_settings: The provider's ``api_settings`` table.

    Returns:
        MLX-LM's host and port as ``http://<host>:<port>/v1`` (TASK-41145),
        else the table's first configured endpoint alias, or None.
    """
    if provider in ADAPTER_ONLY_ENDPOINT_KEYS:
        return mlx_endpoint(provider_settings)
    return first_configured_endpoint(provider_settings)


def adapter_own_endpoint(
    dispatch: str, app_config: Mapping[str, object]
) -> str | None:
    """The endpoint ``dispatch``'s adapter sends to when the request pins none.

    Covers what Console's ``api_settings`` lookup misses: the adapter's own
    table, MLX's host and port, and an engine preset's shipped default.

    Args:
        dispatch: The ``chat_api_call`` endpoint name.
        app_config: The loaded configuration.

    Returns:
        That endpoint, read as Console reads a configured one, or None.
    """
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    legacy_hf = app_config.get("huggingface_api")
    if dispatch == "huggingface" and isinstance(legacy_hf, Mapping):
        if _huggingface_router_mode(legacy_hf):
            # Router mode reads the router base, as chat_with_huggingface does.
            return effective_provider_endpoint(dispatch, None, legacy_hf)
    for path in _ADAPTER_OWN_ENDPOINTS.get(dispatch, ()):
        value: object = app_config
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if isinstance(value, str) and value.strip():
            return _effective_contract_endpoint(dispatch, value.strip())
    if dispatch in ADAPTER_ONLY_ENDPOINT_KEYS:
        api_settings = app_config.get("api_settings")
        table = api_settings.get(dispatch) if isinstance(api_settings, Mapping) else None
        return mlx_endpoint(table if isinstance(table, Mapping) else {})
    record = RECORDS_BY_KEY.get(dispatch)
    return record.default_base_url if record and record.engine_driven else None


#: Console execution keys besides ``CONFIG_READING_LOCAL_EXECUTION_KEYS``
#: whose adapter reads a table Console's own lookup misses (review r14ep-rev3
#: and its follow-up: Local LLM's ``api_ip``, Custom OpenAI API 2's top-level
#: ``api_ip``; rev4 M-4: HuggingFace's ``huggingface_api``).
_OWN_TABLE_CONSOLE_KEYS = frozenset({"local-llm", "custom-openai-api-2", "huggingface"})


def console_own_table_endpoint(
    execution_key: str,
    app_config: Mapping[str, object],
    resolved: str | None,
    provider_settings: Mapping[str, object],
    selected: str | None = None,
) -> str | None:
    """The endpoint a Console send uses, so its send and readiness check it.

    Normally ``resolved``. An adapter that reads its own table when nothing is
    pinned uses that table's endpoint instead when neither the chat nor the
    provider's ``api_settings`` table sets one, so ``[api_settings.ooba_api]
    api_ip`` is checked and pinned rather than sent to unchecked (TASK-40300
    review r14ep-rev3 I-3; refusing would break a config that sets only that
    table). MLX-LM always uses its own host and port (rev4 C-1).

    Args:
        execution_key: Console's resolved adapter key.
        app_config: The loaded configuration.
        resolved: The endpoint Console resolved for the provider.
        provider_settings: The provider's ``api_settings`` table.
        selected: The chat's own endpoint, if any.

    Returns:
        The endpoint the send uses, or None.
    """
    if execution_key in ADAPTER_ONLY_ENDPOINT_KEYS:
        return adapter_own_endpoint(execution_key, app_config)
    if execution_key in CONFIG_READING_LOCAL_EXECUTION_KEYS | _OWN_TABLE_CONSOLE_KEYS and not (
        selected or first_configured_endpoint(provider_settings)
    ):
        return adapter_own_endpoint(execution_key, app_config) or resolved
    return resolved


def adapter_api_base_url(execution_key: str, base_url: str | None) -> str | None:
    """Project the transport base; legacy custom slots accept the full endpoint.

    Args:
        execution_key: Resolved adapter key.
        base_url: Resolved destination URL, without changing its durable identity.

    Returns:
        The API base for the custom engine, otherwise the original nonempty URL.
    """
    if execution_key == "custom-hosted" and base_url is not None:
        base_url = base_url.rstrip("/").removesuffix("/chat/completions")
    return base_url or None


def meaningful_adapter_base_url(execution_key: str, base_url: str | None) -> str | None:
    """Pin the resolved endpoint when it carries real information.

    TASK-2117: a resolution's ``base_url`` always carries SOMETHING -- the
    builtin cloud default when neither a configured
    ``[api_settings.<provider>].api_base_url`` nor a session-selected endpoint
    exists. Forwarding that builtin would pin the adapter to a value it
    already falls back to on its own, so for these first-party cloud keys the
    kwarg is forwarded only when the resolved endpoint differs from the
    provider's shipped default -- i.e. when a configured or session-selected
    URL must reach the request.

    HuggingFace is the exception (TASK-40300 review r14ep-rev5): it is always
    pinned, built-in or not. Its adapter otherwise reads ``huggingface_api``
    (in router mode, its ``router_base_url``), and Console already checks
    that legacy table itself when ``[api_settings.huggingface]`` has no
    endpoint (``console_own_table_endpoint``), so the URL pinned is the one
    Console checked.
    """
    if execution_key not in CONDITIONAL_BASE_URL_EXECUTION_KEYS:
        return None
    if not isinstance(base_url, str) or not base_url.strip():
        return None
    # TASK-40300 review r14ep-rev5: HuggingFace is always pinned. Unpinned, its
    # adapter reads ``huggingface_api`` (and, in router mode, its
    # ``router_base_url``) whatever Console checked.
    if base_url == builtin_provider_endpoint(execution_key) and execution_key != "huggingface":
        return None
    return base_url


def _huggingface_router_mode(provider_settings: Mapping[str, object]) -> bool:
    return (
        str(
            provider_settings.get(
                "use_router_url_format",
                provider_settings.get("huggingface_use_router_url_format", "False"),
            )
        ).lower()
        == "true"
    )


def provider_uses_endpoint(
    provider_key: str, provider_settings: Mapping[str, object]
) -> bool:
    """Return whether a provider should validate saved endpoint overrides.

    Args:
        provider_key: Normalized provider readiness key.
        provider_settings: Provider-specific configuration mapping.

    Returns:
        ``True`` when the provider is known to use endpoint URLs or has a saved
        base URL setting.
    """
    return provider_key in URL_BASED_PROVIDER_KEYS or any(
        key in provider_settings for key in _URL_PROVIDER_SETTING_KEYS
    )


def generic_endpoint_differs(
    base_url: str | None, provider_settings: Mapping[str, object]
) -> bool:
    """Return whether a session endpoint differs from the persisted endpoint.

    Args:
        base_url: Session-selected provider endpoint.
        provider_settings: Provider-specific configuration mapping.

    Returns:
        ``True`` when both values normalize to different endpoint identities.
    """
    selected_base_url = normalize_generic_endpoint_for_compare(base_url)
    if not selected_base_url:
        return False
    configured_base_url = normalize_generic_endpoint_for_compare(
        first_configured_endpoint(provider_settings)
    )
    return selected_base_url != configured_base_url


def unsaved_endpoint_copy(
    base_url: str | None, provider_settings: Mapping[str, object]
) -> str:
    """Return actionable recovery copy with safe endpoint details.

    Args:
        base_url: Session-selected provider endpoint.
        provider_settings: Provider-specific configuration mapping.

    Returns:
        User-visible recovery copy that omits credentials, query strings, and
        fragments from endpoint values.
    """
    selected = safe_endpoint_display(base_url) or "selected session endpoint"
    configured = (
        safe_endpoint_display(first_configured_endpoint(provider_settings))
        or "not saved"
    )
    return f"{UNSAVED_ENDPOINT_COPY} Selected endpoint: {selected}. Saved endpoint: {configured}."


def invalid_llamacpp_base_url_copy(reason: str) -> str:
    """llama.cpp's refusal copy, naming the contract's reason (TASK-40300 M-b).

    Args:
        reason: ``endpoint_refusal``'s reason, which names no part of the value.

    Returns:
        The copy readiness and the send show.
    """
    return f"Provider blocked: invalid llama.cpp base URL: {reason}"


def refuse_a_refused_endpoint(resolution: object) -> None:
    """Refuse a dispatch whose endpoint the contract refuses (TASK-40300 I1).

    ``resolve_for_send`` checks the endpoint it resolves, but an agent
    sub-call or a run snapshot can overlay another ``base_url`` afterwards,
    so the gateway's stream and auxiliary dispatch check the endpoint they
    are about to use, before any request, trace shadow or adapter call.

    Args:
        resolution: The ``ConsoleProviderResolution`` about to be dispatched.

    Raises:
        ChatConfigurationError: Status-less (nothing reached the provider),
            with the contract's reason, which names no part of the value.
    """
    from tldw_chatbook.Chat.custom_endpoint_registry import split_custom_endpoint_id

    provider = str(getattr(resolution, "provider", "") or "")
    base_url = getattr(resolution, "base_url", None)
    if provider in {"llama_cpp", "local_llamacpp"}:
        error = endpoint_refusal("llama_cpp", base_url or DEFAULT_LLAMACPP_BASE_URL)
    elif isinstance(base_url, str) and base_url:
        key = getattr(resolution, "readiness_key", "") or getattr(
            resolution, "execution_key", ""
        )
        error = endpoint_refusal(
            key or provider,
            base_url,
            custom_endpoint=bool(
                split_custom_endpoint_id(getattr(resolution, "selected_provider", ""))
            ),
        )
    else:
        return
    if error:
        raise ChatConfigurationError(
            f"Provider blocked: {error}", provider=provider or None, status_code=None
        )


def _effective_contract_endpoint(provider_key: str, endpoint: str) -> str:
    """Resolve endpoint forms only for adapters governed by the new contract.

    Every provider reads a loopback endpoint typed without a scheme as
    ``http://`` (TASK-39042 review: readiness took ``localhost:8080`` and the
    adapter got it schemeless).
    """
    endpoint = read_schemeless_loopback(endpoint)
    if provider_key not in {
        "custom", "custom_2", "custom_hosted", "llama_cpp", "local_llamacpp"
    }:
        return endpoint
    resolution = resolve_provider_endpoint(provider_key, endpoint)
    return resolution.persisted_endpoint or endpoint


def goal_provider_ref(resolution: ConsoleProviderResolution) -> GoalProviderRef:
    """Pin resolved execution and credential identity without retaining credentials."""
    from tldw_chatbook.Agents.goal_models import GoalProviderRef

    return GoalProviderRef(
        provider=resolution.provider,
        model=resolution.model,
        config_ref="api_settings."
        + (resolution.readiness_key or resolution.execution_key or resolution.provider),
        authority_ref=resolution.api_key_source or "unauthenticated",
        endpoint_ref=normalize_generic_endpoint_for_compare(resolution.base_url)
        or "provider-default",
    )
