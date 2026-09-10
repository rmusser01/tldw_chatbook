"""Shared web-search setup metadata and effective configuration (ADR-012).

This module deliberately imports no UI or provider stack until an explicit probe.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class FieldSpec:
    """Describe one immutable backend setting and its guided editor behavior.

    Attributes:
        key: Canonical key in the local SearchEngines config table.
        label: Human-readable field name used by the form and setup diagnostics.
        secret: Whether saved values are masked and replacement requires input.
        env_var: Environment variable whose nonempty value overrides local config.
        placeholder: Example input shown when the editor is empty.
        required: Whether an effective value is needed before testing the backend.
    """

    key: str
    label: str
    secret: bool = False
    env_var: str = ""
    placeholder: str = ""
    required: bool = True


@dataclass(frozen=True)
class BackendSpec:
    """Define an immutable search backend's setup and presentation contract.

    Attributes:
        id: Canonical identifier accepted by search dispatch and saved defaults.
        label: Display name for selectors and setup messages.
        description: Short explanation of the service and its requirements.
        fields: Ordered settings required or offered by the guided editor.
        docs_url: Provider documentation or account-setup reference.
        notice: Optional availability, retirement, or billing guidance.
    """

    id: str
    label: str
    description: str
    fields: tuple[FieldSpec, ...]
    docs_url: str
    notice: str = ""


def _key(backend: str, env_var: str) -> FieldSpec:
    return FieldSpec(
        f"{backend}_search_api_key", "API key", secret=True, env_var=env_var
    )


BACKENDS: dict[str, BackendSpec] = {
    "duckduckgo": BackendSpec(
        "duckduckgo",
        "DuckDuckGo",
        "Public web search without an API key.",
        (),
        "https://duckduckgo.com/",
    ),
    "brave": BackendSpec(
        "brave",
        "Brave",
        "Independent web index. Requires a Brave Search API subscription.",
        (_key("brave", "BRAVE_SEARCH_API_KEY"),),
        "https://api-dashboard.search.brave.com/documentation/guides/authentication",
    ),
    "serper": BackendSpec(
        "serper",
        "Serper",
        "Google results through the Serper API.",
        (_key("serper", "SERPER_API_KEY"),),
        "https://serper.dev/",
    ),
    "tavily": BackendSpec(
        "tavily",
        "Tavily",
        "Web search with extracted result content.",
        (_key("tavily", "TAVILY_API_KEY"),),
        "https://docs.tavily.com/documentation/quickstart",
    ),
    "exa": BackendSpec(
        "exa",
        "Exa",
        "Web search with result highlights. Content retrieval may add cost.",
        (_key("exa", "EXA_API_KEY"),),
        "https://docs.exa.ai/reference/search",
    ),
    "kagi": BackendSpec(
        "kagi",
        "Kagi",
        "Paid Kagi search using the existing legacy Search API integration.",
        (_key("kagi", "KAGI_API_KEY"),),
        "https://help.kagi.com/kagi/api/search-legacy.html",
        "The Kagi v0 Search API is deprecated; this integration retains the documented legacy contract.",
    ),
    "searx": BackendSpec(
        "searx",
        "SearX / SearXNG",
        "Use your own instance or one that allows JSON API searches.",
        (
            FieldSpec(
                "searx_search_api_url",
                "Instance URL",
                env_var="SEARX_URL",
                placeholder="http://localhost:8080/search",
            ),
        ),
        "https://docs.searxng.org/dev/search_api.html",
    ),
    "google": BackendSpec(
        "google",
        "Google Custom Search",
        "Search with an existing Google Custom Search JSON API account.",
        (
            _key("google", "GOOGLE_SEARCH_API_KEY"),
            FieldSpec(
                "google_search_engine_id",
                "Search engine ID",
                env_var="GOOGLE_SEARCH_ENGINE_ID",
            ),
        ),
        "https://developers.google.com/custom-search/v1/overview",
        "Closed to new customers. Existing customers must transition by January 1, 2027.",
    ),
    "bing": BackendSpec(
        "bing",
        "Bing (retired)",
        "Legacy Bing Search API settings retained for existing configurations.",
        (_key("bing", "BING_SEARCH_API_KEY"),),
        "https://learn.microsoft.com/en-us/lifecycle/announcements/bing-search-api-retirement",
        "Bing Search APIs retired on August 11, 2025. Choose another backend for new setup.",
    ),
    "yandex": BackendSpec(
        "yandex",
        "Yandex",
        "Yandex Cloud Search API v2; requires a key and cloud folder.",
        (
            _key("yandex", "YANDEX_SEARCH_API_KEY"),
            FieldSpec(
                "yandex_search_folder_id", "Folder ID", env_var="YANDEX_FOLDER_ID"
            ),
        ),
        "https://yandex.cloud/en/docs/search-api/quickstart/",
    ),
}

LEGACY_FIELD_ALIASES = {
    "bing_search_api_key": ("search_engine_api_key_bing", "bing_api_key"),
    "searx_search_api_url": ("search_engine_searx_api",),
}


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def saved_field_value(field: FieldSpec, raw: Mapping[str, object]) -> str:
    """Return the saved value, including legacy aliases, without environment overrides."""
    # Raw [SearchEngines] is authoritative. Normalized tables support older callers.
    keys = (field.key, *LEGACY_FIELD_ALIASES.get(field.key, ()))
    for section in (
        "SearchEngines",
        "search_engines_keys",
        "search_engine_specific_settings",
        "search_engines",
    ):
        values = raw.get(section, {})
        if isinstance(values, Mapping):
            for key in keys:
                value = _text(values.get(key))
                if value:
                    return value
    return ""


def resolve_backend_fields(
    backend_id: str, raw: Mapping[str, object]
) -> dict[str, str]:
    """Return effective environment > saved values without modifying configuration."""
    return {
        field.key: _text(os.environ.get(field.env_var)) or saved_field_value(field, raw)
        for field in BACKENDS[backend_id].fields
    }


def field_source(field: FieldSpec, raw: Mapping[str, object]) -> str:
    """Describe the effective source without revealing any field value."""
    if _text(os.environ.get(field.env_var)):
        return f"Environment: {field.env_var}"
    return "Saved in local config" if saved_field_value(field, raw) else "Not set"


def searx_url_issue(value: str) -> str:
    """Validate an explicitly configured endpoint, allowing local/LAN destinations."""
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in ("http", "https")
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
            and "\\" not in value
            and not any(char.isspace() or ord(char) < 32 for char in value)
        )
        _ = parsed.port  # Reject malformed and out-of-range ports.
    except ValueError:
        valid = False
    if not valid:
        return "Enter a valid HTTP(S) SearX instance URL without credentials or a fragment."
    if parsed.hostname.lower().rstrip(".") == "searx.example.com":
        return "Replace the example SearX URL with your instance URL."
    return ""


def setup_issues(backend_id: str, raw: Mapping[str, object]) -> list[str]:
    """Return local setup blockers; no network requests or secret-bearing messages."""
    backend = BACKENDS.get(backend_id)
    if backend is None:
        return ["Choose a supported search backend."]
    values = resolve_backend_fields(backend_id, raw)
    issues = [
        f"Add {field.label} for {backend.label} (or set {field.env_var})."
        for field in backend.fields
        if field.required and not values[field.key]
    ]
    if backend_id == "searx" and values["searx_search_api_url"]:
        issue = searx_url_issue(values["searx_search_api_url"])
        if issue:
            issues.append(issue)
    if backend_id == "duckduckgo":
        from tldw_chatbook.Utils.optional_deps import check_dependency

        if not check_dependency("lxml"):
            issues.append(
                "Install the websearch extra (lxml is required for DuckDuckGo)."
            )
    if backend_id == "bing":
        issues.append(backend.notice)
    return issues


@dataclass(frozen=True)
class ProbeResult:
    """Carry a credential-safe outcome from an explicit saved-settings search.

    Attributes:
        ok: Whether the request completed successfully, including empty results.
        message: Closed UI status text without provider payloads or credentials.
        result_count: Number of returned results; zero for failures or empty results.
    """

    ok: bool
    message: str
    result_count: int = 0


def probe_saved_backend(backend_id: str) -> ProbeResult:
    """Run the visible sample search against saved setup, without synthesis or caching."""
    from tldw_chatbook.config import load_cli_config_and_ensure_existence

    issues = setup_issues(backend_id, load_cli_config_and_ensure_existence())
    if issues:
        return ProbeResult(False, issues[0])
    from tldw_chatbook.Web_Scraping.WebSearch_APIs import perform_websearch

    try:
        result = perform_websearch(backend_id, "tldw chatbook", "US", "en", "en", 1)
    except Exception:  # noqa: BLE001 - UI boundary must never expose provider exception bodies.
        return ProbeResult(
            False, "Search test failed. Check the saved setup and try again."
        )
    if not isinstance(result, dict):
        return ProbeResult(
            False, "Search test failed. Check the saved setup and try again."
        )
    if result.get("processing_error") or result.get("error"):
        messages = {
            "auth": "Authentication failed. Check the effective API key and account permissions.",
            "rate_limit": "Provider rate limit or quota reached. Check your account and try later.",
            "connection": "Could not connect. Check network access and the configured endpoint.",
            "timeout": "Search timed out. Check connectivity and try again.",
        }
        return ProbeResult(
            False,
            messages.get(
                result.get("error_kind"),
                "Search test failed. Check the saved setup and provider availability.",
            ),
        )
    count = len(result.get("results", []))
    return ProbeResult(
        True,
        "Search completed successfully."
        if count
        else "Search completed; no results were returned.",
        count,
    )
