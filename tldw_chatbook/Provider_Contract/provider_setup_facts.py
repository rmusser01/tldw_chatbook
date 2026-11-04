"""Setup facts per provider: one owner the setup form and Settings both read.

TASK-34100.6 (review structural fix SF5): the first-run Provider step's
fields, checks, links and discovery ports came from scattered hand-kept rules
(a 20-key probe whitelist, a 6-entry base-URL table, an 8-entry key-URL map,
hard-coded ports), each drifting on its own. ``ConsoleProviderCatalogEntry``
(the record Settings and setup already share) now exposes these facts as
``entry.setup_facts``; this module is where they live.

The facts are derived from the provider registry (ADR-179) and readiness where
those already own them -- locality, the conventional env var, the per-account
base URL -- and kept here only where nothing else owns them: where to get a
key, the local ports an engine listens on, a route suffix, the legacy flag.

Key URLs are only ever ones taken from the provider's own documentation;
providers whose page was not confirmed sit on ``KEY_URL_UNCONFIRMED`` and get
the generic 'Get a key from your <provider> account dashboard.' instead of an
invented link (``Tests/Chat/test_provider_setup_facts.py`` enforces this).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

Locality = Literal["local", "cloud"]
CheckKind = Literal["model_list", "authenticated", "none"]

#: Where a first-time user gets a key, from each provider's own docs. These
#: eight shipped in the wizard since TASK-21149 (UAT P-3) and were checked
#: live then; huggingface is the token settings page its docs link.
KEY_URLS: dict[str, str] = {
    "openai": "platform.openai.com/api-keys",
    "anthropic": "console.anthropic.com",
    "groq": "console.groq.com/keys",
    "openrouter": "openrouter.ai/keys",
    "mistralai": "console.mistral.ai",
    "mistral": "console.mistral.ai",
    "deepseek": "platform.deepseek.com",
    "cohere": "dashboard.cohere.com",
    "google": "aistudio.google.com/apikey",
    "huggingface": "huggingface.co/settings/tokens",
}

#: Keyed providers whose key page was not confirmed from their docs. They
#: show the dashboard fallback; add a row to ``KEY_URLS`` (and drop it here)
#: once the page is sourced, never a guessed URL (review provider-25).
KEY_URL_UNCONFIRMED: frozenset[str] = frozenset(
    {
        "arcee",
        "azure",
        "baseten",
        "byteplus",
        "cerebras",
        "cloudflare",
        "commandcode",
        "databricks",
        "deepinfra",
        "fireworks",
        "gmi",
        "kilo",
        "meta",
        "mimo",
        "minimax",
        "moonshot",
        "nebius",
        "nous",
        "novita",
        "nvidia",
        "ollama_cloud",
        "opencode_zen",
        "qianfan",
        "qwencloud",
        "sambanova",
        "siliconflow",
        "stepfun",
        "together",
        "tokenhub",
        "upstage",
        "venice",
        "vercel",
        "wandb",
        "zai",
        "zenmux",
    }
)

#: Per-account base URLs (``provider_readiness.PROVIDERS_REQUIRING_BASE_URL_KEYS``):
#: the field's label and an example, shown above the key (review provider-05).
#: Cloudflare's is the full Workers AI base, not the account URL: its record
#: has no suffix to append (pre-merge review 2026-10-29; the endpoint contract
#: completes a URL that stops at ``/accounts/<id>``).
BASE_URL_FIELDS: dict[str, tuple[str, str]] = {
    "azure": ("Azure resource URL", "https://my-resource.openai.azure.com"),
    "databricks": (
        "Databricks workspace URL",
        "https://adb-1234567890123456.7.azuredatabricks.net",
    ),
    "cloudflare": (
        "Workers AI base URL (ends in /ai/v1)",
        "https://api.cloudflare.com/client/v4/accounts/<account-id>/ai/v1",
    ),
}

#: The loopback ports each local engine listens on by default.
LOCAL_PORTS: dict[str, tuple[int, ...]] = {
    "llama_cpp": (8080,),
    "local_llamacpp": (8080,),
    "local_llamafile": (8080,),
    "ollama": (11434,),
    "local_ollama": (11434,),
    "vllm": (8000,),
    "local_vllm": (8000,),
    "koboldcpp": (5001,),
    "tabbyapi": (5000,),
    "oobabooga": (5000,),
    "aphrodite": (2242,),
    "local_mlx_lm": (8080,),
    "custom": (1234,),
    "custom_2": (1234,),
}

#: An engine whose setup address is a route, not its root: KoboldCpp's send
#: posts KoboldAI's native payload to the saved URL, so the example is that
#: route (the endpoint contract keeps it whole and lists at the root's
#: ``/v1/models``, TASK-35010); a typed root resolves as an OpenAI chat URL.
ENDPOINT_EXAMPLES: dict[str, str] = {
    "koboldcpp": "http://127.0.0.1:5001/api/v1/generate",
}

#: Providers whose saved base already ends in its API version, so the chat
#: route is ``<base>/chat/completions`` -- never ``<base>/v1/chat/completions``
#: (review provider-20 / task-33621.32: Z.ai's ``/api/paas/v4``).
ROUTE_SUFFIXES: dict[str, str] = {"zai": "/chat/completions"}

#: Legacy aliases and the '#2' slot: listed by Settings (task-33002.5 AC#6),
#: kept behind 'Show legacy entries' in first-run setup (review provider-12).
LEGACY_PROVIDER_KEYS: frozenset[str] = frozenset(
    {"custom_2", "local_llamacpp", "local_llm", "local_ollama", "local_vllm", "mistral"}
)

#: Providers whose model list answers without a valid key, so listing it
#: checks nothing; an authenticated check (TASK-34100.2) is the only proof.
PUBLIC_LISTING_PROVIDER_KEYS: frozenset[str] = frozenset(
    {"openrouter", "huggingface", "novita"}
)
#: Providers with a dedicated, non-billable authenticated key endpoint
#: (OpenRouter /api/v1/key, Hugging Face whoami); TASK-34100.2 owns the call.
#: ``check_kind == "authenticated"`` is the hook its Check key reads; until it
#: lands, these providers' listings answer without a key and the wizard says
#: so (``listing_requires_auth``).
AUTHENTICATED_CHECK_PROVIDER_KEYS: frozenset[str] = frozenset(
    {"openrouter", "huggingface"}
)


@dataclass(frozen=True, slots=True)
class ProviderSetupFacts:
    """What the setup form needs to know about one provider.

    Attributes:
        provider_key: Canonical readiness key.
        locality: ``"local"`` (runs on this computer) or ``"cloud"``.
        env_var: The conventional API-key environment variable, if any.
        key_url: Where to get a key (no scheme), or ``None``.
        base_url_required: Whether a per-account base URL must be entered.
        base_url_label: The base-URL field's label, when required.
        base_url_example: An example value for that field.
        local_ports: Default loopback ports for a local engine.
        route_suffix: Chat route appended to the saved base, when the base
            already carries the API version.
        check_kind: ``"model_list"`` (a keyed model listing verifies the
            key), ``"authenticated"`` (a dedicated key endpoint does) or
            ``"none"`` (no non-billable check exists).
        listing_requires_auth: Whether the model listing rejects a bad key.
        legacy: A legacy alias or '#2' slot, hidden from first-run setup.
    """

    provider_key: str
    locality: Locality
    env_var: str | None
    key_url: str | None
    base_url_required: bool
    base_url_label: str | None
    base_url_example: str | None
    local_ports: tuple[int, ...]
    route_suffix: str | None
    check_kind: CheckKind
    listing_requires_auth: bool
    legacy: bool


def key_help_text(facts: ProviderSetupFacts, display_name: str) -> str:
    """'New keys: <url>.' or the dashboard fallback for an unconfirmed page."""
    if facts.key_url:
        return f"New keys: {facts.key_url}."
    return f"Get a key from your {display_name} account dashboard."


def endpoint_example(facts: ProviderSetupFacts) -> str:
    """An example address for the endpoint field, or ""."""
    if facts.base_url_example:
        return facts.base_url_example
    if facts.provider_key in ENDPOINT_EXAMPLES:
        return ENDPOINT_EXAMPLES[facts.provider_key]
    if facts.local_ports:
        return f"http://127.0.0.1:{facts.local_ports[0]}"
    return ""


@lru_cache(maxsize=256)
def provider_setup_facts(provider_key: str) -> ProviderSetupFacts:
    """Return the setup facts for ``provider_key`` (canonical readiness key).

    Args:
        provider_key: A readiness key such as ``"openai"`` or ``"llama_cpp"``.

    Returns:
        The provider's facts; an unknown key gets conservative defaults
        (cloud, model-list check, nothing required).
    """
    from tldw_chatbook.Chat.provider_readiness import (
        KEYLESS_PROVIDER_KEYS,
        PROVIDERS_REQUIRING_BASE_URL_KEYS,
        default_api_key_env_var,
    )
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    key = (provider_key or "").strip()
    record = RECORDS_BY_KEY.get(key)
    keyless = key in KEYLESS_PROVIDER_KEYS
    if record is not None:
        locality: Locality = "local" if record.classification == "local" else "cloud"
    else:
        locality = "local" if keyless else "cloud"
    label, example = BASE_URL_FIELDS.get(key, (None, None))
    if key in AUTHENTICATED_CHECK_PROVIDER_KEYS:
        check_kind: CheckKind = "authenticated"
    elif (
        record is not None
        and record.engine_driven
        and getattr(record, "discovery_route", "models") is None
    ):
        check_kind = "none"
    else:
        check_kind = "model_list"
    return ProviderSetupFacts(
        provider_key=key,
        locality=locality,
        env_var=None if keyless else default_api_key_env_var(key),
        key_url=KEY_URLS.get(key),
        base_url_required=key in PROVIDERS_REQUIRING_BASE_URL_KEYS,
        base_url_label=label,
        base_url_example=example,
        local_ports=LOCAL_PORTS.get(key, ()),
        route_suffix=ROUTE_SUFFIXES.get(key),
        check_kind=check_kind,
        listing_requires_auth=key not in PUBLIC_LISTING_PROVIDER_KEYS,
        legacy=key in LEGACY_PROVIDER_KEYS,
    )
