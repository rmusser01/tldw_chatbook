"""Pure presentation records for task-oriented provider Settings."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from ...Chat.console_provider_endpoints import (
    first_configured_endpoint,
    safe_endpoint_display,
)
from ...Chat.custom_endpoint_registry import (
    CUSTOM_ENDPOINT_ID_PREFIX,
    CustomEndpointEntry,
    build_entry_mutation,
    derive_slug,
    family_execution_key,
    load_custom_endpoints,
    split_custom_endpoint_id,
    validate_entry,
)
from ...Chat.provider_catalog import PROVIDER_CUSTOM_GROUP_KEYS, provider_display_name
from ...Chat.provider_readiness import provider_config_key
from ...config import (
    delete_settings_from_cli_config,
    normalize_provider_config_key,
    save_settings_to_cli_config,
)

logger = logging.getLogger(__name__)


class ProviderCatalogEntry(Protocol):
    """Catalog fields required by the provider picker presentation."""

    readiness_key: str
    display_name: str
    requires_api_key: bool


@dataclass(frozen=True, slots=True)
class SettingsOverviewRow:
    """One user-facing Settings overview row."""

    key: str
    label: str
    value: str


@dataclass(frozen=True, slots=True)
class SettingsOverviewPresentation:
    """Primary user tasks and secondary diagnostic Settings rows."""

    primary_rows: tuple[SettingsOverviewRow, ...]
    advanced_rows: tuple[SettingsOverviewRow, ...]


@dataclass(frozen=True, slots=True)
class ProviderPickerOption:
    """One immutable provider choice or picker action."""

    provider_id: str | None
    label: str
    search_text: str
    saved_unknown: bool = False
    action: Literal["select", "enter_provider_id"] = "select"


@dataclass(frozen=True, slots=True)
class ProviderPickerGroup:
    """One stable provider-picker section."""

    group_id: str
    label: str
    options: tuple[ProviderPickerOption, ...]


class CustomEndpointSessionSettings(Protocol):
    """Session-settings fields the custom endpoint reference guard reads."""

    provider: str


class CustomEndpointSession(Protocol):
    """Console session fields the custom endpoint management seams read."""

    id: str
    settings: CustomEndpointSessionSettings | None


class CustomEndpointReferenceStore(Protocol):
    """Store surface the custom endpoint management seams rely on.

    Implemented by ``ConsoleChatStore``; ``replace_session_settings`` is the
    store's existing wholesale settings mutation path (the same apply path
    the Console settings modal result uses), so detached sessions keep every
    store invariant (payload revisions, user-work marks).
    """

    def sessions(self) -> Sequence[CustomEndpointSession]: ...

    def session_settings(
        self, session_id: str
    ) -> CustomEndpointSessionSettings | None: ...

    def replace_session_settings(
        self, session_id: str, settings: Any
    ) -> object: ...


_PRIMARY_OVERVIEW_ROWS = (
    ("configuration", "Configuration"),
    ("last_connection_test", "Last connection test"),
    ("storage_privacy", "Storage & privacy"),
    ("sync", "Sync"),
)

_ADVANCED_OVERVIEW_ROWS = (
    ("runtime_ownership", "Active source"),
    ("server_binding", "Server connection"),
    ("handoff", "Conversation updates"),
)

_PROVIDER_GROUPS = (
    ("cloud", "Cloud"),
    ("local", "Local"),
    ("custom", "Custom & legacy aliases"),
)


def _overview_rows(
    snapshot: Mapping[str, object],
    definitions: tuple[tuple[str, str], ...],
) -> tuple[SettingsOverviewRow, ...]:
    return tuple(
        SettingsOverviewRow(
            key=key,
            label=label,
            value=str(snapshot.get(key) or "Not available"),
        )
        for key, label in definitions
    )


def build_settings_overview(
    snapshot: Mapping[str, object],
) -> SettingsOverviewPresentation:
    """Build the stable user-task overview from already-resolved display values."""

    return SettingsOverviewPresentation(
        primary_rows=_overview_rows(snapshot, _PRIMARY_OVERVIEW_ROWS),
        advanced_rows=_overview_rows(snapshot, _ADVANCED_OVERVIEW_ROWS),
    )


_CUSTOM_PROVIDER_KEYS = frozenset(
    normalize_provider_config_key(key) for key in PROVIDER_CUSTOM_GROUP_KEYS
)


def _provider_group_id(entry: ProviderCatalogEntry) -> str:
    provider_key = normalize_provider_config_key(entry.readiness_key)
    if provider_key in _CUSTOM_PROVIDER_KEYS:
        return "custom"
    return "cloud" if entry.requires_api_key else "local"


def _matches_query(option: ProviderPickerOption, query: str) -> bool:
    return not query or query in option.search_text.casefold()


def build_provider_picker_groups(
    catalog: Sequence[ProviderCatalogEntry],
    saved_provider: object,
    query: object,
) -> tuple[ProviderPickerGroup, ...]:
    """Build stable searchable provider groups without normalizing saved display text."""

    normalized_query = str(query or "").strip().casefold()
    known_provider_keys = {
        normalize_provider_config_key(entry.readiness_key) for entry in catalog
    }
    grouped: dict[str, list[ProviderPickerOption]] = {
        group_id: [] for group_id, _label in _PROVIDER_GROUPS
    }
    for entry in catalog:
        provider_id = str(entry.readiness_key)
        label = str(entry.display_name)
        option = ProviderPickerOption(
            provider_id=provider_id,
            label=label,
            search_text=f"{label} {provider_id}".casefold(),
        )
        if _matches_query(option, normalized_query):
            grouped[_provider_group_id(entry)].append(option)

    groups: list[ProviderPickerGroup] = []
    saved_text = str(saved_provider or "")
    saved_unknown = bool(saved_text.strip()) and (
        normalize_provider_config_key(saved_text) not in known_provider_keys
    )
    if saved_unknown:
        saved_option = ProviderPickerOption(
            provider_id=saved_text,
            label=f"{saved_text} (saved provider)",
            search_text=saved_text.casefold(),
            saved_unknown=True,
        )
        if _matches_query(saved_option, normalized_query):
            groups.append(
                ProviderPickerGroup("saved", "Saved provider", (saved_option,))
            )

    for group_id, label in _PROVIDER_GROUPS:
        options = tuple(
            sorted(
                grouped[group_id],
                key=lambda option: (
                    option.label.casefold(),
                    str(option.provider_id).casefold(),
                ),
            )
        )
        if options:
            groups.append(ProviderPickerGroup(group_id, label, options))

    groups.append(
        ProviderPickerGroup(
            "actions",
            "Other",
            (
                ProviderPickerOption(
                    provider_id=None,
                    label="Enter provider ID",
                    search_text="enter provider id manual custom",
                    action="enter_provider_id",
                ),
            ),
        )
    )
    return tuple(groups)


#: User-facing family labels for the custom endpoints overview rows.
_CUSTOM_ENDPOINT_FAMILY_LABELS = {
    "llama_cpp": "llama.cpp",
    "openai_compatible": "OpenAI-compatible",
    "ollama": "Ollama",
}

#: The built-in provider slots eligible for one-way conversion (ADR-146).
_CONVERTIBLE_SLOT_IDS = frozenset({"custom", "custom_2"})


def custom_endpoint_rows(
    app_config: Mapping[str, object],
) -> tuple[SettingsOverviewRow, ...]:
    """Build one overview row per registry entry (ADR-146 task-7).

    Each row carries the display name, family label,
    ``safe_endpoint_display(base_url)`` (credentials never render), and the
    cached model count. Rows follow ``load_custom_endpoints`` iteration
    order, which is the same creation (config file) order
    ``build_console_provider_options`` renders entries in.

    Args:
        app_config: The full CLI config mapping.

    Returns:
        Rows keyed by the entry's ``custom-ep:<slug>`` provider id; empty
        when the registry has no valid entries.
    """
    rows: list[SettingsOverviewRow] = []
    for entry in load_custom_endpoints(app_config).values():
        family_label = _CUSTOM_ENDPOINT_FAMILY_LABELS.get(
            entry.family, entry.family
        )
        endpoint_display = safe_endpoint_display(entry.base_url) or "not set"
        model_count = len(entry.models)
        models_label = f"{model_count} model" if model_count == 1 else f"{model_count} models"
        rows.append(
            SettingsOverviewRow(
                key=f"{CUSTOM_ENDPOINT_ID_PREFIX}{entry.slug}",
                label=entry.display_name,
                value=f"{family_label} · {endpoint_display} · {models_label}",
            )
        )
    return tuple(rows)


def conversations_referencing_endpoint(
    store: CustomEndpointReferenceStore, provider_id: str
) -> list[str]:
    """Return the ids of sessions whose settings use ``provider_id``.

    Args:
        store: Console store holding the live native sessions.
        provider_id: Candidate provider id (e.g. ``custom-ep:gpu``).

    Returns:
        Session ids, in ``store.sessions()`` order; empty when nothing
        references the provider.
    """
    referencing: list[str] = []
    for session in store.sessions():
        settings = getattr(session, "settings", None)
        if getattr(settings, "provider", None) == provider_id:
            referencing.append(str(session.id))
    return referencing


def _detach_base_url(
    entry: CustomEndpointEntry, settings: CustomEndpointSessionSettings
) -> str | None:
    """Return the ``base_url`` a detached session keeps.

    A session that already carries its own ``base_url`` keeps it
    untouched; a blank one (the registry entry was its only endpoint
    source) takes the entry's ``base_url``, which ``load_custom_endpoints``
    already family-normalized (llama families through the same llama
    normalization the load path uses). Without this, a blank session URL
    would fall back to the family default endpoint, losing the
    conversation's current endpoint — contradicting the detach copy,
    status text, and user guide.
    """
    current = getattr(settings, "base_url", None)
    if isinstance(current, str) and current.strip():
        return current
    return entry.base_url


def detach_and_delete_entry(
    app_config: Mapping[str, object],
    store: CustomEndpointReferenceStore | None,
    provider_id: str,
) -> None:
    """Detach every referencing session, then delete the registry entry.

    Detach re-points each referencing session's ``settings.provider`` at the
    entry's family execution key. A session that already carries a
    ``base_url`` keeps it untouched; a blank one (the registry entry was
    its only endpoint source) takes the entry's ``base_url`` — already
    family-normalized by ``load_custom_endpoints`` — so in both cases the
    endpoint survives as a session-only override instead of falling back
    to the family default. The session settings are frozen dataclasses
    replaced wholesale through the store's ``replace_session_settings``
    path (the settings-modal apply seam). The entry is then removed from
    config via ``delete_settings_from_cli_config("custom_endpoints", [slug])``.

    Args:
        app_config: The full CLI config mapping (entry family source).
        store: Console store holding the live native sessions, or None when
            no Console store exists (no live sessions to detach).
        provider_id: A ``custom-ep:<slug>`` id naming a valid entry.

    Raises:
        KeyError: Unknown session id reported by the store.
        RuntimeError: The config delete failed (referencing sessions that
            were detached keep their conversation-only endpoints; retrying
            the delete is safe).
    """
    slug = split_custom_endpoint_id(provider_id)
    if slug is None:
        return
    entry = load_custom_endpoints(app_config).get(slug)
    if entry is None:
        return
    family_provider = family_execution_key(entry.family)
    if store is not None:
        for session_id in conversations_referencing_endpoint(store, provider_id):
            settings = store.session_settings(session_id)
            if settings is None:
                continue
            store.replace_session_settings(
                session_id,
                replace(
                    settings,
                    provider=family_provider,
                    base_url=_detach_base_url(entry, settings),
                ),
            )
    if not delete_settings_from_cli_config("custom_endpoints", [slug]):
        logger.warning(
            "Custom endpoint '%s' was detached but could not be deleted "
            "from config; it remains on disk.",
            slug,
        )
        raise RuntimeError(
            f"Could not delete custom endpoint '{slug}' from config; it "
            "remains on disk."
        )


def convert_slot_to_named_endpoint(
    app_config: Mapping[str, object], slot_id: str
) -> str:
    """Create a registry entry from a built-in custom slot's config.

    The entry (family ``openai_compatible``, ``created_from`` = the slot id)
    is persisted through ``save_settings_to_cli_config``; the slot's own
    config is left untouched (ADR-146's one-way convert). Only the
    credential *reference* (``api_key_env``, a variable name) is carried
    over -- stored secrets are never duplicated into the registry.

    Args:
        app_config: The full CLI config mapping (slot settings source).
        slot_id: One of ``custom`` / ``custom_2``.

    Returns:
        The new entry's ``custom-ep:<slug>`` provider id.

    Raises:
        ValueError: Unknown slot id, no configured endpoint, or the slot's
            endpoint fails entry validation.
        RuntimeError: The config write failed.
    """
    if slot_id not in _CONVERTIBLE_SLOT_IDS:
        raise ValueError(
            f"Only the custom and custom_2 slots can be converted "
            f"(got '{slot_id}')."
        )
    slot_settings = _slot_provider_settings(app_config, slot_id)
    base_url = first_configured_endpoint(slot_settings)
    if not base_url:
        raise ValueError(
            f"Slot '{slot_id}' has no configured endpoint to convert."
        )
    base_url = base_url.rstrip("/")
    display_name = provider_display_name(slot_id)
    errors = validate_entry(display_name, "openai_compatible", base_url)
    if errors:
        raise ValueError(" ".join(errors))
    slug = derive_slug(slot_id, load_custom_endpoints(app_config).keys())
    entry = CustomEndpointEntry(
        slug=slug,
        display_name=display_name,
        family="openai_compatible",
        base_url=base_url,
        api_key_env=_optional_env_var(slot_settings.get("api_key_env")),
        models=_slot_models(slot_settings),
        created_from=slot_id,
    )
    if not save_settings_to_cli_config(build_entry_mutation(entry)):
        raise RuntimeError(
            f"Could not save the converted endpoint '{slug}' to config."
        )
    return f"{CUSTOM_ENDPOINT_ID_PREFIX}{slug}"


def _slot_provider_settings(
    app_config: Mapping[str, object], slot_id: str
) -> Mapping[str, object]:
    """Return the ``api_settings`` table whose key normalizes to ``slot_id``."""
    api_settings = app_config.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        return {}
    for configured_provider, configured_settings in api_settings.items():
        if (
            provider_config_key(str(configured_provider)) == slot_id
            and isinstance(configured_settings, Mapping)
        ):
            return configured_settings
    return {}


def _slot_models(slot_settings: Mapping[str, object]) -> tuple[str, ...]:
    """Keep the slot's configured model list (``models`` then ``model``)."""
    listed = slot_settings.get("models")
    if isinstance(listed, (list, tuple)):
        models = tuple(
            model.strip()
            for model in listed
            if isinstance(model, str) and model.strip()
        )
        if models:
            return models
    model = slot_settings.get("model")
    if isinstance(model, str) and model.strip():
        return (model.strip(),)
    return ()


def _optional_env_var(value: object) -> str | None:
    """Return a non-blank credential variable name, or None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
