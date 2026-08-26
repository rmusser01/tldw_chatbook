"""Pure presentation records for task-oriented provider Settings."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from ...Chat.provider_catalog import PROVIDER_CUSTOM_GROUP_KEYS
from ...config import normalize_provider_config_key


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
