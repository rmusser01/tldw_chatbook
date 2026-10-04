"""The Settings ▸ Providers & Models card (TASK-33007.1).

The card's composition, moved verbatim out of ``SettingsScreen`` (DESIGN.md's
One Home Rule). Only composition lives here: the card's handlers, workers and
state stay on the screen, which still receives every event the card's
widgets bubble (moving them is task-1378's remaining scope).

``settings_screen`` imports this module inside ``_render_provider_detail``,
so the Settings route's pre-import payload does not grow (ADR-097); by then
``settings_screen`` is fully loaded, which is what makes the import of its
constants below safe.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING

from textual import events
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.errors import NoWidget
from textual.message import Message
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    OptionList,
    Select,
    SelectionList,
    Static,
)

from ...Chat.console_provider_endpoints import (
    first_configured_endpoint,
    safe_endpoint_display,
)
from ...Chat.console_provider_support import MODEL_FIELD_LABELS
from ...Chat.custom_endpoint_registry import (
    CUSTOM_ENDPOINT_ID_PREFIX,
    load_custom_endpoints,
)
from ...Chat.provider_catalog import provider_display_name
from ...Chat.provider_readiness import (
    default_api_key_env_var,
    get_provider_readiness,
    provider_config_key,
)
from ...config import provider_settings_for_key
from ...LLM_Provider_Catalog.model_catalog_settings import (
    AUTO_REFRESH_PROVIDER_LIST_KEYS,
)
from ...Widgets.model_search_picker import ModelSearchPicker, PickerSearchInput
from ..Screens.settings_context_memory import model_context_window_state
from ..Screens.settings_provider_view_model import (
    custom_endpoint_rows,
    provider_picker_summary,
)
from ..Screens.settings_screen import (
    ANTHROPIC_API_KEY_GUIDANCE_COPY,
    ANTHROPIC_SUBSCRIPTION_GUIDANCE_COPY,
    API_URL_PROVIDER_KEYS,
    INSTANT_APPLY_BEHAVIOR_COPY,
    MODEL_DISCOVERY_CAPABILITY_WARNING,
    MODEL_DISCOVERY_EMPTY_COPY,
    MODEL_PROFILE_STREAMING_SELECT_OPTIONS,
    PROVIDER_MANUAL_SELECT_VALUE,
    PROVIDER_MODEL_PROFILE_FIELD_KEYS,
    PROVIDER_TEST_GUIDANCE,
    QWENCLOUD_API_MODE_HELP_COPY,
    QWENCLOUD_API_MODE_INVALID_COPY,
    QWENCLOUD_API_MODE_OPTIONS,
    QWENCLOUD_PROVIDER_TABLE_INVALID_COPY,
    ProviderEndpointURLValidator,
    SettingsRegion,
    SettingsURLInput,
    _anthropic_auth_source_options,
    _anthropic_auth_sources,
    _fold_long_tokens,
)

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from ..Screens.settings_screen import SettingsScreen


#: TASK-33007.2: the Source word for where the shown provider or model comes
#: from (spec §6 row grammar). Keys are exactly the sources
#: ``resolve_effective_provider_model`` returns (TASK-1310: a stale key once
#: rendered a raw fallback label here).
SELECTION_SOURCE_WORDS = {
    "settings_draft": "edited *",
    "chat_defaults": "new-chat default",
    "console_session": "this chat",
    "default": "built-in",
}
#: The Key check row's action, labelled with the key that runs it (AC#6).
KEY_CHECK_ACTION_LABEL = "Test (t)"
#: ADR-012:29: the env var is the safer path, and the field holds a name.
ENV_VAR_HELP_COPY = "safer: keeps keys out of config.toml"
#: TASK-33007.3: the Default model row's help; Custom ID shows while the
#: picker holds focus.
MODEL_PICKER_HELP_COPY = "type to search · Custom ID for others"
PROVIDER_CONTROL_TOOLTIP = (
    "Type to filter providers by name or ID; Up/Down move, Enter chooses, "
    "Esc keeps the current provider."
)
#: Focus can pass to the list's parent on a click; the list closes only
#: once focus has really left the control (ModelSearchPicker's delay).
_BLUR_CLOSE_DELAY_SECONDS = 0.05


class ProviderFilterInput(PickerSearchInput):
    """The one-row Provider control (TASK-33007.2 AC#1, AC#3).

    It shows the chosen provider's name. Typing filters the list under it,
    which never takes focus, so the control is one Tab stop: Up/Down move
    the list's highlight, Enter chooses, and Escape keeps the current
    provider.
    """

    BINDINGS = [
        Binding("down", "move_highlight(1)", "Next provider", show=False),
        Binding("up", "move_highlight(-1)", "Previous provider", show=False),
    ]

    def _picker(self) -> OptionList:
        return self.screen.query_one("#settings-provider-picker", OptionList)

    def action_move_highlight(self, step: int) -> None:
        """Open the list, or move its highlight by one row.

        Args:
            step: 1 for the next row, -1 for the previous one.
        """
        picker = self._picker()
        if not picker.display:
            open_provider_list(self.screen)
        elif step > 0:
            picker.action_cursor_down()
        else:
            picker.action_cursor_up()

    async def action_submit(self) -> None:
        """Choose the highlighted provider while the list is open."""
        picker = self._picker()
        if picker.display:
            picker.action_select()

    def _on_key(self, event: events.Key) -> None:
        if event.key == "escape" and self._picker().display:
            # The list closes and Escape goes on to the screen, which releases
            # the field (task-1560), so the footer's "Esc, s" holds (ADR-031).
            close_provider_list(self.screen)

    def _on_blur(self, event: events.Blur) -> None:
        self.set_timer(_BLUR_CLOSE_DELAY_SECONDS, self._close_after_blur)

    def _close_after_blur(self) -> None:
        if not self.is_mounted or self.has_focus:
            return
        if self._pressed_on_open_list():
            # The press moved focus to the scrolling pane; App sends the Click
            # that chooses only at the release, so keep the list until then.
            self.focus(scroll_visible=False)
            return
        close_provider_list(self.screen)

    def _pressed_on_open_list(self) -> bool:
        """Whether focus left for a mouse press on the open list.

        Returns:
            True when the list is open, focus went to a container that holds
            it, and the pointer is over the list (or its scrollbar).
        """
        try:
            picker = self._picker()
            focused = self.screen.focused
            if not picker.display or focused not in picker.ancestors:
                return False
            under, _region = self.screen.get_widget_at(*self.app.mouse_position)
        except (QueryError, NoWidget):  # e.g. the card unmounted meanwhile
            return False
        return picker in under.ancestors_with_self


class DefaultModelPicker(ModelSearchPicker):
    """The Default model control (TASK-33007.3).

    The shared picker, scoped to the provider the form holds. The hidden
    ``Input#settings-model-value`` beside it stays the value that staging,
    save and revert read; the screen keeps the two in step.
    """

    def on_mount(self) -> None:
        """Give the field the one-row Connect edge (task-1586).

        The base ``on_mount`` still runs after this one (MRO dispatch).
        """
        self.query_one("#model-search-picker-input", Input).add_class(
            "settings-compact-input"
        )

    def _current_provider(self) -> str | None:
        # A manual provider key leaves the provider Select on its manual
        # entry, so ask the form, not the Select.
        return self.screen._provider_widget_value() or None

    def _render_matches(self, query: str, *, show_empty_query: bool = False) -> None:
        if not self.query_one("#model-search-picker-input", Input).has_focus:
            # The field posts Changed for its resting value on mount; the
            # list opens only for the field being edited (as Provider's does).
            return
        super()._render_matches(query, show_empty_query=show_empty_query)
        if self._committed_index is not None:
            # AC#1: the saved default is highlighted as soon as the list opens.
            self.query_one(
                "#model-search-picker-results", OptionList
            ).highlighted = self._committed_index

    def _cancel_edit(self, event: Message) -> None:
        if not self.custom_mode:
            # An unfinished filter is dropped and the list closes.
            super()._cancel_edit(event)
        # task-1560: one Esc also leaves the field, so the footer's "Esc, s"
        # holds -- and saves a typed Custom ID instead of dropping it. Queued
        # after any refocus the cancel itself queued.
        event.stop()
        self.app.call_later(self.screen.set_focus, None)


def open_provider_list(screen: SettingsScreen) -> None:
    """Show the provider list, filtered by what the control holds.

    Args:
        screen: The Settings screen that owns the list.
    """
    try:
        screen.query_one("#settings-provider-picker", OptionList).display = True
    except QueryError:
        return
    screen._refresh_provider_picker()


def close_provider_list(screen: SettingsScreen) -> None:
    """Hide the list and show the chosen provider's name in the control.

    Args:
        screen: The Settings screen that owns the list.
    """
    try:
        screen.query_one("#settings-provider-picker", OptionList).display = False
    except QueryError:
        return
    if not sync_provider_control(screen):
        screen._refresh_provider_picker("")


def shown_provider_label(screen: SettingsScreen) -> str:
    """Return the name the Provider control shows at rest.

    Args:
        screen: The Settings screen holding the draft.

    Returns:
        The display name of the provider the form holds (its draft, else the
        saved or navigated one), or "" when none is chosen.
    """
    provider = str(screen._provider_display_setting_values().get("provider") or "")
    return screen._provider_display_label(provider) if provider.strip() else ""


def sync_provider_control(screen: SettingsScreen) -> bool:
    """Show the chosen provider's display name unless the user is choosing.

    Args:
        screen: The Settings screen that owns the control.

    Returns:
        Whether the name changed (the list was then rebuilt unfiltered).
    """
    try:
        control = screen.query_one("#settings-provider-search", Input)
        picker = screen.query_one("#settings-provider-picker", OptionList)
    except QueryError:
        return False
    if picker.display:
        return False
    label = shown_provider_label(screen)
    if control.value == label:
        return False
    with control.prevent(Input.Changed):
        control.value = label
    if control.has_focus:
        control.select_all()  # After a choice, typing filters afresh.
    screen._refresh_provider_picker("")
    return True


def selection_source_word(source: object) -> str:
    """Return the Source word for a provider or model source.

    Args:
        source: An ``EffectiveProviderModel`` source key.

    Returns:
        E.g. "new-chat default" or "edited *".
    """
    return SELECTION_SOURCE_WORDS.get(str(source or ""), "built-in")


def api_key_row_copy(screen: SettingsScreen, provider: str) -> tuple[str, str]:
    """Say where the provider's key comes from, in words (ADR-012:29).

    Args:
        screen: The Settings screen holding the draft.
        provider: The provider the form holds.

    Returns:
        The API key row's Source word and its one-line help.
    """
    # TASK-34201: an unsaved Sign in with choice shows before Save.
    readiness = get_provider_readiness(
        provider,
        screen._provider_auth_readiness_config(provider),
        background_credentials=True,
    )
    registry = screen._provider_registry_credential_text(provider, readiness)
    if registry is not None:
        return "this endpoint", registry
    if readiness.subscription_status is not None:
        return "subscription", screen._subscription_credential_copy(
            readiness.subscription_status
        )
    if readiness.reason == "Invalid provider settings":
        return "invalid", "repair in Advanced Config or config.toml"
    draft = screen._provider_draft()
    if draft is not None and "api_key" in draft.dirty_keys:
        if str(draft.values.get("api_key") or "").strip():
            return "edited *", "masked · s saves it to config"
        return "cleared *", "s removes the saved key"
    if screen._provider_saved_api_key_present(provider):
        return "saved in config", "masked · used before the env var"
    if (readiness.api_key_source or "").startswith("env:"):
        return "from env var", f"{readiness.env_var} in your shell"
    if not readiness.requires_api_key:
        return "not required", "this provider needs no key"
    if readiness.env_var:
        return "missing", f"paste one, or set {readiness.env_var}"
    return "missing", "paste one to save it in config"


def env_var_source_word(screen: SettingsScreen, provider: str, env_var: str) -> str:
    """Say whether the key's env var is set in this shell.

    Args:
        screen: The Settings screen holding the draft.
        provider: The provider the form holds.
        env_var: The Env var field's value; blank means the default name.

    Returns:
        The Env var row's Source word.
    """
    if screen._provider_is_registry_id(provider):
        return "this endpoint"
    draft = screen._provider_draft()
    if draft is not None and "credential_env_var" in draft.dirty_keys:
        return "edited *"
    name = env_var.strip() or default_api_key_env_var(provider_config_key(provider))
    if not name:
        return "not used"
    return "set in shell" if os.environ.get(name, "").strip() else "not set"


def endpoint_row_copy(
    screen: SettingsScreen, provider: str, endpoint: str
) -> tuple[str, str]:
    """Say where the endpoint comes from and what a blank field means.

    Args:
        screen: The Settings screen holding the draft.
        provider: The provider the form holds.
        endpoint: The Endpoint field's value.

    Returns:
        The Endpoint row's Source word and its one-line help.
    """
    registry = screen._provider_registry_endpoint(provider)
    if registry is not None:
        url = registry[1] or "endpoint not found"
        return "this endpoint", f"{url} · edit in Custom endpoints"
    provider_key = provider_config_key(provider)
    draft = screen._provider_draft()
    if draft is not None and "endpoint" in draft.dirty_keys:
        word = "edited *"
    elif endpoint.strip():
        word = "config"
    elif provider_key in API_URL_PROVIDER_KEYS:
        word = "not set"
    else:
        word = "built-in"
    if provider_key in API_URL_PROVIDER_KEYS:
        return word, "required: the server's base URL"
    return word, "blank uses the provider default"


def key_check_verdict(screen: SettingsScreen) -> str:
    """Return this provider's latest readiness word (AC#6, spec §5).

    The last check's Readiness row while it describes the draft; otherwise
    the word the shared evidence gives the draft, e.g. 'Ready · not tested'.

    Args:
        screen: The Settings screen holding the draft and its evidence.

    Returns:
        The Key check row's verdict.
    """
    label, _gap, word = screen._provider_test_result.partition("\n")[0].partition(" ")
    if label == "Readiness" and word.strip():
        return word.strip()
    provider = screen._provider_widget_value()
    try:
        model = screen.query_one("#settings-model-value", Input).value.strip()
    except QueryError:
        model = str(screen._provider_setting_values_mapping().get("model") or "")
    try:
        staged = screen._provider_test_staged_config(provider)
        readiness = get_provider_readiness(
            provider, staged, background_credentials=True
        )
    except ValueError:
        # A draft env var name readiness rejects is the Env var row's error
        # to show; the verdict falls back to the saved settings meanwhile.
        readiness = get_provider_readiness(
            provider,
            screen._provider_readiness_app_config(),
            background_credentials=True,
        )
    identity = screen._provider_current_draft_identity()
    evidence = (
        screen._provider_evidence_store().evidence_for(identity)
        if identity is not None
        else None
    )
    rows = screen._provider_test_rows(
        readiness, display_name="", model=model.strip(), endpoint="", evidence=evidence
    )
    return rows[0][1]


def refresh_connect_rows(screen: SettingsScreen, provider: str, endpoint: str) -> None:
    """Re-say every Connect row's Source word, help and verdict.

    Args:
        screen: The Settings screen that owns the card.
        provider: The provider the form holds.
        endpoint: The Endpoint field's value.
    """
    resolved = screen._resolve_provider_model_for_settings()
    sync_provider_control(screen)
    screen._set_static_text(
        "#settings-provider-source", selection_source_word(resolved.provider_source)
    )
    screen._set_static_text(
        "#settings-model-source", selection_source_word(resolved.model_source)
    )
    try:
        env_var = screen.query_one(
            "#settings-provider-credential-env-var", Input
        ).value
    except QueryError:
        env_var = ""
    screen._set_static_text(
        "#settings-provider-env-var-source",
        env_var_source_word(screen, provider, env_var),
    )
    endpoint_word, endpoint_help = endpoint_row_copy(screen, provider, endpoint)
    screen._set_static_text("#settings-provider-endpoint-source", endpoint_word)
    screen._set_static_text("#settings-provider-endpoint-help", endpoint_help)
    refresh_key_rows(screen, provider)


def refresh_key_rows(screen: SettingsScreen, provider: str) -> None:
    """Re-say the API key row's source and the Key check verdict.

    Args:
        screen: The Settings screen that owns the card.
        provider: The provider the form holds.
    """
    key_word, key_help = api_key_row_copy(screen, provider)
    screen._set_static_text("#settings-provider-key-status", key_word)
    screen._set_static_text("#settings-provider-api-key-help", key_help)
    screen._set_static_text("#settings-provider-readiness", key_check_verdict(screen))


def compose_providers_models_card(screen: SettingsScreen) -> ComposeResult:
    """Compose the Providers & Models section title and card.

    Args:
        screen: The Settings screen that owns the card's state and handlers.

    Yields:
        The section title, then the card container with every row.
    """
    from tldw_chatbook.LLM_Management import (
        snapshot_settings as snapshot_preferences,
    )

    if screen._snapshot_preferences_loaded is None:
        try:
            screen._snapshot_preferences_loaded = (
                snapshot_preferences.load_snapshot_preferences()
            )
        except (ValueError, OSError):
            screen._snapshot_preferences_loaded = None
        screen._snapshot_preferences_unavailable = (
            screen._snapshot_preferences_loaded is None
        )
        screen._snapshot_preferences_raw = (
            (
                screen._snapshot_preferences_loaded.enabled,
                str(screen._snapshot_preferences_loaded.keep_count),
            )
            if screen._snapshot_preferences_loaded
            else (False, "")
        )
        screen.call_after_refresh(screen._update_guided_action_widgets)
    resolved = screen._resolve_provider_model_for_settings()
    values = screen._provider_display_setting_values()
    provider = str(values["provider"])
    context_window_state = model_context_window_state(
        screen._app_config_mapping(), provider, str(values["model"])
    )
    # Qodo #2876: a registry default is edited in Custom endpoints.
    registry_locked = screen._provider_is_registry_id(provider)
    yield Static(
        "Providers & Models", classes="destination-section settings-column-title"
    )
    provider_card = Vertical(
        id="settings-providers-models-card", classes="settings-focus-card"
    )
    provider_card.disabled = screen._vllm_default_recovery() is not None
    with provider_card:
        with Collapsible(
            title="Prompt-cache snapshots",
            collapsed=True,
            id="settings-snapshot-controls",
        ):
            yield Static(
                "Save processed context to reuse later. Restoring does not change your conversations.",
                classes="settings-help-copy",
            )
            yield Static(
                "Enable/disable applies on next launch.",
                id="settings-snapshot-launch-scope",
                classes="settings-help-copy",
            )
            yield Checkbox(
                "Enable snapshots",
                value=screen._snapshot_preferences_raw[0],
                disabled=screen._snapshot_preferences_unavailable,
                id="settings-snapshot-enabled",
            )
            yield Static(
                "Keep count (1–1000, across all models)",
                classes="settings-input-label",
            )
            yield Input(
                screen._snapshot_preferences_raw[1],
                disabled=screen._snapshot_preferences_unavailable,
                id="settings-snapshot-keep",
                type="integer",
            )
            yield Static(
                screen._SNAPSHOT_PREFERENCES_UNAVAILABLE_COPY
                if screen._snapshot_preferences_unavailable
                else "Draft — use category Save / Revert. Enable/disable applies on next launch.",
                id="settings-snapshot-result",
                classes="settings-help-copy",
            )
        # TASK-33007.2: Connect is one row per fact -- Provider, API key, Env
        # var, Endpoint -- each with a Source word and a one-line help, and it
        # ends in the Key check row. The Test result's labelled rows live in
        # the Inspector's Key block, so a test never pushes Default model down.
        yield Static(
            "Connect",
            id="settings-provider-connect-title",
            classes="destination-section",
        )
        picker_groups = screen._provider_picker_groups()
        with Horizontal(id="settings-provider-row", classes="settings-input-row"):
            yield Static("Provider", classes="settings-input-label")
            yield ProviderFilterInput(
                value=shown_provider_label(screen),
                id="settings-provider-search",
                classes="settings-compact-input",
                placeholder="Type to filter providers",
                tooltip=PROVIDER_CONTROL_TOOLTIP,
            )
            yield Static(
                selection_source_word(resolved.provider_source),
                id="settings-provider-source",
                classes="settings-source-word",
                markup=False,
            )
            yield Static(
                provider_picker_summary(picker_groups),
                id="settings-provider-search-status",
                classes="settings-row-help",
                markup=False,
            )
        picker = OptionList(
            *screen._provider_picker_options(picker_groups),
            id="settings-provider-picker",
            compact=True,
        )
        # One Tab stop (AC#1): the list follows the control's keys and opens
        # only while the user is choosing.
        picker.can_focus = False
        picker.display = False
        # task-16480: compose-time highlight so the configured provider is
        # selected on the very first paint; the post-refresh highlight
        # arrives too early (pre-mount) to serve as the only source.
        screen._apply_provider_picker_highlight(picker)
        yield picker
        with Horizontal(
            classes="settings-input-row settings-provider-manual-hidden"
        ):
            yield Select(
                screen._provider_select_options(),
                value=screen._provider_select_value_for_provider(provider),
                id="settings-provider-value",
                classes="settings-compact-select",
                allow_blank=False,
                compact=True,
            )
        manual_provider_classes = "settings-input-row"
        if (
            screen._provider_select_value_for_provider(provider)
            != PROVIDER_MANUAL_SELECT_VALUE
        ):
            manual_provider_classes += " settings-provider-manual-hidden"
        with Horizontal(
            id="settings-provider-manual-row", classes=manual_provider_classes
        ):
            yield Static("Manual", classes="settings-input-label")
            yield Input(
                value=str(values["provider"])
                if screen._provider_select_value_for_provider(provider)
                == PROVIDER_MANUAL_SELECT_VALUE
                else "",
                id="settings-provider-manual-value",
                classes="settings-compact-input",
                placeholder="Custom provider key",
                disabled=(
                    screen._provider_select_value_for_provider(provider)
                    != PROVIDER_MANUAL_SELECT_VALUE
                ),
            )
        edit_endpoint = Button(
            "Edit this endpoint in Custom endpoints",
            id="settings-provider-edit-custom-endpoint",
            tooltip=(
                "This provider is a named custom endpoint: its URL, "
                "credential, and models are edited there."
            ),
        )
        edit_endpoint.display = registry_locked and (
            screen._provider_registry_entry(provider) is not None
        )
        yield edit_endpoint
        # TASK-34201: Anthropic only -- an API key or the Claude subscription.
        # The choice comes before the key rows it disables (kept visible).
        is_anthropic = provider_config_key(provider) == "anthropic"
        subscription_selected = (
            is_anthropic
            and screen._provider_auth_source_value(provider)
            == _anthropic_auth_sources()[1]
        )
        with Horizontal(
            id="settings-provider-auth-source-row",
            classes=(
                "settings-input-row settings-select-row"
                if is_anthropic
                else "settings-input-row settings-select-row settings-gated-profile-hidden"
            ),
        ):
            yield Static("Sign in with", classes="settings-input-label")
            yield Select(
                _anthropic_auth_source_options(),
                value=screen._provider_auth_source_value(provider),
                id="settings-provider-auth-source",
                classes="settings-compact-select",
                allow_blank=False,
                compact=True,
                disabled=not is_anthropic,
            )
        yield Static(
            ANTHROPIC_SUBSCRIPTION_GUIDANCE_COPY
            if subscription_selected
            else ANTHROPIC_API_KEY_GUIDANCE_COPY,
            id="settings-provider-auth-source-guidance",
            classes=(
                "settings-status-row"
                if is_anthropic
                else "settings-status-row settings-gated-profile-hidden"
            ),
        )
        key_word, key_help = api_key_row_copy(screen, provider)
        with Horizontal(id="settings-provider-api-key-row", classes="settings-input-row"):
            yield Static("API key", classes="settings-input-label")
            yield Input(
                value=str(values.get("api_key") or ""),
                id="settings-provider-api-key",
                classes="settings-compact-input",
                placeholder=screen._provider_api_key_placeholder(provider),
                password=True,
                disabled=registry_locked or subscription_selected,
            )
            yield Button(
                "Clear",
                id="settings-provider-api-key-clear",
                compact=True,
                disabled=subscription_selected or (
                    not screen._provider_saved_api_key_present(provider)
                    and not bool(str(values.get("api_key") or "").strip())
                ),
                tooltip="Clear the API key saved in local config for this provider.",
            )
            yield Static(
                key_word,
                id="settings-provider-key-status",
                classes="settings-source-word",
                markup=False,
            )
            yield Static(
                key_help,
                id="settings-provider-api-key-help",
                classes="settings-row-help",
                markup=False,
            )
        with Horizontal(id="settings-provider-env-var-row", classes="settings-input-row"):
            yield Static("Env var", classes="settings-input-label")
            yield Input(
                value=str(values["credential_env_var"]),
                id="settings-provider-credential-env-var",
                classes="settings-compact-input",
                placeholder=screen._provider_credential_placeholder(provider),
                disabled=registry_locked or subscription_selected,
            )
            yield Static(
                env_var_source_word(
                    screen, provider, str(values["credential_env_var"])
                ),
                id="settings-provider-env-var-source",
                classes="settings-source-word",
                markup=False,
            )
            yield Static(
                ENV_VAR_HELP_COPY,
                id="settings-provider-credential-guidance",
                classes="settings-row-help",
                markup=False,
            )
        endpoint_word, endpoint_help = endpoint_row_copy(
            screen, provider, str(values["endpoint"])
        )
        with Horizontal(id="settings-provider-endpoint-row", classes="settings-input-row"):
            yield Static("Endpoint", classes="settings-input-label")
            yield SettingsURLInput(
                value=str(values["endpoint"]),
                id="settings-provider-endpoint-value",
                classes="settings-compact-input",
                placeholder=screen._provider_endpoint_placeholder(provider),
                validators=[ProviderEndpointURLValidator()],
                validate_on={"blur", "submitted"},
                disabled=registry_locked,
            )
            yield Static(
                endpoint_word,
                id="settings-provider-endpoint-source",
                classes="settings-source-word",
                markup=False,
            )
            yield Static(
                endpoint_help,
                id="settings-provider-endpoint-help",
                classes="settings-row-help",
                markup=False,
            )
        api_mode_value, api_mode_valid = screen._provider_api_mode_display_value(
            provider
        )
        provider_table_malformed = screen._qwencloud_provider_table_is_malformed(
            provider
        )
        api_mode_row_classes = "settings-input-row settings-select-row"
        if provider_config_key(provider) != "qwencloud":
            api_mode_row_classes += " settings-gated-profile-hidden"
        with Horizontal(
            id="settings-provider-api-mode-row", classes=api_mode_row_classes
        ):
            yield Static("API mode", classes="settings-input-label")
            yield Select(
                QWENCLOUD_API_MODE_OPTIONS,
                value=api_mode_value,
                id="settings-provider-api-mode",
                prompt="Choose Responses or Chat Completions",
                classes=(
                    "settings-compact-select"
                    if api_mode_valid and not provider_table_malformed
                    else "settings-compact-select settings-invalid-input"
                ),
                allow_blank=True,
                compact=True,
                disabled=provider_config_key(provider) != "qwencloud",
            )
        yield Static(
            (
                QWENCLOUD_PROVIDER_TABLE_INVALID_COPY
                if provider_config_key(provider) == "qwencloud"
                and provider_table_malformed
                else (
                    QWENCLOUD_API_MODE_INVALID_COPY
                    if provider_config_key(provider) == "qwencloud"
                    and not api_mode_valid
                    else QWENCLOUD_API_MODE_HELP_COPY
                )
            ),
            id="settings-provider-api-mode-guidance",
            classes=(
                "settings-status-row"
                if provider_config_key(provider) == "qwencloud"
                else "settings-status-row settings-gated-profile-hidden"
            ),
        )
        hosted_guidance = screen._hosted_provider_guidance(
            provider, values.get("model")
        )
        yield Static(
            hosted_guidance,
            id="settings-hosted-provider-guidance",
            classes=(
                "settings-status-row"
                if hosted_guidance
                else "settings-status-row settings-gated-profile-hidden"
            ),
        )
        reconnect = Button(
            "Review restored OpenAI connection",
            id="settings-openai-reconnect-review",
        )
        reconnect.display = provider_config_key(provider) == "openai"
        reconnect.disabled = screen._openai_reconnect_busy
        yield reconnect
        # task-189: the Test affordance closes the first-run Connect job.
        # TASK-33005.4: 't' is the non-generating key check (D2).
        with Horizontal(
            id="settings-provider-key-check-row", classes="settings-input-row"
        ):
            yield Static("Key check", classes="settings-input-label")
            yield Static(
                key_check_verdict(screen),
                id="settings-provider-readiness",
                classes="settings-key-check-verdict",
                markup=False,
            )
            test_button = Button(
                KEY_CHECK_ACTION_LABEL,
                id="settings-test-provider",
                compact=True,
                tooltip=PROVIDER_TEST_GUIDANCE,
            )
            # Parent AC#2: t runs it (spec mock (c) shows "t test key"), so it
            # is not a Tab stop between Endpoint and Model; a click still runs it.
            test_button.can_focus = False
            yield test_button
        yield Static(
            screen._provider_save_result,
            id="settings-provider-save-result",
            classes="settings-status-row",
        )
        existing_changes = Static(
            screen._provider_existing_changes_copy(),
            id="settings-provider-existing-changes-summary",
            classes="settings-status-row",
            markup=False,
        )
        existing_changes.display = screen._provider_same_target_has_draft()
        yield existing_changes
        conflict_target = screen._provider_navigation_conflict_target
        conflict = Vertical(id="settings-provider-navigation-conflict")
        conflict.display = conflict_target is not None
        with conflict:
            yield Static(
                screen._provider_navigation_conflict_copy(),
                id="settings-provider-navigation-conflict-summary",
                classes="settings-status-row",
                markup=False,
            )
            yield Button(
                "Review existing changes",
                id="settings-provider-conflict-review",
            )
            yield Button(
                screen._provider_conflict_discard_label(),
                id="settings-provider-conflict-discard",
            )
            yield Button(
                "Return to Chat settings",
                id="settings-provider-conflict-return",
                disabled=screen._provider_return_actions_disabled(),
            )
        continuation = Vertical(id="settings-provider-return-continuation")
        continuation.display = screen._provider_return_outcome is not None
        with continuation:
            yield Static(
                screen._provider_return_continuation_copy(),
                id="settings-provider-return-continuation-status",
                classes="settings-status-row",
                markup=False,
            )
            yield Button(
                "Return to Chat settings",
                id="settings-provider-return",
                variant="primary",
                disabled=screen._provider_return_actions_disabled(),
            )
            yield Button(
                "Stay in Settings",
                id="settings-provider-stay",
            )
        return_without_saving = Button(
            "Return without saving",
            id="settings-provider-return-without-save",
            disabled=screen._provider_return_actions_disabled(),
        )
        return_without_saving.display = screen._provider_can_return_without_saving()
        yield return_without_saving
        yield Static(
            "Default model for new chats",
            id="settings-default-model-title",
            classes="destination-section",
        )
        # TASK-33007.3: a searchable picker over saved, catalog and
        # discovered ids. The hidden Input keeps #settings-model-value and
        # its value for staging, save and revert (the modal's legacy-adapter
        # precedent).
        with Horizontal(id="settings-model-row", classes="settings-input-row"):
            yield Static("Model", classes="settings-input-label")
            app_instance = screen.app_instance
            providers_models = getattr(app_instance, "providers_models", None)
            model_picker = DefaultModelPicker(
                id="settings-model-picker",
                provider_select_id="#settings-provider-value",
                current_model=str(values["model"]),
                # The screen's app owns both, as for every Settings read.
                providers_models=(
                    providers_models if isinstance(providers_models, Mapping) else None
                ),
                show_provenance=True,
                catalog_scope_service=getattr(
                    app_instance, "llm_provider_catalog_scope_service", None
                ),
            )
            model_picker.disabled = registry_locked
            yield model_picker
            yield Static(
                selection_source_word(resolved.model_source),
                id="settings-model-source",
                classes="settings-source-word",
                markup=False,
            )
            yield Static(
                MODEL_PICKER_HELP_COPY,
                id="settings-model-help",
                classes="settings-row-help",
                markup=False,
            )
        model_adapter = Input(
            value=str(values["model"]),
            id="settings-model-value",
            disabled=registry_locked,
        )
        model_adapter.display = False
        model_adapter.can_focus = False
        yield model_adapter
        yield Static("Context capacity", classes="destination-section")
        yield Static(
            screen._provider_model_context_window_status(
                provider,
                str(values["model"]),
                values.get("model_context_window"),
            ),
            id="settings-model-context-window-status",
            classes="settings-status-row",
        )
        with Horizontal(classes="settings-input-row"):
            yield Static("Context window", classes="settings-input-label")
            yield Input(
                value=screen._profile_input_value(
                    values.get("model_context_window", "")
                ),
                id="settings-model-context-window",
                classes="settings-compact-input",
                placeholder="tokens (required when unknown)",
                restrict=r"^[0-9]*$",
                disabled=registry_locked,
            )
        with Horizontal(classes="settings-input-row"):
            yield Static("", classes="settings-input-label")
            yield Button(
                "Reset to detected",
                id="settings-model-context-window-reset",
                disabled=(
                    not context_window_state.has_configured_override
                    or registry_locked
                ),
                tooltip=(
                    "Remove only the configured context-window override and "
                    "return to the detected capability value."
                ),
            )
        yield Static(
            "This is the model's total token capacity, not a conversation "
            "length preference. Repairs update the existing model-capability "
            "registry used by request safety checks.",
            id="settings-model-context-window-help",
            classes="settings-detail-row",
        )
        yield Static("Model discovery", classes="destination-section")
        yield Static(
            screen._model_discovery_status,
            id="settings-model-discovery-status",
            classes="settings-status-row",
        )
        empty_state = Static(
            MODEL_DISCOVERY_EMPTY_COPY,
            id="settings-model-discovery-empty",
            classes="settings-status-row",
        )
        empty_state.display = not screen._model_discovery_models
        yield empty_state
        yield Static(
            MODEL_DISCOVERY_CAPABILITY_WARNING,
            id="settings-model-discovery-capability-warning",
            classes="settings-status-row",
        )
        with Horizontal(classes="settings-input-row"):
            yield Button(
                "Discover models",
                id="settings-discover-provider-models",
                disabled=not screen._model_discovery_available(
                    str(values["provider"])
                ),
                tooltip=(
                    "Query the configured OpenAI-compatible provider endpoint "
                    "for available models."
                ),
            )
            yield Button(
                "Save selected",
                id="settings-save-discovered-provider-models",
                disabled=not screen._model_discovery_models,
                tooltip="Append selected discovered model IDs to the local provider list.",
            )
            yield Button(
                "Clear",
                id="settings-clear-discovered-provider-models",
                disabled=not screen._model_discovery_models,
                tooltip="Clear runtime-discovered models for this provider.",
            )
        yield SelectionList(
            *screen._model_discovery_selection_options(),
            id="settings-discovered-models-list",
            classes="settings-discovered-models-list",
            disabled=not screen._model_discovery_models,
        )
        # ADR-020: [model_catalog] auto-refresh toggles. Values initialize
        # inline from the saved config (the Connect block pattern) and
        # persist immediately on change via the handlers below.
        # task-1341: instant-apply is the labeled exception to the staged
        # default; the bordered group and hint line separate these
        # operational flags visually from the staged Connect fields.
        model_catalog_settings = screen._model_catalog_card_settings()
        # TASK-387: keep the internal decision-record id (ADR-020) out of the
        # user-facing heading; it survives in the code comment above.
        with Vertical(
            id="settings-model-catalog-group",
            classes="settings-instant-apply-group",
        ):
            yield Static("Automatic refresh", classes="destination-section")
            yield Static(
                INSTANT_APPLY_BEHAVIOR_COPY,
                id="settings-model-catalog-instant-hint",
                classes="settings-instant-apply-hint",
            )
            yield Checkbox(
                "Refresh on startup",
                value=model_catalog_settings.auto_refresh_enabled,
                id="settings-model-catalog-auto-refresh",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    "Refresh after (hours)", classes="settings-input-label"
                )
                yield Input(
                    (
                        str(
                            screen._model_catalog_form_values["model_catalog"][
                                "stale_after_hours"
                            ]
                        )
                        if screen._model_catalog_form_values is not None
                        else f"{model_catalog_settings.stale_after_hours:g}"
                    ),
                    id="settings-model-catalog-stale-hours",
                    type="number",
                    tooltip="0 = refetch every launch.",
                )
            yield Static(
                screen._model_catalog_save_status,
                id="settings-model-catalog-save-status",
                classes="settings-status-row",
                markup=False,
            )
            retry = Button("Retry", id="settings-model-catalog-retry")
            retry.display = screen._model_catalog_save_failed
            yield retry
            for _provider in AUTO_REFRESH_PROVIDER_LIST_KEYS:
                _provider_key = provider_config_key(_provider)
                _pid = _provider.lower()
                with Horizontal(classes="settings-input-row"):
                    yield Checkbox(
                        f"{provider_display_name(_provider)}: refresh",
                        value=(
                            _provider_key
                            not in model_catalog_settings.auto_refresh_disabled
                        ),
                        id=f"settings-mc-auto-{_pid}",
                    )
                    yield Checkbox(
                        "save to config",
                        value=_provider_key
                        in model_catalog_settings.write_to_config,
                        id=f"settings-mc-write-{_pid}",
                        tooltip=(
                            "Append newly discovered models to config.toml — "
                            "large catalogs like OpenRouter only add newly released "
                            "models after a first baseline."
                        ),
                    )
        # ADR-146 task-7: named-endpoint management (rename / edit /
        # delete-with-reference-guard / slot conversion). Instant-apply
        # like the catalog block above: threaded config writes, one
        # shared status line, no partial-apply states.
        yield from compose_custom_endpoints_section(screen)
        # task-189: sampling and provider-specific tuning live below the
        # Connect block in a collapsed-by-default disclosure.
        model = str(values["model"])
        # TASK-33001.2: a row the provider+model request does not carry is
        # hidden and disabled (never a focus stop), as the gated rows were.
        row_supported = {
            draft_key: screen._model_profile_field_supported(
                provider, draft_key, model
            )
            for draft_key in PROVIDER_MODEL_PROFILE_FIELD_KEYS
        }
        with Collapsible(
            title="Generation defaults",
            collapsed=screen._generation_defaults_collapsed,
            id="settings-generation-defaults",
            disabled=registry_locked,
        ):
            yield Static(
                "Selected model defaults",
                id="settings-selected-model-defaults-title",
                classes="destination-section",
            )
            yield Static(
                "Global fallbacks live under Console Behavior; these values apply only "
                "to the provider+model above.",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    MODEL_FIELD_LABELS["temperature"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(
                        values["model_profile_temperature"]
                    ),
                    id="settings-model-profile-temperature",
                    classes="settings-compact-input",
                    placeholder="0.0 - 2.0",
                )
            with Horizontal(
                id="settings-model-profile-top-p-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_top_p"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["top_p"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(values["model_profile_top_p"]),
                    id="settings-model-profile-top-p",
                    classes="settings-compact-input",
                    disabled=not row_supported["model_profile_top_p"],
                    placeholder="0.0 - 1.0",
                )
            with Horizontal(
                id="settings-model-profile-min-p-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_min_p"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["min_p"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(values["model_profile_min_p"]),
                    id="settings-model-profile-min-p",
                    classes="settings-compact-input",
                    disabled=not row_supported["model_profile_min_p"],
                    placeholder="optional 0.0 - 1.0",
                )
            with Horizontal(
                id="settings-model-profile-top-k-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_top_k"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["top_k"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(values["model_profile_top_k"]),
                    id="settings-model-profile-top-k",
                    classes="settings-compact-input",
                    disabled=not row_supported["model_profile_top_k"],
                    placeholder="optional whole number",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    MODEL_FIELD_LABELS["max_tokens"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(
                        values["model_profile_max_tokens"]
                    ),
                    id="settings-model-profile-max-tokens",
                    classes="settings-compact-input",
                    placeholder="optional whole number",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(
                id="settings-model-profile-seed-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_seed"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["seed"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(values["model_profile_seed"]),
                    id="settings-model-profile-seed",
                    classes="settings-compact-input",
                    disabled=not row_supported["model_profile_seed"],
                    placeholder="optional whole number",
                    restrict=r"^[0-9]*$",
                )
            with Horizontal(
                id="settings-model-profile-presence-penalty-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_presence_penalty"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["presence_penalty"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(
                        values["model_profile_presence_penalty"]
                    ),
                    id="settings-model-profile-presence-penalty",
                    classes="settings-compact-input",
                    disabled=not row_supported["model_profile_presence_penalty"],
                    placeholder="-2.0 - 2.0",
                )
            with Horizontal(
                id="settings-model-profile-frequency-penalty-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_frequency_penalty"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["frequency_penalty"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._profile_input_value(
                        values["model_profile_frequency_penalty"]
                    ),
                    id="settings-model-profile-frequency-penalty",
                    classes="settings-compact-input",
                    disabled=not row_supported["model_profile_frequency_penalty"],
                    placeholder="-2.0 - 2.0",
                )
            # task-189: one summary line replaces per-row "Unavailable
            # for <provider>" placeholders; unsupported rows are hidden.
            support_copy = screen._provider_generation_support_copy(provider, model)
            support_summary = Static(
                support_copy,
                id="settings-provider-generation-support",
                classes="settings-detail-row",
            )
            support_summary.set_class(
                not support_copy, "settings-gated-profile-hidden"
            )
            yield support_summary
            with Horizontal(
                id="settings-model-profile-reasoning-effort-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_reasoning_effort"]
                )
                + " settings-select-row",
            ):
                yield Static(
                    MODEL_FIELD_LABELS["reasoning_effort"], classes="settings-input-label"
                )
                yield screen._model_profile_enum_select(
                    provider,
                    "model_profile_reasoning_effort",
                    values,
                )
            with Horizontal(
                id="settings-model-profile-reasoning-summary-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_reasoning_summary"]
                )
                + " settings-select-row",
            ):
                yield Static(
                    MODEL_FIELD_LABELS["reasoning_summary"], classes="settings-input-label"
                )
                yield screen._model_profile_enum_select(
                    provider,
                    "model_profile_reasoning_summary",
                    values,
                )
            with Horizontal(
                id="settings-model-profile-verbosity-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_verbosity"]
                )
                + " settings-select-row",
            ):
                yield Static(
                    MODEL_FIELD_LABELS["verbosity"], classes="settings-input-label"
                )
                yield screen._model_profile_enum_select(
                    provider,
                    "model_profile_verbosity",
                    values,
                )
            with Horizontal(
                id="settings-model-profile-thinking-effort-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_thinking_effort"]
                )
                + " settings-select-row",
            ):
                yield Static(
                    MODEL_FIELD_LABELS["thinking_effort"], classes="settings-input-label"
                )
                yield screen._model_profile_enum_select(
                    provider,
                    "model_profile_thinking_effort",
                    values,
                )
            with Horizontal(
                id="settings-model-profile-thinking-budget-tokens-row",
                classes=screen._gated_profile_row_classes(
                    row_supported["model_profile_thinking_budget_tokens"]
                ),
            ):
                yield Static(
                    MODEL_FIELD_LABELS["thinking_budget_tokens"], classes="settings-input-label"
                )
                yield Input(
                    value=screen._model_profile_input_value(
                        provider,
                        "model_profile_thinking_budget_tokens",
                        model,
                        values["model_profile_thinking_budget_tokens"],
                    ),
                    id="settings-model-profile-thinking-budget-tokens",
                    classes="settings-compact-input",
                    placeholder=screen._model_profile_input_placeholder(
                        provider,
                        "model_profile_thinking_budget_tokens",
                        model,
                    ),
                    restrict=r"^[0-9]*$",
                    disabled=not row_supported[
                        "model_profile_thinking_budget_tokens"
                    ],
                )
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static(
                    MODEL_FIELD_LABELS["streaming"], classes="settings-input-label"
                )
                yield Select(
                    list(MODEL_PROFILE_STREAMING_SELECT_OPTIONS),
                    value=screen._streaming_select_value(
                        values["model_profile_streaming"]
                    ),
                    id="settings-model-profile-streaming",
                    classes="settings-compact-select",
                    allow_blank=True,
                    prompt="Inherit default",
                    compact=True,
                )
        yield Static(
            screen._provider_catalog_summary(),
            id="settings-provider-catalog",
            classes="settings-status-row",
        )
        yield Static(
            screen._provider_catalog_key_policy(),
            id="settings-provider-catalog-policy",
            classes="settings-status-row",
        )
        yield Static(
            "Choose a catalog provider (type in the open list to jump to one), "
            "or use Manual / custom provider for other keys.",
            id="settings-provider-manual-entry-policy",
            classes="settings-status-row",
        )
        yield Static(
            "Sampling and transport defaults are routed to Console Behavior.",
            id="settings-provider-sampling-route",
            classes="settings-status-row",
        )
        yield screen._detail_row(
            "Endpoint key",
            screen._provider_endpoint_row(str(values["provider"])).removeprefix(
                "Endpoint key: "
            ),
            identifier="settings-provider-endpoint-key",
        )


def compose_custom_endpoints_section(screen: SettingsScreen) -> ComposeResult:
    """Compose the Custom endpoints section (ADR-146 management).

    Args:
        screen: The Settings screen that owns the section's state.

    Yields:
        The section heading, then the region rebuilt on every refresh.
    """
    yield Static("Custom endpoints", classes="destination-section")
    # SettingsRegion (task-15475): a plain Vertical yielded inline has no
    # compose() of its own, so a region-scoped refresh(recompose=True)
    # would wipe it instead of rebuilding it.
    yield SettingsRegion(
        partial(compose_custom_endpoints_children, screen),
        id="settings-custom-endpoints",
        classes="settings-instant-apply-group",
    )



def compose_custom_endpoints_children(screen: SettingsScreen) -> ComposeResult:
    """Yield the Custom endpoints region children (rebuilt per refresh).

    Args:
        screen: The Settings screen that owns the section's state.

    Yields:
        One row and action row per named endpoint, any open rename or edit
        form, the built-in slot rows, and the shared status line.
    """
    config = screen._custom_endpoints_view_config()
    entries = load_custom_endpoints(config)
    rows = custom_endpoint_rows(config)
    if not rows:
        yield Static(
            "No named endpoints yet. Create one from a template in "
            "Console's Chat settings (New endpoint…), or "
            "convert a built-in custom slot below.",
            id="settings-custom-endpoints-empty",
            classes="settings-status-row",
            markup=False,
        )
    for row in rows:
        slug = row.key[len(CUSTOM_ENDPOINT_ID_PREFIX) :]
        entry = entries.get(slug)
        if entry is None:
            continue
        yield Static(
            f"{row.label}: {_fold_long_tokens(row.value)}",
            id=f"settings-cep-row-{slug}",
            classes="settings-detail-row",
            markup=False,
        )
        with Horizontal(classes="settings-action-row"):
            yield Button(
                "Rename",
                id=f"settings-cep-rename-{slug}",
                tooltip="Rename this endpoint. Its id never changes.",
            )
            yield Button(
                "Edit",
                id=f"settings-cep-edit-{slug}",
                tooltip=(
                    "Edit the endpoint URL, credential variable, or "
                    "model list. Existing conversations re-resolve "
                    "on their next send."
                ),
            )
            yield Button(
                "Duplicate",
                id=f"settings-cep-duplicate-{slug}",
                tooltip=(
                    "Create another endpoint from this one via the "
                    "template flow: same family, new URL (a full copy "
                    "stays one row below the preselected starter)."
                ),
            )
            yield Button(
                "Detach references",
                id=f"settings-cep-detach-{slug}",
                classes=(
                    ""
                    if screen._custom_endpoint_detach_slug == slug
                    else "settings-gated-profile-hidden"
                ),
                tooltip=(
                    "Keep each referencing conversation's current "
                    "endpoint as conversation-only, then delete the "
                    "entry."
                ),
            )
            yield Button(
                "Delete",
                id=f"settings-cep-delete-{slug}",
                tooltip=(
                    "Delete this endpoint. Blocked while any "
                    "conversation still references it."
                ),
            )
        if screen._custom_endpoint_rename_slug == slug:
            with Horizontal(classes="settings-input-row"):
                yield Static("Name", classes="settings-input-label")
                yield Input(
                    value=entry.display_name,
                    id="settings-cep-rename-value",
                    classes="settings-compact-input",
                    placeholder="Display name",
                )
            with Horizontal(classes="settings-action-row"):
                yield Button("Save name", id="settings-cep-rename-save")
                yield Button("Cancel", id="settings-cep-rename-cancel")
        if screen._custom_endpoint_edit_slug == slug:
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    MODEL_FIELD_LABELS["endpoint"], classes="settings-input-label"
                )
                yield Input(
                    value=entry.base_url,
                    id="settings-cep-edit-url",
                    classes="settings-compact-input",
                    placeholder="http://127.0.0.1:8080",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    "Credential env var (name)", classes="settings-input-label"
                )
                yield Input(
                    value=entry.api_key_env or "",
                    id="settings-cep-edit-key-env",
                    classes="settings-compact-input",
                    placeholder="API key environment variable (optional)",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Models", classes="settings-input-label")
                yield Input(
                    value=", ".join(entry.models),
                    id="settings-cep-edit-models",
                    classes="settings-compact-input",
                    placeholder="comma-separated model ids",
                )
            yield Static(
                "",
                id="settings-cep-edit-error",
                classes="settings-status-row",
                markup=False,
            )
            with Horizontal(classes="settings-action-row"):
                yield Button("Save endpoint", id="settings-cep-edit-save")
                yield Button("Cancel", id="settings-cep-edit-cancel")
    for slot_id in ("custom", "custom_2"):
        app_config = screen._app_config_mapping()
        api_settings = app_config.get("api_settings")
        slot_endpoint = first_configured_endpoint(
            provider_settings_for_key(
                api_settings if isinstance(api_settings, Mapping) else {},
                slot_id,
            )
        )
        if not slot_endpoint:
            continue
        yield Static(
            f"{provider_display_name(slot_id)}: "
            f"{safe_endpoint_display(slot_endpoint)} (built-in slot)",
            id=f"settings-cep-slot-{slot_id}",
            classes="settings-detail-row",
            markup=False,
        )
        with Horizontal(classes="settings-action-row"):
            yield Button(
                "Convert to named endpoint",
                id=f"settings-cep-convert-{slot_id}",
                tooltip=(
                    "Create a named endpoint from this slot's "
                    "configured URL and models. The slot itself is "
                    "left untouched."
                ),
            )
    yield Static(
        screen._custom_endpoints_status,
        id="settings-custom-endpoints-status",
        classes="settings-status-row",
        markup=False,
    )
