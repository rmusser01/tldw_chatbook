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
from functools import cache, partial
from typing import TYPE_CHECKING

from textual import events
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
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
    CONFIG_KEY_ROW_LABEL,
    INSTANT_APPLY_BEHAVIOR_COPY,
    MODEL_DISCOVERY_CAPABILITY_WARNING,
    MODEL_DISCOVERY_EMPTY_COPY,
    PROVIDER_MANUAL_SELECT_VALUE,
    PROVIDER_TEST_GUIDANCE,
    QWENCLOUD_API_MODE_HELP_COPY,
    QWENCLOUD_API_MODE_INVALID_COPY,
    QWENCLOUD_API_MODE_OPTIONS,
    QWENCLOUD_PROVIDER_TABLE_INVALID_COPY,
    ProviderEndpointURLValidator,
    SettingsCategoryId,
    SettingsRegion,
    SettingsURLInput,
    _anthropic_auth_source_options,
    _anthropic_auth_sources,
    _fold_long_tokens,
    _ProviderTestResult,
)
from .settings_field_rows import compose_model_defaults, format_value

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
#: Owner ruling 2026-10-04: Clear is a key on the API key field, not a Tab
#: stop. Not ctrl+d: Textual's Input binds it (delete right) and ADR-031
#: rule 2 reserves it. ctrl+l is unbound in Settings, the app and Input, and
#: the Speech playground already clears with it.
API_KEY_CLEAR_KEY = "ctrl+l"
#: The API key row's help names both keys while a saved key can be cleared.
API_KEY_KEYS_HINT = f"(t) test · ({API_KEY_CLEAR_KEY}) clear"
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
#: TASK-33007.4: who a Default model choice reaches (spec mock (c)).
APPLIES_TO_NEW_CHATS = "new chats (Ctrl+T, temporary, workspace)."
NO_CONSOLE_CHAT_COPY = "No Console chat is open."
#: A longer chat title is cut so the pair stays on the row's one line.
_APPLIES_TO_TITLE_LIMIT = 24
#: The Inspector's Applies to block: (id suffix, label, value).
APPLIES_TO_INSPECTOR_ROWS = (
    ("new", "New chats", "yes"),
    ("unused", "Unused open chats", "follow the saved default"),
    ("work", "Chats with work", "keep their own; switch there with Alt+M"),
    ("switch", "Model defaults", "chats that switch to this model pick them up"),
)
UNSAVED_EDITS_NOTE = "Unsaved edits apply only after save (s)."
#: The card's old reference rows, now in the Inspector's config-key disclosure.
MANUAL_ENTRY_POLICY_COPY = (
    "Choose a catalog provider (type in the open list to jump to one), "
    "or use Manual / custom provider for other keys."
)
SAMPLING_ROUTE_COPY = "Sampling and transport defaults are routed to Console Behavior."
#: TASK-33007.6: Advanced follows Model defaults as closed one-row
#: disclosures (spec mock (c)); each title says its state.
ADVANCED_DISCLOSURE_CLASS = "settings-advanced-disclosure"
CONTEXT_WINDOW_DISCLOSURE_ID = "settings-advanced-context-window"
SAVED_MODELS_DISCLOSURE_ID = "settings-advanced-saved-models"
CATALOG_REFRESH_DISCLOSURE_ID = "settings-advanced-catalog-refresh"
CUSTOM_ENDPOINTS_DISCLOSURE_ID = "settings-advanced-custom-endpoints"
#: Prompt-cache snapshots keeps the id its tests already open it by.
SNAPSHOTS_DISCLOSURE_ID = "settings-snapshot-controls"
CATALOG_STARTUP_ID = "settings-model-catalog-auto-refresh"
CATALOG_STARTUP_OFF_ID = "settings-model-catalog-startup-off"
CATALOG_STARTUP_OFF_COPY = (
    "Refresh on startup is Off, so the per-provider choices below are not in effect."
)
#: The head of ADR-033's instant-apply label, for one-row titles.
APPLIES_IMMEDIATELY = "applies immediately"
SNAPSHOTS_SCOPE = "llama.cpp only"


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

    def _typing(self) -> bool:
        # The open list holds a filter; closing it puts the name back.
        return bool(self._picker().display)

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


class ApiKeyInput(Input):
    """The API key field, whose Clear key presses the row's Clear button.

    A saved key is never loaded into the field, and Clear is not a Tab stop
    (parent AC#2), so this key is the keyboard's way to clear one.
    """

    BINDINGS = [
        Binding(API_KEY_CLEAR_KEY, "clear_saved_key", "Clear saved key", show=False)
    ]

    def action_clear_saved_key(self) -> None:
        """Press Clear; a disabled Clear ignores it, as it ignores a click."""
        self.screen.query_one("#settings-provider-api-key-clear", Button).press()


class DefaultModelPicker(ModelSearchPicker):
    """The Default model control (TASK-33007.3).

    The shared picker, scoped to the provider the form holds. The hidden
    ``Input#settings-model-value`` beside it stays the value that staging,
    save and revert read; the screen keeps the two in step.
    """

    def on_mount(self) -> None:
        """Give the field the one-row Connect edge (task-1586).

        Every card compose builds a fresh picker, so the screen's current
        discovery listing is laid over it again here: a pane rebuild or a
        category round trip keeps the Served now rows the listing still
        shows. The base ``on_mount`` still runs after this one (MRO dispatch).
        """
        self.query_one("#model-search-picker-input", Input).add_class(
            "settings-compact-input"
        )
        self.screen._refresh_model_picker_discovered()

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
        if self._holds_invalid_custom_id():
            self._roll_back_invalid_custom_id()
        elif not self.custom_mode:
            # An unfinished filter is dropped and the list closes.
            super()._cancel_edit(event)
        # task-1560: one Esc also leaves the field, so the footer's "Esc, s"
        # holds -- and saves a typed Custom ID instead of dropping it. Queued
        # after any refocus the cancel itself queued.
        event.stop()
        self.app.call_later(self.screen.set_focus, None)

    def _restore_committed_display_after_blur(self) -> None:
        focused = self.app.focused
        left = focused is None or self not in focused.ancestors_with_self
        if left and self._holds_invalid_custom_id():
            self._roll_back_invalid_custom_id()
            return
        super()._restore_committed_display_after_blur()

    def _holds_invalid_custom_id(self) -> bool:
        """Whether the field holds typed text that is not a usable model id.

        Returns:
            True in Custom ID mode when the text is not blank and is not
            bounded single-line text (AC#5); a blank field clears the model.
        """
        if not self.custom_mode or not self.is_mounted:
            return False
        typed = self.query_one("#model-search-picker-input", Input).value
        return bool(typed.strip()) and self.value is None

    def _roll_back_invalid_custom_id(self) -> None:
        """Put back the model held before Custom ID when an invalid id is left.

        The invalid text is only ever explained by the status line, which is
        hidden once the picker loses focus, so it never stays in the row.
        """
        previous = self._model_before_custom
        self.set_model_value(previous)
        self.post_message(self.ModelValueChanged(previous, custom=False))
        self.app.notify(
            f"Invalid model ID not kept; the model is still {previous or 'unset'}.",
            severity="warning",
        )


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
        control = screen.query_one("#settings-provider-search", Input)
    except QueryError:
        return
    if not sync_provider_control(screen):
        screen._refresh_provider_picker("")
    if control.has_focus:
        # A choice or Esc closed it: the next key filters afresh, also when
        # the name typed was the held one's exactly, so nothing was rewritten
        # (a blur close runs after focus has left).
        control.select_all()


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
    changed = control.value != label
    if changed:
        with control.prevent(Input.Changed):
            control.value = label
    if not control.has_focus:
        # At rest the name reads from its head, wherever the caret was left
        # (force: Textual does not scroll a disabled widget without it).
        control.scroll_to(x=0, animate=False, force=True)
    elif changed:
        # A choice landing renames it under focus, so typing filters afresh;
        # close_provider_list selects a name it did not rewrite.
        control.select_all()
    if changed:
        screen._refresh_provider_picker("")
    return changed


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
        return "saved in config", f"masked · {API_KEY_KEYS_HINT}"
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


def provider_model_pair(screen: SettingsScreen, provider: object, model: object) -> str:
    """Name a provider·model pair the way every surface does (spec §4 rule 2).

    Args:
        screen: The Settings screen (it owns the provider display names).
        provider: A provider id or alias.
        model: A model id, or nothing.

    Returns:
        "<Provider> · <model>", with "no model" when none is set.
    """
    name = screen._provider_display_label(str(provider or "")) or "no provider"
    return f"{name} · {str(model or '').strip() or 'no model'}"


def applies_to_copy(screen: SettingsScreen, provider: str, model: str) -> str:
    """Say who the Default model reaches: new chats, then the open Console chat.

    D1 (ADR-095 as amended 2026-09-26): an unused open chat takes the new
    default; a chat with messages or edits keeps its own pair. The test is
    the Console's own (``follows_saved_defaults``), so the row never says
    something the Console then does not do.

    Args:
        screen: The Settings screen; its app holds the live Console store.
        provider: The provider the card shows.
        model: The model the card shows.

    Returns:
        The Applies-to row's text.
    """
    from ...Chat.console_chat_store import ConsoleChatStore
    from ..Console_Modules.session import follows_saved_defaults

    store = screen._custom_endpoints_store()
    session_id = getattr(store, "active_session_id", None)
    session = (
        next((s for s in store.sessions() if s.id == session_id), None)
        if isinstance(store, ConsoleChatStore) and session_id is not None
        else None
    )
    if session is None:
        return f"{APPLIES_TO_NEW_CHATS} {NO_CONSOLE_CHAT_COPY}"
    title = session.title.strip() or "untitled"
    if len(title) > _APPLIES_TO_TITLE_LIMIT:
        # The pair is the point of the row; a long title gives way to it.
        title = title[: _APPLIES_TO_TITLE_LIMIT - 1].rstrip() + "…"
    generation = getattr(screen.app_instance, "console_new_chat_default_generation", 0)
    if follows_saved_defaults(
        store, session, generation if type(generation) is int else 0
    ):
        pair = provider_model_pair(screen, provider, model)
        return (
            f"{APPLIES_TO_NEW_CHATS} Open chat “{title}” is unused and will use {pair}."
        )
    own = session.settings
    pair = provider_model_pair(screen, own.provider, own.model)
    return f"{APPLIES_TO_NEW_CHATS} Open chat “{title}” keeps {pair}."


def next_new_chat_lines(screen: SettingsScreen) -> tuple[str, str]:
    """What a new chat gets from saved config: its pair and core values.

    Built by the function a Console Ctrl+T chat is built with, over the same
    saved mapping, so the two cannot disagree (D3's core values).

    Args:
        screen: The Settings screen whose app holds the saved config.

    Returns:
        The pair line and the "T · max · stream" line.
    """
    from ...Chat.console_session_settings import blank_console_session_settings

    settings = blank_console_session_settings(screen._app_config_mapping())
    max_tokens = "not set" if settings.max_tokens is None else str(settings.max_tokens)
    streaming = "On" if settings.streaming else "Off"
    return (
        provider_model_pair(screen, settings.provider, settings.model),
        f"T {settings.temperature:g} · max {max_tokens} · stream {streaming}",
    )


def refresh_next_new_chat(screen: SettingsScreen) -> None:
    """Re-say the Inspector's Next new chat block after a save.

    Args:
        screen: The Settings screen that owns the Inspector.
    """
    pair, values = next_new_chat_lines(screen)
    screen._set_static_text("#settings-provider-next-chat-pair", pair)
    screen._set_static_text("#settings-provider-next-chat-values", values)


def context_window_summary(screen: SettingsScreen) -> str:
    """Say the context size the card shows and whether an override is set.

    Args:
        screen: The Settings screen that owns the card.

    Returns:
        E.g. "200,000 tokens · detected, no override", "131,072 tokens ·
        override set", "... · edited *" before a save, or "unknown".
    """
    values = screen._provider_display_setting_values()
    model = str(values.get("model") or "").strip()
    if not model:
        return "choose a model first"
    try:
        shown: object = screen.query_one("#settings-model-context-window", Input).value
    except QueryError:
        shown = values.get("model_context_window")
    try:
        tokens = int(str(shown).strip())
    except ValueError:
        tokens = 0
    if tokens <= 0:
        return "unknown · enter the model's documented limit"
    state = model_context_window_state(
        screen._app_config_mapping(), str(values.get("provider") or ""), model
    )
    if tokens != state.effective_tokens:
        return f"{tokens:,} tokens · {SELECTION_SOURCE_WORDS['settings_draft']}"
    if state.has_configured_override:
        return f"{tokens:,} tokens · override set"
    return f"{tokens:,} tokens · detected, no override"


def saved_models_summary(screen: SettingsScreen) -> str:
    """Count the provider's saved models against the discovered ones.

    Args:
        screen: The Settings screen that owns the discovery state.

    Returns:
        E.g. "12 saved in config · 41 discovered, 29 not saved · 2 selected".
    """
    saved_ids = set(screen._provider_saved_model_ids(screen._provider_widget_value()))
    parts = [f"{len(saved_ids)} saved in config"]
    rows = [
        model
        for model in screen._model_discovery_models
        if str(getattr(model, "model_id", "") or "").strip()
    ]
    if not rows:
        parts.append("none discovered")
        return " · ".join(parts)
    unsaved = sum(
        not screen._discovered_model_is_saved(model, saved_ids) for model in rows
    )
    parts.append(f"{len(rows)} discovered, {unsaved} not saved")
    if screen._model_discovery_selected_model_ids:
        parts.append(f"{len(screen._model_discovery_selected_model_ids)} selected")
    return " · ".join(parts)


def catalog_refresh_summary(screen: SettingsScreen) -> str:
    """Say whether startup refresh runs, how often, and for how many providers.

    Args:
        screen: The Settings screen (it resolves the shown catalog settings).

    Returns:
        E.g. "applies immediately · startup refresh On · every 24 h · 29 of 30
        providers"; while startup refresh is Off, that the per-provider
        choices are not in effect.
    """
    settings = screen._model_catalog_card_settings()
    if not settings.auto_refresh_enabled:
        return (
            f"{APPLIES_IMMEDIATELY} · startup refresh Off · "
            "per-provider choices not in effect"
        )
    hours = settings.stale_after_hours
    every = "every launch" if hours <= 0 else f"every {hours:g} h"
    refreshed = sum(
        provider_config_key(provider) not in settings.auto_refresh_disabled
        for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS
    )
    return (
        f"{APPLIES_IMMEDIATELY} · startup refresh On · {every} · "
        f"{refreshed} of {len(AUTO_REFRESH_PROVIDER_LIST_KEYS)} providers"
    )


def custom_endpoints_summary(screen: SettingsScreen) -> str:
    """Count the named custom endpoints.

    Args:
        screen: The Settings screen (it reads the freshest config).

    Returns:
        E.g. "no named endpoints · applies immediately" or "2 named
        endpoints · ..."; built-in slots are listed inside.
    """
    named = len(load_custom_endpoints(screen._custom_endpoints_view_config()))
    count = f"{named or 'no'} named endpoint" + "s" * (named != 1)
    return f"{count} · {APPLIES_IMMEDIATELY}"


def snapshots_summary(screen: SettingsScreen) -> str:
    """Say whether prompt-cache snapshots are on and how many are kept.

    Args:
        screen: The Settings screen that owns the snapshot draft.

    Returns:
        E.g. "llama.cpp only · Off" or "llama.cpp only · On · keep 20".
    """
    if screen._snapshot_preferences_unavailable:
        return f"{SNAPSHOTS_SCOPE} · unavailable"
    enabled, keep = screen._snapshot_preferences_raw or (False, "")
    if not enabled:
        return f"{SNAPSHOTS_SCOPE} · Off"
    keep = str(keep).strip()
    return (
        f"{SNAPSHOTS_SCOPE} · On · keep {keep}" if keep else f"{SNAPSHOTS_SCOPE} · On"
    )


#: The Advanced disclosures in the card's order: id -> (name, summary).
ADVANCED_DISCLOSURES = {
    CONTEXT_WINDOW_DISCLOSURE_ID: ("Context window", context_window_summary),
    SAVED_MODELS_DISCLOSURE_ID: ("Saved model list", saved_models_summary),
    CATALOG_REFRESH_DISCLOSURE_ID: ("Catalog refresh", catalog_refresh_summary),
    CUSTOM_ENDPOINTS_DISCLOSURE_ID: ("Custom endpoints", custom_endpoints_summary),
    SNAPSHOTS_DISCLOSURE_ID: ("Prompt-cache snapshots", snapshots_summary),
}


def advanced_title(screen: SettingsScreen, disclosure_id: str) -> Content:
    """Build an Advanced disclosure's one-row title: its name and state.

    Args:
        screen: The Settings screen that owns the card.
        disclosure_id: A key of ``ADVANCED_DISCLOSURES``.

    Returns:
        A literal title, e.g. "Catalog refresh · applies immediately · ...".
    """
    name, summary = ADVANCED_DISCLOSURES[disclosure_id]
    return Content(f"{name} · {summary(screen)}")


def advanced_disclosure(screen: SettingsScreen, disclosure_id: str) -> Collapsible:
    """Build one Advanced disclosure, closed unless the user left it open.

    Args:
        screen: The Settings screen; it remembers which ones are open.
        disclosure_id: A key of ``ADVANCED_DISCLOSURES``.

    Returns:
        The disclosure, to compose its rows into.
    """
    return Collapsible(
        title=advanced_title(screen, disclosure_id),
        collapsed=disclosure_id not in screen._advanced_disclosures_open,
        id=disclosure_id,
        classes=ADVANCED_DISCLOSURE_CLASS,
    )


@cache
def catalog_toggle_prefixes() -> Mapping[str, str]:
    """Name each Catalog refresh checkbox; its label adds On or Off.

    Returns:
        ``{checkbox id: label without its state word}``.
    """
    prefixes = {CATALOG_STARTUP_ID: "Refresh on startup"}
    for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS:
        provider_id = provider.lower()
        prefixes[f"settings-mc-auto-{provider_id}"] = (
            f"{provider_display_name(provider)}: refresh"
        )
        prefixes[f"settings-mc-write-{provider_id}"] = "save to config"
    return prefixes


def catalog_toggle_label(checkbox_id: str, value: bool) -> str:
    """Label a Catalog refresh checkbox with its state as a word (AC#5, AC#9).

    Args:
        checkbox_id: A key of ``catalog_toggle_prefixes()``.
        value: Whether the box is ticked.

    Returns:
        E.g. "OpenAI: refresh On" or "save to config Off".
    """
    return f"{catalog_toggle_prefixes()[checkbox_id]} {format_value(value)}"


def catalog_toggle(checkbox_id: str, value: bool, **kwargs: object) -> Checkbox:
    """Build a Catalog refresh checkbox; colour only reinforces its word.

    Args:
        checkbox_id: A key of ``catalog_toggle_prefixes()``.
        value: Whether the box starts ticked.
        **kwargs: Further ``Checkbox`` options, e.g. a tooltip.

    Returns:
        The checkbox.
    """
    return Checkbox(
        catalog_toggle_label(checkbox_id, value), value=value, id=checkbox_id, **kwargs
    )


def refresh_advanced(screen: SettingsScreen) -> None:
    """Re-say the Advanced titles and Catalog refresh's state words.

    Runs after anything an Advanced title summarises changes: a draft edit,
    discovery, a catalog toggle or a custom-endpoint change.

    Args:
        screen: The Settings screen that owns the card.
    """
    # Id lookups and plain walks, not selector queries: this runs on every
    # keystroke in the card (via the draft-status refresh).
    try:
        card = screen.query_one("#settings-providers-models-card")
        group = card.query_one("#settings-model-catalog-group")
        startup_off = group.query_one(f"#{CATALOG_STARTUP_OFF_ID}")
    except QueryError:
        return
    for child in card.children:
        if child.has_class(ADVANCED_DISCLOSURE_CLASS):
            child.title = advanced_title(screen, str(child.id))
    for checkbox in group.walk_children(Checkbox):
        if checkbox.id not in catalog_toggle_prefixes():
            continue
        label = catalog_toggle_label(checkbox.id, checkbox.value)
        if checkbox.label.plain != label:
            checkbox.label = label
        if checkbox.id == CATALOG_STARTUP_ID:
            startup_off.display = not checkbox.value


def compose_providers_models_inspector(screen: SettingsScreen) -> ComposeResult:
    """Compose the Inspector's Providers & Models blocks (spec mock (c)).

    Applies to, Next new chat will use, the focused field (help and range,
    its config key in a closed disclosure with the card's old reference
    facts), then Key.

    Args:
        screen: The Settings screen that owns the Inspector.

    Yields:
        The Inspector body's Providers & Models widgets.
    """
    yield Static("Applies to", classes="destination-section")
    for suffix, label, value in APPLIES_TO_INSPECTOR_ROWS:
        yield screen._detail_row(
            label, value, identifier=f"settings-provider-applies-{suffix}"
        )
    yield Static("Next new chat will use", classes="destination-section")
    pair, values = next_new_chat_lines(screen)
    for suffix, text in (("pair", pair), ("values", values)):
        yield Static(
            text,
            id=f"settings-provider-next-chat-{suffix}",
            classes="settings-detail-row",
            markup=False,
        )
    note = Static(
        UNSAVED_EDITS_NOTE,
        id="settings-provider-next-chat-note",
        classes="settings-detail-row",
    )
    note.display = screen._category_has_unsaved_changes(
        SettingsCategoryId.PROVIDERS_MODELS
    )
    yield note
    yield Static("Focused field guide", classes="destination-section")
    shown, config_key = screen._split_config_key_row(
        screen._provider_field_guidance_rows()
    )
    for index, (label, value) in enumerate(shown):
        yield screen._detail_row(
            label, value, identifier=f"settings-provider-field-guide-{index}"
        )
    provider = str(screen._provider_display_setting_values()["provider"])
    yield screen._config_key_disclosure(
        screen._detail_row(
            CONFIG_KEY_ROW_LABEL,
            config_key,
            identifier="settings-provider-config-key-saved-as",
        ),
        screen._detail_row(
            "Endpoint key",
            screen._provider_endpoint_row(provider).removeprefix("Endpoint key: "),
            identifier="settings-provider-endpoint-key",
        ),
        Static(
            screen._provider_catalog_summary(),
            id="settings-provider-catalog",
            classes="settings-detail-row",
            markup=False,
        ),
        Static(
            screen._provider_catalog_key_policy(),
            id="settings-provider-catalog-policy",
            classes="settings-detail-row",
            markup=False,
        ),
        Static(
            MANUAL_ENTRY_POLICY_COPY,
            id="settings-provider-manual-entry-policy",
            classes="settings-detail-row",
            markup=False,
        ),
        Static(
            SAMPLING_ROUTE_COPY,
            id="settings-provider-sampling-route",
            classes="settings-detail-row",
            markup=False,
        ),
        identifier="settings-provider-config-key",
    )
    # TASK-33007.2 (AC#7): the Key block holds what 't' checks and the last
    # check's labelled rows; the card's Key check row says only the verdict,
    # so a result never moves the card's rows.
    yield Static("Key", classes="destination-section")
    yield Static(
        screen._provider_readiness_label(),
        id="settings-provider-inspector-readiness",
        classes="settings-detail-row",
    )
    yield Static(
        PROVIDER_TEST_GUIDANCE,
        id="settings-test-provider-guidance",
        classes="settings-detail-row",
    )
    yield _ProviderTestResult(
        screen._adopt_shared_provider_test_evidence(),
        id="settings-provider-test-result",
        markup=False,
    )


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
                select_on_focus=False,
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
            yield ApiKeyInput(
                value=str(values.get("api_key") or ""),
                id="settings-provider-api-key",
                classes="settings-compact-input",
                placeholder=screen._provider_api_key_placeholder(provider),
                password=True,
                disabled=registry_locked or subscription_selected,
            )
            clear_button = Button(
                "Clear",
                id="settings-provider-api-key-clear",
                compact=True,
                disabled=subscription_selected or (
                    not screen._provider_saved_api_key_present(provider)
                    and not bool(str(values.get("api_key") or "").strip())
                ),
                tooltip=(
                    "Clear the API key saved in local config for this provider "
                    f"({API_KEY_CLEAR_KEY} in the API key field)."
                ),
            )
            # Parent AC#2: like Test (t), a key runs it (ApiKeyInput), so it
            # is not a Tab stop between API key and Env var; a click still does.
            clear_button.can_focus = False
            yield clear_button
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
        # TASK-33007.4 (AC#1): who this choice reaches, under the choice.
        with Horizontal(id="settings-model-applies-row", classes="settings-input-row"):
            yield Static("Applies to", classes="settings-input-label")
            yield Static(
                applies_to_copy(screen, provider, str(values["model"])),
                id="settings-model-applies-to",
                classes="settings-applies-to",
                markup=False,
            )
        # TASK-33007.5: Model defaults follows Default model, open, titled
        # with the pair it edits; core rows first, then a closed Sampling
        # disclosure that names what the provider does not accept.
        yield from compose_model_defaults(
            screen,
            provider,
            str(values["model"]),
            values,
            registry_locked=registry_locked,
        )
        # TASK-33007.6: Advanced -- the rarely used controls, each a closed
        # one-row disclosure whose title says its state (spec mock (c)).
        # Field search opens the one it lands in.
        yield Static(
            "Advanced", id="settings-advanced-title", classes="destination-section"
        )
        with advanced_disclosure(screen, CONTEXT_WINDOW_DISCLOSURE_ID):
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
        with advanced_disclosure(screen, SAVED_MODELS_DISCLOSURE_ID):
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
            discovered_list = SelectionList(
                *screen._model_discovery_selection_options(),
                id="settings-discovered-models-list",
                classes="settings-discovered-models-list",
                disabled=not screen._model_discovery_models,
            )
            # Without its box an empty list is zero rows tall; the empty-state
            # line above stands in for it.
            discovered_list.display = bool(screen._model_discovery_models)
            yield discovered_list
        # ADR-020: [model_catalog] auto-refresh toggles. Values initialize
        # inline from the saved config (the Connect block pattern) and
        # persist immediately on change via the screen's handlers.
        # task-1341 / ADR-033: instant-apply is the labeled exception to the
        # staged default; the group and its hint line keep these operational
        # flags apart from the staged fields (TASK-33007.6: no border, the
        # pane's is the only frame, and its inputs are one row like the
        # card's).
        model_catalog_settings = screen._model_catalog_card_settings()
        with (
            advanced_disclosure(screen, CATALOG_REFRESH_DISCLOSURE_ID),
            Vertical(
                id="settings-model-catalog-group",
                classes="settings-instant-apply-group",
            ),
        ):
            yield Static(
                INSTANT_APPLY_BEHAVIOR_COPY,
                id="settings-model-catalog-instant-hint",
                classes="settings-instant-apply-hint",
            )
            yield catalog_toggle(
                CATALOG_STARTUP_ID, model_catalog_settings.auto_refresh_enabled
            )
            startup_off = Static(
                CATALOG_STARTUP_OFF_COPY,
                id=CATALOG_STARTUP_OFF_ID,
                classes="settings-status-row",
                markup=False,
            )
            startup_off.display = not model_catalog_settings.auto_refresh_enabled
            yield startup_off
            with Horizontal(classes="settings-input-row"):
                yield Static("Refresh after (hours)", classes="settings-input-label")
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
                    classes="settings-compact-input",
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
                    yield catalog_toggle(
                        f"settings-mc-auto-{_pid}",
                        _provider_key
                        not in model_catalog_settings.auto_refresh_disabled,
                    )
                    yield catalog_toggle(
                        f"settings-mc-write-{_pid}",
                        _provider_key in model_catalog_settings.write_to_config,
                        tooltip=(
                            "Append newly discovered models to config.toml — "
                            "large catalogs like OpenRouter only add newly "
                            "released models after a first baseline."
                        ),
                    )
        # ADR-146 task-7: named-endpoint management (rename / edit /
        # delete-with-reference-guard / slot conversion). Instant-apply
        # like the catalog block above: threaded config writes, one
        # shared status line, no partial-apply states.
        with advanced_disclosure(screen, CUSTOM_ENDPOINTS_DISCLOSURE_ID):
            yield from compose_custom_endpoints_section(screen)
        # ADR-119: llama.cpp prompt-cache snapshots, staged with the
        # category's Save / Revert; enable/disable applies on next launch.
        with advanced_disclosure(screen, SNAPSHOTS_DISCLOSURE_ID):
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
                classes="settings-compact-input",
                type="integer",
            )
            yield Static(
                screen._SNAPSHOT_PREFERENCES_UNAVAILABLE_COPY
                if screen._snapshot_preferences_unavailable
                else "Draft — use category Save / Revert. Enable/disable applies on next launch.",
                id="settings-snapshot-result",
                classes="settings-help-copy",
            )
        # TASK-33007.4 (AC#5): the catalog, key-policy, manual-entry,
        # sampling-route and endpoint-key rows moved to the Inspector's
        # config-key disclosure (compose_providers_models_inspector).


def compose_custom_endpoints_section(screen: SettingsScreen) -> ComposeResult:
    """Compose the Custom endpoints section (ADR-146 management).

    Args:
        screen: The Settings screen that owns the section's state.

    Yields:
        The region rebuilt on every refresh; its Advanced disclosure's title
        is the section heading (TASK-33007.6).
    """
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
