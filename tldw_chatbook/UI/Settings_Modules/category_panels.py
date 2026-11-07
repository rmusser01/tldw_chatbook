"""Per-category panels for the Settings screen (TASK-1378.1).

``SettingsScreen`` used to render, label and save every category through
if/elif chains keyed on ``SettingsCategoryId``. A category registered in
``CATEGORY_PANELS`` is served by its panel instead: for it, the screen asks
the panel to render, save, report unsaved state (the rail's ``*``, the
banner count and the quit guard), revert, and give its scope copy, and it
reads what ``s`` and ``r`` may do from the panel's ADR-303 commit model. The
screen's own chains still serve every other category until it moves.

Panels are stateless strategies, so each method takes the screen. Per-visit
edits live in the screen's ``_category_panel_states`` slot, keyed by
category and shaped by the panel that owns them; leaving Settings keeps
them like any other draft. A panel saves through ``screen.
_settings_config_adapter()``, which resolves the adapter in the screen's
module at call time, so tests that patch ``settings_screen.
SettingsConfigAdapter`` reach panel saves too.

Network is served wholly by its panel. Speech & TTS goes through
``SpeechTTSCategoryPanel``, a thin adapter that routes its mounted widget's
own contract (``has_unsaved_changes`` / ``request_save`` /
``revert_to_saved``) through the same registry; its detail pane is still
built by the screen.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.widgets import Input, Select, Static

from ...state.settings_config_models import SettingsCategoryId
from ...Widgets.Settings_Widgets.speech_tts_settings_panel import (
    SpeechTTSSettingsPanel,
)
from ..Screens.settings_network_defaults import (
    SettingsNetworkTLS,
    build_network_save_sections,
    load_network_tls,
    validate_network_tls,
)

if TYPE_CHECKING:
    from ..Screens.settings_screen import SettingsScreen

NO_CHANGES_TO_REVERT_COPY = "No Settings changes to revert."


@dataclass(frozen=True)
class CommitModel:
    """Which ADR-303 commit model a category follows, and its badge copy."""

    kind: Literal["staged", "instant", "raw_toml"]
    badge: str

    @property
    def saves(self) -> bool:
        """Staged drafts and guarded raw TOML commit on ``s`` (ADR-303)."""
        return self.kind != "instant"

    @property
    def reverts(self) -> bool:
        """Staged drafts and guarded raw TOML revert on ``r`` (ADR-303)."""
        return self.kind != "instant"


@dataclass(frozen=True)
class ScopeText:
    """A category's scope copy for the screen's two surfaces."""

    #: After the badge on the State banner (one row at 211x44).
    banner: str
    #: The guided-action row, and the guidance ``s`` gives when it cannot save.
    guided: str


class SettingsCategoryPanel(Protocol):
    """What a category provides so the screen need not special-case it."""

    commit_model: CommitModel

    def render(self, screen: SettingsScreen) -> ComposeResult:
        """Compose the category's detail pane."""
        ...

    def has_unsaved_changes(self, screen: SettingsScreen) -> bool:
        """True while the category holds edits that are not saved."""
        ...

    def request_save(self, screen: SettingsScreen) -> None:
        """Validate and commit the category's edits (the screen's ``s``)."""
        ...

    def revert(self, screen: SettingsScreen) -> None:
        """Discard unsaved edits (the screen's ``r``), confirming as needed."""
        ...

    def scope_text(self, screen: SettingsScreen) -> ScopeText:
        """The category's scope copy for the banner and the guided row."""
        ...


# Network ([network] ssl_verify) TLS trust mode options for the Select.
# Textual Select options are (label, value) pairs, so the mode strings --
# load_network_tls's contract: "verify" | "off" | "custom-ca" ("invalid" is
# a load state, never offered) -- are the VALUES.
NETWORK_TLS_MODE_OPTIONS: list[tuple[str, str]] = [
    ("Verify certificates (default)", "verify"),
    ("Disable verification", "off"),
    ("Custom CA bundle", "custom-ca"),
]
_NETWORK_TLS_MODE_VALUES = frozenset(value for _, value in NETWORK_TLS_MODE_OPTIONS)


class _NetworkCard(Vertical):
    """Network's card; its Select and Input stage edits through the panel."""

    def __init__(
        self, settings_screen: SettingsScreen, seeded: dict[str, str], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._settings_screen = settings_screen
        self._seeded = dict(seeded)

    def _stage(self, key: str, value: str) -> None:
        # Textual posts Changed when a control mounts with its seeded value.
        # That echo is no edit: an invalid ssl_verify seeds "verify", and
        # staging it raised a `*` that `r` could never clear (review N1).
        # Only a control's FIRST Changed can be that echo, so the seed goes
        # with it; every later Changed is the user's, even one back to the
        # seed after a save (review N6).
        if self._seeded.pop(key, None) == value:
            return
        NETWORK_PANEL.stage(self._settings_screen, **{key: value})

    @on(Select.Changed, "#settings-network-ssl-mode")
    def _mode_changed(self, event: Select.Changed) -> None:
        event.stop()
        self._stage("mode", str(event.value or "verify"))
        try:
            warning = self.query_one("#settings-network-warning", Static)
        except QueryError:
            return
        warning.update(
            NETWORK_PANEL.warning_text(
                NETWORK_PANEL.effective_values(self._settings_screen).mode
            )
        )

    @on(Input.Changed, "#settings-network-ca-path")
    def _ca_path_changed(self, event: Input.Changed) -> None:
        self._stage("ca_bundle_path", event.value)


class NetworkPanel:
    """Network: TLS trust for outbound API traffic.

    Staged commit with no ``SettingsDraft``: edits wait in the screen's
    panel-state slot until ``s`` validates and saves them, or ``r`` discards
    them after asking.
    """

    category = SettingsCategoryId.NETWORK
    commit_model = CommitModel("staged", "Pending — save with s")

    def render(self, screen: SettingsScreen) -> ComposeResult:
        values = load_network_tls(screen._app_config_mapping())
        # Seed the Select and CA Input from the EFFECTIVE (pending-aware)
        # state, not the loaded config: the pending edit survives a category
        # hop, so seeding from the loaded values let a hop-away-and-back
        # repaint the Select with the LOADED mode while `s` still saved the
        # PENDING one. The invalid-row banner below stays keyed on the LOADED
        # mode -- a hand-edited invalid config is a config-file fact, not a
        # pending edit.
        effective = self.effective_values(screen)
        seeded = {
            "mode": (
                effective.mode if effective.mode in _NETWORK_TLS_MODE_VALUES else "verify"
            ),
            "ca_bundle_path": (
                effective.ca_bundle_path if effective.mode == "custom-ca" else ""
            ),
        }
        yield Static("Network", classes="destination-section settings-column-title")
        with _NetworkCard(
            screen, seeded, id="settings-network-card", classes="settings-focus-card"
        ):
            yield Static(
                "TLS trust for outbound API traffic", classes="destination-section"
            )
            if values.mode == "invalid":
                yield Static(
                    f"Config has an invalid [network] ssl_verify value"
                    f" ({values.raw!r}); default verification is in use until"
                    " it is fixed.",
                    id="settings-network-invalid-row",
                    classes="settings-network-error",
                )
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static("Certificate verification", classes="settings-input-label")
                yield Select(
                    NETWORK_TLS_MODE_OPTIONS,
                    value=seeded["mode"],
                    id="settings-network-ssl-mode",
                    classes="settings-compact-select",
                    allow_blank=False,
                    compact=True,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("CA bundle path", classes="settings-input-label")
                yield Input(
                    value=seeded["ca_bundle_path"],
                    id="settings-network-ca-path",
                    classes="settings-compact-input",
                    placeholder="/path/to/corp-ca.pem (used by 'Custom CA bundle')",
                )
            yield Static(
                self.warning_text(effective.mode),
                id="settings-network-warning",
                classes="settings-network-warning",
            )

    def has_unsaved_changes(self, screen: SettingsScreen) -> bool:
        # Compared by value, not by "anything staged": a choice set back to
        # the saved value is no edit either.
        return self.effective_values(screen) != self.effective_values(screen, {})

    def request_save(self, screen: SettingsScreen) -> None:
        values = self.effective_values(screen)
        validation = validate_network_tls(values)
        if not validation.valid:
            screen.app.notify(validation.message, severity="error", markup=False)
            return
        section_values = build_network_save_sections(values)
        saved = screen._settings_config_adapter().save_sections(section_values)
        screen.app.notify(
            "Network TLS setting saved."
            if saved
            else "Failed to save Network TLS setting.",
            severity="information" if saved else "error",
        )
        if saved:
            # Sync the app's in-memory config mapping so the next detail
            # render (which seeds from _app_config_mapping, not the runtime
            # config cache) shows the saved value -- same shape as
            # _apply_appearance_save_result (qodo PR #2223, bug 5).
            screen._app_config_update_target().update(
                copy.deepcopy(dict(section_values))
            )
            self._clear(screen)

    def revert(self, screen: SettingsScreen) -> None:
        if not self.has_unsaved_changes(screen):
            screen.app.notify(NO_CHANGES_TO_REVERT_COPY, severity="information")
            return
        screen._confirm_category_revert(self.category, lambda: self._discard(screen))

    def scope_text(self, screen: SettingsScreen) -> ScopeText:
        return ScopeText(
            banner="Trust applies to new connections; open ones keep theirs.",
            guided=(
                "Save with s: the TLS trust setting applies to newly created "
                "clients; already-open connections keep their trust policy "
                "until restart."
            ),
        )

    # -- state helpers ----------------------------------------------------

    def pending(self, screen: SettingsScreen) -> dict[str, object]:
        """The staged edits (read-only view; ``stage`` writes them)."""
        return screen._category_panel_states.get(self.category, {})

    def stage(self, screen: SettingsScreen, **values: object) -> None:
        """Stage an edit from the card, then repaint the dirty markers."""
        screen._category_panel_states.setdefault(self.category, {}).update(values)
        screen._update_draft_status_widgets(self.category)

    def effective_values(
        self, screen: SettingsScreen, pending: dict[str, object] | None = None
    ) -> SettingsNetworkTLS:
        """The loaded config with the staged edits (or ``pending``) applied."""
        pending = self.pending(screen) if pending is None else pending
        loaded = load_network_tls(screen._app_config_mapping())
        mode = pending.get("mode")
        mode = mode if isinstance(mode, str) else loaded.mode
        if mode in ("verify", "off", "invalid"):
            return SettingsNetworkTLS(mode, raw=loaded.raw)
        path = str(pending.get("ca_bundle_path", loaded.ca_bundle_path))
        return SettingsNetworkTLS(mode, ca_bundle_path=path, raw=loaded.raw)

    @staticmethod
    def warning_text(mode: str) -> str:
        if mode == "off":
            return (
                "Verification is DISABLED: API keys and conversation content"
                " can be intercepted by anyone on the network path."
            )
        if mode == "custom-ca":
            return (
                "Verification additionally trusts your custom CA bundle"
                " (corporate root CA)."
            )
        return ""

    def _clear(self, screen: SettingsScreen) -> None:
        screen._category_panel_states.pop(self.category, None)
        screen._update_draft_status_widgets(self.category)

    def _discard(self, screen: SettingsScreen) -> None:
        self._clear(screen)
        # Rebuild the pane so the Select and Input show the loaded values.
        screen.mutate_reactive(type(screen).active_category)
        screen.app.notify("Settings category changes reverted.", severity="information")


class SpeechTTSCategoryPanel:
    """Speech & TTS: routes the mounted widget's own contract through here.

    ponytail: a thin adapter, not a full move. Still in the screen:

    - the detail pane, ``SettingsScreen._render_speech_tts_detail`` (its names
      are patched on the screen module by many tests);
    - the dirty banner's leave wording, the unsaved count and the F1 note;
    - the category-leave guard in ``_select_category``. It asks the live
      widget (``has_unsaved_changes()`` collects visible fields the
      ``_settings_drafts`` mirror has not heard of yet) and the audio_cpp
      cleanup, not this adapter. ``revert`` below reads the widget the same
      way, while ``has_unsaved_changes`` reads the mirror (as at base).

    Routing the guard here first needs one unsaved answer for Speech (live
    widget or mirror) for every caller. Move all of these when Speech moves
    for real.
    """

    category = SettingsCategoryId.SPEECH_TTS
    commit_model = CommitModel("staged", "Draft — save with s")

    def render(self, screen: SettingsScreen) -> ComposeResult:
        yield from screen._render_speech_tts_detail()

    def has_unsaved_changes(self, screen: SettingsScreen) -> bool:
        # Mirrored from the widget's DraftModified messages.
        draft = screen._settings_drafts.get(self.category)
        return bool(draft and draft.is_dirty)

    def request_save(self, screen: SettingsScreen) -> None:
        widget = self._mounted(screen)
        if widget is not None:
            widget.request_save()

    def revert(self, screen: SettingsScreen) -> None:
        widget = self._mounted(screen)
        if widget is None:
            return
        if not widget.has_unsaved_changes():
            screen.app.notify(NO_CHANGES_TO_REVERT_COPY, severity="information")
            return
        screen.run_worker(widget.revert_to_saved(), exclusive=False)

    def scope_text(self, screen: SettingsScreen) -> ScopeText:
        return ScopeText(
            banner="Application-wide defaults; Studio preferences remain separate.",
            guided=(
                "Guided edits: Save or Revert changes."
                if self.has_unsaved_changes(screen)
                else "Guided edits: change a field first."
            ),
        )

    @staticmethod
    def _mounted(screen: SettingsScreen) -> SpeechTTSSettingsPanel | None:
        try:
            return screen.query_one("#settings-speech-tts-panel", SpeechTTSSettingsPanel)
        except QueryError:
            screen.app.notify(
                "Speech & TTS Settings are not available.", severity="warning"
            )
            return None


class ProvidersModelsPanel:
    """Providers & Models: the default provider, model and credentials.

    Staged commit through the category's ``SettingsDraft`` (plus the snapshot
    preferences, which keep their own pair outside it). Its detail pane, the
    card's handlers and its helpers still live on the screen; this panel
    owns the category's save, revert, unsaved state and copy.
    """

    category = SettingsCategoryId.PROVIDERS_MODELS
    commit_model = CommitModel("staged", "Draft — save with s")

    def render(self, screen: SettingsScreen) -> ComposeResult:
        yield from screen._render_provider_detail()

    def has_unsaved_changes(self, screen: SettingsScreen) -> bool:
        draft = screen._settings_drafts.get(self.category)
        return bool(draft and draft.is_dirty) or screen._snapshot_preferences_dirty()

    def request_save(self, screen: SettingsScreen) -> None:
        from dataclasses import replace

        from ...Chat.provider_setup_persistence import (
            MAX_MODEL_ID_CHARS,
            build_provider_setup_mutation,
            is_writable_model_id,
        )
        from ...model_capabilities import reload_capabilities
        from ..Screens.settings_context_memory import (
            model_context_window_reset_entry,
            model_context_window_save_entry,
        )

        # Resolved in settings_screen's namespace at call time, as
        # _settings_config_adapter() is, so tests that patch these names on
        # that module (the writer above all) reach this save.
        from ..Screens.settings_screen import (
            PROVIDER_MODEL_PROFILE_FIELD_KEYS,
            PROVIDER_NO_CHANGES_COPY,
            PROVIDER_SAVE_SCOPE_COPY,
            PROVIDER_SAVED_COPY,
            QWENCLOUD_API_MODE_INVALID_COPY,
            QWENCLOUD_PROVIDER_TABLE_INVALID_COPY,
            ChatConfigurationError,
            ConfigMutationResult,
            ConversationSettingsReturnOutcome,
            ProviderSetupDraft,
            _anthropic_auth_sources,
            _apply_sparse_config_mutation_in_memory,
            normalize_qwencloud_api_mode,
            persist_provider_settings_atomic,
            provider_config_key,
        )

        if screen._snapshot_preferences_saving:
            return
        if screen._snapshot_preferences_dirty():
            screen._snapshot_preferences_saving = True
            draft = screen._settings_drafts.get(self.category)
            screen.run_worker(
                screen._save_snapshot_preferences_draft(
                    screen._snapshot_preferences_raw,
                    screen._snapshot_preferences_loaded,
                    bool(draft and draft.is_dirty),
                ),
                group="settings-snapshot-save",
                exclusive=True,
                exit_on_error=False,
            )
            return
        current_provider = str(
            screen._provider_setting_values_mapping().get("provider") or ""
        ).strip()
        screen._snapshot_provider_api_mode_widget(current_provider)
        screen._snapshot_provider_auth_source_widget(current_provider)
        try:
            values = screen._provider_form_values_from_widgets()
        except ValueError as exc:
            screen._provider_save_result = (
                str(exc) or "Model profile values are invalid."
            )
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(screen._provider_save_result, severity="error")
            return
        loaded_values = screen._provider_loaded_setting_values()
        chat_defaults_keys = {"provider", "model"}
        provider = str(values.get("provider") or "").strip()
        model = str(values.get("model") or "").strip()
        endpoint = str(values.get("endpoint") or "").strip()
        api_key = str(values.get("api_key") or "").strip()
        credential_env_var = str(values.get("credential_env_var") or "").strip()
        draft = screen._settings_drafts.get(self.category)
        provider_key = provider_config_key(provider)
        if not provider_key:
            screen._provider_save_result = "Provider is required."
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(screen._provider_save_result, severity="error")
            return
        normalized_api_mode: str | None = None
        if provider_key == "qwencloud":
            if screen._qwencloud_provider_table_is_malformed(provider):
                screen._provider_save_result = QWENCLOUD_PROVIDER_TABLE_INVALID_COPY
                screen._set_static_text(
                    "#settings-provider-save-result", screen._provider_save_result
                )
                try:
                    screen.query_one("#settings-provider-api-mode", Select).add_class(
                        "settings-invalid-input"
                    )
                except QueryError:
                    pass
                screen.app.notify(screen._provider_save_result, severity="error")
                return
            try:
                normalized_api_mode = normalize_qwencloud_api_mode(
                    values.get("api_mode")
                )
            except ChatConfigurationError:
                screen._provider_save_result = QWENCLOUD_API_MODE_INVALID_COPY
                screen._set_static_text(
                    "#settings-provider-save-result", screen._provider_save_result
                )
                try:
                    screen.query_one("#settings-provider-api-mode", Select).add_class(
                        "settings-invalid-input"
                    )
                except QueryError:
                    pass
                screen.app.notify(screen._provider_save_result, severity="error")
                return
            values["api_mode"] = normalized_api_mode
        endpoint_touched = draft is not None and "endpoint" in draft.dirty_keys
        loaded_endpoint = str(loaded_values.get("endpoint") or "").strip()
        if (
            loaded_values.get("provider") != provider
            and not endpoint_touched
            and endpoint == loaded_endpoint
        ):
            endpoint = screen._provider_endpoint_value(provider)
            values["endpoint"] = endpoint
            try:
                screen.query_one(
                    "#settings-provider-endpoint-value", Input
                ).value = endpoint
            except QueryError:
                pass
        endpoint_validation_error = screen._validate_provider_endpoint(endpoint)
        if endpoint_validation_error:
            screen._provider_save_result = endpoint_validation_error
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(endpoint_validation_error, severity="error")
            return
        credential_validation_error = screen._validate_credential_env_var(
            credential_env_var
        )
        if credential_validation_error:
            screen._provider_save_result = credential_validation_error
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(credential_validation_error, severity="error")
            return
        api_key_validation_error = screen._validate_provider_api_key(api_key)
        if api_key_validation_error:
            screen._provider_save_result = api_key_validation_error
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(api_key_validation_error, severity="error")
            return
        dirty_values = {
            key: value
            for key, value in values.items()
            if key in chat_defaults_keys
            and draft is not None
            and key in draft.dirty_keys
            and loaded_values.get(key) != value
        }
        dirty_keys = draft.dirty_keys if draft is not None else set()
        api_mode_draft_key = screen._provider_api_mode_draft_key(provider)
        api_mode_dirty = bool(
            provider_key == "qwencloud"
            and draft is not None
            and api_mode_draft_key in dirty_keys
            and normalized_api_mode != screen._provider_saved_api_mode_value(provider)
        )
        auth_source = str(values.get("auth_source") or _anthropic_auth_sources()[0])
        auth_source_dirty = bool(
            provider_key == "anthropic"
            and draft is not None
            and screen._provider_auth_source_draft_key(provider) in dirty_keys
            and auth_source in _anthropic_auth_sources()
            and auth_source != screen._provider_saved_auth_source(provider)
        )
        selected_profile = screen._provider_model_profile(provider, model)
        model_profile_dirty = any(
            key in dirty_keys
            and screen._model_profile_field_supported(provider, key, model)
            and values.get(key, "") != selected_profile.get(profile_key, "")
            for key, profile_key in PROVIDER_MODEL_PROFILE_FIELD_KEYS.items()
        )
        context_window_dirty = (
            draft is not None
            and "model_context_window" in dirty_keys
            and values.get("model_context_window", "")
            != loaded_values.get("model_context_window", "")
        )
        context_window_reset = bool(
            draft is not None and draft.values.get("model_context_window_reset")
        )
        context_window_dirty = context_window_dirty or context_window_reset
        provider_section_key, _provider_config = screen._provider_config_entry(
            provider
        )
        current_provider_endpoint = screen._provider_endpoint_value(provider)
        current_credential_env_var = screen._provider_credential_env_var(provider)
        api_key_dirty = draft is not None and "api_key" in draft.dirty_keys
        endpoint_dirty = bool(
            draft is not None
            and "endpoint" in draft.dirty_keys
            and endpoint != current_provider_endpoint
        )
        credential_dirty = bool(
            draft is not None
            and "credential_env_var" in draft.dirty_keys
            and credential_env_var != current_credential_env_var
        )
        connection_dirty = bool(
            dirty_values or endpoint_dirty or credential_dirty or api_key_dirty
        )
        credential_only_save = bool(
            (credential_dirty or api_key_dirty)
            and not dirty_values
            and not endpoint_dirty
            and not api_mode_dirty
            and not auth_source_dirty
            and not model_profile_dirty
            and not context_window_dirty
        )
        provider_return_outcome = (
            ConversationSettingsReturnOutcome.CREDENTIAL_SAVED
            if credential_only_save
            else ConversationSettingsReturnOutcome.PROVIDER_SETTINGS_SAVED
        )
        if endpoint_dirty and not provider_key:
            screen._provider_save_result = (
                "Provider is required before saving an endpoint."
            )
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(screen._provider_save_result, severity="error")
            return
        if credential_dirty and not provider_key:
            screen._provider_save_result = (
                "Provider is required before saving a credential source."
            )
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(screen._provider_save_result, severity="error")
            return
        if api_key_dirty and not provider_key:
            screen._provider_save_result = (
                "Provider is required before saving an API key."
            )
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(screen._provider_save_result, severity="error")
            return
        if model_profile_dirty and not model:
            screen._provider_save_result = (
                "Model is required before saving a model default profile."
            )
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(screen._provider_save_result, severity="error")
            return
        if context_window_dirty and not model:
            screen._provider_save_result = (
                "Model is required before saving its context window."
            )
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(screen._provider_save_result, severity="error")
            return
        if (
            context_window_dirty
            and not context_window_reset
            and not values.get("model_context_window")
        ):
            screen._provider_save_result = (
                "Model context window must be a positive whole number."
            )
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(screen._provider_save_result, severity="error")
            return
        if model_profile_dirty and not provider_key:
            screen._provider_save_result = (
                "Provider is required before saving a model default profile."
            )
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(screen._provider_save_result, severity="error")
            return
        retained_api_mode_draft = screen._provider_api_mode_draft_snapshot(
            excluding_provider=provider
        )
        if (
            not dirty_values
            and not endpoint_dirty
            and not credential_dirty
            and not api_key_dirty
            and not api_mode_dirty
            and not auth_source_dirty
            and not model_profile_dirty
            and not context_window_dirty
        ):
            screen._settings_drafts.pop(self.category, None)
            screen._restore_provider_api_mode_draft_snapshot(retained_api_mode_draft)
            screen._update_provider_dynamic_widgets()
            screen._update_draft_status_widgets(self.category)
            screen._provider_save_result = PROVIDER_NO_CHANGES_COPY
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify("No Settings changes to save.", severity="information")
            return
        if model and not is_writable_model_id(model):  # TASK-35712
            # The save refuses it before touching the file; say why.
            screen._provider_save_result = (
                "Provider settings not saved: the model ID is invalid. Use a "
                f"single-line value of at most {MAX_MODEL_ID_CHARS} characters, "
                "with no invisible or control characters. Draft retained."
            )
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(screen._provider_save_result, severity="error")
            return
        alias_mode_sections = screen._qwencloud_alias_mode_sections()
        qwencloud_mode_requires_canonical_save = bool(
            api_mode_dirty
            and normalized_api_mode is not None
            and (provider_section_key != "qwencloud" or alias_mode_sections)
        )
        next_model_defaults = None
        if model_profile_dirty and provider_key and model:
            next_model_defaults = screen._updated_model_defaults_for_values(
                provider,
                model,
                values,
                dirty_keys=dirty_keys,
            )
        next_model_capabilities = None
        delete_model_capabilities_entry = False
        if context_window_dirty and model:
            if context_window_reset:
                next_model_capabilities = model_context_window_reset_entry(
                    screen._app_config_mapping(), model
                )
                if next_model_capabilities is None:
                    delete_model_capabilities_entry = True
            else:
                try:
                    next_model_capabilities = model_context_window_save_entry(
                        screen._app_config_mapping(),
                        provider,
                        model,
                        values["model_context_window"],
                    )
                except ValueError as exc:
                    screen._provider_save_result = str(exc)
                    screen._set_static_text(
                        "#settings-provider-save-result",
                        screen._provider_save_result,
                    )
                    screen.app.notify(screen._provider_save_result, severity="error")
                    return

        tested_identity = (
            screen._provider_current_draft_identity() if connection_dirty else None
        )
        setup_mutation = None
        section_values: dict[str, dict[str, object]] = {}
        delete_keys: dict[str, list[str]] = {}
        if connection_dirty:
            # Only a save that writes the endpoint refuses (review r8e-rev1).
            from tldw_chatbook.Provider_Contract.provider_endpoint_contract import (
                koboldcpp_route_refusal,
            )

            if refusal := koboldcpp_route_refusal(provider_key, endpoint):
                screen._provider_save_result = refusal
                screen._set_static_text(
                    "#settings-provider-save-result", screen._provider_save_result
                )
                screen.app.notify(refusal, severity="error", markup=False)
                return
            credential_source = screen._provider_current_credential_source(provider)
            try:
                setup_mutation = build_provider_setup_mutation(
                    ProviderSetupDraft(
                        provider=provider,
                        model=model,
                        endpoint=endpoint,
                        credential_source=credential_source,
                        credential_revision=screen._provider_draft_credential_revision(
                            provider,
                            credential_source,
                            {
                                "api_key": api_key,
                                "credential_env_var": credential_env_var,
                            },
                        ),
                        draft_generation=getattr(
                            screen, "_provider_draft_generation", 0
                        ),
                        credential_value=(
                            api_key if credential_source == "draft" else None
                        ),
                        credential_env_var=(
                            credential_env_var
                            if credential_source == "environment"
                            else None
                        ),
                    ),
                    screen._app_config_mapping(),
                    # Settings' "stored" is always the key already saved.
                    keep_stored_credential=True,
                )
            except ValueError:
                screen._provider_save_result = (
                    "Provider settings are invalid; review the endpoint and model."
                )
                screen._set_static_text(
                    "#settings-provider-save-result", screen._provider_save_result
                )
                screen.app.notify(screen._provider_save_result, severity="error")
                return
            section_values = {
                section: dict(section_data)
                for section, section_data in setup_mutation.section_values.items()
            }
            delete_keys = {
                section: list(keys)
                for section, keys in setup_mutation.delete_keys.items()
            }
            provider_section = next(
                section
                for section in section_values
                if section.startswith("api_settings.")
            )
        else:
            provider_section = (
                f"api_settings.{provider_section_key or provider_key}"
            )
        if api_mode_dirty and normalized_api_mode is not None:
            mode_section = (
                "api_settings.qwencloud"
                if qwencloud_mode_requires_canonical_save
                else provider_section
            )
            section_values.setdefault(mode_section, {})["api_mode"] = (
                normalized_api_mode
            )
            if qwencloud_mode_requires_canonical_save:
                for alias_section in alias_mode_sections:
                    delete_keys.setdefault(
                        f"api_settings.{alias_section}", []
                    ).append("api_mode")
        if auth_source_dirty:
            # TASK-34201: Anthropic's sign-in choice; the setup mutation
            # admits it as a routing key of [api_settings.anthropic].
            section_values.setdefault(provider_section, {})["auth_source"] = (
                auth_source
            )
        if next_model_defaults is not None:
            section_values.setdefault(provider_section, {})["model_defaults"] = (
                next_model_defaults
            )
        if next_model_capabilities is not None:
            section_values.setdefault("model_capabilities.models", {})[model] = (
                next_model_capabilities
            )
        elif delete_model_capabilities_entry:
            delete_keys.setdefault("model_capabilities.models", []).append(model)

        normalized_delete_keys = {
            section: tuple(dict.fromkeys(keys))
            for section, keys in delete_keys.items()
            if keys
        }
        from .saved_model_list import LIST_CHANGED_COPY, join_pick

        join = join_pick(screen, provider, model, section_values)
        evidence_store = screen._provider_evidence_store()
        save_lease = (
            evidence_store.begin_save(tested_identity)
            if setup_mutation is not None and tested_identity is not None
            else None
        )
        try:
            mutation_result = persist_provider_settings_atomic(
                setup_mutation,
                provider=provider,
                model=model,
                section_values=section_values,
                delete_keys=normalized_delete_keys,
                locked_snapshot_precondition=join.precondition,
            )
        except ValueError:
            mutation_result = ConfigMutationResult(
                False,
                False,
                "before_replace",
            )
        evidence_preserved = setup_mutation is None
        if (
            setup_mutation is not None
            and save_lease is not None
            and tested_identity is not None
        ):
            saved_identity = setup_mutation.semantic_identity
            if saved_identity is None and not endpoint:
                # TASK-33005.4 (AC#12): no endpoint is saved, so the send
                # uses the built-in one the tested identity already keys
                # (TASK-33005.1); a typed key is now the stored key.
                saved_identity = replace(
                    tested_identity,
                    credential_source=(
                        "stored"
                        if tested_identity.credential_source == "draft"
                        else tested_identity.credential_source
                    ),
                )
            evidence_preserved = evidence_store.rebase_after_save(
                tested_identity,
                saved_identity,
                mutation_result,
                lease=save_lease,
            )

        if mutation_result.fully_applied:
            _apply_sparse_config_mutation_in_memory(
                screen._app_config_update_target(),
                section_values,
                normalized_delete_keys,
            )
            join.applied(screen, provider)
            if (
                next_model_capabilities is not None
                or delete_model_capabilities_entry
            ):
                reload_capabilities()
            screen._settings_drafts.pop(self.category, None)
            screen._provider_api_key_cleared = None
            screen._restore_provider_api_mode_draft_snapshot(retained_api_mode_draft)
            screen._provider_save_result = PROVIDER_SAVED_COPY
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            if not evidence_preserved:
                screen._mark_provider_test_result_stale()
            elif setup_mutation is not None and (  # TASK-33005.8: drop "(draft)"
                screen._provider_test_result not in screen._PROVIDER_TEST_SENTINELS
            ):
                screen._provider_test_result = screen._run_provider_readiness_test()
                screen._update_provider_test_result()
            if screen._navigation_provider:
                # A deep link's form now holds the model that was saved.
                screen._navigation_model = model
                screen._navigation_model_is_explicit = True
            screen._sync_provider_credential_widget(provider)
            screen._sync_provider_context_window_widget(provider, model)
            screen._update_provider_dynamic_widgets()
            from .providers_models_card import (
                refresh_next_new_chat,
                refresh_openai_reconnect_pending,
            )

            refresh_next_new_chat(screen)
            if provider_key == "openai" and connection_dirty:
                # A changed key, env var or endpoint can need a new review.
                refresh_openai_reconnect_pending(screen)
            # TASK-33007.2: a saved key or endpoint can change who leads
            # the provider list and its "configured: ..." help.
            screen._provider_configured_keys = None
            screen._refresh_provider_picker()
            screen._update_draft_status_widgets(self.category)
            conflict_target = screen._provider_navigation_conflict_target
            if conflict_target is not None:
                screen._provider_navigation_conflict_target = None
                screen._provider_return_outcome = None
                screen._navigation_provider = conflict_target.provider
                screen._navigation_model = conflict_target.model
                screen._navigation_model_is_explicit = conflict_target.pins_model
                screen._navigation_field = conflict_target.field
                screen._apply_navigation_provider_context(
                    conflict_target.provider,
                    conflict_target.model,
                    conflict_target.field,
                )
            elif screen._provider_return_target is not None:
                screen._provider_return_outcome = provider_return_outcome
            screen._update_provider_return_widgets()
            if screen._provider_return_outcome is not None:
                screen.call_after_refresh(screen._focus_provider_return_continuation)
            screen.app.notify(
                f"Provider and model settings saved: {PROVIDER_SAVE_SCOPE_COPY}"
                f"{join.receipt}",
                severity="information",
                markup=False,  # the receipt names a model id
            )
        else:
            if mutation_result.file_replaced:
                screen._provider_save_result = (
                    "Provider settings not fully applied: the file was written, "
                    "but the app could not reload it. Draft retained; restart "
                    "before retrying."
                )
            elif mutation_result.conflict:  # only join's check refuses here
                screen._provider_save_result = LIST_CHANGED_COPY
            else:
                screen._provider_save_result = (
                    "Provider settings not fully applied: the file was not "
                    "written. Draft retained; check config access and retry."
                )
            screen._set_static_text(
                "#settings-provider-save-result", screen._provider_save_result
            )
            screen.app.notify(screen._provider_save_result, severity="error")

    def revert(self, screen: SettingsScreen) -> None:
        dirty = self.has_unsaved_changes(screen)
        if screen._snapshot_preferences_unavailable and not dirty:
            screen.run_worker(
                screen._revert_snapshot_preferences(),
                group="settings-snapshot-revert",
                exclusive=True,
                exit_on_error=False,
            )
            return
        if not dirty:
            screen.app.notify(NO_CHANGES_TO_REVERT_COPY, severity="information")
            return
        screen._confirm_category_revert(self.category, lambda: self.discard(screen))

    def discard(self, screen: SettingsScreen) -> None:
        """Drop the draft and repaint the card from saved values (after ``r``)."""
        from ...Widgets.select_values import assign_select_value

        discarded = screen._settings_drafts.pop(self.category, None)
        screen.run_worker(
            screen._revert_snapshot_preferences(),
            group="settings-snapshot-revert",
            exclusive=True,
            exit_on_error=False,
        )
        screen._provider_api_key_cleared = None
        # A deep link's form goes back to the linked provider, not the default.
        values = screen._provider_display_setting_values()
        try:
            provider = str(values["provider"])
            screen._syncing_provider_selection = True
            try:
                revert_select = screen.query_one("#settings-provider-value", Select)
                # task-15740: prevent the posted echo the flag misses.
                with revert_select.prevent(Select.Changed):
                    # TASK-32533: same catalog-vs-composed-options
                    # disagreement as `_sync_provider_manual_widget`; a
                    # refusal needs no handling here because the
                    # `_sync_provider_manual_widget` call below falls the
                    # select back to the manual spelling.
                    assign_select_value(
                        revert_select,
                        screen._provider_select_value_for_provider(provider),
                    )
            finally:
                screen._syncing_provider_selection = False
            screen._sync_provider_manual_widget(provider)
            screen._set_model_field_value(
                str(values["model"]), provider=provider, quiet=False
            )
            endpoint_input = screen.query_one(
                "#settings-provider-endpoint-value", Input
            )
            endpoint_input.value = str(values["endpoint"])
            endpoint_input.placeholder = screen._provider_endpoint_placeholder(
                provider
            )
            api_key_input = screen.query_one("#settings-provider-api-key", Input)
            screen._set_provider_api_key_input_value(
                api_key_input,
                str(values.get("api_key") or ""),
            )
            api_key_input.placeholder = screen._provider_api_key_placeholder(provider)
            credential_input = screen.query_one(
                "#settings-provider-credential-env-var",
                Input,
            )
            credential_input.value = str(values["credential_env_var"])
            credential_input.placeholder = screen._provider_credential_placeholder(
                provider
            )
            # Values, and which rows this provider shows (and the window).
            screen._sync_provider_model_profile_widgets(
                provider, str(values["model"])
            )
        except QueryError:
            pass
        # Capture finding 5: the toast below is the revert's one report;
        # a line left from before went with the draft it described.
        screen._provider_save_result = ""
        screen._set_static_text("#settings-provider-save-result", "")
        # Only a discarded tested field voids the Test rows, and its
        # in-flight probe must not settle onto the saved identity.
        tested = {"provider", "model", "endpoint", "api_key", "credential_env_var"}
        if discarded is not None and discarded.dirty_keys & tested:
            screen._provider_evidence_store().invalidate()
            screen._mark_provider_test_result_stale()
            # A listing discovered from the discarded draft goes with it.
            screen._reset_provider_model_discovery_state(forget_pick=True)
        screen._update_provider_dynamic_widgets()
        screen._update_draft_status_widgets(self.category)
        if screen._active_category_id() is self.category:
            screen._scroll_impact_pane_to_field_guide(self.category)  # rows above it went
        screen.app.notify("Settings category changes reverted.", severity="information")

    def scope_text(self, screen: SettingsScreen) -> ScopeText:
        return ScopeText(
            # One-row form of PROVIDER_SAVE_SCOPE_COPY: change both together.
            banner=(
                "Applies to new and unused open chats · used chats keep theirs "
                "(Console: Alt+M)"
            ),
            guided=(
                "Guided edits: Save or Revert changes."
                if self.has_unsaved_changes(screen)
                else "Guided edits: change a field first."
            ),
        )


NETWORK_PANEL = NetworkPanel()
PROVIDERS_MODELS_PANEL = ProvidersModelsPanel()

#: Categories served through a registered panel. For these the screen asks
#: the panel instead of keeping an arm; add a category here when it moves.
CATEGORY_PANELS: dict[SettingsCategoryId, SettingsCategoryPanel] = {
    SettingsCategoryId.NETWORK: NETWORK_PANEL,
    SettingsCategoryId.PROVIDERS_MODELS: PROVIDERS_MODELS_PANEL,
    SettingsCategoryId.SPEECH_TTS: SpeechTTSCategoryPanel(),
}
