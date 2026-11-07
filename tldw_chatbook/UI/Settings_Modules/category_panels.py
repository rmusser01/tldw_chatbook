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

    def __init__(self, settings_screen: SettingsScreen, **kwargs) -> None:
        super().__init__(**kwargs)
        self._settings_screen = settings_screen

    @on(Select.Changed, "#settings-network-ssl-mode")
    def _mode_changed(self, event: Select.Changed) -> None:
        event.stop()
        NETWORK_PANEL.stage(self._settings_screen, mode=str(event.value or "verify"))
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
        NETWORK_PANEL.stage(self._settings_screen, ca_bundle_path=event.value)


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
        yield Static("Network", classes="destination-section settings-column-title")
        with _NetworkCard(
            screen, id="settings-network-card", classes="settings-focus-card"
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
                    value=(
                        effective.mode
                        if effective.mode in _NETWORK_TLS_MODE_VALUES
                        else "verify"
                    ),
                    id="settings-network-ssl-mode",
                    classes="settings-compact-select",
                    allow_blank=False,
                    compact=True,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("CA bundle path", classes="settings-input-label")
                yield Input(
                    value=(
                        effective.ca_bundle_path
                        if effective.mode == "custom-ca"
                        else ""
                    ),
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
        # Compared by value, not by "anything staged": the Select and Input
        # echo their seeded values on mount, which is no edit.
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

    ponytail: a thin adapter, not a full move. The detail pane is still built
    by ``SettingsScreen._render_speech_tts_detail`` (its names are patched on
    the screen module by many tests), and the dirty banner, unsaved count and
    F1 note keep their Speech arms in the screen. Move them here when Speech
    moves for real.
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


NETWORK_PANEL = NetworkPanel()

#: Categories served through a registered panel. For these the screen asks
#: the panel instead of keeping an arm; add a category here when it moves.
CATEGORY_PANELS: dict[SettingsCategoryId, SettingsCategoryPanel] = {
    SettingsCategoryId.NETWORK: NETWORK_PANEL,
    SettingsCategoryId.SPEECH_TTS: SpeechTTSCategoryPanel(),
}
