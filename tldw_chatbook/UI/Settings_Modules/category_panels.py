"""Per-category panels for the Settings screen (TASK-1378.1).

``SettingsScreen`` used to render, label and save every category through
if/elif chains keyed on ``SettingsCategoryId``. A category registered in
``CATEGORY_PANELS`` is served by its panel instead: the screen asks the
panel to render, save, report unsaved state, revert and describe its scope,
and carries no arm for that category. Network is the first category served
this way; the others still use the screen's chains until they are moved.

Panels are stateless strategies. The screen owns the per-visit state (for
Network, ``screen._network_pending``) and each method takes it as ``screen``.
This generalises the Speech & TTS panel's ``has_unsaved_changes`` /
``request_save`` pair, which stays a widget the screen mounts and queries.

Every panel names its ADR-303 commit model (``CommitModel``) so the screen's
badge is the panel's own words, not a lookup keyed on the category.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Select, Static

from ...state.settings_config_models import SettingsCategoryId
from ..Screens.settings_network_defaults import (
    SettingsNetworkTLS,
    build_network_save_sections,
    load_network_tls,
    validate_network_tls,
)
from .settings_config_adapter import SettingsConfigAdapter

if TYPE_CHECKING:
    from ..Screens.settings_screen import SettingsScreen


@dataclass(frozen=True)
class CommitModel:
    """Which ADR-303 commit model a category follows, and its badge copy."""

    kind: Literal["staged", "instant", "raw_toml"]
    badge: str


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

    def revert(self, screen: SettingsScreen) -> bool:
        """Discard unsaved edits; False when the category offers no revert."""
        ...

    def scope_text(self, screen: SettingsScreen) -> str:
        """The one-line scope/commit guidance shown for the category."""
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


class NetworkPanel:
    """Network: TLS trust for outbound API traffic.

    Staged commit with no ``SettingsDraft``: edits accumulate in
    ``screen._network_pending`` until ``s`` validates and saves them.
    """

    commit_model = CommitModel("staged", "Pending — save with s")

    def render(self, screen: SettingsScreen) -> ComposeResult:
        values = load_network_tls(screen._app_config_mapping())
        # Seed the Select and CA Input from the EFFECTIVE (pending-aware)
        # state, not the loaded config: `_network_pending` survives a
        # category hop, so seeding from the loaded values let a hop-away-and-
        # back repaint the Select with the LOADED mode while `s` still saved
        # the PENDING one. The invalid-row banner below stays keyed on the
        # LOADED mode -- a hand-edited invalid config is a config-file fact,
        # not a pending edit.
        effective = self.effective_values(screen)
        yield Static("Network", classes="destination-section settings-column-title")
        with Vertical(id="settings-network-card", classes="settings-focus-card"):
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
                self.warning_text(self.effective_mode(screen)),
                id="settings-network-warning",
                classes="settings-network-warning",
            )

    def has_unsaved_changes(self, screen: SettingsScreen) -> bool:
        # ponytail: not consulted by the screen's dirty marker / quit guard;
        # Network has never raised them (ADR-303 staged, no SettingsDraft).
        return bool(screen._network_pending)

    def request_save(self, screen: SettingsScreen) -> None:
        values = self.effective_values(screen)
        validation = validate_network_tls(values)
        if not validation.valid:
            screen.app.notify(validation.message, severity="error", markup=False)
            return
        section_values = build_network_save_sections(values)
        saved = SettingsConfigAdapter().save_sections(section_values)
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
            screen._network_pending = {}

    def revert(self, screen: SettingsScreen) -> bool:
        # Network offers no revert: `r` is not advertised and answers with
        # guidance. Switching the Select back is the way to discard.
        return False

    def scope_text(self, screen: SettingsScreen) -> str:
        return (
            "Save with s: the TLS trust setting applies to newly created "
            "clients; already-open connections keep their trust policy "
            "until restart."
        )

    # -- state helpers (also used by the screen's Select/Input handlers) --

    def effective_mode(self, screen: SettingsScreen) -> str:
        loaded = load_network_tls(screen._app_config_mapping())
        pending_mode = screen._network_pending.get("mode")
        return pending_mode if isinstance(pending_mode, str) else loaded.mode

    def effective_values(self, screen: SettingsScreen) -> SettingsNetworkTLS:
        loaded = load_network_tls(screen._app_config_mapping())
        mode = self.effective_mode(screen)
        if mode in ("verify", "off", "invalid"):
            return SettingsNetworkTLS(mode, raw=loaded.raw)
        path = str(screen._network_pending.get("ca_bundle_path", loaded.ca_bundle_path))
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


NETWORK_PANEL = NetworkPanel()

#: Categories served by a registered panel. The screen holds no per-category
#: arm for these; add a category here when it moves.
CATEGORY_PANELS: dict[SettingsCategoryId, SettingsCategoryPanel] = {
    SettingsCategoryId.NETWORK: NETWORK_PANEL,
}
