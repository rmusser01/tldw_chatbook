"""Privacy preflight and destination flow for Trace v2 collaboration export."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import DescendantBlur, DescendantFocus
from textual.screen import ModalScreen
from textual.widgets import Button, Input, RadioButton, RadioSet, Static

from tldw_chatbook.Chat.trace_export_profiles import TraceExportProfile
from tldw_chatbook.Chat.trajectory import TrajectorySnapshot
from tldw_chatbook.Chat.trajectory_export import (
    TraceExportPreflight,
    build_trace_export,
    preflight_trace_export,
    write_trajectory_export,
)
from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

# Re-exported for this dialog's own (deferred-family) consumers. The shared
# copy/labels/confirmation live in `trace_export_profile_ui` (TASK-23020) so
# the Chat-first-paint-leg exchange export dialog can import them WITHOUT
# resolving this module -- this module's imports above drag the whole
# `Chat/trajectory_export.py` engine, which must stay off that leg
# (TASK-22213; guard:
# `Tests/Packaging/test_exchange_export_trajectory_deferral.py`).
from tldw_chatbook.Widgets.Console.trace_export_profile_ui import (
    TRACE_EXPORT_PROFILE_COPY,
    TRACE_EXPORT_PROFILE_LABELS,
    full_trace_confirmation,
)

__all__ = [
    "TRACE_EXPORT_PROFILE_COPY",
    "TRACE_EXPORT_PROFILE_LABELS",
    "TraceExportDialog",
    "full_trace_confirmation",
]


class TraceExportDialog(SafeModalDismissMixin, ModalScreen[Path | None]):
    """Choose a privacy profile, inspect its inventory, and atomically export."""

    SAFE_MODAL_CONTENT = "#trace-export-dialog"
    BINDINGS = [
        Binding("escape", "request_safe_cancel", "Cancel", show=False),
    ]

    BUNDLED_SCREEN_CSS = """
    TraceExportDialog {
        align: center middle;
    }
    #trace-export-dialog {
        width: 78;
        max-width: 96%;
        height: 24;
        max-height: 96%;
        border: tall $primary;
        background: $surface;
        padding: 1 2;
        scrollbar-gutter: stable;
    }
    #trace-export-body {
        height: 1fr;
        scrollbar-gutter: stable;
    }
    #trace-export-title {
        height: 1;
        text-style: bold;
    }
    #trace-export-intro,
    #trace-export-profile-copy {
        height: auto;
        margin-top: 1;
    }
    #trace-export-policy,
    #trace-export-inventory,
    #trace-export-selection,
    #trace-export-status {
        height: auto;
    }
    #trace-export-policy {
        color: $warning;
    }
    #trace-export-profiles {
        height: auto;
        margin-top: 1;
    }
    #trace-export-selection {
        text-style: bold;
    }
    #trace-export-selection.is-selector-focused {
        color: $accent;
    }
    #trace-export-path {
        margin-top: 0;
    }
    #trace-export-actions {
        height: 3;
        min-height: 3;
        margin-top: 0;
        align-horizontal: right;
    }
    #trace-export-actions Button {
        min-width: 10;
        margin-left: 1;
    }
    #trace-export-status.-error {
        color: $error;
    }
    """

    def __init__(self, snapshot: TrajectorySnapshot) -> None:
        super().__init__()
        self._snapshot = snapshot
        self._selected_profile = TraceExportProfile.REDACTED_DIAGNOSTIC
        self._preflight: TraceExportPreflight | None = None
        self._writing = False

    @property
    def selected_profile(self) -> TraceExportProfile:
        return self._selected_profile

    def compose(self) -> ComposeResult:
        with Vertical(id="trace-export-dialog"):
            yield Static("Export shared Trace", id="trace-export-title", markup=False)
            yield Static(
                "Analyzing privacy inventory…", id="trace-export-inventory"
            )
            yield Static(
                "Credentials are always blocked in every profile.",
                id="trace-export-policy",
                markup=False,
            )
            yield Static(
                f"Profile: {TRACE_EXPORT_PROFILE_LABELS[self._selected_profile]}",
                id="trace-export-selection",
                markup=False,
            )
            with VerticalScroll(id="trace-export-body"):
                yield Static(
                    "Review exactly what will leave this machine before writing a portable JSON bundle.",
                    id="trace-export-intro",
                    markup=False,
                )
                with RadioSet(id="trace-export-profiles"):
                    yield RadioButton(
                        TRACE_EXPORT_PROFILE_LABELS[TraceExportProfile.SAFE_SUMMARY],
                        id="trace-export-profile-safe",
                    )
                    yield RadioButton(
                        TRACE_EXPORT_PROFILE_LABELS[
                            TraceExportProfile.REDACTED_DIAGNOSTIC
                        ],
                        id="trace-export-profile-redacted",
                        value=True,
                    )
                    yield RadioButton(
                        TRACE_EXPORT_PROFILE_LABELS[TraceExportProfile.FULL_TRACE],
                        id="trace-export-profile-full",
                    )
                yield Static("", id="trace-export-profile-copy", markup=False)
            yield Input(
                value=str(Path.cwd() / "trace-export.json"),
                placeholder="Destination .json path",
                id="trace-export-path",
            )
            yield Static("", id="trace-export-status", markup=False)
            with Horizontal(id="trace-export-actions"):
                yield Button("Browse…", id="trace-export-browse")
                yield Button("Cancel", id="trace-export-cancel")
                yield Button(
                    "Export",
                    id="trace-export-submit",
                    variant="primary",
                    disabled=True,
                )

    async def on_mount(self) -> None:
        super().on_mount()
        await self.select_profile(self._selected_profile)
        self.query_one("#trace-export-path", Input).focus()

    async def select_profile(self, profile: TraceExportProfile) -> None:
        """Recompute the single-pass preflight off the UI thread."""
        self._selected_profile = profile
        self.query_one("#trace-export-submit", Button).disabled = True
        self._update_profile_selection()
        self.query_one("#trace-export-profile-copy", Static).update(
            TRACE_EXPORT_PROFILE_COPY[profile]
        )
        self.query_one("#trace-export-inventory", Static).update(
            "Analyzing privacy inventory…"
        )
        try:
            preflight = await asyncio.to_thread(
                preflight_trace_export,
                self._snapshot,
                profile=profile,
            )
        except Exception as exc:  # noqa: BLE001 - projection boundary
            self._preflight = None
            self._set_status(f"Preflight failed: {exc}", error=True)
            return
        if profile is not self._selected_profile:
            return
        self._preflight = preflight
        inventory = preflight.privacy_inventory
        self.query_one("#trace-export-inventory", Static).update(
            f"{preflight.event_count} events · {inventory['sensitive']} sensitive fields · "
            f"{inventory['redacted']} redacted · {inventory['omitted']} omitted · "
            f"{inventory['truncated']} truncated · {inventory['missing']} unavailable"
        )
        self._set_status("")
        self.query_one("#trace-export-submit", Button).disabled = False

    def _update_profile_selection(self) -> None:
        """Keep the selected profile legible when its compact selector is hidden."""
        profiles = self.query_one("#trace-export-profiles", RadioSet)
        selection = self.query_one("#trace-export-selection", Static)
        selection.set_class(profiles.has_focus, "is-selector-focused")
        label = TRACE_EXPORT_PROFILE_LABELS[self._selected_profile]
        if profiles.has_focus:
            label = label.replace(" (recommended)", "")
            selection.update(f"Profile: {label} · ↑/↓ · Enter apply")
        else:
            selection.update(f"Profile: {label}")

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Expose a focus cue when the compact profile selector owns focus."""
        if event.widget.id == "trace-export-profiles":
            self._update_profile_selection()

    def on_descendant_blur(self, event: DescendantBlur) -> None:
        """Restore the normal selected-profile summary after selector blur."""
        if event.widget.id == "trace-export-profiles":
            self._update_profile_selection()

    @on(RadioSet.Changed, "#trace-export-profiles")
    async def _profile_changed(self, event: RadioSet.Changed) -> None:
        profile_by_id = {
            "trace-export-profile-safe": TraceExportProfile.SAFE_SUMMARY,
            "trace-export-profile-redacted": TraceExportProfile.REDACTED_DIAGNOSTIC,
            "trace-export-profile-full": TraceExportProfile.FULL_TRACE,
        }
        profile = profile_by_id.get(event.pressed.id or "")
        if profile is not None and profile is not self._selected_profile:
            await self.select_profile(profile)

    @on(Button.Pressed, "#trace-export-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#trace-export-browse")
    def _browse(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(
            self._browse_flow(), group="trace-export-browse", exclusive=True
        )

    async def _browse_flow(self) -> None:
        from tldw_chatbook.Third_Party.textual_fspicker import Filters
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave

        picker = EnhancedFileSave(
            title="Export shared Trace",
            default_filename="trace-export.json",
            filters=Filters(
                ("Trace files", lambda path: path.suffix.lower() == ".json"),
                ("All Files", lambda _path: True),
            ),
            context="trajectory_export",
        )
        selected = await self.app.push_screen_wait(picker)
        if selected is not None:
            self.query_one("#trace-export-path", Input).value = str(selected)

    async def _confirm_full_export(self) -> bool:
        return bool(
            await self.app.push_screen_wait(
                full_trace_confirmation(noun="Trace")
            )
        )

    @on(Button.Pressed, "#trace-export-submit")
    def _export(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(self._export_flow(), group="trace-export-write", exclusive=True)

    @on(Input.Submitted, "#trace-export-path")
    def _submit_path(self, event: Input.Submitted) -> None:
        event.stop()
        self.run_worker(self._export_flow(), group="trace-export-write", exclusive=True)

    async def _confirm_overwrite(self, destination: Path) -> bool:
        return bool(
            await self.app.push_screen_wait(
                ConfirmationDialog(
                    title="Replace existing Trace?",
                    message=(
                        f"{destination.name} already exists. Replacing it cannot be undone."
                    ),
                    confirm_label="Replace file",
                    cancel_label="Keep existing",
                )
            )
        )

    async def _export_flow(self) -> None:
        if self._writing or self._preflight is None:
            return
        raw_path = self.query_one("#trace-export-path", Input).value.strip()
        if not raw_path:
            self._set_status("Choose a destination path.", error=True)
            return
        if self._selected_profile is TraceExportProfile.FULL_TRACE:
            if not await self._confirm_full_export():
                self._set_status("Full export cancelled; review the profile or path.")
                return
        try:
            destination = validate_path_simple(raw_path, require_exists=False)
        except ValueError as exc:
            self._set_status(f"Invalid destination: {exc}", error=True)
            return
        if destination.exists() and not await self._confirm_overwrite(destination):
            self._set_status("Export cancelled; the existing file was kept.")
            return
        self._writing = True
        self._set_controls_disabled(True)
        self._set_status("Writing privacy-governed bundle…")
        try:
            payload = await asyncio.to_thread(
                build_trace_export,
                self._snapshot,
                preflight=self._preflight,
                confirm_full=(self._selected_profile is TraceExportProfile.FULL_TRACE),
            )
            written = await asyncio.to_thread(
                write_trajectory_export, destination, payload
            )
        except Exception as exc:  # noqa: BLE001 - actionable write boundary
            self._writing = False
            self._set_controls_disabled(False)
            self._set_status(f"Export failed: {exc}", error=True)
            return
        self._writing = False
        self.dismiss(written)

    def _set_controls_disabled(self, disabled: bool) -> None:
        for selector in (
            "#trace-export-submit",
            "#trace-export-browse",
            "#trace-export-cancel",
        ):
            self.query_one(selector, Button).disabled = disabled
        self.query_one("#trace-export-path", Input).disabled = disabled

    def _set_status(self, message: str, *, error: bool = False) -> None:
        status = self.query_one("#trace-export-status", Static)
        status.set_class(error, "-error")
        status.update(message)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if self._writing:
            self._set_status("Export is finishing; the destination is still protected.")
            return
        self.dismiss_safe_once(None)
