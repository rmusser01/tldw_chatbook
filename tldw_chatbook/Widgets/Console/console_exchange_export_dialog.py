"""Governed Clipboard/File disclosure for one immutable exchange capture."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Literal

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, RadioButton, RadioSet, Static

from tldw_chatbook.Chat.console_exchange_capture import ExchangeCapture
from tldw_chatbook.Chat.console_exchange_export import (
    ExchangeExportProjection,
    ExchangeExportUnavailable,
    project_exchange_export,
)
from tldw_chatbook.Chat.trajectory_export import TraceExportProfile
from tldw_chatbook.Utils.atomic_file_ops import atomic_write_text
from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Widgets.Console.trace_export_dialog import (
    TRACE_EXPORT_PROFILE_COPY,
    TRACE_EXPORT_PROFILE_LABELS,
    full_trace_confirmation,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

ExportDestination = Literal["clipboard", "file"]

__all__ = ["ConsoleExchangeExportDialog"]


class ConsoleExchangeExportDialog(SafeModalDismissMixin, ModalScreen[None]):
    """Choose one profile and destination without changing capture ownership."""

    SAFE_MODAL_CONTENT = "#exchange-export-dialog"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]

    BUNDLED_SCREEN_CSS = """
    ConsoleExchangeExportDialog { align: center middle; }
    #exchange-export-dialog {
        width: 76;
        max-width: 96%;
        height: 22;
        max-height: 96%;
        border: tall $primary;
        background: $surface;
        padding: 1 2;
    }
    #exchange-export-title { height: 1; text-style: bold; }
    #exchange-export-body { height: 1fr; scrollbar-gutter: stable; }
    #exchange-export-profiles,
    #exchange-export-destinations { height: auto; margin-top: 1; }
    #exchange-export-profile-copy,
    #exchange-export-full-reason,
    #exchange-export-status { height: auto; }
    #exchange-export-full-reason { color: $warning; }
    #exchange-export-path { height: 3; }
    #exchange-export-actions {
        height: 3;
        min-height: 3;
        align-horizontal: right;
    }
    #exchange-export-actions Button { min-width: 10; margin-left: 1; }
    #exchange-export-status.-error { color: $error; }
    """

    def __init__(
        self,
        capture: ExchangeCapture,
        *,
        expected_capture_revision: int,
        capture_revision_provider: Callable[[], int],
    ) -> None:
        super().__init__()
        self._capture = capture
        self._expected_capture_revision = expected_capture_revision
        self._capture_revision_provider = capture_revision_provider
        self._selected_profile = TraceExportProfile.REDACTED_DIAGNOSTIC
        self._destination: ExportDestination = "clipboard"
        self._projection: ExchangeExportProjection | None = None
        self._exporting = False

    @property
    def selected_profile(self) -> TraceExportProfile:
        return self._selected_profile

    def compose(self) -> ComposeResult:
        full_available = self._capture.capture_detail.value == "full"
        with Vertical(id="exchange-export-dialog"):
            yield Static("Export Exchange call", id="exchange-export-title", markup=False)
            with VerticalScroll(id="exchange-export-body"):
                yield Static(
                    "Choose what leaves this machine. Credentials stay structurally blocked.",
                    markup=False,
                )
                with RadioSet(id="exchange-export-profiles"):
                    yield RadioButton(
                        TRACE_EXPORT_PROFILE_LABELS[TraceExportProfile.SAFE_SUMMARY],
                        id="exchange-export-profile-safe",
                    )
                    yield RadioButton(
                        TRACE_EXPORT_PROFILE_LABELS[
                            TraceExportProfile.REDACTED_DIAGNOSTIC
                        ],
                        id="exchange-export-profile-redacted",
                        value=True,
                    )
                    yield RadioButton(
                        TRACE_EXPORT_PROFILE_LABELS[TraceExportProfile.FULL_TRACE],
                        id="exchange-export-profile-full",
                        disabled=not full_available,
                    )
                yield Static(
                    TRACE_EXPORT_PROFILE_COPY[self._selected_profile],
                    id="exchange-export-profile-copy",
                    markup=False,
                )
                yield Static(
                    ""
                    if full_available
                    else "Full trace unavailable: this call was captured in Safe mode.",
                    id="exchange-export-full-reason",
                    markup=False,
                )
                with RadioSet(id="exchange-export-destinations"):
                    yield RadioButton(
                        "Clipboard", id="exchange-export-destination-clipboard", value=True
                    )
                    yield RadioButton("File", id="exchange-export-destination-file")
                path = Input(
                    value=str(Path.cwd() / "exchange-export.json"),
                    placeholder="Destination .json path",
                    id="exchange-export-path",
                )
                path.display = False
                yield path
            yield Static("Ready", id="exchange-export-status", markup=False)
            with Horizontal(id="exchange-export-actions"):
                yield Button("Cancel", id="exchange-export-cancel")
                yield Button(
                    "Export", id="exchange-export-submit", variant="primary"
                )

    def on_mount(self) -> None:
        self.query_one("#exchange-export-submit", Button).focus()

    async def select_profile(self, profile: TraceExportProfile) -> None:
        """Select an available shared profile and refresh its explanation."""
        if (
            profile is TraceExportProfile.FULL_TRACE
            and self._capture.capture_detail.value != "full"
        ):
            self._set_status(
                "Full trace unavailable: this call was captured in Safe mode.",
                error=True,
            )
            return
        self._selected_profile = profile
        self._projection = None
        self.query_one("#exchange-export-profile-copy", Static).update(
            TRACE_EXPORT_PROFILE_COPY[profile]
        )
        for candidate, selector in (
            (TraceExportProfile.SAFE_SUMMARY, "#exchange-export-profile-safe"),
            (
                TraceExportProfile.REDACTED_DIAGNOSTIC,
                "#exchange-export-profile-redacted",
            ),
            (TraceExportProfile.FULL_TRACE, "#exchange-export-profile-full"),
        ):
            self.query_one(selector, RadioButton).value = candidate is profile

    async def select_destination(self, destination: ExportDestination) -> None:
        """Select Clipboard or File without producing a disclosure."""
        if destination not in {"clipboard", "file"}:
            raise ValueError("unsupported export destination")
        self._destination = destination
        self.query_one("#exchange-export-path", Input).display = destination == "file"
        self.query_one(
            "#exchange-export-destination-clipboard", RadioButton
        ).value = destination == "clipboard"
        self.query_one("#exchange-export-destination-file", RadioButton).value = (
            destination == "file"
        )

    @on(RadioSet.Changed, "#exchange-export-profiles")
    async def _profile_changed(self, event: RadioSet.Changed) -> None:
        profile = {
            "exchange-export-profile-safe": TraceExportProfile.SAFE_SUMMARY,
            "exchange-export-profile-redacted": TraceExportProfile.REDACTED_DIAGNOSTIC,
            "exchange-export-profile-full": TraceExportProfile.FULL_TRACE,
        }.get(event.pressed.id or "")
        if profile is not None:
            await self.select_profile(profile)

    @on(RadioSet.Changed, "#exchange-export-destinations")
    async def _destination_changed(self, event: RadioSet.Changed) -> None:
        destination: ExportDestination = (
            "file"
            if event.pressed.id == "exchange-export-destination-file"
            else "clipboard"
        )
        await self.select_destination(destination)

    @on(Button.Pressed, "#exchange-export-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#exchange-export-submit")
    def _export(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(
            self.export_selected(), group="exchange-export", exclusive=True
        )

    @on(Input.Submitted, "#exchange-export-path")
    def _submit_path(self, event: Input.Submitted) -> None:
        event.stop()
        self.run_worker(
            self.export_selected(), group="exchange-export", exclusive=True
        )

    def _revision_is_current(self) -> bool:
        try:
            current = self._capture_revision_provider()
        except Exception:
            current = None
        if current == self._expected_capture_revision:
            return True
        self._projection = None
        self._set_status("Stored captures changed · reopen Export.", error=True)
        return False

    def _project(self, profile: TraceExportProfile) -> ExchangeExportProjection:
        return project_exchange_export(self._capture, profile)

    async def _project_async(
        self, profile: TraceExportProfile
    ) -> ExchangeExportProjection:
        return await asyncio.to_thread(self._project, profile)

    async def _confirm_full_export(self) -> bool:
        return bool(
            await self.app.push_screen_wait(full_trace_confirmation(noun="Exchange"))
        )

    async def _confirm_overwrite(self, destination: Path) -> bool:
        return bool(
            await self.app.push_screen_wait(
                ConfirmationDialog(
                    title="Replace existing exchange export?",
                    message=(
                        f"{destination.name} already exists. Replacing it cannot be undone."
                    ),
                    confirm_label="Replace file",
                    cancel_label="Keep existing",
                )
            )
        )

    async def export_selected(self) -> bool:
        """Confirm, project, revalidate, then disclose one fresh projection."""
        if self._exporting or not self._revision_is_current():
            return False
        if self._selected_profile is TraceExportProfile.FULL_TRACE:
            if not await self._confirm_full_export():
                self._set_status("Full export cancelled.")
                return False

        destination: Path | None = None
        destination_existed = False
        if self._destination == "file":
            raw_path = self.query_one("#exchange-export-path", Input).value.strip()
            if not raw_path:
                self._set_status("Choose a destination path.", error=True)
                return False
            try:
                destination = validate_path_simple(raw_path, require_exists=False)
            except ValueError:
                self._set_status("Invalid destination path.", error=True)
                return False
            destination_existed = destination.exists()
            if destination_existed and not await self._confirm_overwrite(destination):
                self._set_status("Export cancelled; the existing file was kept.")
                return False

        # Confirmations yield to the app. Fence the immutable capture again at
        # the last possible moment before projection begins.
        if not self._revision_is_current():
            return False

        self._exporting = True
        self._set_controls_disabled(True)
        self._set_status("Preparing governed export…")
        try:
            projection = await self._project_async(self._selected_profile)
            self._projection = projection
            if not self._revision_is_current():
                return False
            if destination is None:
                self.app.copy_to_clipboard(projection.json_text)
                self._set_status("Export copied to clipboard.")
            else:
                await asyncio.to_thread(
                    atomic_write_text,
                    destination,
                    projection.json_text,
                    mode=0o600,
                    privacy_safe_log=True,
                    overwrite=destination_existed,
                )
                self._set_status(f"Export written to {destination.name}.")
            return True
        except FileExistsError:
            self._projection = None
            self._set_status(
                "Export cancelled; a file appeared at the destination and was kept.",
                error=True,
            )
            return False
        except ExchangeExportUnavailable:
            self._projection = None
            self._set_status("Full trace is unavailable for this capture.", error=True)
            return False
        except Exception as exc:  # noqa: BLE001 - content-free disclosure boundary
            self._projection = None
            self._set_status(
                f"Export failed ({type(exc).__name__}).", error=True
            )
            return False
        finally:
            self._exporting = False
            self._set_controls_disabled(False)

    def _set_controls_disabled(self, disabled: bool) -> None:
        self.query_one("#exchange-export-submit", Button).disabled = disabled
        self.query_one("#exchange-export-cancel", Button).disabled = disabled
        self.query_one("#exchange-export-path", Input).disabled = disabled
        self.query_one("#exchange-export-profiles", RadioSet).disabled = disabled
        self.query_one("#exchange-export-destinations", RadioSet).disabled = disabled

    def _set_status(self, message: str, *, error: bool = False) -> None:
        status = self.query_one("#exchange-export-status", Static)
        status.set_class(error, "-error")
        status.update(message)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if self._exporting:
            self._set_status("Export is finishing; the destination remains protected.")
            return
        self.dismiss_safe_once(None)
