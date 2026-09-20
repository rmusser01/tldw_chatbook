"""Library presentation for the application-owned Chatbook sharing session."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, SelectionList, Static

if TYPE_CHECKING:
    from ...Library.library_artifacts_state import ArtifactKey
    from ...Web_Server.artifact_share import ArtifactShareController, ShareStatus
    from ..Screens.artifact_share_dialog import ArtifactShareDialog
    from ..Screens.library_screen import LibraryScreen


class _ShareStrip(Horizontal):
    """Keep Manage and Stop independent of the selected Library destination."""

    def __init__(self, owner: LibraryArtifactsShareController) -> None:
        super().__init__(
            id="library-artifacts-share-strip",
            classes="ds-toolbar library-artifacts-share-strip",
        )
        self.owner = owner
        self.display = False

    def compose(self) -> ComposeResult:
        yield Static(
            "",
            id="library-artifacts-share-status",
            classes="status-label",
            markup=False,
        )
        yield Button(
            "Manage", id="library-artifacts-share-manage", classes="action-button"
        )
        yield Button("Stop", id="library-artifacts-share-stop", classes="action-button")

    def on_mount(self) -> None:
        self.owner.refresh_status()

    @on(Button.Pressed, "#library-artifacts-share-manage")
    def manage(self, event: Button.Pressed) -> None:
        event.stop()
        self.owner.open_dialog()

    @on(Button.Pressed, "#library-artifacts-share-stop")
    def stop(self, event: Button.Pressed) -> None:
        event.stop()
        self.owner.stop_share()


class LibraryArtifactsShareController:
    """Fence pending dialogs by visit; keep accepted sharing application-owned."""

    def __init__(self, screen: LibraryScreen) -> None:
        self.screen = screen
        self._owner = screen.app_instance
        self._generation = 0
        self._status_generation = 0
        self._suspended = False
        self._disposed = False
        self._dialog: ArtifactShareDialog | None = None
        self._observed: ArtifactShareController | None = None
        self._strip: _ShareStrip | None = None

    def build_strip(self) -> Horizontal:
        """Build the Library-wide status strip outside route-owned readers."""
        self._strip = _ShareStrip(self)
        return self._strip

    def _profile(self) -> tuple[Any, ...]:
        return (
            id(getattr(self._owner, "app_config", None)),
            getattr(self._owner, "notes_user_id", None),
            id(getattr(self._owner, "local_chatbook_service", None)),
        )

    def _active(self) -> bool:
        return (
            not self._disposed
            and not self._suspended
            and self.screen.is_mounted
            and self.screen.app.screen is self.screen
        )

    def _presentable(self, generation: int, profile: tuple[Any, ...]) -> bool:
        return (
            self._active()
            and generation == self._generation
            and profile == self._profile()
        )

    def open_dialog(self, preselected_key: ArtifactKey | None = None) -> None:
        """Prepare the complete eligible local inventory for explicit consent."""
        from ...Web_Server import is_web_server_available

        if not self._active():
            return
        self.invalidate_pending_presentation()
        if not is_web_server_available():
            self.screen.notify(
                "Install tldw_chatbook[web] to share Chatbooks.", severity="warning"
            )
            return
        service = getattr(self._owner, "local_chatbook_service", None)
        if service is None:
            self.screen.notify("Local Chatbooks are unavailable.", severity="warning")
            return
        controller = self._owner._get_artifact_share_controller()
        self.refresh_status()
        self.screen.run_worker(
            self._prepare_dialog(
                self._generation, self._profile(), service, controller, preselected_key
            ),
            group="library-artifacts-share-listing",
            exclusive=True,
            exit_on_error=False,
        )

    @staticmethod
    def _eligible_records(service: Any) -> list[dict[str, Any]]:
        from ...Chatbooks.artifact_registry_snapshot import usable_chatbook_bundle

        snapshot = service.artifact_read_snapshot()
        return [
            record
            for record in snapshot.iter_records()
            if usable_chatbook_bundle(record.get("file_path"))[0]
        ]

    async def _prepare_dialog(
        self,
        generation: int,
        profile: tuple[Any, ...],
        service: Any,
        controller: ArtifactShareController,
        preselected_key: ArtifactKey | None,
    ) -> None:
        from ...Backup_Recovery.participants import run_finite_local_worker
        from ..Screens.artifact_share_dialog import ArtifactShareDialog

        try:
            records = await asyncio.to_thread(
                run_finite_local_worker, self._eligible_records, service
            )
            status = await asyncio.to_thread(lambda: controller.status)
        except Exception:  # noqa: BLE001 - owner failure must not tear down Library
            logger.warning("Library Chatbook share listing failed")
            if self._presentable(generation, profile):
                self.screen.notify(
                    "Could not load local Chatbooks for sharing.", severity="error"
                )
            return
        # This check and push run together on the UI thread, with no await gap.
        if not self._presentable(generation, profile):
            return
        if getattr(self._owner, "artifact_share_controller", None) is not controller:
            return
        notice = (
            f"Sharing {status.artifact_count} Chatbooks. Starting a new share replaces the current share."
            if status
            else None
        )
        dialog = ArtifactShareDialog(records, active_share_notice=notice)
        # The modal itself suspends Library: result ownership is distinct from
        # the now-invalid pending presentation generation.
        self._dialog = dialog
        self.screen.app.push_screen(
            dialog,
            lambda result: self._accept_dialog(dialog, profile, controller, result),
        )
        if preselected_key is not None and preselected_key.source == "chatbook":
            selected = next(
                (
                    str(record["id"])
                    for record in records
                    if int(record.get("chatbook_id") or record["id"])
                    == preselected_key.native_id
                ),
                None,
            )
            if selected is not None:
                dialog.call_after_refresh(
                    lambda: dialog.query_one(
                        "#share-artifact-list", SelectionList
                    ).select(selected)
                )

    def _accept_dialog(
        self,
        dialog: ArtifactShareDialog,
        profile: tuple[Any, ...],
        controller: ArtifactShareController,
        result: dict[str, Any] | None,
    ) -> None:
        if self._dialog is not dialog:
            return
        self._dialog = None
        if (
            self._disposed
            or profile != self._profile()
            or getattr(self._owner, "artifact_share_controller", None) is not controller
            or result is None
        ):
            return
        options = dict(result)
        options["records"] = options.pop("selected_records")
        # An approved operation outlives this screen and its Textual workers.
        self.screen.app.run_worker(
            self._start_share(controller, options, profile),
            group="library-artifacts-share-action",
            exit_on_error=False,
        )

    async def _start_share(
        self,
        controller: ArtifactShareController,
        options: dict[str, Any],
        profile: tuple[Any, ...],
    ) -> None:
        from ...Backup_Recovery.participants import run_finite_local_worker

        def start_if_current() -> None:
            # A queued app worker may begin after its profile/owner was retired.
            # Screen disposal alone does not revoke an already-approved action.
            if (
                profile != self._profile()
                or getattr(self._owner, "artifact_share_controller", None)
                is not controller
                or getattr(self._owner, "_shutting_down", False)
                or getattr(self._owner, "_closing", False)
            ):
                return
            controller.start_share(**options)

        try:
            await asyncio.to_thread(run_finite_local_worker, start_if_current)
        except Exception:  # noqa: BLE001 - the existing owner reports staging failures
            logger.warning("Library Chatbook share could not start")
            if self._active() and profile == self._profile():
                self.screen.notify(
                    "Could not start sharing. Check the selected exported files and share settings.",
                    severity="error",
                )

    def refresh_status(self) -> None:
        """Observe an existing owner and project status without inventory reads."""
        if self._disposed:
            return
        controller = getattr(self._owner, "artifact_share_controller", None)
        if controller is not self._observed:
            if self._observed is not None:
                self._observed.remove_status_listener(self._status_changed)
            self._observed = controller
            if controller is not None:
                controller.add_status_listener(self._status_changed)
        self._status_generation += 1
        if controller is None:
            self._render_status(None)
        elif self.screen.is_mounted:
            self.screen.run_worker(
                self._read_status(controller, self._status_generation),
                group="library-artifacts-share-status",
                exclusive=True,
                exit_on_error=False,
            )

    async def _read_status(
        self, controller: ArtifactShareController, generation: int
    ) -> None:
        # The lifecycle owner holds its lock while staging: keep that wait off UI.
        status = await asyncio.to_thread(lambda: controller.status)
        if (
            not self._disposed
            and generation == self._status_generation
            and controller is getattr(self._owner, "artifact_share_controller", None)
        ):
            self._render_status(status)

    def _status_changed(self, status: ShareStatus | None) -> None:
        # call_later posts to Textual rather than waiting for the UI while the
        # sharing owner may still hold its lifecycle lock on a worker thread.
        if not self._disposed:
            self.screen.call_later(self.refresh_status)

    def _render_status(self, status: ShareStatus | None) -> None:
        if self._strip is None or not self._strip.is_mounted:
            return
        self._strip.display = status is not None
        description = ""
        if status is not None:
            description = " · ".join(
                (f"Sharing {status.artifact_count} Chatbooks", *status.urls)
            )
        self._strip.query_one("#library-artifacts-share-status", Static).update(
            description
        )

    def stop_share(self) -> None:
        """Stop the app's current share without changing Library selection."""
        from ...Backup_Recovery.participants import run_finite_local_worker

        controller = getattr(self._owner, "artifact_share_controller", None)
        if not self._active() or controller is None:
            return
        self.screen.app.run_worker(
            asyncio.to_thread(run_finite_local_worker, controller.stop_share),
            group="library-artifacts-share-action",
            exit_on_error=False,
        )

    def invalidate_pending_presentation(self) -> None:
        """Discard a pending listing after canvas navigation or a newer request."""
        self._generation += 1

    def suspend(self) -> None:
        """Retire a visit without revoking consent in an already-presented dialog."""
        self._suspended = True
        self.invalidate_pending_presentation()

    def resume(self) -> None:
        """Reconcile the running session without replaying a prior dialog open."""
        self._suspended = False
        self.refresh_status()

    def dispose(self) -> None:
        """Detach presentation only; the application still owns its running share."""
        self._disposed = True
        self.invalidate_pending_presentation()
        self._dialog = None
        if self._observed is not None:
            self._observed.remove_status_listener(self._status_changed)
            self._observed = None
