"""Workspace-scoped Studio generation, history, and recovery controls."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.events import DescendantFocus
from textual.message import Message
from textual.widgets import Button, Input, Select, Static

from ...Research_Workspace.contracts import CapabilityUnavailableError
from ...Research_Workspace.egress_consent import ResearchEgressConsentService
from ...Research_Workspace.layout_state import ResearchPanePreferences
from ...Research_Workspace.overlay_store import OverlayConflictError
from ...Research_Workspace.studio_mapping import (
    PRIMARY_OUTPUT_MAPPINGS,
    project_studio_availability,
)
from ...Research_Workspace.studio_models import (
    ResearchOutputKind,
    ResearchStudioRequest,
    StudioOwnerKind,
)
from ...Widgets.confirmation_dialog import ConfirmationDialog
from .chat_region import ResearchChatRegion
from .quick_notes_section import ResearchQuickNotesSection
from .studio_dialogs import ResearchStudioActions, ResearchStudioOptions
from .studio_history import ResearchStudioHistory, StudioHistoryRow, history_key
from .studio_output_card import ResearchStudioOutputCard, output_label


class ResearchStudioRegion(VerticalScroll):
    """Keep one in-flight operation and a bounded page of canonical references."""

    can_focus = True

    class OwnerAction(Message):
        def __init__(self, action: str, row: StudioHistoryRow) -> None:
            super().__init__()
            self.action = action
            self.row = row

    def __init__(self, *, controller, overlay_store=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.controller = controller
        self.overlay_store = overlay_store
        self.ref = None
        self._availability = {}
        self._capabilities = {}
        self._offset = 0
        self._refresh_revision = 0
        self._cancel = None
        self._retry = None
        self._last_saved = None
        self._dialog_open = False

    def compose(self) -> ComposeResult:
        yield Static(
            "Studio", id="research-studio-heading", classes="research-pane-title"
        )
        yield Button("Quick Notes", id="research-studio-quick-notes", compact=True)
        yield Input(placeholder="Provider", id="research-studio-provider")
        yield Input(placeholder="Model", id="research-studio-model")
        yield Button(
            "Check processing route",
            id="research-studio-check-route",
            compact=True,
            disabled=True,
        )
        yield Static("Select a workspace.", id="research-studio-route", markup=False)
        yield Static("", id="research-studio-status", markup=False)
        yield Button(
            "Stop generation", id="research-studio-stop", compact=True, disabled=True
        )
        yield Button(
            "Retry output", id="research-studio-retry", compact=True, disabled=True
        )
        yield Button(
            "Reopen saved output",
            id="research-studio-reopen-last",
            compact=True,
            disabled=True,
        )
        for kind in ResearchOutputKind:
            yield ResearchStudioOutputCard(kind)
        yield ResearchStudioHistory()
        yield ResearchQuickNotesSection(id="research-quick-notes-section")

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Keep keyboard-selected Studio actions inside the scrolled viewport."""

        event.widget.scroll_visible(animate=False, immediate=True, force=True)

    @property
    def history(self) -> ResearchStudioHistory:
        return self.query_one(ResearchStudioHistory)

    @property
    def is_busy(self) -> bool:
        return self._cancel is not None

    def allow_leave(self) -> bool:
        if self.is_busy:
            self._status(
                "Stop the active Studio generation before leaving this workspace."
            )
            self.notify(
                "Stop the active Studio generation before leaving this workspace.",
                severity="warning",
            )
            return False
        return True

    def set_workspace(self, ref) -> None:
        if ref == self.ref:
            return
        self.ref = ref
        self._availability = {}
        self._capabilities = {}
        self._retry = self._last_saved = None
        self._offset = 0
        self.history.sync_page(())
        for field in ("provider", "model"):
            self.query_one(f"#research-studio-{field}", Input).value = ""
        self._status("")
        self.query_one("#research-studio-route", Static).update(self._route_label(None))
        self._sync_controls()

    def _status(self, text: str) -> None:
        self.query_one("#research-studio-status", Static).update(text)

    def _sync_controls(self) -> None:
        for card in self.query(ResearchStudioOutputCard):
            card.sync_availability(
                self._availability.get(card.kind),
                busy=self.is_busy or self._dialog_open,
            )
        for key, enabled in (
            ("check-route", self.ref is not None and not self.is_busy),
            ("stop", self.is_busy and not self._cancel.is_set()),
            (
                "retry",
                self._retry is not None and not self.is_busy and not self._dialog_open,
            ),
            ("reopen-last", self._last_saved is not None and not self.is_busy),
        ):
            self.query_one(f"#research-studio-{key}", Button).disabled = not enabled

    def _port(self, ref):
        return self.controller.port_for_data_source(ref.data_source)

    def forget_output(self, output) -> None:
        if self._last_saved and history_key(self._last_saved.output) == history_key(
            output
        ):
            self._last_saved = None
            self._sync_controls()

    @on(Button.Pressed, "#research-studio-quick-notes")
    def jump_to_quick_notes(self) -> None:
        section = self.query_one(ResearchQuickNotesSection)
        section.scroll_visible(animate=False, top=True, immediate=True, force=True)
        title = section.query_one("#research-quick-note-title", Input)
        if not title.disabled:
            title.focus()

    def _route_fields(self) -> tuple[str, str]:
        provider = self.query_one("#research-studio-provider", Input).value.strip()
        model = self.query_one("#research-studio-model", Input).value.strip()
        return provider, model

    @staticmethod
    def _route_label(route) -> str:
        if route is None:
            return "Choose a provider and model to check the processing route."
        method = route.endpoint_class.replace("_", " ") or route.processor
        return f"{method} · {route.provider or 'server default'} / {route.model}"

    def _current(self, capture) -> bool:
        return (
            self.is_mounted
            and self.ref == capture.ref
            and self.controller.is_current_request(capture)
        )

    def refresh_studio(self, *, preferred="") -> None:
        if self.ref is None or self.controller.selected_ref != self.ref:
            return
        self.run_worker(
            self._refresh(preferred=preferred),
            group="research-studio-view",
            exclusive=True,
            exit_on_error=False,
        )

    async def _refresh(self, *, preferred="") -> None:
        capture = self.controller.capture_request()
        port = self._port(capture.ref)
        self._refresh_revision += 1
        revision = self._refresh_revision
        if port is None or not hasattr(port, "capture_chat_sources"):
            return
        capabilities = self.controller.visible_capabilities
        sources, route = None, None
        try:
            if (
                studio_capabilities := getattr(port, "studio_capabilities", None)
            ) is not None:
                capabilities = await studio_capabilities(capture.ref)
            sources = await port.capture_chat_sources(capture.ref)
            route = await port.resolve_chat_route(capture.ref, *self._route_fields())
        except (CapabilityUnavailableError, ValueError, RuntimeError) as exc:
            if self._current(capture):
                logger.debug(
                    "Research Studio readiness failed (error_type={})",
                    type(exc).__name__,
                )
        if not self._current(capture) or revision != self._refresh_revision:
            return
        self._capabilities = capabilities
        self._availability = project_studio_availability(
            capture.ref, sources or (), route, capabilities
        )
        self.query_one("#research-studio-route", Static).update(
            self._route_label(route)
        )
        self._sync_controls()
        if not hasattr(port, "list_outputs"):
            return
        try:
            list_records = getattr(port, "list_output_records", None)
            if callable(list_records):
                page, records = await list_records(capture.ref, offset=self._offset)
            else:
                page, records = (
                    await port.list_outputs(capture.ref, offset=self._offset),
                    None,
                )
            current_sources = {row.source_id: row for row in sources or ()}
            rows = []
            for output in page.items:
                if output.workspace != capture.ref:
                    raise ValueError("Output history belongs to a different workspace")
                record = (
                    records[output]
                    if records is not None
                    else await port.resolve_output(output)
                )
                changed = sources is not None and any(
                    current_sources.get(source.source_id) != source
                    for source in output.source_snapshot
                )
                rows.append(StudioHistoryRow(output, record, changed))
            if self._current(capture) and self._refresh_revision == revision:
                self.history.sync_page(
                    rows,
                    preferred=preferred,
                    offset=self._offset,
                    has_more=page.has_more,
                )
        except Exception as exc:
            if self._current(capture) and self._refresh_revision == revision:
                logger.debug(
                    "Research Studio history failed (error_type={})", type(exc).__name__
                )
                self.history.sync_page(())
                self.history.query_one(
                    "#research-studio-history-detail", Static
                ).update(
                    "Output history is unavailable. Check the selected owner and retry."
                )

    @on(Button.Pressed, "#research-studio-check-route")
    def check_route(self) -> None:
        self.refresh_studio()

    @on(Button.Pressed, ".research-studio-create")
    def create(self, event: Button.Pressed) -> None:
        self._configure(event.button.parent.kind)

    def _configure(self, kind, *, replace_output=None, supersedes=None) -> None:
        if (
            self.is_busy
            or self._dialog_open
            or not self._availability.get(kind)
            or not self._availability[kind].available
        ):
            return
        self._dialog_open = True
        self._sync_controls()
        self.run_worker(
            self._configure_and_generate(
                kind, replace_output=replace_output, supersedes=supersedes
            ),
            group="research-studio-generation",
            exit_on_error=False,
        )

    async def _configure_and_generate(
        self, kind, *, replace_output, supersedes
    ) -> None:
        try:
            capture = self.controller.capture_request()
            port = self._port(capture.ref)
            sources = await port.capture_chat_sources(capture.ref)
            route = await port.resolve_chat_route(capture.ref, *self._route_fields())
            if not self._current(capture):
                return
            mapping = PRIMARY_OUTPUT_MAPPINGS[capture.ref.data_source, kind]
            action = (
                "Replace output"
                if replace_output
                else "Generate new version"
                if supersedes
                else "Generate"
            )
            options = await self.app.push_screen_wait(
                ResearchStudioOptions(
                    kind,
                    mapping.owner_label,
                    self._route_label(route),
                    len(sources),
                    action=action,
                )
            )
            if options is None or not self._current(capture):
                return
            request = ResearchStudioRequest(
                uuid4().hex,
                capture.ref,
                kind,
                sources,
                options,
                route,
                capture.context_revision,
            )
            if not await self._approve_route(request) or not self._current(capture):
                return
            self._retry = (request, replace_output, supersedes)
            await self._generate(capture, request, replace_output, supersedes)
        except Exception as exc:
            self._show_error(exc)
        finally:
            self._dialog_open = False
            if self.is_mounted:
                self._sync_controls()

    async def _approve_route(self, request) -> bool:
        store = self.overlay_store
        consent = ResearchEgressConsentService()
        previous = (
            await asyncio.to_thread(store.load, request.workspace) if store else None
        )
        preflight = consent.preflight(
            request, previous.chat_egress_consents if previous else ()
        )
        if preflight is None:
            return True
        if store is None:
            raise ValueError("Device storage is required to approve remote processing.")
        confirm = ConfirmationDialog(
            title="Send Local source evidence?",
            message=f"Send {preflight.source_body_mode} from {preflight.source_count} selected sources to {preflight.provider} on the {preflight.endpoint_class.replace('_', ' ')}. Redaction policy is {preflight.redaction_policy}.",
            confirm_label="Continue",
            cancel_label="Cancel",
        )
        if await self.app.push_screen_wait(confirm) is not True:
            return False

        def save():
            for _ in range(3):
                current = store.load(request.workspace)
                try:
                    return store.save(
                        request.workspace,
                        current.preferences if current else ResearchPanePreferences(),
                        expected_revision=current.revision if current else 0,
                        chat_egress_consents=consent.approve(
                            preflight, current.chat_egress_consents if current else ()
                        ),
                    )
                except OverlayConflictError:
                    continue
            raise ValueError(
                "The route approval changed in another session. Retry after refreshing."
            )

        saved = await asyncio.to_thread(save)
        self.post_message(ResearchChatRegion.OverlayUpdated(saved))
        return True

    async def _generate(self, capture, request, replace_output, supersedes) -> None:
        self._cancel = asyncio.Event()
        self._sync_controls()
        label = output_label(request.output_kind)
        route = self._route_label(request.processing_route)
        self._status(f"Generating {label} · {route}")
        try:
            port = self._port(request.workspace)
            output = await port.generate_output(
                request,
                lambda _chunk: None,
                self._cancel,
                replace_output=replace_output,
                supersedes=supersedes,
            )
            if not self._current(capture):
                return
            if output.status == "complete":
                record = await port.resolve_output(output)
                if not self._current(capture):
                    return
                if not record:
                    raise ValueError(
                        "The canonical owner could not reopen the saved result."
                    )
                self._last_saved = StudioHistoryRow(output, record, False)
                self._retry = None
                self._offset = 0
                owner = PRIMARY_OUTPUT_MAPPINGS[
                    request.workspace.data_source, request.output_kind
                ].owner_label
                self._status(f"Saved to {owner} · {route}")
                self.refresh_studio(preferred=history_key(output))
            else:
                self._status(f"{label} generation stopped. Retry is available.")
        except Exception as exc:
            if self._current(capture):
                self._show_error(exc)
        finally:
            self._cancel = None
            if self.is_mounted:
                self._sync_controls()

    def _show_error(self, exc) -> None:
        logger.warning(
            "Research Studio action failed (error_type={})", type(exc).__name__
        )
        if isinstance(exc, CapabilityUnavailableError):
            self._status(
                f"{exc.capability.user_message} {exc.capability.recovery_action}".strip()
            )
        elif isinstance(exc, ValueError):
            self._status(str(exc))
        else:
            self._status(
                "Studio could not complete the action. Check the selected owner and retry."
            )

    @on(Button.Pressed, "#research-studio-stop")
    def stop(self) -> None:
        if self._cancel:
            self._cancel.set()
            self._status("Stopping generation…")
            self._sync_controls()

    @on(Button.Pressed, "#research-studio-retry")
    def retry(self) -> None:
        if self._retry is None or self.is_busy or self._dialog_open:
            return
        capture = self.controller.capture_request()
        request, target, previous = self._retry
        if not request.is_current(
            workspace=capture.ref, context_revision=capture.context_revision
        ):
            self._retry = None
            self._sync_controls()
            self._status("The workspace changed. Start a new generation.")
            return
        self._dialog_open = True
        self._sync_controls()
        self.run_worker(
            self._retry_generation(capture, request, target, previous),
            group="research-studio-generation",
            exit_on_error=False,
        )

    async def _retry_generation(self, capture, request, target, previous) -> None:
        try:
            await self._generate(capture, request, target, previous)
        finally:
            self._dialog_open = False
            if self.is_mounted:
                self._sync_controls()

    @on(Select.Changed, "#research-studio-history")
    def selected_history(self) -> None:
        self.history.sync_selection()

    @on(Button.Pressed, "#research-studio-previous")
    @on(Button.Pressed, "#research-studio-next")
    def page_history(self, event: Button.Pressed) -> None:
        self._offset = max(
            0, self._offset + (20 if event.button.id.endswith("-next") else -20)
        )
        self.refresh_studio()

    @on(Button.Pressed, "#research-studio-output-actions")
    @on(Button.Pressed, "#research-studio-reopen-last")
    def open_actions(self, event: Button.Pressed) -> None:
        row = (
            self._last_saved
            if event.button.id.endswith("-last")
            else self.history.current
        )
        if row and not self.is_busy and not self._dialog_open:
            self._dialog_open = True
            self._sync_controls()
            self.run_worker(
                self._show_actions(row, reopen=event.button.id.endswith("-last")),
                group="research-studio-actions",
                exit_on_error=False,
            )

    async def _show_actions(self, row: StudioHistoryRow, *, reopen=False) -> None:
        try:
            action = await self._select_action(row, reopen=reopen)
        finally:
            self._dialog_open = False
            if self.is_mounted:
                self._sync_controls()
        if action is None:
            return
        key, row = action
        output = row.output
        if key == "replace":
            self._configure(output.output_kind, replace_output=output)
        elif key == "new-version":
            self._configure(output.output_kind, supersedes=output)
        else:
            self.post_message(self.OwnerAction(key, row))

    async def _select_action(self, row: StudioHistoryRow, *, reopen: bool):
        capture = self.controller.capture_request()
        output = row.output
        if capture.ref != output.workspace:
            return
        owner = PRIMARY_OUTPUT_MAPPINGS[
            output.workspace.data_source, output.output_kind
        ]
        try:
            record = await self._port(output.workspace).resolve_output(output)
        except Exception as exc:
            self._show_error(exc)
            return
        if not self._current(capture):
            return
        row = StudioHistoryRow(output, record, row.sources_changed)
        if reopen and record and owner.reopen_destination != "research_workspace":
            return "reopen", row
        actions = {}
        if record:
            if owner.reopen_destination != "research_workspace":
                actions["reopen"] = f"Open in {owner.reopen_destination.title()}"
            replace_capability = self._capabilities.get(owner.replace_capability)
            if row.content:
                actions["discuss"] = "Discuss in Grounded Chat"
                notes = self.query_one(ResearchQuickNotesSection)
                if notes.can_save_note:
                    actions["note"] = "Save to a new Quick Note"
                if notes.can_append_note:
                    actions["append"] = "Append to Quick Note"
                if replace_capability and replace_capability.available:
                    actions["edit"] = "Edit output"
            if (
                self._availability.get(output.output_kind)
                and self._availability[output.output_kind].available
            ):
                actions["new-version"] = "Regenerate as new version"
                if replace_capability and replace_capability.available:
                    actions["replace"] = "Regenerate and replace"
            if owner.delete_capability and hasattr(
                self._port(output.workspace), "delete_output"
            ):
                actions["delete"] = "Delete output"
            if row.content and hasattr(self._port(output.workspace), "export_output"):
                if output.owner_kind is StudioOwnerKind.LOCAL_CHATBOOK:
                    actions["export"] = "Download as Markdown"
                elif replace_capability and replace_capability.available:
                    if record.get("review_state") == "accepted":
                        actions["export"] = "Export accepted Markdown"
                    elif record.get("review_state") in {"draft", "in_review"}:
                        actions["accept"] = "Accept reviewed version"
            limits = (
                "Undo requires a restorable version from the owner."
                if "delete" in actions
                else "Deletion requires version checks from the owner."
            )
        else:
            limits = "Missing canonical item"
            if hasattr(self._port(output.workspace), "unlink_output"):
                actions["unlink"] = "Remove missing link"
        action = await self.app.push_screen_wait(
            ResearchStudioActions(row, actions, limits, owner.owner_label)
        )
        if action is None or not self._current(capture):
            return None
        return action, row
