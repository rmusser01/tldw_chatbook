"""Launch a native run and review its verified Report in Research Workspace."""

from datetime import UTC, datetime
from uuid import uuid4

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from ...Constants import TAB_RESEARCH
from ...Research_Interop.research_scope_service import ResearchScopeService
from ...Research_Interop.server_research_service import ServerResearchService
from ...Research_Workspace.contracts import WorkspaceDataSource
from ...Research_Workspace.deep_research_bridge import (
    DeepResearchBridge,
    ResearchReturnMismatch,
)
from ...Research_Workspace.deep_research_context import (
    MAX_RESEARCH_QUERY_CHARS,
    ResearchLaunchContext,
    ResearchLaunchOutputRef,
)
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.modal_dismissal import SafeModalDismissMixin
from ...tldw_api.exceptions import APIConnectionError
from ...tldw_api.workspace_research_runs import WorkspaceResearchRunsClient
from ..Navigation.main_navigation import NavigateToScreen
from ..Navigation.pending_handoff_store import HandoffChannel
from ..Navigation.research_run_handoffs import research_run_target_current
from .chat_region import ResearchChatRegion
from .studio_history import StudioHistoryRow, history_key
from .studio_region import ResearchStudioRegion


class ResearchLaunchDialog(SafeModalDismissMixin, ModalScreen[str | None]):
    SAFE_MODAL_CONTENT = "#research-launch-dialog"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]

    def __init__(self, source_count: int, initial_title: str) -> None:
        super().__init__()
        self.source_count = source_count
        self.initial_title = initial_title

    def compose(self) -> ComposeResult:
        with Vertical(id="research-launch-dialog", classes="research-transfer-dialog"):
            yield Static("Launch Deep Research", classes="research-pane-title")
            yield Static(
                f"Runs will record the versions of {self.source_count} selected workspace sources.",
                markup=False,
            )
            yield Input(
                self.initial_title,
                placeholder="Research question",
                max_length=MAX_RESEARCH_QUERY_CHARS,
                id="research-launch-query",
            )
            yield Static("", id="research-launch-status", markup=False)
            yield Button(
                "Launch in Runs",
                id="research-launch-confirm",
                variant="primary",
                compact=True,
            )
            yield Button("Cancel", id="research-launch-cancel", compact=True)

    @on(Input.Submitted, "#research-launch-query")
    @on(Button.Pressed, "#research-launch-confirm")
    def launch(self) -> None:
        query = " ".join(self.query_one("#research-launch-query", Input).value.split())
        if not query:
            self.query_one("#research-launch-status", Static).update(
                "Enter a research question."
            )
            return
        self.dismiss(query)

    @on(Button.Pressed, "#research-launch-cancel")
    def close(self) -> None:
        self.dismiss(None)


class ResearchReportReview(SafeModalDismissMixin, ModalScreen[str | None]):
    SAFE_MODAL_CONTENT = "#research-report-review"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]

    def __init__(self, preview, existing) -> None:
        super().__init__()
        self.preview = preview
        self.existing = existing

    def compose(self) -> ComposeResult:
        with Vertical(
            id="research-report-review",
            classes="research-transfer-dialog research-transfer-scroll-dialog",
        ):
            yield Static("Review Deep Research Report", classes="research-pane-title")
            with VerticalScroll(classes="research-transfer-scroll"):
                yield Static(self.preview.context.query, markup=False)
                target = self.preview.target
                details = f"Run {target.run_id} · {target.origin.data_source.value.title()} workspace {target.origin.workspace_id}"
                if self.preview.source_count is not None:
                    details += f" · {self.preview.source_count} report sources"
                yield Static(details, markup=False)
                text = self.preview.report
                if len(text) > 8000:
                    text = (
                        text[:8000]
                        + "\n\nPreview shortened. Saving retains the full report."
                    )
                yield Static(text, id="research-report-preview", markup=False)
            yield Button(
                "Open saved Report" if self.existing else "Save draft Report",
                id="research-report-save",
                variant="primary",
                compact=True,
            )
            if self.existing:
                yield Button(
                    "Create new version", id="research-report-new-version", compact=True
                )
            yield Button("Cancel", id="research-report-cancel", compact=True)

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        result = {
            "research-report-save": "save",
            "research-report-new-version": "new",
        }.get(event.button.id)
        self.dismiss(result)


def _bound_bridge(screen, ref, port) -> DeepResearchBridge:
    app = screen.app_instance
    scope = getattr(app, "research_scope_service", None)
    if scope is None or screen.overlay_store is None:
        raise ValueError("Research Runs or the device workspace store is unavailable")
    if ref.data_source is WorkspaceDataSource.SERVER:
        client = port.copy_client(ref)
        policy = app.service_policy_enforcer
        scope = ResearchScopeService(
            local_service=None,
            server_service=ServerResearchService(
                client,
                policy_enforcer=policy,
                workspace_runs=WorkspaceResearchRunsClient(
                    client, ref.workspace_id, ref.principal_id
                ),
            ),
            policy_enforcer=policy,
        )
    return DeepResearchBridge(scope, screen.overlay_store)


def _current(screen, capture) -> bool:
    return (
        screen.is_attached
        and screen.app.is_running
        and screen.controller.is_current_request(capture)
        and research_run_target_current(screen.app_instance, capture.ref)
    )


async def show_research_launch(screen, row=None) -> None:
    capture = screen.controller.capture_request()
    ref = capture.ref
    port = screen.controller.port_for_data_source(ref.data_source)
    try:
        sources = await port.capture_chat_sources(ref)
        if not any(source.ready_modes for source in sources):
            raise ValueError(
                "Select a ready workspace source before launching Deep Research"
            )
        if not _current(screen, capture):
            return
        title = row.title if row else ""
        query = await screen.app.push_screen_wait(
            ResearchLaunchDialog(len(sources), title)
        )
        if not query or not _current(screen, capture):
            return
        if await port.capture_chat_sources(ref) != sources:
            raise ValueError(
                "Selected sources changed. Review a new Deep Research launch"
            )
        if not _current(screen, capture):
            return
        conversation = screen.query_one(ResearchChatRegion).conversation_id
        output = None
        if row:
            selected = row.output
            record = await port.resolve_output(selected)
            if (
                selected.workspace != ref
                or not record
                or record.get(
                    "artifact_revision"
                    if ref.data_source is WorkspaceDataSource.LOCAL
                    else "version"
                )
                != selected.owner_version
            ):
                raise ValueError(
                    "The initiating Studio output changed. Reload it first"
                )
            output = ResearchLaunchOutputRef(
                selected.owner_kind, selected.owner_id, selected.owner_version
            )
        local = ref.data_source is WorkspaceDataSource.LOCAL
        context = ResearchLaunchContext(
            ref,
            uuid4().hex,
            sources,
            query,
            datetime.now(UTC).isoformat(),
            initiating_output=output,
            local_conversation_id=conversation if local else "",
            server_chat_id=conversation if not local else "",
        )
        bridge = _bound_bridge(screen, ref, port)
        launch = bridge.launch_local if local else bridge.launch_server
        while _current(screen, capture):
            try:
                target = await launch(context)
            except APIConnectionError:
                if (
                    not _current(screen, capture)
                    or await screen.app.push_screen_wait(
                        ConfirmationDialog(
                            title="The launch response was lost",
                            message="Retry this launch to recover its existing Server run.",
                            confirm_label="Retry launch",
                            cancel_label="Close",
                        )
                    )
                    is not True
                ):
                    return
            else:
                if _current(screen, capture) and screen.app_instance._stage_handoff(
                    HandoffChannel.RESEARCH_RUN_TARGET,
                    target,
                    recovery="The saved run could not open.",
                ):
                    screen.app_instance.post_message(NavigateToScreen(TAB_RESEARCH))
                return
    except Exception as exc:
        if _current(screen, capture):
            _show_error(screen, exc)


def start_research_return(screen) -> bool:
    store = getattr(screen.app_instance, "pending_handoffs", None)
    if store is not None and (
        claim := store.claim(HandoffChannel.RESEARCH_WORKSPACE_RETURN)
    ):
        screen.run_worker(
            _show_research_return(screen, store, claim),
            group="research-return",
            exit_on_error=False,
        )
        return True
    return False


async def _show_research_return(screen, store, claim) -> None:
    target = claim.value
    ref = target.origin
    try:
        if not research_run_target_current(screen.app_instance, ref):
            raise ValueError("Open the run's original Server profile and account")
        screen.controller.select_data_source(ref.data_source)
        screen.query_one("#research-workspace-header").sync_data_source(ref.data_source)
        state = await screen.controller.refresh_workspace_catalog()
        if not screen.is_attached or not store.is_current_claim(claim):
            return
        if not any(row.ref == ref and not row.archived for row in state.workspaces):
            raise ValueError("The run's origin workspace is unavailable")
        await screen._apply_catalog_state(state, preferred_ref=ref)
        if not screen.is_attached or screen.controller.selected_ref != ref:
            return
        capture = screen.controller.capture_request()
        port = screen.controller.port_for_data_source(ref.data_source)
        preview = await _bound_bridge(screen, ref, port).preview_return(target)
        if not _current(screen, capture) or not store.is_current_claim(claim):
            return
        request = preview.studio_request(context_revision=capture.context_revision)
        existing = await port.get_imported_report(request)
        if not _current(screen, capture) or not store.acknowledge_current(claim):
            return
        choice = await screen.app.push_screen_wait(
            ResearchReportReview(preview, existing)
        )
        if choice is None or not _current(screen, capture):
            return
        saved = (
            existing
            if choice == "save" and existing
            else await preview.save_report(
                port,
                request,
                supersedes=existing if choice == "new" else None,
            )
        )
        if _current(screen, capture):
            from .workspace_menu import ResearchPaneModeStrip

            screen.select_pane_mode(ResearchPaneModeStrip.Selected("studio"))
            studio = screen.query_one(ResearchStudioRegion)
            if saved == existing:
                await studio._show_actions(
                    StudioHistoryRow(saved, None, False), reopen=True
                )
            else:
                studio._status("Saved a draft Report in Studio.")
                studio.refresh_studio(preferred=history_key(saved))
    except Exception as exc:
        if screen.controller.selected_ref in {None, ref}:
            _show_error(screen, exc)
    finally:
        if store.is_current_claim(claim):
            store.release(claim)
            if (
                screen.is_attached
                and screen.app.is_running
                and screen.controller.selected_ref is None
            ):
                screen._start_catalog_refresh()


def _show_error(screen, exc: Exception) -> None:
    logger.warning("Research Workspace Runs action failed ({})", type(exc).__name__)
    if screen.is_attached and screen.app.is_running:
        text = (
            str(exc)
            if type(exc) is ValueError or isinstance(exc, ResearchReturnMismatch)
            else "Research Runs are unavailable. Open Runs and retry."
        )
        screen.notify(text, severity="warning")
