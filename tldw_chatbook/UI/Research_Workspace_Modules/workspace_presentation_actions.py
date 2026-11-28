"""Device-only workspace presentation controls and recent selection."""

import asyncio

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Select, Static

from ...Research_Workspace.layout_state import ResearchPanePreferences
from ...Research_Workspace.overlay_store import MAX_WORKSPACE_BANNER_CHARS
from ...Widgets.modal_dismissal import SafeModalDismissMixin
from .header_region import ResearchHeaderRegion
from .workspace_lifecycle_actions import ResearchWorkspaceName, ResearchWorkspacePicker


PRESENTATION_ACTIONS = frozenset(
    {"banner", "split", "pin", "unpin", "pinned", "recent"}
)


class ResearchWorkspaceSplit(
    SafeModalDismissMixin, ModalScreen[ResearchPanePreferences | None]
):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#research-workspace-split-dialog"

    def __init__(self, preferences: ResearchPanePreferences) -> None:
        super().__init__()
        self.preferences = preferences

    def compose(self) -> ComposeResult:
        with Vertical(
            id="research-workspace-split-dialog", classes="research-transfer-dialog"
        ):
            yield Static("Split workspace", classes="research-pane-title")
            yield Static(
                "Saved on this device. Narrow layouts show one pane at a time."
            )
            yield Checkbox(
                "Show Sources",
                value=self.preferences.sources_open,
                id="research-split-sources",
            )
            yield Checkbox(
                "Show Studio",
                value=self.preferences.studio_open,
                id="research-split-studio",
            )
            yield Select(
                [
                    ("Prefer Sources when space is limited", "sources"),
                    ("Prefer Studio when space is limited", "studio"),
                ],
                value=self.preferences.preferred_companion,
                allow_blank=False,
                id="research-split-companion",
            )
            yield Button(
                "Save layout", id="research-workspace-save-split", compact=True
            )
            yield Button("Cancel", id="research-workspace-cancel-split", compact=True)

    @on(Button.Pressed, "#research-workspace-save-split")
    def save(self) -> None:
        self.dismiss(
            ResearchPanePreferences(
                sources_open=self.query_one("#research-split-sources", Checkbox).value,
                studio_open=self.query_one("#research-split-studio", Checkbox).value,
                preferred_companion=self.query_one(Select).value,
            )
        )

    @on(Button.Pressed, "#research-workspace-cancel-split")
    def cancel(self) -> None:
        self.dismiss(None)


async def run_workspace_presentation_action(screen, action, authority, current) -> None:
    store = screen.overlay_store
    if store is None:
        raise ValueError("Device workspace presentation storage is unavailable.")
    ref = screen.controller.selected_ref
    header = screen.query_one(ResearchHeaderRegion) if action == "banner" else None
    if action in {"recent", "pinned"}:
        loaded = await asyncio.to_thread(store.load_all)
        if not current():
            return
        state = screen.controller.catalog_state
        rows = []
        for workspace in state.workspaces if state else ():
            overlay = loaded.records.get(workspace.ref)
            if overlay is None:
                continue
            presentation = overlay.workspace_presentation
            include = (
                presentation.pinned
                if action == "pinned"
                else presentation.last_opened_at
            )
            if include:
                rows.append((presentation.last_opened_at or "", workspace))
        latest = tuple(
            row for _, row in sorted(rows, key=lambda item: item[0], reverse=True)[:25]
        )
        selected = await screen.app.push_screen_wait(
            ResearchWorkspacePicker(authority, latest, mode=action)
        )
        if selected is not None and current():
            await screen._refresh_workspace_catalog(selected)
            await screen.remember_workspace_visit(selected)
        return
    if ref is None:
        return
    if action == "split":
        layout = await screen.app.push_screen_wait(
            ResearchWorkspaceSplit(screen.pane_preferences)
        )
        if layout is not None and current():
            screen.pane_preferences = layout
            screen._apply_pane_layout(
                max(1, screen.size.width), relocate_hidden_focus=True
            )
            screen._start_overlay_save()
        return
    if action == "banner":
        overlay = await asyncio.to_thread(store.load, ref)
        if not current():
            return
        banner = overlay.workspace_presentation.banner if overlay else None
        value = await screen.app.push_screen_wait(
            ResearchWorkspaceName(
                "Customize banner",
                banner or "",
                label="Banner text",
                max_length=MAX_WORKSPACE_BANNER_CHARS,
                description="Saved on this device. Clear the field to remove the banner.",
                allow_blank=True,
            )
        )
        if value is not None and current():
            saved = await asyncio.to_thread(
                store.set_workspace_presentation, ref, banner=value
            )
            if current() and header is not None and header.is_attached:
                header.sync_banner(
                    saved.workspace_presentation.banner
                )
        return
    await asyncio.to_thread(
        store.set_workspace_presentation,
        ref,
        pinned=action == "pin",
        last_opened_at=screen._now_factory(),
    )
    if current():
        screen.notify(
            "Workspace pinned on this device."
            if action == "pin"
            else "Workspace unpinned on this device."
        )
