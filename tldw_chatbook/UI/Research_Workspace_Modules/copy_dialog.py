"""Review an exact Copy destination and the frozen owner decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from ...Research_Workspace.contracts import QualifiedWorkspaceRef, WorkspaceDataSource
from ...Research_Workspace.copy_models import (
    CopyConflictChoice,
    CopyReason,
    CopyTreatment,
    ResearchCopyManifest,
)
from ...Widgets.modal_dismissal import SafeModalDismissMixin

if TYPE_CHECKING:
    from .sharing_dialog import WorkspaceShareDraft


_TREATMENT_LABELS = {
    CopyTreatment.CONTENT: "Copy content",
    CopyTreatment.REFERENCE: "Reference",
    CopyTreatment.METADATA_ONLY: "Metadata only",
    CopyTreatment.OMIT: "Omit",
    CopyTreatment.BLOCKED: "Blocked",
}
_REASONS = {
    CopyReason.NONE: "",
    CopyReason.OWNER_UNSUPPORTED: "This item needs a supported destination owner.",
    CopyReason.LOCAL_ONLY: "This item is restricted to its original device.",
    CopyReason.CHOOSE_CONFLICT: "Choose how to handle the existing destination item.",
    CopyReason.REPLACE_UNAVAILABLE: "Choose Copy as new or Keep destination. Replacement requires an exact owner version.",
    CopyReason.SOURCE_REVISION_REQUIRED: "Refresh the source to capture its owner revision.",
}
_CHOICES = (
    ("Keep destination", CopyConflictChoice.KEEP_DESTINATION),
    ("Copy as new", CopyConflictChoice.COPY_AS_NEW),
    ("Reference existing", CopyConflictChoice.REFERENCE_EXISTING),
    ("Replace destination", CopyConflictChoice.REPLACE_DESTINATION),
    ("Omit", CopyConflictChoice.OMIT),
    ("Cancel Copy", CopyConflictChoice.CANCEL),
)


class ResearchWorkspaceActions(SafeModalDismissMixin, ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close", show=False)]
    SAFE_MODAL_CONTENT = "#research-workspace-actions-dialog"

    def __init__(
        self, ref: QualifiedWorkspaceRef, *, can_copy: bool, can_share: bool
    ) -> None:
        super().__init__()
        self.ref = ref
        self.can_copy = can_copy
        self.can_share = can_share

    def compose(self) -> ComposeResult:
        with Vertical(
            id="research-workspace-actions-dialog", classes="research-transfer-dialog"
        ):
            yield Static("Workspace actions", classes="research-pane-title")
            if self.ref.data_source is WorkspaceDataSource.LOCAL:
                yield Button("Export bundle", id="research-action-export", compact=True)
                yield Button(
                    "Copy to Server",
                    id="research-action-copy-server",
                    compact=True,
                    disabled=not self.can_copy,
                )
                yield Button(
                    "Copy to Server and Share",
                    id="research-action-copy-server-share",
                    compact=True,
                    disabled=not self.can_copy,
                )
            else:
                yield Button(
                    "Copy to Local",
                    id="research-action-copy-local",
                    compact=True,
                    disabled=not self.can_copy,
                )
                yield Button(
                    "Share",
                    id="research-action-share",
                    compact=True,
                    disabled=not self.can_share,
                )
            yield Button(
                "Copy receipts",
                id="research-action-receipts",
                compact=True,
                disabled=not self.can_copy,
            )
            if not self.can_copy:
                yield Static(
                    "Configure both workspace authorities to use Copy.", markup=False
                )
            yield Button("Close", id="research-action-close", compact=True)

    @on(Button.Pressed)
    def select_action(self, event: Button.Pressed) -> None:
        action = (event.button.id or "").removeprefix("research-action-")
        self.dismiss(None if action == "close" else action)


@dataclass(frozen=True, slots=True)
class ResearchCopyTarget:
    ref: QualifiedWorkspaceRef
    name: str = ""
    share_draft: WorkspaceShareDraft | None = None


class ResearchCopyTargetDialog(
    SafeModalDismissMixin, ModalScreen[ResearchCopyTarget | None]
):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#research-copy-target-dialog"

    def __init__(
        self,
        *,
        server_ref: QualifiedWorkspaceRef | None = None,
        default_name: str = "",
        local_workspaces=(),
        allow_sharing_draft: bool = False,
    ) -> None:
        super().__init__()
        self.server_ref = server_ref
        self.default_name = default_name
        self.local_workspaces = tuple(local_workspaces)
        self.allow_sharing_draft = allow_sharing_draft and server_ref is not None
        self._share_draft = None

    def compose(self) -> ComposeResult:
        with Vertical(
            id="research-copy-target-dialog", classes="research-transfer-dialog"
        ):
            yield Static("Copy destination", classes="research-pane-title")
            if self.server_ref:
                yield Static(
                    f"New Server workspace · {self.server_ref.server_profile_id}",
                    markup=False,
                )
                yield Input(
                    self.default_name,
                    placeholder="Workspace name",
                    id="research-copy-name",
                )
            else:
                yield Select(
                    [(row.name, row.ref) for row in self.local_workspaces],
                    prompt="Local workspace",
                    id="research-copy-local-target",
                )
            if self.allow_sharing_draft:
                yield Button(
                    "Share settings", id="research-copy-share-settings", compact=True
                )
            yield Static("", id="research-copy-target-status", markup=False)
            yield Button(
                "Preview Copy",
                id="research-copy-preview",
                compact=True,
                variant="primary",
            )
            yield Button("Cancel", id="research-copy-target-cancel", compact=True)

    @on(Button.Pressed, "#research-copy-preview")
    def preview(self) -> None:
        if self.server_ref is not None:
            name = self.query_one("#research-copy-name", Input).value.strip()
            if name:
                self.dismiss(
                    ResearchCopyTarget(self.server_ref, name, self._share_draft)
                )
                return
        else:
            choice = self.query_one(Select).value
            if isinstance(choice, QualifiedWorkspaceRef):
                self.dismiss(ResearchCopyTarget(choice))
                return
        self.query_one("#research-copy-target-status", Static).update(
            "Choose a destination workspace."
        )

    @on(Button.Pressed, "#research-copy-share-settings")
    def open_share_settings(self) -> None:
        self.run_worker(
            self._configure_sharing(), group="research-copy-share-draft", exclusive=True
        )

    async def _configure_sharing(self) -> None:
        from .sharing_dialog import ResearchSharingDialog

        if self.server_ref is None:
            return
        draft = await self.app.push_screen_wait(
            ResearchSharingDialog(
                self.server_ref,
                None,
                current=lambda: self.is_mounted,
                draft=self._share_draft,
                prepare_only=True,
            )
        )
        if draft is not None and self.is_mounted:
            self._share_draft = draft
            self.query_one("#research-copy-target-status", Static).update(
                "Share settings saved for review after Copy."
            )

    @on(Button.Pressed, "#research-copy-target-cancel")
    def close(self) -> None:
        self.dismiss(None)


class ResearchCopyPreview(
    SafeModalDismissMixin, ModalScreen[bool | dict[int, CopyConflictChoice] | None]
):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#research-copy-preview-dialog"

    def __init__(self, manifest: ResearchCopyManifest) -> None:
        super().__init__()
        self.manifest = manifest

    def compose(self) -> ComposeResult:
        manifest = self.manifest
        destination = manifest.destination
        with Vertical(
            id="research-copy-preview-dialog",
            classes="research-transfer-dialog research-transfer-scroll-dialog",
        ):
            yield Static("Review Copy", classes="research-pane-title")
            with VerticalScroll(classes="research-transfer-scroll"):
                target = manifest.target_name or destination.workspace_id
                qualified = (
                    f" · {destination.server_profile_id} · {destination.principal_id}"
                    if destination.server_profile_id
                    else ""
                )
                yield Static(
                    f"{manifest.source.workspace_id} → {destination.data_source.value.title()} {target}{qualified}",
                    markup=False,
                )
                yield Static(
                    f"{manifest.estimated_bytes:,} bytes · {len(manifest.items)} items",
                    markup=False,
                )
                yield Static(
                    "Device-only pane preferences, folders, annotations, and drafts are excluded.",
                    markup=False,
                )
                for position, item in enumerate(manifest.items):
                    version = (
                        f" · version {item.source.version}"
                        if item.source.version is not None
                        else ""
                    )
                    yield Static(
                        f"{_TREATMENT_LABELS[item.treatment]} · {item.source.kind.value.replace('_', ' ')} {item.source.owner_id}{version} · {item.estimated_bytes:,} bytes",
                        markup=False,
                    )
                    if item.reason is not CopyReason.NONE:
                        yield Static(_REASONS[item.reason], markup=False)
                    if item.destination is not None:
                        version = (
                            f"version {item.destination.version}"
                            if item.destination.has_revision
                            else "owner unavailable"
                        )
                        yield Static(
                            f"Previous destination {item.destination.owner_id} · {version}",
                            markup=False,
                        )
                        yield Select(
                            _CHOICES,
                            id=f"research-copy-choice-{position}",
                            value=item.conflict or Select.NULL,
                            prompt="Choose conflict treatment",
                        )
            if any(row.destination is not None for row in manifest.items):
                yield Button(
                    "Review conflict choices", id="research-copy-resolve", compact=True
                )
            yield Button(
                "Confirm Copy",
                id="research-copy-confirm",
                compact=True,
                variant="primary",
                disabled=not any(row.actionable for row in manifest.items)
                or any(
                    row.reason is CopyReason.CHOOSE_CONFLICT
                    or row.conflict is CopyConflictChoice.CANCEL
                    for row in manifest.items
                ),
            )
            yield Button("Cancel", id="research-copy-preview-cancel", compact=True)

    @on(Button.Pressed, "#research-copy-resolve")
    def resolve(self) -> None:
        self.dismiss(
            {
                int(widget.id.rsplit("-", 1)[-1]): widget.value
                for widget in self.query(Select)
                if isinstance(widget.value, CopyConflictChoice)
            }
        )

    @on(Select.Changed)
    def conflict_changed(self) -> None:
        changed = any(
            (widget.value if isinstance(widget.value, CopyConflictChoice) else None)
            != self.manifest.items[int(widget.id.rsplit("-", 1)[-1])].conflict
            for widget in self.query(Select)
        )
        allowed = any(row.actionable for row in self.manifest.items) and not any(
            row.reason is CopyReason.CHOOSE_CONFLICT
            or row.conflict is CopyConflictChoice.CANCEL
            for row in self.manifest.items
        )
        self.query_one("#research-copy-confirm", Button).disabled = (
            changed or not allowed
        )

    @on(Button.Pressed, "#research-copy-confirm")
    def confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#research-copy-preview-cancel")
    def close(self) -> None:
        self.dismiss(None)
