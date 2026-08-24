"""Textual-free state fencing for Research Workspace requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .contracts import (
    QualifiedWorkspaceRef,
    ResearchWorkspacePort,
    ResearchWorkspaceSummary,
    WorkspaceDataSource,
)


@dataclass(frozen=True, slots=True)
class ResearchRequestContext:
    ref: QualifiedWorkspaceRef
    capability_revision: str
    context_revision: int


class ResearchWorkspaceController:
    """Own normalized state and prevent late async results from repainting."""

    def __init__(
        self, ports: Mapping[WorkspaceDataSource, ResearchWorkspacePort]
    ) -> None:
        self._ports = dict(ports)
        self._context_revision = 0
        self._selected_ref: QualifiedWorkspaceRef | None = None
        self._capability_revision = ""
        self._canonical_workspaces: dict[
            QualifiedWorkspaceRef, ResearchWorkspaceSummary
        ] = {}
        self.visible_workspace: ResearchWorkspaceSummary | None = None

    @property
    def context_revision(self) -> int:
        return self._context_revision

    @property
    def selected_ref(self) -> QualifiedWorkspaceRef | None:
        return self._selected_ref

    def select_workspace(
        self, ref: QualifiedWorkspaceRef, *, capability_revision: str = ""
    ) -> int:
        self._selected_ref = ref
        self._capability_revision = capability_revision.strip()
        self._context_revision += 1
        self.visible_workspace = self._canonical_workspaces.get(ref)
        return self._context_revision

    def set_capability_revision(self, capability_revision: str) -> int:
        normalized = capability_revision.strip()
        if normalized != self._capability_revision:
            self._capability_revision = normalized
            self._context_revision += 1
        return self._context_revision

    def capture_request(self) -> ResearchRequestContext:
        if self._selected_ref is None:
            raise RuntimeError("No Research workspace is selected")
        return ResearchRequestContext(
            ref=self._selected_ref,
            capability_revision=self._capability_revision,
            context_revision=self._context_revision,
        )

    async def refresh_selected_workspace(self) -> bool:
        capture = self.capture_request()
        port = self._ports.get(capture.ref.data_source)
        if port is None:
            raise RuntimeError(
                f"No adapter is configured for {capture.ref.data_source.value}"
            )
        result = await port.get_workspace(capture.ref)
        if result is None:
            return False
        return self.accept_workspace_result(capture, result)

    def accept_workspace_result(
        self,
        capture: ResearchRequestContext,
        result: ResearchWorkspaceSummary,
    ) -> bool:
        if result.ref != capture.ref:
            raise ValueError("Request returned a mismatched workspace ref")
        self._canonical_workspaces[capture.ref] = result
        if (
            capture.context_revision != self._context_revision
            or capture.ref != self._selected_ref
            or capture.capability_revision != self._capability_revision
        ):
            return False
        self.visible_workspace = result
        return True

    def canonical_workspace(
        self, ref: QualifiedWorkspaceRef
    ) -> ResearchWorkspaceSummary | None:
        return self._canonical_workspaces.get(ref)
