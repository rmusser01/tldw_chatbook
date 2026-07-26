"""Console workspace plumbing details tray (Storage, Sync, Handoff)."""

from __future__ import annotations

import re
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceStatusPair,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


_AUTHORITY_LABELS = {
    "local registry ready": "local",
    "local-only": "local",
    "server-backed": "server backed",
    "syncing-to-server": "syncing to server",
    "syncing-from-server": "syncing from server",
    "conflict": "conflict",
    "detached": "detached",
    "remote-only": "remote only",
    "runtime-missing": "runtime missing",
}


class ConsoleWorkspaceDetailsTray(RecomposeCaptureGuard, Vertical):
    """Render workspace plumbing status, readiness, and handoff rows."""

    def __init__(self, state: ConsoleWorkspaceContextState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.styles.height = "auto"
        self.styles.min_height = 0

    def sync_state(self, state: ConsoleWorkspaceContextState) -> None:
        """Refresh the mounted workspace details tray from new display state.

        Args:
            state: Latest workspace context display state.

        Returns:
            None.
        """
        if state == self.state:
            return
        self.state = state
        self.refresh(recompose=True)

    @staticmethod
    def _static(text: str, *, id: str, classes: str = "") -> Static:
        """Create a plain Static widget with markup disabled.

        Args:
            text: Text to display.
            id: Textual widget id.
            classes: Optional CSS classes.

        Returns:
            Static widget configured for plain text.
        """
        return Static(str(text), id=id, classes=classes, markup=False)

    @staticmethod
    def _split_status_row(text: str, fallback_label: str) -> tuple[str, str]:
        """Return a scannable label/value pair from legacy status copy.

        Args:
            text: Existing status copy, optionally containing ``label: value``.
            fallback_label: Label to use when ``text`` has no explicit label.

        Returns:
            Tuple of ``(label, value)`` strings for the status row.
        """
        raw = str(text or "").strip()
        label, separator, value = raw.partition(":")
        if separator:
            clean_label = label.strip()
            clean_value = value.strip()
            if clean_label and clean_value:
                return clean_label, clean_value
            if clean_label:
                return clean_label, "unavailable"
        return fallback_label, raw or "unavailable"

    def _status_pair(
        self,
        text: str,
        *,
        label_id: str,
        value_id: str,
        fallback_label: str,
    ) -> ComposeResult:
        """Build a two-column status row for scan-heavy workspace metadata.

        Args:
            text: Source status copy to split into a label/value pair.
            label_id: Textual widget id for the label cell.
            value_id: Textual widget id for the value cell.
            fallback_label: Label used when ``text`` does not provide one.

        Returns:
            ComposeResult yielding the status-pair widget.
        """
        label, value = self._split_status_row(text, fallback_label)
        yield ConsoleWorkspaceStatusPair(
            label,
            value,
            label_id=label_id,
            value_id=value_id,
        )

    def _server_features_unconfigured(self) -> bool:
        """True when sync, server handoff, and ACP are all factory defaults.

        TASK-715: none of these can be configured anywhere in the UI today
        (no production writer for server adapter state, sync status, or ACP
        handoff state), so their default rows are aspirational jargon. They
        collapse into one plain line until any of them reports real state.
        """
        sync = str(self.state.sync_label or "").strip().lower()
        server = str(self.state.server_readiness_label or "").strip().lower()
        acp = str(self.state.acp_handoff_label or "").strip().lower()
        return (
            sync == "sync: not configured"
            and server.startswith("server: local fallback")
            and acp.startswith("acp task/run: unavailable")
        )

    def compose(self) -> ComposeResult:
        """Render workspace status, readiness, runtime, and handoff rows."""

        collapse_server_features = self._server_features_unconfigured()

        yield from self._status_pair(
            self._friendly_status_label(self.state.authority_label),
            label_id="console-workspace-authority-label",
            value_id="console-workspace-authority-value",
            fallback_label="Storage",
        )
        if not collapse_server_features:
            yield from self._status_pair(
                self._friendly_status_label(self.state.sync_label),
                label_id="console-workspace-sync-label",
                value_id="console-workspace-sync-value",
                fallback_label="Sync",
            )
        yield from self._status_pair(
            self._friendly_status_label(self.state.runtime_label),
            label_id="console-workspace-runtime-label",
            value_id="console-workspace-runtime-value",
            fallback_label="File tools",
        )
        if collapse_server_features:
            yield self._static(
                "Server features (sync, handoff, ACP): not configured. "
                "Chats stay local.",
                id="console-workspace-server-features-collapsed",
                classes="console-workspace-recovery",
            )
        else:
            yield from self._status_pair(
                self._friendly_status_label(self.state.server_readiness_label),
                label_id="console-workspace-server-readiness-label",
                value_id="console-workspace-server-readiness-value",
                fallback_label="Server",
            )
            yield self._static(
                self._friendly_detail_copy(self.state.server_readiness_detail),
                id="console-workspace-server-readiness-detail",
                classes="console-workspace-recovery",
            )
        yield self._static(
            "Handoff",
            id="console-workspace-handoff-title",
            classes="destination-section",
        )
        with Vertical(id="console-workspace-handoff-rows"):
            if self.state.handoff_rows:
                for index, row in enumerate(self.state.handoff_rows):
                    portability = "" if row.portable else " (not portable)"
                    yield self._static(
                        f"{row.title} - {row.transfer_policy.value}{portability}",
                        id=f"console-workspace-handoff-{index}",
                        classes="console-workspace-status-row",
                    )
            else:
                yield self._static(
                    "No handoff package is ready.",
                    id="console-workspace-handoff-empty",
                    classes="console-workspace-empty-copy",
                )
        if not collapse_server_features:
            yield from self._status_pair(
                self._friendly_status_label(self.state.acp_handoff_label),
                label_id="console-workspace-handoff-label",
                value_id="console-workspace-handoff-value",
                fallback_label="ACP",
            )
            yield self._static(
                self._friendly_detail_copy(self.state.acp_handoff_detail),
                id="console-workspace-acp-handoff-detail",
                classes="console-workspace-recovery",
            )
            yield self._static(
                self.state.acp_handoff_audit,
                id="console-workspace-acp-handoff-audit",
                classes="console-workspace-recovery",
            )

    @staticmethod
    def _friendly_status_label(label: str) -> str:
        """Return user-facing workspace status copy for the Console rail."""
        raw = str(label or "").strip()
        normalized = raw.lower()
        if normalized.startswith("authority: unavailable"):
            return "Storage: Unavailable"
        if normalized.startswith("authority:"):
            authority = normalized.partition(":")[2].strip()
            readable = _AUTHORITY_LABELS.get(authority, authority.replace("-", " "))
            return f"Storage: {readable or 'unavailable'}"
        if normalized == "sync: not configured":
            return "Sync: Off"
        if normalized.startswith("runtime: none, file tools disabled"):
            # TASK-715: "Off in Default workspace" ellipsized in the rail's
            # ~23-cell value column even in its own default state.
            return "File tools: Off in Default"
        if normalized.startswith("runtime: none"):
            return "File tools: Off"
        if normalized.startswith("runtime:"):
            readiness = re.search(r"(\d+) ready(?:,\s+(\d+) missing)?", normalized)
            if readiness:
                label = f"File tools: {readiness.group(1)} ready"
                if readiness.group(2):
                    label = f"{label}, {readiness.group(2)} missing"
                return label
            return raw.replace("Runtime:", "File tools:", 1)
        # TASK-715: "Server handoff" (14 cells) wrapped in the 12-cell label
        # column, leaving an orphaned lowercase "handoff" line; and mapping
        # ACP rows to "ACP handoff:" made two different rows share a label
        # with the Handoff package section. Use "Server" / "ACP" - both fit,
        # and the Handoff section title keeps the handoff wording.
        if normalized == "server: local fallback":
            return "Server: Not configured"
        if normalized.startswith("server: unavailable"):
            return "Server: Unavailable"
        if normalized.startswith("server:"):
            return raw
        if normalized.startswith("acp task/run: unavailable"):
            return "ACP: Not configured"
        if normalized.startswith("acp task/run:"):
            return raw.replace("ACP task/run:", "ACP:", 1)
        return raw

    @staticmethod
    def _friendly_detail_copy(copy: str) -> str:
        """Return first-run readable detail while preserving diagnostic intent."""
        raw = str(copy or "").strip()
        normalized = raw.lower()
        if (
            "local registry fallback is active" in normalized
            or "local registry is authoritative" in normalized
        ):
            return "Chats stay local. Connect a server later for explicit handoff."
        if "acp task/run package handoff is not wired" in normalized:
            return "ACP task/run package handoff is not configured yet."
        return raw
