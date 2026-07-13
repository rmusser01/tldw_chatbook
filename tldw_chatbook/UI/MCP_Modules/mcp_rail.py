"""MCP Hub left rail: source switch, server rows with readiness badges, scope."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Label, Select, Static

from tldw_chatbook.MCP.readiness import STATE_GLYPHS, ReadinessSnapshot

MCP_RAIL_ROW_PREFIX = "mcp-rail-row-"
_MAX_ROW_LABEL = 22


def _row_label(snapshot: ReadinessSnapshot) -> str:
    # snapshot.label is user-controlled (local profile ids, server-reported
    # names) and is rendered through Button, which parses str labels as Rich
    # markup — escape it so a profile id like "[bold red]x" can't inject
    # styling or break layout.
    label = snapshot.label
    if len(label) > _MAX_ROW_LABEL:
        label = f"{label[: _MAX_ROW_LABEL - 3].rstrip()}..."
    label = escape_markup(label)
    prefix = "⌂ " if snapshot.source == "builtin" else ""
    suffix = f" · {snapshot.tool_count}" if snapshot.tool_count is not None else ""
    return f"{STATE_GLYPHS[snapshot.state]} {prefix}{label}{suffix}"


class MCPRail(Vertical):
    """Left rail for the MCP workbench. Index-based row ids; keys in a list."""

    DEFAULT_CSS = """
    MCPRail {
        width: 3fr;
        min-width: 24;
        height: 100%;
        min-height: 0;
    }
    Button.mcp-rail-row {
        width: 100%;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }
    """

    class SourceChanged(Message, namespace="mcp_rail"):
        def __init__(self, source: str) -> None:
            super().__init__()
            self.source = source

    class ServerSelected(Message, namespace="mcp_rail"):
        def __init__(self, server_key: str | None) -> None:
            super().__init__()
            self.server_key = server_key

    class ScopeChanged(Message, namespace="mcp_rail"):
        def __init__(self, scope: str, scope_ref: str | None) -> None:
            super().__init__()
            self.scope = scope
            self.scope_ref = scope_ref

    def __init__(
        self,
        *,
        source: str,
        snapshots: list[ReadinessSnapshot],
        selected_server_key: str | None,
        scope_options: list[tuple[str, str]],
        scope_value: str,
        scope_ref_options: list[tuple[str, str]],
        scope_ref_value: str | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.source = source
        self.snapshots = snapshots
        self.selected_server_key = selected_server_key
        self.scope_options = scope_options
        self.scope_value = scope_value
        self.scope_ref_options = scope_ref_options
        self.scope_ref_value = scope_ref_value
        self._row_keys: list[str | None] = []

    def sync_state(
        self,
        *,
        source: str,
        snapshots: list[ReadinessSnapshot],
        selected_server_key: str | None,
        scope_options: list[tuple[str, str]],
        scope_value: str,
        scope_ref_options: list[tuple[str, str]],
        scope_ref_value: str | None,
    ) -> None:
        self.source = source
        self.snapshots = snapshots
        self.selected_server_key = selected_server_key
        self.scope_options = scope_options
        self.scope_value = scope_value
        self.scope_ref_options = scope_ref_options
        self.scope_ref_value = scope_ref_value
        self.refresh(recompose=True)

    def compose(self) -> ComposeResult:
        yield Static("Source", classes="destination-section mcp-rail-heading")
        yield Select(
            [("Local", "local"), ("Server", "server")],
            id="mcp-rail-source",
            allow_blank=False,
            value=self.source if self.source in ("local", "server") else "local",
        )
        yield Static("Servers", classes="destination-section mcp-rail-heading")
        self._row_keys = [None] + [snap.server_key for snap in self.snapshots]
        all_row = Button(
            "All servers",
            id=f"{MCP_RAIL_ROW_PREFIX}0",
            classes="mcp-rail-row console-action-subdued",
            compact=True,
        )
        all_row.set_class(self.selected_server_key is None, "is-active")
        yield all_row
        for index, snap in enumerate(self.snapshots, start=1):
            row = Button(
                _row_label(snap),
                id=f"{MCP_RAIL_ROW_PREFIX}{index}",
                classes="mcp-rail-row console-action-subdued",
                compact=True,
            )
            row.tooltip = escape_markup(snap.message or snap.label)
            row.set_class(snap.server_key == self.selected_server_key, "is-active")
            yield row
        if self.source == "server":
            with Vertical(id="mcp-rail-scope"):
                yield Label("Scope", classes="form-label")
                yield Select(
                    self.scope_options or [("Personal", "personal")],
                    id="mcp-rail-scope-select",
                    allow_blank=False,
                    value=self.scope_value,
                )
                yield Label("Scope Entity", classes="form-label")
                # NOTE: `Select.BLANK` is not a real Select sentinel in this
                # Textual version — it resolves to `Widget.BLANK` (`False`)
                # via MRO, distinct from the actual blank marker `Select.NULL`.
                # It's only safe here as the value of our own synthetic
                # placeholder option (so its custom label isn't replaced by
                # the dim default prompt text). When real options exist but
                # nothing is selected yet, `Select.NULL` is the value that
                # `allow_blank=True` (the default) actually accepts.
                if self.scope_ref_options:
                    ref_options = self.scope_ref_options
                    ref_value = self.scope_ref_value if self.scope_ref_value else Select.NULL
                else:
                    ref_options = [("No scope entities", Select.BLANK)]
                    ref_value = Select.BLANK
                yield Select(
                    ref_options,
                    id="mcp-rail-scope-ref",
                    value=ref_value,
                    disabled=not self.scope_ref_options,
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith(MCP_RAIL_ROW_PREFIX):
            return
        event.stop()
        index = int(button_id.removeprefix(MCP_RAIL_ROW_PREFIX))
        if 0 <= index < len(self._row_keys):
            self.post_message(self.ServerSelected(self._row_keys[index]))

    def on_select_changed(self, event: Select.Changed) -> None:
        select_id = event.select.id or ""
        if select_id == "mcp-rail-source":
            event.stop()
            if event.value in ("local", "server") and event.value != self.source:
                self.post_message(self.SourceChanged(str(event.value)))
        elif select_id == "mcp-rail-scope-select":
            event.stop()
            self.post_message(self.ScopeChanged(str(event.value), None))
        elif select_id == "mcp-rail-scope-ref":
            event.stop()
            # Both our synthetic placeholder sentinel (Select.BLANK, used when
            # there are no ref options) and the auto-added blank row
            # (Select.NULL, present whenever allow_blank=True) mean "no
            # selection" here.
            is_blank = event.value is Select.BLANK or event.value is Select.NULL
            ref = None if is_blank else str(event.value)
            self.post_message(self.ScopeChanged(self.scope_value, ref))
