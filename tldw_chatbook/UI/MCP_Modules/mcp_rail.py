"""MCP Hub left rail: source switch, server rows with readiness badges, scope."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Label, Select, Static

from tldw_chatbook.MCP.readiness import STATE_CSS_CLASSES, STATE_GLYPHS, ReadinessSnapshot

# Task 4: one-shot mount-echo consumption sentinel. `on_select_changed`'s
# scope/scope-ref guards compare an incoming Select.Changed value against the
# value each Select was actually constructed with at the last compose()
# (`_displayed_scope_value`/`_displayed_scope_ref_value`) to swallow that
# constructor-triggered echo. A *standing* sentinel would keep swallowing any
# later user selection that happens to match the same value again (e.g. an
# A -> B -> A round trip's final "A" looks identical to the mount echo), so
# once a guard actually consumes an echo it overwrites the sentinel with this
# unique object instead of leaving the matched value in place -- no real
# Select value can ever equal it, so every subsequent change dispatches.
_ECHO_CONSUMED = object()

MCP_RAIL_ROW_PREFIX = "mcp-rail-row-"
# A4: wide enough that the built-in server's full label ("tldw_chatbook
# (built-in)", 24 chars) always fits without an ellipsis at the rail's real
# rendered width (min-width 24, typically ~35-40 cols at the 3fr share of a
# 140-col QA viewport) -- the old budget of 22 truncated it even though the
# rail had room.
_MAX_ROW_LABEL = 36
# "All servers" carries no readiness glyph but must still line up under the
# same left edge glyph-prefixed rows use ("<glyph> label...", a 2-char-wide
# gutter) -- see _row_label's return and MCPRail.compose()'s "All servers"
# row.
_ALL_SERVERS_GUTTER = "  "


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
    # Task 11 (UX-inputs polish): the tool count sits in a fixed right-side
    # column instead of trailing the label at a variable offset -- the name
    # is left-justified to the same truncation budget every row uses, and
    # the count is right-aligned in a fixed 3-char field (blank, not "0",
    # when no count has ever been discovered) so counts form one scannable
    # column down the rail instead of drifting with label length.
    count = "" if snapshot.tool_count is None else str(snapshot.tool_count)
    return f"{STATE_GLYPHS[snapshot.state]} {prefix}{label:<{_MAX_ROW_LABEL}} {count:>3}"


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
        /* A4: Button defaults to text-align: center; content-align: center
        middle (see Textual's own Button.DEFAULT_CSS) -- left-align rail rows
        instead, mirroring .library-rail-row in _agentic_terminal.tcss. */
        text-align: left;
        content-align: left middle;
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
        # The value each scope/scope-ref Select was actually constructed
        # with on the most recent compose() (post-clamp — see compose()'s
        # own clamping comments). Textual 8.2.7 posts a `Select.Changed` for
        # a Select's own constructor value as part of mounting it; these let
        # `on_select_changed()` recognize and drop that mount-echo instead of
        # forwarding it as a real user-driven ScopeChanged (mirrors the
        # `mcp-rail-source` guard below, which compares against `self.source`
        # directly because that select's displayed value never needs
        # clamping).
        self._displayed_scope_value: str | None = None
        self._displayed_scope_ref_value: Any = None

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
            f"{_ALL_SERVERS_GUTTER}All servers",
            id=f"{MCP_RAIL_ROW_PREFIX}0",
            classes="mcp-rail-row console-action-subdued",
            compact=True,
        )
        all_row.tooltip = "Show every server in the overview table."
        all_row.set_class(self.selected_server_key is None, "is-active")
        yield all_row
        for index, snap in enumerate(self.snapshots, start=1):
            # Task 11: each row carries its readiness state's CSS class
            # (STATE_CSS_CLASSES, Task 3) so it can be colored by status --
            # constructed fresh on every compose() (sync_state() always
            # recomposes), so there is no stale class from a prior render to
            # remove first.
            row = Button(
                _row_label(snap),
                id=f"{MCP_RAIL_ROW_PREFIX}{index}",
                classes=f"mcp-rail-row console-action-subdued {STATE_CSS_CLASSES[snap.state]}",
                compact=True,
            )
            row.tooltip = escape_markup(snap.message or snap.label)
            row.set_class(snap.server_key == self.selected_server_key, "is-active")
            yield row
        if self.source == "server":
            with Vertical(id="mcp-rail-scope"):
                yield Label("Scope", classes="form-label")
                # Phase 1 only ever offers Personal-scope options here; later
                # phases will supply the real option list (team/org scopes,
                # etc.). The workbench keeps tracking the true restored scope
                # in its own state (see MCPWorkbench.get_view_state()) — this
                # clamp only protects the rail's DISPLAY from a restored
                # value (e.g. legacy "team" state) that isn't among the
                # options actually offered, which would otherwise raise
                # InvalidSelectValueError.
                scope_options = self.scope_options or [("Personal", "personal")]
                scope_option_values = [value for _, value in scope_options]
                scope_value = (
                    self.scope_value
                    if self.scope_value in scope_option_values
                    else scope_option_values[0]
                )
                self._displayed_scope_value = scope_value
                yield Select(
                    scope_options,
                    id="mcp-rail-scope-select",
                    allow_blank=False,
                    value=scope_value,
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
                    ref_option_values = [value for _, value in ref_options]
                    if self.scope_ref_value and self.scope_ref_value in ref_option_values:
                        ref_value = self.scope_ref_value
                    else:
                        # Restored/stale value not among the offered scope-ref
                        # options (or no value at all) — no selection.
                        ref_value = Select.NULL
                else:
                    ref_options = [("No scope entities", Select.BLANK)]
                    ref_value = Select.BLANK
                self._displayed_scope_ref_value = ref_value
                yield Select(
                    ref_options,
                    id="mcp-rail-scope-ref",
                    value=ref_value,
                    disabled=not self.scope_ref_options,
                )
        else:
            # No scope selects rendered for this source -- nothing to guard
            # a mount-echo against until the next compose().
            self._displayed_scope_value = None
            self._displayed_scope_ref_value = None

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
            # Mount-echo guard (C1): the value this Select was actually
            # constructed with (post-clamp) at the last compose(). Comparing
            # against `self.scope_value` directly would miss this — that
            # attribute holds the true, un-clamped tracked scope, which can
            # differ from what was actually displayed/selected.
            # Task 4: one-shot -- consume at most the first matching echo,
            # then flip the sentinel to `_ECHO_CONSUMED` so a later user
            # selection that happens to land back on the same value (an
            # A -> B -> A round trip) is never mistaken for a second echo.
            if event.value == self._displayed_scope_value and (
                self._displayed_scope_value is not _ECHO_CONSUMED
            ):
                self._displayed_scope_value = _ECHO_CONSUMED
                return
            self.post_message(self.ScopeChanged(str(event.value), None))
        elif select_id == "mcp-rail-scope-ref":
            event.stop()
            # Same one-shot mount-echo guard as above, for the scope-ref
            # select.
            if event.value == self._displayed_scope_ref_value and (
                self._displayed_scope_ref_value is not _ECHO_CONSUMED
            ):
                self._displayed_scope_ref_value = _ECHO_CONSUMED
                return
            # Both our synthetic placeholder sentinel (Select.BLANK, used when
            # there are no ref options) and the auto-added blank row
            # (Select.NULL, present whenever allow_blank=True) mean "no
            # selection" here.
            is_blank = event.value is Select.BLANK or event.value is Select.NULL
            ref = None if is_blank else str(event.value)
            self.post_message(self.ScopeChanged(self.scope_value, ref))
