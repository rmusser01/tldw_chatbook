"""MCP Hub left rail: source switch, server rows with readiness badges, scope."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Label, Select, Static

from tldw_chatbook.MCP.readiness import (
    STATE_CSS_CLASSES,
    STATE_GLYPHS,
    STATE_LABELS,
    ReadinessSnapshot,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard

# Task 4: one-shot mount-echo consumption sentinel. `on_select_changed`'s
# guards compare an incoming Select.Changed value against the value each
# Select instance was actually constructed with at ITS compose(), pinned on
# the instance itself (`_mcp_mount_echo_value`) to swallow that
# constructor-triggered echo. A *standing* sentinel would keep swallowing
# any later user selection that happens to match the same value again (e.g.
# an A -> B -> A round trip's final "A" looks identical to the mount echo),
# so once a guard actually consumes an echo it overwrites the tag with this
# unique object instead of leaving the matched value in place -- no real
# Select value can ever equal it, so every subsequent change dispatches.
# All three selects (source, scope, scope-ref) use this per-instance
# pattern: a rail-level slot races across back-to-back recompose
# generations -- the rail recomposes on every resync AND on
# budget-changing resizes (F-057), and an older generation's echo consumed
# against a newer generation's reset slot leaks exactly one bogus
# dispatch.
_ECHO_CONSUMED = object()

MCP_RAIL_ROW_PREFIX = "mcp-rail-row-"
# A4: wide enough that the built-in server's full label ("tldw_chatbook
# (built-in)", 24 chars) always fits without an ellipsis at the rail's real
# rendered width (min-width 24, typically ~35-40 cols at the 3fr share of a
# 140-col QA viewport) -- the old budget of 22 truncated it even though the
# rail had room.
_MAX_ROW_LABEL = 36
# F-057: everything around the truncated label on one rail row's rendered
# line -- readiness glyph + space (2), the right-side count field
# (space + 3), and the row Button's own horizontal padding (2). The label
# truncation budget at narrow widths is the rail's rendered width minus
# this chrome, so the line (ellipsis included) always FITS the row instead
# of being cropped mid-word by the Button's clipping.
_ROW_CHROME = 8
# "All servers" carries no readiness glyph but must still line up under the
# same left edge glyph-prefixed rows use ("<glyph> label...", a 2-char-wide
# gutter) -- see _row_label's return and MCPRail.compose()'s "All servers"
# row.
_ALL_SERVERS_GUTTER = "  "


def _present_states_legend(snapshots: list[ReadinessSnapshot]) -> str:
    """Compact glyph legend for the states currently present in the rail.

    task-2243: rail rows show glyph+name only, and decoding them required
    the dim, bottom-of-canvas Servers-mode legend (which wraps at ~100
    cols). Listing ONLY the present states directly under the "Servers"
    heading keeps the decode short in the common case (a fresh install
    reads just "◦ off (opt-in) · ⌂ built-in"). A per-row state-word badge
    was considered and rejected: at the rail's real rendered widths
    (~24-46 cols) a word column of up to 15 chars ("Needs attention")
    would re-truncate the very labels A4 widened the budget to fit (the
    built-in's 24-char label + glyph + count already nearly fills the
    row), and it would have to thread through the F-057 width-aware
    truncation machinery. Derived from STATE_GLYPHS/STATE_LABELS (in
    STATE_GLYPHS order) so the line can never drift from the rows it
    decodes; the ⌂ built-in marker is explained whenever a built-in row
    is present (same copy the Servers-mode legend uses).
    """
    present = {snap.state for snap in snapshots}
    parts = [
        f"{glyph} {STATE_LABELS[state].lower()}"
        for state, glyph in STATE_GLYPHS.items()
        if state in present
    ]
    if any(snap.source == "builtin" for snap in snapshots):
        parts.append("⌂ built-in")
    return " · ".join(parts)


def _row_prefix_and_label(
    snapshot: ReadinessSnapshot, *, budget: int = _MAX_ROW_LABEL
) -> tuple[str, str]:
    """Truncated, UNESCAPED `(prefix, label)` for a rail row.

    Shared by `_row_label()`'s final formatting and `MCPRail.compose()`'s
    per-call adaptive pad-width measurement (A6) -- both need EXACTLY the
    same truncation, so the logic lives in one place rather than two copies
    that could drift. It still runs twice per row per compose() (once to
    measure `pad_width`, once inside `_row_label()` itself) -- rail row
    counts are small and this isn't a hot path, so that repeat call was not
    worth the extra parameter-threading to avoid.

    `budget` (F-057) caps the RENDERED width of `prefix + label` -- the
    fixed `_MAX_ROW_LABEL` at wide rail widths (pre-F-057 behavior), or the
    rail's actual rendered width minus `_ROW_CHROME` when that's narrower,
    so the truncated line (with its "..." marker) fits the row instead of
    being cropped mid-word by the Button's own clipping.

    Deliberately returns the label BEFORE `escape_markup()` -- callers that
    only need the rendered width (`len(prefix) + len(label)`, i.e. this
    function's return value) must measure it here, not on the escaped
    string `_row_label()` actually embeds. `escape_markup()` inserts one
    backslash per markup-special character (e.g. `[` -> `\\[`), and
    Button's own markup parsing consumes exactly that backslash again when
    displaying the label -- so the escaped string is longer than what
    actually renders, and padding/measuring against IT (rather than this
    unescaped, truncated text) misaligns any row whose label contains a
    markup-special character against its sibling rows.
    """
    # snapshot.label is user-controlled (local profile ids, server-reported
    # names) and is rendered through Button, which parses str labels as Rich
    # markup — escape it (in `_row_label()`, at format time) so a profile id
    # like "[bold red]x" can't inject styling or break layout.
    label = snapshot.label
    prefix = "⌂ " if snapshot.source == "builtin" else ""
    label_budget = max(4, budget - len(prefix))
    if len(label) > label_budget:
        label = f"{label[: label_budget - 3].rstrip()}..."
    return prefix, label


def _row_label(
    snapshot: ReadinessSnapshot,
    pad_width: int = _MAX_ROW_LABEL,
    *,
    budget: int = _MAX_ROW_LABEL,
) -> str:
    """Format one rail row's full label, including the glyph and count.

    Args:
        snapshot: The row's readiness snapshot.
        pad_width: A6 -- the column width to left-justify `prefix+label`'s
            RENDERED (post-escape-round-trip) width to before the count
            field. `MCPRail.compose()` passes the per-call adaptive width
            (the longest current rendered label width among its rows) so a
            short label's count isn't stranded far right of a long label's;
            this defaults to the old fixed truncation budget for a
            standalone/direct call (e.g. a unit test exercising truncation
            in isolation, with no sibling rows to adapt to).
        budget: F-057 -- the truncation budget forwarded to
            `_row_prefix_and_label()`; defaults to the fixed
            `_MAX_ROW_LABEL` for standalone/direct calls.
    """
    prefix, label = _row_prefix_and_label(snapshot, budget=budget)
    # Pad using the RENDERED width (prefix + unescaped label), not the
    # escaped string's own (longer, for any markup-special character)
    # length -- see `_row_prefix_and_label()`'s docstring. Python's
    # `f"{s:<{n}}"` format pads based on `len(s)`, which would be wrong
    # here once `s` is escaped, so the padding is built manually instead.
    visual_width = len(prefix) + len(label)
    pad = " " * max(0, pad_width - visual_width)
    text = f"{prefix}{escape_markup(label)}{pad}"
    # Task 11 (UX-inputs polish): the tool count sits in a fixed right-side
    # column instead of trailing the label at a variable offset -- the name
    # is left-justified to `pad_width`, and the count is right-aligned in a
    # fixed 3-char field (blank, not "0", when no count has ever been
    # discovered) so counts form one scannable column down the rail instead
    # of drifting with label length.
    count = "" if snapshot.tool_count is None else str(snapshot.tool_count)
    return f"{STATE_GLYPHS[snapshot.state]} {text} {count:>3}"


class MCPRail(RecomposeCaptureGuard, Vertical):
    """Left rail for the MCP workbench. Index-based row ids; keys in a list.

    ``sync_state()`` (below) drives ``self.refresh(recompose=True)`` on every
    resync; ``RecomposeCaptureGuard`` (task-637) keeps a stale mouse capture
    from leaking app-wide when that recompose tears down a row/Select the
    mouse is still captured on (same bug class as task-627's
    ``BaseAppScreen`` fix, one level down: the rail isn't a screen, so it
    never inherited that guard).
    """

    BUNDLED_CSS = """
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
    /* F-060: the zero-servers empty state reads as quiet guidance, not a
    row -- dim it and align its left edge with the rows' padding. */
    #mcp-rail-empty {
        color: $text-muted;
        padding: 0 1;
    }
    /* task-2243: the in-rail state legend decodes the rows' glyphs right
    under the "Servers" heading -- same quiet dim tier as the empty state,
    hugging its own (possibly wrapped) content. */
    #mcp-rail-state-legend {
        height: auto;
        min-height: 0;
        color: $text-muted;
        padding: 0 1;
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
        # a Select's own constructor value as part of mounting it; the
        # per-instance `_mcp_mount_echo_value` tags (set in compose(), same
        # T9 pattern as the source select) let `on_select_changed()`
        # recognize and drop that mount-echo instead of forwarding it as a
        # real user-driven ScopeChanged. Per-INSTANCE, not a rail-level
        # slot: the rail recomposes on every resync AND on budget-changing
        # resizes (F-057), and a rail-level slot races across back-to-back
        # generations (an older generation's echo consumed against a newer
        # generation's reset slot leaks exactly one bogus dispatch -- the
        # scope-select storm F-057's resize recompose exposed).
        # F-057: the row truncation budget the current compose() used
        # (`_MAX_ROW_LABEL` before the first layout, when the rail's width
        # is still unknown) -- `on_resize` recomposes only when the width-
        # derived budget actually CHANGES, so terminals wide enough for the
        # fixed budget never pay a recompose (or the mount-echo/render
        # churn that comes with it) for a resize that changes nothing.
        self._row_budget: int = _MAX_ROW_LABEL

    def on_resize(self) -> None:
        """F-057: re-truncate row labels when the width-derived budget
        changes (terminal resize into/out of narrow widths) -- otherwise a
        width the first pre-layout compose() couldn't know stays stuck at
        the `_MAX_ROW_LABEL` fallback budget, and rows crop mid-word."""
        width = self.region.width
        if width <= 0:
            return
        budget = max(8, min(_MAX_ROW_LABEL, width - _ROW_CHROME))
        if budget != self._row_budget:
            self._row_budget = budget
            self.refresh(recompose=True)

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
        source_value = self.source if self.source in ("local", "server") else "local"
        source_select = Select(
            [("Local", "local"), ("Server", "server")],
            id="mcp-rail-source",
            allow_blank=False,
            value=source_value,
        )
        # T9 (P4) mount-echo guard for the SOURCE select -- per-INSTANCE
        # (now shared by all three selects; the scope selects used a
        # rail-level slot until F-057's resize recompose made its race
        # fire), because this select's echo can be processed AFTER a
        # newer compose() generation has already been scheduled (verified
        # empirically: the destination-shell restore test's saved "server"
        # source was silently reverted to "local" by exactly this race).
        # The old `event.value != self.source` comparison alone can't catch
        # it: by the time the echo is processed, `self.source` has moved on
        # (e.g. a restored view state switched it to "server"), so the
        # stale "local" echo looks like a genuine user change. A rail-level
        # single slot would have the same hole across generations (each
        # compose() would reset it while an older generation's echo is
        # still queued); pinning the constructed value on the Select
        # instance itself makes the guard track exactly the widget whose
        # mount posted the echo.
        source_select._mcp_mount_echo_value = source_value
        yield source_select
        yield Static("Servers", classes="destination-section mcp-rail-heading")
        # task-2243: decode the rows' state glyphs inline, right under the
        # heading, instead of leaving the decode to the dim bottom-of-
        # canvas Servers-mode legend -- present states only, so the line
        # stays short (a fresh install reads "◦ off (opt-in) · ⌂ built-in").
        # Nothing to decode at zero servers: the F-060 empty state below
        # stands alone.
        if self.snapshots:
            yield Static(
                _present_states_legend(self.snapshots),
                id="mcp-rail-state-legend",
                markup=False,
            )
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
        # F-060: at zero servers the rail needs an empty state in plain
        # language pointing at the Add-server action, not a bare "All
        # servers" row over nothing.
        if not self.snapshots:
            yield Static(
                "No servers yet — Add server to connect one.",
                id="mcp-rail-empty",
                markup=False,
            )
        # A6: the count column's pad width is computed per compose() call as
        # the longest CURRENT row's RENDERED width (post-truncate, still
        # unescaped -- see `_row_prefix_and_label()`'s docstring for why
        # measuring the escaped string instead would misalign any row whose
        # label contains a markup-special character) among this rail's rows,
        # not the fixed `_MAX_ROW_LABEL` truncation budget -- a short label
        # (e.g. "docs") no longer strands its tool count 30+ columns right of
        # where a long label's count lands. The truncation budget itself is
        # unchanged; this only affects the padding applied AFTER truncation.
        # F-057: the truncation budget IS width-aware now -- at narrow
        # rendered widths (below ~120-col terminals) it is the rail's own
        # width minus `_ROW_CHROME`, so rows truncate with an ellipsis that
        # FITS the row instead of being cropped mid-word. Width 0 (first
        # compose, pre-layout) falls back to `_MAX_ROW_LABEL`; `on_resize`
        # recomposes once the real width is known.
        layout_width = self.region.width
        budget = (
            _MAX_ROW_LABEL
            if layout_width <= 0
            else max(8, min(_MAX_ROW_LABEL, layout_width - _ROW_CHROME))
        )
        self._row_budget = budget
        pad_width = max(
            (
                len(f"{prefix}{label}")
                for prefix, label in (
                    _row_prefix_and_label(snap, budget=budget)
                    for snap in self.snapshots
                )
            ),
            default=0,
        )
        for index, snap in enumerate(self.snapshots, start=1):
            # Task 11: each row carries its readiness state's CSS class
            # (STATE_CSS_CLASSES, Task 3) so it can be colored by status --
            # constructed fresh on every compose() (sync_state() always
            # recomposes), so there is no stale class from a prior render to
            # remove first.
            row = Button(
                _row_label(snap, pad_width, budget=budget),
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
                scope_select = Select(
                    scope_options,
                    id="mcp-rail-scope-select",
                    allow_blank=False,
                    value=scope_value,
                )
                # Per-instance mount-echo guard, same T9 pattern as the
                # source select above (and same rationale: echoes can be
                # processed after a newer compose() generation replaced the
                # rail-level slot they would have been compared against).
                scope_select._mcp_mount_echo_value = scope_value
                yield scope_select
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
                    if (
                        self.scope_ref_value
                        and self.scope_ref_value in ref_option_values
                    ):
                        ref_value = self.scope_ref_value
                    else:
                        # Restored/stale value not among the offered scope-ref
                        # options (or no value at all) — no selection.
                        ref_value = Select.NULL
                else:
                    ref_options = [("No scope entities", Select.BLANK)]
                    ref_value = Select.BLANK
                scope_ref_select = Select(
                    ref_options,
                    id="mcp-rail-scope-ref",
                    value=ref_value,
                    disabled=not self.scope_ref_options,
                )
                # Same per-instance mount-echo guard as the scope select.
                scope_ref_select._mcp_mount_echo_value = ref_value
                if not self.scope_ref_options:
                    # F-060: a disabled Select with no explanation reads as
                    # broken -- say why there is nothing to pick.
                    scope_ref_select.tooltip = (
                        "No scope entities to pick for this scope — the "
                        "select stays disabled until one exists."
                    )
                yield scope_ref_select
        # (No scope selects render for non-server sources; the per-instance
        # echo tags only ever exist on the instances that have them, so the
        # handlers below need no rail-level state to consult or reset.)

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
            # T9 (P4): one-shot per-instance mount-echo guard -- see the
            # comment on `_mcp_mount_echo_value` in compose(). The first
            # Changed a Select instance ever posts is its constructor echo
            # (a user can't interact before mount), so consuming at most
            # one matching event per instance drops exactly the echo while
            # a later genuine A -> B -> A round trip still dispatches.
            echo_value = getattr(event.select, "_mcp_mount_echo_value", _ECHO_CONSUMED)
            if echo_value is not _ECHO_CONSUMED:
                event.select._mcp_mount_echo_value = _ECHO_CONSUMED
                if event.value == echo_value:
                    return
            if event.value in ("local", "server") and event.value != self.source:
                self.post_message(self.SourceChanged(str(event.value)))
        elif select_id == "mcp-rail-scope-select":
            event.stop()
            # Mount-echo guard (C1): the value this Select was actually
            # constructed with (post-clamp) at ITS compose. Comparing
            # against `self.scope_value` directly would miss this — that
            # attribute holds the true, un-clamped tracked scope, which can
            # differ from what was actually displayed/selected.
            # Same one-shot per-instance pattern as the source select (T9):
            # a rail-level slot races across back-to-back recompose
            # generations (the F-057 resize recompose made that race fire),
            # pinning the constructed value on the instance itself cannot.
            echo_value = getattr(event.select, "_mcp_mount_echo_value", _ECHO_CONSUMED)
            if echo_value is not _ECHO_CONSUMED:
                event.select._mcp_mount_echo_value = _ECHO_CONSUMED
                if event.value == echo_value:
                    return
            self.post_message(self.ScopeChanged(str(event.value), None))
        elif select_id == "mcp-rail-scope-ref":
            event.stop()
            # Same one-shot per-instance mount-echo guard as above, for the
            # scope-ref select.
            echo_value = getattr(event.select, "_mcp_mount_echo_value", _ECHO_CONSUMED)
            if echo_value is not _ECHO_CONSUMED:
                event.select._mcp_mount_echo_value = _ECHO_CONSUMED
                if event.value == echo_value:
                    return
            # Both our synthetic placeholder sentinel (Select.BLANK, used when
            # there are no ref options) and the auto-added blank row
            # (Select.NULL, present whenever allow_blank=True) mean "no
            # selection" here.
            is_blank = event.value is Select.BLANK or event.value is Select.NULL
            ref = None if is_blank else str(event.value)
            self.post_message(self.ScopeChanged(self.scope_value, ref))
