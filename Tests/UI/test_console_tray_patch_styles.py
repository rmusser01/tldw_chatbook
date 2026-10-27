"""TASK-33620.15.20: the Conversations tray's run-marker patch builds no widget.

TASK-36012 patches a send's "Running" marker into the mounted tray instead of
rebuilding it. It used to decide what to write by diffing two off-DOM
composes of the tray (the state shown and the next one), which were then
dropped, never mounted.

Measured live on origin/dev 3be493413a (2026-10-27, 160x45, Anthropic
claude-haiku-4-5, 12 probed sends): in 8 of 12 sends the longest UI-loop
callback between Enter and the provider POST was the first 0.2 s poll tick
after acceptance (+260-355 ms, 40-57 ms). Inside it, ``App.update_styles``
took 21-34 ms for 64-69 nodes: every widget those two composes built,
restyled as each Button constructor and compose helper set a class. Headless,
with the app stylesheets, one patched sync built 96 widgets and cost about
100 ms of CPU at 12 rows (264 widgets, about 290 ms at 40 rows).

The patch now decides from the row data: compose records, per browser row it
builds, the run-dependent values it wrote (``_BrowserRowPaint``), and the
patch compares that record with the paint the incoming row gives. Nothing is
composed, so nothing is built or styled off the DOM.

The A/B cases below are parity pins: each delta in a patchable browser field
lands what a fresh build of the new state shows, whether the patch takes it
or rebuilds. They hold on origin/dev's compose diff as well; their job is to
keep the data-level decision honest as the compose changes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import pytest
from textual.app import App, ComposeResult
from textual.css.stylesheet import Stylesheet
from textual.widget import Widget
from textual.widgets import Button

# The UI fixture imports the app after selecting a per-test profile. Import it
# under the collection bootstrap first, before config freezes its source owner.
import tldw_chatbook.app  # noqa: F401
from tldw_chatbook.Widgets import glyph_fallback
from Tests.UI.test_console_run_tick_layout_requests import _LayoutRequests, _names
from Tests.UI.test_console_send_transition_layout_requests import (
    _browser_delta,
    _row,
    _settled_tray,
    _tray_state,
    _TrayHost,
    _within,
    _witness,
)
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    _PATCHABLE_BROWSER_FIELDS,
    ConsoleWorkspaceContextTray,
)
from tldw_chatbook.Workspaces.conversation_attention import ConversationAttentionFact
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserGroup,
    ConsoleConversationBrowserSection,
)

pytestmark = [pytest.mark.bootstrap_profile]


def _record_applies(monkeypatch) -> tuple[list[Widget], list[Widget]]:
    """Record every ``Stylesheet.apply``, split by whether the node is in the DOM."""
    detached: list[Widget] = []
    attached: list[Widget] = []
    original = Stylesheet.apply

    def apply(self, node, *args, **kwargs):
        in_dom = isinstance(node, App) or node._parent is not None
        (attached if in_dom else detached).append(node)
        return original(self, node, *args, **kwargs)

    monkeypatch.setattr(Stylesheet, "apply", apply)
    return detached, attached


def _record_builds(monkeypatch) -> list[Widget]:
    """Record every widget constructed (``Widget.__init__``)."""
    built: list[Widget] = []
    original = Widget.__init__

    def init(self, *args, **kwargs):
        built.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Widget, "__init__", init)
    return built


def _record_composes(monkeypatch) -> list[Widget]:
    composes: list[Widget] = []
    original = ConsoleWorkspaceContextTray.compose

    def counted(self) -> ComposeResult:
        composes.append(self)
        yield from original(self)

    monkeypatch.setattr(ConsoleWorkspaceContextTray, "compose", counted)
    return composes


async def test_running_marker_patch_builds_and_styles_no_throwaway_node(
    monkeypatch,
) -> None:
    """The run-marker patch writes the live row and builds nothing else."""
    host = _TrayHost(_tray_state())
    async with host.run_test(size=(40, 30)) as pilot:
        tray = await _settled_tray(host, pilot)
        composes = _record_composes(monkeypatch)
        built = _record_builds(monkeypatch)
        detached, attached = _record_applies(monkeypatch)
        running = _tray_state(running=True)
        tray.sync_state(running)
        # The patch took the delta: no rebuild is pending.
        assert tray.state == running
        assert not getattr(tray, "_recompose_required", False)
        assert tray._composed_publication_state == running
        # Positive control: the live row now shows the marker, and its colour
        # class write was styled.
        menu = tray.query_one("#console-conversation-actions-0", Button)
        assert menu.has_class("conversation-attention-running")
        assert any(_within(node, tray) for node in attached)
        assert composes == [], f"the patch composed the tray {len(composes)} times"
        assert built == [], (
            f"the patch built {len(built)} throwaway widgets: "
            f"{sorted({type(node).__name__ for node in built})}"
        )
        assert detached == [], (
            f"{len(detached)} nodes outside the DOM were styled: "
            f"{sorted({type(node).__name__ for node in detached})}"
        )
        # Positive control for the recorders: a delta that moves rows (an
        # approval adds an activity line) rebuilds, and the rebuild builds.
        tray.sync_state(_tray_state(approval=True))
        for _ in range(4):
            await pilot.pause()
        assert len(composes) == 1
        assert built


# -- Parity: every patchable browser field lands what a rebuild shows ---------


def _grouped(*, collapsed: bool = False, running: bool = False):
    """One workspace group holding rows 0 and 1, row 0 optionally running."""
    state = _tray_state()
    rows = (_row(0, running=running), _row(1))
    facts = (
        (ConversationAttentionFact(kind="running", label="Running"),) if running else ()
    )
    group = ConsoleConversationBrowserGroup(
        group_id="workspace:research",
        label="Research Lab",
        collapsed=collapsed,
        rows=rows,
        count=len(rows),
        workspace_id="research",
        run_marker="⟳" if running else "",
        attention=facts,
    )
    section = ConsoleConversationBrowserSection(
        section_id="workspaces",
        label="Workspaces",
        collapsed=False,
        groups=(group,),
        count=len(rows),
    )
    return replace(
        state,
        conversation_browser=replace(state.conversation_browser, sections=(section,)),
    )


def _row0(**changes):
    return lambda: _browser_delta(
        rows=lambda index, row: replace(row, **changes) if index == 0 else row
    )


def _section(**changes):
    def build():
        state = _tray_state()
        browser = state.conversation_browser
        section = replace(browser.sections[0], **changes)
        return replace(
            state, conversation_browser=replace(browser, sections=(section,))
        )

    return build


#: name -> (shown state, next state, whether the patch takes it)
_PAINT_DELTAS = {
    "row running": (_tray_state, lambda: _tray_state(running=True), True),
    "row icon": (_tray_state, _row0(icon="🔬"), True),
    "row colour": (_tray_state, _row0(color="red"), True),
    # The browser row never shows its own run marker (its menu icon reads the
    # attention facts), so nothing visible moves.
    "row run marker only": (_tray_state, _row0(run_marker="⟳"), True),
    # A section header shows its capped marker in a content-sized title, so
    # the new copy needs a layout: the tray rebuilds.
    "section capped marker": (_tray_state, _section(capped_run_marker="⟳"), False),
    # An expanded group shows each row's own marker, so its header is unchanged.
    "grouped row running": (_grouped, lambda: _grouped(running=True), True),
    # A collapsed group carries its rows' marker on its header: a rebuild.
    "collapsed group marker": (
        lambda: _grouped(collapsed=True),
        lambda: _grouped(collapsed=True, running=True),
        False,
    ),
}


def _changed_patchable_fields(old, new, found: set[str]) -> set[str]:
    if old == new:
        return found
    if isinstance(old, tuple):
        for left, right in zip(old, new):
            _changed_patchable_fields(left, right, found)
        return found
    if dataclasses.is_dataclass(old):
        for field in dataclasses.fields(old):
            left, right = getattr(old, field.name), getattr(new, field.name)
            if left == right:
                continue
            if field.name in _PATCHABLE_BROWSER_FIELDS:
                found.add(field.name)
            else:
                _changed_patchable_fields(left, right, found)
    return found


def test_paint_deltas_cover_every_patchable_browser_field() -> None:
    """A field the patch admits needs an A/B case, or the record can drift."""
    covered: set[str] = set()
    for shown, incoming, _patched in _PAINT_DELTAS.values():
        _changed_patchable_fields(shown(), incoming(), covered)
    assert covered == set(_PATCHABLE_BROWSER_FIELDS)


def _styled_witness(tray: Widget) -> list:
    """``_witness`` plus what it leaves out: each label's and line's styling.

    ``_witness`` compares plain text, so a glyph that keeps its text and
    changes colour (a row's colour) would pass unseen.
    """
    labels = [(w.id, w.label.plain, tuple(w.label.spans)) for w in tray.query(Button)]
    lines = [
        tuple((segment.text, str(segment.style)) for segment in strip)
        for strip in tray.render_lines(tray.region.reset_offset)
    ]
    return [*_witness(tray), labels, lines]


async def _remount(host, pilot, state) -> ConsoleWorkspaceContextTray:
    """Replace the host's tray with a fresh one built from ``state``."""
    await host.tray.remove()
    host.tray = ConsoleWorkspaceContextTray(
        state, content="conversations", id="console-workspace-context"
    )
    await host.mount(host.tray)
    return await _settled_tray(host, pilot)


async def test_browser_paint_deltas_match_a_rebuild(monkeypatch) -> None:
    """Geometry, paint, classes and tooltips equal a fresh build of the state.

    One app serves every case (a fresh tray per step), so the A/B costs one
    stylesheet parse rather than two per case.
    """
    requests = _LayoutRequests(monkeypatch)
    failures: list[str] = []
    host = _TrayHost(_tray_state())
    async with host.run_test(size=(40, 30)) as pilot:
        await _settled_tray(host, pilot)
        for delta, (shown, incoming, patched) in _PAINT_DELTAS.items():
            tray = await _remount(host, pilot, shown())
            with requests.counting() as calls:
                tray.sync_state(incoming())
            rebuilt = tray in calls
            for _ in range(4):
                await pilot.pause()
            if tray.state != incoming():
                failures.append(f"{delta}: the tray did not adopt the state")
            result = _styled_witness(tray)
            built = _styled_witness(await _remount(host, pilot, incoming()))
            if rebuilt is patched:
                failures.append(
                    f"{delta}: expected a {'patch' if patched else 'rebuild'}, "
                    f"layout requests {_names(calls)}"
                )
            if result != built:
                failures.append(f"{delta}: the result differs from a fresh build")
    assert failures == []


def _drift_menu_label(tray) -> None:
    tray.query_one("#console-conversation-actions-0", Button).label = "?"


def _drift_menu_class(tray) -> None:
    menu = tray.query_one("#console-conversation-actions-0", Button)
    menu.add_class("conversation-attention-running")


def _drift_menu_tooltip(tray) -> None:
    menu = tray.query_one("#console-conversation-actions-0", Button)
    menu.tooltip = "edited elsewhere"


def _drift_row_tooltip(tray) -> None:
    row = tray.query_one("#console-workspace-conversation-0", Button)
    row.tooltip = "edited elsewhere"


async def test_rows_that_drifted_from_their_record_are_rebuilt(monkeypatch) -> None:
    """A mounted row that no longer shows what compose wrote is rebuilt.

    So is every row after a glyph-mode flip: the record's paint was written
    in the old mode, and so was what it does not cover (the star prefix, the
    toggles, the ASCII action width).
    """
    requests = _LayoutRequests(monkeypatch)
    failures: list[str] = []
    host = _TrayHost(_tray_state())
    async with host.run_test(size=(40, 30)) as pilot:
        await _settled_tray(host, pilot)
        for drift in (
            _drift_menu_label,
            _drift_menu_class,
            _drift_menu_tooltip,
            _drift_row_tooltip,
        ):
            tray = await _remount(host, pilot, _tray_state())
            drift(tray)
            with requests.counting() as calls:
                tray.sync_state(_tray_state(running=True))
            if tray not in calls:
                failures.append(f"{drift.__name__}: patched a drifted row")
            for _ in range(4):
                await pilot.pause()
            menu = tray.query_one("#console-conversation-actions-0", Button)
            row = tray.query_one("#console-workspace-conversation-0", Button)
            if not (
                menu.has_class("conversation-attention-running")
                and "Running" in str(menu.tooltip)
                and "Running" in str(row.tooltip)
                and menu.label.plain != "?"
            ):
                failures.append(f"{drift.__name__}: the row does not show the run")
        # Last, since it changes the mode for every later compose.
        tray = await _remount(host, pilot, _tray_state())
        monkeypatch.setattr(glyph_fallback, "_ASCII_MODE", True)
        with requests.counting() as calls:
            tray.sync_state(_tray_state(running=True))
        if tray not in calls:
            failures.append("glyph mode flip: patched rows composed in another mode")
        for _ in range(4):
            await pilot.pause()
        menu = tray.query_one("#console-conversation-actions-0", Button)
        if menu.label.plain != "[running]":
            failures.append(f"glyph mode flip: the action shows {menu.label.plain!r}")
    assert failures == []
