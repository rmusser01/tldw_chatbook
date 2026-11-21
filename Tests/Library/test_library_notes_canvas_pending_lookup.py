"""TASK-34000.25 fix round 2: ``_pending_widget_lookup`` reads Textual's
compose-time child registry, so a Textual change to it fails HERE by name.

``LibraryNotesCanvas._compose_editor`` gates every editor surface on its
last line, over the composed-but-unmounted tree, through this lookup: a
child yielded under ``with Container():`` must be reachable from the
container before anything mounts (``compose_add_child`` ->
``_pending_children`` in Textual 8.2.8). Without this pin a Textual bump
would surface as an app crash in the first UI test that opens a note.

PR #3055 review (Important 1): the pin above is the LOUD half. The second
test is the soft half -- when the lookup cannot resolve the gated nodes
(an empty registry is exactly what a renamed attribute yields), compose
must not raise and must not skip the gates: it logs once at WARNING and
gates the mounted tree one refresh later.

TASK-34789: the third test is the same rule for the canvas's OWN recompose.
While it removes and remounts its children the header row can be in the DOM
without its children, and a shell resize handled on the screen's pump in
that window (``apply_work_width``) must not raise -- the shape it asked for
lands on the new children when the recompose completes. The fourth covers
the other half of that window (the row still attached, its children already
pruned), for a recompose and for the canvas being removed outright.

TASK-41703: the last test is the press half of that window, on the
recompose seam every Library canvas shares (``PostRecomposeCallback``): a
press still queued below a canvas child the recompose removes reaches the
screen.

TASK-42014 moved that keeper into ``Widget.recompose`` (the package's Textual
shims), so the tests after it hold a press across every other route a
container's recompose takes: ``refresh(recompose=True)``, a recompose
reactive, an awaited ``recompose()``, an override chained to ``super()``, a
screen, a container removed by its parent's recompose, and the two overrides
that remove their children themselves (``SettingsRegion`` and the Console
tray's kept filter row). A forwarded press runs only the handlers whose
selector the button matches on its own, so two more tests hold a press across
the recompose of the two containers whose handlers were switched to a class
on the button (``ExternalModelView``, ``SpeechTTSSettingsPanel``).

TASK-42150 moved the keeper again, to ``App._prune``, which every removal goes
through: two more routes hold a press across a child swap that is not a
recompose (``remove_children()`` or ``remove()``, then ``mount()``), the
message tests hold Enter in an Input and on a ListView across a recompose, a
Select's mount echo from a removed generation is not handed on, a press below
a popped screen still never reaches the App, and the last test pins the
install. Since review r19rc-rev1 the holds are driven by real input (a click,
or Enter on the focused sender).

Fast lane: a bare Textual app (``run_test``), no product app boot; the
fallback and recompose tests mount one ``LibraryNotesCanvas`` under the app
stylesheets (``ConsolidatedCSSApp``) because their claims are paint-level.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

import pytest
from loguru import logger
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.Widgets.Library.test_library_notes_canvas import _editor_state
from tldw_chatbook import QUEUED_MESSAGE_KEEPER
from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL
from tldw_chatbook.STT.parakeet_sources import (
    ParakeetSourceKey,
    ParakeetSourcePreference,
    ParakeetSourceRecord,
)
from tldw_chatbook.UI.Screens.model_external_view import ExternalModelView
from tldw_chatbook.UI.Screens.settings_screen import SettingsRegion
from tldw_chatbook.UI.Speech.settings_speech_tts import load_global_speech_tts_state
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
    SpeechTTSSettingsPanel,
)
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    _CONVERSATION_FILTER_ROW_ID,
    _KeepsConversationFilterRow,
)
from tldw_chatbook.Widgets.Library import library_notes_canvas as canvas_module
from tldw_chatbook.Widgets.Library.library_canvas_sync import PostRecomposeCallback
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard
from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    LibraryNotesCanvas,
    _pending_widget_lookup,
)


class _Probe(Static):
    """Composes a nested tree and snapshots the lookup before mounting."""

    found: dict[str, object] = {}
    missing: type[Exception] | None = None

    def compose(self) -> ComposeResult:
        roots = []
        with Vertical(id="outer") as outer:
            roots.append(outer)
            with Horizontal(id="row"):
                yield Button("One", id="one")
                yield Button("Two", id="two")
            yield Static("leaf", id="leaf")
        lookup = _pending_widget_lookup(roots)
        _Probe.found = {
            selector: lookup(selector) for selector in ("#outer", "#row", "#one", "#two", "#leaf")
        }
        try:
            lookup("#absent")
        except NoMatches as exc:
            _Probe.missing = type(exc)
        # No re-yield: a closed ``with`` container is composed by its own
        # ``__exit__`` -- exactly how ``_compose_editor``'s ``root()`` only
        # records the node (yielding it again mounts it twice).


class _App(App):
    def compose(self) -> ComposeResult:
        yield _Probe()


@pytest.mark.asyncio
async def test_pending_lookup_finds_every_child_yielded_under_a_with_container():
    _Probe.found = {}
    _Probe.missing = None
    app = _App()
    async with app.run_test(size=(40, 10)):
        assert set(_Probe.found) == {"#outer", "#row", "#one", "#two", "#leaf"}
        # The same objects that then mounted -- not copies, not stale.
        for selector, widget in _Probe.found.items():
            assert app.query_one(selector) is widget, selector
        assert _Probe.missing is NoMatches


class _EditorCanvasApp(ConsolidatedCSSApp):
    """One editor-mode canvas under the app stylesheets (paint-level claim)."""

    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]

    def compose(self) -> ComposeResult:
        shell = Vertical(id="library-shell-grid")
        shell.styles.width = 100
        shell.styles.height = 30
        with shell:
            yield LibraryNotesCanvas(mode="editor", presentation_state=_editor_state())


@pytest.mark.asyncio
async def test_an_empty_pending_lookup_never_aborts_compose_and_gates_after_one_refresh(
    monkeypatch,
):
    """PR #3055 review (Important 1): a lookup that resolves nothing -- the
    shape a renamed Textual registry attribute produces -- must not raise
    inside ``compose`` (a compose exception is the nav-freeze shape) and
    must not silently skip the gates. It logs ONCE at WARNING and the same
    gates land on the mounted tree after one refresh: the Info and Preview
    surfaces of an Edit-region note are not painted.

    RED on bbc7c2c6d2: ``compose`` raised ``NoMatches`` for the first gated
    selector and the app never came up.
    """
    original = canvas_module._pending_widget_lookup
    monkeypatch.setattr(
        canvas_module, "_pending_widget_lookup", lambda roots: original(())
    )
    monkeypatch.setattr(LibraryNotesCanvas, "_pending_lookup_warned", False, raising=False)
    warnings: list[str] = []
    sink = logger.add(
        lambda message: warnings.append(message.record["message"]),
        level="WARNING",
        filter=lambda record: "surface gating" in record["message"],
    )
    try:
        app = _EditorCanvasApp()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.pause()
            canvas = app.query_one(LibraryNotesCanvas)
            painted = app.screen._compositor.visible_widgets
            editor = canvas.query_one("#library-note-editor-region")
            context = canvas.query_one("#library-note-context-region")
            preview = canvas.query_one("#library-note-preview-region")
            assert editor.display is True
            assert editor in painted, "the Edit surface is not painted"
            assert context.display is False
            assert context not in painted, "the Info surface is painted"
            assert preview.display is False
            assert preview not in painted, "the Preview surface is painted"
            assert len(warnings) == 1, warnings
            # A second compose under the same failure stays quiet: once.
            await canvas.remove()
            await app.query_one("#library-shell-grid").mount(
                LibraryNotesCanvas(mode="editor", presentation_state=_editor_state())
            )
            await pilot.pause()
            await pilot.pause()
            again = app.query_one(LibraryNotesCanvas)
            assert again.query_one("#library-note-context-region").display is False
            assert len(warnings) == 1, warnings
    finally:
        logger.remove(sink)


#: Either side of ``_HEADER_ONE_ROW_MIN_WIDTH`` (100): the editor header is
#: one row at the first width and stacked at the second.
ONE_ROW_WORK_WIDTH = 120
STACKED_WORK_WIDTH = 60


@pytest.mark.asyncio
async def test_a_resize_landing_mid_recompose_is_applied_when_the_recompose_completes():
    """TASK-34789: ``apply_work_width`` on a half-remounted canvas never raises.

    The half-mounted state is driven directly, not by timing: the resize is
    delivered from inside the recompose's own ``mount_all``, after the new
    top-level children are registered and before their children exist --
    the state a notes-tree or trash reload leaves the work pane in for as
    long as its recompose runs, while the screen's pump is free to handle
    ``AdaptiveReaderShellResized``.

    RED on dev 1115b492f8: ``NoMatches: No nodes match
    '#library-note-status' on LibraryNotesCanvas(...)`` from
    ``apply_compact_presentation`` -- the UI Fast Lane crash of
    ``test_new_note_with_saved_content_survives_a_rail_round_trip``.
    """
    app = _EditorCanvasApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        canvas = app.query_one(LibraryNotesCanvas)
        canvas.apply_work_width(ONE_ROW_WORK_WIDTH)
        one_row = canvas.query_one("#library-note-header-second-row")
        assert one_row.styles.layout.name == "horizontal"

        real_mount_all = canvas.mount_all
        landed: list[str] = []

        def mount_all_then_resize(widgets, **kwargs):
            pending = real_mount_all(widgets, **kwargs)
            if not landed:
                # Registered, not yet awaited: the row is there, its
                # children are still pending.
                assert canvas.query("#library-note-header-second-row")
                assert not canvas.query("#library-note-status")
                landed.append("resized")
                canvas.apply_work_width(STACKED_WORK_WIDTH)
            return pending

        canvas.mount_all = mount_all_then_resize
        canvas.refresh(recompose=True)
        for _ in range(100):
            if landed and canvas.query("#library-note-status"):
                break
            await pilot.pause(0.02)
        await pilot.pause()

        assert landed == ["resized"]
        stacked = canvas.query_one("#library-note-header-second-row")
        assert stacked is not one_row, "the canvas never recomposed"
        # The shape the mid-recompose resize asked for, on the NEW children.
        assert stacked.styles.layout.name == "vertical"
        assert canvas.query_one("#library-note-status").has_class("h-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("teardown", ["recompose", "remove"])
async def test_a_resize_landing_while_the_header_is_torn_down_never_raises(
    monkeypatch, teardown: str
):
    """TASK-34789 (PR #3090 review): the leaf-first half of the same window.

    Children leave before their parent, so the header row is still in the DOM
    after the widgets it held are gone. The resize is delivered from that
    row's own unmount, the one point where that state is certain. For a
    recompose the readiness flag covers it; a canvas that is being removed
    keeps that flag, and only its pruning state says its children are going.

    RED without the pruning half of the guard for ``remove``, and without
    any guard for both: ``NoMatches`` on ``#library-note-heading``.
    """
    seen: list[tuple[bool, bool]] = []
    outcome: list[str] = []

    def on_unmount(self) -> None:
        if self.id != "library-note-header-second-row" or outcome:
            return
        canvas = next(
            (node for node in self.ancestors if isinstance(node, LibraryNotesCanvas)),
            None,
        )
        if canvas is None:
            return
        seen.append(
            (
                bool(canvas.query("#library-note-header-second-row")),
                bool(canvas.query("#library-note-status")),
            )
        )
        try:
            canvas.apply_work_width(STACKED_WORK_WIDTH)
        except NoMatches as exc:
            outcome.append(f"raised: {exc}")
        else:
            outcome.append("returned")

    monkeypatch.setattr(Horizontal, "on_unmount", on_unmount, raising=False)
    app = _EditorCanvasApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        canvas = app.query_one(LibraryNotesCanvas)
        canvas.apply_work_width(ONE_ROW_WORK_WIDTH)
        if teardown == "remove":
            await canvas.remove()
        else:
            canvas.refresh(recompose=True)
        for _ in range(100):
            rebuilt = teardown == "remove" or canvas.query("#library-note-status")
            if outcome and rebuilt:
                break
            await pilot.pause(0.02)
        await pilot.pause()

        # The row was still attached and its children were already gone.
        assert seen == [(True, False)]
        assert outcome == ["returned"]
        if teardown == "recompose":
            # ... and the shape it asked for lands on the new children.
            row = canvas.query_one("#library-note-header-second-row")
            assert row.styles.layout.name == "vertical"


class _PressCanvas(PostRecomposeCallback, Vertical):
    """The Media toolbar's shape: a button one level below a canvas child."""

    def compose(self) -> ComposeResult:
        with Horizontal(id="toolbar"):
            yield Button("type", id="opener")


class _PressScreen(Screen):
    def __init__(self) -> None:
        super().__init__()
        self.handled = 0

    def compose(self) -> ComposeResult:
        yield _PressCanvas(id="canvas")

    @on(Button.Pressed, "#opener")
    def _count(self, event: Button.Pressed) -> None:
        self.handled += 1


class _PressApp(App):
    def get_default_screen(self) -> Screen:
        return _PressScreen()


@pytest.mark.asyncio
async def test_a_press_queued_below_a_removed_canvas_child_still_reaches_the_screen():
    """TASK-41703: a Library canvas recompose does not drop a pending press.

    Textual removes a recompose's old children top down. The toolbar (a
    canvas child) stops taking messages first; the button below it then
    handles its queued press and would bubble it into the closing toolbar,
    where Textual drops it. The press is held in the button until the
    toolbar is closing, so the order is driven, not timed.
    """
    app = _PressApp()
    async with app.run_test(size=(40, 6)) as pilot:
        await pilot.pause()
        screen = app.screen
        canvas = screen.query_one(_PressCanvas)
        toolbar = screen.query_one("#toolbar")
        opener = screen.query_one("#opener", Button)
        held = asyncio.Event()
        release = asyncio.Event()

        async def hold() -> None:
            held.set()
            await release.wait()

        opener.call_later(hold)
        try:
            await asyncio.wait_for(held.wait(), timeout=5)
            opener.press()  # queued in the button, behind ``hold``
            canvas.refresh(recompose=True)
            deadline = asyncio.get_running_loop().time() + 5
            while not toolbar._closing:
                assert asyncio.get_running_loop().time() < deadline, (
                    "the recompose never closed the toolbar"
                )
                await asyncio.sleep(0)
        finally:
            release.set()
        for _ in range(100):
            if screen.handled:
                break
            await pilot.pause(0.02)

        assert screen.handled == 1, "the press below the removed toolbar was dropped"
        assert screen.query_one("#opener") is not opener, "the canvas never recomposed"


def _toolbar(sender: Widget | None = None) -> Horizontal:
    # Not ``sender or ...``: an unmounted ListView has no children, so it is falsy.
    if sender is None:
        sender = Button("type", id="opener")
    return Horizontal(sender, id="toolbar")


class _Container(Vertical):
    """A plain container: the button one level below the child it removes."""

    bump = reactive(0, recompose=True)

    def compose(self) -> ComposeResult:
        yield _toolbar()


class _GuardedContainer(RecomposeCaptureGuard, _Container):
    """An override that runs its own code around ``super().recompose()``."""


class _Outer(Vertical):
    """Its recompose removes the whole ``_Container`` (the press is deeper)."""

    def compose(self) -> ComposeResult:
        yield _Container(id="container")


class _KeptRowTray(_KeepsConversationFilterRow, Vertical):
    """The Console tray's rebuild around a mounted filter row it keeps."""

    def compose(self) -> ComposeResult:
        yield self._retained_filter_row or Horizontal(
            Static("filter"), id=_CONVERSATION_FILTER_ROW_ID
        )
        yield _toolbar()


def _recompose(container):
    container.refresh(recompose=True)


def _bump(container):
    container.bump += 1


def _awaited(container):
    return asyncio.ensure_future(container.recompose())


async def _swap_children(container) -> None:
    await container.remove_children()
    await container.mount(_toolbar())


async def _swap_child(container) -> None:
    await container.query_one("#toolbar").remove()
    await container.mount(_toolbar())


def _on_its_pump(swap):
    """A swap the container runs on its own pump, as its own method would."""
    return lambda container: container.call_later(swap, container)


# route: (the screen's children, the recomposing node (None: the screen),
#         the child its recompose removes above the button, the trigger)
_ROUTES = {
    "refresh": (lambda: [_Container(id="container")], "#container", "#toolbar", _recompose),
    "reactive": (lambda: [_Container(id="container")], "#container", "#toolbar", _bump),
    "awaited": (lambda: [_Container(id="container")], "#container", "#toolbar", _awaited),
    "chained-override": (
        lambda: [_GuardedContainer(id="container")], "#container", "#toolbar", _recompose
    ),
    "screen": (lambda: [_toolbar()], None, "#toolbar", _recompose),
    "removed-by-its-parent": (lambda: [_Outer(id="outer")], "#outer", "#container", _recompose),
    "settings-region": (
        lambda: [SettingsRegion(lambda: [_toolbar()], id="container")],
        "#container",
        "#toolbar",
        _recompose,
    ),
    "console-kept-filter-row": (
        lambda: [_KeptRowTray(id="container")], "#container", "#toolbar", _recompose
    ),
    # TASK-42150: child swaps that are not recomposes.
    "remove-children-then-mount": (
        lambda: [_Container(id="container")],
        "#container",
        "#toolbar",
        _on_its_pump(_swap_children),
    ),
    "remove-then-mount": (
        lambda: [_Container(id="container")],
        "#container",
        "#toolbar",
        _on_its_pump(_swap_child),
    ),
}


class _RouteScreen(Screen):
    def __init__(self, children) -> None:
        super().__init__()
        self._children = children
        self.handled = 0
        self.by_ancestor = 0

    def compose(self) -> ComposeResult:
        yield from self._children()

    @on(Button.Pressed, "#opener")
    def _count(self, event: Button.Pressed) -> None:
        self.handled += 1

    @on(Button.Pressed, "#toolbar Button")
    def _count_by_ancestor(self, event: Button.Pressed) -> None:
        self.by_ancestor += 1


class _RouteApp(App):
    def __init__(self, children) -> None:
        super().__init__()
        self._children = children

    def get_default_screen(self) -> Screen:
        return _RouteScreen(self._children)


async def _hold_a_press_across(
    pilot, opener: Widget, closing, trigger, action="press", user=None
) -> None:
    """Drive real input on ``opener`` (a click, or ``user``: a key), hold the
    message its ``action`` posts in the sender's own queue, run ``trigger``,
    and release the message only once ``closing`` (the child being removed
    above it) is closing, so the order is driven, not timed."""
    held = asyncio.Event()
    release = asyncio.Event()
    real = getattr(opener, action)

    def hold_then_act(*args, **kwargs):
        delattr(opener, action)  # once

        async def hold() -> None:
            held.set()
            await release.wait()

        opener.call_later(hold)  # ahead of the message ``real`` posts
        return real(*args, **kwargs)

    setattr(opener, action, hold_then_act)
    # Pilot waits for every pump to go idle, so it runs beside the hold.
    acting = asyncio.ensure_future(user() if user else pilot.click(opener))
    pending = None
    try:
        await asyncio.wait_for(held.wait(), timeout=5)
        pending = trigger()
        deadline = asyncio.get_running_loop().time() + 5
        while not closing._closing:
            assert asyncio.get_running_loop().time() < deadline, (
                "the recompose never closed the child above the sender"
            )
            await asyncio.sleep(0)
    finally:
        release.set()
    if isinstance(pending, asyncio.Future):  # an awaited ``recompose()``
        await asyncio.wait_for(pending, timeout=5)
    await asyncio.wait_for(acting, timeout=30)


async def _settle_until(pilot, done) -> None:
    for _ in range(100):
        if done():
            return
        await pilot.pause(0.02)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", sorted(_ROUTES))
async def test_every_removal_route_keeps_a_press_queued_below_a_removed_child(route):
    """TASK-42014/42150: no container's recompose or child swap drops a
    pending press.

    Same hold as the canvas test above. Every route reaches the screen's
    ``#opener`` handler. A forwarded press matches only selectors the
    detached button matches alone: when the container's own pump runs the
    recompose, the button is detached before the container handles the
    press, so ``#toolbar Button`` never runs (the two swaps run on that pump
    too, as a container's own method does). An ``awaited`` recompose runs on
    another task, so the container can handle the press before the button
    detaches, and the ancestor selector may run there; it is not pinned.
    """
    children, recomposing, removed, trigger = _ROUTES[route]
    app = _RouteApp(children)
    async with app.run_test(size=(40, 8)) as pilot:
        await pilot.pause()
        screen = app.screen
        container = screen if recomposing is None else screen.query_one(recomposing)
        opener = screen.query_one("#opener", Button)
        await _hold_a_press_across(
            pilot, opener, screen.query_one(removed), lambda: trigger(container)
        )
        await _settle_until(pilot, lambda: screen.handled)

        assert screen.handled == 1, f"the {route} route dropped the press"
        if route != "awaited":
            assert screen.by_ancestor == 0, "an ancestor selector matched"
        assert screen.query_one("#opener") is not opener, "nothing was rebuilt"


# TASK-42150: the other user actions the app handles above their sender, each
# driven by a real key. message: (the sender, the action the key runs)
_MESSAGES = {
    "Input.Submitted": (lambda: Input("typed", id="opener"), "action_submit"),
    "ListView.Selected": (
        lambda: ListView(ListItem(Label("a")), id="opener"),
        "action_select_cursor",
    ),
}


class _MessageContainer(Vertical):
    def __init__(self, sender, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sender = sender

    def compose(self) -> ComposeResult:
        yield _toolbar(self._sender())


class _MessageScreen(Screen):
    def __init__(self, sender) -> None:
        super().__init__()
        self._sender = sender
        self.seen: list[tuple[str, Widget]] = []

    def compose(self) -> ComposeResult:
        yield _MessageContainer(self._sender, id="container")

    @on(Input.Submitted, "#opener")
    @on(ListView.Selected, "#opener")
    def _record(self, event) -> None:
        self.seen.append((type(event).__qualname__, event.control))


class _MessageApp(App):
    def __init__(self, sender) -> None:
        super().__init__()
        self._sender = sender

    def get_default_screen(self) -> Screen:
        return _MessageScreen(self._sender)


@pytest.mark.asyncio
@pytest.mark.parametrize("message", sorted(_MESSAGES))
async def test_a_user_message_queued_below_a_removed_child_is_kept(message):
    """TASK-42150: Enter in an Input or on a ListView, queued below a removed
    child, reaches the screen once, as a press does. Same hold as the press
    routes, driven by the real key."""
    sender, action = _MESSAGES[message]
    app = _MessageApp(sender)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        screen = app.screen
        opener = screen.query_one("#opener")
        opener.focus()
        await pilot.pause()
        screen.seen.clear()
        await _hold_a_press_across(
            pilot,
            opener,
            screen.query_one("#toolbar"),
            lambda: _recompose(screen.query_one("#container")),
            action=action,
            user=lambda: pilot.press("enter"),
        )
        await _settle_until(pilot, lambda: any(c is opener for _, c in screen.seen))

        mine = [name for name, control in screen.seen if control is opener]
        assert mine == [message], f"the recompose dropped the {message}"
        assert screen.query_one("#opener") is not opener, "nothing recomposed"


class _EchoSelect(Select):
    """Holds its pump, queued ahead of the echo its own mount posts."""

    def __init__(self, *args, held, release, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._held, self._release = held, release

    def _on_mount(self, event) -> None:
        async def hold() -> None:
            self._held.set()
            await self._release.wait()

        self.call_later(hold)
        super()._on_mount(event)


class _EchoScreen(Screen):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[object, bool]] = []

    def compose(self) -> ComposeResult:
        yield Vertical(id="container")

    @on(Select.Changed, "#picker")
    def _record(self, event: Select.Changed) -> None:
        self.seen.append((event.value, event.control.is_attached))


class _EchoApp(App):
    def get_default_screen(self) -> Screen:
        return _EchoScreen()


@pytest.mark.asyncio
async def test_a_mount_echo_below_a_removed_child_is_not_forwarded():
    """TASK-42150 review r19rc-rev1: a Select posts ``Changed`` when it mounts
    and on every programmatic ``value =``, not only when the user picks. An
    echo from a removed generation must not reach the screen as if the user
    had chosen it, so ``Select.Changed`` (like ``Checkbox``'s and ``Switch``'s)
    is not kept: only the new generation's own echo arrives."""
    held, release = asyncio.Event(), asyncio.Event()
    app = _EchoApp()
    async with app.run_test(size=(40, 10)) as pilot:
        await pilot.pause()
        screen = app.screen
        container = screen.query_one("#container")
        stale = _EchoSelect(
            [("a", 1), ("b", 2)], value=1, id="picker", held=held, release=release
        )
        await container.mount(Horizontal(stale, id="toolbar"))
        await asyncio.wait_for(held.wait(), timeout=5)
        toolbar = screen.query_one("#toolbar")

        async def swap() -> None:  # the next generation, on the container's pump
            await container.remove_children()
            await container.mount(
                Horizontal(Select([("a", 1), ("b", 2)], value=2, id="picker"), id="toolbar")
            )

        container.call_later(swap)
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not toolbar._closing:
                assert asyncio.get_running_loop().time() < deadline, "no swap"
                await asyncio.sleep(0)
        finally:
            release.set()
        await _settle_until(pilot, lambda: screen.seen)
        await pilot.pause(0.2)

        assert screen.seen == [(2, True)], "a removed Select's echo was forwarded"


class _PoppedApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.handled = 0

    @on(Button.Pressed)
    def _count(self, event: Button.Pressed) -> None:
        self.handled += 1


@pytest.mark.asyncio
async def test_a_press_queued_in_a_popped_screen_never_reaches_the_app():
    """TASK-42150: every removal now reaches the keeper, a screen's included,
    and a press below a removed screen is still never handed to the App."""
    app = _PoppedApp()
    async with app.run_test(size=(40, 8)) as pilot:
        await app.push_screen(Screen())
        await app.screen.mount(_toolbar())
        await pilot.pause()
        opener = app.screen.query_one("#opener", Button)
        await _hold_a_press_across(
            pilot, opener, app.screen.query_one("#toolbar"), app.pop_screen
        )
        await pilot.pause(0.2)

        assert app.handled == 0, "a press from a removed screen reached the App"


class _RemovedApp(App):
    def compose(self) -> ComposeResult:
        yield _Container(id="container")


@pytest.mark.asyncio
async def test_a_removed_container_recompose_never_waits_for_its_lock():
    """TASK-42014 review r18lib-rev2, kept after TASK-42150 removed the
    ``Widget.recompose`` wrapper: native ``Widget.recompose`` returns at once
    for a removed container, before its lock, and the keeper on ``App._prune``
    never takes the lock at all.

    A child's handler recomposes its container after the container's
    ``remove()``, while another task holds the container's lock and awaits
    that child (``remove_children``). Taking the lock first would wait on a
    task that is waiting on this child, so the removal would never end.
    """
    app = _RemovedApp()
    async with app.run_test(size=(40, 8)) as pilot:
        await pilot.pause()
        container = app.query_one("#container")
        child = container.query_one("#toolbar")
        held = asyncio.Event()
        go = asyncio.Event()
        recomposed: list[bool] = []

        async def hold() -> None:
            held.set()
            await go.wait()

        async def recompose_the_container() -> None:
            await container.recompose()
            recomposed.append(True)

        async def remove_children_under_the_lock() -> None:
            async with container.lock:
                await container.remove_children()

        child.call_later(hold)
        await asyncio.wait_for(held.wait(), timeout=5)
        child.call_later(recompose_the_container)  # runs once ``hold`` ends
        container.remove()
        remover = asyncio.ensure_future(remove_children_under_the_lock())
        try:
            await asyncio.sleep(0)  # the remover takes the lock
            assert container.lock.is_locked, "the remover never took the lock"
        finally:
            go.set()  # a failed check must not leave the child held
        await asyncio.wait_for(remover, timeout=3)
        assert recomposed == [True], "the child's recompose never ran"


class _ExternalSources:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def records(self):
        return {
            ParakeetSourceKey.V2_INT8: ParakeetSourceRecord(
                model_id=PARAKEET_V2_MODEL,
                precision="int8",
                directory=self._directory,
                preferred_source=ParakeetSourcePreference.EXTERNAL,
            )
        }


class _ExternalApp(App):
    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.requests: list[object] = []
        self.directory = directory

    def compose(self) -> ComposeResult:
        yield ExternalModelView(
            _ExternalSources(self.directory), runtime_ready=lambda: True
        )

    @on(ExternalModelView.ChangeRequested)
    def _capture(self, event: ExternalModelView.ChangeRequested) -> None:
        self.requests.append(event)


@pytest.mark.asyncio
async def test_an_external_model_action_pressed_across_a_reload_still_runs(tmp_path):
    """TASK-42014: ``ExternalModelView``'s action handler selects the button
    by its own class, so a press its ``reload()`` forwards still runs it."""
    app = _ExternalApp(tmp_path.absolute())
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        view = app.query_one(ExternalModelView)
        change = view.query_one("#external-model-change-v2_int8", Button)
        await _hold_a_press_across(
            pilot, change, view.query_one(".external-model-list"), view.reload
        )
        await _settle_until(pilot, lambda: app.requests)

        assert [type(event) for event in app.requests] == [
            ExternalModelView.ChangeRequested
        ], "the press across the reload was dropped"
        assert view.query_one("#external-model-change-v2_int8") is not change


class _SpeechApp(App):
    def compose(self) -> ComposeResult:
        yield SpeechTTSSettingsPanel(
            state=load_global_speech_tts_state({}),
            configure_provider="openai",
            config_snapshot={},  # no profile read: this lane has none
            id="panel",
        )


@pytest.mark.asyncio
async def test_a_speech_credential_action_pressed_across_a_recompose_still_runs():
    """TASK-42014: the Speech panel's credential handler selects the button
    by its own class, so a press its recompose forwards still opens the
    credential editor."""
    app = _SpeechApp()
    async with app.run_test(size=(150, 150)) as pilot:
        await pilot.pause()
        panel = app.query_one(SpeechTTSSettingsPanel)
        edit = panel.query_one("#settings-speech-openai-credential-edit", Button)
        closing = next(node for node in edit.ancestors if node.parent is panel)
        # Without the app stylesheet the row covers the button in a mouse hit
        # test, so the real input here is Enter on the focused button.
        edit.focus()
        await pilot.pause()
        await _hold_a_press_across(
            pilot,
            edit,
            closing,
            lambda: panel.refresh(recompose=True),
            user=lambda: pilot.press("enter"),
        )
        await _settle_until(pilot, lambda: app.screen is not app.screen_stack[0])

        assert type(app.screen).__name__ == "_CredentialEditorModal", (
            "the press across the recompose was dropped"
        )


def test_every_removal_reaches_the_message_keeper():
    """TASK-42150: the keeper sits on ``App._prune``, which every removal goes
    through (a recompose, ``remove_children()``, ``remove()``, a query's
    ``remove()``), so no override can remove children ahead of it."""
    assert getattr(App._prune, QUEUED_MESSAGE_KEEPER, False)
