"""An awaited surface swap is refused on the pump of a widget it removes.

TASK-34000.4 (2026-10-02 Library review, finding L-01). The Library's Media
"Export…" froze the whole app because a handler on the Media canvas awaited a
screen recompose that removes that canvas: Textual's teardown waits for the
canvas's pump, and the canvas's pump was waiting for the teardown. These tests
pin the guard that turns that permanent freeze into an immediate, named error
(``UI/Navigation/surface_swap_guard.py``), in two halves:

* the RUNTIME predicate, against a minimal Textual app for every way a widget
  can start a swap. It must refuse exactly the shapes that can never finish
  and none of the ones that can -- the second half matters as much as the
  first, because a guard that refused working code would be removed;
* a STATIC pin over ``Widgets/Library``: no Library widget awaits the
  screen-owned collaborator it forwards its events to. That is the shape the
  freeze shipped in, and this catches it without booting anything.

Which shapes hang was measured, not assumed, on Textual 8.2.8 against a plain
``textual.screen.Screen`` (which has no guard): every ``_REFUSED`` shape below,
and the nested child swap, never landed; every shape these tests expect to
complete did. The probe scripts and their output are kept beside the task's
other evidence (``qa/notes-library-ux-review-2026-10-02/fixes/task-34000.4/``).

Gate-free on purpose, like ``test_base_app_screen_recompose_focus_seam.py``:
a bare ``textual.app.App``, so the recovery gate never runs.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from Tests.UI.pump_probe import (
    describe_parked,
    free_parked_pumps,
    parked_pumps,
    pump_runs,
    until,
)
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.surface_swap_guard import (
    SurfaceSwapSelfAwaitError,
    pump_awaiting_its_own_removal,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY_WIDGETS = _REPO_ROOT / "tldw_chatbook" / "Widgets" / "Library"

#: A widget below a removed root starts the swap on its OWN pump: the teardown
#: waits for that pump and that pump waits for the teardown.
_REFUSED = ["handler-awaits", "own-call-later"]
#: The swap runs somewhere that outlives it. Each of these completes without
#: the guard, so the guard must leave each of them alone.
_COMPLETES = [
    "screen-call-next",
    "screen-call-later",
    "screen-call-after-refresh",
    "own-call-after-refresh",
    "screen-worker",
    "refresh-recompose",
]


class _Canvas(Vertical):
    """A region widget two levels below the screen, like a Library canvas."""

    def __init__(self, shape: str, seen: dict) -> None:
        super().__init__(id="canvas")
        self._shape = shape
        self._seen = seen

    def compose(self) -> ComposeResult:
        yield Button("Export…", id="export")

    @on(Button.Pressed, "#export")
    async def _pressed(self, event: Button.Pressed) -> None:
        screen = self.screen
        shape = self._shape
        if shape == "handler-awaits":
            await self._swap()
        elif shape == "own-call-later":
            self.call_later(self._swap)
        elif shape == "own-call-after-refresh":
            self.call_after_refresh(self._swap)
        elif shape == "screen-call-next":
            screen.call_next(self._swap)
        elif shape == "screen-call-later":
            screen.call_later(self._swap)
        elif shape == "screen-call-after-refresh":
            screen.call_after_refresh(self._swap)
        elif shape == "screen-worker":
            screen.run_worker(self._swap())
        elif shape == "refresh-recompose":
            screen.refresh(recompose=True)
            self._seen["swap"] = "scheduled"

    async def _swap(self) -> None:
        try:
            await self.screen.recompose()
        except SurfaceSwapSelfAwaitError as error:
            self._seen["refused"] = error
            return
        self._seen["swap"] = "returned"


class _Shell(Vertical):
    """The screen's direct child: the removed ROOT the canvas sits below."""

    def __init__(self, shape: str, seen: dict) -> None:
        super().__init__(id="shell")
        self._shape = shape
        self._seen = seen

    def compose(self) -> ComposeResult:
        yield _Canvas(self._shape, self._seen)


class _SwapScreen(BaseAppScreen):
    """A minimal ``BaseAppScreen``; ``compose`` is overridden so the real nav
    bar and footer (which want a live ``TldwCli``) are never built."""

    def __init__(self, shape: str, seen: dict) -> None:
        super().__init__(SimpleNamespace(), "probe")
        self._shape = shape
        self._seen = seen
        self.composed = 0

    def compose(self) -> ComposeResult:
        self.composed += 1
        yield _Shell(self._shape, self._seen)


class _SwapApp(App):
    def __init__(self, shape: str) -> None:
        super().__init__()
        self.seen: dict = {}
        self._shape = shape

    def on_mount(self) -> None:
        self.push_screen(_SwapScreen(self._shape, self.seen))


#: How long a shape gets to land or be refused. Each takes a few milliseconds;
#: a deadlocked one never ends, so this is what a red run spends finding out.
_BUDGET_SECONDS = 5.0


async def _fail_loudly_not_forever(app) -> None:
    """Free a parked pump so a red run tears down instead of hanging."""
    for line in await free_parked_pumps(app):
        print(f"\nfreed for tear-down: {line}")


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _REFUSED)
async def test_a_recompose_awaited_on_a_removed_widgets_pump_is_refused(
    shape: str,
) -> None:
    """Refused at once, by name, before anything is torn down."""
    app = _SwapApp(shape)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        canvas = screen.query_one("#canvas", _Canvas)
        composed_before = screen.composed

        screen.query_one("#export", Button).press()

        try:
            assert await until(lambda: "refused" in app.seen, _BUDGET_SECONDS), (
                f"{shape}: the guard did not refuse a recompose awaited on the "
                f"pump of a widget the recompose removes (seen: {app.seen!r}; "
                f"parked: {describe_parked(await parked_pumps(app, 0.5))})"
            )
        except AssertionError:
            await _fail_loudly_not_forever(app)
            raise
        message = str(app.seen["refused"])
        assert "_SwapScreen.recompose" in message, message
        assert "_Canvas(id='canvas')" in message, message
        # Nothing was torn down: same widgets, still attached, still running.
        assert screen.composed == composed_before
        assert screen.query_one("#canvas") is canvas
        assert canvas.is_attached and not canvas._pruning
        for pump in (app, screen, canvas):
            assert await pump_runs(pump), f"{shape}: {pump!r} is parked"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _COMPLETES)
async def test_a_swap_handed_to_the_screen_is_left_alone(shape: str) -> None:
    """The half that keeps the guard honest: working shapes are not refused.

    ``own-call-after-refresh`` is the one a context-variable check gets wrong:
    Textual runs the callback on the SCREEN's task with the scheduling widget
    still in ``active_message_pump``, and it completes.
    """
    app = _SwapApp(shape)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        composed_before = screen.composed

        screen.query_one("#export", Button).press()

        try:
            assert await until(
                lambda: (
                    screen.composed > composed_before and bool(screen.query("#export"))
                ),
                _BUDGET_SECONDS,
            ), f"{shape}: the recompose never landed (seen: {app.seen!r})"
        except AssertionError:
            await _fail_loudly_not_forever(app)
            raise
        assert "refused" not in app.seen, app.seen["refused"]
        for pump in (app, screen, screen.query_one("#canvas")):
            assert await pump_runs(pump), f"{shape}: {pump!r} is parked"


class _HostScreen(BaseAppScreen):
    """A screen that swaps one host's children, the region-scoped seam."""

    def __init__(self) -> None:
        super().__init__(SimpleNamespace(), "probe")
        self.seen: dict = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="host"):
            with Vertical(id="outgoing"):
                yield Static("nested", id="nested")

    async def swap_from(self, widget) -> None:
        """Run the child swap on ``widget``'s own pump and record the result."""

        async def swap() -> None:
            host = self.query_one("#host", Vertical)
            try:
                outgoing = self.children_safe_to_await_removing(host)
            except SurfaceSwapSelfAwaitError as error:
                self.seen["refused"] = error
                return
            await host.remove_children(outgoing)
            await host.mount(Static("incoming", id="incoming"))
            self.seen["swap"] = "returned"

        widget.call_later(swap)


class _HostApp(App):
    def on_mount(self) -> None:
        self.push_screen(_HostScreen())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_child_swap_awaited_below_an_outgoing_child_is_refused() -> None:
    """The region-scoped twin: same predicate, one host's children."""
    async with _HostApp().run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        nested = screen.query_one("#nested", Static)

        await screen.swap_from(nested)

        try:
            assert await until(lambda: "refused" in screen.seen, _BUDGET_SECONDS), (
                "the child-swap seam did not refuse a removal awaited below an "
                f"outgoing child (seen: {screen.seen!r}; parked: "
                f"{describe_parked(await parked_pumps(pilot.app, 0.5))})"
            )
        except AssertionError:
            await _fail_loudly_not_forever(pilot.app)
            raise
        assert "Static(id='nested')" in str(screen.seen["refused"])
        assert screen.query_one("#nested") is nested and nested.is_attached
        assert not screen.query("#incoming")


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("runner", ["#outgoing", "#host", "screen"])
async def test_a_child_swap_that_can_finish_is_left_alone(runner: str) -> None:
    """A removed ROOT may await its own removal -- ``AwaitRemove`` leaves the
    current task out of what it waits on -- and so may anything above it."""
    async with _HostApp().run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        widget = screen if runner == "screen" else screen.query_one(runner)

        await screen.swap_from(widget)

        try:
            assert await until(
                lambda: screen.seen.get("swap") == "returned", _BUDGET_SECONDS
            ), f"{runner}: the child swap never finished (seen: {screen.seen!r})"
        except AssertionError:
            await _fail_loudly_not_forever(pilot.app)
            raise
        assert "refused" not in screen.seen
        assert screen.query("#incoming") and not screen.query("#nested")


@pytest.mark.unit
def test_the_predicate_is_quiet_outside_an_event_loop() -> None:
    """Composition-time and thread callers have no pump to be parked."""
    assert pump_awaiting_its_own_removal([]) is None


# ---- static pin: Library widgets never await the screen's work -------------

#: Receivers whose awaited calls are screen-owned work. ``actions`` is the
#: controller a region widget forwards its events to (``LibraryMediaCanvas``);
#: ``screen`` is the screen itself.
_SCREEN_OWNED_RECEIVERS = frozenset({"actions", "screen"})


def _receiver_chain(node: ast.AST) -> list[str]:
    """``self.actions.handle_x`` -> ``["self", "actions", "handle_x"]``."""
    names: list[str] = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    return list(reversed(names))


def _awaits_of_screen_owned_work(path: Path) -> list[str]:
    """Every ``await <screen-owned receiver>.<method>(...)`` in a module."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
            continue
        chain = _receiver_chain(node.value.func)
        # The last name is the method; anything before it is the receiver.
        if _SCREEN_OWNED_RECEIVERS.intersection(chain[:-1]):
            found.append(f"{path.name}:{node.lineno}: await {'.'.join(chain)}(...)")
    return found


@pytest.mark.unit
def test_no_library_widget_awaits_the_screen_owned_work_it_forwards_to() -> None:
    """A region widget forwards an event and returns; it never awaits the work.

    The work can swap the surface the widget lives in, and a widget that is
    awaiting its own removal never gets it. ``LibraryMediaCanvas``'s Export
    row was the one ``async`` forwarder of its sixteen and froze the app
    (TASK-34000.4). A new row that needs screen-owned async work calls the
    controller synchronously; the controller hands the coroutine to the
    screen's pump (``call_next``) or to a screen-owned worker.
    """
    modules = sorted(_LIBRARY_WIDGETS.glob("*.py"))
    assert len(modules) > 20, "the Library widget package moved; re-point this pin"
    offenders = [
        line for module in modules for line in _awaits_of_screen_owned_work(module)
    ]
    assert not offenders, (
        "Library widget(s) await screen-owned work on their own message pump. "
        "If that work swaps the surface, the widget waits on its own removal "
        "and the app freezes:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.unit
def test_the_static_pin_sees_the_shape_that_froze_the_app() -> None:
    """Negative control: the scan flags the forwarder exactly as it shipped."""
    shipped = (
        "class LibraryMediaCanvas:\n"
        "    async def handle_library_media_export(self, event):\n"
        "        actions = self._media_actions_for_press(event)\n"
        "        if actions is not None:\n"
        "            await actions.handle_library_media_export(event)\n"
        "    async def also(self):\n"
        "        await self.actions.open(1)\n"
        "        await self.screen.recompose()\n"
        "        await self.app.screen.recompose()\n"
        "        await self.mount(1)\n"
        "        await asyncio.to_thread(len, ())\n"
    )
    flagged = [
        ".".join(_receiver_chain(node.value.func))
        for node in ast.walk(ast.parse(shipped))
        if isinstance(node, ast.Await)
        and _SCREEN_OWNED_RECEIVERS.intersection(_receiver_chain(node.value.func)[:-1])
    ]
    assert sorted(flagged) == [
        "actions.handle_library_media_export",
        "self.actions.open",
        "self.app.screen.recompose",
        "self.screen.recompose",
    ]
