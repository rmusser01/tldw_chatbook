"""TASK-34000.25 fix round 2: ``_pending_widget_lookup`` reads Textual's
compose-time child registry, so a Textual change to it fails HERE by name.

``LibraryNotesCanvas._compose_editor`` gates every editor surface on its
last line, over the composed-but-unmounted tree, through this lookup: a
child yielded under ``with Container():`` must be reachable from the
container before anything mounts (``compose_add_child`` ->
``_pending_children`` in Textual 8.2.8). Without this pin a Textual bump
would surface as an app crash in the first UI test that opens a note.

Fast lane: a bare Textual app (``run_test``), no product app boot.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Library.library_notes_canvas import _pending_widget_lookup


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
