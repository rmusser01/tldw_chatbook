"""TASK-33620.15.20: the Conversations tray's diff composes are never styled.

TASK-36012 patches a send's "Running" marker into the mounted tray instead of
rebuilding it. ``plan_in_place_delta`` decides by diffing two off-DOM composes
(the state shown and the next one), which are then dropped, never mounted.

Measured live on origin/dev 3be493413a (2026-10-27, 160x45, Anthropic
claude-haiku-4-5, 12 probed sends): in 8 of 12 sends the longest UI-loop
callback between Enter and the provider POST was the first 0.2 s poll tick
after acceptance (+260-355 ms, 40-57 ms). Inside it, ``App.update_styles``
took 21-34 ms for 64-69 nodes: every widget those two composes built,
restyled as each Button constructor and compose helper set a class. Textual
styles a widget again when it mounts (``App._register`` applies the
stylesheet after attaching it), so for a tree that is never mounted none of
that styling is ever read.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.css.stylesheet import Stylesheet
from textual.widget import Widget
from textual.widgets import Static

# The UI fixture imports the app after selecting a per-test profile. Import it
# under the collection bootstrap first, before config freezes its source owner.
import tldw_chatbook.app  # noqa: F401
from Tests.UI.test_console_send_transition_layout_requests import (
    _settled_tray,
    _tray_state,
    _TrayHost,
    _within,
)
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


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


async def test_running_marker_patch_styles_no_throwaway_node(monkeypatch) -> None:
    """The run-marker patch styles the live rows it writes, and nothing else."""
    composes: list[Widget] = []
    original = ConsoleWorkspaceContextTray.compose

    def counted(self) -> ComposeResult:
        composes.append(self)
        yield from original(self)

    host = _TrayHost(_tray_state())
    async with host.run_test(size=(40, 30)) as pilot:
        tray = await _settled_tray(host, pilot)
        monkeypatch.setattr(ConsoleWorkspaceContextTray, "compose", counted)
        detached, attached = _record_applies(monkeypatch)
        running = _tray_state(running=True)
        tray.sync_state(running)
        # The patch took the delta: two diff composes and no rebuild.
        assert len(composes) == 2
        assert tray.state == running
        assert not getattr(tray, "_recompose_required", False)
        # Positive control: the live row's colour-class write was styled.
        assert any(_within(node, tray) for node in attached)
        assert detached == [], (
            f"{len(detached)} nodes of the throwaway diff composes were styled: "
            f"{sorted({type(node).__name__ for node in detached})}"
        )


async def test_detached_styles_skipped_styles_live_nodes_and_restores() -> None:
    """Only nodes outside the DOM skip styling, and only inside the block."""
    # Imported here so the test above still runs against a base build that
    # has no such helper.
    from tldw_chatbook.Widgets.Console.compose_delta import detached_styles_skipped

    host = _TrayHost(_tray_state())
    async with host.run_test(size=(40, 30)) as pilot:
        tray = await _settled_tray(host, pilot)
        with pytest.MonkeyPatch.context() as patch:
            detached, attached = _record_applies(patch)
            with pytest.raises(RuntimeError, match="compose failed"):
                with detached_styles_skipped(host):
                    Static("off the DOM").add_class("console-rail-section-title")
                    tray.add_class("p20-live-probe")
                    raise RuntimeError("compose failed")
            assert detached == []
            assert tray in attached
            # Restored on the raising exit: Textual styles an off-DOM node again.
            Static("after").add_class("console-rail-section-title")
            assert len(detached) == 1
        assert "update_styles" not in vars(host)
