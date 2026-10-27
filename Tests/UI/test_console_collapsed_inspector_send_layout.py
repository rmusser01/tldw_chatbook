"""TASK-37000: a send leaves the collapsed Inspector alone until it opens.

Textual answers every ``refresh(layout=True)`` or recompose with a layout of
the whole screen, whether or not the requester is drawn. TASK-36012 measured
what the collapsed Inspector rail (``#console-right-rail``) still asked for
during a send. Re-measured headless on origin/dev 3be493413a (2026-10-27,
160x45, collapsed Inspector, one run):

* a chat's first send: 56 layout requests from widgets inside the hidden
  rail. Persisting the conversation turns the dictionary and world-book rows
  into "No ... in play" and enables their Attach actions, which the run
  inspector treats as structural: it recomposed, re-mounting its More
  control and every bounded section and viewport. The Agents section, whose
  summary becomes "Run history", recomposed as well and set its own
  ``display`` inside the hidden rail;
* every send, start and end: the run inspector's auto-height "Live work"
  row re-wrote its copy with a layout;
* a chat's first agent run: it publishes the MCP catalog, and the sync tick
  re-wrote the readiness card's MCP row with a layout. Found live (pf2probe,
  160x45, one request per launch on base and branch alike); this harness
  runs without the agent runtime, so the send publishes the catalog itself.

The Agents section's display write also lost the section: the rail's collapse cascade
keeps each child's own display in ``_console_rail_prior_display`` and
restores it when the rail opens, so a section that showed itself inside the
hidden rail was hidden again by the opening cascade. A rail opened after a
collapsed first send had no Agents section; one open throughout had it.

The two tests pin the two halves of the contract: no layout request from
inside the collapsed rail during a send, and the rail opened afterwards
matching one that was open throughout, rows, actions and geometry alike.
"""

from __future__ import annotations

import asyncio
import re
import time

import pytest
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button, Static

# The UI fixture imports the app after selecting a per-test profile. Import it
# under the collection bootstrap first, before config freezes its source owner.
import tldw_chatbook.app  # noqa: F401
from Tests.private_profile import private_profile_test
from Tests.UI.test_console_run_tick_layout_requests import _LayoutRequests, _names
from Tests.UI.test_console_send_transition_layout_requests import (
    _assistant_turns,
    _console_host,
)
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _inside(widget: Widget, rail: Widget) -> bool:
    return widget is rail or rail in widget.ancestors


async def _wait_for(pilot, predicate, what: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    assert predicate(), what


_MCP_TOOLS = 4
_MCP_ROW = "MCP: Connected - 4 tools ready."


async def _send(
    console, composer, pilot, gateway, text: str, requests=None, *, mcp=False
):
    """One held send; return the requesters of its start and end windows.

    The start window runs from Send to the first streamed chunk, the end
    window from the stream's release to the run poll stopping. With ``mcp``,
    the run publishes an MCP catalog as a chat's first agent run does live
    (``_publish_mcp_inspector_counts``; this harness runs without the agent
    runtime), and the start window lasts until the sync tick has written the
    readiness card's MCP row.
    """
    store = console._ensure_console_chat_store()
    send = console.query_one("#console-send-message", Button)
    stop = composer.query_one("#console-stop-generation", Button)
    gateway.release = asyncio.Event()
    # A finished turn can re-sync the session draft after it ends; load until
    # Send has the draft and keeps it.
    for _ in range(40):
        composer.load_draft(text)
        await pilot.pause(0.2)
        if not send.disabled:
            await pilot.pause(0.3)
            if not send.disabled:
                break
    assert not send.disabled, "Send never took the draft"
    before = _assistant_turns(store)
    windows: dict[str, list[Widget]] = {}

    async def start() -> None:
        send.press()
        if mcp:
            console._ensure_console_chat_controller()._publish_mcp_inspector_counts(
                _MCP_TOOLS, 0
            )
        await _wait_for(pilot, lambda: _assistant_turns(store) > before, "no chunk")
        await _wait_for(pilot, lambda: stop.display, "Stop never displayed")
        if mcp:
            row = console.query_one("#console-live-work-source-mcp", Static)
            await _wait_for(
                pilot,
                lambda: str(row.render()) == _MCP_ROW,
                "the sync tick never wrote the MCP readiness row",
            )
        await pilot.pause()
        await pilot.pause()

    async def end() -> None:
        gateway.release.set()
        await _wait_for(pilot, lambda: not stop.display, "Stop never hid")
        await _wait_for(
            pilot,
            lambda: console._console_transcript_sync_timer is None,
            "the run poll never stopped",
        )
        await pilot.pause(0.3)

    for name, window in (("start", start), ("end", end)):
        if requests is None:
            await window()
        else:
            with requests.counting() as calls:
                await window()
                windows[name] = list(calls)
        await pilot.pause(0.3)
    return windows


async def _open_inspector(console, pilot) -> None:
    rail = console.query_one("#console-right-rail")
    await console.action_toggle_console_inspector_rail()
    await _wait_for(pilot, lambda: rail.display, "the Inspector never opened")
    await pilot.pause(0.5)


def _painted(widget: Widget) -> str:
    if isinstance(widget, Button):
        return widget.label.plain
    if isinstance(widget, Static):
        return _UUID.sub("<id>", str(widget.render()))
    return ""


def _shown(widget: Widget, rail: Widget) -> bool:
    return all(
        node.display
        for node in (widget, *widget.ancestors)
        if node is not rail and _inside(node, rail)
    )


def _witness(rail: Widget) -> list[tuple]:
    """Every widget the open rail shows: what it says, its state, its box."""
    origin = rail.region.offset
    return [
        (
            type(widget).__name__,
            widget.id or "",
            _painted(widget),
            bool(widget.disabled),
            tuple(sorted(widget.classes)),
            (
                widget.region.x - origin.x,
                widget.region.y - origin.y,
                widget.region.width,
                widget.region.height,
            ),
        )
        for widget in rail.query("*")
        if _shown(widget, rail)
    ]


def _action(actions, widget_id: str):
    return next(action for action in actions if action.widget_id == widget_id)


@private_profile_test
async def test_a_send_asks_no_layout_of_the_collapsed_inspector(
    monkeypatch, request, tmp_path
) -> None:
    """A first and a later send, Inspector collapsed: nothing inside it asks.

    The later send follows an open-and-close of the rail, so no rebuild is
    held and the run inspector re-writes its rows in place inside the hidden
    rail. Fails on the base build at the first send's start window (the run
    inspector's and the Agents section's recomposes); its later windows fail
    there on the "Live work" row alone.
    """
    from Tests.UI.test_console_composer_run_controls import _mounted

    requests = _LayoutRequests(monkeypatch)
    gateway, host = _console_host(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        await pilot.pause(0.5)
        rail = console.query_one("#console-right-rail")
        inspector = console.query_one(
            "#console-run-inspector-state", ConsoleRunInspector
        )
        assert not rail.display, "the layout under test is a collapsed Inspector"
        before = inspector.state
        first = await _send(
            console, composer, pilot, gateway, "stream and hold 0", requests, mcp=True
        )
        # Readers of the inspector see the persisted chat at once: the
        # dictionary row and its enabled Attach action are in its state,
        # though nothing inside the hidden rail was rebuilt.
        assert inspector.state != before
        assert _action(
            inspector.state.dictionary_actions, "console-inspector-dictionaries-attach"
        ).enabled
        assert not _action(
            before.dictionary_actions, "console-inspector-dictionaries-attach"
        ).enabled
        # Open and close: the opening runs what the hidden rail held, so the
        # later send meets a rail with nothing pending.
        await _open_inspector(console, pilot)
        assert (
            console.query_one("#console-inspector-dictionaries-attach", Button).disabled
            is False
        )
        await console.action_toggle_console_inspector_rail()
        await _wait_for(pilot, lambda: not rail.display, "the Inspector never closed")
        await pilot.pause(0.3)
        settled = inspector.recompose_count
        later = await _send(
            console, composer, pilot, gateway, "stream and hold 1", requests
        )
        assert inspector.recompose_count == settled, (
            "the later send rebuilt the hidden inspector"
        )
        for send, windows in (("first", first), ("later", later)):
            for window, calls in windows.items():
                assert calls, f"{send} send {window}: the counter saw nothing"
                in_rail = [widget for widget in calls if _inside(widget, rail)]
                assert _names(in_rail) == [], (
                    f"{send} send {window}: the collapsed Inspector asked for "
                    f"{len(in_rail)} layouts: {_names(in_rail)}"
                )

        # Positive control: with the Inspector open, the same send does lay
        # out widgets inside it (its "Live work" row changes height).
        await _open_inspector(console, pilot)
        shown = await _send(
            console, composer, pilot, gateway, "stream and hold 2", requests
        )
        assert any(_inside(widget, rail) for widget in shown["start"] + shown["end"]), (
            "the counter cannot see the open Inspector's layouts"
        )


def _frame_facts(screen) -> tuple:
    """What a painted frame of the opened rail says about this send's rows."""
    attach = screen.query("#console-inspector-dictionaries-attach")
    fleet = screen.query("#console-inspector-section-agent-fleet-summary")
    return (
        tuple(bool(button.disabled) for button in attach),
        tuple(str(summary.render()) for summary in fleet if _shown(summary, screen)),
    )


async def _rail_after_first_send(
    monkeypatch, tmp_path, *, open_first: bool
) -> tuple[list[tuple], list[tuple]]:
    """The rail's witness after a first send, and each frame the opening painted."""
    from Tests.UI.test_console_composer_run_controls import _mounted

    frames: list[tuple] = []
    recording = [False]
    paint = Screen._compositor_refresh

    def painted(screen) -> None:
        if recording[0] and screen.query("#console-right-rail").first().display:
            frames.append(_frame_facts(screen))
        return paint(screen)

    monkeypatch.setattr(Screen, "_compositor_refresh", painted)
    gateway, host = _console_host(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        await pilot.pause(0.5)
        rail = console.query_one("#console-right-rail")
        assert not rail.display
        if open_first:
            await _open_inspector(console, pilot)
        await _send(console, composer, pilot, gateway, "stream and hold 0", mcp=True)
        if not open_first:
            recording[0] = True
            await _open_inspector(console, pilot)
        await pilot.pause(0.5)
        recording[0] = False
        witness = _witness(rail)
        # Leave the stored rail preference collapsed for the next arm.
        await console.action_toggle_console_inspector_rail()
        await _wait_for(pilot, lambda: not rail.display, "the Inspector never closed")
        await pilot.pause(0.3)
        return witness, frames


@private_profile_test
async def test_an_inspector_opened_after_a_send_matches_one_open_throughout(
    monkeypatch, request, tmp_path
) -> None:
    """Rows, actions and geometry: the late-opened rail lost nothing.

    Fails on the base build: the late-opened rail has no Agents section
    ("Run history" / "View all runs"), and everything below it sits three
    rows higher. Also pins every frame the opening paints, so a rebuild that
    landed after the first paint (one frame of the rows from before the send,
    TASK-35970's lesson) would fail too.
    """
    (tmp_path / "open").mkdir()
    (tmp_path / "late").mkdir()
    open_throughout, _ = await _rail_after_first_send(
        monkeypatch, tmp_path / "open", open_first=True
    )
    opened_late, frames = await _rail_after_first_send(
        monkeypatch, tmp_path / "late", open_first=False
    )
    ids = {row[1] for row in open_throughout}
    # The open rail shows the rows this transition changes, so the witness
    # covers them.
    assert {
        "console-inspector-dictionaries-row-0",
        "console-inspector-dictionaries-attach",
        "console-inspector-worldbooks-attach",
        "console-agent-section-subagents",
        "console-live-work-source-mcp",
    } <= ids
    assert ("Static", "console-live-work-source-mcp", _MCP_ROW) in {
        row[:3] for row in open_throughout
    }
    assert opened_late == open_throughout
    # Not one painted frame of the opening showed the held rows: the rebuild
    # lands before the opened rail's first paint (Attach enabled, the Agents
    # section's "Run history" shown).
    assert frames, "the opening painted no frame"
    assert set(frames) == {((False,), ("Run history",))}, frames
