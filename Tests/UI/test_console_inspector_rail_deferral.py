"""TASK-37000: Inspector widgets under the collapsed rail's cascade.

``ChatScreen._sync_console_rail_descendant_visibility`` hides every
descendant of a collapsed Console rail and records the display each had in
``_console_rail_prior_display``; the rail's opening restores it. A widget in
that state is not drawn, yet any layout it requests lays out the whole
screen. These cases drive the production cascade around the two Inspector
widgets a send changes, ``ConsoleRunInspector`` and ``ConsoleInspectorSection``:
while hidden, a change requests no layout (the copy is re-written in place,
a rebuild waits), and once the cascade opens the rail again, the widget
matches one that was never hidden, widget for widget and box for box.

On the base build each change below requests layouts inside the hidden
rail, and the section's secondary line, shown by a direct ``display`` write
the opening then undid, stays hidden after the rail opens.
"""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Button, Static

# The UI fixture imports the app after selecting a per-test profile. Import it
# under the collection bootstrap first, before config freezes its source owner.
import tldw_chatbook.app  # noqa: F401
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_run_tick_layout_requests import _LayoutRequests, _names
from tldw_chatbook.Chat.console_display_state import (
    ConsoleDisplayRow,
    ConsoleInspectorAction,
    ConsoleInspectorState,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
    ConsoleInspectorSectionState,
    InspectorSectionRow,
)
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _inspector_state(*, provider: str, chat: bool) -> ConsoleInspectorState:
    """The run inspector before (``chat=False``) and after a first send."""
    reason = "Start or load a conversation first"
    return ConsoleInspectorState(
        rows=(
            ConsoleDisplayRow("Run recipe", "Chat with provider"),
            ConsoleDisplayRow("Live work", provider),
        ),
        actions=(),
        dictionary_rows=(
            ConsoleDisplayRow(
                "No dictionaries in play" if chat else "No active chat", ""
            ),
        ),
        dictionary_actions=(
            ConsoleInspectorAction(
                "console-inspector-dictionaries-attach",
                "Attach dictionary…",
                enabled=chat,
                disabled_reason=reason,
            ),
            ConsoleInspectorAction(
                "console-inspector-dictionaries-detach",
                "Detach dictionary…",
                enabled=False,
            ),
        ),
    )


def _section_state(*, summary: str, detail: str) -> ConsoleInspectorSectionState:
    return ConsoleInspectorSectionState(
        rows=(InspectorSectionRow("run-1", "Run one", secondary_text=detail),),
        summary=summary,
    )


class _RailHost(ConsolidatedCSSApp):
    """A rail holding one production widget, as the Inspector rail holds it."""

    def __init__(self, widget: Widget) -> None:
        super().__init__()
        self.app_config = {}
        self.widget = widget

    def compose(self) -> ComposeResult:
        with Vertical(id="rail"):
            yield self.widget


def _painted(widget: Widget) -> str:
    if isinstance(widget, Button):
        return widget.label.plain
    if isinstance(widget, Static):
        return str(widget.render())
    return ""


def _witness(rail: Widget) -> list[tuple]:
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
        if all(
            node.display
            for node in (widget, *widget.ancestors)
            if node is not rail and rail in node.ancestors
        )
    ]


async def _never_hidden(widget: Widget) -> list[tuple]:
    host = _RailHost(widget)
    async with host.run_test(size=(40, 30)) as pilot:
        await pilot.pause()
        await pilot.pause()
        return _witness(host.query_one("#rail"))


async def _changed_while_hidden(monkeypatch, widget: Widget, change) -> tuple:
    """Hide the rail, apply ``change``, open it; return requests and witness."""
    requests = _LayoutRequests(monkeypatch)
    host = _RailHost(widget)
    async with host.run_test(size=(40, 30)) as pilot:
        await pilot.pause()
        rail = host.query_one("#rail")
        ChatScreen._sync_console_rail_descendant_visibility(rail, False)
        await pilot.pause()
        with requests.counting() as calls:
            change()
            await pilot.pause()
            await pilot.pause()
            hidden_calls = list(calls)
        ChatScreen._sync_console_rail_descendant_visibility(rail, True)
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        return hidden_calls, _witness(rail)


async def test_run_inspector_holds_a_rebuild_for_the_rails_opening(monkeypatch) -> None:
    """A first send's dictionary change, made while the rail is collapsed."""
    after = _inspector_state(provider="No active work", chat=True)
    inspector = ConsoleRunInspector(
        _inspector_state(provider="No active work", chat=False)
    )
    calls, opened = await _changed_while_hidden(
        monkeypatch, inspector, lambda: inspector.sync_state(after)
    )
    assert _names(calls) == [], (
        f"the hidden inspector asked for layouts: {_names(calls)}"
    )
    assert inspector.state == after
    assert opened == await _never_hidden(ConsoleRunInspector(after))


async def test_run_inspector_rewrites_a_row_in_place_without_a_layout(
    monkeypatch,
) -> None:
    """A "Live work" copy change, the in-place path, while the rail is collapsed."""
    after = _inspector_state(provider="1 run streaming in this chat now", chat=True)
    inspector = ConsoleRunInspector(
        _inspector_state(provider="No active work", chat=True)
    )
    calls, opened = await _changed_while_hidden(
        monkeypatch, inspector, lambda: inspector.sync_state(after)
    )
    assert _names(calls) == [], (
        f"the hidden inspector asked for layouts: {_names(calls)}"
    )
    assert inspector.recompose_count == 0, "an in-place change was rebuilt"
    assert opened == await _never_hidden(ConsoleRunInspector(after))


async def test_section_holds_a_rebuild_for_the_rails_opening(monkeypatch) -> None:
    """A summary that appears (the Agents section's "Run history")."""
    after = _section_state(summary="Run history", detail="")
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        **_kwargs(_section_state(summary="", detail="")),
    )
    calls, opened = await _changed_while_hidden(
        monkeypatch, section, lambda: section.sync_state(after)
    )
    assert _names(calls) == [], f"the hidden section asked for layouts: {_names(calls)}"
    assert (section.rows, section.summary) == (after.rows, after.summary)
    assert opened == await _never_hidden(
        ConsoleInspectorSection(title="Agents", section_id="agents", **_kwargs(after))
    )


async def test_section_patch_in_a_hidden_rail_keeps_its_new_secondary_line(
    monkeypatch,
) -> None:
    """An in-place patch that gives a row its detail line while hidden."""
    # Too long to share the primary's line, so the row keeps its two-line
    # shape and the patch only re-shows the line it already has.
    after = _section_state(summary="1 run", detail="finished after reading four files")
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        **_kwargs(_section_state(summary="1 run", detail="")),
    )
    calls, opened = await _changed_while_hidden(
        monkeypatch, section, lambda: section.sync_state(after)
    )
    assert _names(calls) == [], f"the hidden section asked for layouts: {_names(calls)}"
    assert section.recompose_count == 0, "an in-place change was rebuilt"
    assert opened == await _never_hidden(
        ConsoleInspectorSection(title="Agents", section_id="agents", **_kwargs(after))
    )


class _FocusRailHost(_RailHost):
    """The rail beside a control outside it, as the Console composer sits."""

    def compose(self) -> ComposeResult:
        yield Button("Outside", id="outside")
        with Vertical(id="rail"):
            yield self.widget


@pytest.mark.parametrize("same_tick", [False, True], ids=["later", "same-tick"])
async def test_a_control_focused_at_the_collapse_leaves_no_focus_to_carry(
    same_tick,
) -> None:
    """The held rebuild drops the focus recovery an immediate rebuild sets up.

    That is safe because Textual blurs a focused control the collapse hides:
    the control gets a Hide event at the collapse's first layout, as on the
    base build, before the opening could need its focus back. A change in
    the same tick as the collapse, made while the control still holds focus,
    is held all the same.
    """
    inspector = ConsoleRunInspector(
        _inspector_state(provider="No active work", chat=True)
    )
    after = _inspector_state(provider="No active work", chat=False)
    host = _FocusRailHost(inspector)
    async with host.run_test(size=(40, 30)) as pilot:
        await pilot.pause()
        attach = host.query_one("#console-inspector-dictionaries-attach", Button)
        attach.focus()
        await pilot.pause()
        assert host.focused is attach
        rail = host.query_one("#rail")
        ChatScreen._sync_console_rail_descendant_visibility(rail, False)
        if same_tick:
            assert host.focused is attach, "the collapse blurred at once"
            inspector.sync_state(after)
        await pilot.pause()
        await pilot.pause()
        assert host.focused is None or rail not in host.focused.ancestors, (
            f"focus stayed in the collapsed rail: {host.focused!r}"
        )
        if not same_tick:
            inspector.sync_state(after)
            await pilot.pause()
        assert inspector._rail_rebuild_deferred
        assert inspector.recompose_count == 0
        ChatScreen._sync_console_rail_descendant_visibility(rail, True)
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        assert inspector.recompose_count == 1
        assert host.query_one(
            "#console-inspector-dictionaries-attach", Button
        ).disabled, "the opening did not paint the held state"
        focused = host.focused
        assert focused is None or (
            focused.is_mounted and rail not in focused.ancestors
        ), f"the opening left focus on {focused!r}"


def _kwargs(state: ConsoleInspectorSectionState) -> dict:
    return {"rows": state.rows, "summary": state.summary, "notice": state.notice}
