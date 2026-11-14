"""Plain session tabs must not round-trip their native label through Rich Text."""

import pytest
from rich.text import Text
from textual.containers import HorizontalScroll
from textual.content import Content
from textual.widgets import Button

from Tests.UI.test_console_session_tab_strip import TabStripHost
from tldw_chatbook.Chat.console_chat_models import ConsoleRunMarker
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Widgets.Console.console_session_surface import ConsoleSessionSurface

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("size", [(52, 20), (120, 50)])
async def test_native_tab_labels_avoid_rich_copies_and_preserve_current_paint(
    monkeypatch, size
):
    app = TabStripHost()
    async with app.run_test(size=size) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        session = ConsoleChatSession(
            title="[bold]界🙂 literal[/bold]", id="literal-tab"
        )
        await surface.sync_sessions(sessions=[session], active_session_id=session.id)
        await pilot.pause()
        strip = app.query_one("#console-native-tab-strip", HorizontalScroll)
        button = app.query_one("#console-session-tab-literal-tab", Button)
        button.focus()
        await pilot.pause()
        allocations = []
        original = Text.__init__

        def construct(self, *args, **kwargs):
            allocations.append(1)
            original(self, *args, **kwargs)

        for marker in ConsoleRunMarker:
            for ephemeral, queued in ((False, 0), (True, 3)):
                session.ephemeral = ephemeral
                expected = Content.from_text(
                    Text(
                        surface._tab_label(
                            session.title,
                            marker=marker,
                            ephemeral=ephemeral,
                            queued_count=queued,
                        )
                    )
                )
                with monkeypatch.context() as scope:
                    scope.setattr(Text, "__init__", construct)
                    created = surface._build_session_tab_button(
                        session, active=True, marker=marker, queued_count=queued
                    )
                    surface._update_existing_tab_strip(
                        tab_strip=strip,
                        sessions=[session],
                        active_session_id=session.id,
                        run_markers={session.id: marker},
                        queue_counts={session.id: queued},
                    )
                assert not allocations
                assert created.label == button.label == expected
                assert created._session_id == session.id
                assert app.focused is button
                tooltip = button.tooltip
                await pilot.pause()
                frame = tuple(
                    tuple((s.text, s.style, s.control) for s in line)
                    for line in app.screen._compositor.render_strips()
                )
                button.label = Text(
                    surface._tab_label(
                        session.title,
                        marker=marker,
                        ephemeral=ephemeral,
                        queued_count=queued,
                    )
                )
                await pilot.pause()
                assert (
                    tuple(
                        tuple((s.text, s.style, s.control) for s in line)
                        for line in app.screen._compositor.render_strips()
                    )
                    == frame
                )
                assert button.tooltip == tooltip
                assert app.focused is button
