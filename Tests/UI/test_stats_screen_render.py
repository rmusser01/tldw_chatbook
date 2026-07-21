"""Stats screen render regression test.

The section helpers used to build detached ``Grid``/``Container`` widgets and
``.mount()`` children into them before the container itself was mounted —
current Textual raises ``MountError`` for that, so the screen rendered only
the first section label and nothing else. This mounts the real screen against
a real (file-backed — the load worker runs on a thread) ChaChaNotes DB and
asserts stat cards actually appear.
"""

from pathlib import Path

import pytest
from textual.app import App

import tldw_chatbook
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.stats_screen import StatCard, StatsScreen

_APP_CSS = str(Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss")


@pytest.mark.asyncio
async def test_stats_screen_renders_stat_cards_with_real_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "stats.db", client_id="test-client")

    class HostApp(App):
        # Real stylesheet: the cards once rendered but clipped their value
        # labels to zero height under the app CSS, which a bare App misses.
        CSS_PATH = _APP_CSS

        def __init__(self):
            super().__init__()
            self.chachanotes_db = db
            self.notes_service = None

        async def on_mount(self):
            await self.push_screen(StatsScreen(self))

    app = HostApp()
    async with app.run_test(size=(120, 60)) as pilot:
        # Wait for the load worker AND a layout pass over the mounted cards:
        # widgets exist in the DOM before layout assigns them regions.
        laid_out = False
        for _ in range(50):
            await pilot.pause(0.1)
            values = list(app.screen.query(".stat-card-value"))
            if values and values[0].region.height > 0:
                laid_out = True
                break

        assert list(app.screen.query(StatCard)), (
            "no StatCard widgets rendered after statistics load"
        )
        assert not list(app.screen.query(".error-container"))
        assert laid_out, "stat card value label never laid out (clipped?)"
