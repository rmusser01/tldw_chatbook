"""Regression tests for task-292: media search display path.

``perform_media_search_and_display`` used to import the legacy
``UI.MediaWindow`` module (deleted in the Library redesign) from inside its
body; the resulting ``ModuleNotFoundError`` was swallowed by the handler's
broad ``except Exception`` and rendered as a permanent "Error loading" list
item, so media search always failed on dev even though ``search_media_db``
itself worked. These tests drive the REAL handler end-to-end -- real
``MediaDatabase`` (file-backed, so the ``asyncio.to_thread`` path runs),
no mocks -- which is exactly the coverage that would have caught the
deleted-module import.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Label, ListView
from loguru import logger as loguru_logger

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Event_Handlers.media_events import perform_media_search_and_display

TYPE_SLUG = "video"


class _MediaSearchHarness(App):
    """Minimal host exposing exactly the app surface the handler touches."""

    def __init__(self, media_db: MediaDatabase):
        super().__init__()
        self.media_db = media_db
        self.loguru_logger = loguru_logger
        self.media_current_page = 1
        self._media_search_generation = {}

    def compose(self) -> ComposeResult:
        yield ListView(id=f"media-list-view-{TYPE_SLUG}")


def _seeded_db(tmp_path) -> MediaDatabase:
    db = MediaDatabase(tmp_path / "media-292.db", client_id="task-292-test")
    media_id, _uuid, msg = db.add_media_with_keywords(
        title="Tides Explained",
        media_type=TYPE_SLUG,
        content="Tides are driven by the moon's gravity.",
        author="QA",
    )
    assert media_id is not None, msg
    return db


def _rendered_labels(list_view: ListView) -> list[str]:
    return [str(label.renderable) for label in list_view.query(Label)]


@pytest.mark.asyncio
async def test_media_search_end_to_end_renders_results_not_error(tmp_path):
    """The unmocked handler renders the seeded item, never "Error loading".

    Pins the deleted-module import class: any import of a nonexistent
    module inside the handler body resurfaces here as the "Error loading"
    row this asserts against.
    """
    db = _seeded_db(tmp_path)
    app = _MediaSearchHarness(db)
    async with app.run_test():
        await perform_media_search_and_display(app, TYPE_SLUG, "")

        list_view = app.query_one(ListView)
        labels = _rendered_labels(list_view)
        assert any("Tides Explained" in text for text in labels), labels
        assert not any("Error loading" in text for text in labels), labels
        # The row the click handlers consume carries the DB record.
        assert list_view.children, "expected at least one result row"
        assert getattr(list_view.children[0], "media_data", {}).get("title") == "Tides Explained"


@pytest.mark.asyncio
async def test_media_search_end_to_end_search_term_filters(tmp_path):
    """A real FTS search term flows through search_media_db unmocked."""
    db = _seeded_db(tmp_path)
    app = _MediaSearchHarness(db)
    async with app.run_test():
        await perform_media_search_and_display(app, TYPE_SLUG, "gravity")
        labels = _rendered_labels(app.query_one(ListView))
        assert any("Tides Explained" in text for text in labels), labels

        await perform_media_search_and_display(app, TYPE_SLUG, "zebra-nonmatch")
        labels = _rendered_labels(app.query_one(ListView))
        assert any("No media items found" in text for text in labels), labels
        assert not any("Error loading" in text for text in labels), labels
