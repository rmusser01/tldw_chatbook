"""The Reader's review-set banner (task-30045).

Critique 2026-09-03 P2 + user ruling on its Q2: a review set is a workflow
object and deserves a real runtime surface — the set's name and progress in
the Reader chrome, plus the current item's reviewed state at a glance,
instead of only a footer string.
"""

from __future__ import annotations

import itertools
from types import MethodType, SimpleNamespace

import pytest
from textual.widgets import Static

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.library_media_viewer_state import (
    build_library_media_viewer_state,
)
from tldw_chatbook.Library.review_set_service import ReviewSetService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_media_viewer import LibraryMediaViewer
from Tests.UI.consolidated_css import ConsolidatedCSSApp


def _service(tmp_path) -> ReviewSetService:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    counter = itertools.count(1)
    return ReviewSetService(
        db,
        id_factory=lambda: f"set-{next(counter)}",
        now=lambda: "2026-09-03T00:00:00Z",
    )


def _banner_fake(service, *, loaded=None) -> SimpleNamespace:
    fake = SimpleNamespace(
        _review_set_service=lambda: service,
        _review_set_live_ids=lambda ids: {int(i) for i in ids},
        _library_media_reader_session=SimpleNamespace(loaded_backing_id=loaded),
    )
    fake._active_review_set_banner = MethodType(
        LibraryScreen._active_review_set_banner, fake
    )
    return fake


def test_banner_names_the_set_and_flags_the_loaded_items_state(tmp_path):
    """Name + live progress + the loaded item's own reviewed state."""
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "All media", origin="browse", items=[(10, "A"), (11, "B")]
    )
    service.mark_item_done(set_id, backing_media_id=10, done=True)

    done_banner = LibraryScreen._active_review_set_banner(
        _banner_fake(service, loaded=10)
    )
    assert done_banner is not None
    assert "All media" in done_banner
    assert "1 of 2" in done_banner
    assert "1 reviewed" in done_banner
    assert "✓ reviewed" in done_banner

    fresh_banner = LibraryScreen._active_review_set_banner(
        _banner_fake(service, loaded=11)
    )
    assert "not yet reviewed" in fresh_banner


def test_banner_omits_item_state_when_the_reader_is_off_set(tmp_path):
    service = _service(tmp_path)
    service.create_review_set("All media", origin="browse", items=[(10, "A")])

    banner = LibraryScreen._active_review_set_banner(
        _banner_fake(service, loaded=999)
    )
    assert banner is not None and "All media" in banner
    assert "reviewed" in banner  # progress still shows
    assert "✓ reviewed" not in banner and "not yet" not in banner


def test_banner_is_none_without_an_active_set(tmp_path):
    assert (
        LibraryScreen._active_review_set_banner(_banner_fake(_service(tmp_path)))
        is None
    )


def test_banner_fails_closed_on_storage_errors(tmp_path):
    """A storage error never crashes the Reader build (task-30042 doctrine:
    the gate fails closed; the explicit gesture paths carry the notices)."""

    class _Boom:
        def get_active_review_set(self):
            raise RuntimeError("db gone")

    fake = _banner_fake(_service(tmp_path))
    fake._review_set_service = lambda: _Boom()
    assert LibraryScreen._active_review_set_banner(fake) is None


class _ViewerApp(ConsolidatedCSSApp):
    def __init__(self, banner: str) -> None:
        super().__init__()
        self._banner = banner

    def compose(self):
        yield LibraryMediaViewer(
            build_library_media_viewer_state(
                {"id": "local:media:10", "title": "Doc", "content": "text"}
            ),
            review_banner=self._banner,
            id="library-media-viewer",
        )


@pytest.mark.asyncio
async def test_viewer_renders_the_banner_when_a_set_is_active():
    app = _ViewerApp("Reviewing: All media — 1 of 2 · 1 reviewed · ✓ reviewed")
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        banner = app.query_one("#library-media-review-banner", Static)
        assert "Reviewing: All media" in str(banner.renderable)


@pytest.mark.asyncio
async def test_viewer_omits_the_banner_without_an_active_set():
    app = _ViewerApp("")
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        assert not app.query("#library-media-review-banner")


def test_banner_omits_item_state_for_a_tombstoned_loaded_item(tmp_path):
    """A deleted-but-still-open item shows no per-item state (Qodo #2351).

    Progress is live-only, so claiming "✓ reviewed" for an item excluded
    from "X of M" would contradict the set's own arithmetic.
    """
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "All media", origin="browse", items=[(10, "A"), (11, "B")]
    )
    service.mark_item_done(set_id, backing_media_id=10, done=True)
    fake = _banner_fake(service, loaded=10)
    fake._review_set_live_ids = lambda ids: {11}  # 10 is a tombstone

    banner = LibraryScreen._active_review_set_banner(fake)

    assert banner is not None and "All media" in banner
    assert "✓ reviewed" not in banner and "not yet" not in banner
