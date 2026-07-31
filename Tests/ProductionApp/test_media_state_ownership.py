from __future__ import annotations

import asyncio
import inspect
import logging

import pytest
from loguru import logger
from textual.message import Message

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import TAB_MEDIA
from tldw_chatbook.Event_Handlers.media_events import (
    MediaAnalysisDeleteEvent,
    MediaAnalysisOverwriteEvent,
    MediaAnalysisRequestEvent,
    MediaAnalysisSaveAsNoteEvent,
    MediaAnalysisSaveEvent,
    MediaDeleteConfirmationEvent,
    MediaListCollapseEvent,
    MediaMetadataUpdateEvent,
    MediaReadingHighlightCreateEvent,
    MediaReadingHighlightDeleteEvent,
    MediaReadingHighlightUpdateEvent,
    MediaReadItLaterToggleEvent,
    MediaUndeleteEvent,
    SidebarCollapseEvent,
)
from tldw_chatbook.RAG_Search import ingestion_indexing
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.media_screen import MediaScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.Media.media_navigation_panel import (
    MediaTypeSelectedEvent,
)
from tldw_chatbook.Widgets.delete_confirmation_dialog import (
    create_delete_confirmation,
)
from tldw_chatbook.Widgets.Media.media_list_panel import MediaItemSelectedEvent
from tldw_chatbook.Widgets.Media.media_search_panel import (
    MediaBrowseSubviewChangedEvent,
    MediaSearchEvent,
)


def _disable_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


def _production_app(monkeypatch: pytest.MonkeyPatch) -> TldwCli:
    _disable_splash(monkeypatch)
    monkeypatch.setattr(
        ingestion_indexing,
        "semantic_indexing_available",
        lambda: False,
    )
    app = TldwCli()
    app.app_config["_first_run"] = False
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
    return app


async def _wait_for_media_screen(
    app: TldwCli,
    pilot,
) -> MediaScreen:
    for _ in range(300):
        if isinstance(app.screen, MediaScreen) and app.current_tab == TAB_MEDIA:
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError("production TldwCli did not finish routing to MediaScreen")


async def _wait_for_screen(app: TldwCli, pilot, screen_type):
    for _ in range(300):
        if isinstance(app.screen, screen_type):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError(
        f"production TldwCli did not finish routing to {screen_type.__name__}"
    )


async def _wait_until(pilot, predicate, failure: str) -> None:
    for _ in range(300):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError(failure)


async def _close_production_app(app: TldwCli) -> None:
    try:
        if app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()
        await app.on_shutdown_request()
        await app.on_unmount()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_real_media_metadata_event_mutates_and_refreshes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One real destination event must never fall through to the root handler."""
    app = _production_app(monkeypatch)

    try:
        async with app.run_test(size=(160, 50)) as pilot:
            app.post_message(NavigateToScreen("media"))
            screen = await _wait_for_media_screen(app, pilot)
            window = screen.media_window
            assert window is not None

            media_id, _media_uuid, message = app.media_db.add_media_with_keywords(
                title="TASK-652 original",
                media_type="document",
                content="TASK-652 production Media record",
                author="TASK-652",
                keywords=["task-652"],
            )
            assert media_id is not None, message
            detail = await app.media_reading_scope_service.get_media_detail(
                mode="local",
                media_id=media_id,
            )
            record_id = str(detail["id"])
            window.active_media_type = "all-media"
            window.selected_media_id = record_id
            window.runtime_state.active_media_type = "all-media"
            window.runtime_state.selected_record_id = record_id
            window.runtime_state.detail_by_record_id[record_id] = dict(detail)
            window.viewer_panel.load_media(detail)

            mutation_calls: list[int] = []
            real_update = app.media_db.update_media_metadata

            def record_update(*args, **kwargs):
                media_arg = kwargs["media_id"] if "media_id" in kwargs else args[0]
                mutation_calls.append(int(media_arg))
                return real_update(*args, **kwargs)

            monkeypatch.setattr(app.media_db, "update_media_metadata", record_update)

            refresh_calls: list[tuple[str, str, str]] = []
            monkeypatch.setattr(
                window,
                "_perform_search",
                lambda type_slug, search_term, keyword_filter: refresh_calls.append(
                    (type_slug, search_term, keyword_filter)
                ),
            )

            event = MediaMetadataUpdateEvent(
                media_id=media_id,
                record_id=record_id,
                backing_media_id=media_id,
                title="TASK-652 updated",
                media_type="document",
                author="TASK-652",
                url="",
                keywords=["task-652", "updated"],
                type_slug="all-media",
            )
            window.viewer_panel.post_message(event)

            await _wait_until(
                pilot,
                lambda: bool(mutation_calls),
                "metadata mutation did not reach the production scoped service",
            )
            await _wait_until(
                pilot,
                lambda: bool(refresh_calls),
                "metadata mutation did not refresh the production destination",
            )
            await _wait_until(
                pilot,
                lambda: event._stop_propagation,
                "metadata event did not stop at the production destination",
            )
            await pilot.pause()

            assert mutation_calls == [media_id]
            assert refresh_calls == [("all-media", "", "")]
            assert event._stop_propagation is True
    finally:
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_real_metadata_updates_are_last_edit_wins_in_durable_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slower earlier edit must not overwrite a newer edit for one record."""
    app = _production_app(monkeypatch)
    release_first = asyncio.Event()

    try:
        async with app.run_test(size=(160, 50)) as pilot:
            app.post_message(NavigateToScreen("media"))
            screen = await _wait_for_media_screen(app, pilot)
            window = screen.media_window
            assert window is not None

            media_id, _media_uuid, message = app.media_db.add_media_with_keywords(
                title="TASK-652 original metadata",
                media_type="document",
                content="TASK-652 metadata ordering record",
                author="TASK-652",
                keywords=["task-652"],
            )
            assert media_id is not None, message
            detail = await app.media_reading_scope_service.get_media_detail(
                mode="local",
                media_id=media_id,
            )
            record_id = str(detail["id"])
            window.active_media_type = "all-media"
            window.selected_media_id = record_id
            window.runtime_state.active_media_type = "all-media"
            window.runtime_state.selected_record_id = record_id
            window.runtime_state.detail_by_record_id[record_id] = dict(detail)
            window.viewer_panel.load_media(detail)

            first_started = asyncio.Event()
            second_started = asyncio.Event()
            finished_titles: list[str] = []
            real_update = app.local_media_reading_service.update_media_metadata

            async def reordered_update(media_id, **kwargs):
                title = str(kwargs["title"])
                if title == "TASK-652 older edit":
                    first_started.set()
                    await release_first.wait()
                else:
                    second_started.set()
                result = real_update(media_id, **kwargs)
                finished_titles.append(title)
                return result

            monkeypatch.setattr(
                app.local_media_reading_service,
                "update_media_metadata",
                reordered_update,
            )
            monkeypatch.setattr(window, "_perform_search", lambda *_args: None)

            def post_metadata(title: str) -> None:
                window.viewer_panel.post_message(
                    MediaMetadataUpdateEvent(
                        media_id=media_id,
                        record_id=record_id,
                        backing_media_id=media_id,
                        title=title,
                        media_type="document",
                        author="TASK-652",
                        url="",
                        keywords=["task-652"],
                        type_slug="all-media",
                    )
                )

            post_metadata("TASK-652 older edit")
            await _wait_until(
                pilot,
                first_started.is_set,
                "the older metadata update did not start",
            )

            post_metadata("TASK-652 newer edit")
            for _ in range(30):
                if second_started.is_set():
                    break
                await pilot.pause(0.01)
            if second_started.is_set():
                await _wait_until(
                    pilot,
                    lambda: "TASK-652 newer edit" in finished_titles,
                    "the concurrently started newer edit did not finish",
                )

            release_first.set()
            await _wait_until(
                pilot,
                lambda: len(finished_titles) == 2,
                "both metadata updates did not settle",
            )

            stored = app.media_db.get_media_by_id(media_id)
            assert stored is not None
            assert stored["title"] == "TASK-652 newer edit"
    finally:
        release_first.set()
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_real_metadata_ordering_survives_media_window_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement MediaWindow must share durable write ordering."""
    app = _production_app(monkeypatch)
    release_first = asyncio.Event()

    try:
        async with app.run_test(size=(160, 50)) as pilot:
            app.post_message(NavigateToScreen("media"))
            first_screen = await _wait_for_media_screen(app, pilot)
            first_window = first_screen.media_window
            assert first_window is not None

            media_id, _media_uuid, message = app.media_db.add_media_with_keywords(
                title="TASK-652 original replacement metadata",
                media_type="document",
                content="TASK-652 replacement-window ordering record",
                author="TASK-652",
                keywords=["task-652"],
            )
            assert media_id is not None, message
            detail = await app.media_reading_scope_service.get_media_detail(
                mode="local",
                media_id=media_id,
            )
            record_id = str(detail["id"])

            def select_record(window) -> None:
                window.active_media_type = "all-media"
                window.selected_media_id = record_id
                window.runtime_state.active_media_type = "all-media"
                window.runtime_state.selected_record_id = record_id
                window.runtime_state.detail_by_record_id[record_id] = dict(detail)
                window.viewer_panel.load_media(detail)
                monkeypatch.setattr(window, "_perform_search", lambda *_args: None)

            select_record(first_window)

            first_started = asyncio.Event()
            second_started = asyncio.Event()
            finished_titles: list[str] = []
            real_update = app.local_media_reading_service.update_media_metadata

            async def reordered_local_update(media_id, **metadata):
                title = str(metadata["title"])
                if title == "TASK-652 older replaced-window edit":
                    first_started.set()
                    await release_first.wait()
                else:
                    second_started.set()
                result = real_update(media_id, **metadata)
                finished_titles.append(title)
                return result

            monkeypatch.setattr(
                app.local_media_reading_service,
                "update_media_metadata",
                reordered_local_update,
            )

            def post_metadata(window, title: str) -> None:
                window.viewer_panel.post_message(
                    MediaMetadataUpdateEvent(
                        media_id=media_id,
                        record_id=record_id,
                        backing_media_id=media_id,
                        title=title,
                        media_type="document",
                        author="TASK-652",
                        url="",
                        keywords=["task-652"],
                        type_slug="all-media",
                    )
                )

            post_metadata(first_window, "TASK-652 older replaced-window edit")
            await _wait_until(
                pilot,
                first_started.is_set,
                "the older replacement-window edit did not start",
            )

            app.post_message(NavigateToScreen("settings"))
            await _wait_for_screen(app, pilot, SettingsScreen)
            await _wait_until(
                pilot,
                lambda: first_window._closed and first_window._parent is None,
                "the replaced Media owner did not finish teardown",
            )

            app.post_message(NavigateToScreen("media"))
            second_screen = await _wait_for_media_screen(app, pilot)
            second_window = second_screen.media_window
            assert second_window is not None
            assert second_window is not first_window
            select_record(second_window)
            post_metadata(second_window, "TASK-652 newer replaced-window edit")

            for _ in range(30):
                if second_started.is_set():
                    break
                await pilot.pause(0.01)
            if second_started.is_set():
                await _wait_until(
                    pilot,
                    lambda: "TASK-652 newer replaced-window edit" in finished_titles,
                    "the newer replacement-window edit did not finish",
                )

            release_first.set()
            await _wait_until(
                pilot,
                lambda: len(finished_titles) == 2,
                "both replacement-window metadata updates did not settle",
            )

            stored = app.media_db.get_media_by_id(media_id)
            assert stored is not None
            assert stored["title"] == "TASK-652 newer replaced-window edit"
    finally:
        release_first.set()
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_real_media_destination_stops_handled_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handled navigation and mutation messages stop at the production owner."""
    app = _production_app(monkeypatch)

    try:
        async with app.run_test(size=(160, 50)) as pilot:
            app.post_message(NavigateToScreen("media"))
            screen = await _wait_for_media_screen(app, pilot)
            window = screen.media_window
            assert window is not None
            window.active_media_type = "all-media"
            window.runtime_state.active_media_type = "all-media"

            monkeypatch.setattr(window, "_perform_search", lambda *args: None)

            def discard_worker(awaitable, *args, **kwargs):
                if inspect.iscoroutine(awaitable):
                    awaitable.close()
                return None

            monkeypatch.setattr(window, "run_worker", discard_worker)

            async def accept_undelete(**_kwargs):
                return True

            monkeypatch.setattr(
                app.media_reading_scope_service,
                "undelete_media",
                accept_undelete,
            )

            bubble_targets: dict[int, list[object]] = {}
            real_bubble_to = Message._bubble_to

            def record_bubble_to(message: Message, target) -> None:
                bubble_targets.setdefault(id(message), []).append(target)
                real_bubble_to(message, target)

            monkeypatch.setattr(Message, "_bubble_to", record_bubble_to)

            record_id = "local:media:652"
            media_data = {
                "id": record_id,
                "source_id": "652",
                "backing_media_id": 652,
                "backend": "local",
                "title": "TASK-652 propagation",
            }
            events = [
                (
                    window.nav_panel,
                    MediaTypeSelectedEvent("all-media", "All Media"),
                ),
                (window.search_panel, MediaSearchEvent("", "", False)),
                (
                    window.search_panel,
                    MediaBrowseSubviewChangedEvent("all"),
                ),
                (
                    window.list_panel,
                    MediaItemSelectedEvent(record_id, media_data),
                ),
                (
                    window.viewer_panel,
                    MediaDeleteConfirmationEvent(
                        652,
                        "TASK-652 propagation",
                        "all-media",
                        record_id=record_id,
                        backing_media_id=652,
                    ),
                ),
                (
                    window.viewer_panel,
                    MediaUndeleteEvent(
                        652,
                        "all-media",
                        record_id=record_id,
                    ),
                ),
                (
                    window.viewer_panel,
                    MediaReadItLaterToggleEvent(
                        652,
                        record_id=record_id,
                        save_for_later=True,
                    ),
                ),
                (
                    window.viewer_panel,
                    MediaReadingHighlightCreateEvent(
                        652,
                        record_id=record_id,
                        quote="TASK-652",
                        media_data=media_data,
                    ),
                ),
                (
                    window.viewer_panel,
                    MediaReadingHighlightUpdateEvent(
                        652,
                        record_id=record_id,
                        highlight_id=1,
                        note="TASK-652",
                        media_data=media_data,
                    ),
                ),
                (
                    window.viewer_panel,
                    MediaReadingHighlightDeleteEvent(
                        652,
                        record_id=record_id,
                        highlight_id=1,
                        media_data=media_data,
                    ),
                ),
                (
                    window.viewer_panel,
                    MediaAnalysisRequestEvent(
                        652,
                        provider="OpenAI",
                        model="task-652",
                        system_prompt="TASK-652 private system",
                        user_prompt="TASK-652 private user",
                        type_slug="all-media",
                        record_id=record_id,
                        backing_media_id=652,
                    ),
                ),
                (
                    window.viewer_panel,
                    MediaAnalysisSaveEvent(
                        652,
                        "TASK-652 private analysis",
                        "all-media",
                        record_id=record_id,
                    ),
                ),
                (
                    window.viewer_panel,
                    MediaAnalysisSaveAsNoteEvent(
                        652,
                        "TASK-652 propagation",
                        "TASK-652 private analysis",
                        record_id=record_id,
                    ),
                ),
                (
                    window.viewer_panel,
                    MediaAnalysisOverwriteEvent(
                        652,
                        "TASK-652 private analysis",
                        "all-media",
                        record_id=record_id,
                    ),
                ),
                (
                    window.viewer_panel,
                    MediaAnalysisDeleteEvent(
                        652,
                        "task-652-version",
                        "all-media",
                        record_id=record_id,
                        version_number=1,
                    ),
                ),
                (window.viewer_panel, MediaListCollapseEvent()),
                (window.search_panel, SidebarCollapseEvent()),
            ]

            for origin, event in events:
                origin.post_message(event)
                await _wait_until(
                    pilot,
                    lambda: window in bubble_targets.get(id(event), []),
                    f"{type(event).__name__} did not reach the production owner",
                )
                await pilot.pause()
                assert bubble_targets[id(event)][-1] is window, type(event).__name__
                assert event._stop_propagation is True, type(event).__name__
    finally:
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_real_metadata_mutation_survives_media_screen_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Durable metadata work completes while a stale owner performs no refresh."""
    app = _production_app(monkeypatch)

    try:
        async with app.run_test(size=(160, 50)) as pilot:
            app.post_message(NavigateToScreen("media"))
            screen = await _wait_for_media_screen(app, pilot)
            window = screen.media_window
            assert window is not None

            media_id, _media_uuid, message = app.media_db.add_media_with_keywords(
                title="TASK-652 before teardown",
                media_type="document",
                content="TASK-652 durable metadata record",
                author="TASK-652",
                keywords=["task-652"],
            )
            assert media_id is not None, message
            detail = await app.media_reading_scope_service.get_media_detail(
                mode="local",
                media_id=media_id,
            )
            record_id = str(detail["id"])
            window.active_media_type = "all-media"
            window.selected_media_id = record_id
            window.runtime_state.active_media_type = "all-media"
            window.runtime_state.selected_record_id = record_id
            window.runtime_state.detail_by_record_id[record_id] = dict(detail)
            window.viewer_panel.load_media(detail)

            update_started = asyncio.Event()
            release_update = asyncio.Event()
            update_finished = asyncio.Event()
            committed = False
            real_update = app.local_media_reading_service.update_media_metadata

            async def delayed_real_update(media_id, **kwargs):
                nonlocal committed
                update_started.set()
                try:
                    await release_update.wait()
                    result = real_update(media_id, **kwargs)
                    committed = True
                    return result
                finally:
                    update_finished.set()

            monkeypatch.setattr(
                app.local_media_reading_service,
                "update_media_metadata",
                delayed_real_update,
            )
            stale_refreshes: list[tuple[str, str, str]] = []
            monkeypatch.setattr(
                window,
                "_perform_search",
                lambda type_slug, search_term, keyword_filter: stale_refreshes.append(
                    (type_slug, search_term, keyword_filter)
                ),
            )

            event = MediaMetadataUpdateEvent(
                media_id=media_id,
                record_id=record_id,
                backing_media_id=media_id,
                title="TASK-652 after teardown",
                media_type="document",
                author="TASK-652",
                url="",
                keywords=["task-652", "durable"],
                type_slug="all-media",
            )
            window.viewer_panel.post_message(event)
            await _wait_until(
                pilot,
                update_started.is_set,
                "metadata update did not start on the production destination",
            )

            app.post_message(NavigateToScreen("settings"))
            await _wait_for_screen(app, pilot, SettingsScreen)
            await _wait_until(
                pilot,
                lambda: window._closed and window._parent is None,
                "the stale Media owner did not finish teardown",
            )

            release_update.set()
            await _wait_until(
                pilot,
                update_finished.is_set,
                "metadata update did not settle after owner teardown",
            )

            assert committed is True
            assert stale_refreshes == []
            stored = app.media_db.get_media_by_id(media_id)
            assert stored is not None
            assert stored["title"] == "TASK-652 after teardown"
    finally:
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_real_metadata_completion_ignores_changed_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable mutation may settle, but a changed selection owns presentation."""
    app = _production_app(monkeypatch)
    release_update = asyncio.Event()

    try:
        async with app.run_test(size=(160, 50)) as pilot:
            app.post_message(NavigateToScreen("media"))
            screen = await _wait_for_media_screen(app, pilot)
            window = screen.media_window
            assert window is not None

            old_id = "local:media:old-metadata"
            new_id = "local:media:new-selection"
            old_record = {
                "id": old_id,
                "source_id": "old-metadata",
                "backing_media_id": "old-metadata",
                "backend": "local",
                "title": "TASK-652 old selection",
            }
            window.active_media_type = "all-media"
            window.selected_media_id = old_id
            window.runtime_state.active_media_type = "all-media"
            window.runtime_state.selected_record_id = old_id
            window.runtime_state.detail_by_record_id[old_id] = dict(old_record)
            window.viewer_panel.load_media(old_record)

            update_started = asyncio.Event()
            update_finished = asyncio.Event()

            async def delayed_update(_media_id, **_kwargs):
                update_started.set()
                await release_update.wait()
                update_finished.set()
                return True

            async def load_new_detail(*, mode, media_id):
                return {
                    "id": new_id,
                    "source_id": str(media_id),
                    "backing_media_id": str(media_id),
                    "backend": mode,
                    "title": "TASK-652 new selection",
                    "reading_progress": {},
                    "reading_highlights": [],
                }

            monkeypatch.setattr(
                app.local_media_reading_service,
                "update_media_metadata",
                delayed_update,
            )
            monkeypatch.setattr(
                app.media_reading_scope_service,
                "get_media_detail",
                load_new_detail,
            )

            async def no_versions(_record):
                return []

            monkeypatch.setattr(window, "_fetch_document_versions", no_versions)
            stale_refreshes: list[tuple[str, str, str]] = []
            monkeypatch.setattr(
                window,
                "_perform_search",
                lambda type_slug, search_term, keyword_filter: stale_refreshes.append(
                    (type_slug, search_term, keyword_filter)
                ),
            )

            window.viewer_panel.post_message(
                MediaMetadataUpdateEvent(
                    media_id="old-metadata",
                    record_id=old_id,
                    backing_media_id="old-metadata",
                    title="TASK-652 updated old selection",
                    media_type="document",
                    author="TASK-652",
                    url="",
                    keywords=["task-652"],
                    type_slug="all-media",
                )
            )
            await _wait_until(
                pilot,
                update_started.is_set,
                "metadata update did not start",
            )

            window.list_panel.post_message(
                MediaItemSelectedEvent(
                    new_id,
                    {
                        "id": new_id,
                        "source_id": "new-selection",
                        "backing_media_id": "new-selection",
                        "backend": "local",
                    },
                )
            )
            await _wait_until(
                pilot,
                lambda: (
                    window.selected_media_id == new_id
                    and getattr(window.viewer_panel, "media_data", {}).get("id")
                    == new_id
                ),
                "newer selection was not presented",
            )

            release_update.set()
            await _wait_until(
                pilot,
                update_finished.is_set,
                "metadata update did not settle",
            )
            await pilot.pause()

            assert window.selected_media_id == new_id
            assert window.runtime_state.selected_record_id == new_id
            assert getattr(window.viewer_panel, "media_data", {}).get("id") == new_id
            assert stale_refreshes == []
    finally:
        release_update.set()
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_real_metadata_failure_is_bounded_and_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata failures expose a recovery action, not payload or exception text."""
    app = _production_app(monkeypatch)
    private_value = "TASK-652-PRIVATE-METADATA-DO-NOT-LOG"

    try:
        async with app.run_test(size=(160, 50)) as pilot:
            app.post_message(NavigateToScreen("media"))
            screen = await _wait_for_media_screen(app, pilot)
            window = screen.media_window
            assert window is not None

            record_id = "local:media:private-failure"
            record = {
                "id": record_id,
                "source_id": "private-failure",
                "backing_media_id": "private-failure",
                "backend": "local",
                "title": "TASK-652 pre-failure",
            }
            window.active_media_type = "all-media"
            window.selected_media_id = record_id
            window.runtime_state.active_media_type = "all-media"
            window.runtime_state.selected_record_id = record_id
            window.runtime_state.detail_by_record_id[record_id] = dict(record)
            window.viewer_panel.load_media(record)

            async def fail_update(_media_id, **_kwargs):
                raise RuntimeError(private_value)

            monkeypatch.setattr(
                app.local_media_reading_service,
                "update_media_metadata",
                fail_update,
            )
            notifications: list[str] = []
            monkeypatch.setattr(
                app,
                "notify",
                lambda message, **_kwargs: notifications.append(str(message)),
            )
            captured_logs: list[str] = []
            sink_id = logger.add(captured_logs.append, level="DEBUG")
            try:
                event = MediaMetadataUpdateEvent(
                    media_id="private-failure",
                    record_id=record_id,
                    backing_media_id="private-failure",
                    title=private_value,
                    media_type="document",
                    author=private_value,
                    url=private_value,
                    keywords=[private_value],
                    type_slug="all-media",
                )
                window.viewer_panel.post_message(event)
                await _wait_until(
                    pilot,
                    lambda: bool(notifications),
                    "metadata failure did not produce bounded recovery",
                )
            finally:
                logger.remove(sink_id)

            rendered = "\n".join([*captured_logs, *notifications])
            assert event._stop_propagation is True
            assert notifications == [
                "Media metadata could not be updated; retry the edit."
            ]
            assert private_value not in rendered
            assert "RuntimeError" in rendered
    finally:
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_real_media_detail_reverse_completion_keeps_newest_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older detail completion must not overwrite the newer real selection."""
    app = _production_app(monkeypatch)
    release_old = asyncio.Event()

    try:
        async with app.run_test(size=(160, 50)) as pilot:
            app.post_message(NavigateToScreen("media"))
            screen = await _wait_for_media_screen(app, pilot)
            window = screen.media_window
            assert window is not None
            window.active_media_type = "all-media"
            window.runtime_state.active_media_type = "all-media"

            old_started = asyncio.Event()
            new_started = asyncio.Event()

            async def controlled_detail(*, mode, media_id):
                record_id = f"{mode}:media:{media_id}"
                if str(media_id) == "old":
                    old_started.set()
                    await release_old.wait()
                    title = "TASK-652 stale detail"
                else:
                    new_started.set()
                    title = "TASK-652 newest detail"
                return {
                    "id": record_id,
                    "source_id": str(media_id),
                    "backing_media_id": str(media_id),
                    "backend": mode,
                    "title": title,
                    "reading_progress": {},
                    "reading_highlights": [],
                }

            monkeypatch.setattr(
                app.media_reading_scope_service,
                "get_media_detail",
                controlled_detail,
            )

            async def no_versions(_record):
                return []

            monkeypatch.setattr(window, "_fetch_document_versions", no_versions)

            presented: list[str] = []
            real_load_media = window.viewer_panel.load_media

            def record_presentation(media_data):
                presented.append(str(media_data.get("id")))
                real_load_media(media_data)

            monkeypatch.setattr(
                window.viewer_panel,
                "load_media",
                record_presentation,
            )

            old_id = "local:media:old"
            new_id = "local:media:new"
            window.list_panel.post_message(
                MediaItemSelectedEvent(
                    old_id,
                    {
                        "id": old_id,
                        "source_id": "old",
                        "backing_media_id": "old",
                        "backend": "local",
                    },
                )
            )
            await _wait_until(
                pilot,
                old_started.is_set,
                "older detail request did not start",
            )

            try:
                window.list_panel.post_message(
                    MediaItemSelectedEvent(
                        new_id,
                        {
                            "id": new_id,
                            "source_id": "new",
                            "backing_media_id": "new",
                            "backend": "local",
                        },
                    )
                )
                try:
                    await asyncio.wait_for(new_started.wait(), timeout=0.5)
                except TimeoutError as exc:
                    raise AssertionError(
                        "newer detail request was blocked behind the older request"
                    ) from exc
                await _wait_until(
                    pilot,
                    lambda: presented and presented[-1] == new_id,
                    "newer detail was not presented",
                )

                release_old.set()
                await pilot.pause()
                await pilot.pause()

                assert window.selected_media_id == new_id
                assert window.runtime_state.selected_record_id == new_id
                assert presented[-1] == new_id
                assert (
                    getattr(window.viewer_panel, "media_data", {}).get("id") == new_id
                )
            finally:
                release_old.set()
    finally:
        release_old.set()
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_real_media_search_reverse_completion_keeps_newest_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older search completion must not overwrite the newer real query."""
    app = _production_app(monkeypatch)
    release_old = asyncio.Event()

    try:
        async with app.run_test(size=(160, 50)) as pilot:
            app.post_message(NavigateToScreen("media"))
            screen = await _wait_for_media_screen(app, pilot)
            window = screen.media_window
            assert window is not None
            window.active_media_type = "all-media"
            window.runtime_state.active_media_type = "all-media"

            old_started = asyncio.Event()
            old_finished = asyncio.Event()
            new_started = asyncio.Event()

            async def controlled_search(*, mode, query, limit, offset, **filters):
                del limit, offset, filters
                if query == "TASK-652 old query":
                    old_started.set()
                    try:
                        await release_old.wait()
                    except asyncio.CancelledError:
                        await release_old.wait()
                    old_finished.set()
                    item_id = "local:media:old-search"
                else:
                    new_started.set()
                    item_id = "local:media:new-search"
                return {
                    "items": [
                        {
                            "id": item_id,
                            "source_id": item_id.rsplit(":", 1)[-1],
                            "backend": mode,
                            "title": item_id,
                        }
                    ],
                    "total": 1,
                }

            monkeypatch.setattr(
                app.media_reading_scope_service,
                "search_media",
                controlled_search,
            )

            window.search_panel.search_term = "TASK-652 old query"
            window.search_panel.keyword_filter = ""
            window.search_panel.post_message(
                MediaSearchEvent("TASK-652 old query", "", False)
            )
            await _wait_until(
                pilot,
                old_started.is_set,
                "older search request did not start",
            )

            window.search_panel.search_term = "TASK-652 new query"
            window.search_panel.post_message(
                MediaSearchEvent("TASK-652 new query", "", False)
            )
            await _wait_until(
                pilot,
                new_started.is_set,
                "newer search request did not start",
            )
            await _wait_until(
                pilot,
                lambda: (
                    [item.get("id") for item in window.runtime_state.browse_items]
                    == ["local:media:new-search"]
                ),
                "newer query results were not presented",
            )

            release_old.set()
            await _wait_until(
                pilot,
                old_finished.is_set,
                "older search did not settle after release",
            )
            await pilot.pause()

            assert window.runtime_state.search_term == "TASK-652 new query"
            assert [item.get("id") for item in window.runtime_state.browse_items] == [
                "local:media:new-search"
            ]
    finally:
        release_old.set()
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_real_media_search_completion_survives_temporary_modal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A modal above the mounted Media route must not invalidate its owner."""
    app = _production_app(monkeypatch)
    release_search = asyncio.Event()

    try:
        async with app.run_test(size=(160, 50)) as pilot:
            app.post_message(NavigateToScreen("media"))
            screen = await _wait_for_media_screen(app, pilot)
            window = screen.media_window
            assert window is not None
            window.active_media_type = "all-media"
            window.runtime_state.active_media_type = "all-media"

            search_started = asyncio.Event()
            search_finished = asyncio.Event()

            async def delayed_search(*, mode, query, limit, offset, **filters):
                del query, limit, offset, filters
                search_started.set()
                await release_search.wait()
                search_finished.set()
                return {
                    "items": [
                        {
                            "id": "local:media:modal-result",
                            "source_id": "modal-result",
                            "backend": mode,
                            "title": "TASK-652 modal result",
                        }
                    ],
                    "total": 1,
                }

            monkeypatch.setattr(
                app.media_reading_scope_service,
                "search_media",
                delayed_search,
            )
            window.search_panel.search_term = "modal query"
            window.search_panel.post_message(MediaSearchEvent("modal query", "", False))
            await _wait_until(
                pilot,
                search_started.is_set,
                "Media search did not start before the modal opened",
            )

            dialog = create_delete_confirmation(
                item_type="Media",
                item_name="TASK-652 modal",
                permanent=True,
            )
            app.push_screen(dialog)
            await _wait_until(
                pilot,
                lambda: app.screen is dialog,
                "production confirmation dialog did not become active",
            )
            assert window.screen is screen
            assert screen in app.screen_stack

            release_search.set()
            await _wait_until(
                pilot,
                search_finished.is_set,
                "Media search did not settle beneath the modal",
            )
            await _wait_until(
                pilot,
                lambda: (
                    [item.get("id") for item in window.runtime_state.browse_items]
                    == ["local:media:modal-result"]
                    and window.list_panel.loading is False
                ),
                "mounted Media owner discarded its completion beneath the modal",
            )

            app.pop_screen()
            await _wait_until(
                pilot,
                lambda: app.screen is screen,
                "production confirmation dialog did not close",
            )
            assert window.runtime_state.search_term == "modal query"
    finally:
        release_search.set()
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_real_read_later_refresh_cannot_overwrite_newer_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mutation-triggered refresh uses the same stale-safe search contract."""
    app = _production_app(monkeypatch)
    release_refresh = asyncio.Event()

    try:
        async with app.run_test(size=(160, 50)) as pilot:
            app.post_message(NavigateToScreen("media"))
            screen = await _wait_for_media_screen(app, pilot)
            window = screen.media_window
            assert window is not None

            record_id = "local:media:read-later"
            record = {
                "id": record_id,
                "source_id": "read-later",
                "backing_media_id": "read-later",
                "backend": "local",
                "title": "TASK-652 read-later",
            }
            window.active_media_type = "all-media"
            window.selected_media_id = record_id
            window.runtime_state.active_media_type = "all-media"
            window.runtime_state.selected_record_id = record_id
            window.runtime_state.detail_by_record_id[record_id] = dict(record)
            window.viewer_panel.load_media(record)

            async def save_for_later(**_kwargs):
                return {"is_read_it_later": True}

            refresh_started = asyncio.Event()
            refresh_finished = asyncio.Event()
            newest_started = asyncio.Event()

            async def controlled_search(*, mode, query, limit, offset, **filters):
                del limit, offset, filters
                if query == "TASK-652 mutation refresh":
                    refresh_started.set()
                    try:
                        await release_refresh.wait()
                    except asyncio.CancelledError:
                        await release_refresh.wait()
                    refresh_finished.set()
                    item_id = "local:media:stale-refresh"
                else:
                    newest_started.set()
                    item_id = "local:media:newest-search"
                return {
                    "items": [
                        {
                            "id": item_id,
                            "source_id": item_id.rsplit(":", 1)[-1],
                            "backend": mode,
                            "title": item_id,
                        }
                    ],
                    "total": 1,
                }

            monkeypatch.setattr(
                app.media_reading_scope_service,
                "save_to_read_it_later",
                save_for_later,
            )
            monkeypatch.setattr(
                app.media_reading_scope_service,
                "search_media",
                controlled_search,
            )

            window.search_panel.search_term = "TASK-652 mutation refresh"
            window.viewer_panel.post_message(
                MediaReadItLaterToggleEvent(
                    "read-later",
                    record_id=record_id,
                    save_for_later=True,
                )
            )
            await _wait_until(
                pilot,
                refresh_started.is_set,
                "mutation-triggered browse refresh did not start",
            )

            window.search_panel.search_term = "TASK-652 newest search"
            window.search_panel.post_message(
                MediaSearchEvent("TASK-652 newest search", "", False)
            )
            await _wait_until(
                pilot,
                newest_started.is_set,
                "newer user search did not start",
            )
            await _wait_until(
                pilot,
                lambda: (
                    [item.get("id") for item in window.runtime_state.browse_items]
                    == ["local:media:newest-search"]
                ),
                "newer user search was not presented",
            )

            release_refresh.set()
            await _wait_until(
                pilot,
                refresh_finished.is_set,
                "mutation-triggered refresh did not settle",
            )
            await pilot.pause()

            assert window.runtime_state.search_term == "TASK-652 newest search"
            assert [item.get("id") for item in window.runtime_state.browse_items] == [
                "local:media:newest-search"
            ]
    finally:
        release_refresh.set()
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_real_media_window_owns_state_and_restores_actual_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real destination owns Media state while screen snapshots survive visits."""
    app = _production_app(monkeypatch)
    retired_root_names = (
        "media_active_view",
        "_initial_media_view_slug",
        "current_media_type_filter_slug",
        "current_media_type_filter_display_name",
        "media_current_page",
        "current_loaded_media_item",
        "_media_search_timers",
        "_media_search_generation",
        "_initial_media_view",
        "media_runtime_state",
    )

    try:
        async with app.run_test(size=(160, 50)) as pilot:
            app.post_message(NavigateToScreen("media"))
            first_screen = await _wait_for_media_screen(app, pilot)
            first_window = first_screen.media_window
            assert first_window is not None

            assert all(not hasattr(app, name) for name in retired_root_names)
            assert not hasattr(first_screen, "media_runtime_state")
            assert first_window.runtime_state.runtime_backend == (
                app.get_authoritative_runtime_source()
            )

            media_id, _media_uuid, message = app.media_db.add_media_with_keywords(
                title="restored query",
                media_type="document",
                content="TASK-652 snapshot-owned selection",
                author="TASK-652",
                keywords=["task-652"],
            )
            assert media_id is not None, message
            detail = await app.media_reading_scope_service.get_media_detail(
                mode="local",
                media_id=media_id,
            )
            record_id = str(detail["id"])
            first_window.active_media_type = "all-media"
            first_window.runtime_state.active_media_type = "all-media"
            first_window.selected_media_id = record_id
            first_window.runtime_state.selected_record_id = record_id
            first_window.search_panel.search_term = "restored query"
            first_window.search_panel.keyword_filter = "task-652"
            matching_results, _total = await first_window._execute_browse_query_async(
                type_slug="all-media",
                search_term="restored query",
                keyword_filter="task-652",
            )
            assert record_id in {str(item.get("id")) for item in matching_results}

            app.post_message(NavigateToScreen("settings"))
            await _wait_for_screen(app, pilot, SettingsScreen)
            app.post_message(NavigateToScreen("media"))
            second_screen = await _wait_for_media_screen(app, pilot)
            second_window = second_screen.media_window
            assert second_window is not None
            assert second_window is not first_window

            await _wait_until(
                pilot,
                lambda: (
                    second_window.active_media_type == "all-media"
                    and second_window.search_panel.search_term == "restored query"
                    and second_window.search_panel.keyword_filter == "task-652"
                ),
                "MediaScreen did not restore the actual destination snapshot",
            )
            await _wait_until(
                pilot,
                lambda: (
                    (getattr(second_window.viewer_panel, "media_data", None) or {}).get(
                        "id"
                    )
                    == record_id
                ),
                "MediaScreen did not restore the selected destination record",
            )
            assert second_window.selected_media_id == record_id
            assert second_window.runtime_state.selected_record_id == record_id
            assert all(not hasattr(app, name) for name in retired_root_names)
            assert not hasattr(second_screen, "media_runtime_state")
    finally:
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_real_media_snapshot_clears_a_missing_restored_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deleted record cannot remain the destination's restored selection."""
    app = _production_app(monkeypatch)
    missing_record_id = "local:media:missing-task-652"

    try:
        async with app.run_test(size=(160, 50)) as pilot:
            app.post_message(NavigateToScreen("media"))
            first_screen = await _wait_for_media_screen(app, pilot)
            first_window = first_screen.media_window
            assert first_window is not None
            first_window.active_media_type = "all-media"
            first_window.runtime_state.active_media_type = "all-media"
            first_window.selected_media_id = missing_record_id
            first_window.runtime_state.selected_record_id = missing_record_id

            app.post_message(NavigateToScreen("settings"))
            await _wait_for_screen(app, pilot, SettingsScreen)
            app.post_message(NavigateToScreen("media"))
            second_screen = await _wait_for_media_screen(app, pilot)
            second_window = second_screen.media_window
            assert second_window is not None

            await _wait_until(
                pilot,
                lambda: (
                    second_window._pending_restored_selection_id is None
                    and second_window.list_panel.loading is False
                ),
                "missing restored selection did not settle",
            )

            assert second_window.selected_media_id is None
            assert second_window.runtime_state.selected_record_id is None
            assert second_window.list_panel.selected_id is None
            assert second_window.viewer_panel.media_data is None
    finally:
        await _close_production_app(app)
