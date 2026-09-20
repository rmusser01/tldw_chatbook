"""Registered Chatbooks exercise the actual Library canvas and owner contracts."""

from __future__ import annotations

import pytest
from rich.text import Text
from textual.widgets import Button, Input, Markdown, OptionList, SelectionList, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_library_artifacts_canvas import artifact_library
from Tests.UI.test_library_artifacts_sharing import exported_registry
from Tests.UI.test_library_artifacts_sharing import (
    staged_controller as staged_controller,  # noqa: PLC0414 - re-export shared pytest fixture
)
from tldw_chatbook.Chat.console_save_targets import console_chatbook_artifact_payload
from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
from tldw_chatbook.Library.library_artifacts_state import ArtifactKey

pytestmark = pytest.mark.ui


async def settled(pilot, predicate):
    for _ in range(160):
        if predicate():
            return
        await pilot.pause(0.025)
    raise AssertionError("The Library Chatbooks canvas did not settle")


def registry_for(screen, tmp_path):
    service = LocalChatbookService(registry_path=tmp_path / "chatbooks.json")
    screen.app_instance.local_chatbook_service = service
    return service


async def show_chatbooks(screen, pilot, key=None):
    await screen._select_library_rail_row("artifacts-chatbooks")
    c = screen._artifacts_controller
    await settled(
        pilot, lambda: c.page and c.page.scope.view == "chatbooks" and not c.loading
    )
    if key is not None:
        rows = screen.query_one("#library-artifacts-list", OptionList)
        rows.highlighted = next(
            i for i, row in enumerate(c.page.items) if row.key == key
        )
    await settled(
        pilot,
        lambda: (
            c.detail and not c.detail_loading and (key is None or c.detail.key == key)
        ),
    )
    await pilot.pause()
    return c


def visible_text(widget):
    rendered = widget.renderable
    return rendered.plain if isinstance(rendered, Text) else str(rendered)


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [1800, 21000])
@private_profile_test
async def test_chatbooks_reader_shows_full_stored_response_and_truthful_excerpt(
    request,
    tmp_path,
    length,
):
    async with artifact_library(tmp_path) as (screen, pilot):
        service = registry_for(screen, tmp_path)
        body = ("A complete saved paragraph with a tail marker.\n\n" * 500)[:length]
        payload = console_chatbook_artifact_payload(
            title="Saved field notes",
            message_text=body,
            message_role="Assistant",
            message_id="saved-message",
            provider="Test provider",
            model="Test model",
        )
        record = await service.create_chatbook(**payload)
        key = ArtifactKey("chatbook", record["chatbook_id"])
        c = await show_chatbooks(screen, pilot, key)
        expected = payload["metadata"]["content"]
        assert len(expected) > 1000
        assert c.detail.body == expected
        markdown = screen.query_one("#library-artifacts-markdown", Markdown)
        assert markdown._markdown == expected
        retention = visible_text(
            screen.query_one("#library-artifacts-retention", Static)
        ).lower()
        assert "saved response" in retention
        assert ("excerpt" in retention) is (length > 20000)
        assert "no exported bundle" in retention
        assert not screen.query_one("#library-artifacts-share", Button).display
        assert not screen.query_one("#library-artifacts-export", Button).display
        assert not screen.query_one("#library-artifacts-keep", Button).display


@pytest.mark.asyncio
@private_profile_test
async def test_missing_export_still_previews_metadata_and_explains_sharing_limit(
    request,
    tmp_path,
):
    async with artifact_library(tmp_path) as (screen, pilot):
        service = registry_for(screen, tmp_path)
        description = "Registered expedition notes. The original pack has moved."
        record = await service.create_chatbook(
            name="Moved expedition pack",
            description=description,
            file_path=tmp_path / "missing.zip",
        )
        c = await show_chatbooks(
            screen, pilot, ArtifactKey("chatbook", record["chatbook_id"])
        )
        assert c.detail.body == description
        assert (
            screen.query_one("#library-artifacts-markdown", Markdown)._markdown
            == description
        )
        assert not screen.query_one("#library-artifacts-share", Button).display
        retention = visible_text(
            screen.query_one("#library-artifacts-retention", Static)
        ).lower()
        assert "missing" in retention
        assert "existing exported zip" not in retention
        screen.query_one("#library-artifacts-details", Button).press()
        await pilot.pause()
        details = visible_text(screen.query_one("#library-artifacts-content", Static))
        assert "Exported bundle is missing" in details
        assert "Source conversation unavailable" in details
        assert not screen.query_one("#library-artifacts-source", Button).display


@pytest.mark.asyncio
@private_profile_test
async def test_selected_chatbook_console_and_source_actions_preserve_exact_identity(
    request,
    tmp_path,
    monkeypatch,
):
    async with artifact_library(tmp_path) as (screen, pilot):
        service = registry_for(screen, tmp_path)
        app = screen.app_instance
        source = app.chachanotes_db.add_conversation(
            {"title": "Original field conversation"}
        )
        body = "Exact selected answer. " * 90
        payload = console_chatbook_artifact_payload(
            title="Selected saved answer",
            message_text=body,
            message_role="Assistant",
            conversation_id=source,
            message_id="exact-selected-message",
            provider="Exact provider",
            model="Exact model",
        )
        target = await service.create_chatbook(**payload)
        await service.create_chatbook(
            name="Newer unrelated pack", description="Never launch this record"
        )
        launches, sources = [], []
        monkeypatch.setattr(
            app, "open_console_for_live_work", lambda **kwargs: launches.append(kwargs)
        )
        monkeypatch.setattr(app, "resume_console_conversation", sources.append)
        c = await show_chatbooks(
            screen, pilot, ArtifactKey("chatbook", target["chatbook_id"])
        )
        screen.query_one("#library-artifacts-console", Button).press()
        await settled(pilot, lambda: bool(launches) and not c.busy)
        launch = launches[0]
        assert launch["source"] == "artifacts"
        assert launch["title"] == "Selected saved answer"
        context = launch["payload"]
        assert context["target_id"] == f"local:chatbook:{target['chatbook_id']}"
        assert context["chatbook_id"] == target["chatbook_id"]
        assert context["record_id"] == target["id"]
        assert context["conversation_id"] == source
        assert context["message_id"] == "exact-selected-message"
        assert context["provider"] == "Exact provider"
        assert context["model"] == "Exact model"
        # Existing Console handoff intentionally uses a bounded preview; reading
        # above verifies that the Library body is the complete stored response.
        assert context["content_preview"] == body[:1000].strip()
        assert context["content_truncated"] is False
        screen.query_one("#library-artifacts-source", Button).press()
        await settled(pilot, lambda: bool(sources) and not c.busy)
        assert sources == [source]
        assert c.selected.native_id == target["chatbook_id"]


@pytest.mark.asyncio
@private_profile_test
async def test_chatbooks_manager_link_keeps_existing_chatbooks_route(request, tmp_path):
    async with artifact_library(tmp_path) as (screen, pilot):
        registry_for(screen, tmp_path)
        await screen._select_library_rail_row("artifacts-chatbooks")
        c = screen._artifacts_controller
        await settled(
            pilot, lambda: c.page and c.page.scope.view == "chatbooks" and not c.loading
        )
        assert c.page.total == 0
        manage = screen.query_one("#library-artifacts-manage", Button)
        assert manage.display and not manage.disabled
        manage.press()
        await settled(pilot, lambda: bool(screen.app.seen_routes))
        assert screen.app.seen_routes == ["chatbooks"]


@pytest.mark.asyncio
@private_profile_test
async def test_chatbooks_and_reports_restore_independent_query_selection_and_scroll(
    request,
    tmp_path,
):
    async with artifact_library(tmp_path, size=(160, 38)) as (screen, pilot):
        service = registry_for(screen, tmp_path)
        long_body = "\n\n".join(
            f"Paragraph {i}: readable field observations." for i in range(100)
        )
        for index in range(24):
            await service.create_chatbook(
                **console_chatbook_artifact_payload(
                    title=f"Notebook {index:02d}",
                    message_text=long_body,
                    message_role="Assistant",
                )
            )
        db = screen.app_instance.subscriptions_db
        existing = screen._artifacts_controller.selected.native_id
        watch = db.get_briefing(existing)["watchlist_id"]
        for _ in range(24):
            report = db.insert_briefing(watch)
            db.update_briefing(report, status="complete", body_markdown=long_body)
        c = await show_chatbooks(screen, pilot)
        states = {}
        for view, query, index, list_y, body_y in (
            ("chatbooks", "Notebook", 5, 9, 25),
            ("reports", "Weekly", 7, 12, 40),
        ):
            if c.scope.view != view:
                await screen._select_library_rail_row(f"artifacts-{view}")
            screen.query_one("#library-artifacts-search", Input).value = query
            await settled(
                pilot,
                lambda view=view, query=query: (
                    c.page
                    and c.page.scope.view == view
                    and c.page.scope.query == query
                    and not c.loading
                ),
            )
            rows = screen.query_one("#library-artifacts-list", OptionList)
            rows.highlighted = index
            target = c.page.items[index].key
            await settled(
                pilot,
                lambda target=target: (
                    c.detail and c.detail.key == target and not c.detail_loading
                ),
            )
            c.focus_reader()
            await pilot.pause()
            rows.scroll_to(y=list_y, animate=False)
            body = screen.query_one("#library-artifacts-body")
            body.scroll_to(y=body_y, animate=False)
            await pilot.pause()
            assert rows.scroll_y > 0 and body.scroll_y > 0
            states[view] = (c.selected, rows.scroll_y, body.scroll_y)
        for view, query in (("chatbooks", "Notebook"), ("reports", "Weekly")):
            expected_key, expected_items, expected_body = states[view]
            await screen._select_library_rail_row(f"artifacts-{view}")
            await settled(
                pilot,
                lambda expected_key=expected_key: (
                    c.detail and c.detail.key == expected_key and not c.loading
                ),
            )
            # Markdown blocks mount asynchronously; allow restoration to settle,
            # then assert actual offsets rather than the saved controller tuple.
            for _ in range(80):
                if (
                    abs(
                        screen.query_one("#library-artifacts-body").scroll_y
                        - expected_body
                    )
                    <= 0.5
                ):
                    break
                await pilot.pause(0.025)
            assert c.scope.query == query
            assert screen.query_one("#library-artifacts-search", Input).value == query
            assert c.selected == expected_key
            assert screen.query_one(
                "#library-artifacts-list"
            ).scroll_y == pytest.approx(expected_items, abs=0.5)
            assert screen.query_one(
                "#library-artifacts-body"
            ).scroll_y == pytest.approx(expected_body, abs=0.5)


@pytest.mark.asyncio
@private_profile_test
async def test_full_library_keeps_active_share_manage_and_stop_across_panes_and_views(
    request,
    tmp_path,
    staged_controller,
):
    from tldw_chatbook.UI.Screens.artifact_share_dialog import ArtifactShareDialog

    service, records = await exported_registry(tmp_path)
    async with artifact_library(tmp_path, size=(160, 50)) as (screen, pilot):
        screen.app_instance.local_chatbook_service = service
        screen.app_instance.artifact_share_controller = staged_controller
        c = await show_chatbooks(
            screen, pilot, ArtifactKey("chatbook", records[0]["chatbook_id"])
        )
        screen.query_one("#library-artifacts-share", Button).press()
        await settled(
            pilot,
            lambda: (
                isinstance(screen.app.screen, ArtifactShareDialog)
                and screen.app.screen.is_mounted
            ),
        )
        dialog = screen.app.screen
        dialog.query_one("#share-artifact-list", SelectionList).select_all()
        dialog.query_one("#share-start", Button).press()
        await settled(pilot, lambda: staged_controller.status is not None)
        status = staged_controller.status
        assert status.artifact_count == 2
        assert (status.share_dir / "manifest.json").is_file()
        await settled(
            pilot, lambda: screen.query_one("#library-artifacts-share-strip").display
        )
        for pane in ("items", "library"):
            c.toggle_pane(pane)
            await pilot.pause()
            strip = screen.query_one("#library-artifacts-share-strip")
            stop = screen.query_one("#library-artifacts-share-stop", Button)
            assert strip.display and strip.region.height > 0
            assert stop.region.width > 0 and not stop.disabled
            assert staged_controller.status.share_dir == status.share_dir
        # Restore the navigation pane before following the actual Library rows.
        c.toggle_pane("library")
        c.toggle_pane("items")
        await pilot.pause()
        for row in (
            "artifacts-reports",
            "browse-notes",
            "artifacts-chatbooks",
            "browse-notes",
        ):
            await screen._select_library_rail_row(row)
            await settled(
                pilot,
                lambda: (
                    screen.query_one("#library-artifacts-share-strip").display
                    and screen.query_one(
                        "#library-artifacts-share-stop", Button
                    ).region.width
                    > 0
                ),
            )
            assert "Sharing 2 Chatbooks" in visible_text(
                screen.query_one("#library-artifacts-share-status", Static)
            )
            assert screen.app_instance.artifact_share_controller is staged_controller
            assert staged_controller.status.share_dir == status.share_dir
            assert (
                screen.query_one("#library-artifacts-share-stop", Button).region.width
                > 0
            )
        screen.query_one("#library-artifacts-share-manage", Button).press()
        await settled(
            pilot,
            lambda: (
                isinstance(screen.app.screen, ArtifactShareDialog)
                and screen.app.screen.is_mounted
            ),
        )
        assert "Sharing 2 Chatbooks" in visible_text(
            screen.app.screen.query_one("#share-active-note", Static)
        )
        await pilot.press("escape")
        await pilot.pause()
        assert staged_controller.status.share_dir == status.share_dir
        screen.query_one("#library-artifacts-share-stop", Button).press()
        await settled(pilot, lambda: staged_controller.status is None)
        await settled(
            pilot,
            lambda: not screen.query_one("#library-artifacts-share-strip").display,
        )
        assert not status.share_dir.exists()
