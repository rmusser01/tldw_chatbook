"""Draft Shelf behavior inside the existing Console Prompt Workbench."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from textual.app import App
from textual.widgets import Button, Input, Select, Static, TextArea

from tldw_chatbook.Widgets.Console.console_prompt_draft_editor import (
    ConsolePromptDraftEditor,
)
from tldw_chatbook.Widgets.Console.console_prompts_browse import ConsolePromptsBrowse
from tldw_chatbook.Widgets.Console.console_prompts_modal import ConsolePromptsModal
from tldw_chatbook.Widgets.Console.console_prompts_state import ConsolePromptsState


def _draft(
    draft_id: int = 7, *, content: str = "First line\nfull body", version: int = 1
):
    return {
        "id": f"draft_shelf:{draft_id}",
        "draft_id": draft_id,
        "content": content,
        "display_name": content.splitlines()[0],
        "preview": " ".join(content.split()),
        "created_at": "2026-09-25T10:00:00.000Z",
        "updated_at": "2026-09-25T11:00:00.000Z",
        "version": version,
        "backend": "draft_shelf",
        "artifact_type": "draft",
    }


class _Backend:
    def __init__(self) -> None:
        self.drafts = {7: _draft()}
        self.update_calls = []
        self.delete_calls = []
        self.insert_calls = []
        self.save_calls = []
        self.membership_calls = []

    async def capabilities(self, source):
        return SimpleNamespace(
            structured_kinds=frozenset(),
            artifact_types=frozenset({"prompt", "recipe"}),
            conditional_update=True,
        )

    async def list_page(self, source, page):
        if source == "draft_shelf":
            items = tuple(self.drafts.values())
            return {
                "items": items,
                "page": page,
                "total_pages": 1,
                "total_items": len(items),
            }
        return {"items": [], "page": page, "total_pages": 1, "total_items": 0}

    async def search(self, source, query):
        if source != "draft_shelf":
            return []
        return [
            item
            for item in self.drafts.values()
            if query.casefold() in item["content"].casefold()
        ]

    async def detail(self, source, identifier):
        assert source == "draft_shelf"
        return self.drafts[int(str(identifier).rsplit(":", 1)[-1])]

    async def save(self, **payload):
        self.save_calls.append(payload)
        return {"id": "local:prompt:41", "local_id": 41, "source_id": "41"}

    async def update_draft(self, *, draft_id, content, expected_version):
        self.update_calls.append((draft_id, content, expected_version))
        updated = _draft(draft_id, content=content, version=expected_version + 1)
        self.drafts[draft_id] = updated
        return updated

    async def delete_draft(self, *, draft_id, expected_version):
        self.delete_calls.append((draft_id, expected_version))
        del self.drafts[draft_id]
        return True

    async def insert_draft(self, content):
        self.insert_calls.append(content)
        return True

    async def list_collections(self):
        return {
            "collections": [
                {"collection_id": 5, "display_name": "Writing"},
            ]
        }

    async def assign_collection(self, *, prompt_id, collection_ids):
        self.membership_calls.append((prompt_id, tuple(collection_ids)))
        return {"prompt_id": prompt_id, "collection_ids": tuple(collection_ids)}


def _modal(backend: _Backend) -> ConsolePromptsModal:
    return ConsolePromptsModal(
        capabilities=backend.capabilities,
        list_page=backend.list_page,
        search=backend.search,
        detail=backend.detail,
        save=backend.save,
        update_draft=backend.update_draft,
        delete_draft=backend.delete_draft,
        insert_draft=backend.insert_draft,
        list_draft_collections=backend.list_collections,
        assign_draft_collection=backend.assign_collection,
        initial_source="draft_shelf",
    )


class _ProductionCssApp(App):
    _CSS_ROOT = Path(__file__).parents[2] / "tldw_chatbook" / "css"
    CSS_PATH: ClassVar[list[str]] = [
        str(_CSS_ROOT / "screen_css_scoped.tcss"),
        str(_CSS_ROOT / "tldw_cli_modular.tcss"),
        str(_CSS_ROOT / "screen_css_self.tcss"),
    ]


def test_state_accepts_draft_shelf_and_rejects_stale_source_results():
    state = ConsolePromptsState.initial().with_source("draft_shelf").begin_search()
    token = state.search_token

    assert state.source == "draft_shelf"
    assert state.accepts(token, "draft_shelf")

    switched = state.with_source("local")
    assert not switched.accepts(token, "draft_shelf")


@pytest.mark.asyncio
async def test_draft_shelf_search_keeps_focus_and_arrow_enter_opens_highlighted_row():
    backend = _Backend()
    modal = _modal(backend)
    app = _ProductionCssApp()

    async with app.run_test(size=(100, 32)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        browse = modal.query_one(ConsolePromptsBrowse)
        search = browse.query_one("#console-prompts-search", Input)
        assert app.focused is search
        assert "Draft Shelf" in str(
            browse.query_one("#console-prompts-browse-status", Static).render()
        )

        await pilot.press("down")
        assert app.focused is search
        row = browse.query_one(".console-prompts-result", Button)
        assert row.has_class("highlighted")

        await pilot.press("enter")
        await pilot.pause()

        assert modal.state.mode == "draft_edit"
        assert modal.query_one(ConsolePromptDraftEditor)


@pytest.mark.asyncio
async def test_stale_draft_detail_cannot_open_after_source_switch_during_collections():
    backend = _Backend()
    collections_started = asyncio.Event()
    release_collections = asyncio.Event()

    async def delayed_collections():
        collections_started.set()
        await release_collections.wait()
        return {"collections": []}

    backend.list_collections = delayed_collections
    modal = _modal(backend)
    app = _ProductionCssApp()

    async with app.run_test(size=(100, 32)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        detail_task = asyncio.create_task(modal.open_artifact("draft_shelf:7"))
        await collections_started.wait()

        await modal.switch_source("local")
        release_collections.set()
        await detail_task
        await pilot.pause()

        assert modal.state.source == "local"
        assert modal.state.mode == "browse"
        assert not modal.query(ConsolePromptDraftEditor)


@pytest.mark.asyncio
async def test_out_of_range_page_reloads_last_real_page_instead_of_false_empty_state():
    backend = _Backend()
    requested_pages = []

    async def paged_list(source, page):
        assert source == "draft_shelf"
        requested_pages.append(page)
        return {
            "items": () if page == 10 else (_draft(),),
            "page": page,
            "total_pages": 9,
            "total_items": 90,
        }

    backend.list_page = paged_list
    modal = _modal(backend)
    app = _ProductionCssApp()

    async with app.run_test(size=(100, 32)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        requested_pages.clear()
        modal.state = modal.state.with_page(10)

        await modal.reload_browse()
        await pilot.pause()

        assert requested_pages == [10, 9]
        assert modal.state.page == 9
        assert modal.browse_result.page == 9
        page_copy = str(modal.query_one("#console-prompts-page", Static).render())
        status = str(
            modal.query_one("#console-prompts-browse-status", Static).render()
        )
        assert page_copy == "Page 9 of 9"
        assert "empty" not in status


@pytest.mark.asyncio
async def test_bounded_draft_search_reports_visible_and_total_match_counts():
    backend = _Backend()

    async def bounded_search(source, query):
        assert source == "draft_shelf"
        assert query == "needle"
        return {
            "items": (_draft(content="needle result"),),
            "total_items": 100,
        }

    backend.search = bounded_search
    modal = _modal(backend)
    app = _ProductionCssApp()

    async with app.run_test(size=(100, 32)) as pilot:
        app.push_screen(modal)
        await pilot.pause()

        await modal.set_query("needle")
        await pilot.pause()

        status = str(
            modal.query_one("#console-prompts-browse-status", Static).render()
        )
        assert modal.browse_result.total_items == 100
        assert "Showing 1 of 100 matches" in status
        assert "saved drafts" not in status


@pytest.mark.asyncio
async def test_production_styles_keep_source_selector_visible_beside_search():
    backend = _Backend()
    modal = _modal(backend)
    app = _ProductionCssApp()

    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        source = modal.query_one("#console-prompts-source", Select)
        search = modal.query_one("#console-prompts-search", Input)
        assert source.region.width >= 22
        assert source.region.height >= 3
        assert search.region.x > source.region.x

        source.focus()
        await modal.switch_source("local")
        await pilot.pause()
        assert app.focused is search


@pytest.mark.asyncio
async def test_narrow_editor_scrolls_to_reveal_promotion_action():
    backend = _Backend()
    modal = _modal(backend)
    app = _ProductionCssApp()

    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        await modal.open_artifact("draft_shelf:7")
        await pilot.pause()

        editor = modal.query_one(ConsolePromptDraftEditor)
        promote = modal.query_one("#console-prompt-draft-promote", Button)
        assert editor.max_scroll_y > 0

        promote.focus()
        promote.scroll_visible(animate=False, immediate=True)
        await pilot.pause()

        assert editor.scroll_y > 0
        assert editor.scrollable_content_region.contains_region(promote.region)


@pytest.mark.asyncio
async def test_full_shelf_names_the_block_and_required_recovery():
    backend = _Backend()

    async def full_page(source, page):
        return {
            "items": (_draft(),),
            "page": page,
            "total_pages": 10,
            "total_items": 100,
        }

    backend.list_page = full_page
    modal = _modal(backend)
    app = _ProductionCssApp()

    async with app.run_test(size=(100, 32)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        status = str(modal.query_one("#console-prompts-browse-status", Static).render())
        assert "full" in status
        assert "100 of 100" in status
        assert "Delete an entry" in status


@pytest.mark.asyncio
async def test_draft_editor_updates_inserts_at_caret_and_requires_two_delete_presses():
    backend = _Backend()
    modal = _modal(backend)
    app = _ProductionCssApp()

    async with app.run_test(size=(100, 32)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        await modal.open_artifact("draft_shelf:7")

        editor = modal.query_one(ConsolePromptDraftEditor)
        content = editor.query_one("#console-prompt-draft-content", TextArea)
        content.text = "edited body"
        await pilot.pause()
        await pilot.click("#console-prompt-draft-update")
        await pilot.pause()

        assert backend.update_calls == [(7, "edited body", 1)]
        assert modal.state.dirty is False

        await pilot.click("#console-prompt-draft-insert")
        await pilot.pause()
        assert backend.insert_calls == ["edited body"]

    backend = _Backend()
    modal = _modal(backend)
    app = _ProductionCssApp()
    async with app.run_test(size=(100, 32)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        await modal.open_artifact("draft_shelf:7")
        await pilot.pause()
        delete = modal.query_one("#console-prompt-draft-delete", Button)

        await pilot.click("#console-prompt-draft-delete")
        assert backend.delete_calls == []
        assert delete.label.plain == "Press again to delete"
        await pilot.pause(0.3)
        assert modal.query_one(ConsolePromptDraftEditor).delete_armed is True

        clicked = await pilot.click("#console-prompt-draft-delete")
        assert clicked is True
        await pilot.pause()
        assert backend.delete_calls == [(7, 1)], str(
            modal.query_one("#console-prompt-draft-status", Static).render()
        )
        assert modal.state.mode == "browse"


@pytest.mark.asyncio
async def test_delete_confirmation_resets_after_field_edit_or_focus_navigation():
    backend = _Backend()
    modal = _modal(backend)
    app = _ProductionCssApp()

    async with app.run_test(size=(100, 32)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        await modal.open_artifact("draft_shelf:7")
        await pilot.pause()
        editor = modal.query_one(ConsolePromptDraftEditor)

        clicked = await pilot.click("#console-prompt-draft-delete")
        assert clicked is True
        await pilot.pause()
        assert editor.delete_armed is True, (
            backend.delete_calls,
            modal.state.mode,
            modal.query_one("#console-prompt-draft-delete", Button).label.plain,
        )
        name = modal.query_one("#console-prompt-draft-library-name", Input)
        name.value = "Reusable"
        await pilot.pause()
        assert editor.delete_armed is False
        await pilot.pause(0.3)

        clicked = await pilot.click("#console-prompt-draft-delete")
        assert clicked is True
        await pilot.pause()
        assert editor.delete_armed is True, (
            backend.delete_calls,
            modal.state.mode,
            modal.query_one("#console-prompt-draft-delete", Button).label.plain,
        )
        collection = modal.query_one(
            "#console-prompt-draft-library-collection", Select
        )
        collection.value = 5
        await pilot.pause()
        assert editor.delete_armed is False
        await pilot.pause(0.3)

        await pilot.click("#console-prompt-draft-delete")
        await pilot.pause()
        assert editor.delete_armed is True
        modal.query_one("#console-prompt-draft-update", Button).focus()
        await pilot.pause()
        assert editor.delete_armed is False
        assert backend.delete_calls == []


@pytest.mark.asyncio
async def test_escape_returns_clean_draft_editor_to_shelf_then_closes_root():
    backend = _Backend()
    modal = _modal(backend)
    app = _ProductionCssApp()
    results = []

    async with app.run_test(size=(100, 32)) as pilot:
        app.push_screen(modal, callback=results.append)
        await pilot.pause()
        await modal.open_artifact("draft_shelf:7")
        await pilot.pause()
        assert modal.state.mode == "draft_edit"

        await pilot.press("escape")
        await pilot.pause()
        assert modal.state.mode == "browse"
        assert results == []

        await pilot.press("escape")
        await pilot.pause()
        assert results == [None]


@pytest.mark.asyncio
async def test_updated_draft_replaces_the_cached_shelf_row_before_back():
    backend = _Backend()
    modal = _modal(backend)
    app = _ProductionCssApp()

    async with app.run_test(size=(100, 32)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        await modal.open_artifact("draft_shelf:7")
        await pilot.pause()
        modal.query_one("#console-prompt-draft-content", TextArea).text = (
            "New title\nupdated body"
        )
        await pilot.click("#console-prompt-draft-update")
        await pilot.pause()

        await modal._back_internal()
        await pilot.pause()

        row = modal.query_one(".console-prompts-result", Button)
        assert "New title" in row.label.plain
        assert "First line" not in row.label.plain


@pytest.mark.asyncio
async def test_promotion_assigns_optional_collection_and_keeps_shelf_entry():
    backend = _Backend()
    modal = _modal(backend)
    app = _ProductionCssApp()

    async with app.run_test(size=(100, 32)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        await modal.open_artifact("draft_shelf:7")
        await pilot.pause()

        modal.query_one("#console-prompt-draft-library-name", Input).value = "Reusable"
        modal.query_one("#console-prompt-draft-library-collection", Select).value = 5
        await pilot.click("#console-prompt-draft-promote")
        await pilot.pause()

        assert backend.save_calls == [
            {
                "source": "local",
                "name": "Reusable",
                "system_prompt": "",
                "user_prompt": "First line\nfull body",
                "artifact_type": "prompt",
                "prompt_format": "legacy",
            }
        ]
        assert backend.membership_calls == [(41, (5,))]
        assert 7 in backend.drafts
        status = modal.query_one("#console-prompt-draft-status", Static)
        assert "saved to Library" in str(status.render())
