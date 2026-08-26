"""Mounted Add Sources modal vocabulary and stable controls."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select, Static, TabbedContent, TextArea

from tldw_chatbook.Research_Workspace import (
    BoundedPageResult,
    QualifiedWorkspaceRef,
    ResearchCatalogItem,
    WorkspaceDataSource,
)
from tldw_chatbook.UI.Research_Workspace_Modules.add_source_modal import (
    ResearchAddSourceModal,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("authority", "labels"),
    [
        (
            WorkspaceDataSource.LOCAL,
            ("Import Files", "Local Library", "URL", "Paste", "Search Local"),
        ),
        (
            WorkspaceDataSource.SERVER,
            ("Upload", "My Media", "URL", "Paste", "Search Server"),
        ),
    ],
)
async def test_add_source_modal_keeps_exact_authority_tab_order(
    authority: WorkspaceDataSource, labels: tuple[str, ...]
) -> None:
    app = App()
    async with app.run_test(size=(100, 32)) as pilot:
        modal = ResearchAddSourceModal(authority)
        await app.push_screen(modal)
        await pilot.pause()

        tabs = modal.query_one("#research-add-source-tabs", TabbedContent)
        assert tuple(tab.label.plain for tab in tabs.query("Tab")) == labels
        expected_ids = {
            "research-add-upload-path",
            "research-add-upload-browse",
            "research-add-upload-submit",
            "research-add-existing-query",
            "research-add-existing-search",
            "research-add-existing-type",
            "research-add-existing-sort",
            "research-add-existing-prev",
            "research-add-existing-next",
            "research-add-existing-select",
            "research-add-existing-submit",
            "research-add-existing-selection-scope",
            "research-add-url-mode-single",
            "research-add-url-mode-batch",
            "research-add-url-single",
            "research-add-url-batch",
            "research-add-url-submit",
            "research-add-paste-title",
            "research-add-paste-body",
            "research-add-paste-submit",
            "research-add-search-query",
            "research-add-search-search",
            "research-add-search-type",
            "research-add-search-sort",
            "research-add-search-prev",
            "research-add-search-next",
            "research-add-search-select",
            "research-add-search-submit",
            "research-add-error",
            "research-add-close",
        }
        assert {
            widget.id for widget in modal.walk_children() if widget.id
        } >= expected_ids
        assert isinstance(modal.query_one("#research-add-upload-path"), Input)
        assert isinstance(modal.query_one("#research-add-url-batch"), TextArea)
        assert isinstance(modal.query_one("#research-add-paste-body"), TextArea)
        assert not modal.query_one("#research-add-error", Static).display
        assert modal.query_one("#research-add-existing-submit", Button).disabled
        assert modal.query_one("#research-add-search-submit", Button).disabled
        assert "Choose one source file" in str(
            modal.query_one("#research-add-upload-scope", Static).render()
        )
        assert "one existing item" in str(
            modal.query_one("#research-add-existing-selection-scope", Static).render()
        )


@pytest.mark.asyncio
async def test_url_batch_mode_is_real_and_returns_one_item_per_line() -> None:
    app = App()
    results = []
    async with app.run_test(size=(100, 32)) as pilot:
        modal = ResearchAddSourceModal(WorkspaceDataSource.LOCAL)
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()

        modal.query_one("#research-add-url-mode-batch", Button).press()
        await pilot.pause()
        assert not modal.query_one("#research-add-url-single", Input).display
        batch = modal.query_one("#research-add-url-batch", TextArea)
        assert batch.display
        batch.text = "https://example.invalid/a\nhttps://example.invalid/b"
        modal.query_one("#research-add-url-submit", Button).press()
        await pilot.pause()

    assert results[0].kind == "url"
    assert results[0].values == (
        "https://example.invalid/a",
        "https://example.invalid/b",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate",
    [
        "https://user:PRIVATE@example.invalid/paper",
        "file:///private/research.txt",
        "/private/research.txt",
        "https://example.invalid/zero\x00width",
        "https://example.invalid/zero\u200bwidth",
        "https://example.invalid/" + "x" * 2000,
    ],
)
async def test_url_intake_rejects_unsafe_or_unbounded_values_without_dismissal(
    candidate: str,
) -> None:
    """Malformed URLs stay in the modal and never become intake requests."""

    app = App()
    results = []
    async with app.run_test(size=(100, 32)) as pilot:
        modal = ResearchAddSourceModal(WorkspaceDataSource.LOCAL)
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()

        modal.query_one("#research-add-url-single", Input).value = candidate
        modal.query_one("#research-add-url-submit", Button).press()
        await pilot.pause()

        assert app.screen is modal
        error = modal.query_one("#research-add-error", Static)
        assert error.display
        assert "valid HTTP or HTTPS URL" in str(error.render())

    assert results == []


@pytest.mark.asyncio
async def test_url_intake_accepts_supported_unicode_and_percent_encoding() -> None:
    """The security gate must not reject syntax the ingest owner supports."""

    app = App()
    results = []
    candidate = "https://例え.テスト/über?mark=%E2%9C%93"
    async with app.run_test(size=(100, 32)) as pilot:
        modal = ResearchAddSourceModal(WorkspaceDataSource.SERVER)
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()

        modal.query_one("#research-add-url-single", Input).value = candidate
        modal.query_one("#research-add-url-submit", Button).press()
        await pilot.pause()

    assert results[0].values == (candidate,)


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["existing", "search"])
async def test_catalog_tabs_search_select_and_return_exact_catalog_id(
    prefix: str,
) -> None:
    app = App()
    results = []
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-local")

    async def search(**kwargs):
        assert kwargs["limit"] == 25
        return BoundedPageResult(
            items=(ResearchCatalogItem(ref, "7", "Paper", "pdf"),),
            limit=25,
            total=1,
        )

    async with app.run_test(size=(100, 32)) as pilot:
        modal = ResearchAddSourceModal(WorkspaceDataSource.LOCAL, catalog_search=search)
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()

        modal.query_one(f"#research-add-{prefix}-search", Button).press()
        await pilot.pause()
        result_select = modal.query_one(f"#research-add-{prefix}-results", Select)
        result_select.value = "7"
        await pilot.pause()
        submit = modal.query_one(f"#research-add-{prefix}-submit", Button)
        assert not submit.disabled
        submit.press()
        await pilot.pause()

    assert results[0].kind == ("existing" if prefix == "existing" else "catalog")
    assert results[0].values == ("7",)


@pytest.mark.asyncio
async def test_catalog_search_without_owner_callback_reports_gated_reason() -> None:
    app = App()
    async with app.run_test(size=(100, 32)) as pilot:
        modal = ResearchAddSourceModal(WorkspaceDataSource.SERVER)
        await app.push_screen(modal)
        await pilot.pause()

        modal.query_one("#research-add-existing-search", Button).press()
        await pilot.pause()

        error = modal.query_one("#research-add-error", Static)
        assert error.display
        assert "unavailable for this authority" in str(error.render())


@pytest.mark.asyncio
async def test_server_web_search_stays_visible_but_never_reuses_media_catalog() -> None:
    app = App()
    searches = []

    async def media_search(**kwargs):
        searches.append(kwargs)
        return BoundedPageResult(items=(), limit=25, total=0)

    async with app.run_test(size=(100, 32)) as pilot:
        modal = ResearchAddSourceModal(
            WorkspaceDataSource.SERVER, catalog_search=media_search
        )
        await app.push_screen(modal)
        await pilot.pause()

        assert not modal.query_one("#research-add-existing-search", Button).disabled
        assert "one source file" in str(
            modal.query_one("#research-add-upload-path", Input).placeholder
        )
        assert (
            modal.query_one("#research-add-existing-submit", Button).label.plain
            == "Add selected"
        )
        for suffix in (
            "query",
            "type",
            "sort",
            "search",
            "results",
            "prev",
            "next",
            "select",
            "submit",
        ):
            assert modal.query_one(f"#research-add-search-{suffix}").disabled
        assert "Web search is unavailable" in str(
            modal.query_one("#research-add-search-unavailable", Static).render()
        )
        modal._start_catalog_search("search")
        await pilot.pause()

    assert searches == []


@pytest.mark.asyncio
async def test_add_modal_escape_cancels_without_losing_draft_before_callback() -> None:
    app = App()
    results = []
    async with app.run_test(size=(100, 32)) as pilot:
        modal = ResearchAddSourceModal(WorkspaceDataSource.LOCAL)
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()
        modal.query_one("#research-add-paste-title", Input).value = "Unsaved title"
        assert (
            modal.query_one("#research-add-paste-title", Input).value == "Unsaved title"
        )

        await pilot.press("escape")
        await pilot.pause()

    assert results == [None]


class _ModalOpenerApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Button("Open Add Sources", id="modal-opener")


@pytest.mark.asyncio
async def test_add_modal_escape_traps_then_restores_opener_focus() -> None:
    app = _ModalOpenerApp()
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        opener = app.query_one("#modal-opener", Button)
        opener.focus()
        modal = ResearchAddSourceModal(WorkspaceDataSource.LOCAL)
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("shift+tab")
        assert app.focused is not None and app.focused.screen is modal
        await pilot.press("escape")
        await pilot.pause()

        assert app.focused is opener


@pytest.mark.asyncio
async def test_upload_and_paste_tabs_return_validated_requests() -> None:
    app = App()
    results = []
    async with app.run_test(size=(100, 32)) as pilot:
        upload = ResearchAddSourceModal(WorkspaceDataSource.SERVER)
        await app.push_screen(upload, callback=results.append)
        await pilot.pause()
        upload.query_one("#research-add-upload-path", Input).value = "/tmp/paper.pdf"
        upload.query_one("#research-add-upload-submit", Button).press()
        await pilot.pause()

        paste = ResearchAddSourceModal(WorkspaceDataSource.LOCAL)
        await app.push_screen(paste, callback=results.append)
        await pilot.pause()
        paste.query_one("#research-add-paste-title", Input).value = "Field notes"
        paste.query_one("#research-add-paste-body", TextArea).text = "Evidence body"
        paste.query_one("#research-add-paste-submit", Button).press()
        await pilot.pause()

    assert [(item.kind, item.values, item.title) for item in results] == [
        ("file", ("/tmp/paper.pdf",), ""),
        ("paste", ("Evidence body",), "Field notes"),
    ]
