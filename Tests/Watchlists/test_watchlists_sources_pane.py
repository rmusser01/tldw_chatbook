"""Tests for the Watchlists sources pane."""

import pytest
from rich.style import Style
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, Select, Switch, TextArea

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import LocalWatchlistsService
from tldw_chatbook.Subscriptions.noise_defaults import default_ignore_selectors_text
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
    CheckNowRequested,
    PreviewRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import (
    CreateSourceRequested,
    ExportOpmlRequested,
    ImportOpmlRequested,
    SourceSelected,
    SourcesPane,
)


class SourcesPaneHarness(App):
    def __init__(self):
        super().__init__()
        self.captured_messages = []

    def compose(self) -> ComposeResult:
        yield SourcesPane()

    def on_source_selected(self, message: SourceSelected) -> None:
        self.captured_messages.append(("source_selected", message.source))

    def on_create_source_requested(self, message: CreateSourceRequested) -> None:
        self.captured_messages.append(("create_source_requested", message.payload))

    def on_preview_requested(self, message: PreviewRequested) -> None:
        self.captured_messages.append(("preview_requested", message.entity))

    def on_check_now_requested(self, message: CheckNowRequested) -> None:
        self.captured_messages.append(("check_now_requested", message.entity))

    def on_import_opml_requested(self, message: ImportOpmlRequested) -> None:
        self.captured_messages.append(("import_opml_requested", None))

    def on_export_opml_requested(self, message: ExportOpmlRequested) -> None:
        self.captured_messages.append(("export_opml_requested", None))


class PersistingSourcesPaneHarness(SourcesPaneHarness):
    """`SourcesPaneHarness`, but the create request reaches a real database.

    TASK-1362. The noise field's contract is about what gets *stored* -- the
    selector text a source is created with, and the empty text a user who
    cleared the field is entitled to. A captured payload cannot show that: it
    would still look correct if the field never reached
    `_subscription_config_fields`, or if the column were dropped. So this
    harness forwards `CreateSourceRequested` to the real
    `LocalWatchlistsService` over a real `SubscriptionsDB`, exactly as
    `WatchlistsCollectionsScreen.handle_create_source_requested` does, and the
    assertions read the row back.
    """

    def __init__(self, service: LocalWatchlistsService) -> None:
        super().__init__()
        self._service = service
        self.created_sources: list[dict] = []
        self.create_error: BaseException | None = None

    def on_create_source_requested(self, message: CreateSourceRequested) -> None:
        super().on_create_source_requested(message)
        self.run_worker(self._create(message.payload), exclusive=True)

    async def _create(self, payload: dict) -> None:
        try:
            self.created_sources.append(await self._service.create_source(payload))
        except BaseException as exc:  # surfaced by the tests, never swallowed
            self.create_error = exc


async def _create_through_the_form(pilot, app, **field_values) -> dict:
    """Fill the create form and press `Create`, then return the stored row.

    Uses the same direct-assignment style as the rest of this module (an
    `Input.value` write posts `Input.Changed` exactly as a keystroke does) and
    the pane's own submit button, so the payload is built by
    `_submit_create_form` rather than by the test.

    Args:
        pilot: The running app's pilot.
        app: A `PersistingSourcesPaneHarness`.
        field_values: `id suffix -> value` for any extra field to set.

    Returns:
        The `subscriptions` row as stored.
    """
    pane = app.query_one(SourcesPane)
    pane.query_one("#sources-new-button", Button).press()
    await pilot.pause()

    pane.query_one("#sources-create-name", Input).value = field_values.pop(
        "name", "Noisy Page"
    )
    pane.query_one("#sources-create-url", Input).value = field_values.pop(
        "url", "https://example.com/page"
    )
    assert not field_values, f"unhandled field values: {field_values}"

    pane.query_one("#sources-create-submit", Button).press()
    for _ in range(50):
        await pilot.pause()
        if app.created_sources or app.create_error is not None:
            break
    if app.create_error is not None:
        raise app.create_error
    assert app.created_sources, "the create request never reached the service"
    stored = app._service._db().get_subscription(
        int(app.created_sources[0]["source_id"])
    )
    assert stored is not None, "the source was not persisted"
    return stored


@pytest.mark.asyncio
async def test_create_form_stores_the_default_noise_selectors(tmp_path):
    """TASK-1362, spec §2: the prefill has to land in the database.

    The suppression defaults exist so that a first source does not report a
    change every check because its ad slot or view counter rewrote itself. If
    the field is decorative -- prefilled in the form but dropped on the way to
    `create_source` -- the user sees the rules, believes them applied, and gets
    the noise anyway.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    app = PersistingSourcesPaneHarness(LocalWatchlistsService(db_factory=lambda: db))
    async with app.run_test(size=(120, 40)) as pilot:
        stored = await _create_through_the_form(pilot, app)

        assert stored["ignore_selectors"] == default_ignore_selectors_text(), (
            "a source created without touching the noise field must be stored "
            "with the shipped default selectors; the row holds "
            f"{stored['ignore_selectors']!r}"
        )


@pytest.mark.asyncio
async def test_clearing_the_noise_field_stores_no_selectors(tmp_path):
    """TASK-1362, spec §2: deliberate emptiness is honoured.

    Emptying the field is a real instruction -- "report every change on this
    page, including the furniture" -- and it is the only way to watch a page
    whose payload happens to live in an element the defaults strip. Re-applying
    the default at save time would overrule the user silently, and they would
    have no way to tell: the form they submitted said empty.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    app = PersistingSourcesPaneHarness(LocalWatchlistsService(db_factory=lambda: db))
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.query_one("#sources-new-button", Button).press()
        await pilot.pause()
        # The real clearing edit, not a `.text` assignment: this is what
        # select-all-and-delete performs, and it posts `TextArea.Changed`.
        pane.query_one("#sources-create-ignore-selectors", TextArea).clear()
        await pilot.pause()

        stored = await _create_through_the_form(pilot, app)

        assert not (stored["ignore_selectors"] or ""), (
            "a source created with the noise field cleared must be stored with "
            f"no selectors; the row holds {stored['ignore_selectors']!r}"
        )
        assert stored["ignore_selectors"] != default_ignore_selectors_text(), (
            "the cleared field was re-filled with the shipped default behind "
            "the user's back"
        )


@pytest.mark.asyncio
async def test_noise_field_is_visible_prefilled_and_labelled():
    """The control itself: on screen, filled in, and named (TASK-1362).

    Spec §2 puts the prefill in the *form* rather than applying it invisibly at
    save time, so what the field shows and what it is called are part of the
    contract, not decoration.
    """
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.query_one("#sources-new-button", Button).press()
        await pilot.pause()

        field = pane.query_one("#sources-create-ignore-selectors", TextArea)
        assert field.display and field.region.height > 0, (
            "the noise field is not on screen"
        )
        assert field.text == default_ignore_selectors_text()
        assert field.border_title == (
            "Ignore elements (CSS selectors — one rule per line; commas group)"
        )
        # The spam -> add-a-selector loop has to be stated where the field is.
        assert "silence" in str(field.border_subtitle)


@pytest.mark.asyncio
async def test_noise_field_is_seeded_from_the_draft_not_the_default():
    """A rebuild must not restore rules the user deleted (TASK-1362).

    `SourcesPane` is reconstructed whenever the workbench recomposes -- any
    region collapse does it -- which is why the name/url/tags drafts are
    mirrored to the screen and seeded back. The noise field needs the same
    treatment for a stronger reason: its untouched state is not empty, so a
    pane rebuilt without the draft would re-prefill a field the user had
    emptied on purpose.
    """
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.create_draft_ignore_selectors = ""
        pane.query_one("#sources-new-button", Button).press()
        await pilot.pause()

        field = pane.query_one("#sources-create-ignore-selectors", TextArea)
        assert field.text == "", (
            f"the rebuilt form re-prefilled a cleared field with {field.text!r}"
        )


@pytest.fixture
def sample_sources():
    return [
        {
            "id": "source-1",
            "name": "AI News RSS",
            "source_type": "rss",
            "status": "ok",
            "last_scraped": "2026-07-18",
            "active": True,
        },
        {
            "id": "source-2",
            "name": "Tech Atom Feed",
            "source_type": "atom",
            "status": "error",
            "last_scraped": "2026-07-17",
            "active": False,
        },
        {
            "id": "source-3",
            "name": "Playlist Watch",
            "source_type": "playlist",
            "status": "ok",
            "last_scraped": "-",
            "active": True,
        },
    ]


@pytest.mark.asyncio
async def test_sources_pane_renders_table_and_toolbar():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        assert pane.query_one("#sources-search-input", Input)
        assert pane.query_one("#sources-type-select", Select)
        assert pane.query_one("#sources-new-button", Button)
        assert pane.query_one("#sources-table", DataTable)


@pytest.mark.asyncio
async def test_sources_pane_populates_table(sample_sources):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert table.row_count == 3


@pytest.mark.asyncio
async def test_sources_pane_filters_by_search(sample_sources):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        pane.search_query = "AI"
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert table.row_count == 1
        assert "AI News RSS" in str(table.get_row_at(0)[0])


@pytest.mark.asyncio
async def test_sources_pane_filters_by_type(sample_sources):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        pane.source_type_filter = "atom"
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert table.row_count == 1
        assert "Tech Atom Feed" in str(table.get_row_at(0)[0])


@pytest.mark.asyncio
async def test_sources_pane_selects_source_and_posts_message(sample_sources):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert "source-1" in [str(key.value) for key in table.rows]

        pane.select_source_by_id("source-1")
        await pilot.pause()

        assert pane.selected_source == sample_sources[0]
        assert app.captured_messages == [("source_selected", sample_sources[0])]


@pytest.mark.asyncio
async def test_sources_pane_new_source_form_posts_request():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.query_one("#sources-new-button", Button).press()
        await pilot.pause()

        assert pane.query_one("#sources-create-form")
        pane.query_one("#sources-create-name", Input).value = "New Feed"
        pane.query_one("#sources-create-url", Input).value = "http://example.com/feed"
        pane.query_one("#sources-create-type", Select).value = "rss"
        pane.query_one("#sources-create-active", Switch).value = True
        pane.query_one("#sources-create-tags", Input).value = "ai, news"
        pane.query_one("#sources-create-submit", Button).press()
        await pilot.pause()

        assert not pane.query("#sources-create-form")
        assert len(app.captured_messages) == 1
        kind, payload = app.captured_messages[0]
        assert kind == "create_source_requested"
        assert payload["name"] == "New Feed"
        assert payload["url"] == "http://example.com/feed"
        assert payload["source_type"] == "rss"
        assert payload["active"] is True
        assert payload["tags"] == ["ai", "news"]
        # Untouched, the cadence control reproduces the Subscriptions_DB
        # column default rather than leaving the source unscheduled.
        assert payload["check_frequency"] == 3600


@pytest.mark.asyncio
async def test_sources_pane_new_source_form_carries_selected_check_frequency():
    """TASK-1210: the cadence the user picks has to reach the payload.

    Without it every source lands on the database default and there is no way
    to say "check this hourly" or "check this daily" from the screen at all.
    """
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.query_one("#sources-new-button", Button).press()
        await pilot.pause()

        pane.query_one("#sources-create-name", Input).value = "Daily Feed"
        pane.query_one("#sources-create-url", Input).value = "http://example.com/d"
        pane.query_one("#sources-create-frequency", Select).value = 86_400
        pane.query_one("#sources-create-submit", Button).press()
        await pilot.pause()

        _kind, payload = app.captured_messages[0]
        assert payload["check_frequency"] == 86_400


@pytest.mark.asyncio
async def test_sources_pane_action_buttons_exist():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        assert pane.query_one("#sources-preview-button", Button)
        assert pane.query_one("#sources-check-now-button", Button)
        assert pane.query_one("#sources-import-opml-button", Button)
        assert pane.query_one("#sources-export-opml-button", Button)


@pytest.mark.asyncio
async def test_sources_pane_preview_and_check_now_disabled_without_selection():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        preview = pane.query_one("#sources-preview-button", Button)
        check_now = pane.query_one("#sources-check-now-button", Button)
        assert preview.disabled
        assert check_now.disabled


@pytest.mark.asyncio
async def test_sources_pane_preview_and_check_now_enabled_with_selection(sample_sources):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        pane.select_source_by_id("source-1")
        await pilot.pause()

        preview = pane.query_one("#sources-preview-button", Button)
        check_now = pane.query_one("#sources-check-now-button", Button)
        assert not preview.disabled
        assert not check_now.disabled


@pytest.mark.asyncio
async def test_sources_pane_posts_preview_and_check_now_messages(sample_sources):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        pane.select_source_by_id("source-1")
        await pilot.pause()

        pane.query_one("#sources-preview-button", Button).press()
        pane.query_one("#sources-check-now-button", Button).press()
        await pilot.pause()

        assert app.captured_messages == [
            ("source_selected", sample_sources[0]),
            ("preview_requested", sample_sources[0]),
            ("check_now_requested", sample_sources[0]),
        ]


@pytest.mark.asyncio
async def test_sources_pane_posts_opml_messages():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.query_one("#sources-import-opml-button", Button).press()
        pane.query_one("#sources-export-opml-button", Button).press()
        await pilot.pause()

        assert app.captured_messages == [
            ("import_opml_requested", None),
            ("export_opml_requested", None),
        ]


@pytest.mark.asyncio
async def test_sources_pane_filter_editor_toggles():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        assert not pane.query("#sources-filter-editor")
        pane.query_one("#sources-filter-toggle", Button).press()
        await pilot.pause()
        assert pane.query_one("#sources-filter-editor")
        pane.query_one("#sources-filter-toggle", Button).press()
        await pilot.pause()
        assert not pane.query("#sources-filter-editor")


@pytest.mark.asyncio
async def test_sources_pane_filters_by_status():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = [
            {"id": "s1", "name": "A", "source_type": "rss", "status": "ok", "active": True},
            {"id": "s2", "name": "B", "source_type": "rss", "status": "error", "active": True},
        ]
        pane.status_filter = "error"
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert table.row_count == 1
        assert "B" in str(table.get_row_at(0)[0])


@pytest.mark.asyncio
async def test_sources_pane_filters_by_active_state():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = [
            {"id": "s1", "name": "A", "source_type": "rss", "status": "ok", "active": True},
            {"id": "s2", "name": "B", "source_type": "rss", "status": "ok", "active": False},
        ]
        pane.active_filter = "active"
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert table.row_count == 1
        assert "A" in str(table.get_row_at(0)[0])


@pytest.mark.asyncio
async def test_sources_pane_filters_by_tags():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = [
            {"id": "s1", "name": "A", "source_type": "rss", "status": "ok", "active": True, "tags": ["ai"]},
            {"id": "s2", "name": "B", "source_type": "rss", "status": "ok", "active": True, "tags": ["tech"]},
        ]
        pane.tags_filter = "tech"
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert table.row_count == 1
        assert "B" in str(table.get_row_at(0)[0])


# --- task-876: selected row is distinguishable from a merely-focused one ---
#
# `DataTable`'s own cursor is a keyboard-focus affordance that always sits
# somewhere -- including on a row this pane does not consider selected. The
# actual selection (`selected_source`) is marked with Rich's own
# terminal-agnostic "reverse bold" idiom directly on the cell `Text`, the
# same approach `snippet_editor.py`/`library_media_viewer.py` already use for
# a DataTable/Static cell that cannot reference Textual CSS variables.


def _cell_style(table: DataTable, row_key: str, column_index: int) -> Style:
    """The Rich `Style` a cell's `Text` carries.

    `Text.style` stores whatever was passed to its constructor verbatim --
    a plain string here, not a parsed `Style` -- so this parses it the same
    way Rich itself would at render time.
    """
    column_key = list(table.columns.keys())[column_index]
    raw_style = table.get_cell(row_key, column_key).style
    return Style.parse(raw_style) if isinstance(raw_style, str) else raw_style


@pytest.mark.asyncio
async def test_selected_source_row_is_styled_distinctly_from_others(sample_sources):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        pane.select_source_by_id("source-1")
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        selected_style = _cell_style(table, "source-1", 0)
        other_style = _cell_style(table, "source-2", 0)
        assert selected_style.reverse, "the selected row must carry the highlight style"
        assert not other_style.reverse, "an unselected row must not"


@pytest.mark.asyncio
async def test_selection_highlight_moves_without_rebuilding_the_table(sample_sources):
    """Selecting a different row moves the highlight via a targeted
    `update_cell`, not a table rebuild -- `selected_source` is deliberately
    NOT `recompose=True` (a selection must not discard the DataTable's own
    scroll position/cursor).
    """
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        pane.select_source_by_id("source-1")
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert _cell_style(table, "source-1", 0).reverse

        pane.select_source_by_id("source-2")
        await pilot.pause()

        # Same table instance throughout (no recompose destroyed and
        # rebuilt it), the old row reverted, and the new one highlighted.
        assert pane.query_one("#sources-table", DataTable) is table
        assert not _cell_style(table, "source-1", 0).reverse
        assert _cell_style(table, "source-2", 0).reverse


@pytest.mark.asyncio
async def test_clearing_the_selection_removes_the_highlight(sample_sources):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        pane.select_source_by_id("source-1")
        await pilot.pause()

        pane.selected_source = None
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert not _cell_style(table, "source-1", 0).reverse
