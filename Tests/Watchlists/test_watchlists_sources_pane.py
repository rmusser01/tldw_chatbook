"""Tests for the Watchlists sources pane."""

import pytest
from rich.style import Style
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, Select, Static, Switch, TextArea

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import LocalWatchlistsService
from tldw_chatbook.Subscriptions.noise_defaults import default_ignore_selectors_text
from tldw_chatbook.Widgets.prune_safe_select import PruneSafeSelect
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
    CheckNowRequested,
    PreviewRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.bulk_sources_modal import (
    OpenBulkSourcesRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import (
    CreateWatchlistFromSelectedRequested,
    CreateSourceRequested,
    ExportOpmlRequested,
    ImportOpmlRequested,
    SourceSelected,
    SourceSelectionChanged,
    SourcesPane,
)
from tldw_chatbook.UI.Watchlists_Modules.table_selection import IdSelectionModel


def test_id_selection_model_keeps_filtered_and_sorted_selection_by_id():
    """Catches row-index selection moving to a different source after sorting."""
    selection = IdSelectionModel()
    selection.set_visible_ids(("source-3", "source-1", "source-2"))

    selection.toggle("source-1")
    selection.set_visible_ids(("source-2", "source-1", "source-3"))
    assert selection.selected_ids == frozenset({"source-1"})
    selection.set_visible_ids(("source-2", "source-3"))

    assert selection.selected_ids == frozenset({"source-1"})
    assert selection.status_text == "1 selected · 1 hidden by filters"


def test_id_selection_model_extends_and_contracts_range_in_visible_order():
    """Catches range selection growing by stale row indexes or not contracting."""
    selection = IdSelectionModel()
    selection.set_visible_ids(("source-4", "source-2", "source-9", "source-1"))
    selection.toggle("source-2")

    assert selection.shift("source-2", 1) == "source-9"
    assert selection.selected_ids == frozenset({"source-2", "source-9"})
    assert selection.shift("source-9", 1) == "source-1"
    assert selection.selected_ids == frozenset(
        {"source-2", "source-9", "source-1"}
    )
    assert selection.shift("source-1", -1) == "source-9"
    assert selection.selected_ids == frozenset({"source-2", "source-9"})


def test_id_selection_model_visible_toggle_preserves_hidden_and_clear_removes_all():
    """Catches visible-select accidentally clearing filtered-out selections."""
    selection = IdSelectionModel()
    selection.set_visible_ids(("source-1", "source-2", "source-3"))
    selection.toggle("source-3")
    selection.set_visible_ids(("source-1", "source-2"))

    selection.toggle_visible()
    assert selection.selected_ids == frozenset(
        {"source-1", "source-2", "source-3"}
    )
    selection.toggle_visible()
    assert selection.selected_ids == frozenset({"source-3"})
    selection.clear()
    assert selection.selected_ids == frozenset()


def test_id_selection_model_prunes_only_deleted_source_ids():
    """Catches reload pruning selected sources that still exist but are hidden."""
    selection = IdSelectionModel()
    selection.set_visible_ids(("source-1", "source-2", "source-3"))
    selection.toggle_visible()

    selection.prune(("source-1", "source-3", "source-4"))

    assert selection.selected_ids == frozenset({"source-1", "source-3"})


def test_id_selection_model_reanchors_range_after_anchor_is_deleted():
    """Catches a removed anchor making the next Shift act on a stale row."""
    selection = IdSelectionModel()
    selection.set_visible_ids(("source-1", "source-2", "source-3"))
    selection.toggle("source-1")
    selection.prune(("source-2", "source-3"))
    selection.set_visible_ids(("source-2", "source-3"))

    assert selection.anchor_id is None
    assert selection.shift("source-2", 1) == "source-3"
    assert selection.selected_ids == frozenset({"source-2", "source-3"})


@pytest.mark.asyncio
async def test_scoped_source_rows_do_not_prune_authoritative_selection(sample_sources):
    """A pane row subset is visibility input, not source-deletion truth."""
    app = SourcesPaneHarness()
    async with app.run_test(size=(160, 42)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        pane.set_authoritative_source_ids(
            tuple(str(source["id"]) for source in sample_sources)
        )
        await pilot.pause()
        pane.set_selected_source_ids(("source-1", "source-3"))
        await pilot.pause()
        app.captured_messages.clear()

        pane.sources = sample_sources[:2]
        await pilot.pause()

        assert pane.selected_source_ids == frozenset({"source-1", "source-3"})
        assert app.captured_messages == []


class SourcesPaneHarness(App):
    def __init__(self):
        super().__init__()
        self.captured_messages = []

    def compose(self) -> ComposeResult:
        yield SourcesPane()

    def on_source_selected(self, message: SourceSelected) -> None:
        self.captured_messages.append(("source_selected", message.source))

    def on_source_selection_changed(self, message: SourceSelectionChanged) -> None:
        self.captured_messages.append(("source_selection_changed", message.source_ids))

    def on_create_source_requested(self, message: CreateSourceRequested) -> None:
        self.captured_messages.append(
            ("create_source_requested", message.runtime_backend, message.payload)
        )

    def on_preview_requested(self, message: PreviewRequested) -> None:
        self.captured_messages.append(("preview_requested", message.entity))

    def on_check_now_requested(self, message: CheckNowRequested) -> None:
        self.captured_messages.append(("check_now_requested", message.entity))

    def on_import_opml_requested(self, message: ImportOpmlRequested) -> None:
        self.captured_messages.append(("import_opml_requested", None))

    def on_export_opml_requested(self, message: ExportOpmlRequested) -> None:
        self.captured_messages.append(("export_opml_requested", None))

    def on_open_bulk_sources_requested(
        self, message: OpenBulkSourcesRequested
    ) -> None:
        self.captured_messages.append(("open_bulk_sources_requested", None))

    def on_create_watchlist_from_selected_requested(
        self, message: CreateWatchlistFromSelectedRequested
    ) -> None:
        self.captured_messages.append(
            ("create_watchlist_from_selected_requested", message.source_ids)
        )


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


async def _open_page_create_form(pilot, app) -> SourcesPane:
    """Open the create form with a page-scrape type chosen.

    TASK-2302 renders the noise field only for the url family (CSS selectors
    describe elements on a page; an RSS feed has none), so every test in this
    module that is about that field has to put the form in the state the
    field exists in. `Select.value` is the same state change a click through
    the overlay makes, and the pane recomposes around it.
    """
    pane = app.query_one(SourcesPane)
    pane.query_one("#sources-new-button", Button).press()
    await pilot.pause()
    pane.query_one("#sources-create-type", Select).value = "url"
    for _ in range(100):
        await pilot.pause()
        if pane.query("#sources-create-ignore-selectors"):
            break
    assert pane.query("#sources-create-ignore-selectors"), (
        "choosing a page type did not bring the noise field back"
    )
    return pane


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
    pane = await _open_page_create_form(pilot, app)

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
    assert app.captured_messages[-1][1] == "local"
    stored = app._service._db().get_subscription(
        int(app.created_sources[0]["source_id"])
    )
    assert stored is not None, "the source was not persisted"
    return stored


def _option_pairs(select: Select) -> list[tuple[str, object]]:
    """Return a Select's labels and values in display order."""
    return [(str(label), value) for label, value in select._options]


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
        pane = await _open_page_create_form(pilot, app)
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
async def test_create_is_refused_when_a_noise_line_is_not_valid_css(tmp_path):
    """Whole-branch fix F1, UI side: name the bad line, at the keyboard.

    `soup.select` raises on anything CSS cannot parse. The extraction side now
    survives that, but a silently-skipped rule is still a rule the user
    believes is suppressing noise and that does nothing, and nothing else in
    the product would ever tell them -- the only place the mistake is cheap to
    fix is the form they are still looking at. Mutation: delete the
    `first_invalid_selector` check in `_submit_create_form` and this reddens
    (a row appears and no toast is delivered).
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    app = PersistingSourcesPaneHarness(LocalWatchlistsService(db_factory=lambda: db))
    toasts: list[tuple[str, dict]] = []
    async with app.run_test(size=(120, 40)) as pilot:
        app.notify = lambda message, **kwargs: toasts.append((str(message), kwargs))
        pane = await _open_page_create_form(pilot, app)

        field = pane.query_one("#sources-create-ignore-selectors", TextArea)
        # One good line, one unparseable one: the refusal must not need the
        # whole field to be broken.
        field.text = ".ad\ndiv[\n.promo"
        await pilot.pause()
        pane.query_one("#sources-create-name", Input).value = "Noisy Page"
        pane.query_one("#sources-create-url", Input).value = "https://example.com/page"
        await pilot.pause()

        pane.query_one("#sources-create-submit", Button).press()
        for _ in range(20):
            await pilot.pause()

        assert app.create_error is None, app.create_error
        assert not app.created_sources, (
            "a source with an unparseable ignore rule must not be created"
        )
        assert not db.get_all_subscriptions(), (
            "nothing may reach the subscriptions table"
        )

        assert toasts, "the refusal must be visible, not only a return"
        message, kwargs = toasts[-1]
        assert kwargs.get("severity") == "error"
        assert "div[" in message, (
            "the toast must name the offending LINE -- 'a selector is invalid' "
            f"is unactionable when the field holds three; got {message!r}"
        )
        # Selectors are full of `[`, which Textual's toast markup parses.
        assert kwargs.get("markup") is False, (
            "the message must be delivered with markup off or the bracket in "
            "the selector is eaten before the user sees it"
        )

        # And the form stays open with the text intact, so the fix is one edit
        # away rather than a retyped form.
        assert pane.show_create_form, "the form must stay open to be corrected"
        assert (
            pane.query_one("#sources-create-ignore-selectors", TextArea).text
            == ".ad\ndiv[\n.promo"
        )


@pytest.mark.asyncio
async def test_a_valid_multi_line_noise_field_still_creates(tmp_path):
    """The refusal's other direction: real selectors must not be rejected.

    Commas inside a line are CSS selector GROUPS, and `:is(...)` /
    attribute-substring forms are exactly what the shipped defaults use -- a
    validator that split lines on commas, or that rejected anything exotic,
    would refuse every source created with the prefill untouched.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    app = PersistingSourcesPaneHarness(LocalWatchlistsService(db_factory=lambda: db))
    toasts: list[tuple[str, dict]] = []
    async with app.run_test(size=(120, 40)) as pilot:
        app.notify = lambda message, **kwargs: toasts.append((str(message), kwargs))
        pane = await _open_page_create_form(pilot, app)

        exotic = (
            default_ignore_selectors_text()
            + '\n:is(.a, .b)\n[data-x="a,b"]\ndiv:has(> p)\n\n'
        )
        pane.query_one("#sources-create-ignore-selectors", TextArea).text = exotic
        await pilot.pause()

        stored = await _create_through_the_form(pilot, app)

        assert stored["ignore_selectors"] == exotic.strip(), (
            "every one of these lines is valid CSS and must be stored verbatim"
        )
        assert not [t for t in toasts if t[1].get("severity") == "error"], (
            f"a valid field produced an error toast: {toasts!r}"
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
        pane = await _open_page_create_form(pilot, app)

        field = pane.query_one("#sources-create-ignore-selectors", TextArea)
        assert field.display and field.region.height > 0, (
            "the noise field is not on screen"
        )
        assert field.text == default_ignore_selectors_text()
        # TASK-2302 shortened both strings to fit the field's REAL width (see
        # `_IGNORE_SELECTORS_LABEL`); what they have to say is asserted here,
        # and that they fit is asserted against the mounted field's own width
        # in `Tests/UI/test_watchlists_create_form_destination.py`.
        assert "Ignore elements" in str(field.border_title)
        assert "CSS selectors" in str(field.border_title)
        # The spam -> add-a-selector loop has to be stated where the field is.
        assert "silence" in str(field.border_subtitle).lower()
        # And the syntax detail the shortening displaced is still reachable.
        assert "comma" in str(field.tooltip)


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
        app.query_one(SourcesPane).create_draft_ignore_selectors = ""
        pane = await _open_page_create_form(pilot, app)

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
        assert pane.query_one("#sources-new-button", Button)
        assert pane.query_one("#sources-add-several-button", Button)
        assert pane.query_one("#sources-table", DataTable)
        assert not pane.query("#sources-type-select")

        pane.query_one("#sources-filter-toggle", Button).press()
        await pilot.pause()

        editor = pane.query_one("#sources-filter-editor")
        labels = [
            str(label.render())
            for label in editor.query(".sources-filter-label").results(Static)
        ]
        assert labels == ["Type", "Status", "Active", "Tags"]
        assert editor.query_one("#sources-type-select", Select)
        assert editor.query_one("#sources-status-filter", Select)
        assert editor.query_one("#sources-active-filter", Select)
        assert editor.query_one("#sources-tags-filter", Input)


@pytest.mark.asyncio
async def test_sources_pane_add_several_posts_one_bulk_open_request():
    """Catches the peer bulk action being decorative or routed through single-add."""
    app = SourcesPaneHarness()
    async with app.run_test(size=(160, 42)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.query_one("#sources-add-several-button", Button).press()
        await pilot.pause()

        assert app.captured_messages == [("open_bulk_sources_requested", None)]


@pytest.mark.asyncio
async def test_sources_table_keyboard_selection_is_focus_scoped_and_id_based(sample_sources):
    """Catches global key interception or selection stored by cursor row."""
    app = SourcesPaneHarness()
    async with app.run_test(size=(160, 42)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        await pilot.pause()
        table = pane.query_one("#sources-table", DataTable)
        table.focus()
        table.move_cursor(row=0, animate=False)

        await pilot.press("space")
        await pilot.press("shift+down")
        await pilot.pause()

        assert pane.selected_source_ids == frozenset({"source-1", "source-2"})
        assert "2 selected" in str(
            pane.query_one("#sources-selection-status", Static).render()
        )
        assert str(table.get_row("source-1")[0]).startswith("[x] ")
        assert str(table.get_row("source-2")[0]).startswith("[x] ")
        assert not pane.query_one(
            "#sources-create-watchlist-selected", Button
        ).disabled

        pane.query_one("#sources-search-input", Input).focus()
        await pilot.press("x")
        await pilot.pause()
        assert pane.selected_source_ids == frozenset({"source-1", "source-2"})


@pytest.mark.asyncio
async def test_sources_visible_toggle_keeps_hidden_selection_and_clear_removes_it(sample_sources):
    """Catches v applying globally or x clearing visible rows only."""
    app = SourcesPaneHarness()
    async with app.run_test(size=(160, 42)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        await pilot.pause()
        table = pane.query_one("#sources-table", DataTable)
        table.focus()
        table.move_cursor(row=0, animate=False)
        await pilot.press("space")

        pane.search_query = "Tech"
        await pilot.pause()
        assert "1 hidden by filters" in str(
            pane.query_one("#sources-selection-status", Static).render()
        )
        await pilot.press("v")
        await pilot.pause()
        assert pane.selected_source_ids == frozenset({"source-1", "source-2"})
        assert "1 hidden by filters" in str(
            pane.query_one("#sources-selection-status", Static).render()
        )

        await pilot.press("x")
        await pilot.pause()
        assert pane.selected_source_ids == frozenset()


@pytest.mark.asyncio
async def test_create_watchlist_from_selected_posts_canonical_ids(sample_sources):
    """Catches collection creation falling back to repeated row dialogs."""
    app = SourcesPaneHarness()
    async with app.run_test(size=(160, 42)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        await pilot.pause()
        pane.set_selected_source_ids(("source-3", "source-1"))
        pane.query_one("#sources-create-watchlist-selected", Button).press()
        await pilot.pause()

        assert app.captured_messages[-1] == (
            "create_watchlist_from_selected_requested",
            ("source-1", "source-3"),
        )


@pytest.mark.asyncio
async def test_create_watchlist_from_selected_disables_above_domain_limit():
    """Catches the UI offering a collection request the domain must reject."""
    app = SourcesPaneHarness()
    async with app.run_test(size=(160, 42)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = [
            {
                "id": f"local:subscription:{index}",
                "name": f"Source {index}",
                "source_type": "rss",
                "active": True,
            }
            for index in range(1, 102)
        ]
        await pilot.pause()
        pane.set_selected_source_ids(
            tuple(f"local:subscription:{index}" for index in range(1, 102))
        )

        assert pane.query_one(
            "#sources-create-watchlist-selected", Button
        ).disabled
        assert "100" in str(
            pane.query_one("#sources-selection-status", Static).render()
        )


@pytest.mark.asyncio
async def test_the_last_checked_column_uses_the_check_vocabulary():
    """TASK-2313, AC#2: "scraped" vs "checked" terminology drift -- this
    column was the one holdout still saying "Last scraped" while every
    button/toast elsewhere on this screen says "check"/"Check now"."""
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)):
        pane = app.query_one(SourcesPane)
        table = pane.query_one("#sources-table", DataTable)
        columns = [str(col.label) for col in table.columns.values()]
        assert "Last checked" in columns
        assert "Last scraped" not in columns


@pytest.mark.asyncio
async def test_toolbar_filter_selects_each_carry_a_visible_label():
    """Every filter has a persistent label inside the focused disclosure."""
    app = SourcesPaneHarness()
    async with app.run_test(size=(160, 42)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.query_one("#sources-filter-toggle", Button).press()
        await pilot.pause()

        editor = pane.query_one("#sources-filter-editor")
        assert [
            str(label.render())
            for label in editor.query(".sources-filter-label").results(Static)
        ] == [
            "Type",
            "Status",
            "Active",
            "Tags",
        ]
        assert editor.region.right <= pane.region.right


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
        kind, runtime_backend, payload = app.captured_messages[0]
        assert kind == "create_source_requested"
        assert runtime_backend == "local"
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

        _kind, runtime_backend, payload = app.captured_messages[0]
        assert runtime_backend == "local"
        assert payload["check_frequency"] == 86_400


@pytest.mark.asyncio
async def test_backend_specific_create_type_options_and_filter_options():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.query_one("#sources-new-button", Button).press()
        await pilot.pause()

        assert _option_pairs(pane.query_one("#sources-create-type", Select)) == [
            ("RSS", "rss"),
            ("Atom", "atom"),
            ("Web page", "url"),
        ]
        pane.query_one("#sources-filter-toggle", Button).press()
        await pilot.pause()
        table = pane.query_one("#sources-table", DataTable)
        type_select = pane.query_one("#sources-create-type", Select)
        assert _option_pairs(pane.query_one("#sources-type-select", Select)) == [
            ("All", "all"),
            ("RSS", "rss"),
            ("Atom", "atom"),
            ("Feed", "feed"),
            ("Playlist", "playlist"),
            ("Channel", "channel"),
            ("Web page", "url"),
        ]

        pane.configure_create_backend("server", ("rss", "site", "forum"))
        await pilot.pause()

        assert pane.query_one("#sources-table", DataTable) is table
        assert pane.query_one("#sources-create-type", Select) is type_select
        assert _option_pairs(type_select) == [
            ("RSS", "rss"),
            ("Site", "site"),
            ("Forum", "forum"),
        ]
        assert _option_pairs(pane.query_one("#sources-type-select", Select)) == [
            ("All", "all"),
            ("RSS", "rss"),
            ("Atom", "atom"),
            ("Feed", "feed"),
            ("Playlist", "playlist"),
            ("Channel", "channel"),
            ("Web page", "url"),
        ]


@pytest.mark.asyncio
async def test_backend_switch_preserves_complete_draft_and_open_form():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.watchlist_choices = [{"id": 7, "name": "Research"}]
        pane.query_one("#sources-new-button", Button).press()
        await pilot.pause()

        pane.query_one("#sources-create-type", Select).value = "url"
        for _ in range(20):
            await pilot.pause()
            if pane.query("#sources-create-ignore-selectors"):
                break
        pane.query_one("#sources-create-name", Input).value = "Draft source"
        pane.query_one("#sources-create-url", Input).value = "https://example.com"
        pane.query_one("#sources-create-active", Switch).value = False
        pane.query_one("#sources-create-watchlist", Select).value = 7
        pane.query_one("#sources-create-tags", Input).value = "alpha, beta"
        pane.query_one("#sources-create-frequency", Select).value = 86_400
        pane.query_one("#sources-create-ignore-selectors", TextArea).text = (
            ".advert\n.promo"
        )
        destination = pane.query_one("#sources-create-watchlist", Select)
        await pilot.pause()

        pane.configure_create_backend("server", ("rss", "site", "forum"))
        await pilot.pause()

        assert pane.show_create_form
        assert pane.query_one("#sources-create-name", Input).value == "Draft source"
        assert pane.query_one("#sources-create-url", Input).value == "https://example.com"
        assert pane.query_one("#sources-create-active", Switch).value is False
        assert pane.query_one("#sources-create-watchlist", Select) is destination
        assert destination.disabled is True
        assert destination.value == SourcesPane.UNASSIGNED_DESTINATION
        assert pane.create_draft_destination == 7
        assert pane.query_one("#sources-create-tags", Input).value == "alpha, beta"
        assert pane.query_one("#sources-create-type", Select).value == "rss"
        assert pane.query_one("#sources-create-frequency").display is False
        assert pane.query_one("#sources-create-ignore-selectors").display is False

        pane.configure_create_backend("local", ("rss", "atom", "url"))
        await pilot.pause()

        assert pane.show_create_form
        assert pane.query_one("#sources-create-name", Input).value == "Draft source"
        assert pane.query_one("#sources-create-url", Input).value == "https://example.com"
        assert pane.query_one("#sources-create-active", Switch).value is False
        assert pane.query_one("#sources-create-watchlist", Select) is destination
        assert destination.disabled is False
        assert destination.value == 7
        assert pane.query_one("#sources-create-tags", Input).value == "alpha, beta"
        frequency = pane.query_one("#sources-create-frequency", Select)
        assert frequency.display is True
        assert frequency.value == 86_400
        assert pane.create_draft_ignore_selectors == ".advert\n.promo"
        assert pane.query_one("#sources-create-ignore-selectors").display is False
        pane.query_one("#sources-create-type", Select).value = "url"
        await pilot.pause()
        ignore_selectors = pane.query_one(
            "#sources-create-ignore-selectors", TextArea
        )
        assert ignore_selectors.display is True
        assert ignore_selectors.text == ".advert\n.promo"


@pytest.mark.asyncio
async def test_server_to_local_submit_without_recompose_uses_saved_frequency():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.query_one("#sources-new-button", Button).press()
        await pilot.pause()

        pane.query_one("#sources-create-name", Input).value = "Draft source"
        pane.query_one("#sources-create-url", Input).value = "https://example.com"
        pane.query_one("#sources-create-frequency", Select).value = 86_400
        await pilot.pause()

        pane.configure_create_backend("server", ("rss", "site", "forum"))
        await pilot.pause()
        assert pane.query_one("#sources-create-frequency").display is False

        pane.configure_create_backend("local", ("rss", "atom", "url"))
        pane._submit_create_form()
        await pilot.pause()

        assert app.captured_messages == [
            (
                "create_source_requested",
                "local",
                {
                    "name": "Draft source",
                    "url": "https://example.com",
                    "source_type": "rss",
                    "active": True,
                    "tags": [],
                    "watchlist_id": None,
                    "check_frequency": 86_400,
                    "ignore_selectors": "",
                },
            )
        ]


@pytest.mark.asyncio
async def test_server_payload_omits_local_fields_and_captures_backend():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.configure_create_backend("server", ("rss", "site", "forum"))
        await pilot.pause()
        pane.query_one("#sources-new-button", Button).press()
        await pilot.pause()

        pane.query_one("#sources-create-name", Input).value = "Forum source"
        pane.query_one("#sources-create-url", Input).value = "https://example.com/forum"
        pane.query_one("#sources-create-type", Select).value = "forum"
        await pilot.pause()
        pane.query_one("#sources-create-active", Switch).value = False
        pane.query_one("#sources-create-tags", Input).value = "community, updates"
        await pilot.pause()
        pane.query_one("#sources-create-submit", Button).press()
        await pilot.pause()

        assert app.captured_messages == [
            (
                "create_source_requested",
                "server",
                {
                    "name": "Forum source",
                    "url": "https://example.com/forum",
                    "source_type": "forum",
                    "active": False,
                    "tags": ["community", "updates"],
                    "watchlist_id": None,
                },
            )
        ]
        assert pane.create_draft_active is True
        assert pane.create_draft_frequency == 3600


async def _submit_unsupported_form_type(
    pilot,
    app: SourcesPaneHarness,
    *,
    backend: str,
    source_types: tuple[str, ...],
    value: object,
) -> tuple[SourcesPane, list[tuple[str, dict]]]:
    pane = app.query_one(SourcesPane)
    pane.configure_create_backend(backend, source_types)
    await pilot.pause()
    pane.query_one("#sources-new-button", Button).press()
    await pilot.pause()
    pane.query_one("#sources-create-name", Input).value = "Stale source"
    pane.query_one("#sources-create-url", Input).value = "https://example.com/stale"
    type_select = pane.query_one("#sources-create-type", Select)
    type_select.set_options([("Stale", value)])
    type_select.value = value
    toasts: list[tuple[str, dict]] = []
    app.notify = lambda message, **kwargs: toasts.append((str(message), kwargs))

    pane._submit_create_form()
    await pilot.pause()
    return pane, toasts


@pytest.mark.asyncio
async def test_unsupported_form_type_sitemap_is_rejected_before_event(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    accepted = await service.create_source(
        {
            "name": "Imported sitemap",
            "url": "https://example.com/sitemap.xml",
            "source_type": "sitemap",
            "active": True,
            "tags": [],
            "watchlist_id": None,
        }
    )
    assert accepted["source_type"] == "sitemap"

    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane, _toasts = await _submit_unsupported_form_type(
            pilot,
            app,
            backend="local",
            source_types=("rss", "atom", "url"),
            value="sitemap",
        )

        assert pane.show_create_form
        assert not app.captured_messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "source_types", "expected"),
    [
        (
            "local",
            ("rss", "atom", "url"),
            "Local sources don't support 'Playlist'. Choose RSS, Atom, or Web page.",
        ),
        (
            "server",
            ("rss", "site", "forum"),
            "Server sources don't support 'Playlist'. Choose RSS, Site, or Forum.",
        ),
    ],
)
async def test_source_type_recovery_uses_exact_registered_copy(
    backend, source_types, expected
):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane, toasts = await _submit_unsupported_form_type(
            pilot,
            app,
            backend=backend,
            source_types=source_types,
            value="playlist",
        )

        assert pane.show_create_form
        assert not app.captured_messages
        assert toasts == [(expected, {"severity": "error", "markup": False})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "display"),
    [
        ("\x00  Strange\t Type\n ", "Strange Type"),
        ("x" * 41, f"{'x' * 39}…"),
        ("\x00\t\n", "Unknown"),
    ],
)
async def test_source_type_recovery_normalizes_unregistered_values(value, display):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane, toasts = await _submit_unsupported_form_type(
            pilot,
            app,
            backend="local",
            source_types=("rss", "atom", "url"),
            value=value,
        )

        assert pane.show_create_form
        assert not app.captured_messages
        assert toasts == [
            (
                f"Local sources don't support '{display}'. "
                "Choose RSS, Atom, or Web page.",
                {"severity": "error", "markup": False},
            )
        ]


@pytest.mark.asyncio
async def test_sources_pane_action_buttons_exist():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)):
        pane = app.query_one(SourcesPane)
        assert pane.query_one("#sources-preview-button", Button)
        assert pane.query_one("#sources-check-now-button", Button)
        assert pane.query_one("#sources-import-opml-button", Button)
        assert pane.query_one("#sources-export-opml-button", Button)


@pytest.mark.asyncio
async def test_sources_pane_preview_and_check_now_disabled_without_selection():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)):
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
async def test_paused_sources_stay_in_the_error_bucket_and_get_their_own():
    """task-2050 review: an auto-paused source's `status_summary` now reads
    "paused" (paused wins the precedence over error), so without an explicit
    branch the Error filter -- the triage view for broken feeds -- would
    silently skip exactly the most-broken sources. Paused sources must appear
    under BOTH the Error bucket (broken-feed triage keeps working) and the
    new dedicated Paused bucket. Reds if the error branch loses its paused
    arm, or the Paused option vanishes.
    """
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = [
            {"id": "s1", "name": "Healthy", "source_type": "rss", "status_summary": "active", "active": True},
            {"id": "s2", "name": "Erroring", "source_type": "rss", "status_summary": "error (3)", "active": True},
            {"id": "s3", "name": "AutoPaused", "source_type": "rss", "status_summary": "paused", "active": False},
        ]

        pane.status_filter = "error"
        await pilot.pause()
        table = pane.query_one("#sources-table", DataTable)
        names = {str(table.get_row_at(i)[0]) for i in range(table.row_count)}
        assert any("Erroring" in n for n in names)
        assert any("AutoPaused" in n for n in names), (
            "the Error triage bucket must include auto-paused sources -- they "
            "failed PAST the threshold"
        )
        assert not any("Healthy" in n for n in names)

        pane.status_filter = "paused"
        await pilot.pause()
        table = pane.query_one("#sources-table", DataTable)
        names = {str(table.get_row_at(i)[0]) for i in range(table.row_count)}
        assert names and all("AutoPaused" in n for n in names)

        # The Paused option genuinely exists in the dropdown.
        assert ("Paused", "paused") in SourcesPane._STATUS_OPTIONS


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


@pytest.mark.asyncio
async def test_every_select_in_the_pane_is_prune_safe():
    """TASK-1960: the toolbar filters must survive a prune mid-mount.

    This pane is torn down and rebuilt by two independent recomposes that
    can overlap -- its own `show_create_form` toggle, and the owning
    screen's `refresh(recompose=True)` from `_apply_local_wc_snapshot` /
    `_load_tree_data`. When the screen's prune lands between one of these
    `Select`s registering its `SelectCurrent` and that `SelectCurrent`
    composing, stock `Select._on_mount` raises `NoMatches` on `#label` and
    Textual turns it into an app-level crash.

    Pinned structurally rather than by reproducing the race: the race itself
    is timing-dependent (that is the whole of TASK-1960), so a call site
    quietly reverted to a stock `Select` would otherwise only ever be caught
    by an intermittent failure in a different test file.
    """
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.show_create_form = True
        await pilot.pause()

        selects = list(pane.query(Select))
        assert selects, "the pane should compose Selects to check"
        offenders = [
            select.id for select in selects
            if not isinstance(select, PruneSafeSelect)
        ]
        assert not offenders, f"stock Select still used for: {offenders}"


def test_source_row_name_strips_control_characters():
    """Batch-4 review, I1. `name` is remote-derived (OPML import hands an
    `<outline text=...>` attribute straight through with zero sanitization
    -- see `Tests/Subscriptions/test_watchlist_opml_service.py` for the full
    delivery-path proof), and `Text(...)` protects only against Rich markup,
    not a raw control byte.
    """
    cells = SourcesPane._source_row_cells(
        {"name": "Evil\x9b31mFeed", "source_type": "rss"}, False
    )
    assert "\x9b" not in cells[0].plain
    assert "Evil" in cells[0].plain and "31mFeed" in cells[0].plain
