"""Tests for ``ConsoleScopePickerModal`` (RAG scope narrowing picker, Task 8).

Harness mirrors ``Tests/UI/test_console_prompt_picker.py``'s ``ModalHarness``
(bare ``App[None]`` subclass, push the modal directly, capture ``on_save``
calls into a list). Fake ``media_lister``/``notes_lister``/``tag_lister``
implement the protocols documented in ``console_scope_picker_modal.py`` over
small in-memory fixtures -- no DB, no Screens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
from textual.app import App
from textual.widgets import Button, Checkbox, Input, Select, Static

from tldw_chatbook.Chat.rag_scope import SOURCE_TYPE_MEDIA, RagScope, ScopeItem
from tldw_chatbook.Widgets.Console.console_scope_picker_modal import (
    CANCEL_BTN_ID,
    CLEAR_SCOPE_BTN_ID,
    CLEAR_SHOWN_BTN_ID,
    CONFIRM_NO_BTN_ID,
    CONFIRM_STATIC_ID,
    CONFIRM_YES_BTN_ID,
    COUNT_STATIC_ID,
    EMPTY_STATIC_ID,
    FILTER_DEBOUNCE_SECONDS,
    LIST_CONTAINER_ID,
    OUTSIDE_UNIVERSE_SUFFIX,
    PAGE_LABEL_ID,
    PAGE_NEXT_ID,
    SAVE_BTN_ID,
    SELECT_ALL_BTN_ID,
    SORT_SELECT_ID,
    SORT_TITLE,
    SORT_TYPE,
    TAB_MEDIA_BTN_ID,
    TEXT_FILTER_ID,
    VIEW_ALL_BTN_ID,
    VIEW_SELECTED_BTN_ID,
    ConsoleScopePickerModal,
    ScopeListItem,
    ScopeListPage,
    TagCount,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTIC_TERMINAL = (
    REPO_ROOT / "tldw_chatbook" / "css" / "components" / "_agentic_terminal.tcss"
)
BUNDLED_STYLESHEET = REPO_ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"


# -- fake listers ---------------------------------------------------------------


class FakeSourceLister:
    """In-memory ``SourceLister`` over a fixed item list, for a single type."""

    def __init__(self, items: list[ScopeListItem]) -> None:
        self.items = items
        self.list_page_calls: list[dict] = []
        self.list_ids_calls: list[dict] = []

    def _matching(self, *, text: str, tags: tuple[str, ...]) -> list[ScopeListItem]:
        result = []
        for item in self.items:
            if text and text.lower() not in item.title.lower():
                continue
            if tags and not (set(tags) & set(item.tags)):
                continue
            result.append(item)
        return result

    async def list_page(
        self, *, text: str, tags: tuple[str, ...], sort: str, offset: int, limit: int
    ) -> ScopeListPage:
        self.list_page_calls.append(
            {"text": text, "tags": tags, "sort": sort, "offset": offset, "limit": limit}
        )
        matched = self._matching(text=text, tags=tags)
        if sort == SORT_TITLE:
            matched = sorted(matched, key=lambda i: i.title.lower())
        elif sort == SORT_TYPE:
            pass  # single-type lister; type-sort is a no-op here
        else:
            matched = sorted(matched, key=lambda i: i.updated_at, reverse=True)
        page = tuple(matched[offset : offset + limit])
        return ScopeListPage(items=page, total_matching=len(matched))

    async def list_ids(self, *, text: str, tags: tuple[str, ...]) -> tuple[str, ...]:
        self.list_ids_calls.append({"text": text, "tags": tags})
        return tuple(i.source_id for i in self._matching(text=text, tags=tags))


class FakeTagLister:
    """In-memory tag vocabulary, callable per the ``TagLister`` protocol."""

    def __init__(self, counts: list[TagCount]) -> None:
        self.counts = sorted(counts, key=lambda tc: tc.count, reverse=True)
        self.calls: list[str] = []

    async def __call__(self, query: str) -> tuple[TagCount, ...]:
        self.calls.append(query)
        if not query:
            return tuple(self.counts[:10])
        q = query.lower()
        return tuple(tc for tc in self.counts if q in tc.tag.lower())


def _media(source_id: str, title: str, updated_at: str = "2026-01-01", tags=()) -> ScopeListItem:
    return ScopeListItem(source_id=source_id, title=title, updated_at=updated_at, tags=tuple(tags))


def _note(source_id: str, title: str, updated_at: str = "2026-01-01", tags=()) -> ScopeListItem:
    return ScopeListItem(source_id=source_id, title=title, updated_at=updated_at, tags=tuple(tags))


# -- harness --------------------------------------------------------------------


class ModalHarness(App[None]):
    """Loads the real bundled stylesheet (mirrors ``test_console_session_
    settings.py``'s ``StyledModalHarness``) -- this modal's footer/list
    layout relies on real CSS (``1fr``/``auto`` width interplay) to keep
    the Save/Clear/Cancel row within screen bounds for ``pilot.click``."""

    CSS_PATH = str(BUNDLED_STYLESHEET)

    def __init__(self) -> None:
        super().__init__()
        self.saved: list[Optional[RagScope]] = []

    def capture_save(self, scope: Optional[RagScope]) -> None:
        self.saved.append(scope)


async def _settle(pilot) -> None:
    """Advance past debounce timers and let any in-flight worker settle."""
    await pilot.pause(FILTER_DEBOUNCE_SECONDS + 0.1)
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


def _row_labels(app) -> list[str]:
    container = app.screen.query_one(f"#{LIST_CONTAINER_ID}")
    return [str(cb.label) for cb in container.query(Checkbox)]


async def _open(
    app: ModalHarness,
    pilot,
    *,
    media: list[ScopeListItem] | None = None,
    notes: list[ScopeListItem] | None = None,
    tags: list[TagCount] | None = None,
    universe=None,
    initial: Optional[RagScope] = None,
    target_label: str = "this conversation",
    page_size: int = 20,
) -> tuple[FakeSourceLister, FakeSourceLister, FakeTagLister]:
    media_lister = FakeSourceLister(media or [])
    notes_lister = FakeSourceLister(notes or [])
    tag_lister = FakeTagLister(tags or [])
    await app.push_screen(
        ConsoleScopePickerModal(
            target_label,
            universe,
            initial,
            app.capture_save,
            media_lister=media_lister,
            notes_lister=notes_lister,
            tag_lister=tag_lister,
            page_size=page_size,
        )
    )
    await pilot.pause()
    await _settle(pilot)
    return media_lister, notes_lister, tag_lister


# -- list loads + renders (types/glyphs) -----------------------------------------


@pytest.mark.asyncio
async def test_list_loads_off_loop_and_renders_with_type_glyphs() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(
            app,
            pilot,
            media=[_media("m1", "Quarterly report")],
            notes=[_note("n1", "Meeting notes")],
        )
        labels = _row_labels(app)
        assert any("Quarterly report" in label for label in labels)
        assert any("Meeting notes" in label for label in labels)
        # Distinct glyphs identify each source type.
        media_label = next(label for label in labels if "Quarterly report" in label)
        note_label = next(label for label in labels if "Meeting notes" in label)
        assert media_label[0] != note_label[0]


@pytest.mark.asyncio
async def test_title_names_the_target() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(app, pilot, target_label="workspace 'hunt X'")
        header = app.screen.query(".console-modal-header").first(Static)
        assert "Narrow RAG scope — workspace 'hunt X'" == str(header.renderable)


@pytest.mark.asyncio
async def test_empty_store_shows_copy() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(app, pilot, media=[], notes=[])
        empty = app.screen.query_one(f"#{EMPTY_STATIC_ID}", Static)
        assert "No matching" in str(empty.renderable)


# -- selection survives filter/sort/tab changes ----------------------------------


@pytest.mark.asyncio
async def test_selection_survives_filter_sort_and_tab_changes() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(
            app,
            pilot,
            media=[_media("m1", "Alpha report"), _media("m2", "Beta report")],
            notes=[_note("n1", "Gamma notes")],
        )
        # Select the first rendered row (order may vary; grab by content).
        container = app.screen.query_one(f"#{LIST_CONTAINER_ID}")
        alpha_checkbox = next(
            cb for cb in container.query(Checkbox) if "Alpha report" in str(cb.label)
        )
        alpha_checkbox.value = True
        await pilot.pause()

        count = app.screen.query_one(f"#{COUNT_STATIC_ID}", Static)
        assert str(count.renderable).startswith("1 selected of")

        # Change text filter -- selection (footer count) must survive.
        filter_input = app.screen.query_one(f"#{TEXT_FILTER_ID}", Input)
        filter_input.value = "Beta"
        await _settle(pilot)
        count = app.screen.query_one(f"#{COUNT_STATIC_ID}", Static)
        assert str(count.renderable).startswith("1 selected of")

        filter_input.value = ""
        await _settle(pilot)

        # Change sort.
        sort_select = app.screen.query_one(f"#{SORT_SELECT_ID}", Select)
        sort_select.value = SORT_TITLE
        await _settle(pilot)
        count = app.screen.query_one(f"#{COUNT_STATIC_ID}", Static)
        assert str(count.renderable).startswith("1 selected of")

        # Change tab.
        await pilot.click(f"#{TAB_MEDIA_BTN_ID}")
        await _settle(pilot)
        count = app.screen.query_one(f"#{COUNT_STATIC_ID}", Static)
        assert str(count.renderable).startswith("1 selected of")

        # Confirm the Selected view still shows exactly Alpha.
        await pilot.click(f"#{VIEW_SELECTED_BTN_ID}")
        await pilot.pause()
        labels = _row_labels(app)
        assert len(labels) == 1
        assert "Alpha report" in labels[0]


# -- Selected-view default + out-of-universe marking -----------------------------


@pytest.mark.asyncio
async def test_opens_in_all_view_when_unscoped() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(app, pilot, media=[_media("m1", "Alpha")])
        all_btn = app.screen.query_one(f"#{VIEW_ALL_BTN_ID}", Button)
        selected_btn = app.screen.query_one(f"#{VIEW_SELECTED_BTN_ID}", Button)
        assert all_btn.has_class("console-scope-picker-view-active")
        assert not selected_btn.has_class("console-scope-picker-view-active")


@pytest.mark.asyncio
async def test_opens_in_selected_view_when_initial_scope_present() -> None:
    app = ModalHarness()
    initial = RagScope(
        items=(ScopeItem(SOURCE_TYPE_MEDIA, "m1"),), updated_at="2026-01-01T00:00:00Z"
    )
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(app, pilot, media=[_media("m1", "Alpha report")], initial=initial)
        selected_btn = app.screen.query_one(f"#{VIEW_SELECTED_BTN_ID}", Button)
        assert selected_btn.has_class("console-scope-picker-view-active")
        # Selected view renders immediately, without waiting on lister I/O.
        labels = _row_labels(app)
        assert len(labels) == 1


@pytest.mark.asyncio
async def test_selected_view_marks_out_of_universe_items() -> None:
    app = ModalHarness()
    initial = RagScope(
        items=(
            ScopeItem(SOURCE_TYPE_MEDIA, "m1"),
            ScopeItem(SOURCE_TYPE_MEDIA, "m2"),
        ),
        updated_at="2026-01-01T00:00:00Z",
    )
    universe = frozenset({(SOURCE_TYPE_MEDIA, "m1")})  # m2 is NOT in the universe
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(
            app,
            pilot,
            media=[_media("m1", "In scope"), _media("m2", "Out of scope")],
            initial=initial,
            universe=universe,
        )
        container = app.screen.query_one(f"#{LIST_CONTAINER_ID}")
        checkboxes = list(container.query(Checkbox))
        assert len(checkboxes) == 2
        # Locate rows by label content directly (more robust than id order).
        in_scope_row = next(cb for cb in checkboxes if "In scope" in str(cb.label))
        out_scope_row = next(cb for cb in checkboxes if "Out of scope" in str(cb.label))
        assert OUTSIDE_UNIVERSE_SUFFIX.strip() not in str(in_scope_row.label)
        assert OUTSIDE_UNIVERSE_SUFFIX.strip() in str(out_scope_row.label)
        assert out_scope_row.has_class("console-scope-picker-row-greyed")
        assert not in_scope_row.has_class("console-scope-picker-row-greyed")


# -- select-all-matching honors universe + confirmation --------------------------


@pytest.mark.asyncio
async def test_select_all_matching_honors_universe_and_requires_confirmation() -> None:
    app = ModalHarness()
    universe = frozenset(
        {(SOURCE_TYPE_MEDIA, "m1"), (SOURCE_TYPE_MEDIA, "m2")}
    )  # m3 excluded
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(
            app,
            pilot,
            media=[
                _media("m1", "Alpha"),
                _media("m2", "Beta"),
                _media("m3", "Gamma outside"),
            ],
            universe=universe,
        )
        await pilot.click(f"#{SELECT_ALL_BTN_ID}")
        await pilot.pause()

        confirm = app.screen.query_one(f"#{CONFIRM_STATIC_ID}", Static)
        assert confirm.display is True
        assert "Select all 2 matching?" == str(confirm.renderable)

        # Not yet applied.
        count = app.screen.query_one(f"#{COUNT_STATIC_ID}", Static)
        assert str(count.renderable).startswith("0 selected of")

        await pilot.click(f"#{CONFIRM_YES_BTN_ID}")
        await pilot.pause()

        count = app.screen.query_one(f"#{COUNT_STATIC_ID}", Static)
        assert str(count.renderable).startswith("2 selected of")

        await pilot.click(f"#{VIEW_SELECTED_BTN_ID}")
        await pilot.pause()
        labels = _row_labels(app)
        assert len(labels) == 2
        assert not any("Gamma" in label for label in labels)


@pytest.mark.asyncio
async def test_select_all_matching_cancel_applies_nothing() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(app, pilot, media=[_media("m1", "Alpha"), _media("m2", "Beta")])
        await pilot.click(f"#{SELECT_ALL_BTN_ID}")
        await pilot.pause()
        await pilot.click(f"#{CONFIRM_NO_BTN_ID}")
        await pilot.pause()

        confirm = app.screen.query_one(f"#{CONFIRM_STATIC_ID}", Static)
        assert confirm.display is False
        count = app.screen.query_one(f"#{COUNT_STATIC_ID}", Static)
        assert str(count.renderable).startswith("0 selected of")


@pytest.mark.asyncio
async def test_clear_shown_removes_only_currently_matching_selection() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(
            app,
            pilot,
            media=[_media("m1", "Alpha"), _media("m2", "Beta")],
            notes=[_note("n1", "Gamma")],
        )
        # Select all media+notes first.
        await pilot.click(f"#{SELECT_ALL_BTN_ID}")
        await pilot.pause()
        await pilot.click(f"#{CONFIRM_YES_BTN_ID}")
        await pilot.pause()
        count = app.screen.query_one(f"#{COUNT_STATIC_ID}", Static)
        assert str(count.renderable).startswith("3 selected of")

        # Narrow to Media tab only, then Clear shown -- notes selection
        # (not "shown" under this tab) must survive.
        await pilot.click(f"#{TAB_MEDIA_BTN_ID}")
        await _settle(pilot)
        await pilot.click(f"#{CLEAR_SHOWN_BTN_ID}")
        await pilot.pause()

        await pilot.click(f"#{VIEW_SELECTED_BTN_ID}")
        await pilot.pause()
        labels = _row_labels(app)
        assert len(labels) == 1
        assert "Gamma" in labels[0]


# -- zero-selection save calls on_save(None) / cancel never calls it ------------


@pytest.mark.asyncio
async def test_save_with_selection_calls_on_save_with_scope() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(app, pilot, media=[_media("m1", "Alpha")])
        container = app.screen.query_one(f"#{LIST_CONTAINER_ID}")
        checkbox = list(container.query(Checkbox))[0]
        checkbox.value = True
        await pilot.pause()
        await pilot.click(f"#{SAVE_BTN_ID}")
        await pilot.pause()

    assert len(app.saved) == 1
    scope = app.saved[0]
    assert isinstance(scope, RagScope)
    assert scope.items == (ScopeItem(SOURCE_TYPE_MEDIA, "m1"),)


@pytest.mark.asyncio
async def test_save_with_zero_selected_calls_on_save_none() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(app, pilot, media=[_media("m1", "Alpha")])
        await pilot.click(f"#{SAVE_BTN_ID}")
        await pilot.pause()

    assert app.saved == [None]


@pytest.mark.asyncio
async def test_clear_scope_button_calls_on_save_none_even_with_selection() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(app, pilot, media=[_media("m1", "Alpha")])
        container = app.screen.query_one(f"#{LIST_CONTAINER_ID}")
        checkbox = list(container.query(Checkbox))[0]
        checkbox.value = True
        await pilot.pause()
        await pilot.click(f"#{CLEAR_SCOPE_BTN_ID}")
        await pilot.pause()

    assert app.saved == [None]


@pytest.mark.asyncio
async def test_cancel_button_never_calls_on_save() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(app, pilot, media=[_media("m1", "Alpha")])
        container = app.screen.query_one(f"#{LIST_CONTAINER_ID}")
        checkbox = list(container.query(Checkbox))[0]
        checkbox.value = True
        await pilot.pause()
        await pilot.click(f"#{CANCEL_BTN_ID}")
        await pilot.pause()

    assert app.saved == []


@pytest.mark.asyncio
async def test_escape_never_calls_on_save() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(app, pilot, media=[_media("m1", "Alpha")])
        await pilot.press("escape")
        await pilot.pause()

    assert app.saved == []


# -- tag OR/AND semantics ---------------------------------------------------------


@pytest.mark.asyncio
async def test_tag_multi_select_is_or_and_ands_with_text_filter() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(
            app,
            pilot,
            media=[
                _media("m1", "Sales Q1", tags=["sales"]),
                _media("m2", "Marketing plan", tags=["marketing"]),
                _media("m3", "Sales Q2", tags=["sales", "finance"]),
            ],
            tags=[TagCount("sales", 2), TagCount("marketing", 1), TagCount("finance", 1)],
        )
        # Top tag chips loaded.
        chips = app.screen.query(".console-scope-picker-tag-chip")
        sales_chip = next(b for b in chips if "sales" in str(b.label))
        marketing_chip = next(b for b in chips if "marketing" in str(b.label))

        await pilot.click(sales_chip)
        await _settle(pilot)
        labels = _row_labels(app)
        assert len(labels) == 2
        assert all("Sales" in label for label in labels)

        # OR: adding "marketing" widens the match to include it too.
        await pilot.click(marketing_chip)
        await _settle(pilot)
        labels = _row_labels(app)
        assert len(labels) == 3

        # AND with text filter: narrows within the tag-OR set.
        filter_input = app.screen.query_one(f"#{TEXT_FILTER_ID}", Input)
        filter_input.value = "Q2"
        await _settle(pilot)
        labels = _row_labels(app)
        assert len(labels) == 1
        assert "Sales Q2" in labels[0]


@pytest.mark.asyncio
async def test_tag_union_across_types() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(
            app,
            pilot,
            media=[_media("m1", "Sales media", tags=["sales"])],
            notes=[_note("n1", "Sales note", tags=["sales"])],
            tags=[TagCount("sales", 2)],
        )
        chips = app.screen.query(".console-scope-picker-tag-chip")
        sales_chip = next(b for b in chips if "sales" in str(b.label))
        await pilot.click(sales_chip)
        await _settle(pilot)
        labels = _row_labels(app)
        assert len(labels) == 2
        assert any("Sales media" in label for label in labels)
        assert any("Sales note" in label for label in labels)


# -- sort orders --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sort_by_title_orders_rows_alphabetically() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(
            app,
            pilot,
            media=[
                _media("m1", "Zebra report", updated_at="2026-03-01"),
                _media("m2", "Alpha report", updated_at="2026-01-01"),
                _media("m3", "Mango report", updated_at="2026-02-01"),
            ],
        )
        sort_select = app.screen.query_one(f"#{SORT_SELECT_ID}", Select)
        sort_select.value = SORT_TITLE
        await _settle(pilot)
        labels = _row_labels(app)
        assert [label.split(" ", 1)[1] for label in labels] == [
            "Alpha report",
            "Mango report",
            "Zebra report",
        ]


@pytest.mark.asyncio
async def test_sort_recent_orders_by_updated_at_descending() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(
            app,
            pilot,
            media=[
                _media("m1", "Oldest", updated_at="2026-01-01"),
                _media("m2", "Newest", updated_at="2026-03-01"),
                _media("m3", "Middle", updated_at="2026-02-01"),
            ],
        )
        labels = _row_labels(app)  # default sort is "recent"
        assert [label.split(" ", 1)[1] for label in labels] == ["Newest", "Middle", "Oldest"]


@pytest.mark.asyncio
async def test_sort_by_type_groups_media_before_notes() -> None:
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await _open(
            app,
            pilot,
            media=[_media("m1", "Zeta media")],
            notes=[_note("n1", "Alpha note")],
        )
        sort_select = app.screen.query_one(f"#{SORT_SELECT_ID}", Select)
        sort_select.value = SORT_TYPE
        await _settle(pilot)
        labels = _row_labels(app)
        # "media" < "note" lexicographically -- media rows first regardless
        # of title, since type is the primary sort key.
        assert "Zeta media" in labels[0]
        assert "Alpha note" in labels[1]


# -- target_label escaping + universe-aware pagination ---------------------------


@pytest.mark.asyncio
async def test_target_label_with_markup_characters_mounts_without_error() -> None:
    """Verify that target_label containing markup chars like '[/bad]' doesn't
    raise MarkupError and renders the literal text instead."""
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        # These markup chars would cause MarkupError if not escaped.
        await _open(app, pilot, media=[], target_label="workspace [/bad]")
        header = app.screen.query(".console-modal-header").first(Static)
        header_text = str(header.renderable)
        # Header should contain escaped markup (e.g., \\[/bad]), not raise an error.
        # The important thing is that it mounts without MarkupError.
        assert "workspace" in header_text
        # Verify escaping happened: backslash before the bracket.
        assert "\\[/bad]" in header_text


@pytest.mark.asyncio
async def test_pagination_respects_universe_size_not_raw_total() -> None:
    """Verify pagination uses len(matching_keys) (universe-intersected) instead
    of raw lister totals. A universe smaller than one page should show 1 page."""
    app = ModalHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        # Create 50-item media store, but universe restricts to just 3 items.
        media_items = [_media(f"m{i:02d}", f"Item {i}") for i in range(50)]
        universe = frozenset(
            {(SOURCE_TYPE_MEDIA, "m00"), (SOURCE_TYPE_MEDIA, "m01"), (SOURCE_TYPE_MEDIA, "m02")}
        )
        await _open(app, pilot, media=media_items, universe=universe, page_size=10)
        # With universe-aware pagination, "1 of 1" page (3 items < 10 page_size).
        label = app.screen.query_one(f"#{PAGE_LABEL_ID}", Static)
        page_text = str(label.renderable)
        assert "Page 1 of 1" == page_text
        # Prev/Next buttons must both be disabled (single page).
        next_btn = app.screen.query_one(f"#{PAGE_NEXT_ID}", Button)
        assert next_btn.disabled is True


# -- CSS pinned in source + bundle ---------------------------------------------


def test_scope_picker_css_pinned_in_source_and_bundle() -> None:
    """The scope-picker ids/classes must be styled in BOTH the module source
    (``_agentic_terminal.tcss``) and the generated bundle
    (``tldw_cli_modular.tcss``) -- proves ``build_css.py`` was re-run after
    the source edit (dual-file CSS-parity discipline used by the sibling
    prompt-picker test)."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        for selector in (
            "ConsoleScopePickerModal {",
            f"#{LIST_CONTAINER_ID} {{",
            ".console-scope-picker-row {",
            ".console-scope-picker-row-greyed {",
            ".console-scope-picker-tab-active {",
            ".console-scope-picker-view-active {",
            ".console-scope-picker-tag-chip-active {",
        ):
            assert selector in text, f"missing CSS for {selector!r}"
