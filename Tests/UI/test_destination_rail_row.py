"""``fit_rail_row_label`` and ``DestinationRailRowButton`` (Roleplay frame B0).

Rail text width is pane width minus 4 (round border + row padding): 24 / 31 /
35 cells at 120 / 160 / >=180 columns (spec section 1.4.1). The fallback order
is spec section 1.4.3: title + count + key -> drop the key -> short count ->
short title -> short title + short count -> ellipsis. The canonical noun
outlives the key hint, and the count is never clipped. The two-cell row
prefix counts toward the width.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from rich.cells import cell_len
from textual.app import ComposeResult
from textual.containers import Vertical

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Widgets import glyph_fallback
from tldw_chatbook.Widgets.adaptive_pane_shell import (
    DestinationRailRow,
    DestinationRailRowButton,
    fit_rail_row_label,
    rail_row_content,
)

CHARACTERS = DestinationRailRow(row_id="characters", title="Characters", count="(28)", key="c")
PERSONAS = DestinationRailRow(row_id="personas", title="Personas", count="(4)", key="p")
LORE = DestinationRailRow(
    row_id="lore",
    title="Lore books",
    short_title="Lore",
    count="(3 · 2 on)",
    short_count="(3)",
    key="l",
)
DICTIONARIES = DestinationRailRow(
    row_id="dictionaries",
    title="Chat dictionaries",
    short_title="Dictionaries",
    count="(3 · 2 on)",
    short_count="(3)",
    key="d",
)
ROWS = (CHARACTERS, PERSONAS, LORE, DICTIONARIES)

#: A row whose short title is much shorter than its title, so every one of
#: the six steps is reachable at some width.
FALLBACK = DestinationRailRow(
    row_id="fallback",
    title="Chat dictionaries",
    short_title="Dicts",
    count="(3 · 2 on)",
    short_count="(3)",
    key="d",
)

WIDE = (
    DestinationRailRow(row_id="cjk", title="漢字のキャラクター名", count="(12)", key="c"),
    DestinationRailRow(row_id="emoji", title="🎭 Masks 🎭 Theatre", count="(3)", key="m"),
    DestinationRailRow(row_id="zero-width", title="Zoë\u200b Zero\u200bwidth", count="(1)", key="z"),
)


@pytest.mark.parametrize(
    ("row", "width", "current", "expected"),
    [
        (CHARACTERS, 24, True, "▸ Characters (28)  c"),
        (PERSONAS, 24, False, "  Personas (4)  p"),
        (LORE, 24, False, "  Lore books (3 · 2 on)"),
        (DICTIONARIES, 24, False, "  Chat dictionaries (3)"),
        (LORE, 31, False, "  Lore books (3 · 2 on)  l"),
        (DICTIONARIES, 31, False, "  Chat dictionaries (3 · 2 on)"),
        (CHARACTERS, 35, False, "  Characters (28)  c"),
        (PERSONAS, 35, False, "  Personas (4)  p"),
        (LORE, 35, False, "  Lore books (3 · 2 on)  l"),
        (DICTIONARIES, 35, False, "  Chat dictionaries (3 · 2 on)  d"),
    ],
)
def test_spec_rows_fit_the_bordered_rail_at_24_31_35(row, width, current, expected) -> None:
    label = fit_rail_row_label(row, width, current=current)
    assert label.plain == expected
    assert cell_len(label.plain) <= width


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (33, "  Chat dictionaries (3 · 2 on)  d"),  # 1. title + count + key
        (32, "  Chat dictionaries (3 · 2 on)"),  # 2. the key drops first
        (29, "  Chat dictionaries (3)"),  # 3. short count
        (22, "  Dicts (3 · 2 on)"),  # 4. short title
        (17, "  Dicts (3)"),  # 5. short title + short count
        (10, "  Dic… (3)"),  # 6. ellipsis, count whole
    ],
)
def test_fallback_order_drops_the_key_first_and_never_the_count(width, expected) -> None:
    assert fit_rail_row_label(FALLBACK, width).plain == expected


@pytest.mark.parametrize(
    ("row", "width", "current", "expected"),
    [
        (LORE, 20, False, "  Lore books (3)"),
        (DICTIONARIES, 20, False, "  Dictionaries (3)"),
        (CHARACTERS, 16, True, "▸ Characte… (28)"),
        (DICTIONARIES, 16, False, "  Dictionar… (3)"),
    ],
)
def test_narrow_rails_keep_the_noun_before_the_detail(row, width, current, expected) -> None:
    assert fit_rail_row_label(row, width, current=current).plain == expected


@pytest.mark.parametrize("width", range(1, 41))
@pytest.mark.parametrize("row", (*ROWS, FALLBACK), ids=lambda row: row.row_id)
def test_the_count_is_never_clipped(row, width) -> None:
    label = fit_rail_row_label(row, width)
    assert label.count in {row.count, row.short_count or row.count}
    # Independent of the label's own parts (controller ruling C9): the tails
    # a row may legitimately end with, written from the row's inputs.
    allowed_tails = {row.count, row.short_count or row.count}
    allowed_tails |= {f"{count}  {row.key}" for count in allowed_tails}
    assert any(label.plain.endswith(tail) for tail in allowed_tails), label.plain
    if row is CHARACTERS:
        assert label.plain.endswith("(28)") or label.plain.endswith("(28)  c")
    for current in (True, False):
        assert (
            rail_row_content(row, width, current=current).plain
            == fit_rail_row_label(row, width, current=current).plain
        )


def test_a_loading_count_never_paints_the_key_hint() -> None:
    """RC-11: a hint painted while loading would vanish on every arrival."""
    loading = DestinationRailRow(
        row_id="lore",
        title="Lore books",
        short_title="Lore",
        count="(…)",
        key="l",
        count_loading=True,
    )
    for width in (0, 16, 24, 31, 35, 60):
        assert fit_rail_row_label(loading, width).key == ""
    assert fit_rail_row_label(loading, 24).plain == "  Lore books (…)"


def test_an_unknown_width_renders_the_full_label() -> None:
    assert fit_rail_row_label(DICTIONARIES, 0).plain == "  Chat dictionaries (3 · 2 on)  d"


def test_the_current_marker_follows_ascii_glyph_mode() -> None:
    glyph_fallback.set_ascii_glyph_mode(True)
    try:
        assert fit_rail_row_label(CHARACTERS, 24, current=True).plain == "> Characters (28)  c"
    finally:
        glyph_fallback.set_ascii_glyph_mode(False)


@pytest.mark.parametrize("width", range(1, 41))
@pytest.mark.parametrize("row", WIDE, ids=lambda row: row.row_id)
def test_wide_and_zero_width_titles_fit_by_terminal_cells(row, width) -> None:
    label = fit_rail_row_label(row, width)
    if width >= 2 + cell_len(row.short_count or row.count):
        assert cell_len(label.plain) <= width, (label.plain, width)
    assert label.count in {row.count, row.short_count or row.count}


class _RailApp(ConsolidatedCSSApp):
    """Real CSS so the row's ``w-full``/``h-1`` utilities apply."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, rows, current_id: str | None = None) -> None:
        super().__init__()
        self.rows = rows
        self.current_id = current_id

    def compose(self) -> ComposeResult:
        with Vertical(id="rail"):
            for row in self.rows:
                yield DestinationRailRowButton(
                    row,
                    current=row.row_id == self.current_id,
                    id=f"rail-row-{row.row_id}",
                )


async def test_the_row_button_refits_its_label_when_its_width_changes() -> None:
    app = _RailApp(ROWS, current_id="characters")
    async with app.run_test(size=(60, 12)) as pilot:
        rail = app.query_one("#rail")
        painted = {}
        for width in (35, 31, 24):
            rail.styles.width = width
            await pilot.pause()
            for button in app.query(DestinationRailRowButton):
                assert button.content_region.width == width
                expected = fit_rail_row_label(
                    button.rail_row, width, current=button.is_current
                )
                assert button.label.plain == expected.plain
            painted[width] = app.query_one(
                "#rail-row-dictionaries", DestinationRailRowButton
            ).label.plain
        assert painted == {
            35: "  Chat dictionaries (3 · 2 on)  d",
            31: "  Chat dictionaries (3 · 2 on)",
            24: "  Chat dictionaries (3)",
        }


async def test_the_row_button_renders_untrusted_text_literally() -> None:
    """R33: a ``[/]`` name would raise MarkupError; ``[@click=…]`` would act.

    The ``sync_row`` and the second refit force real label assignments: at
    width 40 the first refit has the same plain text as the label built at
    construction, so it assigns nothing and could not catch a markup parse.
    """
    row = DestinationRailRow(row_id="hostile", title="[/] [@click=app.quit]Boom", count="(1)")
    app = _RailApp((row,))
    async with app.run_test(size=(60, 8)) as pilot:
        app.query_one("#rail").styles.width = 40
        await pilot.pause()
        button = app.query_one(DestinationRailRowButton)
        assert "[/] [@click=app.quit]Boom" in button.label.plain
        button.sync_row(
            DestinationRailRow(row_id="hostile", title="[b]x [@click=app.quit]Boom", count="(2)"),
            current=True,
        )
        await pilot.pause()
        assert button.label.plain == "▸ [b]x [@click=app.quit]Boom (2)"
        app.query_one("#rail").styles.width = 20
        await pilot.pause()
        assert "[b]x" in button.label.plain
        assert not any("@click" in str(span.style) for span in button.label.spans)


async def test_a_style_only_change_repaints_the_row() -> None:
    """``Content.__eq__`` and the ``label`` reactive compare plain text only.

    A count that starts loading with unchanged text changes only the count's
    style (dim); a plain-text comparison alone would keep the stale style.
    """
    known = DestinationRailRow(row_id="chats", title="Chats", count="(3)")
    app = _RailApp((known,))
    async with app.run_test(size=(60, 8)) as pilot:
        app.query_one("#rail").styles.width = 24
        await pilot.pause()
        button = app.query_one(DestinationRailRowButton)
        assert not any("dim" in str(span.style) for span in button.label.spans)
        button.sync_row(replace(known, count_loading=True), current=False)
        await pilot.pause()
        assert button.label.plain == "  Chats (3)"
        assert any("dim" in str(span.style) for span in button.label.spans)


async def test_sync_row_patches_in_place_and_skips_an_unchanged_row() -> None:
    app = _RailApp((LORE,))
    async with app.run_test(size=(60, 8)) as pilot:
        app.query_one("#rail").styles.width = 24
        await pilot.pause()
        button = app.query_one(DestinationRailRowButton)
        refreshes: list[tuple] = []
        original = button.refresh

        def counting_refresh(*args, **kwargs):
            refreshes.append(args)
            return original(*args, **kwargs)

        button.refresh = counting_refresh
        button.sync_row(LORE, current=False)
        assert refreshes == []
        button.refresh = original
        loading = DestinationRailRow(
            row_id="lore",
            title="Lore books",
            short_title="Lore",
            count="(…)",
            key="l",
            count_loading=True,
        )
        button.sync_row(loading, current=False)
        await pilot.pause()
        assert app.query_one(DestinationRailRowButton) is button
        assert button.label.plain == "  Lore books (…)"
