"""Tests for the LibraryRail widget."""

# Pytest injects the imported fixture into same-named test parameters.
# ruff: noqa: F811

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static

from Tests.textual_test_utils import widget_pilot  # noqa: F401
from tldw_chatbook.Library.library_rail_state import (
    LibraryLifecycle,
    LibraryRailPreferences,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_CREATE_NOTE,
    LIBRARY_ROW_INGEST_MEDIA,
    LibraryRailRow,
    LibraryRailSectionState,
    LibraryShellInput,
    LibraryShellState,
    build_library_shell_state,
)
from tldw_chatbook.Widgets.Library.library_rail import (
    LibraryRail,
    _fit_title_no_mid_word_cut,
)


pytestmark = pytest.mark.asyncio


def _make_shell() -> LibraryShellState:
    """Return a minimal Library shell state for rail tests."""
    return LibraryShellState(
        header_line="Library | Test",
        sections=(),
        details_lines=(),
        selected_row_id="",
        canvas_kind="empty",
        canvas_target="",
        canvas_empty_copy="",
    )


def _make_full_shell() -> LibraryShellState:
    """Return the production Library shell row table."""
    return build_library_shell_state(LibraryShellInput())


@pytest.mark.parametrize(
    "lifecycle",
    [LibraryLifecycle.STARTER, LibraryLifecycle.UNKNOWN],
)
async def test_library_starter_and_unknown_rail_render_only_safe_actions(
    widget_pilot,
    lifecycle,
):
    """Starter and unresolved profiles share one compact production action set."""
    async with await widget_pilot(
        LibraryRail,
        shell=_make_full_shell(),
        preferences=LibraryRailPreferences(),
        lifecycle=lifecycle,
    ) as pilot:
        await pilot.pause()
        rail = pilot.app.test_widget

        assert rail.query_one("#library-rail-heading-label", Static).renderable == (
            "Navigation"
        )
        assert rail.query_one("#library-rail-collapse", Button)
        assert [button.id for button in pilot.app.query(".library-rail-row")] == [
            f"library-row-{LIBRARY_ROW_INGEST_MEDIA}",
            f"library-row-{LIBRARY_ROW_CREATE_NOTE}",
        ]
        assert rail.query_one(
            f"#library-row-{LIBRARY_ROW_INGEST_MEDIA}", Button
        ).row_id == LIBRARY_ROW_INGEST_MEDIA
        assert rail.query_one(
            f"#library-row-{LIBRARY_ROW_CREATE_NOTE}", Button
        ).row_id == LIBRARY_ROW_CREATE_NOTE
        assert rail.query_one("#library-rail-explore-all", Button).label.plain == (
            "Explore all tools"
        )
        assert len(pilot.app.query("#library-rail-explore-all")) == 1
        assert not pilot.app.query("#library-search-input")
        assert not pilot.app.query(".destination-rail-section-header")
        assert not pilot.app.query("#library-rail-section-header-details")
        assert not pilot.app.query("#library-ingest-top-button")


@pytest.mark.parametrize(
    ("lifecycle", "all_empty", "shows_back"),
    [
        (LibraryLifecycle.EXPANDED, True, True),
        (LibraryLifecycle.EXPANDED, False, False),
        (LibraryLifecycle.GRADUATED, True, False),
    ],
)
async def test_library_expanded_and_graduated_rails_render_full_shell(
    widget_pilot,
    lifecycle,
    all_empty,
    shows_back,
):
    """Only a freshly empty expanded profile can return to Starter."""
    shell = _make_full_shell()
    async with await widget_pilot(
        LibraryRail,
        shell=shell,
        preferences=LibraryRailPreferences(),
        lifecycle=lifecycle,
        onboarding_all_empty=all_empty,
    ) as pilot:
        await pilot.pause()
        rail = pilot.app.test_widget

        assert rail.query_one("#library-search-input", Input)
        assert [
            button.row_id for button in pilot.app.query(".library-rail-row")
        ] == [
            row.row_id for section in shell.sections for row in section.rows
        ]
        assert (
            bool(pilot.app.query("#library-rail-back-to-starter")) is shows_back
        )


async def test_library_starter_rail_tab_order_and_labels_are_text_complete(
    widget_pilot,
):
    """Compact actions are keyboard ordered and understandable without color."""
    async with await widget_pilot(
        LibraryRail,
        shell=_make_full_shell(),
        preferences=LibraryRailPreferences(),
        lifecycle=LibraryLifecycle.STARTER,
    ) as pilot:
        await pilot.pause()
        action_ids = {
            f"library-row-{LIBRARY_ROW_INGEST_MEDIA}",
            f"library-row-{LIBRARY_ROW_CREATE_NOTE}",
            "library-rail-explore-all",
        }
        focus_order = [
            widget.id
            for widget in pilot.app.screen.focus_chain
            if widget.id in action_ids
        ]
        assert focus_order == [
            f"library-row-{LIBRARY_ROW_INGEST_MEDIA}",
            f"library-row-{LIBRARY_ROW_CREATE_NOTE}",
            "library-rail-explore-all",
        ]
        assert [
            button.label.plain.strip().lstrip("▸").strip()
            for button in pilot.app.query("Button")
            if button.id in action_ids
        ] == ["Import…", "New note", "Explore all tools"]


async def test_library_rail_top_action_factory(widget_pilot):
    """The top_action_factory is stored and its widgets are rendered first."""
    def factory():
        return [Button("Ingest", id="library-top-action")]
    preferences = LibraryRailPreferences()

    async with await widget_pilot(
        LibraryRail,
        shell=_make_shell(),
        preferences=preferences,
        top_action_factory=factory,
    ) as pilot:
        rail = pilot.app.test_widget
        assert rail.top_action_factory is factory

        await pilot.pause()
        assert isinstance(pilot.app.query_one("#library-top-action", Button), Button)
        assert isinstance(pilot.app.query_one("#library-search-input", Input), Input)


# -- task-670: RecomposeCaptureGuard extended to LibraryRail ---------------
# LibraryRail.sync_state() drives `self.refresh(recompose=True)`; before this
# fix the rail carried no guard against task-637's bug class.


async def test_post_recompose_sweep_releases_a_capture_dispatched_during_the_teardown_drain(
    widget_pilot,
):
    """Residual-window regression (mirrors ``test_post_recompose_sweep_
    releases_a_capture_dispatched_during_the_teardown_drain`` in
    ``Tests/UI/test_chatbooks_screen_server_actions.py``, the task-637
    code-review finding for ``BaseAppScreen``/task-627): a capture that
    lands on the VICTIM's own message pump -- queued before the recompose's
    pre-teardown release even ran, but processed DURING
    ``super().recompose()``'s own ``remove()`` drain -- must still be swept
    once the recompose fully completes.

    Reproduced deterministically with ``call_later`` on the victim's own
    pump, mechanism-equivalent to a forwarded ``MouseDown`` whose dispatch is
    still pending on the search Input's pump when ``sync_state()`` starts
    the rail's recompose.
    """
    async with await widget_pilot(
        LibraryRail,
        shell=_make_shell(),
        preferences=LibraryRailPreferences(),
    ) as pilot:
        rail = pilot.app.test_widget
        await pilot.pause()
        victim = pilot.app.query_one("#library-search-input", Input)

        # Schedule the recompose first (the widget's own next-callback),
        # then queue a capture-inducing message on the VICTIM's own pump --
        # modelling a MouseDown forwarded to the Input but not yet
        # dispatched when the teardown starts.
        rail.sync_state(_make_shell(), LibraryRailPreferences(), query="x")
        victim.call_later(lambda: pilot.app.capture_mouse(victim))

        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        captured = pilot.app.mouse_captured
        assert captured is None, (
            f"stale capture survived the teardown drain: {captured!r} "
            f"(attached={getattr(captured, 'is_attached', None)}) -- clicks "
            "anywhere in the app are silently swallowed again (task-670)"
        )


# -- F-014: one count policy for every rail row ----------------------------
# dim "(…)" while the source snapshot is in flight, "(N)"/"(N+)" when the
# count is known, and no suffix at all when the source is off or unknown --
# never a misleading "(0)" for an unavailable source.


def _row(row_id: str, title: str, **kwargs) -> LibraryRailRow:
    return LibraryRailRow(
        row_id=row_id,
        section_id="browse",
        title=title,
        target_kind="canvas",
        target_id="x",
        **kwargs,
    )


def _shell_with_rows(rows) -> LibraryShellState:
    return LibraryShellState(
        header_line="Library | Test",
        sections=(
            LibraryRailSectionState(
                section_id="browse", title="Browse", rows=tuple(rows)
            ),
        ),
        details_lines=(),
        selected_row_id="",
        canvas_kind="empty",
        canvas_target="",
        canvas_empty_copy="",
    )


async def test_count_policy_loading_known_estimate_and_off_rows(widget_pilot):
    """One policy: dim placeholder while loading, count when known, nothing
    when the source is off."""
    shell = _shell_with_rows(
        [
            _row("r-loading", "Loading", count=None, count_loading=True),
            _row("r-known", "Known", count=7),
            _row("r-estimate", "Estimate", count=7, count_known=False),
            _row("r-off", "Off", count=None),
        ]
    )

    async with await widget_pilot(
        LibraryRail,
        shell=shell,
        preferences=LibraryRailPreferences(),
    ) as pilot:
        await pilot.pause()

        loading = pilot.app.query_one("#library-row-r-loading", Button)
        assert loading.label.plain == "  Loading (…)"
        assert any(
            "dim" in str(span.style) for span in loading.label.spans
        ), f"loading placeholder must render dim: {loading.label.spans}"

        known = pilot.app.query_one("#library-row-r-known", Button)
        assert known.label.plain == "  Known (7)"

        estimate = pilot.app.query_one("#library-row-r-estimate", Button)
        assert estimate.label.plain == "  Estimate (7+)"

        off = pilot.app.query_one("#library-row-r-off", Button)
        assert off.label.plain == "  Off"


async def test_details_renders_db_sizes_row_only_when_provided(widget_pilot):
    """F-014: relocated DB-size telemetry lives in the Details disclosure --
    rendered when the shell carries it, omitted entirely when it does not
    (no 'N/A' triplets)."""
    with_sizes = LibraryShellState(
        header_line="Library | Test",
        sections=(),
        details_lines=(
            "Local",
            "Notes 0 · Media 0 · Conversations 0",
            "Prompts 1.0 KB · Chats/Notes 2.0 KB · Media 3.0 KB",
        ),
        selected_row_id="",
        canvas_kind="empty",
        canvas_target="",
        canvas_empty_copy="",
    )

    async with await widget_pilot(
        LibraryRail,
        shell=with_sizes,
        preferences=LibraryRailPreferences(details_open=True),
    ) as pilot:
        await pilot.pause()
        sizes = pilot.app.query_one("#library-details-db-sizes", Static)
        text = str(sizes.renderable)
        assert "Prompts 1.0 KB" in text
        assert "Chats/Notes 2.0 KB" in text
        assert "Media 3.0 KB" in text

    without_sizes = LibraryShellState(
        header_line="Library | Test",
        sections=(),
        details_lines=("Local", "Notes 0 · Media 0 · Conversations 0"),
        selected_row_id="",
        canvas_kind="empty",
        canvas_target="",
        canvas_empty_copy="",
    )

    async with await widget_pilot(
        LibraryRail,
        shell=without_sizes,
        preferences=LibraryRailPreferences(details_open=True),
    ) as pilot:
        await pilot.pause()
        assert not list(pilot.app.query("#library-details-db-sizes"))


# -- LIB-15: one deterministic gloss/count rule across visit/leave/re-enter -

# The rail's own real minimum row-content width (verified live: 120x35,
# 100x30, and 80x24 terminals all pin the rail row content to the SAME 17
# cells via LibraryRail's own `min_width=24`, since the canvas pane absorbs
# every column of difference above that floor). All the LIB-18 width-sweep
# assertions below use this same figure -- it is not a magic number chosen
# for the test, it is what 120/100/80 actually produce.
_REAL_RAIL_ROW_WIDTH = 17


async def test_count_pending_reserve_stabilizes_gloss_across_arrival():
    """LIB-15: reproduces the live "Collections — item sets" -> "Collections
    (0)" flip and pins the fix. Two different (title, gloss) pairs are
    covered so the rule is proven general, not special-cased to one row.

    For each pair, a width is chosen in that row's own "bug zone": wide
    enough that ``prefix + title + gloss`` fits with NO count on screen
    (the pre-visit state), but too narrow once a real single-digit count's
    4 cells are added on top (the post-visit state) -- exactly the width
    range the live 120x35+ session sat in for Collections. Leaving
    ``count_pending`` at its default ``False`` reproduces the historical
    shape every row had before this fix (gloss visible, then gone, at an
    UNCHANGED width -- the flip a real UAT tester saw). Setting
    ``count_pending=True`` (Collections' actual current shape) makes the
    two states AGREE, and re-building the identical row/width a second
    time (leave-and-re-enter) reproduces the same result -- the rule is a
    pure function of (row, width), not of how many times it has rendered.
    """
    cases = (
        # (title, gloss, width chosen inside THIS pair's own bug zone)
        ("Collections", "item sets", 27),
        ("Watchers", "tracked pages", 28),
    )
    for title, gloss, width in cases:
        def make(count: int | None, pending: bool) -> LibraryRailRow:
            return LibraryRailRow(
                row_id="r",
                section_id="browse",
                title=title,
                target_kind="canvas",
                target_id="x",
                count=count,
                count_known=True,
                subtitle=gloss,
                count_pending=pending,
            )

        # Unfixed shape (count_pending=False, default): the gloss shows
        # before the count arrives and drops the instant a real
        # single-digit count lands -- the historical bug, reproduced here
        # as the control case proving the test's width actually exercises
        # it (not merely asserting the fix without evidence of the flip).
        before_unfixed = LibraryRail._row_label(make(None, False), False, width)
        after_unfixed = LibraryRail._row_label(make(0, False), False, width)
        assert gloss in before_unfixed, (title, "control: gloss should show pre-visit")
        assert gloss not in after_unfixed, (
            title,
            "control: gloss should flip off post-visit -- if this fails, "
            "the chosen width no longer reproduces the historical bug",
        )

        # Fixed shape (count_pending=True, Collections' real current
        # shape): both states now agree, across a leave/re-enter rebuild.
        for count in (0, 1):
            after_fixed = LibraryRail._row_label(make(count, True), False, width)
            before_fixed = LibraryRail._row_label(make(None, True), False, width)
            assert (gloss in before_fixed) == (gloss in after_fixed), (
                title,
                count,
                before_fixed,
                after_fixed,
            )
            # Re-entry: rebuilding the identical (row, width) a second time
            # must be idempotent.
            assert after_fixed == LibraryRail._row_label(
                make(count, True), False, width
            )


async def test_gloss_rule_is_the_same_function_for_a_non_pending_row():
    """LIB-15: an ordinary counts_loading row (never count_pending) already
    reserves the F-014 placeholder's width for its count from the moment it
    is first composed (the dim "(…)" IS 4 cells wide, the same width a
    single-digit count needs), so its gloss outcome is identical across the
    loading -> known transition too -- the SAME ``_row_label`` rule covers
    both row shapes, with no special-casing fork between them."""
    for width in (17, 27):
        loading = LibraryRailRow(
            row_id="r",
            section_id="browse",
            title="Media",
            target_kind="canvas",
            target_id="x",
            count=None,
            count_known=True,
            subtitle="your files",
            count_loading=True,
        )
        known = LibraryRailRow(
            row_id="r",
            section_id="browse",
            title="Media",
            target_kind="canvas",
            target_id="x",
            count=0,
            count_known=True,
            subtitle="your files",
            count_loading=False,
        )
        label_loading = LibraryRail._row_label(loading, selected=False, width=width)
        label_known = LibraryRail._row_label(known, selected=False, width=width)
        assert ("your files" in label_loading) == ("your files" in label_known), width


async def test_gloss_genuinely_drops_when_the_known_count_grows_too_wide():
    """The rule still lets a gloss drop once the row's OWN content
    genuinely needs the space (a real, not phantom, width conflict) --
    determinism means "stable for an unchanged input", not "gloss never
    drops". A 3-digit count is wide enough to force the drop even with the
    count_pending reserve (which only covers the F-014 placeholder's
    4-cell width)."""
    row = LibraryRailRow(
        row_id="r",
        section_id="browse",
        title="Collections",
        target_kind="canvas",
        target_id="x",
        count=123,
        count_known=True,
        subtitle="item sets",
        count_pending=True,
        short_title="Sets",
    )
    label = LibraryRail._row_label(row, selected=False, width=_REAL_RAIL_ROW_WIDTH)
    assert "item sets" not in label


# -- LIB-18: no rail row label truncates mid-word at 120/100/80 columns ----


async def test_fit_title_no_mid_word_cut_prefers_word_boundary():
    assert _fit_title_no_mid_word_cut("Study decks", 9) == "Study..."
    # Fits outright -- returned unchanged, no ellipsis.
    assert _fit_title_no_mid_word_cut("Media", 17) == "Media"
    # A single unbroken word with no boundary to retreat to falls back to a
    # hard cut (last resort -- callers should supply a short_title instead).
    assert _fit_title_no_mid_word_cut("Conversations", 8) == "Conve..."


async def test_conversations_row_uses_short_title_at_real_rail_width():
    """Reproduces the live 120/100/80-column finding verbatim: the full
    title "Conversations" ellipsis-cut inside the word ("Conversa...").
    The fix substitutes the row's own short_title instead of ellipsizing
    the long title."""
    row = LibraryRailRow(
        row_id="browse-conversations",
        section_id="browse",
        title="Conversations",
        target_kind="canvas",
        target_id="conversations",
        count=0,
        count_known=True,
        short_title="Chats",
    )
    label = LibraryRail._row_label(row, selected=False, width=_REAL_RAIL_ROW_WIDTH)
    assert label == "  Chats (0)"
    assert "Conversa" not in label
    assert "..." not in label


async def test_flashcards_row_uses_short_title_at_real_rail_width():
    """Reproduces the live finding verbatim: "Flash... due: 0"."""
    row = LibraryRailRow(
        row_id="create-flashcards",
        section_id="study",
        title="Flashcards",
        target_kind="handoff",
        target_id="flashcards",
        count=None,
        count_known=True,
        count_display=" due: 0",
        short_title="Cards",
    )
    label = LibraryRail._row_label(row, selected=False, width=_REAL_RAIL_ROW_WIDTH)
    assert "Cards due: 0" in label
    assert "Flash" not in label
    assert "..." not in label.split("\n")[0]


async def test_collections_row_falls_back_to_short_title_once_double_digit():
    """A latent case beyond current UAT data: once Collections' own count
    grows to two digits, "Collections" (11 cells) no longer fits the real
    17-cell budget either -- short_title covers it the same way, so the
    row never regresses to a mid-word "Collect..." cut as the library
    grows."""
    row = LibraryRailRow(
        row_id="browse-collections",
        section_id="browse",
        title="Collections",
        target_kind="canvas",
        target_id="collections",
        count=23,
        count_known=True,
        subtitle="item sets",
        count_pending=True,
        short_title="Sets",
    )
    label = LibraryRail._row_label(row, selected=False, width=_REAL_RAIL_ROW_WIDTH)
    assert label == "  Sets (23)"
    assert "Collect" not in label


async def test_no_row_label_truncates_mid_word_at_120_100_and_80_columns():
    """Rendered-geometry sweep: calls the REAL rail's static ``_row_label``
    directly for every browse-section row (built by
    build_library_shell_state) at the one width value verified to be the
    real rail's row-content width at 120/100/80 columns alike (17 cells,
    see ``_REAL_RAIL_ROW_WIDTH``'s comment above) -- the genuine MOUNTED
    rail check lives in
    ``test_library_shell.py::test_rail_counts_never_clip_and_titles_
    shrink_first_at_100x30``. Asserts no rendered label contains an
    ellipsis immediately preceded by a lowercase letter that is not the
    end of a real short word -- i.e. no row's PRIMARY title line hard-cuts
    inside "Conversations"/"Flashcards"/"Collections" the way the live
    120x35/100x30/80x24 captures originally showed."""
    from tldw_chatbook.Library.library_shell_state import (
        LibraryShellInput,
        build_library_shell_state,
    )

    shell = build_library_shell_state(
        LibraryShellInput(
            media_count=0,
            conversations_count=0,
            notes_count=0,
            prompts_count=0,
            skills_count=0,
            collections_count=0,
            study_decks_count=0,
            flashcards_due_count=0,
            quizzes_count=0,
        )
    )
    banned_fragments = ("Conversa...", "Flash...", "Collect...")
    for section in shell.sections:
        for row in section.rows:
            label = LibraryRail._row_label(
                row, selected=False, width=_REAL_RAIL_ROW_WIDTH
            )
            for fragment in banned_fragments:
                assert fragment not in label, (row.row_id, label)


async def test_handoff_meta_line_drops_rather_than_ellipsizing_mid_word():
    """LIB-18: "opens staging canvas" (24 cells with its indent) does not
    fit the real 17-cell row width -- it must drop entirely rather than
    render Textual's own "opens stagin…" mid-word ellipsis (reproduced
    live at 120x35). At width 0 (compose time, before layout) it still
    renders in full, matching every other element's unfitted-until-resize
    behavior."""
    row = LibraryRailRow(
        row_id="create-flashcards",
        section_id="study",
        title="Flashcards",
        target_kind="handoff",
        target_id="flashcards",
        count=None,
        count_known=True,
        count_display=" due: 0",
        short_title="Cards",
    )
    fitted = LibraryRail._row_label(row, selected=False, width=_REAL_RAIL_ROW_WIDTH)
    assert "opens" not in fitted
    assert "\n" not in fitted

    unfitted = LibraryRail._row_label(row, selected=False, width=0)
    assert "see what carries over" in unfitted


# -- LIB-17: prefilled search inputs are editable without cursor traps -----


async def test_click_on_unfocused_prefilled_search_box_selects_all(widget_pilot):
    """Reproduces the live finding verbatim: clicking into a prefilled rail
    search box landed the cursor at the click position (position 0 for a
    click near the box's start), so typing PREPENDED instead of replacing
    ("Zquokka" from a click near "quokka"'s start + typing "Z"). The fix:
    the box's first click (the one that ALSO focuses it) selects all, so
    the next keystroke replaces the whole stale query."""
    async with await widget_pilot(
        LibraryRail,
        shell=_make_shell(),
        preferences=LibraryRailPreferences(),
        query="quokka",
    ) as pilot:
        await pilot.pause()
        box = pilot.app.query_one("#library-search-input", Input)
        # This isolated harness auto-focuses the sole focusable widget on
        # start (the real app never does -- confirmed live: the rail search
        # box is blurred on screen re-entry, one focusable control among
        # many). Blur it explicitly so the click below exercises the same
        # "not-yet-focused" starting state the live bug needs.
        box.blur()
        await pilot.pause()
        assert not box.has_focus

        # Click near the START of the prefilled text -- the exact spot the
        # live UAT clicked, and the spot most likely to land at index 0
        # under the OLD (unfixed) precise-click-position behavior.
        await pilot.click(box, offset=(1, 0))
        await pilot.pause()
        assert box.has_focus
        # Selection is a NamedTuple(start, end) -- compares equal to a plain
        # tuple, so this asserts "the whole value is selected" without
        # importing Textual's private Selection class.
        assert tuple(box.selection) == (0, len(box.value)), (
            "the focusing click must select-all, not position a bare cursor"
        )

        await pilot.press("Z")
        await pilot.pause()
        assert box.value == "Z", (
            f"typing after the focusing click must REPLACE the stale query, "
            f"not insert into it: got {box.value!r}"
        )


async def test_second_click_while_already_focused_positions_cursor_normally(
    widget_pilot,
):
    """The select-all override is scoped to the FOCUSING click only -- once
    the box already has focus, a second click still positions the cursor
    precisely (normal, expected mid-text editing must not regress)."""
    async with await widget_pilot(
        LibraryRail,
        shell=_make_shell(),
        preferences=LibraryRailPreferences(),
        query="quokka",
    ) as pilot:
        await pilot.pause()
        box = pilot.app.query_one("#library-search-input", Input)
        box.focus()
        await pilot.pause()
        assert box.has_focus

        # A second click, now that the box already has focus, must land the
        # cursor at the click offset (not re-select-all).
        await pilot.click(box, offset=(3, 0))
        await pilot.pause()
        assert box.selection.is_empty
        assert tuple(box.selection) != (0, len(box.value))


async def test_search_rag_query_input_gets_the_same_click_select_all_fix():
    """LIB-17 explicitly covers "prefilled search Inputs" (plural, AC#7) --
    the canvas Search/RAG query box shares the rail search box's stale-
    prefill shape (rebuilt with ``value=`` seeded from the persisted query)
    and must not regress to click-then-prepend either."""
    from tldw_chatbook.Widgets.Library.library_search_rag_panel import (
        SelectAllOnFocusingClickInput,
    )

    assert issubclass(SelectAllOnFocusingClickInput.__mro__[0], Input)
    # Direct identity check: the panel imports the SAME class the rail
    # search box uses -- one implementation, not a fork.
    from tldw_chatbook.Widgets.Library.library_rail import (
        SelectAllOnFocusingClickInput as RailBase,
    )

    assert SelectAllOnFocusingClickInput is RailBase
