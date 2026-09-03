"""task-14902: Library cycle controls offer a discoverable option menu.

The Notes Sort chooser (press -> one-row choice strip, ``✓`` on the active
option, pick applies + closes, Escape cancels) is the in-house discoverable
pattern; the sync panel's direction/conflict groups are the always-visible
variant of the same grammar. This module pins the convergence of the
remaining cyclers on that pattern via ONE shared mechanism:

- shared builders in ``library_shell_state.py`` (``library_choice_label`` for
  chooser-openers, ``library_toggle_label`` for the kept one-press toggles,
  ``library_choice_tooltip``), and
- one strip composer in ``Widgets/Library/library_choice_strip.py``.

Converged to strips: media type filter (both task-14900 layouts), prompts
sort, skills sort, export quality. Kept as one-press toggles with the full
option set ON the label (``name: ✓ a ⇄ b``): Search/RAG mode and the three
skill-editor toggles -- a two-option cycler is a genuine toggle, and a strip
would add a press to the most common action for zero information. The
prompts collection control already opens a direct-pick surface (the
collection manager modal); its work here is vocabulary honesty -- it is a
chooser, not a cycler, so it loses the ``⇄``.
"""

from types import SimpleNamespace

import pytest
# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, OptionList, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)

#: task-14900's two width regimes (mirrors test_library_media_side_by_side).
WIDE_SIZE = LIBRARY_TEST_SIZE
NARROW_SIZE = (100, 30)


# ---------------------------------------------------------------------------
# Shared builders: one labelling grammar (AC#1/AC#2).
# ---------------------------------------------------------------------------


def test_library_choice_label_and_tooltip_shapes():
    """Chooser-openers use the Notes Sort spelling: ``name: value`` with NO
    cycle glyph (press opens choices; it no longer advances), and the
    tooltip names the pick interaction instead of a false "Cycles"."""
    from tldw_chatbook.Library.library_shell_state import (
        library_choice_label,
        library_choice_tooltip,
    )

    assert library_choice_label("type", "All") == "type: All"
    assert "⇄" not in library_choice_label("quality", "thumbnail")
    assert library_choice_tooltip("media type", ("All", "audio", "video")) == (
        "Press to pick media type: All · audio · video."
    )
    assert library_choice_tooltip("media type", ()) == "Press to pick media type."


def test_library_toggle_label_marks_active_option():
    """Kept one-press toggles satisfy AC#1 at the label: BOTH options are on
    screen, the active one carries the ``✓`` marker (AC#2, same marker as
    the strips), and the ``⇄`` between them keeps its press-advances
    meaning."""
    from tldw_chatbook.Library.library_shell_state import library_toggle_label

    assert library_toggle_label("mode", ("Search", "RAG Answer"), 0) == (
        "mode: ✓ Search ⇄ RAG Answer"
    )
    assert library_toggle_label("mode", ("Search", "RAG Answer"), 1) == (
        "mode: Search ⇄ ✓ RAG Answer"
    )


def test_skill_toggle_labels_show_both_options_with_active_marker():
    """The three skill-editor toggles are kept toggles: full option set on
    the label, ``✓`` on the active option. The context toggle keeps its
    task-418 plain-language hint on the ACTIVE option only (60-col-safe)."""
    from tldw_chatbook.Widgets.Library.library_skills_canvas import (
        skill_context_toggle_label,
        skill_disable_model_label,
        skill_user_invocable_label,
    )

    assert skill_user_invocable_label(True) == "User can invoke: ✓ yes ⇄ no"
    assert skill_user_invocable_label(False) == "User can invoke: yes ⇄ ✓ no"
    # Stored polarity is disable_model_invocation; the label answers
    # "can the agent invoke?" (task-418), so True -> "no" active.
    assert skill_disable_model_label(False) == "Agent can invoke: ✓ yes ⇄ no"
    assert skill_disable_model_label(True) == "Agent can invoke: yes ⇄ ✓ no"
    assert skill_context_toggle_label("inline") == (
        "Runs in: ✓ inline (this conversation) ⇄ fork"
    )
    assert skill_context_toggle_label("fork") == (
        "Runs in: inline ⇄ ✓ fork (sub-agent)"
    )


def test_rag_mode_toggle_label_shows_both_modes():
    """The Search/RAG mode control is the kept toggle par excellence (a
    two-state mode flip that resets retrieval state): its full option set
    moves onto the label."""
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
    from tldw_chatbook.Widgets.Library.library_search_rag_panel import (
        _mode_toggle_label,
    )

    def state_for(mode: str):
        return SimpleNamespace(
            query_state=SimpleNamespace(
                mode=mode,
                mode_label="Search" if mode == "search" else "RAG Answer",
            )
        )

    assert LibraryRagPanelState is not None
    assert _mode_toggle_label(state_for("search")) == "mode: ✓ Search ⇄ RAG Answer"
    assert _mode_toggle_label(state_for("rag")) == "mode: Search ⇄ ✓ RAG Answer"


@pytest.mark.asyncio
async def test_prompts_collection_control_is_a_chooser_not_a_cycler():
    """Pressing the collection control opens the manager modal (a full
    direct-pick surface) -- it does NOT cycle, so carrying the ``⇄``
    press-advances glyph (and a "Cycles the prompt scope" tooltip) was
    dishonest vocabulary. It now uses the shared chooser label."""
    from tldw_chatbook.Library.library_prompts_state import PromptsListState
    from tldw_chatbook.Widgets.Library.library_prompts_canvas import (
        LibraryPromptsListCanvas,
    )

    canvas = LibraryPromptsListCanvas(
        PromptsListState(rows=(), count=0, sort="newest"),
        collection_label="All prompts",
    )

    class Host(ConsolidatedCSSApp):
        def compose(self):
            yield canvas

    async with Host().run_test() as pilot:
        button = pilot.app.query_one("#library-prompts-collection", Button)
        label = str(button.label)
        tooltip = str(button.tooltip or "")
    assert label == "collection: All prompts"
    assert "⇄" not in label
    assert not tooltip.startswith("Cycles")


# ---------------------------------------------------------------------------
# Media type strip: full set on screen, direct pick, Escape, both layouts,
# keyboard-only (AC#1-#3).
# ---------------------------------------------------------------------------


async def _open_media_list(host, pilot):
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-media").press()
    await _wait_for_selector(screen, pilot, "#library-media-type-filter")
    return screen


@pytest.mark.asyncio
async def test_media_type_strip_opens_full_set_marks_active_and_picks():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)

        opener = screen.query_one("#library-media-type-filter", Button)
        assert str(opener.label) == "type: All types"

        opener.press()
        chooser = await _wait_for_selector(
            screen, pilot, "#library-media-type-choices"
        )
        assert isinstance(chooser, OptionList)
        labels = {str(option.prompt) for option in chooser.options}
        # Full option set on screen, ✓ on the active option only.
        assert labels == {"✓ All types", "audio", "video"}

        chooser.highlighted = next(
            index
            for index, option in enumerate(chooser.options)
            if getattr(option, "choice_value", None) == "audio"
        )
        chooser.action_select()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_type_filter == "audio"
        assert not screen.query("#library-media-type-choices")
        opener = screen.query_one("#library-media-type-filter", Button)
        assert str(opener.label) == "type: audio"
        rows = list(screen.query(".library-media-row"))
        assert len(rows) == 1
        assert "Interview Recording" in str(rows[0].label)


@pytest.mark.asyncio
async def test_media_type_strip_escape_closes_without_change():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)

        screen.query_one("#library-media-type-filter", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-type-choices")

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert not screen.query("#library-media-type-choices")
        assert screen._library_media_type_filter is None
        # The opener regains focus so Escape round-trips for keyboard users.
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None)
            == "library-media-type-filter",
            message=(
                "Escape never refocused the type opener; focused: "
                f"{screen.focused!r}"
            ),
        )
        assert len(list(screen.query(".library-media-row"))) == 2


@pytest.mark.asyncio
async def test_media_type_strip_works_in_both_layouts():
    """task-14900 interaction: the strip must render and pick in BOTH the
    wide side-by-side regime and the stacked compact regime."""
    for size, compact in ((WIDE_SIZE, False), (NARROW_SIZE, True)):
        app = _build_test_app()
        _seed_conversations(app, _two_conversations(), media=_two_media_items())
        host = LibraryHarness(app)

        async with host.run_test(size=size) as pilot:
            screen = await _open_media_list(host, pilot)
            host_pane = screen.query_one("#library-canvas")
            await _wait_for_condition(
                pilot,
                lambda: host_pane.has_class("library-notes-compact") is compact,
                message=f"compact class never reached {compact} at {size}",
            )

            screen.query_one("#library-media-type-filter", Button).press()
            chooser = await _wait_for_selector(
                screen, pilot, "#library-media-type-choices"
            )
            assert isinstance(chooser, OptionList)
            assert chooser.option_count == 3
            assert chooser.region.width > 0 and chooser.region.height > 0
            chooser.highlighted = next(
                index
                for index, option in enumerate(chooser.options)
                if getattr(option, "choice_value", None) == "video"
            )
            chooser.action_select()
            await pilot.pause()
            await pilot.pause()
            assert screen._library_media_type_filter == "video"
            assert not screen.query("#library-media-type-choices")


@pytest.mark.asyncio
async def test_media_type_strip_keyboard_only_path():
    """Open the strip and pick an option using Tab/Enter only. Opening
    focuses the active choice so the strip is immediately traversable."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)

        opener = screen.query_one("#library-media-type-filter", Button)
        for _ in range(120):
            if screen.focused is opener:
                break
            await pilot.press("tab")
            await pilot.pause()
        else:
            raise AssertionError("Tab never reached the type opener.")

        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-media-type-choices")
        # Focus lands in the chooser with the active unfiltered option highlighted.
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", "")
            == "library-media-type-choices",
            message=f"Focus never entered the strip; focused: {screen.focused!r}",
        )
        chooser = screen.query_one("#library-media-type-choices", OptionList)
        assert chooser.highlighted == 0
        assert str(chooser.highlighted_option.prompt) == "✓ All types"

        await pilot.press("down")
        await pilot.pause()
        assert str(chooser.highlighted_option.prompt) == "audio"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_type_filter == "audio"
        assert not screen.query("#library-media-type-choices")


@pytest.mark.asyncio
async def test_media_type_chooser_keeps_complete_facets_in_one_bounded_widget():
    media = [
        {
            "id": f"media-{index}",
            "title": f"Media {index:02d}",
            "type": "All" if index == 0 else f"type-{index:02d}",
            "last_modified": f"2026-08-01T00:{index:02d}:00Z",
        }
        for index in range(63)
    ]
    app = _build_test_app()
    _seed_conversations(app, [], media=media)
    host = LibraryHarness(app)

    async with host.run_test(size=NARROW_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        controller = screen._library_media_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: len(controller.type_options) == 63,
            message="Complete Media facets never settled.",
        )
        applied_before = controller.applied_scope
        requested_before = controller.requested_scope

        screen.query_one("#library-media-type-filter", Button).press()
        chooser = await _wait_for_selector(
            screen, pilot, "#library-media-type-choices"
        )
        assert isinstance(chooser, OptionList)
        assert len(screen.query("#library-media-type-choices")) == 1
        assert chooser.option_count == 64
        assert len(chooser.children) == 0
        assert chooser.region.height <= 10
        assert getattr(chooser.get_option_at_index(0), "choice_value") is None
        assert str(chooser.get_option_at_index(0).prompt) == "✓ All types"
        assert getattr(chooser.get_option_at_index(1), "choice_value") == "All"
        assert str(chooser.get_option_at_index(1).prompt) == "All"
        assert controller.applied_scope == applied_before
        assert controller.requested_scope == requested_before

        chooser.focus()
        await pilot.press("end")
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: controller.applied_scope is not None
            and controller.applied_scope.media_type == "type-62",
            message="Keyboard commit never selected the final complete type.",
        )
        assert not screen.query("#library-media-type-choices")


# ---------------------------------------------------------------------------
# Footer/F1 (AC#3): the one shared seam advertises the open strip.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_footer_advertises_open_media_type_strip():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)

        before = screen._library_footer_shortcuts_for_current_state()
        assert ("enter", "choose type") not in before

        screen.query_one("#library-media-type-filter", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-type-choices")

        shortcuts = screen._library_footer_shortcuts_for_current_state()
        assert ("enter", "choose type") in shortcuts
        assert ("esc", "cancel") in shortcuts

        await pilot.press("escape")
        await pilot.pause()
        after = screen._library_footer_shortcuts_for_current_state()
        assert ("enter", "choose type") not in after


# ---------------------------------------------------------------------------
# Export quality strip: opener stays visible (second press closes), pick
# updates the helper line.
# ---------------------------------------------------------------------------


async def _open_media_export(host, pilot):
    screen = await _open_media_list(host, pilot)
    screen.query_one("#library-media-export").press()
    await _wait_for_selector(screen, pilot, "#library-export-quality")
    return screen


@pytest.mark.asyncio
async def test_export_quality_strip_opens_picks_and_second_press_closes():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_media_export(host, pilot)

        opener = screen.query_one("#library-export-quality", Button)
        assert str(opener.label) == "quality: thumbnail"

        opener.press()
        await _wait_for_selector(screen, pilot, "#library-export-quality-choices")
        labels = [
            str(button.label)
            for button in screen.query(".library-export-quality-choice")
        ]
        assert labels == ["✓ thumbnail", "compressed", "original"]

        # Second press on the still-visible opener closes without change.
        screen.query_one("#library-export-quality", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-export-quality-choices")
        assert screen._export_state.form.get("quality", "thumbnail") == "thumbnail"

        # Reopen and pick directly: value + helper line update, strip closes.
        screen.query_one("#library-export-quality", Button).press()
        await _wait_for_selector(screen, pilot, "#library-export-quality-choices")
        original = next(
            button
            for button in screen.query(".library-export-quality-choice")
            if str(button.label) == "original"
        )
        original.press()
        await pilot.pause()
        await pilot.pause()

        assert screen._export_state.form["quality"] == "original"
        assert not screen.query("#library-export-quality-choices")
        assert (
            str(screen.query_one("#library-export-quality", Button).label)
            == "quality: original"
        )
        helper = str(
            screen.query_one("#library-export-quality-helper", Static).renderable
        )
        assert "original" in helper.lower() or "full" in helper.lower()


@pytest.mark.asyncio
async def test_footer_advertises_open_export_quality_strip():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_media_export(host, pilot)

        screen.query_one("#library-export-quality", Button).press()
        await _wait_for_selector(screen, pilot, "#library-export-quality-choices")
        shortcuts = screen._library_footer_shortcuts_for_current_state()
        assert ("enter", "choose quality") in shortcuts
        assert ("esc", "cancel") in shortcuts

        # Escape closes the strip FIRST -- it must not leave the Export
        # canvas while a strip is open.
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-export-quality-choices")
        assert screen.query("#library-export-quality")


# ---------------------------------------------------------------------------
# Skills sort strip (shell-level).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skills_sort_strip_opens_and_applies():
    from Tests.UI.test_destination_shells import (
        StaticLibraryConversationScopeService,
        StaticLibraryMediaScopeService,
        StaticLibraryNotesListScopeService,
    )
    from Tests.UI.test_library_skills_canvas import _FakeSkillsScopeService

    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = _FakeSkillsScopeService(
        available=[{"name": "code-review"}],
        blocked=[{"name": "summarize"}],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-skills").press()
        await _wait_for_selector(screen, pilot, "#library-skills-sort")

        assert screen._library_skills_sort == "name"
        opener = screen.query_one("#library-skills-sort", Button)
        assert str(opener.label) == "sort: Name"

        opener.press()
        await _wait_for_selector(screen, pilot, "#library-skills-sort-choices")
        labels = [
            str(button.label)
            for button in screen.query(".library-skills-sort-choice")
        ]
        assert labels == ["✓ Name", "Status"]

        screen.query_one("#library-skills-sort-status", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_skills_sort == "status"
        assert not screen.query("#library-skills-sort-choices")
        assert (
            str(screen.query_one("#library-skills-sort", Button).label)
            == "sort: Status"
        )


# ---------------------------------------------------------------------------
# Prompts sort strip: pick requests the EXACT service scope.
# ---------------------------------------------------------------------------


def test_prompts_sort_choice_requests_exact_scope():
    """A direct pick maps to the exact browse scope the old cycle produced
    for that value (name -> name/asc, newest -> last_modified/desc), always
    resetting to page 1; picking the already-active value only closes."""
    from tldw_chatbook.Library.library_prompts_state import PromptBrowseScope
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    requests = []
    refreshes = []

    def make_fake(sort_by: str):
        applied_scope = PromptBrowseScope(sort_by=sort_by)
        return SimpleNamespace(
            # task-15790: production gained this in-flight guard; stale double.
            _library_prompts_mutation_in_flight=False,
            _library_prompts_sort_choices_visible=True,
            _library_prompt_browse_controller=SimpleNamespace(
                scope=applied_scope,
                visible_result=SimpleNamespace(scope=applied_scope),
            ),
            _request_library_prompts_browse=lambda scope, focus_identity=None: (
                requests.append((scope, focus_identity))
            ),
            refresh=lambda recompose=False: refreshes.append(recompose),
            call_after_refresh=lambda *args, **kwargs: None,
            _focus_library_control=lambda selector: None,
        )

    fake = make_fake("last_modified")
    event = SimpleNamespace(
        stop=lambda: None,
        button=SimpleNamespace(choice_value="name"),
    )
    LibraryScreen.handle_library_prompts_sort_choice(fake, event)
    assert fake._library_prompts_sort_choices_visible is False
    assert len(requests) == 1
    scope, focus_identity = requests[0]
    assert (scope.sort_by, scope.sort_order, scope.page) == ("name", "asc", 1)
    assert focus_identity == "library-prompts-sort"

    # Picking the already-active value closes without a service request.
    requests.clear()
    fake = make_fake("last_modified")
    event = SimpleNamespace(
        stop=lambda: None,
        button=SimpleNamespace(choice_value="newest"),
    )
    LibraryScreen.handle_library_prompts_sort_choice(fake, event)
    assert fake._library_prompts_sort_choices_visible is False
    assert requests == []
    assert refreshes == [True]


@pytest.mark.asyncio
async def test_prompts_canvas_composes_sort_strip_when_visible():
    from tldw_chatbook.Library.library_prompts_state import PromptsListState
    from tldw_chatbook.Widgets.Library.library_prompts_canvas import (
        LibraryPromptsListCanvas,
    )

    canvas = LibraryPromptsListCanvas(
        PromptsListState(rows=(), count=0, sort="newest"),
        sort_mode="newest",
        sort_choices_visible=True,
    )

    class Host(ConsolidatedCSSApp):
        def compose(self):
            yield canvas

    async with Host().run_test() as pilot:
        labels = [
            str(button.label)
            for button in pilot.app.query(".library-prompts-sort-choice")
        ]
        assert labels == ["✓ Newest", "Name"]
        # The toolbar row is swapped out while the strip is open (the Notes
        # browse-actions precedent).
        toolbar_sort = pilot.app.query_one("#library-prompts-sort", Button)
        assert toolbar_sort.parent.display is False
