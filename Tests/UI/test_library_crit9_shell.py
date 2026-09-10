"""Critique-9 shell fixes: density, narrow Escape, Conversations footer, trust.

Covers tasks 32217, 32223, 32225 and 32228 -- the shell group of the
critique-9 fix wave.
"""

from __future__ import annotations

import pytest
from textual._context import NoActiveAppError
from textual.widget import Widget

from tldw_chatbook.UI.Library_Modules.screen_constants import (
    _LIBRARY_READER_SHELL_SELECTOR,
    LIBRARY_COLLECTIONS_READER_PROFILE,
    LIBRARY_CONVERSATION_READER_PROFILE,
    LIBRARY_PROMPTS_READER_PROFILE,
    LIBRARY_SKILLS_READER_PROFILE,
)
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    LibraryAdaptiveReaderShell,
)
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _FakeSkillsScopeService,
    _active_library_screen,
    _build_test_app,
    _wait_for_condition,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)


pytestmark = pytest.mark.asyncio


WIDE_TEST_SIZE = (235, 52)
NARROW_TEST_SIZE = (60, 24)


def _library_host() -> LibraryHarness:
    return LibraryHarness(_build_test_app())


def _panes_match_layout(screen, shell_id: str) -> bool:
    """Whether both optional panes are painted at their resolved widths."""
    shells = screen.query(shell_id)
    if not shells:
        return False
    shell = shells.first(LibraryAdaptiveReaderShell)
    layout = shell.effective_layout
    return (
        layout.items_open
        and layout.reader_width > 0
        and shell.items.region.width == layout.items_width
        and shell.work.region.width == layout.reader_width
    )


def _narrow_stage_layout(screen):
    """Return the route shell's settled layout while the rail pane is closed."""
    shells = screen.query(_LIBRARY_READER_SHELL_SELECTOR)
    if not shells:
        return None
    layout = shells.first(Widget).effective_layout
    if layout.library_open or layout.items_width + layout.reader_width == 0:
        return None
    return layout


@pytest.mark.parametrize(
    ("row_id", "shell_id", "profile"),
    [
        ("browse-prompts", "#library-prompts-reader-shell", LIBRARY_PROMPTS_READER_PROFILE),
        ("browse-skills", "#library-skills-reader-shell", LIBRARY_SKILLS_READER_PROFILE),
        (
            "browse-collections",
            "#library-collections-reader-shell",
            LIBRARY_COLLECTIONS_READER_PROFILE,
        ),
        (
            "browse-conversations",
            "#library-conversations-reader-shell",
            LIBRARY_CONVERSATION_READER_PROFILE,
        ),
    ],
)
async def test_a_canvas_with_nothing_open_gives_its_columns_to_the_list(
    row_id: str, shell_id: str, profile
) -> None:
    """task-32217 AC#1: an empty work pane hands its columns to the list.

    The rule Media and Notes already carry (``resolve_adaptive_reader_layout``'s
    ``reader_has_item`` block): a work pane showing only "Select ... to read it
    here." keeps its own floor and nothing more. Measured at the width the
    critique-9 live review ran at.
    """
    host = _library_host()
    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one(f"#library-row-{row_id}").press()
        await _wait_for_selector(screen, pilot, shell_id)
        # Settled means PAINTED, not merely resolved: an intermediate frame
        # can hand the work pane the whole stage while the list pane is still
        # 0x0, which reads exactly like the defect this test pins.
        await _wait_for_condition(
            pilot,
            lambda: _panes_match_layout(screen, shell_id),
            message=f"{shell_id} never settled its allocation.",
        )
        shell = screen.query_one(shell_id, LibraryAdaptiveReaderShell)
        items = shell.items
        work = shell.work
        print(
            f"MEASURED {row_id}: items={items.region.width} work={work.region.width} "
            f"work_min={profile.work_min_width} shell={shell.region.width}"
        )
        assert work.region.width == profile.work_min_width, (
            items.region,
            work.region,
        )
        assert items.region.width > work.region.width, (items.region, work.region)


async def test_the_landing_hub_keeps_a_readable_measure_on_a_wide_terminal() -> None:
    """task-32217 AC#3: the landing is capped, not stretched to 190 cells."""
    host = _library_host()
    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        landing = await _wait_for_selector(screen, pilot, "#library-landing-canvas")
        await pilot.pause()
        canvas = screen.query_one("#library-canvas")
        print(
            f"MEASURED landing: landing={landing.region.width} "
            f"canvas={canvas.region.width}"
        )
        assert landing.region.width == 96, (landing.region, canvas.region)
        # A landing collapsed to nothing would satisfy a bare upper bound.
        assert landing.region.height > 0, landing.region


async def test_the_skills_list_shows_each_row_s_trust_state() -> None:
    """task-32223: an approved and an unapproved skill no longer paint alike."""
    app = _build_test_app()
    app.skills_scope_service = _FakeSkillsScopeService(
        available=[{"name": "code-review", "trust_status": "trusted"}],
        blocked=[{"name": "summarize", "trust_status": "quarantined_added"}],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-skills").press()
        await _wait_for_selector(screen, pilot, "#library-skill-row-code-review")
        labels = {
            str(button.label)
            for button in screen.query("#library-skills-list Button")
        }
        print(f"MEASURED skills rows: {sorted(labels)}")
        assert any("code-review · trusted" in label for label in labels), labels
        assert any("summarize · needs review" in label for label in labels), labels


@pytest.mark.parametrize(
    "row_id",
    [
        "browse-media",
        "browse-conversations",
        "browse-prompts",
        "browse-skills",
        "browse-collections",
    ],
)
async def test_escape_returns_to_the_library_pane_below_64_columns(
    row_id: str,
) -> None:
    """task-32225: at 60 columns the rail pane is gone and Escape was inert.

    The footer advertised "esc focus rail" while the hop's destination
    (``#library-search-input``) lived inside a closed pane, so the key moved
    nothing on every adaptive-reader route. The chip now names the control
    that IS on screen ("‹ Library") and Escape does what that control does.
    """
    host = _library_host()
    async with host.run_test(size=NARROW_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one(f"#library-row-{row_id}").press()
        # Wait for a SETTLED allocation: the first frames carry an all-zero
        # layout whose rail is already hidden, so "the rail is not displayed"
        # alone reports the transition, not the stage.
        # Settled means the stage AND the key: pressing Escape into a
        # half-resolved arrival races the entry-focus arm still landing.
        await _wait_for_condition(
            pilot,
            lambda: _narrow_stage_layout(screen) is not None
            and screen.check_action("library_narrow_stage_return", ()),
            message="The Library pane never closed at 60 columns.",
        )
        chips = screen._library_footer_shortcuts_for_current_state()
        assert ("esc", "back to Library") in chips, chips
        assert not any(pair == ("esc", "focus rail") for pair in chips), chips

        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one("#library-rail").display,
            message="Escape never reopened the Library pane.",
        )


@pytest.mark.parametrize(
    "row_id", ["browse-media", "browse-prompts", "browse-collections"]
)
async def test_the_registered_footer_names_the_return_below_64_columns(
    row_id: str,
) -> None:
    """task-32225 AC#1: the chip reaches the RENDERED footer, not just the set.

    The footer is registered from ``compose_content``, before the shell has
    resolved its allocation, so the live app at 60x24 showed only "F6 next
    pane" while ``_library_footer_shortcuts_for_current_state()`` already
    carried the chip. Read back what was registered.
    """
    host = _library_host()
    async with host.run_test(size=NARROW_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one(f"#library-row-{row_id}").press()
        await _wait_for_condition(
            pilot,
            lambda: ("esc", "back to Library")
            in tuple((screen._footer_shortcut_registration or ("", ()))[1]),
            message=lambda: (
                "The registered footer never named the return: "
                f"{screen._footer_shortcut_registration}"
            ),
        )


async def test_the_return_chip_disappears_once_the_library_pane_is_back() -> None:
    """task-32225: the chip follows the pane, not just the width."""
    host = _library_host()
    async with host.run_test(size=NARROW_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts").press()
        # Settled first: pressing Escape against the all-zero opening layout
        # toggles a pane the resolver has not decided yet.
        await _wait_for_condition(
            pilot,
            lambda: _narrow_stage_layout(screen) is not None
            and ("esc", "back to Library")
            in tuple((screen._footer_shortcut_registration or ("", ()))[1]),
            message="The registered footer never named the return.",
        )
        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: ("esc", "back to Library")
            not in tuple((screen._footer_shortcut_registration or ("", ()))[1]),
            message=lambda: (
                "The return chip outlived the pane it names: "
                f"{screen._footer_shortcut_registration}"
            ),
        )
        assert screen.query_one("#library-rail").display


async def test_the_conversations_footer_advertises_the_filter_key_it_honours() -> None:
    """task-32228 AC#1: "/" is advertised exactly while it works.

    Live at 100x30 the Conversations canvas showed its Filter box, "/" focused
    it, and the footer said only "F6 next pane": the chip reads
    ``reader_layout.items_open``, which was still False when the footer was
    registered during compose. Every OTHER Conversations chip is honest -- the
    Escape hop is advertised under the label it will actually perform
    ("focus Items" / "focus Library", pinned by
    test_conversations_escape_moves_to_nearest_visible_prior_role), and it is
    absent exactly while the hop would move nothing.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")
        await _wait_for_condition(
            pilot,
            lambda: ("/", "focus filter")
            in tuple((screen._footer_shortcut_registration or ("", ()))[1]),
            message=lambda: (
                "The Conversations footer never advertised its Filter key: "
                f"{screen._footer_shortcut_registration}"
            ),
        )
        # ...and the key it advertises does what the chip says.
        await pilot.press("slash")
        await pilot.pause()
        # By id: the canvas recomposes, so the mounted box is a new instance.
        assert getattr(screen.focused, "id", None) == "library-conversations-filter"


async def test_the_return_chip_stands_down_where_escape_belongs_to_the_surface() -> None:
    """task-32225 fix round 1: the chip may not override an earlier Escape.

    Every Escape binding declared ABOVE ``library_narrow_stage_return``
    (viewer back, editor back, trash back, an armed delete confirm, ...), and
    Textual gives the key to the first gate that passes. Below 64 columns the
    Media viewer therefore painted "esc back to Library" while Escape went to
    the media list and the Library pane stayed shut -- exactly the dishonest
    chip this wave exists to remove. The gate now stands down whenever an
    earlier Escape action is live, the way the ``emergency.enabled`` branch
    beside it stands down through its own ``guarded`` projection.
    """
    from Tests.UI.test_library_media_reader_flow import _flow_app, _load_row_0
    from Tests.UI.test_library_media_side_by_side import _open_media_list
    from Tests.UI.test_library_shell import LibraryProductionCSSHarness

    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=NARROW_TEST_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen._media_state.view == "viewer"
            and screen.check_action("library_media_viewer_back", ()),
            message="The Media viewer never took Escape at 60 columns.",
        )

        chips = screen._library_footer_shortcuts_for_current_state()
        assert ("esc", "back to Library") not in chips, chips
        assert screen._library_narrow_stage_return_active() is False

        # ...and the key still does what the surface's own chip promises.
        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: screen._media_state.view == "list",
            message="Escape did not return the Media viewer to its list.",
        )


def test_the_narrow_stage_gate_survives_a_screen_with_no_active_app() -> None:
    """task-32225 fix round 2: the gate must not read ``Screen.size`` too early.

    Re-review N1: ``Screen.size`` is ``self.app.size - gutter``, NOT
    ``Widget.size``. Read with no active app it RAISES ``NoActiveAppError`` --
    it never returns ``Size(0, 0)``, which is what round 1's guard and its
    ``SimpleNamespace(size=Size(0, 0))`` fake pinned. The real state is an
    unmounted screen driven directly, which is how
    ``Tests/UI/test_library_entry_compose_once.py`` exercises
    ``apply_navigation_context`` -> ``_register_footer_shortcuts`` -> this gate,
    and where eight of its cases went red.

    Pinned against a REAL ``LibraryScreen`` outside any app context, so the
    fake can no longer disagree with the class it stands for.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    screen = LibraryScreen(_build_test_app())
    assert screen.is_mounted is False
    with pytest.raises(NoActiveAppError):
        _ = screen.size  # the read the gate must not reach unguarded
    assert screen._library_narrow_stage_return_active() is False
    # ...and the footer that reads it survives the same context.
    assert isinstance(screen._library_footer_shortcuts_for_current_state(), tuple)


async def test_the_conversations_footer_advertises_escape_on_arrival() -> None:
    """task-32228 AC#1 (ruling R1): arrive with focus on the list, like siblings.

    The Conversations footer was "nearly empty" next to its siblings for a
    reason nobody had traced: this was the one browse list that did NOT arm
    task-2856's entry focus, so on arrival focus sat outside the shell, the
    Escape hop had nowhere to go from, and its chip was (correctly) withheld.
    The keys were never missing -- the state that makes them live was.

    The label is "focus Library", not the siblings' "focus rail": on this
    canvas Escape steps back through the visible panes (Items -> Library), the
    grammar it shares with the Media Reader and that
    test_conversations_escape_moves_to_nearest_visible_prior_role pins. Parity
    is on the keys, which is what the AC asks for.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", "") == "library-conversation-row-0",
            message=lambda: (
                "Conversations never took entry focus like its siblings: "
                f"{screen.focused!r}"
            ),
        )
        chips = screen._library_footer_shortcuts_for_current_state()
        assert ("/", "focus filter") in chips, chips
        assert ("esc", "focus Library") in chips, chips
        assert screen.check_action("library_list_focus_rail", ()) is True

        await pilot.press("escape")
        await pilot.pause()
        assert getattr(screen.focused, "id", "") == "library-search-input"


def test_an_unrecognised_trust_status_never_claims_trust() -> None:
    """task-32223 fix round 1 (review finding 10): the word is a claim.

    The row's glyph is decoration and may default; the word must not. An
    unrecognised status on a record that is not blocked now says nothing
    rather than asserting "trusted".
    """
    from tldw_chatbook.Library.library_skills_state import _trust_row_label

    assert _trust_row_label("trusted", blocked=False) == "trusted"
    assert _trust_row_label("trust_locked", blocked=True) == "locked"
    assert _trust_row_label("quarantined_added", blocked=True) == "needs review"
    assert _trust_row_label("trust_uninitialized", blocked=False) == "needs review"
    # Unknown to us: blocked still says so; not-blocked says nothing.
    assert _trust_row_label("some_future_status", blocked=True) == "needs review"
    assert _trust_row_label("some_future_status", blocked=False) == ""
    assert _trust_row_label("", blocked=False) == ""


async def test_the_slash_chip_stands_down_when_its_box_is_in_a_closed_pane() -> None:
    """task-32225 re-review N4: the same dead-chip test, one key over.

    Below 64 columns the box "/" jumps to can be inside a CLOSED pane --
    mounted, so the handler's query finds it, but out of the focus chain, so
    ``focus()`` is a no-op. Measured in the Media Reader at 60x24:
    "/ focus search" was the FIRST painted chip (the P1 stand-down promoted it
    there), ``#library-media-filter`` was mounted inside the collapsed Items
    pane, and the key moved nothing.
    """
    from Tests.UI.test_library_media_reader_flow import _flow_app, _load_row_0
    from Tests.UI.test_library_media_side_by_side import _open_media_list
    from Tests.UI.test_library_shell import LibraryProductionCSSHarness

    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=NARROW_TEST_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen._media_state.view == "viewer",
            message="The Media viewer never opened at 60 columns.",
        )
        assert screen.query_one("#library-rail").display is False
        # The filter IS mounted -- being findable is not being focusable.
        assert screen.query("#library-media-filter")

        chips = screen._library_footer_shortcuts_for_current_state()
        assert not any(pair[0] == "/" for pair in chips), chips

        before = screen.focused
        await pilot.press("slash")
        await pilot.pause()
        assert screen.focused is before, screen.focused


async def test_the_slash_chip_survives_where_the_key_still_works() -> None:
    """Negative control for the chip above: a reachable box keeps its chip.

    Same canvas, same harness, one width apart -- so the difference under test
    is the closed pane and nothing else.
    """
    from Tests.UI.test_library_media_reader_flow import _flow_app
    from Tests.UI.test_library_media_side_by_side import _open_media_list
    from Tests.UI.test_library_shell import LibraryProductionCSSHarness

    app, _service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one("#library-rail").display
            and any(
                pair[0] == "/"
                for pair in screen._library_footer_shortcuts_for_current_state()
            ),
            message=lambda: (
                "The Media footer dropped a working / key: "
                f"{screen._library_footer_shortcuts_for_current_state()}"
            ),
        )
        await pilot.press("slash")
        await pilot.pause()
        assert getattr(screen.focused, "id", "") == "library-media-filter"
