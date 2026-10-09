"""Mounted contracts of Roleplay's one-row header (frame slice B1, spec 1.3).

Geometry runs under BOTH styled tiers (``Tests/UI/roleplay_frame_harness.py``)
at every B1 size, measured relative to the nav bar and the header (spec 5.7.2
item 1), never as absolute rows. Behaviour checks that need no geometry run on
the styled mock tier only; the leave-guard check needs the real app's
navigation and runs on the full tier.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from rich.cells import cell_len
from textual.color import Color
from textual.widgets import Static

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from Tests.UI.roleplay_frame_harness import (
    assert_painted_inside,
    chrome_bottoms,
    first_list_item,
    open_styled_roleplay,
    painted_rows,
    painted_text,
    seed_mock_characters,
    settle,
    size_matrix,
    styled_tiers,
    wait_until,
)
from Tests.UI.test_personas_workbench import (
    _conversation_record,
    _install_conversation_db,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Persona_Modules import roleplay_frame_state as fs
from tldw_chatbook.UI.Screens.personas_screen import (
    PERSONAS_COMPACT_WORKBENCH_MAX_WIDTH,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import DestinationHeader, FittedText
from tldw_chatbook.Widgets.glyph_fallback import ASCII_GLYPH_FALLBACKS
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

#: Twelve characters with a dated meta line, so every row is two lines tall
#: like a real library (the date is the meta when a card has no description).
CHARACTERS = [
    {
        "id": index,
        "name": f"Character {index:02d}",
        "version": 1,
        "last_modified": f"2026-09-{index:02d}T12:00:00",
    }
    for index in range(1, 13)
]

#: 213 cells: longer than the item label is wide at every B1 size (220x55 included).
LONG_NAME = (
    "Ser Bartholomew Ignatius Fairweather-Montgomery of the Seventeen Silver "
    "Towers, Keeper of the Long Keys, Warden of the Quiet Marches and Sworn "
    "Shield of the Ninefold Orchard Gates beyond the Amber Hills of Lowmere"
)


@pytest.fixture
def roleplay_data(monkeypatch):
    """Twelve characters and one conversation through the screen's seams."""
    seed_mock_characters(monkeypatch, CHARACTERS)
    monkeypatch.setattr(
        character_handler_module, "_default_character_db", lambda: object()
    )
    _install_conversation_db(monkeypatch, [_conversation_record(1)])
    return CHARACTERS


def _parts(screen):
    header = screen.query_one("#personas-header")
    return {
        "header": header,
        "title": header.query_one("#workbench-header-title", Static),
        "subtitle": header.query_one("#workbench-header-subtitle", Static),
        "item": header.query_one("#personas-header-item", FittedText),
        "unsaved": header.query_one("#personas-header-unsaved", FittedText),
        "blocked": header.query_one("#personas-header-blocked", FittedText),
        "status": header.query_one("#workbench-header-status", Static),
    }


@styled_tiers
@size_matrix
async def test_header_is_one_row_and_the_kind_stays_visible(
    styled_tier, roleplay_size, mock_app_instance, roleplay_data
):
    """AC#1: one row at every size; the kind shows even at 24 rows or fewer."""
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=roleplay_size
    ) as pilot:
        screen = pilot.app.screen
        parts = _parts(screen)
        nav_bottom, _header_bottom = chrome_bottoms(screen)
        assert parts["header"].region.y == nav_bottom
        assert parts["header"].region.height == 1
        assert parts["subtitle"].display
        assert painted_text(screen, parts["subtitle"].region) == "Characters"
        if roleplay_size[1] <= 24:
            # The compact-height path that used to hide the subtitle is live.
            assert screen.has_class("shell-header-compact")


@styled_tiers
@size_matrix
async def test_first_list_item_sits_right_under_the_unchanged_band(
    styled_tier, roleplay_size, mock_app_instance, roleplay_data
):
    """AC#2: B1 moves only the header. The purpose line, the mode strip and
    the workbench chrome keep their rows, so the first item sits a fixed
    number of rows under the header: with today's 3-row nav that is row 19
    at the design centre (y 18) and row 18 at 80x24, whose compact workbench
    has one row less chrome."""
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=roleplay_size
    ) as pilot:
        screen = pilot.app.screen
        nav_bottom, header_bottom = chrome_bottoms(screen)
        assert header_bottom == nav_bottom + 1
        band = 13 if roleplay_size[0] <= PERSONAS_COMPACT_WORKBENCH_MAX_WIDTH else 14
        assert first_list_item(screen).region.y == header_bottom + band


@styled_tiers
async def test_seven_characters_show_at_120x36(
    styled_tier, mock_app_instance, roleplay_data
):
    """AC#2: at least 7 two-line rows show their name line at 120x36 (5 before)."""
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=(120, 36)
    ) as pilot:
        screen = pilot.app.screen
        window = screen.query_one("#personas-library-rows").scrollable_content_region
        items = list(screen.query("#personas-library-rows > ListItem"))
        assert items and all(item.region.height == 2 for item in items[:3])
        shown = [
            item for item in items if window.contains(item.region.x, item.region.y)
        ]
        assert len(shown) >= 7, [item.region for item in items]


@styled_tiers
@pytest.mark.parametrize(
    ("roleplay_size", "work_width", "inspector_width"),
    [((120, 36), 58, 30), ((160, 45), 78, 39), ((220, 55), 108, 54)],
    ids=["120x36", "160x45", "220x55"],
)
async def test_the_work_pane_and_inspector_keep_their_widths(
    styled_tier,
    roleplay_size,
    work_width,
    inspector_width,
    mock_app_instance,
    roleplay_data,
):
    """Spec 5.3's interim geometry: B1 moves only the header, so the work pane
    and the Inspector keep the base arm's widths (spec 5.3's "today" row,
    measured on both styled tiers on the base arm)."""
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=roleplay_size
    ) as pilot:
        screen = pilot.app.screen
        work = screen.query_one("#personas-work-area")
        inspector = screen.query_one("#personas-inspector-pane")
        assert (work.region.width, inspector.region.width) == (
            work_width,
            inspector_width,
        )


@styled_tiers
@size_matrix
async def test_every_header_part_fits_in_the_worst_case(
    styled_tier, roleplay_size, mock_app_instance, roleplay_data, monkeypatch
):
    """No part is clipped with every chip, a server label, a long name and the
    longest kind; the kind is painted whole (AC#4)."""
    inputs = fs.RoleplayHeaderInputs(
        mode="dictionaries",
        edit_mode="edit",
        item_name=LONG_NAME,
        unsaved=True,
        provider_blocked=True,
        runtime_source="server",
        server_label="home-tldw.example.internal",
    )
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=roleplay_size
    ) as pilot:
        screen = pilot.app.screen
        monkeypatch.setattr(screen, "_gather_header_inputs", lambda: inputs)
        screen._update_title()
        await settle(pilot)
        parts = _parts(screen)
        header = parts["header"]
        view = fs.build_header_view(inputs, header.region.width)
        expected = {
            "title": "Roleplay",
            "subtitle": "Dictionaries",
            "unsaved": view.unsaved_chip,
            "blocked": view.blocked_chip,
            "status": view.status_plain,
        }
        for name, text in expected.items():
            part = parts[name]
            assert part.display and part.region.width > 0, name
            # Inside the header's painted window and covered by no sibling.
            assert_painted_inside(part, header)
            assert painted_text(screen, part.region).strip() == text, name
        item = parts["item"]
        assert_painted_inside(item, header)
        painted = painted_text(screen, item.region).rstrip()
        assert painted == item.fitted_text
        assert cell_len(painted) <= item.region.width
        if painted.startswith("›"):
            assert painted.endswith("… · editing")
        else:
            assert painted in ("· editing", "")


@styled_tiers
async def test_header_chrome_cells_match_the_lazy_sheet(
    styled_tier, mock_app_instance, roleplay_data, monkeypatch
):
    """The fit's HEADER_*_CELLS constants are the sheet's real spacing, under
    both the harness CSS_PATH and the app's route loader."""
    inputs = fs.RoleplayHeaderInputs(
        mode="characters", unsaved=True, provider_blocked=True
    )
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=(160, 45)
    ) as pilot:
        screen = pilot.app.screen
        monkeypatch.setattr(screen, "_gather_header_inputs", lambda: inputs)
        screen._update_title()
        await settle(pilot)
        parts = _parts(screen)

        def chrome(widget, *, margin=True):
            styles = widget.styles
            left = styles.margin.left if margin else 0
            return left + styles.padding.left + styles.padding.right

        assert chrome(parts["header"], margin=False) == fs.HEADER_PADDING_CELLS
        assert parts["subtitle"].styles.margin.left == fs.KIND_GAP_CELLS
        assert parts["item"].styles.margin.left == fs.ITEM_GAP_CELLS
        assert chrome(parts["unsaved"]) == fs.CHIP_CHROME_CELLS
        assert chrome(parts["blocked"]) == fs.CHIP_CHROME_CELLS
        assert chrome(parts["status"]) == fs.STATUS_CHROME_CELLS


@pytest.mark.parametrize(
    "target", [(90, 45), (80, 24), (220, 55)], ids=lambda s: f"{s[0]}x{s[1]}"
)
async def test_a_resize_refits_the_header_without_gathering_inputs(
    target, mock_app_instance, roleplay_data, monkeypatch
):
    """Spec 2.13 ("resize does no data work"): a width-only change refits the
    chips and the status from the cached inputs (the short unsaved chip below
    100 columns, the degrade order below that) and never re-reads readiness
    or the draft aggregate."""
    inputs = fs.RoleplayHeaderInputs(
        mode="characters",
        unsaved=True,
        provider_blocked=True,
        runtime_source="server",
        server_label="home-tldw.example.internal",
    )
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        # The 0.25 s readiness poll re-gathers on its own: stop it, so only
        # the resize can repaint.
        screen._console_readiness_poll_timer.stop()
        screen._header_inputs = inputs
        screen._paint_header()
        await settle(pilot)
        parts = _parts(screen)
        gathered = []
        monkeypatch.setattr(
            screen, "_gather_header_inputs", lambda: gathered.append(1) or inputs
        )
        await pilot.resize_terminal(*target)
        await settle(pilot)
        view = fs.build_header_view(inputs, parts["header"].region.width)
        assert parts["unsaved"].value == view.unsaved_chip
        assert parts["blocked"].value == view.blocked_chip
        assert str(parts["status"].renderable) == view.status_plain
        assert gathered == []


async def test_gathering_header_inputs_walks_no_dom(
    mock_app_instance, roleplay_data, monkeypatch
):
    """The 0.25 s readiness poll gathers the header's inputs on every tick, so
    the gather must not query the DOM: a ``query_one`` for the demand-mounted
    character editor walks the whole screen and raises ``NoMatches`` while
    browsing (about 4 ms a tick, measured in review). The aggregate reads the
    cached editor instead."""
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        queried = []
        original = screen.query_one

        def counting(*args, **kwargs):
            queried.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(screen, "query_one", counting)
        screen._gather_header_inputs()
        assert queried == []


@styled_tiers
@size_matrix
async def test_a_long_name_ellipsises_and_the_kind_is_never_cut(
    styled_tier, roleplay_size, mock_app_instance, monkeypatch
):
    """AC#4, on a real first-paint auto-selection of a long-named character."""
    # Alone in the library, so the first-paint auto-selection (F-031) picks it.
    seed_mock_characters(monkeypatch, [{"id": 1, "name": LONG_NAME, "version": 1}])
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=roleplay_size
    ) as pilot:
        screen = pilot.app.screen
        parts = _parts(screen)
        await wait_until(
            pilot,
            lambda: parts["item"].value == (LONG_NAME, False),
            what="the long name in the header",
        )
        await settle(pilot)
        assert painted_text(screen, parts["subtitle"].region) == "Characters"
        painted = painted_text(screen, parts["item"].region).rstrip()
        assert painted.startswith("› Ser ") and painted.endswith("…"), painted
        assert painted == parts["item"].fitted_text


async def test_status_names_the_data_source_and_never_says_ready(
    mock_app_instance, roleplay_data
):
    """RP-067 and DESIGN.md:115: "Local", or "Server: <label> · read-only"."""
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        status = _parts(screen)["status"]
        assert painted_text(screen, status.region).strip() == "Local"
        pilot.app.runtime_policy = SimpleNamespace(
            state=SimpleNamespace(
                last_known_server_label="home-tldw", active_server_id=None
            )
        )
        screen._set_persona_editor_runtime_source("server")
        screen._update_title()
        await settle(pilot)
        assert (
            painted_text(screen, status.region).strip()
            == "Server: home-tldw · read-only"
        )
        assert "Ready" not in painted_rows(screen)[status.region.y]


async def test_the_header_composes_without_a_ready_chip(
    mock_app_instance, roleplay_data, monkeypatch
):
    """The composed state already names the kind and the data source, so the
    shared header's default "Ready" chip never paints, whatever the order of
    ``DestinationHeader.on_mount`` and the screen's first ``_update_title``."""
    composed = []
    original = DestinationHeader.__init__

    def recording_init(self, state, *args, **kwargs):
        if kwargs.get("id") == "personas-header":
            composed.append(state)
        original(self, state, *args, **kwargs)

    monkeypatch.setattr(DestinationHeader, "__init__", recording_init)
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)):
        pass
    assert [(state.subtitle, state.status_label) for state in composed] == [
        ("Characters", "Local")
    ]


async def test_blocked_chip_follows_destination_readiness_through_the_poll(
    mock_app_instance, roleplay_data
):
    """Shown only while no chat provider is ready for character chats; the
    0.25 s readiness poll clears it with no other refresh."""
    mock_app_instance.app_config = {
        "chat_defaults": {"provider": "openai", "model": "gpt-4o"}
    }
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        blocked = _parts(screen)["blocked"]
        assert blocked.display
        assert painted_text(screen, blocked.region).strip() == (
            "No chat provider · Settings ›"
        )
        mock_app_instance.app_config["api_settings"] = {
            "openai": {"api_key": "unit-test-placeholder-key"}
        }
        await wait_until(pilot, lambda: not blocked.display, what="the chip to go")


async def test_chip_words_take_the_readable_hues_and_the_status_is_neutral(
    mock_app_instance, roleplay_data, monkeypatch
):
    """The colours come from the lazy sheet: the readable status foregrounds
    ($text-warning, $text-error: AA on the panel in every theme, unlike the
    decorative $warning and $error) on the chips, and a neutral status word
    on the panel (no badge)."""
    inputs = fs.RoleplayHeaderInputs(
        mode="characters", unsaved=True, provider_blocked=True
    )
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        monkeypatch.setattr(screen, "_gather_header_inputs", lambda: inputs)
        screen._update_title()
        await settle(pilot)
        parts = _parts(screen)
        variables = pilot.app.get_css_variables()
        assert parts["unsaved"].display and parts["blocked"].display
        unsaved, blocked = parts["unsaved"].styles.color, parts["blocked"].styles.color
        assert unsaved.rgb == Color.parse(variables["text-warning"]).rgb
        assert blocked.rgb == Color.parse(variables["text-error"]).rgb
        assert parts["status"].styles.color not in (unsaved, blocked)
        assert parts["status"].styles.background == parts["header"].styles.background


async def test_clicking_the_blocked_chip_opens_settings_for_the_chat_provider(
    mock_app_instance, roleplay_data, monkeypatch
):
    """The chip deep-links to the provider ``console_handoff_readiness`` checks
    (chat_defaults), through app navigation and so through the leave guard."""
    mock_app_instance.app_config = {
        "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
        "chat_defaults": {"provider": "openai", "model": "gpt-4o"},
    }
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        posted = []
        original = screen.post_message

        def spy(message):
            posted.append(message)
            return original(message)

        monkeypatch.setattr(screen, "post_message", spy)
        assert await pilot.click("#personas-header-blocked")
        await pilot.pause()
        navigations = [m for m in posted if isinstance(m, NavigateToScreen)]
        assert [m.screen_name for m in navigations] == ["settings"]
        assert navigations[0].screen_context.get("provider") == "openai"


async def test_the_blocked_chip_passes_the_leave_guard(
    mock_app_instance, roleplay_data, monkeypatch
):
    """Spec 1.3: the chip "passes the leave guard". On the real app, a click
    with an unsaved draft asks Save / Discard / Stay before leaving, and Stay
    keeps Roleplay and the draft (the mock tier handles no navigation).

    The blocked state is set on the screen's own readiness seam, never taken
    from the process's provider config: in the PR Fast Lane's admission step
    this file runs after test_console_runtime_ownership.py in one pytest
    process, and there the real app read a ready provider and never showed
    the chip (2026-10-08 dry run). A chip that appears only then must be laid
    out before the click, or the click lands elsewhere.
    """
    from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
        RoleplayDraftNavigationDialog,
    )

    async with open_styled_roleplay("full", mock_app_instance, size=(160, 45)) as pilot:
        app = pilot.app
        screen = app.screen
        monkeypatch.setattr(
            screen.preview,
            "console_handoff_readiness",
            lambda: (False, "No chat provider is configured."),
        )
        screen._update_title()
        blocked = _parts(screen)["blocked"]
        await wait_until(
            pilot,
            lambda: blocked.display and blocked.region.width > 0,
            what="the blocked chip, laid out",
        )
        await settle(pilot)
        screen.state.has_unsaved_changes = True
        assert await pilot.click("#personas-header-blocked")
        await wait_until(
            pilot,
            lambda: isinstance(app.screen, RoleplayDraftNavigationDialog),
            what="the Save / Discard / Stay question",
        )
        await pilot.press("escape")  # Stay
        await wait_until(pilot, lambda: app.screen is screen, what="Stay")
        await settle(pilot)
        assert screen.state.has_unsaved_changes is True


@pytest.mark.parametrize(
    "case",
    [
        "attachment",
        "character_visual",
        "persona_visual",
        "persona_shared_visual",
        "visual_operation_inflight",
        "character_save_inflight",
        "persona_save_inflight",
    ],
)
async def test_unsaved_chip_follows_the_aggregate_not_has_unsaved_changes(
    case, mock_app_instance, roleplay_data
):
    """AC#3 (R24): every case here leaves ``has_unsaved_changes`` False, yet
    the aggregate is not clean, so the chip shows; it goes when the domain
    is clean again. The 0.25 s poll is the only refresh used."""
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        chip = _parts(screen)["unsaved"]
        assert not chip.display

        async def dirty():
            if case == "attachment":
                await screen._ensure_center_view("character-editor")
                editor = screen.query_one(PersonasCharacterEditorWidget)
                editor.load_character({"id": 1, "name": "Ada", "image": b"old"})
                editor.set_avatar_image(b"new")
            elif case == "character_visual":
                screen._visual_identity_authoring = object()
            elif case == "persona_visual":
                screen._persona_visual_authoring = SimpleNamespace(dirty=True)
            elif case == "persona_shared_visual":
                screen._persona_shared_visual_identity_authoring = object()
            elif case == "visual_operation_inflight":
                loop = asyncio.get_running_loop()
                screen._visual_identity_operation_task = loop.create_future()
            elif case == "character_save_inflight":
                screen._character_save_inflight = True
            else:
                screen._profile_save_operation_inflight = True

        def clean():
            if case == "attachment":
                screen.query_one(
                    PersonasCharacterEditorWidget
                ).discard_unsaved_attachment()
            elif case == "character_visual":
                screen._visual_identity_authoring = None
            elif case == "persona_visual":
                screen._persona_visual_authoring = None
            elif case == "persona_shared_visual":
                screen._persona_shared_visual_identity_authoring = None
            elif case == "visual_operation_inflight":
                screen._visual_identity_operation_task.cancel()
                screen._visual_identity_operation_task = None
            elif case == "character_save_inflight":
                screen._character_save_inflight = False
            else:
                screen._profile_save_operation_inflight = False

        await dirty()
        assert screen.state.has_unsaved_changes is False
        assert fs.roleplay_has_unsaved_work(screen._aggregate_roleplay_draft_snapshot())
        await wait_until(pilot, lambda: chip.display, what="the unsaved chip")
        assert chip.value == "Unsaved changes"
        clean()
        assert not fs.roleplay_has_unsaved_work(
            screen._aggregate_roleplay_draft_snapshot()
        )
        await wait_until(pilot, lambda: not chip.display, what="the chip to go")


@styled_tiers
@size_matrix
async def test_header_row_paints_ascii_markers_in_ascii_mode(
    styled_tier, roleplay_size, mock_app_instance, monkeypatch
):
    """Spec 4.12: every header glyph goes through the glyph map (no CSS "…")."""
    # Alone in the library, so the first-paint auto-selection (F-031) picks it.
    seed_mock_characters(monkeypatch, [{"id": 1, "name": LONG_NAME, "version": 1}])
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=roleplay_size, ascii_glyphs=True
    ) as pilot:
        screen = pilot.app.screen
        parts = _parts(screen)
        await wait_until(
            pilot,
            lambda: parts["item"].value == (LONG_NAME, False),
            what="the long name in the header",
        )
        await settle(pilot)
        row = painted_rows(screen)[parts["header"].region.y]
        unmapped = set(row) & (set(ASCII_GLYPH_FALLBACKS) | {"…"})
        assert not unmapped, (unmapped, row)
        assert "> Ser " in row and "..." in row, row
