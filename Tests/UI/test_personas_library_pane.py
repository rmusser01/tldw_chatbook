"""Mounted tests for the Personas library pane."""

from dataclasses import fields

import pytest
from textual.app import App

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Input, Static

from Tests.UI.roleplay_frame_harness import (
    highlighted_item_id,
    item_at,
    select_roleplay_item,
    visible_item_ids,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    LibraryRow,
    PersonasLibraryPane,
    RoleplayItemList,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    PersonaActionRequested,
    PersonaEntitySelected,
    PersonaSearchChanged,
)

pytestmark = pytest.mark.asyncio


class LibraryPaneApp(ConsolidatedCSSApp):
    def compose(self):
        yield PersonasLibraryPane(id="personas-library-pane")


async def test_library_row_has_only_selection_and_display_state():
    """Library rows carry selection state, never the human user's identity."""
    assert tuple(field.name for field in fields(LibraryRow)) == (
        "item_id",
        "kind",
        "name",
        "meta",
        "state",
        "short",
    )


async def test_pane_renders_search_toolbar_and_empty_state():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        assert pilot.app.query_one("#personas-library-search", Input)
        assert pilot.app.query_one("#personas-library-new", Button)
        assert pilot.app.query_one("#personas-library-import", Button)
        await pane.update_rows((), total=0, noun="characters")
        await pilot.pause()
        empty = pilot.app.query_one("#personas-library-empty", Static)
        assert "No characters yet" in str(empty.renderable)


async def test_update_rows_renders_rows_and_count():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        rows = (
            LibraryRow(item_id="1", kind="character", name="Detective Sam"),
            LibraryRow(item_id="2", kind="character", name="Tutor"),
        )
        await pane.update_rows(rows, total=2, noun="characters")
        await pilot.pause()
        assert len(visible_item_ids(pilot.app.screen)) == 2
        # OBSOLETE (D14-A): spec 1.5.3 (R24) drops the list's unsaved marker;
        # the old ``is-unsaved`` row-class assertion is deleted.
        # F-033: the plain total moved up into the screen's merged purpose
        # line; the pane count line only speaks for filtered states now.
        count = pilot.app.query_one("#personas-library-count", Static)
        assert str(count.renderable) == ""


async def test_unfiltered_count_line_stays_empty():
    """F-033: the unfiltered total renders once (header purpose line), never
    duplicated at the bottom of the library pane. Singularization of the
    total (task-445) is still covered by the filtered-count tests."""
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            (LibraryRow(item_id="1", kind="character", name="Detective Sam"),),
            total=1,
            noun="characters",
        )
        await pilot.pause()
        count = pilot.app.query_one("#personas-library-count", Static)
        assert str(count.renderable) == ""


async def test_singular_filtered_count_uses_singular_noun():
    """A filtered total of 1 (e.g. '1 of 1 dictionaries') must also read
    singular: '1 of 1 dictionary'."""
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            (LibraryRow(item_id="1", kind="character", name="Only One"),),
            total=1,
            noun="dictionaries",
            filtered=True,
        )
        await pilot.pause()
        count = pilot.app.query_one("#personas-library-count", Static)
        assert str(count.renderable) == "1 of 1 dictionary"


async def test_filtered_count_shows_n_of_m():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            (LibraryRow(item_id="1", kind="character", name="Detective Sam"),),
            total=12,
            noun="characters",
            filtered=True,
        )
        await pilot.pause()
        count = pilot.app.query_one("#personas-library-count", Static)
        assert "1 of 12 characters" in str(count.renderable)


async def test_row_press_posts_persona_entity_selected():
    received = []

    class CaptureApp(LibraryPaneApp):
        def on_persona_entity_selected(self, message: PersonaEntitySelected) -> None:
            received.append(
                (message.entity_kind, message.entity_id, message.entity_name)
            )

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            (LibraryRow(item_id="7", kind="character", name="Detective Sam"),),
            total=1,
            noun="characters",
        )
        await pilot.pause()
        await select_roleplay_item(pilot, pilot.app.screen, "character:7", via="click")
        await pilot.pause()
    assert received == [("character", "7", "Detective Sam")]


async def test_search_input_posts_search_changed_and_new_posts_create_action():
    searches = []
    actions = []

    class CaptureApp(LibraryPaneApp):
        def on_persona_search_changed(self, message: PersonaSearchChanged) -> None:
            searches.append(message.query)

        def on_persona_action_requested(self, message: PersonaActionRequested) -> None:
            actions.append(message.action)

    app = CaptureApp()
    async with app.run_test() as pilot:
        search = pilot.app.query_one("#personas-library-search", Input)
        search.value = "sam"
        await pilot.pause()
        await pilot.click("#personas-library-new")
        await pilot.pause()
        await pilot.click("#personas-library-import")
        await pilot.pause()
    assert searches[-1] == "sam"
    assert actions == ["create", "import"]


async def test_mark_active_row_applies_is_active_to_selected_only():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            (
                LibraryRow(item_id="1", kind="character", name="Detective Sam"),
                LibraryRow(item_id="2", kind="character", name="Tutor"),
            ),
            total=2,
            noun="characters",
        )
        await pilot.pause()
        pane.mark_active_row("character", "2")
        # SEMANTIC (D14-A): the active marker is the gutter glyph, not a class.
        assert item_at(pilot.app.screen, "character:2").active
        assert not item_at(pilot.app.screen, "character:1").active
        # The list highlight follows the active row for keyboard continuity.
        assert highlighted_item_id(pilot.app.screen) == "character:2"


async def test_toolbar_and_rows_carry_shared_flat_button_classes():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        assert pilot.app.query_one("#personas-library-new", Button).has_class(
            "console-action-secondary"
        )
        assert pilot.app.query_one("#personas-library-import", Button).has_class(
            "console-action-secondary"
        )
        await pane.update_rows(
            (LibraryRow(item_id="1", kind="character", name="Detective Sam"),),
            total=1,
            noun="characters",
        )
        await pilot.pause()
        # SEMANTIC (D14-A): rows are option lines styled by the list's
        # component classes; the per-row widget classes no longer exist.
        assert isinstance(
            pilot.app.query_one("#personas-library-rows"), RoleplayItemList
        )
        assert visible_item_ids(pilot.app.screen) == ("character:1",)


async def test_active_row_keeps_subdued_and_is_active_markers():
    """The loaded row reads as loaded without a per-row widget class.

    On dev the row was a ``ListItem`` whose ``is-active`` class a sheet rule
    styled; since frame slice B2a (spec 1.5.3) rows are option lines, and
    the loaded one carries the gutter's loaded glyph and a bold name in its
    own prompt, which this test reads.
    """
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            (
                LibraryRow(item_id="1", kind="character", name="Detective Sam"),
                LibraryRow(item_id="2", kind="character", name="Tutor"),
            ),
            total=2,
            noun="characters",
        )
        await pilot.pause()
        pane.mark_active_row("character", "1")
        # SEMANTIC (D14-A): spec 1.5.3 renders the loaded row as gutter
        # glyph + bold name, so the class-cascade contract becomes a prompt check.
        active = item_at(pilot.app.screen, "character:1")
        assert active.active
        prompt = (
            pilot.app.query_one("#personas-library-rows").get_option_at_index(0).prompt
        )
        assert any(str(span.style) == "bold" for span in prompt.spans)
        assert not item_at(pilot.app.screen, "character:2").active


async def test_set_mode_toggles_import_button_and_empty_copy():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        import_button = pilot.app.query_one("#personas-library-import", Button)
        pane.set_mode("personas")
        assert import_button.display is False
        pane.set_mode("characters")
        assert import_button.display is True
        pane.set_mode("personas")
        await pane.update_rows((), total=0, noun="personas")
        await pilot.pause()
        empty = pilot.app.query_one("#personas-library-empty", Static)
        copy = str(empty.renderable)
        assert "No personas yet" in copy
        assert "Import" not in copy


async def test_update_rows_twice_in_same_tick_does_not_crash():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        rows_a = (LibraryRow(item_id="1", kind="character", name="First"),)
        rows_b = (LibraryRow(item_id="2", kind="character", name="Second"),)
        await pane.update_rows(rows_a, total=1, noun="characters")
        await pane.update_rows(rows_b, total=1, noun="characters")
        await pilot.pause()
        assert visible_item_ids(pilot.app.screen) == ("character:2",)
        assert item_at(pilot.app.screen, 0).name == "Second"


async def test_arrow_navigation_and_enter_selects():
    """Down/Down highlights the second row without selecting; Enter selects it."""
    received = []

    class CaptureApp(LibraryPaneApp):
        def on_persona_entity_selected(self, message: PersonaEntitySelected) -> None:
            received.append(
                (message.entity_kind, message.entity_id, message.entity_name)
            )

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            (
                LibraryRow(item_id="1", kind="character", name="Detective Sam"),
                LibraryRow(item_id="2", kind="character", name="Tutor"),
                LibraryRow(item_id="3", kind="character", name="Navigator"),
            ),
            total=3,
            noun="characters",
        )
        await pilot.pause()
        pilot.app.query_one("#personas-library-rows").focus()
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("down")
        await pilot.pause()
        # Arrow browsing alone must never fire a selection (guards/dirty
        # prompts stay quiet until the user explicitly commits with Enter).
        assert received == []
        await pilot.press("enter")
        await pilot.pause()
    assert received == [("character", "2", "Tutor")]


async def test_search_enter_jumps_to_results():
    """Enter in the search input focuses the list and highlights the first row."""
    received = []

    class CaptureApp(LibraryPaneApp):
        def on_persona_entity_selected(self, message: PersonaEntitySelected) -> None:
            received.append(message.entity_id)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            (
                LibraryRow(item_id="1", kind="character", name="Detective Sam"),
                LibraryRow(item_id="2", kind="character", name="Tutor"),
            ),
            total=2,
            noun="characters",
        )
        await pilot.pause()
        search = pilot.app.query_one("#personas-library-search", Input)
        search.focus()
        await pilot.pause()
        search.value = "sam"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.focused is pilot.app.query_one("#personas-library-rows")
        assert highlighted_item_id(pilot.app.screen) == "character:1"
        # Jumping into the results highlights only; it must not select.
        assert received == []


async def test_search_enter_with_no_rows_does_not_crash():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await pane.update_rows((), total=0, noun="characters")
        await pilot.pause()
        search = pilot.app.query_one("#personas-library-search", Input)
        search.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.focused is pilot.app.query_one("#personas-library-rows")


async def test_highlighted_row_carries_textual_highlight_class():
    """The browse cursor is ``OptionList.highlighted``, painted by a component class.

    On dev Textual set ``-highlight`` on the browsed ``ListItem``; since frame
    slice B2a the cursor is the list's highlighted option, and the Roleplay
    sheet paints ``option-list--option-highlighted`` only while the list has
    focus (spec 1.5.3). This pins the component-class name, so a Textual
    upgrade that renames it fails here instead of silently unpainting.
    """
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            (
                LibraryRow(item_id="1", kind="character", name="Alpha"),
                LibraryRow(item_id="2", kind="character", name="Beta"),
            ),
            total=2,
            noun="characters",
        )
        await pilot.pause()
        pilot.app.query_one("#personas-library-rows").focus()
        await pilot.pause()
        # After pressing down the cursor moves to index 0
        # (Textual convention: first press sets the highlight).
        await pilot.press("down")
        await pilot.pause()
        # SEMANTIC (D14-A): the cursor is OptionList.highlighted, painted with
        # the ``option-list--option-highlighted`` component class (spec 1.5.3
        # pins an id-scoped :focus rule for it in B2a's own tests).
        assert highlighted_item_id(pilot.app.screen) == "character:1"
        assert "option-list--option-highlighted" in RoleplayItemList.COMPONENT_CLASSES


async def test_is_active_and_highlight_can_coexist():
    """The loaded row can also hold the browse cursor (frame slice B2a).

    Loaded is the gutter glyph in the row's prompt; the cursor is the list's
    highlighted option. ``mark_active_row`` moves the cursor to the loaded
    row, so both hold for the same row at once.
    """
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            (
                LibraryRow(item_id="1", kind="character", name="Alpha"),
                LibraryRow(item_id="2", kind="character", name="Beta"),
            ),
            total=2,
            noun="characters",
        )
        await pilot.pause()
        pilot.app.query_one("#personas-library-rows").focus()
        await pilot.pause()
        # Press down twice so the cursor is on index 1 (Beta), then mark it
        # active — both ``is-active`` and ``-highlight`` must coexist.
        await pilot.press("down")
        await pilot.press("down")
        await pilot.pause()
        pane.mark_active_row("character", "2")
        await pilot.pause()
        # SEMANTIC (D14-A): active = gutter glyph, cursor = highlighted option.
        assert item_at(pilot.app.screen, "character:2").active
        assert highlighted_item_id(pilot.app.screen) == "character:2"


async def test_colliding_item_ids_render_without_crash():
    received = []

    class CaptureApp(LibraryPaneApp):
        def on_persona_entity_selected(self, message: PersonaEntitySelected) -> None:
            received.append(message.entity_id)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            (
                LibraryRow(item_id="a.b", kind="character", name="Dotted"),
                LibraryRow(item_id="a b", kind="character", name="Spaced"),
            ),
            total=2,
            noun="characters",
        )
        await pilot.pause()
        ids = visible_item_ids(pilot.app.screen)
        assert len(ids) == 2
        for option_id in ids:
            await select_roleplay_item(pilot, pilot.app.screen, option_id, via="click")
            await pilot.pause()
    assert received == ["a.b", "a b"]


# ===== F-040: marks (multi-select) and keyboard sort =====


async def _two_character_rows(pane: PersonasLibraryPane) -> None:
    await pane.update_rows(
        (
            LibraryRow(item_id="1", kind="character", name="Detective Sam"),
            LibraryRow(item_id="2", kind="character", name="Lab Assistant"),
        ),
        total=2,
        noun="characters",
    )


async def test_m_key_marks_rows_and_posts_marks_changed():
    received = []

    class CaptureApp(LibraryPaneApp):
        def on_persona_marks_changed(self, message) -> None:
            received.append(message.marks)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await _two_character_rows(pane)
        await pilot.pause()
        pilot.app.query_one("#personas-library-rows").focus()
        await select_roleplay_item(pilot, pilot.app.screen, index=0, load=False)
        await pilot.pause()
        await pilot.press("m")
        await pilot.pause()
        assert received[-1] == (("character", "1", "Detective Sam"),)
        # The marked row paints the mark glyph in its gutter, the unmarked row
        # does not, and the pane reports the mark.
        assert item_at(pilot.app.screen, 0).marked
        assert item_at(pilot.app.screen, 0).painted_mark
        assert not item_at(pilot.app.screen, 1).painted_mark
        count = pilot.app.query_one("#personas-library-count", Static)
        assert str(count.renderable) == "1 marked"
        await pilot.press("m")  # toggles off
        await pilot.pause()
        assert received[-1] == ()
        assert not item_at(pilot.app.screen, 0).marked
        assert not item_at(pilot.app.screen, 0).painted_mark
        assert str(count.renderable) == ""


async def test_marks_prune_when_rows_vanish_on_refresh():
    received = []

    class CaptureApp(LibraryPaneApp):
        def on_persona_marks_changed(self, message) -> None:
            received.append(message.marks)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await _two_character_rows(pane)
        await pilot.pause()
        pilot.app.query_one("#personas-library-rows").focus()
        await select_roleplay_item(pilot, pilot.app.screen, index=0, load=False)
        await pilot.pause()
        await pilot.press("m")
        await pilot.pause()
        assert received[-1] == (("character", "1", "Detective Sam"),)
        # A refresh that no longer contains the marked row drops the mark.
        await pane.update_rows(
            (LibraryRow(item_id="2", kind="character", name="Lab Assistant"),),
            total=1,
            noun="characters",
        )
        await pilot.pause()
        assert received[-1] == ()
        count = pilot.app.query_one("#personas-library-count", Static)
        assert str(count.renderable) == ""


async def test_marks_clear_on_mode_switch():
    received = []

    class CaptureApp(LibraryPaneApp):
        def on_persona_marks_changed(self, message) -> None:
            received.append(message.marks)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await _two_character_rows(pane)
        await pilot.pause()
        pilot.app.query_one("#personas-library-rows").focus()
        await select_roleplay_item(pilot, pilot.app.screen, index=1, load=False)
        await pilot.pause()
        await pilot.press("m")
        await pilot.pause()
        assert received[-1] == (("character", "2", "Lab Assistant"),)
        pane.set_mode("dictionaries")
        await pilot.pause()
        assert received[-1] == ()


async def test_s_key_cycles_sort_only_when_sort_applies():
    received = []

    class CaptureApp(LibraryPaneApp):
        def on_persona_sort_cycle_requested(self, message) -> None:
            received.append(message)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        await _two_character_rows(pane)
        await pilot.pause()
        pilot.app.query_one("#personas-library-rows").focus()
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        assert len(received) == 1
        # Dictionaries mode has no sort control - the key is a no-op there.
        pane.set_mode("dictionaries")
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        assert len(received) == 1
        # The sort button discloses the key.
        assert "(s)" in str(
            pilot.app.query_one("#personas-library-sort", Button).tooltip
        )


async def test_sync_control_layout_logs_and_keeps_state_on_failure(monkeypatch):
    """Qodo review: a toolbar width-measurement failure must be logged (not
    silently swallowed) and must leave the previous layout in place."""
    from unittest.mock import Mock

    from loguru import logger as loguru_logger

    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        pane.set_class(True, "personas-library-stacked-controls")
        records: list[str] = []
        sink = loguru_logger.add(lambda m: records.append(str(m)), level="DEBUG")
        try:
            monkeypatch.setattr(
                pane,
                "_required_toolbar_row_width",
                Mock(side_effect=RuntimeError("boom")),
            )
            pane._sync_control_layout()  # must not raise
            # Fallback: the previous layout class is untouched.
            assert pane.has_class("personas-library-stacked-controls")
        finally:
            loguru_logger.remove(sink)
    assert any("toolbar" in record.lower() for record in records), (
        f"expected a debug/warning log about the toolbar layout failure; got {records}"
    )
