"""Library critique #10, group ``layout``: panes, grips and the hand-off.

Covers task-32355 (grip labels, rail beside the note editor), task-32359
(rail focus is a shape, not a second blue), task-32360 (the narrow-stage
return and clipped copy), task-32361 (the Conversations width split) and
task-32107 (link-on-use for the Conversations hand-off).
"""

from __future__ import annotations

import dataclasses

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    LibraryAdaptiveReaderPaneGrip,
)


CONVERSATION_ID = "conv-1"
CONVERSATION_TITLE = "Alpha planning"
ACTIVE_WORKSPACE_NAME = "Local Default"


SECOND_CONVERSATION_ID = "conv-2"
SECOND_CONVERSATION_TITLE = "Beta planning"


def _conversations_host(
    *,
    linked: bool = False,
    reason_code: str = "not_in_active_workspace",
    second: bool = False,
):
    """Build a Library harness sitting on Conversations with one saved chat.

    The real ``workspace_registry_service`` is used rather than a recording
    fake: the point of link-on-use is that the link actually changes what
    ``library_item_context_handoff`` answers, which only a real registry can
    prove.

    Args:
        linked: Whether the conversation already belongs to the active
            workspace (so the hand-off is not blocked at all).
        reason_code: ``"not_in_active_workspace"`` for the link-resolvable
            block; ``"no_active_workspace"`` leaves the profile without an
            active workspace, the block a link cannot resolve.
        second: Seed a second conversation too, so a test can drive a real
            second selection through the load path.

    Returns:
        The mounted-ready ``LibraryHarness``, with ``registry`` attached.
    """
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _build_test_app,
        _seed_conversations,
    )
    from tldw_chatbook.UI.Screens.library_screen import (
        LIBRARY_ROW_BROWSE_CONVERSATIONS,
        LibraryScreen,
    )

    app = _build_test_app()
    rows = [
        {
            "id": CONVERSATION_ID,
            "title": CONVERSATION_TITLE,
            "version": 4,
            "message_count": 1,
            "last_modified": "2026-09-01T12:00:00Z",
        }
    ]
    if second:
        rows.append(
            {
                "id": SECOND_CONVERSATION_ID,
                "title": SECOND_CONVERSATION_TITLE,
                "version": 2,
                "message_count": 1,
                "last_modified": "2026-09-01T11:00:00Z",
            }
        )
    _seed_conversations(app, rows)
    registry = app.workspace_registry_service
    registry.clear_active_workspace()
    if reason_code != "no_active_workspace":
        registry.create_workspace(
            workspace_id="workspace-a", name=ACTIVE_WORKSPACE_NAME
        )
        registry.set_active_workspace("workspace-a")
        if linked:
            registry.link_membership(
                "workspace-a",
                item_type="conversation",
                item_id=CONVERSATION_ID,
                title=CONVERSATION_TITLE,
            )
    screen = LibraryScreen(app)
    screen.restore_state({"library_selected_row_id": LIBRARY_ROW_BROWSE_CONVERSATIONS})
    host = LibraryHarness(app, screen=screen)
    host.registry = registry
    host.tldw_app = app
    return host


async def _open_first_conversation(host, pilot):
    """Settle the Conversations shell with the seeded conversation loaded."""
    from Tests.UI.test_library_shell import (
        _active_library_screen,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    await _wait_for_selector(screen, pilot, "#library-conversation-reader")
    for _ in range(40):
        if screen._conversations_state.reader_state.loaded_id:
            break
        await pilot.pause(0.01)
    await pilot.pause()
    return screen


# --------------------------------------------------------------------------
# task-32361: the Conversations width split
# --------------------------------------------------------------------------


# The AC says "200 columns and wider", so the boundary is pinned, not just
# the comfortable case. Floors are the measured widths, one per size.
_MAJORITY_SIZES = {(200, 52): 90, (235, 52): 100}


@pytest.mark.parametrize("size, reader_floor", sorted(_MAJORITY_SIZES.items()))
@pytest.mark.asyncio
async def test_the_conversation_reader_takes_the_majority_of_a_wide_terminal(
    size, reader_floor
) -> None:
    """task-32361 AC#1: the reader wins the width once something is open.

    Measured before the fix at 235x52: reader 44, list 137 -- the layout
    resolved while the Reader was still empty (``reader_has_item=False``
    hands its columns to the list) and nothing re-resolved it once the
    selection settled. After: 132/49 at 235 and 97/49 at the AC's own 200.
    """
    host = _conversations_host(linked=True)
    async with host.run_test(size=size) as pilot:
        screen = await _open_first_conversation(host, pilot)
        reader = screen.query_one("#library-conversation-reader")
        items = screen.query_one("#library-conversations-list")
        assert reader.region.width > items.region.width, (reader.region, items.region)
        assert reader.region.width >= reader_floor, reader.region


@pytest.mark.asyncio
async def test_an_empty_conversation_reader_still_gives_its_width_away() -> None:
    """task-32217 stays intact: nothing open means the list takes the stage."""
    host = _conversations_host(linked=True)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_conversation(host, pilot)
        screen._conversations_state.reader_state = dataclasses.replace(
            screen._conversations_state.reader_state, selected_id=None
        )
        screen._sync_library_conversation_reader()
        await pilot.pause()
        await pilot.pause()
        items = screen.query_one("#library-conversations-list")
        reader = screen.query_one("#library-conversation-reader")
        assert items.region.width > reader.region.width, (items.region, reader.region)


# --------------------------------------------------------------------------
# task-32355 / task-32360 AC#3: the pane grips
# --------------------------------------------------------------------------


def _painted_column(app, widget) -> str:
    """Return what a one-column-wide strip of ``widget``'s region paints."""
    strips = list(app.screen._compositor.render_strips())
    return "".join(
        strips[y].crop(widget.region.x, widget.region.right).text.strip()
        for y in range(widget.region.y, widget.region.bottom)
    )


@pytest.mark.parametrize("size", [(235, 52), (100, 30), (60, 24)])
@pytest.mark.asyncio
async def test_both_pane_grips_paint_their_name(size) -> None:
    """task-32355 AC#1/AC#2: the handle says what it opens, at every width.

    The name goes DOWN the five-cell column -- "Nav" for the Library pane
    (the name ``Docs/User_Guide/library.md`` uses for this handle) and the
    pane's own label for the items pane.
    """
    host = _conversations_host(linked=True)
    async with host.run_test(size=size) as pilot:
        screen = await _open_first_conversation(host, pilot)
        grips = {
            grip.pane: grip for grip in screen.query(LibraryAdaptiveReaderPaneGrip)
        }
        assert set(grips) == {"library", "items"}
        assert grips["library"].painted_name() == "Nav"
        assert grips["items"].painted_name() == "Items"
        assert _painted_column(host.app, grips["library"]).startswith("Nav")
        assert _painted_column(host.app, grips["items"]).startswith("Items")


@pytest.mark.parametrize("size", [(235, 52), (100, 30), (60, 24)])
@pytest.mark.asyncio
async def test_the_grips_never_overlap_the_panes_beside_them(size) -> None:
    """task-32360 AC#3: measured geometry, before any paint claim."""
    host = _conversations_host(linked=True)
    async with host.run_test(size=size) as pilot:
        screen = await _open_first_conversation(host, pilot)
        grips = {
            grip.pane: grip for grip in screen.query(LibraryAdaptiveReaderPaneGrip)
        }
        regions = {
            pane: grip.region
            for pane, grip in grips.items()
        }
        # (review fix round 1) The list pane and the rail are the neighbours
        # B reported overpainted, so they have to be IN the measurement --
        # round 0 measured only the grips against the reader.
        neighbours = [
            widget
            for selector in (
                "#library-rail",
                "#library-canvas",
                "#library-conversation-reader",
            )
            for widget in screen.query(selector)
        ]
        occupied = sorted(
            (widget.region.x, widget.region.right, widget.id)
            for widget in (*grips.values(), *neighbours)
            if widget.region.width and widget.display
        )
        assert len(occupied) >= 3, occupied
        for (_, left_right, left_id), (right_x, _, right_id) in zip(
            occupied, occupied[1:]
        ):
            assert left_right <= right_x, (regions, left_id, right_id, occupied)

        # Region assertions are blind to paint-over
        # (backlog/docs/lessons-live-verification.md), so read the frame too:
        # the arrow run must not appear on any column outside a grip. B's
        # capture put "<---" at char column 42 of a 235-cell frame -- which
        # is the Library grip's OWN column (x=41), not the content area.
        strips = list(host.app.screen._compositor.render_strips())
        grip_columns = {
            column
            for grip in grips.values()
            for column in range(grip.region.x, grip.region.right)
        }
        for row, strip in enumerate(strips):
            text = strip.text
            for column in range(len(text) - 3):
                if text[column : column + 4] in ("<---", "--->"):
                    assert column in grip_columns, (row, column, text)


# --------------------------------------------------------------------------
# task-32107: link-on-use for the Conversations hand-off
# --------------------------------------------------------------------------


def _loaded_reader_state():
    """A settled, complete transcript -- the crit8 fixture, reused."""
    from tldw_chatbook.Library.library_conversation_reader_state import (
        ConversationMessageView,
        ConversationReaderState,
    )

    return ConversationReaderState(
        selected_id=CONVERSATION_ID,
        selected_version=4,
        loaded_id=CONVERSATION_ID,
        loaded_version=4,
        loaded_generation=2,
        generation=2,
        messages=(
            ConversationMessageView(
                "message-a", "user", "2026-09-01T12:01:00Z", "revision-a", 5, "hello"
            ),
        ),
        message_total=1,
        complete=True,
    )


BLOCK_SENTENCE = (
    "This conversation is not in this workspace. Pressing this adds it to "
    "the active workspace first, and you can undo that."
)

LINK_RECEIPT = (
    f"✓ linked · {ACTIVE_WORKSPACE_NAME} · this conversation can now be "
    "used in Console"
)


def _memberships(host) -> list:
    return list(
        host.registry.get_item_memberships(
            item_type="conversation", item_id=CONVERSATION_ID
        )
    )


@pytest.mark.asyncio
async def test_use_as_source_links_an_unlinked_conversation_and_proceeds() -> None:
    """task-32107 AC#2: one gesture links and stages, with a receipt."""
    host = _conversations_host(linked=False)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_conversation(host, pilot)
        source = screen.query_one("#library-conversation-use-source", Button)
        assert not source.disabled, "a link-resolvable block no longer blocks"
        assert str(source.label) == "Use as source"
        blocked = screen.query_one(
            "#library-conversation-open-console-blocked", Static
        )
        assert str(blocked.renderable) == BLOCK_SENTENCE, str(blocked.renderable)
        assert not _memberships(host)
        staged: list = []
        host.tldw_app.open_chat_with_handoff = (
            lambda payload, **kwargs: staged.append(payload)
        )

        source.press()
        await pilot.pause()
        await pilot.pause()

        receipt = screen.query_one("#library-conversation-link-receipt", Static)
        assert str(receipt.renderable) == LINK_RECEIPT, str(receipt.renderable)
        assert receipt.display is True
        assert screen.query_one("#library-conversation-link-undo", Button).display
        assert _memberships(host)
        assert staged, "the hand-off still ran"
        assert staged[0].source_id == CONVERSATION_ID


@pytest.mark.asyncio
async def test_undo_removes_the_membership_the_press_added() -> None:
    """task-32107 AC#2: the widening is reversible from where it happened."""
    host = _conversations_host(linked=False)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_conversation(host, pilot)
        screen.query_one("#library-conversation-use-source", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert _memberships(host)

        screen.query_one("#library-conversation-link-undo", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert not _memberships(host)
        assert not screen.query_one(
            "#library-conversation-link-receipt", Static
        ).display
        assert not screen.query_one(
            "#library-conversation-link-undo", Button
        ).display


@pytest.mark.asyncio
async def test_a_block_a_link_cannot_resolve_still_refuses(widget_pilot) -> None:
    """task-32107 AC#2: the aggregate fallback keeps refusing, exactly as before.

    Driven through the reader's own metadata seam because that is where the
    non-linkable block arrives: ``library_item_context_handoff`` treats "no
    active workspace" as nothing to gate against, so the block a link cannot
    resolve is the aggregate ``LIBRARY_GENERIC_WORKSPACE_BLOCK`` fallback for
    an item missing from the row model.
    """
    from tldw_chatbook.Widgets.Library import LibraryConversationReader

    async with await widget_pilot(
        LibraryConversationReader,
        state=_loaded_reader_state(),
        loaded_metadata={
            "title": CONVERSATION_TITLE,
            "_workspace_block": "blocked for this workspace",
            "_workspace_block_linkable": False,
            "_workspace_block_detail": (
                "Select an active workspace before using this item in Console."
            ),
        },
        id="library-conversation-reader",
    ) as pilot:
        source = pilot.app.query_one("#library-conversation-use-source", Button)
        assert source.disabled
        assert str(source.label).startswith("○ ")
        assert not pilot.app.query_one(
            "#library-conversation-link-workspace", Button
        ).display
        assert "Select an active workspace" in str(source.tooltip)
        assert not pilot.app.query_one(
            "#library-conversation-link-receipt", Static
        ).display


@pytest.mark.asyncio
async def test_the_receipt_does_not_survive_a_different_conversation() -> None:
    """The receipt belongs to the conversation it was written for."""
    host = _conversations_host(linked=False, second=True)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_conversation(host, pilot)
        screen.query_one("#library-conversation-use-source", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert screen.query_one("#library-conversation-link-receipt", Static).display

        # (review fix round 1) Drive a REAL second selection through the load
        # path, not a hand-assigned mapping: the claim is that the load
        # replaces the whole mapping, so the load is what has to be exercised.
        screen.query_one("#library-conversation-row-1", Button).press()
        for _ in range(60):
            if (
                screen._conversations_state.reader_state.loaded_id
                == SECOND_CONVERSATION_ID
            ):
                break
            await pilot.pause(0.01)
        await pilot.pause()
        assert (
            screen._conversations_state.reader_state.loaded_id
            == SECOND_CONVERSATION_ID
        ), screen._conversations_state.reader_state.loaded_id
        assert not screen.query_one(
            "#library-conversation-link-receipt", Static
        ).display


# --------------------------------------------------------------------------
# task-32360 AC#1/AC#2: the narrow stage's return, and clipped copy
# --------------------------------------------------------------------------


def _painted_footer(chips, size) -> str:
    """Return what the REAL footer widget paints for ``chips`` at ``size``.

    The Library harness mounts no app chrome, so the widget that actually
    elides -- ``AppFooterStatus`` -- is driven with the very set the screen
    registered. This is the rendered text, not the registered tuple: the
    registered tuple is exactly what hid the AC#1 regression in round 1.
    """
    from Tests.UI.test_chrome_ux_fixes import _FooterHarness, _shown_text

    async def _run() -> str:
        app = _FooterHarness()
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            app.footer.set_workbench_shortcuts(source="library", shortcuts=chips)
            await pilot.pause()
            return _shown_text(app.footer)

    return _run()


@pytest.mark.parametrize("field_focused", [False, True])
@pytest.mark.asyncio
async def test_the_narrow_stage_return_is_painted_in_both_focus_states(
    field_focused: bool,
) -> None:
    """task-32360 AC#1: the return is PAINTED at 60x24, typing or not.

    Fix round 1 (review P1). The round-0 pin read the registered tuple, which
    is blind to the thing that actually decides: at width 60 the real footer
    paints exactly ONE context chip and only about 24 rendered characters of
    it. Measured there: "esc back to Library" (19) survives; "esc typing ·
    back to Library" (28) and Task 1's wide-footer form collapse the WHOLE
    context to "…". So the field-state marker and the return cannot both
    paint, and the chip that survives has to name what Escape actually does.

    It does not leave the field here. ``library_narrow_stage_return`` is
    declared above ``library_blur_text_field``, and with an Input focused on
    this stage ``check_action("library_blur_text_field")`` is measurably
    False while ``check_action("library_narrow_stage_return")`` is True --
    so "esc leaves field" would be a dead key. Nothing the caret would
    swallow is advertised either: the typing block above has already dropped
    every printable-key chip.
    """
    from unittest.mock import patch

    from textual.widgets import Input

    from Tests.UI.test_library_crit9_shell import (
        NARROW_TEST_SIZE,
        _active_library_screen,
        _library_host,
        _wait_for_condition,
        _wait_for_library_shell,
    )

    host = _library_host()
    async with host.run_test(size=NARROW_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media").press()
        await _wait_for_condition(
            pilot,
            lambda: ("esc", "back to Library")
            in tuple((screen._footer_shortcut_registration or ("", ()))[1]),
            message="The registered footer never named the return.",
        )
        if field_focused:
            # This harness's own entry-focus arm takes the key back from any
            # field a test focuses, and the branch under test reads
            # ``self.focused`` and nothing else -- state it directly rather
            # than racing that arm.
            with patch.object(type(screen), "focused", property(lambda _s: Input())):
                chips = screen._library_footer_shortcuts_for_current_state()
                # The key the chip names must be the one Escape performs.
                assert screen.check_action("library_narrow_stage_return", ()) is True
                assert screen.check_action("library_blur_text_field", ()) is False
        else:
            chips = screen._library_footer_shortcuts_for_current_state()
        assert chips[0] == ("esc", "back to Library"), chips

    shown = await _painted_footer(chips, NARROW_TEST_SIZE)
    assert "esc back to Library" in shown, shown
    assert "…" not in shown, shown


# --------------------------------------------------------------------------
# task-32359: the focused rail row differs from the active one by shape
# --------------------------------------------------------------------------


_THICK_LEFT_GLYPH = "█"


class _RailRowsHost(App):
    """Two real rail rows under the production stylesheet: one active.

    The shape of ``library_rail.py``'s rows exactly: a ``Button`` classed
    ``library-rail-row``, the active destination additionally carrying
    ``library-rail-row-selected``. ``AUTO_FOCUS = None`` keeps both genuinely
    blurred until a test moves focus, so the observed cue is the one a real
    Tab/arrow produces.
    """

    AUTO_FOCUS = None

    def __init__(self) -> None:
        from Tests.UI.consolidated_css import APP_STYLESHEETS

        self.CSS_PATH = [str(path) for path in APP_STYLESHEETS]
        super().__init__()

    def compose(self):
        from textual.containers import Vertical

        with Vertical():
            yield Button("Media", id="rail-media", classes="library-rail-row")
            yield Button(
                "▸ Conversations",
                id="rail-conversations",
                classes="library-rail-row library-rail-row-selected",
            )


def _rail_rows(app) -> list[str]:
    return [
        "".join(segment.text for segment in strip)
        for strip in app.screen._compositor.render_strips()
    ]


def _leftmost(rows: list[str], widget) -> str:
    line = rows[widget.region.y]
    return line[widget.region.x] if widget.region.x < len(line) else ""


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(235, 52), (100, 30)])
async def test_the_focused_rail_row_is_a_shape_not_a_second_blue(size) -> None:
    """task-32359 AC#1/AC#2: focus paints the house bar; active does not.

    B measured both states as bold + underline over backgrounds three RGB
    units apart (rgb(25,68,102) vs rgb(28,70,102)) -- indistinguishable, and
    colour-only. Focus everywhere else on this screen is the ``█`` left bar
    (task-31983), so the rail stops being the exception.
    """
    app = _RailRowsHost()
    async with app.run_test(size=size) as pilot:
        media = app.query_one("#rail-media", Button)
        conversations = app.query_one("#rail-conversations", Button)

        media.focus()
        await pilot.pause()
        rows = _rail_rows(app)
        assert _leftmost(rows, media) == _THICK_LEFT_GLYPH, rows[media.region.y]
        assert _leftmost(rows, conversations) != _THICK_LEFT_GLYPH
        assert media.styles.border_left[0]
        assert not conversations.styles.border_left[0]

        conversations.focus()
        await pilot.pause()
        rows = _rail_rows(app)
        assert _leftmost(rows, conversations) == _THICK_LEFT_GLYPH
        assert _leftmost(rows, media) != _THICK_LEFT_GLYPH
        # The active row keeps its own treatment and its whole label.
        assert "Conversations" in rows[conversations.region.y]


@pytest.mark.asyncio
async def test_only_one_context_chip_paints_at_sixty_columns() -> None:
    """task-32360 AC#1, fix round 1: the budget the chip choice turns on.

    Round 1 of this branch put the field-state marker at the head so it would
    be first, as every wider surface has it. Measured here, that silently
    dropped the return: the footer keeps ONE chip at this width, so "first"
    and "only" are the same thing. Pinned so a future re-ordering has to see
    the cost, and so the ~24-character budget is a measurement in the suite
    rather than a number in a report.
    """
    cases = {
        (("esc", "back to Library"),): True,
        (("esc", "back to Library"), ("", "typing in field")): True,
        (("", "typing in field"), ("esc", "back to Library")): False,
        (("esc", "typing · back to Library"),): False,
        (("", "typing in field · after esc: back to Library"),): False,
    }
    for chips, return_survives in cases.items():
        shown = await _painted_footer(chips, (60, 24))
        assert ("esc back to Library" in shown) is return_survives, (chips, shown)


@pytest.mark.asyncio
async def test_undo_unlinks_the_workspace_the_receipt_names() -> None:
    """task-32107 (review fix round 1): Undo reverses THIS press, not the moment.

    Creating a workspace from the rail activates it and recomposes the
    reader from the same metadata mapping, so a standing receipt can outlive
    the active workspace it was written for. Undo read
    ``get_active_workspace()`` and would have removed a membership the press
    never added -- a data effect in the opposite direction of the one the
    user asked to reverse.
    """
    host = _conversations_host(linked=False)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_conversation(host, pilot)
        screen.query_one("#library-conversation-use-source", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert _memberships(host)

        # The active workspace moves on while the receipt stands.
        host.registry.create_workspace(workspace_id="workspace-b", name="Other")
        host.registry.set_active_workspace("workspace-b")
        host.registry.link_membership(
            "workspace-b",
            item_type="conversation",
            item_id=CONVERSATION_ID,
            title=CONVERSATION_TITLE,
        )
        screen._invalidate_library_workspace_depth_state()
        await pilot.pause()

        screen.query_one("#library-conversation-link-undo", Button).press()
        await pilot.pause()
        await pilot.pause()

        remaining = {
            membership.workspace_id for membership in _memberships(host)
        }
        assert remaining == {"workspace-b"}, remaining


@pytest.mark.asyncio
async def test_the_reason_line_sits_directly_under_the_action_it_describes(
    widget_pilot,
) -> None:
    """task-32107 (review fix round 1): "Pressing this" needs the right "this".

    The sentence used to be yielded after the Archive/Restore pair, so the
    control directly above it was "Archive conversation" and the deixis
    pointed at the wrong button.
    """
    from tldw_chatbook.Widgets.Library import LibraryConversationReader

    async with await widget_pilot(
        LibraryConversationReader,
        state=_loaded_reader_state(),
        loaded_metadata={
            "title": CONVERSATION_TITLE,
            "_workspace_block": "not in this workspace",
            "_workspace_block_linkable": True,
        },
        id="library-conversation-reader",
    ) as pilot:
        reader = pilot.app.query_one(
            "#library-conversation-reader", LibraryConversationReader
        )
        order = [child.id for child in reader.walk_children() if child.id]
        assert (
            order.index("library-conversation-open-console-blocked")
            == order.index("library-conversation-use-source") + 1
        ), order


@pytest.mark.asyncio
async def test_the_receipt_never_stands_beside_the_refusal_it_resolved() -> None:
    """task-32107 (re-review P3): "can now be used in Console" has to stay true.

    The other half of the stale-receipt problem the id-based Undo fix does not
    cover: activate a workspace this conversation is NOT in and the block
    comes straight back, so the reader would paint the refusal sentence and a
    receipt claiming the opposite, one under the other.
    """
    host = _conversations_host(linked=False)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_conversation(host, pilot)
        screen.query_one("#library-conversation-use-source", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert screen.query_one("#library-conversation-link-receipt", Static).display
        assert not screen.query_one(
            "#library-conversation-open-console-blocked", Static
        ).display

        # The active workspace moves to one this conversation is not in.
        host.registry.create_workspace(workspace_id="workspace-b", name="Other")
        host.registry.set_active_workspace("workspace-b")
        screen._invalidate_library_workspace_depth_state()
        screen._sync_library_conversation_reader()
        await pilot.pause()

        blocked = screen.query_one(
            "#library-conversation-open-console-blocked", Static
        )
        receipt = screen.query_one("#library-conversation-link-receipt", Static)
        assert blocked.display is True, "the block came back"
        assert receipt.display is False, str(receipt.renderable)
        assert screen.query_one(
            "#library-conversation-link-undo", Button
        ).display is False
