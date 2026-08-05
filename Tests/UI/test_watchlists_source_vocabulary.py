"""TASK-2303: create and assign stop sharing a vocabulary.

The 2026-08-04 UAT (F1, high) found three near-synonym labels covering two
DIFFERENT operations on one screen:

* the rail's ``Add source`` ASSIGNS a source that already exists,
* the centre's ``Create source`` and the pane's ``New Source`` CREATE one.

Every assertion here reads a label off a MOUNTED widget in the production
screen, not off a module constant: a constant renamed in one place and not
another is exactly the drift this suite exists to catch, and a constant can
be renamed without the button the user presses changing at all.

The rule the suite enforces, stated once:

* **NEW** brings a source into existence. Every create affordance says
  ``New source``.
* **ADD** files a source that already exists into a watchlist. Every
  membership affordance starts with ``Add``.
* No affordance may use the other family's verb.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
from tldw_chatbook.UI.Watchlists_Modules.opml_dialogs import (
    WatchlistPickerDialog,
    WatchlistSourcePickerDialog,
)
from tldw_chatbook.UI.Watchlists_Modules.overview_pane import OverviewPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope

pytestmark = pytest.mark.unit

#: The two verb families, as the user reads them off the screen.
CREATE_LABEL = "New source"
ASSIGN_PREFIX = "Add"

#: Words that must never appear on a membership affordance, because each one
#: is the promise that something will be brought into existence.
CREATE_WORDS = ("new", "create")

#: A membership affordance must also say, in the label itself, that the thing
#: being moved already exists. ``Add source`` -- the label the UAT misread --
#: passes the "starts with Add, contains no create word" test and is still a
#: near-synonym of ``New source``: both are two words ending in the same
#: noun. Naming the pre-existence is what actually separates the families, so
#: it is asserted rather than left to the reader.
EXISTENCE_PHRASES = ("existing", "to watchlist")


def _seed(app) -> tuple[int, int]:
    """One watchlist and one unassigned source.

    Returns:
        ``(watchlist_id, source_id)``.
    """
    db = app.local_watchlists_service._db()
    source_id = db.add_subscription(
        name="Loose Feed", type="rss", source="https://loose.test/feed.xml"
    )
    watchlist = app.watchlist_bundle_service.create("Reading")
    return int(watchlist["id"]), int(source_id)


async def _mounted(host, pilot):
    await pilot.pause(0.3)
    screen = host.screen_stack[-1]
    for _ in range(40):
        await pilot.pause()
        if screen._tree_watchlists:
            break
    return screen


def _label(widget) -> str:
    return str(widget.label)


# --- AC#1: one verb per operation, everywhere -------------------------------


@pytest.mark.asyncio
async def test_the_rail_and_the_pane_do_not_share_a_verb():
    """AC#1: the two labels the UAT confused, read off the live screen."""
    app = _build_test_app()
    _seed(app)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen.active_section = "sources"
        await pilot.pause(0.2)

        assign = _label(screen.query_one("#wl-tree-add-source", Button))
        create = _label(screen.query_one("#sources-new-button", Button))

        assert create == CREATE_LABEL, (
            f"the create affordance reads {create!r}; every create affordance "
            f"must read {CREATE_LABEL!r}"
        )
        assert assign.startswith(ASSIGN_PREFIX), (
            f"the membership affordance reads {assign!r}; it must start with "
            f"{ASSIGN_PREFIX!r}"
        )
        for word in CREATE_WORDS:
            assert word not in assign.lower(), (
                f"the membership affordance {assign!r} uses the create word "
                f"{word!r} -- that is the near-synonym defect this task removed"
            )
        assert any(phrase in assign.lower() for phrase in EXISTENCE_PHRASES), (
            f"the membership affordance {assign!r} does not say the source "
            f"already exists; expected one of {EXISTENCE_PHRASES}"
        )
        assert assign.lower() != create.lower()


@pytest.mark.asyncio
async def test_the_empty_state_create_button_uses_the_create_verb():
    """AC#1: the centre's own create affordance, which said "Create source".

    This button is only rendered on a profile with nothing in it, so the
    harness deliberately seeds NOTHING.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        for _ in range(60):
            await pilot.pause(0.02)
            if screen.query("#wc-empty-create-source"):
                break
        button = screen.query_one("#wc-empty-create-source", Button)
        assert _label(button) == CREATE_LABEL


@pytest.mark.asyncio
async def test_the_assign_affordances_agree_with_each_other():
    """AC#1/#2: rail and Inspector name the same operation the same way."""
    app = _build_test_app()
    watchlist_id, _source_id = _seed(app)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        )
        await pilot.pause(0.2)

        rail = _label(screen.query_one("#wl-tree-add-source", Button))
        inspector = _label(
            screen.query_one("#inspector-add-existing-source-button", Button)
        )
        assert rail == inspector, (
            "the rail and the Inspector must name the same write identically; "
            f"got {rail!r} and {inspector!r}"
        )
        for label in (rail, inspector):
            assert any(
                phrase in label.lower() for phrase in EXISTENCE_PHRASES
            ), f"{label!r} does not say the source already exists"


# --- AC#2: the Inspector can assign -----------------------------------------


@pytest.mark.asyncio
async def test_a_selected_source_can_be_filed_from_its_inspector():
    """AC#2 + the data actually lands.

    Presses the Inspector's own button, answers the picker with a real
    click, and then asks the BUNDLE SERVICE -- not the toast, not the rail --
    whether the membership row exists. A test that asserted the dialog opened
    would stay green with the write deleted.
    """
    app = _build_test_app()
    watchlist_id, source_id = _seed(app)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen.selected_entity = {
            "backend": "local",
            "entity_kind": "subscription",
            "source_id": source_id,
            "name": "Loose Feed",
        }
        await pilot.pause(0.2)

        assert app.watchlist_bundle_service.list_sources(watchlist_id) == [], (
            "precondition: the source is unassigned"
        )

        screen.query_one("#inspector-add-to-watchlist-button", Button).press()
        for _ in range(200):
            await pilot.pause(0.02)
            if isinstance(host.screen_stack[-1], WatchlistPickerDialog):
                break
        dialog = host.screen_stack[-1]
        assert isinstance(dialog, WatchlistPickerDialog), (
            "pressing Add to watchlist… never opened the watchlist picker"
        )
        dialog.query_one(f"#wl-pick-option-{watchlist_id}", Button).press()

        for _ in range(200):
            await pilot.pause(0.02)
            if app.watchlist_bundle_service.list_sources(watchlist_id):
                break

        assert app.watchlist_bundle_service.list_sources(watchlist_id) == [
            source_id
        ], "the source never joined the watchlist the user picked"


@pytest.mark.asyncio
async def test_the_watchlist_inspector_opens_the_source_picker():
    """AC#2, the watchlist-first direction, driven by a real press.

    The button was rendered disabled with "no message type exists" as the
    reason; it posts the rail's own message now, so the assertion is that
    the picker really opens -- naming the watchlist in scope, which is what
    proves the id came off the level rather than off a stale entity.
    """
    app = _build_test_app()
    watchlist_id, _source_id = _seed(app)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        )
        await pilot.pause(0.2)

        screen.query_one("#inspector-add-existing-source-button", Button).press()
        for _ in range(200):
            await pilot.pause(0.02)
            if isinstance(host.screen_stack[-1], WatchlistSourcePickerDialog):
                break
        dialog = host.screen_stack[-1]
        assert isinstance(dialog, WatchlistSourcePickerDialog), (
            "the Inspector's Add existing… never opened the source picker"
        )
        assert dialog.watchlist_name == "Reading"


@pytest.mark.asyncio
async def test_the_inspector_offers_no_assign_action_for_a_server_source():
    """AC#2's guard: membership rows key on a LOCAL subscription id.

    The button still renders (the Inspector cannot know the backend refuses
    it until the press is dispatched), but the press must not write, and it
    must say so rather than failing silently.
    """
    app = _build_test_app()
    watchlist_id, _source_id = _seed(app)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        notices: list[str] = []
        screen._notify_watchlists = (
            lambda message, severity="information", **kwargs: notices.append(message)
        )
        screen.selected_entity = {
            "backend": "server",
            "entity_kind": "watchlist_source",
            "source_id": 999,
            "name": "Remote Feed",
        }
        await pilot.pause(0.2)

        screen.query_one("#inspector-add-to-watchlist-button", Button).press()
        await pilot.pause(0.3)

        assert app.watchlist_bundle_service.list_sources(watchlist_id) == []
        assert notices, "a refused press must explain itself"


# --- AC#3: the modals explain themselves ------------------------------------


@pytest.mark.asyncio
async def test_both_pickers_state_what_a_row_does_and_that_nothing_is_created():
    """AC#3, on both directions of the same write.

    Mounted, not composed in isolation: the instruction line has to reach
    the screen, and a `Static` returned by `compose()` proves only that it
    was constructed.
    """
    app = _build_test_app()
    _seed(app)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await _mounted(host, pilot)
        for dialog, instruction_id, subject in (
            (
                WatchlistSourcePickerDialog(
                    "Reading", [{"id": 1, "name": "Loose Feed", "type": "rss"}]
                ),
                "watchlist-add-source-instructions",
                "watchlist",
            ),
            (
                WatchlistPickerDialog("Loose Feed", [{"id": 1, "name": "Reading"}]),
                "watchlist-pick-instructions",
                "source",
            ),
        ):
            name = type(dialog).__name__
            host.push_screen(dialog)
            await pilot.pause(0.2)
            text = str(
                host.screen_stack[-1]
                .query_one(f"#{instruction_id}", Static)
                .renderable
            )
            assert "Choose a" in text, (
                f"{name}'s instruction does not say what to do: {text!r}"
            )
            assert "No new source is created" in text, (
                f"{name} does not say that nothing is created: {text!r}"
            )
            assert subject in text
            host.pop_screen()
            await pilot.pause(0.1)


# --- AC#4: guidance names labels that exist ---------------------------------


@pytest.mark.asyncio
async def test_first_run_guidance_names_the_button_that_is_on_screen():
    """AC#4. The copy and the button are read in the same run, so a rename
    of either alone fails this."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        for _ in range(60):
            await pilot.pause(0.02)
            if screen.query("#overview-first-run-body"):
                break
        body = str(
            screen.query_one("#overview-first-run-body", Static).renderable
        )
        hint = str(
            screen.query_one("#inspector-first-run-hint", Static).renderable
        )

        screen.active_section = "sources"
        await pilot.pause(0.3)
        button = _label(screen.query_one("#sources-new-button", Button))

        assert button in body, (
            f"the first-run copy tells the user to press something that is "
            f"not on screen: copy={body!r}, button={button!r}"
        )
        assert button in hint, (
            f"the Inspector's first-run hint names a control that does not "
            f"exist: hint={hint!r}, button={button!r}"
        )


def test_both_first_run_variants_name_the_create_button():
    """AC#4 for the variant the mounted test above cannot reach: the copy
    shown once a watchlist exists but has no sources."""
    pane = OverviewPane()
    pane.watchlist_count = 0
    assert CREATE_LABEL in pane._first_run_body()
    pane.watchlist_count = 1
    assert CREATE_LABEL in pane._first_run_body()


def test_the_inspector_first_run_hint_is_not_stale_copy():
    """A unit-level twin of the mounted assertion, so a failure says which
    of the two halves moved."""
    pane = InspectorPane()
    pane.profile_state = OverviewPane.EMPTY
    hints = [
        str(widget.renderable)
        for widget in pane.compose()
        if getattr(widget, "id", None) == "inspector-first-run-hint"
    ]
    assert hints and CREATE_LABEL in hints[0]
