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
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
    AssignSourceToWatchlistRequested,
    InspectorPane,
)
from tldw_chatbook.UI.Watchlists_Modules.opml_dialogs import (
    WatchlistPickerDialog,
    WatchlistSourcePickerDialog,
)
from tldw_chatbook.UI.Watchlists_Modules.overview_pane import OverviewPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
    AddSourceToWatchlistRequested,
    TreeScope,
)

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

        # And the picker does not offer a watchlist it is already in (review
        # wave, M5: this is the single-query `list_watchlists_for_source`
        # replacing a per-watchlist fan-out -- the answer has to be the same).
        screen.query_one("#inspector-add-to-watchlist-button", Button).press()
        for _ in range(200):
            await pilot.pause(0.02)
            if isinstance(host.screen_stack[-1], WatchlistPickerDialog):
                break
        reopened = host.screen_stack[-1]
        assert isinstance(reopened, WatchlistPickerDialog)
        assert [int(row["id"]) for row in reopened.candidates] == [], (
            "the picker offered a watchlist the source already belongs to: "
            f"{reopened.candidates}"
        )
        assert reopened.total_watchlists == 1, (
            "the dialog cannot tell 'in all of them' from 'there are none' "
            "without this"
        )


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
            "the Inspector's Add existing never opened the source picker"
        )
        assert dialog.watchlist_name == "Reading"


@pytest.mark.asyncio
async def test_neither_assign_path_writes_on_a_backend_that_cannot_take_it():
    """Review wave, I1. The rail has refused this since task-895; the two
    Inspector actions this branch added did not.

    On the Server backend there is no wire path for watchlist membership, so
    the rail's `Add existing` renders disabled with that reason painted
    beneath it. The Inspector's copy of the same verb shipped ungated: it
    opened a picker full of LOCAL sources the screen was not showing, wrote a
    local `watchlist_sources` row and reported success -- one control away
    from a button explaining that the write is impossible.

    Asserted in both directions and at both layers, the render AND the
    handler, because a backend switch can land between the two.
    """
    app = _build_test_app()
    watchlist_id, source_id = _seed(app)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        notices: list[str] = []
        screen._notify_watchlists = (
            lambda message, severity="information", **kwargs: notices.append(message)
        )
        screen.runtime_backend = "server"
        await pilot.pause(0.3)
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        )
        await pilot.pause(0.3)

        rail = screen.query_one("#wl-tree-add-source", Button)
        inspector = screen.query_one("#inspector-add-existing-source-button", Button)
        assert rail.disabled, "precondition: the rail refuses this backend"
        assert inspector.disabled, (
            "the Inspector's Add existing is enabled on a backend whose rail "
            "copy of the same verb is disabled"
        )
        assert str(inspector.tooltip) == str(rail.tooltip), (
            "the two controls must give the same reason; got "
            f"{inspector.tooltip!r} vs {rail.tooltip!r}"
        )

        # Survives a workbench rebuild: the push that armed this happens on
        # the backend switch, and a `[`/`]` toggle builds a brand new
        # Inspector that has to be seeded from the same value.
        screen.refresh(recompose=True)
        for _ in range(200):
            await pilot.pause(0.02)
            if screen.query("#inspector-add-existing-source-button"):
                break
        assert screen.query_one(
            "#inspector-add-existing-source-button", Button
        ).disabled, "a rebuilt Inspector forgot the backend cannot take this write"

        # The pane refuses to POST, not merely to paint: a backend switch can
        # land between compose and the press.
        pane = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        posted: list[object] = []
        # try/finally, not a bare patch-then-assert: a failing assertion in
        # between would leave this widget's `post_message` replaced by a
        # lambda, and Textual's message pump then hangs the whole run on
        # teardown instead of reporting the failure.
        try:
            pane.post_message = lambda message: posted.append(message)
            pane._post_add_existing_source()
        finally:
            del pane.post_message
        assert not posted, (
            "the Inspector posted a membership request on a blocked backend"
        )

        # The handler refuses too. A disabled button is a render; the message
        # it posts is one call away from a durable write.
        screen.post_message(AddSourceToWatchlistRequested(watchlist_id))
        screen.post_message(
            AssignSourceToWatchlistRequested(
                {
                    "backend": "local",
                    "entity_kind": "subscription",
                    "source_id": source_id,
                    "name": "Loose Feed",
                }
            )
        )
        await pilot.pause(0.5)

        assert app.watchlist_bundle_service.list_sources(watchlist_id) == [], (
            "a membership row was written on a backend that cannot carry one"
        )
        assert not isinstance(
            host.screen_stack[-1],
            (WatchlistPickerDialog, WatchlistSourcePickerDialog),
        ), "a picker opened on a backend that cannot service the write"
        assert len(notices) >= 2, (
            f"both refusals must explain themselves; got {notices}"
        )


@pytest.mark.asyncio
async def test_both_inspector_assign_buttons_read_the_same_backend_gate():
    """Review wave, M6. One write, one condition, one tooltip.

    The watchlist-side button was gated first; the source-side one was left
    refused at the HANDLER only, so on the Server backend it rendered live
    two rows below a greyed-out twin -- the drift this wave exists to remove,
    reintroduced by fixing only half of it.

    Both states are asserted in one run so neither can be satisfied by a
    button that is simply always disabled (or always enabled).
    """
    app = _build_test_app()
    watchlist_id, source_id = _seed(app)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        )
        screen.selected_entity = {
            "backend": "local",
            "entity_kind": "subscription",
            "source_id": source_id,
            "name": "Loose Feed",
        }
        await pilot.pause(0.3)

        # Local: the source-side button is live, with its own copy.
        source_side = screen.query_one(
            "#inspector-add-to-watchlist-button", Button
        )
        assert not source_side.disabled, (
            "the source-side assign action is disabled on a backend that can "
            "service it"
        )
        assert "watchlist" in str(source_side.tooltip).lower()

        # Server: the same reactive, the same reason, on both directions.
        screen.runtime_backend = "server"
        await pilot.pause(0.3)
        screen.selected_entity = {
            "backend": "local",
            "entity_kind": "subscription",
            "source_id": source_id,
            "name": "Loose Feed",
        }
        await pilot.pause(0.3)

        source_side = screen.query_one(
            "#inspector-add-to-watchlist-button", Button
        )
        rail = screen.query_one("#wl-tree-add-source", Button)
        assert source_side.disabled, (
            "the source-side assign action is enabled on a backend whose "
            "rail copy of the same write is greyed out"
        )
        assert str(source_side.tooltip) == str(rail.tooltip), (
            "both directions must give the SAME reason; got "
            f"{source_side.tooltip!r} vs {rail.tooltip!r}"
        )

        # Belt and braces stays: the handler still refuses a message posted
        # around the disabled render.
        notices: list[str] = []
        screen._notify_watchlists = (
            lambda message, severity="information", **kwargs: notices.append(message)
        )
        screen.post_message(
            AssignSourceToWatchlistRequested(
                {
                    "backend": "local",
                    "entity_kind": "subscription",
                    "source_id": source_id,
                    "name": "Loose Feed",
                }
            )
        )
        await pilot.pause(0.4)
        assert app.watchlist_bundle_service.list_sources(watchlist_id) == []
        assert notices, "the handler refusal must still explain itself"


@pytest.mark.asyncio
async def test_the_inspector_offers_no_assign_action_for_a_server_source():
    """AC#2's guard: membership rows key on a LOCAL subscription id.

    Distinct from the backend gate next door, and the reason both exist. The
    RUNTIME backend here is `local`, so `write_disabled_reason` is None and
    the button is correctly ENABLED; what is refused is an ENTITY whose own
    `backend` is `server` -- a property of the selection, not of the screen,
    which no render-time gate can see. So this one is caught at the handler,
    and it must say so rather than failing silently.

    Review wave, M6: the previous docstring justified the enabled button with
    "the Inspector cannot know the backend refuses it until the press is
    dispatched", which stopped being true for the RUNTIME backend the moment
    `write_disabled_reason` reached this pane. It is still true for the
    entity, which is what this test is actually about.
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


@pytest.mark.asyncio
async def test_the_watchlist_picker_does_not_claim_membership_that_cannot_exist():
    """Review wave, M2. An empty candidate list has two causes.

    "This source already belongs to every watchlist" is simply false on a
    profile that has none -- which is the profile a first-run user reaches
    this dialog on, since the source-side entry point does not require one.
    """
    app = _build_test_app()
    _seed(app)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await _mounted(host, pilot)
        for dialog, must_say, must_not_say in (
            (
                WatchlistPickerDialog("Loose Feed", [], total_watchlists=0),
                "no watchlists yet",
                "already belongs",
            ),
            (
                WatchlistPickerDialog("Loose Feed", [], total_watchlists=2),
                "already belongs",
                "no watchlists yet",
            ),
        ):
            host.push_screen(dialog)
            await pilot.pause(0.2)
            text = str(
                host.screen_stack[-1]
                .query_one("#watchlist-pick-empty", Static)
                .renderable
            ).lower()
            assert must_say in text, f"expected {must_say!r} in {text!r}"
            assert must_not_say not in text, (
                f"{must_not_say!r} is not true here: {text!r}"
            )
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
        # The default section is Read since task-2511; the Overview pane
        # lives behind its own tab now.
        screen.active_section = "overview"
        await pilot.pause(0.3)
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
