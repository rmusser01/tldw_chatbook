"""TASK-2302: the create form says where the source is going, and it goes there.

The 2026-08-04 UAT (F13, high) created a source with the watchlist
"AI Research News" selected in the rail -- one screen after first-run
guidance that said "press New source to add a feed to it" -- and the source
landed in Unassigned. Nothing in the form named a destination beforehand and
nothing named one afterwards.

Every assertion about where a source LANDED is read from the bundle service's
own membership query, never from a toast, a rail count or a table row: those
are all downstream of the write, and a test that watched them would stay
green with the write deleted and a refresh left in place.

The polish findings in the same task are pinned here too, because they are
the same form: the Type Select's missing label (F17), the noise help copy's
truncation (F11/F12) and the ignore-selectors block being prefilled and
prominent for feed types it cannot affect (F14).
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Select, Static, TextArea

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope

pytestmark = pytest.mark.unit

#: The UAT ran at 235x52; 160x42 is the small end the rest of the watchlists
#: parity work covers and the size that actually constrains this form.
SIZES = [(160, 42), (235, 52)]


def _seed_watchlists(app, *names: str) -> dict[str, int]:
    bundle = app.watchlist_bundle_service
    return {name: int(bundle.create(name)["id"]) for name in names}


async def _mounted(host, pilot):
    await pilot.pause(0.3)
    screen = host.screen_stack[-1]
    for _ in range(60):
        await pilot.pause()
        if screen._tree_watchlists:
            break
    return screen


def _form_is_settled(pane: SourcesPane) -> bool:
    """Whether the form's controls have mounted AND been laid out.

    Both halves are load-bearing. `Select.value` is only resolved from the
    constructor argument in `Select._on_mount`, so a Select queried the
    instant `compose()` has run still reports `Select.NULL`; and its region
    is `0x0` until layout, which every geometry assertion here reads. A
    helper that returned as soon as the widget could be QUERIED would make
    those assertions measure a widget that is not on screen yet.
    """
    controls = pane.query("#sources-create-watchlist")
    if not controls:
        return False
    control = controls.first()
    return control.region.width > 0 and control.value is not Select.NULL


async def _open_create_form(host, pilot, screen) -> SourcesPane:
    """Open the form the way a user does -- through the pane's own button."""
    screen.active_section = "sources"
    for _ in range(200):
        await pilot.pause(0.02)
        if screen.query("#watchlists-sources-pane"):
            break
    screen.query_one("#watchlists-sources-pane", SourcesPane).query_one(
        "#sources-new-button", Button
    ).press()
    pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
    for _ in range(300):
        await pilot.pause(0.02)
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        if _form_is_settled(pane):
            break
    assert _form_is_settled(pane), "the create form never finished opening"
    return pane


async def _choose_page_type(pilot, screen) -> SourcesPane:
    """Switch the open form to a page-scrape type and wait out the recompose.

    Waits for the noise field to be LAID OUT, not merely mounted: its width
    is what the truncation assertions measure against, and a freshly composed
    widget reports a 0x0 region.
    """
    pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
    pane.query_one("#sources-create-type", Select).value = "url"
    for _ in range(300):
        await pilot.pause(0.02)
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        fields = pane.query("#sources-create-ignore-selectors")
        if fields and fields.first().region.width > 0:
            break
    return pane


async def _submit(pane, screen, pilot, name: str, url: str) -> None:
    pane.query_one("#sources-create-name", Input).value = name
    pane.query_one("#sources-create-url", Input).value = url
    await pilot.pause(0.1)
    pane.query_one("#sources-create-submit", Button).press()
    for _ in range(300):
        await pilot.pause(0.02)
        if screen._loaded_sources and any(
            str(row.get("title") or row.get("name")) == name
            for row in screen._loaded_sources
        ):
            break


def _destination_label(pane: SourcesPane) -> str:
    """What the destination control is SHOWING, off the painted label."""
    select = pane.query_one("#sources-create-watchlist", Select)
    current = select.query_one("SelectCurrent")
    return str(current.query_one("#label", Static).renderable)


# --- AC#1: the destination is visible before submit -------------------------


@pytest.mark.asyncio
async def test_the_form_shows_the_active_scope_as_its_destination():
    """AC#1, and the UAT's exact setup: a watchlist is selected in the rail."""
    app = _build_test_app()
    ids = _seed_watchlists(app, "AI Research News")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=ids["AI Research News"])
        )
        await pilot.pause(0.2)

        pane = await _open_create_form(host, pilot, screen)

        assert _destination_label(pane) == "AI Research News", (
            "the form does not name the watchlist the rail has selected; it "
            f"shows {_destination_label(pane)!r}"
        )
        assert (
            pane.query_one("#sources-create-watchlist", Select).value
            == ids["AI Research News"]
        )


@pytest.mark.asyncio
async def test_the_form_shows_unassigned_when_no_watchlist_is_in_scope():
    """AC#1's other half: Unassigned is a destination, stated, not a silence."""
    app = _build_test_app()
    _seed_watchlists(app, "Reading")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen._apply_tree_scope(TreeScope(kind="all"))
        await pilot.pause(0.2)

        pane = await _open_create_form(host, pilot, screen)
        assert "Unassigned" in _destination_label(pane)
        # And the watchlist is still reachable -- the default is a default,
        # not a restriction.
        options = pane._destination_options()
        assert any(str(label) == "Reading" for label, _value in options)


@pytest.mark.asyncio
async def test_the_destination_control_carries_a_visible_label():
    """AC#1: a bare Select is what the UAT could not read on the Type row."""
    app = _build_test_app()
    _seed_watchlists(app, "Reading")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        pane = await _open_create_form(host, pilot, screen)

        row = pane.query_one(".sources-create-destination-row")
        labels = [
            str(child.renderable)
            for child in row.query(Static)
            if child.id != "label"
        ]
        assert any("Watchlist" in label for label in labels), (
            f"the destination row paints no label of its own: {labels}"
        )


# --- AC#2: it lands where the form said --------------------------------------


@pytest.mark.asyncio
async def test_a_source_created_under_a_watchlist_scope_joins_that_watchlist():
    """AC#5, the regression the UAT asked for, asserted at the store."""
    app = _build_test_app()
    ids = _seed_watchlists(app, "AI Research News")
    watchlist_id = ids["AI Research News"]
    notices: list[str] = []
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen._notify_watchlists = (
            lambda message, severity="information", **kwargs: notices.append(message)
        )
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        )
        await pilot.pause(0.2)

        pane = await _open_create_form(host, pilot, screen)
        assert app.watchlist_bundle_service.list_sources(watchlist_id) == []

        await _submit(pane, screen, pilot, "HN Front Page", "https://hn.test/rss")
        for _ in range(200):
            await pilot.pause(0.02)
            if app.watchlist_bundle_service.list_sources(watchlist_id):
                break

        members = app.watchlist_bundle_service.list_sources(watchlist_id)
        assert len(members) == 1, (
            "the source created under an active watchlist scope did not join "
            f"it; membership is {members}"
        )
        row = app.local_watchlists_service._db().get_subscription(members[0])
        assert row["name"] == "HN Front Page"

        # AC#2: and the user is told where it went, by name.
        assert any("AI Research News" in notice for notice in notices), (
            f"no confirmation named the destination: {notices}"
        )


@pytest.mark.asyncio
async def test_choosing_unassigned_really_leaves_the_source_unassigned():
    """AC#2 in the other direction: the control is not decorative.

    Same scope as the test above -- a watchlist IS selected -- so the only
    thing that can produce an unassigned source is the choice made in the
    form.
    """
    app = _build_test_app()
    ids = _seed_watchlists(app, "AI Research News")
    watchlist_id = ids["AI Research News"]
    notices: list[str] = []
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen._notify_watchlists = (
            lambda message, severity="information", **kwargs: notices.append(message)
        )
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        )
        await pilot.pause(0.2)

        pane = await _open_create_form(host, pilot, screen)
        pane.query_one("#sources-create-watchlist", Select).value = (
            SourcesPane.UNASSIGNED_DESTINATION
        )
        await pilot.pause(0.2)
        assert "Unassigned" in _destination_label(pane)

        await _submit(pane, screen, pilot, "Loose Feed", "https://loose.test/rss")
        await pilot.pause(0.3)

        assert app.watchlist_bundle_service.list_sources(watchlist_id) == [], (
            "a source the form said was Unassigned joined the scoped watchlist"
        )
        assert [
            row["name"]
            for row in app.watchlist_bundle_service.list_unassigned_source_rows()
        ] == ["Loose Feed"]
        assert any("Unassigned" in notice for notice in notices), (
            f"no confirmation named the destination: {notices}"
        )


@pytest.mark.asyncio
async def test_changing_the_destination_survives_a_workbench_rebuild():
    """The destination is draft state, like the name and the url.

    Any region collapse constructs a brand new `SourcesPane`; a destination
    that reset to the scope default there would silently re-aim a form the
    user had already pointed somewhere else.
    """
    app = _build_test_app()
    ids = _seed_watchlists(app, "AI Research News", "Reading")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=ids["AI Research News"])
        )
        await pilot.pause(0.2)

        pane = await _open_create_form(host, pilot, screen)
        pane.query_one("#sources-create-name", Input).value = "Half typed"
        pane.query_one("#sources-create-watchlist", Select).value = ids["Reading"]
        await pilot.pause(0.3)

        # A rebuild that has nothing to do with Sources.
        screen.refresh(recompose=True)
        for _ in range(200):
            await pilot.pause(0.02)
            panes = list(screen.query(SourcesPane))
            if panes and panes[0].query("#sources-create-watchlist"):
                break
        rebuilt = screen.query_one("#watchlists-sources-pane", SourcesPane)

        assert rebuilt.query_one("#sources-create-name", Input).value == "Half typed"
        assert _destination_label(rebuilt) == "Reading", (
            "the rebuilt form re-aimed itself at the scope, discarding the "
            f"user's choice; it shows {_destination_label(rebuilt)!r}"
        )


@pytest.mark.asyncio
async def test_a_new_form_re_reads_the_scope_rather_than_the_last_submission():
    """Opening the form again aims it at where the user is NOW."""
    app = _build_test_app()
    ids = _seed_watchlists(app, "AI Research News", "Reading")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=ids["AI Research News"])
        )
        await pilot.pause(0.2)
        pane = await _open_create_form(host, pilot, screen)
        pane.query_one("#sources-create-cancel", Button).press()
        await pilot.pause(0.2)

        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=ids["Reading"])
        )
        await pilot.pause(0.3)
        pane = await _open_create_form(host, pilot, screen)

        assert _destination_label(pane) == "Reading"


# --- AC#3/#4: the form's own polish ------------------------------------------


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_the_type_select_carries_a_visible_label(size):
    """AC#3. The UAT read a bare "RSS ▼" with nothing naming the field."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=size) as pilot:
        screen = await _mounted(host, pilot)
        pane = await _open_create_form(host, pilot, screen)

        select = pane.query_one("#sources-create-type", Select)
        row = pane.query_one(".sources-create-type-row")
        labels = [
            child
            for child in row.query(Static)
            if child.id != "label" and "Type" in str(child.renderable)
        ]
        assert labels, "the Type Select has no label beside it"

        # It has to reach the screen, on the Select's own rows, and inside
        # the pane: a label pushed off the right edge is not a label.
        label = labels[0]
        assert label.region.width > 0 and label.region.height > 0
        assert label.region.right <= pane.region.right
        assert label.region.y <= select.region.y < label.region.bottom
        strips = screen._compositor.render_strips()
        painted = "".join(
            seg.text for seg in strips[label.region.y + label.region.height // 2]
        )
        assert "Type" in painted, (
            f"the Type label never reaches the screen at {size}: {painted.strip()!r}"
        )


@pytest.mark.asyncio
async def test_the_ignore_selectors_block_is_absent_for_feed_types():
    """AC#4. CSS selectors cannot affect an RSS feed, so the form does not
    prefill four rows of them over one."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        pane = await _open_create_form(host, pilot, screen)

        assert (
            pane.query_one("#sources-create-type", Select).value == "rss"
        ), "precondition: the form opens on a feed type"
        assert not pane.query("#sources-create-ignore-selectors"), (
            "the noise field is on screen for a feed type it cannot affect"
        )


@pytest.mark.asyncio
async def test_the_ignore_selectors_block_returns_for_a_page_type():
    """AC#4's other half: gated, not deleted. A page type is reachable from
    this form at all only because TASK-2302 added one."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        pane = await _open_create_form(host, pilot, screen)

        pane = await _choose_page_type(pilot, screen)

        assert pane.query("#sources-create-ignore-selectors"), (
            "choosing a page type did not bring the noise field back"
        )


@pytest.mark.asyncio
async def test_a_feed_source_is_not_created_with_selectors_it_cannot_use():
    """The gate has to reach the payload, not just the paint.

    `_clear_create_draft` keeps `create_draft_ignore_selectors` prefilled for
    the next form, so a submit that read the DRAFT rather than the DOM would
    file the shipped default against every RSS source ever created here.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        pane = await _open_create_form(host, pilot, screen)

        await _submit(pane, screen, pilot, "Plain Feed", "https://plain.test/rss")
        for _ in range(200):
            await pilot.pause(0.02)
            if app.local_watchlists_service._db().get_all_subscriptions():
                break

        rows = app.local_watchlists_service._db().get_all_subscriptions()
        assert [row["name"] for row in rows] == ["Plain Feed"]
        assert not (rows[0]["ignore_selectors"] or ""), (
            "an RSS source was stored with page selectors it can never use: "
            f"{rows[0]['ignore_selectors']!r}"
        )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_the_noise_help_text_fits_the_field_it_is_painted_on(size):
    """AC#4. The UAT read "…changes always report; change_threshold" at
    235x52 -- Textual's border-label renderer truncates silently, so the copy
    is measured against the field's REAL width rather than assumed to fit.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=size) as pilot:
        screen = await _mounted(host, pilot)
        await _open_create_form(host, pilot, screen)
        pane = await _choose_page_type(pilot, screen)
        field = pane.query_one("#sources-create-ignore-selectors", TextArea)

        # Textual paints a border label between the two corner cells and
        # pads it by one column on each side, so the budget is width - 4.
        budget = field.region.width - 4
        for role, text in (
            ("label", str(field.border_title)),
            ("help", str(field.border_subtitle)),
        ):
            assert len(text) <= budget, (
                f"the noise field's {role} is {len(text)} characters in a "
                f"{field.region.width}-column field at {size}: Textual will "
                f"truncate it. Text: {text!r}"
            )

        # And it really is on screen, not merely short enough in principle.
        strips = screen._compositor.render_strips()
        painted = "".join(seg.text for seg in strips[field.region.bottom - 1])
        assert "…" not in painted, (
            f"the noise field's bottom border is truncated at {size}: "
            f"{painted.strip()!r}"
        )
        assert str(field.border_subtitle).split(";")[-1].strip() in painted, (
            f"the tail of the help copy never reaches the screen at {size}: "
            f"{painted.strip()!r}"
        )


# --- degradation --------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_destination_offers_nothing_when_membership_cannot_be_written():
    """The server backend has no wire path for membership, so the form must
    not offer a destination it would then silently drop."""
    app = _build_test_app()
    _seed_watchlists(app, "Reading")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen.runtime_backend = "server"
        await pilot.pause(0.3)

        assert screen._create_form_watchlist_choices() == []
        assert screen._scope_default_destination() == (
            SourcesPane.UNASSIGNED_DESTINATION
        )
