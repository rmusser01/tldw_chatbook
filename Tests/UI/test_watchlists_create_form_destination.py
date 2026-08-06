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

from Tests.UI.full_app_destination_context import (
    active_destination_screen as _production_screen,
    full_app_destination_context as _production_host,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope

pytestmark = pytest.mark.unit

#: The UAT ran at 235x52; 160x42 is the small end the rest of the watchlists
#: parity work covers and the size that actually constrains this form.
SIZES = [(160, 42), (235, 52)]

# The PRODUCTION-CSS harness throughout (whole-branch review, I2).
#
# The first version of this file used `DestinationHarness`, which mounts the
# production screen inside a bare `App` with NO stylesheet. That is harmless
# for "where did the source land", which is a database question -- and
# actively misleading for anything about width. Measured, same screen, same
# size: the Sources pane is 53 columns there and **93** under the production
# stylesheet at 160x42 (78 vs **168** at 235x52), the `Type` label reports
# the whole row rather than `width: auto`'s few columns, and the destination
# Select satisfies "laid out" at `w=1 h=0`. The noise field's truncation
# budget was measured in that harness and the resulting number written into
# a shipped code comment, where it was wrong by roughly 2x.
#
# One harness for the whole file rather than two, so no future test can pick
# the wrong one: it is the same context `test_watchlists_source_create_form.py`
# and `test_destination_visual_parity_correction.py` already use.


def _seed_watchlists(app, *names: str) -> dict[str, int]:
    bundle = app.watchlist_bundle_service
    return {name: int(bundle.create(name)["id"]) for name in names}


async def _mounted_with_css(host, pilot):
    """The same wait, against the production-CSS context.

    `FullAppDestinationContext` keeps the destination screen as an attribute
    rather than leaving it on top of the stack, so the screen is read from
    there instead of `screen_stack[-1]` (which a pushed modal would also
    satisfy).
    """
    await pilot.pause(0.3)
    screen = _production_screen(host)
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
    return (
        control.region.width > 0
        # HEIGHT too (review wave): in a stylesheet-less harness this control
        # satisfies `width > 0` at `w=1 h=0` -- present, sized, and painting
        # nothing. "Laid out" has to mean both dimensions.
        and control.region.height > 0
        and control.value is not Select.NULL
    )


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
    host = _production_host(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted_with_css(host, pilot)
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
    host = _production_host(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted_with_css(host, pilot)
        screen._apply_tree_scope(TreeScope(kind="all"))
        await pilot.pause(0.2)

        pane = await _open_create_form(host, pilot, screen)
        assert "Unassigned" in _destination_label(pane)
        # And the watchlist is still reachable -- the default is a default,
        # not a restriction.
        options = pane._destination_options()
        assert any(str(label) == "Reading" for label, _value in options)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_the_destination_control_carries_a_visible_label(size):
    """AC#1: a bare Select is what the UAT could not read on the Type row.

    Production CSS (review wave, I2): the label's `width: auto` and the row's
    `height: 1` are both stylesheet rules, so a stylesheet-less harness
    cannot tell a label that fits from one claiming the whole row.
    """
    host = _production_host(_build_test_app(), "watchlists_collections")
    _seed_watchlists(host.app, "Reading")
    async with host.run_test(size=size) as pilot:
        screen = await _mounted_with_css(host, pilot)
        pane = await _open_create_form(host, pilot, screen)

        row = pane.query_one(".sources-create-destination-row")
        labels = [
            child
            for child in row.query(Static)
            if child.id != "label" and "Watchlist" in str(child.renderable)
        ]
        assert labels, (
            "the destination row paints no label of its own: "
            f"{[str(c.renderable) for c in row.query(Static)]}"
        )
        label = labels[0]
        select = pane.query_one("#sources-create-watchlist", Select)
        assert label.region.width < select.region.width, (
            f"the label claims {label.region.width} columns beside a "
            f"{select.region.width}-column control -- `width: auto` is not "
            "winning, and the control it names gets pushed off the row"
        )
        assert label.region.right <= pane.region.right
        strips = screen._compositor.render_strips()
        painted = "".join(seg.text for seg in strips[label.region.y])
        assert "Watchlist" in painted, (
            f"the Watchlist label never reaches the screen at {size}: "
            f"{painted.strip()!r}"
        )


# --- AC#2: it lands where the form said --------------------------------------


@pytest.mark.asyncio
async def test_a_source_created_under_a_watchlist_scope_joins_that_watchlist():
    """AC#5, the regression the UAT asked for, asserted at the store."""
    app = _build_test_app()
    ids = _seed_watchlists(app, "AI Research News")
    watchlist_id = ids["AI Research News"]
    notices: list[str] = []
    host = _production_host(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted_with_css(host, pilot)
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
    host = _production_host(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted_with_css(host, pilot)
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
    host = _production_host(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted_with_css(host, pilot)
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
    host = _production_host(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted_with_css(host, pilot)
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
    host = _production_host(_build_test_app(), "watchlists_collections")
    async with host.run_test(size=size) as pilot:
        screen = await _mounted_with_css(host, pilot)
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
        # The CSS comment claims `width: auto` is what stops this label
        # claiming the row and pushing the Select it names off the pane's
        # right edge. Only a stylesheet-loading harness can check that
        # (review wave, I2): without CSS the label IS the whole row.
        assert label.region.width < select.region.width, (
            f"the Type label claims {label.region.width} columns beside a "
            f"{select.region.width}-column Select"
        )
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
    host = _production_host(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted_with_css(host, pilot)
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
    host = _production_host(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted_with_css(host, pilot)
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
    host = _production_host(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted_with_css(host, pilot)
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
    host = _production_host(_build_test_app(), "watchlists_collections")
    async with host.run_test(size=size) as pilot:
        screen = await _mounted_with_css(host, pilot)
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
    not offer a destination it would then silently drop.

    Review wave: this used to assert on two screen helpers only, which is one
    indirection away from its own claim. It now OPENS the form under that
    backend and reads the mounted control's options.
    """
    app = _build_test_app()
    _seed_watchlists(app, "Reading")
    host = _production_host(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted_with_css(host, pilot)
        screen.runtime_backend = "server"
        await pilot.pause(0.3)

        assert screen._create_form_watchlist_choices() == []
        assert screen._scope_default_destination() == (
            SourcesPane.UNASSIGNED_DESTINATION
        )

        pane = await _open_create_form(host, pilot, screen)
        assert [value for _label, value in pane._destination_options()] == [
            SourcesPane.UNASSIGNED_DESTINATION
        ], "the form offered a watchlist the backend cannot write membership for"
        assert "Unassigned" in _destination_label(pane)


@pytest.mark.asyncio
async def test_a_destination_that_vanished_before_submit_is_reported_as_news():
    """Review wave, M3, driven through the real FK.

    `watchlist_sources` carries a FOREIGN KEY on both columns and
    `SubscriptionsDB` sets `PRAGMA foreign_keys = ON`, so deleting the chosen
    watchlist between form-open and submit makes `add_source` raise and the
    source lands in Unassigned. That is true and it is also NOT what the user
    asked for, so it cannot arrive at the same severity as a create that went
    to plan.
    """
    app = _build_test_app()
    ids = _seed_watchlists(app, "Doomed")
    notices: list[tuple[str, str]] = []
    host = _production_host(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted_with_css(host, pilot)
        screen._notify_watchlists = (
            lambda message, severity="information", **kwargs: notices.append(
                (message, severity)
            )
        )
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=ids["Doomed"])
        )
        await pilot.pause(0.2)
        pane = await _open_create_form(host, pilot, screen)
        assert _destination_label(pane) == "Doomed"

        # Gone from under the form, without touching the form.
        app.watchlist_bundle_service.delete(ids["Doomed"])

        await _submit(pane, screen, pilot, "Orphan", "https://orphan.test/rss")
        for _ in range(200):
            await pilot.pause(0.02)
            if notices:
                break

        assert notices, "the create produced no confirmation at all"
        message, severity = notices[-1]
        assert "Unassigned" in message, (
            f"the toast must name where the source really is: {message!r}"
        )
        assert severity == "warning", (
            "a destination the user chose and did not get is news, not "
            f"routine; got severity {severity!r}"
        )
        assert [
            row["name"]
            for row in app.watchlist_bundle_service.list_unassigned_source_rows()
        ] == ["Orphan"]


# --- the watchlist set changes while a form is alive -------------------------


@pytest.mark.asyncio
async def test_deleting_the_destination_watchlist_under_an_open_form_degrades():
    """Review wave, I3(a). `_resolved_destination`'s stale-draft fallback.

    `Select` with `allow_blank=False` raises `InvalidSelectValueError` on a
    value it has no option for, and that raise comes out of `compose()` --
    which takes the whole pane down: no create form, no sources table. The
    sequence is reachable without leaving the screen: aim the form at a
    watchlist, delete it in the rail, then let anything recompose the pane.
    """
    app = _build_test_app()
    ids = _seed_watchlists(app, "Doomed", "Survivor")
    host = _production_host(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted_with_css(host, pilot)
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=ids["Doomed"])
        )
        await pilot.pause(0.2)
        pane = await _open_create_form(host, pilot, screen)
        pane.query_one("#sources-create-name", Input).value = "Half typed"
        await pilot.pause(0.2)
        assert _destination_label(pane) == "Doomed"

        # The delete, then the push that tells the live pane about it.
        app.watchlist_bundle_service.delete(ids["Doomed"])
        screen._load_tree_data()
        for _ in range(200):
            await pilot.pause(0.02)
            if ids["Doomed"] not in {
                int(w["id"]) for w in screen._tree_watchlists
            }:
                break

        # Any recompose of the pane -- here the Filters toggle, a real button.
        pane.query_one("#sources-filter-toggle", Button).press()
        for _ in range(200):
            await pilot.pause(0.02)
            live = screen.query(SourcesPane)
            if live and _form_is_settled(live.first()):
                break

        live_pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        assert live_pane.query("#sources-table"), (
            "the pane failed to rebuild after its destination watchlist was "
            "deleted -- compose() raised"
        )
        assert "Unassigned" in _destination_label(live_pane), (
            "a form aimed at a deleted watchlist must fall back to Unassigned;"
            f" it shows {_destination_label(live_pane)!r}"
        )
        assert (
            live_pane.query_one("#sources-create-name", Input).value == "Half typed"
        ), "the degrade must not cost the user the rest of their draft"


@pytest.mark.asyncio
async def test_a_watchlist_created_mid_session_reaches_the_next_form():
    """Review wave, I3(b). The `watchlist_choices` push.

    This is the ONLY thing that makes a watchlist created after the pane was
    built selectable without a full screen rebuild. Asserted at the shipped
    limit, not an aspirational one: a form left open ACROSS the creation is
    documented as missing it (rebuilding the Select would cost a half-typed
    draft), so the assertion is that the next OPEN offers it -- and that the
    open form is left alone in the meantime.
    """
    app = _build_test_app()
    _seed_watchlists(app, "Reading")
    host = _production_host(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted_with_css(host, pilot)
        pane = await _open_create_form(host, pilot, screen)
        assert [str(label) for label, _v in pane._destination_options()] == [
            "Unassigned (no watchlist)",
            "Reading",
        ]

        created = app.watchlist_bundle_service.create("Later")
        screen._load_tree_data()
        for _ in range(200):
            await pilot.pause(0.02)
            if int(created["id"]) in {
                int(w["id"]) for w in screen._tree_watchlists
            }:
                break
        await pilot.pause(0.2)

        # The live pane knows, without having been rebuilt...
        live_pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        assert [str(w["name"]) for w in live_pane.watchlist_choices] == [
            "Later",
            "Reading",
        ], (
            "the new watchlist never reached the mounted pane, so no reopen "
            "of the form can offer it"
        )

        # ...and the next open of the form offers it.
        live_pane.query_one("#sources-create-cancel", Button).press()
        await pilot.pause(0.2)
        reopened = await _open_create_form(host, pilot, screen)
        assert [str(label) for label, _v in reopened._destination_options()] == [
            "Unassigned (no watchlist)",
            "Later",
            "Reading",
        ]
