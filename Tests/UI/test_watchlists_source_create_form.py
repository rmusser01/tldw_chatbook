"""TASK-1035: the Watchlists create-source form, driven the way a user drives it.

The 2026-07-28 experienced-user UAT reported that the create-source form
"cannot be filled in": opening it and typing, and opening it and pressing
`Tab` five times then typing, both left `Name` and `URL` empty.

Everything here runs against the **production stylesheet in the full shell**
(`_visual_destination_harness`), because the form's real failure is a
geometry/focus interaction that a bare `App` with no CSS cannot reproduce --
the same reason `test_destination_visual_parity_correction.py` exists.

What the UAT saw, established in code:

* Pressing `New Source` sets `SourcesPane.show_create_form`, which is
  `reactive(..., recompose=True)`. The recompose tears down and remounts
  every child of the pane -- including the `New Source` `Button` that was
  holding focus. Textual does not re-home focus after a descendant
  recompose, so `Screen.focused` lands on `None` and the form opens with
  nothing focused anywhere on the screen. Typing goes to the void.
* From `Screen.focused is None`, `Tab` restarts at the head of the screen's
  focus chain, which is the top navigation bar. Measured at 235x52 the first
  form field is **37** tabs away, so five tabs land on `nav-personas` and
  typing still goes nowhere.
* Clicking the `Name` input does focus it and typing does land, on every one
  of its three rows including its borders -- see
  `test_clicking_any_row_of_the_name_input_focuses_it`. That part of the UAT
  report did not reproduce and is recorded here so the next run does not
  re-file it.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button, Input, Select, Switch, TextArea

from Tests.UI.full_app_destination_context import (
    StaticWatchlistsScopeService,
    active_destination_screen as _active_destination_screen,
    full_app_destination_context as _visual_destination_harness,
    wait_for_selector as _wait_for_selector,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane

# 235x52 is the size the UAT ran at, so the geometry here is the geometry it
# saw. 160x42 is the small end the rest of the Watchlists parity suite covers,
# and it is the size that actually constrains this form: the Sources pane is
# 16 rows there and its toolbar takes two, so a form of five full-height rows
# puts `Create`/`Cancel` past the bottom edge -- present in the DOM, reported
# by `.region`, and unreachable.
SIZES = [(160, 42), (235, 52)]

# The form as it opens: type `rss`, so no noise control (TASK-2302 renders it
# only for the url family -- see `URL_FAMILY_FIELD_ORDER` below, which is the
# shape every geometry assertion here also has to hold for).
CREATE_FIELD_ORDER = [
    "sources-create-name",
    "sources-create-url",
    "sources-create-type",
    "sources-create-active",
    # TASK-2302: the destination. Focusable, so it joins the Tab walk, and
    # placed here because a source's watchlist is a top-level property of it,
    # not a detail below the tags.
    "sources-create-watchlist",
    "sources-create-tags",
    "sources-create-frequency",
    "sources-create-submit",
    "sources-create-cancel",
]

# The same form with a page-scrape type chosen, which is the only state the
# noise control exists in (TASK-1362's field, TASK-2302's gate). Kept as a
# separate list rather than an `if` inside the walk so the two shapes are
# both stated, and so the taller one is measured at both sizes -- the form
# had exactly zero spare rows at 160x42 when that field was added.
URL_FAMILY_FIELD_ORDER = [
    "sources-create-name",
    "sources-create-url",
    "sources-create-type",
    "sources-create-active",
    "sources-create-watchlist",
    "sources-create-tags",
    "sources-create-frequency",
    "sources-create-ignore-selectors",
    "sources-create-submit",
    "sources-create-cancel",
]

SERVER_FIELD_ORDER = [
    "sources-create-name",
    "sources-create-url",
    "sources-create-type",
    "sources-create-active",
    "sources-create-watchlist",
    "sources-create-tags",
    "sources-create-submit",
    "sources-create-cancel",
]

FORM_CASES = [
    ("local", "rss", CREATE_FIELD_ORDER),
    ("local", "url", URL_FAMILY_FIELD_ORDER),
    ("server", "rss", SERVER_FIELD_ORDER),
]


def _watchlists_host():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    return _visual_destination_harness(app, "watchlists_collections")


async def _settled_sources_pane(screen, pilot, timeout: float = 5.0) -> SourcesPane:
    """The live `SourcesPane` once the post-submit recompose storm has settled.

    TASK-1960 review, Important 2. A submit fires the pane's own form-close
    recompose *and* a full-screen recompose from `_create_source`'s worker
    chain, and for a measured 0.14-0.32s there is no `SourcesPane` mounted at
    all. Anything asserted against a pane captured before that window is
    asserted against a pruned, zero-child widget and is vacuously true.

    Waits for a pane that is attached to this screen and has finished
    mounting its own children, so callers observe the state the user
    actually ends up looking at.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        panes = list(screen.query(SourcesPane))
        if len(panes) == 1 and panes[0].is_mounted and panes[0].query("#sources-table"):
            await pilot.pause()
            return panes[0]
        await pilot.pause(0.02)
    raise AssertionError(
        f"no settled SourcesPane after {timeout}s: "
        f"found {len(list(screen.query(SourcesPane)))}"
    )


async def _open_sources_create_form(pilot, host):
    """Open the form exactly as a user does: click through `New source`."""
    screen = _active_destination_screen(host)
    screen.active_section = "sources"
    await _wait_for_selector(
        screen,
        pilot,
        "#watchlists-sources-pane",
        timeout=5.0,
    )
    pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
    screen.query_one("#sources-new-button", Button).press()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        focused = screen.focused
        selects = list(pane.query(Select))
        if (
            pane.query("#sources-create-form")
            and focused is not None
            and focused.id == "sources-create-name"
            and selects
            and all(bool(select.query("#label")) for select in selects)
        ):
            break
        await pilot.pause(0.02)
    assert pane.query("#sources-create-form"), "the create form never opened"
    assert screen.focused is not None and (
        screen.focused.id == "sources-create-name"
    ), "the create form mounted but did not focus its Name field"
    await pilot.pause()
    return screen, pane


async def _choose_source_type(pilot, pane, value: str) -> None:
    """Pick a source type, and wait out the recompose the pane runs for it.

    Assigning `Select.value` is exactly the state change a click through the
    overlay produces (`Select.Changed` either way); what is under test is
    what the pane does with it, which is rebuild the form around the noise
    control.
    """
    pane.query_one("#sources-create-type", Select).value = value
    for _ in range(200):
        await pilot.pause(0.02)
        if pane.create_draft_source_type == value:
            break
    assert pane.create_draft_source_type == value, (
        f"the pane never took source type {value!r}"
    )
    await pilot.pause(0.1)


async def _choose_runtime_backend(screen, pilot, value: str) -> None:
    """Choose a backend through the production selector and await its watcher."""
    screen.query_one("#watchlists-backend-select", Select).value = value
    for _ in range(200):
        await pilot.pause(0.02)
        if screen.runtime_backend == value:
            break
    assert screen.runtime_backend == value
    await pilot.pause(0.1)


def _option_pairs(select: Select) -> list[tuple[str, object]]:
    return [(str(label), value) for label, value in select._options]


@pytest.mark.asyncio
async def test_backend_switch_preserves_the_complete_open_create_draft():
    app = _build_test_app()
    watchlist_id = int(app.watchlist_bundle_service.create("Reading")["id"])
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")

    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_destination_screen(host)
        for _ in range(200):
            await pilot.pause(0.02)
            if screen._tree_watchlists:
                break
        assert screen._tree_watchlists

        screen, pane = await _open_sources_create_form(pilot, host)
        await _choose_source_type(pilot, pane, "url")
        pane.query_one("#sources-create-name", Input).value = "Morning"
        pane.query_one("#sources-create-url", Input).value = "https://example.test"
        pane.query_one("#sources-create-active", Switch).value = False
        pane.query_one("#sources-create-watchlist", Select).value = watchlist_id
        pane.query_one("#sources-create-tags", Input).value = "news, daily"
        pane.query_one("#sources-create-frequency", Select).value = 86_400
        pane.query_one("#sources-create-ignore-selectors", TextArea).text = (
            ".ad\n.counter"
        )
        await pilot.pause(0.2)

        await _choose_runtime_backend(screen, pilot, "server")

        assert screen.query_one("#watchlists-sources-pane", SourcesPane) is pane
        assert pane.show_create_form is True
        assert _option_pairs(pane.query_one("#sources-create-type", Select)) == [
            ("RSS", "rss"),
            ("Site", "site"),
            ("Forum", "forum"),
        ]
        assert pane.query_one("#sources-create-type", Select).value == "rss"
        assert not pane.query("#sources-create-frequency")
        assert not pane.query("#sources-create-ignore-selectors")
        assert pane.query_one("#sources-create-name", Input).value == "Morning"
        assert (
            pane.query_one("#sources-create-url", Input).value
            == "https://example.test"
        )
        assert pane.query_one("#sources-create-active", Switch).value is False
        assert (
            pane.query_one("#sources-create-watchlist", Select).value
            == watchlist_id
        )
        assert pane.query_one("#sources-create-tags", Input).value == "news, daily"

        await _choose_runtime_backend(screen, pilot, "local")

        assert screen.query_one("#watchlists-sources-pane", SourcesPane) is pane
        assert pane.show_create_form is True
        assert _option_pairs(pane.query_one("#sources-create-type", Select)) == [
            ("RSS", "rss"),
            ("Atom", "atom"),
            ("Web page", "url"),
        ]
        assert pane.query_one("#sources-create-type", Select).value == "rss"
        assert pane.query_one("#sources-create-frequency", Select).value == 86_400
        await _choose_source_type(pilot, pane, "url")
        assert (
            pane.query_one("#sources-create-ignore-selectors", TextArea).text
            == ".ad\n.counter"
        )
        assert pane.query_one("#sources-create-active", Switch).value is False
        assert (
            pane.query_one("#sources-create-watchlist", Select).value
            == watchlist_id
        )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_create_form_focuses_its_first_field_when_it_opens(size):
    """AC#3: match `WatchlistNameDialog`, which focuses its input `on_mount`."""
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        screen, _pane = await _open_sources_create_form(pilot, host)

        focused = screen.focused
        assert focused is not None, (
            "the create form opened with nothing focused anywhere on the "
            "screen: the recompose that mounts the form destroyed the "
            "`New source` Button that held focus, and nothing took its place"
        )
        assert focused.id == "sources-create-name", (
            f"expected the Name field to be focused on open, got {focused.id!r}"
        )


@pytest.mark.asyncio
async def test_a_sources_reload_interleaving_the_open_does_not_lose_focus():
    """TASK-1345: force the exact interleave the recompose/focus race needs.

    `watch_show_create_form` arms `_pending_create_focus` when the form
    opens, and `recompose()` calls `.focus()` on the first field -- but
    `Widget.focus()` only *schedules* the actual focus change (it posts to
    `app.call_later`), so it has not landed by the time `recompose()`
    returns. If a SECOND `recompose=True` assignment on the same pane
    (here: `sources` reloading, exactly what `_load_sources` does in
    production) lands in that gap, it remounts the create form's fields
    out from under the still-pending focus callback: the callback then
    fires on a **detached** widget and is silently dropped. Before
    TASK-1345's sticky-until-confirmed fix, the intent was already cleared
    to `None` before the interleave landed, so the second recompose had
    nothing to reapply either -- focus landed nowhere. See
    `SourcesPane.recompose`.

    Forces the interleave deterministically -- no sleep, no retry count.
    `_check_recompose` is the exact internal seam Textual's own
    `call_next`-driven scheduling uses to invoke `recompose()` (see
    `Widget.refresh(recompose=True)`); calling it directly, twice, back to
    back with no intervening `pilot.pause()`, reproduces the ordering the
    race depends on without guessing at asyncio scheduling: the second
    recompose starts -- and reads whatever focus-restoration state is
    current -- strictly BEFORE the first recompose's own scheduled
    `.focus()` has had any chance to land (nothing has yielded control
    back to the app's own message queue yet), exactly as `_load_sources`
    racing the form opening does with two independent asyncio tasks in
    production.
    """
    host = _watchlists_host()
    async with host.run_test(size=(160, 42)) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = "sources"
        await _wait_for_selector(
            screen, pilot, "#watchlists-sources-pane", timeout=5.0
        )
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)

        # Open the form: arms `_pending_create_focus` and queues a
        # recompose. Force that recompose to run NOW (rather than waiting
        # for Textual's own `call_next` scheduling) so the next statement
        # lands exactly in the gap this race depends on.
        pane.show_create_form = True
        await pane._check_recompose()
        assert pane.query("#sources-create-form"), "the create form should be open"
        assert pane._recompose_required is False, (
            "setup invariant: the recompose triggered by opening the form "
            "must already be consumed before the interleave below"
        )

        # The forced interleave itself: a second `recompose=True` reactive
        # assignment on the SAME pane -- exactly what `_load_sources` does
        # in production -- forced to run immediately, before the first
        # recompose's own scheduled `.focus()` has landed (nothing above
        # has awaited anything that would let the app's message queue run
        # that scheduled callback).
        pane.sources = [
            {"id": 1, "name": "AI News RSS", "source_type": "rss", "active": True},
        ]
        await pane._check_recompose()

        # NOW let everything settle: both recomposes' scheduled `.focus()`
        # calls get a chance to land, whichever is last wins.
        await pilot.pause()
        await pilot.pause()

        assert pane.query("#sources-create-form"), "the create form should still be open"
        assert screen.focused is not None and screen.focused.id == "sources-create-name", (
            "a `sources` reload interleaving the create form's own opening "
            "lost the focus intent: expected 'sources-create-name', got "
            f"{screen.focused.id if screen.focused else None!r}"
        )


@pytest.mark.asyncio
async def test_an_external_rebuild_does_not_yank_focus_back_to_the_first_field():
    """TASK-1345 regression guard for case 2 in `SourcesPane.recompose`.

    The sticky-until-confirmed fix must not regress the OTHER case
    `recompose` handles: once the form-opening focus has CONFIRMED landed
    (`_confirm_create_focus` has cleared `_pending_create_focus`), the user
    is free to Tab/click to a different field. A LATER, unrelated rebuild
    of this pane -- `sources` reloading, a filter changing -- must restore
    wherever the user actually is, not yank them back to field 0. This is
    exactly what the confirm-clear exists for: once it fires,
    `_pending_create_focus` is `None` again, so later recomposes fall
    through to `_focused_create_field_id()`, which reports the user's
    CURRENT focus rather than the stale opening intent.
    """
    host = _watchlists_host()
    async with host.run_test(size=(160, 42)) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)
        # `_open_sources_create_form` already waits for the opening focus
        # to land; give the confirm callback a moment too before asserting
        # the setup invariant below.
        for _ in range(10):
            if pane._pending_create_focus is None:
                break
            await pilot.pause(0.02)
        assert pane._pending_create_focus is None, (
            "setup invariant: the opening intent must already be confirmed "
            "and cleared before the external rebuild below, or this test "
            "cannot tell case 2 apart from case 1"
        )

        # Move to a different field, as a user would.
        pane.query_one("#sources-create-url", Input).focus()
        await pilot.pause()
        assert screen.focused is not None and screen.focused.id == "sources-create-url"

        # An unrelated external rebuild: `sources` reloading, exactly like
        # `_load_sources` after a background refresh -- nothing to do with
        # the create form itself.
        pane.sources = [
            {"id": 1, "name": "AI News RSS", "source_type": "rss", "active": True},
        ]
        await pilot.pause()
        await pilot.pause()

        assert pane.query("#sources-create-form"), "the create form should still be open"
        assert screen.focused is not None and screen.focused.id == "sources-create-url", (
            "an unrelated `sources` reload yanked focus back to the first "
            "field instead of leaving it on the user's field: got "
            f"{screen.focused.id if screen.focused else None!r}"
        )


@pytest.mark.asyncio
async def test_an_external_rebuild_does_not_yank_focus_to_a_stale_pending_target():
    """TASK-1345 Qodo follow-up: FIX A's `recompose` ordering.

    Before this fix, `recompose` computed
    `restore = self._pending_create_focus or self._focused_create_field_id()`
    -- the sticky intent WON over the user's actual current focus. That is
    correct while the intent is still mid-burst (`screen.focused` is `None`,
    see `test_a_sources_reload_interleaving_the_open_does_not_lose_focus`),
    but wrong the moment the user has Tabbed to a real in-form field WHILE
    a stale intent from an earlier open is still armed and has not yet been
    observed as confirmed: a later, unrelated rebuild would restore the
    stale target and yank the user off the field they moved to.

    This forces exactly that: arm `_pending_create_focus` back to field 0
    by hand (simulating a confirm callback that has not caught up yet),
    move real focus to field 1, then force an external rebuild
    (`pane._check_recompose()`, the same seam
    `test_a_sources_reload_interleaving_the_open_does_not_lose_focus` uses)
    and confirm focus stays on field 1 -- NOT yanked back to field 0.
    """
    host = _watchlists_host()
    async with host.run_test(size=(160, 42)) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)

        # Move to a different field, as a user would.
        pane.query_one("#sources-create-url", Input).focus()
        await pilot.pause()
        assert screen.focused is not None and screen.focused.id == "sources-create-url"

        # Arm the sticky intent back to field 0 directly -- simulating a
        # confirm callback for an earlier open that has not yet observed
        # focus landing (the exact state a stale `_pending_create_focus`
        # would be in if it survived past the user's own Tab).
        pane._pending_create_focus = "sources-create-name"

        # An external rebuild -- `sources` reloading, exactly like
        # `_load_sources` after a background refresh -- forced to run NOW
        # via the same internal seam the interleaving test above uses, so
        # this does not depend on asyncio scheduling order.
        pane.sources = [
            {"id": 1, "name": "AI News RSS", "source_type": "rss", "active": True},
        ]
        await pane._check_recompose()
        await pilot.pause()
        await pilot.pause()

        assert pane.query("#sources-create-form"), "the create form should still be open"
        assert screen.focused is not None and screen.focused.id == "sources-create-url", (
            "a stale pending focus intent yanked the user back to field 0 "
            "instead of leaving them on the field they had Tabbed to: got "
            f"{screen.focused.id if screen.focused else None!r}"
        )


@pytest.mark.asyncio
async def test_confirm_create_focus_gives_up_after_max_attempts():
    """TASK-1345 Qodo follow-up: FIX B's bound on `_confirm_create_focus`.

    Before this fix, `_confirm_create_focus` only ever cleared
    `_pending_create_focus` when `screen.focused.id == target` EXACTLY, and
    otherwise rescheduled itself via `call_after_refresh` unconditionally --
    forever, if focus never becomes that exact target (the form stays open,
    focus parked somewhere that is never the target). This drives the
    method at the bound directly -- with `screen.focused` still `None`, the
    one case that must reschedule below the bound -- and asserts it gives
    up instead: the intent clears and no further reschedule is queued.
    """
    host = _watchlists_host()
    async with host.run_test(size=(160, 42)) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)
        # Let the form's own opening confirmation land first, so the
        # manual arm below is not racing it.
        for _ in range(10):
            if pane._pending_create_focus is None:
                break
            await pilot.pause(0.02)
        assert pane._pending_create_focus is None

        screen.set_focus(None)
        await pilot.pause()
        assert screen.focused is None, "setup invariant: focus must be genuinely absent"

        target = "sources-create-name"
        pane._pending_create_focus = target

        calls: list[tuple] = []
        original_call_after_refresh = pane.call_after_refresh

        def _spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original_call_after_refresh(*args, **kwargs)

        pane.call_after_refresh = _spy

        pane._confirm_create_focus(target, attempts=pane._CREATE_FOCUS_CONFIRM_MAX_ATTEMPTS)

        assert pane._pending_create_focus is None, (
            "the intent must clear once the reschedule bound is reached, "
            "or it stays armed forever with focus never confirmed"
        )
        assert calls == [], (
            "_confirm_create_focus rescheduled itself past its own bound: "
            f"{calls}"
        )


@pytest.mark.asyncio
async def test_confirm_create_focus_clears_on_any_in_form_field_not_only_target():
    """TASK-1345 Qodo follow-up: FIX B's any-in-form clear.

    The bound above only fires while `screen.focused` stays `None`. The
    OTHER half of FIX B is that landing on any in-form field -- not only
    the original `target` -- also clears the intent and does not
    reschedule, which is what lets FIX A's ordering
    (`test_an_external_rebuild_does_not_yank_focus_to_a_stale_pending_target`)
    actually stick: if this still cleared on `== target` only, the stale
    intent would keep re-arming.
    """
    host = _watchlists_host()
    async with host.run_test(size=(160, 42)) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)
        for _ in range(10):
            if pane._pending_create_focus is None:
                break
            await pilot.pause(0.02)
        assert pane._pending_create_focus is None

        pane.query_one("#sources-create-url", Input).focus()
        await pilot.pause()
        assert screen.focused is not None and screen.focused.id == "sources-create-url"

        target = "sources-create-name"
        pane._pending_create_focus = target

        calls: list[tuple] = []
        original_call_after_refresh = pane.call_after_refresh

        def _spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original_call_after_refresh(*args, **kwargs)

        pane.call_after_refresh = _spy

        pane._confirm_create_focus(target)

        assert pane._pending_create_focus is None, (
            "landing on a sibling in-form field (not the original target) "
            "must still clear the pending intent"
        )
        assert calls == [], (
            "_confirm_create_focus rescheduled instead of recognising the "
            f"user is already on an in-form field: {calls}"
        )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_typing_straight_after_opening_the_form_lands_in_name(size):
    """AC#5, the headline UAT step: open the form and type. No clicking."""
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        _screen, pane = await _open_sources_create_form(pilot, host)

        await pilot.press(*"Morning")
        await pilot.pause(0.1)

        assert pane.query_one("#sources-create-name", Input).value == "Morning", (
            "typing straight after opening the create form put the text "
            "nowhere -- exactly what the UAT reported"
        )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize(
    "runtime_backend,source_type,field_order",
    FORM_CASES,
    ids=["local-feed", "local-page", "server-feed"],
)
@pytest.mark.asyncio
async def test_tab_walks_the_create_form_in_visual_order(
    size, runtime_backend, source_type, field_order
):
    """AC#4: from the focused first field, `Tab` follows what the eye sees."""
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        screen = _active_destination_screen(host)
        if runtime_backend != "local":
            await _choose_runtime_backend(screen, pilot, runtime_backend)
        screen, pane = await _open_sources_create_form(pilot, host)
        if runtime_backend == "server":
            assert not pane.query("#sources-create-frequency")
            assert not pane.query("#sources-create-ignore-selectors")
        if source_type != "rss":
            await _choose_source_type(pilot, pane, source_type)
            pane.query_one("#sources-create-name", Input).focus()
            await pilot.pause(0.1)

        assert screen.focused is not None, "nothing focused; see AC#3 test"
        seen = [screen.focused.id]
        for _ in range(len(field_order) - 1):
            await pilot.press("tab")
            await pilot.pause(0.05)
            seen.append(screen.focused.id if screen.focused else None)

        assert seen == field_order, (
            f"Tab order through the create form is {seen}, expected "
            f"{field_order}"
        )

        # "Visual order" is not just DOM order: every step must move down the
        # form, or right along the same row, and must stay inside the pane.
        previous = None
        for field_id in field_order:
            widget = pane.query_one(f"#{field_id}")
            region = widget.region
            assert region.x >= pane.region.x and region.right <= pane.region.right, (
                f"#{field_id} is outside the Sources pane horizontally: "
                f"{region} vs {pane.region}"
            )
            if previous is not None:
                assert (region.y, region.x) >= previous, (
                    f"#{field_id} at {region} comes before the field that "
                    f"precedes it in Tab order (previous top-left {previous})"
                )
            previous = (region.y, region.x)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_create_and_cancel_sit_side_by_side_like_the_dialog(size):
    """AC#6: `WatchlistNameDialog` pairs its buttons in one `.dialog-buttons`
    row. The form stacked them with blank rows in between."""
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)

        submit = pane.query_one("#sources-create-submit", Button)
        cancel = pane.query_one("#sources-create-cancel", Button)
        assert submit.region.y == cancel.region.y, (
            f"Create and Cancel are on different rows: {submit.region} vs "
            f"{cancel.region} -- the New-watchlist dialog puts its pair on one"
        )
        assert cancel.region.x > submit.region.x, (
            "Cancel should sit to the right of Create, as in the dialog"
        )

        # Regions alone would not catch a pair that is laid out on one row and
        # then clipped, so require both labels on the same painted row.
        strips = screen._compositor.render_strips()
        painted = "".join(seg.text for seg in strips[submit.region.y])
        assert "Create" in painted and "Cancel" in painted, (
            f"row {submit.region.y} paints {painted.strip()!r}; both buttons "
            "must reach the screen on one row"
        )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize(
    "runtime_backend,source_type,field_order",
    FORM_CASES,
    ids=["local-feed", "local-page", "server-feed"],
)
@pytest.mark.asyncio
async def test_the_whole_create_form_fits_inside_the_sources_pane(
    size, runtime_backend, source_type, field_order
):
    """Every control has to be reachable, not merely present.

    The `Grid` this form used to be took `height: 1fr` and spread seven
    controls over 23 rows. `Vertical` + `height: auto` fixes the spread, but
    an auto-height form can just as easily grow past the bottom of a pane
    that is 16 rows tall at 160x42 -- and a `Create` button below the fold
    is exactly as unusable as one that was never focusable. Asserted on both
    the geometry and the paint, because a widget clipped by an ancestor
    still reports a perfectly sensible `.region`.
    """
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        screen = _active_destination_screen(host)
        if runtime_backend != "local":
            await _choose_runtime_backend(screen, pilot, runtime_backend)
        screen, pane = await _open_sources_create_form(pilot, host)
        if runtime_backend == "server":
            assert not pane.query("#sources-create-frequency")
            assert not pane.query("#sources-create-ignore-selectors")
        if source_type != "rss":
            await _choose_source_type(pilot, pane, source_type)

        strips = screen._compositor.render_strips()
        for field_id in field_order:
            region = pane.query_one(f"#{field_id}").region
            assert region.bottom <= pane.region.bottom, (
                f"#{field_id} at {region} hangs below the Sources pane "
                f"{pane.region} at {size} -- the user cannot reach it"
            )
            assert region.y >= pane.region.y, (
                f"#{field_id} at {region} starts above the Sources pane "
                f"{pane.region}"
            )

        # The table must survive too: the form is transient, the list is not.
        table_region = pane.query_one("#sources-table").region
        assert table_region.height >= 1 and table_region.bottom <= pane.region.bottom, (
            f"#sources-table is {table_region} inside {pane.region} at {size}"
        )
        header = "".join(seg.text for seg in strips[table_region.y])
        assert "Name" in header, (
            f"the sources table header never reaches the screen at {size}: "
            f"row {table_region.y} paints {header.strip()!r}"
        )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_a_source_can_be_created_end_to_end_through_the_form(size):
    """AC#2: open, type, tab, type, submit -- no widget `.value =` anywhere."""
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)
        created = AsyncMock(return_value={"id": 1, "name": "Morning"})
        screen._controller.create_source = created

        await pilot.press(*"Morning")
        await pilot.press("tab")
        await pilot.pause(0.05)
        await pilot.press(*"https://example.com/feed")
        await pilot.pause(0.05)

        assert pane.query_one("#sources-create-name", Input).value == "Morning"
        assert (
            pane.query_one("#sources-create-url", Input).value
            == "https://example.com/feed"
        )

        pane.query_one("#sources-create-submit", Button).press()
        for _ in range(200):
            if created.await_count == 1 and not pane.query("#sources-create-form"):
                break
            await pilot.pause(0.01)

        assert created.await_count == 1, "pressing Create never reached the controller"
        payload = created.await_args.kwargs["payload"]
        assert payload["name"] == "Morning"
        assert payload["url"] == "https://example.com/feed"

        # Re-query the pane from the SCREEN rather than reusing the captured
        # `pane` (TASK-1960 review, Important 2). Submitting fires the pane's
        # own form-close recompose AND -- through `_create_source`'s worker
        # chain -- a full-screen recompose that replaces the SourcesPane
        # wholesale. Measured during that review, the captured `pane` is
        # `_pruning=True, children=0, is_running=False` by the time this runs,
        # and there is NO SourcesPane on screen at all for 0.14-0.32s. So
        # `assert not pane.query(...)` against the captured object was true no
        # matter what: verified during this fix wave that it stayed GREEN under
        # a mutation which left a create form on the settled pane.
        live_pane = await _settled_sources_pane(screen, pilot)
        # Positively: the rebuilt pane really is populated, so "no form" below
        # cannot be satisfied by an empty or half-mounted pane.
        assert live_pane.query("#sources-table"), (
            "the settled Sources pane never remounted its table"
        )
        assert not live_pane.query("#sources-create-form"), (
            "the form should be gone from the settled Sources pane once the "
            "source is submitted"
        )
        assert not screen.query("#sources-create-form"), (
            "a create form survived somewhere else on the screen"
        )


@pytest.mark.asyncio
async def test_submission_backend_governs_creation_filing_and_confirmation():
    app = _build_test_app()
    bundle = app.watchlist_bundle_service
    watchlist_id = int(bundle.create("Reading")["id"])
    source_id = int(
        bundle._db.add_subscription(
            name="Race Feed",
            type="rss",
            source="https://race.example/feed",
        )
    )
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")

    entered = asyncio.Event()
    release = asyncio.Event()
    create_calls: list[dict[str, object]] = []
    reload_backends: list[str] = []
    notices: list[str] = []

    async def gated_create(*, runtime_backend, payload):
        create_calls.append(
            {"runtime_backend": runtime_backend, "payload": dict(payload)}
        )
        entered.set()
        await release.wait()
        return {"id": f"local:subscription:{source_id}", "source_id": source_id}

    async def record_source_reload(*, runtime_backend, **_kwargs):
        reload_backends.append(runtime_backend)
        return []

    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_destination_screen(host)
        for _ in range(200):
            await pilot.pause(0.02)
            if screen._tree_watchlists:
                break
        assert screen._tree_watchlists
        screen._controller.create_source = gated_create
        screen._controller.list_sources = record_source_reload
        screen._notify_watchlists = (
            lambda message, severity="information", **_kwargs: notices.append(message)
        )

        _screen, pane = await _open_sources_create_form(pilot, host)
        pane.query_one("#sources-create-name", Input).value = "Race Feed"
        pane.query_one("#sources-create-url", Input).value = (
            "https://race.example/feed"
        )
        pane.query_one("#sources-create-watchlist", Select).value = watchlist_id
        await pilot.pause(0.2)
        pane.query_one("#sources-create-submit", Button).press()

        await asyncio.wait_for(entered.wait(), timeout=2)
        assert create_calls[0]["runtime_backend"] == "local"
        await _choose_runtime_backend(screen, pilot, "server")
        release.set()
        await host.workers.wait_for_complete()
        for _ in range(200):
            await pilot.pause(0.02)
            if reload_backends and notices:
                break

        assert bundle.list_sources(watchlist_id) == [source_id]
        assert any('Source created in "Reading".' == notice for notice in notices)
        assert reload_backends[-1] == "server"


@pytest.mark.asyncio
async def test_unrelated_create_failure_keeps_the_generic_error_copy():
    host = _watchlists_host()
    notices: list[tuple[str, str]] = []

    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)

        async def fail_create(*, runtime_backend, payload):
            raise ValueError("the supported RSS source failed for another reason")

        screen._controller.create_source = fail_create
        screen.app_instance.notify = (
            lambda message, severity="information", **_kwargs: notices.append(
                (message, severity)
            )
        )
        pane.query_one("#sources-create-name", Input).value = "Broken RSS"
        pane.query_one("#sources-create-url", Input).value = "https://broken.test/rss"
        await pilot.pause(0.1)
        pane.query_one("#sources-create-submit", Button).press()
        for _ in range(200):
            await pilot.pause(0.02)
            if notices:
                break

        assert notices == [("Failed to create source.", "error")]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_clicking_any_row_of_the_name_input_focuses_it(size):
    """AC#1, the part of the UAT report that did NOT reproduce.

    The UAT clicked "the `Name` input's own row" and reported nothing landed.
    A bordered `Input` is three rows, and only the middle one is content, so
    the natural suspicion was that the click hit a border row. It does not
    matter: `Screen.get_widget_at` returns the `Input` for all three rows and
    Textual focuses on mouse-down, so a click anywhere in the widget focuses
    it and typing lands. Pinned so the next UAT does not re-file this.
    """
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)

        name_region = pane.query_one("#sources-create-name", Input).region
        column = name_region.x + 10
        for row in range(name_region.y, name_region.bottom):
            screen.set_focus(None)
            await pilot.pause(0.05)
            await pilot.click(offset=(column, row))
            await pilot.pause(0.1)
            assert screen.focused is not None and (
                screen.focused.id == "sources-create-name"
            ), f"clicking ({column},{row}) did not focus Name"
            await pilot.press("x")
            await pilot.pause(0.05)
            assert pane.query_one("#sources-create-name", Input).value.endswith("x"), (
                f"typing after clicking ({column},{row}) did not land"
            )
            pane.query_one("#sources-create-name", Input).value = ""
            await pilot.pause(0.05)


@pytest.mark.asyncio
async def test_creating_a_source_refreshes_the_table_and_the_tree_counts():
    """TASK-1040: creation only refreshed the view that read the data directly.

    `_create_source` called `_refresh_local_wc_snapshot` and
    `_refresh_overview_data` but never `_load_sources` or `_load_tree_data`,
    so `#sources-table` kept the list from before and the rail's counts stayed
    behind. Measured live: the rail read `All sources  0` while the centre read
    `Feeds in All sources (1)` -- the same thing, disagreeing on one screen.

    A first-time user is told nothing happened, and reasonably tries again.
    """
    host = _watchlists_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen, _pane = await _open_sources_create_form(pilot, host)

        created: list[dict] = []

        async def fake_create(*, runtime_backend, payload):
            created.append(payload)

        async def fake_list(*, runtime_backend, limit=100):
            return [
                {"id": 1, "name": "AI News RSS", "source_type": "rss", "active": True}
            ] if created else []

        screen._controller.create_source = fake_create
        screen._controller.list_sources = fake_list

        reloaded: list[str] = []
        real_sources = screen._load_sources
        real_tree = screen._load_tree_data

        async def watch_sources():
            reloaded.append("sources")
            await real_sources()

        def watch_tree():
            reloaded.append("tree")
            return real_tree()

        screen._load_sources = watch_sources
        screen._load_tree_data = watch_tree

        await screen._create_source(
            {"name": "AI News RSS", "url": "https://x/f"},
            runtime_backend="local",
        )
        for _ in range(20):
            await pilot.pause()
            if {"sources", "tree"} <= set(reloaded):
                break

        assert "sources" in reloaded, (
            "creating a source must reload the sources table; without it the "
            "table keeps the old list until the user leaves and comes back"
        )
        assert "tree" in reloaded, (
            "creating a source must reload the tree counts; without it the "
            "rail says 0 while the centre says 1"
        )


# --- TASK-2200 -------------------------------------------------------------
#
# TASK-1960 proved the crash class here came from the SCREEN recomposing out of
# `_apply_local_wc_snapshot`/`_load_tree_data` while this pane was mid-recompose
# of its own, and recorded a second, masked defect on the same path: the pane's
# form-close recompose can mount *nothing*, invisible only because the screen
# rebuilt the pane wholesale straight afterwards. Both tests below assert
# against the SAME pane instance the user was already looking at, so neither can
# be satisfied by a screen-level rebuild papering over an empty pane.


@pytest.mark.asyncio
async def test_a_background_refresh_does_not_tear_down_an_open_create_form():
    """AC#1: a half-typed create form survives a background load landing.

    Before TASK-2200 both loaders ended in `refresh(recompose=True)`, which
    replaced the pane, the form and every widget in it. The draft *text*
    survived (the screen mirrors it), so text alone does not discriminate --
    these assertions are on widget identity, which only survives if nothing
    rebuilt the region.
    """
    host = _watchlists_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)

        await pilot.press(*"Morning")
        await pilot.pause(0.05)
        name_input = pane.query_one("#sources-create-name", Input)
        assert name_input.value == "Morning", "precondition: the draft was typed"
        assert not screen.query("#wc-service-error")

        # Both background loaders, together -- the exact pair `_create_source`
        # fires, and the pair whose recomposes TASK-1960 caught destroying this
        # pane.
        screen._apply_local_wc_snapshot(
            (), 0, True, "Watchlists services unavailable; retry Watchlists later.", None
        )
        screen._load_tree_data()
        for _ in range(300):
            await pilot.pause(0.01)
            if screen.query("#wc-service-error"):
                break

        assert screen.query("#wc-service-error"), (
            "precondition: the background snapshot really did land and repaint "
            "the centre header"
        )
        assert screen.query_one("#watchlists-sources-pane", SourcesPane) is pane, (
            "the background load replaced the Sources pane"
        )
        assert pane.query_one("#sources-create-name", Input) is name_input, (
            "the background load rebuilt the create form's Name field out from "
            "under the user"
        )
        assert name_input.value == "Morning"
        assert pane.query("#sources-create-form"), "the form must still be open"


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_closing_the_create_form_repopulates_the_same_pane(size):
    """AC#4: form-close no longer leans on a screen rebuild to look right.

    The masked defect TASK-1960 recorded: `SourcesPane`'s form-close recompose
    silently mounts nothing when the screen's own recompose prunes it
    mid-flight, and that was invisible only because the screen immediately
    rebuilt the pane. With the screen recompose gone, the pane the user keeps
    looking at is the pane that has to come back correct -- so this asserts
    identity first, then that the same instance really did remount its table
    and drop its form.
    """
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)
        created = AsyncMock(return_value={"id": 1, "name": "Morning"})
        screen._controller.create_source = created

        await pilot.press(*"Morning")
        await pilot.press("tab")
        await pilot.pause(0.05)
        await pilot.press(*"https://example.com/feed")
        await pilot.pause(0.05)

        pane.query_one("#sources-create-submit", Button).press()
        settled = await _settled_sources_pane(screen, pilot)

        assert created.await_count == 1, "precondition: Create reached the controller"
        assert settled is pane, (
            "the create flow's background loaders must not replace the pane -- "
            "the whole point is that the pane's own form-close is what has to "
            "come back correct"
        )
        assert pane.query("#sources-table"), (
            "the pane's own form-close recompose mounted nothing: the masked "
            "TASK-1960 defect, now unmasked"
        )
        assert not pane.query("#sources-create-form"), (
            "the create form must be gone from the pane the user is looking at"
        )
