"""Briefing preset manager modal (spec #2 phase 2a, Task 3).

Task 1 shipped `briefing_presets` CRUD; Task 2 shipped `validate_roster` /
`dump_roster` / `load_roster` / `ScriptCastError`. This is the one surface
that puts those together for a person: `BriefingPresetModal` lists, creates,
edits and deletes presets, and `WatchlistsCollectionsScreen
._open_briefing_preset_manager` is the mount/dismiss wiring around it (Task
4 adds the toolbar button that calls it; nothing here depends on that
button existing).

Every DB-touching test uses a real, file-backed `SubscriptionsDB` (never
`:memory:`) -- the modal's writes go through `asyncio.to_thread`, and
`SubscriptionsDB.conn` is thread-local, so an in-memory database opened on
the test's own thread would be empty and unmigrated on the executor thread
(`test_briefing_cast.py`'s own `_db` helper documents this exact trap).

Geometry assertions run against the REAL generated stylesheet
(`ProductionCSSDestinationHarness.CSS_PATH`), not a widget-embedded
`DEFAULT_CSS` -- the modal's styles live in `css/features/_watchlists.tcss`
and are only real once the bundle is regenerated from them.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select, Static

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.test_destination_visual_parity_correction import (
    ProductionCSSDestinationHarness,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.briefing_cast import (
    dump_roster,
    load_roster,
    validate_roster,
)
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules.briefing_preset_modal import (
    BriefingPresetModal,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

# Marked so CI actually runs this file: the unit job selects `-m unit` and
# the UI job runs `Tests/UI` plus `Tests -m ui --ignore=Tests/UI`, so an
# unmarked test in `Tests/Watchlists` is collected by nothing (matches
# `test_watchlists_artifacts_pane.py`'s own header note).
pytestmark = pytest.mark.ui


def _db(tmp_path) -> SubscriptionsDB:
    """A real, file-backed `SubscriptionsDB` -- not `:memory:`.

    Every write this modal issues goes through `asyncio.to_thread`.
    `SubscriptionsDB.conn` is thread-local, so an in-memory connection
    opened on the test's own thread would be a brand-new, unmigrated,
    empty database on the executor thread. Matches
    `test_briefing_cast.py`'s own `_db(tmp_path)`.
    """
    return SubscriptionsDB(tmp_path / "subs.db", "test")


async def _wait_until(pilot, predicate, *, ticks: int = 80) -> bool:
    for _ in range(ticks):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _wait_for_screen_type(host, screen_type, pilot, *, ticks: int = 80):
    for _ in range(ticks):
        await pilot.pause()
        if isinstance(host.screen, screen_type):
            return host.screen
    raise AssertionError(f"{screen_type.__name__} never opened")


class _ModalHost(App[None]):
    """A bare host mounting the REAL generated bundle, not a widget's own
    `DEFAULT_CSS` -- `BriefingPresetModal`'s styles live in
    `css/features/_watchlists.tcss` and are only real once regenerated into
    this exact file (the three-way-vacuity lesson: a geometry assertion
    against a stylesheet the shipped code does not actually load proves
    nothing).
    """

    CSS_PATH = ProductionCSSDestinationHarness.CSS_PATH

    def compose(self) -> ComposeResult:
        yield Static("host")


# --- Modal: listing, create, validation, delete, editing -------------------


@pytest.mark.asyncio
async def test_modal_lists_presets_name_ascending(tmp_path):
    db = _db(tmp_path)
    one_speaker = dump_roster(validate_roster([{"name": "Host"}]))
    db.insert_briefing_preset("Zebra hour", roster_json=one_speaker)
    db.insert_briefing_preset("Alpha digest", roster_json=one_speaker)
    modal = BriefingPresetModal(db, character_options=[], voice_options=[])

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal)
        assert await _wait_until(
            pilot, lambda: bool(modal.query("#bpm-preset-list Button"))
        )
        labels = [str(button.label) for button in modal.query("#bpm-preset-list Button")]

    assert labels == ["Alpha digest", "Zebra hour"]


@pytest.mark.asyncio
async def test_create_with_two_speakers_persists_exactly_the_roster_entered(tmp_path):
    db = _db(tmp_path)
    modal = BriefingPresetModal(
        db,
        character_options=[("Ada", 7)],
        voice_options=[("Warm narrator", "voice-1")],
    )
    results: list[bool] = []

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal, results.append)
        assert await _wait_until(pilot, lambda: modal.is_mounted)

        modal.query_one("#bpm-name-input", Input).value = "Morning digest"
        modal.query_one("#bpm-speaker-name-0", Input).value = "Host"
        modal.query_one("#bpm-speaker-role-0", Input).value = "Leads the segment"
        modal.query_one("#bpm-speaker-character-0", Select).value = 7
        modal.query_one("#bpm-speaker-voice-0", Select).value = "voice-1"

        await pilot.click("#bpm-add-speaker")
        await pilot.pause()
        modal.query_one("#bpm-speaker-name-1", Input).value = "Guest"

        await pilot.click("#bpm-save")
        assert await _wait_until(pilot, lambda: bool(db.list_briefing_presets()))

        await pilot.click("#bpm-close")
        assert await _wait_until(pilot, lambda: results != [])

    assert results == [True]
    rows = db.list_briefing_presets()
    assert len(rows) == 1
    assert rows[0]["name"] == "Morning digest"
    assert load_roster(rows[0]["roster_json"]) == [
        {
            "name": "Host",
            "role_prompt": "Leads the segment",
            "character_card_id": 7,
            "voice_profile_id": "voice-1",
        },
        {
            "name": "Guest",
            "role_prompt": "",
            "character_card_id": None,
            "voice_profile_id": None,
        },
    ]


@pytest.mark.asyncio
async def test_duplicate_speaker_name_shows_inline_error_and_does_not_persist(
    tmp_path,
):
    db = _db(tmp_path)
    modal = BriefingPresetModal(db, character_options=[], voice_options=[])

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal)
        assert await _wait_until(pilot, lambda: modal.is_mounted)

        modal.query_one("#bpm-name-input", Input).value = "Duplicate names"
        modal.query_one("#bpm-speaker-name-0", Input).value = "Sam"
        await pilot.click("#bpm-add-speaker")
        await pilot.pause()
        modal.query_one("#bpm-speaker-name-1", Input).value = "Sam"

        await pilot.click("#bpm-save")
        assert await _wait_until(
            pilot,
            lambda: bool(
                str(modal.query_one("#bpm-error", Static).renderable).strip()
            ),
        )

        error_content = modal.query_one("#bpm-error", Static).renderable
        # `validate_roster`'s `ScriptCastError` message interpolates the
        # speaker name the user typed -- it must render `markup=False`-safe
        # (a `Text` object, never `Text.from_markup` or a bare string).
        assert isinstance(error_content, Text)
        assert "sam" in str(error_content).lower()
        assert "duplicate" in str(error_content).lower()

    # `validate_roster` is the gate: row count is unchanged.
    assert db.list_briefing_presets() == []


@pytest.mark.asyncio
async def test_inline_error_paints_a_markup_shaped_speaker_name_literally(tmp_path):
    """Review round 1, Minor 2: probed as safe by the reviewer, now pinned.

    `validate_roster`'s `ScriptCastError` message interpolates the
    offending speaker's name verbatim. A name that is itself markup-shaped
    must still paint as literal characters -- proof, not just code-reading,
    that `_show_error` never reaches `Text.from_markup` or a bare `str`.
    """
    db = _db(tmp_path)
    modal = BriefingPresetModal(db, character_options=[], voice_options=[])
    markup_name = "has [bold red]markup[/] inside"

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal)
        assert await _wait_until(pilot, lambda: modal.is_mounted)

        modal.query_one("#bpm-name-input", Input).value = "Markup names"
        modal.query_one("#bpm-speaker-name-0", Input).value = markup_name
        await pilot.click("#bpm-add-speaker")
        await pilot.pause()
        modal.query_one("#bpm-speaker-name-1", Input).value = markup_name

        await pilot.click("#bpm-save")
        assert await _wait_until(
            pilot,
            lambda: bool(
                str(modal.query_one("#bpm-error", Static).renderable).strip()
            ),
        )

        error_content = modal.query_one("#bpm-error", Static).renderable
        assert isinstance(error_content, Text)
        # Literal, unparsed: the exact source string -- brackets included --
        # appears verbatim. A markup-parsing surface would instead consume
        # `[bold red]...[/]` as a style span, and the plain text read back
        # would NOT contain the brackets at all.
        assert markup_name in str(error_content)
        assert "duplicate" in str(error_content).lower()

    assert db.list_briefing_presets() == []


@pytest.mark.asyncio
async def test_delete_asks_confirmation_and_hard_deletes(tmp_path):
    db = _db(tmp_path)
    preset_id = db.insert_briefing_preset(
        "Old preset", roster_json=dump_roster(validate_roster([{"name": "Host"}]))
    )
    modal = BriefingPresetModal(db, character_options=[], voice_options=[])
    results: list[bool] = []

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal, results.append)
        assert await _wait_until(
            pilot, lambda: bool(modal.query("#bpm-preset-list Button"))
        )

        await pilot.click(f"#bpm-preset-btn-{preset_id}")
        await pilot.pause()
        assert modal.query_one("#bpm-delete", Button).disabled is False

        await pilot.click("#bpm-delete")
        assert await _wait_until(
            pilot, lambda: isinstance(app.screen, ConfirmationDialog)
        )
        await pilot.click("#confirm-button")

        assert await _wait_until(pilot, lambda: db.get_briefing_preset(preset_id) is None)

        await pilot.click("#bpm-close")
        assert await _wait_until(pilot, lambda: results != [])

    assert results == [True]
    assert db.get_briefing_preset(preset_id) is None


@pytest.mark.asyncio
async def test_delete_cancelled_leaves_the_preset_in_place(tmp_path):
    db = _db(tmp_path)
    preset_id = db.insert_briefing_preset(
        "Keep me", roster_json=dump_roster(validate_roster([{"name": "Host"}]))
    )
    modal = BriefingPresetModal(db, character_options=[], voice_options=[])

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal)
        assert await _wait_until(
            pilot, lambda: bool(modal.query("#bpm-preset-list Button"))
        )

        await pilot.click(f"#bpm-preset-btn-{preset_id}")
        await pilot.pause()
        await pilot.click("#bpm-delete")
        assert await _wait_until(
            pilot, lambda: isinstance(app.screen, ConfirmationDialog)
        )
        await pilot.click("#cancel-button")
        assert await _wait_until(pilot, lambda: modal.is_mounted and modal.is_current)

    assert db.get_briefing_preset(preset_id) is not None


@pytest.mark.asyncio
async def test_editing_preserves_untouched_fields(tmp_path):
    db = _db(tmp_path)
    original_roster = validate_roster(
        [{"name": "Host", "role_prompt": "Warm", "character_card_id": None}]
    )
    preset_id = db.insert_briefing_preset(
        "Weekly wrap",
        roster_json=dump_roster(original_roster),
        style_notes="Keep it upbeat.",
        provider="openai",
        model="gpt-4o",
    )
    modal = BriefingPresetModal(db, character_options=[], voice_options=[])

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal)
        assert await _wait_until(
            pilot, lambda: bool(modal.query("#bpm-preset-list Button"))
        )

        await pilot.click(f"#bpm-preset-btn-{preset_id}")
        await pilot.pause()
        assert modal.query_one("#bpm-model-input", Input).value == "gpt-4o"
        assert modal.query_one("#bpm-name-input", Input).value == "Weekly wrap"

        # Only the model field is touched.
        modal.query_one("#bpm-model-input", Input).value = "gpt-4o-mini"
        await pilot.click("#bpm-save")
        assert await _wait_until(
            pilot,
            lambda: (db.get_briefing_preset(preset_id) or {}).get("model")
            == "gpt-4o-mini",
        )

    row = db.get_briefing_preset(preset_id)
    assert row["name"] == "Weekly wrap"
    assert row["style_notes"] == "Keep it upbeat."
    assert row["provider"] == "openai"
    assert row["model"] == "gpt-4o-mini"
    assert load_roster(row["roster_json"]) == original_roster


@pytest.mark.asyncio
async def test_editing_an_existing_preset_and_saving_dismisses_true(tmp_path):
    """Review round 1, Minor 3: the other half of the dismiss contract.

    `test_close_without_any_change_dismisses_false` already pins "no
    changes -> False". This pins the mirror case: a session that opens on
    an EXISTING preset, edits it, saves, and closes must dismiss `True` --
    editing a row is still "something changed", not only "something
    created".
    """
    db = _db(tmp_path)
    preset_id = db.insert_briefing_preset(
        "Old name", roster_json=dump_roster(validate_roster([{"name": "Host"}]))
    )
    modal = BriefingPresetModal(db, character_options=[], voice_options=[])
    results: list[bool] = []

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal, results.append)
        assert await _wait_until(
            pilot, lambda: bool(modal.query("#bpm-preset-list Button"))
        )

        await pilot.click(f"#bpm-preset-btn-{preset_id}")
        await pilot.pause()
        modal.query_one("#bpm-name-input", Input).value = "New name"

        await pilot.click("#bpm-save")
        assert await _wait_until(
            pilot,
            lambda: (db.get_briefing_preset(preset_id) or {}).get("name")
            == "New name",
        )

        await pilot.click("#bpm-close")
        assert await _wait_until(pilot, lambda: results != [])

    assert results == [True]


# --- Character / voice Selects ----------------------------------------------


@pytest.mark.asyncio
async def test_character_select_offers_the_passed_options_and_stores_the_card_id(
    tmp_path,
):
    db = _db(tmp_path)
    modal = BriefingPresetModal(
        db, character_options=[("Ada", 3), ("Grace", 9)], voice_options=[]
    )

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal)
        assert await _wait_until(pilot, lambda: modal.is_mounted)

        select = modal.query_one("#bpm-speaker-character-0", Select)
        assert select.disabled is False

        # A value NOT among the passed options is illegal -- proving the
        # Select's legal values are exactly what the screen supplied, not
        # an arbitrary integer.
        with pytest.raises(Exception):
            select.value = 12345

        select.value = 9
        modal.query_one("#bpm-speaker-name-0", Input).value = "Narrator"
        modal.query_one("#bpm-name-input", Input).value = "With character"

        await pilot.click("#bpm-save")
        assert await _wait_until(pilot, lambda: bool(db.list_briefing_presets()))

    roster = load_roster(db.list_briefing_presets()[0]["roster_json"])
    assert roster[0]["character_card_id"] == 9


@pytest.mark.asyncio
async def test_voice_select_stores_voice_profile_id_inert(tmp_path):
    db = _db(tmp_path)
    modal = BriefingPresetModal(
        db, character_options=[], voice_options=[("Warm narrator", "voice-abc")]
    )

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal)
        assert await _wait_until(pilot, lambda: modal.is_mounted)

        select = modal.query_one("#bpm-speaker-voice-0", Select)
        assert select.disabled is False
        select.value = "voice-abc"
        modal.query_one("#bpm-speaker-name-0", Input).value = "Narrator"
        modal.query_one("#bpm-name-input", Input).value = "Voice test"

        await pilot.click("#bpm-save")
        assert await _wait_until(pilot, lambda: bool(db.list_briefing_presets()))

    # Stored, exactly as chosen -- 2a records `voice_profile_id`; nothing
    # in this phase reads it back out to synthesize audio. Phase 2b is what
    # consumes it (see `briefing_preset_modal.py`'s module docstring).
    roster = load_roster(db.list_briefing_presets()[0]["roster_json"])
    assert roster[0]["voice_profile_id"] == "voice-abc"


@pytest.mark.asyncio
async def test_character_and_voice_selects_disable_with_a_tooltip_when_unsupplied(
    tmp_path,
):
    db = _db(tmp_path)
    modal = BriefingPresetModal(db, character_options=[], voice_options=[])

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal)
        assert await _wait_until(pilot, lambda: modal.is_mounted)

        character_select = modal.query_one("#bpm-speaker-character-0", Select)
        voice_select = modal.query_one("#bpm-speaker-voice-0", Select)
        assert character_select.disabled is True
        assert voice_select.disabled is True
        assert character_select.tooltip
        assert voice_select.tooltip


# --- Dynamic speaker rows ----------------------------------------------------


@pytest.mark.asyncio
async def test_add_speaker_appends_a_row_and_remove_deletes_it(tmp_path):
    db = _db(tmp_path)
    modal = BriefingPresetModal(db, character_options=[], voice_options=[])

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal)
        assert await _wait_until(pilot, lambda: modal.is_mounted)
        assert len(modal.query(".bpm-speaker-row")) == 1

        await pilot.click("#bpm-add-speaker")
        await pilot.pause()
        assert len(modal.query(".bpm-speaker-row")) == 2

        await pilot.click("#bpm-speaker-remove-0")
        await pilot.pause()
        assert len(modal.query(".bpm-speaker-row")) == 1


@pytest.mark.asyncio
async def test_remove_speaker_enforces_a_one_row_minimum(tmp_path):
    db = _db(tmp_path)
    modal = BriefingPresetModal(db, character_options=[], voice_options=[])

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal)
        assert await _wait_until(pilot, lambda: modal.is_mounted)
        assert modal.query_one("#bpm-speaker-remove-0", Button).disabled is True

        await pilot.click("#bpm-speaker-remove-0")
        await pilot.pause()
        assert len(modal.query(".bpm-speaker-row")) == 1


@pytest.mark.asyncio
async def test_close_without_any_change_dismisses_false(tmp_path):
    db = _db(tmp_path)
    modal = BriefingPresetModal(db, character_options=[], voice_options=[])
    results: list[bool] = []

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal, results.append)
        assert await _wait_until(pilot, lambda: modal.is_mounted)
        await pilot.click("#bpm-close")
        assert await _wait_until(pilot, lambda: results != [])

    assert results == [False]


# --- Review round 1: write-in-flight race guards ---------------------------


@pytest.mark.asyncio
async def test_switching_the_selected_preset_mid_write_does_not_let_the_next_save_overwrite_it(
    tmp_path,
):
    """Review round 1, Important finding, driven exactly as reported.

    Deterministic control over exactly when the insert's `asyncio.to_thread`
    call resolves comes from a `threading.Event` the fake write blocks on
    (the phase-1 controllable-seam pattern, not a sleep/poll race) --
    `threading.Event`, not `asyncio.Event`, because the fake runs INSIDE
    the `to_thread` executor thread, and an `asyncio.Event` cannot be
    waited on safely from a thread other than its owning loop's.

    Sequence, exactly as the review names it: start a brand-new preset
    ("Preset A") and press Save -- the insert blocks. While it is still in
    flight, attempt to switch to an existing preset ("Preset B") in the
    list: this must be REFUSED (the form must still show Preset A's own,
    unsaved draft) -- not silently succeed and repoint the editor at B
    while the about-to-be-created row is still in flight. Release the
    write; let it finish. THEN legitimately switch to Preset B and press
    Save again ("the user's next Save") -- this must update B with its own
    content; Preset A's row must survive with ITS OWN content, never
    silently overwritten by B's.
    """
    db = _db(tmp_path)
    b_id = db.insert_briefing_preset(
        "Preset B", roster_json=dump_roster(validate_roster([{"name": "Guest"}]))
    )
    modal = BriefingPresetModal(db, character_options=[], voice_options=[])

    release_write = threading.Event()
    real_insert = db.insert_briefing_preset

    def _blocking_insert(*args, **kwargs):
        assert release_write.wait(timeout=5), "test setup: write never released"
        return real_insert(*args, **kwargs)

    db.insert_briefing_preset = _blocking_insert

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal)
        assert await _wait_until(
            pilot, lambda: bool(modal.query("#bpm-preset-list Button"))
        )

        # Start a brand-new preset and Save it -- the insert blocks.
        modal.query_one("#bpm-name-input", Input).value = "Preset A"
        modal.query_one("#bpm-speaker-name-0", Input).value = "Host A"
        await pilot.click("#bpm-save")
        assert await _wait_until(pilot, lambda: modal._write_in_flight)

        # While that write is in flight, attempt to switch to Preset B.
        await pilot.click(f"#bpm-preset-btn-{b_id}")
        await pilot.pause()

        # Refused: the form must still show Preset A's own, unsaved draft --
        # not Preset B's, sitting "over" whatever id the insert is about to
        # claim (the exact corruption shape the review names).
        assert modal.query_one("#bpm-name-input", Input).value == "Preset A"
        assert modal._editing_id is None

        # Let the insert finish.
        release_write.set()
        assert await _wait_until(pilot, lambda: not modal._write_in_flight)
        assert await _wait_until(pilot, lambda: len(db.list_briefing_presets()) == 2)

        # NOW legitimately switch to Preset B (the write is no longer in
        # flight, so this is allowed) and Save again -- "the user's next
        # Save".
        await pilot.click(f"#bpm-preset-btn-{b_id}")
        await pilot.pause()
        assert modal.query_one("#bpm-name-input", Input).value == "Preset B"
        await pilot.click("#bpm-save")
        assert await _wait_until(pilot, lambda: not modal._write_in_flight)

    rows = {row["name"]: row for row in db.list_briefing_presets()}
    assert set(rows) == {"Preset A", "Preset B"}, (
        "Preset A must survive as its own row -- not get clobbered by "
        "Preset B's content, and not vanish"
    )
    assert rows["Preset A"]["id"] != b_id
    assert load_roster(rows["Preset A"]["roster_json"])[0]["name"] == "Host A"
    assert rows["Preset B"]["id"] == b_id
    assert load_roster(rows["Preset B"]["roster_json"])[0]["name"] == "Guest"


# --- Geometry (real generated CSS, per the three-way-vacuity lesson) -------


@pytest.mark.asyncio
async def test_modal_dialog_and_columns_fit_on_screen_with_production_css(tmp_path):
    db = _db(tmp_path)
    modal = BriefingPresetModal(db, character_options=[], voice_options=[])

    app = _ModalHost()
    async with app.run_test(size=(160, 42)) as pilot:
        app.push_screen(modal)
        assert await _wait_until(pilot, lambda: modal.is_mounted)

        dialog = modal.query_one("#bpm-dialog")
        # Exact width, not just "> 0": a bare `ModalScreen` with no sizing
        # rule at all still reports a nonzero region (its `Vertical` child
        # falls back to filling the whole screen, 160x42 here) -- a loose
        # `> 0`/"fits in viewport" assertion is true in that failure mode
        # too (the three-way-vacuity trap this brief names). Pinning the
        # exact `width: 96` from `_watchlists.tcss` and requiring the
        # dialog be narrower AND shorter than the full screen is what
        # actually distinguishes "sized and centred" from "unstyled and
        # full-bleed".
        assert dialog.region.width == 96
        assert dialog.region.height > 0
        assert dialog.region.width < app.size.width
        assert dialog.region.height < app.size.height
        assert dialog.region.x > 0
        assert dialog.region.y >= 0
        assert dialog.region.right <= app.size.width
        assert dialog.region.bottom <= app.size.height

        for widget_id in (
            "#bpm-list-column",
            "#bpm-editor-column",
            "#bpm-actions",
            "#bpm-save",
            "#bpm-name-input",
        ):
            widget = modal.query_one(widget_id)
            assert widget.region.width > 0, widget_id
            assert widget.region.height > 0, widget_id
            assert app.screen.region.contains_region(widget.region), widget_id


# --- Screen wiring: option-building loaders ---------------------------------


def _screen_stub(app_instance: SimpleNamespace) -> WatchlistsCollectionsScreen:
    """A `WatchlistsCollectionsScreen` constructed but never mounted.

    Legal for methods that touch only `self.app_instance` -- Screen/Widget
    construction has no App requirement; only `.app`/`is_mounted`-style
    access does. `_load_character_options`/`_load_voice_options` are exactly
    such methods.
    """
    return WatchlistsCollectionsScreen(app_instance)


def _bare_app_instance(**overrides: object) -> SimpleNamespace:
    base = dict(
        chachanotes_db=None,
        _tts_profile_service=None,
        watchlist_scope_service=None,
        server_watchlists_service=None,
        client_notifications_db=None,
        watchlist_bundle_service=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_load_character_options_degrades_to_empty_when_chachanotes_db_unbound():
    screen = _screen_stub(_bare_app_instance())
    assert await screen._load_character_options() == []


@pytest.mark.asyncio
async def test_load_character_options_returns_name_id_pairs_when_bound():
    fake_db = Mock()
    fake_db.list_character_cards = Mock(
        return_value=[{"id": 3, "name": "Ada"}, {"id": 9, "name": "Grace"}]
    )
    screen = _screen_stub(_bare_app_instance(chachanotes_db=fake_db))
    assert await screen._load_character_options() == [("Ada", 3), ("Grace", 9)]


@pytest.mark.asyncio
async def test_load_character_options_degrades_to_empty_on_lookup_failure():
    fake_db = Mock()
    fake_db.list_character_cards = Mock(side_effect=RuntimeError("boom"))
    screen = _screen_stub(_bare_app_instance(chachanotes_db=fake_db))
    assert await screen._load_character_options() == []


@pytest.mark.asyncio
async def test_load_voice_options_degrades_to_empty_when_service_unbound():
    screen = _screen_stub(_bare_app_instance())
    assert await screen._load_voice_options() == []


@pytest.mark.asyncio
async def test_load_voice_options_returns_name_id_pairs_when_bound():
    profile_a = SimpleNamespace(display_name="Warm narrator", profile_id="voice-1")
    profile_b = SimpleNamespace(display_name="Crisp anchor", profile_id="voice-2")
    fake_service = Mock()
    fake_service.list_profiles = AsyncMock(
        return_value=SimpleNamespace(profiles=[profile_a, profile_b])
    )
    screen = _screen_stub(_bare_app_instance(_tts_profile_service=fake_service))
    assert await screen._load_voice_options() == [
        ("Warm narrator", "voice-1"),
        ("Crisp anchor", "voice-2"),
    ]


@pytest.mark.asyncio
async def test_load_voice_options_degrades_to_empty_on_lookup_failure():
    fake_service = Mock()
    fake_service.list_profiles = AsyncMock(side_effect=RuntimeError("boom"))
    screen = _screen_stub(_bare_app_instance(_tts_profile_service=fake_service))
    assert await screen._load_voice_options() == []


# --- Screen wiring: the mount/dismiss round trip ----------------------------


@pytest.mark.asyncio
async def test_open_briefing_preset_manager_notifies_when_briefings_db_unavailable():
    app = _build_test_app()
    app.watchlist_bundle_service = None
    app.notify = Mock()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = host.screen_stack[-1]
        await screen._open_briefing_preset_manager()
        await pilot.pause()

    assert app.notify.called
    _args, kwargs = app.notify.call_args
    assert kwargs.get("severity") == "error"


@pytest.mark.asyncio
async def test_open_briefing_preset_manager_passes_built_options_to_the_modal():
    app = _build_test_app()
    fake_characters = Mock()
    fake_characters.list_character_cards = Mock(
        return_value=[{"id": 5, "name": "Ada"}]
    )
    app.chachanotes_db = fake_characters
    fake_voice_service = Mock()
    fake_voice_service.list_profiles = AsyncMock(
        return_value=SimpleNamespace(
            profiles=[SimpleNamespace(display_name="Warm", profile_id="v1")]
        )
    )
    app._tts_profile_service = fake_voice_service

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = host.screen_stack[-1]
        screen.run_worker(
            screen._open_briefing_preset_manager(),
            exclusive=True,
            name="open-briefing-presets",
        )
        modal = await _wait_for_screen_type(host, BriefingPresetModal, pilot)
        assert modal.character_options == [("Ada", 5)]
        assert modal.voice_options == [("Warm", "v1")]

        await pilot.click("#bpm-close")
        assert await _wait_until(
            pilot, lambda: not isinstance(host.screen, BriefingPresetModal)
        )


@pytest.mark.asyncio
async def test_open_briefing_preset_manager_reloads_presets_only_when_something_changed():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service.db

        screen.run_worker(
            screen._open_briefing_preset_manager(),
            exclusive=True,
            name="open-briefing-presets",
        )
        modal = await _wait_for_screen_type(host, BriefingPresetModal, pilot)

        modal.query_one("#bpm-name-input", Input).value = "New preset"
        modal.query_one("#bpm-speaker-name-0", Input).value = "Host"
        await pilot.click("#bpm-save")
        assert await _wait_until(pilot, lambda: bool(db.list_briefing_presets()))

        await pilot.click("#bpm-close")
        assert await _wait_until(pilot, lambda: bool(screen._loaded_briefing_presets))

    assert [row["name"] for row in screen._loaded_briefing_presets] == ["New preset"]


@pytest.mark.asyncio
async def test_open_briefing_preset_manager_leaves_preset_list_untouched_when_cancelled():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = host.screen_stack[-1]
        sentinel = [{"id": 1, "name": "sentinel"}]
        screen._loaded_briefing_presets = list(sentinel)

        screen.run_worker(
            screen._open_briefing_preset_manager(),
            exclusive=True,
            name="open-briefing-presets",
        )
        await _wait_for_screen_type(host, BriefingPresetModal, pilot)

        await pilot.click("#bpm-close")
        assert await _wait_until(
            pilot, lambda: not isinstance(host.screen, BriefingPresetModal)
        )

    assert screen._loaded_briefing_presets == sentinel
