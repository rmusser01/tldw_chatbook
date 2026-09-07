"""Kept briefings modal (task-1780, Task 5).

`KeptBriefingsModal` is the one surface that lists, inspects, hard-deletes,
and casts from kept briefings (`briefing_keep.keep_briefing`'s own targets,
Task 2). Mirrors `test_watchlists_briefing_presets_ui.py`'s own testing
shape: every DB-touching test uses a real, file-backed `CharactersRAGDB`
(never `:memory:` -- the modal's reads/writes go through `asyncio.
to_thread`, and `CharactersRAGDB._get_thread_connection` is thread-local,
so an in-memory connection opened on the test's own thread would be a
brand-new, unmigrated, empty database on the executor thread) and, where a
preset is involved, a real, file-backed `SubscriptionsDB` for the identical
reason.

Geometry assertions run against the REAL generated stylesheet
(`ProductionCSSDestinationHarness.CSS_PATH`), not a widget-embedded
`DEFAULT_CSS` -- the modal's styles live in `css/features/_watchlists.tcss`
and are only real once the bundle is regenerated from them (the
three-way-vacuity lesson `test_watchlists_briefing_presets_ui.py`'s own
geometry test names).
"""

from __future__ import annotations

import json
import threading
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from rich.console import Console
from textual.app import App, ComposeResult
from textual.widgets import Button, Select, Static

from Tests.UI.test_destination_visual_parity_correction import (
    ProductionCSSDestinationHarness,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import briefing_cast
from tldw_chatbook.Subscriptions.briefing_cast import (
    APP_DEFAULT_PRESET_NAME,
    dump_roster,
    validate_roster,
)
from tldw_chatbook.UI.Watchlists_Modules import kept_briefings_modal as kbm_module
from tldw_chatbook.UI.Watchlists_Modules.kept_briefings_modal import KeptBriefingsModal
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

# Marked so CI actually runs this file: the unit job selects `-m unit` and
# the UI job runs `Tests/UI` plus `Tests -m ui --ignore=Tests/UI`, so an
# unmarked test in `Tests/Watchlists` is collected by nothing (matches
# `test_watchlists_briefing_presets_ui.py`'s own header note).
pytestmark = pytest.mark.ui


def _subs_db(tmp_path: Path) -> SubscriptionsDB:
    """A real, file-backed `SubscriptionsDB` -- not `:memory:`. See the
    module docstring."""
    return SubscriptionsDB(tmp_path / "subs.db", "test")


def _chacha_db(tmp_path: Path) -> CharactersRAGDB:
    """A real, file-backed `CharactersRAGDB` -- not `:memory:`. See the
    module docstring."""
    return CharactersRAGDB(tmp_path / "chacha.sqlite", client_id="kept-modal-test")


def _kept_briefing(
    chacha_db: CharactersRAGDB,
    *,
    source_briefing_id: int,
    body: str = "## Kept\n\nSomething worth keeping happened.\n",
    watchlist_name: str = "Security Watch",
    origin: str = "manual",
) -> int:
    """A minimal `kept_briefings` row. `source_briefing_id` is `UNIQUE`, so
    a test that keeps more than one briefing must pass distinct values.
    """
    return chacha_db.create_kept_briefing(
        source_briefing_id=source_briefing_id,
        watchlist_name=watchlist_name,
        body_markdown=body,
        origin=origin,
    )


def _preset(
    subs_db: SubscriptionsDB,
    *,
    roster: list[dict] | None = None,
    name: str = "Solo",
) -> int:
    return subs_db.insert_briefing_preset(
        name,
        roster_json=dump_roster(validate_roster(roster or [{"name": "Narrator"}])),
    )


def _use_fake_kept_cast_chat(monkeypatch, chat) -> None:
    """Fake the chat call at the service boundary, nothing else.

    `generate_script_from_text` binds its `chat` default at definition
    time, so patching `briefing_cast.chat_api_call` would not reach it.
    Wrapping the MODAL's own imported reference instead (mirrors
    `test_watchlists_artifacts_pane._use_fake_cast_chat`'s identical
    technique for the screen's own live-cast path) keeps the whole
    service real -- roster validation, prompt building, strict turn
    parsing -- with only the provider call replaced.
    """

    async def _generate(chacha_db, kept_id, **kwargs):
        return await briefing_cast.generate_script_from_text(
            chacha_db, kept_id, chat=chat, **kwargs
        )

    monkeypatch.setattr(kbm_module, "generate_script_from_text", _generate)


class _FakeChat:
    """The one faked seam, mirroring `test_briefing_cast._FakeChat`."""

    def __init__(self, *, reply: object = None):
        self.reply = (
            json.dumps([{"speaker": "Narrator", "text": "Hi."}])
            if reply is None
            else reply
        )
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


async def _wait_until(pilot, predicate, *, ticks: int = 80) -> bool:
    for _ in range(ticks):
        await pilot.pause()
        if predicate():
            return True
    return False


def _render_to_console(renderable, *, width: int = 100) -> tuple[str, str]:
    """Render through a real Console and return (plain, ansi).

    Mirrors `test_watchlists_artifacts_pane._render_to_console` exactly --
    duplicated rather than imported, since it is a private helper of a
    sibling test module (matches this stream's own precedent of each
    module owning its own small test vocabulary).
    """
    console = Console(
        width=width,
        record=True,
        color_system="standard",
        force_terminal=True,
        file=StringIO(),
    )
    console.print(renderable)
    return console.export_text(clear=False), console.export_text(styles=True)


class _ModalHost(App[None]):
    """A bare host mounting the REAL generated bundle, not a widget's own
    `DEFAULT_CSS` -- see the module docstring's three-way-vacuity note.
    """

    CSS_PATH = ProductionCSSDestinationHarness.CSS_PATH

    def compose(self) -> ComposeResult:
        yield Static("host")


# --- Listing, selection, rendering -----------------------------------------


@pytest.mark.asyncio
async def test_modal_lists_kept_briefings_newest_kept_first(tmp_path):
    chacha_db = _chacha_db(tmp_path)
    try:
        first_id = _kept_briefing(
            chacha_db, source_briefing_id=1, watchlist_name="Alpha Watch"
        )
        second_id = _kept_briefing(
            chacha_db, source_briefing_id=2, watchlist_name="Zeta Watch"
        )
        modal = KeptBriefingsModal(chacha_db)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            button_ids = [
                str(button.id) for button in modal.query("#kbm-kept-list Button")
            ]

        # `kept_at DESC, id DESC` -- the later insert (`second_id`) sorts
        # first, even though both share the same `CURRENT_TIMESTAMP`
        # second-resolution `kept_at`.
        assert button_ids == [
            f"kbm-kept-btn-{second_id}",
            f"kbm-kept-btn-{first_id}",
        ]
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_selecting_a_kept_briefing_renders_its_body_and_header(tmp_path):
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(
            chacha_db,
            source_briefing_id=1,
            body="## Digest\n\nAcme shipped a thing.\n",
            watchlist_name="Security Watch",
            origin="scheduled",
        )
        modal = KeptBriefingsModal(chacha_db)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(pilot, lambda: modal.is_mounted)

            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            await pilot.pause()

            plain, _ansi = _render_to_console(
                modal.query_one("#kbm-detail", Static).renderable, width=120
            )

        assert "Security Watch" in plain
        assert "scheduled" in plain
        assert "Acme shipped a thing." in plain
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_a_hostile_watchlist_name_paints_literally_never_as_markup(tmp_path):
    """Mutation target (ii): route a hostile string through markup
    parsing and this REDs. Watchlist names (and bodies) are hostile,
    user-authored text -- a markup-shaped name must paint as literal
    characters in BOTH the kept-list button and the detail header, never
    be interpreted or silently swallow an unclosed tag.
    """
    hostile = "[bold red]x[/]"
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(
            chacha_db, source_briefing_id=1, watchlist_name=hostile
        )
        modal = KeptBriefingsModal(chacha_db)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )

            # `Button` converts whatever renderable it is given into its
            # own `textual.content.Content` -- constructed here from a
            # `rich.text.Text` (`_kept_list_label`), which preserves the
            # source string verbatim rather than re-parsing it as Textual
            # markup (the fate a bare `str` label would suffer instead).
            list_button = modal.query_one(f"#kbm-kept-btn-{kept_id}", Button)
            assert hostile in list_button.label.plain, (
                "the name must paint exactly as it was typed, not be "
                "parsed or silently swallowed"
            )

            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            await pilot.pause()

            detail_plain, detail_ansi = _render_to_console(
                modal.query_one("#kbm-detail", Static).renderable, width=120
            )

        assert hostile in detail_plain
        assert "\x1b[1;31m" not in detail_ansi
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_no_kept_briefings_shows_the_empty_placeholder(tmp_path):
    chacha_db = _chacha_db(tmp_path)
    try:
        modal = KeptBriefingsModal(chacha_db)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(pilot, lambda: modal.is_mounted)
            assert modal.query_one("#kbm-kept-list-empty", Static)

            plain, _ansi = _render_to_console(
                modal.query_one("#kbm-detail", Static).renderable
            )
        assert "No kept briefings yet" in plain
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_existing_kept_scripts_render_under_the_detail(tmp_path):
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, source_briefing_id=1)
        chacha_db.create_kept_script(
            kept_id,
            source_script_id=None,
            preset_name="Duo",
            roster_snapshot_json='[{"name": "Host"}]',
            turns_json='[{"speaker": "Host", "text": "Welcome back."}]',
        )
        modal = KeptBriefingsModal(chacha_db)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(pilot, lambda: modal.is_mounted)
            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            assert await _wait_until(pilot, lambda: bool(modal.query(".kbm-script")))

            script_widgets = modal.query(".kbm-script")
            plain, _ansi = _render_to_console(
                script_widgets[0].renderable, width=120
            )
        assert "Duo" in plain
        assert "Welcome back." in plain
    finally:
        chacha_db.close_connection()


# --- Delete: confirms, hard-deletes, cascades -------------------------------


@pytest.mark.asyncio
async def test_delete_asks_confirmation_and_hard_deletes_cascading_scripts(tmp_path):
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, source_briefing_id=1)
        chacha_db.create_kept_script(
            kept_id,
            source_script_id=None,
            preset_name="Duo",
            roster_snapshot_json='[{"name": "Host"}]',
            turns_json='[{"speaker": "Host", "text": "Hi."}]',
        )
        modal = KeptBriefingsModal(chacha_db)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )

            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            await pilot.pause()
            assert modal.query_one("#kbm-delete-button", Button).disabled is False

            await pilot.click("#kbm-delete-button")
            assert await _wait_until(
                pilot, lambda: isinstance(app.screen, ConfirmationDialog)
            )
            await pilot.click("#confirm-button")

            assert await _wait_until(
                pilot, lambda: chacha_db.get_kept_briefing(kept_id) is None
            )

        assert chacha_db.get_kept_briefing(kept_id) is None
        assert chacha_db.list_kept_scripts(kept_id) == [], (
            "kept scripts must cascade with their kept briefing"
        )
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_delete_cancelled_leaves_the_kept_briefing_in_place(tmp_path):
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, source_briefing_id=1)
        modal = KeptBriefingsModal(chacha_db)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )

            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            await pilot.pause()
            await pilot.click("#kbm-delete-button")
            assert await _wait_until(
                pilot, lambda: isinstance(app.screen, ConfirmationDialog)
            )
            await pilot.click("#cancel-button")
            assert await _wait_until(pilot, lambda: modal.is_mounted and modal.is_current)

        assert chacha_db.get_kept_briefing(kept_id) is not None
    finally:
        chacha_db.close_connection()


# --- Delete: robustness (task-1780 whole-branch review) ---------------------


@pytest.mark.asyncio
async def test_a_failed_delete_toasts_and_does_not_exit_the_app(monkeypatch, tmp_path):
    """FIX 1 (Important): `_run_delete` used to be a bare `try/finally`
    with no `except` at all -- any exception surfacing from either of
    `_handle_delete`'s two `await`s (the confirmation dialog, or the hard
    delete itself: a `SQLITE_BUSY`/`CharactersRAGDBError` mid-delete is no
    longer theoretical now that auto-keep writes ChaChaNotes concurrently
    from the scheduler) took the WHOLE APPLICATION down via a Textual
    worker's default `exit_on_error=True`.

    Mutation target: remove the `except Exception` branch (or the guard's
    `finally` re-arm) and this REDs -- either the app stops running mid-test
    or `modal._delete_in_flight` never clears.
    """
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, source_briefing_id=1)
        modal = KeptBriefingsModal(chacha_db)

        def _boom(_kept_id):
            raise RuntimeError("simulated SQLITE_BUSY mid-delete")

        monkeypatch.setattr(chacha_db, "delete_kept_briefing", _boom)

        app = _ModalHost()
        app.notify = Mock()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            await pilot.pause()

            await pilot.click("#kbm-delete-button")
            assert await _wait_until(
                pilot, lambda: isinstance(app.screen, ConfirmationDialog)
            )
            await pilot.click("#confirm-button")

            assert await _wait_until(pilot, lambda: not modal._delete_in_flight)

            assert app.is_running, "a delete failure must not exit the application"
            assert modal.is_mounted and modal.is_current, (
                "the modal must survive and stay re-armed"
            )
            app.notify.assert_called_once()
            _args, kwargs = app.notify.call_args
            assert kwargs.get("severity") == "error"
            assert kwargs.get("markup") is False

        # The failed delete never reached ChaChaNotes: the row is intact,
        # unrefreshed but consistent (never claimed to be anything else).
        assert chacha_db.get_kept_briefing(kept_id) is not None
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_a_second_delete_press_while_in_flight_is_refused_with_a_toast(
    monkeypatch, tmp_path
):
    """FIX 2 (Minor): a second Delete press while the first is still in
    flight used to be a silent no-op -- unlike the screen's own Keep ("A
    keep is already in progress. Nothing else was started."). A blocking
    real `delete_kept_briefing` (a `threading.Event`, not a sleep/poll
    race -- mirrors `test_a_second_cast_press_while_in_flight_is_refused_
    by_the_modals_own_guard`'s own controllable-seam pattern) holds the
    delete in flight deterministically through the one `await` window
    `_dispatch_cast`'s own comment names: after the confirmation dialog
    resolves, this modal is again the top screen and its own buttons are
    clickable while the delete itself is still running.

    Mutation target: drop the toast call and this REDs.
    """
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, source_briefing_id=1)
        modal = KeptBriefingsModal(chacha_db)

        release = threading.Event()
        calls: list[int] = []
        real_delete = chacha_db.delete_kept_briefing

        def _blocking_delete(target_id):
            calls.append(target_id)
            assert release.wait(timeout=5), "test setup: delete never released"
            return real_delete(target_id)

        monkeypatch.setattr(chacha_db, "delete_kept_briefing", _blocking_delete)

        app = _ModalHost()
        app.notify = Mock()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            await pilot.pause()

            await pilot.click("#kbm-delete-button")
            assert await _wait_until(
                pilot, lambda: isinstance(app.screen, ConfirmationDialog)
            )
            await pilot.click("#confirm-button")
            assert await _wait_until(pilot, lambda: bool(calls))

            # A second press while the first delete is still blocked
            # inside `to_thread` -- the modal is the top screen again.
            app.notify.reset_mock()
            await pilot.click("#kbm-delete-button")
            await pilot.pause()
            assert len(calls) == 1, (
                "the second press must not have dispatched a second delete"
            )
            app.notify.assert_called_once()
            _args, kwargs = app.notify.call_args
            assert "already in progress" in _args[0]
            assert kwargs.get("severity") == "warning"
            assert kwargs.get("markup") is False

            release.set()
            assert await _wait_until(pilot, lambda: not modal._delete_in_flight)
            assert await _wait_until(
                pilot, lambda: chacha_db.get_kept_briefing(kept_id) is None
            )

        assert chacha_db.get_kept_briefing(kept_id) is None
    finally:
        chacha_db.close_connection()


# --- Cast from kept ----------------------------------------------------------


@pytest.mark.asyncio
async def test_cast_with_a_real_preset_writes_a_new_kept_script_and_it_appears(
    monkeypatch, tmp_path
):
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(
            chacha_db, source_briefing_id=1, body="## Kept\n\nAcme shipped a thing."
        )
        preset_id = _preset(subs_db, name="Duo", roster=[{"name": "Host"}])
        modal = KeptBriefingsModal(chacha_db, subs_db=subs_db)

        # The fake reply must name a speaker the PRESET's own roster
        # actually has ("Host") -- `_FakeChat`'s own default ("Narrator")
        # is what `test_cast_with_app_default_...` below wants instead.
        chat = _FakeChat(reply=json.dumps([{"speaker": "Host", "text": "Hi."}]))
        _use_fake_kept_cast_chat(monkeypatch, chat)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            await pilot.pause()

            modal.query_one("#kbm-preset-select", Select).value = preset_id
            await pilot.pause()

            modal.query_one("#kbm-cast-button", Button).press()
            assert await _wait_until(
                pilot, lambda: bool(chacha_db.list_kept_scripts(kept_id))
            )
            assert await _wait_until(pilot, lambda: bool(modal.query(".kbm-script")))

        scripts = chacha_db.list_kept_scripts(kept_id)
        assert len(scripts) == 1
        assert scripts[0]["preset_name"] == "Duo"
        assert scripts[0]["source_script_id"] is None
        assert len(chat.calls) == 1
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_cast_with_app_default_uses_the_single_narrator_roster(
    monkeypatch, tmp_path
):
    """`preset_id=None`, the "App default (single narrator)" option: the
    kept script's `preset_name` is the literal `APP_DEFAULT_PRESET_NAME` --
    the carried decision this task's brief states verbatim.
    """
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, source_briefing_id=1)
        modal = KeptBriefingsModal(chacha_db, subs_db=None)

        chat = _FakeChat()
        _use_fake_kept_cast_chat(monkeypatch, chat)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            await pilot.pause()

            # Default value is already `None` ("App default…") -- no need
            # to touch the Select at all.
            modal.query_one("#kbm-cast-button", Button).press()
            assert await _wait_until(
                pilot, lambda: bool(chacha_db.list_kept_scripts(kept_id))
            )

        scripts = chacha_db.list_kept_scripts(kept_id)
        assert scripts[0]["preset_name"] == APP_DEFAULT_PRESET_NAME
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_generation_in_flight_error_gets_its_specific_toast(monkeypatch, tmp_path):
    """The phase-4 lesson: `GenerationInFlightError` is caught and shown
    with its OWN specific message, never folded into the generic
    "could not cast a script" fallback.
    """
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, source_briefing_id=1)
        modal = KeptBriefingsModal(chacha_db, subs_db=None)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            await pilot.pause()

            with briefing_cast._claim_kept_cast(kept_id):
                modal.query_one("#kbm-cast-button", Button).press()
                assert await _wait_until(
                    pilot,
                    lambda: bool(
                        str(modal.query_one("#kbm-error", Static).renderable).strip()
                    ),
                )
                error_text = str(modal.query_one("#kbm-error", Static).renderable)

        assert f"kept briefing {kept_id}" in error_text
        assert "already being cast" in error_text
        assert chacha_db.list_kept_scripts(kept_id) == []
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_a_second_cast_press_while_in_flight_is_refused_by_the_modals_own_guard(
    monkeypatch, tmp_path
):
    """Mutation target (iii): drop the modal's own `_cast_in_flight` guard
    and this REDs. A blocking fake chat (a `threading.Event`, not a
    sleep/poll race -- the same controllable-seam pattern `test_
    watchlists_briefing_presets_ui.py`'s own write-race test uses) holds
    the FIRST cast in flight deterministically while a second press is
    attempted; only ONE chat call may have happened by the time the second
    press returns.

    Also pins FIX 2 (task-1780 whole-branch review): the second press must
    now toast a refusal -- it used to be a silent no-op.
    """
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, source_briefing_id=1)
        modal = KeptBriefingsModal(chacha_db, subs_db=None)

        release = threading.Event()
        calls: list[dict] = []

        def _blocking_chat(**kwargs):
            calls.append(kwargs)
            assert release.wait(timeout=5), "test setup: cast never released"
            return json.dumps([{"speaker": "Narrator", "text": "Hi."}])

        _use_fake_kept_cast_chat(monkeypatch, _blocking_chat)

        app = _ModalHost()
        app.notify = Mock()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            await pilot.pause()

            modal.query_one("#kbm-cast-button", Button).press()
            assert await _wait_until(pilot, lambda: modal._cast_in_flight)

            # A second press while the first is still in flight.
            modal.query_one("#kbm-cast-button", Button).press()
            await pilot.pause()
            assert len(calls) == 1, (
                "the second press must not have dispatched a second cast"
            )
            app.notify.assert_called_once()
            _args, kwargs = app.notify.call_args
            assert "already in progress" in _args[0]
            assert kwargs.get("severity") == "warning"
            assert kwargs.get("markup") is False

            release.set()
            assert await _wait_until(pilot, lambda: not modal._cast_in_flight)
            assert await _wait_until(
                pilot, lambda: bool(chacha_db.list_kept_scripts(kept_id))
            )

        assert len(chacha_db.list_kept_scripts(kept_id)) == 1
    finally:
        chacha_db.close_connection()


# --- Cast/delete mutual exclusion (task-1780 whole-branch review, FIX 3) ----


@pytest.mark.asyncio
async def test_cast_is_refused_while_a_delete_is_in_flight(monkeypatch, tmp_path):
    """Pins `_dispatch_cast`'s own mutual-exclusion guard: a cast that
    outlived a delete would try to `create_kept_script` against a
    `kept_briefing_id` the foreign key no longer has. A blocking real
    `delete_kept_briefing` holds `_delete_in_flight` through the one
    `await` window after the confirmation dialog resolves (the identical
    window `test_a_second_delete_press_...` exercises for a same-action
    double-press); Cast is pressed in that same window instead.

    Mutation target: drop the `_delete_in_flight` check from `_dispatch_
    cast` and this REDs -- either a cast actually runs (the fake chat gets
    called) or no refusal toast appears.
    """
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, source_briefing_id=1)
        modal = KeptBriefingsModal(chacha_db, subs_db=None)

        release = threading.Event()
        delete_calls: list[int] = []
        real_delete = chacha_db.delete_kept_briefing

        def _blocking_delete(target_id):
            delete_calls.append(target_id)
            assert release.wait(timeout=5), "test setup: delete never released"
            return real_delete(target_id)

        monkeypatch.setattr(chacha_db, "delete_kept_briefing", _blocking_delete)

        chat = _FakeChat()
        _use_fake_kept_cast_chat(monkeypatch, chat)

        app = _ModalHost()
        app.notify = Mock()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            await pilot.pause()

            await pilot.click("#kbm-delete-button")
            assert await _wait_until(
                pilot, lambda: isinstance(app.screen, ConfirmationDialog)
            )
            await pilot.click("#confirm-button")
            assert await _wait_until(pilot, lambda: bool(delete_calls))

            await pilot.click("#kbm-cast-button")
            await pilot.pause()
            assert not chat.calls, "a cast must not run while a delete is in flight"
            app.notify.assert_called_once()
            _args, kwargs = app.notify.call_args
            assert "delete is in progress" in _args[0]
            assert kwargs.get("severity") == "warning"
            assert kwargs.get("markup") is False

            release.set()
            assert await _wait_until(pilot, lambda: not modal._delete_in_flight)
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_delete_is_refused_while_a_cast_is_in_flight(monkeypatch, tmp_path):
    """Mirror of the test above, the other direction: `_dispatch_delete`'s
    own `_cast_in_flight` check. Mutation target: drop it and this REDs --
    a delete would either open its confirmation dialog or actually run.
    """
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, source_briefing_id=1)
        modal = KeptBriefingsModal(chacha_db, subs_db=None)

        release = threading.Event()
        cast_calls: list[dict] = []

        def _blocking_chat(**kwargs):
            cast_calls.append(kwargs)
            assert release.wait(timeout=5), "test setup: cast never released"
            return json.dumps([{"speaker": "Narrator", "text": "Hi."}])

        _use_fake_kept_cast_chat(monkeypatch, _blocking_chat)

        app = _ModalHost()
        app.notify = Mock()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            await pilot.pause()

            await pilot.click("#kbm-cast-button")
            assert await _wait_until(pilot, lambda: modal._cast_in_flight)

            await pilot.click("#kbm-delete-button")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmationDialog), (
                "a delete must not even open its confirmation while a cast "
                "is in flight"
            )
            app.notify.assert_called_once()
            _args, kwargs = app.notify.call_args
            assert "cast is in progress" in _args[0]
            assert kwargs.get("severity") == "warning"
            assert kwargs.get("markup") is False

            release.set()
            assert await _wait_until(pilot, lambda: not modal._cast_in_flight)

        assert chacha_db.get_kept_briefing(kept_id) is not None
    finally:
        chacha_db.close_connection()


# --- Listing bounds (task-1780 whole-branch review, FIX 4) ------------------


@pytest.mark.asyncio
async def test_kept_list_stays_quiet_exactly_at_its_display_cap(monkeypatch, tmp_path):
    chacha_db = _chacha_db(tmp_path)
    try:
        monkeypatch.setattr(kbm_module, "_KEPT_LIST_DISPLAY_CAP", 2)
        for i in range(2):
            _kept_briefing(
                chacha_db, source_briefing_id=i, watchlist_name=f"Watch {i}"
            )
        modal = KeptBriefingsModal(chacha_db)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            assert len(modal.query("#kbm-kept-list Button")) == 2
            assert not modal.query("#kbm-kept-list-overflow"), (
                "exactly at the cap must not claim there is more"
            )
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_kept_list_shows_an_overflow_line_past_its_display_cap(
    monkeypatch, tmp_path
):
    """Shrinks `_KEPT_LIST_DISPLAY_CAP` rather than seeding hundreds of
    real rows -- the established idiom for this class of test (mirrors
    `_all_briefing_scripts`'s own page-size-shrink test in `test_
    briefing_keep.py`). Mutation target: drop the overflow line and this
    REDs.
    """
    chacha_db = _chacha_db(tmp_path)
    try:
        monkeypatch.setattr(kbm_module, "_KEPT_LIST_DISPLAY_CAP", 2)
        for i in range(3):
            _kept_briefing(
                chacha_db, source_briefing_id=i, watchlist_name=f"Watch {i}"
            )
        modal = KeptBriefingsModal(chacha_db)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            assert len(modal.query("#kbm-kept-list Button")) == 2, (
                "the display cap itself must still hold"
            )
            assert modal.query("#kbm-kept-list-overflow"), (
                "more kept briefings than shown must say so honestly"
            )
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_kept_scripts_list_shows_an_overflow_line_past_its_display_cap(
    monkeypatch, tmp_path
):
    """Mirror of the kept-list overflow test, for one kept briefing's own
    kept-scripts list. Mutation target: drop the overflow line and this
    REDs.
    """
    chacha_db = _chacha_db(tmp_path)
    try:
        monkeypatch.setattr(kbm_module, "_KEPT_SCRIPTS_DISPLAY_CAP", 2)
        kept_id = _kept_briefing(chacha_db, source_briefing_id=1)
        for i in range(3):
            chacha_db.create_kept_script(
                kept_id,
                source_script_id=None,
                preset_name=f"Preset {i}",
                roster_snapshot_json='[{"name": "Host"}]',
                turns_json='[{"speaker": "Host", "text": "Hi."}]',
            )
        modal = KeptBriefingsModal(chacha_db)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            assert await _wait_until(pilot, lambda: bool(modal.query(".kbm-script")))
            assert len(modal.query(".kbm-script")) == 2
            assert modal.query("#kbm-scripts-overflow")
    finally:
        chacha_db.close_connection()


# --- Geometry (real generated CSS, per the three-way-vacuity lesson) -------


@pytest.mark.asyncio
async def test_modal_dialog_and_columns_fit_on_screen_with_production_css(tmp_path):
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, source_briefing_id=1)
        chacha_db.create_kept_script(
            kept_id,
            source_script_id=None,
            preset_name="Duo",
            roster_snapshot_json='[{"name": "Host"}]',
            turns_json='[{"speaker": "Host", "text": "Hi."}]',
        )
        modal = KeptBriefingsModal(chacha_db)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            assert await _wait_until(pilot, lambda: bool(modal.query(".kbm-script")))

            dialog = modal.query_one("#kbm-dialog")
            # Exact width, not just "> 0" -- see the module docstring's own
            # three-way-vacuity note: a bare `ModalScreen` with no sizing
            # rule at all still reports a nonzero, in-viewport region (its
            # `Vertical` child falls back to filling the whole screen),
            # which a loose assertion would not distinguish from "sized and
            # centred".
            assert dialog.region.width == 104
            assert dialog.region.height > 0
            assert dialog.region.width < app.size.width
            assert dialog.region.height < app.size.height
            assert dialog.region.x > 0
            assert dialog.region.y >= 0
            assert dialog.region.right <= app.size.width
            assert dialog.region.bottom <= app.size.height

            for widget_id in (
                "#kbm-list-column",
                "#kbm-detail-column",
                "#kbm-actions",
                "#kbm-close",
                "#kbm-detail",
                "#kbm-cast-toolbar",
                "#kbm-preset-select",
                "#kbm-cast-button",
            ):
                widget = modal.query_one(widget_id)
                assert widget.region.width > 0, widget_id
                assert widget.region.height > 0, widget_id
                assert app.screen.region.contains_region(widget.region), widget_id
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_preset_select_carries_a_visible_preset_label(tmp_path):
    """UAT batch-5 review, m1: `#kbm-preset-select` was tooltip-only, with
    no persistent label -- the same "bare value, hover-only meaning"
    pattern task-2310 removed from the Artifacts pane's own, structurally
    identical `#artifacts-preset-select` (same `.destination-filter-strip`
    row shape: a Select next to an action button). Geometry checked
    against the REAL generated stylesheet, per this file's own
    three-way-vacuity note -- a `> 0` region on an unstyled row would not
    distinguish "labeled" from "present but invisible."
    """
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, source_briefing_id=1)
        chacha_db.create_kept_script(
            kept_id,
            source_script_id=None,
            preset_name="Duo",
            roster_snapshot_json='[{"name": "Host"}]',
            turns_json='[{"speaker": "Host", "text": "Hi."}]',
        )
        modal = KeptBriefingsModal(chacha_db)

        app = _ModalHost()
        async with app.run_test(size=(160, 42)) as pilot:
            app.push_screen(modal)
            assert await _wait_until(
                pilot, lambda: bool(modal.query("#kbm-kept-list Button"))
            )
            await pilot.click(f"#kbm-kept-btn-{kept_id}")
            assert await _wait_until(pilot, lambda: bool(modal.query(".kbm-script")))

            labels = [
                widget
                for widget in modal.query("#kbm-cast-toolbar Static")
                if str(widget.renderable) == "Preset"
            ]
            assert len(labels) == 1, (
                "expected exactly one visible 'Preset' label in the cast "
                "toolbar"
            )
            label = labels[0]
            assert label.region.width > 0
            assert label.region.height > 0

            preset_select = modal.query_one("#kbm-preset-select")
            # The label must precede the Select it names, not merely exist
            # somewhere in the toolbar.
            assert label.region.x < preset_select.region.x
    finally:
        chacha_db.close_connection()
