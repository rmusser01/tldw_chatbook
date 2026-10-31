"""TASK-33620.15.17: keys typed before the Console composer's first focus.

When the Console is pushed, Textual focuses the first focusable widget, the
nav bar's Home button. The composer takes focus only when the activation's
own restore (`ChatScreen._restore_console_workbench_focus`) runs after the
first attach sync: live, 0.1-1.9 s after the Console's first frame. Keys typed
in that window went to Home, bubbled to `ChatScreen.on_key`, failed the
composer capture check and were dropped with no cue. A 29-character prompt
typed 0.5 s after the first frame was lost whole in 3 of 10 launches.

The mounted cases hold the activation restore back, which keeps the window
open for as long as the case needs, and type through the app's driver queue
(`pump_probe.key`), the path a terminal's keys take. A printable key typed in
the window now lands where that restore is about to put focus (the draft), in
order; a modal, the setup card, a collapsed composer and a focus the user
moved keep their input.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from textual.events import Key

from Tests.private_profile import private_profile_test
from Tests.UI import pump_probe
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_console_native_chat_flow import (
    _build_console_send_test_app,
    _configure_native_ready_console,
)
from tldw_chatbook.UI.Console_Modules import composer_run_controls as run_controls
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.model_catalog_consent import ModelCatalogConsentModal
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


class _HeldRestoreHarness(ConsolidatedCSSApp):
    """Push a ready Console whose activation focus restore never runs.

    The restore is replaced on the instance before the push, so every call
    the activation schedules is counted and held: focus stays on whatever
    Textual's automatic placement chose, the state a live launch is in until
    its restore lands.
    """

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance
        self.held_restores = 0

    async def on_mount(self) -> None:
        self.app_instance._ui_ready = True
        console = ChatScreen(self.app_instance)
        console._restore_console_workbench_focus = self._hold_restore
        await self.push_screen(console)

    def _hold_restore(self) -> None:
        self.held_restores += 1


def _ready_host() -> _HeldRestoreHarness:
    app = _build_console_send_test_app()
    _configure_native_ready_console(app)
    return _HeldRestoreHarness(app)


async def _wait_for(pilot, predicate, what, timeout: float = 5.0) -> None:
    """Poll ``predicate``; ``what`` may be a callable for late diagnostics."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    assert predicate(), what() if callable(what) else what


def _name(widget) -> str:
    if widget is None:
        return "nothing"
    return f"{type(widget).__name__}#{widget.id}"


async def _arrived(host, pilot):
    """Wait until the activation has run and scheduled its (held) restore."""
    console = host.screen_stack[-1]
    await _wait_for(
        pilot,
        lambda: (
            bool(console.query("#console-native-composer"))
            and console._console_attach_reconciled
            and host.held_restores > 0
        ),
        lambda: (
            "the Console never finished activating "
            f"(reconciled={console._console_attach_reconciled}, "
            f"held restores={host.held_restores})"
        ),
    )
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    # The window this task is about: focus is where Textual's automatic
    # placement put it -- outside every Console pane -- and not on the draft.
    assert console.focused is not None, "nothing is focused at arrival"
    assert console._console_workbench_focus_id_for_widget(console.focused) is None, (
        f"arrival focus {_name(console.focused)} is already inside the Console"
    )
    assert not console._console_setup_modal_blocking()
    return console, composer


def _type(host, text: str) -> None:
    """Queue ``text`` through the app's driver, as a terminal burst arrives."""
    for character in text:
        name = "space" if character == " " else character
        pump_probe.key(host, name, character)


@pytest.mark.asyncio
@private_profile_test
async def test_keys_typed_before_the_activation_restore_land_in_the_draft_in_order(
    request,
):
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _arrived(host, pilot)
        arrival = console.focused

        _type(host, "name one colour")
        await _wait_for(
            pilot,
            lambda: composer.draft_text() == "name one colour",
            lambda: (
                "keys typed before the composer's first focus were not typed "
                f"into it: draft {composer.draft_text()!r}, focus "
                f"{_name(console.focused)}"
            ),
        )
        assert console.focused is composer

        # The restore, once it lands, keeps the focus the keys moved.
        ChatScreen._restore_console_workbench_focus(console)
        await pilot.pause()
        assert console.focused is composer
        assert composer.draft_text() == "name one colour"

        # The window is over: Home, focused on purpose now, keeps its keys.
        console.set_focus(arrival)
        await pilot.pause()
        _type(host, "x")
        await pilot.pause(0.3)
        assert composer.draft_text() == "name one colour"
        assert console.focused is arrival


@pytest.mark.asyncio
@private_profile_test
async def test_a_focus_the_user_moves_before_the_restore_keeps_its_input(request):
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _arrived(host, pilot)
        arrival = console.focused
        elsewhere = next(
            widget
            for widget in console.focus_chain
            if widget is not arrival
            and console._console_workbench_focus_id_for_widget(widget) is None
        )

        console.set_focus(elsewhere)
        await pilot.pause()
        _type(host, "x")
        await pilot.pause(0.3)
        assert composer.draft_text() == ""
        assert console.focused is elsewhere

        # The attach reconcile replays on_screen_resume directly; that is not
        # an arrival, so the focus the user moved is not re-armed.
        console.on_screen_resume()
        await pilot.pause()
        _type(host, "z")
        await pilot.pause(0.3)
        assert composer.draft_text() == ""
        assert console.focused is elsewhere

        # Back on the arrival widget by choice: still the user's focus.
        console.set_focus(arrival)
        await pilot.pause()
        _type(host, "y")
        await pilot.pause(0.3)
        assert composer.draft_text() == ""
        assert console.focused is arrival


@pytest.mark.asyncio
@private_profile_test
async def test_a_modal_keeps_its_input_and_typing_after_it_closes_lands_in_the_draft(
    request,
):
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _arrived(host, pilot)
        arrival = console.focused

        # A guard, not a pin: the pushed modal receives the keys itself and
        # `ChatScreen.on_key` never sees them, with or without the claim.
        # The second half, after the modal closes, is the regression case.
        await host.push_screen(ModelCatalogConsentModal())
        await pilot.pause()
        _type(host, "x")
        await pilot.pause(0.3)
        assert composer.draft_text() == ""

        # Escape answers the modal; the Console resumes with focus where it
        # was, and its (held) resume restore is the arrival window again.
        await pilot.press("escape")
        await _wait_for(
            pilot,
            lambda: host.screen is console,
            "the modal never closed",
        )
        assert console.focused is arrival
        _type(host, "hi")
        await _wait_for(
            pilot,
            lambda: composer.draft_text() == "hi",
            lambda: (
                "keys typed on return from the modal were not typed into the "
                f"draft: draft {composer.draft_text()!r}, focus "
                f"{_name(console.focused)}"
            ),
        )
        assert console.focused is composer


@pytest.mark.asyncio
@private_profile_test
async def test_a_collapsed_composer_keeps_arrival_keys_out_of_the_draft(request):
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _arrived(host, pilot)
        arrival = console.focused
        console._set_console_composer_collapsed(True)
        # Collapsing hands focus to the transcript; let that land first.
        await _wait_for(
            pilot,
            lambda: (
                composer.collapsed
                and console._console_workbench_focus_id_for_widget(console.focused)
                == "console-transcript-surface"
            ),
            lambda: f"never collapsed: focus {_name(console.focused)}",
        )
        # Leave the Console and come back with focus on the arrival widget:
        # the revisit's own restore is held, so its window is open.
        console.set_focus(arrival)
        await host.push_screen(ModelCatalogConsentModal())
        await pilot.pause()
        await pilot.press("escape")
        await _wait_for(pilot, lambda: host.screen is console, "never returned")
        assert console.focused is arrival

        _type(host, "x")
        await pilot.pause(0.3)
        assert composer.draft_text() == ""
        assert console.focused is arrival


# ---------------------------------------------------------------------------
# The claim's gates, without a mounted app
# ---------------------------------------------------------------------------


def _claim_case(*, card_mode: str = "quiet", **screen_fields):
    """A screen and composer double for `claim_arrival_key` alone."""
    arrival = object()
    focused_calls: list = []
    composer = SimpleNamespace(
        collapsed=False,
        focus_draft_from=lambda control: focused_calls.append(control),
    )
    displayed = {"console-native-composer", "console-transcript-surface"}
    fields = {
        "_console_arrival_focus": arrival,
        "focused": arrival,
        "_resume_navigation_startup_in_progress": False,
        "_last_console_workbench_focus_id": None,
        "_is_console_widget_displayed": lambda widget_id: widget_id in displayed,
        "_build_console_setup_card_state": lambda: SimpleNamespace(mode=card_mode),
    }
    fields.update(screen_fields)
    return SimpleNamespace(**fields), composer, arrival, focused_calls


def test_an_arrival_key_moves_focus_to_the_draft_when_the_restore_would():
    screen, composer, arrival, focused_calls = _claim_case()
    run_controls.claim_arrival_key(screen, composer, Key("x", "x"))
    assert focused_calls == [arrival]


@pytest.mark.parametrize(
    ("label", "overrides", "key"),
    [
        ("setup card about to block", {"card_mode": "card"}, Key("x", "x")),
        (
            "explicit Resume owns arrival",
            {"_resume_navigation_startup_in_progress": True},
            Key("x", "x"),
        ),
        (
            "restore would pick the transcript",
            {"_last_console_workbench_focus_id": "console-transcript-surface"},
            Key("x", "x"),
        ),
        ("focus already moved", {"focused": object()}, Key("x", "x")),
        ("window over", {"_console_arrival_focus": None}, Key("x", "x")),
        ("not typing", {}, Key("enter", None)),
        ("a chord", {}, Key("ctrl+w", None)),
    ],
)
def test_an_arrival_key_leaves_focus_alone_when(label, overrides, key):
    screen, composer, _arrival, focused_calls = _claim_case(**overrides)
    run_controls.claim_arrival_key(screen, composer, key)
    assert focused_calls == [], label


def test_a_collapsed_composer_takes_no_arrival_key():
    screen, composer, _arrival, focused_calls = _claim_case()
    composer.collapsed = True
    run_controls.claim_arrival_key(screen, composer, Key("x", "x"))
    assert focused_calls == []


def _arm_case(*, focused, pane_of=lambda _widget: None, **fields):
    watches: list = []
    screen = SimpleNamespace(
        focused=focused,
        _console_workbench_focus_id_for_widget=pane_of,
        watch=lambda *args, **kwargs: watches.append((args, kwargs)),
        **fields,
    )
    return screen, watches


def test_an_activation_arms_the_focus_it_found_and_watches_it_once():
    arrival = object()
    screen, watches = _arm_case(focused=arrival)
    run_controls.arm_arrival_focus(screen)
    run_controls.arm_arrival_focus(screen)
    assert screen._console_arrival_focus is arrival
    assert len(watches) == 1

    # Focus landing anywhere else closes the window; the same widget does not.
    _args, _kwargs = watches[0]
    on_change = _args[2]
    on_change(arrival)
    assert screen._console_arrival_focus is arrival
    on_change(object())
    assert screen._console_arrival_focus is None


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        (
            "focus already in a Console pane",
            {"pane_of": lambda _widget: "console-left-rail"},
        ),
        ("nothing focused", {"focused": None}),
    ],
)
def test_an_activation_arms_nothing_for(label, kwargs):
    fields = {"focused": object(), "_console_arrival_focus": "unchanged"}
    fields.update(kwargs)
    screen, watches = _arm_case(**fields)
    run_controls.arm_arrival_focus(screen)
    assert screen._console_arrival_focus is None, label
    assert watches == [], label
