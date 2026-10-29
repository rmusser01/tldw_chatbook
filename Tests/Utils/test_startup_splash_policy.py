"""TASK-34100.13 (entry-exit-handoff-17, new-coverage-power-01, E7).

A profile's first launch played 10-16 s of a random splash -- the JULES
intro, a high-score table with URMOM on it -- with no skip hint, after raw
dependency warnings on the terminal. Now: no splash on a first launch, joke
and persona cards out of the random pool, a dim "Press any key to skip" line
whenever the skip works, reduced motion from ``TLDW_REDUCE_MOTION`` /
``NO_MOTION``, and pre-TUI warnings routed to the log.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from textual.app import ComposeResult

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Static

from tldw_chatbook.Utils import startup_logging
from tldw_chatbook.Widgets import splash_screen as startup_splash
from tldw_chatbook.Widgets.splash_screen import SplashScreen

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The splash widget reads [splash_screen] through the loaded config, so the
#: tests run on the shared bootstrap profile (Tests/conftest.py).
pytestmark = pytest.mark.bootstrap_profile


# -- AC#10: first launch, card pool ------------------------------------------


def test_a_first_launch_plays_no_splash() -> None:
    """The launch that created the profile's config file is its first."""
    options = startup_splash.splash_options(
        {"splash_screen": {"enabled": True}}, first_launch=True
    )
    assert options is None


def test_a_later_launch_plays_the_configured_splash() -> None:
    options = startup_splash.splash_options(
        {"splash_screen": {"enabled": True, "duration": 2.0, "card_selection": "matrix"}},
        environ={},
        first_launch=False,
    )
    assert options == {
        "card_name": "matrix",
        "duration": 2.0,
        "skip_on_keypress": True,
        "show_progress": True,
        "reduced_motion": False,
    }


def test_splash_off_plays_nothing_after_the_first_launch() -> None:
    config = {"splash_screen": {"enabled": False}}
    assert startup_splash.splash_options(config, first_launch=False) is None


def test_a_configured_profile_that_never_ran_setup_plays_its_splash() -> None:
    """Review F1 (TASK-34100.13): "first launch" used to mean "no [first_run]
    flag", and a hand-written or pre-wizard config with an inline provider key
    never gets one -- the wizard is not offered and no env-key notice fires.
    Such a profile lost its configured splash on every launch, forever."""
    config = {
        "splash_screen": {"enabled": True, "card_selection": "matrix"},
        "api_settings": {"openai": {"api_key": "sk-configured-inline"}},
    }
    options = startup_splash.splash_options(config, environ={}, first_launch=False)
    assert options is not None
    assert options["card_name"] == "matrix"


#: The cards the AC names, written out here rather than read from the
#: implementation's set: dropping one from OPT_IN_ONLY_CARDS must fail this
#: test, not quietly take its case away (review-tests F1).
JOKE_AND_PERSONA_CARDS = (
    "retro_gaming_intro",  # the "A JULES PRODUCTION" intro
    "character_select",  # a JULES player card
    "arcade_high_score",  # the URMOM high-score table
    "old_film",  # "Starring YOU!"
)


def test_the_opt_in_set_covers_every_named_card() -> None:
    assert set(JOKE_AND_PERSONA_CARDS) <= startup_splash.OPT_IN_ONLY_CARDS


@pytest.mark.parametrize("card", JOKE_AND_PERSONA_CARDS)
def test_joke_and_persona_cards_never_play_at_random(card: str, monkeypatch) -> None:
    pools: list[list[str]] = []
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.splash_screen.random.choice",
        lambda pool: pools.append(list(pool)) or pool[0],
    )
    splash = SplashScreen(card_name=None, skip_on_keypress=False)
    assert splash.card_name != card
    assert pools and card not in pools[0]


def test_a_joke_card_still_plays_when_chosen_by_name() -> None:
    splash = SplashScreen(card_name="arcade_high_score", skip_on_keypress=False)
    assert splash.card_name == "arcade_high_score"


# -- AC#11: the skip hint ------------------------------------------------------


class _SplashHost(ConsolidatedCSSApp):
    """The app's real stylesheet: #splash-display takes the full height there,
    which pushed an in-flow hint off screen on the first build."""

    CSS_PATH = str(REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss")

    def __init__(self, splash: SplashScreen) -> None:
        super().__init__()
        self._splash = splash

    def compose(self) -> ComposeResult:
        yield self._splash


@pytest.mark.parametrize("skip", [True, False])
@pytest.mark.asyncio
async def test_the_splash_draws_a_skip_hint_whenever_skip_works(skip: bool) -> None:
    splash = SplashScreen(
        card_name="default", duration=30, skip_on_keypress=skip, show_progress=True
    )
    async with _SplashHost(splash).run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        hints = list(splash.query("#splash-skip-hint"))
        if skip:
            assert len(hints) == 1
            assert str(hints[0].render()) == "Press any key to skip"
            region = hints[0].region
            assert region.height >= 1
            assert region.bottom <= pilot.app.size.height, "hint is off screen"
        else:
            assert hints == []


# -- AC#16: reduce motion from the environment ---------------------------------


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"TLDW_REDUCE_MOTION": "1"}, True),
        ({"NO_MOTION": "yes"}, True),
        ({"TLDW_REDUCE_MOTION": "0"}, False),
        ({}, False),
    ],
)
def test_reduce_motion_comes_from_the_environment(environ, expected) -> None:
    assert startup_logging.motion_reduced_by_environment(environ) is expected
    appearance = startup_logging.with_motion_environment({}, environ)
    assert appearance.get("reduce_motion", False) is expected


def test_the_environment_reduces_the_splash_motion() -> None:
    options = startup_splash.splash_options(
        {"splash_screen": {}}, environ={"NO_MOTION": "1"}, first_launch=False
    )
    assert options is not None and options["reduced_motion"] is True


def test_the_environment_motion_override_is_never_saved_by_settings(monkeypatch) -> None:
    """Review F5 (TASK-34100.13): the override was folded into load_settings'
    ``[appearance]``, the table Settings ▸ Appearance loads and writes back
    whole -- so saving any Appearance change (density, say) under
    TLDW_REDUCE_MOTION persisted reduce_motion = true. It is applied where
    motion is read instead, so the loaded settings stay what is saved."""
    from tldw_chatbook.UI.Screens.settings_appearance_defaults import (
        build_appearance_save_sections,
        load_appearance_defaults,
    )
    from tldw_chatbook.config import load_settings

    monkeypatch.setenv("TLDW_REDUCE_MOTION", "1")
    try:
        settings = load_settings(force_reload=True)
        assert settings["appearance"].get("reduce_motion", False) is False
        values = load_appearance_defaults(settings)
        sections = build_appearance_save_sections(settings, values)
        assert sections["appearance"]["reduce_motion"] is False
    finally:
        monkeypatch.delenv("TLDW_REDUCE_MOTION")
        load_settings(force_reload=True)


@pytest.mark.parametrize(
    ("environ", "saved", "expected"),
    [
        ({"NO_MOTION": "1"}, False, True),
        ({}, True, True),
        ({}, "false", False),
        ({"TLDW_REDUCE_MOTION": "0"}, False, False),
    ],
)
def test_motion_is_reduced_by_the_saved_setting_or_the_environment(
    environ, saved, expected
) -> None:
    reduced = startup_logging.motion_reduced({"reduce_motion": saved}, environ)
    assert reduced is expected


def test_character_expressions_hold_still_under_the_environment_override(
    monkeypatch,
) -> None:
    from tldw_chatbook.Chat.character_expression_playback import (
        expression_motion_enabled,
    )

    config = {"appearance": {"reduce_motion": False}}
    assert expression_motion_enabled(config, react=True, manual=False) is True
    monkeypatch.setenv("NO_MOTION", "1")
    assert expression_motion_enabled(config, react=True, manual=False) is False


# -- AC#10: dependency warnings stay off the pre-TUI terminal ------------------


_PROBE = r"""
import logging, warnings
from tldw_chatbook.Utils.startup_logging import quiet_startup_stderr, replay_startup_notices
quiet_startup_stderr()
warnings.warn("urllib3 (2.6.3) doesn't match a supported version!", UserWarning)
from loguru import logger
logger.warning("python-frontmatter not installed. Markdown import will not be available.")
logging.getLogger("some.dependency").warning("optional dependency missing")
records = []
class Capture(logging.Handler):
    def emit(self, record):
        records.append(record.getMessage())
logging.getLogger().addHandler(Capture())
logging.getLogger().setLevel(logging.INFO)
replay_startup_notices()
import sys
print("\n".join(records), file=sys.stdout)
"""


def test_pre_tui_warnings_go_to_the_log_not_the_terminal() -> None:
    env = {k: v for k, v in os.environ.items() if k != "TLDW_VERBOSE_STARTUP"}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.strip() == ""
    replayed = result.stdout
    assert "doesn't match a supported version" in replayed
    assert "python-frontmatter not installed" in replayed
    assert "optional dependency missing" in replayed


_EXIT_WITHOUT_REPLAY = r"""
import logging, warnings
from tldw_chatbook.Utils.startup_logging import quiet_startup_stderr
quiet_startup_stderr()
warnings.warn("held python warning", UserWarning)
from loguru import logger
logger.warning("held loguru notice")
logging.getLogger("some.dependency").warning("held stdlib notice")
# A recovery subcommand, a refused unlock or an early crash: the app's
# logging is never configured, so nothing replays the held notices.
"""


def test_held_warnings_reach_stderr_when_the_app_never_configures_logging() -> None:
    """Review F10 (TASK-34100.13): quiet_startup_stderr() holds WARNINGs for the
    log, but only Logging_Config replays them. ``tldw-cli recovery``, a refused
    startup unlock or a crash before the TUI never got that far, so their
    warnings -- which used to reach stderr -- were silently dropped."""
    env = {k: v for k, v in os.environ.items() if k != "TLDW_VERBOSE_STARTUP"}
    result = subprocess.run(
        [sys.executable, "-c", _EXIT_WITHOUT_REPLAY],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    for notice in ("held python warning", "held loguru notice", "held stdlib notice"):
        assert notice in result.stderr


_STDLIB_ERRORS_BEFORE_THE_TUI = r"""
import logging, sys
from tldw_chatbook.Utils.startup_logging import quiet_startup_stderr
quiet_startup_stderr()
if sys.argv[1] == "other-handler":
    class Other(logging.Handler):
        def emit(self, record):
            sys.stderr.write("OTHER " + record.getMessage() + "\n")
    logging.getLogger().addHandler(Other(level=logging.ERROR))
logging.getLogger("some.module").error("STDLIB ERROR pre-TUI")
logging.getLogger("other.module").critical("STDLIB CRITICAL pre-TUI")
logging.getLogger("third.module").warning("held stdlib warning")
sys.stderr.write("--exit--\n")
sys.stderr.flush()
"""


@pytest.mark.parametrize("handlers", ["none", "other-handler"])
def test_a_stdlib_error_before_the_tui_still_reaches_stderr(handlers: str) -> None:
    """Review R3 (TASK-34100.13): the holding handler on the root logger stops
    ``logging.lastResort`` from printing held WARNINGs, but it dropped ERROR
    and CRITICAL records outright -- printed by dev through lastResort, now
    lost. Errors print when they happen, once, and only when no other handler
    took them (lastResort's own rule); warnings are still held for the exit
    flush."""
    env = {k: v for k, v in os.environ.items() if k != "TLDW_VERBOSE_STARTUP"}
    result = subprocess.run(
        [sys.executable, "-c", _STDLIB_ERRORS_BEFORE_THE_TUI, handlers],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    before, _, at_exit = result.stderr.partition("--exit--")
    for message in ("STDLIB ERROR pre-TUI", "STDLIB CRITICAL pre-TUI"):
        assert result.stderr.count(message) == 1, result.stderr
        assert message in before
    if handlers == "other-handler":
        assert "OTHER STDLIB ERROR pre-TUI" in before
    assert "held stdlib warning" not in before
    assert "held stdlib warning" in at_exit


_CONFIGURE_LOGGING = r"""
import logging, warnings
from types import SimpleNamespace
from tldw_chatbook.Utils.startup_logging import quiet_startup_stderr
quiet_startup_stderr()
warnings.warn("urllib3 (2.6.3) doesn't match a supported version!", UserWarning)
from tldw_chatbook import Logging_Config

records = []
class Capture(logging.Handler):
    def emit(self, record):
        records.append(record.getMessage())
logging.getLogger("tldw_chatbook.startup").addHandler(Capture())

def no_log_widget(*_args, **_kwargs):
    raise LookupError("no log widget in this probe")

app = SimpleNamespace(
    app_config={"general": {"log_level": "INFO"}, "logging": {}},
    _rich_log_handler=None,
    query_one=no_log_widget,
)
Logging_Config.configure_application_logging(app)
print("\n".join(records))
"""


def test_configuring_the_app_log_replays_the_held_notices() -> None:
    """review-tests F5: the app's own logging setup is what writes the held
    notices to the log; removing that call must fail a test."""
    env = {k: v for k, v in os.environ.items() if k != "TLDW_VERBOSE_STARTUP"}
    result = subprocess.run(
        [sys.executable, "-c", _CONFIGURE_LOGGING],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    replayed = [
        line for line in result.stdout.splitlines() if "Before the app opened" in line
    ]
    assert any("doesn't match a supported version" in line for line in replayed), (
        result.stdout[-2000:]
    )
    assert "doesn't match a supported version" not in result.stderr


def test_buddy_holds_still_under_the_environment_override(monkeypatch) -> None:
    """Buddy read reduced motion from the env-folded settings; with the fold
    gone (review F5) it checks the environment where it reads motion."""
    from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController

    monkeypatch.delenv("NO_MOTION", raising=False)
    monkeypatch.delenv("TLDW_REDUCE_MOTION", raising=False)
    controller = PersonaBuddyController(reduced_motion=lambda: False)
    assert controller._current_reduced_motion() is False
    monkeypatch.setenv("TLDW_REDUCE_MOTION", "1")
    assert controller._current_reduced_motion() is True
