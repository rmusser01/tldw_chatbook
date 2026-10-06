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
from textual.app import App, ComposeResult
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
    assert startup_splash.splash_options({"splash_screen": {"enabled": True}}) is None


def test_a_later_launch_plays_the_configured_splash() -> None:
    options = startup_splash.splash_options(
        {
            "first_run": {"setup_started": True},
            "splash_screen": {"enabled": True, "duration": 2.0, "card_selection": "matrix"},
        },
        environ={},
    )
    assert options == {
        "card_name": "matrix",
        "duration": 2.0,
        "skip_on_keypress": True,
        "show_progress": True,
        "reduced_motion": False,
    }


def test_splash_off_plays_nothing_after_the_first_launch() -> None:
    config = {"first_run": {"setup_completed": True}, "splash_screen": {"enabled": False}}
    assert startup_splash.splash_options(config) is None


@pytest.mark.parametrize("card", sorted(startup_splash.OPT_IN_ONLY_CARDS))
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


class _SplashHost(App):
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
            assert hints[0].region.height >= 1
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
        {"first_run": {"setup_started": True}, "splash_screen": {}},
        environ={"NO_MOTION": "1"},
    )
    assert options is not None and options["reduced_motion"] is True


def test_loaded_settings_carry_the_environment_motion_override(monkeypatch) -> None:
    from tldw_chatbook.config import load_settings

    monkeypatch.setenv("TLDW_REDUCE_MOTION", "1")
    settings = load_settings(force_reload=True)
    assert settings["appearance"]["reduce_motion"] is True
    monkeypatch.delenv("TLDW_REDUCE_MOTION")
    load_settings(force_reload=True)


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
