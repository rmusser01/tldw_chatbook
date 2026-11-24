"""UI latency regression guardrails (TASK-18908).

Born of the 2026-08 Windows 3 s-lag incident: after the run-path fixes
landed (PR #1824), the remaining regression vectors were structural
(screen-mount weight) or silent (the Textual CSS parse-cache cliff). These
tests set off the alarm before users do.

Two guardrails, both measured on the same Pilot destination tour:

1. **Screen-switch arrival budgets.** Every hot destination must ARRIVE
   (the expected screen class is active) and settle within a generous
   budget. The budgets are deliberately loose — an order-of-magnitude
   regression (this week's class) trips them; normal drift does not.
   Measured baseline on an M-series Mac (2026-08-19, dev ``f6ae7d23e``):
   Home 0.76 s, Console 1.55 s, Library 1.39 s, Settings 0.89–1.38 s.
   CI runners are slower; budgets sit far above baseline and far below
   the incident.

2. **CSS source-count cliff.** Textual 8.2.8's stylesheet parse cache is
   an ``LRUCache(64)``. TASK-15450 consolidated widget CSS into single
   registered sources (a full tour lands at 44 today); this guard fails
   the first PR whose new ``CSS_PATH``/``DEFAULT_CSS`` declarations push a
   full destination tour past the cliff, where every first mount pays a
   full cold reparse (125–380 ms measured per unseen widget class).

The arrival check asserts the SCREEN CLASS NAME, not a label: the
2026-08-19 probe that motivated this file originally waited for a
"Console" label that never matched (ctrl+2 routes to ``ChatScreen``), and
its 30 s deadline expiry was misread as app latency.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
from textual.widgets import Static

from Tests.private_profile import private_profile_test

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Textual 8.2.8 ``Stylesheet`` parse cache size (``textual/css/stylesheet.py``).
#: Going past this evicts every source; a tour that ends above it means cold
#: reparses on first mounts.
CSS_PARSE_CACHE_LIMIT = 64

#: Head-room below the cliff: failing at exactly 64 would leave no room to
#: review. 56 leaves one review cycle (~8 sources) before the cliff.
CSS_SOURCE_SOFT_LIMIT = CSS_PARSE_CACHE_LIMIT - 8

#: Seconds from keypress to (arrival + settle) per destination. Generous by
#: design: baselines are 0.76–1.9 s on a fast Mac; the incident class is 3 s+
#: on constrained hardware (≈3–5× these numbers, i.e. 6–10 s here).
SCREEN_SWITCH_BUDGET_SECONDS = 10.0

#: The audit tour, hotkey -> expected screen class (``type(app.screen)``).
#: SHELL_DESTINATION_ORDER: ctrl+1 home, ctrl+2 console(chat), ctrl+3 library,
#: ctrl+4 personas, ctrl+7 schedules, f2 lab(llm), f4 settings. ADR-240:
#: ctrl+9 is the retained Tools & MCP compatibility deep-link, whose target
#: screen is Settings now -- the tour still measures the hotkey's switch.
DESTINATION_TOUR: tuple[tuple[str, str], ...] = (
    ("ctrl+1", "HomeScreen"),
    ("ctrl+2", "ChatScreen"),
    ("ctrl+3", "LibraryScreen"),
    ("ctrl+4", "PersonasScreen"),
    ("ctrl+7", "SchedulesWorkbench"),
    ("ctrl+9", "SettingsScreen"),
    ("f2", "LLMScreen"),
    ("f4", "SettingsScreen"),
)

_SETTLE_PASSES = 6
_SETTLE_INTERVAL = 0.05

#: TASK-38687: how long the idle app is watched, once with Console active and
#: once after the tour, and how many calls to ``_sync_native_console_chat_ui``
#: it may start in each window. The readiness projection and the
#: sync once fed each other ~20 times a second on every screen; each idle
#: ``pilot.pause()`` then waited out its 1 s ceiling and the readings crossed
#: the budget. A count, not a clock: the loop is many per second on any CPU.
IDLE_WINDOW_SECONDS = 1.0
IDLE_CONSOLE_SYNC_LIMIT = 1


def _scratch_env(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Populate the scratch profile selected before this interpreter's imports."""
    home = Path(os.environ["HOME"])
    data = Path(os.environ["XDG_DATA_HOME"])
    config = Path(os.environ["XDG_CONFIG_HOME"])
    for sub in (home, data, config):
        sub.mkdir(parents=True, exist_ok=True)
    config_file = Path(os.environ["TLDW_CONFIG_PATH"])
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "[first_run]\nsetup_completed = true\n\n[splash_screen]\nenabled = false\n"
    )
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    # Same signal the app's own screen-preimport gate reads: keep the probe
    # environment free of the background import thread.
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "latency_guardrails")
    from tldw_chatbook.config import load_settings

    load_settings(force_reload=True)
    return home


async def _wait_for_screen(pilot, expected: str, budget: float) -> tuple[bool, float]:
    """Press-and-wait helper; returns (arrived, elapsed_seconds)."""
    deadline = asyncio.get_running_loop().time() + budget
    while asyncio.get_running_loop().time() < deadline:
        await pilot.pause()
        if type(pilot.app.screen).__name__ == expected:
            break
    arrived = type(pilot.app.screen).__name__ == expected
    for _ in range(_SETTLE_PASSES):
        await asyncio.sleep(_SETTLE_INTERVAL)
        await pilot.pause()
    return arrived, budget - (deadline - asyncio.get_running_loop().time())


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_destination_tour_stays_under_switch_budgets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Every hot destination arrives and settles inside its budget."""
    _scratch_env(monkeypatch)
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        # Boot settle: the initial screen's own mount work must finish so it
        # is not billed to the first destination.
        for _ in range(20):
            await asyncio.sleep(0.05)
            await pilot.pause()

        chat_screen = sys.modules["tldw_chatbook.UI.Screens.chat_screen"].ChatScreen
        sync = chat_screen._sync_native_console_chat_ui
        syncs = 0

        async def counted(self, *args, **kwargs):
            nonlocal syncs
            syncs += 1
            return await sync(self, *args, **kwargs)

        async def check_idle(where: str) -> None:
            nonlocal syncs
            syncs = 0
            await asyncio.sleep(IDLE_WINDOW_SECONDS)
            if syncs > IDLE_CONSOLE_SYNC_LIMIT:
                failures.append(
                    f"idle {where} made {syncs} _sync_native_console_chat_ui calls in "
                    f"{IDLE_WINDOW_SECONDS}s (limit {IDLE_CONSOLE_SYNC_LIMIT}): a sync is re-arming itself"
                )

        monkeypatch.setattr(chat_screen, "_sync_native_console_chat_ui", counted)
        failures: list[str] = []
        for key, expected in DESTINATION_TOUR:
            t0 = asyncio.get_running_loop().time()
            await pilot.press(key)
            arrived, _ = await _wait_for_screen(
                pilot, expected, SCREEN_SWITCH_BUDGET_SECONDS
            )
            elapsed = asyncio.get_running_loop().time() - t0
            if not arrived:
                failures.append(
                    f"{key} -> {expected}: NEVER ARRIVED "
                    f"(stuck on {type(pilot.app.screen).__name__})"
                )
            elif elapsed > SCREEN_SWITCH_BUDGET_SECONDS:
                failures.append(f"{key} -> {expected}: {elapsed:.1f}s > budget")
            if arrived and expected == "ChatScreen":
                await check_idle("Console")

        await check_idle("app after the tour")
        assert not failures, "Screen-switch latency guardrail tripped:\n" + "\n".join(
            failures
        )


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_destination_tour_css_sources_stay_below_parse_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """A full destination tour must not cross Textual's LRU-64 parse cliff."""
    _scratch_env(monkeypatch)
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(20):
            await asyncio.sleep(0.05)
            await pilot.pause()
        for key, expected in DESTINATION_TOUR:
            await pilot.press(key)
            arrived, _ = await _wait_for_screen(
                pilot, expected, SCREEN_SWITCH_BUDGET_SECONDS
            )
            assert arrived, f"{key} never reached {expected}"
        sources = len(app.stylesheet.source)
        assert sources < CSS_SOURCE_SOFT_LIMIT, (
            f"CSS sources after a full tour: {sources} "
            f"(soft limit {CSS_SOURCE_SOFT_LIMIT}, cliff at "
            f"{CSS_PARSE_CACHE_LIMIT} = Textual's LRU parse cache). New "
            f"CSS_PATH/DEFAULT_CSS declarations must consolidate per "
            f"TASK-15450 (css/build_css.py)."
        )


# ---------------------------------------------------------------------------
# task-32260: the seeded Library open. The critique measured 12.6 s on a
# 27-item profile against 2.7 s on an empty one and could not trace it. On
# this machine the gap is 0.2 s (see the task's Implementation Notes for the
# full numbers, including a re-measure at the critique's own base commit), so
# there is no slow path to remove -- but nothing pinned the open time on a
# profile that HAS content, which is why a 12x claim could stand unchecked
# for a week. This is that pin: the same generous budget the empty tour uses,
# now also on a seeded profile, where a regression of the reported class
# (12.6 s) fails the build.
# ---------------------------------------------------------------------------

#: What the critique's profile held: 10 notes, 11 media, 6 conversations.
SEEDED_PROFILE_NOTES = 10
SEEDED_PROFILE_MEDIA = 11
SEEDED_PROFILE_CONVERSATIONS = 6


def _seed_library_profile(home: Path) -> None:
    """Write a 27-item Library into the scratch env's own databases.

    Content shape, not volume, is the point: 27 items is the corpus the
    critique measured, and the open path it exercises (per-source counts,
    the recent-items strip, one private-SQLite connection per database) is
    the work an empty profile skips entirely.

    Args:
        home: The scratch HOME the config file was written under.

    Returns:
        None.
    """
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    data = home / "library-budget-data"
    data.mkdir(parents=True, exist_ok=True)
    chacha = CharactersRAGDB(data / "chachanotes.db", "budget")
    for index in range(SEEDED_PROFILE_NOTES):
        chacha.add_note(f"Seeded note {index}", f"Body of seeded note {index}.\n" * 8)
    for index in range(SEEDED_PROFILE_CONVERSATIONS):
        conversation_id = chacha.add_conversation({"title": f"Seeded chat {index}"})
        for turn in range(4):
            chacha.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user" if turn % 2 == 0 else "assistant",
                    "content": f"Turn {turn} of seeded chat {index}.",
                }
            )
    media = MediaDatabase(data / "media.db", "budget")
    for index in range(SEEDED_PROFILE_MEDIA):
        media.add_media_with_keywords(
            title=f"Seeded media {index}",
            media_type="article",
            content=f"Body of seeded media {index}.\n" * 20,
            keywords=["seeded"],
        )


def _point_config_at_seeded_databases(home: Path) -> None:
    """Append explicit database paths so the seed and the app agree."""
    config_file = Path(os.environ["TLDW_CONFIG_PATH"])
    data = home / "library-budget-data"
    config_file.write_text(
        config_file.read_text()
        + '\n[general]\nusers_name = "budget"\n\n[database]\n'
        + f'chachanotes_db_path = "{data / "chachanotes.db"}"\n'
        + f'media_db_path = "{data / "media.db"}"\n'
    )
    from tldw_chatbook.config import load_settings

    load_settings(force_reload=True)


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_library_opens_within_budget_on_a_seeded_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Library arrives and settles inside its budget with 27 items in it."""
    home = _scratch_env(monkeypatch)
    _point_config_at_seeded_databases(home)
    _seed_library_profile(home)

    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    async with app.run_test(size=(235, 52)) as pilot:
        for _ in range(20):
            await asyncio.sleep(0.05)
            await pilot.pause()

        started = asyncio.get_running_loop().time()
        await pilot.press("ctrl+3")
        arrived, _ = await _wait_for_screen(
            pilot, "LibraryScreen", SCREEN_SWITCH_BUDGET_SECONDS
        )
        arrival = asyncio.get_running_loop().time() - started
        # The counts are the SLOW half -- 0.68 s to the painted shell against
        # 1.01 s to resolved counts -- so the clock only stops once they are
        # on screen. Waiting a fixed number of passes and billing none of it
        # would leave the reported regression class unmeasured. They are also
        # what proves this pin is not quietly measuring an empty profile.
        counted: list[str] = []
        deadline = started + SCREEN_SWITCH_BUDGET_SECONDS
        while not counted and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(_SETTLE_INTERVAL)
            await pilot.pause()
            counted = [
                str(widget.renderable)
                for widget in app.screen.query(Static)
                if "Notes (" in str(widget.renderable)
            ]
        settled = asyncio.get_running_loop().time() - started
        stuck_on = type(app.screen).__name__

    item_count = (
        SEEDED_PROFILE_NOTES + SEEDED_PROFILE_MEDIA + SEEDED_PROFILE_CONVERSATIONS
    )
    budget_note = (
        f"budget {SCREEN_SWITCH_BUDGET_SECONDS}s; measured 1.1 s on an "
        "M-series Mac at dev 4a14b3f36f, task-32260"
    )
    # Latency first, content second: a real overrun must report as an
    # overrun, not as a seeding failure.
    assert arrived, (
        f"Library never opened on a {item_count}-item profile "
        f"(stuck on {stuck_on}; {budget_note})"
    )
    assert counted, (
        f"Library arrived in {arrival:.1f}s but its counts never resolved "
        f"within the budget on a {item_count}-item profile ({budget_note})"
    )
    assert settled <= SCREEN_SWITCH_BUDGET_SECONDS, (
        f"Library took {settled:.1f}s to open and settle (arrival "
        f"{arrival:.1f}s) on a {item_count}-item profile ({budget_note})"
    )
    assert (
        f"Notes ({SEEDED_PROFILE_NOTES})" in counted[0]
        and f"Media ({SEEDED_PROFILE_MEDIA})" in counted[0]
    ), f"the seeded profile never reached the screen: {counted[:3]}"


# ---------------------------------------------------------------------------
# TASK-34100.13 (E7, entry-exit-handoff-17): fresh-profile time to Welcome.
# A first launch played the full splash (7 s plus a random joke card) before
# the setup wizard could appear; reviewers measured 10-16 s to Welcome. A
# profile's first launch now plays no splash. This pins both halves: the
# splash never mounts on a first launch (deterministic: that half fails on
# any machine), and Welcome is on screen within the budget. The deterministic
# half is the real splash guard: a returning splash fails it at once.
# The clock starts once run_test() has the app up. Review F13 asked for
# construction to count too; tried 2026-10-27 and reverted, because building
# TldwCli is import-dominated and load-sensitive: on the same M-series Mac at
# load average ~57 the construction-inclusive clock read 12.3 s against 4.5 s
# for this one (and 4.0-4.4 s at load 15-25), and a ratchet that may never
# be raised must not fail on a busy runner. Budget: its own literal, ~3.5x
# this clock's measurement (2.5-2.9 s, 2026-10-05), the same CI headroom as
# the screen-switch guard. RATCHET (ADR-097): never raise it; a breach means new
# work landed between the app coming up and Welcome.
# ---------------------------------------------------------------------------

#: Seconds from run_test()'s running app to the first-run wizard's Welcome step.
TIME_TO_WELCOME_BUDGET_SECONDS = 10.0


def _fresh_profile_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scratch profile with no config file: the real first launch."""
    for name in ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    config_file = Path(os.environ["TLDW_CONFIG_PATH"])
    config_file.parent.mkdir(parents=True, exist_ok=True)
    if config_file.exists():
        config_file.unlink()
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "latency_guardrails")
    from tldw_chatbook.config import load_settings

    load_settings(force_reload=True)


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_fresh_profile_reaches_welcome_without_a_splash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """First launch: no splash, and Welcome inside the time-to-Welcome budget."""
    _fresh_profile_env(monkeypatch)
    from tldw_chatbook.UI.Wizards.first_run_welcome_step import WelcomeStep
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    splash_seen = False
    welcome_after: float | None = None
    async with app.run_test(size=(120, 40)) as pilot:
        started = asyncio.get_running_loop().time()
        deadline = started + 30.0
        while asyncio.get_running_loop().time() < deadline:
            await pilot.pause(0.05)
            splash_seen = splash_seen or bool(app.splash_screen_active)
            steps = list(app.screen.query(WelcomeStep))
            if steps and steps[0].display and steps[0].region.height > 0:
                welcome_after = asyncio.get_running_loop().time() - started
                break
        stuck_on = type(app.screen).__name__

    assert not splash_seen, "a profile's first launch played the splash"
    assert welcome_after is not None, f"Welcome never appeared (on {stuck_on})"
    assert welcome_after <= TIME_TO_WELCOME_BUDGET_SECONDS, (
        f"fresh profile reached Welcome in {welcome_after:.1f}s "
        f"(budget {TIME_TO_WELCOME_BUDGET_SECONDS}s)"
    )
