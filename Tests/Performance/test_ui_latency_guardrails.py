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
import tempfile
from pathlib import Path

import pytest

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
#: ctrl+5 personas, ctrl+7 schedules, ctrl+9 mcp, f7 lab(llm), f9 settings.
DESTINATION_TOUR: tuple[tuple[str, str], ...] = (
    ("ctrl+1", "HomeScreen"),
    ("ctrl+2", "ChatScreen"),
    ("ctrl+3", "LibraryScreen"),
    ("ctrl+5", "PersonasScreen"),
    ("ctrl+7", "SchedulesWorkbench"),
    ("ctrl+9", "MCPScreen"),
    ("f7", "LLMScreen"),
    ("f9", "SettingsScreen"),
)

_SETTLE_PASSES = 6
_SETTLE_INTERVAL = 0.05


def _scratch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point every config/data seam at a scratch tree with setup completed."""
    home = tmp_path / "home"
    data = tmp_path / "data"
    config = tmp_path / "config"
    for sub in (home, data, config):
        sub.mkdir(parents=True, exist_ok=True)
    config_file = config / "tldw_cli" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "[first_run]\nsetup_completed = true\n\n[splash_screen]\nenabled = false\n"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_file))
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    # Same signal the app's own screen-preimport gate reads: keep the probe
    # environment free of the background import thread.
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "latency_guardrails")
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
async def test_destination_tour_stays_under_switch_budgets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every hot destination arrives and settles inside its budget."""
    _scratch_env(monkeypatch, tmp_path)
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        # Boot settle: the initial screen's own mount work must finish so it
        # is not billed to the first destination.
        for _ in range(20):
            await asyncio.sleep(0.05)
            await pilot.pause()

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
        assert not failures, "Screen-switch latency guardrail tripped:\n" + "\n".join(
            failures
        )


@pytest.mark.ui
@pytest.mark.asyncio
async def test_destination_tour_css_sources_stay_below_parse_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A full destination tour must not cross Textual's LRU-64 parse cliff."""
    _scratch_env(monkeypatch, tmp_path)
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
