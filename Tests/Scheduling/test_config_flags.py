"""Tests for scheduling feature-flag defaults."""

import inspect
import re

import pytest

from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML


def test_watchlist_checks_run_by_default() -> None:
    """The shipped config must execute watchlist checks and persist their results.

    Both flags were staged for the ADR-019 shadow-mode dual-run against the legacy
    ``SubscriptionScheduler``. That scheduler is unreachable, so the staging values
    meant no watchlist was ever checked on a schedule (TASK-1210).
    """
    scheduling = DEFAULT_CONFIG_FROM_TOML["scheduling"]
    assert scheduling["watchlist_checks_enabled"] is True
    assert scheduling["watchlist_checks_shadow"] is False


def test_briefing_schedules_run_by_default() -> None:
    """Briefings phase 4: the shipped config must fire scheduled briefings.

    Unlike the watchlist flags, this one has no staged/dead history -- it is
    a brand new key (Locked Decision 4: opt-in *per watchlist* via
    ``briefing_cadence_seconds``, not a reason for the *app-level* gate to
    default off too; a watchlist with no cadence set already opts itself out).
    """
    scheduling = DEFAULT_CONFIG_FROM_TOML["scheduling"]
    assert scheduling["briefing_schedules_enabled"] is True


@pytest.mark.parametrize(
    "flag, expected",
    [
        ("watchlist_checks_enabled", True),
        ("watchlist_checks_shadow", False),
        ("briefing_schedules_enabled", True),
    ],
)
def test_watchlist_flag_fallbacks_match_shipped_config(flag, expected) -> None:
    """A config.toml missing these keys must behave like the shipped one.

    ``app.py`` reads both flags with ``get_cli_setting(..., default)``. If those
    in-code fallbacks drift from the shipped TOML, users whose config predates the
    keys silently get different scheduling behaviour from new users -- which is how
    TASK-1210 stayed invisible.
    """
    from tldw_chatbook import app as app_module

    source = inspect.getsource(app_module)
    match = re.search(
        r'get_cli_setting\(\s*"scheduling",\s*"' + flag + r'",\s*(True|False)\s*,?\s*\)',
        source,
    )
    assert match is not None, f"no get_cli_setting call found for {flag}"
    assert (match.group(1) == "True") is expected


def test_briefing_projection_is_only_wired_when_the_flag_is_on() -> None:
    """The "config gate off -> no projection wired" seam, pinned honestly.

    Booting the whole ``TldwCli`` app to prove the flag really disables
    scheduled generation is prohibitively heavy for a unit test, so this
    pins the *pattern* ``app.py`` must keep instead -- the exact shape
    already established for ``watchlist_handler``/``watchlist_projection``
    (``watchlist_projection if watchlist_handler is not None else None``):
    a handler is constructed only inside the flag's own ``if``, and
    ``SchedulerLoop`` is handed the projection only when that handler
    exists. ``Tests/Scheduling/test_scheduler_loop.py``'s
    ``test_queue_with_no_briefing_projection_loads_no_briefing_jobs`` pins
    the other half: that passing ``None`` genuinely loads nothing,
    regardless of what the projection would otherwise report.
    """
    from tldw_chatbook import app as app_module

    source = inspect.getsource(app_module)

    assert "briefing_projection = None" in source
    assert "briefing_handler = None" in source

    # Both assignments live inside one `if briefing_schedules_enabled:` block,
    # with nothing but simple statements between the `if` and each
    # assignment -- i.e. neither is reachable when the flag is off.
    gate_block = re.search(
        r"if\s+briefing_schedules_enabled:\n((?:[ \t]+\S.*\n)+)",
        source,
    )
    assert gate_block is not None, "no `if briefing_schedules_enabled:` block found"
    body = gate_block.group(1)
    assert "briefing_projection = BriefingProjection(" in body
    assert "briefing_handler = BriefingJobHandler(" in body

    assert re.search(
        r"briefing_projection\s+if\s+briefing_handler is not None\s+else\s+None",
        source,
    ), "SchedulerLoop must only receive the projection when the handler exists"
