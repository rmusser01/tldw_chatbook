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


@pytest.mark.parametrize(
    "flag, expected",
    [("watchlist_checks_enabled", True), ("watchlist_checks_shadow", False)],
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
