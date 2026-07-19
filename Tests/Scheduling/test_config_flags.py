"""Tests for scheduling feature-flag defaults."""

from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML


def test_watchlist_checks_flags_have_defaults() -> None:
    scheduling = DEFAULT_CONFIG_FROM_TOML["scheduling"]
    assert scheduling["watchlist_checks_enabled"] is False
    assert scheduling["watchlist_checks_shadow"] is True
