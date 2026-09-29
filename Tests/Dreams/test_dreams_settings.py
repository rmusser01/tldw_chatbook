from tldw_chatbook.Dreams.settings import DREAMS_DEFAULTS, dreams_setting


def test_defaults_cover_spec_config_keys():
    expected = {
        "enabled", "provider", "model", "stories_per_cycle", "queries_per_cycle",
        "exploration_slots", "search_engine", "web_search_enabled", "region",
        "cadence_hours", "catchup_enabled", "watchlist_freshness_hours",
        "seen_item_ttl_days", "max_searches_per_day", "max_llm_calls_per_day",
        "tracked_item_cap", "track_min_check_interval_hours",
        "track_quiet_retire_count",
    }
    assert expected <= set(DREAMS_DEFAULTS)


def test_dreams_setting_returns_default_when_unset(monkeypatch):
    monkeypatch.setattr("tldw_chatbook.Dreams.settings.get_cli_setting",
                        lambda section, key, default: default)
    assert dreams_setting("stories_per_cycle") == 5
    assert dreams_setting("enabled") is False


def test_track_lifecycle_keys_have_exact_defaults():
    """Phase 2 Task 6: the caps/floors the track loop reads, single-sourced.

    Task 3's fallback literals (20/12) and the quiet-retire count resolve
    through ``DREAMS_DEFAULTS`` now; these exact values are what
    ``track_service`` and the user guide both depend on.
    """
    assert DREAMS_DEFAULTS["tracked_item_cap"] == 20
    assert DREAMS_DEFAULTS["track_min_check_interval_hours"] == 12
    assert DREAMS_DEFAULTS["track_quiet_retire_count"] == 14


def test_track_lifecycle_keys_resolve_through_defaults_when_unset(monkeypatch):
    monkeypatch.setattr("tldw_chatbook.Dreams.settings.get_cli_setting",
                        lambda section, key, default: default)
    assert dreams_setting("tracked_item_cap") == 20
    assert dreams_setting("track_min_check_interval_hours") == 12
    assert dreams_setting("track_quiet_retire_count") == 14
