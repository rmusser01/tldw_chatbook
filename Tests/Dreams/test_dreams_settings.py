from tldw_chatbook.Dreams.settings import DREAMS_DEFAULTS, dreams_setting


def test_defaults_cover_spec_config_keys():
    expected = {
        "enabled", "provider", "model", "stories_per_cycle", "queries_per_cycle",
        "exploration_slots", "search_engine", "web_search_enabled", "region",
        "cadence_hours", "catchup_enabled", "watchlist_freshness_hours",
        "seen_item_ttl_days", "max_searches_per_day", "max_llm_calls_per_day",
    }
    assert expected <= set(DREAMS_DEFAULTS)


def test_dreams_setting_returns_default_when_unset(monkeypatch):
    monkeypatch.setattr("tldw_chatbook.Dreams.settings.get_cli_setting",
                        lambda section, key, default: default)
    assert dreams_setting("stories_per_cycle") == 5
    assert dreams_setting("enabled") is False
