"""Inspect current web-search configuration keys without revealing secrets."""

from tldw_chatbook.Web_Scraping.WebSearch_APIs import initialize_config


def test_current_search_configuration_contains_supported_fields():
    keys = initialize_config()["search_engines"]
    assert {
        "bing_search_api_key",
        "google_search_api_key",
        "brave_search_api_key",
        "bing_search_api_url",
        "google_search_api_url",
        "searx_search_api_url",
    } <= keys.keys()
