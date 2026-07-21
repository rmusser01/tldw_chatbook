import pytest


# Copied from Tests/Internal_Prompts/conftest.py (not imported via
# pytest_plugins) because declaring pytest_plugins = ["Tests.Internal_Prompts.
# conftest"] — either in the test module or here — collides with pytest's
# implicit auto-load of Tests/Internal_Prompts/conftest.py under
# --import-mode=importlib when both test directories are collected in the
# same session: "ValueError: Plugin already registered under a different
# name". See Tests/Web_Scraping/test_websearch_internal_prompts.py.
@pytest.fixture
def scratch_config(tmp_path, monkeypatch):
    """Point the app at a throwaway config file. Yields write(toml_text)."""
    from tldw_chatbook import config
    from tldw_chatbook.Internal_Prompts import resolver

    config_file = tmp_path / "config.toml"

    def write(toml_text: str) -> None:
        config_file.write_text(toml_text, encoding="utf-8")
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_file))
        config.load_settings(force_reload=True)

    resolver._warned_ids.clear()
    write("")
    yield write
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    resolver._warned_ids.clear()
    config.load_settings(force_reload=True)
