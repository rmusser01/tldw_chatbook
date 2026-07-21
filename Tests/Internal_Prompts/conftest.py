import pytest


# Automatically applies to every test module in Tests/Internal_Prompts/ so caplog captures loguru output.
@pytest.fixture(autouse=True)
def _loguru_to_caplog(caplog):
    import logging
    from loguru import logger as loguru_logger

    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = loguru_logger.add(PropagateHandler(), format="{message}")
    yield
    loguru_logger.remove(handler_id)


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
    resolver._warned_ids.clear()
