from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import toml

from tldw_chatbook import config


def _reset_config_state() -> None:
    config._CONFIG_CACHE = None
    config._CONFIG_CACHE_SOURCE = None
    config._SETTINGS_CACHE = None
    config._SETTINGS_CACHE_SOURCE = None


def test_runtime_snapshot_is_defensive_and_advances_after_successful_save(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_text(
        toml.dumps(
            {
                "chat_defaults": {"provider": "before"},
                "api_settings": {"openai": {"api_key": "before-key"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _reset_config_state()

    before = config.get_runtime_config_snapshot(force_reload=True)
    before.values["api_settings"]["openai"]["api_key"] = "mutated-copy"
    assert config.save_setting_to_cli_config(
        "api_settings.openai",
        "api_key",
        "after-key",
    )
    after = config.get_runtime_config_snapshot()

    assert after.generation > before.generation
    assert after.values["api_settings"]["openai"]["api_key"] == "after-key"
    after.values["api_settings"]["openai"]["api_key"] = "mutated-again"
    current = config.get_runtime_config_snapshot()
    assert current.generation == after.generation
    assert current.values["api_settings"]["openai"]["api_key"] == "after-key"


def test_concurrent_runtime_reads_never_observe_file_cache_split(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_text('[chat_defaults]\nprovider = "before"\n', encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _reset_config_state()
    initial = config.get_runtime_config_snapshot(force_reload=True)

    def writer() -> None:
        for index in range(12):
            assert config.save_setting_to_cli_config(
                "chat_defaults",
                "provider",
                f"provider-{index}",
            )

    def reader() -> list[tuple[int, str]]:
        observed = []
        for _ in range(40):
            snapshot = config.get_runtime_config_snapshot()
            observed.append(
                (snapshot.generation, snapshot.values["chat_defaults"]["provider"])
            )
        return observed

    with ThreadPoolExecutor(max_workers=4) as executor:
        writer_future = executor.submit(writer)
        reader_futures = [executor.submit(reader) for _ in range(3)]
        writer_future.result()
        observations = [
            observation
            for future in reader_futures
            for observation in future.result()
        ]

    assert all(generation >= initial.generation for generation, _ in observations)
    assert all(
        value == "before" or value.startswith("provider-")
        for _, value in observations
    )
    assert config.get_runtime_config_snapshot().values["chat_defaults"]["provider"] == (
        "provider-11"
    )


def test_storage_default_and_console_session_contract_sources_remain_separate():
    package_root = Path(config.__file__).parent
    storage_source = (
        package_root / "UI" / "Screens" / "settings_screen.py"
    ).read_text(encoding="utf-8")
    session_source = (
        package_root / "Chat" / "console_session_settings.py"
    ).read_text(encoding="utf-8")

    assert "changes apply on next launch" in storage_source.lower()
    assert "session" in session_source.lower()
    assert "refreshable when config changes" in session_source.lower()
