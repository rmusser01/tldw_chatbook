"""Exact raw-file ownership for retained Advanced Config drafts."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import toml

from tldw_chatbook import config
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.Utils.config_encryption import config_encryption


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    config._invalidate_config_caches()
    config.clear_encryption_password()
    yield target
    config._invalidate_config_caches()
    config.clear_encryption_password()


@pytest.mark.parametrize("serialized", [None, "", "# comment\r\n[broken\r\n"])
def test_snapshot_reads_exact_raw_text_without_creating_or_parsing_config(
    config_path: Path, serialized: str | None
) -> None:
    if serialized is not None:
        config_path.write_bytes(serialized.encode("utf-8"))

    snapshot = SettingsConfigAdapter().read_snapshot()

    assert snapshot.path == config_path
    assert snapshot.serialized == serialized
    assert config_path.exists() is (serialized is not None)


@pytest.mark.parametrize(
    ("original", "intervening"),
    [
        ('[global]\nvalue = "old"\n', '[global]\nvalue = "new"\n'),
        ('[global]\nvalue = "old"\n', '# new comment\n[global]\nvalue = "old"\n'),
        ('[global]\nvalue = "old"\n', '[global]\r\nvalue = "old"\r\n'),
        ('[global]\nvalue = "old"\n', None),
        (None, '[global]\nvalue = "created"\n'),
        (None, ""),
    ],
)
def test_snapshot_save_rejects_intervening_file_change_without_touching_backup(
    config_path: Path, original: str | None, intervening: str | None
) -> None:
    if original is not None:
        config_path.write_bytes(original.encode("utf-8"))
    adapter = SettingsConfigAdapter()
    snapshot = adapter.read_snapshot()
    backup = config_path.with_suffix(".toml.bak")
    backup.write_text("prior-backup-SENTINEL", encoding="utf-8")
    if intervening is None:
        config_path.unlink()
    else:
        config_path.write_bytes(intervening.encode("utf-8"))

    with pytest.raises(config.ConfigSnapshotConflictError) as caught:
        adapter.replace_snapshot('[global]\nvalue = "draft-SENTINEL"\n', snapshot)

    assert isinstance(caught.value, ValueError)
    assert str(config_path) not in str(caught.value)
    assert "SENTINEL" not in str(caught.value)
    assert config_path.exists() is (intervening is not None)
    if intervening is not None:
        assert config_path.read_bytes().decode("utf-8") == intervening
    assert backup.read_text(encoding="utf-8") == "prior-backup-SENTINEL"


def test_snapshot_save_rejects_profile_change_even_when_files_match(
    config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = '[global]\nvalue = "same"\n'
    config_path.write_text(original, encoding="utf-8")
    snapshot = config.read_cli_config_snapshot()
    other_path = tmp_path / "other.toml"
    other_path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(other_path))

    with pytest.raises(config.ConfigSnapshotConflictError):
        config.replace_cli_config_snapshot('[global]\nvalue = "draft"\n', snapshot)

    assert config_path.read_text(encoding="utf-8") == original
    assert other_path.read_text(encoding="utf-8") == original
    assert not config_path.with_suffix(".toml.bak").exists()
    assert not other_path.with_suffix(".toml.bak").exists()


def test_snapshot_save_checks_file_after_acquiring_write_lock(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path.write_text('[global]\nvalue = "old"\n', encoding="utf-8")
    snapshot = config.read_cli_config_snapshot()
    real_lock = config._config_write_lock
    intervening = '[global]\nvalue = "won-lock-first"\n'

    @contextmanager
    def lock_after_other_writer(path: Path):
        config_path.write_text(intervening, encoding="utf-8")
        with real_lock(path):
            yield

    monkeypatch.setattr(config, "_config_write_lock", lock_after_other_writer)

    with pytest.raises(config.ConfigSnapshotConflictError):
        config.replace_cli_config_snapshot('[global]\nvalue = "draft"\n', snapshot)

    assert config_path.read_text(encoding="utf-8") == intervening
    assert not config_path.with_suffix(".toml.bak").exists()


def test_snapshot_save_rejects_profile_switch_while_waiting_for_write_lock(
    config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = '[global]\nvalue = "old"\n'
    config_path.write_text(original, encoding="utf-8")
    snapshot = config.read_cli_config_snapshot()
    other_path = tmp_path / "other.toml"
    other_path.write_text(original, encoding="utf-8")
    real_lock = config._config_write_lock

    @contextmanager
    def switch_profile_at_lock(path: Path):
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(other_path))
        with real_lock(path):
            yield

    monkeypatch.setattr(config, "_config_write_lock", switch_profile_at_lock)

    with pytest.raises(config.ConfigSnapshotConflictError):
        config.replace_cli_config_snapshot('[global]\nvalue = "draft"\n', snapshot)

    assert config_path.read_text(encoding="utf-8") == original
    assert other_path.read_text(encoding="utf-8") == original
    assert not config_path.with_suffix(".toml.bak").exists()
    assert not other_path.with_suffix(".toml.bak").exists()


def test_snapshot_save_cannot_bypass_conflict_check_with_missing_baseline(
    config_path: Path,
) -> None:
    original = '[global]\nvalue = "external"\n'
    config_path.write_text(original, encoding="utf-8")

    with pytest.raises(TypeError):
        config.replace_cli_config_snapshot('[global]\nvalue = "draft"\n', None)

    assert config_path.read_text(encoding="utf-8") == original
    assert not config_path.with_suffix(".toml.bak").exists()


def test_snapshot_save_returns_normalized_file_and_preserves_revision_owned_data(
    config_path: Path,
) -> None:
    original = (
        '[global]\nvalue = "old"\n'
        "[speech_studio]\nschema_version = 1\nrevision = 4\n"
        '[speech_studio.selection]\nprovider_id = "audio_cpp"\n'
    )
    config_path.write_text(original, encoding="utf-8")
    adapter = SettingsConfigAdapter()
    before = adapter.read_snapshot()
    draft = '# draft formatting\n[global]\nvalue="new"\n[speech_studio]\nrevision=1\n'

    loaded, backup, saved = adapter.replace_snapshot(draft, before)

    assert loaded["global"]["value"] == "new"
    assert backup == config_path.with_suffix(".toml.bak")
    assert backup.read_text(encoding="utf-8") == original
    assert saved.path == config_path
    assert saved.serialized == config_path.read_bytes().decode("utf-8")
    assert saved.serialized != draft
    assert toml.loads(saved.serialized)["speech_studio"] == {
        "schema_version": 1,
        "revision": 4,
        "selection": {"provider_id": "audio_cpp"},
    }
    _, _, next_snapshot = config.replace_cli_config_snapshot(
        '[global]\nvalue = "next"\n', saved, create_backup=False
    )
    assert toml.loads(next_snapshot.serialized)["global"]["value"] == "next"
    assert backup.read_text(encoding="utf-8") == original


def test_snapshot_save_from_missing_file_creates_config_without_backup(
    config_path: Path,
) -> None:
    before = config.read_cli_config_snapshot()

    loaded, backup, saved = config.replace_cli_config_snapshot(
        '[global]\nvalue = "created"\n', before
    )

    assert loaded["global"]["value"] == "created"
    assert backup is None
    assert saved.serialized == config_path.read_bytes().decode("utf-8")


def test_snapshot_is_captured_before_another_writer_can_acquire_lock(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path.write_text('[global]\nvalue = "old"\n', encoding="utf-8")
    before = config.read_cli_config_snapshot()
    real_lock = config._config_write_lock

    @contextmanager
    def overwrite_at_lock_release(path: Path):
        with real_lock(path):
            yield
        config_path.write_text('[global]\nvalue = "later"\n', encoding="utf-8")

    monkeypatch.setattr(config, "_config_write_lock", overwrite_at_lock_release)

    loaded, _, saved = config.replace_cli_config_snapshot(
        '[global]\nvalue = "submitted"\n', before
    )

    assert loaded["global"]["value"] == "submitted"
    assert toml.loads(saved.serialized)["global"]["value"] == "submitted"
    assert toml.load(config_path)["global"]["value"] == "later"


def test_snapshot_save_preserves_encryption_and_rejects_downgrade(
    config_path: Path,
) -> None:
    password = "raw-snapshot-password"
    original = config.encrypt_api_keys_in_config(
        {
            "encryption": {"enabled": True},
            "api_settings": {"openai": {"api_key": "old-secret-SENTINEL"}},
        },
        password,
    )
    config_path.write_text(toml.dumps(original), encoding="utf-8")
    config.set_encryption_password(password)
    before = config.read_cli_config_snapshot()

    loaded, backup, saved = config.replace_cli_config_snapshot(
        "[encryption]\nenabled = true\n"
        '[api_settings.openai]\napi_key = "new-secret-SENTINEL"\n',
        before,
    )

    assert loaded["api_settings"]["openai"]["api_key"] == "new-secret-SENTINEL"
    assert saved.serialized == config_path.read_bytes().decode("utf-8")
    assert "SENTINEL" not in saved.serialized
    assert backup.read_text(encoding="utf-8") == before.serialized
    assert config_encryption.is_encrypted(
        toml.loads(saved.serialized)["api_settings"]["openai"]["api_key"]
    )
    with pytest.raises(ValueError, match="disable encryption explicitly"):
        config.replace_cli_config_snapshot(
            '[api_settings.openai]\napi_key = "plaintext-SENTINEL"\n', saved
        )
    assert config_path.read_bytes().decode("utf-8") == saved.serialized
    assert backup.read_text(encoding="utf-8") == before.serialized
