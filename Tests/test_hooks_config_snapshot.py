"""Hook writes compare the current section and preserve unrelated raw data."""

import sys

import toml

from Tests.Backup_Recovery.config_test_support import install_config_source
from tldw_chatbook import config


def test_hooks_save_rejects_stale_section_without_replacing_file(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps({"hooks": {"enabled": True, "hook": []}}))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    monkeypatch.setattr(
        sys.modules[__name__], "config", install_config_source(monkeypatch)
    )
    original = config.read_hooks_config_snapshot()
    path.write_text(toml.dumps({"hooks": {"enabled": False, "hook": []}}))
    before = path.read_bytes()
    result = config.replace_hooks_config_snapshot(
        original, {"enabled": True, "hook": []}
    )
    assert not result.file_replaced
    assert path.read_bytes() == before


def test_hooks_save_preserves_unknown_keys_and_concurrent_unrelated_edit(
    tmp_path, monkeypatch
):
    path = tmp_path / "config.toml"
    path.write_text(
        toml.dumps(
            {
                "hooks": {
                    "custom": "retained",
                    "hook": [{"event": "Stop", "command": ["python3"], "custom": 4}],
                },
                "other": {"value": 1},
            }
        )
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    monkeypatch.setattr(
        sys.modules[__name__], "config", install_config_source(monkeypatch)
    )
    original = config.read_hooks_config_snapshot()
    raw = toml.loads(path.read_text())
    raw["other"]["value"] = 2
    path.write_text(toml.dumps(raw))
    replacement = {**original.section, "enabled": False}
    result = config.replace_hooks_config_snapshot(original, replacement)
    assert result.file_replaced and result.caches_reloaded
    saved = toml.loads(path.read_text())
    assert saved["other"]["value"] == 2
    assert saved["hooks"]["custom"] == "retained"
    assert saved["hooks"]["hook"][0]["custom"] == 4


def test_hooks_snapshot_is_detached_and_scope_change_rejects_save(
    tmp_path, monkeypatch
):
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    text = toml.dumps({"hooks": {"hook": [{"event": "Stop", "command": ["python3"]}]}})
    first.write_text(text)
    second.write_text(text)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(first))
    monkeypatch.setattr(
        sys.modules[__name__], "config", install_config_source(monkeypatch)
    )
    original = config.read_hooks_config_snapshot()
    original.section["hook"][0]["command"].append("new.py")
    assert "new.py" not in first.read_text()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(second))
    result = config.replace_hooks_config_snapshot(original, original.section)
    assert not result.file_replaced
    assert second.read_text() == text


def test_nan_original_can_be_compared_and_disabled(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(
        '[hooks]\n[[hooks.hook]]\nevent="Stop"\ncommand=["python3"]\ntimeout_s=nan\n'
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    monkeypatch.setattr(
        sys.modules[__name__], "config", install_config_source(monkeypatch)
    )
    original = config.read_hooks_config_snapshot()
    result = config.replace_hooks_config_snapshot(
        original, {**original.section, "enabled": False}
    )
    assert result.file_replaced


def test_hooks_save_reports_replacement_when_publication_fails(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text("[hooks]\nenabled=true\n")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    monkeypatch.setattr(
        sys.modules[__name__], "config", install_config_source(monkeypatch)
    )
    original = config.read_hooks_config_snapshot()

    def fail_publication(*args, **kwargs):
        raise RuntimeError("test refresh failure")

    monkeypatch.setattr(config, "_publish_runtime_config_unlocked", fail_publication)
    result = config.replace_hooks_config_snapshot(original, {"enabled": False})
    assert result.file_replaced and not result.caches_reloaded
    assert toml.loads(path.read_text())["hooks"]["enabled"] is False


def test_save_compares_writer_file_even_if_effective_path_changes_during_lock(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    text = "[hooks]\nenabled=true\n"
    first.write_text(text)
    second.write_text(text)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(second))
    monkeypatch.setattr(
        sys.modules[__name__], "config", install_config_source(monkeypatch)
    )
    original = config.read_hooks_config_snapshot()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(first))
    lock = config._config_write_lock

    @contextmanager
    def retarget(path):
        with lock(path):
            monkeypatch.setenv("TLDW_CONFIG_PATH", str(second))
            yield

    monkeypatch.setattr(config, "_config_write_lock", retarget)
    result = config.replace_hooks_config_snapshot(original, {"enabled": False})
    assert not result.file_replaced
    assert first.read_text() == text
    assert second.read_text() == text


def test_guided_replace_cannot_discard_unknown_hook_section_keys(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[hooks]\nenabled=true\nfuture_option="retain"\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    monkeypatch.setattr(
        sys.modules[__name__], "config", install_config_source(monkeypatch)
    )
    original = config.read_hooks_config_snapshot()
    result = config.replace_hooks_config_snapshot(
        original, {"enabled": False, "hook": []}
    )
    assert result.file_replaced
    assert toml.loads(path.read_text())["hooks"]["future_option"] == "retain"


def test_absent_hooks_section_is_readable_and_can_be_added(tmp_path, monkeypatch):
    path = tmp_path / "Q3 P&L; config.toml"
    path.write_text("[other]\nvalue=1\n")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    monkeypatch.setattr(
        sys.modules[__name__], "config", install_config_source(monkeypatch)
    )
    absent = config.read_hooks_config_snapshot()
    assert not absent.section_present and absent.section is None
    result = config.replace_hooks_config_snapshot(absent, {"enabled": False})
    assert result.file_replaced and result.caches_reloaded
    current = config.read_hooks_config_snapshot()
    assert current.section_present and current.section == {"enabled": False}
    assert current.section_stamp != absent.section_stamp
    assert toml.loads(path.read_text())["other"]["value"] == 1


def test_hooks_save_rejects_profile_change_inside_the_same_config(
    tmp_path, monkeypatch
):
    path = tmp_path / "config.toml"
    raw = {"general": {"users_name": "first"}, "hooks": {"enabled": False}}
    path.write_text(toml.dumps(raw))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    monkeypatch.setattr(
        sys.modules[__name__], "config", install_config_source(monkeypatch)
    )
    original = config.read_hooks_config_snapshot()
    raw["general"]["users_name"] = "second"
    path.write_text(toml.dumps(raw))
    before = path.read_bytes()
    result = config.replace_hooks_config_snapshot(original, {"enabled": True})
    assert not result.file_replaced
    assert path.read_bytes() == before
