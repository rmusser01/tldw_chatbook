"""Test profile selection must create a new source, not weaken an old one."""

import pytest

from Tests.Backup_Recovery import config_test_support
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired


def test_selected_config_uses_real_custody_and_refuses_later_retarget(
    tmp_path, monkeypatch
):
    select = getattr(config_test_support, "select_config_source", None)
    assert callable(select), "explicit test profile selection helper is missing"
    from tldw_chatbook import config

    namespace = {"config": config, "read": config.get_cli_setting}
    target = tmp_path / "selected.toml"
    target.write_text('[general]\nusers_name = "Selected"\n', encoding="utf-8")
    source = select(monkeypatch, target, namespace)
    assert namespace["read"]("general", "users_name") == "Selected"
    assert namespace["config"] is source

    other = tmp_path / "other.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(other))
    with pytest.raises(RecoveryRequired, match="raw_source_selection_changed"):
        source.load_cli_config_and_ensure_existence(force_reload=True)
    assert not other.exists()
