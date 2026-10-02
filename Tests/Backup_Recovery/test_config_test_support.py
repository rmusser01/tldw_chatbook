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


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("nested", [False, True])
def test_newly_imported_guarded_consumer_restores_after_source_teardown(
    tmp_path, monkeypatch, nested
):
    """A real getter imported during selection must work in the next lifetime."""
    import importlib
    import sys

    from tldw_chatbook import TTS, config

    expected = config.get_user_data_dir()
    consumer_name = "tldw_chatbook.TTS.adapter_bootstrap"
    monkeypatch.delitem(sys.modules, consumer_name, raising=False)
    # Preserve an existing package binding independently of the selected scope.
    monkeypatch.setattr(
        TTS, "adapter_bootstrap", getattr(TTS, "adapter_bootstrap", None), raising=False
    )
    with pytest.MonkeyPatch.context() as selected_patch:
        first = config_test_support.select_config_source(
            selected_patch, tmp_path / "first.toml"
        )
        consumer = importlib.import_module(consumer_name)
        # Import aliases and module imports share the same owned lifetime.
        consumer._test_data_dir_alias = consumer.get_user_data_dir
        consumer._test_config_source = sys.modules[config.__name__]
        consumer._test_unrelated = object()
        unrelated = consumer._test_unrelated
        if nested:
            config_test_support.select_config_source(
                selected_patch, tmp_path / "second.toml", vars(consumer)
            )
            assert consumer._test_config_source is not first
        retired_getter = consumer.get_user_data_dir
        assert consumer.get_user_data_dir() == expected

    config_test_support.restore_config_source_consumers()

    assert consumer.get_user_data_dir() == expected
    assert consumer._test_data_dir_alias() == expected
    assert consumer._test_config_source is config
    assert consumer._test_unrelated is unrelated
    # Cleanup restores consumers; it never makes a retired wrapper admissible.
    with pytest.raises(RecoveryRequired, match="^config_source_not_installed$"):
        retired_getter()
    for name in ("_test_data_dir_alias", "_test_config_source", "_test_unrelated"):
        delattr(consumer, name)
