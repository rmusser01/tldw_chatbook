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


def test_selection_releases_only_the_startup_admission_it_added(tmp_path):
    """TASK-39059: a fresh-HOME selection's config import files a startup
    admission for its own bootstrap root, and nothing released it before
    process exit, so an in-process RuntimeMaintenance backup after such tests
    refused (``runtime_startup_scope_unqualified``). A nested selection on
    the same root adds none, so it releases none."""
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    for name in ("outer.toml", "inner.toml"):
        (tmp_path / name).write_text('[general]\nusers_name = "Selected"\n')
    before = dict(storage._startups)
    with pytest.MonkeyPatch.context() as outer:
        config_test_support.select_config_source(outer, tmp_path / "outer.toml")
        added = dict(storage._startups)
        with pytest.MonkeyPatch.context() as inner:
            config_test_support.select_config_source(inner, tmp_path / "inner.toml")
        after_inner = dict(storage._startups)
    config_test_support.restore_config_source_consumers()

    # (admissions added, the inner undo kept the outer's, all released)
    assert (len(added.keys() - before.keys()), after_inner == added) == (1, True)
    assert storage._startups == before


@pytest.mark.bootstrap_profile
def test_selection_in_the_bootstrap_profile_keeps_the_process_admission(tmp_path):
    """TASK-39059: the process's own startup admission was not added by the
    selection, so undoing the selection must leave it in place."""
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    before = dict(storage._startups)
    assert before
    with pytest.MonkeyPatch.context() as selected:
        config_test_support.select_config_source(selected, tmp_path / "selected.toml")
        during = dict(storage._startups)
    config_test_support.restore_config_source_consumers()

    assert (during, storage._startups) == (before, before)


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


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("nested", [False, True])
def test_newly_imported_consumer_still_catches_config_errors_after_teardown(
    tmp_path, monkeypatch, nested
):
    """Restore real error-catching consumers with their config functions."""
    import importlib
    import sys

    from tldw_chatbook import Chat, config

    consumer_name = "tldw_chatbook.Chat.provider_readiness"
    monkeypatch.delitem(sys.modules, consumer_name, raising=False)
    monkeypatch.setattr(
        Chat,
        "provider_readiness",
        getattr(Chat, "provider_readiness", None),
        raising=False,
    )
    malformed = {"api_settings": {"moonshot": "not a table"}}
    with pytest.MonkeyPatch.context() as selected_patch:
        config_test_support.select_config_source(
            selected_patch, tmp_path / "first.toml"
        )
        consumer = importlib.import_module(consumer_name)
        consumer._test_error_alias = consumer.ProviderSettingsError
        consumer._test_library_class = ValueError
        if nested:
            config_test_support.select_config_source(
                selected_patch, tmp_path / "second.toml", vars(consumer)
            )
        before = consumer.get_provider_readiness("Moonshot", malformed, environ={})
        assert not before.ready and before.reason == "Invalid provider settings"

    config_test_support.restore_config_source_consumers()

    try:
        after = consumer.get_provider_readiness("Moonshot", malformed, environ={})
        assert after == before
        assert consumer.ProviderSettingsError is config.ProviderSettingsError
        assert consumer._test_error_alias is config.ProviderSettingsError
        assert consumer._test_library_class is ValueError
    finally:
        del consumer._test_error_alias
        del consumer._test_library_class
