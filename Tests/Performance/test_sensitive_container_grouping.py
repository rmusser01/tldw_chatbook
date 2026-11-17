"""Original container readers retain their checks in one CONFIG operation."""

from collections import Counter
import inspect
import os

import pytest

from Tests.private_profile import private_profile_test
from Tests.Performance.test_sensitive_db_path_grouping import _starts


@private_profile_test
def test_original_container_readers_share_one_retired_owner(
    request, monkeypatch, record_property
):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Utils import rag_state_paths as rag
    from tldw_chatbook.Utils import sensitive_paths as sensitive

    monkeypatch.delenv("RAG_PERSIST_DIR", raising=False)
    config.get_user_data_dir()
    readers = {
        rag.default_chroma_persist_directory.__code__: "chroma",
        rag.default_rag_profiles_dir.__code__: "profiles",
        config.get_user_data_dir.__wrapped__.__code__: "directory",
        storage._acquire_storage.__code__: "acquisition",
    }
    if os.name == "nt":
        from tldw_chatbook.Utils.windows_files import _Native

        readers[inspect.getattr_static(_Native, "open_handle").__code__] = "native"

    def measured():
        counts, owners = Counter(), []

        def entered(code, offset):
            name = readers[code]
            counts[name] += 1
            if name == "directory":
                owners.append(getattr(raw._local, "operation", None))

        with _starts(readers, entered):
            result = sensitive._direct_child_rule_container_dirs()
        assert all(owner is not None and owner not in raw._states for owner in owners)
        return result, counts, owners

    grouped, grouped_counts, grouped_owners = measured()
    monkeypatch.setattr(rag, "_SENSITIVE_CONTAINER_SOURCE", None, raising=False)
    separate, separate_counts, separate_owners = measured()
    assert grouped == separate
    assert grouped_counts["directory"] == separate_counts["directory"] == 3
    assert grouped_counts["chroma"] == separate_counts["chroma"] == 1
    assert grouped_counts["profiles"] == separate_counts["profiles"] == 1
    assert len(set(grouped_owners)) == 1
    assert len(set(separate_owners)) == 3
    assert 0 < grouped_counts["acquisition"] < separate_counts["acquisition"]
    if os.name == "nt":
        assert 0 < grouped_counts["native"] < separate_counts["native"]
    record_property("grouped_original_counts", dict(grouped_counts))
    record_property("separate_original_counts", dict(separate_counts))


@private_profile_test
def test_container_readers_keep_live_override_and_custom_failure(
    request, monkeypatch, tmp_path
):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Utils import rag_state_paths as rag
    from tldw_chatbook.Utils import sensitive_paths as sensitive

    config.get_user_data_dir()
    selected = tmp_path / "configured vectors"
    assert config.save_setting_to_cli_config(
        "AppRAGSearchConfig", "rag", {"vector_store": {"persist_directory": str(selected)}}
    )
    monkeypatch.delenv("RAG_PERSIST_DIR", raising=False)
    assert selected in sensitive._direct_child_rule_container_dirs()
    env_selected = tmp_path / "environment vectors"
    monkeypatch.setenv("RAG_PERSIST_DIR", str(env_selected))
    assert env_selected in sensitive._direct_child_rule_container_dirs()
    assert selected not in sensitive._direct_child_rule_container_dirs()
    owners = []

    def unavailable():
        owners.append(getattr(raw._local, "operation", None))
        raise ValueError("optional container unavailable")

    monkeypatch.setattr(rag, "default_rag_profiles_dir", unavailable)
    before = getattr(sensitive._failures, "count", 0)
    result = sensitive._direct_child_rule_container_dirs()
    assert env_selected in result and len(result) == 3
    assert owners == [None]
    assert getattr(sensitive._failures, "count", 0) == before + 1


@private_profile_test
def test_grouped_container_accessor_failure_stays_additive(request):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Utils import rag_state_paths as rag
    from tldw_chatbook.Utils import sensitive_paths as sensitive

    config.get_user_data_dir()
    original_error = config._CONFIG_PERSISTENCE_ERROR
    owners = []

    def unavailable(code, offset):
        owners.append(getattr(raw._local, "operation", None))
        raise ValueError("optional container unavailable")

    before = getattr(sensitive._failures, "count", 0)
    with _starts({rag.default_rag_profiles_dir.__code__}, unavailable):
        result = sensitive._direct_child_rule_container_dirs()
    assert len(result) == 3
    assert len(owners) == 1 and owners[0] is not None
    assert owners[0] not in raw._states
    assert getattr(sensitive._failures, "count", 0) == before + 1
    assert config._CONFIG_PERSISTENCE_ERROR == original_error


@private_profile_test
def test_changed_container_reader_refuses_before_memo_publication(request, monkeypatch):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Utils import rag_state_paths as rag
    from tldw_chatbook.Utils import sensitive_paths as sensitive

    config.get_user_data_dir()
    monkeypatch.setattr(sensitive, "_RAW_INPUTS_MEMO", None)
    code = rag.default_chroma_persist_directory.__code__
    called = []

    def changed():
        called.append(True)
        raise AssertionError("Changed reader must not enter grouped operation")

    def replace_later(entered, offset):
        monkeypatch.setattr(rag, "default_rag_profiles_dir", changed)

    with _starts({code}, replace_later):
        with pytest.raises(RecoveryRequired, match="sensitive_container_sources_changed"):
            sensitive._raw_inputs()
    assert not called
    assert sensitive._RAW_INPUTS_MEMO is None


@private_profile_test
def test_container_retirement_failure_cannot_return_paths(request, monkeypatch):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Utils import sensitive_paths as sensitive

    config.get_user_data_dir()
    retire, retained = raw._retire, []

    def hold_first(state):
        if state.source is config and not retained:
            retained.extend((owner, state) for owner, current in raw._states.items()
                            if current is state)
            return False
        return retire(state)

    monkeypatch.setattr(raw, "_retire", hold_first)
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="raw_resources_not_retired"):
            sensitive._direct_child_rule_container_dirs()
        assert len(retained) == 1 and retained[0][0] in raw._states
    finally:
        monkeypatch.setattr(raw, "_retire", retire)
        for owner, state in retained:
            assert retire(state)
            with storage._changed:
                del raw._states[owner]
                storage._raw_operations.discard(owner)
                storage._changed.notify_all()


@private_profile_test
def test_container_package_config_drift_refuses_before_later_reader(
    request, monkeypatch
):
    from types import SimpleNamespace

    import tldw_chatbook as package
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Utils import rag_state_paths as rag
    from tldw_chatbook.Utils import sensitive_paths as sensitive

    config.get_user_data_dir()
    monkeypatch.setattr(sensitive, "_RAW_INPUTS_MEMO", None)
    called = []

    def changed(*args, **kwargs):
        called.append(True)
        raise AssertionError("Changed package config must not receive grouped authority")

    fake = SimpleNamespace(
        get_user_data_dir=changed,
        get_cli_setting=changed,
        _get_effective_config_path=config._get_effective_config_path,
        _CONFIG_CACHE=config._CONFIG_CACHE,
        _CONFIG_GENERATION=config._CONFIG_GENERATION,
        _CONFIG_CACHE_SOURCE=config._CONFIG_CACHE_SOURCE,
    )

    def replace_package_alias(code, offset):
        # The original first RAG reader has already captured its real config
        # and original callback arguments. Drift only the later lazy import.
        monkeypatch.setattr(package, "config", fake)

    with _starts({rag._resolve_chroma_persist_directory.__code__}, replace_package_alias):
        with pytest.raises(RecoveryRequired, match="sensitive_container_sources_changed"):
            sensitive._raw_inputs()
    assert not called
    assert sensitive._RAW_INPUTS_MEMO is None
