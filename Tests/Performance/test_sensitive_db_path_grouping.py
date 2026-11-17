"""Original DB denylist readers share a finite, fully retired CONFIG owner."""

from collections import Counter
from contextlib import contextmanager
import inspect
import os
import sys

import pytest

from Tests.private_profile import private_profile_test


@contextmanager
def _starts(codes, callback):
    """Observe exact original code entries without replacing any callable."""
    monitor = sys.monitoring
    tool = next((index for index in range(5, -1, -1)
                 if monitor.get_tool(index) is None), None)
    assert tool is not None, "finite original-code counter needs a free monitoring ID"
    monitor.use_tool_id(tool, "sensitive-db-cohort-test")
    try:
        monitor.register_callback(tool, monitor.events.PY_START, callback)
        for code in codes:
            monitor.set_local_events(tool, code, monitor.events.PY_START)
        yield
    finally:
        for code in codes:
            monitor.set_local_events(tool, code, 0)
        monitor.register_callback(tool, monitor.events.PY_START, None)
        monitor.free_tool_id(tool)


@private_profile_test
def test_original_db_getters_share_one_retired_owner(request, monkeypatch, record_property):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Utils import sensitive_paths as sensitive

    config.get_user_data_dir()
    getters = {getattr(config, name).__code__: name
               for name in sensitive._DB_PATH_ACCESSOR_NAMES}
    directory = config.get_user_data_dir.__wrapped__.__code__
    acquisition = storage._acquire_storage.__code__
    codes = {*getters, directory, acquisition}
    native = None
    if os.name == "nt":
        from tldw_chatbook.Utils.windows_files import _Native
        native = inspect.getattr_static(_Native, "open_handle").__code__
        codes.add(native)

    def measured():
        counts, owners = Counter(), []

        def entered(code, offset):
            if code in getters:
                counts[getters[code]] += 1
            elif code is directory:
                counts["directory"] += 1
                owners.append(getattr(raw._local, "operation", None))
            elif code is acquisition:
                counts["acquisition"] += 1
            elif code is native:
                counts["native"] += 1

        with _starts(codes, entered):
            paths = sensitive._sensitive_db_paths()
        assert all(owner is not None and owner not in raw._states for owner in owners)
        return paths, counts, owners

    grouped, grouped_counts, grouped_owners = measured()
    # Only opt out of the new grouping; every original callable remains installed.
    monkeypatch.setattr(config, "_SENSITIVE_DB_PATH_SOURCE", None, raising=False)
    separate, separate_counts, separate_owners = measured()
    assert grouped == separate
    for name in sensitive._DB_PATH_ACCESSOR_NAMES:
        assert grouped_counts[name] == separate_counts[name] == 1
    assert grouped_counts["directory"] == separate_counts["directory"] == 13
    assert len(set(grouped_owners)) == 1
    assert len(set(separate_owners)) == 13
    assert 0 < grouped_counts["acquisition"] < separate_counts["acquisition"]
    if native is not None:
        assert 0 < grouped_counts["native"] < separate_counts["native"]
    record_property("grouped_original_counts", dict(grouped_counts))
    record_property("separate_original_counts", dict(separate_counts))


@private_profile_test
def test_custom_db_getter_keeps_ungrouped_additive_failure(request, monkeypatch, tmp_path):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Utils import sensitive_paths as sensitive

    config.get_user_data_dir()
    original_error = config._CONFIG_PERSISTENCE_ERROR
    selected, owners = tmp_path / "custom.sqlite", []

    def custom():
        owners.append(getattr(raw._local, "operation", None))
        return selected

    def unavailable():
        raise ValueError("optional accessor unavailable")

    monkeypatch.setattr(config, "get_chachanotes_db_path", custom)
    monkeypatch.setattr(config, "get_prompts_db_path", unavailable)
    before = getattr(sensitive._failures, "count", 0)
    result = sensitive._sensitive_db_paths()
    assert selected in result and len(result) == 12
    assert owners == [None]
    assert getattr(sensitive._failures, "count", 0) == before + 1
    assert config._CONFIG_PERSISTENCE_ERROR == original_error


@private_profile_test
def test_grouped_accessor_failure_does_not_fail_config_operation(request):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Utils import sensitive_paths as sensitive

    config.get_user_data_dir()
    original_error = config._CONFIG_PERSISTENCE_ERROR
    code = config.get_prompts_db_path.__code__
    owners = []

    def unavailable(entered, offset):
        owners.append(getattr(raw._local, "operation", None))
        raise ValueError("optional accessor unavailable")

    with _starts({code}, unavailable):
        result = sensitive._sensitive_db_paths()
    assert len(result) == 12
    assert len(owners) == 1 and owners[0] is not None
    assert owners[0] not in raw._states
    assert config._CONFIG_PERSISTENCE_ERROR == original_error


@private_profile_test
def test_changed_later_getter_never_runs_under_group_owner(request, monkeypatch, tmp_path):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Utils import sensitive_paths as sensitive

    config.get_user_data_dir()
    monkeypatch.setattr(sensitive, "_RAW_INPUTS_MEMO", None)
    code = config.get_chachanotes_db_path.__code__
    called = []

    def changed():
        called.append(True)
        return tmp_path / "changed.sqlite"

    def replace_later(entered, offset):
        monkeypatch.setattr(config, "get_prompts_db_path", changed)

    with _starts({code}, replace_later):
        with pytest.raises(bootstrap.RecoveryRequired, match="sensitive_db_sources_changed"):
            sensitive._raw_inputs()
    assert not called
    assert sensitive._RAW_INPUTS_MEMO is None


@private_profile_test
def test_original_getter_keeps_configured_override_and_sidecars(request, tmp_path):
    from tldw_chatbook import config
    from tldw_chatbook.Utils import sensitive_paths as sensitive

    selected = tmp_path / "selected.sqlite"
    assert config.save_setting_to_cli_config("database", "chachanotes_db_path", str(selected))
    assert selected in sensitive._sensitive_db_paths()
    context = sensitive.resolve_sensitive_context()
    for path in (selected, *sensitive._db_sidecar_paths(selected)):
        assert sensitive.is_sensitive_path(path, context)


@private_profile_test
def test_group_admission_failure_cannot_publish_partial_custom_denylist(
    request, monkeypatch, tmp_path
):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Utils import sensitive_paths as sensitive

    selected = tmp_path / "selected.sqlite"
    assert config.save_setting_to_cli_config("database", "chachanotes_db_path", str(selected))
    assert config.get_chachanotes_db_path() == selected
    monkeypatch.setattr(sensitive, "_RAW_INPUTS_MEMO", None)

    def refuse(*args, **kwargs):
        raise bootstrap.RecoveryRequired("sensitive_test_admission_refused")

    monkeypatch.setattr(storage, "_acquire_storage", refuse)
    with pytest.raises(bootstrap.RecoveryRequired, match="sensitive_test_admission_refused"):
        sensitive._raw_inputs()
    assert sensitive._RAW_INPUTS_MEMO is None


@private_profile_test
def test_group_retirement_failure_cannot_return_collected_db_paths(request, monkeypatch):
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
            sensitive._sensitive_db_paths()
        assert len(retained) == 1
        assert retained[0][0] in raw._states
    finally:
        monkeypatch.setattr(raw, "_retire", retire)
        for owner, state in retained:
            # Release actual held resources through their original owner before
            # matching the registry removal performed by raw.scope itself.
            assert retire(state)
            with storage._changed:
                del raw._states[owner]
                storage._raw_operations.discard(owner)
                storage._changed.notify_all()
