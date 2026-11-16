"""A sensitive DB-path set owns one finite, checked configuration read."""

import sys
import threading
from contextlib import contextmanager

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.timeout(120)
@private_profile_test
def test_sensitive_database_paths_share_one_native_config_read(request):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import raw_participants, storage_admission
    from tldw_chatbook.Utils import sensitive_paths

    config.get_user_data_dir()
    expected = tuple(
        getattr(config, name)() for name in sensitive_paths._DB_PATH_ACCESSOR_NAMES
    )
    previous = getattr(raw_participants._local, "operation", None)
    thread = threading.get_ident()
    starts = []
    code = storage_admission._acquire_storage.__code__
    tool = next(index for index in range(6) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "sensitive-config-admission-count")

    def started(_code, _offset):
        if threading.get_ident() == thread:
            starts.append(None)

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        actual = sensitive_paths._sensitive_db_paths()
    finally:
        sys.monitoring.set_local_events(tool, code, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, None)
        sys.monitoring.free_tool_id(tool)

    assert actual == expected
    assert getattr(raw_participants._local, "operation", None) is previous
    # A config operation may separately admit its selected companion directory.
    assert 1 <= len(starts) <= 2, (
        f"DB-path assembly acquired storage {len(starts)} times"
    )


@pytest.mark.timeout(120)
@private_profile_test
def test_custom_database_read_failure_rolls_back_and_prevents_memo(
    request, monkeypatch, tmp_path
):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import config_participants, raw_participants
    from tldw_chatbook.Utils import sensitive_paths

    config.get_user_data_dir()
    before = config._CONFIG_CACHE
    previous = getattr(raw_participants._local, "operation", None)
    custom = tmp_path / "custom.db"
    observed = []

    def custom_getter():
        active = getattr(raw_participants._local, "operation", None)
        observed.append(
            active is not None and raw_participants.check(active).source is config
        )
        return custom

    def failed_getter():
        with config_participants.operation(config):
            config._CONFIG_CACHE = {**before, "failed_read_marker": True}
            raise ValueError("incomplete database path read")

    monkeypatch.setattr(sensitive_paths, "_RAW_INPUTS_MEMO", None)
    monkeypatch.setattr(
        sensitive_paths,
        "_DB_PATH_ACCESSOR_NAMES",
        ("get_prompts_db_path", "get_media_db_path"),
    )
    monkeypatch.setattr(config, "get_prompts_db_path", custom_getter)
    monkeypatch.setattr(config, "get_media_db_path", failed_getter)
    failures = getattr(sensitive_paths._failures, "count", 0)
    inputs = sensitive_paths._raw_inputs()

    assert inputs[3] == (custom,)
    assert observed == [True]
    assert config._CONFIG_CACHE is before
    assert sensitive_paths._failures.count > failures
    assert sensitive_paths._RAW_INPUTS_MEMO is None
    assert getattr(raw_participants._local, "operation", None) is previous


@pytest.mark.timeout(120)
@pytest.mark.parametrize("boundary", ["admission", "retirement"])
@private_profile_test
def test_database_path_scope_failure_cannot_publish_a_partial_memo(
    request, monkeypatch, boundary
):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import (
        bootstrap,
        config_participants,
        raw_participants,
    )
    from tldw_chatbook.Utils import sensitive_paths

    config.get_user_data_dir()
    previous = getattr(raw_participants._local, "operation", None)
    original = config_participants.operation
    depth = 0

    @contextmanager
    def operation(source, **kwargs):
        nonlocal depth
        outer = depth == 0
        depth += 1
        try:
            if outer and boundary == "admission":
                raise bootstrap.RecoveryRequired("test_path_read_admission")
            with original(source, **kwargs) as active:
                yield active
            if outer:
                raise bootstrap.RecoveryRequired("test_path_read_retirement")
        finally:
            depth -= 1

    # Keep the input key stable to isolate failure of the DB assembly boundary.
    key = sensitive_paths._raw_inputs_key()
    monkeypatch.setattr(sensitive_paths, "_raw_inputs_key", lambda: key)
    monkeypatch.setattr(sensitive_paths, "_RAW_INPUTS_MEMO", None)
    monkeypatch.setattr(config_participants, "operation", operation)
    with pytest.raises(bootstrap.RecoveryRequired, match="test_path_read_" + boundary):
        sensitive_paths._raw_inputs()
    assert sensitive_paths._RAW_INPUTS_MEMO is None
    assert getattr(raw_participants._local, "operation", None) is previous
    assert depth == 0


@pytest.mark.timeout(120)
@private_profile_test
def test_cold_sensitive_input_assembly_shares_one_native_config_read(
    request, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import raw_participants, storage_admission
    from tldw_chatbook.Utils import sensitive_paths

    expected = (
        sensitive_paths._raw_inputs_key()[0],
        sensitive_paths._sensitive_single_file_paths(),
        sensitive_paths._sensitive_skill_trust_dir(),
        sensitive_paths._sensitive_db_paths(),
        sensitive_paths._direct_child_rule_container_dirs(),
    )
    monkeypatch.setattr(sensitive_paths, "_RAW_INPUTS_MEMO", None)
    previous = getattr(raw_participants._local, "operation", None)
    thread = threading.get_ident()
    starts = []
    code = storage_admission._acquire_storage.__code__
    tool = next(index for index in range(6) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "sensitive-input-admission-count")

    def started(_code, _offset):
        if threading.get_ident() == thread:
            starts.append(None)

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        actual = sensitive_paths._raw_inputs()
    finally:
        sys.monitoring.set_local_events(tool, code, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, None)
        sys.monitoring.free_tool_id(tool)

    assert actual == expected
    assert getattr(raw_participants._local, "operation", None) is previous
    memo = sensitive_paths._RAW_INPUTS_MEMO
    assert memo is not None and memo[1] == actual
    assert sensitive_paths._same_key(memo[0], sensitive_paths._raw_inputs_key()[2])
    # Fresh before/after keys retain their separate admission; only the cold
    # synchronous assembly shares a scope and its optional companion admission.
    assert 1 <= len(starts) <= 6, (
        f"cold input assembly acquired storage {len(starts)} times"
    )


@pytest.mark.timeout(120)
@private_profile_test
def test_sensitive_input_environment_change_does_not_publish_a_memo(
    request, monkeypatch, tmp_path
):
    from tldw_chatbook.Utils import sensitive_paths

    original = sensitive_paths._sensitive_skill_trust_dir
    changed = tmp_path / "changed-chroma"
    monkeypatch.setattr(sensitive_paths, "_RAW_INPUTS_MEMO", None)
    monkeypatch.delenv("RAG_PERSIST_DIR", raising=False)

    def change_environment():
        result = original()
        monkeypatch.setenv("RAG_PERSIST_DIR", str(changed))
        return result

    monkeypatch.setattr(
        sensitive_paths, "_sensitive_skill_trust_dir", change_environment
    )
    result = sensitive_paths._raw_inputs()
    assert changed in result[4]
    assert sensitive_paths._RAW_INPUTS_MEMO is None
