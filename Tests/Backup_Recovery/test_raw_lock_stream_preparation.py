"""Actual config and hook lock acquisition prepares its owned parent once."""

import inspect
import json
import os
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_raw_native_retirement as retirement_cases
from Tests.Backup_Recovery.config_test_support import install_config_source
from tldw_chatbook.Agents import hook_permissions
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Utils import private_paths

configured_source = retirement_cases.configured_source
local_root = retirement_cases.local_root


@pytest.fixture
def default_config_source(tmp_path, monkeypatch, local_root):
    # An explicit TLDW_CONFIG_PATH intentionally has no application-owned parent.
    # Use the established default-profile fixture, without replacing that policy.
    home = tmp_path / "default-home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    source = install_config_source(monkeypatch)
    target = home / ".config" / "tldw_cli" / "config.toml"
    assert source._get_effective_config_path() == target
    assert source.application_owned_config_directory(target) == target.parent
    return source


@contextmanager
def _original_lock_preparation(source, lock_path, lock_function):
    """Observe original bodies, including exception-unwound create attempts."""
    names = (
        "create_private_text",
        "open_private_text_append_stream",
        "open_private_lock_stream",
        "secure_private_directory",
    )
    functions = {
        name: getattr(private_paths, name)
        for name in names
        if hasattr(private_paths, name)
    }
    originals = {
        name: (function, inspect.unwrap(function).__code__)
        for name, function in functions.items()
    }
    helper_codes = {
        code: name
        for name, (_, code) in originals.items()
        if name != "secure_private_directory"
    }
    secure_code = originals["secure_private_directory"][1]
    lock_code = inspect.unwrap(lock_function).__code__
    native_module = sys.modules.get("tldw_chatbook.Utils.windows_files")
    native_type = getattr(native_module, "_Native", None)
    native_open = (
        inspect.getattr_static(native_type, "open_handle")
        if native_type is not None
        else None
    )
    native_code = native_open.__code__ if native_open is not None else None
    observed = SimpleNamespace(
        parent_calls=0,
        parent_callers=[],
        helper_calls=dict.fromkeys(helper_codes.values(), 0),
        helper_returns=dict.fromkeys(helper_codes.values(), 0),
        create_unwinds=0,
        native_opens=0,
        native_available=native_open is not None,
        locks=[],
    )
    active = {}
    previous = sys.getprofile()

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        code = frame.f_code
        if code is native_code and event == "call":
            observed.native_opens += 1
        elif code in helper_codes:
            name = helper_codes[code]
            if event == "call" and frame.f_locals["path"] == lock_path:
                observed.helper_calls[name] += 1
                active[id(frame)] = name
            elif event == "return" and id(frame) in active:
                observed.helper_returns[active.pop(id(frame))] += 1
                if name == "create_private_text" and result is None:
                    observed.create_unwinds += 1
        elif code is secure_code and event == "call":
            if frame.f_locals["path"] != lock_path.parent:
                return
            parent = frame.f_back
            while parent is not None and parent.f_code is not lock_code:
                name = helper_codes.get(parent.f_code)
                if name is not None and parent.f_locals["path"] == lock_path:
                    observed.parent_calls += 1
                    observed.parent_callers.append(name)
                    break
                parent = parent.f_back
        elif code is lock_code and event == "return":
            stream = frame.f_locals.get("stream")
            # Profile emits return at a generator yield as well as its end.
            # The actual open stream distinguishes the held-lock boundary.
            if stream is None or stream.closed:
                return
            operation = getattr(raw._local, "operation", None)
            state = raw._states.get(operation)
            assert state is not None and state.source is source
            assert state.active and not state.uncertain
            assert state.pins and state.leases and state.files
            assert stream.fileno() in state.descriptors
            assert all(lease in storage._live_leases for lease in state.leases)
            if "locked" in frame.f_locals:
                assert frame.f_locals["locked"] is True
            observed.locks.append((operation, state, tuple(state.leases), stream))

    sys.setprofile(observe)
    try:
        yield observed
    finally:
        sys.setprofile(previous)
        assert not active
        for name, (function, code) in originals.items():
            assert getattr(private_paths, name) is function
            assert inspect.unwrap(function).__code__ is code
        assert inspect.unwrap(lock_function).__code__ is lock_code
        if native_open is not None:
            assert inspect.getattr_static(native_type, "open_handle") is native_open
            assert native_open.__code__ is native_code


@pytest.mark.parametrize("domain", ["config", "hook"])
@pytest.mark.parametrize("lock_state", ["cold", "warm"])
def test_real_lock_prepares_application_parent_once(
    domain, lock_state, request, monkeypatch
):
    config = request.getfixturevalue(
        "default_config_source" if domain == "config" else "configured_source"
    )
    if domain == "config":
        source = config
        selected = config._get_effective_config_path()
        lock_function = config._config_interprocess_lock

        def read():
            with config.locked_hooks_config_snapshot() as snapshot:
                assert snapshot.config_path == selected
                return snapshot

    else:
        monkeypatch.setattr(hook_permissions, "config", config)
        source = hook_permissions.HookPermissions()
        selected = config.get_user_data_dir() / "hook_permissions.json"
        lock_function = source._store_lock
        read = source.snapshot
    lock_path = selected.with_name(selected.name + ".lock")
    config_before = config._get_effective_config_path().read_bytes()
    try:
        if lock_state == "warm":
            read()
            assert lock_path.is_file()
            before = selected.read_bytes()
        else:
            assert not lock_path.exists(), "cold means the real lock file is absent"
            before = selected.read_bytes() if selected.exists() else None
        metadata_probe = None
        if (
            os.environ.get("TLDW_RAW_PARENT_METADATA_PROBE") == "1"
            and domain == "hook"
            and lock_state == "warm"
        ):
            from Tests.Performance.raw_parent_metadata_probe import (
                RawParentMetadataProbe,
            )

            metadata_probe = RawParentMetadataProbe()
        try:
            with retirement_cases._original_tracked_closes(source) as retirement:
                with _original_lock_preparation(
                    source, lock_path, lock_function
                ) as observed:
                    with metadata_probe if metadata_probe is not None else nullcontext():
                        output = read()
        finally:
            if metadata_probe is not None:
                metadata_receipt = metadata_probe.receipt()
                request.node.user_properties.append(
                    ("raw_parent_metadata_probe", json.dumps(metadata_receipt))
                )
        if domain == "hook":
            assert output.ready and output.rows == () and output.store_path == selected
            assert selected.is_file()
        else:
            assert output.config_path == selected
        if before is not None:
            assert selected.read_bytes() == before
        assert config._get_effective_config_path().read_bytes() == config_before
        assert lock_path.read_bytes() == b""
        assert len(observed.locks) == 1
        operation, state, leases, stream = observed.locks[0]
        assert stream.closed
        assert not state.active and not state.uncertain
        assert not state.pins and not state.files and not state.descriptors
        assert operation not in raw._states and operation not in storage._raw_operations
        assert all(lease not in storage._live_leases for lease in leases)
        counts, closed, removed, active, operations = retirement
        assert counts["closes"] > 0 and counts["parent_walk_closes"] > 0
        assert not active and len(closed) == counts["closes"] and all(closed)
        assert all(removed) and operations
        assert all(operation not in raw._states for operation in operations)
        assert all(operation not in storage._raw_operations for operation in operations)
        assert observed.helper_returns == observed.helper_calls
        stream_calls = sum(
            count
            for name, count in observed.helper_calls.items()
            if name != "create_private_text"
        )
        assert stream_calls >= 1
        if lock_state == "warm":
            assert (
                observed.create_unwinds == observed.helper_calls["create_private_text"]
            )
        request.node.user_properties.extend(
            [
                ("lock_domain", domain),
                ("lock_state", lock_state),
                ("original_parent_preparations", observed.parent_calls),
                ("original_parent_callers", ",".join(observed.parent_callers)),
                ("original_create_unwinds", observed.create_unwinds),
                ("original_native_attempts_available", observed.native_available),
                ("original_whole_read_native_attempts", observed.native_opens),
                ("original_descriptor_closes", counts["closes"]),
            ]
        )
        if metadata_probe is not None:
            assert metadata_receipt["selected_defining_bodies_and_source_files_current"]
            assert metadata_receipt["monitoring_retired"]
            assert metadata_receipt["global_event_mask_after_read"] == 0
            assert metadata_receipt["overflow"] == metadata_receipt["unmatched"] == 0
            assert (
                metadata_receipt["exit_order_mismatches"]
                == metadata_receipt["unfinished"]
                == 0
            )
            rows = metadata_receipt["rows"]
            assert rows["parent_check"]["starts"] > 0
            assert all(
                row["starts"] == row["returns"] + row["unwinds"]
                for row in rows.values()
            )
            assert (
                sum(
                    row["starts"]
                    for key, row in rows.items()
                    if key.endswith(".open_handle")
                )
                == observed.native_opens
            )
        # All payload, lock, FD and lease assertions precede this causal RED.
        assert observed.parent_calls == 1, observed.parent_callers
    finally:
        if domain == "hook":
            source.close()
