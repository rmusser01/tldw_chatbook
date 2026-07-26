from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import tldw_chatbook.Utils.private_paths as private_paths
import tldw_chatbook.runtime_policy.source_state as source_state
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathStatus,
)
from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore
from tldw_chatbook.runtime_policy.types import RuntimeSourceState


def test_missing_runtime_policy_returns_safe_default(tmp_path: Path) -> None:
    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")

    assert store.load() == RuntimeSourceState()


@pytest.mark.skipif(os.name != "posix", reason="POSIX link contract")
def test_runtime_policy_store_rejects_symlink_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text('{"active_source": "server"}', encoding="utf-8")
    outside.chmod(0o644)
    selected = tmp_path / "runtime_policy.json"
    selected.symlink_to(outside)
    parse_called = False

    def fail_if_parsed(*args, **kwargs):
        nonlocal parse_called
        parse_called = True
        raise AssertionError("unsafe target must not be parsed")

    monkeypatch.setattr(source_state.json, "load", fail_if_parsed)

    with pytest.raises(PrivatePathError):
        RuntimeSourceStateStore(selected).load()

    assert parse_called is False
    assert stat.S_IMODE(outside.stat().st_mode) == 0o644


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
@pytest.mark.timeout(2, method="signal")
def test_runtime_policy_store_rejects_fifo_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "runtime_policy.json"
    os.mkfifo(selected, mode=0o644)
    parse_called = False

    def fail_if_parsed(*args, **kwargs):
        nonlocal parse_called
        parse_called = True
        raise AssertionError("non-regular target must not be parsed")

    monkeypatch.setattr(source_state.json, "load", fail_if_parsed)

    with pytest.raises(PrivatePathError):
        RuntimeSourceStateStore(selected).load()

    assert parse_called is False


@pytest.mark.skipif(os.name != "posix", reason="POSIX identity contract")
def test_runtime_policy_store_rejects_replaced_target_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "runtime_policy.json"
    selected.write_text('{"active_source": "server"}', encoding="utf-8")
    parse_called = False

    def fail_if_parsed(*args, **kwargs):
        nonlocal parse_called
        parse_called = True
        raise AssertionError("replaced target must not be parsed")

    monkeypatch.setattr(source_state.json, "load", fail_if_parsed)
    monkeypatch.setattr(
        private_paths,
        "_private_file_postcondition_holds",
        lambda *args, **kwargs: False,
    )

    with pytest.raises(PrivatePathError) as caught:
        RuntimeSourceStateStore(selected).load()

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert parse_called is False


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_runtime_policy_store_hardens_existing_file_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "runtime_policy.json"
    selected.write_text('{"active_source": "local"}', encoding="utf-8")
    selected.chmod(0o644)
    observed_modes: list[int] = []
    real_load = source_state.json.load

    def observe_mode(handle):
        observed_modes.append(stat.S_IMODE(os.fstat(handle.fileno()).st_mode))
        return real_load(handle)

    monkeypatch.setattr(source_state.json, "load", observe_mode)

    restored = RuntimeSourceStateStore(selected).load()

    assert restored.active_source == "local"
    assert observed_modes == [0o600]
    assert stat.S_IMODE(selected.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_malformed_runtime_policy_defaults_only_after_verified_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "runtime_policy.json"
    selected.write_text("{not-json", encoding="utf-8")
    selected.chmod(0o644)
    verified_modes: list[int] = []
    real_load = source_state.json.load

    def observe_verified_open(handle):
        verified_modes.append(stat.S_IMODE(os.fstat(handle.fileno()).st_mode))
        return real_load(handle)

    monkeypatch.setattr(source_state.json, "load", observe_verified_open)

    restored = RuntimeSourceStateStore(selected).load()

    assert restored == RuntimeSourceState()
    assert verified_modes == [0o600]


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory mode contract")
def test_application_owned_runtime_policy_parent_is_created_private(
    tmp_path: Path,
) -> None:
    owned_parent = tmp_path / "application-config"
    store = RuntimeSourceStateStore(
        owned_parent / "runtime_policy.json",
        application_owned_directory=owned_parent,
    )

    store.save(RuntimeSourceState())

    assert stat.S_IMODE(owned_parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory mode contract")
def test_application_owned_runtime_policy_parent_is_hardened(
    tmp_path: Path,
) -> None:
    owned_parent = tmp_path / "application-config"
    owned_parent.mkdir(mode=0o755)
    store = RuntimeSourceStateStore(
        owned_parent / "runtime_policy.json",
        application_owned_directory=owned_parent,
    )

    store.save(RuntimeSourceState())

    assert stat.S_IMODE(owned_parent.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
def test_custom_runtime_policy_parent_is_never_created(tmp_path: Path) -> None:
    custom_parent = tmp_path / "custom"
    store = RuntimeSourceStateStore(custom_parent / "runtime_policy.json")

    with pytest.raises(PrivatePathError):
        store.save(RuntimeSourceState())

    assert not custom_parent.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory mode contract")
def test_custom_runtime_policy_parent_is_never_chmodded(tmp_path: Path) -> None:
    custom_parent = tmp_path / "custom"
    custom_parent.mkdir(mode=0o751)
    store = RuntimeSourceStateStore(custom_parent / "runtime_policy.json")

    store.save(RuntimeSourceState())

    assert stat.S_IMODE(custom_parent.stat().st_mode) == 0o751


def test_runtime_policy_save_does_not_use_predictable_tmp_name(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "runtime_policy.json"
    predictable = selected.with_suffix(".json.tmp")
    predictable.write_text("do-not-touch", encoding="utf-8")

    RuntimeSourceStateStore(selected).save(RuntimeSourceState())

    assert predictable.read_text(encoding="utf-8") == "do-not-touch"


def test_runtime_policy_store_reports_windows_posture_as_unverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "runtime_policy.json"
    selected.write_text('{"active_source": "local"}', encoding="utf-8")
    monkeypatch.setattr(private_paths, "_posix_guards_available", lambda: False)
    monkeypatch.setattr(private_paths, "_atomic_posix_guards_available", lambda: False)
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", True)
    messages: list[str] = []
    sink = source_state.logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        store = RuntimeSourceStateStore(selected)
        assert store.load() == RuntimeSourceState()
        store.save(RuntimeSourceState())
    finally:
        source_state.logger.remove(sink)

    assert len(messages) == 2
    assert all("unverified" in message.lower() for message in messages)
    assert all("private" not in message.lower() for message in messages)


def test_runtime_policy_diagnostics_omit_path_and_state_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_sentinel = "RUNTIME-PATH-SENTINEL-37f9"
    state_sentinel = "RUNTIME-STATE-SENTINEL-a11c"
    selected = tmp_path / path_sentinel / "runtime_policy.json"
    selected.parent.mkdir()
    selected.write_text(
        '{"active_source": "server", '
        f'"active_server_id": "{state_sentinel}", "server_configured": true}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(private_paths, "_posix_guards_available", lambda: False)
    monkeypatch.setattr(private_paths, "_atomic_posix_guards_available", lambda: False)
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", True)
    messages: list[str] = []
    sink = source_state.logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        store = RuntimeSourceStateStore(selected)
        restored = store.load()
        store.save(restored)
    finally:
        source_state.logger.remove(sink)

    rendered = "\n".join(messages)
    assert path_sentinel not in rendered
    assert state_sentinel not in rendered
