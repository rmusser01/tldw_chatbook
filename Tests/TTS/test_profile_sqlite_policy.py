"""Runtime admission tests for the TTS profile SQLite close policy."""

from __future__ import annotations

import inspect
import pickle
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.TTS import profile_sqlite_policy as policy
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError


class _ConnectionProxy:
    """Small public-capability proxy with observable borrowed ownership."""

    def __init__(
        self,
        *,
        set_error: BaseException | None = None,
        get_error: BaseException | None = None,
        configured: object = True,
    ) -> None:
        self.set_error = set_error
        self.get_error = get_error
        self.configured = configured
        self.set_calls: list[tuple[int, bool]] = []
        self.get_calls: list[int] = []
        self.close_calls = 0

    def setconfig(self, option: int, enabled: bool) -> None:
        self.set_calls.append((option, enabled))
        if self.set_error is not None:
            raise self.set_error

    def getconfig(self, option: int) -> object:
        self.get_calls.append(option)
        if self.get_error is not None:
            raise self.get_error
        return self.configured

    def close(self) -> None:
        self.close_calls += 1


def _assert_runtime_unsupported(error: ProfileRepositoryError, *secrets: str) -> None:
    assert error.code == "runtime_unsupported"
    assert str(error) == (
        "TTS profile repository unavailable: SQLite runtime lacks required "
        "close-policy support."
    )
    rendered = repr(error), str(error), repr(error.args)
    for secret in secrets:
        assert all(secret not in value for value in rendered)


def test_configure_native_policy_verifies_flag_without_sql() -> None:
    connection = sqlite3.connect(":memory:")
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    try:
        policy.configure_native_close_policy(connection)
        assert connection.getconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE) is True
        assert statements == []
        connection.execute("SELECT 1")  # The borrowed handle remains open.
    finally:
        connection.close()


def test_missing_constant_refuses_before_opening_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(policy.sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE")

    def unexpected_connect(*_args: object, **_kwargs: object) -> Any:
        pytest.fail("unsupported runtime opened a handle")

    monkeypatch.setattr(policy.sqlite3, "connect", unexpected_connect)
    with pytest.raises(ProfileRepositoryError) as failure:
        policy.require_native_close_policy_support()
    _assert_runtime_unsupported(failure.value)


@pytest.mark.parametrize("missing_method", ("setconfig", "getconfig"))
def test_missing_public_method_refuses_before_opening_probe(
    monkeypatch: pytest.MonkeyPatch,
    missing_method: str,
) -> None:
    methods = {
        name: lambda *_args, **_kwargs: None
        for name in {"setconfig", "getconfig"} - {missing_method}
    }
    fake_connection_type = type("MissingCapabilityConnection", (), methods)
    fake_sqlite = SimpleNamespace(
        Connection=fake_connection_type,
        SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE=sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE,
        Error=sqlite3.Error,
        connect=lambda *_args, **_kwargs: pytest.fail(
            "unsupported runtime opened a handle"
        ),
    )
    monkeypatch.setattr(policy, "sqlite3", fake_sqlite)

    with pytest.raises(ProfileRepositoryError) as failure:
        policy.require_native_close_policy_support()
    _assert_runtime_unsupported(failure.value)


@pytest.mark.parametrize("missing_method", ("setconfig", "getconfig"))
def test_borrowed_connection_missing_method_is_refused_without_close(
    monkeypatch: pytest.MonkeyPatch,
    missing_method: str,
) -> None:
    methods = {
        name: lambda *_args, **_kwargs: True
        for name in {"setconfig", "getconfig"} - {missing_method}
    }

    def close(connection):
        connection.close_calls += 1

    methods["close"] = close
    connection = type("MissingCapabilityConnection", (), methods)()
    connection.close_calls = 0

    with pytest.raises(ProfileRepositoryError) as failure:
        policy.configure_native_close_policy(connection)  # type: ignore[arg-type]
    _assert_runtime_unsupported(failure.value)
    assert connection.close_calls == 0


@pytest.mark.parametrize("missing_method", ("setconfig", "getconfig"))
def test_missing_method_ownership_assertion_detects_borrowed_close(
    monkeypatch, missing_method
):
    configure = policy.configure_native_close_policy

    def incorrectly_close(connection):
        try:
            configure(connection)
        finally:
            connection.close()

    monkeypatch.setattr(policy, "configure_native_close_policy", incorrectly_close)
    with pytest.raises(AssertionError):
        test_borrowed_connection_missing_method_is_refused_without_close(
            monkeypatch, missing_method
        )


@pytest.mark.parametrize("phase", ("set", "get"))
def test_rejected_public_configuration_is_bounded_and_borrowed_handle_stays_open(
    phase: str,
) -> None:
    secret = f"/private/runtime-{phase}-sentinel.sqlite3"
    rejection = sqlite3.NotSupportedError(secret)
    connection = _ConnectionProxy(
        set_error=rejection if phase == "set" else None,
        get_error=rejection if phase == "get" else None,
    )

    with pytest.raises(ProfileRepositoryError) as failure:
        policy.configure_native_close_policy(connection)  # type: ignore[arg-type]

    _assert_runtime_unsupported(failure.value, secret)
    assert connection.close_calls == 0


def test_false_configuration_verification_is_refused_without_closing_borrowed_handle() -> (
    None
):
    connection = _ConnectionProxy(configured=False)

    with pytest.raises(ProfileRepositoryError) as failure:
        policy.configure_native_close_policy(connection)  # type: ignore[arg-type]

    _assert_runtime_unsupported(failure.value)
    assert connection.set_calls == [(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)]
    assert connection.get_calls == [sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE]
    assert connection.close_calls == 0


@pytest.mark.parametrize("configured", (True, False))
def test_owned_probe_closes_after_success_or_configuration_refusal(
    monkeypatch: pytest.MonkeyPatch,
    configured: bool,
) -> None:
    connection = _ConnectionProxy(configured=configured)
    monkeypatch.setattr(policy.sqlite3, "connect", lambda target: connection)

    if configured:
        policy.require_native_close_policy_support()
    else:
        with pytest.raises(ProfileRepositoryError) as failure:
            policy.require_native_close_policy_support()
        _assert_runtime_unsupported(failure.value)

    assert connection.close_calls == 1


def test_programming_error_is_not_translated_or_closed_by_borrowed_helper() -> None:
    bug = TypeError("caller supplied a broken connection")
    connection = _ConnectionProxy(set_error=bug)

    with pytest.raises(TypeError) as failure:
        policy.configure_native_close_policy(connection)  # type: ignore[arg-type]

    assert failure.value is bug
    assert connection.close_calls == 0


def test_policy_source_has_no_private_or_numeric_fallback() -> None:
    source = inspect.getsource(policy)

    assert "PRAGMA" not in source.upper()
    assert "ctypes" not in source
    assert "connect_private_sqlite" not in source
    assert "tldw_chatbook.config" not in source
    assert "tldw_chatbook.app" not in source
    assert "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE =" not in source
    assert "1006" not in source


def test_runtime_unsupported_error_round_trips_and_unknown_code_stays_closed() -> None:
    original = ProfileRepositoryError("runtime_unsupported")
    restored = pickle.loads(pickle.dumps(original))

    _assert_runtime_unsupported(restored)
    unknown = ProfileRepositoryError("/private/unknown-runtime-sentinel")
    assert unknown.code == "operation_failed"
    assert str(unknown) == "TTS profile repository failed: operation_failed"
