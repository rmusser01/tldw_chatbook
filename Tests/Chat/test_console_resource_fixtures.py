"""Failure containment for the explicitly imported Console resource fixture."""

import asyncio
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests import console_resource_fixtures as resources


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "shutdown",
        "shutdown_cancel",
        "quiescence",
        "quiescence_cancel",
        "count",
        "auxiliary",
        "auxiliary_cancel",
        "all",
    ],
)
async def test_resource_cleanup_attempts_every_owner_before_reporting_errors(
    monkeypatch, tmp_path, failure
):
    events = []
    shutdown_error = RuntimeError("shutdown failed")
    shutdown_cancel = asyncio.CancelledError("shutdown cancelled")
    quiescence_error = TimeoutError("database still active")
    quiescence_cancel = asyncio.CancelledError("database cleanup cancelled")
    auxiliary_error = RuntimeError("auxiliary cleanup failed")
    auxiliary_cancel = asyncio.CancelledError("auxiliary cleanup cancelled")

    # These narrow owners inject lifecycle failures; the fixture's constructor
    # tracking, path filter, teardown ordering and error reporting remain real.
    class Controller:
        def __init__(self, name):
            self.name = name

        async def shutdown(self):
            events.append(("shutdown", self.name))
            if self.name == "second" and failure in {"shutdown", "all"}:
                raise shutdown_error
            if self.name == "second" and failure == "shutdown_cancel":
                raise shutdown_cancel

    class Database:
        def __init__(self, path):
            self.db_path = path
            self.name = path.stem

        @contextmanager
        def quiesce_connections(self, *, timeout_seconds):
            assert timeout_seconds == 2.0
            events.append(("quiesce", self.name))
            if self.name == "busy" and failure in {"quiescence", "all"}:
                raise quiescence_error
            if self.name == "busy" and failure == "quiescence_cancel":
                raise quiescence_cancel
            try:
                yield
            finally:
                events.append(("release", self.name))

        def registered_connection_count(self):
            events.append(("count", self.name))
            return int(self.name == "retained" and failure in {"count", "all"})

    monkeypatch.setattr(resources, "ConsoleChatController", Controller)
    monkeypatch.setattr(resources, "CharactersRAGDB", Database)
    fixture = resources.close_owned_console_resources.__wrapped__(
        monkeypatch, tmp_path, None
    )
    auxiliary = await anext(fixture)
    assert isinstance(auxiliary, ExitStack)

    def close_last_auxiliary():
        events.append(("auxiliary", "last"))
        if failure in {"auxiliary", "all"}:
            raise auxiliary_error
        if failure == "auxiliary_cancel":
            raise auxiliary_cancel

    auxiliary.callback(events.append, ("auxiliary", "first"))
    auxiliary.callback(close_last_auxiliary)
    Controller("first")
    Controller("second")
    for name in ("healthy", "retained", "busy"):
        Database(tmp_path / f"{name}.db")
    Database(tmp_path.parent / "foreign.db")

    caught = None
    try:
        await anext(fixture)
    except StopAsyncIteration:
        pass
    except BaseException as exc:
        caught = exc
    finally:
        await fixture.aclose()

    expected = [("shutdown", "second"), ("shutdown", "first")]
    for name in ("busy", "retained", "healthy"):
        expected.append(("quiesce", name))
        if name != "busy" or failure not in {
            "quiescence",
            "quiescence_cancel",
            "all",
        }:
            expected.extend([("release", name), ("count", name)])
    expected.extend([("auxiliary", "last"), ("auxiliary", "first")])
    assert events == expected
    if failure is None:
        assert caught is None
    else:
        expected_group = (
            BaseExceptionGroup
            if failure in {"shutdown_cancel", "quiescence_cancel", "auxiliary_cancel"}
            else ExceptionGroup
        )
        assert isinstance(caught, expected_group)
        expected_types = []
        if failure in {"shutdown", "all"}:
            expected_types.append(RuntimeError)
            assert caught.exceptions[0] is shutdown_error
        if failure == "shutdown_cancel":
            expected_types.append(asyncio.CancelledError)
            assert caught.exceptions[0] is shutdown_cancel
        if failure in {"quiescence", "all"}:
            expected_types.append(TimeoutError)
            assert quiescence_error in caught.exceptions
        if failure == "quiescence_cancel":
            expected_types.append(asyncio.CancelledError)
            assert caught.exceptions[0] is quiescence_cancel
        if failure in {"count", "all"}:
            expected_types.append(AssertionError)
        if failure in {"auxiliary", "all"}:
            expected_types.append(RuntimeError)
            assert caught.exceptions[-1] is auxiliary_error
        if failure == "auxiliary_cancel":
            expected_types.append(asyncio.CancelledError)
            assert caught.exceptions[-1] is auxiliary_cancel
        assert [type(error) for error in caught.exceptions] == expected_types


@pytest.mark.parametrize(
    "fail_before_teardown", [False, True], ids=("normal", "forced")
)
async def test_resource_cleanup_retires_real_sqlite_after_earlier_owner_failure(
    monkeypatch, tmp_path, fail_before_teardown
):
    """A failed controller shutdown cannot strand later real SQLite owners."""
    shutdown_error = RuntimeError("shutdown failed")

    class Controller:
        def __init__(self):
            pass

        async def shutdown(self):
            raise shutdown_error

    monkeypatch.setattr(resources, "ConsoleChatController", Controller)
    fixture = resources.close_owned_console_resources.__wrapped__(
        monkeypatch, tmp_path, None
    )
    await anext(fixture)
    databases = []
    forced_error = (
        AssertionError("forced failure before fixture teardown")
        if fail_before_teardown
        else None
    )
    operation_error = None
    caught = None
    try:
        Controller()
        databases = [
            resources.CharactersRAGDB(
                tmp_path / f"real-{index}.sqlite", client_id=f"real-{index}"
            )
            for index in range(2)
        ]
        for database in databases:
            with database.transaction() as cursor:
                assert cursor.execute("SELECT 1").fetchone()[0] == 1
            assert database.registered_connection_count() == 1
        if forced_error is not None:
            raise forced_error
    except BaseException as exc:
        operation_error = exc
    finally:
        try:
            await anext(fixture)
        except StopAsyncIteration:
            pass
        except BaseException as exc:
            caught = exc
        finally:
            await fixture.aclose()

    assert isinstance(caught, ExceptionGroup)
    assert caught.exceptions == (shutdown_error,)
    if operation_error is not None and operation_error is not forced_error:
        raise operation_error
    assert [database.registered_connection_count() for database in databases] == [
        0,
        0,
    ]
    assert operation_error is forced_error


@pytest.mark.parametrize("failure", [None, "runtime", "runtime_cancel", "database"])
async def test_app_cleanup_owns_only_importing_module_builder_products(
    monkeypatch, failure
):
    events = []
    build_calls = []
    runtime_error = (
        asyncio.CancelledError("runtime cancelled")
        if failure == "runtime_cancel"
        else RuntimeError("runtime failed")
    )
    database_error = RuntimeError("database close failed")

    def build_app(name, *, marker):
        build_calls.append((name, marker))

        def database(kind):
            def close():
                events.append(("close", name, kind))
                if name == "second" and kind == "evals" and failure == "database":
                    raise database_error

            return SimpleNamespace(close=close)

        async def shutdown():
            events.append(("runtime", name))
            if name == "second" and failure in {"runtime", "runtime_cancel"}:
                raise runtime_error

        return SimpleNamespace(
            _shutdown_console_runtime=shutdown,
            local_workspace_db=database("workspace"),
            subscriptions_db=database("subscriptions"),
            local_library_collections_db=database("collections"),
            evaluation_orchestrator=SimpleNamespace(db=database("evals")),
        )

    module = SimpleNamespace(_build_test_app=build_app)
    foreign_module = SimpleNamespace(_build_test_app=build_app)
    existing = build_app("preexisting", marker=0)
    auxiliary = ExitStack()
    with monkeypatch.context() as patch:
        fixture = resources.close_owned_console_test_apps.__wrapped__(
            SimpleNamespace(module=module), patch, auxiliary
        )
        await anext(fixture)
        module._build_test_app("first", marker=1)
        module._build_test_app("second", marker=2)
        foreign = foreign_module._build_test_app("foreign", marker=3)
        assert foreign_module._build_test_app is build_app
        assert events == []

        caught = None
        try:
            await anext(fixture)
        except StopAsyncIteration:
            pass
        except BaseException as exc:
            caught = exc
        finally:
            await fixture.aclose()

        assert events == [("runtime", "second"), ("runtime", "first")]
        if failure in {"runtime", "runtime_cancel"}:
            assert isinstance(caught, BaseExceptionGroup)
            assert caught.exceptions == (runtime_error,)
        else:
            assert caught is None

        # Shared controller/ChaChaNotes cleanup owns closing this stack later.
        if failure == "database":
            with pytest.raises(RuntimeError) as raised:
                auxiliary.close()
            assert raised.value is database_error
        else:
            auxiliary.close()

    assert module._build_test_app is build_app
    assert build_calls == [
        ("preexisting", 0),
        ("first", 1),
        ("second", 2),
        ("foreign", 3),
    ]
    assert events == [
        ("runtime", "second"),
        ("runtime", "first"),
        *[
            ("close", name, kind)
            for name in ("second", "first")
            for kind in ("evals", "collections", "subscriptions", "workspace")
        ],
    ]
    # Retain both foreign owners throughout teardown so GC cannot hide a close.
    assert existing is not foreign


@pytest.mark.parametrize("failure", [None, "runtime", "runtime_cancel", "auxiliary"])
async def test_app_cleanup_releases_only_owned_real_instance_locks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str | None
) -> None:
    """Release exact app locks despite teardown faults, retaining foreign locks.

    Args:
        monkeypatch: Fixture isolating the importing module's app builder.
        tmp_path: Private directories for real advisory lock files.
        failure: Optional runtime or auxiliary teardown failure to inject.
    """
    from tldw_chatbook.Utils.instance_lock import acquire_profile_instance_lock

    error = (
        asyncio.CancelledError("runtime cancelled")
        if failure == "runtime_cancel"
        else RuntimeError("teardown failed")
    )
    with ExitStack() as cleanup:
        def build_app(name):
            directory = tmp_path / name
            directory.mkdir()
            status = acquire_profile_instance_lock(directory)
            assert status.acquired and status.handle is not None
            cleanup.callback(status.handle.close)

            async def shutdown():
                if failure in {"runtime", "runtime_cancel"}:
                    raise error

            return SimpleNamespace(
                _instance_lock_status=status,
                _shutdown_console_runtime=shutdown,
                local_workspace_db=None,
                subscriptions_db=None,
                local_library_collections_db=None,
                evaluation_orchestrator=None,
            )

        existing = build_app("existing")
        module = SimpleNamespace(_build_test_app=build_app)
        foreign_module = SimpleNamespace(_build_test_app=build_app)
        auxiliary = ExitStack()
        cleanup.callback(auxiliary.close)
        fixture = resources.close_owned_console_test_apps.__wrapped__(
            SimpleNamespace(module=module), monkeypatch, auxiliary
        )
        await anext(fixture)
        owned = module._build_test_app("owned")
        foreign = foreign_module._build_test_app("foreign")

        def fail_auxiliary():
            raise error

        if failure == "auxiliary":
            auxiliary.callback(fail_auxiliary)
        try:
            if failure in {"runtime", "runtime_cancel"}:
                with pytest.raises(BaseExceptionGroup) as caught:
                    await anext(fixture)
                assert caught.value.exceptions == (error,)
            else:
                with pytest.raises(StopAsyncIteration):
                    await anext(fixture)
        finally:
            await fixture.aclose()
        if failure == "auxiliary":
            with pytest.raises(RuntimeError) as caught:
                auxiliary.close()
            assert caught.value is error
        else:
            auxiliary.close()

        # Keep all owners referenced so garbage collection cannot release them.
        assert owned._instance_lock_status.handle.closed
        assert not existing._instance_lock_status.handle.closed
        assert not foreign._instance_lock_status.handle.closed
        assert all(
            (tmp_path / name / ".instance.lock").is_file()
            for name in ("owned", "existing", "foreign")
        )
        for name in ("owned", "existing", "foreign"):
            probe = acquire_profile_instance_lock(tmp_path / name)
            if probe.handle is not None:
                cleanup.callback(probe.handle.close)
            assert probe.acquired is (name == "owned")
