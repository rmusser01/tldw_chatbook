"""Failure containment for the explicitly imported Console resource fixture."""

import asyncio
from contextlib import ExitStack, contextmanager

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
