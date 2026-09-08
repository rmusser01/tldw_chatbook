"""Opt-in thread and initializer ownership for factory-built test apps."""

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from typing import Any

import pytest


class OwnedAppExecutor(ThreadPoolExecutor):
    """Close newly opened exact app handles on their creator thread.

    Keep actual concurrent futures: cancelling Textual's asyncio wrapper does
    not finish its already-running thread. The per-test asyncio Runner owns
    executor shutdown; the resource fixture drains work before closing DBs.
    """

    def __init__(self, databases: list[Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.databases = databases
        self.submitted = []
        self.cleanup_errors = []

    def submit(
        self, callback: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        def invoke():
            previous = [
                (database, getattr(database._thread_local, "conn", None))
                for database in self.databases
            ]
            try:
                return callback(*args, **kwargs)
            finally:
                for database, connection in previous:
                    current = getattr(database._thread_local, "conn", None)
                    if current is not None and current is not connection:
                        try:
                            database.close()
                        except BaseException as error:
                            # Preserve the worker's own return/exception, and
                            # attempt every handle before reporting at teardown.
                            self.cleanup_errors.append(error)

        future = super().submit(invoke)
        self.submitted.append(future)
        return future

    async def drain(self, *, timeout_seconds: float = 10.0) -> list[BaseException]:
        """Boundedly await real thread completion without cancelling its work."""
        async with asyncio.timeout(timeout_seconds):
            while pending := [future for future in self.submitted if not future.done()]:
                await asyncio.shield(
                    asyncio.gather(
                        *(asyncio.wrap_future(future) for future in pending),
                        return_exceptions=True,
                    )
                )
        return list(self.cleanup_errors)


@pytest.fixture(autouse=True)
async def close_owned_console_workers(
    request, monkeypatch, close_owned_console_resources, close_owned_console_test_apps
):
    """Give importing-module app workers an exact, drained connection owner."""
    databases = []
    loop = asyncio.get_running_loop()
    executor = None
    build = request.module._build_test_app

    def build_owned(*args, **kwargs):
        nonlocal executor
        app = build(*args, **kwargs)
        databases.extend(
            database
            for database in (app.local_workspace_db, app.local_library_collections_db)
            if database is not None
        )
        if executor is None:
            executor = OwnedAppExecutor(databases)
            loop.set_default_executor(executor)
            close_owned_console_resources.before_close.append(executor.drain)
        return app

    monkeypatch.setattr(request.module, "_build_test_app", build_owned)


def capture_app_initialization(
    build: Callable[[], Any], monkeypatch: pytest.MonkeyPatch, cleanup: ExitStack
) -> Any:
    """Capture constructor-thread handles, then select the returned app's DBs.

    Prompts and Media permit cross-thread close. Their constructor pool has
    joined before build returns; ordinary db.close() would only see the test
    thread's local handle. No unrelated instance is registered for cleanup.
    """
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

    connections = {}
    with monkeypatch.context() as initializers:
        for cls in (PromptsDatabase, MediaDatabase):
            original = cls._get_thread_connection

            def capture(database, _original=original):
                connection = _original(database)
                connections.setdefault(database, set()).add(connection)
                return connection

            initializers.setattr(cls, "_get_thread_connection", capture)
        app = build()
    for database in (app.prompts_db, app.media_db):
        for connection in connections.get(database, ()):
            cleanup.callback(connection.close)
    return app


@pytest.fixture(autouse=True)
def close_owned_app_initialization_connections(
    request, monkeypatch, close_owned_console_resources, close_owned_console_test_apps
):
    """Capture real constructor DBs only for importing-module builder products."""
    build = request.module._build_test_app

    def build_owned(*args, **kwargs):
        return capture_app_initialization(
            lambda: build(*args, **kwargs), monkeypatch, close_owned_console_resources
        )

    monkeypatch.setattr(request.module, "_build_test_app", build_owned)
