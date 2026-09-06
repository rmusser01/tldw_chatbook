"""Explicit ownership cleanup for real Console integration fixtures."""

from contextlib import ExitStack
from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture(autouse=True)
async def close_owned_console_resources(
    monkeypatch, tmp_path, cleanup_file_descriptors
):
    """Drain controllers and SQLite, then close explicitly registered auxiliaries."""
    controllers, databases = [], []
    for cls, instances in (
        (ConsoleChatController, controllers),
        (CharactersRAGDB, databases),
    ):
        original_init = cls.__init__

        def record_instance(
            instance, *args, _initialize=original_init, _instances=instances, **kwargs
        ):
            _initialize(instance, *args, **kwargs)
            if _instances is controllers or Path(str(instance.db_path)).is_relative_to(
                tmp_path
            ):
                _instances.append(instance)

        monkeypatch.setattr(cls, "__init__", record_instance)

    auxiliary = ExitStack()
    yield auxiliary

    errors: list[BaseException] = []
    try:
        for controller in reversed(controllers):
            try:
                await controller.shutdown()
            except BaseException as exc:
                errors.append(exc)
        for database in reversed(databases):
            try:
                with database.quiesce_connections(timeout_seconds=2.0):
                    pass
                assert database.registered_connection_count() == 0
            except BaseException as exc:
                errors.append(exc)
        try:
            auxiliary.close()
        except BaseException as exc:
            errors.append(exc)
    finally:
        controllers.clear()
        databases.clear()
    if errors:
        raise BaseExceptionGroup("Console resource cleanup failed", errors)


@pytest.fixture(autouse=True)
async def close_owned_console_test_apps(
    request, monkeypatch, close_owned_console_resources
):
    """Drain only importing-module builder products before their exact DBs close."""
    apps = []
    build_app = request.module._build_test_app

    def build_owned_app(*args, **kwargs):
        app = build_app(*args, **kwargs)
        apps.append(app)
        for database in (
            app.local_workspace_db,
            app.subscriptions_db,
            app.local_library_collections_db,
            app.evaluation_orchestrator.db if app.evaluation_orchestrator else None,
        ):
            if database is not None:
                close_owned_console_resources.callback(database.close)
        return app

    monkeypatch.setattr(request.module, "_build_test_app", build_owned_app)
    yield

    errors = []
    try:
        for app in reversed(apps):
            try:
                await app._shutdown_console_runtime()
            except BaseException as exc:
                errors.append(exc)
    finally:
        apps.clear()
    if errors:
        raise BaseExceptionGroup("Console test app runtime cleanup failed", errors)
