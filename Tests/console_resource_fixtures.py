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
