"""A durable local callback owns one operation without merging transactions."""

from functools import partial
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Chat.console_automatic_submission import run_owned_chat_operation
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.asyncio
async def test_durable_callback_batches_admission_and_preserves_transactions(
    tmp_path, monkeypatch
):
    database = CharactersRAGDB(tmp_path / "durable.sqlite", "durable-batch")
    controller = SimpleNamespace(
        store=SimpleNamespace(persistence=SimpleNamespace(db=database)),
        _durable_db_call_offloadable=lambda: True,
    )
    controller._run_owned_chat_db_operation = partial(
        run_owned_chat_operation, controller
    )
    with database.transaction() as cursor:
        cursor.execute("CREATE TABLE batch_effect (value INTEGER)")
    admissions = []
    original = storage.acquire_storage

    def counted(path=None, **kwargs):
        admissions.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(storage, "acquire_storage", counted)

    def command():
        operation = getattr(storage._operation_local, "operation", None)
        assert operation is not None
        with database.transaction() as cursor:
            cursor.execute("INSERT INTO batch_effect VALUES (1)")
        try:
            with database.transaction() as cursor:
                cursor.execute("INSERT INTO batch_effect VALUES (2)")
                raise ValueError("roll back only this transaction")
        except ValueError:
            pass
        with database.transaction() as cursor:
            values = [
                row[0] for row in cursor.execute("SELECT value FROM batch_effect")
            ]
        assert storage._operation_local.operation is operation
        return values, operation

    try:
        values, operation = await ConsoleChatController._run_durable_db_call(
            controller, command
        )
        assert values == [1]
        # One operation plus the independent SQLite handle's resource admission.
        assert admissions == [database.db_path, database.db_path]
        assert operation not in storage._operations
        assert database.registered_connection_count() == 1  # original caller handle
    finally:
        database.close_connection()
