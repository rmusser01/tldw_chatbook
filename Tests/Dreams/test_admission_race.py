"""TASK-33132 migration 7: admission recheck parity (Guardian race test ported).

A pause during native open/setup must close the unadmitted handle — the
repaired fleet standard; the old Dreams _get_connection carried the same
double gate the migration must not lose.
"""

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB


@pytest.mark.parametrize("pause_point", ("allocation", "setup"))
@pytest.mark.parametrize("getter_name", ("_get_connection", "_held_connection"))
def test_raw_dreams_connection_rechecks_admission_before_return(
    tmp_path, monkeypatch, pause_point, getter_name
):
    """A pause during native open/setup must close the unadmitted handle."""
    import sqlite3

    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant
    from tldw_chatbook.DB.base_db import BaseDB

    database = DreamsDB(tmp_path / "dreams-race.sqlite", "admission-race")
    database.close()
    participant = _repository_participant(database)
    original = BaseDB._get_connection
    allocated = []

    def pause_after_native(*args, **kwargs):
        connection = original(*args, **kwargs)
        allocated.append(connection)
        if pause_point == "allocation":
            participant.close_admission()
        else:
            execute = connection.execute

            def pause_in_setup(sql, *params, **options):
                result = execute(sql, *params, **options)
                if sql.startswith("PRAGMA"):
                    participant.close_admission()
                return result

            monkeypatch.setattr(connection, "execute", pause_in_setup)
        return connection

    monkeypatch.setattr(BaseDB, "_get_connection", pause_after_native)
    try:
        with pytest.raises(RecoveryRequired, match="storage_locally_paused"):
            getattr(database, getter_name)()
        assert len(allocated) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            allocated[0].execute("SELECT 42")
    finally:
        participant.resume()
        for connection in allocated:
            connection.close()
        database.close()
