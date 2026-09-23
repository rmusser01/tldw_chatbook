"""Pet storage failures go to the log, never to the terminal.

TASK-32901 (tier-2 S15 P2): ``tamagotchi_storage.py`` reported six storage
failures with bare ``print()`` -- including ``"Error saving to SQLite"`` on a
data-write path -- while the same module already has a ``logger`` at module
scope and uses it at seven other lines. Under a Textual app stdout is the
surface the TUI is drawing on, not a log sink, so a persistent write failure
produced no log line anywhere and corrupted the screen instead.

This half of ``Widgets/Tamagotchi/`` is live infrastructure: it is registered
in ``DB/private_sqlite.py``'s owner policy and has three ``Backup_Recovery``
participants, unlike the unmounted widget half.
"""

from __future__ import annotations

import logging

import pytest

from tldw_chatbook.Widgets.Tamagotchi import tamagotchi_storage


@pytest.fixture()
def storage(tmp_path):
    return tamagotchi_storage.SQLiteStorage(tmp_path / "pets.sqlite")


def _break_connection(monkeypatch, storage):
    def _fail(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(storage, "_connect", _fail)


@pytest.mark.parametrize(
    "call, expected",
    [
        (lambda s: s.list_pets(), []),
        (lambda s: s.get_statistics(), {}),
        (lambda s: s.delete("pet-1"), False),
    ],
)
def test_sqlite_storage_failures_are_logged_not_printed(
    storage, monkeypatch, capsys, caplog, call, expected
):
    _break_connection(monkeypatch, storage)

    with caplog.at_level(logging.ERROR, logger=tamagotchi_storage.logger.name):
        assert call(storage) == expected

    assert capsys.readouterr().out == ""
    assert caplog.records, "storage failure left no log record"
    assert "database is locked" in caplog.text


def test_no_print_calls_remain_in_the_storage_module():
    source = tamagotchi_storage.__file__
    with open(source, encoding="utf-8") as handle:
        offenders = [
            (number, line.strip())
            for number, line in enumerate(handle, start=1)
            if line.lstrip().startswith("print(")
        ]
    assert offenders == []
