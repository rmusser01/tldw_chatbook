"""TASK-36949.3: the DB layer reaches semantic-revision coordination through a port.

The coordinator runs inside the same write transaction as the write it
coordinates, a process with nothing installed fails closed, and every
composition root installs the port during startup.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_semantic_revision import (
    SemanticRevisionCoordinator,
    install_semantic_coordination,
)
from tldw_chatbook.DB import transaction_observer as port
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "port.db", client_id="port-test")
    yield database
    database.close_connection()


def _message(conversation_id):
    return {"conversation_id": conversation_id, "sender": "user", "content": "hi"}


def _message_count(db) -> int:
    return db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0]


def _content_and_version(db, message_id):
    row = db.get_connection().execute(
        "SELECT content, version FROM messages WHERE id = ?", (message_id,)
    ).fetchone()
    return row["content"], row["version"]


class _Recording:
    """Real coordinator that records whether its cursor's transaction was open."""

    def __init__(self, database, seen):
        self._real = SemanticRevisionCoordinator(database)
        self._seen = seen

    def ensure_current_revision(self, cursor, **kwargs):
        self._seen.append(cursor.connection.in_transaction)
        return self._real.ensure_current_revision(cursor, **kwargs)

    def mutate_message(self, cursor, **kwargs):
        self._seen.append(cursor.connection.in_transaction)
        return self._real.mutate_message(cursor, **kwargs)


def test_create_and_update_coordinate_inside_the_write_transaction(db, monkeypatch):
    seen: list[bool] = []
    monkeypatch.setattr(port, "_factory", lambda d: _Recording(d, seen))
    conversation_id = db.add_conversation({"title": "t"})
    message_id = db.add_message(_message(conversation_id))
    assert db.update_message(message_id, {"content": "edited"}, 1)
    assert seen == [True, True]


def test_coordinator_failure_rolls_the_create_back(db, monkeypatch):
    class Boom:
        def __init__(self, _db): ...

        def ensure_current_revision(self, cursor, **kwargs):
            raise RuntimeError("coordinator failed")

    conversation_id = db.add_conversation({"title": "t"})
    monkeypatch.setattr(port, "_factory", Boom)
    with pytest.raises(Exception, match="coordinator failed"):
        db.add_message(_message(conversation_id))
    assert _message_count(db) == 0


def test_coordinator_failure_after_the_mutation_rolls_the_update_back(db, monkeypatch):
    class MutateThenFail:
        """The real coordinator, failing after the guarded write has run."""

        def __init__(self, database):
            self._real = SemanticRevisionCoordinator(database)

        def mutate_message(self, cursor, *, mutate, **kwargs):
            def write_then_fail(mutation_cursor):
                mutate(mutation_cursor)
                raise RuntimeError("coordinator failed after the write")

            return self._real.mutate_message(cursor, mutate=write_then_fail, **kwargs)

    conversation_id = db.add_conversation({"title": "t"})
    message_id = db.add_message(_message(conversation_id))
    monkeypatch.setattr(port, "_factory", MutateThenFail)
    with pytest.raises(Exception, match="coordinator failed after the write"):
        db.update_message(message_id, {"content": "edited"}, 1)
    assert _content_and_version(db, message_id) == ("hi", 1)


def test_unset_port_fails_closed_on_create_and_persists_nothing(db, monkeypatch):
    conversation_id = db.add_conversation({"title": "t"})
    monkeypatch.setattr(port, "_factory", None)
    with pytest.raises(port.SemanticCoordinationNotInstalledError) as info:
        db.add_message(_message(conversation_id))
    assert "install_semantic_coordination" in str(info.value)
    assert _message_count(db) == 0


def test_unset_port_fails_closed_on_update_and_leaves_content(db, monkeypatch):
    conversation_id = db.add_conversation({"title": "t"})
    message_id = db.add_message(_message(conversation_id))
    monkeypatch.setattr(port, "_factory", None)
    with pytest.raises(port.SemanticCoordinationNotInstalledError):
        db.update_message(message_id, {"content": "edited"}, 1)
    assert _content_and_version(db, message_id) == ("hi", 1)


def test_install_is_idempotent(monkeypatch):
    monkeypatch.setattr(port, "_factory", None)
    install_semantic_coordination()
    install_semantic_coordination()
    assert port._factory is SemanticRevisionCoordinator


_UNCOMPOSED_PROCESS = """
from Tests.network_guard import install
install()
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.transaction_observer import SemanticCoordinationNotInstalledError
db = CharactersRAGDB(':memory:', 'uncomposed')
conversation = db.add_conversation({'title': 'bare library use'})
message = {'conversation_id': conversation, 'sender': 'user', 'content': 'hi'}
try:
    db.add_message(message)
except SemanticCoordinationNotInstalledError:
    pass
else:
    raise AssertionError('an uncomposed process coordinated a write')
count = 'SELECT COUNT(*) FROM messages'
assert db.get_connection().execute(count).fetchone()[0] == 0
import tldw_chatbook.Chat.console_semantic_revision  # registers itself on import
db.add_message(message)
assert db.get_connection().execute(count).fetchone()[0] == 1
print('fail-closed then registered')
"""


def test_uncomposed_process_fails_closed_until_the_coordinator_is_imported():
    result = subprocess.run(
        [sys.executable, "-c", _UNCOMPOSED_PROCESS],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "fail-closed then registered" in result.stdout


def test_db_module_does_not_import_the_console_coordinator():
    source = (REPO / "tldw_chatbook/DB/ChaChaNotes_DB.py").read_text()
    assert "console_semantic_revision" not in source
    assert "SemanticRevisionCoordinator" not in source


class _StartupReached(Exception):
    """The first startup step after the root's install ran."""


def _stop(*_args, **_kwargs):
    raise _StartupReached


def _tldw_cli(monkeypatch, tmp_path):
    # `tldw-cli` and `python -m tldw_chatbook` reach this through cli.main_cli_runner.
    from tldw_chatbook import app_entry
    from tldw_chatbook.Tools import workspace_file_roots

    monkeypatch.setattr(workspace_file_roots, "set_launch_cwd", lambda: None)
    monkeypatch.setattr(app_entry, "install_termination_handlers", _stop)
    return app_entry.main_cli_runner


def _python_m_app(monkeypatch, tmp_path):
    # `python -m tldw_chatbook.app`; textual-serve sessions launch this too.
    from tldw_chatbook import app_entry
    from tldw_chatbook.Tools import workspace_file_roots

    monkeypatch.setattr(workspace_file_roots, "set_launch_cwd", lambda: None)
    monkeypatch.setattr(app_entry, "initialize_early_logging", _stop)
    return app_entry._run_module_main


def _tldw_serve(monkeypatch, tmp_path):
    from tldw_chatbook.Web_Server import serve

    monkeypatch.setattr(sys, "argv", ["tldw-serve"])
    monkeypatch.setattr(serve, "apply_config_eager_flags", lambda: None)
    monkeypatch.setattr(serve, "check_web_server_available", lambda: True)
    monkeypatch.setattr(serve, "get_cli_setting", _stop)
    return serve.main


def _mobile_service(monkeypatch, tmp_path):
    from tldw_chatbook.Web_Server import mobile_service

    service = mobile_service.MobileChatService.__new__(mobile_service.MobileChatService)
    service._data_dir = tmp_path
    monkeypatch.setattr(mobile_service, "WorkspaceDB", _stop)
    return service._build_engine


def _mcp_server(monkeypatch, tmp_path):
    # `python -m tldw_chatbook.MCP` and `python -m tldw_chatbook.MCP.server`.
    from tldw_chatbook.MCP import server
    from tldw_chatbook.Utils import optional_deps

    monkeypatch.setattr(optional_deps, "apply_config_eager_flags", lambda: None)
    monkeypatch.setattr(server, "TldwMCPServer", _stop)
    return lambda: asyncio.run(server.main())


def _rag_backfill(monkeypatch, tmp_path):
    # `python -m tldw_chatbook.RAG_Search.backfill`.
    from tldw_chatbook import config
    from tldw_chatbook.RAG_Search import backfill

    monkeypatch.setattr(backfill, "semantic_indexing_available", lambda: True)
    monkeypatch.setattr(config, "get_media_db_path", _stop)
    return lambda: backfill.main(["--types", "media"])


@pytest.mark.parametrize(
    "root",
    [_tldw_cli, _python_m_app, _tldw_serve, _mobile_service, _mcp_server, _rag_backfill],
    ids=lambda root: root.__name__.lstrip("_"),
)
def test_every_composition_root_installs_the_port_during_startup(
    root, monkeypatch, tmp_path
):
    start = root(monkeypatch, tmp_path)
    monkeypatch.setattr(port, "_factory", None)
    with pytest.raises(_StartupReached):
        start()
    assert port._factory is SemanticRevisionCoordinator
