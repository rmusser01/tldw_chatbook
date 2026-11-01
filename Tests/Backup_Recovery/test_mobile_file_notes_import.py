"""Fresh native File Notes construction must not import unrelated config owners."""

import hashlib
import json
import os
import pwd
import sqlite3
import sys
import tempfile
from pathlib import Path


def cold_native_owner(result_path):
    from Tests import network_guard, real_profile_guard

    entry = {key: os.environ.get(key) for key in ("HOME", "USERPROFILE")}
    uid_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert entry == {"HOME": str(uid_home), "USERPROFILE": None}
    real_profile_guard.install()
    network_guard.install()
    assert Path(os.environ["TLDW_TEST_REAL_HOME"]) == uid_home
    assert (
        str(uid_home / ".config/tldw_cli").lower()
        in real_profile_guard.protected_roots()
    )
    assert "tldw_chatbook.config" not in sys.modules
    root = Path(
        tempfile.mkdtemp(
            prefix="native-file-notes-", dir=Path(os.environ["TMPDIR"]).resolve()
        )
    ).resolve()
    selector = root / "config.toml"
    selector.write_text('[general]\nusers_name="shared"\n')
    selector.chmod(0o600)
    before = selector.read_bytes()
    os.environ["TLDW_CONFIG_PATH"] = str(selector)
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    bootstrap.default_bootstrap_root = lambda: root / "bootstrap"
    report = {
        "scope": "standalone cold native FileNotes class/schema owner",
        "root": str(root),
        "entry_home": entry,
    }
    owner = None

    def refuse_config(event, args):
        if event == "import" and args[0] == "tldw_chatbook.config":
            raise RuntimeError("cold_file_notes_imported_configuration")

    sys.addaudithook(refuse_config)
    try:
        from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica

        assert "tldw_chatbook.config" not in sys.modules and not storage._startups
        owner = FileNotesReplica(root / "file_notes.sqlite")
        connection = owner._connection
        assert isinstance(connection, sqlite3.Connection)
        with owner._locked_connection():
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table'"
                )
            }
        assert {"files", "revisions", "protected_paths", "files_fts"} <= tables
        assert owner.list_active_files("root") == []
        owner.close()
        owner = None
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError as error:
            assert str(error) == "Cannot operate on a closed database."
        else:
            raise AssertionError("native FileNotes handle remains open")
        assert (
            not storage._live_leases
            and not storage._operations
            and not storage._raw_operations
        )
        assert (
            not storage._holds and not storage._retiring_holds and not storage._startups
        )
        assert selector.read_bytes() == before
        report.update(
            completed=True,
            connection_closed=True,
            tables=sorted(tables),
            selector_sha256=hashlib.sha256(before).hexdigest(),
        )
    finally:
        if owner is not None:
            owner.close()
        report.update(
            profile_guard=real_profile_guard.take_violations(),
            network_guard=network_guard.blocked_attempts(),
            exit_home={key: os.environ.get(key) for key in entry},
            config_loaded="tldw_chatbook.config" in sys.modules,
            startups=len(storage._startups),
            selector_unchanged=selector.read_bytes() == before,
        )
        result_path.write_text(json.dumps(report, indent=2))
    assert report["entry_home"] == report["exit_home"]
    assert (
        not report["profile_guard"]
        and not report["network_guard"]
        and not report["config_loaded"]
    )


if __name__ == "__main__":
    result = Path(sys.argv[1])
    assert result.is_absolute() and result.parent.is_dir() and not result.exists()
    cold_native_owner(result)
