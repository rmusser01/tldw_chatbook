"""Keep the normal CLI startup/recovery path inside the embedded runtime."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


def probe() -> None:
    """Exercise installed config, migration resources and persistent SQLite."""
    from tldw_chatbook.Utils.startup_logging import quiet_startup_stderr

    quiet_startup_stderr()
    from tldw_chatbook.Backup_Recovery.storage_admission import admit_startup

    admit_startup()
    import tldw_chatbook
    from tldw_chatbook.config import (
        get_chachanotes_db_path,
        get_cli_config_path,
        get_cli_log_file_path,
        get_model_cache_dir,
        get_user_data_dir,
    )

    root = Path(__file__).absolute().parent
    data = root / "Data"
    paths = {
        "config": get_cli_config_path(),
        "profile": get_user_data_dir(),
        "database": get_chachanotes_db_path(),
        "log": get_cli_log_file_path(),
        "model_cache": get_model_cache_dir(),
    }
    for path in paths.values():
        if not path.is_relative_to(data):
            raise RuntimeError("portable_path_escaped")
    package = Path(tldw_chatbook.__file__).absolute().parent
    if not package.is_relative_to(root / "runtime"):
        raise RuntimeError("package_is_not_bundled")
    migration_count = len(list((package / "DB" / "migrations").glob("*.sql")))
    if not migration_count:
        raise RuntimeError("packaged_migrations_missing")
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    notes = CharactersRAGDB(paths["database"], client_id="portable-packaging-probe")
    note_id = "2e5eab9b-12fe-4d87-aa63-1ccb25b75c12"
    note_created = notes.get_note_by_id(note_id) is None
    try:
        if note_created:
            notes.add_note(
                title="Portable packaging proof",
                content="This note must survive moving the complete portable folder.",
                note_id=note_id,
            )
        if notes.get_note_by_id(note_id)["content"] != (
            "This note must survive moving the complete portable folder."
        ):
            raise RuntimeError("portable_note_did_not_persist")
    finally:
        notes.close()
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "package": str(package.relative_to(root)),
                "paths": {
                    name: str(path.relative_to(root)) for name, path in paths.items()
                },
                "migration_count": migration_count,
                "chatbook_note_created": note_created,
                "chatbook_note_persisted": True,
                "sqlite": sqlite3.sqlite_version,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    import keyring
    from keyring.backends.null import Keyring

    keyring.set_keyring(Keyring())
    from tldw_chatbook.Backup_Recovery.portable_relocation import (
        WORK_NAME,
        command_main,
        read_selection,
        verified_selection,
    )

    data = Path(__file__).absolute().parent / "Data"
    selection = read_selection(data)
    if selection is not None and selection.location == str(data.parent):
        selection = verified_selection(data)
        from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile

        select_profile(
            selection.profile_id, data / WORK_NAME / selection.attempt / "control"
        )
        sys.argv = [value for value in sys.argv if value != "--portable-relocate"]
    elif "--portable-relocate" in sys.argv or selection is not None:
        raise SystemExit(command_main(data))
    if sys.argv[1:2] == ["--portable-probe"]:
        probe()
    else:
        from tldw_chatbook.cli import main_cli_runner

        main_cli_runner()
