"""Pre-boot "upgrading database..." notice (task-21100, AC #2).

The ChaChaNotes migration chain replays inside ``TldwCli.__init__`` (thread
pool -> ``_init_notes_service``), BEFORE Textual's ``run()`` -- so nothing can
paint while it runs: no splash, no screen, nothing. The only surface that
exists at that phase is the terminal the user launched from, so the honest
minimal "upgrading database..." state is a line printed there by the entry
points right before the app object is constructed. With the task-21100 FTS
deferral the remaining chain is short, but it is not free (sync_log purges,
DDL for 12 steps), and a user on a slow disk deserves to know the pause is
deliberate.

The probe is read-only and failure-proof by construction: it opens the
database through the registered private-SQLite seam (owner
``utils.db_upgrade_notice``, read-only URI -- ADR-029; never a raw
``sqlite3.connect``), reads one row, and swallows every exception -- a
corrupt file, a locked WAL, or a missing table must surface through the real
initialization path with its real error handling, never here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, TextIO

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite


def print_db_upgrade_notice_if_pending(
    db_path: Optional[Path] = None,
    stream: Optional[TextIO] = None,
) -> bool:
    """Print an "upgrading database" line if ChaChaNotes migrations are pending.

    Args:
        db_path: The ChaChaNotes database file to probe. Defaults to the
            configured path (resolved lazily so importing this module never
            touches config).
        stream: Where to print. Defaults to ``sys.stdout``.

    Returns:
        True if a pending upgrade was detected and the notice printed;
        False otherwise (fresh install, up to date, or probe failure --
        a fresh install builds the schema from scratch quickly and gets no
        notice).
    """
    try:
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        if db_path is None:
            from tldw_chatbook.config import get_chachanotes_db_path

            db_path = get_chachanotes_db_path()
        db_path = Path(db_path)
        if not db_path.is_file():
            return False
        conn = connect_private_sqlite(
            "utils.db_upgrade_notice",
            db_path,
            read_only=True,
            must_exist=True,
            timeout=1.0,
        )
        try:
            row = conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ? LIMIT 1",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()
        finally:
            conn.close()
        target = CharactersRAGDB._CURRENT_SCHEMA_VERSION
        if row is None or row[0] >= target:
            return False
        print(
            f"Upgrading database (schema v{row[0]} -> v{target})... "
            "the app will start when the upgrade completes.",
            file=stream if stream is not None else sys.stdout,
            flush=True,
        )
        return True
    except Exception:
        # Never let the courtesy notice break or delay boot; the real
        # initialization path owns error reporting.
        return False
