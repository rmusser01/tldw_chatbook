"""Bootstrap genuinely historical ChaChaNotes DBs for migration fixtures.

Successor to the hand-maintained rollback registry (``schema_rollback.py``,
retired in task-16840). The registry rewound a current-version DB by removing
every newer migration's artifacts from a per-version drop list — knowledge
that had to be extended on EVERY schema bump (a ratchet enforced it) and that
could silently be wrong (the task-15765 review proved its parity sweep blind
to a wrong ``DROP COLUMN`` entry until column sets were added, and found a
trigger drop in the V28 entry that corrupted every V20..V27 replay target).

This module replaces that with the knowledge-free primitive the repo already
used in ``Tests/DB/test_chachanotes_note_folders_migration.py``: patch
``CharactersRAGDB._CURRENT_SCHEMA_VERSION`` to ``version`` and let the
production bootstrap build the DB. ``_initialize_schema`` applies the v4 base
schema and replays the REAL migration chain up to the patched target, so the
result is a genuinely ``version``-shaped DB: real sync triggers, real column
order, and no artifact of any later migration — which makes the "table
already exists" fixture-breakage class (task-15730/15765/16197) impossible by
construction, with zero per-version maintenance when the schema grows.

One caveat, measured in task-16840 and censused in its review: SEVEN
artifacts across four steps pre-exist their declaring migration — not only
via base drift (``conversation_local_marks`` + its index in the v4 base) but
because an EARLIER migration's DDL was retro-edited to include them
(``flashcard_templates``/``flashcard_assets``/an index declared by V15->V16
but created by V14->V15; ``world_book_entries.priority``/``regex`` declared
by V20->V21/V21->V22 but inline in V8->V9's CREATE TABLE). The mechanism is
"the artifact predates the step that declares it", base or not. The set is
static (7 today), does NOT grow with schema bumps, and each case is local to
the one test pinning that migration. A fixture that needs its migration-under-test to genuinely CREATE an
artifact the base schema also ships must drop that specific artifact itself —
knowledge about the single migration the test pins, owned by that test, which
no future schema bump can invalidate.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

SCHEMA_NAME = CharactersRAGDB._SCHEMA_NAME

#: The v4 base schema is the oldest state the production chain can build:
#: `_initialize_schema` stamps a fresh DB at 4 before replaying migrations,
#: so a target below 4 would fail its own final version check.
MINIMUM_BOOTSTRAP_VERSION = 4


@contextmanager
def chachanotes_db_at_version(
    db_path: str | os.PathLike[str],
    version: int,
    *,
    client_id: str = "historical-bootstrap",
) -> Iterator[CharactersRAGDB]:
    """Yield an open ``CharactersRAGDB`` genuinely at schema ``version``.

    Bootstraps ``db_path`` with ``_CURRENT_SCHEMA_VERSION`` patched to
    ``version``, so the production migration chain itself builds and stamps a
    real historical schema. Seed data inside the ``with`` block; the DB is
    closed on exit. Reopening the path with an unpatched ``CharactersRAGDB``
    afterwards replays the chain from ``version`` to the current version.

    Args:
        db_path: Path for the SQLite file (should not already hold a newer
            DB — opening a newer DB under a patched older version raises).
        version: The historical schema version to stop the chain at. Must be
            within ``[MINIMUM_BOOTSTRAP_VERSION, _CURRENT_SCHEMA_VERSION]``.
        client_id: Client id for the bootstrap connection.

    Yields:
        The open ``CharactersRAGDB`` instance, recorded at ``version``.

    Raises:
        AssertionError: If ``version`` is outside the supported range.
    """
    current = CharactersRAGDB._CURRENT_SCHEMA_VERSION
    assert MINIMUM_BOOTSTRAP_VERSION <= version <= current, (
        f"bootstrap version {version} must be in "
        f"[{MINIMUM_BOOTSTRAP_VERSION}, {current}]"
    )
    with patch.object(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", version):
        db = CharactersRAGDB(str(db_path), client_id=client_id)
        try:
            yield db
        finally:
            db.close_connection()
