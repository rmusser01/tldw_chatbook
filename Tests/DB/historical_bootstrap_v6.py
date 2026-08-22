"""Bootstrap genuinely historical Media DBs for migration fixtures.

The ``Tests/ChaChaNotesDB/historical_bootstrap.py`` pattern, applied to
``MediaDatabase`` (task-7, spec 2026-08-21 chunking-template-parity §5.3):
patch ``MediaDatabase._CURRENT_SCHEMA_VERSION`` to ``version`` and let the
production bootstrap build the DB. ``_initialize_schema`` applies the v1 base
schema and replays the REAL migration chain up to the patched target, so the
result is a genuinely ``version``-shaped DB — real five-seed
``ChunkingTemplates`` rows, real trigger, real column set — never a
hand-rolled "drop the new artifacts and stamp the version back" fixture.
That hand-stamped style is explicitly forbidden for the v6→v7 conversion
tests (AC 19): it broke serially across four repair tasks in this repo.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

#: The chain boots a fresh file at version 0 and stamps v1 via
#: ``_apply_schema_v1``; anything below 1 has nothing to replay from.
MINIMUM_BOOTSTRAP_VERSION = 1


@contextmanager
def media_db_at_version(
    db_path: str | os.PathLike[str],
    version: int,
    *,
    client_id: str = "historical-bootstrap",
) -> Iterator[MediaDatabase]:
    """Yield an open ``MediaDatabase`` genuinely at schema ``version``.

    Bootstraps ``db_path`` with ``_CURRENT_SCHEMA_VERSION`` patched to
    ``version``, so the production migration chain itself builds and stamps a
    real historical schema. Seed data inside the ``with`` block; the DB is
    closed on exit. Reopening the path with an unpatched ``MediaDatabase``
    afterwards replays the chain from ``version`` to the current version.

    Args:
        db_path: Path for the SQLite file (must not already hold a newer DB —
            opening a newer DB under a patched older version raises).
        version: The historical schema version to stop the chain at. Must be
            within ``[MINIMUM_BOOTSTRAP_VERSION, _CURRENT_SCHEMA_VERSION]``.
        client_id: Client id for the bootstrap connection.

    Yields:
        The open ``MediaDatabase`` instance, recorded at ``version``.

    Raises:
        AssertionError: If ``version`` is outside the supported range.
    """
    current = MediaDatabase._CURRENT_SCHEMA_VERSION
    assert MINIMUM_BOOTSTRAP_VERSION <= version <= current, (
        f"bootstrap version {version} must be in "
        f"[{MINIMUM_BOOTSTRAP_VERSION}, {current}]"
    )
    with patch.object(MediaDatabase, "_CURRENT_SCHEMA_VERSION", version):
        db = MediaDatabase(str(db_path), client_id=client_id)
        try:
            yield db
        finally:
            db.close_connection()


@contextmanager
def media_db_at_v6(
    db_path: str | os.PathLike[str],
    *,
    client_id: str = "historical-bootstrap",
) -> Iterator[MediaDatabase]:
    """Yield an open ``MediaDatabase`` genuinely at schema v6 (AC 19 fixture).

    The v6 shape the v6→v7 rebuild is tested against: built by the real
    chain (base v1 + every migration through v5→v6), carrying the five
    original ``is_system = 1`` chunking-template seeds.
    """
    with media_db_at_version(db_path, 6, client_id=client_id) as db:
        yield db
