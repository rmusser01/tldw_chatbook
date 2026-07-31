# conftest.py for Tests/ChaChaNotesDB (task-1460).
#
# CharactersRAGDB's full schema DDL (v28, FTS5 tables, triggers) costs ~137ms
# per construction; copying a pre-built template and reopening it costs
# ~10.5ms (92% less, measured on this harness). The session-scoped template is
# built once per (xdist worker's) session and function fixtures copy it.
# Tests that exercise construction/migration/versioning semantics themselves
# keep building from scratch on purpose.

import shutil
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture(scope="session")
def chachanotes_template_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the schema-complete template database once per session.

    Args:
        tmp_path_factory: pytest's session-scoped temp directory factory.

    Returns:
        Path to a closed, WAL-checkpointed template file with the current
        schema and no test rows (client_id is per-row attribution, so an
        empty template carries no identity to re-stamp).
    """
    template = tmp_path_factory.mktemp("chachanotes-template") / "template.sqlite"
    db = CharactersRAGDB(template, "template_builder")
    db.close_connection()
    leftovers = [
        s for s in ("-wal", "-shm") if Path(f"{template}{s}").exists()
    ]
    assert not leftovers, f"template close left WAL sidecars: {leftovers}"
    return template
