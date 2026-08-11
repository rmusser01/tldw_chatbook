# test_client_media_debug_logging.py
# Description: RED-first-turned-green regression coverage for task-15474
# (lazy debug logging in Client_Media_DB_v2.py).
"""
task-15474 audit finding: `Client_Media_DB_v2.py` already had the lazy
`logger.opt(lazy=True)` + `preview_params` pattern in its shared
`execute_query` (task-246), but three call sites in `search_media_db` and
one in `get_all_document_versions` built their own eager
`logging.debug(f"... {params}")`/`logging.info(f"... {params}")` lines on
every call, ahead of / in addition to `execute_query`'s own (already-lazy)
logging. None of these particular sites carry a raw image/document BLOB
today (media content bytes are never passed as a bound *search* filter
param), but they were still unconditional `str(...)` builds of a
params-like collection on every call regardless of log level -- exactly
what this module's own `execute_query` precedent (and `DB/sql_logging.py`)
exists to avoid, and the shape a future caller could turn into a real BLOB
cost.

Two kinds of evidence, matching the two things worth proving:

1. Behavioral sanity (`TestConvertedSitesStillLogUnderDebug`): with a DEBUG
   sink explicitly attached, the converted lines still fire, so the
   conversion didn't silently delete the log line.
2. Structural regression coverage (`TestNoEagerFStringParamsRemain`): the
   method source no longer contains the old eager
   `f"...: {params}"`-style literal, and does route the same params through
   `preview_params`. This is the durable, environment-independent form of
   "no eager params stringification remains" -- loguru's own default sink
   (level DEBUG, active for the whole pytest session unless a test removes
   it) makes a call-counting/"never invoked without an explicit sink" style
   test unreliable here: that ambient sink means `opt(lazy=True)` lambdas
   legitimately DO run in this process even when a test adds no sink of its
   own, so the meaningful, stable assertion is over the source shape, not
   over invocation counts. BLOB-safety itself (bytes never repr()'d/str()'d
   by `preview_params`) is covered directly by `test_sql_debug_logging.py`,
   including for the one call site in this codebase that genuinely carries
   an image BLOB (`ChaChaNotes_DB.update_character_card`).
"""

import inspect
import sys

import pytest
from loguru import logger

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


@pytest.fixture
def db(tmp_path):
    database = MediaDatabase(db_path=tmp_path / "media.db", client_id="test-client")
    yield database
    database.close_connection()


def _seed_one_media_item(db: MediaDatabase) -> int:
    media_id, _uuid, _msg = db.add_media_with_keywords(
        title="Task 15474 Fixture",
        media_type="article",
        content="Content mentioning fifteen thousand four hundred seventy four.",
        keywords=["task15474"],
    )
    assert media_id is not None
    return media_id


class TestConvertedSitesStillLogUnderDebug:
    """Sanity check: the lazy conversion didn't silently delete these lines."""

    def test_search_media_db_log_lines_fire_under_debug(self, db, capsys):
        _seed_one_media_item(db)
        sink_id = logger.add(sys.stderr, level="DEBUG")
        try:
            results, total = db.search_media_db(search_query="fifteen thousand")
        finally:
            logger.remove(sink_id)
        assert total >= 1
        captured = capsys.readouterr().err
        assert "Search Count Params" in captured
        assert "Search Results Params" in captured
        assert "Search using FTS with query parts" in captured
        assert "Search using LIKE with patterns" in captured

    def test_get_all_document_versions_log_line_fires_under_debug(self, db, capsys):
        media_id = _seed_one_media_item(db)
        sink_id = logger.add(sys.stderr, level="DEBUG")
        try:
            db.get_all_document_versions(media_id)
        finally:
            logger.remove(sink_id)
        captured = capsys.readouterr().err
        assert "Executing get_all_document_versions query" in captured


class TestNoEagerFStringParamsRemain:
    """Structural regression coverage: the old eager `f"...: {params}"`
    literals are gone, and the same values now go through `preview_params`.

    Reads live source via `inspect.getsource` rather than re-grepping the
    file on disk, so this test breaks (not silently passes) if a refactor
    moves the code without preserving the guarantee.
    """

    def test_search_media_db_no_eager_params_fstrings(self):
        source = inspect.getsource(MediaDatabase.search_media_db)
        for eager_literal in (
            '{params}")',
            "{params})",
            "{paginated_params}",
            "{fts_query_parts}",
            "{like_params}",
        ):
            assert eager_literal not in source, (
                f"Eager params f-string literal {eager_literal!r} reintroduced "
                "into search_media_db -- route it through preview_params "
                "under logger.opt(lazy=True) instead (task-15474)."
            )
        assert source.count("preview_params(") >= 3, (
            "Expected preview_params() to cover the count/results/LIKE "
            "params sites in search_media_db."
        )
        assert "logger.opt(lazy=True)" in source

    def test_get_all_document_versions_no_eager_params_fstring(self):
        source = inspect.getsource(MediaDatabase.get_all_document_versions)
        assert "{params}" not in source, (
            "Eager params f-string literal reintroduced into "
            "get_all_document_versions -- route it through preview_params "
            "under logger.opt(lazy=True) instead (task-15474)."
        )
        assert "preview_params(params)" in source
        assert "logger.opt(lazy=True)" in source
