"""Privacy regressions for Media database diagnostics.

Search and type-facet reads may log bounded operation metadata, but never
queries, row values, stable IDs, database paths, SQL parameters, or raw error
text. The document-version assertions retain task-15474's lazy logging check.
"""

import inspect
import io
import logging
import sqlite3
import sys

import pytest
from loguru import logger

from tldw_chatbook.DB import Client_Media_DB_v2 as media_db_module
from tldw_chatbook.DB.Client_Media_DB_v2 import DatabaseError, MediaDatabase


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

    def test_search_media_db_logs_only_metadata_under_debug(self, db):
        query_sentinel = "PRIVATE_QUERY_TASK_16483"
        title_sentinel = "PRIVATE_TITLE_TASK_16483"
        sequence_sentinel = 987_654_320
        sequence_cursor = db.get_connection().execute(
            "UPDATE sqlite_sequence SET seq = ? WHERE name = 'Media'",
            (sequence_sentinel,),
        )
        if sequence_cursor.rowcount == 0:
            db.get_connection().execute(
                "INSERT INTO sqlite_sequence(name, seq) VALUES ('Media', ?)",
                (sequence_sentinel,),
            )
        db.get_connection().commit()
        media_id, _uuid, _msg = db.add_media_with_keywords(
            title=title_sentinel,
            media_type="article",
            content=query_sentinel,
            keywords=[],
        )
        assert media_id is not None
        loguru_output = io.StringIO()
        stdlib_output = io.StringIO()
        stdlib_handler = logging.StreamHandler(stdlib_output)
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(stdlib_handler)
        sink_id = logger.add(loguru_output, level="DEBUG")
        try:
            results, total = db.search_media_db(
                search_query=query_sentinel,
                results_per_page=20,
                offset=0,
                library_summary=True,
            )
        finally:
            logger.remove(sink_id)
            root_logger.removeHandler(stdlib_handler)
            root_logger.setLevel(previous_level)
        assert total == 1
        assert len(results) == 1
        captured = loguru_output.getvalue() + stdlib_output.getvalue()
        assert "Media search completed" in captured
        for private_value in (
            query_sentinel,
            title_sentinel,
            str(media_id),
            db.db_path_str,
        ):
            assert private_value not in captured

    def test_search_media_db_error_logs_only_fixed_metadata(self, db):
        query_sentinel = "PRIVATE_ERROR_QUERY_TASK_16483"
        raw_error_sentinel = "no such table: media_fts"
        db.get_connection().execute("DROP TABLE media_fts")
        db.get_connection().commit()
        loguru_output = io.StringIO()
        stdlib_output = io.StringIO()
        stdlib_handler = logging.StreamHandler(stdlib_output)
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(stdlib_handler)
        sink_id = logger.add(loguru_output, level="DEBUG")
        try:
            with pytest.raises(DatabaseError, match="Media search failed"):
                db.search_media_db(query_sentinel, offset=0, library_summary=True)
        finally:
            logger.remove(sink_id)
            root_logger.removeHandler(stdlib_handler)
            root_logger.setLevel(previous_level)
        captured = loguru_output.getvalue() + stdlib_output.getvalue()
        assert "Media search failed" in captured
        for private_value in (
            query_sentinel,
            raw_error_sentinel,
            db.db_path_str,
        ):
            assert private_value not in captured

    def test_distinct_media_type_logs_are_metadata_only(self, db):
        type_sentinel = "PRIVATE_TYPE_TASK_16483"
        raw_error_sentinel = "no such table: Media"
        db.add_media_with_keywords(
            title="Private type fixture",
            media_type=type_sentinel,
            content="private type fixture body",
            keywords=[],
        )
        loguru_output = io.StringIO()
        stdlib_output = io.StringIO()
        stdlib_handler = logging.StreamHandler(stdlib_output)
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(stdlib_handler)
        sink_id = logger.add(loguru_output, level="DEBUG")
        try:
            assert db.get_distinct_media_types() == [type_sentinel]
            db.get_connection().execute("DROP TABLE Media")
            db.get_connection().commit()
            with pytest.raises(DatabaseError, match="Failed to fetch distinct"):
                db.get_distinct_media_types()
        finally:
            logger.remove(sink_id)
            root_logger.removeHandler(stdlib_handler)
            root_logger.setLevel(previous_level)
        captured = loguru_output.getvalue() + stdlib_output.getvalue()
        assert "Distinct media types loaded" in captured
        assert "Distinct media types failed" in captured
        for private_value in (
            type_sentinel,
            raw_error_sentinel,
            db.db_path_str,
        ):
            assert private_value not in captured

    def test_closed_connection_reopens_without_private_diagnostics(self, db):
        database_path = db.db_path_str
        connection = db.get_connection()
        connection.close()
        db._local.conn_last_used = None
        loguru_output = io.StringIO()
        stdlib_output = io.StringIO()
        stdlib_handler = logging.StreamHandler(stdlib_output)
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(stdlib_handler)
        sink_id = logger.add(loguru_output, level="DEBUG")
        try:
            rows, total = db.search_media_db(None, library_summary=True)
        finally:
            logger.remove(sink_id)
            root_logger.removeHandler(stdlib_handler)
            root_logger.setLevel(previous_level)

        assert rows == []
        assert total == 0
        captured = loguru_output.getvalue() + stdlib_output.getvalue()
        assert "Media database connection was closed; reopening." in captured
        assert database_path not in captured
        assert "test-client" not in captured

    @pytest.mark.parametrize("operation", ["search", "types"])
    def test_connection_open_failure_is_wrapped_without_private_diagnostics(
        self, db, monkeypatch, operation
    ):
        error_sentinel = "PRIVATE_CONNECTION_ERROR_TASK_16483"
        database_path = db.db_path_str
        db.close_connection()

        def fail_connection(*_args, **_kwargs):
            raise sqlite3.OperationalError(
                f"{error_sentinel} path={database_path} credential=PRIVATE_TOKEN"
            )

        monkeypatch.setattr(
            media_db_module, "connect_private_sqlite", fail_connection
        )
        loguru_output = io.StringIO()
        stdlib_output = io.StringIO()
        stdlib_handler = logging.StreamHandler(stdlib_output)
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(stdlib_handler)
        sink_id = logger.add(loguru_output, level="DEBUG")
        try:
            expected_message = (
                "Media search failed"
                if operation == "search"
                else "Failed to fetch distinct media types"
            )
            with pytest.raises(DatabaseError, match=expected_message) as raised:
                if operation == "search":
                    db.search_media_db(None, library_summary=True)
                else:
                    db.get_distinct_media_types()
        finally:
            logger.remove(sink_id)
            root_logger.removeHandler(stdlib_handler)
            root_logger.setLevel(previous_level)

        captured = loguru_output.getvalue() + stdlib_output.getvalue()
        assert "error_type=OperationalError" in captured
        assert raised.value.__cause__ is None
        for private_value in (
            error_sentinel,
            database_path,
            "PRIVATE_TOKEN",
            "test-client",
        ):
            assert private_value not in captured
            assert private_value not in str(raised.value)

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
                "into search_media_db -- log metadata only."
            )
        assert "preview_params(" not in source
        assert "self.db_path_str" not in source
        assert "exception=True" not in source

    def test_get_all_document_versions_no_eager_params_fstring(self):
        source = inspect.getsource(MediaDatabase.get_all_document_versions)
        assert "{params}" not in source, (
            "Eager params f-string literal reintroduced into "
            "get_all_document_versions -- route it through preview_params "
            "under logger.opt(lazy=True) instead (task-15474)."
        )
        assert "preview_params(params)" in source
        assert "logger.opt(lazy=True)" in source
