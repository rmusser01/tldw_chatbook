# test_sql_debug_logging.py
# Description: RED-first regression coverage for task-246 (lazy DB debug logging).
"""
Task-246: the debug-log line in the DB layer's ``execute_query`` methods used
to build an eager f-string -- including ``str(params)`` -- on *every* query,
regardless of whether debug logging was actually enabled. For a BLOB param
(e.g. an image message insert) this could cost double-digit milliseconds per
query for a string that is thrown away unread.

These tests prove:
  * a BLOB-like param is never stringified in full at the default log level;
  * it is *still* never stringified in full even with a DEBUG sink attached
    (lazy logging only defers the *decision*; once made, the preview helper
    must summarize bytes rather than repr() them);
  * the shared ``preview_params`` helper produces the documented shapes.
"""

import sys

import pytest
from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.sql_logging import preview_params


class CountingBytes(bytes):
    """A bytes subclass that counts how many times __repr__ is called.

    Stringifying a large ``bytes`` value (via ``str()``/``repr()``/an
    f-string) is the exact cost this task eliminates, so the counter is the
    ground truth for "was this BLOB ever stringified".
    """

    repr_calls = 0

    def __repr__(self) -> str:  # pragma: no cover - trivial
        CountingBytes.repr_calls += 1
        return super().__repr__()

    def __str__(self) -> str:  # pragma: no cover - trivial
        CountingBytes.repr_calls += 1
        return super().__str__()


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
    yield database
    database.close_connection()


@pytest.fixture(autouse=True)
def _reset_counter():
    CountingBytes.repr_calls = 0
    yield
    CountingBytes.repr_calls = 0


def _make_blob_param(size: int = 1024) -> CountingBytes:
    return CountingBytes(b"x" * size)


class TestNoStringifyAtDefaultLevel:
    def test_blob_param_not_stringified_at_default_log_level(self, db):
        """No debug sink attached -> the BLOB must never be repr()'d/str()'d."""
        blob = _make_blob_param()
        db.execute_query(
            "CREATE TABLE IF NOT EXISTS scratch_blob (id INTEGER PRIMARY KEY, data BLOB)"
        )
        db.execute_query(
            "INSERT INTO scratch_blob (id, data) VALUES (1, ?)",
            (blob,),
            commit=True,
        )
        assert CountingBytes.repr_calls == 0

    def test_blob_param_not_stringified_even_with_debug_sink_attached(self, db, capsys):
        """A DEBUG sink IS attached -- the log line fires, but the preview
        helper must summarize the BLOB by length, never repr()/str() it."""
        blob = _make_blob_param()
        sink_id = logger.add(sys.stderr, level="DEBUG")
        try:
            db.execute_query(
                "CREATE TABLE IF NOT EXISTS scratch_blob2 (id INTEGER PRIMARY KEY, data BLOB)"
            )
            db.execute_query(
                "INSERT INTO scratch_blob2 (id, data) VALUES (1, ?)",
                (blob,),
                commit=True,
            )
        finally:
            logger.remove(sink_id)
        assert CountingBytes.repr_calls == 0

    def test_debug_log_line_is_actually_emitted_when_enabled(self, db, capsys):
        """Sanity check that the lazy form still logs something at DEBUG,
        so the "never emits" case above isn't passing by accident (e.g. the
        debug call was deleted rather than made lazy)."""
        sink_id = logger.add(sys.stderr, level="DEBUG")
        try:
            db.execute_query(
                "CREATE TABLE IF NOT EXISTS scratch_plain (id INTEGER PRIMARY KEY, val TEXT)"
            )
            db.execute_query(
                "INSERT INTO scratch_plain (id, val) VALUES (1, ?)",
                ("hello",),
                commit=True,
            )
        finally:
            logger.remove(sink_id)
        captured = capsys.readouterr()
        assert "Executing SQL" in captured.err


class TestPreviewParamsHelperShapes:
    def test_none_params(self):
        assert preview_params(None) == "None"

    def test_tuple_of_scalars(self):
        assert preview_params((1, "abc", None, True)) == "(1, abc, None, True)"

    def test_bytes_param_summarized_by_length_never_repr(self):
        preview = preview_params((b"x" * 5_000_000,))
        assert "5000000 bytes" in preview
        assert "x" not in preview  # never contains the actual byte content

    def test_bytearray_and_memoryview_summarized(self):
        assert "<3 bytes>" in preview_params((bytearray(b"abc"),))
        assert "<3 bytes>" in preview_params((memoryview(b"abc"),))

    def test_long_string_truncated(self):
        long_str = "y" * 500
        preview = preview_params((long_str,))
        assert "500 chars" in preview
        assert len(preview) < 250

    def test_dict_params(self):
        preview = preview_params({"a": 1, "b": "x"})
        assert preview.startswith("{")
        assert "a=1" in preview
        assert "b=x" in preview

    def test_whole_preview_capped_even_with_many_small_params(self):
        many_params = tuple(str(i) for i in range(500))
        preview = preview_params(many_params)
        assert len(preview) <= 210  # _MAX_PREVIEW_CHARS + "..." slack

    def test_unreprable_params_do_not_raise(self):
        class Explodes:
            def __repr__(self):
                raise RuntimeError("boom")

        # Must not raise -- a broken repr() on a param must never break the
        # query/logging path.
        preview_params((Explodes(),))
