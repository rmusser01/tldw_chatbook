"""Exact BASE: soft deletion then native purge corrupts a shared FTS posting."""
import pytest
from Tests.Chat.test_console_trace_runtime import make_database as make_database
from tldw_chatbook.Canvas.repository import CanvasRepository

pytestmark = pytest.mark.bootstrap_profile


def test_soft_delete_then_purge_keeps_other_conversation_usable(make_database):
    db = make_database(":memory:", "mixin-baseline-probe")
    first = db.add_conversation({"title": "Mix-ins"})
    second = db.add_conversation({"title": "Mix-ins"})
    assert db._CURRENT_SCHEMA_VERSION == 81
    assert db.soft_delete_conversation(first, expected_version=1)
    CanvasRepository(db).hard_purge_conversation(first)
    assert db.soft_delete_conversation(second, expected_version=1)
