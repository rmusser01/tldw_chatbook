"""Historical crisis classification survives rule changes and native migrations."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Guardian_DB import GuardianDB, GuardianSchemaError
from tldw_chatbook.Dreams.profile_sources import read_guardian_topics
from tldw_chatbook.Guardian import settings
from tldw_chatbook.Guardian.check_pipeline import GuardianChecker
from tldw_chatbook.Guardian.crisis_resources import render_crisis_block
from tldw_chatbook.Guardian.trend_analyzer import analyze
from tldw_chatbook.Utils.timestamps import utc_now_iso


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "guardian_setting", lambda key, default=None:
        {"enabled": True, "fixation_min_hits": 1}.get(key, default))


def _legacy(path):
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript((Path(__file__).parent / "fixtures/guardian_v1_schema.sql").read_text())
        conn.execute("INSERT INTO guardian_rules (id,name,topic,pattern,feeds_discovery,created_at,updated_at) VALUES (90,'legacy','private','private',1,'2000-01-01T00:00:00+00:00','2000-01-01T00:00:00+00:00')")
        conn.execute("INSERT INTO guardian_alerts VALUES (90,90,'session-a','visit-a','private','digest',?)", (utc_now_iso(),))
        conn.commit()


def test_populated_v1_migrates_without_guessing_historical_classification(tmp_path):
    path = tmp_path / "guardian.db"
    _legacy(path)
    with closing(sqlite3.connect(path)) as conn:
        before = conn.execute("SELECT * FROM guardian_alerts").fetchall()
    db = GuardianDB(path)
    try:
        with db.connection() as conn:
            assert conn.execute("SELECT version FROM schema_version").fetchall()[0][0] == 2
            after = [tuple(row) for row in conn.execute("SELECT * FROM guardian_alerts")]
        assert [row[:-1] for row in after] == before
        assert after[0][-1] == -1
        assert read_guardian_topics(db) == []
        notices = analyze(db, now=utc_now_iso(), visit_id="trend-a", session_id="session-a")
        assert notices and notices[0].is_crisis
        assert notices[0].message.endswith(render_crisis_block())
    finally:
        db.close()
    reopened = GuardianDB(path)
    try:
        assert read_guardian_topics(reopened) == []
    finally:
        reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("edit", ["reclassify", "delete"])
async def test_crisis_history_cannot_be_reclassified_by_current_rule(tmp_path, edit):
    db = GuardianDB(tmp_path / "guardian.db")
    rule_id = db.upsert_rule(name="private rule", topic="private", pattern="private phrase", is_crisis=1)
    checker = GuardianChecker(db_getter=lambda: db, session_id_getter=lambda: "session-a",
        notify=lambda *args: None, append_system_row=lambda text: None, now=utc_now_iso)
    try:
        outcome = await checker.check("private phrase")
        assert outcome["notice"]["text"].endswith(render_crisis_block())
        if edit == "reclassify":
            db.upsert_rule(id=rule_id, is_crisis=0, feeds_discovery=1)
        else:
            db.delete_rule(rule_id)
        assert read_guardian_topics(db) == []
        payload = await checker.finalize_visit_async()
        assert payload["is_crisis"] is True
        assert payload["per_topic_counts"] == {"private": 1}
        assert payload["trend_notices"] and all(n["is_crisis"] for n in payload["trend_notices"])
        assert all(n["message"].endswith(render_crisis_block()) for n in payload["trend_notices"])
        stored = db.get_visit_summary(payload["visit_id"])
        assert json.loads(stored["payload"])["is_crisis"] is True
        with db.connection() as conn:
            assert {row[0] for row in conn.execute("SELECT crisis_provenance FROM guardian_alerts")} == {1}
    finally:
        db.close()


@pytest.mark.parametrize("marker", [None, True])
def test_next_visit_summary_includes_resources_for_crisis_or_unproven(marker):
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    shown = []
    payload = {"session_id": "session-a", "per_topic_counts": {"private": 1}}
    if marker is not None:
        payload["is_crisis"] = marker
    app = SimpleNamespace(_guardian_visit_summary_notice={"session-a": payload},
        notify=lambda text, severity: shown.append(text))
    screen = SimpleNamespace(app_instance=app,
        _ensure_console_chat_store=lambda: SimpleNamespace(active_session_id="session-a"))
    ChatScreen._notify_guardian_visit_summary_if_any(screen)
    assert len(shown) == 1 and shown[0].endswith(render_crisis_block())


@pytest.mark.parametrize("damage", ["future", "empty", "missing", "v2-without-column", "v1-with-v2-column"])
def test_version_schema_refusal_preserves_native_rows(tmp_path, damage):
    path = tmp_path / "guardian.db"
    _legacy(path)
    with closing(sqlite3.connect(path)) as conn:
        if damage == "future":
            conn.execute("UPDATE schema_version SET version=99")
        elif damage == "empty":
            conn.execute("DELETE FROM schema_version")
        elif damage == "missing":
            conn.execute("DROP TABLE schema_version")
        elif damage == "v2-without-column":
            conn.execute("UPDATE schema_version SET version=2")
        else:
            conn.execute("ALTER TABLE guardian_alerts ADD COLUMN crisis_provenance INTEGER NOT NULL DEFAULT -1 CHECK(crisis_provenance IN (-1,0,1))")
        conn.commit()
        before = list(conn.iterdump())
    with pytest.raises(GuardianSchemaError):
        GuardianDB(path)
    with closing(sqlite3.connect(path)) as conn:
        assert list(conn.iterdump()) == before


def test_v1_migration_fault_rolls_back_and_reopens(tmp_path, monkeypatch):
    path = tmp_path / "guardian.db"
    _legacy(path)
    with closing(sqlite3.connect(path)) as conn:
        before = list(conn.iterdump())
    statements = GuardianDB._migration_statements()
    with monkeypatch.context() as patch:
        patch.setattr(GuardianDB, "_migration_statements", staticmethod(lambda: statements + ("INSERT INTO absent_guardian_table VALUES (1)",)))
        with pytest.raises(sqlite3.OperationalError):
            GuardianDB(path)
    with closing(sqlite3.connect(path)) as conn:
        assert list(conn.iterdump()) == before
    reopened = GuardianDB(path)
    try:
        with reopened.connection() as conn:
            assert conn.execute("SELECT crisis_provenance FROM guardian_alerts WHERE id=90").fetchone()[0] == -1
            assert conn.execute("SELECT updated_at FROM guardian_rules WHERE id=90").fetchone()[0] == "2000-01-01T00:00:00+00:00"
    finally:
        reopened.close()


def test_explicit_new_noncrisis_observation_remains_discovery_eligible(tmp_path):
    db = GuardianDB(tmp_path / "guardian.db")
    try:
        rule_id = db.upsert_rule(name="gardening", topic="gardening", pattern="garden", feeds_discovery=1)
        db.insert_alert(rule_id=rule_id, session_id="session-a", visit_id="visit-a", topic="gardening", message_digest="digest")
        assert [row["text"] for row in read_guardian_topics(db)] == ["gardening"]
        notices = analyze(db, now=utc_now_iso(), visit_id="trend-a", session_id="session-a")
        assert len(notices) == 1 and notices[0].is_crisis is False
        assert render_crisis_block() not in notices[0].message
    finally:
        db.close()


@pytest.mark.asyncio
async def test_captured_crisis_match_never_becomes_block_after_rule_edit(tmp_path, monkeypatch):
    db = GuardianDB(tmp_path / "guardian.db")
    rule_id = db.upsert_rule(name="private rule", topic="private", pattern="private phrase", is_crisis=1)
    checker = GuardianChecker(db_getter=lambda: db, session_id_getter=lambda: "session-a",
        notify=lambda *args: None, append_system_row=lambda text: None, now=utc_now_iso)
    original = checker._record_matches
    def change_before_record(*args):
        db.upsert_rule(id=rule_id, is_crisis=0, action="block")
        return original(*args)
    monkeypatch.setattr(checker, "_record_matches", change_before_record)
    try:
        result = await checker.check("private phrase")
        assert result["action"] == "notify"
        assert result["notice"]["is_crisis"] is True
        assert result["notice"]["text"].endswith(render_crisis_block())
    finally:
        db.close()
