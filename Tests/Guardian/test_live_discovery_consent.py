"""Real-store revocation of Guardian-only outbound interests."""
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from Tests.Dreams.test_cycle_service import _deps, dreams_db, settings
from tldw_chatbook.DB.Guardian_DB import GuardianDB
from tldw_chatbook.Dreams.cycle_service import _refresh_profile_signals
from tldw_chatbook.Dreams.interest_profile import snapshot
from tldw_chatbook.Dreams.query_synthesis import synthesize_queries
from tldw_chatbook.Guardian import settings as guardian_settings
from tldw_chatbook.app_service_wiring import ServiceWiringMixin


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["revoked", "crisis", "inactive", "disabled"])
async def test_revoking_live_guardian_consent_removes_only_its_contribution(
    tmp_path, dreams_db, settings, monkeypatch, change,
):
    enabled = [True]
    monkeypatch.setattr(guardian_settings, "guardian_setting",
                        lambda key, default=None: enabled[0] if key == "enabled" else default)
    guardian = GuardianDB(tmp_path / "guardian.sqlite", "consent")
    try:
        now = datetime.now(UTC)
        rule = guardian.upsert_rule(name="topic", topic="guardian private topic",
                                    pattern="x", feeds_discovery=1)
        guardian.insert_alert(rule_id=rule, session_id="a", visit_id="v",
                              topic="guardian private topic", message_digest="a" * 64)
        dreams_db.upsert_profile_entry("topic", "independent user topic", source="user", weight=0.5, searchable=1)
        dreams_db.upsert_profile_entry("topic", "independent note topic", source="notes", weight=0.5, searchable=1)
        deps = _deps(dreams_db)
        deps.guardian_db_getter = lambda: guardian
        await _refresh_profile_signals(deps, now)
        assert "guardian private topic" in json.dumps(snapshot(dreams_db, now_epoch=now.timestamp()))
        if change == "disabled":
            enabled[0] = False
        else:
            changes = {"feeds_discovery": 0} if change == "revoked" else (
                {"feeds_discovery": 0, "is_crisis": 1} if change == "crisis" else {"enabled": 0})
            guardian.upsert_rule(id=rule, **changes)
        await _refresh_profile_signals(deps, now)
        snap = snapshot(dreams_db, now_epoch=now.timestamp())
        calls = []
        def chat(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "one query\ntwo query"}}]}
        await synthesize_queries(chat, snapshot=snap, count=2, exploration_slots=1)
        assert "guardian private topic" not in json.dumps(calls)
        assert {row["text"] for row in snap["topics"]} == {
            "independent user topic", "independent note topic"}
    finally:
        guardian.close()


def test_disabled_runtime_getter_never_returns_cached_store(monkeypatch):
    monkeypatch.setattr(guardian_settings, "guardian_setting", lambda *args: False)
    app = SimpleNamespace(guardian_db=object())
    assert ServiceWiringMixin.get_guardian_db(app) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["provider", "search", "story"])
async def test_revocation_during_cycle_wait_prevents_next_external_call(
    tmp_path, dreams_db, settings, monkeypatch, boundary,
):
    from tldw_chatbook.Dreams.cycle_service import run_cycle
    from Tests.Dreams.test_cycle_service import _fake_chat, _fake_perform

    monkeypatch.setattr(guardian_settings, "guardian_setting",
                        lambda key, default=None: True if key == "enabled" else default)
    guardian = GuardianDB(tmp_path / "guardian.sqlite", "consent")
    try:
        rule = guardian.upsert_rule(name="topic", topic="guardian private topic",
                                    pattern="x", feeds_discovery=1)
        guardian.insert_alert(rule_id=rule, session_id="a", visit_id="v",
                              topic="guardian private topic", message_digest="a" * 64)
        revoked = [False]
        calls = []
        reservations = [0]
        original = dreams_db.usage_reserve

        def revoke():
            guardian.upsert_rule(id=rule, feeds_discovery=0)
            revoked[0] = True

        def reserve(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs["counter"] == "llm_calls":
                reservations[0] += 1
            if ((boundary == "search" and kwargs["counter"] == "searches")
                or (boundary == "story" and reservations[0] == 2)):
                revoke()
            return result

        def chat(**kwargs):
            calls.append(("model", revoked[0]))
            return _fake_chat(**kwargs)

        def search(engine, query, **kwargs):
            calls.append(("search", revoked[0]))
            return _fake_perform(engine, query, **kwargs)

        def resolve():
            if boundary == "provider":
                revoke()
            return chat

        monkeypatch.setattr(dreams_db, "usage_reserve", reserve)
        deps = _deps(dreams_db, perform=search)
        deps.chat_getter = resolve
        deps.guardian_db_getter = lambda: guardian
        await run_cycle(deps, trigger="scheduled")
        assert revoked[0]
        assert not any(was_revoked for _kind, was_revoked in calls), calls
    finally:
        guardian.close()
