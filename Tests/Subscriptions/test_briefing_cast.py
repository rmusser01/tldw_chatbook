"""Tests for the cast service (spec #2 phase 2a, Task 2).

`briefing_cast` casts an already-`complete` briefing into an N-speaker
script: a strict JSON array of `{"speaker", "text"}` turns. Two rules shape
every test here, both load-bearing enough to be named invariants in the
plan:

- **Validation is strict and failure is honest, by name.** An unknown
  speaker, a malformed payload, or a roster naming a deleted character card
  each fail naming the specific defect -- never a generic "cast failed".
- **The briefing is never touched by a script's outcome.** Every
  `generate_script` test that reaches a `briefings` row asserts this
  directly (byte-for-byte dict equality before/after), not merely that the
  script row looks right.

Same testing rule as `briefing_service` (spec #Testing): the only faked
seam is `chat`; `load_character` is a plain Python callable (never a real
DB in these tests, so no second seam is actually being faked -- it is
exercised as an ordinary dict lookup). Everything else is a real,
file-backed `SubscriptionsDB` (required by `generate_script`'s own
`asyncio.to_thread` hops -- see `_db`, copied from
`test_briefing_service.py`'s own docstring explaining why `:memory:` won't
do for an async, thread-hopping caller).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timezone

import pytest
from loguru import logger

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import briefing_cast
from tldw_chatbook.Subscriptions.briefing_cast import (
    ScriptCastError,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_GENERATING,
    VALID_STATUSES,
    build_cast_prompt,
    dump_roster,
    fail_interrupted_scripts,
    generate_script,
    load_roster,
    parse_script_turns,
    validate_roster,
)
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit


TWO_SPEAKER_ROSTER = [
    {"name": "Host", "role_prompt": "Warm, curious interviewer."},
    {"name": "Analyst", "role_prompt": "Dry, precise, cites specifics."},
]

ONE_SPEAKER_ROSTER = [{"name": "Narrator", "role_prompt": "Calm, even narration."}]

CANNED_TURNS = [
    {"speaker": "Host", "text": "Welcome back -- what happened this week?"},
    {"speaker": "Analyst", "text": "Acme shipped a thing, per item 1."},
]


class _FakeChat:
    """The one faked seam, mirroring `test_briefing_service._FakeChat` exactly."""

    def __init__(self, *, reply: object = None, error: Exception | None = None):
        self.reply = json.dumps(CANNED_TURNS) if reply is None else reply
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.reply


def _db(tmp_path) -> SubscriptionsDB:
    """A real, file-backed `SubscriptionsDB` -- see the module docstring."""
    return SubscriptionsDB(tmp_path / "subs.db", "test")


def _complete_briefing(db, watchlist_id, *, body: str = "## This week\n\nSomething happened.\n") -> int:
    """A `complete` `briefings` row with a body -- the only status a cast may start from."""
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(briefing_id, status="complete", body_markdown=body)
    return briefing_id


def _preset(
    db,
    *,
    roster: list[dict] = TWO_SPEAKER_ROSTER,
    style_notes: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    name: str = "Duo",
) -> int:
    return db.insert_briefing_preset(
        name,
        roster_json=dump_roster(roster),
        style_notes=style_notes,
        provider=provider,
        model=model,
    )


# --- validate_roster ---------------------------------------------------------


def test_validate_roster_rejects_duplicate_speaker_name():
    roster = [{"name": "Host"}, {"name": "Host"}]
    with pytest.raises(ScriptCastError, match="Host"):
        validate_roster(roster)


def test_validate_roster_rejects_an_empty_roster():
    with pytest.raises(ScriptCastError):
        validate_roster([])


def test_validate_roster_accepts_a_single_speaker_roster():
    """Spec: "a roster of one produces narration through the identical path"."""
    normalized = validate_roster(ONE_SPEAKER_ROSTER)
    assert normalized == [
        {
            "name": "Narrator",
            "role_prompt": "Calm, even narration.",
            "character_card_id": None,
            "voice_profile_id": None,
        }
    ]


def test_validate_roster_rejects_non_list_input():
    with pytest.raises(ScriptCastError):
        validate_roster({"name": "Host"})
    with pytest.raises(ScriptCastError):
        validate_roster("not a roster")


def test_validate_roster_normalizes_character_card_id_and_voice_profile_id():
    normalized = validate_roster(
        [{"name": "Host", "character_card_id": "7", "voice_profile_id": 42}]
    )
    assert normalized[0]["character_card_id"] == 7
    assert normalized[0]["voice_profile_id"] == "42"


def test_dump_roster_and_load_roster_round_trip():
    roster = validate_roster(TWO_SPEAKER_ROSTER)
    assert load_roster(dump_roster(roster)) == roster


def test_load_roster_rejects_junk():
    with pytest.raises(ScriptCastError):
        load_roster("not json")
    with pytest.raises(ScriptCastError):
        load_roster('{"not": "a list"}')
    with pytest.raises(ScriptCastError):
        load_roster('["a string, not a speaker object"]')


# --- parse_script_turns -------------------------------------------------------


def test_parse_script_turns_valid_array_round_trips():
    turns = parse_script_turns(json.dumps(CANNED_TURNS), {"Host", "Analyst"})
    assert turns == CANNED_TURNS


def test_parse_script_turns_recovers_a_fenced_json_array():
    fenced = "```json\n" + json.dumps(CANNED_TURNS) + "\n```"
    assert parse_script_turns(fenced, {"Host", "Analyst"}) == CANNED_TURNS


def test_parse_script_turns_recovers_a_prose_wrapped_array_via_slice():
    prose = "Sure, here is the script:\n" + json.dumps(CANNED_TURNS) + "\nHope that helps!"
    assert parse_script_turns(prose, {"Host", "Analyst"}) == CANNED_TURNS


def test_an_unknown_speaker_fails_the_script_by_name():
    """Named invariant (spec + plan): an unknown speaker names ITSELF."""
    turns = [{"speaker": "Dave", "text": "Hi, I'm not on the roster."}]
    with pytest.raises(ScriptCastError, match="Dave"):
        parse_script_turns(json.dumps(turns), {"Host", "Analyst"})


def test_parse_script_turns_non_string_text_fails_naming_the_turn_index():
    turns = [
        {"speaker": "Host", "text": "fine"},
        {"speaker": "Analyst", "text": 12345},
    ]
    with pytest.raises(ScriptCastError, match="turn 1"):
        parse_script_turns(json.dumps(turns), {"Host", "Analyst"})


def test_parse_script_turns_rejects_a_non_array_payload():
    with pytest.raises(ScriptCastError):
        parse_script_turns(json.dumps({"speaker": "Host", "text": "hi"}), {"Host"})


def test_parse_script_turns_rejects_a_turn_missing_required_keys():
    with pytest.raises(ScriptCastError):
        parse_script_turns(json.dumps([{"speaker": "Host"}]), {"Host"})


def test_parse_script_turns_rejects_junk_text():
    with pytest.raises(ScriptCastError):
        parse_script_turns("not json at all", {"Host"})


# --- build_cast_prompt --------------------------------------------------------


def test_build_cast_prompt_includes_roster_style_notes_and_output_contract():
    roster = validate_roster(TWO_SPEAKER_ROSTER)
    system, user = build_cast_prompt(
        "## Body\n\nThe thing.",
        roster,
        "Keep it under two minutes.",
        {"Host": "warm, quick to laugh"},
    )
    assert "Host" in system and "Analyst" in system
    assert "Warm, curious interviewer." in system
    assert "warm, quick to laugh" in system
    assert "Keep it under two minutes." in system
    assert "JSON array" in system
    assert "Host" in system.split("speaker must be one of:")[-1]
    assert user == "## Body\n\nThe thing."


# --- generate_script -----------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_script_happy_path_writes_everything(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist, body="## Body\n\nAcme shipped a thing.")
    preset_id = _preset(db, roster=TWO_SPEAKER_ROSTER, style_notes="Keep it brisk.")

    chat = _FakeChat()
    row = await generate_script(db, briefing_id, preset_id=preset_id, chat=chat)

    assert row["status"] == STATUS_COMPLETE
    assert row["error"] is None
    assert json.loads(row["turns_json"]) == CANNED_TURNS
    assert row["model_used"]
    assert row["preset_id"] == preset_id
    assert row["preset_name"] == "Duo"
    snapshot = load_roster(row["roster_snapshot_json"])
    assert [speaker["name"] for speaker in snapshot] == ["Host", "Analyst"]

    # Exactly one call, non-streaming, style notes reached the system prompt.
    assert len(chat.calls) == 1
    call = chat.calls[0]
    assert call["streaming"] is False
    assert "Keep it brisk." in call["system_message"]
    assert call["messages_payload"][0]["content"] == "## Body\n\nAcme shipped a thing."

    # Findable via the listing interface, newest first.
    assert db.list_briefing_scripts(briefing_id)[0]["id"] == row["id"]


@pytest.mark.asyncio
async def test_generate_script_snapshot_embeds_the_resolved_character_name(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    roster = [{"name": "Host", "role_prompt": "Warm.", "character_card_id": 7}]
    preset_id = _preset(db, roster=roster)

    def _load_character(card_id):
        assert card_id == 7
        return {"name": "Ada", "personality": "curious", "description": "a host"}

    row = await generate_script(
        db, briefing_id, preset_id=preset_id, chat=_FakeChat(reply=json.dumps(
            [{"speaker": "Host", "text": "Hello!"}]
        )), load_character=_load_character
    )

    assert row["status"] == STATUS_COMPLETE
    snapshot = load_roster(row["roster_snapshot_json"])
    assert snapshot[0]["character_name"] == "Ada"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["generating", "empty", "failed"])
async def test_generate_script_refuses_when_the_briefing_is_not_complete(tmp_path, status):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = db.insert_briefing(watchlist)
    if status != "generating":
        db.update_briefing(briefing_id, status=status)
    preset_id = _preset(db)

    with pytest.raises(ScriptCastError, match=status):
        await generate_script(db, briefing_id, preset_id=preset_id, chat=_FakeChat())

    assert db.list_briefing_scripts(briefing_id) == [], "no row on a pre-flight refusal"


@pytest.mark.asyncio
async def test_generate_script_refuses_when_the_preset_is_missing(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)

    with pytest.raises(ScriptCastError, match="9999"):
        await generate_script(db, briefing_id, preset_id=9999, chat=_FakeChat())

    assert db.list_briefing_scripts(briefing_id) == [], "no row on a pre-flight refusal"


@pytest.mark.asyncio
async def test_generate_script_fails_naming_the_card_when_load_character_returns_none(tmp_path):
    """Spec: "fails the cast at that point, naming the card"."""
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    roster = [{"name": "Host", "role_prompt": "Warm.", "character_card_id": 7}]
    preset_id = _preset(db, roster=roster)

    row = await generate_script(
        db, briefing_id, preset_id=preset_id, chat=_FakeChat(), load_character=lambda _card_id: None
    )

    assert row["status"] == STATUS_FAILED
    assert "7" in row["error"]
    assert db.list_briefing_scripts(briefing_id)[0]["id"] == row["id"]


@pytest.mark.asyncio
async def test_generate_script_fails_naming_the_card_when_load_character_is_none(tmp_path):
    """Same failed-naming path when no character lookup is available at all."""
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    roster = [{"name": "Host", "role_prompt": "Warm.", "character_card_id": 7}]
    preset_id = _preset(db, roster=roster)

    row = await generate_script(db, briefing_id, preset_id=preset_id, chat=_FakeChat())

    assert row["status"] == STATUS_FAILED
    assert "7" in row["error"]


@pytest.mark.asyncio
async def test_generate_script_chat_failure_is_a_failed_row_and_leaves_the_briefing_untouched(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)

    before = db.get_briefing(briefing_id)
    boom = _FakeChat(error=RuntimeError("provider exploded: 503 upstream"))
    row = await generate_script(db, briefing_id, preset_id=preset_id, chat=boom)
    after = db.get_briefing(briefing_id)

    assert row["status"] == STATUS_FAILED
    assert "provider exploded: 503 upstream" in row["error"]
    assert "Traceback" not in row["error"]
    # THE named invariant: the briefing is never touched by a script outcome.
    assert before == after


@pytest.mark.asyncio
async def test_generate_script_parse_failure_is_a_failed_row_naming_the_defect(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)

    row = await generate_script(
        db, briefing_id, preset_id=preset_id, chat=_FakeChat(reply="not a JSON array at all")
    )

    assert row["status"] == STATUS_FAILED
    assert row["error"]
    assert "Traceback" not in row["error"]


@pytest.mark.asyncio
async def test_generate_script_logs_no_cast_content_on_failure(tmp_path):
    """Egress pin, mirroring `test_a_failed_generation_logs_no_item_content`.

    This app's file sink runs `diagnose=True`, which dumps a failing
    frame's locals -- and the frame at the cast failure holds the prompt.
    """
    canary = "ZEBRACANARY"
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist, body=f"## Body\n\n{canary}\n")
    preset_id = _preset(db)

    captured: list[str] = []
    handler = logger.add(captured.append, level="DEBUG", diagnose=True, backtrace=True, catch=False)
    try:
        row = await generate_script(
            db, briefing_id, preset_id=preset_id,
            chat=_FakeChat(error=RuntimeError("upstream 503")),
        )
    finally:
        logger.remove(handler)

    assert row["status"] == STATUS_FAILED
    log_text = "".join(captured)
    assert log_text
    assert "cast failed" in log_text
    assert "RuntimeError" in log_text
    assert canary not in log_text
    assert "messages_payload" not in log_text


@pytest.mark.asyncio
async def test_generate_script_propagates_a_real_db_error(tmp_path, monkeypatch):
    """`generate_script` must not swallow a genuine DB failure into a row.

    Simulated with a real, closed `sqlite3.Connection` rather than merely
    monkeypatching a DB method to raise: `_get_connection` is replaced so
    that ANY thread's first `.conn` access (including the executor thread
    `asyncio.to_thread` dispatches onto -- `SubscriptionsDB.conn` is
    thread-local) reaches the same already-closed connection, sidestepping
    the nondeterminism of whether the default executor happens to reuse the
    calling thread.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)

    closed_connection = db._get_connection()
    closed_connection.close()
    monkeypatch.setattr(db, "_get_connection", lambda: closed_connection)
    db._local = threading.local()  # evict every thread's cached (open) connection

    with pytest.raises(sqlite3.ProgrammingError):
        await generate_script(db, briefing_id, preset_id=preset_id, chat=_FakeChat())


@pytest.mark.asyncio
async def test_generate_script_db_work_runs_off_the_event_loop_thread(tmp_path):
    """Mirrors `test_the_db_work_runs_off_the_event_loop_thread` (phase 1):
    `generate_script` is dispatched from a Textual worker in the real
    caller (Task 5), so its DB work must not run directly on the event
    loop -- an untested regression here would silently reintroduce the
    exact bug phase 1's "whole-branch review fix 1" fixed for
    `generate_briefing`.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)

    loop_thread_id = threading.get_ident()
    write_thread_ids: list[int] = []

    for name in ("get_briefing", "get_briefing_preset", "insert_briefing_script", "update_briefing_script"):
        original = getattr(db, name)

        def _spy(*args, __original=original, **kwargs):
            write_thread_ids.append(threading.get_ident())
            return __original(*args, **kwargs)

        setattr(db, name, _spy)

    row = await generate_script(db, briefing_id, preset_id=preset_id, chat=_FakeChat())

    assert row["status"] == STATUS_COMPLETE
    assert len(write_thread_ids) >= 4
    assert all(tid != loop_thread_id for tid in write_thread_ids)


# --- fail_interrupted_scripts --------------------------------------------------


def test_fail_interrupted_scripts_only_touches_generating_rows(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    other_briefing_id = _complete_briefing(db, watchlist)

    zombie = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Duo", roster_snapshot_json="[]"
    )
    other_zombie = db.insert_briefing_script(
        other_briefing_id, preset_id=None, preset_name="Duo", roster_snapshot_json="[]"
    )
    done = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Duo", roster_snapshot_json="[]"
    )
    db.update_briefing_script(done, status="complete", turns_json="[]")
    already_failed = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Duo", roster_snapshot_json="[]"
    )
    db.update_briefing_script(already_failed, status="failed", error="provider said no")

    assert fail_interrupted_scripts(db, briefing_id=briefing_id) == 1

    assert db.get_briefing_script(zombie)["status"] == "failed"
    assert db.get_briefing_script(zombie)["error"] == "interrupted"
    assert db.get_briefing_script(other_zombie)["status"] == "generating"
    assert db.get_briefing_script(done)["status"] == "complete"
    assert db.get_briefing_script(already_failed)["error"] == "provider said no"

    assert fail_interrupted_scripts(db, briefing_id=briefing_id) == 0

    assert fail_interrupted_scripts(db) == 1
    assert db.get_briefing_script(other_zombie)["status"] == "failed"
    assert db.get_briefing_script(other_zombie)["error"] == "interrupted"
