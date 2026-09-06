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
from pathlib import Path
import sqlite3
import threading

import pytest
from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import briefing_cast, briefing_service
from tldw_chatbook.Subscriptions.briefing_cast import (
    APP_DEFAULT_PRESET_NAME,
    ScriptCastError,
    STATUS_COMPLETE,
    STATUS_FAILED,
    active_cast_claim_row_ids,
    active_cast_claims,
    active_kept_cast_claims,
    build_cast_prompt,
    dump_roster,
    fail_interrupted_scripts,
    generate_script,
    generate_script_from_text,
    load_roster,
    parse_script_turns,
    pending_cast_claim_briefing_ids,
    validate_roster,
)
from tldw_chatbook.Subscriptions.briefing_keep import keep_briefing
from tldw_chatbook.Subscriptions.briefing_service import GenerationInFlightError
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


def _complete_briefing(
    db, watchlist_id, *, body: str = "## This week\n\nSomething happened.\n"
) -> int:
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


def _chacha_db(tmp_path: Path) -> CharactersRAGDB:
    """A real, file-backed `CharactersRAGDB` -- mirrors `test_briefing_keep.py`'s
    own `_chacha_db` fixture exactly; `generate_script_from_text`'s DB work
    also runs through `asyncio.to_thread`, so `:memory:` would not do here
    either."""
    return CharactersRAGDB(tmp_path / "chacha.sqlite", client_id="cast-from-kept-test")


def _kept_briefing(
    chacha_db: CharactersRAGDB,
    *,
    source_briefing_id: int = 101,
    body: str = "## Kept body\n\nSomething worth keeping happened.\n",
    watchlist_name: str = "Security",
    origin: str = "manual",
) -> int:
    """A minimal `kept_briefings` row -- the only state `generate_script_from_text`
    may start a cast from. `source_briefing_id` is `UNIQUE`, so a test that
    keeps more than one briefing must pass distinct values explicitly."""
    return chacha_db.create_kept_briefing(
        source_briefing_id=source_briefing_id,
        watchlist_name=watchlist_name,
        body_markdown=body,
        origin=origin,
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
    prose = (
        "Sure, here is the script:\n" + json.dumps(CANNED_TURNS) + "\nHope that helps!"
    )
    assert parse_script_turns(prose, {"Host", "Analyst"}) == CANNED_TURNS


def test_an_unknown_speaker_fails_the_script_by_name():
    """Named invariant (spec + plan): an unknown speaker names ITSELF."""
    turns = [{"speaker": "Dave", "text": "Hi, I'm not on the roster."}]
    with pytest.raises(ScriptCastError, match="Dave"):
        parse_script_turns(json.dumps(turns), {"Host", "Analyst"})


def test_a_speakers_incidental_whitespace_is_stripped_before_the_roster_check():
    """`validate_roster` stores canonical (stripped) speaker names, but the
    model's raw JSON reply is never guaranteed to match that exactly -- a
    turn naming `"Alice "` must not fail the WHOLE cast as an unknown
    speaker just because of padding. The stored turn also carries the
    canonical name, not the raw padded one, so downstream rendering matches
    the roster."""
    turns = [
        {"speaker": "Alice ", "text": "Hi there."},
        {"speaker": " Bob", "text": "Hey."},
    ]
    parsed = parse_script_turns(json.dumps(turns), {"Alice", "Bob"})
    assert parsed == [
        {"speaker": "Alice", "text": "Hi there."},
        {"speaker": "Bob", "text": "Hey."},
    ]


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
    briefing_id = _complete_briefing(
        db, watchlist, body="## Body\n\nAcme shipped a thing."
    )
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
        db,
        briefing_id,
        preset_id=preset_id,
        chat=_FakeChat(reply=json.dumps([{"speaker": "Host", "text": "Hello!"}])),
        load_character=_load_character,
    )

    assert row["status"] == STATUS_COMPLETE
    snapshot = load_roster(row["roster_snapshot_json"])
    assert snapshot[0]["character_name"] == "Ada"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["generating", "empty", "failed"])
async def test_generate_script_refuses_when_the_briefing_is_not_complete(
    tmp_path: Path,
    status: str,
) -> None:
    """Refuse every incomplete briefing state without creating a script.

    Args:
        tmp_path: Private root for the real file-backed subscriptions database.
        status: Incomplete briefing status exercised by this parametrized case.
    """
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


# --- Provider/model resolution (whole-branch review fix wave, Important #2) --
#
# `generate_script` resolves `endpoint`/`resolved_model` correctly
# (briefing_cast.py:600-601: explicit args, then the preset's own
# provider/model, then the app default), but nothing pinned it -- the
# reviewer mutated both lines away (dropping the preset resolution
# entirely) and every one of the file's other tests stayed green, because
# every other test either passes no preset provider/model at all or never
# inspects the chat call's own kwargs. Mirrors `test_briefing_service.py`'s
# own three pins for `generate_briefing` (lines 629, 649, and the bare
# default asserted by `test_generation_happy_path_writes_everything`)
# exactly, on the cast side.


@pytest.mark.asyncio
async def test_a_presets_provider_and_model_are_used_with_no_explicit_override(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        briefing_service,
        "resolve_persisted_briefing_defaults",
        lambda: ("local-llama", "local-model"),
    )
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db, provider="anthropic", model="claude-x")

    chat = _FakeChat()
    row = await generate_script(db, briefing_id, preset_id=preset_id, chat=chat)

    assert chat.calls[0]["api_endpoint"] == "anthropic"
    assert chat.calls[0]["model"] == "claude-x"
    assert row["model_used"] == "anthropic/claude-x"


@pytest.mark.asyncio
async def test_explicit_provider_and_model_win_over_the_presets_own(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db, provider="anthropic", model="claude-x")

    chat = _FakeChat()
    row = await generate_script(
        db,
        briefing_id,
        preset_id=preset_id,
        chat=chat,
        provider="openai",
        model="gpt-x",
    )

    assert chat.calls[0]["api_endpoint"] == "openai"
    assert chat.calls[0]["model"] == "gpt-x"
    assert row["model_used"] == "openai/gpt-x"


@pytest.mark.asyncio
async def test_no_preset_provider_or_model_falls_back_to_the_app_default(
    tmp_path, monkeypatch
):
    """A preset with no provider/model of its own (and no explicit
    override) must reach the app's configured default endpoint -- the
    third leg of the fallback chain `_default_provider`'s own docstring
    names.
    """
    monkeypatch.setattr(
        briefing_service,
        "resolve_persisted_briefing_defaults",
        lambda: ("local-llama", "local-model"),
    )
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)  # no provider/model set

    chat = _FakeChat()
    row = await generate_script(db, briefing_id, preset_id=preset_id, chat=chat)

    assert chat.calls[0]["api_endpoint"] == "local-llama"
    assert chat.calls[0].get("model") is None
    assert row["model_used"] == "local-llama"


@pytest.mark.asyncio
async def test_generate_script_fails_naming_the_card_when_load_character_returns_none(
    tmp_path,
):
    """Spec: "fails the cast at that point, naming the card"."""
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    roster = [{"name": "Host", "role_prompt": "Warm.", "character_card_id": 7}]
    preset_id = _preset(db, roster=roster)

    row = await generate_script(
        db,
        briefing_id,
        preset_id=preset_id,
        chat=_FakeChat(),
        load_character=lambda _card_id: None,
    )

    assert row["status"] == STATUS_FAILED
    assert "7" in row["error"]
    assert db.list_briefing_scripts(briefing_id)[0]["id"] == row["id"]


@pytest.mark.asyncio
async def test_generate_script_fails_naming_the_card_when_load_character_is_none(
    tmp_path,
):
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
async def test_generate_script_chat_failure_is_a_failed_row_and_leaves_the_briefing_untouched(
    tmp_path,
):
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
async def test_generate_script_parse_failure_is_a_failed_row_naming_the_defect(
    tmp_path,
):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)

    row = await generate_script(
        db,
        briefing_id,
        preset_id=preset_id,
        chat=_FakeChat(reply="not a JSON array at all"),
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
    private_endpoint = "PRIVATE-ENDPOINT-SENTINEL"
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist, body=f"## Body\n\n{canary}\n")
    preset_id = _preset(db)

    captured: list[str] = []
    handler = logger.add(
        captured.append, level="DEBUG", diagnose=True, backtrace=True, catch=False
    )
    try:
        row = await generate_script(
            db,
            briefing_id,
            preset_id=preset_id,
            provider=private_endpoint,
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
    assert private_endpoint not in log_text
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

    Extended (Task 5 review round 1, Important): the roster here binds a
    speaker to a character card, so `load_character` is exercised through
    BOTH of its call sites -- `_snapshot_roster` (inside `_start_script`'s
    own `asyncio.to_thread` wrapper, already off-loop before this fix) and
    `_resolve_character_texts` (called directly from `generate_script`'s
    own coroutine body before this fix -- ON the event loop thread, since
    the real implementation is a blocking `ChaChaNotesDB.get_character_
    card_by_id` SELECT). Pinning `load_character`'s own thread identity,
    not just the DB writes above, is what catches a regression back to the
    direct (unthreaded) call.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    roster = [{"name": "Host", "role_prompt": "Warm.", "character_card_id": 7}]
    preset_id = _preset(db, roster=roster)

    loop_thread_id = threading.get_ident()
    write_thread_ids: list[int] = []

    for name in (
        "get_briefing",
        "get_briefing_preset",
        "insert_briefing_script",
        "update_briefing_script",
    ):
        original = getattr(db, name)

        def _spy(*args, __original=original, **kwargs):
            write_thread_ids.append(threading.get_ident())
            return __original(*args, **kwargs)

        setattr(db, name, _spy)

    load_character_thread_ids: list[int] = []

    def _load_character(card_id):
        assert card_id == 7
        load_character_thread_ids.append(threading.get_ident())
        return {"name": "Ada", "personality": "curious", "description": "a host"}

    row = await generate_script(
        db,
        briefing_id,
        preset_id=preset_id,
        chat=_FakeChat(reply=json.dumps([{"speaker": "Host", "text": "Hello!"}])),
        load_character=_load_character,
    )

    assert row["status"] == STATUS_COMPLETE
    assert len(write_thread_ids) >= 4
    assert all(tid != loop_thread_id for tid in write_thread_ids)
    # `load_character` is called twice per cast (snapshot, then strict
    # resolve) -- both must be off the event loop thread.
    assert len(load_character_thread_ids) >= 2
    assert all(tid != loop_thread_id for tid in load_character_thread_ids)


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


# --- In-process cast claims (spec #2 phase 4, Task 1) ------------------------
#
# Mirrors `test_briefing_service.py`'s own claims section exactly, scoped to
# a briefing id (a cast's collision unit) instead of a watchlist id.


def test_active_cast_claims_is_an_empty_snapshot_by_default():
    assert active_cast_claims() == frozenset()


def test_fail_interrupted_scripts_spares_an_excluded_row_both_directions(tmp_path):
    """Survey finding (a)'s cast-scoped sibling, updated for task-1890
    (mirroring task-1812): `exclude` now names the script ROW to spare, not
    its briefing -- a row survives the sweep when its id is passed via
    `exclude`, and is swept once it is not.

    A padding row is inserted (and finished) first so the row under test's
    own id is NOT the same integer as the briefing's id -- proving
    `exclude={live_row}` protects because of the ROW id, not a
    coincidental match with the briefing id (see `test_briefing_service.
    py`'s identical fixture reasoning).
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    padding = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Duo", roster_snapshot_json="[]"
    )
    db.update_briefing_script(padding, status="complete", turns_json="[]")
    live_row = db.insert_briefing_script(  # stands in for a live claim's own row
        briefing_id, preset_id=None, preset_name="Duo", roster_snapshot_json="[]"
    )
    assert live_row != briefing_id, "the fixture must not coincidentally align the ids"

    assert fail_interrupted_scripts(db, exclude={live_row}) == 0
    assert db.get_briefing_script(live_row)["status"] == "generating"

    assert fail_interrupted_scripts(db, exclude={live_row}) == 0
    assert fail_interrupted_scripts(db) == 1
    assert db.get_briefing_script(live_row)["status"] == "failed"
    assert db.get_briefing_script(live_row)["error"] == "interrupted"


@pytest.mark.asyncio
async def test_a_second_cast_for_a_claimed_briefing_raises_before_any_row_insert(
    tmp_path,
):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)
    rows_before = len(db.list_briefing_scripts(briefing_id))

    with briefing_cast._claim_cast(briefing_id):
        assert briefing_id in active_cast_claims()
        with pytest.raises(GenerationInFlightError, match=str(briefing_id)):
            await generate_script(
                db, briefing_id, preset_id=preset_id, chat=_FakeChat()
            )

    assert len(db.list_briefing_scripts(briefing_id)) == rows_before
    assert briefing_id not in active_cast_claims()


@pytest.mark.asyncio
async def test_the_cast_claim_is_released_after_a_successful_cast(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)

    row = await generate_script(db, briefing_id, preset_id=preset_id, chat=_FakeChat())

    assert row["status"] == STATUS_COMPLETE
    assert briefing_id not in active_cast_claims()


@pytest.mark.asyncio
async def test_the_cast_claim_is_released_after_a_cast_failure(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)

    row = await generate_script(
        db,
        briefing_id,
        preset_id=preset_id,
        chat=_FakeChat(error=RuntimeError("upstream 503")),
    )

    assert row["status"] == STATUS_FAILED
    assert briefing_id not in active_cast_claims()


@pytest.mark.asyncio
async def test_the_cast_claim_is_released_when_a_db_error_escapes(tmp_path, monkeypatch):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)

    closed_connection = db._get_connection()
    closed_connection.close()
    monkeypatch.setattr(db, "_get_connection", lambda: closed_connection)
    db._local = threading.local()

    with pytest.raises(sqlite3.ProgrammingError):
        await generate_script(db, briefing_id, preset_id=preset_id, chat=_FakeChat())

    assert briefing_id not in active_cast_claims()


@pytest.mark.asyncio
async def test_a_concurrent_cast_for_the_same_briefing_is_refused(tmp_path):
    """Pins that the claim is held through the chat call, exactly like
    `test_briefing_service.py`'s own concurrency pin."""
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def _slow_chat(**kwargs):
        entered.set()
        await release.wait()
        return json.dumps(CANNED_TURNS)

    first = asyncio.ensure_future(
        generate_script(db, briefing_id, preset_id=preset_id, chat=_slow_chat)
    )
    await entered.wait()

    assert briefing_id in active_cast_claims()
    with pytest.raises(GenerationInFlightError, match=str(briefing_id)):
        await generate_script(db, briefing_id, preset_id=preset_id, chat=_FakeChat())

    release.set()
    row = await first
    assert row["status"] == STATUS_COMPLETE
    assert briefing_id not in active_cast_claims()


# --- Row-scoped sweep exclusion (task-1890, generalizing task-1812) ---------
#
# `fail_interrupted_scripts`'s `exclude` used to be briefing-granular: ANY
# `generating` script row for a briefing named in `exclude` survived, on the
# reasoning that such a row is "a LIVE, in-process cast" -- true only if a
# briefing can have at most one `generating` script at a time. It cannot: a
# crash-zombie row left by a PRIOR process (that process's own claim died
# with it) can coexist with a freshly-claimed live row for the SAME
# briefing, and briefing-scoped exclusion incidentally shielded the zombie
# too. `active_cast_claim_row_ids()` names the live ROW instead, so a
# same-briefing zombie is swept exactly as if nothing were claimed at all.


def test_active_cast_claim_row_ids_is_an_empty_snapshot_by_default():
    """With no cast claims taken, the row-id snapshot is an empty frozenset."""
    assert active_cast_claim_row_ids() == frozenset()


@pytest.mark.asyncio
async def test_row_scoped_exclude_sweeps_a_same_briefing_zombie_while_sparing_the_live_row(
    tmp_path,
):
    """The coexistence case the docstring used to over-claim protection
    for: a crash-zombie script row and a genuinely live claim, both
    `generating`, both cast from the SAME briefing, in the same sweep.
    Row-scoped `exclude` must fail the zombie and leave the live row alone
    -- briefing-scoped `exclude` (the pre-task-1890 shape) cannot tell them
    apart and would spare both.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)

    # Stands in for a row a PRIOR process left behind mid-cast: its own
    # claim died with that process, so nothing in THIS process's
    # `_ACTIVE_CAST_CLAIM_ROW_IDS` will ever name it.
    zombie_id = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Duo", roster_snapshot_json="[]"
    )

    entered = asyncio.Event()
    release = asyncio.Event()

    async def _slow_chat(**kwargs):
        entered.set()
        await release.wait()
        return json.dumps(CANNED_TURNS)

    first = asyncio.ensure_future(
        generate_script(db, briefing_id, preset_id=preset_id, chat=_slow_chat)
    )
    await entered.wait()

    # The live claim now protects its OWN row, not merely its briefing.
    live_ids = active_cast_claim_row_ids()
    assert live_ids, "a live claim's row must be recorded by the time chat runs"
    assert zombie_id not in live_ids, (
        "the zombie's id must never be recorded by a claim it did not make"
    )

    swept = fail_interrupted_scripts(db, exclude=live_ids)

    assert swept == 1
    assert db.get_briefing_script(zombie_id)["status"] == "failed"
    assert db.get_briefing_script(zombie_id)["error"] == "interrupted"
    live_id = next(iter(live_ids))
    assert db.get_briefing_script(live_id)["status"] == "generating", (
        "row-scoped exclude must not falsify the row a live claim is "
        "actually writing"
    )

    release.set()
    row = await first
    assert row["status"] == STATUS_COMPLETE


# --- The unrecorded-claim sweep window (task-1890, generalizing the ---------
# whole-branch review fix in chore/briefings-residuals-1810-1812) -----------
#
# `generate_script` records `_ACTIVE_CAST_CLAIM_ROW_IDS[briefing_id]` only
# after the ENTIRE `_start_script` `to_thread` hop returns, but that hop's
# `INSERT` is its LAST statement -- for the rest of the hop (briefing
# lookup, preset lookup, roster validation, the roster snapshot) the
# briefing is claimed but no script row exists yet to protect, and once the
# `INSERT` itself has run, the row exists, reads `generating`, and `active_
# cast_claim_row_ids()` alone names nothing that protects it. A sweep at
# that exact instant used to falsify the row. `pending_cast_claim_
# briefing_ids()` closes the window: `_claim_cast(briefing_id)` with no
# `script_id` reproduces the registry state mid-window directly, with no
# need to block inside `_start_script` itself.


def test_pending_cast_claim_briefing_ids_is_an_empty_snapshot_by_default():
    """With no cast claims taken, the pending (unrecorded) set is empty."""
    assert pending_cast_claim_briefing_ids() == frozenset()


def test_a_cast_claim_with_no_recorded_row_id_yet_is_named_pending(tmp_path):
    """Direct pin of the accessor: a claim taken via `_claim_cast` without a
    `script_id` (exactly the registry state for the span inside `_start_
    script`'s `to_thread` hop, before `generate_script` resumes to record
    one) must appear in `pending_cast_claim_briefing_ids()`, and must
    disappear the instant a row id IS recorded.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)

    with briefing_cast._claim_cast(briefing_id):
        assert active_cast_claim_row_ids() == frozenset(), (
            "no row id has been recorded yet -- this is the window itself"
        )
        assert briefing_id in pending_cast_claim_briefing_ids()

    assert briefing_id not in pending_cast_claim_briefing_ids(), (
        "the claim released -- nothing should still read as pending"
    )

    live_row = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Duo", roster_snapshot_json="[]"
    )
    with briefing_cast._claim_cast(briefing_id, script_id=live_row):
        assert briefing_id not in pending_cast_claim_briefing_ids(), (
            "a claim whose row id IS recorded is no longer pending"
        )


def test_a_cast_claim_with_no_recorded_row_id_yet_survives_a_sweep_of_its_own_row(
    tmp_path,
):
    """The window itself, closed: a `generating` script row for a briefing
    whose claim exists but whose row id is not yet recorded (mid-`_start_
    script`, before `generate_script` resumes to record it) must survive a
    sweep run at that exact instant -- reproducing the same regression
    `test_briefing_service.py`'s own analogous test pins for briefings.

    Without `exclude_briefings`, `active_cast_claim_row_ids()` alone is
    empty here (nothing recorded yet) and the row would be swept as a false
    zombie -- exactly the regression this pins.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    # Stands in for the row `_start_script`'s `INSERT` just wrote, before
    # `generate_script` resumes on the event loop to record its id.
    live_row = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Duo", roster_snapshot_json="[]"
    )

    with briefing_cast._claim_cast(briefing_id):
        row_ids = active_cast_claim_row_ids()
        pending = pending_cast_claim_briefing_ids()
        assert row_ids == frozenset(), "the row id is not recorded in this window"
        assert briefing_id in pending

        swept = fail_interrupted_scripts(
            db, exclude=row_ids, exclude_briefings=pending
        )

    assert swept == 0
    assert db.get_briefing_script(live_row)["status"] == "generating", (
        "a claim whose row id has not been recorded yet must still spare "
        "its own row from a concurrent sweep"
    )


@pytest.mark.asyncio
async def test_row_scoped_exclude_still_sweeps_a_same_briefing_zombie_once_the_id_lands(
    tmp_path,
):
    """The task-1890 coexistence fix, re-asserted alongside the new guard:
    once a claim's row id IS recorded, `pending_cast_claim_briefing_ids()`
    no longer names its briefing, so `exclude_briefings` goes back to being
    a no-op for it and row-scoped `exclude` alone decides -- a
    same-briefing crash zombie is still swept even though the briefing
    itself has a live claim.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)

    zombie_id = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Duo", roster_snapshot_json="[]"
    )

    entered = asyncio.Event()
    release = asyncio.Event()

    async def _slow_chat(**kwargs):
        entered.set()
        await release.wait()
        return json.dumps(CANNED_TURNS)

    first = asyncio.ensure_future(
        generate_script(db, briefing_id, preset_id=preset_id, chat=_slow_chat)
    )
    await entered.wait()

    row_ids = active_cast_claim_row_ids()
    pending = pending_cast_claim_briefing_ids()
    assert row_ids, "the live claim's row id must be recorded by the time chat runs"
    assert briefing_id not in pending, (
        "once the row id lands, the briefing is no longer 'pending'"
    )

    swept = fail_interrupted_scripts(db, exclude=row_ids, exclude_briefings=pending)

    assert swept == 1
    assert db.get_briefing_script(zombie_id)["status"] == "failed"
    live_id = next(iter(row_ids))
    assert db.get_briefing_script(live_id)["status"] == "generating", (
        "the live row must survive even with exclude_briefings passed "
        "alongside row-scoped exclude"
    )

    release.set()
    row = await first
    assert row["status"] == STATUS_COMPLETE


# --- generate_script_from_text (task-1780, Task 4) ---------------------------
#
# Casts directly from a ChaChaNotes `kept_briefings` row into `kept_scripts`.
# Two rules shape these tests, both restated in `briefing_cast.py`'s own
# section comment (above `_APP_DEFAULT_ROSTER`):
#
# - **The asymmetry against `generate_script`.** `kept_scripts` has no
#   `status` column, so a chat or parse failure RAISES instead of becoming
#   a `failed` row, and nothing is written until a cast fully succeeds.
# - **The kept briefing is never touched by a cast's outcome**, success or
#   failure -- pinned by full-dict equality, exactly like `generate_
#   script`'s own "the briefing is never touched" tests pin `briefings`.
#
# Every test here closes its own `CharactersRAGDB` in a `finally`, matching
# `test_briefing_keep.py`'s own convention for that class.


@pytest.mark.asyncio
async def test_generate_script_from_text_happy_path_writes_into_kept_scripts(tmp_path):
    subs_db = _db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, body="## Kept\n\nAcme shipped a thing.")
        preset_id = _preset(
            subs_db, roster=TWO_SPEAKER_ROSTER, style_notes="Keep it brisk."
        )

        chat = _FakeChat()
        row = await generate_script_from_text(
            chacha_db, kept_id, preset_id=preset_id, subs_db=subs_db, chat=chat
        )

        assert row["kept_briefing_id"] == kept_id
        assert row["source_script_id"] is None
        assert row["preset_name"] == "Duo"
        assert json.loads(row["turns_json"]) == CANNED_TURNS
        assert row["model_used"]
        snapshot = load_roster(row["roster_snapshot_json"])
        assert [speaker["name"] for speaker in snapshot] == ["Host", "Analyst"]

        assert len(chat.calls) == 1
        assert chat.calls[0]["streaming"] is False
        assert "Keep it brisk." in chat.calls[0]["system_message"]
        assert (
            chat.calls[0]["messages_payload"][0]["content"]
            == "## Kept\n\nAcme shipped a thing."
        )

        # Findable via the listing interface, newest first.
        assert chacha_db.list_kept_scripts(kept_id)[0]["id"] == row["id"]
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_generate_script_from_text_app_default_when_preset_id_is_none(
    tmp_path, monkeypatch
):
    """`preset_id=None`: app default provider/model, no style notes,
    `preset_name` is the literal `APP_DEFAULT_PRESET_NAME`, and the roster
    is a single unbound "Narrator" speaker (there is no other source for a
    roster when no preset supplies one)."""
    monkeypatch.setattr(
        briefing_service,
        "resolve_persisted_briefing_defaults",
        lambda: ("local-llama", "local-model"),
    )
    subs_db = _db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db)

        chat = _FakeChat(reply=json.dumps([{"speaker": "Narrator", "text": "Hi."}]))
        row = await generate_script_from_text(
            chacha_db, kept_id, preset_id=None, subs_db=subs_db, chat=chat
        )

        assert row["preset_name"] == APP_DEFAULT_PRESET_NAME
        assert row["model_used"] == "local-llama"
        assert chat.calls[0]["api_endpoint"] == "local-llama"
        assert chat.calls[0].get("model") is None
        snapshot = load_roster(row["roster_snapshot_json"])
        assert [speaker["name"] for speaker in snapshot] == ["Narrator"]
        # No style notes reached the prompt for an app-default cast.
        assert "Style notes" not in chat.calls[0]["system_message"]
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_generate_script_from_text_refuses_when_kept_briefing_is_missing(
    tmp_path,
):
    subs_db = _db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        preset_id = _preset(subs_db)

        with pytest.raises(ScriptCastError, match="9999"):
            await generate_script_from_text(
                chacha_db, 9999, preset_id=preset_id, subs_db=subs_db, chat=_FakeChat()
            )

        assert chacha_db.list_kept_scripts(9999) == []
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_generate_script_from_text_refuses_when_body_is_empty(tmp_path):
    subs_db = _db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db, body="   ")
        preset_id = _preset(subs_db)

        with pytest.raises(ScriptCastError, match="empty"):
            await generate_script_from_text(
                chacha_db,
                kept_id,
                preset_id=preset_id,
                subs_db=subs_db,
                chat=_FakeChat(),
            )

        assert chacha_db.list_kept_scripts(kept_id) == []
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_generate_script_from_text_refuses_when_preset_is_missing(tmp_path):
    subs_db = _db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db)

        with pytest.raises(ScriptCastError, match="9999"):
            await generate_script_from_text(
                chacha_db,
                kept_id,
                preset_id=9999,
                subs_db=subs_db,
                chat=_FakeChat(),
            )

        assert chacha_db.list_kept_scripts(kept_id) == []
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_generate_script_from_text_fails_naming_the_card_when_load_character_is_none(
    tmp_path,
):
    """The roster always comes from the PRESET being cast with, never from
    the kept briefing's own (roster-less) snapshot -- so a card-bound
    speaker fails exactly like a live cast's does."""
    subs_db = _db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db)
        roster = [{"name": "Host", "role_prompt": "Warm.", "character_card_id": 7}]
        preset_id = _preset(subs_db, roster=roster)

        with pytest.raises(ScriptCastError, match="7"):
            await generate_script_from_text(
                chacha_db,
                kept_id,
                preset_id=preset_id,
                subs_db=subs_db,
                chat=_FakeChat(),
            )

        assert chacha_db.list_kept_scripts(kept_id) == []
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_generate_script_from_text_chat_failure_raises_and_writes_no_row(
    tmp_path,
):
    """The named asymmetry: unlike `generate_script`, a chat failure here
    RAISES rather than becoming a `failed` row."""
    subs_db = _db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db)
        preset_id = _preset(subs_db)

        boom = _FakeChat(error=RuntimeError("provider exploded: 503 upstream"))
        with pytest.raises(RuntimeError, match="503 upstream"):
            await generate_script_from_text(
                chacha_db, kept_id, preset_id=preset_id, subs_db=subs_db, chat=boom
            )

        assert chacha_db.list_kept_scripts(kept_id) == []
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_generate_script_from_text_parse_failure_raises_and_writes_no_row(
    tmp_path,
):
    subs_db = _db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db)
        preset_id = _preset(subs_db)

        with pytest.raises(ScriptCastError):
            await generate_script_from_text(
                chacha_db,
                kept_id,
                preset_id=preset_id,
                subs_db=subs_db,
                chat=_FakeChat(reply="not a JSON array at all"),
            )

        assert chacha_db.list_kept_scripts(kept_id) == []
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_generate_script_from_text_never_touches_the_kept_briefing_row(
    tmp_path,
):
    """THE named invariant, both directions: a cast failure AND a cast
    success leave `kept_briefings` byte-for-byte unchanged."""
    subs_db = _db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db)
        preset_id = _preset(subs_db)

        before = chacha_db.get_kept_briefing(kept_id)
        boom = _FakeChat(error=RuntimeError("provider exploded"))
        with pytest.raises(RuntimeError):
            await generate_script_from_text(
                chacha_db, kept_id, preset_id=preset_id, subs_db=subs_db, chat=boom
            )
        after_failure = chacha_db.get_kept_briefing(kept_id)
        assert before == after_failure

        row = await generate_script_from_text(
            chacha_db,
            kept_id,
            preset_id=preset_id,
            subs_db=subs_db,
            chat=_FakeChat(),
        )
        after_success = chacha_db.get_kept_briefing(kept_id)
        assert row["kept_briefing_id"] == kept_id
        assert before == after_success
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_recast_needs_no_subscriptions_rows(tmp_path):
    """Named test (AC #4): keep a briefing, delete the watchlist AND the
    original preset through real paths, then cast with a DIFFERENT
    currently-existing preset -- succeeds, lands in `kept_scripts` with
    the new preset's name and `source_script_id=NULL`."""
    subs_db = _db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = WatchlistBundleService(subs_db).create(name="Security")["id"]
        briefing_id = _complete_briefing(
            subs_db, watchlist_id, body="## Body\n\nAcme shipped a thing."
        )
        original_preset_id = _preset(
            subs_db, roster=TWO_SPEAKER_ROSTER, name="Original"
        )

        kept = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")
        kept_id = kept["kept_id"]

        # Delete the watchlist AND the original preset -- both through
        # real, app-level paths, never raw SQL.
        WatchlistBundleService(subs_db).delete(watchlist_id)
        assert subs_db.delete_briefing_preset(original_preset_id) is True
        assert subs_db.get_briefing_preset(original_preset_id) is None
        # Prove the watchlist deletion actually cascaded the briefing away
        # -- otherwise AC #4 would not be exercising anything real.
        assert subs_db.get_briefing(briefing_id) is None

        new_preset_id = _preset(subs_db, roster=ONE_SPEAKER_ROSTER, name="Fresh")

        row = await generate_script_from_text(
            chacha_db,
            kept_id,
            preset_id=new_preset_id,
            subs_db=subs_db,
            chat=_FakeChat(reply=json.dumps([{"speaker": "Narrator", "text": "Hi."}])),
        )

        assert row["preset_name"] == "Fresh"
        assert row["source_script_id"] is None
        assert row["kept_briefing_id"] == kept_id
        # The kept briefing itself is untouched by the recast.
        assert chacha_db.get_kept_briefing(kept_id) is not None
    finally:
        chacha_db.close_connection()


# --- In-process kept-cast claims (task-1780, Task 4) -------------------------
#
# Mirrors the live claims section above exactly, scoped to
# `_ACTIVE_KEPT_CAST_CLAIMS` -- except for the id-space test, whose whole
# point is proving the two sets do NOT interact.


def test_active_kept_cast_claims_is_an_empty_snapshot_by_default():
    assert active_kept_cast_claims() == frozenset()


def test_kept_cast_claim_blocks_a_second_concurrent_claim_of_the_same_kept_briefing():
    with briefing_cast._claim_kept_cast(42):
        assert 42 in active_kept_cast_claims()
        with pytest.raises(GenerationInFlightError, match="42"):
            with briefing_cast._claim_kept_cast(42):
                pass
    assert 42 not in active_kept_cast_claims()


def test_kept_and_live_cast_claims_do_not_collide_across_id_spaces():
    """The id-space guard this task's own mutation targets: a live claim
    and a kept claim sharing the same integer id must never see each
    other. Seeds BOTH directions -- a live claim on id 5 must not block a
    kept cast of kept_briefing_id 5, and vice versa."""
    with briefing_cast._claim_cast(5):
        # A live claim on briefing 5 does not block casting kept briefing 5.
        with briefing_cast._claim_kept_cast(5):
            assert 5 in active_cast_claims()
            assert 5 in active_kept_cast_claims()
    assert 5 not in active_cast_claims()
    assert 5 not in active_kept_cast_claims()

    with briefing_cast._claim_kept_cast(7):
        # A kept claim on kept_briefing 7 does not block casting live briefing 7.
        with briefing_cast._claim_cast(7):
            assert 7 in active_kept_cast_claims()
            assert 7 in active_cast_claims()
    assert 7 not in active_cast_claims()
    assert 7 not in active_kept_cast_claims()


@pytest.mark.asyncio
async def test_a_concurrent_kept_cast_for_the_same_kept_briefing_is_refused(tmp_path):
    """Pins that the kept claim is held through the chat call, mirroring
    `test_a_concurrent_cast_for_the_same_briefing_is_refused` exactly."""
    subs_db = _db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        kept_id = _kept_briefing(chacha_db)
        preset_id = _preset(subs_db)

        entered = asyncio.Event()
        release = asyncio.Event()

        async def _slow_chat(**kwargs):
            entered.set()
            await release.wait()
            return json.dumps(CANNED_TURNS)

        first = asyncio.ensure_future(
            generate_script_from_text(
                chacha_db,
                kept_id,
                preset_id=preset_id,
                subs_db=subs_db,
                chat=_slow_chat,
            )
        )
        await entered.wait()

        assert kept_id in active_kept_cast_claims()
        with pytest.raises(GenerationInFlightError, match=str(kept_id)):
            await generate_script_from_text(
                chacha_db,
                kept_id,
                preset_id=preset_id,
                subs_db=subs_db,
                chat=_FakeChat(),
            )

        release.set()
        row = await first
        assert row["kept_briefing_id"] == kept_id
        assert kept_id not in active_kept_cast_claims()
    finally:
        chacha_db.close_connection()


# --- TASK-21515: reasoning-aware completion budget (cast side) ---------------
#
# The cast call inherits the same defect a briefing's own call has: the
# DeepSeek handler's ``max_tokens`` is reasoning-inclusive with no effort
# lever, so a reasoning-typed default model spends a plain 3000-token budget
# on thinking and returns an empty completion, failing the script row.


@pytest.mark.asyncio
async def test_a_reasoning_typed_deepseek_cast_gets_a_larger_completion_budget(
    tmp_path,
):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)

    chat = _FakeChat()
    row = await generate_script(
        db,
        briefing_id,
        preset_id=preset_id,
        chat=chat,
        provider="deepseek",
        model="deepseek-v4-flash",
    )

    assert row["status"] == STATUS_COMPLETE
    assert chat.calls[0]["max_tokens"] == briefing_cast.CAST_REASONING_MAX_TOKENS == 12000


@pytest.mark.asyncio
async def test_a_plain_deepseek_chat_cast_keeps_the_plain_completion_budget(
    tmp_path,
):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    briefing_id = _complete_briefing(db, watchlist)
    preset_id = _preset(db)

    chat = _FakeChat()
    row = await generate_script(
        db,
        briefing_id,
        preset_id=preset_id,
        chat=chat,
        provider="deepseek",
        model="deepseek-chat",
    )

    assert row["status"] == STATUS_COMPLETE
    assert chat.calls[0]["max_tokens"] == briefing_cast.CAST_MAX_TOKENS == 3000


def test_cast_effective_max_tokens_gates_on_the_deepseek_endpoint_and_model():
    helper = briefing_cast._effective_max_tokens
    assert helper("deepseek", "deepseek-v4-flash") == (
        briefing_cast.CAST_REASONING_MAX_TOKENS
    )
    assert helper("DeepSeek", "deepseek-reasoner") == (
        briefing_cast.CAST_REASONING_MAX_TOKENS
    )
    assert helper("openai", "deepseek-v4-flash") == briefing_cast.CAST_MAX_TOKENS
    assert helper("deepseek", "deepseek-chat") == briefing_cast.CAST_MAX_TOKENS


@pytest.mark.parametrize(
    ("resolved_default", "expected"),
    [
        # Qodo #7/#8: a model=None cast on the deepseek endpoint runs the
        # handler's own configured (reasoning-typed) default, so the gate
        # resolves it instead of testing the literal None.
        ("deepseek-v4-flash", briefing_cast.CAST_REASONING_MAX_TOKENS),
        ("deepseek-reasoner", briefing_cast.CAST_REASONING_MAX_TOKENS),
        ("deepseek-chat", briefing_cast.CAST_MAX_TOKENS),
    ],
)
def test_cast_effective_max_tokens_resolves_the_deepseek_default_when_model_is_none(
    monkeypatch, resolved_default, expected
):
    monkeypatch.setattr(
        briefing_cast, "resolve_deepseek_effective_model", lambda m: resolved_default
    )
    assert briefing_cast._effective_max_tokens("deepseek", None) == expected
