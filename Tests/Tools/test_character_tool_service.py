"""CharacterToolService: handlers, bounding, guard, approval summary.

TASK-32954 Task 3 (+ fix round 1). Real in-memory ``CharactersRAGDB`` +
Task 1's ``LocalCharacterPersonaService`` -- no DB mocking, per repo
convention.
"""

from __future__ import annotations

import base64
import json

import pytest

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_chatbook.Tools import character_tool_service as cts

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def env():
    db = CharactersRAGDB(":memory:", client_id="test")
    local = LocalCharacterPersonaService(db=db)
    changed = []
    state = {"source": "local"}
    tool = cts.CharacterToolService(
        service_loader=lambda: local,
        runtime_source_loader=lambda: state["source"],
        read_guard=cts.CharacterReadGuard(),
        on_changed=changed.append,
        generate_avatar=lambda prompt: PNG,
    )
    return tool, local, db, changed, state


def j(text):
    return json.loads(text)


def test_create_then_get(env):
    tool, *_ = env
    saved = j(tool.save({"name": "Aria", "description": "A pilot"}))
    assert saved["status"] == "saved" and saved["version"] >= 1
    got = j(tool.get({"id": saved["id"]}))
    assert got["fields"]["description"]["text"] == "A pilot"
    assert "extensions" not in got["fields"]


def test_update_changes_only_given_fields(env):
    tool, _local, db, changed, _ = env
    saved = j(tool.save({"name": "Aria", "description": "A pilot", "scenario": "Space"}))
    out = j(tool.save({"id": saved["id"], "expected_version": saved["version"],
                       "description": "A captain"}))
    card = db.get_character_card_by_id(saved["id"])
    assert card["description"] == "A captain" and card["scenario"] == "Space"
    assert out["changed_fields"] == ["description"]
    assert changed[-1] == saved["id"]


def test_save_rejects_stale_version(env):
    tool, local, db, *_ = env
    saved = j(tool.save({"name": "Aria"}))
    local.update_character(saved["id"], {"scenario": "edited in Personas"},
                           expected_version=saved["version"])
    out = j(tool.save({"id": saved["id"], "expected_version": saved["version"],
                       "description": "x"}))
    assert out["status"] == "stale_version"
    assert db.get_character_card_by_id(saved["id"])["description"] in (None, "")


def test_duplicate_name(env):
    tool, *_ = env
    first = j(tool.save({"name": "Aria"}))
    out = j(tool.save({"name": "Aria"}))
    assert out["status"] == "duplicate_name" and str(first["id"]) in out["message"]


def test_update_rename_to_duplicate_name_reports_duplicate(env):
    # A race/late-arriving DB-layer ConflictError (name UNIQUE constraint on
    # UPDATE) must map to duplicate_name, not stale_version -- the two are
    # different DB.ConflictError messages disambiguated by the "already
    # exists" substring in CharacterToolService._save. The message must also
    # name the existing id (fix-round-1 item 7).
    tool, *_ = env
    first = j(tool.save({"name": "Aria"}))
    second = j(tool.save({"name": "Bianca"}))
    out = j(tool.save({"id": second["id"], "expected_version": second["version"],
                       "name": "Aria"}))
    assert out["status"] == "duplicate_name"
    assert str(first["id"]) in out["message"]


def test_create_race_duplicate_name_includes_id(env):
    # Simulate a true create-time race: the pre-check lookup momentarily
    # reports "no conflict" (as a concurrent writer's insert had not yet
    # committed when we checked), but the DB layer's own UNIQUE constraint
    # still raises ConflictError on the actual insert. The except-branch's
    # re-lookup (using the real, unpatched lookup on all later calls) must
    # still find and report the existing id (fix-round-1 item 7).
    tool, local, db, *_ = env
    first = j(tool.save({"name": "Aria"}))

    real_lookup = db.get_character_card_by_name
    calls = {"n": 0}

    def flaky_lookup(name):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return real_lookup(name)

    db.get_character_card_by_name = flaky_lookup

    def flaky_create(_payload):
        raise ConflictError("Character card with name 'Aria' already exists.",
                            entity="character_cards", entity_id="Aria")

    local.create_character = flaky_create

    out = j(tool.save({"name": "Aria"}))
    assert out["status"] == "duplicate_name"
    assert str(first["id"]) in out["message"]


def test_whitespace_name_rejected(env):
    tool, *_ = env
    assert j(tool.save({"name": "   "}))["status"] == "invalid_argument"


def test_field_over_limit(env):
    # CharacterCreateRequest.name has max_length=500; pydantic's ValidationError
    # must surface as invalid_argument naming the field, not a raw exception.
    tool, *_ = env
    out = j(tool.save({"name": "x" * 501}))
    assert out["status"] == "invalid_argument" and "name" in out["message"]


def test_update_field_over_limit_is_invalid_argument(env):
    # Fix-round-1 item 2: pydantic ValidationError IS a ValueError, and the
    # update path's ConflictError/ValueError handling used to swallow it as
    # stale_version. Must report invalid_argument naming the field instead.
    tool, *_ = env
    saved = j(tool.save({"name": "Aria"}))
    out = j(tool.save({"id": saved["id"], "expected_version": saved["version"],
                       "name": "y" * 501}))
    assert out["status"] == "invalid_argument" and "name" in out["message"]


def test_search_bad_limit_type_is_invalid_argument(env):
    tool, *_ = env
    assert j(tool.search({"limit": "abc"}))["status"] == "invalid_argument"
    assert j(tool.search({"offset": "abc"}))["status"] == "invalid_argument"


def test_not_found_and_bad_ids(env):
    tool, _local, db, *_ = env
    assert j(tool.get({"id": 999}))["status"] == "not_found"
    assert j(tool.get({"id": "abc"}))["status"] == "invalid_argument"
    saved = j(tool.save({"name": "Gone"}))
    # LocalCharacterPersonaService.delete_character requires expected_version
    # as a keyword-only arg (no bare no-arg delete exists on the real
    # service), so soft-delete directly through the DB instead, matching the
    # brief's "replace with a real soft-delete via the DB" fallback.
    db.soft_delete_character_card(saved["id"], saved["version"])
    assert j(tool.get({"id": saved["id"]}))["status"] == "not_found"


def test_get_pages_reassemble_exactly(env):
    tool, *_ = env
    text = ("é🙂á" * 3000)[: cts.CHARACTER_FIELD_READ_BOUND * 2 + 17]
    saved = j(tool.save({"name": "Long", "description": text}))
    first = j(tool.get({"id": saved["id"]}))["fields"]["description"]
    parts, offset = [first["text"]], first.get("next_offset")
    while offset is not None:
        page = j(tool.get({"id": saved["id"], "field": "description", "offset": offset}))
        parts.append(page["text"])
        offset = page.get("next_offset")
    assert "".join(parts) == text


def test_truncation_guard_blocks_until_full_read(env):
    tool, *_ = env
    long_text = "x" * (cts.CHARACTER_FIELD_READ_BOUND + 10)
    saved = j(tool.save({"name": "Long", "description": long_text}))
    tool.get({"id": saved["id"]})  # truncated read only
    blocked = j(tool.save({"id": saved["id"], "expected_version": saved["version"],
                           "description": "rewrite"}))
    assert blocked["status"] == "read_full_field_first"
    tool.get({"id": saved["id"], "field": "description",
              "offset": cts.CHARACTER_FIELD_READ_BOUND})  # reaches the end
    ok = j(tool.save({"id": saved["id"], "expected_version": saved["version"],
                      "description": "rewrite"}))
    assert ok["status"] == "saved"


def test_truncation_guard_requires_contiguous_coverage(env):
    # Fix-round-1 item 1 (data-loss finding): reading page 1 then jumping to
    # the LAST page (skipping the middle one) must NOT unlock a write --
    # the old implementation marked a field "full" whenever any single
    # page's end reached the field's length, regardless of what was
    # skipped. Filling the missing middle page closes the gap up to where
    # that page ends, but the earlier out-of-order tail read was a no-op
    # (coverage tracks a single contiguous frontier, not a bitmap of every
    # page ever read) -- the tail must be re-read, now contiguous, to
    # actually unlock.
    tool, *_ = env
    bound = cts.CHARACTER_FIELD_READ_BOUND
    text = "x" * (bound * 2 + 17)  # exactly 3 pages: [0,8000) [8000,16000) [16000,16017)
    saved = j(tool.save({"name": "Long3", "description": text}))

    tool.get({"id": saved["id"], "field": "description", "offset": 0})       # page 1
    tool.get({"id": saved["id"], "field": "description", "offset": bound * 2})  # page 3, skip 2

    blocked = j(tool.save({"id": saved["id"], "expected_version": saved["version"],
                           "description": "rewrite"}))
    assert blocked["status"] == "read_full_field_first"

    tool.get({"id": saved["id"], "field": "description", "offset": bound})  # the missing page 2
    still_blocked = j(tool.save({"id": saved["id"], "expected_version": saved["version"],
                                 "description": "rewrite"}))
    assert still_blocked["status"] == "read_full_field_first"  # still short the final 17 chars

    tool.get({"id": saved["id"], "field": "description", "offset": bound * 2})  # re-read the tail
    ok = j(tool.save({"id": saved["id"], "expected_version": saved["version"],
                      "description": "rewrite"}))
    assert ok["status"] == "saved"


def test_get_offset_beyond_length_is_invalid_argument(env):
    # Fix-round-1 item 1: an absurd offset (e.g. 10**9) must be rejected
    # outright, not silently return "" and be treated as a valid read that
    # could unlock a write.
    tool, *_ = env
    saved = j(tool.save({"name": "Short", "description": "hi"}))
    out = j(tool.get({"id": saved["id"], "field": "description", "offset": 10**9}))
    assert out["status"] == "invalid_argument"


def test_server_mode_refuses(env):
    tool, *_, state = env
    state["source"] = "server"
    out = j(tool.search({}))
    assert out["status"] == "unsupported" and out["message"] == cts.SERVER_REFUSAL


def test_save_with_generated_avatar_is_one_write(env):
    # A description is required for resolve_avatar to compose a prompt when
    # none is given -- Task 3 no longer papers over a missing description
    # with a synthetic prompt (fix-round-1 item 4 removed that fallback).
    tool, _local, db, *_ = env
    out = j(tool.save({"name": "Aria", "description": "A witty starship pilot",
                       "avatar": {"source": "generate"}}))
    card = db.get_character_card_by_id(out["id"])
    assert out["avatar"] == "saved" and card["image"] == PNG and card["version"] == out["version"]


def test_create_generate_avatar_without_description_or_prompt_fails_but_saves_text(env):
    # Fix-round-1 item 4: no fallback prompt. Task 2's original contract
    # applies unmodified -- the text still saves, the avatar reports failed.
    tool, _local, db, *_ = env
    out = j(tool.save({"name": "Aria", "avatar": {"source": "generate"}}))
    assert out["status"] == "saved"
    assert out["avatar"] == "failed: add a description or give an avatar prompt"
    assert db.get_character_card_by_id(out["id"])["image"] is None


def test_create_over_limit_validates_before_avatar_generation(env):
    # Fix-round-1 item 3: validate BEFORE resolving the avatar, so a paid
    # generation is never spent on a save that then fails validation.
    _, local, _db, changed, state = env
    calls = []
    tool2 = cts.CharacterToolService(
        service_loader=lambda: local,
        runtime_source_loader=lambda: state["source"],
        read_guard=cts.CharacterReadGuard(),
        on_changed=changed.append,
        generate_avatar=lambda prompt: (calls.append(prompt), PNG)[1],
    )
    out = j(tool2.save({"name": "y" * 501, "avatar": {"source": "generate"}}))
    assert out["status"] == "invalid_argument"
    assert calls == []


def test_create_with_remove_avatar_reports_none(env):
    # A brand-new character has nothing to remove; resolve_avatar's "remove"
    # outcome is a no-op on create, reported as avatar: "none" (per ruling).
    tool, _local, db, *_ = env
    out = j(tool.save({"name": "Aria", "avatar": {"source": "remove"}}))
    assert out["status"] == "saved" and out["avatar"] == "none"
    assert db.get_character_card_by_id(out["id"])["image"] is None


def test_save_reports_avatar_failure_but_keeps_text(env, tmp_path):
    tool, _local, db, *_ = env
    bad = tmp_path / "fake.png"
    bad.write_text("nope")
    out = j(tool.save({"name": "Aria", "description": "kept",
                       "avatar": {"source": "file", "path": str(bad)}}))
    assert out["status"] == "saved" and out["avatar"].startswith("failed:")
    assert db.get_character_card_by_id(out["id"])["description"] == "kept"


def test_non_mapping_avatar_is_invalid_argument(env):
    # Fix-round-1 item 8.
    tool, *_ = env
    out = j(tool.save({"name": "Aria", "avatar": "generate"}))
    assert out["status"] == "invalid_argument"


def test_on_changed_exception_does_not_fail_the_save(env):
    # Fix-round-1 item 9: a notification failure must not turn an already-
    # committed save into a reported failure.
    tool, _local, db, *_ = env

    def _boom(_character_id):
        raise RuntimeError("notification boom")

    tool._on_changed = _boom
    out = j(tool.save({"name": "Aria"}))
    assert out["status"] == "saved"
    assert db.get_character_card_by_id(out["id"]) is not None


def test_runtime_source_loader_exception_is_handled(env):
    # Fix-round-1 item 9: runtime_source_loader() must live inside _run's
    # guarded region -- an exception from it is handled like any other,
    # not left to crash out of _run uncaught.
    tool, *_ = env

    def _boom():
        raise RuntimeError("source lookup boom")

    tool._runtime_source_loader = _boom
    with pytest.raises(RuntimeError, match=cts._PUBLIC_EXECUTION_ERROR):
        tool.search({})


def test_search_lists_and_bounds(env):
    tool, *_ = env
    for n in range(3):
        tool.save({"name": f"C{n}", "description": "d" * 5000})
    out = j(tool.search({"limit": 2}))
    assert len(out["items"]) == 2 and out["next_offset"] == 2
    assert all(len(i["description"]) <= 160 for i in out["items"])


def test_search_no_query_reports_has_avatar_unknown(env):
    # Fix-round-1 item 5: list_characters omits the image column, so the
    # browse (no-query) path cannot know -- report null, never False. A
    # fresh CharactersRAGDB seeds a "Default Assistant" card that can sort
    # ahead of ours alphabetically, so find our row by name explicitly
    # rather than assuming index 0.
    tool, *_ = env
    tool.save({"name": "NoQueryAvatar", "description": "d",
              "avatar": {"source": "generate"}})
    out = j(tool.search({"limit": 25}))
    item = next(i for i in out["items"] if i["name"] == "NoQueryAvatar")
    assert item["has_avatar"] is None


def test_search_with_query_reports_real_has_avatar(env):
    # search_characters does SELECT cc.* -- the real value is known.
    tool, *_ = env
    tool.save({"name": "QueryAvatarZzz", "description": "unique marker text zzzqqq",
              "avatar": {"source": "generate"}})
    out = j(tool.search({"query": "zzzqqq"}))
    assert out["items"] and out["items"][0]["has_avatar"] is True


def test_search_query_honors_offset_and_next_offset(env):
    # Fix-round-1 item 6: offset was previously ignored for the query path.
    tool, *_ = env
    for n in range(3):
        tool.save({"name": f"Wob{n}", "description": "wobblefish matching text"})
    page1 = j(tool.search({"query": "wobblefish", "limit": 1, "offset": 0}))
    page2 = j(tool.search({"query": "wobblefish", "limit": 1, "offset": 1}))
    page3 = j(tool.search({"query": "wobblefish", "limit": 1, "offset": 2}))
    assert page1["next_offset"] == 1 and page2["next_offset"] == 2
    assert "next_offset" not in page3
    ids = {page1["items"][0]["id"], page2["items"][0]["id"], page3["items"][0]["id"]}
    assert len(ids) == 3


def test_search_bounds_tags(env):
    # Fix-round-1 item 10: at most 20 tags, each truncated to 64 chars. A
    # fresh CharactersRAGDB seeds a "Default Assistant" card that can sort
    # ahead of ours alphabetically, so find our row by name explicitly
    # rather than assuming index 0.
    tool, *_ = env
    many_tags = [f"tag-{i}-{'x' * 100}" for i in range(30)]
    tool.save({"name": "Tagged", "tags": many_tags})
    out = j(tool.search({"limit": 25}))
    item = next(i for i in out["items"] if i["name"] == "Tagged")
    tags = item["tags"]
    assert len(tags) == 20
    assert all(len(t) <= 64 for t in tags)


def test_approval_summary_has_no_field_text():
    summary = cts.save_approval_summary(
        {"id": 3, "expected_version": 2, "description": "SECRET TEXT",
         "avatar": {"source": "file", "path": "/tmp/a.png"}}
    )
    flat = json.dumps(summary)
    assert "SECRET TEXT" not in flat and "description" in flat and "/tmp/a.png" in flat
