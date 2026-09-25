"""CharacterToolService: handlers, bounding, guard, approval summary.

TASK-32954 Task 3. Real in-memory ``CharactersRAGDB`` + Task 1's
``LocalCharacterPersonaService`` -- no DB mocking, per repo convention.
"""

from __future__ import annotations

import base64
import json

import pytest

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
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
    # exists" substring in CharacterToolService._save.
    tool, *_ = env
    j(tool.save({"name": "Aria"}))
    second = j(tool.save({"name": "Bianca"}))
    out = j(tool.save({"id": second["id"], "expected_version": second["version"],
                       "name": "Aria"}))
    assert out["status"] == "duplicate_name"


def test_whitespace_name_rejected(env):
    tool, *_ = env
    assert j(tool.save({"name": "   "}))["status"] == "invalid_argument"


def test_field_over_limit(env):
    # CharacterCreateRequest.name has max_length=500; pydantic's ValidationError
    # must surface as invalid_argument naming the field, not a raw exception.
    tool, *_ = env
    out = j(tool.save({"name": "x" * 501}))
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


def test_server_mode_refuses(env):
    tool, *_, state = env
    state["source"] = "server"
    out = j(tool.search({}))
    assert out["status"] == "unsupported" and out["message"] == cts.SERVER_REFUSAL


def test_save_with_generated_avatar_is_one_write(env):
    tool, _local, db, *_ = env
    out = j(tool.save({"name": "Aria", "avatar": {"source": "generate"}}))
    card = db.get_character_card_by_id(out["id"])
    assert out["avatar"] == "saved" and card["image"] == PNG and card["version"] == out["version"]


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


def test_search_lists_and_bounds(env):
    tool, *_ = env
    for n in range(3):
        tool.save({"name": f"C{n}", "description": "d" * 5000})
    out = j(tool.search({"limit": 2}))
    assert len(out["items"]) == 2 and out["next_offset"] == 2
    assert all(len(i["description"]) <= 160 for i in out["items"])


def test_approval_summary_has_no_field_text():
    summary = cts.save_approval_summary(
        {"id": 3, "expected_version": 2, "description": "SECRET TEXT",
         "avatar": {"source": "file", "path": "/tmp/a.png"}}
    )
    flat = json.dumps(summary)
    assert "SECRET TEXT" not in flat and "description" in flat and "/tmp/a.png" in flat
