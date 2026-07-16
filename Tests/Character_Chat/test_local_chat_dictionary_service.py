import json as _json

import pytest

from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError


@pytest.fixture
def dictionary_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "chat_dictionaries.db", "test-client")
    yield db
    db.close_connection()


def test_local_chat_dictionary_service_routes_core_crud(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)

    created = service.create_dictionary(
        {
            "name": "Local Lore",
            "description": "Local dictionary",
            "default_token_budget": 750,
        }
    )
    detail = service.get_dictionary(created["id"])
    updated = service.update_dictionary(
        created["id"],
        {
            "name": "Local Lore v2",
            "description": "Updated",
            "is_active": False,
            "default_token_budget": 900,
        },
        expected_version=detail["version"],
    )
    listed = service.list_dictionaries(include_inactive=True)
    deleted = service.delete_dictionary(updated["id"], expected_version=updated["version"])

    assert created["source"] == "local"
    assert detail["name"] == "Local Lore"
    assert detail["max_tokens"] == 750
    assert updated["name"] == "Local Lore v2"
    assert updated["enabled"] is False
    assert updated["max_tokens"] == 900
    assert listed["dictionaries"][0]["id"] == created["id"]
    assert deleted == {"status": "deleted", "dictionary_id": created["id"], "source": "local"}
    assert service.get_dictionary(created["id"]) is None


def test_list_dictionaries_reports_entry_count_without_inflating_entries(dictionary_db):
    """The rail meta relies on list_dictionaries() reporting a real entry
    count without paying to materialize every entry's ChatDictionary object.

    Regression for PR #622 finding 1: list_chat_dictionaries() used to
    hardcode 'entries': [] with no way for callers to learn the true count,
    so the rail always rendered "0 entries" no matter what was saved.
    """
    service = LocalChatDictionaryService(dictionary_db)

    created = service.create_dictionary(
        {
            "name": "Entry Count Lore",
            "entries": [
                {"pattern": "BP", "replacement": "blood pressure"},
                {"pattern": "HR", "replacement": "heart rate"},
            ],
        }
    )
    assert created["entry_count"] == 2

    listed = service.list_dictionaries(include_inactive=True)
    record = next(r for r in listed["dictionaries"] if r["id"] == created["id"])
    assert record["entry_count"] == 2
    # List path stays cheap: entries themselves are not materialized here.
    assert record["entries"] == []

    detail = service.get_dictionary(created["id"])
    assert detail["entry_count"] == 2
    assert len(detail["entries"]) == 2


def test_local_chat_dictionary_service_imports_exports_and_processes_markdown(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)

    imported = service.import_markdown(
        {
            "name": "Imported Lore",
            "content": "Ada: Dr. Ada\nTuring: |\nDr. Turing\n---@@@---\n",
            "activate": True,
        }
    )
    exported = service.export_markdown(imported["dictionary_id"])
    processed = service.process_text({"text": "Ada met Turing.", "dictionary_id": imported["dictionary_id"]})

    assert imported["source"] == "local"
    assert exported["name"] == "Imported Lore"
    assert "Ada: Dr. Ada" in exported["content"]
    assert processed["processed_text"] == "Dr. Ada met Dr. Turing."


def test_local_chat_dictionary_service_uses_source_scoped_entry_ids(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    dictionary = service.create_dictionary({"name": "Entry Lore"})

    created_entry = service.add_entry(
        dictionary["id"],
        {
            "pattern": "Ada",
            "replacement": "Dr. Ada",
            "probability": 1.0,
            "group": "people",
        },
    )
    entries = service.list_entries(dictionary["id"], group="people")
    updated_entry = service.update_entry(created_entry["id"], {"replacement": "Professor Ada"})
    reordered = service.reorder_entries(dictionary["id"], {"entry_ids": [created_entry["id"]]})
    deleted = service.delete_entry(created_entry["id"])

    assert created_entry["id"] == f"local:chat_dictionary_entry:{dictionary['id']}:0"
    assert entries["entries"][0]["pattern"] == "Ada"
    assert entries["entries"][0]["replacement"] == "Dr. Ada"
    assert updated_entry["replacement"] == "Professor Ada"
    assert reordered["entry_ids"] == [created_entry["id"]]
    assert deleted == {"status": "deleted", "entry_id": created_entry["id"], "source": "local"}
    assert service.list_entries(dictionary["id"])["entries"] == []


def test_local_chat_dictionary_service_reports_basic_statistics(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    dictionary = service.create_dictionary(
        {
            "name": "Stats Lore",
            "entries": [
                {"pattern": "Ada", "replacement": "Dr. Ada"},
                {"pattern": "Turing", "replacement": "Dr. Turing"},
            ],
        }
    )

    statistics = service.get_statistics(dictionary["id"])

    assert statistics == {
        "dictionary_id": dictionary["id"],
        "entry_count": 2,
        "enabled": True,
        "source": "local",
    }


def test_local_chat_dictionary_service_records_activity_versions_and_reverts(dictionary_db, tmp_path):
    history_path = tmp_path / "chat_dictionary_history.json"
    service = LocalChatDictionaryService(dictionary_db, history_store_path=history_path)

    created = service.create_dictionary({"name": "Versioned Lore", "description": "v1"})
    updated = service.update_dictionary(
        created["id"],
        {"name": "Versioned Lore v2", "description": "v2"},
        expected_version=created["version"],
    )
    activity = service.list_activity(created["id"], limit=10)
    versions = service.list_versions(created["id"], limit=10)
    version_one = service.get_version(created["id"], 1)
    reverted = service.revert_version(created["id"], 1)
    reloaded = LocalChatDictionaryService(dictionary_db, history_store_path=history_path)

    assert updated["version"] == 2
    assert [item["action"] for item in activity["activity"]] == ["update", "create"]
    assert [item["revision"] for item in versions["versions"]] == [2, 1]
    assert version_one["snapshot"]["name"] == "Versioned Lore"
    assert reverted["name"] == "Versioned Lore"
    assert reverted["reverted_to_revision"] == 1
    assert reloaded.list_versions(created["id"], limit=10)["total"] == 3


def test_local_chat_dictionary_service_update_raises_conflict_error_on_stale_version(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    dictionary = service.create_dictionary({"name": "Conflict Lore"})

    with pytest.raises(ConflictError):
        service.update_dictionary(
            dictionary["id"],
            {"name": "Conflict Lore v2"},
            expected_version=dictionary["version"] + 1,
        )


def test_local_chat_dictionary_service_delete_raises_conflict_error_on_stale_version(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    dictionary = service.create_dictionary({"name": "Conflict Lore Delete"})

    with pytest.raises(ConflictError):
        service.delete_dictionary(dictionary["id"], expected_version=dictionary["version"] + 1)


def test_local_chat_dictionary_service_repairs_legacy_fts_trigger_before_delete(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    dictionary = service.create_dictionary({"name": "Legacy Trigger Lore"})
    conn = dictionary_db.get_connection()
    conn.execute("DROP TRIGGER IF EXISTS chat_dictionaries_au")
    conn.execute(
        """
        CREATE TRIGGER chat_dictionaries_au
        AFTER UPDATE ON chat_dictionaries BEGIN
          UPDATE chat_dictionaries_fts
          SET name = NEW.name, description = NEW.description, content = NEW.content
          WHERE rowid = NEW.id;
        END;
        """
    )
    conn.commit()

    deleted = service.delete_dictionary(dictionary["id"], expected_version=dictionary["version"])

    assert deleted["status"] == "deleted"


def _create_two_entry_dictionary(service, *, name="Diagnose Me"):
    return service.create_dictionary(
        {
            "name": name,
            "entries": [
                {"pattern": "BP", "replacement": "blood pressure"},
                {"pattern": "HR", "replacement": "heart rate", "group": "vitals"},
            ],
        }
    )


def test_process_text_carries_enriched_diagnostics(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    created = _create_two_entry_dictionary(service)

    response = service.process_text({"text": "BP and HR", "dictionary_id": created["id"]})

    # Existing keys byte-identical in name and meaning.
    assert response["text"] == "BP and HR"
    assert response["processed_text"] == "blood pressure and heart rate"
    assert response["dictionary_id"] == created["id"]
    assert response["source"] == "local"

    diagnostics = response["diagnostics"]
    assert diagnostics["matched"] == 2 and diagnostics["fired"] == 2
    ids = {r["pattern"]: r["entry_id"] for r in diagnostics["entries"]}
    assert ids["BP"] == f"local:chat_dictionary_entry:{created['id']}:0"
    assert ids["HR"] == f"local:chat_dictionary_entry:{created['id']}:1"


def test_process_text_group_filter_keeps_ids_correct(dictionary_db):
    # The group filter drops entry 0 BEFORE the engine runs; append-time
    # tracking must still map the surviving entry to stored index 1.
    service = LocalChatDictionaryService(dictionary_db)
    created = _create_two_entry_dictionary(service, name="Grouped")

    response = service.process_text(
        {"text": "BP and HR", "dictionary_id": created["id"], "group": "vitals"}
    )

    records = response["diagnostics"]["entries"]
    assert [r["pattern"] for r in records] == ["HR"]
    assert records[0]["entry_id"] == f"local:chat_dictionary_entry:{created['id']}:1"
    assert records[0]["input_index"] == 0  # engine saw a 1-element list


def test_process_text_all_dictionaries_path_ids_carry_own_dict(dictionary_db):
    # dictionary_id=None concatenates entries across ALL dictionaries; each
    # record's entry_id must carry its OWN dictionary's id + stored index.
    service = LocalChatDictionaryService(dictionary_db)
    first = _create_two_entry_dictionary(service, name="First")
    second = service.create_dictionary(
        {"name": "Second", "entries": [{"pattern": "RR", "replacement": "respiratory rate"}]}
    )

    response = service.process_text({"text": "BP and RR"})

    ids = {r["pattern"]: r["entry_id"] for r in response["diagnostics"]["entries"]}
    assert ids["BP"] == f"local:chat_dictionary_entry:{first['id']}:0"
    assert ids["RR"] == f"local:chat_dictionary_entry:{second['id']}:0"
    assert response["dictionary_id"] is None


def test_process_text_omits_diagnostics_on_assembly_failure(dictionary_db, monkeypatch):
    service = LocalChatDictionaryService(dictionary_db)
    created = _create_two_entry_dictionary(service, name="Degrade")

    import tldw_chatbook.Character_Chat.Chat_Dictionary_Lib as cdl_module

    monkeypatch.setattr(
        cdl_module.DictionaryProcessDiagnostics,
        "to_dict",
        lambda self: (_ for _ in ()).throw(RuntimeError("assembly boom")),
    )
    response = service.process_text({"text": "BP now", "dictionary_id": created["id"]})
    assert response["processed_text"] == "blood pressure now"
    assert "diagnostics" not in response


def test_entry_new_fields_roundtrip_through_service(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    created = service.create_dictionary(
        {
            "name": "Fields",
            "entries": [
                {"pattern": "BP", "replacement": "blood pressure",
                 "enabled": False, "case_sensitive": True, "priority": 9},
            ],
        }
    )
    record = service.get_dictionary(created["id"])
    entry = record["entries"][0]
    assert entry["enabled"] is False          # no longer hardcoded True
    assert entry["case_sensitive"] is True
    assert entry["priority"] == 9
    # Partial update touching only the replacement preserves the three fields.
    updated = service.update_entry(entry["id"], {"replacement": "arterial pressure"})
    assert (updated["enabled"], updated["case_sensitive"], updated["priority"]) == (False, True, 9)


def test_entry_loose_typed_priority_and_enabled_do_not_crash(dictionary_db):
    """A malformed ``priority`` string must not crash entry creation, and a
    quoted ``"false"`` must be honored as False rather than truthy-``bool()``."""
    service = LocalChatDictionaryService(dictionary_db)
    created = service.create_dictionary(
        {
            "name": "Loose Types",
            "entries": [
                {"pattern": "BP", "replacement": "blood pressure",
                 "priority": "garbage", "enabled": "false"},
            ],
        }
    )
    record = service.get_dictionary(created["id"])
    entry = record["entries"][0]
    assert entry["priority"] == 0
    assert entry["enabled"] is False


def test_json_export_import_roundtrips_every_field(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    created = service.create_dictionary(
        {
            "name": "Round Trip",
            "description": "all fields",
            "entries": [
                {"pattern": "BP", "replacement": "blood pressure", "probability": 0.85,
                 "group": "med", "timed_effects": {"sticky": 0, "cooldown": 5, "delay": 0},
                 "max_replacements": 3, "type": "literal",
                 "enabled": False, "case_sensitive": True, "priority": 7},
            ],
            "max_tokens": 750,
        }
    )
    # strategy is settable only via update (create ignores it — P1a ground truth).
    service.update_dictionary(created["id"], {"strategy": "character_lore_first"})

    exported = service.export_json(created["id"])
    assert exported["data"]["strategy"] == "character_lore_first"  # the fixed seam

    exported["data"]["name"] = "Round Trip (imported)"  # data.name WINS — mutate it
    imported = service.import_json({"data": exported["data"]})
    record = service.get_dictionary(imported["dictionary_id"])

    assert record["name"] == "Round Trip (imported)"
    assert record["description"] == "all fields"
    assert record["strategy"] == "character_lore_first"
    assert record["max_tokens"] == 750
    src_entry = service.get_dictionary(created["id"])["entries"][0]
    dup_entry = record["entries"][0]
    for field in ("pattern", "replacement", "probability", "group", "timed_effects",
                  "max_replacements", "type", "enabled", "case_sensitive", "priority"):
        assert dup_entry.get(field) == src_entry.get(field), field


def _seed_conversation(db, title="Chat"):
    return db.add_conversation({"title": title})


def _active(db, conv_id):
    meta = _json.loads(db.get_conversation_by_id(conv_id).get("metadata") or "{}")
    return meta.get("active_dictionaries", [])


def test_attach_is_idempotent_and_dedups(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    d = service.create_dictionary({"name": "Meds"})
    conv = _seed_conversation(dictionary_db)

    r1 = service.attach_to_conversation(d["id"], conv)
    assert r1["active_dictionaries"] == [d["id"]]
    r2 = service.attach_to_conversation(d["id"], conv)          # idempotent
    assert r2["active_dictionaries"] == [d["id"]]               # no duplicate
    assert _active(dictionary_db, conv) == [d["id"]]            # persisted as int


def test_detach_removes_and_noop_when_absent(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    d = service.create_dictionary({"name": "Meds"})
    conv = _seed_conversation(dictionary_db)
    service.attach_to_conversation(d["id"], conv)
    service.detach_from_conversation(d["id"], conv)
    assert _active(dictionary_db, conv) == []
    # not-attached -> no-op success
    again = service.detach_from_conversation(d["id"], conv)
    assert again["active_dictionaries"] == []


def test_attach_missing_conversation_raises(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    d = service.create_dictionary({"name": "Meds"})
    with pytest.raises(ValueError):
        service.attach_to_conversation(d["id"], "does-not-exist")


def test_used_by_exact_int_membership_not_substring(dictionary_db):
    # THE 1-vs-11 trap: dict id 1 must NOT match a conversation holding only id 11.
    # Ids auto-increment from 1, so create 11 dictionaries to force a real id-11
    # (a function-scoped 2-dictionary fixture only ever yields ids 1 and 2, which
    # can't distinguish exact-int membership from substring matching).
    service = LocalChatDictionaryService(dictionary_db)
    created = [service.create_dictionary({"name": f"Dict {n}"}) for n in range(1, 12)]
    d1, d11 = created[0], created[10]
    assert d1["id"] == 1
    assert d11["id"] == 11

    # Attach ONLY the id-11 dictionary to a conversation.
    conv_with_11 = _seed_conversation(dictionary_db, "has 11")
    service.attach_to_conversation(d11["id"], conv_with_11)

    # Dict id 1 is not attached anywhere: a substring LIKE match ('%1%') would
    # wrongly catch conv_with_11 (metadata contains "11", which contains "1").
    used_by_1 = service.list_dictionary_conversations(d1["id"])
    assert used_by_1["conversations"] == []

    # Attach id 1 elsewhere and confirm its used-by returns exactly that one
    # conversation, not the id-11 conversation.
    conv_with_1 = _seed_conversation(dictionary_db, "has 1")
    service.attach_to_conversation(d1["id"], conv_with_1)
    used_by_1 = service.list_dictionary_conversations(d1["id"])
    ids = {c["conversation_id"] for c in used_by_1["conversations"]}
    assert ids == {conv_with_1}
    titles = {c["title"] for c in used_by_1["conversations"]}
    assert titles == {"has 1"}


def test_used_by_tolerates_non_list_active_dictionaries_value(dictionary_db):
    # Well-formed JSON, but a malformed *value* under a good key:
    # {"active_dictionaries": 5} is valid JSON yet not iterable as a member
    # list. list_dictionary_conversations() scans ALL matching rows, so one
    # pathological row like this must not crash used-by for every dictionary.
    service = LocalChatDictionaryService(dictionary_db)
    d = service.create_dictionary({"name": "Meds"})
    good_conv = _seed_conversation(dictionary_db, "good")
    service.attach_to_conversation(d["id"], good_conv)

    bad_conv = dictionary_db.add_conversation({"title": "bad metadata"})
    bad_record = dictionary_db.get_conversation_by_id(bad_conv)
    dictionary_db.update_conversation(
        bad_conv,
        {"metadata": _json.dumps({"active_dictionaries": 5})},
        expected_version=bad_record["version"],
    )

    result = service.list_dictionary_conversations(d["id"])  # must not raise
    ids = {c["conversation_id"] for c in result["conversations"]}
    assert ids == {good_conv}
    assert bad_conv not in ids


def test_attach_survives_non_dict_metadata_json(dictionary_db):
    """Regression for Roleplay P1e final-review #2.

    A conversation's `metadata` can be valid JSON but not a JSON object
    (e.g. a bare scalar like ``"5"``). `_active_dictionaries`/
    `_write_active_dictionaries` used to call `.get()`/`__setitem__` on
    whatever `json.loads()` returned, which raises `AttributeError` /
    `TypeError` for a non-dict value (only `json.loads()` itself was
    guarded, not the shape of its result). Attach must recover by treating
    a non-dict decode result as an empty dict.
    """
    service = LocalChatDictionaryService(dictionary_db)
    d = service.create_dictionary({"name": "Meds"})
    conv = _seed_conversation(dictionary_db)
    record = dictionary_db.get_conversation_by_id(conv)
    dictionary_db.update_conversation(conv, {"metadata": "5"}, expected_version=record["version"])

    result = service.attach_to_conversation(d["id"], conv)
    assert result["active_dictionaries"] == [d["id"]]
    assert _active(dictionary_db, conv) == [d["id"]]


def test_attach_conflict_on_stale_version(dictionary_db, monkeypatch):
    service = LocalChatDictionaryService(dictionary_db)
    d = service.create_dictionary({"name": "Meds"})
    conv = _seed_conversation(dictionary_db)
    # Make update_conversation report a version mismatch.
    real_update = dictionary_db.update_conversation

    def _stale(conversation_id, update_data, expected_version):
        raise ConflictError("version mismatch")

    monkeypatch.setattr(dictionary_db, "update_conversation", _stale)
    with pytest.raises(ConflictError):
        service.attach_to_conversation(d["id"], conv)


def _make_dict_with_entries(service, name):
    created = service.create_dictionary(
        {"name": name, "entries": [{"pattern": "BP", "replacement": "blood pressure"}]}
    )
    return created["id"]


def test_attach_to_character_embeds_content_snapshot(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    dict_id = _make_dict_with_entries(service, "Slang")
    char_id = dictionary_db.add_character_card({"name": "Noir"})

    result = service.attach_to_character(dict_id, char_id)

    assert result["dictionary_name"] == "Slang"
    assert result["character_dictionaries"] == ["Slang"]
    # The full content snapshot (not just an id) is embedded in extensions.
    record = dictionary_db.get_character_card_by_id(char_id)
    blocks = record["extensions"]["chat_dictionaries"]
    assert len(blocks) == 1
    assert blocks[0]["name"] == "Slang"
    assert blocks[0]["entries"], "entries content must be embedded"
    # Version bumped by the optimistic-locked write.
    assert record["version"] == 2


def test_attach_to_character_is_idempotent_by_name(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    dict_id = _make_dict_with_entries(service, "Slang")
    char_id = dictionary_db.add_character_card({"name": "Noir"})

    service.attach_to_character(dict_id, char_id)
    result = service.attach_to_character(dict_id, char_id)  # second attach = no-op

    assert result["character_dictionaries"] == ["Slang"]
    record = dictionary_db.get_character_card_by_id(char_id)
    assert len(record["extensions"]["chat_dictionaries"]) == 1
    assert record["version"] == 2  # no extra write on the idempotent re-attach


def test_detach_from_character_removes_by_name(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    dict_id = _make_dict_with_entries(service, "Slang")
    char_id = dictionary_db.add_character_card({"name": "Noir"})
    service.attach_to_character(dict_id, char_id)

    result = service.detach_from_character(char_id, "Slang")

    assert result["character_dictionaries"] == []
    record = dictionary_db.get_character_card_by_id(char_id)
    assert record["extensions"].get("chat_dictionaries") == []


def test_list_character_dictionaries_summarizes_embedded(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    dict_id = _make_dict_with_entries(service, "Slang")
    char_id = dictionary_db.add_character_card({"name": "Noir"})
    service.attach_to_character(dict_id, char_id)

    listing = service.list_character_dictionaries(char_id)

    assert listing["dictionaries"] == [{"name": "Slang", "entry_count": 1, "enabled": True}]


def test_attach_to_missing_character_raises_value_error(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    dict_id = _make_dict_with_entries(service, "Slang")
    with pytest.raises(ValueError):
        service.attach_to_character(dict_id, 999999)


def test_attach_to_character_raises_conflict_on_stale_version(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    dict_id = _make_dict_with_entries(service, "Slang")
    char_id = dictionary_db.add_character_card({"name": "Noir"})
    # Bump the character version out from under a captured-stale record.
    stale = dictionary_db.get_character_card_by_id(char_id)
    dictionary_db.update_character_card(char_id, {"name": "Noir2"}, expected_version=stale["version"])
    # Monkeypatch the load to return the stale record so the write uses version 1.
    service._load_character_or_raise = lambda cid: stale  # type: ignore[assignment]
    with pytest.raises(ConflictError):
        service.attach_to_character(dict_id, char_id)
