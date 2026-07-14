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
