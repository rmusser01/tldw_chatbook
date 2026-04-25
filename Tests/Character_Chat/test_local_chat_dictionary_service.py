import pytest

from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


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
