# Roleplay P1f — Character-Level Dictionary Attachment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a character own dictionaries as portable embedded snapshots (`character_cards.extensions['chat_dictionaries']`) that apply to all of that character's chats, attached/detached instantly from the character view.

**Architecture:** A character embeds `export_json(id)["data"]` content blocks in its card `extensions` (portable, snapshot semantics, deduped by dictionary name). At send time a testable helper unions the conversation's by-id dictionaries with the character's embedded dictionaries — deduped by name, conversation wins a collision — and `chat_events.py` calls it in one line. The attach UI is an I/O-free panel + a dictionary picker; the screen owns all service/DB work in `personas-io` workers and keeps the character editor's in-memory base coherent after an out-of-band attach.

**Tech Stack:** Python 3.11+, Textual, SQLite (CharactersRAGDB / ChaChaNotes schema v20), pytest.

## Global Constraints

- **No schema migration.** `update_character_card(character_id, card_data, expected_version)` (`ChaChaNotes_DB.py:3853`) already whitelists `extensions` (JSON field) and is optimistic-locked; `get_character_card_by_id(character_id)` (`:3691`) deserializes `extensions` to a dict. Do NOT add a table or migration.
- **Embedded snapshot = `export_json(dictionary_id)["data"]`** — `{name, description, content, entries[...], strategy, max_tokens, enabled, version}`; entries are `ChatDictionary.to_dict()` shape (via `_entries_payload`). Re-parse with `ChatDictionary.from_dict(e)` (covers `enabled`/`case_sensitive`/`priority` losslessly).
- **Identity = dictionary name.** `chat_dictionaries.name` is `UNIQUE NOT NULL`. Dedup, detach, and the runtime union all key on name. A character never embeds two dicts of the same name.
- **Runtime dedup is at the dictionary level, before entries are flattened.** Additive union; the conversation's by-id dict wins a name collision; only `enabled` dicts contribute.
- **Typing seam:** dictionary ids are `int`; character ids are `int` (the DB uses int character ids, but `state.selected_entity_id` is a `str` — cast with `int(entity_id)`). Conversation attach (P1e) is unchanged.
- **Untrusted embedded content:** an imported card's `chat_dictionaries` is hostile input — the runtime parse must never raise into the send path (skip malformed).
- **Widget is I/O-free:** panels post intent messages and render fed rows; the screen owns every service/DB call in `personas-io` workers.
- **Test env (venv-only):** prefix every pytest run with `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share` and use `-p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`. Python: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`.

---

### Task 1: Character attach/detach service seam

**Files:**
- Modify: `tldw_chatbook/Character_Chat/local_chat_dictionary_service.py` (add methods near the P1e `attach_to_conversation` block, ~line 700)
- Test: `Tests/Character_Chat/test_local_chat_dictionary_service.py` (append)

**Interfaces:**
- Consumes: existing `self.export_json(dictionary_id) -> {"data": {...}}`, `self._require_db()` → `CharactersRAGDB` with `get_character_card_by_id(int) -> dict|None` (extensions deserialized to a dict) and `update_character_card(int, {"extensions": dict}, expected_version=int)` (raises `ConflictError` on stale version). `Mapping` and `json` are already imported in this module.
- Produces:
  - `attach_to_character(dictionary_id: int, character_id: int) -> {"dictionary_id": int, "character_id": int, "dictionary_name": str, "character_dictionaries": list[str], "source": "local"}`
  - `detach_from_character(character_id: int, dictionary_name: str) -> {"character_id": int, "dictionary_name": str, "character_dictionaries": list[str], "source": "local"}`
  - `list_character_dictionaries(character_id: int) -> {"dictionaries": [{"name": str, "entry_count": int, "enabled": bool}], "source": "local"}`

- [ ] **Step 1: Write the failing tests.** Append to `Tests/Character_Chat/test_local_chat_dictionary_service.py`:

```python
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
```

Ensure `import pytest` and `from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError` are present at the top (they are, per the existing fixture).

- [ ] **Step 2: Run — expect FAIL** (`AttributeError: ... has no attribute 'attach_to_character'`).

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_local_chat_dictionary_service.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 3: Implement.** Add to `LocalChatDictionaryService` (place directly after `list_dictionary_conversations`):

```python
    def _load_character_or_raise(self, character_id: int) -> dict[str, Any]:
        record = self._require_db().get_character_card_by_id(int(character_id))
        if record is None:
            raise ValueError(f"Character '{character_id}' was not found.")
        return record

    @staticmethod
    def _embedded_dictionaries(record: Mapping[str, Any]) -> list[dict[str, Any]]:
        ext = record.get("extensions")
        if isinstance(ext, str):
            try:
                ext = json.loads(ext or "{}")
            except (TypeError, ValueError):
                ext = {}
        if not isinstance(ext, dict):
            ext = {}
        raw = ext.get("chat_dictionaries") or []
        if not isinstance(raw, list):
            raw = []
        return [b for b in raw if isinstance(b, dict) and b.get("name")]

    def _write_embedded_dictionaries(
        self, record: dict[str, Any], character_id: int, blocks: list[dict[str, Any]]
    ) -> None:
        ext = record.get("extensions")
        if isinstance(ext, str):
            try:
                ext = json.loads(ext or "{}")
            except (TypeError, ValueError):
                ext = {}
        if not isinstance(ext, dict):
            ext = {}
        ext["chat_dictionaries"] = blocks
        self._require_db().update_character_card(
            int(character_id), {"extensions": ext}, expected_version=record["version"]
        )

    def attach_to_character(self, dictionary_id: int, character_id: int) -> dict[str, Any]:
        """Embed a dictionary's content snapshot into a character (idempotent by name).

        Raises:
            ValueError: If the dictionary or the character does not exist.
            ConflictError: If the character's version is stale at write time.
        """
        block = self.export_json(int(dictionary_id))["data"]
        name = block.get("name")
        record = self._load_character_or_raise(character_id)
        blocks = self._embedded_dictionaries(record)
        if not any(b.get("name") == name for b in blocks):
            blocks = blocks + [block]
            self._write_embedded_dictionaries(record, character_id, blocks)
        return {
            "dictionary_id": int(dictionary_id),
            "character_id": int(character_id),
            "dictionary_name": name,
            "character_dictionaries": [b.get("name") for b in blocks],
            "source": "local",
        }

    def detach_from_character(self, character_id: int, dictionary_name: str) -> dict[str, Any]:
        """Remove an embedded dictionary from a character by name (no-op when absent).

        Raises:
            ValueError: If the character does not exist.
            ConflictError: If the character's version is stale at write time.
        """
        record = self._load_character_or_raise(character_id)
        blocks = self._embedded_dictionaries(record)
        if any(b.get("name") == dictionary_name for b in blocks):
            blocks = [b for b in blocks if b.get("name") != dictionary_name]
            self._write_embedded_dictionaries(record, character_id, blocks)
        return {
            "character_id": int(character_id),
            "dictionary_name": str(dictionary_name),
            "character_dictionaries": [b.get("name") for b in blocks],
            "source": "local",
        }

    def list_character_dictionaries(self, character_id: int) -> dict[str, Any]:
        """Summarize a character's embedded dictionaries (from the snapshots only)."""
        record = self._load_character_or_raise(character_id)
        dictionaries = [
            {
                "name": b.get("name"),
                "entry_count": len(b.get("entries") or []),
                "enabled": bool(b.get("enabled", True)),
            }
            for b in self._embedded_dictionaries(record)
        ]
        return {"dictionaries": dictionaries, "source": "local"}
```

- [ ] **Step 4: Run — all PASS.**

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_local_chat_dictionary_service.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 5: Commit** — `feat(chat-dictionaries): character attach/detach/list over embedded snapshots` + `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 2: Scope-service async wrappers

**Files:**
- Modify: `tldw_chatbook/Character_Chat/chat_dictionary_scope_service.py` (after `list_dictionary_conversations`, ~line 355)
- Test: `Tests/Character_Chat/test_chat_dictionary_scope_service.py` (append)

**Interfaces:**
- Consumes: Task 1's `LocalChatDictionaryService.attach_to_character` / `detach_from_character` / `list_character_dictionaries`; existing `self._normalize_mode`, `self._dictionary_action(mode, "update")`, `self._statistics_action(mode, "detail")`, `self._invoke(mode, action_id, method_name, *args)`.
- Produces: async `attach_to_character(dictionary_id, character_id, mode="local")`, `detach_from_character(character_id, dictionary_name, mode="local")`, `list_character_dictionaries(character_id, mode="local")`.

- [ ] **Step 1: Write the failing test.** Append to `Tests/Character_Chat/test_chat_dictionary_scope_service.py`:

```python
async def test_scope_service_character_attach_roundtrip(tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
    from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService

    db = CharactersRAGDB(tmp_path / "scope.db", "test-client")
    local = LocalChatDictionaryService(db)
    scope = ChatDictionaryScopeService(local_service=local)
    dict_id = local.create_dictionary({"name": "Slang", "entries": [{"pattern": "x", "replacement": "y"}]})["id"]
    char_id = db.add_character_card({"name": "Noir"})

    await scope.attach_to_character(dict_id, char_id, mode="local")
    listing = await scope.list_character_dictionaries(char_id, mode="local")
    assert [d["name"] for d in listing["dictionaries"]] == ["Slang"]

    await scope.detach_from_character(char_id, "Slang", mode="local")
    listing = await scope.list_character_dictionaries(char_id, mode="local")
    assert listing["dictionaries"] == []
```

Match the existing file's `ChatDictionaryScopeService(...)` construction (check how sibling tests build it — the constructor kwarg name may be `local_service=` or positional; mirror the existing tests).

- [ ] **Step 2: Run — expect FAIL** (`AttributeError: ... 'attach_to_character'`).

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_chat_dictionary_scope_service.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 3: Implement.** Add to `ChatDictionaryScopeService` (after `list_dictionary_conversations`):

```python
    async def attach_to_character(self, dictionary_id: int, character_id: int, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._dictionary_action(normalized_mode, "update"),
            "attach_to_character",
            dictionary_id,
            character_id,
        )

    async def detach_from_character(self, character_id: int, dictionary_name: str, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._dictionary_action(normalized_mode, "update"),
            "detach_from_character",
            character_id,
            dictionary_name,
        )

    async def list_character_dictionaries(self, character_id: int, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._statistics_action(normalized_mode, "detail"),
            "list_character_dictionaries",
            character_id,
        )
```

- [ ] **Step 4: Run — PASS.** (same command as Step 2)

- [ ] **Step 5: Commit** — `feat(chat-dictionaries): scope-service async wrappers for character attach`.

---

### Task 3: Defensive parse of a character's embedded dictionaries

**Files:**
- Modify: `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py` (add a module-level function near `load_chat_dictionary`, ~line 1021)
- Test: `Tests/Character_Chat/test_load_character_dictionaries.py` (create)

**Interfaces:**
- Consumes: `ChatDictionary.from_dict(e)` (same file); `json` (already imported).
- Produces: `load_character_dictionaries(char_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]`, each element `{"name": str, "enabled": bool, "entries": list[ChatDictionary]}`. Never raises.

- [ ] **Step 1: Write the failing tests.** Create `Tests/Character_Chat/test_load_character_dictionaries.py`:

```python
"""P1f: defensive parse of a character card's embedded chat dictionaries."""

from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import load_character_dictionaries, ChatDictionary


def test_parses_embedded_blocks_into_chatdictionary_entries():
    char = {"extensions": {"chat_dictionaries": [
        {"name": "Slang", "enabled": True, "entries": [
            {"key": "BP", "content": "blood pressure", "probability": 100},
        ]},
    ]}}
    blocks = load_character_dictionaries(char)
    assert len(blocks) == 1
    assert blocks[0]["name"] == "Slang"
    assert blocks[0]["enabled"] is True
    assert len(blocks[0]["entries"]) == 1
    assert isinstance(blocks[0]["entries"][0], ChatDictionary)


def test_skips_malformed_blocks_and_entries_without_raising():
    char = {"extensions": {"chat_dictionaries": [
        "not-a-dict",
        {"name": "", "entries": []},                 # no name → skipped
        {"name": "Bad", "entries": [{"content": "x"}]},  # entry missing 'key' → entry skipped
        {"name": "Good", "entries": [{"key": "k", "content": "c"}]},
    ]}}
    blocks = load_character_dictionaries(char)
    names = [b["name"] for b in blocks]
    assert names == ["Bad", "Good"]
    assert load_character_dictionaries(char)[0]["entries"] == []  # 'Bad' dropped its bad entry


def test_tolerates_none_and_missing_and_string_extensions():
    assert load_character_dictionaries(None) == []
    assert load_character_dictionaries({}) == []
    assert load_character_dictionaries({"extensions": {}}) == []
    assert load_character_dictionaries({"extensions": '{"chat_dictionaries": []}'}) == []
    assert load_character_dictionaries({"extensions": "not json"}) == []
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: cannot import name 'load_character_dictionaries'`).

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_load_character_dictionaries.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 3: Implement.** Add to `Chat_Dictionary_Lib.py` (after `load_chat_dictionary`):

```python
def load_character_dictionaries(char_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse a character card's embedded chat dictionaries into runtime blocks.

    Reads ``extensions['chat_dictionaries']`` (a list of ``export_json`` ``data``
    blocks) and returns one ``{"name", "enabled", "entries": [ChatDictionary...]}``
    per well-formed block. Malformed blocks/entries are skipped. This runs on the
    chat send path over untrusted (imported) card content, so it MUST NOT raise.
    """
    result: List[Dict[str, Any]] = []
    if not isinstance(char_data, dict):
        return result
    ext = char_data.get('extensions')
    if isinstance(ext, str):
        try:
            ext = json.loads(ext or "{}")
        except (TypeError, ValueError):
            ext = {}
    if not isinstance(ext, dict):
        return result
    raw = ext.get('chat_dictionaries') or []
    if not isinstance(raw, list):
        return result
    for block in raw:
        if not isinstance(block, dict):
            continue
        name = block.get('name')
        if not name:
            continue
        entries: List[ChatDictionary] = []
        for entry in block.get('entries') or []:
            try:
                entries.append(ChatDictionary.from_dict(entry))
            except Exception:
                continue
        result.append({
            "name": str(name),
            "enabled": bool(block.get('enabled', True)),
            "entries": entries,
        })
    return result
```

Confirm `Optional`, `Dict`, `Any`, `List` are imported at the top of the module (they are — `load_chat_dictionary` uses them).

- [ ] **Step 4: Run — PASS.** (same command as Step 2)

- [ ] **Step 5: Commit** — `feat(chat-dictionaries): defensive parse of a character's embedded dictionaries`.

---

### Task 4: Runtime union at the send path

**Files:**
- Modify: `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py` (add `collect_active_chatdict_entries` after `load_character_dictionaries`)
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py:~977-1002` (replace the inline dictionary-loading loop with a call to the new helper)
- Test: `Tests/Character_Chat/test_collect_active_chatdict_entries.py` (create)

**Interfaces:**
- Consumes: `load_chat_dictionary(db, dict_id)` (returns `{"name", "enabled", "entries": [ChatDictionary...]}`), Task 3's `load_character_dictionaries(char_data)`, `CharactersRAGDB.get_conversation_by_id(str)`.
- Produces: `collect_active_chatdict_entries(db, conversation_id: Optional[str], char_data: Optional[Dict[str, Any]]) -> List[ChatDictionary]` — the exact list the send path assigns to `chatdict_entries`. Never raises.

- [ ] **Step 1: Write the failing tests.** Create `Tests/Character_Chat/test_collect_active_chatdict_entries.py`:

```python
"""P1f: the runtime union (conversation + character dictionaries, dedup by name)."""

import json
import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import collect_active_chatdict_entries


@pytest.fixture
def db(tmp_path):
    return CharactersRAGDB(tmp_path / "collect.db", "test-client")


def _attach_conv_dict(db, service, conv_id, name):
    dict_id = service.create_dictionary({"name": name, "entries": [{"pattern": name, "replacement": name.lower()}]})["id"]
    conv = db.get_conversation_by_id(conv_id)
    meta = json.loads(conv.get("metadata") or "{}")
    meta["active_dictionaries"] = meta.get("active_dictionaries", []) + [dict_id]
    db.update_conversation(conv_id, {"metadata": json.dumps(meta)}, expected_version=conv["version"])
    return dict_id


def test_character_dictionary_fires_in_a_send(db):
    service = LocalChatDictionaryService(db)
    dict_id = service.create_dictionary({"name": "CharDict", "entries": [{"pattern": "hi", "replacement": "hello"}]})["id"]
    char_id = db.add_character_card({"name": "Noir"})
    service.attach_to_character(dict_id, char_id)
    char_data = db.get_character_card_by_id(char_id)

    entries = collect_active_chatdict_entries(db, None, char_data)

    assert len(entries) == 1
    assert entries[0].content == "hello"


def test_additive_union_conversation_wins_on_name_collision(db):
    service = LocalChatDictionaryService(db)
    conv_id = db.add_conversation({"title": "chat"})
    # Conversation attaches "Shared" (live). Character embeds "Shared" (snapshot) + "Extra".
    _attach_conv_dict(db, service, conv_id, "Shared")
    d_shared = service.create_dictionary({"name": "SharedSnap"})  # distinct source name is fine; embed under "Shared" below
    char_id = db.add_character_card({"name": "Noir"})
    # Embed a character "Shared" (same name as the conversation dict) + "Extra".
    extra_id = service.create_dictionary({"name": "Extra", "entries": [{"pattern": "e", "replacement": "x"}]})["id"]
    # Manually embed a same-named "Shared" snapshot to force the collision.
    rec = db.get_character_card_by_id(char_id)
    rec_ext = rec["extensions"] if isinstance(rec["extensions"], dict) else {}
    rec_ext["chat_dictionaries"] = [
        {"name": "Shared", "enabled": True, "entries": [{"key": "s", "content": "CHAR"}]},
        {"name": "Extra", "enabled": True, "entries": [{"key": "e", "content": "x"}]},
    ]
    db.update_character_card(char_id, {"extensions": rec_ext}, expected_version=rec["version"])
    char_data = db.get_character_card_by_id(char_id)

    entries = collect_active_chatdict_entries(db, conv_id, char_data)
    contents = sorted(e.content for e in entries)
    # "Shared" comes from the conversation (its content, "shared"), NOT the char snapshot ("CHAR");
    # "Extra" from the character. The char's "Shared" snapshot is dropped.
    assert "CHAR" not in contents
    assert "x" in contents  # Extra applied
    assert any(c == "shared" for c in contents)  # conversation Shared applied


def test_disabled_embedded_dictionary_is_skipped(db):
    char_id = db.add_character_card({"name": "Noir"})
    rec = db.get_character_card_by_id(char_id)
    rec_ext = rec["extensions"] if isinstance(rec["extensions"], dict) else {}
    rec_ext["chat_dictionaries"] = [{"name": "Off", "enabled": False, "entries": [{"key": "k", "content": "c"}]}]
    db.update_character_card(char_id, {"extensions": rec_ext}, expected_version=rec["version"])
    char_data = db.get_character_card_by_id(char_id)
    assert collect_active_chatdict_entries(db, None, char_data) == []


def test_malformed_embedded_content_never_raises(db):
    char_id = db.add_character_card({"name": "Noir"})
    rec = db.get_character_card_by_id(char_id)
    rec_ext = rec["extensions"] if isinstance(rec["extensions"], dict) else {}
    rec_ext["chat_dictionaries"] = "not-a-list"
    db.update_character_card(char_id, {"extensions": rec_ext}, expected_version=rec["version"])
    char_data = db.get_character_card_by_id(char_id)
    assert collect_active_chatdict_entries(db, None, char_data) == []


def test_conversation_only_output_unchanged(db):
    service = LocalChatDictionaryService(db)
    conv_id = db.add_conversation({"title": "chat"})
    _attach_conv_dict(db, service, conv_id, "OnlyConv")
    # No active character.
    entries = collect_active_chatdict_entries(db, conv_id, None)
    assert len(entries) == 1
    assert entries[0].content == "onlyconv"
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: cannot import name 'collect_active_chatdict_entries'`).

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_collect_active_chatdict_entries.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 3a: Implement the helper.** Add to `Chat_Dictionary_Lib.py` (after `load_character_dictionaries`):

```python
def collect_active_chatdict_entries(
    db: "CharactersRAGDB",
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
) -> List[ChatDictionary]:
    """Collect the ChatDictionary entries that apply to the current send.

    Additive union of the conversation's attached dictionaries (by id, from
    ``metadata.active_dictionaries``) and the active character's embedded
    dictionaries (snapshots in ``extensions.chat_dictionaries``), deduped at the
    dictionary level by name — the conversation's dictionary WINS a name
    collision. Only enabled dictionaries contribute. Never raises: any bad row is
    skipped so a chat send is never broken by dictionary loading.
    """
    entries: List[ChatDictionary] = []
    conversation_dict_names: set = set()
    if conversation_id and db is not None:
        try:
            conv_details = db.get_conversation_by_id(conversation_id)
        except Exception:
            conv_details = None
        if conv_details:
            try:
                metadata = json.loads(conv_details.get('metadata') or '{}')
            except (TypeError, ValueError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            for dict_id in metadata.get('active_dictionaries') or []:
                try:
                    dict_data = load_chat_dictionary(db, dict_id)
                except Exception:
                    continue
                if dict_data and dict_data.get('enabled', True):
                    conversation_dict_names.add(dict_data.get('name'))
                    entries.extend(dict_data.get('entries') or [])
    for block in load_character_dictionaries(char_data):
        if not block.get('enabled', True):
            continue
        if block.get('name') in conversation_dict_names:
            continue
        entries.extend(block.get('entries') or [])
    return entries
```

- [ ] **Step 3b: Rewire `chat_events.py`.** Replace the block at `~977-1002` (the `chatdict_entries = []` through the `except Exception as e:` that logs "Error loading chat dictionaries") with:

```python
    # --- 10.6. Apply Chat Dictionaries (conversation + active character) ---
    chatdict_entries = []
    if (app.current_chat_conversation_id or app.current_chat_active_character_data) and db:
        try:
            from ...Character_Chat import Chat_Dictionary_Lib as cdl
            chatdict_entries = cdl.collect_active_chatdict_entries(
                db,
                app.current_chat_conversation_id,
                app.current_chat_active_character_data,
            )
            if chatdict_entries:
                loguru_logger.info(f"Total chat dictionary entries loaded: {len(chatdict_entries)}")
        except Exception as e:
            loguru_logger.opt(exception=True).error(f"Error loading chat dictionaries: {e}")
            # Continue without dictionaries on error
```

- [ ] **Step 4: Run — helper tests PASS; then confirm chat_events still imports + its own tests are green.**

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_collect_active_chatdict_entries.py Tests/Chat/test_chat_functions.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.Event_Handlers.Chat_Events.chat_events; print('IMPORT OK')"
```

- [ ] **Step 5: Commit** — `feat(chat-dictionaries): runtime union of conversation + character dictionaries (conversation-wins)`.

---

### Task 5: Dictionary picker modal

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/dictionary_picker.py`
- Test: `Tests/UI/test_dictionary_picker.py` (create)

**Interfaces:**
- Produces: `DictionaryPicker(ModalScreen[int | None])`, `__init__(self, dictionaries: list[dict], **kwargs)` where each row is `{"dictionary_id": int, "name": str}`; a search `Input` filters by name; **Attach** → `dismiss(int)`, **Cancel** → `dismiss(None)`. DOM ids `#dict-pick-search`, `#dict-pick-list`, `#dict-pick-confirm`, `#dict-pick-cancel`. Ids are ints — never str-cast. Exclusion of already-attached dictionaries is done by the caller (the screen feeds a pre-filtered list).

- [ ] **Step 1: Write the failing tests.** Create `Tests/UI/test_dictionary_picker.py`:

```python
"""P1f: the dictionary picker returns int dictionary ids."""

import pytest
from textual.app import App
from textual.widgets import Input, ListView

from tldw_chatbook.Widgets.Persona_Widgets.dictionary_picker import DictionaryPicker

pytestmark = pytest.mark.asyncio

DICTS = [
    {"dictionary_id": 3, "name": "Slang"},
    {"dictionary_id": 7, "name": "Period Vocab"},
]


class _Host(App):
    def __init__(self):
        super().__init__()
        self.result = "unset"

    def on_mount(self):
        self.run_worker(self._drive)

    async def _drive(self):
        self.result = await self.push_screen_wait(DictionaryPicker(list(DICTS)))


async def test_picker_returns_selected_int_id():
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        picker = app.screen
        picker.query_one("#dict-pick-list", ListView).index = 1  # Period Vocab
        await pilot.pause()
        await pilot.click("#dict-pick-confirm")
        await pilot.pause()
    assert app.result == 7
    assert isinstance(app.result, int)


async def test_picker_search_filters_by_name():
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        picker = app.screen
        picker.query_one("#dict-pick-search", Input).value = "slang"
        await pilot.pause()
        assert len(picker.query_one("#dict-pick-list", ListView).children) == 1
        picker.query_one("#dict-pick-list", ListView).index = 0
        await pilot.pause()
        await pilot.click("#dict-pick-confirm")
        await pilot.pause()
    assert app.result == 3


async def test_picker_cancel_returns_none():
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#dict-pick-cancel")
        await pilot.pause()
    assert app.result is None
```

- [ ] **Step 2: Run — expect FAIL** (module missing).

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_dictionary_picker.py -q -p no:cacheprovider -o addopts="" --timeout=180 --timeout-method=thread
```

- [ ] **Step 3: Implement.** Create `tldw_chatbook/Widgets/Persona_Widgets/dictionary_picker.py`:

```python
"""A small modal for picking a dictionary to attach to a character (Roleplay P1f).

Distinct from ``DictionaryAttachPicker`` (which picks a conversation and returns a
string id); this one lists dictionaries and returns the picked int dictionary id.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static


class DictionaryPicker(ModalScreen[int | None]):
    """Pick one dictionary (by int id) to attach to the current character.

    Args:
        dictionaries: ``{"dictionary_id": int, "name": str}`` rows to choose from
            (already filtered to those not yet attached to the character).
    """

    DEFAULT_CSS = """
    DictionaryPicker { align: center middle; }
    DictionaryPicker > Vertical {
        width: 60%; max-width: 80; height: auto; max-height: 80%;
        padding: 1 2; border: round $panel;
    }
    DictionaryPicker #dict-pick-list { height: auto; max-height: 16; }
    """

    def __init__(self, dictionaries: list[dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._dictionaries = list(dictionaries)
        self._row_ids: list[int] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Attach dictionary", markup=False)
            yield Input(placeholder="Search dictionaries…", id="dict-pick-search")
            yield ListView(id="dict-pick-list")
            with Vertical(id="dict-pick-actions"):
                yield Button("Attach", id="dict-pick-confirm", classes="console-action-secondary")
                yield Button("Cancel", id="dict-pick-cancel", classes="console-action-secondary")

    def on_mount(self) -> None:
        self._populate(self._dictionaries)

    def _populate(self, rows: list[dict[str, Any]]) -> None:
        listing = self.query_one("#dict-pick-list", ListView)
        listing.clear()
        self._row_ids = []
        for row in rows:
            listing.append(ListItem(Static(str(row.get("name") or "(unnamed)"), markup=False)))
            self._row_ids.append(int(row.get("dictionary_id")))
        listing.index = None

    @on(Input.Changed, "#dict-pick-search")
    def _filter(self, event: Input.Changed) -> None:
        event.stop()
        needle = event.value.strip().lower()
        rows = [d for d in self._dictionaries if needle in str(d.get("name") or "").lower()] if needle else self._dictionaries
        self._populate(rows)

    def _selected_id(self) -> int | None:
        listing = self.query_one("#dict-pick-list", ListView)
        index = listing.index
        if index is None or not 0 <= index < len(self._row_ids):
            return None
        return self._row_ids[index]

    @on(Button.Pressed, "#dict-pick-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(self._selected_id())

    @on(Button.Pressed, "#dict-pick-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)


__all__ = ["DictionaryPicker"]
```

- [ ] **Step 4: Run — all 3 PASS.** (same command as Step 2)

- [ ] **Step 5: Commit** — `feat(personas): dictionary picker modal (int ids) for character attach`.

---

### Task 6: Character dictionaries panel (I/O-free)

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_dictionaries.py`
- Test: `Tests/UI/test_personas_character_dictionaries.py` (create)

**Interfaces:**
- Produces: `PersonasCharacterDictionariesWidget(Container)` with DOM id `#personas-character-dictionaries`; a DataTable `#personas-char-dicts-table` (columns `dictionary · entries`, `cursor_type="row"`, row key = dictionary name), an empty-state Static `#personas-char-dicts-empty`, an **Attach dictionary…** Button `#personas-char-dicts-add`, a **Detach** Button `#personas-char-dicts-detach`. Module-level messages `CharacterDictionaryAttachRequested(Message)` and `CharacterDictionaryDetachRequested(Message)` (stores `dictionary_name: str`). API `load_character_dictionaries(rows: list[dict])` where each row is `{"name": str, "entry_count": int, "enabled": bool}`.

- [ ] **Step 1: Write the failing tests.** Create `Tests/UI/test_personas_character_dictionaries.py`:

```python
"""P1f: the I/O-free character dictionaries panel."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_character_dictionaries import (
    PersonasCharacterDictionariesWidget,
    CharacterDictionaryAttachRequested,
    CharacterDictionaryDetachRequested,
)

pytestmark = pytest.mark.asyncio


class _Host(App):
    def compose(self) -> ComposeResult:
        yield PersonasCharacterDictionariesWidget()


async def test_empty_state_when_no_dictionaries():
    async with _Host().run_test(size=(120, 40)) as pilot:
        panel = pilot.app.query_one(PersonasCharacterDictionariesWidget)
        panel.load_character_dictionaries([])
        await pilot.pause()
        empty = pilot.app.query_one("#personas-char-dicts-empty", Static)
        assert empty.display is True
        assert pilot.app.query_one("#personas-char-dicts-table", DataTable).display is False


async def test_load_renders_rows():
    async with _Host().run_test(size=(120, 40)) as pilot:
        panel = pilot.app.query_one(PersonasCharacterDictionariesWidget)
        panel.load_character_dictionaries([{"name": "Slang", "entry_count": 2, "enabled": True}])
        await pilot.pause()
        table = pilot.app.query_one("#personas-char-dicts-table", DataTable)
        assert table.row_count == 1
        assert "Slang" in str(table.get_cell_at((0, 0)))


async def test_attach_button_posts_intent():
    posted = []

    class _CaptureHost(_Host):
        def on_character_dictionary_attach_requested(self, m: CharacterDictionaryAttachRequested):
            posted.append(m)

    async with _CaptureHost().run_test(size=(120, 40)) as pilot:
        await pilot.click("#personas-char-dicts-add")
        await pilot.pause()
    assert len(posted) == 1


async def test_detach_button_posts_intent_with_name():
    posted = []

    class _CaptureHost(_Host):
        def on_character_dictionary_detach_requested(self, m: CharacterDictionaryDetachRequested):
            posted.append(m.dictionary_name)

    async with _CaptureHost().run_test(size=(120, 40)) as pilot:
        panel = pilot.app.query_one(PersonasCharacterDictionariesWidget)
        panel.load_character_dictionaries([{"name": "Slang", "entry_count": 1, "enabled": True}])
        await pilot.pause()
        pilot.app.query_one("#personas-char-dicts-table", DataTable).move_cursor(row=0)
        await pilot.click("#personas-char-dicts-detach")
        await pilot.pause()
    assert posted == ["Slang"]
```

- [ ] **Step 2: Run — expect FAIL** (module missing).

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_character_dictionaries.py -q -p no:cacheprovider -o addopts="" --timeout=180 --timeout-method=thread
```

- [ ] **Step 3: Implement.** Create `tldw_chatbook/Widgets/Persona_Widgets/personas_character_dictionaries.py`:

```python
"""Roleplay P1f: an I/O-free panel listing a character's embedded dictionaries.

The panel renders what the screen feeds via ``load_character_dictionaries`` and
posts intent messages; the screen owns all service/DB work. Each embedded
dictionary is a snapshot (an embedded copy — editing the source dictionary does
not update it).
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Button, DataTable, Static


class CharacterDictionaryAttachRequested(Message):
    """Request the attach-dictionary picker for the current character."""


class CharacterDictionaryDetachRequested(Message):
    """Detach one embedded dictionary from the current character.

    Args:
        dictionary_name: The embedded dictionary to remove (by name).
    """

    def __init__(self, dictionary_name: str) -> None:
        super().__init__()
        self.dictionary_name = dictionary_name


class PersonasCharacterDictionariesWidget(Container):
    """List + attach/detach a character's embedded dictionaries (snapshots)."""

    DEFAULT_CSS = """
    PersonasCharacterDictionariesWidget #personas-char-dicts-table { height: auto; max-height: 8; }
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "personas-character-dictionaries")
        super().__init__(**kwargs)
        self._rows: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Static("Dictionaries (embedded copies)", classes="destination-section")
        yield Static(
            "No dictionaries attached to this character yet.",
            id="personas-char-dicts-empty",
            markup=False,
        )
        yield DataTable(id="personas-char-dicts-table", cursor_type="row")
        with Horizontal(classes="personas-dict-form-row"):
            yield Button("Attach dictionary…", id="personas-char-dicts-add", classes="console-action-secondary")
            yield Button("Detach", id="personas-char-dicts-detach", classes="console-action-secondary")

    def on_mount(self) -> None:
        self.query_one("#personas-char-dicts-table", DataTable).add_columns("dictionary", "entries")
        self.load_character_dictionaries([])

    def load_character_dictionaries(self, rows: list[dict[str, Any]]) -> None:
        """Render the character's embedded dictionaries.

        Args:
            rows: ``{"name": str, "entry_count": int, "enabled": bool}`` entries.
        """
        self._rows = list(rows)
        table = self.query_one("#personas-char-dicts-table", DataTable)
        table.clear()
        for row in self._rows:
            table.add_row(
                Text(str(row.get("name") or "(unnamed)")),
                Text(str(row.get("entry_count") if row.get("entry_count") is not None else "")),
                key=str(row.get("name")),
            )
        empty = self.query_one("#personas-char-dicts-empty", Static)
        empty.display = not self._rows
        table.display = bool(self._rows)

    def _selected_name(self) -> str | None:
        table = self.query_one("#personas-char-dicts-table", DataTable)
        if table.row_count == 0 or table.cursor_row is None or table.cursor_row < 0:
            return None
        try:
            return str(table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value)
        except Exception:
            return None

    @on(Button.Pressed, "#personas-char-dicts-add")
    def _attach(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(CharacterDictionaryAttachRequested())

    @on(Button.Pressed, "#personas-char-dicts-detach")
    def _detach(self, event: Button.Pressed) -> None:
        event.stop()
        name = self._selected_name()
        if name is not None:
            self.post_message(CharacterDictionaryDetachRequested(name))


__all__ = [
    "PersonasCharacterDictionariesWidget",
    "CharacterDictionaryAttachRequested",
    "CharacterDictionaryDetachRequested",
]
```

- [ ] **Step 4: Run — all 4 PASS.** (same command as Step 2)

- [ ] **Step 5: Commit** — `feat(personas): I/O-free character dictionaries panel + attach/detach intents`.

---

### Task 7: Editor base-copy coherence

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py` (add a method)
- Test: `Tests/UI/test_personas_character_editor_sync.py` (create)

**Interfaces:**
- Consumes: the editor's `self._character_data: Dict[str, Any]` base copy (set in `load_character`); `get_character_data()` starts from `dict(self._character_data)`.
- Produces: `PersonasCharacterEditorWidget.sync_attached_dictionaries(chat_dictionaries: list, new_version: int) -> None` — patches the base copy's `extensions['chat_dictionaries']` and `version` in place (leaving form-edited fields untouched), so a subsequent Save preserves an out-of-band attach without a version conflict.

- [ ] **Step 1: Write the failing test.** Create `Tests/UI/test_personas_character_editor_sync.py`:

```python
"""P1f: an out-of-band attach patches the editor base without a clobber."""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)

pytestmark = pytest.mark.asyncio


class _Host(App):
    def compose(self) -> ComposeResult:
        yield PersonasCharacterEditorWidget()


async def test_sync_patches_base_and_survives_get_character_data():
    async with _Host().run_test(size=(120, 40)) as pilot:
        editor = pilot.app.query_one(PersonasCharacterEditorWidget)
        editor.load_character({"id": 5, "name": "Noir", "version": 1, "extensions": {}})
        await pilot.pause()

        editor.sync_attached_dictionaries(
            [{"name": "Slang", "enabled": True, "entries": []}], new_version=2
        )
        data = editor.get_character_data()
        assert data["version"] == 2
        assert data["extensions"]["chat_dictionaries"][0]["name"] == "Slang"


async def test_sync_is_noop_without_a_loaded_character():
    async with _Host().run_test(size=(120, 40)) as pilot:
        editor = pilot.app.query_one(PersonasCharacterEditorWidget)
        # no load_character
        editor.sync_attached_dictionaries([{"name": "X", "entries": []}], new_version=9)
        assert editor._character_data == {}
```

Adjust the second test if `new_character()`/mount seeds `_character_data` — in that case assert the sync did not add `chat_dictionaries` when no id is present (the guard below keys on a truthy base). Keep the first (primary) assertion intact.

- [ ] **Step 2: Run — expect FAIL** (`AttributeError: ... 'sync_attached_dictionaries'`).

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_character_editor_sync.py -q -p no:cacheprovider -o addopts="" --timeout=180 --timeout-method=thread
```

- [ ] **Step 3: Implement.** Add to `PersonasCharacterEditorWidget` (near `get_character_data`):

```python
    def sync_attached_dictionaries(self, chat_dictionaries: list, new_version: int) -> None:
        """Patch the loaded base after an out-of-band dictionary attach/detach.

        Updates only ``extensions['chat_dictionaries']`` and ``version`` on the
        base copy the Save path starts from, so an instant attach is neither
        clobbered by a later Save nor forces a version conflict — and the user's
        in-progress form edits are left untouched. No-op when no character is
        loaded (empty base).
        """
        if not self._character_data:
            return
        ext = self._character_data.get("extensions")
        if not isinstance(ext, dict):
            ext = {}
        ext["chat_dictionaries"] = list(chat_dictionaries)
        self._character_data["extensions"] = ext
        self._character_data["version"] = new_version
```

- [ ] **Step 4: Run — PASS.** (same command as Step 2)

- [ ] **Step 5: Commit** — `feat(personas): editor base-copy coherence for out-of-band character attach`.

---

### Task 8: Screen wiring + fake

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_character_attach.py` (create)

**Interfaces:**
- Consumes: Task 1/2 service (`attach_to_character` / `detach_from_character` / `list_character_dictionaries` via `self._dictionary_scope_service()`), Task 5 `DictionaryPicker`, Task 6 panel + messages, Task 7 `sync_attached_dictionaries`. Existing: `self.state.selected_entity_id` (str), `self.state.selected_entity_kind == "character"`, `self._io_dialog_active`, `self.app_instance.chachanotes_db`, `cdl.list_chat_dictionaries(db, ...)`, `get_character_card_by_id`.
- Produces: `_refresh_character_dictionaries()`, `@on(CharacterDictionaryAttachRequested) _handle_character_dictionary_attach`, `_character_dictionary_attach_worker`, `_list_attachable_dictionaries(character_id)`, `@on(CharacterDictionaryDetachRequested) _handle_character_dictionary_detach`. The panel is mounted in `#personas-detail-stack` and refreshed from `_select_character`.

- [ ] **Step 1: Write the failing test.** Create `Tests/UI/test_personas_character_attach.py`. Mirror the harness of `Tests/UI/test_personas_dictionaries.py` (import its `PersonasTestApp`, `mock_app_instance`, `stub_characters`, and its `fake_dict_service`; read that file first). Extend the fake with an in-memory character store and the three character methods, then drive the flow:

```python
# In the fake dictionary scope service (mirror real shapes), add:
#   self.characters: dict[int, dict] = {}   # in __init__: {char_id: {"extensions": {"chat_dictionaries": [...]}, "version": N}}
#
#   async def list_character_dictionaries(self, character_id, mode="local"):
#       blocks = self.characters.get(int(character_id), {}).get("extensions", {}).get("chat_dictionaries", [])
#       return {"dictionaries": [{"name": b["name"], "entry_count": len(b.get("entries") or []),
#                                 "enabled": bool(b.get("enabled", True))} for b in blocks], "source": "local"}
#
#   async def attach_to_character(self, dictionary_id, character_id, mode="local"):
#       char = self.characters.setdefault(int(character_id), {"extensions": {"chat_dictionaries": []}, "version": 1})
#       blocks = char["extensions"].setdefault("chat_dictionaries", [])
#       name = self._names.get(int(dictionary_id), f"dict-{dictionary_id}")
#       if not any(b["name"] == name for b in blocks):
#           blocks.append({"name": name, "enabled": True, "entries": []})
#           char["version"] += 1
#       return {"dictionary_id": int(dictionary_id), "character_id": int(character_id),
#               "dictionary_name": name, "character_dictionaries": [b["name"] for b in blocks], "source": "local"}
#
#   async def detach_from_character(self, character_id, dictionary_name, mode="local"):
#       char = self.characters.get(int(character_id), {"extensions": {"chat_dictionaries": []}, "version": 1})
#       blocks = char["extensions"].get("chat_dictionaries", [])
#       char["extensions"]["chat_dictionaries"] = [b for b in blocks if b["name"] != dictionary_name]
#       char["version"] += 1
#       return {"character_id": int(character_id), "dictionary_name": dictionary_name,
#               "character_dictionaries": [b["name"] for b in char["extensions"]["chat_dictionaries"]], "source": "local"}

async def test_character_attach_via_picker_then_detach(mock_app_instance, stub_characters, fake_dict_service, monkeypatch):
    from textual.widgets import DataTable
    from tldw_chatbook.Widgets.Persona_Widgets.dictionary_picker import DictionaryPicker
    from tldw_chatbook.Widgets.Persona_Widgets.personas_character_dictionaries import (
        PersonasCharacterDictionariesWidget,
    )
    fake_dict_service._names = {1: "Slang"}  # dict id 1 → "Slang"

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_characters(pilot)          # helper: enter characters mode + select first char (id 1)
        # Auto-pick dictionary id 1 instead of showing the modal.
        async def _fake_push(screen_obj):
            return 1 if isinstance(screen_obj, DictionaryPicker) else None
        monkeypatch.setattr(screen.app, "push_screen_wait", _fake_push, raising=False)
        monkeypatch.setattr(screen, "_list_attachable_dictionaries",
                            lambda cid: [{"dictionary_id": 1, "name": "Slang"}])
        await pilot.click("#personas-char-dicts-add")
        await pilot.pause(); await pilot.app.workers.wait_for_complete(); await pilot.pause()
        assert any(b["name"] == "Slang" for b in fake_dict_service.characters[1]["extensions"]["chat_dictionaries"])
        table = screen.query_one("#personas-char-dicts-table", DataTable)
        assert table.row_count == 1
        table.move_cursor(row=0)
        await pilot.click("#personas-char-dicts-detach")
        await pilot.pause(); await pilot.app.workers.wait_for_complete(); await pilot.pause()
        assert fake_dict_service.characters[1]["extensions"]["chat_dictionaries"] == []
```

Write `_enter_characters(pilot)` mirroring `test_personas_dictionaries.py`'s `_enter_dictionaries` (set mode to "characters", select the first library row so `state.selected_entity_kind == "character"` and `selected_entity_id == "1"`). If the character panel lives behind a specific center view, ensure the test makes it visible before clicking (mirror how the dictionaries test reveals its detail).

- [ ] **Step 2: Run — expect FAIL** (`NoMatches: #personas-char-dicts-add` / handler missing).

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_character_attach.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 3a: Mount the panel.** In `personas_screen.py` `compose`, inside `#personas-detail-stack` (after `PersonasCharacterEditorWidget()`, ~line 452), add:

```python
                        yield PersonasCharacterDictionariesWidget()
```

Add the imports near the other `Persona_Widgets` imports:

```python
from ...Widgets.Persona_Widgets.personas_character_dictionaries import (
    PersonasCharacterDictionariesWidget,
    CharacterDictionaryAttachRequested,
    CharacterDictionaryDetachRequested,
)
from ...Widgets.Persona_Widgets.dictionary_picker import DictionaryPicker
```

- [ ] **Step 3b: Add the handlers.** Place next to the P1e `_refresh_dictionary_attachments` block (~line 1368):

```python
    async def _refresh_character_dictionaries(self) -> None:
        """Re-feed the character dictionaries panel (best-effort)."""
        entity_id = self.state.selected_entity_id
        service = self._dictionary_scope_service()
        if service is None or self.state.selected_entity_kind != "character" or not entity_id:
            return
        panel = self.query_one(PersonasCharacterDictionariesWidget)
        try:
            response = await service.list_character_dictionaries(int(entity_id), mode="local")
        except Exception:
            logger.opt(exception=True).warning(f"Could not list dictionaries for character {entity_id}.")
            panel.load_character_dictionaries([])
            return
        panel.load_character_dictionaries(list(response.get("dictionaries") or []))

    @on(CharacterDictionaryAttachRequested)
    async def _handle_character_dictionary_attach(self, message: CharacterDictionaryAttachRequested) -> None:
        message.stop()
        if self.state.selected_entity_kind != "character" or not self.state.selected_entity_id:
            return
        if self._io_dialog_active:
            return
        self._io_dialog_active = True
        self.run_worker(self._character_dictionary_attach_worker(), group="personas-io")

    async def _character_dictionary_attach_worker(self) -> None:
        try:
            entity_id = self.state.selected_entity_id
            service = self._dictionary_scope_service()
            if service is None or not entity_id:
                return
            char_id = int(entity_id)
            try:
                dicts = await asyncio.to_thread(self._list_attachable_dictionaries, char_id)
            except Exception:
                logger.opt(exception=True).warning("Could not load dictionaries for the attach picker.")
                return
            try:
                picked = await self.app.push_screen_wait(DictionaryPicker(dicts))
            except Exception:
                logger.opt(exception=True).warning("Could not show the dictionary picker.")
                return
            if not picked:
                return
            try:
                await service.attach_to_character(int(picked), char_id, mode="local")
            except ConflictError:
                self.notify("Attach failed: the character changed since it was loaded. Try again.", severity="warning")
                return
            except Exception:
                logger.opt(exception=True).warning(f"Could not attach dictionary to character {char_id}.")
                return
            await self._refresh_character_dictionaries()
            await self._sync_character_editor_dictionaries(char_id)
        finally:
            self._io_dialog_active = False

    def _list_attachable_dictionaries(self, character_id: int) -> list[dict]:
        """Local dictionaries NOT already attached to this character (sync DB read)."""
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            return []
        from ...Character_Chat import Chat_Dictionary_Lib as cdl
        attached = set()
        record = db.get_character_card_by_id(int(character_id))
        for block in cdl.load_character_dictionaries(record):
            attached.add(block.get("name"))
        rows = []
        for d in cdl.list_chat_dictionaries(db, limit=1000, include_disabled=True) or []:
            name = d.get("name") if isinstance(d, dict) else d[1]
            did = d.get("id") if isinstance(d, dict) else d[0]
            if name in attached:
                continue
            rows.append({"dictionary_id": int(did), "name": str(name)})
        return rows

    async def _sync_character_editor_dictionaries(self, character_id: int) -> None:
        """Keep the editor's base coherent after an out-of-band attach/detach."""
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            return
        try:
            record = await asyncio.to_thread(db.get_character_card_by_id, int(character_id))
        except Exception:
            return
        if not record:
            return
        ext = record.get("extensions") if isinstance(record.get("extensions"), dict) else {}
        try:
            editor = self.query_one(PersonasCharacterEditorWidget)
        except Exception:
            return
        if int(editor._character_data.get("id") or 0) == int(character_id):
            editor.sync_attached_dictionaries(ext.get("chat_dictionaries") or [], record.get("version"))

    @on(CharacterDictionaryDetachRequested)
    async def _handle_character_dictionary_detach(self, message: CharacterDictionaryDetachRequested) -> None:
        message.stop()
        entity_id = self.state.selected_entity_id
        service = self._dictionary_scope_service()
        if service is None or self.state.selected_entity_kind != "character" or not entity_id:
            return
        char_id = int(entity_id)
        try:
            await service.detach_from_character(char_id, str(message.dictionary_name), mode="local")
        except Exception:
            logger.opt(exception=True).warning(f"Could not detach dictionary from character {char_id}.")
            return
        await self._refresh_character_dictionaries()
        await self._sync_character_editor_dictionaries(char_id)
```

Verify the exact `cdl.list_chat_dictionaries` signature during implementation (mirror how `_dictionary_scope_service`/`list_dictionaries` already enumerates dictionaries — reuse that path if simpler, e.g. `await service.list_dictionaries(mode="local", include_inactive=True)` from within the worker, then filter). The row shape (`dict` vs tuple) is handled defensively above; simplify to the real shape once confirmed.

- [ ] **Step 3c: Refresh the panel on character select.** At the end of `_select_character` (~line 1090+), add:

```python
        await self._refresh_character_dictionaries()
```

- [ ] **Step 4: Run — PASS; then the whole personas UI suite for regressions.**

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_character_attach.py Tests/UI/test_personas_workbench.py Tests/UI/test_personas_dictionaries.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 5: Commit** — `feat(personas): wire character dictionary attach/detach + panel refresh + editor sync`.

---

### Task 9: Editor-coherence integration test, full gate + spec status

**Files:**
- Test: `Tests/UI/test_personas_character_attach.py` (append the coherence test)
- Modify: `Docs/superpowers/specs/2026-07-15-roleplay-p1f-character-attach-design.md` (status line)

- [ ] **Step 1: Add the editor-coherence integration test.** Append to `Tests/UI/test_personas_character_attach.py` — attach a dictionary, edit an unrelated editor field, Save, and assert BOTH the attachment and the edit persist (the co-owned-`extensions` clobber guard). Use the same `PersonasTestApp` harness; drive an attach (as in Task 8), then set an editor field value (e.g. `#personas-char-editor-personality`), click `#personas-char-editor-save`, and assert the fake character store shows both the embedded `chat_dictionaries` (Slang) and the new personality. If the save path in the test harness is heavy, assert at the editor level instead: after attach + `sync`, `editor.get_character_data()["extensions"]["chat_dictionaries"]` includes Slang AND the edited field value is present — proving the base patch didn't drop the form edit.

- [ ] **Step 1b: Add the AC6 export/import portability test.** Create `Tests/Character_Chat/test_character_dictionaries_portability.py` (auto-covered by the Task 9 gate, which runs `Tests/Character_Chat/`):

```python
"""P1f AC6: a character's embedded chat_dictionaries survive export → import."""

import json

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    export_character_card_to_json,
    import_character_card_from_json_string,
)


def test_embedded_chat_dictionaries_survive_export_import(tmp_path):
    db = CharactersRAGDB(tmp_path / "port.db", "test-client")
    service = LocalChatDictionaryService(db)
    dict_id = service.create_dictionary(
        {"name": "Slang", "entries": [{"pattern": "hi", "replacement": "hello"}]}
    )["id"]
    char_id = db.add_character_card({"name": "Noir"})
    service.attach_to_character(dict_id, char_id)

    exported = export_character_card_to_json(db, char_id, include_image=False)
    payload = exported if isinstance(exported, str) else json.dumps(exported)
    imported = import_character_card_from_json_string(payload)

    ext = (imported or {}).get("extensions") or {}
    names = [b.get("name") for b in (ext.get("chat_dictionaries") or [])]
    assert "Slang" in names, "embedded chat_dictionaries must ride inside extensions across export/import"
```

Run it; if the `export_character_card_to_json` return or the imported-card structure differs (e.g. extensions nested under a `data` key on the imported side), adjust the unwrapping to the real shape — the assertion (Slang survives) stays. This exercises the *existing* export/import machinery (no P1f code change to that path); it is a characterization guard that the portable-snapshot promise holds.

- [ ] **Step 2: Run the new test — PASS.**

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_character_attach.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 3: Full gate.**

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Character_Chat/ Tests/UI/test_dictionary_picker.py Tests/UI/test_personas_character_dictionaries.py \
  Tests/UI/test_personas_character_editor_sync.py Tests/UI/test_personas_character_attach.py \
  Tests/UI/test_personas_dictionaries.py Tests/UI/test_personas_workbench.py Tests/Chat/test_chat_functions.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

Expected: all pass (record exact counts in the report). Then the `import tldw_chatbook.app` smoke (the runtime touch is in the send path):

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('IMPORT OK')"
```

- [ ] **Step 4: Flip spec status** — in the spec, `**Status:** Design approved (brainstorm), pending spec review.` → `**Status:** Implemented (P1f).`

- [ ] **Step 5: Commit** — `docs(roleplay): mark P1f character-attach spec implemented`.

---

## Notes for the executor

- **Load-bearing tests** (do not let an implementer substitute a fake): Task 1 (real-DB embed + dedup + version bump), Task 4 (real-DB union: character-fires / conversation-wins / disabled-skip / malformed-skip / conversation-only-unchanged), Task 9 (editor-coherence). These are exactly where the `to_dict`/`from_dict` shape assumption and the co-owned-`extensions` clobber live — a fake alone would give a false green.
- **Widget stays I/O-free** (Tasks 5–6): no service/DB import in the panel or picker; the screen owns all I/O in `personas-io` workers.
- **No schema migration, no new table** — if you find yourself writing one, stop: `extensions` is already writable.
- The `_list_attachable_dictionaries` dictionary-enumeration path (Task 8) should reuse the screen's existing dictionary-listing seam if that is cleaner than `cdl.list_chat_dictionaries`; verify the real signature/return shape and simplify the defensive dict/tuple handling to the real one.
