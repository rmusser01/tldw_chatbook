# Roleplay P1e — Dictionary Conversation-Attach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attach/detach a dictionary to a conversation on the real runtime contract (the conversation's `active_dictionaries` metadata), surface the reverse used-by, and remove three dead attach paths.

**Architecture:** Make conversation `metadata` writable (DB), build the attach/detach/used-by seam on `LocalChatDictionaryService` + async scope wrappers, add an I/O-free Attachments tab + a dedicated conversation picker, wire the screen (which owns all service/DB calls in `personas-io` workers), and delete the dead code.

**Tech Stack:** Python ≥3.11, Textual (TabPane/DataTable/ModalScreen), pytest with a REAL `CharactersRAGDB` for the backend (the traps live below the fake).

**Spec:** `Docs/superpowers/specs/2026-07-14-roleplay-p1e-conversation-attach-design.md` (committed `64513288`) — its "Ground truths" section is binding; read once before Task 1.

## Global Constraints

- **The metadata-ignore trap:** `update_conversation` whitelists only `title`/`rating` today — a `metadata` field is silently dropped. Task 1 fixes this FIRST; attach/detach depend on it. Any attach test written against a fake alone is a false green — the backend seam is tested against a **real `CharactersRAGDB` seeded with real conversations**.
- **Typing seam:** dictionary ids are **ints** (`active_dictionaries` holds int dict ids; runtime does `load_chat_dictionary(db, dict_id)`); conversation ids are **strings** (never int-cast). Attach appends `int(dictionary_id)`.
- **Used-by is exact int membership**, never substring: `LIKE '%active_dictionaries%'` prefilter → JSON-parse → `int(dictionary_id) in parsed_list` (the id-1-matches-11 trap).
- **The existing `ConversationSelectionDialog` is a TTS dialog that int-casts ids — do NOT reuse it.** Build the dedicated picker (Task 4).
- **Widget I/O-free:** the Attachments tab and picker emit intent messages / return values; the SCREEN owns every service and DB call, in `personas-io` workers with the `_io_dialog_active` guard where a modal is involved.
- **Dead-code removal keeps the junction TABLE DDL** (dropping it is a migration — out of scope). Removal is preceded by a safety check that nothing live imports/dispatches the removed symbols.
- Scope: Personas-owned + Character_Chat dictionary files, plus the `update_conversation` whitelist + used-by read query (conversation-metadata touch — **AC-clause reported**) and the dead-code removal across `Chat_Events`/`conv_char_events`/`event_dispatcher`/CSS. No character-level attachment, no "What's in play" (P1f). Google docstrings on new public callables; widget CSS structure-only. Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (isolated HOME; from this worktree use the main checkout's absolute `.venv/bin/python`):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    .venv/bin/python -m pytest <target> \
    -q -p no:cacheprovider -o addopts="" --timeout=180 --timeout-method=thread
  ```
- UI tests that click inside the detail widget use `app.run_test(size=(200, 60))`.

---

### Task 1: Conversation `metadata` becomes writable

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` (`update_conversation`, the `fields_to_update_sql` block after the `rating` clause)
- Test: `Tests/DB/test_chachanotes_conversation_metadata.py` (create) — or append to an existing `Tests/DB/` conversation test if present; a new file is fine.

**Interfaces:**
- Produces: `update_conversation(conversation_id: str, {"metadata": <json str>}, expected_version)` persists the metadata blob, bumps version, updates last_modified/sync_log (inherited). Tasks 2 relies on this.

- [ ] **Step 1: Write the failing test.** Create `Tests/DB/test_chachanotes_conversation_metadata.py`:

```python
"""P1e: conversation metadata must be writable via update_conversation."""

import json

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
    yield database
    database.close_connection()


def _new_conversation(db) -> str:
    return db.add_conversation({"title": "Chat A"})


def test_update_conversation_persists_metadata_and_bumps_version(db):
    conv_id = _new_conversation(db)
    before = db.get_conversation_by_id(conv_id)
    blob = json.dumps({"active_dictionaries": [3, 7]})

    ok = db.update_conversation(conv_id, {"metadata": blob}, expected_version=before["version"])
    assert ok

    after = db.get_conversation_by_id(conv_id)
    assert json.loads(after["metadata"]) == {"active_dictionaries": [3, 7]}
    assert after["version"] == before["version"] + 1


def test_title_only_update_still_works(db):
    conv_id = _new_conversation(db)
    v = db.get_conversation_by_id(conv_id)["version"]
    db.update_conversation(conv_id, {"title": "Renamed"}, expected_version=v)
    assert db.get_conversation_by_id(conv_id)["title"] == "Renamed"
```

- [ ] **Step 2: Run — expect FAIL** (`test_update_conversation_persists_metadata...` — the metadata comes back None/unchanged, the whitelist ignored it).

- [ ] **Step 3: Implement.** In `update_conversation`, immediately AFTER the existing `rating` block:

```python
                if 'rating' in update_data:
                    fields_to_update_sql.append("rating = ?")
                    params_for_set_clause.append(update_data.get('rating'))
                if 'metadata' in update_data:                       # ADDED (P1e)
                    fields_to_update_sql.append("metadata = ?")     # ADDED
                    params_for_set_clause.append(update_data.get('metadata'))  # ADDED
```

Update the method's docstring line "Updatable fields from `update_data`: 'title', 'rating'." → "'title', 'rating', 'metadata'." (`metadata` expects a JSON string.)

- [ ] **Step 4: Run — both PASS.** Then run the DB conversation regression set: `Tests/DB/ -k conversation` (all pass — title/rating paths unchanged).

- [ ] **Step 5: Commit** — `feat(db): update_conversation can write the metadata field (P1e attach prerequisite)` + trailer.

---

### Task 2: Backend attach seam — service methods + scope wrappers

**Files:**
- Modify: `tldw_chatbook/Character_Chat/local_chat_dictionary_service.py` (new methods; the `include_usage` stub)
- Modify: `tldw_chatbook/Character_Chat/chat_dictionary_scope_service.py` (async pass-throughs)
- Test: `Tests/Character_Chat/test_local_chat_dictionary_service.py` (real-DB)

**Interfaces:**
- Consumes: Task 1's writable metadata; `self.db` (a `CharactersRAGDB` with `get_conversation_by_id`/`update_conversation`/`get_connection`).
- Produces (the UI + fake mirror these EXACTLY):
  - `LocalChatDictionaryService.attach_to_conversation(dictionary_id: int, conversation_id: str) -> {"dictionary_id": int, "conversation_id": str, "active_dictionaries": list[int], "source": "local"}` (idempotent/dedup; missing conv → `ValueError`; stale version → `ConflictError`).
  - `.detach_from_conversation(dictionary_id: int, conversation_id: str) -> <same shape>` (not-attached → no-op success).
  - `.list_dictionary_conversations(dictionary_id: int) -> {"conversations": [{"conversation_id": str, "title": str}], "source": "local"}` (exact int membership).
  - Scope service: `async attach_to_conversation(dictionary_id, conversation_id, mode="local")`, `async detach_from_conversation(...)`, `async list_dictionary_conversations(dictionary_id, mode="local")` — thin `_invoke` wrappers.

- [ ] **Step 1: Write the failing tests** (append to `test_local_chat_dictionary_service.py`; the file's `dictionary_db` fixture is a real `CharactersRAGDB`):

```python
import json as _json
from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError


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
    service = LocalChatDictionaryService(dictionary_db)
    d1 = service.create_dictionary({"name": "One"})
    d11 = service.create_dictionary({"name": "Eleven"})
    # Force the ids we need by attaching to distinct conversations.
    conv_with_1 = _seed_conversation(dictionary_db, "has 1")
    conv_with_11 = _seed_conversation(dictionary_db, "has 11")
    service.attach_to_conversation(d1["id"], conv_with_1)
    service.attach_to_conversation(d11["id"], conv_with_11)

    used_by_1 = service.list_dictionary_conversations(d1["id"])
    ids = {c["conversation_id"] for c in used_by_1["conversations"]}
    assert conv_with_1 in ids
    assert conv_with_11 not in ids     # would fail under a substring LIKE match
    titles = {c["title"] for c in used_by_1["conversations"]}
    assert "has 1" in titles


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
```

- [ ] **Step 2: Run — expect FAIL** (`AttributeError: attach_to_conversation`).

- [ ] **Step 3: Implement.** In `local_chat_dictionary_service.py` (import `json` if not already; it uses it elsewhere):

```python
    def _load_conversation_or_raise(self, conversation_id: str) -> dict:
        record = self._require_db().get_conversation_by_id(str(conversation_id))
        if record is None:
            raise ValueError(f"Conversation '{conversation_id}' was not found.")
        return record

    @staticmethod
    def _active_dictionaries(record: dict) -> list[int]:
        try:
            meta = json.loads(record.get("metadata") or "{}")
        except (TypeError, ValueError):
            meta = {}
        raw = meta.get("active_dictionaries") or []
        result: list[int] = []
        for value in raw:
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                continue
        return result

    def _write_active_dictionaries(self, record: dict, conversation_id: str, ids: list[int]) -> None:
        try:
            meta = json.loads(record.get("metadata") or "{}")
        except (TypeError, ValueError):
            meta = {}
        meta["active_dictionaries"] = ids
        self._require_db().update_conversation(
            str(conversation_id), {"metadata": json.dumps(meta)}, expected_version=record["version"]
        )

    def attach_to_conversation(self, dictionary_id: int, conversation_id: str) -> dict[str, Any]:
        """Attach a dictionary to a conversation's active_dictionaries (idempotent)."""
        record = self._load_conversation_or_raise(conversation_id)
        ids = self._active_dictionaries(record)
        did = int(dictionary_id)
        if did not in ids:
            ids.append(did)
            self._write_active_dictionaries(record, conversation_id, ids)
        return {"dictionary_id": did, "conversation_id": str(conversation_id),
                "active_dictionaries": ids, "source": "local"}

    def detach_from_conversation(self, dictionary_id: int, conversation_id: str) -> dict[str, Any]:
        """Detach a dictionary from a conversation (no-op when not attached)."""
        record = self._load_conversation_or_raise(conversation_id)
        ids = self._active_dictionaries(record)
        did = int(dictionary_id)
        if did in ids:
            ids = [i for i in ids if i != did]
            self._write_active_dictionaries(record, conversation_id, ids)
        return {"dictionary_id": did, "conversation_id": str(conversation_id),
                "active_dictionaries": ids, "source": "local"}

    def list_dictionary_conversations(self, dictionary_id: int) -> dict[str, Any]:
        """Reverse used-by: conversations whose active_dictionaries include this id.

        Args:
            dictionary_id: The dictionary to find attachments for.

        Returns:
            ``{"conversations": [{"conversation_id": str, "title": str}], "source": "local"}``.
        """
        did = int(dictionary_id)
        conn = self._require_db().get_connection()
        # LIKE prefilter shrinks the scan; exact int membership below avoids the
        # id-1-matches-11 substring trap. metadata is a column on conversations.
        rows = conn.execute(
            "SELECT id, title, metadata FROM conversations "
            "WHERE deleted = 0 AND metadata LIKE '%active_dictionaries%'"
        ).fetchall()
        conversations: list[dict[str, Any]] = []
        for row in rows:
            if did in self._active_dictionaries({"metadata": row["metadata"]}):
                conversations.append({"conversation_id": str(row["id"]), "title": str(row["title"] or "")})
        return {"conversations": conversations, "source": "local"}
```

(`row["metadata"]`/`row["title"]` assume the connection uses `sqlite3.Row` — the DB module does; if the rows are plain tuples in this codebase, index positionally `row[2]`/`row[1]`. The implementer confirms the row type and reports.)

Also update the `include_usage` stub (currently `{"conversation_count": None}`): make it report the real count via `len(self.list_dictionary_conversations(record_id)["conversations"])` when the dictionary's id is known at that point; if wiring that is awkward in the list loop, leave the stub but change `None` to `0` and note it in the report (the real count lives on the used-by call). Prefer the real count if clean.

Then the scope wrappers in `chat_dictionary_scope_service.py` (mirror `get_statistics`'s `_invoke` pattern; use a local-only action id — reuse the same `action_id` family the CRUD/statistics methods use for local; the implementer picks the closest existing constant and notes it):

```python
    async def attach_to_conversation(self, dictionary_id: int, conversation_id: str, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(normalized_mode, self._statistics_action(normalized_mode, "detail"),
                                  "attach_to_conversation", dictionary_id, conversation_id)

    async def detach_from_conversation(self, dictionary_id: int, conversation_id: str, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(normalized_mode, self._statistics_action(normalized_mode, "detail"),
                                  "detach_from_conversation", dictionary_id, conversation_id)

    async def list_dictionary_conversations(self, dictionary_id: int, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(normalized_mode, self._statistics_action(normalized_mode, "detail"),
                                  "list_dictionary_conversations", dictionary_id)
```

(If `_invoke` requires the local backend to actually have the method — it does — these route straight through. Confirm `_statistics_action` exists and returns a local action id; if the codebase gates on a policy action that would deny these, use whichever existing local action the CRUD methods use. Report the action id chosen.)

- [ ] **Step 4: Run — all PASS** (5 new + the existing service suite). The real-DB round-trip proves attach actually persists (Task 1 made it writable).

- [ ] **Step 5: Commit** — `feat(chat-dictionaries): attach/detach + reverse used-by over the active_dictionaries contract` + trailer.

---

### Task 3: Remove the dead attach code

**Files:**
- Delete: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events_dictionaries.py`
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py` (remove the `CHAT_DICTIONARY_BUTTON_HANDLERS` merge ~:4923 and the `refresh_active_dictionaries` calls ~:2309/2385/3204), `tldw_chatbook/Event_Handlers/Chat_Events/conv_char_events.py` (remove the 3 orphaned CCP handlers + their `CCP_BUTTON_HANDLERS` entries), `tldw_chatbook/Event_Handlers/event_dispatcher.py` (remove the dead `CHAT_DICTIONARIES_BUTTON_HANDLERS` line ~:72), `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py` (remove `associate_dictionary_with_conversation` + `get_conversation_dictionaries`), the dead CSS (`css/layout/_sidebars.tcss:404` region, `css/tldw_cli_modular.tcss:643` region — the `chat-dictionary-*` rules)
- Test: `Tests/Character_Chat/test_dead_attach_removed.py` (create — removal-safety)

**Interfaces:** none produced; this is deletion. The removal-safety test guards it.

- [ ] **Step 1: Write the failing test.** Create `Tests/Character_Chat/test_dead_attach_removed.py`:

```python
"""P1e: the dead dictionary-attach code paths are gone and nothing imports them."""

import importlib

import pytest


def test_chat_events_dictionaries_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tldw_chatbook.Event_Handlers.Chat_Events.chat_events_dictionaries")


def test_dead_junction_functions_removed():
    import tldw_chatbook.Character_Chat.Chat_Dictionary_Lib as cdl
    assert not hasattr(cdl, "associate_dictionary_with_conversation")
    assert not hasattr(cdl, "get_conversation_dictionaries")


def test_app_and_chat_events_still_import():
    # The wiring removal didn't break the modules that referenced the dead handlers.
    importlib.import_module("tldw_chatbook.Event_Handlers.Chat_Events.chat_events")
    importlib.import_module("tldw_chatbook.Event_Handlers.Chat_Events.conv_char_events")
    importlib.import_module("tldw_chatbook.Event_Handlers.event_dispatcher")
```

- [ ] **Step 2: Run — expect FAIL** (module still imports; junction fns still present).

- [ ] **Step 3: Implement — remove, safety-first.** Before deleting each symbol, grep for LIVE importers/dispatchers and confirm none remain (the audit found none; re-verify). Order:
  1. `grep -rn "chat_events_dictionaries\|CHAT_DICTIONARY_BUTTON_HANDLERS\|CHAT_DICTIONARIES_BUTTON_HANDLERS\|refresh_active_dictionaries\|handle_ccp_dict_apply_button_pressed\|handle_ccp_dict_remove_from_conv_button_pressed\|populate_active_dictionaries_list\|associate_dictionary_with_conversation\|get_conversation_dictionaries" tldw_chatbook/` — every hit must be either (a) the definition/wiring you're removing, or (b) already-dead. Report the grep result in the report; if a LIVE non-dead caller appears, STOP and report BLOCKED.
  2. Delete `chat_events_dictionaries.py`; remove its import + the `CHAT_DICTIONARY_BUTTON_HANDLERS` merge in `chat_events.py`; remove the 3 `refresh_active_dictionaries(...)` call sites in `chat_events.py`.
  3. Remove the 3 CCP handler defs + their `CCP_BUTTON_HANDLERS` map entries in `conv_char_events.py`.
  4. Remove the `event_dispatcher.py` dead line.
  5. Remove the two junction functions from `Chat_Dictionary_Lib.py`.
  6. Remove the `chat-dictionary-*` CSS rule blocks in the two CSS files.

- [ ] **Step 4: Run** the removal-safety test (PASS) + import smoke on `app` (`.venv/bin/python -c "import tldw_chatbook.app; print('ok')"`) + the chat-events regression (`Tests/ -k "chat_events or conv_char" -q` — no import errors, existing behavior intact).

- [ ] **Step 5: Commit** — `refactor(chat-dictionaries): remove three dead dictionary-attach paths (uncomposed/non-functional)` + trailer.

---

### Task 4: Dedicated conversation attach picker

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/dictionary_attach_picker.py`
- Test: `Tests/UI/test_dictionary_attach_picker.py` (create — bare-App modal harness)

**Interfaces:**
- Produces: `DictionaryAttachPicker(ModalScreen[str | None])` — `__init__(self, conversations: list[dict], **kwargs)` where each `conversation` is `{"conversation_id": str, "title": str}`; a search `Input` filters the list; single-select; **Attach** button `dismiss(selected_conversation_id: str)`, **Cancel** `dismiss(None)`. Ids stay strings — NO int cast. DOM ids `#dict-attach-search`, `#dict-attach-list` (ListView), `#dict-attach-confirm`, `#dict-attach-cancel`.

- [ ] **Step 1: Write the failing tests.** Create `Tests/UI/test_dictionary_attach_picker.py`:

```python
"""P1e: the dedicated (non-TTS) conversation attach picker returns string ids."""

import pytest
from textual.app import App
from textual.widgets import Input, ListView

from tldw_chatbook.Widgets.Persona_Widgets.dictionary_attach_picker import DictionaryAttachPicker

pytestmark = pytest.mark.asyncio

CONVS = [
    {"conversation_id": "conv-uuid-a", "title": "Noir case"},
    {"conversation_id": "conv-uuid-b", "title": "Lab notes"},
]


class _Host(App):
    def __init__(self):
        super().__init__()
        self.result = "unset"

    async def on_mount(self):
        self.result = await self.push_screen_wait(DictionaryAttachPicker(list(CONVS)))


async def test_picker_returns_selected_string_id():
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        picker = app.screen
        picker.query_one("#dict-attach-list", ListView).index = 1  # Lab notes
        await pilot.pause()
        await pilot.click("#dict-attach-confirm")
        await pilot.pause()
    assert app.result == "conv-uuid-b"          # string id, not int


async def test_picker_search_filters():
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        picker = app.screen
        picker.query_one("#dict-attach-search", Input).value = "noir"
        await pilot.pause()
        rows = picker.query_one("#dict-attach-list", ListView).children
        assert len(rows) == 1
        # select the only match, confirm
        picker.query_one("#dict-attach-list", ListView).index = 0
        await pilot.pause()
        await pilot.click("#dict-attach-confirm")
        await pilot.pause()
    assert app.result == "conv-uuid-a"


async def test_picker_cancel_returns_none():
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#dict-attach-cancel")
        await pilot.pause()
    assert app.result is None
```

- [ ] **Step 2: Run — expect FAIL** (module missing).

- [ ] **Step 3: Implement.** Create `dictionary_attach_picker.py`:

```python
"""A small modal for picking a conversation to attach a dictionary to (Roleplay P1e).

Distinct from ``ConversationSelectionDialog`` (a TTS dialog that int-casts ids);
this one keeps conversation ids as strings and returns the picked id.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static


class DictionaryAttachPicker(ModalScreen[str | None]):
    """Pick one conversation (by string id) to attach the current dictionary to.

    Args:
        conversations: ``{"conversation_id": str, "title": str}`` rows to choose from.
    """

    DEFAULT_CSS = """
    DictionaryAttachPicker { align: center middle; }
    DictionaryAttachPicker > Vertical {
        width: 60%; max-width: 80; height: auto; max-height: 80%;
        padding: 1 2; border: round $panel;
    }
    DictionaryAttachPicker #dict-attach-list { height: auto; max-height: 16; }
    """

    def __init__(self, conversations: list[dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._conversations = list(conversations)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Attach to conversation", markup=False)
            yield Input(placeholder="Search conversations…", id="dict-attach-search")
            yield ListView(id="dict-attach-list")
            with Vertical(id="dict-attach-actions"):
                yield Button("Attach", id="dict-attach-confirm", classes="console-action-secondary")
                yield Button("Cancel", id="dict-attach-cancel", classes="console-action-secondary")

    def on_mount(self) -> None:
        self._render(self._conversations)

    def _render(self, rows: list[dict[str, Any]]) -> None:
        listing = self.query_one("#dict-attach-list", ListView)
        listing.clear()
        for row in rows:
            item = ListItem(Static(str(row.get("title") or "(untitled)"), markup=False))
            item.dict_conversation_id = str(row.get("conversation_id"))  # stash the string id
            listing.append(item)

    @on(Input.Changed, "#dict-attach-search")
    def _filter(self, event: Input.Changed) -> None:
        event.stop()
        needle = event.value.strip().lower()
        rows = [c for c in self._conversations if needle in str(c.get("title") or "").lower()] if needle else self._conversations
        self._render(rows)

    def _selected_id(self) -> str | None:
        listing = self.query_one("#dict-attach-list", ListView)
        index = listing.index
        if index is None or not 0 <= index < len(listing.children):
            return None
        return getattr(listing.children[index], "dict_conversation_id", None)

    @on(Button.Pressed, "#dict-attach-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(self._selected_id())

    @on(Button.Pressed, "#dict-attach-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)


__all__ = ["DictionaryAttachPicker"]
```

- [ ] **Step 4: Run — all 3 PASS.** (If `ListItem` won't accept an arbitrary attribute in the installed Textual, stash ids in a parallel `self._row_ids` list keyed by index instead — keep the tests green.)

- [ ] **Step 5: Commit** — `feat(personas): dedicated conversation attach picker (string ids, non-TTS)` + trailer.

---

### Task 5: Attachments tab (I/O-free widget surface)

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Produces: TabPane `#personas-dict-tab-attachments` after Versions; DataTable `#personas-dict-attachments-table` (columns `conversation · id`), a **Detach** button `#personas-dict-attach-detach`, an **Attach to conversation…** button `#personas-dict-attach-add`, empty-state Static `#personas-dict-attachments-empty`; messages `DictionaryAttachRequested`, `DictionaryDetachRequested(conversation_id: str)`; API `load_attachments(rows: list[dict])` where each row is `{"conversation_id": str, "title": str}`.

- [ ] **Step 1: Failing tests** (append; `size=(200,60)`):

```python
class TestDictionaryAttachmentsTab:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_empty_state_when_unattached(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            empty = screen.query_one("#personas-dict-attachments-empty", Static)
            assert "Not attached" in str(empty.renderable)

    async def test_load_attachments_renders_rows(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import DataTable
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            detail = screen.query_one("#personas-dictionary-detail")
            detail.load_attachments([{"conversation_id": "c1", "title": "Noir case"}])
            await pilot.pause()
            table = screen.query_one("#personas-dict-attachments-table", DataTable)
            assert table.row_count == 1
            assert "Noir case" in str(table.get_cell_at((0, 0)))
```

- [ ] **Step 2: Run — FAIL** (`NoMatches: #personas-dict-attachments-empty`).

- [ ] **Step 3: Implement.** Messages (module level):

```python
class DictionaryAttachRequested(Message):
    """Request the attach-to-conversation picker for the current dictionary."""


class DictionaryDetachRequested(Message):
    """Detach the current dictionary from one conversation.

    Args:
        conversation_id: The conversation to detach from (string id).
    """

    def __init__(self, conversation_id: str) -> None:
        super().__init__()
        self.conversation_id = conversation_id
```

Compose (after the Versions TabPane), with `from rich.text import Text` already imported:

```python
            with TabPane("Attachments", id="personas-dict-tab-attachments"):
                yield Static("Not attached to any conversation yet.", id="personas-dict-attachments-empty", markup=False)
                yield DataTable(id="personas-dict-attachments-table", cursor_type="row")
                with Horizontal(classes="personas-dict-form-row"):
                    yield Button("Attach to conversation…", id="personas-dict-attach-add", classes="console-action-secondary")
                    yield Button("Detach", id="personas-dict-attach-detach", classes="console-action-secondary")
```

`on_mount` adds columns: `self.query_one("#personas-dict-attachments-table", DataTable).add_columns("conversation", "id")`. CSS: table `height: auto; max-height: 8;`.

```python
    def load_attachments(self, rows: list[dict]) -> None:
        """Render the conversations this dictionary is attached to.

        Args:
            rows: ``{"conversation_id": str, "title": str}`` entries.
        """
        self._attachment_rows = list(rows)
        table = self.query_one("#personas-dict-attachments-table", DataTable)
        table.clear()
        for row in self._attachment_rows:
            table.add_row(
                Text(str(row.get("title") or "(untitled)")),
                Text(str(row.get("conversation_id") or "")),
                key=str(row.get("conversation_id")),
            )
        empty = self.query_one("#personas-dict-attachments-empty", Static)
        empty.display = not self._attachment_rows
        table.display = bool(self._attachment_rows)

    def _selected_attachment_id(self) -> str | None:
        table = self.query_one("#personas-dict-attachments-table", DataTable)
        if table.row_count == 0 or table.cursor_row is None or table.cursor_row < 0:
            return None
        try:
            return str(table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value)
        except Exception:
            return None

    @on(Button.Pressed, "#personas-dict-attach-add")
    def _attach_add(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(DictionaryAttachRequested())

    @on(Button.Pressed, "#personas-dict-attach-detach")
    def _attach_detach(self, event: Button.Pressed) -> None:
        event.stop()
        conversation_id = self._selected_attachment_id()
        if conversation_id is not None:
            self.post_message(DictionaryDetachRequested(conversation_id))
```

Init `self._attachment_rows: list[dict] = []` in `__init__`; `clear()` also empties the attachments table + `_attachment_rows` (symmetry with the other tabs).

- [ ] **Step 4: Run the UI file — PASS.**

- [ ] **Step 5: Commit** — `feat(personas): dictionary Attachments tab (used-by list + attach/detach intents)` + trailer.

---

### Task 6: Screen wiring — attach/detach + refresh + fake

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: Task 2's scope methods, Task 4's picker, Task 5's messages/`load_attachments`.
- Produces: `_refresh_dictionary_attachments()` (fetch `list_dictionary_conversations` → `load_attachments`), `@on(DictionaryAttachRequested)` (opens the picker in a `personas-io` worker → attach → refresh), `@on(DictionaryDetachRequested)` (detach → refresh); attachments refreshed in `_select_dictionary`. Fake gains `attach_to_conversation`/`detach_from_conversation`/`list_dictionary_conversations` mirroring the real shapes, backed by an in-memory `active_dictionaries` per fake conversation.

- [ ] **Step 1: Failing tests** (append):

```python
class TestDictionaryAttachFlow:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_attach_via_picker_then_detach(self, mock_app_instance, stub_characters, fake_dict_service, monkeypatch):
        from textual.widgets import DataTable
        from tldw_chatbook.Widgets.Persona_Widgets.dictionary_attach_picker import DictionaryAttachPicker

        # Seed a conversation the picker can offer + the attach can target.
        fake_dict_service.conversations = {"c1": {"id": "c1", "title": "Noir case", "active_dictionaries": []}}

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)

            # Auto-pick "c1" instead of showing the modal (the picker itself is
            # covered by Task 4's dedicated test).
            async def _fake_push(screen_obj):
                return "c1" if isinstance(screen_obj, DictionaryAttachPicker) else None
            monkeypatch.setattr(screen.app, "push_screen_wait", _fake_push, raising=False)
            # The attach worker also does a sync DB read for the conversation list;
            # stub it to the fake's seeded conversation.
            monkeypatch.setattr(
                screen, "_list_attachable_conversations",
                lambda: [{"conversation_id": "c1", "title": "Noir case"}],
            )
            await pilot.click("#personas-dict-attach-add")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert 1 in fake_dict_service.conversations["c1"]["active_dictionaries"]  # attached (dict id 1)
            table = screen.query_one("#personas-dict-attachments-table", DataTable)
            assert table.row_count == 1
            # detach
            table.move_cursor(row=0)
            await pilot.click("#personas-dict-attach-detach")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.conversations["c1"]["active_dictionaries"] == []
```

Fake additions (mirroring real shapes; `self.conversations: dict[str, dict] = {}` in `__init__`):

```python
    async def list_dictionary_conversations(self, dictionary_id: int, mode: str = "local") -> dict:
        did = int(dictionary_id)
        rows = [{"conversation_id": cid, "title": c.get("title") or ""}
                for cid, c in self.conversations.items()
                if did in (c.get("active_dictionaries") or [])]
        return {"conversations": rows, "source": "local"}

    async def attach_to_conversation(self, dictionary_id: int, conversation_id: str, mode: str = "local") -> dict:
        conv = self.conversations[str(conversation_id)]
        did = int(dictionary_id)
        ids = conv.setdefault("active_dictionaries", [])
        if did not in ids:
            ids.append(did)
        return {"dictionary_id": did, "conversation_id": str(conversation_id),
                "active_dictionaries": list(ids), "source": "local"}

    async def detach_from_conversation(self, dictionary_id: int, conversation_id: str, mode: str = "local") -> dict:
        conv = self.conversations[str(conversation_id)]
        did = int(dictionary_id)
        conv["active_dictionaries"] = [i for i in conv.get("active_dictionaries") or [] if i != did]
        return {"dictionary_id": did, "conversation_id": str(conversation_id),
                "active_dictionaries": list(conv["active_dictionaries"]), "source": "local"}
```

- [ ] **Step 2: Run — FAIL** (`NoMatches`/`AttributeError` on the handlers).

- [ ] **Step 3: Implement** (screen). Import the picker + messages. Handlers:

```python
    async def _refresh_dictionary_attachments(self) -> None:
        """Re-feed the Attachments tab for the selected dictionary (best-effort)."""
        entity_id = self.state.selected_entity_id
        service = self._dictionary_scope_service()
        if service is None or self.state.selected_entity_kind != "dictionary" or not entity_id:
            return
        detail = self.query_one(PersonasDictionaryDetailWidget)
        try:
            response = await service.list_dictionary_conversations(int(entity_id), mode="local")
        except Exception:
            logger.opt(exception=True).warning(f"Could not list conversations for dictionary {entity_id}.")
            detail.load_attachments([])
            return
        detail.load_attachments(list(response.get("conversations") or []))

    @on(DictionaryAttachRequested)
    async def _handle_dictionary_attach(self, message: DictionaryAttachRequested) -> None:
        message.stop()
        if self.state.selected_entity_kind != "dictionary" or not self.state.selected_entity_id:
            return
        if self._io_dialog_active:
            return
        self._io_dialog_active = True
        self.run_worker(self._dictionary_attach_worker(), group="personas-io")

    async def _dictionary_attach_worker(self) -> None:
        try:
            entity_id = self.state.selected_entity_id
            service = self._dictionary_scope_service()
            if service is None or not entity_id:
                return
            detail = self.query_one(PersonasDictionaryDetailWidget)
            try:
                convs = await asyncio.to_thread(self._list_attachable_conversations)
            except Exception as exc:
                logger.opt(exception=True).warning("Could not load conversations for the attach picker.")
                detail.set_status(f"Attach failed: {exc}")
                return
            from ...Widgets.Persona_Widgets.dictionary_attach_picker import DictionaryAttachPicker
            try:
                picked = await self.app.push_screen_wait(DictionaryAttachPicker(convs))
            except Exception:
                logger.opt(exception=True).warning("Could not show the attach picker.")
                return
            if not picked:
                return
            try:
                await service.attach_to_conversation(int(entity_id), str(picked), mode="local")
            except ConflictError:
                detail.set_status("Attach failed: the conversation changed since it was loaded. Try again.")
                return
            except Exception as exc:
                logger.opt(exception=True).warning(f"Could not attach dictionary {entity_id}.")
                detail.set_status(f"Attach failed: {exc}")
                return
            await self._refresh_dictionary_attachments()
        finally:
            self._io_dialog_active = False

    def _list_attachable_conversations(self) -> list[dict]:
        """Conversations offered by the attach picker (title + string id). Sync DB read."""
        db = getattr(self.app_instance, "chachanotes_db", None) or getattr(self.app_instance, "db", None)
        if db is None:
            return []
        page = db.search_conversations_page(search_term="", page=1, results_per_page=200) if hasattr(db, "search_conversations_page") else None
        results = (page[0] if isinstance(page, tuple) else page) or []
        rows = []
        for conv in results:
            rows.append({"conversation_id": str(conv.get("id")), "title": str(conv.get("title") or "(untitled)")})
        return rows

    @on(DictionaryDetachRequested)
    async def _handle_dictionary_detach(self, message: DictionaryDetachRequested) -> None:
        message.stop()
        entity_id = self.state.selected_entity_id
        service = self._dictionary_scope_service()
        if service is None or not entity_id:
            return
        detail = self.query_one(PersonasDictionaryDetailWidget)
        try:
            await service.detach_from_conversation(int(entity_id), str(message.conversation_id), mode="local")
        except Exception as exc:
            logger.opt(exception=True).warning(f"Could not detach dictionary {entity_id}.")
            detail.set_status(f"Detach failed: {exc}")
            return
        await self._refresh_dictionary_attachments()
```

Call `await self._refresh_dictionary_attachments()` at the end of `_select_dictionary`.

**Implementer notes:** `_list_attachable_conversations` reaches the app's ChaChaNotes DB — verify the attribute name on the real app (`chachanotes_db` per app.py) and the exact `search_conversations_page` signature/return (it may return `(rows, total)`); adapt and report. In the test, `mock_app_instance` won't have a real DB, so the test monkeypatches `push_screen_wait` to return `"c1"` directly (the picker/conversation-list path isn't exercised there — a separate picker test covers that in Task 4). Keep `asyncio` imported (it is).

- [ ] **Step 4: Run the whole UI file — PASS** (attach/detach flow + all prior).

- [ ] **Step 5: Commit** — `feat(personas): wire dictionary attach/detach + attachments refresh` + trailer.

---

### Task 7: Full gate + spec status

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-14-roleplay-p1e-conversation-attach-design.md` (status line)

- [ ] **Step 1: Full gate**

```
HOME=... .venv/bin/python -m pytest \
  Tests/UI/test_personas_dictionaries.py Tests/UI/test_dictionary_attach_picker.py \
  Tests/UI/test_personas_dictionary_validation.py Tests/UI/test_personas_workbench.py \
  Tests/Character_Chat/ Tests/DB/test_chachanotes_conversation_metadata.py \
  Tests/Chat/test_chat_functions.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

Expected: all pass (exact counts in the report). Then `import tldw_chatbook.app` smoke (the dead-code removal touched app wiring).

- [ ] **Step 2: Spec status** — `**Status:** Design approved (brainstorm), pending spec review.` → `**Status:** Implemented (P1e).`

- [ ] **Step 3: Commit** — `docs(roleplay): mark P1e conversation-attach spec implemented` + trailer.
