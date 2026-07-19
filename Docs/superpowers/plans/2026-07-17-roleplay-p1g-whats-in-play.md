# Roleplay P1g — Console "What's in play" Dictionaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the conversation+character dictionary union for the current chat in the Console inspector rail, and let the user attach/detach conversation-level dictionaries from there.

**Architecture:** One shared union core in `Chat_Dictionary_Lib` feeds both the send path (`collect_active_chatdict_entries`, behavior-unchanged) and a new read-model (`summarize_active_dictionaries`), so "shown" can't drift from "applied". A new "Chat Dictionaries" block in the Console inspector renders from a screen-cached summary (recomputed only on conversation/character change — zero DB on recompose) and exposes group-level attach/detach action buttons that open a modal dictionary picker and write over the P1e conversation-attach seam in a Console worker.

**Tech Stack:** Python 3.11+, Textual, SQLite (CharactersRAGDB / schema v20), pytest.

## Global Constraints

- **Shared union core is the single source of truth.** `_resolve_active_dictionaries(db, conversation_id, char_data) -> list[dict]` (each `{name, source, enabled, entries, shadowed}`, **conversation-then-character order**) applies dedup-by-name, conversation-wins, enabled-only, never-raises. `collect_active_chatdict_entries` is reimplemented on it and stays **byte-unchanged** (regression-pinned, including the disabled-conversation-dict edge). `summarize_active_dictionaries` projects `{name, source, enabled, entry_count, shadowed}`.
- **`shadowed` = collision with an ENABLED conversation dict.** A character dict colliding only with a *disabled* conversation dict is NOT shadowed (it applies) — matching `collect`'s exact name-claiming (`collect` adds a conversation dict's name to the dedup set only when `dict_data.get('enabled', True)`).
- **The inspector renders from cached state with ZERO DB I/O.** `ConsoleRunInspector` is `refresh(recompose=True)`; `ChatScreen._build_console_inspector_state` builds a **fresh** `ConsoleInspectorState` each pass (3 call sites). The summary is cached on a persistent **screen attribute** and merely read into each build. It is recomputed only on change, via the reactives `current_chat_conversation_id` / `current_chat_active_character_data` (`app.py:2296/2299`).
- **Inspector actions are group-level, not per-row.** Attach and detach are two `ConsoleInspectorAction` buttons; each opens the reused P1f `DictionaryPicker(ModalScreen[int | None])` (int ids). Attach filters to dicts NOT already attached to the current conversation; detach filters to the conversation's *attached* dicts (conversation-source only). **Character dicts are read-only** (never offered for detach).
- **Writes run in a Console worker with `exit_on_error` discipline** (no uncaught worker exceptions), over P1e's `attach_to_conversation`/`detach_from_conversation`; `ConflictError` → notify + refresh; no active conversation → attach disabled + notify.
- **No dead-code cleanup** (backlog). No character-attach from the Console. No lore (P2).
- **Test env (venv-only):** prefix every pytest run with `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share` and use `-p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`. Python: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`. **Stage only your task's files — never `git add -A` or stage `.superpowers/`.**

---

### Task 1: Shared union core + `summarize_active_dictionaries` (+ `collect` byte-unchanged)

**Files:**
- Modify: `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py` (the `collect_active_chatdict_entries` region, ~1146-1205)
- Test: `Tests/Character_Chat/test_resolve_active_dictionaries.py` (create); `Tests/Character_Chat/test_collect_active_chatdict_entries.py` (extend)

**Interfaces:**
- Consumes: existing `load_chat_dictionary(db, dict_id) -> {"name","enabled","entries":[ChatDictionary]}`, `load_character_dictionaries(char_data) -> [{"name","enabled","entries":[ChatDictionary]}]`, `CharactersRAGDB.get_conversation_by_id(str)`.
- Produces:
  - `_resolve_active_dictionaries(db, conversation_id: Optional[str], char_data: Optional[dict]) -> list[dict]` — each `{"name": str, "source": "conversation"|"character", "enabled": bool, "entries": list[ChatDictionary], "shadowed": bool}`, conversation rows first then character rows, never raises.
  - `collect_active_chatdict_entries(db, conversation_id, char_data) -> list[ChatDictionary]` (reimplemented on the core; behavior byte-unchanged).
  - `summarize_active_dictionaries(db, conversation_id, char_data) -> {"dictionaries": [{"name","source","enabled","entry_count","shadowed"}], "source": "local"}`.

- [ ] **Step 1: Write the failing tests.** Create `Tests/Character_Chat/test_resolve_active_dictionaries.py`:

```python
"""P1g: the shared union core + summary projection."""

import json
import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import (
    _resolve_active_dictionaries,
    summarize_active_dictionaries,
    collect_active_chatdict_entries,
)


@pytest.fixture
def db(tmp_path):
    return CharactersRAGDB(tmp_path / "resolve.db", "test-client")


def _attach_conv_dict(db, service, conv_id, name, enabled=True):
    dict_id = service.create_dictionary({"name": name, "entries": [{"pattern": name, "replacement": name.lower()}]})["id"]
    if not enabled:
        rec = service.get_dictionary(dict_id)
        service.update_dictionary(dict_id, {"is_active": False}, expected_version=rec["version"])
    conv = db.get_conversation_by_id(conv_id)
    meta = json.loads(conv.get("metadata") or "{}")
    meta["active_dictionaries"] = meta.get("active_dictionaries", []) + [dict_id]
    db.update_conversation(conv_id, {"metadata": json.dumps(meta)}, expected_version=conv["version"])
    return dict_id


def _embed_char_dict(db, char_id, name, enabled=True, entries=None):
    rec = db.get_character_card_by_id(char_id)
    ext = rec["extensions"] if isinstance(rec["extensions"], dict) else {}
    ext.setdefault("chat_dictionaries", []).append(
        {"name": name, "enabled": enabled, "entries": entries or [{"key": name, "content": name.lower()}]}
    )
    db.update_character_card(char_id, {"extensions": ext}, expected_version=rec["version"])


def test_summary_applied_set_equals_collect(db):
    service = LocalChatDictionaryService(db)
    conv_id = db.add_conversation({"title": "chat"})
    _attach_conv_dict(db, service, conv_id, "ConvDict")
    char_id = db.add_character_card({"name": "Noir"})
    _embed_char_dict(db, char_id, "CharDict")
    char_data = db.get_character_card_by_id(char_id)

    summary = summarize_active_dictionaries(db, conv_id, char_data)["dictionaries"]
    applied_names_summary = {d["name"] for d in summary if d["enabled"] and not d["shadowed"]}
    # The applied set from the summary must equal the dicts collect actually loads.
    collect_names = {"ConvDict", "CharDict"}
    assert applied_names_summary == collect_names
    assert len(collect_active_chatdict_entries(db, conv_id, char_data)) == 2  # one entry each


def test_shadowed_only_by_enabled_conversation_dict(db):
    service = LocalChatDictionaryService(db)
    conv_id = db.add_conversation({"title": "chat"})
    _attach_conv_dict(db, service, conv_id, "Shared", enabled=True)
    char_id = db.add_character_card({"name": "Noir"})
    _embed_char_dict(db, char_id, "Shared")  # same name as an ENABLED conversation dict
    char_data = db.get_character_card_by_id(char_id)

    rows = _resolve_active_dictionaries(db, conv_id, char_data)
    char_shared = [r for r in rows if r["source"] == "character" and r["name"] == "Shared"][0]
    assert char_shared["shadowed"] is True
    # Only the conversation "Shared" applies.
    assert collect_active_chatdict_entries(db, conv_id, char_data)[0].content == "shared"


def test_not_shadowed_by_disabled_conversation_dict(db):
    service = LocalChatDictionaryService(db)
    conv_id = db.add_conversation({"title": "chat"})
    _attach_conv_dict(db, service, conv_id, "Shared", enabled=False)  # DISABLED
    char_id = db.add_character_card({"name": "Noir"})
    _embed_char_dict(db, char_id, "Shared", entries=[{"key": "s", "content": "CHAR"}])
    char_data = db.get_character_card_by_id(char_id)

    rows = _resolve_active_dictionaries(db, conv_id, char_data)
    char_shared = [r for r in rows if r["source"] == "character" and r["name"] == "Shared"][0]
    assert char_shared["shadowed"] is False  # disabled conv dict does NOT shadow
    # The character "Shared" applies (the disabled conversation one does not).
    assert collect_active_chatdict_entries(db, conv_id, char_data)[0].content == "CHAR"


def test_summary_entry_count_and_source_and_never_raises(db):
    char_id = db.add_character_card({"name": "Noir"})
    rec = db.get_character_card_by_id(char_id)
    ext = rec["extensions"] if isinstance(rec["extensions"], dict) else {}
    ext["chat_dictionaries"] = "not-a-list"  # hostile
    db.update_character_card(char_id, {"extensions": ext}, expected_version=rec["version"])
    char_data = db.get_character_card_by_id(char_id)
    # never raises on hostile embedded content
    assert summarize_active_dictionaries(db, None, char_data) == {"dictionaries": [], "source": "local"}
```

Also append a byte-unchanged regression to `Tests/Character_Chat/test_collect_active_chatdict_entries.py` (the file already exercises collect; add this):

```python
def test_collect_disabled_conversation_dict_does_not_shadow_character(db):
    from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
    import json as _json
    service = LocalChatDictionaryService(db)
    conv_id = db.add_conversation({"title": "chat"})
    did = service.create_dictionary({"name": "Dup", "entries": [{"pattern": "a", "replacement": "conv"}]})["id"]
    rec = service.get_dictionary(did)
    service.update_dictionary(did, {"is_active": False}, expected_version=rec["version"])  # disable it
    conv = db.get_conversation_by_id(conv_id)
    meta = _json.loads(conv.get("metadata") or "{}"); meta["active_dictionaries"] = [did]
    db.update_conversation(conv_id, {"metadata": _json.dumps(meta)}, expected_version=conv["version"])
    char_id = db.add_character_card({"name": "C"})
    r = db.get_character_card_by_id(char_id); ext = r["extensions"] if isinstance(r["extensions"], dict) else {}
    ext["chat_dictionaries"] = [{"name": "Dup", "enabled": True, "entries": [{"key": "a", "content": "char"}]}]
    db.update_character_card(char_id, {"extensions": ext}, expected_version=r["version"])
    entries = collect_active_chatdict_entries(db, conv_id, db.get_character_card_by_id(char_id))
    assert [e.content for e in entries] == ["char"]  # character applies; disabled conv dict didn't shadow
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: cannot import name '_resolve_active_dictionaries'`).

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_resolve_active_dictionaries.py Tests/Character_Chat/test_collect_active_chatdict_entries.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 3: Implement.** In `Chat_Dictionary_Lib.py`, REPLACE the body of `collect_active_chatdict_entries` (keep its signature + docstring) and add the two new functions. The current collect body (conversation loop that adds names to `conversation_dict_names` only when enabled, then the character loop that skips disabled + name-collisions) is preserved *exactly* by the core:

```python
def _resolve_active_dictionaries(
    db: "CharactersRAGDB",
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve the dict-level union for the current chat (never raises).

    Returns one row per dictionary in conversation-then-character order:
    ``{"name", "source": "conversation"|"character", "enabled", "entries":
    [ChatDictionary], "shadowed"}``. ``shadowed`` marks a character dictionary
    whose name collides with an ENABLED conversation dictionary (present but not
    applied); a disabled conversation dict does not claim its name, so a
    same-named character dict is not shadowed. This is the single source of truth
    for both :func:`collect_active_chatdict_entries` (send path) and
    :func:`summarize_active_dictionaries` (read model).
    """
    rows: List[Dict[str, Any]] = []
    enabled_conversation_names: set = set()
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
            active = metadata.get('active_dictionaries')
            if not isinstance(active, list):
                active = []
            for dict_id in active:
                try:
                    dict_data = load_chat_dictionary(db, dict_id)
                except Exception:
                    continue
                if not dict_data:
                    continue
                enabled = bool(dict_data.get('enabled', True))
                name = dict_data.get('name')
                if enabled and name is not None:
                    enabled_conversation_names.add(name)
                rows.append({
                    "name": name,
                    "source": "conversation",
                    "enabled": enabled,
                    "entries": dict_data.get('entries') or [],
                    "shadowed": False,  # a conversation dict is never shadowed
                })
    for block in load_character_dictionaries(char_data):
        rows.append({
            "name": block.get('name'),
            "source": "character",
            "enabled": bool(block.get('enabled', True)),
            "entries": block.get('entries') or [],
            "shadowed": block.get('name') in enabled_conversation_names,
        })
    return rows


def collect_active_chatdict_entries(
    db: "CharactersRAGDB",
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
) -> List[ChatDictionary]:
    """Collect the ChatDictionary entries that apply to the current send.

    Additive union of conversation + character dictionaries, deduped by name
    (conversation wins), enabled-only, never raises. Reimplemented on
    :func:`_resolve_active_dictionaries`; behavior is unchanged.
    """
    entries: List[ChatDictionary] = []
    for row in _resolve_active_dictionaries(db, conversation_id, char_data):
        if not row["enabled"] or row["shadowed"]:
            continue
        entries.extend(row["entries"])
    return entries


def summarize_active_dictionaries(
    db: "CharactersRAGDB",
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Dict-level "what's in play" summary for the current chat (never raises)."""
    dictionaries = [
        {
            "name": row["name"],
            "source": row["source"],
            "enabled": row["enabled"],
            "entry_count": len(row["entries"]),
            "shadowed": row["shadowed"],
        }
        for row in _resolve_active_dictionaries(db, conversation_id, char_data)
    ]
    return {"dictionaries": dictionaries, "source": "local"}
```

Note: `collect`'s old behavior added the name to the dedup set only for *enabled* conversation dicts and skipped disabled/colliding character dicts — the core preserves this exactly (a disabled conversation row is `enabled=False` so it's skipped by `collect` and doesn't populate `enabled_conversation_names`; a character row is `shadowed` only against `enabled_conversation_names`). Confirm `_resolve_active_dictionaries` sits after `load_character_dictionaries` in the file so it can call it.

- [ ] **Step 4: Run — all PASS** (including the existing `test_collect_active_chatdict_entries.py` suite, proving byte-unchanged).

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_resolve_active_dictionaries.py Tests/Character_Chat/test_collect_active_chatdict_entries.py Tests/Chat/test_chat_functions.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 5: Commit** — `feat(chat-dictionaries): shared union core + what's-in-play summary (collect byte-unchanged)` + `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Stage only `Chat_Dictionary_Lib.py` + the two test files.

---

### Task 2: Scope-service `summarize_active_dictionaries` wrapper

**Files:**
- Modify: `tldw_chatbook/Character_Chat/chat_dictionary_scope_service.py`
- Test: `Tests/Character_Chat/test_chat_dictionary_scope_service.py` (append)

**Interfaces:**
- Consumes: Task 1's `LocalChatDictionaryService`-hosted summary. **First** add a thin sync passthrough on `LocalChatDictionaryService` so the scope service can `_invoke` it: `summarize_active_dictionaries(self, conversation_id, character_id_or_data...)` — but the summary needs `char_data` (a dict), not an id. To keep the scope-service seam uniform (which passes primitives), the screen will call the module function directly for the char_data case; the scope wrapper is provided for symmetry over a conversation id + a character id. Simpler: expose the read on `LocalChatDictionaryService` taking `(conversation_id: Optional[str], character_id: Optional[int])`, resolving `char_data` via `get_character_card_by_id` internally, then calling the module `summarize_active_dictionaries`.
- Produces: `LocalChatDictionaryService.summarize_active_dictionaries(conversation_id, character_id) -> {...}`; async `ChatDictionaryScopeService.summarize_active_dictionaries(conversation_id, character_id, mode="local")` (read/detail action).

- [ ] **Step 1: Write the failing test.** Append to `Tests/Character_Chat/test_chat_dictionary_scope_service.py`:

```python
async def test_scope_service_summarize_active_dictionaries(tmp_path):
    import json as _json
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
    from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService

    db = CharactersRAGDB(tmp_path / "sum.db", "test-client")
    local = LocalChatDictionaryService(db)
    scope = ChatDictionaryScopeService(local_service=local, server_service=None)
    conv_id = db.add_conversation({"title": "c"})
    did = local.create_dictionary({"name": "Conv", "entries": [{"pattern": "x", "replacement": "y"}]})["id"]
    conv = db.get_conversation_by_id(conv_id)
    meta = _json.loads(conv.get("metadata") or "{}"); meta["active_dictionaries"] = [did]
    db.update_conversation(conv_id, {"metadata": _json.dumps(meta)}, expected_version=conv["version"])
    char_id = db.add_character_card({"name": "N"})
    local.attach_to_character(local.create_dictionary({"name": "Char"})["id"], char_id)

    out = await scope.summarize_active_dictionaries(conv_id, char_id, mode="local")
    names = {(d["name"], d["source"]) for d in out["dictionaries"]}
    assert ("Conv", "conversation") in names and ("Char", "character") in names
```

- [ ] **Step 2: Run — expect FAIL** (`AttributeError`).

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_chat_dictionary_scope_service.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 3: Implement.** On `LocalChatDictionaryService` (near `list_character_dictionaries`):

```python
    def summarize_active_dictionaries(self, conversation_id, character_id) -> dict[str, Any]:
        """What's-in-play summary for a chat: conversation dicts (by id) + a character's embedded dicts."""
        from . import Chat_Dictionary_Lib as cdl
        char_data = None
        if character_id is not None:
            char_data = self._require_db().get_character_card_by_id(int(character_id))
        conv_id = str(conversation_id) if conversation_id is not None else None
        return cdl.summarize_active_dictionaries(self._require_db(), conv_id, char_data)
```

On `ChatDictionaryScopeService` (after `list_character_dictionaries`):

```python
    async def summarize_active_dictionaries(self, conversation_id, character_id, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._statistics_action(normalized_mode, "detail"),
            "summarize_active_dictionaries",
            conversation_id,
            character_id,
        )
```

- [ ] **Step 4: Run — PASS.** (same command as Step 2)

- [ ] **Step 5: Commit** — `feat(chat-dictionaries): scope-service what's-in-play summary wrapper`. Stage only the two files.

---

### Task 3: Inspector "Chat Dictionaries" block (state fields + render, read-only)

**Files:**
- Modify: `tldw_chatbook/Chat/console_display_state.py` (`ConsoleInspectorState`)
- Modify: `tldw_chatbook/Widgets/Console/console_run_inspector.py` (`compose`)
- Test: `Tests/UI/test_console_dictionaries_inspector.py` (create)

**Interfaces:**
- Consumes: `ConsoleDisplayRow(label, value, status="ready")` and `ConsoleInspectorAction(widget_id, label, enabled, disabled_reason="")` (existing, `Chat/console_display_state.py`).
- Produces: two new fields on `ConsoleInspectorState` — `dictionary_rows: tuple[ConsoleDisplayRow, ...] = ()` and `dictionary_actions: tuple[ConsoleInspectorAction, ...] = ()`; the inspector renders a dedicated **"Chat Dictionaries"** heading + those rows + those action buttons (a self-contained block, NOT via `_ROW_GROUPS` — dictionary rows are dynamic and `_ROW_GROUPS` matches fixed labels).

- [ ] **Step 1: Write the failing test.** Create `Tests/UI/test_console_dictionaries_inspector.py`:

```python
"""P1g: the inspector renders a Chat Dictionaries block from state (no I/O)."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static, Button

from tldw_chatbook.Chat.console_display_state import (
    ConsoleInspectorState, ConsoleDisplayRow, ConsoleInspectorAction,
)
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector

pytestmark = pytest.mark.asyncio


def _state(**kw):
    return ConsoleInspectorState.from_values(**kw)


class _Host(App):
    def __init__(self, state):
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield ConsoleRunInspector(self._state)


async def test_dictionaries_block_renders_rows_and_actions():
    from dataclasses import replace
    state = replace(
        _state(),
        dictionary_rows=(
            ConsoleDisplayRow("Slang", "from conversation"),
            ConsoleDisplayRow("Period", "from character (shadowed)"),
        ),
        dictionary_actions=(
            ConsoleInspectorAction("console-inspector-dictionaries-attach", "Attach dictionary…", True),
            ConsoleInspectorAction("console-inspector-dictionaries-detach", "Detach dictionary…", True),
        ),
    )
    async with _Host(state).run_test(size=(120, 50)) as pilot:
        texts = [str(s.renderable) for s in pilot.app.query(Static)]
        assert any("Chat Dictionaries" in t for t in texts)
        assert any("Slang" in t for t in texts) and any("shadowed" in t for t in texts)
        assert pilot.app.query_one("#console-inspector-dictionaries-attach", Button)
        assert pilot.app.query_one("#console-inspector-dictionaries-detach", Button)


async def test_dictionaries_block_absent_when_empty():
    async with _Host(_state()).run_test(size=(120, 50)) as pilot:
        texts = [str(s.renderable) for s in pilot.app.query(Static)]
        assert not any("Chat Dictionaries" in t for t in texts)
```

- [ ] **Step 2: Run — expect FAIL** (`TypeError: ... unexpected keyword argument 'dictionary_rows'`).

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_dictionaries_inspector.py -q -p no:cacheprovider -o addopts="" --timeout=180 --timeout-method=thread
```

- [ ] **Step 3a: Add the state fields.** In `Chat/console_display_state.py`, in the `ConsoleInspectorState` dataclass, add (after `actions`):

```python
    dictionary_rows: tuple[ConsoleDisplayRow, ...] = ()
    dictionary_actions: tuple[ConsoleInspectorAction, ...] = ()
```

(The dataclass is `frozen`; `replace(...)` is how they're set — no `from_values` change needed.)

- [ ] **Step 3b: Render the block.** In `console_run_inspector.py` `compose`, AFTER the trailing ungrouped-actions loop (the final `for action in self.state.actions:` block), append a dedicated block:

```python
        dict_rows = getattr(self.state, "dictionary_rows", ())
        dict_actions = getattr(self.state, "dictionary_actions", ())
        if dict_rows or dict_actions:
            yield Static(
                "Chat Dictionaries",
                id="console-inspector-dictionaries-heading",
                classes="console-inspector-group-heading destination-section",
            )
            for index, row in enumerate(dict_rows):
                yield Static(
                    row.text,
                    id=f"console-inspector-dictionaries-row-{index}",
                    classes=f"console-inspector-row console-inspector-row-{row.status}",
                    markup=False,
                )
            for action in dict_actions:
                yield from self._compose_action(action)
```

- [ ] **Step 4: Run — both PASS.** (same command as Step 2)

- [ ] **Step 5: Commit** — `feat(console): inspector Chat Dictionaries block (read-only, from state)`. Stage only the three files.

---

### Task 4: Screen cache + reactive recompute + feed into the inspector build

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_dictionaries_screen.py` (create)

**Interfaces:**
- Consumes: Task 2's `service.summarize_active_dictionaries(conv_id, char_id, mode="local")` via the screen's dictionary scope service accessor; `app.current_chat_conversation_id` / `app.current_chat_active_character_data` (reactives); `ConsoleDisplayRow`/`ConsoleInspectorAction`; `_build_console_inspector_state` (`chat_screen.py:4926`, builds a fresh `ConsoleInspectorState`, uses `replace(...)` to append rows).
- Produces: a persistent screen attr `self._active_dictionaries_summary: dict | None`; `refresh_active_dictionaries_summary()` (recompute+cache, sync DB read via `asyncio.to_thread`-style worker or direct sync read on change); `_console_dictionary_inspector_rows()` + `_console_dictionary_inspector_actions()` (project the cache into rows/actions); `_build_console_inspector_state` appends them via `replace(inspector_state, dictionary_rows=..., dictionary_actions=...)`; `watch_current_chat_conversation_id`/`watch_current_chat_active_character_data` (or Console-side equivalents) call `refresh_active_dictionaries_summary()`.

- [ ] **Step 1: Write the failing test.** Create `Tests/UI/test_console_dictionaries_screen.py`. Mirror the Console `ChatScreen` harness in **`Tests/UI/test_chat_screen_state.py`** (how it constructs the screen + provides `app_instance`/db + builds/inspects `ConsoleInspectorState`). The test must prove: (a) after setting the current conversation with an attached dict and calling `refresh_active_dictionaries_summary()`, the built inspector state carries a `dictionary_rows` entry for that dict; (b) a second `_build_console_inspector_state()` call does NOT re-query the DB (patch the summarize call to count invocations — it must be called on `refresh_*`, not on build). Assert the cache attr holds the summary and the projected rows include the dict name/source. (Concrete construction depends on the harness; mirror the existing Console screen test's fixtures — do not invent a new harness.)

- [ ] **Step 2: Run — expect FAIL** (attr/method missing).

- [ ] **Step 3: Implement.**
  1. In `ChatScreen.__init__` add `self._active_dictionaries_summary: dict | None = None`.
  2. `refresh_active_dictionaries_summary()`: read `self.app_instance.current_chat_conversation_id` + the active character id (from `current_chat_active_character_data.get('id')`), call the scope service `summarize_active_dictionaries(conv_id, char_id, mode="local")` (in a worker / `asyncio.to_thread` — never on the UI thread synchronously if it hits the DB), store the result in `self._active_dictionaries_summary`, then request an inspector rebuild (call the same method the other change-paths use to rebuild + feed the inspector — find how `_build_console_inspector_state` results reach the inspector widget and trigger that). Guard: no scope service / no conversation and no character → cache `{"dictionaries": []}`.
  3. `_console_dictionary_inspector_rows() -> tuple[ConsoleDisplayRow, ...]`: from `self._active_dictionaries_summary`, one `ConsoleDisplayRow(name, "from conversation"|"from character" + (" (shadowed)" if shadowed else "") + (" (disabled)" if not enabled else ""))` per dictionary; empty → a single `ConsoleDisplayRow("No dictionaries in play", "")`; no conversation and no character → `ConsoleDisplayRow("No active chat", "")`.
  4. `_console_dictionary_inspector_actions() -> tuple[ConsoleInspectorAction, ...]`: `ConsoleInspectorAction("console-inspector-dictionaries-attach", "Attach dictionary…", enabled=bool(current_conversation_id), disabled_reason="Start or load a conversation first")` and `ConsoleInspectorAction("console-inspector-dictionaries-detach", "Detach dictionary…", enabled=<has conversation-source dicts>)`.
  5. In `_build_console_inspector_state`, before returning, append: `inspector_state = replace(inspector_state, dictionary_rows=self._console_dictionary_inspector_rows(), dictionary_actions=self._console_dictionary_inspector_actions())`. This reads ONLY the cache — no DB.
  6. Add `watch_current_chat_conversation_id(self, ...)` and `watch_current_chat_active_character_data(self, ...)` on `ChatScreen` (or hook the Console-side change points) that call `self.refresh_active_dictionaries_summary()`. Confirm the reactives are on the app; if the watcher must live on the app, add a small screen-notify instead — mirror how the Console already reacts to conversation changes.

- [ ] **Step 4: Run — PASS.** Then confirm the Console screen regression suite is green.

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_dictionaries_screen.py Tests/UI/test_chat_screen_state.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 5: Commit** — `feat(console): cache what's-in-play summary on change; feed inspector (no DB on recompose)`. Stage only `chat_screen.py` + the new test.

---

### Task 5: Console attach/detach handlers + picker wiring

**Files:**
- Create: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events_console_dictionaries.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (action-button `@on` handlers)
- Test: `Tests/UI/test_console_dictionaries_attach.py` (create)

**Interfaces:**
- Consumes: the scope service `attach_to_conversation(dict_id, conv_id, mode="local")` / `detach_from_conversation(dict_id, conv_id, mode="local")` (P1e); `Widgets/Persona_Widgets/dictionary_picker.py` `DictionaryPicker(ModalScreen[int | None])` (int ids); `cdl.list_chat_dictionaries` (attachable list) + the conversation's `active_dictionaries` (detachable list); Task 4's `refresh_active_dictionaries_summary()`.
- Produces: `handle_console_dictionary_attach(app)` / `handle_console_dictionary_detach(app)` (mirror `chat_events_worldbooks.py`: guard conversation/db, run in `app.run_worker(...)` with `exit_on_error` handling, over the P1e seam, then `refresh`); `ChatScreen` `@on(Button.Pressed, "#console-inspector-dictionaries-attach")` / `"#console-inspector-dictionaries-detach"` that open `DictionaryPicker` (filtered) and dispatch to the handlers.

- [ ] **Step 1: Write the failing test.** Create `Tests/UI/test_console_dictionaries_attach.py`, mirroring the Console screen harness (`Tests/UI/test_chat_screen_state.py`, as in Task 4). Seed a real DB with a conversation + a dictionary; monkeypatch `push_screen_wait` to auto-return the dict id (as the P1e/P1f attach tests do); click `#console-inspector-dictionaries-attach`; assert the conversation's `active_dictionaries` gained the id AND the cached summary now lists it as `source: "conversation"`. Then detach (monkeypatched picker returns the same id) and assert it's removed. Do NOT weaken the round-trip assertions; character-source dicts must never appear in the detach picker's list.

- [ ] **Step 2: Run — expect FAIL** (`NoMatches`/handler missing).

- [ ] **Step 3: Implement.**
  - Create `chat_events_console_dictionaries.py` with `handle_console_dictionary_attach(app, dictionary_id)` and `handle_console_dictionary_detach(app, dictionary_id)`: guard `app.current_chat_conversation_id` (else `app.notify("No active conversation…", severity="warning")`) and the scope service; call `await service.attach_to_conversation(int(dictionary_id), str(conv_id), mode="local")` (or detach); on `ConflictError` → `app.notify("Dictionaries changed since loaded. Try again.", severity="warning")`; on success → `screen.refresh_active_dictionaries_summary()`. Wrap the DB list read used to build the picker (`_console_attachable_dictionaries` / `_console_attached_dictionaries`) so a bad row never raises.
  - In `ChatScreen`, add `@on(Button.Pressed, "#console-inspector-dictionaries-attach")` → open `DictionaryPicker(list_of_attachable)` via `push_screen_wait` in a worker (`exit_on_error` discipline), then `handle_console_dictionary_attach(self.app_instance, picked)`. The attachable list = `cdl.list_chat_dictionaries(db, limit=1000, include_disabled=True)` (rows `{"dictionary_id": int, "name": str}`) MINUS the conversation's current `active_dictionaries` ids. Detach handler: `@on(Button.Pressed, "#console-inspector-dictionaries-detach")` → `DictionaryPicker(list_of_attached_conversation_dicts)` (built from the conversation's `active_dictionaries`, each `{"dictionary_id": id, "name": load_chat_dictionary(db, id)["name"]}`; character dicts are NOT included) → `handle_console_dictionary_detach`.

- [ ] **Step 4: Run — PASS.** Then the Console regression suite.

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_dictionaries_attach.py Tests/UI/test_console_dictionaries_inspector.py Tests/UI/test_console_dictionaries_screen.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 5: Commit** — `feat(console): attach/detach conversation dictionaries from the inspector (over P1e seam)`. Stage only the handler file, `chat_screen.py`, and the new test.

---

### Task 6: Full gate + spec status

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-17-roleplay-p1g-whats-in-play-design.md` (status line)

- [ ] **Step 1: Full gate.**

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Character_Chat/ Tests/Chat/test_chat_functions.py \
  Tests/UI/test_console_dictionaries_inspector.py Tests/UI/test_console_dictionaries_screen.py Tests/UI/test_console_dictionaries_attach.py \
  Tests/UI/test_chat_screen_state.py Tests/UI/test_console_run_gate.py Tests/UI/test_console_persistent_rails.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

Expected: all pass (record exact counts). Then `import tldw_chatbook.app` smoke:

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('IMPORT OK')"
```

- [ ] **Step 2: Flip spec status** — `**Status:** Design approved (brainstorm), pending spec review.` → `**Status:** Implemented (P1g).`

- [ ] **Step 3: Commit** — `docs(roleplay): mark P1g Console what's-in-play spec implemented`.

---

## Notes for the executor

- **Load-bearing tests** (do not accept a fake substitute): Task 1 real-DB shared-core parity (`summarize` applied-set == what `collect` loads) + shadowed-only-by-enabled-conversation-dict + collect byte-unchanged; Task 4 compute-on-change/cache (summary updates on conversation switch, NO DB query on a plain inspector rebuild); Task 5 real-seam attach/detach round-trip. These pin the two invariants the feature exists for: "shown = applied" and "no DB on recompose".
- **Tasks 4 & 5 are Console-integration tasks** into a large `chat_screen.py`. The plan gives exact seams (`_build_console_inspector_state` `replace(...)` append at `:4926`; inspector-action `@on(Button.Pressed, "#<widget_id>")` routing; the `chat_events_worldbooks.py` handler pattern; the P1f `DictionaryPicker`), but the implementer must READ the surrounding `chat_screen.py` methods (how the built inspector state reaches the widget, how the Console already reacts to conversation change) and integrate — mirror existing patterns, don't invent. If the reactive `watch_*` can't live on the screen, hook the Console's existing conversation-change path and call `refresh_active_dictionaries_summary()` there.
- **Zero DB on recompose is a hard rule** — `_build_console_inspector_state` and the inspector's `compose` must only READ `self._active_dictionaries_summary` / the state fields; the DB read happens solely in `refresh_active_dictionaries_summary()` (on change).
- The inspector "Chat Dictionaries" block is rendered by a **dedicated compose block + dedicated state fields**, NOT `_ROW_GROUPS` (dictionary rows are dynamic; `_ROW_GROUPS` matches fixed labels) — this refines the spec's mechanism while meeting its intent.
