# Roleplay P1a — Dictionaries Mode Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Roleplay strip's Dictionaries chip a working three-pane workbench (List · Detail · Try-it) over the existing local `chat_dictionary_scope_service` — no backend changes.

**Architecture:** Two new self-contained widgets (`PersonasDictionaryDetailWidget` center detail with Entries/Settings tabs; `PersonasDictionaryTryItWidget` right-pane substitution preview) that emit intent messages; `PersonasScreen` performs all persistence via `app_instance.chat_dictionary_scope_service` and re-feeds the widgets (reload-after-mutation — entry ids are positional). `PersonasLibraryPane` grows optional height-2 meta rows.

**Tech Stack:** Python ≥3.11, Textual (TabbedContent/DataTable/ListView/TextArea), pytest-asyncio `app.run_test()` harness, difflib for the word-diff.

**Spec:** `Docs/superpowers/specs/2026-07-13-roleplay-p1a-dictionaries-foundation-design.md` (read it once before Task 1; its "Backend reality" and "Verified integration facts" sections are binding).

## Global Constraints

- **Collision guard:** do NOT touch `screen_registry.py`, `shell_destinations.py`, `route_inventory.py`, the route id `personas`, or `MODE_CHIP_ORDER` (the `"prompts"` entry stays; a parallel branch owns its removal). No backend/schema changes.
- **API naming only:** the UI speaks `pattern` / `replacement` / `probability` (**0–1 float** in payloads/responses; **displayed as 0–100 %**) / `type` (`"literal"|"regex"`) / `group` / `max_replacements`. Never `key`/`content`.
- **`list_dictionaries` must be called with `include_inactive=True`** — the default silently drops disabled dictionaries (a toggled-off row would vanish). Its return key is `"dictionaries"` (not `"items"`).
- **Entry ids are positional** (`local:chat_dictionary_entry:<dict_id>:<index>`): after EVERY entry mutation, reload entries from the service before the next action; never cache row→id across a mutation.
- **`reorder_entries` semantics are move-to-front** (`selected + remainder`): always pass the FULL current id list in the desired final order as `{"entry_ids": [...]}`.
- **`create_dictionary` ignores `strategy`** (column default `sorted_evenly`): Duplicate must follow with `update_dictionary(new_id, {"strategy": src})` when the source's differs. `name` is UNIQUE — New/Duplicate must disambiguate (`"Untitled dictionary"`, `"… 2"`; `"{name} (copy)"`, `"… (copy 2)"`).
- **Dictionary-level copy:** list row meta line is exactly `"{n} entries · on"` / `"{n} entries · off"`.
- Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (venv, isolated HOME; from a git worktree substitute the main checkout's absolute `.venv/bin/python`):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    .venv/bin/python -m pytest <target> \
    -q -p no:cacheprovider -o addopts="" --timeout=180 --timeout-method=thread
  ```
- Widget `DEFAULT_CSS` is structure-only (sizes/layout); colors come from the app stylesheet. Diff highlighting uses theme-safe inline Rich styles (`strike`, `bold underline`), no colors.

---

### Task 1: `LibraryRow.meta` + height-2 rows in `PersonasLibraryPane`

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py`
- Test: `Tests/UI/test_personas_dictionaries.py` (create)

**Interfaces:**
- Consumes: existing `LibraryRow` dataclass (`item_id/kind/name/is_unsaved`), `update_rows` row loop (`personas_library_pane.py:190-206`), row CSS block (`:59-73`).
- Produces: `LibraryRow.meta: str | None = None` (new optional field). Rows with `meta` render as a 2-line ListItem: name Static + `Static(meta, classes="personas-library-row-meta")`. Later tasks build rows via `LibraryRow(item_id=..., kind="dictionary", name=..., meta="3 entries · on")`.

- [ ] **Step 1: Create the test file with harness + failing meta-row tests**

Create `Tests/UI/test_personas_dictionaries.py`:

```python
# Tests/UI/test_personas_dictionaries.py
"""Mounted tests for the Roleplay Dictionaries mode (P1a)."""

import copy
from pathlib import Path
from typing import Any

import pytest
from textual.app import App
from textual.widgets import Button, Input, ListItem, ListView, Static

from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    LibraryRow,
    PersonasLibraryPane,
)

pytestmark = pytest.mark.asyncio


def _entry_response(dictionary_id: int, index: int, entry: dict) -> dict:
    """Shape one entry exactly like local _entry_to_response (API naming)."""
    return {
        "id": f"local:chat_dictionary_entry:{dictionary_id}:{index}",
        "dictionary_id": dictionary_id,
        "index": index,
        "pattern": entry.get("pattern", ""),
        "replacement": entry.get("replacement", ""),
        "probability": entry.get("probability", 1.0),
        "group": entry.get("group"),
        "timed_effects": entry.get("timed_effects"),
        "max_replacements": entry.get("max_replacements", 1),
        "type": entry.get("type", "literal"),
        "enabled": True,
        "source": "local",
    }


class FakeDictScopeService:
    """In-memory stand-in mirroring the real local shapes and quirks."""

    def __init__(self, records: list[dict] | None = None) -> None:
        self.records: dict[int, dict] = {}
        self.calls: list[tuple] = []
        for record in records or []:
            self.records[int(record["id"])] = copy.deepcopy(record)
        self._next_id = max(self.records, default=0) + 1

    def _summary(self, record: dict) -> dict:
        out = copy.deepcopy(record)
        out["entries"] = [
            _entry_response(int(record["id"]), i, e)
            for i, e in enumerate(record.get("entries") or [])
        ]
        out["is_active"] = bool(record.get("enabled", True))
        return out

    async def list_dictionaries(self, mode: str = "local", **kwargs: Any) -> dict:
        self.calls.append(("list", kwargs))
        assert mode == "local"
        # Real local list drops disabled records unless include_inactive=True.
        include_inactive = bool(kwargs.get("include_inactive", False))
        items = [
            self._summary(r)
            for r in self.records.values()
            if include_inactive or r.get("enabled", True)
        ]
        return {"dictionaries": items, "source": "local"}

    async def get_dictionary(self, dictionary_id: int, mode: str = "local") -> dict:
        record = self.records.get(int(dictionary_id))
        if record is None:
            raise ValueError(f"Local chat dictionary '{dictionary_id}' was not found.")
        return self._summary(record)

    async def create_dictionary(self, request_data: Any, mode: str = "local") -> dict:
        payload = dict(request_data)
        if any(r["name"] == payload["name"] for r in self.records.values()):
            from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError

            raise ConflictError(f"duplicate name {payload['name']}")
        new_id = self._next_id
        self._next_id += 1
        record = {
            "id": new_id,
            "name": payload["name"],
            "description": payload.get("description") or "",
            "strategy": "sorted_evenly",  # create_dictionary ignores strategy
            "max_tokens": int(payload.get("max_tokens") or 1000),
            "enabled": bool(payload.get("enabled", True)),
            "version": 1,
            "entries": [
                {
                    "pattern": e.get("pattern", ""),
                    "replacement": e.get("replacement", ""),
                    "probability": e.get("probability", 1.0),
                    "group": e.get("group"),
                    "timed_effects": e.get("timed_effects"),
                    "max_replacements": e.get("max_replacements", 1),
                    "type": e.get("type", "literal"),
                }
                for e in payload.get("entries") or []
            ],
        }
        self.records[new_id] = record
        self.calls.append(("create", payload["name"]))
        return self._summary(record)

    async def update_dictionary(
        self, dictionary_id: int, request_data: Any, mode: str = "local", **kwargs: Any
    ) -> dict:
        record = self.records[int(dictionary_id)]
        expected = kwargs.get("expected_version")
        if expected is not None and expected != record["version"]:
            from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError

            raise ConflictError("version mismatch")
        payload = dict(request_data)
        for field in ("name", "description", "strategy", "max_tokens", "enabled"):
            if field in payload:
                record[field] = payload[field]
        if "entries" in payload:
            record["entries"] = [dict(e) for e in payload["entries"] or []]
        record["version"] += 1
        self.calls.append(("update", int(dictionary_id), payload))
        return self._summary(record)

    async def delete_dictionary(
        self, dictionary_id: int, mode: str = "local", **kwargs: Any
    ) -> dict:
        record = self.records.pop(int(dictionary_id))
        self.calls.append(("delete", int(dictionary_id)))
        return {"status": "deleted", "dictionary_id": record["id"], "source": "local"}

    async def add_entry(self, dictionary_id: int, request_data: Any, mode: str = "local") -> dict:
        record = self.records[int(dictionary_id)]
        record["entries"].append(dict(request_data))
        record["version"] += 1
        self.calls.append(("add_entry", int(dictionary_id)))
        return _entry_response(int(dictionary_id), len(record["entries"]) - 1, dict(request_data))

    async def list_entries(self, dictionary_id: int, mode: str = "local", **kwargs: Any) -> dict:
        record = self.records[int(dictionary_id)]
        return {
            "dictionary_id": int(dictionary_id),
            "entries": [
                _entry_response(int(dictionary_id), i, e)
                for i, e in enumerate(record["entries"])
            ],
            "source": "local",
        }

    @staticmethod
    def _parse_entry_id(entry_id: str) -> tuple[int, int]:
        parts = str(entry_id).split(":")
        return int(parts[-2]), int(parts[-1])

    async def update_entry(self, entry_id: str, request_data: Any, mode: str = "local") -> dict:
        dictionary_id, index = self._parse_entry_id(entry_id)
        record = self.records[dictionary_id]
        record["entries"][index].update(dict(request_data))
        record["version"] += 1
        self.calls.append(("update_entry", entry_id))
        return _entry_response(dictionary_id, index, record["entries"][index])

    async def delete_entry(self, entry_id: str, mode: str = "local") -> dict:
        dictionary_id, index = self._parse_entry_id(entry_id)
        record = self.records[dictionary_id]
        del record["entries"][index]
        record["version"] += 1
        self.calls.append(("delete_entry", entry_id))
        return {"status": "deleted", "entry_id": entry_id, "source": "local"}

    async def reorder_entries(self, dictionary_id: int, request_data: Any, mode: str = "local") -> dict:
        # Mirrors the real move-to-front semantics: selected + remainder.
        payload = dict(request_data)
        record = self.records[int(dictionary_id)]
        entries = list(record["entries"])
        selected_indexes = [self._parse_entry_id(eid)[1] for eid in payload.get("entry_ids") or []]
        selected = [entries[i] for i in selected_indexes if i < len(entries)]
        remainder = [e for i, e in enumerate(entries) if i not in set(selected_indexes)]
        record["entries"] = selected + remainder
        record["version"] += 1
        self.calls.append(("reorder", int(dictionary_id), list(payload.get("entry_ids") or [])))
        return {"dictionary_id": int(dictionary_id), "entry_ids": payload.get("entry_ids"), "source": "local"}

    async def process_text(self, request_data: Any, mode: str = "local") -> dict:
        payload = dict(request_data)
        text = payload["text"]
        record = self.records.get(int(payload.get("dictionary_id") or 0))
        if record is None:
            raise ValueError("Local chat dictionary was not found.")
        processed = text
        for entry in record["entries"]:
            if entry.get("type") != "regex" and entry.get("pattern"):
                processed = processed.replace(entry["pattern"], entry["replacement"])
        self.calls.append(("process", text))
        return {
            "text": text,
            "processed_text": processed,
            "dictionary_id": record["id"],
            "source": "local",
        }


def make_dict_record(
    record_id: int = 1,
    name: str = "Medical Abbrev",
    *,
    enabled: bool = True,
    strategy: str = "sorted_evenly",
    entries: list[dict] | None = None,
) -> dict:
    return {
        "id": record_id,
        "name": name,
        "description": "",
        "strategy": strategy,
        "max_tokens": 1000,
        "enabled": enabled,
        "version": 1,
        "entries": entries
        if entries is not None
        else [
            {"pattern": "BP", "replacement": "blood pressure", "probability": 1.0,
             "group": None, "timed_effects": None, "max_replacements": 1, "type": "literal"},
            {"pattern": "HR", "replacement": "heart rate", "probability": 1.0,
             "group": None, "timed_effects": None, "max_replacements": 1, "type": "literal"},
        ],
    }


@pytest.fixture
def fake_dict_service(mock_app_instance):
    service = FakeDictScopeService([make_dict_record(1), make_dict_record(2, "Chat-Speak", enabled=False, entries=[])])
    mock_app_instance.chat_dictionary_scope_service = service
    return service


@pytest.fixture
def stub_characters(monkeypatch):
    import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module

    monkeypatch.setattr(character_handler_module, "fetch_all_characters", lambda: [])
    monkeypatch.setattr(
        character_handler_module, "fetch_character_by_id", lambda character_id: None
    )


class PersonasTestApp(App):
    """Same harness as test_personas_workbench.py (delegating App)."""

    def __init__(self, mock_app_instance):
        super().__init__()
        self._mock = mock_app_instance
        self.character_persona_scope_service = mock_app_instance.character_persona_scope_service

    _NON_DELEGATED_PREFIXES = ("_", "watch_", "compute_", "validate_", "action_", "key_", "on_")

    def __getattr__(self, name):
        if name.startswith(self._NON_DELEGATED_PREFIXES):
            raise AttributeError(name)
        return getattr(self.__dict__["_mock"], name)

    def compose(self):
        yield AppFooterStatus(id="app-footer-status")

    def on_mount(self) -> None:
        self.push_screen(PersonasScreen(self))


class StyledPersonasTestApp(PersonasTestApp):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"
    )


async def _mounted(pilot):
    await pilot.pause()
    return pilot.app.screen


async def _enter_dictionaries(pilot):
    screen = await _mounted(pilot)
    await pilot.click("#personas-mode-dictionaries")
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return screen


class TestLibraryMetaRows:
    async def test_meta_row_renders_two_lines(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            pane = screen.query_one(PersonasLibraryPane)
            await pane.update_rows(
                (LibraryRow(item_id="1", kind="dictionary", name="Medical Abbrev", meta="2 entries · on"),),
                total=1,
                noun="dictionaries",
            )
            await pilot.pause()
            row = screen.query_one("#personas-library-rows", ListView).children[0]
            statics = row.query(Static).results()
            texts = [str(s.renderable) for s in statics]
            assert texts == ["Medical Abbrev", "2 entries · on"]
            meta = row.query_one(".personas-library-row-meta", Static)
            assert meta is not None

    async def test_metaless_rows_stay_single_line(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            pane = screen.query_one(PersonasLibraryPane)
            await pane.update_rows(
                (LibraryRow(item_id="1", kind="character", name="Detective Sam"),),
                total=1,
                noun="characters",
            )
            await pilot.pause()
            row = screen.query_one("#personas-library-rows", ListView).children[0]
            assert len(list(row.query(Static).results())) == 1

    async def test_meta_row_geometry_not_clipped(self, mock_app_instance, stub_characters, fake_dict_service):
        """Geometry pilot: the meta line must actually render (rail CSS pins height:1)."""
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 48)) as pilot:
            screen = await _mounted(pilot)
            pane = screen.query_one(PersonasLibraryPane)
            await pane.update_rows(
                (LibraryRow(item_id="1", kind="dictionary", name="Medical Abbrev", meta="2 entries · on"),),
                total=1,
                noun="dictionaries",
            )
            await pilot.pause()
            row = screen.query_one("#personas-library-rows", ListView).children[0]
            assert row.region.height >= 2
            meta = row.query_one(".personas-library-row-meta", Static)
            assert meta.region.height >= 1
```

- [ ] **Step 2: Run to verify the meta tests fail**

Run: `HOME=... .venv/bin/python -m pytest Tests/UI/test_personas_dictionaries.py -k MetaRows -q ...` (Global Constraints command)
Expected: FAIL — `TypeError: LibraryRow.__init__() got an unexpected keyword argument 'meta'`.

- [ ] **Step 3: Implement `meta` on `LibraryRow` + 2-line rendering**

In `personas_library_pane.py`, extend the dataclass (`:38-44`):

```python
class LibraryRow:
    """One selectable row in the workbench library list."""

    item_id: str
    kind: PersonaEntityKind
    name: str
    is_unsaved: bool = False
    meta: str | None = None
```

In `update_rows`, replace the single-Static append (`:203-205`) with:

```python
            if row.meta:
                classes += " personas-library-row-tall"
                items.append(
                    ListItem(
                        Vertical(
                            Static(row.name, markup=False),
                            Static(
                                row.meta,
                                markup=False,
                                classes="personas-library-row-meta destination-purpose",
                            ),
                        ),
                        id=dom_id,
                        classes=classes,
                    )
                )
            else:
                items.append(
                    ListItem(Static(row.name, markup=False), id=dom_id, classes=classes)
                )
```

Add `Vertical` to the module's textual imports. In the `DEFAULT_CSS` block (after the existing `ListItem` rules), add structure-only rules so tall rows escape the pinned `height: 1`:

```css
    PersonasLibraryPane #personas-library-rows ListItem.personas-library-row-tall {
        height: 2;
        min-height: 2;
    }

    PersonasLibraryPane #personas-library-rows ListItem.personas-library-row-tall Vertical {
        height: 2;
    }

    PersonasLibraryPane #personas-library-rows ListItem.personas-library-row-tall Static {
        height: 1;
    }
```

- [ ] **Step 4: Run the three MetaRows tests — expect PASS.** If the geometry test fails with height 1, the tall-row CSS selector isn't beating the pinned `height: 1` rule — raise its specificity (keep the class-based selector, same block), do not touch the app stylesheet.

- [ ] **Step 5: Run the existing pane consumers for regressions**

Run: `... -m pytest Tests/UI/test_personas_workbench.py -q ...`
Expected: PASS (170 tests; metaless rows render exactly as before).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py Tests/UI/test_personas_dictionaries.py
git commit -m "feat(personas): optional height-2 meta rows in the library rail

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Dictionaries mode lists dictionaries (chip live, P0 debt, search)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_dictionaries.py`, `Tests/UI/test_personas_workbench.py` (P0 debt)

**Interfaces:**
- Consumes: Task 1's `LibraryRow(meta=...)`; `FakeDictScopeService` fixture; `_apply_mode` else-branch (`personas_screen.py:899-902`); `_COMING_SOON_MODES`/`_MODE_PLACEHOLDER_BODY` (`:96-106`); `_render_search_query` (`:768`).
- Produces: `PersonasScreen._dictionary_scope_service` (returns service or None), `PersonasScreen._dictionaries_cache: list[dict]` (summaries incl. `version`), `_dictionary_row(record) -> LibraryRow`, `async _render_dictionary_rows(query: str = "") -> None`. Later tasks reuse all four.

- [ ] **Step 1: Write the failing tests** (append to `Tests/UI/test_personas_dictionaries.py`):

```python
class TestDictionariesList:
    async def test_mode_lists_dictionaries_with_meta(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            rows = screen.query_one("#personas-library-rows", ListView).children
            texts = [str(s.renderable) for r in rows for s in r.query(Static).results()]
            assert "Medical Abbrev" in texts and "2 entries · on" in texts
            # Disabled dictionaries must still be listed (include_inactive=True).
            assert "Chat-Speak" in texts and "0 entries · off" in texts
            # No placeholder: the mode is live.
            assert screen.query_one("#personas-mode-placeholder", Static).display is False

    async def test_list_requests_inactive_records(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            await _enter_dictionaries(pilot)
            list_calls = [c for c in fake_dict_service.calls if c[0] == "list"]
            assert list_calls and list_calls[-1][1].get("include_inactive") is True

    async def test_search_filters_by_name(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            search = screen.query_one("#personas-library-search", Input)
            search.value = "medic"
            await pilot.pause(0.4)  # debounce
            rows = screen.query_one("#personas-library-rows", ListView).children
            names = [str(r.query(Static).first().renderable) for r in rows]
            assert names == ["Medical Abbrev"]

    async def test_dictionaries_chip_no_longer_coming_soon(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            chip = screen.query_one("#personas-mode-dictionaries", Button)
            assert "soon" not in str(chip.label).lower()

    async def test_empty_state_prompts_create(self, mock_app_instance, stub_characters):
        service = FakeDictScopeService([])
        mock_app_instance.chat_dictionary_scope_service = service
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            empty = screen.query_one("#personas-library-empty", Static)
            assert "No dictionaries yet" in str(empty.renderable)
```

- [ ] **Step 2: Run — expect FAIL** (rows empty / placeholder still shown / chip still "· soon").

- [ ] **Step 3: Implement**

In `personas_screen.py`:

(a) `_COMING_SOON_MODES` (`:98`) → `frozenset({"lore"})`. In `_MODE_PLACEHOLDER_BODY` (`:101-105`) delete the `"dictionaries"` entry (keep `"lore"` and `"prompts"`).

(b) Compose default placeholder (`:439`): `self._mode_placeholder_text("dictionaries")` → `self._mode_placeholder_text("lore")`.

(c) Add after `_profile_record` (or near the other service getters):

```python
    def _dictionary_scope_service(self) -> Any:
        """The app-level dictionaries scope service, or None when absent."""
        return getattr(self.app_instance, "chat_dictionary_scope_service", None)

    @staticmethod
    def _dictionary_row(record: dict) -> LibraryRow:
        entries = record.get("entries") or []
        state = "on" if record.get("enabled", record.get("is_active", True)) else "off"
        return LibraryRow(
            item_id=str(record.get("id")),
            kind="dictionary",
            name=str(record.get("name") or "Unnamed"),
            meta=f"{len(entries)} entries · {state}",
        )

    async def _render_dictionary_rows(self, query: str = "") -> None:
        """Fetch and render dictionary rows; degrade to recovery copy on failure."""
        library = self.query_one(PersonasLibraryPane)
        service = self._dictionary_scope_service()
        if service is None:
            await library.update_rows(
                (), total=0, noun="dictionaries",
                recovery_copy="Dictionaries are unavailable: the service is not configured.",
            )
            return
        try:
            response = await service.list_dictionaries(mode="local", include_inactive=True)
            records = list(response.get("dictionaries") or [])
        except Exception:
            logger.opt(exception=True).warning("Could not list chat dictionaries.")
            await library.update_rows(
                (), total=0, noun="dictionaries",
                recovery_copy="Dictionaries could not be loaded.\nSwitch modes and back to retry.",
            )
            return
        self._dictionaries_cache = records
        needle = query.strip().lower()
        visible = [r for r in records if needle in str(r.get("name", "")).lower()] if needle else records
        rows = tuple(self._dictionary_row(r) for r in visible)
        await library.update_rows(
            rows, total=len(records), noun="dictionaries", filtered=bool(needle),
        )
```

Initialize `self._dictionaries_cache: list[dict] = []` in `__init__`.

(d) In `_apply_mode`, insert a dictionaries branch before the final `else` (`:899`):

```python
        elif mode == "dictionaries":
            await self._render_dictionary_rows()
            self._show_center(None)
```

(e) In `_render_search_query` (`:768`), add after the `personas` branch:

```python
        elif mode == "dictionaries":
            try:
                await self._render_dictionary_rows(query=query)
            except Exception:
                logger.opt(exception=True).warning("Could not re-render dictionary rows after search.")
```

- [ ] **Step 4: Run the new tests — expect PASS.**

- [ ] **Step 5: Fix the two P0-debt tests in `Tests/UI/test_personas_workbench.py`**

In `test_mode_chips_are_self_explaining_and_mark_coming_soon` (~`:384`): change the "· soon"-marked chip from dictionaries to lore —

```python
            lore_chip = screen.query_one("#personas-mode-lore", Button)
            assert lore_chip.tooltip == "Lore — world facts injected on keywords."
            assert "soon" in str(lore_chip.label).lower()
            char_chip = screen.query_one("#personas-mode-characters", Button)
            assert "soon" not in str(char_chip.label).lower()
```

In `test_coming_soon_mode_shows_inviting_copy` (~`:394`): drive with `lore` — `await screen._apply_mode("lore")` (assertions unchanged).

- [ ] **Step 6: Run both files — expect PASS** (`Tests/UI/test_personas_dictionaries.py Tests/UI/test_personas_workbench.py`).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_dictionaries.py Tests/UI/test_personas_workbench.py
git commit -m "feat(personas): dictionaries mode lists real records; chip leaves coming-soon

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `PersonasDictionaryDetailWidget` (Entries + Settings tabs, standalone)

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: nothing from other tasks (self-contained widget).
- Produces (Tasks 4–9 rely on these EXACT names):
  - `PersonasDictionaryDetailWidget(Vertical)` with `id` passed by caller; methods `load_dictionary(record: dict) -> None` (settings fields + entries table from a get_dictionary/summary dict), `update_entries(entries: list[dict]) -> None`, `clear() -> None`, `settings_payload() -> dict` (`{"name","description","strategy","max_tokens","enabled"}`), `form_payload() -> dict | None` (API-named entry payload, probability %→0–1 float; `None` + inline error when pattern empty), `selected_entry_id: str | None` (property; the table-cursor entry id).
  - Messages (module-level, all `Message` subclasses): `DictionaryEntryAddRequested(payload: dict)`, `DictionaryEntryUpdateRequested(entry_id: str, payload: dict)`, `DictionaryEntryDeleteRequested(entry_id: str)`, `DictionaryEntriesReorderRequested(entry_ids: list[str])`, `DictionarySettingsSaveRequested(payload: dict)`, `DictionarySettingsEdited()`.
  - DOM ids: `#personas-dict-tabs`, tab panes `#personas-dict-tab-entries` / `#personas-dict-tab-settings`, table `#personas-dict-entries-table`, form inputs `#personas-dict-entry-pattern`, `#personas-dict-entry-replacement` (TextArea), `#personas-dict-entry-regex` (Switch), `#personas-dict-entry-probability`, `#personas-dict-entry-group`, `#personas-dict-entry-max-repl`, buttons `#personas-dict-entry-add`, `#personas-dict-entry-update`, `#personas-dict-entry-delete`, `#personas-dict-entry-up`, `#personas-dict-entry-down`, form error `#personas-dict-entry-error`; settings `#personas-dict-name`, `#personas-dict-description` (TextArea), `#personas-dict-strategy` (Select: `sorted_evenly`/`character_lore_first`/`global_lore_first`), `#personas-dict-max-tokens`, `#personas-dict-enabled` (Switch), `#personas-dict-settings-save`, status `#personas-dict-status`.

- [ ] **Step 1: Write the failing tests** (append; a bare-App harness — the widget is screen-independent):

```python
from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_detail import (
    DictionaryEntryAddRequested,
    DictionaryEntryDeleteRequested,
    DictionaryEntriesReorderRequested,
    DictionaryEntryUpdateRequested,
    DictionarySettingsSaveRequested,
    PersonasDictionaryDetailWidget,
)


class DetailHarnessApp(App):
    def __init__(self):
        super().__init__()
        self.messages = []

    def compose(self):
        yield PersonasDictionaryDetailWidget(id="personas-dictionary-detail")

    def on_dictionary_entry_add_requested(self, m): self.messages.append(m)
    def on_dictionary_entry_update_requested(self, m): self.messages.append(m)
    def on_dictionary_entry_delete_requested(self, m): self.messages.append(m)
    def on_dictionary_entries_reorder_requested(self, m): self.messages.append(m)
    def on_dictionary_settings_save_requested(self, m): self.messages.append(m)


class TestDictionaryDetailWidget:
    async def test_load_populates_settings_and_entries(self):
        from textual.widgets import DataTable, Select, Switch, TextArea

        app = DetailHarnessApp()
        async with app.run_test() as pilot:
            widget = app.query_one(PersonasDictionaryDetailWidget)
            record = FakeDictScopeService([make_dict_record(1)])._summary(make_dict_record(1))
            widget.load_dictionary(record)
            await pilot.pause()
            assert app.query_one("#personas-dict-name", Input).value == "Medical Abbrev"
            table = app.query_one("#personas-dict-entries-table", DataTable)
            assert table.row_count == 2
            assert app.query_one("#personas-dict-max-tokens", Input).value == "1000"
            assert app.query_one("#personas-dict-enabled", Switch).value is True

    async def test_form_payload_converts_probability_percent(self):
        app = DetailHarnessApp()
        async with app.run_test() as pilot:
            widget = app.query_one(PersonasDictionaryDetailWidget)
            app.query_one("#personas-dict-entry-pattern", Input).value = "ASAP"
            app.query_one("#personas-dict-entry-probability", Input).value = "85"
            payload = widget.form_payload()
            assert payload["pattern"] == "ASAP"
            assert payload["probability"] == pytest.approx(0.85)
            assert payload["type"] == "literal"

    async def test_form_payload_requires_pattern(self):
        app = DetailHarnessApp()
        async with app.run_test() as pilot:
            widget = app.query_one(PersonasDictionaryDetailWidget)
            app.query_one("#personas-dict-entry-pattern", Input).value = "  "
            assert widget.form_payload() is None
            error = app.query_one("#personas-dict-entry-error", Static)
            assert "pattern" in str(error.renderable).lower()

    async def test_add_button_posts_message(self):
        from textual.widgets import TextArea

        app = DetailHarnessApp()
        async with app.run_test() as pilot:
            app.query_one("#personas-dict-entry-pattern", Input).value = "BP"
            app.query_one("#personas-dict-entry-replacement", TextArea).text = "blood pressure"
            await pilot.click("#personas-dict-entry-add")
            await pilot.pause()
            adds = [m for m in app.messages if isinstance(m, DictionaryEntryAddRequested)]
            assert adds and adds[0].payload["pattern"] == "BP"

    async def test_settings_save_posts_payload(self):
        app = DetailHarnessApp()
        async with app.run_test() as pilot:
            widget = app.query_one(PersonasDictionaryDetailWidget)
            record = FakeDictScopeService([make_dict_record(1)])._summary(make_dict_record(1))
            widget.load_dictionary(record)
            await pilot.pause()
            app.query_one("#personas-dict-name", Input).value = "Renamed"
            await pilot.click("#personas-dict-settings-save")
            await pilot.pause()
            saves = [m for m in app.messages if isinstance(m, DictionarySettingsSaveRequested)]
            assert saves and saves[0].payload["name"] == "Renamed"
            assert saves[0].payload["max_tokens"] == 1000
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: personas_dictionary_detail`).

- [ ] **Step 3: Implement the widget**

Create `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py`:

```python
"""Center detail widget for the Roleplay Dictionaries mode (Entries + Settings)."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    TextArea,
)

STRATEGIES = ("sorted_evenly", "character_lore_first", "global_lore_first")


class DictionaryEntryAddRequested(Message):
    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.payload = payload


class DictionaryEntryUpdateRequested(Message):
    def __init__(self, entry_id: str, payload: dict) -> None:
        super().__init__()
        self.entry_id = entry_id
        self.payload = payload


class DictionaryEntryDeleteRequested(Message):
    def __init__(self, entry_id: str) -> None:
        super().__init__()
        self.entry_id = entry_id


class DictionaryEntriesReorderRequested(Message):
    def __init__(self, entry_ids: list[str]) -> None:
        super().__init__()
        self.entry_ids = entry_ids


class DictionarySettingsSaveRequested(Message):
    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.payload = payload


class DictionarySettingsEdited(Message):
    """A settings field changed (dirty marker); no payload."""


class PersonasDictionaryDetailWidget(Vertical):
    """Entries + Settings tabs for one dictionary. Emits intents; owns no I/O."""

    DEFAULT_CSS = """
    PersonasDictionaryDetailWidget {
        height: 1fr;
        min-height: 0;
    }
    PersonasDictionaryDetailWidget #personas-dict-entries-table {
        height: 1fr;
        min-height: 4;
    }
    PersonasDictionaryDetailWidget .personas-dict-form-row {
        height: auto;
    }
    PersonasDictionaryDetailWidget #personas-dict-entry-replacement,
    PersonasDictionaryDetailWidget #personas-dict-description {
        height: 4;
    }
    PersonasDictionaryDetailWidget #personas-dict-entry-error,
    PersonasDictionaryDetailWidget #personas-dict-status {
        height: 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._entries: list[dict] = []
        self._loading = False  # suppress dirty events while programmatically filling fields

    # ----- compose -----

    def compose(self) -> ComposeResult:
        with TabbedContent(id="personas-dict-tabs"):
            with TabPane("Entries", id="personas-dict-tab-entries"):
                yield DataTable(id="personas-dict-entries-table", cursor_type="row")
                yield Static("", id="personas-dict-entry-error", markup=False)
                with Horizontal(classes="personas-dict-form-row"):
                    yield Input(placeholder="Pattern", id="personas-dict-entry-pattern")
                    yield Switch(value=False, id="personas-dict-entry-regex", tooltip="Regex pattern")
                    yield Input(placeholder="Probability %", id="personas-dict-entry-probability", value="100")
                    yield Input(placeholder="Group", id="personas-dict-entry-group")
                    yield Input(placeholder="Max repl.", id="personas-dict-entry-max-repl", value="1")
                yield TextArea(id="personas-dict-entry-replacement")
                with Horizontal(classes="personas-dict-form-row"):
                    yield Button("Add", id="personas-dict-entry-add", classes="console-action-secondary")
                    yield Button("Update", id="personas-dict-entry-update", classes="console-action-secondary")
                    yield Button("Delete", id="personas-dict-entry-delete", classes="console-action-secondary")
                    yield Button("Move up", id="personas-dict-entry-up", classes="console-action-secondary")
                    yield Button("Move down", id="personas-dict-entry-down", classes="console-action-secondary")
            with TabPane("Settings", id="personas-dict-tab-settings"):
                yield Input(placeholder="Name", id="personas-dict-name")
                yield TextArea(id="personas-dict-description")
                yield Select(
                    ((s, s) for s in STRATEGIES),
                    id="personas-dict-strategy",
                    value="sorted_evenly",
                    allow_blank=False,
                )
                yield Input(placeholder="Token budget", id="personas-dict-max-tokens", value="1000")
                with Horizontal(classes="personas-dict-form-row"):
                    yield Static("Enabled", markup=False)
                    yield Switch(value=True, id="personas-dict-enabled")
                yield Button("Save settings", id="personas-dict-settings-save", classes="console-action-secondary")
        yield Static("", id="personas-dict-status", markup=False)

    def on_mount(self) -> None:
        table = self.query_one("#personas-dict-entries-table", DataTable)
        table.add_columns("pattern", "replacement", "type", "prob %", "group")

    # ----- public API -----

    def load_dictionary(self, record: dict) -> None:
        """Fill settings + entries from a get_dictionary()/summary dict."""
        self._loading = True
        try:
            self.query_one("#personas-dict-name", Input).value = str(record.get("name") or "")
            self.query_one("#personas-dict-description", TextArea).text = str(record.get("description") or "")
            strategy = str(record.get("strategy") or "sorted_evenly")
            if strategy in STRATEGIES:
                self.query_one("#personas-dict-strategy", Select).value = strategy
            self.query_one("#personas-dict-max-tokens", Input).value = str(record.get("max_tokens") or 1000)
            self.query_one("#personas-dict-enabled", Switch).value = bool(
                record.get("enabled", record.get("is_active", True))
            )
        finally:
            self._loading = False
        self.update_entries(list(record.get("entries") or []))
        self.query_one("#personas-dict-status", Static).update("")

    def update_entries(self, entries: list[dict]) -> None:
        """Re-render the entries table from a fresh service response."""
        self._entries = list(entries)
        table = self.query_one("#personas-dict-entries-table", DataTable)
        table.clear()
        for entry in self._entries:
            probability = entry.get("probability")
            prob_pct = round(float(probability if probability is not None else 1.0) * 100)
            table.add_row(
                str(entry.get("pattern") or ""),
                str(entry.get("replacement") or ""),
                str(entry.get("type") or "literal"),
                str(prob_pct),
                str(entry.get("group") or ""),
                key=str(entry.get("id")),
            )
        self.query_one("#personas-dict-entry-error", Static).update("")

    def clear(self) -> None:
        self._entries = []
        self.query_one("#personas-dict-entries-table", DataTable).clear()

    @property
    def selected_entry_id(self) -> str | None:
        table = self.query_one("#personas-dict-entries-table", DataTable)
        if not self._entries or table.cursor_row is None or table.cursor_row < 0:
            return None
        try:
            row_key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key
            return str(row_key.value)
        except Exception:
            return None

    def entry_ids_in_order(self) -> list[str]:
        return [str(e.get("id")) for e in self._entries]

    def form_payload(self) -> dict | None:
        """API-named entry payload from the form; None + inline error when invalid."""
        error = self.query_one("#personas-dict-entry-error", Static)
        pattern = self.query_one("#personas-dict-entry-pattern", Input).value.strip()
        if not pattern:
            error.update("A pattern is required (an empty pattern can never fire).")
            return None
        raw_prob = self.query_one("#personas-dict-entry-probability", Input).value.strip() or "100"
        try:
            prob_pct = max(0, min(100, int(raw_prob)))
        except ValueError:
            error.update("Probability must be a whole number 0-100.")
            return None
        raw_max = self.query_one("#personas-dict-entry-max-repl", Input).value.strip() or "1"
        try:
            max_repl = max(1, int(raw_max))
        except ValueError:
            error.update("Max replacements must be a whole number.")
            return None
        error.update("")
        group = self.query_one("#personas-dict-entry-group", Input).value.strip()
        return {
            "pattern": pattern,
            "replacement": self.query_one("#personas-dict-entry-replacement", TextArea).text,
            "type": "regex" if self.query_one("#personas-dict-entry-regex", Switch).value else "literal",
            "probability": prob_pct / 100,
            "group": group or None,
            "max_replacements": max_repl,
        }

    def settings_payload(self) -> dict:
        raw_tokens = self.query_one("#personas-dict-max-tokens", Input).value.strip() or "1000"
        try:
            max_tokens = max(1, int(raw_tokens))
        except ValueError:
            max_tokens = 1000
        return {
            "name": self.query_one("#personas-dict-name", Input).value.strip(),
            "description": self.query_one("#personas-dict-description", TextArea).text,
            "strategy": str(self.query_one("#personas-dict-strategy", Select).value),
            "max_tokens": max_tokens,
            "enabled": bool(self.query_one("#personas-dict-enabled", Switch).value),
        }

    def set_status(self, message: str) -> None:
        self.query_one("#personas-dict-status", Static).update(message)

    # ----- events -----

    @on(DataTable.RowSelected, "#personas-dict-entries-table")
    def _row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        entry_id = str(event.row_key.value)
        entry = next((e for e in self._entries if str(e.get("id")) == entry_id), None)
        if entry is None:
            return
        self._loading = True
        try:
            self.query_one("#personas-dict-entry-pattern", Input).value = str(entry.get("pattern") or "")
            self.query_one("#personas-dict-entry-replacement", TextArea).text = str(entry.get("replacement") or "")
            self.query_one("#personas-dict-entry-regex", Switch).value = entry.get("type") == "regex"
            probability = entry.get("probability")
            self.query_one("#personas-dict-entry-probability", Input).value = str(
                round(float(probability if probability is not None else 1.0) * 100)
            )
            self.query_one("#personas-dict-entry-group", Input).value = str(entry.get("group") or "")
            self.query_one("#personas-dict-entry-max-repl", Input).value = str(entry.get("max_replacements") or 1)
        finally:
            self._loading = False

    @on(Button.Pressed, "#personas-dict-entry-add")
    def _add_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        payload = self.form_payload()
        if payload is not None:
            self.post_message(DictionaryEntryAddRequested(payload))

    @on(Button.Pressed, "#personas-dict-entry-update")
    def _update_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        entry_id = self.selected_entry_id
        if entry_id is None:
            self.query_one("#personas-dict-entry-error", Static).update("Select an entry row first.")
            return
        payload = self.form_payload()
        if payload is not None:
            self.post_message(DictionaryEntryUpdateRequested(entry_id, payload))

    @on(Button.Pressed, "#personas-dict-entry-delete")
    def _delete_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        entry_id = self.selected_entry_id
        if entry_id is None:
            self.query_one("#personas-dict-entry-error", Static).update("Select an entry row first.")
            return
        self.post_message(DictionaryEntryDeleteRequested(entry_id))

    def _post_reorder(self, offset: int) -> None:
        entry_id = self.selected_entry_id
        ids = self.entry_ids_in_order()
        if entry_id is None or entry_id not in ids:
            self.query_one("#personas-dict-entry-error", Static).update("Select an entry row first.")
            return
        index = ids.index(entry_id)
        target = index + offset
        if not 0 <= target < len(ids):
            return
        ids[index], ids[target] = ids[target], ids[index]
        self.post_message(DictionaryEntriesReorderRequested(ids))

    @on(Button.Pressed, "#personas-dict-entry-up")
    def _up_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._post_reorder(-1)

    @on(Button.Pressed, "#personas-dict-entry-down")
    def _down_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._post_reorder(1)

    @on(Button.Pressed, "#personas-dict-settings-save")
    def _settings_save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(DictionarySettingsSaveRequested(self.settings_payload()))

    @on(Input.Changed, "#personas-dict-name")
    @on(Input.Changed, "#personas-dict-max-tokens")
    @on(TextArea.Changed, "#personas-dict-description")
    @on(Select.Changed, "#personas-dict-strategy")
    @on(Switch.Changed, "#personas-dict-enabled")
    def _settings_edited(self, event: Message) -> None:
        if not self._loading:
            self.post_message(DictionarySettingsEdited())


__all__ = [
    "DictionaryEntriesReorderRequested",
    "DictionaryEntryAddRequested",
    "DictionaryEntryDeleteRequested",
    "DictionaryEntryUpdateRequested",
    "DictionarySettingsEdited",
    "DictionarySettingsSaveRequested",
    "PersonasDictionaryDetailWidget",
]
```

- [ ] **Step 4: Run the widget tests — expect PASS.** (If `Select.Changed` fires during compose despite `_loading`, move the `_loading = False` reset into `call_after_refresh`.)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py Tests/UI/test_personas_dictionaries.py
git commit -m "feat(personas): dictionary detail widget (Entries + Settings tabs)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Selection wiring — detail in the center, inspector, state

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: Task 3's widget + `load_dictionary`/`clear`; Task 2's `_dictionaries_cache`/`_render_dictionary_rows`; `_CENTER_VIEW_IDS` (`:121-128`), `_handle_entity_selected` (`:953`), `PersonasInspectorPane.show_selection(name=,kind=,authority=)`, `state.select_entity`, `library.mark_active_row(kind, item_id)`.
- Produces: `async _select_dictionary(entity_id: str, entity_name: str) -> None`; center id `"#personas-dictionary-detail"` registered; the composed widget (display-toggled by `_show_center`). Tasks 5–9 assume a selected dictionary means: `state.selected_entity_kind == "dictionary"`, detail loaded, `self._selected_dictionary_version: int | None` set.

- [ ] **Step 1: Failing tests** (append):

```python
class TestDictionarySelection:
    async def test_select_shows_detail_and_inspector(self, mock_app_instance, stub_characters, fake_dict_service):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_inspector_pane import PersonasInspectorPane
        from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_detail import (
            PersonasDictionaryDetailWidget,
        )
        from textual.widgets import DataTable

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            rows = screen.query_one("#personas-library-rows", ListView)
            rows.index = 0
            rows.action_select_cursor()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            detail = screen.query_one(PersonasDictionaryDetailWidget)
            assert detail.display is True
            assert screen.query_one("#personas-dict-name", Input).value == "Medical Abbrev"
            assert screen.query_one("#personas-dict-entries-table", DataTable).row_count == 2
            inspector = screen.query_one(PersonasInspectorPane)
            assert "Medical Abbrev" in str(
                inspector.query_one("#personas-selected-name", Static).renderable
            )
            assert "Dictionary" in str(
                inspector.query_one("#personas-selected-kind", Static).renderable
            )
            assert screen.state.selected_entity_kind == "dictionary"
            # Console actions blocked with the HONEST dictionary reason, not
            # "select a character or persona" while a dictionary IS selected.
            assert not screen._console_action_allowed()
            assert screen._console_action_block_reason() == "attach arrives in a later update"

    async def test_mode_switch_clears_detail(self, mock_app_instance, stub_characters, fake_dict_service):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_detail import (
            PersonasDictionaryDetailWidget,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            rows = screen.query_one("#personas-library-rows", ListView)
            rows.index = 0
            rows.action_select_cursor()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.click("#personas-mode-characters")
            await pilot.pause()
            assert screen.query_one(PersonasDictionaryDetailWidget).display is False
```

- [ ] **Step 2: Run — expect FAIL** (no `PersonasDictionaryDetailWidget` mounted on the screen).

- [ ] **Step 3: Implement**

In `personas_screen.py`:

(a) Import the widget + messages:

```python
from ...Widgets.Persona_Widgets.personas_dictionary_detail import (
    DictionaryEntriesReorderRequested,
    DictionaryEntryAddRequested,
    DictionaryEntryDeleteRequested,
    DictionaryEntryUpdateRequested,
    DictionarySettingsEdited,
    DictionarySettingsSaveRequested,
    PersonasDictionaryDetailWidget,
)
```

(b) `_CENTER_VIEW_IDS` (`:121`): add `"#personas-dictionary-detail",` before the placeholder entry.

(c) In `compose_content`, inside `#personas-detail-stack` right before the placeholder Static (`:439`):

```python
                        yield PersonasDictionaryDetailWidget(id="personas-dictionary-detail")
```

(d) `__init__`: add `self._selected_dictionary_version: int | None = None`.

(e) Extend `_handle_entity_selected` (`:963`, replacing the trailing comment):

```python
        elif message.entity_kind == "dictionary":
            await self._run_guarded(
                lambda: self._select_dictionary(message.entity_id, message.entity_name)
            )
        # Prompts and lore are wired in follow-up tasks.
```

(f) Add `_select_dictionary` (next to `_select_profile`):

```python
    async def _select_dictionary(self, entity_id: str, entity_name: str) -> None:
        """Load one dictionary into the center detail; inspector shows the selection."""
        service = self._dictionary_scope_service()
        if service is None:
            self._notify("Dictionaries service is not configured.", "error")
            return
        try:
            record = await service.get_dictionary(int(entity_id), mode="local")
        except Exception as exc:
            logger.opt(exception=True).warning(f"Could not load dictionary {entity_id}.")
            self._notify(f"Could not load dictionary: {exc}", "error")
            return
        self._edit_mode = "view"
        self.state.has_unsaved_changes = False
        raw_version = record.get("version")
        self._selected_dictionary_version = int(raw_version) if raw_version is not None else None
        self.state.select_entity(
            entity_kind="dictionary", entity_id=entity_id, entity_name=entity_name
        )
        detail = self.query_one(PersonasDictionaryDetailWidget)
        detail.load_dictionary(record)
        self._show_center("#personas-dictionary-detail")
        library = self.query_one(PersonasLibraryPane)
        library.mark_active_row("dictionary", entity_id)
        inspector = self.query_one(PersonasInspectorPane)
        inspector.show_selection(name=entity_name, kind="Dictionary", authority="Local")
        self._sync_inspector_console_actions()
        self._update_title()
        self._update_status_row()
```

(g) In `_apply_mode`, the dictionaries branch stays `self._show_center(None)` — `_show_center` already hides every registered center view, so leaving the mode hides the detail (the existing loop covers the new id).

(h) In `_console_action_block_reason` (`:1112`), insert an honest dictionary branch BEFORE the generic kind check:

```python
        if self.state.selected_entity_kind == "dictionary":
            return "attach arrives in a later update"
```

(so a selected dictionary never reads "select a character or persona" — the P0 honesty rule; Attachments are P1d).

- [ ] **Step 4: Run the selection tests — expect PASS.** Also rerun `TestDictionariesList` (compose change) — expect PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_dictionaries.py
git commit -m "feat(personas): dictionary selection loads the center detail + inspector

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Settings save + dirty guard + ConflictError

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: `DictionarySettingsSaveRequested(payload)` / `DictionarySettingsEdited` (Task 3); `_selected_dictionary_version` (Task 4); `state.has_unsaved_changes` + `_run_guarded` + `_update_title` (existing); `ConflictError` (imported at `:31` area from `DB.ChaChaNotes_DB`).
- Produces: `@on(DictionarySettingsSaveRequested)` / `@on(DictionarySettingsEdited)` handlers. Save = `update_dictionary(id, payload, mode="local", expected_version=…)` → reload detail + rows.

- [ ] **Step 1: Failing tests** (append):

```python
class TestDictionarySettings:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_save_persists_and_refreshes_row(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-name", Input).value = "Medical Terms"
            await pilot.click("#personas-dict-settings-save")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.records[1]["name"] == "Medical Terms"
            assert screen.state.has_unsaved_changes is False
            rows = screen.query_one("#personas-library-rows", ListView).children
            names = [str(r.query(Static).first().renderable) for r in rows]
            assert "Medical Terms" in names

    async def test_settings_edit_marks_dirty_and_guards_navigation(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-name", Input).value = "Half-renamed"
            await pilot.pause()
            assert screen.state.has_unsaved_changes is True
            assert "- unsaved" in str(screen.query_one("#personas-title", Static).renderable)

    async def test_conflict_surfaces_status_not_crash(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            fake_dict_service.records[1]["version"] = 99  # concurrent writer
            screen.query_one("#personas-dict-name", Input).value = "Medical Terms"
            await pilot.click("#personas-dict-settings-save")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            status = screen.query_one("#personas-dict-status", Static)
            assert "changed since it was loaded" in str(status.renderable)
```

- [ ] **Step 2: Run — expect FAIL** (no handlers; dirty flag untouched).

- [ ] **Step 3: Implement** (screen handlers, near the other `@on` blocks):

```python
    @on(DictionarySettingsEdited)
    def _handle_dictionary_settings_edited(self, message: DictionarySettingsEdited) -> None:
        message.stop()
        if self.state.selected_entity_kind != "dictionary":
            return
        if not self.state.has_unsaved_changes:
            self.state.has_unsaved_changes = True
            self._update_title()
            self._sync_inspector_console_actions()

    @on(DictionarySettingsSaveRequested)
    async def _handle_dictionary_settings_save(self, message: DictionarySettingsSaveRequested) -> None:
        message.stop()
        if self.state.selected_entity_kind != "dictionary" or not self.state.selected_entity_id:
            return
        detail = self.query_one(PersonasDictionaryDetailWidget)
        payload = dict(message.payload)
        if not payload.get("name"):
            detail.set_status("A name is required.")
            return
        service = self._dictionary_scope_service()
        if service is None:
            self._notify("Dictionaries service is not configured.", "error")
            return
        entity_id = self.state.selected_entity_id
        try:
            record = await service.update_dictionary(
                int(entity_id), payload, mode="local",
                expected_version=self._selected_dictionary_version,
            )
        except ConflictError:
            detail.set_status(
                "Save failed: the dictionary changed since it was loaded. Reselect and try again."
            )
            return
        except Exception as exc:
            logger.opt(exception=True).warning(f"Could not save dictionary {entity_id}.")
            detail.set_status(f"Save failed: {exc}")
            return
        raw_version = record.get("version")
        self._selected_dictionary_version = int(raw_version) if raw_version is not None else None
        self.state.has_unsaved_changes = False
        self.state.selected_entity_name = str(record.get("name") or "")
        detail.load_dictionary(record)
        detail.set_status("Saved.")
        self._update_title()
        await self._render_dictionary_rows(query=self.state.search_query)
        self.query_one(PersonasLibraryPane).mark_active_row("dictionary", entity_id)
        self._sync_inspector_console_actions()
```

Navigation guarding needs no new code: mode chips, row selection, New, and Delete already route through `_run_guarded`, which reads `state.has_unsaved_changes`.

- [ ] **Step 4: Run the settings tests — expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_dictionaries.py
git commit -m "feat(personas): dictionary settings save with optimistic-lock + dirty guard

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Entry add / update / delete (reload-after-mutation)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: Task 3 messages `DictionaryEntryAddRequested/UpdateRequested/DeleteRequested`; `detail.update_entries(entries)`; fake `add_entry/update_entry/delete_entry/list_entries`.
- Produces: `async _reload_selected_dictionary_entries() -> None` (calls `list_entries`, feeds `detail.update_entries`, refreshes the row meta + `_selected_dictionary_version` via `get_dictionary`). Tasks 7–9 reuse it.

- [ ] **Step 1: Failing tests** (append):

```python
class TestDictionaryEntries:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_add_entry_roundtrip(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import DataTable, TextArea

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-entry-pattern", Input).value = "ASAP"
            screen.query_one("#personas-dict-entry-replacement", TextArea).text = "as soon as possible"
            await pilot.click("#personas-dict-entry-add")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert len(fake_dict_service.records[1]["entries"]) == 3
            table = screen.query_one("#personas-dict-entries-table", DataTable)
            assert table.row_count == 3
            # Row meta refreshed too.
            rows = screen.query_one("#personas-library-rows", ListView).children
            metas = [str(s.renderable) for r in rows for s in r.query(".personas-library-row-meta").results()]
            assert "3 entries · on" in metas

    async def test_delete_entry_reindexes(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import DataTable

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            table = screen.query_one("#personas-dict-entries-table", DataTable)
            table.move_cursor(row=0)
            await pilot.click("#personas-dict-entry-delete")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert [e["pattern"] for e in fake_dict_service.records[1]["entries"]] == ["HR"]
            assert table.row_count == 1
            # Positional ids re-derived after reload: remaining row is index 0.
            detail = screen.query_one("#personas-dictionary-detail")
            assert detail.entry_ids_in_order() == ["local:chat_dictionary_entry:1:0"]

    async def test_update_entry_persists(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import DataTable, TextArea

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            table = screen.query_one("#personas-dict-entries-table", DataTable)
            table.move_cursor(row=1)
            screen.query_one("#personas-dict-entry-pattern", Input).value = "HRV"
            screen.query_one("#personas-dict-entry-replacement", TextArea).text = "heart rate variability"
            await pilot.click("#personas-dict-entry-update")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.records[1]["entries"][1]["pattern"] == "HRV"
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** (screen):

```python
    async def _reload_selected_dictionary_entries(self) -> None:
        """Re-fetch entries + version after a mutation (positional ids shift)."""
        entity_id = self.state.selected_entity_id
        service = self._dictionary_scope_service()
        if service is None or self.state.selected_entity_kind != "dictionary" or not entity_id:
            return
        detail = self.query_one(PersonasDictionaryDetailWidget)
        try:
            response = await service.list_entries(int(entity_id), mode="local")
            record = await service.get_dictionary(int(entity_id), mode="local")
        except Exception as exc:
            logger.opt(exception=True).warning(f"Could not reload dictionary {entity_id} entries.")
            detail.set_status(f"Reload failed: {exc}")
            return
        raw_version = record.get("version")
        self._selected_dictionary_version = int(raw_version) if raw_version is not None else None
        detail.update_entries(list(response.get("entries") or []))
        await self._render_dictionary_rows(query=self.state.search_query)
        self.query_one(PersonasLibraryPane).mark_active_row("dictionary", entity_id)

    async def _run_dictionary_entry_op(self, op: Callable[[Any], Awaitable[Any]], failure: str) -> None:
        """One guarded service mutation + the mandatory entries reload."""
        service = self._dictionary_scope_service()
        detail = self.query_one(PersonasDictionaryDetailWidget)
        if service is None or self.state.selected_entity_kind != "dictionary":
            return
        try:
            await op(service)
        except ConflictError:
            detail.set_status(
                "Change failed: the dictionary changed since it was loaded. Reselect and try again."
            )
            return
        except Exception as exc:
            logger.opt(exception=True).warning(failure)
            detail.set_status(f"{failure}: {exc}")
            return
        await self._reload_selected_dictionary_entries()
        detail.set_status("")

    @on(DictionaryEntryAddRequested)
    async def _handle_dictionary_entry_add(self, message: DictionaryEntryAddRequested) -> None:
        message.stop()
        entity_id = self.state.selected_entity_id
        if not entity_id:
            return
        await self._run_dictionary_entry_op(
            lambda service: service.add_entry(int(entity_id), message.payload, mode="local"),
            "Could not add the entry",
        )

    @on(DictionaryEntryUpdateRequested)
    async def _handle_dictionary_entry_update(self, message: DictionaryEntryUpdateRequested) -> None:
        message.stop()
        await self._run_dictionary_entry_op(
            lambda service: service.update_entry(message.entry_id, message.payload, mode="local"),
            "Could not update the entry",
        )

    @on(DictionaryEntryDeleteRequested)
    async def _handle_dictionary_entry_delete(self, message: DictionaryEntryDeleteRequested) -> None:
        message.stop()
        await self._run_dictionary_entry_op(
            lambda service: service.delete_entry(message.entry_id, mode="local"),
            "Could not delete the entry",
        )
```

(`Callable`/`Awaitable` are already imported at the top of the screen module.)

- [ ] **Step 4: Run the entry tests — expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_dictionaries.py
git commit -m "feat(personas): dictionary entry CRUD with reload-after-mutation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Reorder (move-up/down with full-order ids)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: `DictionaryEntriesReorderRequested(entry_ids)` (Task 3 posts the FULL swapped order); `_run_dictionary_entry_op` (Task 6); fake `reorder_entries` (move-to-front mirror).
- Produces: the reorder handler; the pinned semantics test.

- [ ] **Step 1: Failing test** (append to `TestDictionaryEntries`):

```python
    async def test_move_down_sends_full_order_and_reorders(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import DataTable

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            table = screen.query_one("#personas-dict-entries-table", DataTable)
            table.move_cursor(row=0)
            await pilot.click("#personas-dict-entry-down")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # Pins the move-to-front backend semantics: the FULL id list, in
            # the desired final order, was sent - selected+remainder then
            # yields exactly that order.
            reorders = [c for c in fake_dict_service.calls if c[0] == "reorder"]
            assert reorders and reorders[-1][2] == [
                "local:chat_dictionary_entry:1:1",
                "local:chat_dictionary_entry:1:0",
            ]
            assert [e["pattern"] for e in fake_dict_service.records[1]["entries"]] == ["HR", "BP"]
            assert str(table.get_cell_at((0, 0))) == "HR"
```

- [ ] **Step 2: Run — expect FAIL** (message unhandled).

- [ ] **Step 3: Implement** (screen):

```python
    @on(DictionaryEntriesReorderRequested)
    async def _handle_dictionary_entries_reorder(self, message: DictionaryEntriesReorderRequested) -> None:
        message.stop()
        entity_id = self.state.selected_entity_id
        if not entity_id:
            return
        await self._run_dictionary_entry_op(
            lambda service: service.reorder_entries(
                int(entity_id), {"entry_ids": list(message.entry_ids)}, mode="local"
            ),
            "Could not reorder entries",
        )
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_dictionaries.py
git commit -m "feat(personas): entry reorder passes the full desired order (move-to-front backend)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: New + Duplicate (disambiguated names, strategy follow-up)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`, `tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py`, `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: `_handle_action_requested` create branch (`:1252`); `_select_dictionary` (Task 4); `_dictionaries_cache` (Task 2); fake `create_dictionary` (raises `ConflictError` on duplicate names, ignores strategy) + `update_dictionary`.
- Produces: `PersonaAction` Literal gains `"duplicate"`; pane gains a mode-gated `#personas-library-duplicate` button posting `PersonaActionRequested(action="duplicate")`; screen `_begin_create_dictionary()` / `_duplicate_selected_dictionary()`; `_unique_dictionary_name(base: str) -> str`.

- [ ] **Step 1: Failing tests** (append):

```python
class TestDictionaryNewDuplicate:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_new_creates_disambiguated_and_selects(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await pilot.click("#personas-library-new")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            names = [r["name"] for r in fake_dict_service.records.values()]
            assert "Untitled dictionary" in names
            assert screen.state.selected_entity_name == "Untitled dictionary"
            # Second New must disambiguate against the first.
            await pilot.click("#personas-library-new")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            names = [r["name"] for r in fake_dict_service.records.values()]
            assert "Untitled dictionary 2" in names

    async def test_duplicate_copies_entries_and_strategy(self, mock_app_instance, stub_characters, fake_dict_service):
        fake_dict_service.records[1]["strategy"] = "character_lore_first"
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            await pilot.click("#personas-library-duplicate")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            copy_rec = next(r for r in fake_dict_service.records.values() if r["name"] == "Medical Abbrev (copy)")
            assert [e["pattern"] for e in copy_rec["entries"]] == ["BP", "HR"]
            # create_dictionary ignores strategy - the follow-up update must set it.
            assert copy_rec["strategy"] == "character_lore_first"

    async def test_duplicate_button_hidden_outside_dictionaries(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            dup = screen.query_one("#personas-library-duplicate", Button)
            assert dup.display is False  # characters mode
            await pilot.click("#personas-mode-dictionaries")
            await pilot.pause()
            assert dup.display is True
```

- [ ] **Step 2: Run — expect FAIL** (`#personas-library-duplicate` missing).

- [ ] **Step 3: Implement**

(a) `personas_messages.py`: add `"duplicate",` to the `PersonaAction` Literal (after `"export"`).

(b) `personas_library_pane.py` — in `compose` after the Import button:

```python
            yield Button(
                "Duplicate",
                id="personas-library-duplicate",
                tooltip="Duplicate the selected dictionary.",
                classes="console-action-secondary",
            )
```

In `set_mode`:

```python
    def set_mode(self, mode: str) -> None:
        """Show Import only where it applies (Characters mode); Duplicate is dictionaries-only."""
        self._import_visible = mode == "characters"
        self.query_one("#personas-library-import", Button).display = self._import_visible
        self.query_one("#personas-library-duplicate", Button).display = mode == "dictionaries"
```

And the handler (below `_import_pressed`):

```python
    @on(Button.Pressed, "#personas-library-duplicate")
    def _duplicate_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaActionRequested(action="duplicate"))
```

Also set `display = False` on the button at compose time for the default characters mode: after `yield`-composition Textual applies `set_mode` only on explicit calls, so in the pane's `on_mount` add `self.query_one("#personas-library-duplicate", Button).display = False` (or call `set_mode("characters")` — pick the former, minimal).

(c) Screen — extend `_handle_action_requested`:

```python
            elif self.state.active_mode == "dictionaries":
                await self._run_guarded(self._begin_create_dictionary)
```

(inside the `create` branch, after the personas elif) and add:

```python
        elif message.action == "duplicate":
            if self.state.active_mode == "dictionaries":
                await self._run_guarded(self._duplicate_selected_dictionary)
```

(d) Screen — the flows:

```python
    def _unique_dictionary_name(self, base: str) -> str:
        """Disambiguate against the loaded list (name column is UNIQUE)."""
        existing = {str(r.get("name") or "") for r in self._dictionaries_cache}
        if base not in existing:
            return base
        suffix = 2
        while f"{base} {suffix}" in existing:
            suffix += 1
        return f"{base} {suffix}"

    async def _begin_create_dictionary(self) -> None:
        service = self._dictionary_scope_service()
        if service is None:
            self._notify("Dictionaries service is not configured.", "error")
            return
        name = self._unique_dictionary_name("Untitled dictionary")
        try:
            record = await service.create_dictionary({"name": name}, mode="local")
        except ConflictError:
            self._notify("A dictionary with that name already exists.", "error")
            return
        except Exception as exc:
            logger.opt(exception=True).warning("Could not create a dictionary.")
            self._notify(f"Create failed: {exc}", "error")
            return
        await self._render_dictionary_rows(query="")
        await self._select_dictionary(str(record.get("id")), str(record.get("name") or name))
        # Land the user in Settings to rename immediately.
        try:
            self.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-settings"
            self.query_one("#personas-dict-name", Input).focus()
        except QueryError:
            pass

    async def _duplicate_selected_dictionary(self) -> None:
        service = self._dictionary_scope_service()
        entity_id = self.state.selected_entity_id
        if service is None or self.state.selected_entity_kind != "dictionary" or not entity_id:
            self._notify("Select a dictionary to duplicate.", "warning")
            return
        try:
            source = await service.get_dictionary(int(entity_id), mode="local")
        except Exception as exc:
            logger.opt(exception=True).warning(f"Could not load dictionary {entity_id} to duplicate.")
            self._notify(f"Duplicate failed: {exc}", "error")
            return
        base = f"{source.get('name') or 'Dictionary'} (copy)"
        existing = {str(r.get("name") or "") for r in self._dictionaries_cache}
        name = base
        suffix = 2
        while name in existing:
            name = f"{source.get('name') or 'Dictionary'} (copy {suffix})"
            suffix += 1
        payload = {
            "name": name,
            "description": source.get("description") or "",
            "max_tokens": source.get("max_tokens") or 1000,
            "enabled": bool(source.get("enabled", source.get("is_active", True))),
            "entries": [
                {
                    "pattern": e.get("pattern"),
                    "replacement": e.get("replacement"),
                    "probability": e.get("probability"),
                    "group": e.get("group"),
                    "timed_effects": e.get("timed_effects"),
                    "max_replacements": e.get("max_replacements"),
                    "type": e.get("type"),
                }
                for e in source.get("entries") or []
            ],
        }
        try:
            record = await service.create_dictionary(payload, mode="local")
            # create_dictionary ignores strategy (column default); set it after.
            source_strategy = str(source.get("strategy") or "sorted_evenly")
            if source_strategy != "sorted_evenly":
                record = await service.update_dictionary(
                    int(record["id"]), {"strategy": source_strategy}, mode="local"
                )
        except ConflictError:
            self._notify("A dictionary with that name already exists.", "error")
            return
        except Exception as exc:
            logger.opt(exception=True).warning("Could not duplicate the dictionary.")
            self._notify(f"Duplicate failed: {exc}", "error")
            return
        await self._render_dictionary_rows(query="")
        await self._select_dictionary(str(record.get("id")), str(record.get("name") or name))
```

Add `TabbedContent` to the screen's textual imports.

- [ ] **Step 4: Run the New/Duplicate tests — expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py Tests/UI/test_personas_dictionaries.py
git commit -m "feat(personas): New + Duplicate dictionaries with unique names and strategy carry-over

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Space enable-toggle + Delete via the existing confirm flow

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`, `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py`, `tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: `_begin_delete_selection` kind gate (`:1758`) + `_delete_entity` (`:1810`); `ConfirmationDialog` via `_confirm_delete` (unchanged); pane `_row_lookup` + ListView highlight; `PersonaAction` Literal (Task 8 already touched the file).
- Produces: `PersonaAction` gains `"toggle_enabled"`; pane `space` binding posting `PersonaActionRequested(action="toggle_enabled", entity_kind=..., entity_id=...)` for the highlighted row; screen toggle handler + dictionary branch in the delete flow.

- [ ] **Step 1: Failing tests** (append):

```python
class TestDictionaryToggleDelete:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_space_toggles_enabled_and_meta(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            rows = screen.query_one("#personas-library-rows", ListView)
            rows.focus()
            rows.index = 0
            await pilot.press("space")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.records[1]["enabled"] is False
            metas = [
                str(s.renderable)
                for r in screen.query_one("#personas-library-rows", ListView).children
                for s in r.query(".personas-library-row-meta").results()
            ]
            assert "2 entries · off" in metas

    async def test_delete_confirms_then_removes(self, mock_app_instance, stub_characters, fake_dict_service, monkeypatch):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_detail import (
            PersonasDictionaryDetailWidget,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)

            async def _fake_confirm(name):
                return True

            monkeypatch.setattr(screen, "_confirm_delete", _fake_confirm)
            await pilot.click("#personas-delete")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert 1 not in fake_dict_service.records
            assert screen.query_one(PersonasDictionaryDetailWidget).display is False
            assert screen.state.selected_entity_id is None

    async def test_delete_cancelled_keeps_record(self, mock_app_instance, stub_characters, fake_dict_service, monkeypatch):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)

            async def _fake_confirm(name):
                return False

            monkeypatch.setattr(screen, "_confirm_delete", _fake_confirm)
            await pilot.click("#personas-delete")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert 1 in fake_dict_service.records
```

- [ ] **Step 2: Run — expect FAIL** (space unhandled; delete flow warns "Select a saved item").

- [ ] **Step 3: Implement**

(a) `personas_messages.py`: add `"toggle_enabled",` to `PersonaAction`.

(b) `personas_library_pane.py`: give the pane a binding + handler —

```python
    BINDINGS = [
        ("space", "toggle_highlighted", "Toggle on/off"),
    ]

    def action_toggle_highlighted(self) -> None:
        """Space on a highlighted dictionary row requests an enable-toggle."""
        list_view = self.query_one("#personas-library-rows", ListView)
        index = list_view.index
        if index is None or not 0 <= index < len(list_view.children):
            return
        row = self._row_lookup.get(str(list_view.children[index].id or ""))
        if row is None or row.kind != "dictionary":
            return
        self.post_message(
            PersonaActionRequested(
                action="toggle_enabled", entity_kind=row.kind, entity_id=row.item_id
            )
        )
```

(Space is a printable key: with focus in the search Input it types a space; the binding only acts from list focus — same trade-off as the screen's `[`/`]` bindings.)

(c) Screen — extend `_handle_action_requested`:

```python
        elif message.action == "toggle_enabled":
            if self.state.active_mode == "dictionaries" and message.entity_id:
                await self._toggle_dictionary_enabled(message.entity_id)
```

and add:

```python
    async def _toggle_dictionary_enabled(self, entity_id: str) -> None:
        """Flip a dictionary's enabled flag from the rail (space on the row)."""
        service = self._dictionary_scope_service()
        if service is None:
            return
        record = next(
            (r for r in self._dictionaries_cache if str(r.get("id")) == str(entity_id)), None
        )
        if record is None:
            return
        target = not bool(record.get("enabled", record.get("is_active", True)))
        try:
            updated = await service.update_dictionary(
                int(entity_id), {"enabled": target}, mode="local"
            )
        except Exception as exc:
            logger.opt(exception=True).warning(f"Could not toggle dictionary {entity_id}.")
            self._notify(f"Toggle failed: {exc}", "error")
            return
        if str(self.state.selected_entity_id) == str(entity_id):
            raw_version = updated.get("version")
            self._selected_dictionary_version = int(raw_version) if raw_version is not None else None
            self.query_one(PersonasDictionaryDetailWidget).load_dictionary(updated)
        await self._render_dictionary_rows(query=self.state.search_query)
        if self.state.selected_entity_id:
            self.query_one(PersonasLibraryPane).mark_active_row(
                "dictionary", self.state.selected_entity_id
            )
```

(d) Screen — delete flow. In `_begin_delete_selection` change the gate (`:1758`) to `if not entity_id or kind not in ("character", "persona_profile", "dictionary"):` and extend the version-fetch chain:

```python
        elif kind == "dictionary":
            record = next(
                (r for r in self._dictionaries_cache if str(r.get("id")) == entity_id), None
            )
            if record is None:
                self._notify("Dictionary data is not loaded yet.", "warning")
                return
            raw_version = record.get("version")
            version = int(raw_version) if raw_version is not None else None
```

(as a new branch between the `character` branch and the profile `else`, making the profile branch `elif kind == "persona_profile":`). In `_delete_entity`, add a dictionary branch (before or after the character branch, same shape):

```python
        if kind == "dictionary":
            service = self._dictionary_scope_service()
            if service is None:
                self._notify("Dictionaries service is not configured.", "error")
                return
            try:
                await service.delete_dictionary(
                    int(entity_id), mode="local", expected_version=version
                )
            except ConflictError:
                self._notify(conflict_copy.format(noun="dictionary"), "error")
                return
            except Exception as exc:
                logger.opt(exception=True).error(f"Error deleting dictionary {entity_id}: {exc}")
                self._notify(f"Delete failed: {exc}", "error")
                return
            self.state.clear_selection()
            self.state.has_unsaved_changes = False
            self._selected_dictionary_version = None
            self.query_one(PersonasDictionaryDetailWidget).clear()
            self._show_center(None)
            await self.query_one(PersonasInspectorPane).clear_selection()
            await self._render_dictionary_rows(query=self.state.search_query)
            self._update_title()
            self._update_status_row()
            return
```

Also check the inspector's Delete-enable gate: in `personas_inspector_pane.py` `_apply_action_state`, Delete is enabled on `self._has_selection` — kind-agnostic, so dictionary selections enable it with no change. Verify this while implementing; if it filters by kind, extend it.

- [ ] **Step 4: Run the toggle/delete tests — expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py Tests/UI/test_personas_dictionaries.py
git commit -m "feat(personas): rail space-toggle + confirmed dictionary delete

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Try-it substitution preview (word-diff)

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_tryit.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: `process_text({"text","dictionary_id","token_budget"}, mode="local") -> {"text","processed_text",...}`; selection state (Task 4); `PersonasPreviewPane` (to hide in dictionaries mode).
- Produces: `PersonasDictionaryTryItWidget(Vertical)` with `set_ready(ready: bool, hint: str = "") -> None`, `render_result(original: str, processed: str) -> None`, `show_error(message: str) -> None`, `sample_text() -> str`; message `DictionaryTryItRunRequested(text: str)`; module function `word_diff(original: str, processed: str) -> tuple[Text, Text]` (Rich `Text` pair: original with removed spans styled `"strike dim"`, processed with added spans styled `"bold underline"`). DOM ids: `#personas-dict-tryit` (the widget instance id), `#personas-dict-tryit-sample` (TextArea), `#personas-dict-tryit-run` (Button), `#personas-dict-tryit-original`, `#personas-dict-tryit-processed`, `#personas-dict-tryit-status` (Statics).

- [ ] **Step 1: Failing tests** (append):

```python
class TestDictionaryTryIt:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    def test_word_diff_marks_changes(self):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_tryit import word_diff

        original, processed = word_diff("check BP now", "check blood pressure now")
        assert "BP" in original.plain and "blood pressure" in processed.plain
        assert any(span.style == "strike dim" for span in original.spans)
        assert any(span.style == "bold underline" for span in processed.spans)

    def test_word_diff_no_changes_has_no_spans(self):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_tryit import word_diff

        original, processed = word_diff("same text", "same text")
        assert not original.spans and not processed.spans

    async def test_run_renders_diff(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TextArea
        from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_tryit import (
            PersonasDictionaryTryItWidget,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            tryit = screen.query_one(PersonasDictionaryTryItWidget)
            assert tryit.display is True
            run = screen.query_one("#personas-dict-tryit-run", Button)
            assert run.disabled is True  # nothing selected yet
            await self._select_first(pilot, screen)
            assert run.disabled is False
            screen.query_one("#personas-dict-tryit-sample", TextArea).text = "check BP now"
            await pilot.click("#personas-dict-tryit-run")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            processed = screen.query_one("#personas-dict-tryit-processed", Static)
            assert "blood pressure" in str(processed.renderable)
            # token_budget rode along from the dict's max_tokens.
            run_calls = [c for c in fake_dict_service.calls if c[0] == "process"]
            assert run_calls

    async def test_no_differences_state(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TextArea

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-tryit-sample", TextArea).text = "no matches here"
            await pilot.click("#personas-dict-tryit-run")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            status = screen.query_one("#personas-dict-tryit-status", Static)
            assert "No differences" in str(status.renderable)

    async def test_preview_pane_swapped_by_mode(self, mock_app_instance, stub_characters, fake_dict_service):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import PersonasPreviewPane
        from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_tryit import (
            PersonasDictionaryTryItWidget,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            assert screen.query_one(PersonasDictionaryTryItWidget).display is False
            assert screen.query_one(PersonasPreviewPane).display is True
            await pilot.click("#personas-mode-dictionaries")
            await pilot.pause()
            assert screen.query_one(PersonasDictionaryTryItWidget).display is True
            assert screen.query_one(PersonasPreviewPane).display is False
            await pilot.click("#personas-mode-characters")
            await pilot.pause()
            assert screen.query_one(PersonasDictionaryTryItWidget).display is False
            assert screen.query_one(PersonasPreviewPane).display is True
```

- [ ] **Step 2: Run — expect FAIL** (module missing).

- [ ] **Step 3: Implement the widget**

Create `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_tryit.py`:

```python
"""Try-it substitution preview for the Roleplay Dictionaries mode."""

from __future__ import annotations

import difflib
import re

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Static, TextArea

_TOKEN_RE = re.compile(r"(\s+)")


def word_diff(original: str, processed: str) -> tuple[Text, Text]:
    """Word-level diff: removals styled 'strike dim' in the original, additions
    'bold underline' in the processed text. Theme-safe (no colors)."""
    left = Text()
    right = Text()
    a = _TOKEN_RE.split(original)
    b = _TOKEN_RE.split(processed)
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        a_chunk = "".join(a[a0:a1])
        b_chunk = "".join(b[b0:b1])
        if op == "equal":
            left.append(a_chunk)
            right.append(b_chunk)
        elif op == "delete":
            left.append(a_chunk, style="strike dim")
        elif op == "insert":
            right.append(b_chunk, style="bold underline")
        else:  # replace
            left.append(a_chunk, style="strike dim")
            right.append(b_chunk, style="bold underline")
    return left, right


class DictionaryTryItRunRequested(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class PersonasDictionaryTryItWidget(Vertical):
    """Sample text in, highlighted before/after diff out. Owns no I/O."""

    BINDINGS = [
        Binding("ctrl+enter", "run_preview", "Run preview", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    PersonasDictionaryTryItWidget {
        height: auto;
        max-height: 60%;
    }
    PersonasDictionaryTryItWidget #personas-dict-tryit-sample {
        height: 4;
    }
    PersonasDictionaryTryItWidget #personas-dict-tryit-original,
    PersonasDictionaryTryItWidget #personas-dict-tryit-processed {
        height: auto;
        max-height: 8;
    }
    PersonasDictionaryTryItWidget #personas-dict-tryit-status {
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Try it — substitution preview", markup=False)
        yield TextArea(id="personas-dict-tryit-sample")
        yield Button(
            "Run preview",
            id="personas-dict-tryit-run",
            classes="console-action-secondary",
            disabled=True,
            tooltip="Run the selected dictionary against the sample text (Ctrl+Enter).",
        )
        yield Static("", id="personas-dict-tryit-status", markup=False)
        yield Static("", id="personas-dict-tryit-original")
        yield Static("", id="personas-dict-tryit-processed")

    def set_ready(self, ready: bool, hint: str = "") -> None:
        self.query_one("#personas-dict-tryit-run", Button).disabled = not ready
        if hint:
            self.query_one("#personas-dict-tryit-status", Static).update(hint)

    def sample_text(self) -> str:
        return self.query_one("#personas-dict-tryit-sample", TextArea).text

    def render_result(self, original: str, processed: str) -> None:
        left, right = word_diff(original, processed)
        status = self.query_one("#personas-dict-tryit-status", Static)
        if original == processed:
            status.update("No differences - no entry changed the sample.")
        else:
            status.update("Changed spans highlighted below.")
        self.query_one("#personas-dict-tryit-original", Static).update(left)
        self.query_one("#personas-dict-tryit-processed", Static).update(right)

    def show_error(self, message: str) -> None:
        self.query_one("#personas-dict-tryit-status", Static).update(message)

    def _post_run(self) -> None:
        text = self.sample_text()
        if not text.strip():
            self.show_error("Type some sample text first.")
            return
        self.post_message(DictionaryTryItRunRequested(text))

    def action_run_preview(self) -> None:
        if not self.query_one("#personas-dict-tryit-run", Button).disabled:
            self._post_run()

    @on(Button.Pressed, "#personas-dict-tryit-run")
    def _run_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._post_run()


__all__ = [
    "DictionaryTryItRunRequested",
    "PersonasDictionaryTryItWidget",
    "word_diff",
]
```

(If the priority `ctrl+enter` binding is still consumed by `TextArea` in practice, drop the binding and keep the Run button — the spec's accepted fallback. The screen's own `ctrl+enter → personas_attach` binding stays; this widget-scoped binding shadows it only while focus is inside Try-it, which is the intended precedence.)

- [ ] **Step 4: Wire the screen**

(a) Imports:

```python
from ...Widgets.Persona_Widgets.personas_dictionary_tryit import (
    DictionaryTryItRunRequested,
    PersonasDictionaryTryItWidget,
)
```

(b) In `compose_content`, right after `yield PersonasPreviewPane(id="personas-preview-pane")` (`:440`):

```python
                    tryit = PersonasDictionaryTryItWidget(id="personas-dict-tryit")
                    tryit.display = False
                    yield tryit
```

(c) In `_apply_mode`, make the pane swap explicit. In the dictionaries branch:

```python
        elif mode == "dictionaries":
            await self._render_dictionary_rows()
            self._show_center(None)
```

becomes part of a swap done for ALL branches — add immediately after the `library.set_mode(mode)` line:

```python
        is_dictionaries = mode == "dictionaries"
        self.query_one(PersonasPreviewPane).display = not is_dictionaries
        tryit = self.query_one(PersonasDictionaryTryItWidget)
        tryit.display = is_dictionaries
        if is_dictionaries:
            tryit.set_ready(False, "Select a dictionary to preview substitutions.")
```

(`PersonasPreviewPane` is already imported at `:63`.)

(d) In `_select_dictionary` (Task 4), after `inspector.show_selection(...)` add:

```python
        self.query_one(PersonasDictionaryTryItWidget).set_ready(
            True, "Run the preview to see what this dictionary changes."
        )
```

And in the Task 9 delete-branch cleanup (after `self._show_center(None)`), add `self.query_one(PersonasDictionaryTryItWidget).set_ready(False, "Select a dictionary to preview substitutions.")`.

(e) The run handler:

```python
    @on(DictionaryTryItRunRequested)
    async def _handle_dictionary_tryit_run(self, message: DictionaryTryItRunRequested) -> None:
        message.stop()
        tryit = self.query_one(PersonasDictionaryTryItWidget)
        entity_id = self.state.selected_entity_id
        service = self._dictionary_scope_service()
        if service is None or self.state.selected_entity_kind != "dictionary" or not entity_id:
            tryit.show_error("Select a dictionary first.")
            return
        record = next(
            (r for r in self._dictionaries_cache if str(r.get("id")) == str(entity_id)), None
        )
        token_budget = int((record or {}).get("max_tokens") or 1000)
        try:
            response = await service.process_text(
                {"text": message.text, "dictionary_id": int(entity_id), "token_budget": token_budget},
                mode="local",
            )
        except Exception as exc:
            logger.opt(exception=True).warning(f"Try-it preview failed for dictionary {entity_id}.")
            tryit.show_error(f"Couldn't run the preview: {exc}")
            return
        tryit.render_result(
            str(response.get("text") or message.text),
            str(response.get("processed_text") or ""),
        )
```

- [ ] **Step 5: Run the Try-it tests — expect PASS.**

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_tryit.py tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_dictionaries.py
git commit -m "feat(personas): Try-it substitution preview with highlighted word-diff

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Full-suite gate + spec status

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-13-roleplay-p1a-dictionaries-foundation-design.md` (status line only)
- Test: whole personas surface

- [ ] **Step 1: Run the full gate**

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  .venv/bin/python -m pytest \
  Tests/UI/test_personas_dictionaries.py Tests/UI/test_personas_workbench.py \
  Tests/UI/test_personas_workbench_foundation.py \
  "Tests/UI/test_unified_shell_phase6_first_time_replay.py" \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

Expected: all pass EXCEPT the pre-existing `nav-library` failure in the phase6 file (`test_first_time_shell_replay_exposes_home_console_and_orientation_paths`, asserts "Import/Export Sources" — unrelated, pre-dates this branch). The phase6 `nav-personas` test must pass.

- [ ] **Step 2: Import smoke**

```
... .venv/bin/python -c "import tldw_chatbook.UI.Screens.personas_screen; import tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_detail; import tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_tryit; print('import ok')"
```

- [ ] **Step 3: Update the spec status line** — `**Status:** Design approved (brainstorm), pending spec review.` → `**Status:** Implemented (P1a).`

- [ ] **Step 4: Commit**

```bash
git add Docs/superpowers/specs/2026-07-13-roleplay-p1a-dictionaries-foundation-design.md
git commit -m "docs(roleplay): mark P1a dictionaries-foundation spec implemented

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
