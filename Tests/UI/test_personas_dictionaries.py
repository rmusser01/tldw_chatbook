"""Mounted tests for the Roleplay Dictionaries mode (P1a)."""

import copy
from pathlib import Path
from typing import Any

import pytest
from textual.app import App
from textual.widgets import Button, DataTable, Input, ListItem, ListView, Select, Static, Switch, TextArea

from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_detail import (
    DictionaryEntryAddRequested,
    DictionaryEntryDeleteRequested,
    DictionaryEntriesReorderRequested,
    DictionaryEntryUpdateRequested,
    DictionarySettingsSaveRequested,
    PersonasDictionaryDetailWidget,
)
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
        "enabled": bool(entry.get("enabled", True)),
        "case_sensitive": bool(entry.get("case_sensitive", False)),
        "priority": int(entry.get("priority", 0)),
        "source": "local",
    }


class FakeDictScopeService:
    """In-memory stand-in mirroring the real local shapes and quirks."""

    emit_diagnostics = True

    def __init__(self, records: list[dict] | None = None) -> None:
        self.records: dict[int, dict] = {}
        self.calls: list[tuple] = []
        self.history: dict[int, list[dict]] = {}
        self.conversations: dict[str, dict] = {}
        for record in records or []:
            self.records[int(record["id"])] = copy.deepcopy(record)
        self._next_id = max(self.records, default=0) + 1

    def _summary(self, record: dict, *, for_list: bool = False) -> dict:
        out = copy.deepcopy(record)
        true_entries = record.get("entries") or []
        if for_list:
            # Mirrors the real list_chat_dictionaries(): entries stay empty
            # (list stays O(1) per row) but entry_count reports the truth,
            # same as list_chat_dictionaries()'s json_array_length column.
            out["entries"] = []
        else:
            out["entries"] = [
                _entry_response(int(record["id"]), i, e)
                for i, e in enumerate(true_entries)
            ]
        out["entry_count"] = len(true_entries)
        out["is_active"] = bool(record.get("enabled", True))
        return out

    async def list_dictionaries(self, mode: str = "local", **kwargs: Any) -> dict:
        self.calls.append(("list", kwargs))
        assert mode == "local"
        # Real local list drops disabled records unless include_inactive=True.
        include_inactive = bool(kwargs.get("include_inactive", False))
        items = [
            self._summary(r, for_list=True)
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
                    "enabled": bool(e.get("enabled", True)),
                    "case_sensitive": bool(e.get("case_sensitive", False)),
                    "priority": int(e.get("priority", 0)),
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
            record["entries"] = [
                {
                    **dict(e),
                    "enabled": bool(e.get("enabled", True)),
                    "case_sensitive": bool(e.get("case_sensitive", False)),
                    "priority": int(e.get("priority", 0)),
                }
                for e in payload["entries"] or []
            ]
        record["version"] += 1
        self._record_version(record, "update")
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
        entry = dict(request_data)
        entry.setdefault("enabled", True)
        entry.setdefault("case_sensitive", False)
        entry.setdefault("priority", 0)
        record["entries"].append(entry)
        record["version"] += 1
        self.calls.append(("add_entry", int(dictionary_id)))
        return _entry_response(int(dictionary_id), len(record["entries"]) - 1, entry)

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
        # Ensure the three new fields have their defaults if not set
        record["entries"][index].setdefault("enabled", True)
        record["entries"][index].setdefault("case_sensitive", False)
        record["entries"][index].setdefault("priority", 0)
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

    async def get_statistics(self, dictionary_id: int, mode: str = "local") -> dict:
        record = self.records[int(dictionary_id)]
        return {
            "dictionary_id": int(dictionary_id),
            "entry_count": len(record["entries"]),
            "enabled": bool(record.get("enabled", True)),
            "source": "local",
        }

    def _record_version(self, record: dict, action: str) -> None:
        history = self.history.setdefault(int(record["id"]), [])
        entry = {
            "dictionary_id": int(record["id"]),
            "revision": int(record["version"]),
            "action": action,
            "name": record["name"],
            "created_at": "2026-07-15T00:00:00Z",
            "snapshot": copy.deepcopy(
                {k: record[k] for k in ("id", "name", "description", "strategy",
                                         "max_tokens", "enabled", "version", "entries")}
            ),
        }
        history[:] = [h for h in history if h["revision"] != entry["revision"]]
        history.append(entry)

    async def list_versions(self, dictionary_id: int, mode: str = "local", **kwargs: Any) -> dict:
        self._ensure_baseline(int(dictionary_id))
        versions = sorted(self.history.get(int(dictionary_id), []), key=lambda h: -h["revision"])
        return {
            "dictionary_id": int(dictionary_id),
            "versions": [{k: h[k] for k in ("dictionary_id", "revision", "action", "name", "created_at")}
                          for h in versions],
            "total": len(versions), "limit": 20, "offset": 0, "source": "local",
        }

    async def get_version(self, dictionary_id: int, revision: int, mode: str = "local") -> dict:
        self._ensure_baseline(int(dictionary_id))
        for h in self.history.get(int(dictionary_id), []):
            if h["revision"] == int(revision):
                return {**h, "source": "local"}
        raise ValueError(f"local_chat_dictionary_version_not_found:{revision}")

    async def revert_version(self, dictionary_id: int, revision: int, mode: str = "local") -> dict:
        target = await self.get_version(dictionary_id, revision)
        record = self.records[int(dictionary_id)]
        snapshot = target["snapshot"]
        for k in ("name", "description", "strategy", "max_tokens", "enabled"):
            record[k] = snapshot[k]
        record["entries"] = copy.deepcopy(snapshot["entries"])
        record["version"] += 1
        self._record_version(record, "revert")
        return {**self._summary(record), "reverted_to_revision": int(revision)}

    def _ensure_baseline(self, dictionary_id: int) -> None:
        if not self.history.get(dictionary_id):
            self._record_version(self.records[dictionary_id], "baseline")

    async def process_text(self, request_data: Any, mode: str = "local") -> dict:
        payload = dict(request_data)
        text = payload["text"]
        record = self.records.get(int(payload.get("dictionary_id") or 0))
        if record is None:
            raise ValueError("Local chat dictionary was not found.")
        token_budget = int(payload.get("token_budget") or 5000)
        processed = text
        diag_entries = []
        tokens_used = 0
        budget_exceeded = False
        applied_order = 0

        def _base(stored_index: int, pattern: str, entry: dict) -> dict:
            return {
                "input_index": stored_index,
                "pattern": pattern,
                "replacements": 0,
                "token_cost": len((entry.get("replacement") or "").split()),
                "applied_order": None,
                "content_preview": (entry.get("replacement") or "")[:40],
                "entry_id": f"local:chat_dictionary_entry:{record['id']}:{stored_index}",
            }

        # Priority order (higher first), ties broken by original stored
        # index — mirrors the real engine's within-scope priority sort.
        ordered = sorted(
            enumerate(record["entries"]),
            key=lambda pair: (-int(pair[1].get("priority") or 0), pair[0]),
        )
        # Never-matched entries are omitted from diagnostics entirely (real shape).
        candidates = [
            (stored_index, entry, entry.get("pattern") or "")
            for stored_index, entry in ordered
            if entry.get("type") != "regex"
            and (entry.get("pattern") or "")
            and (entry.get("pattern") or "") in text
        ]
        for i, (stored_index, entry, pattern) in enumerate(candidates):
            base = _base(stored_index, pattern, entry)
            token_cost = base["token_cost"]
            if not entry.get("enabled", True):
                diag_entries.append({**base, "status": "skipped:disabled"})
                continue
            if float(entry.get("probability", 1.0)) == 0.0:
                diag_entries.append({**base, "status": "skipped:probability"})
                continue
            if tokens_used + token_cost > token_budget:
                budget_exceeded = True
                diag_entries.append({**base, "status": "skipped:token_budget"})
                # Walk-and-stop: everything after the first non-fitting
                # candidate is unreachable too — mark the rest, then stop.
                for r_index, r_entry, r_pattern in candidates[i + 1 :]:
                    diag_entries.append(
                        {**_base(r_index, r_pattern, r_entry), "status": "skipped:token_budget"}
                    )
                break
            tokens_used += token_cost
            count = processed.count(pattern)
            processed = processed.replace(pattern, entry.get("replacement") or "")
            diag_entries.append(
                {**base, "status": "fired", "replacements": count, "applied_order": applied_order}
            )
            applied_order += 1
        self.calls.append(("process", text))
        response = {
            "text": text,
            "processed_text": processed,
            "dictionary_id": record["id"],
            "source": "local",
        }
        if self.emit_diagnostics:
            fired = sum(1 for r in diag_entries if r["status"] == "fired")
            response["diagnostics"] = {
                "entries": diag_entries,
                "matched": len(diag_entries),
                "fired": fired,
                "skipped": len(diag_entries) - fired,
                "total_replacements": sum(r["replacements"] for r in diag_entries),
                "tokens_used": tokens_used,
                "token_budget": token_budget,
                "budget_exceeded": budget_exceeded,
            }
        return response

    async def export_json(self, dictionary_id: int, mode: str = "local") -> dict:
        record = self.records[int(dictionary_id)]
        return {
            "dictionary_id": int(dictionary_id),
            "data": {
                "name": record["name"], "description": record.get("description") or "",
                "content": None,
                "entries": copy.deepcopy(record["entries"]),
                "strategy": record.get("strategy") or "sorted_evenly",
                "max_tokens": record.get("max_tokens") or 1000,
                "enabled": bool(record.get("enabled", True)),
                "version": record.get("version", 1),
            },
            "source": "local",
        }

    async def export_markdown(self, dictionary_id: int, mode: str = "local") -> dict:
        record = self.records[int(dictionary_id)]
        lines = [f"{e.get('pattern')}: {e.get('replacement')}" for e in record["entries"]]
        return {"dictionary_id": int(dictionary_id), "name": record["name"],
                "content": "\n".join(lines) + ("\n" if lines else ""), "source": "local"}

    async def import_json(self, request_data: Any, mode: str = "local") -> dict:
        payload = dict(request_data)
        data = dict(payload.get("data") or {})
        name = data.get("name") or payload.get("name") or "Imported Dictionary"  # data.name WINS
        created = await self.create_dictionary(
            {"name": name, "description": data.get("description") or "",
             "max_tokens": data.get("max_tokens") or 1000,
             "enabled": bool(data.get("enabled", True)),
             "entries": data.get("entries") or []}
        )
        if (data.get("strategy") or "sorted_evenly") != "sorted_evenly":
            await self.update_dictionary(created["id"], {"strategy": data["strategy"]})
        self.calls.append(("import_json", name))
        return {"dictionary_id": created["id"], "source": "local"}

    async def import_markdown(self, request_data: Any, mode: str = "local") -> dict:
        payload = dict(request_data)
        name = payload["name"]  # REQUIRED, like the real service
        entries = []
        for line in str(payload.get("content") or "").splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                entries.append({"pattern": key.strip(), "replacement": value.strip(),
                                "probability": 1.0, "group": None, "timed_effects": None,
                                "max_replacements": 1, "type": "literal",
                                "enabled": True, "case_sensitive": False, "priority": 0})
        created = await self.create_dictionary({"name": name, "entries": entries})
        self.calls.append(("import_markdown", name))
        return {"dictionary_id": created["id"], "source": "local"}

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
             "group": None, "timed_effects": None, "max_replacements": 1, "type": "literal",
             "enabled": True, "case_sensitive": False, "priority": 0},
            {"pattern": "HR", "replacement": "heart rate", "probability": 1.0,
             "group": None, "timed_effects": None, "max_replacements": 1, "type": "literal",
             "enabled": True, "case_sensitive": False, "priority": 0},
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
    def on_dictionary_settings_edited(self, m): self.messages.append(m)


class TestDictionaryDetailWidget:
    async def test_load_populates_settings_and_entries(self):
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

    async def test_form_payload_rejects_out_of_range_probability(self):
        """Validate-don't-clamp: 150 is refused, not silently coerced to 100."""
        app = DetailHarnessApp()
        async with app.run_test() as pilot:
            widget = app.query_one(PersonasDictionaryDetailWidget)
            app.query_one("#personas-dict-entry-pattern", Input).value = "ASAP"
            app.query_one("#personas-dict-entry-probability", Input).value = "150"
            assert widget.form_payload() is None
            error = app.query_one("#personas-dict-entry-error", Static)
            assert "0-100" in str(error.renderable)

    async def test_form_payload_rejects_non_positive_max_replacements(self):
        """Validate-don't-clamp: 0 is refused, not silently coerced to 1."""
        app = DetailHarnessApp()
        async with app.run_test() as pilot:
            widget = app.query_one(PersonasDictionaryDetailWidget)
            app.query_one("#personas-dict-entry-pattern", Input).value = "ASAP"
            app.query_one("#personas-dict-entry-max-repl", Input).value = "0"
            assert widget.form_payload() is None
            error = app.query_one("#personas-dict-entry-error", Static)
            assert "positive" in str(error.renderable).lower()

    async def test_add_button_posts_message(self):
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
            # Switch to Settings tab to make the save button clickable
            from textual.widgets import TabbedContent
            tabs = app.query_one("#personas-dict-tabs", TabbedContent)
            tabs.active = "personas-dict-tab-settings"
            await pilot.pause()
            app.query_one("#personas-dict-name", Input).value = "Renamed"
            await pilot.pause()
            await pilot.click("#personas-dict-settings-save")
            await pilot.pause()
            saves = [m for m in app.messages if isinstance(m, DictionarySettingsSaveRequested)]
            assert saves and saves[0].payload["name"] == "Renamed"
            assert saves[0].payload["max_tokens"] == 1000

    async def test_settings_edited_fires_only_on_real_user_change(self):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_detail import (
            DictionarySettingsEdited,
        )

        app = DetailHarnessApp()
        async with app.run_test() as pilot:
            widget = app.query_one(PersonasDictionaryDetailWidget)
            await pilot.pause()
            record = FakeDictScopeService([make_dict_record(1)])._summary(make_dict_record(1))
            widget.load_dictionary(record)
            await pilot.pause()
            await pilot.pause()
            edited = [m for m in app.messages if isinstance(m, DictionarySettingsEdited)]
            assert edited == []  # mount + programmatic load fire nothing
            app.query_one("#personas-dict-name", Input).value = "User typed this"
            await pilot.pause()
            edited = [m for m in app.messages if isinstance(m, DictionarySettingsEdited)]
            assert len(edited) >= 1  # a real user change fires


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
            assert "dictionary" in str(
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
        # Default 80x24 clips the entry-form button row off-screen (the Add
        # button's center lands at y=24), so pilot.click would miss it.
        async with app.run_test(size=(200, 60)) as pilot:
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
        async with app.run_test(size=(200, 60)) as pilot:
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
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            table = screen.query_one("#personas-dict-entries-table", DataTable)
            table.move_cursor(row=1)
            # Let the RowHighlighted-driven form sync land (HR/heart rate)
            # before typing over it, mirroring real keyboard navigation.
            await pilot.pause()
            screen.query_one("#personas-dict-entry-pattern", Input).value = "HRV"
            screen.query_one("#personas-dict-entry-replacement", TextArea).text = "heart rate variability"
            await pilot.click("#personas-dict-entry-update")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.records[1]["entries"][1]["pattern"] == "HRV"

    async def test_arrow_key_highlight_syncs_form_before_update(
        self, mock_app_instance, stub_characters, fake_dict_service
    ):
        """Keyboard nav (RowHighlighted only) must resync the form, not just
        the cursor - else Update saves a stale, previously-selected row's
        form values onto whatever row the cursor now sits on."""
        from textual.widgets import DataTable

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            table = screen.query_one("#personas-dict-entries-table", DataTable)
            # Click row 0 (BP) - form fills with BP via RowSelected.
            table.move_cursor(row=0)
            await pilot.pause()
            assert screen.query_one("#personas-dict-entry-pattern", Input).value == "BP"
            # Arrow-down to row 1 (HR) - only RowHighlighted fires here.
            table.move_cursor(row=1)
            await pilot.pause()
            await pilot.click("#personas-dict-entry-update")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # B kept ITS OWN pattern: the form resynced to B on highlight,
            # so Update saved B's data, not A's stale "BP".
            assert fake_dict_service.records[1]["entries"][1]["pattern"] == "HR"
            assert fake_dict_service.records[1]["entries"][0]["pattern"] == "BP"

    async def test_move_down_sends_full_order_and_reorders(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import DataTable

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
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

    async def test_form_roundtrips_new_fields(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import Switch, TextArea

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-entry-pattern", Input).value = "ICU"
            screen.query_one("#personas-dict-entry-replacement", TextArea).text = "intensive care"
            screen.query_one("#personas-dict-entry-enabled", Switch).value = False
            screen.query_one("#personas-dict-entry-case", Switch).value = True
            screen.query_one("#personas-dict-entry-priority", Input).value = "5"
            await pilot.click("#personas-dict-entry-add")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            added = fake_dict_service.records[1]["entries"][-1]
            assert (added["enabled"], added["case_sensitive"], added["priority"]) == (False, True, 5)

    async def test_priority_input_validates_integer(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-entry-pattern", Input).value = "X"
            screen.query_one("#personas-dict-entry-priority", Input).value = "high"
            detail = screen.query_one("#personas-dictionary-detail")
            assert detail.form_payload() is None
            error = str(screen.query_one("#personas-dict-entry-error", Static).renderable)
            assert "whole number" in error.lower()


class TestDictionarySettings:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_save_persists_and_refreshes_row(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TabbedContent
        from tldw_chatbook.Widgets.Persona_Widgets.personas_inspector_pane import PersonasInspectorPane

        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 48)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            tabs = screen.query_one("#personas-dict-tabs", TabbedContent)
            tabs.active = "personas-dict-tab-settings"
            await pilot.pause()
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
            # Inspector must re-drive with the new name too, not just the row.
            inspector = screen.query_one(PersonasInspectorPane)
            assert "Medical Terms" in str(
                inspector.query_one("#personas-selected-name", Static).renderable
            )

    async def test_settings_edit_marks_dirty_and_guards_navigation(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TabbedContent

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            tabs = screen.query_one("#personas-dict-tabs", TabbedContent)
            tabs.active = "personas-dict-tab-settings"
            await pilot.pause()
            screen.query_one("#personas-dict-name", Input).value = "Half-renamed"
            await pilot.pause()
            assert screen.state.has_unsaved_changes is True
            assert "- unsaved" in str(screen.query_one("#personas-title", Static).renderable)

    async def test_reverting_edit_clears_dirty_flag(self, mock_app_instance, stub_characters, fake_dict_service):
        """Transition-based dirty signaling: editing back to the loaded value
        must clear has_unsaved_changes, not just leave it stuck True."""
        from textual.widgets import TabbedContent

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            tabs = screen.query_one("#personas-dict-tabs", TabbedContent)
            tabs.active = "personas-dict-tab-settings"
            await pilot.pause()
            name_input = screen.query_one("#personas-dict-name", Input)
            original = name_input.value
            name_input.value = "Half-renamed"
            await pilot.pause()
            assert screen.state.has_unsaved_changes is True
            assert "- unsaved" in str(screen.query_one("#personas-title", Static).renderable)
            name_input.value = original
            await pilot.pause()
            assert screen.state.has_unsaved_changes is False
            assert "- unsaved" not in str(screen.query_one("#personas-title", Static).renderable)

    async def test_conflict_surfaces_status_not_crash(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TabbedContent

        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 48)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            tabs = screen.query_one("#personas-dict-tabs", TabbedContent)
            tabs.active = "personas-dict-tab-settings"
            await pilot.pause()
            fake_dict_service.records[1]["version"] = 99  # concurrent writer
            screen.query_one("#personas-dict-name", Input).value = "Medical Terms"
            await pilot.click("#personas-dict-settings-save")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            status = screen.query_one("#personas-dict-status", Static)
            assert "changed since it was loaded" in str(status.renderable)


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
        async with app.run_test(size=(200, 60)) as pilot:
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
        fake_dict_service.records[1]["entries"][0].update(
            {"enabled": False, "case_sensitive": True, "priority": 4}
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
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
            # Check all nine fields are carried
            src = fake_dict_service.records[1]["entries"][0]
            dup = copy_rec["entries"][0]
            for field in ("pattern", "replacement", "probability", "group", "timed_effects",
                          "max_replacements", "type", "enabled", "case_sensitive", "priority"):
                assert dup.get(field) == src.get(field), field

    async def test_duplicate_button_hidden_outside_dictionaries(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _mounted(pilot)
            dup = screen.query_one("#personas-library-duplicate", Button)
            assert dup.display is False  # characters mode
            await pilot.click("#personas-mode-dictionaries")
            await pilot.pause()
            assert dup.display is True

    async def test_duplicate_survives_failed_strategy_copy(self, mock_app_instance, stub_characters, fake_dict_service):
        """Verify that a failed strategy copy doesn't orphan the new dictionary."""
        # Set non-default strategy on source
        fake_dict_service.records[1]["strategy"] = "character_lore_first"

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)

            # Monkeypatch update_dictionary to fail
            original_update = fake_dict_service.update_dictionary
            async def _failing_update(dictionary_id, request_data, mode="local", **kwargs):
                raise RuntimeError("strategy update boom")

            fake_dict_service.update_dictionary = _failing_update

            # Click duplicate
            await pilot.click("#personas-library-duplicate")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            # Restore the original update method
            fake_dict_service.update_dictionary = original_update

            # Verify the copy EXISTS in the service records
            copy_rec = next(
                (r for r in fake_dict_service.records.values() if r["name"] == "Medical Abbrev (copy)"),
                None
            )
            assert copy_rec is not None, "Duplicate dictionary should exist despite strategy update failure"

            # Verify its strategy is still default (update failed, so strategy wasn't changed)
            assert copy_rec["strategy"] == "sorted_evenly", \
                f"Strategy should remain default after failed update, but got {copy_rec['strategy']}"

            # Verify the screen selected the new copy (not orphaned)
            assert screen.state.selected_entity_name == "Medical Abbrev (copy)", \
                f"Screen should have selected the copy despite update failure, but selected {screen.state.selected_entity_name}"


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
        async with app.run_test(size=(200, 60)) as pilot:
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
        async with app.run_test(size=(200, 60)) as pilot:
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

    async def test_space_toggle_keeps_cursor_on_unselected_row(
        self, mock_app_instance, stub_characters, fake_dict_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            rows = screen.query_one("#personas-library-rows", ListView)
            rows.focus()
            rows.index = 1  # Chat-Speak - not selected
            await pilot.press("space")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.records[2]["enabled"] is True
            assert rows.index is not None
            row = rows.children[rows.index]
            assert str(row.query(Static).first().renderable) == "Chat-Speak"

            await pilot.press("space")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.records[2]["enabled"] is False

    async def test_space_toggle_on_selected_preserves_unsaved_edits(
        self, mock_app_instance, stub_characters, fake_dict_service
    ):
        from textual.widgets import TabbedContent

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            tabs = screen.query_one("#personas-dict-tabs", TabbedContent)
            tabs.active = "personas-dict-tab-settings"
            await pilot.pause()
            screen.query_one("#personas-dict-name", Input).value = "Half-renamed"
            await pilot.pause()

            rows = screen.query_one("#personas-library-rows", ListView)
            rows.focus()
            rows.index = 0
            await pilot.press("space")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert fake_dict_service.records[1]["enabled"] is False
            assert screen.query_one("#personas-dict-name", Input).value == "Half-renamed"
            assert screen.state.has_unsaved_changes is True
            assert screen.query_one("#personas-dict-enabled", Switch).value is False


class TestDictionaryTryIt:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_word_diff_marks_changes(self):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_tryit import word_diff

        original, processed = word_diff("check BP now", "check blood pressure now")
        assert "BP" in original.plain and "blood pressure" in processed.plain
        assert any(span.style == "strike dim" for span in original.spans)
        assert any(span.style == "bold underline" for span in processed.spans)

    async def test_word_diff_no_changes_has_no_spans(self):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_tryit import word_diff

        original, processed = word_diff("same text", "same text")
        assert not original.spans and not processed.spans

    async def test_run_renders_diff(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TextArea
        from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_tryit import (
            PersonasDictionaryTryItWidget,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
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

    async def test_tryit_renders_summary_and_fired_lines(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TextArea

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-tryit-sample", TextArea).text = "check BP now"
            await pilot.click("#personas-dict-tryit-run")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            summary = str(screen.query_one("#personas-dict-tryit-summary", Static).renderable)
            assert "1 fired" in summary and "0 skipped" in summary
            assert "/1000 tokens" in summary  # dict max_tokens=1000 rode along
            fired = str(screen.query_one("#personas-dict-tryit-fired", Static).renderable)
            assert "BP" in fired and "blood pressure" in fired and "×1" in fired

    async def test_tryit_renders_near_miss_with_reason(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TextArea

        fake_dict_service.records[1]["entries"][1]["probability"] = 0.0  # HR never fires
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-tryit-sample", TextArea).text = "BP and HR"
            await pilot.click("#personas-dict-tryit-run")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            nearmiss = str(screen.query_one("#personas-dict-tryit-nearmiss", Static).renderable)
            assert "HR" in nearmiss and "probability roll" in nearmiss
            summary = str(screen.query_one("#personas-dict-tryit-summary", Static).renderable)
            assert "1 fired" in summary and "1 skipped" in summary

    async def test_tryit_budget_flag_shows(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TextArea

        fake_dict_service.records[1]["max_tokens"] = 2  # both entries cost 2 -> second is dropped
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-tryit-sample", TextArea).text = "BP and HR"
            await pilot.click("#personas-dict-tryit-run")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            summary = str(screen.query_one("#personas-dict-tryit-summary", Static).renderable)
            assert "over budget" in summary
            nearmiss = str(screen.query_one("#personas-dict-tryit-nearmiss", Static).renderable)
            assert "token budget" in nearmiss

    async def test_tryit_degrades_without_diagnostics(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TextArea

        fake_dict_service.emit_diagnostics = False
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-tryit-sample", TextArea).text = "check BP now"
            await pilot.click("#personas-dict-tryit-run")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # Diff still renders (P1a behavior)...
            processed = str(screen.query_one("#personas-dict-tryit-processed", Static).renderable)
            assert "blood pressure" in processed
            # ...and the summary carries the honest unavailable note.
            summary = str(screen.query_one("#personas-dict-tryit-summary", Static).renderable)
            assert "diagnostics unavailable" in summary


class TestDictionaryValidationPanel:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_findings_listed_and_jump_moves_cursor(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import DataTable, OptionList

        fake_dict_service.records[1]["entries"].append(
            {"pattern": "BP", "replacement": "dup", "probability": 1.0, "group": None,
             "timed_effects": None, "max_replacements": 1, "type": "literal",
             "enabled": True, "case_sensitive": False, "priority": 0}
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            panel = screen.query_one("#personas-dict-validation", OptionList)
            assert panel.option_count == 1  # the duplicate BP
            panel.highlighted = 0
            panel.action_select()
            await pilot.pause()
            table = screen.query_one("#personas-dict-entries-table", DataTable)
            assert table.cursor_row == 2  # jumped to the duplicate (index 2)

    async def test_multi_finding_entry_does_not_crash_and_jump_selects_by_index(
        self, mock_app_instance, stub_characters, fake_dict_service
    ):
        """One entry tripping two rules (duplicate_pattern + probability_zero)
        must not crash the panel. Before the fix, options were keyed by
        ``id=str(finding.entry_id)`` and the add_option loop sat outside the
        try/except, so a second finding sharing an entry_id raised Textual's
        DuplicateID and took the whole selection flow down with it."""
        from textual.widgets import DataTable, Input, OptionList

        fake_dict_service.records[1]["entries"].append(
            {"pattern": "BP", "replacement": "dup-and-dead", "probability": 0.0,
             "group": None, "timed_effects": None, "max_replacements": 1, "type": "literal",
             "enabled": True, "case_sensitive": False, "priority": 0}
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)  # must not raise DuplicateID
            panel = screen.query_one("#personas-dict-validation", OptionList)
            assert panel.option_count == 2  # duplicate_pattern + probability_zero, same entry
            panel.highlighted = 1
            panel.action_select()
            await pilot.pause()
            table = screen.query_one("#personas-dict-entries-table", DataTable)
            assert table.cursor_row == 2  # jumped to the new (index 2) entry
            assert screen.query_one("#personas-dict-entry-probability", Input).value == "0"

    async def test_panel_clears_on_clean_dictionary(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import OptionList

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            assert screen.query_one("#personas-dict-validation", OptionList).option_count == 0

    async def test_entries_tab_scrolls_buttons_reachable_when_short(self, mock_app_instance, stub_characters, fake_dict_service):
        """AC5b geometry: at a height that can't fit the whole tab, the scroll
        container exists and the button row is scrollable into view."""
        from textual.containers import VerticalScroll
        from textual.widgets import Button

        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 34)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            scroll = screen.query_one("#personas-dict-entries-scroll", VerticalScroll)
            add = screen.query_one("#personas-dict-entry-add", Button)
            add.scroll_visible(animate=False)
            await pilot.pause()
            container_region = scroll.region
            add_region = add.region
            # The button's row must actually land inside the scroll container's
            # visible viewport — a broken scroll leaves it below the container.
            assert container_region.y <= add_region.y < container_region.y + container_region.height
            assert scroll.scroll_offset.y > 0  # the scroll genuinely moved

    async def test_clear_empties_validation_panel(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import OptionList

        fake_dict_service.records[1]["entries"].append(
            {"pattern": "BP", "replacement": "dup", "probability": 1.0, "group": None,
             "timed_effects": None, "max_replacements": 1, "type": "literal",
             "enabled": True, "case_sensitive": False, "priority": 0}
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            panel = screen.query_one("#personas-dict-validation", OptionList)
            assert panel.option_count == 1
            detail = screen.query_one("#personas-dictionary-detail")
            detail.clear()
            await pilot.pause()
            assert panel.option_count == 0
            assert detail._validation_findings == []


class TestTryItDisabledReason:
    async def test_disabled_entry_renders_reason(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TextArea

        fake_dict_service.records[1]["entries"][1]["enabled"] = False  # HR off
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            rows = screen.query_one("#personas-library-rows", ListView)
            rows.index = 0
            rows.action_select_cursor()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            screen.query_one("#personas-dict-tryit-sample", TextArea).text = "BP and HR"
            await pilot.click("#personas-dict-tryit-run")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            nearmiss = str(screen.query_one("#personas-dict-tryit-nearmiss", Static).renderable)
            assert "HR" in nearmiss and "skipped: disabled" in nearmiss


class TestDictionaryMalformedProbability:
    """A malformed ``probability`` (corrupt row, hand-edited DB, etc.) must
    not crash the widget - the display falls back to 100%."""

    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_garbage_probability_does_not_crash_and_falls_back_to_100(
        self, mock_app_instance, stub_characters, fake_dict_service
    ):
        from textual.widgets import DataTable

        fake_dict_service.records[1]["entries"].append(
            {"pattern": "XYZ", "replacement": "garbage-prob", "probability": "garbage",
             "group": None, "timed_effects": None, "max_replacements": 1, "type": "literal",
             "enabled": True, "case_sensitive": False, "priority": 0}
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)  # must not raise
            table = screen.query_one("#personas-dict-entries-table", DataTable)
            assert table.row_count == 3
            assert "100" in str(table.get_cell_at((2, 3)))


class TestDictionaryValidationMarkupSafety:
    """Rich markup ("[tag]") must never be interpreted in validation-panel
    option text - both the leading "[code]" marker and any "[" inside a
    user's own pattern must render literally."""

    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_validation_option_renders_markup_inert(
        self, mock_app_instance, stub_characters, fake_dict_service
    ):
        from textual.widgets import OptionList

        fake_dict_service.records[1]["entries"] = [
            {"pattern": "AB[x]CD", "replacement": "one", "probability": 1.0, "group": None,
             "timed_effects": None, "max_replacements": 1, "type": "literal",
             "enabled": True, "case_sensitive": False, "priority": 0},
            {"pattern": "AB[x]CD", "replacement": "two", "probability": 1.0, "group": None,
             "timed_effects": None, "max_replacements": 1, "type": "literal",
             "enabled": True, "case_sensitive": False, "priority": 0},
        ]
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            panel = screen.query_one("#personas-dict-validation", OptionList)
            assert panel.option_count == 1  # the duplicate-pattern finding
            # render_line is OptionList's actual on-screen rendering path -
            # unlike Option.prompt (which just echoes back whatever was
            # passed in, str or Text, unchanged), this is what a real user
            # would see, and is the only place OptionList's markup=True
            # parsing actually bites.
            rendered = panel.render_line(0).text
            assert "duplicate_pattern" in rendered
            assert "[x]" in rendered


class TestDictionaryStatsTab:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_stats_render_service_plus_client_enrichment(self, mock_app_instance, stub_characters, fake_dict_service):
        fake_dict_service.records[1]["entries"][1].update({"type": "regex", "pattern": "/hr/i", "enabled": False, "priority": 5})
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            body = str(screen.query_one("#personas-dict-stats-body", Static).renderable)
            assert "Entries: 2" in body
            assert "literal: 1" in body and "regex: 1" in body
            assert "disabled: 1" in body
            assert "Priority: 0..5" in body
            assert "tokens" in body  # approximate token total line

    async def test_stats_refreshed_after_settings_save(
        self, mock_app_instance, stub_characters, fake_dict_service
    ):
        """load_statistics() runs on select and on entry mutations, but a
        settings-only save (e.g. toggling enabled) must also re-feed the
        Stats tab - otherwise it goes stale without requiring a reselect."""
        from textual.widgets import TabbedContent

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            body_before = str(screen.query_one("#personas-dict-stats-body", Static).renderable)
            assert "Dictionary enabled: yes" in body_before

            screen.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-settings"
            await pilot.pause()
            screen.query_one("#personas-dict-enabled", Switch).value = False
            await pilot.click("#personas-dict-settings-save")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert fake_dict_service.records[1]["enabled"] is False
            body_after = str(screen.query_one("#personas-dict-stats-body", Static).renderable)
            assert "Dictionary enabled: no" in body_after


class TestDictionaryVersionsTab:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_versions_listed_with_baseline(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import DataTable

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            table = screen.query_one("#personas-dict-versions-table", DataTable)
            assert table.row_count >= 1
            assert str(table.get_cell_at((0, 1))) == "baseline"

    async def test_view_shows_snapshot_summary(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import DataTable, TabbedContent

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-versions"
            await pilot.pause()
            screen.query_one("#personas-dict-versions-table", DataTable).move_cursor(row=0)
            await pilot.click("#personas-dict-version-view")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            snapshot = str(screen.query_one("#personas-dict-version-snapshot", Static).renderable)
            assert "rev 1" in snapshot and "entries: 2" in snapshot and "sorted_evenly" in snapshot

    async def test_revert_confirms_then_restores(self, mock_app_instance, stub_characters, fake_dict_service, monkeypatch):
        from textual.widgets import DataTable, TabbedContent

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            # Mutate: rename via settings save to create revision 2.
            screen.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-settings"
            await pilot.pause()
            screen.query_one("#personas-dict-name", Input).value = "Renamed"
            await pilot.click("#personas-dict-settings-save")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.records[1]["name"] == "Renamed"

            async def _yes(name):
                return True

            monkeypatch.setattr(screen, "_confirm_dictionary_revert", _yes)
            # The Save button still holds focus from the click above; a
            # DescendantFocus event it triggers is still working through the
            # message queue and would otherwise re-fire TabPane.Focused ->
            # TabbedContent._on_tab_pane_focused, snapping .active back to
            # "personas-dict-tab-settings" on the very next pause. Blur first
            # so the tab switch below actually sticks.
            screen.set_focus(None)
            await pilot.pause()
            screen.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-versions"
            await pilot.pause()
            table = screen.query_one("#personas-dict-versions-table", DataTable)
            # list_versions() sorts newest-first (matching the real local
            # backend's reverse=True), so after the rename row 0 is revision
            # 2 ("update") and row 1 is revision 1 (baseline, pre-rename).
            assert str(table.get_cell_at((1, 1))) == "baseline"
            table.move_cursor(row=1)  # revision 1 (baseline, pre-rename)
            await pilot.click("#personas-dict-version-revert")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.records[1]["name"] == "Medical Abbrev"  # restored
            assert screen.query_one("#personas-dict-name", Input).value == "Medical Abbrev"  # detail reloaded

    async def test_revert_refreshes_inspector_name(
        self, mock_app_instance, stub_characters, fake_dict_service, monkeypatch
    ):
        """A revert restores the pre-rename name in the center detail (as
        covered above), but the inspector's 'Selected:' line is a separate
        widget the revert worker must also re-drive - otherwise it keeps
        showing the renamed value even though the record itself reverted."""
        from textual.widgets import DataTable, TabbedContent

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            # Mutate: rename via settings save to create revision 2.
            screen.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-settings"
            await pilot.pause()
            screen.query_one("#personas-dict-name", Input).value = "Renamed"
            await pilot.click("#personas-dict-settings-save")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.records[1]["name"] == "Renamed"
            assert "Renamed" in str(
                screen.query_one("#personas-selected-name", Static).renderable
            )

            async def _yes(name):
                return True

            monkeypatch.setattr(screen, "_confirm_dictionary_revert", _yes)
            # See the comment in test_revert_confirms_then_restores: blur
            # first so the tab switch below actually sticks.
            screen.set_focus(None)
            await pilot.pause()
            screen.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-versions"
            await pilot.pause()
            table = screen.query_one("#personas-dict-versions-table", DataTable)
            table.move_cursor(row=1)  # revision 1 (baseline, pre-rename)
            await pilot.click("#personas-dict-version-revert")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.records[1]["name"] == "Medical Abbrev"  # restored
            assert str(
                screen.query_one("#personas-selected-name", Static).renderable
            ) == "Selected: Medical Abbrev"

    async def test_versions_table_renders_bracketed_name_safely(
        self, mock_app_instance, stub_characters, fake_dict_service
    ):
        """Bare str cells are markup-parsed by DataTable's idle render pass.
        A dictionary name containing an unmatched closing tag (e.g. '[/]')
        raises an uncaught rich.errors.MarkupError there - not caught by
        anything, so it corrupts/crashes the widget on selection. Cells must
        be pre-wrapped in Text() so they're never markup-parsed. This name
        also carries a plain '[WIP]' prefix to confirm ordinary bracketed
        text still renders through intact.
        """
        from textual.widgets import DataTable

        fake_dict_service.records[1]["name"] = "[WIP] [/] Meds"
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            table = screen.query_one("#personas-dict-versions-table", DataTable)
            cell = table.get_cell_at((0, 2))
            assert "[WIP]" in str(cell)


class TestDictionaryExport:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_export_json_writes_file_and_reports_path(self, mock_app_instance, stub_characters, fake_dict_service, tmp_path, monkeypatch):
        from textual.widgets import TabbedContent
        import tldw_chatbook.UI.Screens.personas_screen as screen_module

        monkeypatch.setattr(screen_module, "get_user_data_dir", lambda: tmp_path)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-settings"
            await pilot.pause()
            await pilot.click("#personas-dict-export-json")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            files = list((tmp_path / "exports").glob("medical-abbrev-*.json"))
            assert len(files) == 1
            import json as jsonlib
            payload = jsonlib.loads(files[0].read_text())
            assert payload["data"]["name"] == "Medical Abbrev"
            assert payload["data"]["strategy"] == "sorted_evenly"
            status = str(screen.query_one("#personas-dict-status", Static).renderable)
            assert "Exported to" in status and str(files[0]) in status

    async def test_markdown_export_gates_on_lossy_confirm(self, mock_app_instance, stub_characters, fake_dict_service, tmp_path, monkeypatch):
        from textual.widgets import TabbedContent
        import tldw_chatbook.UI.Screens.personas_screen as screen_module

        monkeypatch.setattr(screen_module, "get_user_data_dir", lambda: tmp_path)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)

            async def _declined():
                return False

            monkeypatch.setattr(screen, "_confirm_lossy_markdown_export", _declined)
            screen.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-settings"
            await pilot.pause()
            await pilot.click("#personas-dict-export-md")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert list((tmp_path / "exports").glob("*.md")) == []  # declined -> no file

            async def _accepted():
                return True

            monkeypatch.setattr(screen, "_confirm_lossy_markdown_export", _accepted)
            # Button.press() flags "-active" for active_effect_duration (0.2s) and
            # _on_click() no-ops a second click while that class is set; a real-time
            # pause (pilot.pause() alone only yields to the event loop, it doesn't
            # advance the clock) lets the flash clear so this second click registers.
            await pilot.pause(0.3)
            await pilot.click("#personas-dict-export-md")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            files = list((tmp_path / "exports").glob("medical-abbrev-*.md"))
            assert len(files) == 1
            assert "BP: blood pressure" in files[0].read_text()


class TestDictionaryImport:
    @staticmethod
    def _capture_notifications(app) -> list[tuple[str, str]]:
        """Shadow App.notify with an instance attribute, like _notify resolves it."""
        captured: list[tuple[str, str]] = []
        app.notify = lambda message, severity="information", **kwargs: captured.append(
            (str(message), severity)
        )
        return captured

    async def test_import_button_visible_in_dictionaries_mode(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            btn = screen.query_one("#personas-library-import", Button)
            assert btn.display is True
            assert "dictionary" in str(btn.tooltip).lower()

    async def test_import_json_file_selects_new_dictionary(self, mock_app_instance, stub_characters, fake_dict_service, tmp_path):
        import json as jsonlib

        payload = {"data": {"name": "Shipped In", "description": "", "content": None,
                             "entries": [{"pattern": "RR", "replacement": "resp rate",
                                          "probability": 1.0, "group": None, "timed_effects": None,
                                          "max_replacements": 1, "type": "literal",
                                          "enabled": True, "case_sensitive": False, "priority": 0}],
                             "strategy": "character_lore_first", "max_tokens": 500,
                             "enabled": True, "version": 3}}
        source = tmp_path / "shipped.json"
        source.write_text(jsonlib.dumps(payload))
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            names = [r["name"] for r in fake_dict_service.records.values()]
            assert "Shipped In" in names
            assert screen.state.selected_entity_name == "Shipped In"
            imported = next(r for r in fake_dict_service.records.values() if r["name"] == "Shipped In")
            assert imported["strategy"] == "character_lore_first"

    async def test_import_conflict_renames_and_succeeds(self, mock_app_instance, stub_characters, fake_dict_service, tmp_path):
        import json as jsonlib

        payload = {"data": {"name": "Medical Abbrev", "description": "", "content": None,
                             "entries": [], "strategy": "sorted_evenly", "max_tokens": 1000,
                             "enabled": True, "version": 1}}
        source = tmp_path / "clash.json"
        source.write_text(jsonlib.dumps(payload))
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            names = [r["name"] for r in fake_dict_service.records.values()]
            assert "Medical Abbrev (imported)" in names  # data.name mutated, retried

    async def test_import_markdown_uses_filename_stem(self, mock_app_instance, stub_characters, fake_dict_service, tmp_path):
        source = tmp_path / "field-notes.md"
        source.write_text("RR: resp rate\n")
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            names = [r["name"] for r in fake_dict_service.records.values()]
            assert "field-notes" in names

    async def test_import_bad_json_creates_nothing(self, mock_app_instance, stub_characters, fake_dict_service, tmp_path):
        source = tmp_path / "broken.json"
        source.write_text("{not json")
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            before = len(fake_dict_service.records)
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            assert len(fake_dict_service.records) == before

    async def test_import_non_utf8_file_notifies_no_crash(
        self, mock_app_instance, stub_characters, fake_dict_service, tmp_path
    ):
        """UnicodeDecodeError is a ValueError, not an OSError - the read
        try/except must catch it too or this worker-hosted coroutine
        crashes the whole TUI (exit_on_error=True on the default worker)."""
        source = tmp_path / "binary.json"
        source.write_bytes(b"\xff\xfe bad")
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            before = len(fake_dict_service.records)
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            assert len(fake_dict_service.records) == before
            assert any(severity == "error" for _, severity in notifications)

    async def test_import_json_array_notifies_no_crash(
        self, mock_app_instance, stub_characters, fake_dict_service, tmp_path
    ):
        """A structurally-valid JSON array has no 'data' key, so the coercion
        falls through to dict(parsed) - which raises on a non-dict payload
        unless it's explicitly guarded."""
        source = tmp_path / "array.json"
        source.write_text("[1, 2, 3]")
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            before = len(fake_dict_service.records)
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            assert len(fake_dict_service.records) == before
            assert any(severity == "error" for _, severity in notifications)

    async def test_import_json_null_data_notifies_no_crash(
        self, mock_app_instance, stub_characters, fake_dict_service, tmp_path
    ):
        """An envelope with a non-dict 'data' field (here: null) must also be
        guarded - dict(None) raises the same as dict(parsed) on a list."""
        import json as jsonlib

        source = tmp_path / "null-data.json"
        source.write_text(jsonlib.dumps({"data": None}))
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            before = len(fake_dict_service.records)
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            assert len(fake_dict_service.records) == before
            assert any(severity == "error" for _, severity in notifications)

    async def test_import_oversized_file_notifies_no_crash(
        self, mock_app_instance, stub_characters, fake_dict_service, tmp_path, monkeypatch
    ):
        """A huge import file must be rejected before the read - MemoryError
        from `source.read_text` on a genuinely oversized file is not an
        OSError, so it would otherwise escape the existing read guard and
        crash the whole TUI (exit_on_error=True on the default worker).
        Exercises the guard by lowering the byte ceiling instead of writing
        a multi-MB fixture."""
        import tldw_chatbook.UI.Screens.personas_screen as screen_module

        monkeypatch.setattr(screen_module, "PERSONAS_DICTIONARY_IMPORT_MAX_BYTES", 5)
        source = tmp_path / "toobig.json"
        source.write_text('{"data": {}}')
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            before = len(fake_dict_service.records)
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            assert len(fake_dict_service.records) == before
            assert any(
                "larger than" in message and severity == "error"
                for message, severity in notifications
            )

    async def test_import_deeply_nested_json_notifies_no_crash(
        self, mock_app_instance, stub_characters, fake_dict_service, tmp_path
    ):
        """Deeply nested JSON blows the recursive-descent parser's stack -
        RecursionError is not a JSONDecodeError, so the parse guard must
        catch it too or the worker crashes the whole TUI."""
        source = tmp_path / "nested.json"
        source.write_text("[" * 20000 + "]" * 20000)
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            before = len(fake_dict_service.records)
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            assert len(fake_dict_service.records) == before
            assert any(severity == "error" for _, severity in notifications)

    async def test_import_completion_skipped_when_mode_changed(
        self, mock_app_instance, stub_characters, fake_dict_service, tmp_path
    ):
        """The import coroutine awaits the DB call mid-flight; if the user
        switches away from Dictionaries mode before it resumes, completion
        must not yank them back into Dictionaries mode and select the newly
        imported dictionary out from under them. The import itself (a DB
        write, already committed by the time the mode is checked) must
        still succeed."""
        import json as jsonlib

        payload = {"data": {"name": "Late Arrival", "description": "", "content": None,
                             "entries": [], "strategy": "sorted_evenly", "max_tokens": 1000,
                             "enabled": True, "version": 1}}
        source = tmp_path / "late.json"
        source.write_text(jsonlib.dumps(payload))
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            # Simulate the user having navigated away while the import ran.
            screen.state.active_mode = "characters"
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            names = [r["name"] for r in fake_dict_service.records.values()]
            assert "Late Arrival" in names  # the import itself still succeeded
            assert screen.state.selected_entity_kind != "dictionary"  # no yank-back
            assert any(
                "open Dictionaries" in message and severity == "information"
                for message, severity in notifications
            )


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


class TestDictionaryAttachFlow:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_attach_via_picker_then_detach(self, mock_app_instance, stub_characters, fake_dict_service, monkeypatch):
        from textual.widgets import DataTable, TabbedContent
        from tldw_chatbook.Widgets.Persona_Widgets.dictionary_attach_picker import DictionaryAttachPicker

        # Seed a conversation the picker can offer + the attach can target.
        fake_dict_service.conversations = {"c1": {"id": "c1", "title": "Noir case", "active_dictionaries": []}}

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            # The attach/detach buttons live in the Attachments TabPane; the
            # default active tab is Entries, so a click won't land on them
            # (and may fall through to whatever's underneath) until switched.
            screen.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-attachments"
            await pilot.pause()

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
