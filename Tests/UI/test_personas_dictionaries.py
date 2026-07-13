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
