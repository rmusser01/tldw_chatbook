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
        self._loaded_settings: dict | None = None  # last-loaded settings snapshot; dirty = value diff

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
        self.query_one("#personas-dict-name", Input).value = str(record.get("name") or "")
        self.query_one("#personas-dict-description", TextArea).text = str(record.get("description") or "")
        strategy = str(record.get("strategy") or "sorted_evenly")
        if strategy in STRATEGIES:
            self.query_one("#personas-dict-strategy", Select).value = strategy
        self.query_one("#personas-dict-max-tokens", Input).value = str(record.get("max_tokens") or 1000)
        self.query_one("#personas-dict-enabled", Switch).value = bool(
            record.get("enabled", record.get("is_active", True))
        )
        self.update_entries(list(record.get("entries") or []))
        self.query_one("#personas-dict-status", Static).update("")
        self._loaded_settings = self.settings_payload()

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

    def apply_enabled(self, enabled: bool) -> None:
        """Reflect an externally-toggled enabled flag without touching other fields.

        Keeps the dirty-detection snapshot consistent so the flip itself never
        counts as a user edit, while any in-flight edits stay intact.
        """
        self.query_one("#personas-dict-enabled", Switch).value = bool(enabled)
        if self._loaded_settings is not None:
            self._loaded_settings["enabled"] = bool(enabled)

    def clear(self) -> None:
        self._entries = []
        self._loaded_settings = None
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

    def _fill_form_from_entry(self, entry_id: str) -> None:
        """Sync the entry form fields from ``entry_id``'s current data."""
        entry = next((e for e in self._entries if str(e.get("id")) == entry_id), None)
        if entry is None:
            return
        self.query_one("#personas-dict-entry-pattern", Input).value = str(entry.get("pattern") or "")
        self.query_one("#personas-dict-entry-replacement", TextArea).text = str(entry.get("replacement") or "")
        self.query_one("#personas-dict-entry-regex", Switch).value = entry.get("type") == "regex"
        probability = entry.get("probability")
        self.query_one("#personas-dict-entry-probability", Input).value = str(
            round(float(probability if probability is not None else 1.0) * 100)
        )
        self.query_one("#personas-dict-entry-group", Input).value = str(entry.get("group") or "")
        self.query_one("#personas-dict-entry-max-repl", Input).value = str(entry.get("max_replacements") or 1)

    @on(DataTable.RowSelected, "#personas-dict-entries-table")
    def _row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self._fill_form_from_entry(str(event.row_key.value))

    @on(DataTable.RowHighlighted, "#personas-dict-entries-table")
    def _row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        # Arrow-key navigation only fires RowHighlighted (not RowSelected), so
        # without this the form silently keeps stale values from a prior row
        # while selected_entry_id tracks the cursor - Update would then save
        # the wrong entry's old form data onto the newly-highlighted entry.
        # This also fires on the programmatic cursor moves update_entries
        # triggers on reload, which is fine: it refreshes from the fresh list.
        event.stop()
        if event.row_key is not None and event.row_key.value is not None:
            self._fill_form_from_entry(str(event.row_key.value))

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
        if self._loaded_settings is None:
            return  # nothing loaded yet - mount-time Changed noise
        if self.settings_payload() != self._loaded_settings:
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
