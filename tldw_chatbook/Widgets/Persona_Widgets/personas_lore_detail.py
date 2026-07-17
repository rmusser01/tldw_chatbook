"""Center detail widget for the Roleplay Lore/world-book mode (Entries + Settings)."""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
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

POSITIONS = (
    ("before_char", "Before character"),
    ("after_char", "After character"),
    ("at_start", "At start"),
    ("at_end", "At end"),
)


class LoreBookCreateRequested(Message):
    """Request to create a new lore book."""


class LoreBookDuplicateRequested(Message):
    """Request to duplicate the current lore book."""


class LoreBookDeleteRequested(Message):
    """Request to delete the current lore book."""


class LoreBookEnableToggled(Message):
    def __init__(self, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled


class LoreEntryAddRequested(Message):
    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.payload = payload


class LoreEntryUpdateRequested(Message):
    def __init__(self, entry_id: str, payload: dict) -> None:
        super().__init__()
        self.entry_id = entry_id
        self.payload = payload


class LoreEntryDeleteRequested(Message):
    def __init__(self, entry_id: str) -> None:
        super().__init__()
        self.entry_id = entry_id


class LoreEntriesReorderRequested(Message):
    def __init__(self, entry_ids: list[str]) -> None:
        super().__init__()
        self.entry_ids = entry_ids


class LoreBookSettingsSaveRequested(Message):
    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.payload = payload


class PersonasLoreDetailWidget(Vertical):
    """Entries + Settings tabs for one lore/world book. Emits intents; owns no I/O."""

    DEFAULT_CSS = """
    PersonasLoreDetailWidget {
        height: 1fr;
        min-height: 0;
    }
    PersonasLoreDetailWidget #personas-lore-entries-scroll {
        height: 1fr;
    }
    PersonasLoreDetailWidget #personas-lore-entries-table {
        min-height: 6;
        max-height: 12;
    }
    PersonasLoreDetailWidget .personas-lore-form-row {
        height: auto;
    }
    PersonasLoreDetailWidget #personas-lore-entry-content,
    PersonasLoreDetailWidget #personas-lore-description {
        height: 4;
    }
    PersonasLoreDetailWidget #personas-lore-status {
        height: 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._entries: list[dict] = []
        self._suppress_enabled_toggle: bool = False  # True while we set the Switch programmatically

    # ----- compose -----

    def compose(self) -> ComposeResult:
        """Build the Entries and Settings tab panes.

        Returns:
            ComposeResult: The child widgets that make up this container
            (the tabbed Entries/Settings layout plus the status line).
        """
        with TabbedContent(id="personas-lore-tabs"):
            with TabPane("Entries", id="personas-lore-tab-entries"):
                with VerticalScroll(id="personas-lore-entries-scroll"):
                    yield DataTable(id="personas-lore-entries-table", cursor_type="row")
                    with Horizontal(classes="personas-lore-form-row"):
                        yield Input(placeholder="Keys (comma-separated)", id="personas-lore-entry-keys")
                        yield Select(
                            [(label, value) for value, label in POSITIONS],
                            id="personas-lore-entry-position",
                            value="before_char",
                            allow_blank=False,
                        )
                        yield Switch(value=True, id="personas-lore-entry-enabled", tooltip="Entry enabled")
                    yield TextArea(id="personas-lore-entry-content")
                    with Horizontal(classes="personas-lore-form-row"):
                        yield Button("Add", id="personas-lore-entry-add", classes="console-action-secondary")
                        yield Button("Update", id="personas-lore-entry-update", classes="console-action-secondary")
                        yield Button("Delete", id="personas-lore-entry-delete", classes="console-action-secondary")
                        yield Button("Move up", id="personas-lore-entry-move-up", classes="console-action-secondary")
                        yield Button("Move down", id="personas-lore-entry-move-down", classes="console-action-secondary")
            with TabPane("Settings", id="personas-lore-tab-settings"):
                yield Input(placeholder="Name", id="personas-lore-name")
                yield TextArea(id="personas-lore-description")
                yield Input(placeholder="Scan depth", id="personas-lore-scan-depth", value="3")
                yield Input(placeholder="Token budget", id="personas-lore-token-budget", value="500")
                with Horizontal(classes="personas-lore-form-row"):
                    yield Static("Recursive scanning", markup=False)
                    yield Switch(value=False, id="personas-lore-recursive")
                with Horizontal(classes="personas-lore-form-row"):
                    yield Static("Enabled", markup=False)
                    yield Switch(value=True, id="personas-lore-enabled")
                yield Button("Save settings", id="personas-lore-settings-save", classes="console-action-secondary")
        yield Static("", id="personas-lore-status", markup=False)

    def on_mount(self) -> None:
        """Register the entries table's columns once the widget is mounted.

        Returns:
            None.
        """
        table = self.query_one("#personas-lore-entries-table", DataTable)
        table.add_columns("keys", "content", "position", "enabled")

    # ----- public API -----

    def load_book(self, record: dict) -> None:
        """Fill settings + entries from a get_world_book()/summary dict.

        Args:
            record: A normalized world-book dict with name/description/
                scan_depth/token_budget/recursive_scanning/enabled and
                optionally entries.

        Returns:
            None.
        """
        self.query_one("#personas-lore-name", Input).value = str(record.get("name") or "")
        self.query_one("#personas-lore-description", TextArea).text = str(record.get("description") or "")
        self.query_one("#personas-lore-scan-depth", Input).value = str(record.get("scan_depth") or 3)
        self.query_one("#personas-lore-token-budget", Input).value = str(record.get("token_budget") or 500)
        self.query_one("#personas-lore-recursive", Switch).value = bool(record.get("recursive_scanning", False))
        self._set_enabled_switch(bool(record.get("enabled", True)))
        self.update_entries(list(record.get("entries") or []))
        self.query_one("#personas-lore-status", Static).update("")

    def update_entries(self, entries: list[dict]) -> None:
        """Re-render the entries table from a fresh service response.

        Args:
            entries: Normalized entry dicts (as returned by the world-book
                service) to display, in application order.
        """
        self._entries = list(entries)
        table = self.query_one("#personas-lore-entries-table", DataTable)
        table.clear()
        for entry in self._entries:
            enabled = bool(entry.get("enabled", True))
            style = "dim" if not enabled else ""
            keys = ", ".join(str(k) for k in (entry.get("keys") or []))
            content = str(entry.get("content") or "")
            preview = content if len(content) <= 60 else content[:57] + "..."
            table.add_row(
                Text(escape(keys), style=style),
                Text(escape(preview), style=style),
                Text(escape(str(entry.get("position") or "before_char")), style=style),
                Text("yes" if enabled else "no", style=style),
                key=str(entry.get("id")),
            )

    def apply_enabled(self, enabled: bool) -> None:
        """Reflect an externally-toggled enabled flag without touching other fields."""
        self._set_enabled_switch(bool(enabled))

    def clear(self) -> None:
        """Reset the widget to its unloaded state.

        Returns:
            None.
        """
        self._entries = []
        self.query_one("#personas-lore-entries-table", DataTable).clear()
        self.query_one("#personas-lore-name", Input).value = ""
        self.query_one("#personas-lore-description", TextArea).text = ""
        self.query_one("#personas-lore-scan-depth", Input).value = "3"
        self.query_one("#personas-lore-token-budget", Input).value = "500"
        self.query_one("#personas-lore-recursive", Switch).value = False
        self._set_enabled_switch(True)
        self.query_one("#personas-lore-status", Static).update("")

    def _set_enabled_switch(self, value: bool) -> None:
        """Set the Settings-tab Enabled switch without echoing a Toggled message.

        Guards with ``_suppress_enabled_toggle`` so programmatic updates
        (load_book/apply_enabled/clear) never round-trip back out as a
        user-driven ``LoreBookEnableToggled``. When the switch's value does
        not actually change, Textual never fires ``Switch.Changed`` at all,
        so the deferred reset below clears the guard on the next refresh
        instead of leaving it stuck (which would otherwise swallow the next
        real user toggle).
        """
        self._suppress_enabled_toggle = True
        self.query_one("#personas-lore-enabled", Switch).value = value
        self.call_after_refresh(self._clear_enabled_toggle_guard)

    def _clear_enabled_toggle_guard(self) -> None:
        self._suppress_enabled_toggle = False

    @property
    def selected_entry_id(self) -> str | None:
        table = self.query_one("#personas-lore-entries-table", DataTable)
        if not self._entries or table.cursor_row is None or table.cursor_row < 0:
            return None
        try:
            row_key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key
            return str(row_key.value)
        except Exception:
            return None

    def entry_ids_in_order(self) -> list[str]:
        return [str(e.get("id")) for e in self._entries]

    def entry_form_payload(self) -> dict | None:
        """API-named entry payload from the form; None if keys or content is empty.

        Returns:
            dict | None: The entry payload keyed by API field names
            (``keys``, ``content``, ``position``, ``enabled``,
            ``insertion_order``), or ``None`` if keys or content are empty.
        """
        raw_keys = self.query_one("#personas-lore-entry-keys", Input).value
        keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        content = self.query_one("#personas-lore-entry-content", TextArea).text
        if not keys or not content.strip():
            return None
        position = str(self.query_one("#personas-lore-entry-position", Select).value)
        enabled = bool(self.query_one("#personas-lore-entry-enabled", Switch).value)
        selected_id = self.selected_entry_id
        if selected_id is not None:
            entry = next((e for e in self._entries if str(e.get("id")) == selected_id), None)
            insertion_order = int(entry.get("insertion_order") or 0) if entry else len(self._entries)
        else:
            insertion_order = len(self._entries)
        return {
            "keys": keys,
            "content": content,
            "position": position,
            "enabled": enabled,
            "insertion_order": insertion_order,
        }

    def settings_payload(self) -> dict:
        raw_scan_depth = self.query_one("#personas-lore-scan-depth", Input).value.strip() or "3"
        try:
            scan_depth = int(raw_scan_depth)
        except ValueError:
            scan_depth = 3
        raw_token_budget = self.query_one("#personas-lore-token-budget", Input).value.strip() or "500"
        try:
            token_budget = int(raw_token_budget)
        except ValueError:
            token_budget = 500
        return {
            "name": self.query_one("#personas-lore-name", Input).value.strip(),
            "description": self.query_one("#personas-lore-description", TextArea).text,
            "scan_depth": scan_depth,
            "token_budget": token_budget,
            "recursive_scanning": bool(self.query_one("#personas-lore-recursive", Switch).value),
            "enabled": bool(self.query_one("#personas-lore-enabled", Switch).value),
        }

    def set_status(self, message: str) -> None:
        self.query_one("#personas-lore-status", Static).update(message)

    # ----- events -----

    def _fill_form_from_entry(self, entry_id: str) -> None:
        """Sync the entry form fields from ``entry_id``'s current data."""
        entry = next((e for e in self._entries if str(e.get("id")) == entry_id), None)
        if entry is None:
            return
        self.query_one("#personas-lore-entry-keys", Input).value = ", ".join(
            str(k) for k in (entry.get("keys") or [])
        )
        self.query_one("#personas-lore-entry-content", TextArea).text = str(entry.get("content") or "")
        position = str(entry.get("position") or "before_char")
        if position in {p[0] for p in POSITIONS}:
            self.query_one("#personas-lore-entry-position", Select).value = position
        self.query_one("#personas-lore-entry-enabled", Switch).value = bool(entry.get("enabled", True))

    @on(DataTable.RowSelected, "#personas-lore-entries-table")
    def _row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self._fill_form_from_entry(str(event.row_key.value))

    @on(DataTable.RowHighlighted, "#personas-lore-entries-table")
    def _row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        # Arrow-key navigation only fires RowHighlighted (not RowSelected), so
        # without this the form silently keeps stale values from a prior row
        # while selected_entry_id tracks the cursor. This also fires on the
        # programmatic cursor moves update_entries triggers on reload, which
        # is fine: it refreshes from the fresh list.
        event.stop()
        if event.row_key is not None and event.row_key.value is not None:
            self._fill_form_from_entry(str(event.row_key.value))

    @on(Button.Pressed, "#personas-lore-entry-add")
    def _add_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        payload = self.entry_form_payload()
        if payload is not None:
            self.post_message(LoreEntryAddRequested(payload))

    @on(Button.Pressed, "#personas-lore-entry-update")
    def _update_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        entry_id = self.selected_entry_id
        if entry_id is None:
            self.set_status("Select an entry row first.")
            return
        payload = self.entry_form_payload()
        if payload is not None:
            self.post_message(LoreEntryUpdateRequested(entry_id, payload))

    @on(Button.Pressed, "#personas-lore-entry-delete")
    def _delete_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        entry_id = self.selected_entry_id
        if entry_id is None:
            self.set_status("Select an entry row first.")
            return
        self.post_message(LoreEntryDeleteRequested(entry_id))

    def _post_reorder(self, offset: int) -> None:
        entry_id = self.selected_entry_id
        ids = self.entry_ids_in_order()
        if entry_id is None or entry_id not in ids:
            self.set_status("Select an entry row first.")
            return
        index = ids.index(entry_id)
        target = index + offset
        if not 0 <= target < len(ids):
            return
        ids[index], ids[target] = ids[target], ids[index]
        self.post_message(LoreEntriesReorderRequested(ids))

    @on(Button.Pressed, "#personas-lore-entry-move-up")
    def _up_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._post_reorder(-1)

    @on(Button.Pressed, "#personas-lore-entry-move-down")
    def _down_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._post_reorder(1)

    @on(Button.Pressed, "#personas-lore-settings-save")
    def _settings_save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(LoreBookSettingsSaveRequested(self.settings_payload()))

    @on(Switch.Changed, "#personas-lore-enabled")
    def _enabled_changed(self, event: Switch.Changed) -> None:
        event.stop()
        if self._suppress_enabled_toggle:
            # A programmatic set (load_book/apply_enabled/clear) triggered this
            # Changed message; consume the flag without echoing it back out.
            self._suppress_enabled_toggle = False
            return
        self.post_message(LoreBookEnableToggled(bool(event.value)))


__all__ = [
    "LoreBookCreateRequested",
    "LoreBookDeleteRequested",
    "LoreBookDuplicateRequested",
    "LoreBookEnableToggled",
    "LoreBookSettingsSaveRequested",
    "LoreEntriesReorderRequested",
    "LoreEntryAddRequested",
    "LoreEntryDeleteRequested",
    "LoreEntryUpdateRequested",
    "PersonasLoreDetailWidget",
]
