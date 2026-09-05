"""Manual prompt-cache controls in the incumbent terminal workbench.

Operate-mode extension: compact shared actions lead selectable tables; secondary
telemetry stays in details. The app owns work, and this widget owns only selection.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.widgets import Button, Checkbox, Collapsible, DataTable, Input, Static

from tldw_chatbook.LLM_Management import snapshot_settings as preferences
from tldw_chatbook.LLM_Management.snapshot_models import SnapshotError
from tldw_chatbook.LLM_Management.snapshot_service import LlamaCppSnapshotService
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

PREFERENCES_UNAVAILABLE = (
    "Snapshot preferences unavailable. In F9 Advanced Config, correct "
    "llamacpp_snapshots.enabled (true/false) and keep_count (1–1000), then Reload."
)


def retention_copy(keep_count: int) -> str:
    return f"Keeps the newest {keep_count} across all models"


def token_copy(tokens: int | None) -> str:
    return "Unknown" if tokens is None else str(tokens)


def local_time(value: str) -> str:
    return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M:%S")


class LlamaCppSnapshotManager(Vertical):
    """Project the app-owned service without taking ownership of its operations."""

    def __init__(self, service: LlamaCppSnapshotService):
        super().__init__(id="llamacpp-snapshot-manager")
        self.service = service
        self._slot_id: int | None = None
        self._snapshot_id: str | None = None
        self._attachment = None
        self._unsubscribe = None
        self._elapsed_timer = None
        self._confirming = False
        self._offsets = [0]
        try:
            self._preferences = preferences.load_snapshot_preferences()
        except (ValueError, OSError):
            self._preferences = None
        self._effective_keep = (
            self._preferences.keep_count if self._preferences else None
        )

    def compose(self) -> ComposeResult:
        yield Static("Prompt-cache snapshots", classes="section-title")
        yield Static(
            "Save processed context to reuse later. Restoring does not change your conversations.",
            classes="snapshot-copy",
        )
        yield Static(
            retention_copy(self._effective_keep)
            if self._effective_keep is not None
            else "Retention unavailable",
            id="snapshot-retention",
        )
        with Horizontal(classes="snapshot-actions"):
            yield Button("Save", id="snapshot-save")
            yield Button("Restore", id="snapshot-restore")
            yield Button("Delete", id="snapshot-delete")
            yield Button("Refresh", id="snapshot-refresh")
        yield Static("Preparing observations…", id="snapshot-operation-status")
        yield Static("", id="snapshot-disabled-reason")
        yield Static("Slots — select a destination", classes="snapshot-copy")
        yield DataTable(id="snapshot-slots", cursor_type="row")
        yield Static("Snapshots — select a saved context", classes="snapshot-copy")
        yield DataTable(id="snapshot-records", cursor_type="row")
        with Horizontal(classes="snapshot-actions"):
            yield Button("Previous", id="snapshot-previous")
            yield Button("Next", id="snapshot-next")
        with Collapsible(
            title="Details & preferences",
            collapsed=True,
            id="snapshot-details-panel",
        ):
            yield Static("", id="snapshot-details", markup=False)
            yield Static(
                "Enable/disable applies on next launch.", classes="snapshot-copy"
            )
            yield Checkbox(
                "Enable snapshots",
                value=self._preferences.enabled if self._preferences else False,
                disabled=self._preferences is None,
                id="snapshot-enabled",
            )
            with Horizontal(classes="snapshot-actions"):
                yield Static("Keep count", classes="snapshot-keep-label")
                yield Input(
                    str(self._preferences.keep_count) if self._preferences else "",
                    disabled=self._preferences is None,
                    id="snapshot-keep",
                    type="integer",
                )
                yield Button(
                    "Apply", id="snapshot-apply", disabled=self._preferences is None
                )
                yield Button("Reload", id="snapshot-reload")
            yield Static(
                "Enable/disable applies on next launch. A lower count takes effect after the next completed save."
                if self._preferences
                else PREFERENCES_UNAVAILABLE,
                id="snapshot-preferences-result",
            )
        yield Static(
            "On tables/actions: s Save · r Restore · d Delete · f Refresh",
            classes="snapshot-copy",
        )

    def on_mount(self) -> None:
        self._attachment = object()
        self.query_one("#snapshot-slots", DataTable).add_columns(
            "Slot", "State", "Tokens", "Context"
        )
        self._unsubscribe = self.service.subscribe(self._paint)
        self._elapsed_timer = self.set_interval(1, self._paint_elapsed)
        self.watch(self.screen, "stack_updates", self._screen_reentered, init=False)
        self.request_refresh()

    def _screen_reentered(self) -> None:
        if self.is_mounted and self.screen.is_active:
            self.request_refresh()

    def on_unmount(self) -> None:
        self._attachment = None
        if self._unsubscribe:
            self._unsubscribe()
        if self._elapsed_timer:
            self._elapsed_timer.stop()

    def on_resize(self) -> None:
        self._paint()

    def request_refresh(self) -> None:
        """Refresh on entry or explicit action; never poll on a timer."""
        if self.is_mounted:
            self._offsets = [0]
            self.app.run_worker(
                self._refresh(), group="snapshot-view-refresh", exit_on_error=False
            )

    async def _refresh(self) -> None:
        attachment = self._attachment
        await self.service.refresh()
        try:
            loaded = await asyncio.to_thread(preferences.load_snapshot_preferences)
        except (ValueError, OSError):
            loaded = None
        if attachment is self._attachment and self.is_mounted:
            if loaded is None or self._preferences is None:
                self._set_loaded_preferences(loaded)
            else:
                self._effective_keep = loaded.keep_count
            self._paint()

    def _set_loaded_preferences(
        self, value: preferences.SnapshotPreferences | None
    ) -> None:
        self._preferences = value
        self._effective_keep = value.keep_count if value else None
        self.query_one("#snapshot-enabled", Checkbox).value = (
            value.enabled if value else False
        )
        self.query_one("#snapshot-keep", Input).value = (
            str(value.keep_count) if value else ""
        )
        self.query_one("#snapshot-preferences-result", Static).update(
            "Preferences loaded. Enable/disable applies on next launch."
            if value
            else PREFERENCES_UNAVAILABLE
        )

    def _paint_elapsed(self) -> None:
        if not self.is_mounted or self._attachment is None:
            return
        view = self.service.view()
        elapsed = (
            ""
            if view.started_at is None
            else f" · {int(time.monotonic() - view.started_at)}s"
        )
        status = view.status.replace("_", " ").capitalize()
        if view.status == "awaiting_ack":
            status = "Waiting for server"
        if view.status == "outcome_unknown":
            status = "Outcome unknown — Stop the server before trying again"
        elif view.message:
            status += " · " + view.message.replace("_", " ")
        self.query_one("#snapshot-operation-status", Static).update(status + elapsed)

    def _paint(self) -> None:
        if not self.is_mounted or self._attachment is None:
            return
        view = self.service.view()
        slots = self.query_one("#snapshot-slots", DataTable)
        records = self.query_one("#snapshot-records", DataTable)
        narrow = self.size.width < 60
        slots.styles.height = min(5, max(2, 1 + len(view.slots))) if narrow else 5
        records.styles.height = (
            min(5, max(2, 1 + 2 * len(view.catalog.records))) if narrow else 5
        )
        valid_slots = {slot.slot_id for slot in view.slots}
        if self._slot_id not in valid_slots:
            self._slot_id = None
            if not self._confirming:
                idle = sorted(
                    (slot for slot in view.slots if slot.busy is False),
                    key=lambda slot: (slot.tokens != 0, slot.slot_id),
                )
                if idle:
                    self._slot_id = idle[0].slot_id
        ids = [record.snapshot_id for record in view.catalog.records]
        if self._snapshot_id not in ids:
            self._snapshot_id = ids[0] if ids and not self._confirming else None
        with slots.prevent(DataTable.RowHighlighted):
            slots.clear()
            slots.show_cursor = self._slot_id is not None
            for index, slot in enumerate(view.slots):
                slots.add_row(
                    str(slot.slot_id),
                    "Busy"
                    if slot.busy
                    else ("Empty" if slot.tokens == 0 else "Idle")
                    if slot.busy is False
                    else "Unknown",
                    token_copy(slot.tokens),
                    token_copy(slot.context_size),
                    key=str(slot.slot_id),
                )
                if slot.slot_id == self._slot_id:
                    slots.move_cursor(row=index)
        labels = {
            "matching": "Matching configuration",
            "different": "Different configuration",
            "unknown": "Compatibility unknown",
        }
        compatibility = dict(view.snapshot_compatibility)
        with records.prevent(DataTable.RowHighlighted):
            records.clear(columns=True)
            if narrow:
                records.add_column(
                    "Saved context (local)", width=max(12, self.size.width - 4)
                )
            else:
                records.add_column("Saved (local)", width=19)
                records.add_column("Model", width=16)
                records.add_columns("Slot", "Tokens", "Bytes", "Compatibility")
            records.show_cursor = self._snapshot_id is not None
            for index, record in enumerate(view.catalog.records):
                status = labels[compatibility.get(record.snapshot_id, "unknown")]
                if narrow:
                    records.add_row(
                        local_time(record.created_utc) + "\n" + status,
                        key=record.snapshot_id,
                        height=2,
                    )
                else:
                    records.add_row(
                        local_time(record.created_utc),
                        record.model_label,
                        str(record.source_slot),
                        str(record.tokens),
                        str(record.bytes),
                        status,
                        key=record.snapshot_id,
                    )
                if record.snapshot_id == self._snapshot_id:
                    records.move_cursor(row=index)
        self.query_one("#snapshot-retention", Static).update(
            retention_copy(self._effective_keep)
            if self._effective_keep is not None
            else "Retention unavailable"
        )
        reason = view.disabled_reason or (
            "No snapshots yet — Save an idle slot to reuse its processed context."
            if not ids
            else ""
        )
        self.query_one("#snapshot-disabled-reason", Static).update(reason)
        selected = next(
            (slot for slot in view.slots if slot.slot_id == self._slot_id), None
        )
        ready = (
            view.status == "idle"
            and self._preferences is not None
            and not view.disabled_reason
            and selected is not None
            and selected.busy is False
        )
        if not view.disabled_reason:
            if view.status == "outcome_unknown":
                self.query_one("#snapshot-disabled-reason", Static).update(
                    "Stop the server to settle the previous operation."
                )
            elif view.status == "idle" and (
                selected is None or selected.busy is not False
            ):
                self.query_one("#snapshot-disabled-reason", Static).update(
                    "Select an idle slot before saving or restoring."
                )
        self.query_one("#snapshot-save", Button).disabled = (
            not ready or selected.tokens == 0
        )
        if ready and selected.tokens == 0:
            self.query_one("#snapshot-disabled-reason", Static).update(
                "Selected slot is empty — process context before saving."
            )
        elif (
            ready
            and self._snapshot_id
            and compatibility.get(self._snapshot_id) != "matching"
        ):
            self.query_one("#snapshot-disabled-reason", Static).update(
                "Restore unavailable: "
                + labels[compatibility.get(self._snapshot_id, "unknown")]
            )
        self.query_one("#snapshot-restore", Button).disabled = not (
            ready and compatibility.get(self._snapshot_id) == "matching"
        )
        self.query_one("#snapshot-delete", Button).disabled = self._snapshot_id is None
        for selector in ("#snapshot-enabled", "#snapshot-keep", "#snapshot-apply"):
            self.query_one(selector).disabled = self._preferences is None
        if self._preferences is None:
            self.query_one("#snapshot-disabled-reason", Static).update(
                PREFERENCES_UNAVAILABLE
            )
        self.query_one("#snapshot-previous", Button).disabled = len(self._offsets) == 1
        self.query_one("#snapshot-next", Button).disabled = (
            view.catalog.next_offset is None
        )
        observed = max((slot.observed_at for slot in view.slots), default=None)
        age = (
            "Unknown"
            if observed is None
            else f"{int(max(0, time.monotonic() - observed))}s ago"
        )
        record = next(
            (
                record
                for record in view.catalog.records
                if record.snapshot_id == self._snapshot_id
            ),
            None,
        )
        details = f"Updated {age}\nStored on this device: {token_copy(view.catalog.stored_bytes)} bytes · Residual: {token_copy(view.catalog.residual_bytes)} bytes"
        if (
            not self.query_one("#snapshot-details-panel", Collapsible).collapsed
            and view.storage_location
        ):
            details += "\nStorage location (read-only): " + view.storage_location
        if not view.catalog.scan_complete:
            details += "\nPartial scan — totals may be incomplete."
        if record:
            details += f"\n{record.model_label} · {record.tokens} tokens · {record.bytes} bytes · source slot {record.source_slot}"
        self.query_one("#snapshot-details", Static).update(details)
        self._paint_elapsed()

    @on(Collapsible.Expanded, "#snapshot-details-panel")
    @on(Collapsible.Collapsed, "#snapshot-details-panel")
    def _details_changed(self) -> None:
        self._paint()

    @on(DataTable.RowHighlighted, "#snapshot-slots")
    def _select_slot(self, event: DataTable.RowHighlighted) -> None:
        value = int(event.row_key.value)
        if (
            any(slot.slot_id == value for slot in self.service.view().slots)
            and value != self._slot_id
        ):
            self._slot_id = value
            self._paint()

    @on(DataTable.RowHighlighted, "#snapshot-records")
    def _select_record(self, event: DataTable.RowHighlighted) -> None:
        value = event.row_key.value
        if (
            any(
                record.snapshot_id == value
                for record in self.service.view().catalog.records
            )
            and value != self._snapshot_id
        ):
            self._snapshot_id = value
            self._paint()

    def on_key(self, event: Key) -> None:
        if isinstance(
            self.app.focused, (Button, DataTable)
        ) and self.app.focused in self.query("*"):
            action = {"s": "save", "r": "restore", "d": "delete", "f": "refresh"}.get(
                event.key
            )
            if action:
                event.stop()
                event.prevent_default()
                self.query_one(f"#snapshot-{action}", Button).press()

    @on(Button.Pressed)
    def _action(self, event: Button.Pressed) -> None:
        action = event.button.id or ""
        if not action.startswith("snapshot-"):
            return
        event.stop()
        if action == "snapshot-refresh":
            self.request_refresh()
        elif action == "snapshot-save":
            if self._preferences is None:
                return
            self._offsets = [0]
            try:
                self.service.start_save(self._slot_id)
            except SnapshotError as error:
                self.app.notify(error.code.replace("_", " "), severity="warning")
        elif action in {"snapshot-restore", "snapshot-delete"}:
            self._confirm(action == "snapshot-restore")
        elif action in {"snapshot-next", "snapshot-previous"}:
            if action == "snapshot-next":
                offset = self.service.view().catalog.next_offset
                if offset is None:
                    return
                self._offsets.append(offset)
            elif len(self._offsets) > 1:
                self._offsets.pop()
            self.app.run_worker(
                self.service.browse_catalog(self._offsets[-1]), exit_on_error=False
            )
        elif action in {"snapshot-apply", "snapshot-reload"}:
            self.app.run_worker(
                self._save_preferences(action == "snapshot-reload"),
                group="snapshot-preferences",
                exclusive=True,
                exit_on_error=False,
            )

    def _confirm(self, restore: bool) -> None:
        view = self.service.view()
        record = next(
            (
                record
                for record in view.catalog.records
                if record.snapshot_id == self._snapshot_id
            ),
            None,
        )
        if record is None or self._confirming:
            return
        attachment, slot_id, launch_id = self._attachment, self._slot_id, view.launch_id
        self._confirming = True
        message = (
            f"Restore {local_time(record.created_utc)} into slot {slot_id}? This replaces its processed context. A restore failure may clear the destination slot. Conversations are unchanged."
            if restore
            else f"Permanently delete {local_time(record.created_utc)} ({record.bytes} bytes)? This cannot be undone."
        )

        def confirmed(accepted: bool) -> None:
            self._confirming = False
            current = self.service.view()
            if (
                not accepted
                or attachment is not self._attachment
                or not self.is_mounted
            ):
                return
            if self._snapshot_id != record.snapshot_id or not any(
                item == record for item in current.catalog.records
            ):
                self.app.notify(
                    "Snapshot selection changed; select it again.", severity="warning"
                )
                return
            if restore:
                if self._preferences is None:
                    return
                if current.launch_id != launch_id or self._slot_id != slot_id:
                    self.app.notify(
                        "Destination changed; select it again.", severity="warning"
                    )
                    return
                try:
                    self.service.start_restore(record.snapshot_id, slot_id)
                except SnapshotError as error:
                    self.app.notify(error.code.replace("_", " "), severity="warning")
            else:
                self._offsets = [0]
                self.app.run_worker(
                    self.service.delete_snapshot(record.snapshot_id),
                    exit_on_error=False,
                )

        self.app.push_screen(
            ConfirmationDialog(
                title="Restore snapshot" if restore else "Delete snapshot",
                message=message,
                confirm_label="Restore" if restore else "Delete",
            ),
            confirmed,
        )

    async def _save_preferences(self, reload: bool) -> None:
        attachment = self._attachment
        try:
            if reload:
                value = await asyncio.to_thread(preferences.load_snapshot_preferences)
            else:
                if self._preferences is None:
                    return
                value = preferences.SnapshotPreferences(
                    enabled=self.query_one("#snapshot-enabled", Checkbox).value,
                    keep_count=int(self.query_one("#snapshot-keep", Input).value),
                )
                if not await asyncio.to_thread(
                    preferences.save_snapshot_preferences,
                    value,
                    expected=self._preferences,
                ):
                    raise ValueError("save failed")
            if attachment is self._attachment and self.is_mounted:
                self._set_loaded_preferences(value)
                self.query_one("#snapshot-preferences-result", Static).update(
                    "Saved. Enable/disable applies on next launch."
                )
                self._paint()
        except preferences.SnapshotPreferencesConflict:
            if attachment is self._attachment and self.is_mounted:
                self.query_one("#snapshot-preferences-result", Static).update(
                    "Preferences changed elsewhere. Reload before applying."
                )
        except (ValueError, OSError):
            if attachment is self._attachment and self.is_mounted:
                if reload:
                    self._set_loaded_preferences(None)
                    self._paint()
                    return
                self.query_one("#snapshot-preferences-result", Static).update(
                    "Not saved. Use a keep count from 1 to 1000; check config access."
                )
