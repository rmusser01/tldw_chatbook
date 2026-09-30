"""The canonical Settings draft editor for standalone Console hooks."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, get_args
from uuid import uuid4

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    OptionList,
    Select,
    Static,
    TextArea,
)

from tldw_chatbook.UI.Screens.settings_config_models import SettingsDraft
from tldw_chatbook.Utils.input_validation import (
    CONSOLE_HOOK_DEFAULT_TIMEOUT_S as HOOK_DEFAULT_TIMEOUT_S,
)
from tldw_chatbook.Utils.input_validation import (
    ConsoleHookEvent,
)

if TYPE_CHECKING:
    from tldw_chatbook.Agents.hook_permissions import HookReviewSnapshot

HOOK_EVENTS = frozenset(get_args(ConsoleHookEvent))


class HooksSettingsPanel(Vertical):
    """Edit one shared SettingsDraft. Persistence stays with SettingsScreen."""

    class Requested(Message):
        def __init__(self, action: str) -> None:
            self.action = action
            super().__init__()

    def __init__(
        self, draft: SettingsDraft, snapshot: HookReviewSnapshot | None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.add_class("settings-hooks-panel")
        self.draft = draft
        self.snapshot = snapshot
        self.selected = 0
        self._display: dict[str, object] = {}

    def on_mount(self) -> None:
        self.post_message(self.Requested("load"))

    @property
    def section(self) -> object:
        return self.draft.values.get("section")

    def load(self, snapshot: HookReviewSnapshot, *, reset: bool = False) -> None:
        if not reset and "section" in self.draft.values and self.snapshot == snapshot:
            return
        self.snapshot = snapshot
        if reset or not self.draft.is_dirty or "section" not in self.draft.values:
            section = (
                copy.deepcopy(snapshot.config.section)
                if snapshot.config.section_present
                else {"enabled": False, "hook": []}
            )
            count = (
                len(section.get("hook", []))
                if isinstance(section, dict)
                and isinstance(section.get("hook", []), list)
                else 0
            )
            self.draft.originals = {
                "section": copy.deepcopy(section),
                "origins": list(range(count)),
            }
            self.draft.values = {"section": section, "origins": list(range(count))}
        self.refresh(recompose=True)

    def _rows(self) -> list:
        section = self.section
        return (
            section.get("hook", [])
            if isinstance(section, dict) and isinstance(section.get("hook", []), list)
            else []
        )

    def compose(self) -> ComposeResult:
        from tldw_chatbook.Agents.run_hooks import inspect_hooks_config

        yield Static("Hooks", classes="destination-section")
        yield Static(
            "External commands for Console events. Edits stay here until Save. Review applies to saved definitions.",
            classes="settings-hooks-copy",
            markup=False,
        )
        with Horizontal(classes="settings-hooks-actions"):
            yield Button(
                "Save",
                id="settings-hooks-save",
                variant="primary",
                classes="settings-hooks-action",
            )
            yield Button(
                "Revert", id="settings-hooks-revert", classes="settings-hooks-action"
            )
        yield Button(
            f"Review saved hooks · {self.snapshot.pending_count if self.snapshot else 0}",
            id="settings-hooks-review",
            classes="settings-hooks-action",
        )
        yield Static(
            self.snapshot.notice
            or self.snapshot.blocked_reason
            or "Saved hooks are ready."
            if self.snapshot
            else "Loading saved hooks…",
            id="settings-hooks-status",
            classes="settings-hooks-copy",
            markup=False,
        )
        if "section" not in self.draft.values:
            return
        section = self.section
        if not isinstance(section, dict) or not isinstance(
            section.get("hook", []), list
        ):
            yield Static(
                "Malformed Hooks section. Repair it in Advanced Config; this editor keeps the original intact.",
                markup=False,
            )
            yield Button(
                "Advanced Config",
                id="settings-hooks-advanced",
                classes="settings-hooks-action",
            )
            return
        yield Checkbox(
            "Enable Console hooks",
            value=section.get("enabled", True) is True,
            id="settings-hooks-enabled",
        )
        with Horizontal(classes="settings-hooks-actions"):
            yield Button(
                "Add hook", id="settings-hooks-add", classes="settings-hooks-action"
            )
            yield Button(
                "Remove selected",
                id="settings-hooks-remove",
                disabled=not self._rows(),
                classes="settings-hooks-action",
            )
        rows = self._rows()
        inventory = inspect_hooks_config({"hooks": section})
        labels = []
        for index, raw in enumerate(rows):
            origin = self.draft.values["origins"][index]
            original = self.draft.originals["section"].get("hook", [])
            unchanged = (
                origin is not None
                and origin < len(original)
                and raw == original[origin]
            )
            state = (
                next(
                    (
                        r.state
                        for r in self.snapshot.rows
                        if r.entry
                        and r.entry.key == inventory.rows[index].key
                        and r.entry.spec == inventory.rows[index].spec
                    ),
                    "Unsaved",
                )
                if self.snapshot and unchanged
                else "Unsaved"
            )
            label = (
                raw.get("name") or raw.get("event", "Invalid")
                if isinstance(raw, dict)
                else "Invalid entry"
            )
            if (
                inventory.rows[index].spec
                and isinstance(raw, dict)
                and not raw.get("name")
            ):
                label += " · " + raw["command"][0].rsplit("/", 1)[-1]
            issue = inventory.rows[index].error
            labels.append(
                f"{index + 1}. {json.dumps(str(label), ensure_ascii=True)} · {state}{' · Invalid' if issue else ''}"
            )
        self.selected = min(self.selected, max(0, len(rows) - 1))
        options = OptionList(*labels, id="settings-hooks-list", markup=False)
        options.highlighted = self.selected if rows else None
        yield options
        if not rows:
            yield Static(
                "No hooks configured. Events: " + ", ".join(sorted(HOOK_EVENTS)) + ".",
                classes="settings-hooks-copy",
            )
            return
        raw = rows[self.selected]
        if not isinstance(raw, dict):
            yield Static(
                "This entry is not a table. Remove it explicitly or repair in Advanced Config.",
                markup=False,
            )
            return
        self._display = {
            "event": str(raw.get("event", "")),
            "command": json.dumps(raw.get("command", []), ensure_ascii=True, indent=2),
            "matcher": str(raw.get("matcher", "")),
            "timeout_s": str(raw.get("timeout_s", HOOK_DEFAULT_TIMEOUT_S)),
            "enabled": raw.get("enabled", True) is True,
        }
        with Vertical(id="settings-hooks-editor"):
            yield Static(f"Hook {self.selected + 1}", classes="destination-section")
            yield Checkbox(
                "Enable hook",
                value=self._display["enabled"],
                id="settings-hooks-row-enabled",
            )
            options = [(event, event) for event in sorted(HOOK_EVENTS)]
            if self._display["event"] not in HOOK_EVENTS:
                options.append(("Invalid saved event", self._display["event"]))
            yield Select(
                options,
                value=self._display["event"],
                allow_blank=False,
                id="settings-hooks-event",
                classes="settings-hooks-field",
            )
            yield Static(
                "Command · JSON argument array (no shell)",
                classes="settings-hooks-copy",
            )
            yield TextArea(str(self._display["command"]), id="settings-hooks-command")
            yield Static(
                "Tool matcher · optional glob for PreToolUse / PostToolUse",
                classes="settings-hooks-copy",
            )
            yield Input(
                str(self._display["matcher"]),
                id="settings-hooks-matcher",
                classes="settings-hooks-field",
            )
            yield Static("Timeout · seconds", classes="settings-hooks-copy")
            yield Input(
                str(self._display["timeout_s"]),
                id="settings-hooks-timeout",
                classes="settings-hooks-field",
            )
            yield Static(
                inventory.rows[self.selected].error
                or "Saving a command change requires review before the next Send.",
                id="settings-hooks-validation",
                classes="settings-hooks-copy",
                markup=False,
            )

    def capture(self) -> None:
        """Flush current controls without changing untouched raw values."""
        from tldw_chatbook.Agents.run_hooks import inspect_hooks_config

        if (
            not self._rows()
            or not isinstance(self._rows()[self.selected], dict)
            or not self.query("#settings-hooks-editor")
        ):
            return
        raw = self._rows()[self.selected]
        values = {
            "event": self.query_one("#settings-hooks-event", Select).value,
            "command": self.query_one("#settings-hooks-command", TextArea).text,
            "matcher": self.query_one("#settings-hooks-matcher", Input).value,
            "timeout_s": self.query_one("#settings-hooks-timeout", Input).value,
            "enabled": self.query_one("#settings-hooks-row-enabled", Checkbox).value,
        }
        for key, value in values.items():
            if value == self._display.get(key):
                continue
            if key == "command":
                try:
                    raw[key] = json.loads(str(value))
                except ValueError:
                    raw[key] = value
            elif key == "timeout_s":
                try:
                    raw[key] = float(str(value))
                except ValueError:
                    raw[key] = value
            elif key == "matcher" and value == "":
                raw.pop(key, None)
            else:
                raw[key] = value
        self._display = values
        inventory = inspect_hooks_config({"hooks": self.section})
        self.query_one("#settings-hooks-validation", Static).update(
            inventory.rows[self.selected].error
            or "Changes are staged. Save before reviewing."
        )

    def submission(self) -> tuple[dict, dict[str, str]]:
        """Validate edited rows and pin unchanged legacy sources for ID assignment."""
        from tldw_chatbook.Agents.run_hooks import (
            fingerprint_hook,
            inspect_hooks_config,
        )

        self.capture()
        section = copy.deepcopy(self.section)
        if not isinstance(section, dict) or not isinstance(
            section.get("hook", []), list
        ):
            raise ValueError("Repair malformed Hooks section in Advanced Config.")  # noqa: TRY004 -- user-facing validation result
        current = inspect_hooks_config({"hooks": section})
        original = self.draft.originals["section"]
        old_rows = original.get("hook", []) if isinstance(original, dict) else []
        origins = self.draft.values["origins"]
        for row in current.rows:
            raw = section["hook"][row.index]
            origin = origins[row.index]
            old = (
                old_rows[origin]
                if origin is not None and origin < len(old_rows)
                else None
            )
            disabled_only = (
                isinstance(raw, dict)
                and isinstance(old, dict)
                and raw.get("enabled") is False
                and {k: v for k, v in raw.items() if k != "enabled"}
                == {k: v for k, v in old.items() if k != "enabled"}
            )
            if row.error and raw != old and not disabled_only:
                raise ValueError(f"Hook {row.index + 1}: {row.error}")
        if current.container_error:
            raise ValueError(current.container_error)
        legacy_ids = {}
        original_inventory = inspect_hooks_config({"hooks": original})
        old_entries = {r.index: r for r in original_inventory.rows}
        for row in current.rows:
            raw = section["hook"][row.index]
            if row.spec and isinstance(raw, dict) and "id" not in raw:
                raw["id"] = str(uuid4())
                prior = old_entries.get(origins[row.index])
                if (
                    prior
                    and prior.spec
                    and fingerprint_hook(prior.spec) == fingerprint_hook(row.spec)
                ):
                    legacy_ids[prior.key] = raw["id"]
        return section, legacy_ids

    @on(OptionList.OptionSelected, "#settings-hooks-list")
    def select_row(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.capture()
        self.selected = event.option_index
        self.refresh(recompose=True)

    @on(Checkbox.Changed)
    @on(Input.Changed)
    @on(Select.Changed)
    @on(TextArea.Changed)
    def edited(self, event: Message) -> None:
        event.stop()
        if (
            getattr(getattr(event, "control", None), "id", None)
            == "settings-hooks-enabled"
        ):
            if event.value != (self.section.get("enabled", True) is True):
                self.section["enabled"] = event.value
        else:
            self.capture()
        self.post_message(self.Requested("edited"))

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        event.stop()
        action = (event.button.id or "").removeprefix("settings-hooks-")
        if action in {"add", "remove"}:
            self.capture()
            section = self.section
            if action == "add":
                section.setdefault("hook", []).append(
                    {
                        "id": str(uuid4()),
                        "enabled": False,
                        "event": "UserPromptSubmit",
                        "command": [],
                        "timeout_s": HOOK_DEFAULT_TIMEOUT_S,
                    }
                )
                self.draft.values["origins"].append(None)
                self.selected = len(self._rows()) - 1
            elif self._rows():
                self._rows().pop(self.selected)
                self.draft.values["origins"].pop(self.selected)
                self.selected = min(self.selected, max(0, len(self._rows()) - 1))
            self.refresh(recompose=True)
            self.post_message(self.Requested("edited"))
        else:
            self.post_message(self.Requested(action))
