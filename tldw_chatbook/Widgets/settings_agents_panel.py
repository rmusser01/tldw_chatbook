"""Settings ▸ Agents: CRUD editor for named sub-agent definitions.

Edits the AgentRuns DB directly (immediate CRUD) — unlike TOML-backed
Settings categories there is no draft/Save-with-`s` cycle; each Save/Delete
applies at once. Fleet spec §4.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Input,
    ListItem,
    ListView,
    Select,
    Static,
    Switch,
    TextArea,
)

from tldw_chatbook.Agents.agent_models import (
    RUNTIME_TOOL_NAMES,
    AgentDefinition,
)
from tldw_chatbook.Agents.agent_presets import AGENT_PRESETS
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

#: Soft ceiling before the status line warns about spawn-schema bloat
#: (spec §4: every enabled definition rides the spawn tool's schema).
ENABLED_DEFINITIONS_SOFT_CAP = 20


def _parse_requested_tools(raw: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split ordered tool names into stored and runtime-only selections."""

    requested = tuple(
        dict.fromkeys(name.strip() for name in raw.split(",") if name.strip())
    )
    stored = tuple(name for name in requested if name not in RUNTIME_TOOL_NAMES)
    omitted = tuple(name for name in requested if name in RUNTIME_TOOL_NAMES)
    return stored, omitted


def _derive_runs_db(app_instance) -> AgentRunsDB | None:
    """Same derivation as UI/Console_Modules/agent.py:337 — the runs DB
    lives next to the ChaChaNotes file; a :memory: ChaChaNotes (tests,
    ephemeral) means no durable definitions store."""
    db = getattr(app_instance, "chachanotes_db", None)
    db_path = getattr(db, "db_path", None) if db is not None else None
    if not db_path or str(db_path) == ":memory:":
        return None
    try:
        return AgentRunsDB(Path(db_path).parent / "agent_runs.db")
    except Exception as exc:  # noqa: BLE001 - any failure means "no DB"
        logger.warning(
            "Settings ▸ Agents: could not open agent runs database (error_type={})",
            type(exc).__name__,
        )
        return None


class AgentsSettingsPanel(Vertical):
    """List + form editor over the agent_definitions table."""

    def __init__(self, app_instance, runs_db: AgentRunsDB | None = None, **kwargs):
        super().__init__(**kwargs)
        self._runs_db = (
            runs_db if runs_db is not None else _derive_runs_db(app_instance)
        )
        self._owns_runs_db = runs_db is None and self._runs_db is not None
        self._selected_id: str | None = None
        self._rows: list[dict] = []

    def compose(self) -> ComposeResult:
        if self._runs_db is None:
            yield Static(
                "Agent definitions need a saved (non-temporary) profile "
                "database; none is available in this session.",
                id="agents-no-db-notice",
                classes="settings-detail-row",
            )
            return
        yield Static(
            "Named sub-agents the Console supervisor can spawn. Changes "
            "apply immediately (stored in agent_runs.db, not config.toml) "
            "and take effect on the next reply.",
            classes="settings-detail-row",
        )
        yield ListView(id="agents-definition-list")
        with VerticalScroll(id="agents-form"):
            with Horizontal(classes="settings-input-row agents-field-row"):
                yield Static("Name", classes="settings-input-label")
                yield Input(
                    placeholder="researcher (lowercase slug)",
                    id="agents-name-input",
                    classes="settings-compact-input",
                )
            with Horizontal(classes="settings-input-row agents-field-row"):
                yield Static("Description", classes="settings-input-label")
                yield Input(
                    placeholder="One line the supervisor reads (max 200 chars)",
                    id="agents-description-input",
                    classes="settings-compact-input",
                )
            yield Static(
                "Instructions (appended to the sub-agent prompt)",
                classes="settings-input-label",
            )
            yield TextArea(id="agents-instructions-area")
            with Horizontal(classes="settings-input-row agents-field-row"):
                yield Static("Model override", classes="settings-input-label")
                yield Input(
                    placeholder="empty = parent's model (same provider)",
                    id="agents-model-input",
                    classes="settings-compact-input",
                )
            with Horizontal(classes="settings-input-row agents-field-row"):
                yield Static("Child time cap (seconds)", classes="settings-input-label")
                yield Input(
                    placeholder="empty = existing child limit",
                    id="agents-wall-seconds-input",
                    classes="settings-compact-input",
                )
            with Horizontal(classes="settings-input-row agents-field-row"):
                yield Static(
                    "Tools (comma-separated; empty = inherit all; names "
                    "only narrow, never grant)",
                    classes="settings-input-label",
                )
                yield Input(id="agents-tools-input", classes="settings-compact-input")
            with Horizontal(classes="settings-input-row agents-enabled-row"):
                yield Static("Enabled", classes="settings-input-label")
                yield Switch(value=True, id="agents-enabled-switch")
            with Horizontal(
                classes="settings-input-row settings-select-row agents-preset-row"
            ):
                yield Static("Preset", classes="settings-input-label")
                yield Select(
                    [(preset.name, preset.name) for preset in AGENT_PRESETS],
                    value=AGENT_PRESETS[0].name,
                    allow_blank=False,
                    id="agents-preset-select",
                    classes="settings-compact-select",
                )
                yield Button("Load preset", id="agents-load-preset-button")
            with Horizontal(classes="settings-input-row"):
                yield Button("New", id="agents-new-button")
                yield Button("Save", variant="primary", id="agents-save-button")
                yield Button("Delete", variant="error", id="agents-delete-button")
        yield Static(
            "", id="agents-status", classes="settings-detail-row", markup=False
        )

    async def on_mount(self) -> None:
        await self._reload_list()

    # -- list / selection -------------------------------------------------
    async def _reload_list(self) -> None:
        if self._runs_db is None:
            return
        lv = self.query_one("#agents-definition-list", ListView)
        # Await both the removal and the appends -- ListView.clear()/append()
        # return AwaitRemove/AwaitMount, not plain None; a fire-and-forget
        # call leaves a freshly-appended row un-laid-out (Region(0,0,0,0))
        # for up to a tick, so a click/select right after reload can miss it
        # (review finding, task-6 fix round 1).
        await lv.clear()
        self._rows = self._runs_db.list_agent_definitions()
        for row in self._rows:
            marker = "" if row["enabled"] else " (disabled)"
            await lv.append(
                ListItem(Static(f"{row['name']}{marker}", markup=False), name=row["id"])
            )
        warning = self._enabled_count_warning()
        if warning:
            self._set_status(warning)

    def on_unmount(self) -> None:
        if self._owns_runs_db:
            self._owns_runs_db = False
            db, self._runs_db = self._runs_db, None
            if db is not None:
                db.close()

    def _enabled_count_warning(self) -> str:
        """Return the spawn-schema warning for the current loaded rows."""

        enabled_count = sum(1 for row in self._rows if row["enabled"])
        if enabled_count <= ENABLED_DEFINITIONS_SOFT_CAP:
            return ""
        return (
            f"{enabled_count} enabled definitions — every one rides the "
            "spawn schema each turn; consider disabling some."
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        definition_id = event.item.name
        row = next((r for r in self._rows if r["id"] == definition_id), None)
        if row is None:
            return
        self._selected_id = definition_id
        self.query_one("#agents-name-input", Input).value = row["name"]
        self.query_one("#agents-description-input", Input).value = row["description"]
        self.query_one("#agents-instructions-area", TextArea).text = row["instructions"]
        self.query_one("#agents-model-input", Input).value = row["model"]
        wall_seconds = row.get("max_wall_seconds")
        self.query_one("#agents-wall-seconds-input", Input).value = (
            "" if wall_seconds is None else str(wall_seconds)
        )
        self.query_one("#agents-tools-input", Input).value = ", ".join(
            row["tool_allowlist"]
        )
        self.query_one("#agents-enabled-switch", Switch).value = bool(row["enabled"])

    # -- buttons ----------------------------------------------------------
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agents-new-button":
            self._clear_form()
        elif event.button.id == "agents-load-preset-button":
            selected_name = self.query_one("#agents-preset-select", Select).value
            preset = next(
                preset for preset in AGENT_PRESETS if preset.name == selected_name
            )
            self._load_preset(preset)
        elif event.button.id == "agents-save-button":
            await self._save()
        elif event.button.id == "agents-delete-button":
            await self._delete()

    def _clear_form(self) -> None:
        self._selected_id = None
        self.query_one("#agents-name-input", Input).value = ""
        self.query_one("#agents-description-input", Input).value = ""
        self.query_one("#agents-instructions-area", TextArea).text = ""
        self.query_one("#agents-model-input", Input).value = ""
        self.query_one("#agents-wall-seconds-input", Input).value = ""
        self.query_one("#agents-tools-input", Input).value = ""
        self.query_one("#agents-enabled-switch", Switch).value = True
        self._set_status("")

    def _load_preset(self, preset: AgentDefinition) -> None:
        self._selected_id = None
        name_input = self.query_one("#agents-name-input", Input)
        name_input.value = preset.name
        description_input = self.query_one("#agents-description-input", Input)
        description_input.value = preset.description
        self.query_one("#agents-instructions-area", TextArea).text = preset.instructions
        model_input = self.query_one("#agents-model-input", Input)
        model_input.value = ""
        self.query_one("#agents-wall-seconds-input", Input).value = ""
        tools_input = self.query_one("#agents-tools-input", Input)
        tools_input.value = ", ".join(preset.tool_allowlist)
        self.query_one("#agents-enabled-switch", Switch).value = preset.enabled
        if preset.name == "bulk-reader":
            self._set_status(
                "Choose a cheaper model from the same provider, then Save."
            )
        else:
            self._set_status("Editable preset loaded; choose Save to create it.")

    def _form_definition(
        self, *, stored_tools: tuple[str, ...] | None = None
    ) -> AgentDefinition:
        if stored_tools is None:
            stored_tools, _ = _parse_requested_tools(
                self.query_one("#agents-tools-input", Input).value
            )
        raw_wall_seconds = self.query_one(
            "#agents-wall-seconds-input", Input
        ).value.strip()
        try:
            max_wall_seconds = None if not raw_wall_seconds else float(raw_wall_seconds)
        except ValueError as exc:
            raise ValueError("Child time cap (seconds) must be a number.") from exc
        return AgentDefinition(
            name=self.query_one("#agents-name-input", Input).value.strip(),
            description=self.query_one(
                "#agents-description-input", Input
            ).value.strip(),
            instructions=self.query_one(
                "#agents-instructions-area", TextArea
            ).text.strip(),
            tool_allowlist=stored_tools,
            model=self.query_one("#agents-model-input", Input).value.strip(),
            enabled=self.query_one("#agents-enabled-switch", Switch).value,
            max_wall_seconds=max_wall_seconds,
        )

    async def _save(self) -> None:
        stored_tools, omitted_runtime_tools = _parse_requested_tools(
            self.query_one("#agents-tools-input", Input).value
        )
        try:
            defn = self._form_definition(stored_tools=stored_tools)
            if self._selected_id is None:
                self._runs_db.create_agent_definition(defn)
            else:
                self._runs_db.update_agent_definition(self._selected_id, defn)
        except (ValueError, sqlite3.Error) as exc:
            # A locked/corrupt agent_runs.db must surface as a status-line
            # message, not an uncaught exception that would crash the
            # Settings screen's compose (compose-exception lesson: a crash
            # there kills navigation for the whole app).
            message = str(exc).replace("max_wall_seconds", "Child time cap (seconds)")
            self._set_status(message)
            return
        notice = f"Saved '{defn.name}'."
        if omitted_runtime_tools:
            notice += (
                " Ignored runtime-only tools: " + ", ".join(omitted_runtime_tools) + "."
            )
            if not stored_tools:
                notice += " No tool filter remains; parent tools are inherited."
        await self._reload_list()
        warning = self._enabled_count_warning()
        self._set_status(" ".join(part for part in (notice, warning) if part))

    async def _delete(self) -> None:
        if self._selected_id is None:
            self._set_status("Select a definition to delete.")
            return
        try:
            self._runs_db.soft_delete_agent_definition(self._selected_id)
        except (ValueError, sqlite3.Error) as exc:
            self._set_status(str(exc))
            return
        self._clear_form()
        self._set_status("Deleted.")
        await self._reload_list()

    def _set_status(self, text: str) -> None:
        self.query_one("#agents-status", Static).update(text)
