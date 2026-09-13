"""Settings ▸ Agents: CRUD editor for named sub-agent definitions.

Edits the AgentRuns DB directly (immediate CRUD) — unlike TOML-backed
Settings categories there is no draft/Save-with-`s` cycle; each Save/Delete
applies at once. Fleet spec §4.

ADR-147 (TASK-32477 task 9): the preset form also edits the routing fields
(``provider`` / ``params``), and a routing block below the form edits the
four ``[agents]`` routing keys (sub-agent defaults + spawn-override policy)
and offers a "Test routing" dry-run over the pure resolver. The routing keys
persist through the atomic config writer; presets persist through the DB.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
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
    definition_from_row,
)
from tldw_chatbook.Agents.agent_presets import BULK_READER_PRESET
from tldw_chatbook.Agents.agent_routing import (
    AgentsRoutingConfig,
    RoutingError,
    load_agents_routing_config,
    resolve_spawn_target,
)
from tldw_chatbook.Chat.console_provider_support import (
    supported_console_provider_readiness_keys,
)
from tldw_chatbook.Chat.console_session_settings import (
    build_console_provider_options,
)
from tldw_chatbook.Chat.custom_endpoint_registry import (
    CUSTOM_ENDPOINT_ID_PREFIX,
    load_custom_endpoints,
    split_custom_endpoint_id,
)
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.Chat.sampling_params import (
    params_to_tuple,
    validate_sampling_params,
)
from tldw_chatbook.config import save_settings_to_cli_config
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

#: Soft ceiling before the status line warns about spawn-schema bloat
#: (spec §4: every enabled definition rides the spawn tool's schema).
ENABLED_DEFINITIONS_SOFT_CAP = 20

#: Prompt shown by the routing provider Selects for the empty (inherit) value.
INHERIT_PARENT_PROMPT = "(inherit parent)"

_INT_LITERAL = re.compile(r"[+-]?\d+")


def parse_params_text(text: str) -> tuple[dict[str, object], list[str]]:
    """Parse a ``key = value``-per-line sampling-params draft.

    Numeric literals become ``int``/``float``; anything else stays a string.
    Non-finite float literals (``inf``/``nan``) stay strings so validation
    rejects them as non-numbers rather than smuggling an un-JSON-able float
    into the DB or the config file.

    Args:
        text: The raw TextArea draft.

    Returns:
        ``(params, errors)``: the parsed mapping (insertion-ordered by line)
        and grammar errors only — validate the mapping itself with
        ``validate_sampling_params``. Empty ``errors`` means every non-blank
        line parsed.
    """
    params: dict[str, object] = {}
    errors: list[str] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        key, sep, raw_value = line.partition("=")
        key = key.strip()
        value = raw_value.strip()
        if not sep or not key:
            errors.append(f"params line {lineno}: expected 'key = value'")
            continue
        if _INT_LITERAL.fullmatch(value):
            params[key] = int(value)
            continue
        try:
            number = float(value)
        except ValueError:
            params[key] = value
        else:
            params[key] = number if math.isfinite(number) else value
    return params, errors


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

    def __init__(
        self,
        app_instance: Any | None,
        runs_db: AgentRunsDB | None = None,
        *,
        routing_readiness: Callable[[Mapping[str, Any], str], str | None] | None = None,
        **kwargs: Any,
    ):
        """Initialize the panel.

        Args:
            app_instance: The running app (its ``app_config`` feeds the
                provider options, the routing dry-run, and gates the
                ``[agents]`` config write); ``None`` in bare harnesses, where
                the routing controls still render but Save skips the config
                write (a bare harness must never touch the real config file).
            runs_db: Injectable definitions store (tests); derived from the
                app's ChaChaNotes path when omitted.
            routing_readiness: Injectable readiness probe for the "Test
                routing" dry-run (same seam as the resolver's own
                ``readiness=`` parameter); ``None`` uses real readiness.
        """
        super().__init__(**kwargs)
        self._app_instance = app_instance
        self._runs_db = (
            runs_db if runs_db is not None else _derive_runs_db(app_instance)
        )
        self._selected_id: str | None = None
        self._rows: list[dict] = []
        self._routing_readiness = routing_readiness

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
            with Horizontal(classes="settings-input-row"):
                yield Static("Name", classes="settings-input-label")
                yield Input(
                    placeholder="researcher (lowercase slug)",
                    id="agents-name-input",
                    classes="settings-compact-input",
                )
            with Horizontal(classes="settings-input-row"):
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
            with Horizontal(classes="settings-input-row"):
                yield Static("Model override", classes="settings-input-label")
                yield Input(
                    placeholder="empty = parent's model (same provider)",
                    id="agents-model-input",
                    classes="settings-compact-input",
                )
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static("Provider", classes="settings-input-label")
                yield Select(
                    self._provider_select_options(),
                    value=Select.NULL,
                    prompt=INHERIT_PARENT_PROMPT,
                    allow_blank=True,
                    compact=True,
                    id="agents-provider-select",
                    classes="settings-compact-select",
                )
            yield Static(
                "Params (one key = value per line; empty = inherit the "
                "resolved stack)",
                classes="settings-input-label",
            )
            yield TextArea(id="agents-params-area")
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    "Tools (comma-separated; empty = inherit all; names "
                    "only narrow, never grant)",
                    classes="settings-input-label",
                )
                yield Input(id="agents-tools-input", classes="settings-compact-input")
            with Horizontal(classes="settings-input-row"):
                yield Static("Enabled", classes="settings-input-label")
                yield Switch(value=True, id="agents-enabled-switch")
            with Horizontal(classes="settings-input-row"):
                yield Button("New", id="agents-new-button")
                yield Button("Bulk reader", id="agents-bulk-reader-button")
                yield Button("Save", variant="primary", id="agents-save-button")
                yield Button("Delete", variant="error", id="agents-delete-button")
            yield Static(
                "Routing — sub-agent defaults and spawn-override policy "
                "(stored in config.toml [agents], applied at spawn)",
                classes="settings-detail-row",
            )
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static("Default provider", classes="settings-input-label")
                yield Select(
                    self._provider_select_options(),
                    value=Select.NULL,
                    prompt=INHERIT_PARENT_PROMPT,
                    allow_blank=True,
                    compact=True,
                    id="agents-default-provider-select",
                    classes="settings-compact-select",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Default model", classes="settings-input-label")
                yield Input(
                    placeholder="empty = provider's configured model",
                    id="agents-default-model-input",
                    classes="settings-compact-input",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    "Allow ad-hoc spawn overrides", classes="settings-input-label"
                )
                yield Checkbox(value=False, id="agents-override-enabled-checkbox")
            yield Static(
                "Override allowlist (one per line: provider or "
                "provider/model-glob)",
                classes="settings-input-label",
            )
            yield TextArea(id="agents-override-allowlist-area")
            yield Static(
                "", id="agents-allowlist-warning", classes="settings-detail-row"
            )
            with Horizontal(classes="settings-input-row"):
                yield Button("Test routing", id="agents-test-routing-button")
            yield Static(
                "", id="agents-routing-report", classes="settings-detail-row"
            )
        yield Static("", id="agents-status", classes="settings-detail-row")

    async def on_mount(self) -> None:
        await self._reload_list()
        self._load_routing_controls()

    # -- app config / provider options ------------------------------------
    def _app_config(self) -> Mapping[str, Any]:
        """Return the app's config mapping (empty when harness-mounted)."""
        config = getattr(self._app_instance, "app_config", None)
        return config if isinstance(config, Mapping) else {}

    def _provider_select_options(self) -> list[tuple[str, str]]:
        """Build provider Select options via the shared Console builder so
        custom-ep entries appear by display_name (ADR-146/147)."""
        return [
            (option.label, option.value)
            for option in build_console_provider_options(
                {}, app_config=self._app_config()
            )
        ]

    def _set_select_provider(self, select: Select, provider: str) -> None:
        """Set a provider Select's value, keeping a stored-but-stale id
        selectable (flagged ``(missing)``) instead of raising or dropping it."""
        if not provider:
            select.value = Select.NULL
            return
        options = self._provider_select_options()
        if provider not in {value for _, value in options}:
            options = [*options, (f"{provider} (missing)", provider)]
            select.set_options(options)
        select.value = provider

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
            await lv.append(ListItem(Static(f"{row['name']}{marker}"), name=row["id"]))
        enabled_count = sum(1 for r in self._rows if r["enabled"])
        if enabled_count > ENABLED_DEFINITIONS_SOFT_CAP:
            self._set_status(
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
        self._set_select_provider(
            self.query_one("#agents-provider-select", Select),
            row.get("provider", ""),
        )
        self.query_one("#agents-params-area", TextArea).text = "\n".join(
            f"{key} = {value}"
            for key, value in definition_from_row(row).params
        )
        self.query_one("#agents-tools-input", Input).value = ", ".join(
            row["tool_allowlist"]
        )
        self.query_one("#agents-enabled-switch", Switch).value = bool(row["enabled"])

    # -- buttons ----------------------------------------------------------
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agents-new-button":
            self._clear_form()
        elif event.button.id == "agents-bulk-reader-button":
            self._load_bulk_reader_preset()
        elif event.button.id == "agents-save-button":
            await self._save()
        elif event.button.id == "agents-delete-button":
            await self._delete()
        elif event.button.id == "agents-test-routing-button":
            self._test_routing()

    def _clear_form(self) -> None:
        self._selected_id = None
        self.query_one("#agents-name-input", Input).value = ""
        self.query_one("#agents-description-input", Input).value = ""
        self.query_one("#agents-instructions-area", TextArea).text = ""
        self.query_one("#agents-model-input", Input).value = ""
        self.query_one("#agents-provider-select", Select).value = Select.NULL
        self.query_one("#agents-params-area", TextArea).text = ""
        self.query_one("#agents-tools-input", Input).value = ""
        self.query_one("#agents-enabled-switch", Switch).value = True
        self._set_status("")

    def _load_bulk_reader_preset(self) -> None:
        self._selected_id = None
        self.query_one("#agents-name-input", Input).value = BULK_READER_PRESET.name
        self.query_one(
            "#agents-description-input", Input
        ).value = BULK_READER_PRESET.description
        self.query_one(
            "#agents-instructions-area", TextArea
        ).text = BULK_READER_PRESET.instructions
        self.query_one("#agents-model-input", Input).value = ""
        self.query_one("#agents-tools-input", Input).value = ", ".join(
            BULK_READER_PRESET.tool_allowlist
        )
        self.query_one(
            "#agents-enabled-switch", Switch
        ).value = BULK_READER_PRESET.enabled
        self._set_status("Choose a cheaper model from the same provider, then Save.")

    def _form_definition(self) -> AgentDefinition:
        # dict.fromkeys dedupes while preserving first-seen order -- "a, a"
        # must not produce a tool_allowlist with a repeated entry (it feeds
        # definition_fingerprint's sorted() list, so a dupe there would be a
        # silent identity divergence from what was actually typed).
        tools = tuple(
            dict.fromkeys(
                name.strip()
                for name in self.query_one("#agents-tools-input", Input).value.split(
                    ","
                )
                if name.strip() and name.strip() not in RUNTIME_TOOL_NAMES
            )
        )
        params, param_errors = parse_params_text(
            self.query_one("#agents-params-area", TextArea).text
        )
        param_errors.extend(validate_sampling_params(params))
        if param_errors:
            # Raised here so _save's existing ValueError channel renders the
            # message and gates the whole save (preset row AND [agents] keys).
            raise ValueError("; ".join(param_errors))
        provider_value = self.query_one("#agents-provider-select", Select).value
        return AgentDefinition(
            name=self.query_one("#agents-name-input", Input).value.strip(),
            description=self.query_one(
                "#agents-description-input", Input
            ).value.strip(),
            instructions=self.query_one(
                "#agents-instructions-area", TextArea
            ).text.strip(),
            tool_allowlist=tools,
            model=self.query_one("#agents-model-input", Input).value.strip(),
            enabled=self.query_one("#agents-enabled-switch", Switch).value,
            provider=(
                "" if provider_value is Select.NULL else str(provider_value)
            ),
            params=params_to_tuple(params),
        )

    async def _save(self) -> None:
        try:
            defn = self._form_definition()
            if self._selected_id is None:
                self._runs_db.create_agent_definition(defn)
            else:
                self._runs_db.update_agent_definition(self._selected_id, defn)
        except (ValueError, sqlite3.Error) as exc:
            # A locked/corrupt agent_runs.db must surface as a status-line
            # message, not an uncaught exception that would crash the
            # Settings screen's compose (compose-exception lesson: a crash
            # there kills navigation for the whole app).
            self._set_status(str(exc))
            return
        routing_note = self._save_routing_config()
        status = f"Saved '{defn.name}'."
        if routing_note:
            status = f"{status} {routing_note}"
        self._set_status(status)
        await self._reload_list()

    # -- routing config (the four [agents] keys) ---------------------------
    def _load_routing_controls(self) -> None:
        """Populate the routing controls from the current [agents] config."""
        if self._runs_db is None:
            return
        routing = load_agents_routing_config()
        self._set_select_provider(
            self.query_one("#agents-default-provider-select", Select),
            routing.subagent_default_provider,
        )
        self.query_one(
            "#agents-default-model-input", Input
        ).value = routing.subagent_default_model
        self.query_one(
            "#agents-override-enabled-checkbox", Checkbox
        ).value = routing.spawn_override_enabled
        self.query_one(
            "#agents-override-allowlist-area", TextArea
        ).text = "\n".join(routing.spawn_override_allowlist)

    def _parse_allowlist(self, text: str) -> tuple[list[str], list[str]]:
        """Split the allowlist draft into ``(entries, problems)``.

        Each non-blank line is ``provider`` or ``provider/model-glob``; the
        provider part must be a known provider id or a live ``custom-ep:``
        slug. Problem entries are named verbatim so the warning can point at
        them — they are never silently dropped from the stored value (the
        whole allowlist write is skipped instead).
        """
        known_providers = set(supported_console_provider_readiness_keys())
        endpoints = load_custom_endpoints(self._app_config())
        entries: list[str] = []
        problems: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            provider_part, sep, glob = line.partition("/")
            provider_part = provider_part.strip()
            glob = glob.strip()
            if sep and not glob:
                problems.append(f"'{line}' (empty model glob)")
                continue
            if provider_part.startswith(CUSTOM_ENDPOINT_ID_PREFIX):
                slug = split_custom_endpoint_id(provider_part)
                if slug is None or slug not in endpoints:
                    problems.append(f"'{line}' (unknown endpoint slug)")
                    continue
            elif provider_config_key(provider_part) not in known_providers:
                problems.append(f"'{line}' (unknown provider id)")
                continue
            entries.append(f"{provider_part}/{glob}" if sep else provider_part)
        return entries, problems

    def _save_routing_config(self) -> str | None:
        """Persist the routing controls to the four ``[agents]`` config keys.

        Returns:
            A status-line note (config write failed or allowlist skipped),
            or ``None`` when everything persisted — including when the panel
            has no app config context (bare harness), where the write is
            skipped entirely so tests can never touch the real config file.
        """
        warning = self.query_one("#agents-allowlist-warning", Static)
        warning.update("")
        if self._app_instance is None:
            return None
        entries, problems = self._parse_allowlist(
            self.query_one("#agents-override-allowlist-area", TextArea).text
        )
        default_provider_value = self.query_one(
            "#agents-default-provider-select", Select
        ).value
        values: dict[str, object] = {
            "subagent_default_provider": (
                ""
                if default_provider_value is Select.NULL
                else str(default_provider_value)
            ),
            "subagent_default_model": self.query_one(
                "#agents-default-model-input", Input
            ).value.strip(),
            "spawn_override_enabled": bool(
                self.query_one("#agents-override-enabled-checkbox", Checkbox).value
            ),
        }
        if not problems:
            values["spawn_override_allowlist"] = entries
        note = None
        if not save_settings_to_cli_config({"agents": values}):
            note = "Could not write the [agents] routing keys to config.toml."
        if problems:
            warning.update(
                "Override allowlist NOT saved — fix or remove: "
                + "; ".join(problems)
            )
        return note

    # -- "Test routing" dry-run --------------------------------------------
    def _test_routing(self) -> None:
        """Dry-run the spawn resolver for every enabled preset plus the
        configured default. Pure resolution — no child is spawned."""
        routing = load_agents_routing_config()
        app_config = self._app_config()
        lines = [
            self._routing_report_line(preset.name, app_config, routing, preset=preset)
            for row in self._rows
            if row["enabled"]
            for preset in [definition_from_row(row)]
        ]
        lines.append(
            self._routing_report_line("(default)", app_config, routing, preset=None)
        )
        self.query_one("#agents-routing-report", Static).update("\n".join(lines))

    def _routing_report_line(
        self,
        label: str,
        app_config: Mapping[str, Any],
        routing: AgentsRoutingConfig,
        *,
        preset: AgentDefinition | None,
    ) -> str:
        """Render one dry-run line: ``name -> provider / model — ready`` or
        ``name -> [code] message``; inherit-fallthroughs are reported as such."""
        try:
            target = resolve_spawn_target(
                app_config,
                parent_provider="",
                parent_model="",
                preset=preset,
                routing=routing,
                readiness=self._routing_readiness,
            )
        except RoutingError as exc:
            if exc.level == "inherit":
                return f"{label} -> inherit parent's endpoint at spawn"
            return f"{label} -> [{exc.code}] {exc}"
        return f"{label} -> {target.provider} / {target.model} — ready"

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
