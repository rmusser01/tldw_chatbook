"""Explicit plugin management with retained nonsecret drafts and opaque reviews."""

from __future__ import annotations

import json
from functools import partial

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Input,
    OptionList,
    Select,
    SelectionList,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option

from ..UI.Plugins_Modules.details import display_text
from ..UI.Plugins_Modules.operations import app_operations
from ..UI.Plugins_Modules.review import (
    PluginConnectionDraft,
    PluginReviewDraft,
    review_text,
)
from .password_dialog import PasswordDialog


def _configuration_references(details):
    """Project every closed reviewed row, or refuse the whole prefill."""
    import hashlib

    from ..Plugins.host_references import (
        _reference_form,
        declared_tools,
        tool_constraint_id,
    )
    from ..Plugins.host_references import (
        mapping_id as tool_mapping_id,
    )
    from ..Plugins.variables import consumed, mapping_id, reference

    inspection = details["inspection"]
    installation_id = details["installation"]["installation_id"]
    rows = details["mappings"]
    if not isinstance(rows, (tuple, list)):
        raise ValueError(  # noqa: TRY004 -- stable refusal API
            "plugin_configuration_unavailable"
        )
    result = {
        key: {}
        for key in (
            "connections",
            "tools",
            "models",
            "component_tools",
            "tool_references",
            "variables",
        )
    }
    declarations = json.loads(inspection.variables_json)
    by_id, forms, labels = {}, {}, set()
    fields = {
        "installation_id",
        "revision_digest",
        "component_id",
        "mapping_id",
        "kind",
        "target_reference",
        "definition_digest",
        "configuration_digest",
        "credential_bindings",
    }
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != fields
            or row["installation_id"] != installation_id
            or row["revision_digest"] != inspection.effective_digest
            or row["component_id"] not in inspection.inventory
            or not isinstance(row["mapping_id"], str)
            or not isinstance(row["target_reference"], str)
            or row["mapping_id"] in by_id
        ):
            raise ValueError("plugin_configuration_identity_invalid")
        identity = row["mapping_id"]
        by_id[identity] = row
        forms[identity] = (
            "variable"
            if identity.startswith("variable:")
            else _reference_form(row, inspection)
        )

    # A tool spelling is reconstructed only after joining its actual connection.
    connections = {}
    for identity, row in by_id.items():
        if forms[identity] != "mcp" or row["kind"] != "connection":
            continue
        component = row["component_id"]
        target = row["target_reference"]
        if (
            identity != component
            or component in connections
            or not target.startswith("local:")
            or not target[6:]
        ):
            raise ValueError("plugin_configuration_connection_invalid")
        connections[component] = row
        result["connections"][component] = target[6:]

    def tool_target(row):
        if forms.get(row["mapping_id"]) != "mcp" or row["kind"] != "tool":
            raise ValueError("plugin_configuration_tool_invalid")
        connection = connections.get(row["component_id"])
        if connection is None:
            raise ValueError("plugin_configuration_connection_required")
        prefix = connection["target_reference"] + "::"
        target = row["target_reference"]
        if not target.startswith(prefix) or not target[len(prefix) :]:
            raise ValueError("plugin_configuration_tool_invalid")
        name = target[len(prefix) :]
        expected = (
            "tool-"
            + hashlib.sha256(
                (connection["mapping_id"] + "\0" + name).encode()
            ).hexdigest()
        )
        if row["mapping_id"] != expected:
            raise ValueError("plugin_configuration_tool_invalid")
        return row["component_id"] + "::" + name, name

    for identity, row in by_id.items():
        form = forms[identity]
        component_id, target = row["component_id"], row["target_reference"]
        component = inspection.inventory[component_id]
        if form == "variable":
            names = [
                name
                for name in consumed(component)
                if name in declarations and mapping_id(component_id, name) == identity
            ]
            if row["kind"] != "configuration" or len(names) != 1:
                raise ValueError("plugin_configuration_variable_invalid")
            reference(target)
            name = names[0]
            previous = result["variables"].get(name)
            if previous is not None and previous != target:
                raise ValueError("plugin_configuration_variable_ambiguous")
            result["variables"][name] = target
        elif form == "mcp":
            if row["kind"] == "tool":
                _target, name = tool_target(row)
                result["tools"].setdefault(component_id, []).append(name)
        elif form in {"canonical_model", "dev_model"}:
            if component_id in result["models"]:
                raise ValueError("plugin_configuration_model_ambiguous")
            result["models"][component_id] = target
        elif form in {"canonical_tool", "dev_tool"}:
            names = [
                name
                for name in declared_tools(component) or ()
                if identity
                == (
                    tool_constraint_id(component_id, name)
                    if form == "canonical_tool"
                    else tool_mapping_id(component_id, name)
                )
            ]
            if len(names) != 1 or (component_id, names[0]) in labels:
                raise ValueError("plugin_configuration_alias_ambiguous")
            labels.add((component_id, names[0]))
            if form == "dev_tool" and target.startswith("builtin:"):
                if not target[len("builtin:") :]:
                    raise ValueError("plugin_configuration_alias_invalid")
                projected = target
            else:
                sources = (
                    ([by_id[target]] if target in by_id else [])
                    if form == "canonical_tool"
                    else [
                        source
                        for source in rows
                        if forms[source["mapping_id"]] == "mcp"
                        and source["kind"] == "tool"
                        and source["target_reference"] == target
                    ]
                )
                if len(sources) != 1:
                    raise ValueError("plugin_configuration_alias_ambiguous")
                projected, _name = tool_target(sources[0])
            field = "component_tools" if form == "canonical_tool" else "tool_references"
            result[field].setdefault(component_id, {})[names[0]] = projected
        else:
            raise ValueError("plugin_configuration_identity_invalid")
    return result


class PluginVariableDialog(ModalScreen):
    """Typed entry is transient; only protected reference metadata leaves this dialog."""

    BINDINGS = (Binding("escape", "close", "Close", show=False),)

    def __init__(self, operations, installation_id, details):
        super().__init__()
        self.operations, self.installation_id = operations, installation_id
        self.declarations = json.loads(details["inspection"].variables_json)
        self.references = _configuration_references(details)["variables"]
        from ..Plugins.variables import handle

        for metadata in operations.variable_results.values():
            if (
                metadata
                and metadata["installation_id"] == installation_id
                and metadata["name"] in self.declarations
            ):
                self.references[metadata["name"]] = handle(metadata)
        self._busy = False
        self._revoke_reference = None

    def compose(self):
        with VerticalScroll(id="plugin-review-shell"):
            yield Static("Declared plugin value", classes="dialog-title")
            yield Select(
                [(name, name) for name in self.declarations],
                allow_blank=False,
                id="plugin-variable-name",
            )
            yield Static("", id="plugin-variable-type", markup=False)
            self._value_input = Input(
                id="plugin-variable-value",
                password=True,
                max_length=16384,
                classes="form-input",
            )
            yield self._value_input
            yield Static("", id="plugin-variable-reference", markup=False)
            yield Static(
                "Save creates a protected reference. Review and confirm its mapping afterward.",
                id="plugin-variable-status",
                markup=False,
            )
            with Horizontal(classes="dialog-buttons"):
                yield Button("Save value", id="plugin-variable-save")
                yield Button("Review revoke", id="plugin-variable-revoke")
            with Horizontal(classes="dialog-buttons"):
                yield Button(
                    "Confirm revoke", id="plugin-variable-confirm", disabled=True
                )
                yield Button("Close", id="plugin-variable-close")

    def on_mount(self):
        self._declaration()
        self.call_after_refresh(self.query_one("#plugin-variable-value", Input).focus)

    def _clear(self):
        self._value_input.value = ""

    @on(Select.Changed, "#plugin-variable-name")
    def _declaration(self):
        self._clear()
        name = str(self.query_one("#plugin-variable-name", Select).value)
        declaration = self.declarations[name]
        field = self.query_one("#plugin-variable-value", Input)
        field.password = declaration["secret"]
        field.placeholder = {
            "string": "Enter text",
            "integer": "Enter a whole number",
            "boolean": "Enter true or false",
        }[declaration["type"]]
        self.query_one("#plugin-variable-type", Static).update(
            f"{name} · {declaration['type']} · {'secret (masked)' if declaration['secret'] else 'nonsecret'}"
        )
        self.query_one("#plugin-variable-reference", Static).update(
            "Selected reference: " + self.references.get(name, "none")
        )
        self.query_one("#plugin-variable-revoke", Button).disabled = (
            name not in self.references
        )
        self.query_one("#plugin-variable-confirm", Button).disabled = True
        self._revoke_reference = None

    def action_close(self):
        self._clear()
        self.dismiss(None)

    def on_unmount(self):
        self._clear()

    @on(Button.Pressed)
    async def _button(self, event):
        event.stop()
        identity = event.button.id
        if identity == "plugin-variable-close":
            self.action_close()
            return
        if self._busy:
            return
        name = str(self.query_one("#plugin-variable-name", Select).value)
        if identity == "plugin-variable-revoke":
            self._revoke_reference = self.references[name]
            self.query_one("#plugin-variable-status", Static).update(
                "Revoke this reference? Consuming components will need configuration."
            )
            self.query_one("#plugin-variable-confirm", Button).disabled = False
            self.query_one("#plugin-variable-confirm", Button).focus()
            return
        self._busy = True
        value = None
        try:
            if identity == "plugin-variable-save":
                from uuid import uuid4

                value = self.query_one("#plugin-variable-value", Input).value
                kind = self.declarations[name]["type"]
                if kind == "integer":
                    value = int(value)
                elif kind == "boolean":
                    if value not in {"true", "false"}:
                        raise ValueError("boolean_required")
                    value = value == "true"
                self._clear()
                metadata = await self.operations.save_variable(
                    uuid4().hex, self.installation_id, name, value
                )
                if self.is_mounted and self.app.screen is self:
                    self.dismiss(("saved", name, metadata))
            elif (
                identity == "plugin-variable-confirm"
                and self._revoke_reference == self.references.get(name)
            ):
                from ..Plugins.variables import reference

                await self.operations.revoke_variable(
                    self.installation_id, reference(self._revoke_reference)[0]
                )
                if self.is_mounted and self.app.screen is self:
                    self.dismiss(("revoked", name, None))
        except (ValueError, OSError, RuntimeError):
            if self.is_mounted:
                self.query_one("#plugin-variable-status", Static).update(
                    "Value unavailable or invalid for its declared type. Check the protected backend and enter it again."
                )
                self.query_one("#plugin-variable-value", Input).focus()
        finally:
            value = None
            if self.is_mounted:
                self._clear()
            self._busy = False


class PluginAuthorityDialog(PasswordDialog):
    """Keep passphrases in the existing masked credential flow only."""

    def __init__(self, service, mode):
        super().__init__(
            mode=mode,
            title=(
                "Set up plugin authority"
                if mode == "setup"
                else "Unlock plugin authority"
            ),
            message="Plugin authority uses its own passphrase and protected marker. This does not change config encryption or standalone skill trust.",
        )
        self.service = service
        self._submitting = False

    def on_mount(self):
        self.call_after_refresh(self.query_one("#password-input", Input).focus)

    def _clear(self):
        for field in self.query(".password-input").results(Input):
            with field.prevent(Input.Changed):
                field.value = ""

    @on(Button.Pressed, "#submit-button")
    async def on_submit(self):
        if self._submitting:
            return
        password = self.query_one("#password-input", Input).value
        if len(password) < 8:
            self.show_error("Use at least 8 characters.")
            return
        if (
            self.mode == "setup"
            and password != self.query_one("#confirm-input", Input).value
        ):
            self.show_error("Passwords do not match.")
            return
        self._submitting = True
        try:
            await (
                self.service.bootstrap(password)
                if self.mode == "setup"
                else self.service.unlock(password)
            )
            self._clear()
            self.dismiss(True)
        except (ValueError, OSError, RuntimeError):
            self._clear()
            self.show_error(
                "Authority unavailable or passphrase incorrect. Retry after checking the protected backend."
            )
        finally:
            password = ""
            self._submitting = False

    @on(Input.Submitted)
    async def on_input_submitted(self, event):
        if self.mode == "setup" and event.input.id == "password-input":
            self.query_one("#confirm-input", Input).focus()
        else:
            await self.on_submit()

    def on_cancel(self):
        self._clear()
        self.dismiss(None)


class PluginAuthorityResetModal(ModalScreen):
    """Two explicit steps bound to the plugin namespace and retained consequences."""

    BINDINGS = (Binding("escape", "close", "Close", show=False),)

    def __init__(self, service):
        super().__init__()
        self.service, self.review = service, None

    def compose(self):
        with VerticalScroll(id="plugin-review-shell"):
            yield Static("Reset plugin authority", classes="dialog-title")
            yield Static(
                "This archives only plugin trust authority. Installed records and saved data remain. "
                "Old saved roots may require recovery; reset does not make them safe to reuse or delete. "
                "Wrong passwords and an unavailable backend should be retried without reset.",
                markup=False,
            )
            yield Static(
                "Review exact consequences first.",
                id="plugin-reset-status",
                markup=False,
            )
            with Horizontal(classes="dialog-buttons"):
                yield Button("Review reset", id="plugin-reset-review")
                yield Button(
                    "Confirm plugin reset", id="plugin-reset-confirm", disabled=True
                )
                yield Button("Close", id="plugin-reset-close")

    def action_close(self):
        self.dismiss(None)

    @on(Button.Pressed)
    async def _button(self, event):
        event.stop()
        identity = event.button.id
        if identity == "plugin-reset-close":
            self.action_close()
            return
        try:
            if identity == "plugin-reset-review":
                self.review = await self.service.review_authority_reset()
                self.query_one("#plugin-reset-status", Static).update(
                    display_text(
                        f"Namespace: {self.review.namespace_id}\nPosture: {self.review.posture}\n"
                        f"Retained root consequences (metadata, not proof of safe cleanup): {self.review.roots_json}",
                        limit=16384,
                    )
                )
                self.query_one("#plugin-reset-confirm", Button).disabled = False
            elif identity == "plugin-reset-confirm" and self.review is not None:
                self.query_one("#plugin-reset-confirm", Button).disabled = True
                await self.service.reset_authority(self.review)
                self.review = None
                self.query_one("#plugin-reset-status", Static).update(
                    "Plugin authority reset. Use Authority to set up again. Retained data still requires its own qualified recovery."
                )
        except (OSError, ValueError, RuntimeError):
            self.query_one("#plugin-reset-confirm", Button).disabled = True
            self.query_one("#plugin-reset-status", Static).update(
                "Reset unavailable or consequences changed. Restore the protected backend, wait for existing work, then review again."
            )


class PluginStatusModal(ModalScreen):
    """Bounded operation/candidate views holding original public owner objects."""

    BINDINGS = (Binding("escape", "close", "Close", show=False),)

    def __init__(self, sources, *, candidates=False):
        super().__init__()
        self.sources, self.candidates = sources, candidates
        self.rows = ()
        self.review = None
        self.page = 0

    def compose(self):
        with VerticalScroll(id="plugin-review-shell"):
            yield Static(
                "Persisted candidates" if self.candidates else "Plugin operations",
                classes="dialog-title",
            )
            yield OptionList(id="plugin-status-list")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Previous", id="plugin-status-previous", disabled=True)
                yield Button("Next", id="plugin-status-next")
            yield Static("Loading…", id="plugin-review-status", markup=False)
            with Horizontal(classes="dialog-buttons"):
                yield Button("Refresh", id="plugin-status-refresh")
                yield Button(
                    "Review discard",
                    id="plugin-status-review",
                    disabled=not self.candidates,
                )
                yield Button(
                    "Confirm discard", id="plugin-status-confirm", disabled=True
                )
            with Horizontal(classes="dialog-buttons"):
                yield Button(
                    "Manage saved data",
                    id="plugin-status-data",
                    disabled=self.candidates,
                )
                yield Button("Close", id="plugin-status-close")

    async def on_mount(self):
        self.operations = app_operations(self.app, self.sources.plugins)
        await self._load()

    async def _load(self):
        self.review = None
        self.query_one("#plugin-status-confirm", Button).disabled = True
        try:
            if self.candidates:
                self.rows = await self.sources.list_persisted_candidates(
                    offset=self.page * 50, limit=50
                )
                labels = [
                    (
                        row["operation_id"],
                        f"{row['operation_id']} · {row['reason'] or 'eligible for reviewed reference discard'}",
                    )
                    for row in self.rows
                ]
            else:
                self.rows = await self.operations.refresh(
                    offset=self.page * 50, limit=50
                )
                labels = [
                    (str(index), job.label) for index, job in enumerate(self.rows)
                ]
            self.query_one("#plugin-status-previous", Button).disabled = self.page == 0
            self.query_one("#plugin-status-next", Button).disabled = len(self.rows) < 50
            listing = self.query_one(OptionList)
            listing.clear_options()
            listing.add_options(
                [
                    Option(display_text(label, limit=1024), id=identity)
                    for identity, label in labels
                ]
            )
            self.query_one("#plugin-review-status", Static).update(
                "Reference release does not delete bytes. Pending or unknown custody stays protected."
                if self.candidates
                else "Durable history may omit expired entries. Runtime completion remains separate."
            )
        except (OSError, ValueError, RuntimeError):
            self.query_one("#plugin-review-status", Static).update(
                "Recovery required or authority locked. No unknown candidate is eligible for discard."
            )

    def action_close(self):
        self.dismiss(None)

    @on(Button.Pressed)
    async def _button(self, event):
        event.stop()
        identity = event.button.id
        try:
            if identity == "plugin-status-close":
                self.action_close()
            elif identity in {"plugin-status-previous", "plugin-status-next"}:
                self.page = max(0, self.page + (1 if identity.endswith("next") else -1))
                await self._load()
            elif identity == "plugin-status-refresh":
                await self._load()
            elif identity == "plugin-status-data" and not self.candidates:
                index = self.query_one(OptionList).highlighted
                if index is not None and self.rows[index].installation_id:
                    job = self.rows[index]
                    await self.app.push_screen(
                        PluginReviewModal(
                            self.sources,
                            installation_id=job.installation_id,
                            workspace_id=job.workspace_id,
                        )
                    )
            elif identity == "plugin-status-review":
                listing = self.query_one(OptionList)
                if listing.highlighted is None:
                    return
                row = self.rows[listing.highlighted]
                if row["reason"] or row["reference"] is None:
                    raise ValueError("candidate_protected")
                self.review = await self.sources.review_candidate_discard(
                    row["reference"]
                )
                self.query_one("#plugin-status-confirm", Button).disabled = False
                self.query_one("#plugin-review-status", Static).update(
                    f"Review exact reference: {row['operation_id']}. Confirm releases its recovery reference only."
                )
            elif identity == "plugin-status-confirm" and self.review is not None:
                await self.sources.discard_candidate_reference(self.review)
                self.review = None
                self.query_one("#plugin-status-confirm", Button).disabled = True
                await self._load()
        except (OSError, ValueError, RuntimeError):
            self.query_one("#plugin-review-status", Static).update(
                "Reference changed or protected. Refresh and review again."
            )


class PluginReviewModal(ModalScreen):
    BINDINGS = (Binding("escape", "close", "Close", show=False),)

    def __init__(
        self,
        sources,
        *,
        workspace_id,
        installation_id=None,
        producer=None,
        component_id=None,
    ):
        super().__init__()
        self.sources, self.service = sources, sources.plugins
        self.workspace_id, self.installation_id = workspace_id, installation_id
        self.producer = producer
        self.component_id = component_id
        self.details = None
        self._busy = False
        self._job_id = None
        self._loading = True
        self._input_generation = 0

    def compose(self):
        with VerticalScroll(id="plugin-review-shell"):
            yield Static("Manage plugin", classes="dialog-title", markup=False)
            yield Static(
                f"Captured workspace: {self.workspace_id or 'global default'}",
                markup=False,
            )
            yield Static(
                "Loading verified management inputs…",
                id="plugin-review-status",
                markup=False,
            )
            yield Select(
                (
                    ("Automatic interpretation", "auto"),
                    ("Portable", "portable"),
                    ("OpenAI", "openai"),
                    ("Cursor", "cursor"),
                ),
                value="auto",
                allow_blank=False,
                disabled=True,
                id="plugin-review-dialect",
            )
            yield SelectionList(id="plugin-review-selection")
            actions = (
                [
                    ("Review trust", "trust"),
                    ("Save component selection", "select"),
                    ("Enable in captured workspace", "enable"),
                    ("Inherit global default", "inherit"),
                    ("Disable here", "disable"),
                    (
                        "Enable global default (all inheriting workspaces)",
                        "global-enable",
                    ),
                    (
                        "Disable global default (all inheriting workspaces)",
                        "global-disable",
                    ),
                    ("Disable everywhere", "disable-all"),
                    ("Check for update", "update"),
                    ("Rollback selected revision", "rollback"),
                    ("Save MCP configuration (does not connect)", "save-profile"),
                    ("Save or revoke declared value", "save-variable"),
                    ("Save reviewed mapping references", "configure"),
                    ("Review explicit hook test", "test-hook"),
                    ("Review connection and discovery", "test-connection"),
                    ("Create saved-data root", "create-data"),
                    ("Uninstall; keep saved data", "uninstall"),
                    ("Delete exact saved roots", "delete-data"),
                    ("Resume pending saved-data cleanup", "resume-data"),
                    ("Review qualified reboot reconciliation", "reconcile-data"),
                ]
                if self.installation_id
                else [
                    ("Install disabled", "install"),
                    ("Install and enable in captured workspace", "install-enable"),
                ]
            )
            yield Select(
                actions,
                value=actions[0][1],
                allow_blank=False,
                id="plugin-review-action",
            )
            yield Input(
                placeholder="Revision digest, mapping JSON, or comma-separated exact root IDs",
                id="plugin-review-reference",
                max_length=4096,
                classes="form-input",
            )
            yield Static(
                "",
                id="plugin-review-input-help",
                markup=False,
            )
            yield TextArea("{}", id="plugin-review-test-data")
            yield Static("", id="plugin-review-body", markup=False)
            with Horizontal(classes="dialog-buttons"):
                yield Button("Review action", id="plugin-review-prepare")
                yield Button(
                    "Confirm reviewed action", id="plugin-review-confirm", disabled=True
                )
            with Horizontal(classes="dialog-buttons"):
                yield Button("Refresh status", id="plugin-review-refresh")
                yield Button(
                    "Cancel operation", id="plugin-review-cancel", disabled=True
                )
                yield Button(
                    "Cancel affected work",
                    id="plugin-review-cancel-work",
                    disabled=True,
                )
            with Horizontal(classes="dialog-buttons"):
                yield Button("Console", id="plugin-review-console")
                yield Button("Close", id="plugin-review-close")
                yield Button("Discard draft", id="plugin-review-discard")

    def on_mount(self):
        self.operations = app_operations(self.app, self.service)
        identity = (
            self.installation_id
            or getattr(self.producer, "operation_id", None)
            or getattr(self.producer, "token", None)
        )
        self.key = (identity, self.workspace_id)
        self.draft = self.operations.drafts.setdefault(
            self.key,
            PluginReviewDraft(self.workspace_id, self.installation_id, self.producer),
        )
        self.query_one("#plugin-review-reference", Input).value = (
            self.draft.configuration
        )
        self.query_one("#plugin-review-test-data", TextArea).text = self.draft.test_data
        self._job_id = self.draft.job_id
        if self.draft.action:
            self.query_one("#plugin-review-action", Select).value = self.draft.action
        self.query_one("#plugin-review-dialect", Select).value = (
            self.draft.dialect or "auto"
        )
        self.run_worker(self._load(), exclusive=True, group="plugin-review-load")

    def _status(self, value):
        self.query_one("#plugin-review-status", Static).update(
            display_text(value, limit=4096)
        )

    async def _load(self):
        try:
            if self.installation_id:
                try:
                    self.details = await self.service.management_details(
                        self.installation_id, workspace_id=self.workspace_id
                    )
                except ValueError as error:
                    if str(error) != "installation_unavailable":
                        raise
                    await self._load_retained_data()
                    return
                inspection = self.details["inspection"]
                selected = (
                    self.draft.selection
                    if self.draft.selection is not None
                    else self.details["selection"]
                )
            else:
                from ..Plugins.models import AcquiredSnapshot

                if isinstance(self.producer, AcquiredSnapshot):
                    inspection = await self.sources.inspect_acquired(
                        self.producer, dialect=self.draft.dialect
                    )
                else:
                    inspection = self.producer.inspection
                selected = self.draft.selection or ()
            self.inspection = inspection
            listing = self.query_one("#plugin-review-selection", SelectionList)
            with listing.prevent(SelectionList.SelectedChanged):
                listing.clear_options()
                listing.add_options(
                    [
                        (display_text(key, limit=256), key, key in selected)
                        for key in inspection.inventory
                    ]
                )
            self.query_one("#plugin-review-dialect", Select).display = (
                not self.installation_id
            )
            if self.draft.review is not None:
                self._render_review()
            else:
                self.query_one("#plugin-review-body", Static).update(
                    "Verified retained inputs. Executable components run with host privileges.\n"
                    + "\n".join(inspection.activation_blockers)
                    + (
                        "\nRetained revisions: " + ", ".join(self.details["revisions"])
                        if self.details
                        else ""
                    )
                )
            self._status(
                f"Review binds {self.component_id or self.installation_id or 'this retained package'}. Close preserves this draft."
            )
            self.call_after_refresh(self._loaded)
        except (OSError, ValueError, RuntimeError):
            self._status(
                "Management inputs unavailable. Unlock authority, check workspace and refresh."
            )
            self._loading = False
            self.query_one("#plugin-review-dialect", Select).disabled = False

    async def _data_rows(self):
        if self.details is not None:
            try:
                self.details = await self.service.management_details(
                    self.installation_id, workspace_id=self.workspace_id
                )
            except ValueError as error:
                if str(error) != "installation_unavailable":
                    raise
                self.details = None
            else:
                refs = self.details["roots"]
                if len(refs) > 256:
                    raise ValueError("retained_root_inventory_incomplete")
                rows = []
                for offset in range(0, len(refs), 50):
                    rows.extend(
                        await self.service.data_status(refs, limit=50, offset=offset)
                    )
                return tuple(rows)
        # The root owner caps an exact review at256 targets. Never select a
        # partial inventory, or find a replacement installation by package name.
        rows = []
        while len(rows) <= 256:
            page = await self.service.data_status(
                installation_id=self.installation_id,
                limit=min(50, 257 - len(rows)),
                offset=len(rows),
            )
            rows.extend(page)
            if len(rows) > 256:
                raise ValueError("retained_root_inventory_incomplete")
            if len(page) < min(50, 257 - (len(rows) - len(page))):
                return tuple(
                    row
                    for row in rows
                    if row["root"].installation_id == self.installation_id
                )
        raise ValueError("retained_root_inventory_incomplete")

    async def _load_retained_data(self):
        self.details = None
        rows = await self._data_rows()
        actions = [
            ("Review saved-data deletion", "delete-data"),
            ("Resume saved-data cleanup", "resume-data"),
            ("Review reboot reconciliation", "reconcile-data"),
        ]
        self.query_one("#plugin-review-action", Select).set_options(actions)
        if self.draft.action not in {value for _, value in actions}:
            self.draft.action = "delete-data"
            self.draft.review = None
        self.query_one("#plugin-review-action", Select).value = self.draft.action
        self.query_one("#plugin-review-selection", SelectionList).display = False
        self.query_one("#plugin-review-dialect", Select).display = False
        if self.draft.review is not None:
            self._render_review()
        else:
            self.query_one("#plugin-review-body", Static).update(
                display_text(
                    f"Installation {self.installation_id} is no longer installed.\n"
                    "Retained data is managed by its original exact root identities.\n"
                    + "\n".join(
                        f"{row['root'].root_id} · generation {row['root'].generation} · {row['root'].path}"
                        for row in rows
                    ),
                    limit=65536,
                )
            )
        self._status(
            "Retained data only. Review exact roots before cleanup; no package is reinstalled."
        )
        self.call_after_refresh(self._loaded)

    def _loaded(self):
        self._loading = False
        self.query_one("#plugin-review-dialect", Select).disabled = False
        self._show_action_inputs()
        matches = self.query(f"#{self.draft.focus_id}")
        if matches and matches.first().display:
            matches.first().focus()
        else:
            self.query_one("#plugin-review-action", Select).focus()

    def _show_action_inputs(self):
        action = self.query_one("#plugin-review-action", Select).value
        references = {
            "rollback": "Exact retained revision digest",
            "configure": "Complete mapping JSON: connections, tools, tool_references, models, component_tools, variables (reference handles only)",
            "save-profile": "MCP profile JSON with component_id and reviewed references",
            "delete-data": "Comma-separated exact root IDs; blank selects listed roots",
            "resume-data": "Comma-separated exact root IDs; blank selects listed roots",
            "reconcile-data": "Comma-separated exact root IDs; blank selects listed roots",
        }
        test = action in {"test-hook", "test-connection"}
        reference = self.query_one("#plugin-review-reference", Input)
        reference.display = action in references
        reference.placeholder = references.get(action, "")
        self.query_one("#plugin-review-test-data", TextArea).display = test
        self.query_one("#plugin-review-selection", SelectionList).display = (
            action in {"install", "install-enable", "select"} or test
        )
        self.query_one("#plugin-review-dialect", Select).display = action in {
            "install",
            "install-enable",
        }
        help_text = self.query_one("#plugin-review-input-help", Static)
        help_text.display = test or reference.display
        help_text.update(
            "Select one hook/server and enter synthetic event data. Tests require an "
            "existing idle Console chat in this workspace and may initialize SessionStart hooks."
            if test
            else references.get(action, "")
        )
        if self.focused is not None and not self.focused.display:
            self.query_one("#plugin-review-action", Select).focus()

    @on(Select.Changed, "#plugin-review-action")
    @on(SelectionList.SelectedChanged, "#plugin-review-selection")
    @on(Input.Changed, "#plugin-review-reference")
    @on(TextArea.Changed, "#plugin-review-test-data")
    def _review_input_changed(self, event):
        if self._loading or not hasattr(self, "draft"):
            return
        if (
            tuple(self.query_one("#plugin-review-selection", SelectionList).selected)
            == (self.draft.selection or ())
            and str(self.query_one("#plugin-review-action", Select).value)
            == self.draft.action
            and self.query_one("#plugin-review-reference", Input).value
            == self.draft.configuration
            and self.query_one("#plugin-review-test-data", TextArea).text
            == self.draft.test_data
        ):
            return
        self._input_generation += 1
        self._discard_pending_review()
        self.draft.action = str(self.query_one("#plugin-review-action", Select).value)
        self._show_action_inputs()
        self.query_one("#plugin-review-confirm", Button).disabled = True
        self._status("Inputs changed. Review again before confirming.")

    def _discard_pending_review(self):
        review = self.draft.review
        if getattr(review, "kind", None) in {"select", "update"} and not any(
            job.review is review for job in self.operations.jobs.values()
        ):
            self.service.cancel_revision(review)
        self.draft.review = None
        self.draft.root_impact = ()
        self.draft.root_before_refs = ()

    def _capture(self):
        self.draft.selection = tuple(
            self.query_one("#plugin-review-selection", SelectionList).selected
        )
        self.draft.configuration = self.query_one(
            "#plugin-review-reference", Input
        ).value
        self.draft.test_data = self.query_one("#plugin-review-test-data", TextArea).text
        self.draft.focus_id = getattr(self.focused, "id", None) or "plugin-review-close"

    def action_close(self):
        self._capture()
        self.dismiss(None)

    def _render_review(self):
        review = self.draft.review
        if isinstance(review, PluginConnectionDraft):
            component = review.inspection.inventory[review.component_id]
            body = (
                f"Save MCP configuration only\nInstallation: {review.installation_id}\n"
                f"Component: {review.component_id}\nRevision: {review.inspection.effective_digest}\n"
                f"Credential reference: {review.credential_reference or 'none'}\n"
                f"Saved data: {getattr(review.data_root, 'path', None) or 'none'}\n"
                f"Session isolation: {review.session_isolation}\nProtocol: {review.protocol_version}\nDefinition: {component.definition_json}\n"
                "This does not connect or grant tools. Review its mapping, connect/discover, then review exact tool mappings separately."
            )
        elif hasattr(review, "session_id"):
            body = (
                f"Explicit synthetic {review.kind} test\nInstallation: {review.installation_id}\n"
                f"Component: {review.component_id}\nRevision: {review.revision}\n"
                f"Console chat: {review.session_id}\nWorkspace: {review.workspace_id}\n"
                f"Event: {review.event_name}\nInput: {review.data_json}\n"
                "Host-user privileges; normal authority and permission gates apply. "
                "SessionStart hooks may initialize. Test effects are displayed here only."
            )
        elif hasattr(review, "inspection"):
            body = review_text(review)
        elif hasattr(review, "targets"):
            heading = (
                "Proposed saved-root bindings:"
                if self.draft.action == "reconcile-data"
                else "Exact saved roots:"
            )
            body = (
                heading
                + "\n"
                + "\n".join(
                    f"{row['root_id']} · generation {row['generation']} · {row['path']}"
                    for row in review.targets
                )
            )
            if self.draft.action == "create-data":
                body += "\nProposed new roots; no existing root users."
            else:
                for row in self.draft.root_impact:
                    if self.draft.action == "reconcile-data":
                        proposed = next(
                            target
                            for target in review.targets
                            if target["root_id"] == row["root"].root_id
                        )
                        body += (
                            f"\nCurrent generation {row['root'].generation} → "
                            f"proposed generation {proposed['generation']}"
                        )
                    body += (
                        f"\nRoot {row['root'].root_id}: recovery={row['recovery_reason']}; "
                        f"cleanup={row['cleanup']}; complete impact={row.get('blockers_complete', False)}"
                    )
                    for user in row["blockers"]:
                        body += (
                            f"\nOwner {user.get('owner_token', 'unknown')} · "
                            f"workspace {user.get('workspace_id') or 'unknown'} · {user['reason']}"
                        )
                    if not row["blockers"]:
                        body += "\nNo current users in this projection."
                body += (
                    "\nCurrent impact is observational. New users remain subject to owner fencing and drain. "
                    "Wait and review again if impact is incomplete or unknown. "
                    "After submission, Cancel operation leaves affected work owned; Cancel affected work asks its owners to stop."
                )
        else:
            body = f"{self.draft.action}: installation {self.installation_id}\nCaptured workspace: {self.workspace_id}\n"
            if self.details:
                body += "Activation overrides: " + str(self.details["activation"])
        if self.details:
            body += (
                "\nActivation overrides: "
                + str(self.details["activation"])
                + "\nGlobal-default actions affect every inheriting workspace; explicit overrides remain. "
                "Workspace actions affect only the captured workspace. Uninstall affects every workspace."
            )
        self.query_one("#plugin-review-body", Static).update(
            display_text(body, limit=65536)
        )
        self.query_one("#plugin-review-confirm", Button).disabled = (
            review is None
            or any(job.review is review for job in self.operations.jobs.values())
            or (
                hasattr(review, "targets")
                and (not self._impact_complete() or len(body) > 65536)
            )
        )
        # The body queues its own layout request; mark this screen before scrolling.
        self.refresh(layout=True)
        self.call_after_refresh(self._keep_focused_action_visible)

    def _keep_focused_action_visible(self):
        button = self.focused
        if (
            isinstance(button, Button)
            and button.is_mounted
            and button.screen is self
            and button.display
            and not button.disabled
        ):
            button.scroll_visible(animate=False, immediate=True)

    def _impact_complete(self):
        if self.draft.action == "create-data":
            return True
        allowed = {"live_root_user", "root_user"}
        if self.draft.action == "reconcile-data":
            allowed.add("root_boot_identity_changed")
        return bool(self.draft.root_impact) and all(
            row.get("blockers_complete") is True
            and all(user.get("reason") in allowed for user in row["blockers"])
            for row in self.draft.root_impact
        )

    async def _root_impact(self, review):
        refs = self.draft.root_before_refs
        if (
            review is not self.draft.review
            or not refs
            or {ref.root_id for ref in refs}
            != {row["root_id"] for row in review.targets}
        ):
            raise ValueError("root_impact_review_required")
        # Use the owner-returned inputs of this exact review, not its proposed
        # bindings/generations. Exact-ref validation precedes page slicing in the owning facade.
        rows = []
        for offset in range(0, len(refs), 50):
            rows.extend(await self.service.data_status(refs, limit=50, offset=offset))
        return tuple(rows)

    async def _validate_root_impact(self):
        review = self.draft.review
        if not hasattr(review, "targets") or self.draft.action == "create-data":
            return
        try:
            current = await self._root_impact(review)
            if not self._impact_complete() or current != self.draft.root_impact:
                raise ValueError("root_impact_changed_review_again")
        except (OSError, ValueError, RuntimeError):
            self._discard_pending_review()
            self.query_one("#plugin-review-confirm", Button).disabled = True
            raise

    async def _prepare(self):
        self._discard_pending_review()
        self._capture()
        action = str(self.query_one("#plugin-review-action", Select).value)
        root_before_refs = ()
        generation = self._input_generation
        selection, configuration = self.draft.selection, self.draft.configuration
        await self.service.management_snapshot(self.workspace_id, limit=1)
        if action.startswith("install"):
            from ..Plugins.models import AcquiredSnapshot

            kwargs = {
                "selection": selection,
                "workspace_id": self.workspace_id,
                "enable_in_workspace": action == "install-enable",
            }
            if isinstance(self.producer, AcquiredSnapshot):
                review = await self.sources.review_acquired(
                    self.producer, dialect=self.draft.dialect, **kwargs
                )
            else:
                review = await self.sources.review_candidate(
                    self.producer.inspection, **kwargs
                )
        elif action == "trust":
            review = await self.service.review_trust(self.installation_id)
        elif action == "select":
            review = await self.service.review_selection(
                self.installation_id, selection=selection
            )
        elif action in {"enable", "inherit", "global-enable", "global-disable"}:
            review = await self.service.review_activation(
                self.installation_id,
                workspace_id=None if action.startswith("global") else self.workspace_id,
                intent={
                    "enable": "enabled",
                    "inherit": "inherit",
                    "global-enable": "enabled",
                    "global-disable": "disabled",
                }[action],
            )
        elif action == "disable":
            review = await self.service.review_activation(
                self.installation_id, workspace_id=self.workspace_id, intent="disabled"
            )
        elif action in {"disable-all", "uninstall"}:
            review = await self.service.review_removal(
                self.installation_id, uninstall=action == "uninstall"
            )
        elif action == "update":
            candidates = await self.sources.check_updates(self.installation_id)
            if len(candidates) != 1:
                raise ValueError("update_candidate_unavailable")
            review = await self.sources.review_update(
                self.installation_id, candidates[0]
            )
        elif action == "rollback":
            review = await self.service.review_rollback(
                self.installation_id, configuration.strip()
            )
        elif action == "save-profile":
            from uuid import uuid4

            values = json.loads(configuration or "{}")
            allowed = {
                "component_id",
                "data_root_id",
                "credential_reference",
                "credential_generation",
                "session_isolation",
                "protocol_version",
            }
            if not isinstance(values, dict) or set(values) - allowed:
                raise ValueError("profile_requires_reference_fields_only")
            key = values.get("component_id") or self.component_id
            component = self.details["inspection"].inventory.get(key)
            if component is None or component.kind != "mcp":
                raise ValueError("choose_exact_mcp_component_id")
            root_id = values.get("data_root_id")
            root = next(
                (row for row in self.details["roots"] if row.root_id == root_id), None
            )
            if root_id is not None and root is None:
                raise ValueError("saved_data_root_unavailable")
            review = PluginConnectionDraft(
                uuid4().hex,
                self.installation_id,
                self.workspace_id,
                key,
                self.details["inspection"],
                root,
                values.get("credential_reference"),
                values.get("credential_generation"),
                values.get("session_isolation", "separate"),
                values.get("protocol_version", "2026-07-28"),
            )
        elif action == "configure":
            values = json.loads(configuration)
            if not isinstance(values, dict) or set(values) - {
                "connections",
                "tools",
                "tool_references",
                "models",
                "component_tools",
                "variables",
            }:
                raise ValueError("configuration_requires_mapping_references")
            review = await self.service.review_configuration(
                self.installation_id, **values
            )
        elif action.startswith("test-"):
            runtime = getattr(self.app, "console_runtime", None)
            store = runtime.chat_store if runtime is not None else None
            session = (
                next(
                    (
                        row
                        for row in store.sessions()
                        if row.id == store.active_session_id
                        and row.workspace_id == self.workspace_id
                    ),
                    None,
                )
                if store
                else None
            )
            if session is None:
                raise ValueError(
                    "open_existing_console_chat_in_captured_workspace_then_return"
                )
            kind = "hook" if action == "test-hook" else "mcp"
            candidates = [
                key
                for key in selection
                if self.details["inspection"].inventory[key].kind == kind
            ]
            if self.component_id in candidates:
                candidates = [self.component_id]
            if len(candidates) != 1:
                raise ValueError("select_exactly_one_test_component")
            review = await runtime.prepare_plugin_test(
                session.id,
                self.installation_id,
                candidates[0],
                kind="hook" if kind == "hook" else "connection",
                data=json.loads(
                    self.query_one("#plugin-review-test-data", TextArea).text
                ),
            )
        elif action == "create-data":
            review = await self.service.review_data_creation(
                self.installation_id, workspace_id=self.workspace_id
            )
        else:
            roots = tuple(row["root"] for row in await self._data_rows())
            if configuration.strip():
                wanted = {value.strip() for value in configuration.split(",")}
                roots = tuple(root for root in roots if root.root_id in wanted)
                if {root.root_id for root in roots} != wanted:
                    raise ValueError("choose_exact_retained_root_ids")
            method = {
                "delete-data": self.service.review_data_deletion,
                "resume-data": self.service.review_data_cleanup_resume,
                "reconcile-data": self.service.review_data_boot_reconciliation,
            }[action]
            review = await method(roots)
            root_before_refs = roots
        if generation != self._input_generation:
            if getattr(review, "kind", None) in {"select", "update"}:
                self.service.cancel_revision(review)
            raise ValueError("review_inputs_changed")
        self.draft.review, self.draft.action = review, action
        self.draft.root_before_refs = root_before_refs
        if hasattr(review, "targets") and action != "create-data":
            self.draft.root_impact = await self._root_impact(review)
        self._render_review()
        self._status(
            "Impact incomplete or unknown. Wait for owned work to settle, then review again."
            if self.query_one("#plugin-review-confirm", Button).disabled
            else "Review ready. Confirm applies these exact inputs; Close does not approve."
        )

    async def _save_profile(self, review):
        draft = self.draft
        generation = self._input_generation
        original = self.query_one("#plugin-review-reference", Input).value
        try:
            owner = getattr(self.app, "local_mcp_control_service", None)
            if owner is None:
                raise ValueError("mcp_configuration_owner_unavailable")
            result = await self.operations.save_profile(review, owner)
        except (OSError, ValueError, RuntimeError) as error:
            if self.is_mounted:
                self._status(
                    f"Configuration save unavailable: {type(error).__name__}. No mapping or connection was approved."
                )
            return
        # Saving the profile succeeded independently of the following refresh.
        # Never turn an unknown or nonempty mapping set into a partial replacement.
        saved = f"MCP profile {result['profile_id']} saved without connecting. "
        try:
            details = await self.service.management_details(
                review.installation_id, workspace_id=review.workspace_id
            )
            from ..Plugins.review import inspection_identity

            if inspection_identity(details["inspection"]) != inspection_identity(
                review.inspection
            ):
                raise ValueError("plugin_configuration_review_changed")
            mappings = details["mappings"]
            if not isinstance(mappings, (tuple, list)):
                raise ValueError(  # noqa: TRY004 -- stable refusal API
                    "plugin_configuration_unavailable"
                )
        except (OSError, ValueError, RuntimeError, KeyError):
            if self.is_mounted:
                self._status(
                    saved
                    + "Mapping refresh unavailable. Draft preserved; refresh and review the complete configuration."
                )
            return
        if not self.is_mounted:
            return
        same_draft = (
            self.draft is draft
            and draft.review is review
            and generation == self._input_generation
            and self.query_one("#plugin-review-reference", Input).value == original
        )
        if mappings or not same_draft:
            self._status(
                saved
                + "Draft preserved. Supply and review the complete configuration, including every retained mapping."
            )
            return
        self.query_one("#plugin-review-action", Select).value = "configure"
        self.query_one("#plugin-review-reference", Input).value = json.dumps(
            {"connections": {review.component_id: result["profile_id"]}}
        )
        self._status(
            saved
            + "Review the filled mapping separately; then connect/discover and review tool mappings."
        )

    async def _variable_result(self, result, *, request=None):
        if result is None or not self.is_mounted:
            return
        action, name, metadata = result
        saved = "Reference saved. " if action == "saved" else "Reference revoked. "
        if request is None:
            request = (
                self.draft,
                self._input_generation,
                self.query_one("#plugin-review-reference", Input).value,
            )
        draft, generation, original = request
        try:
            details = await self.service.management_details(
                self.installation_id, workspace_id=self.workspace_id
            )
            configuration = _configuration_references(details)
            if action == "saved":
                from ..Plugins.variables import handle

                declaration = json.loads(details["inspection"].variables_json).get(name)
                if (
                    declaration is None
                    or metadata["installation_id"] != self.installation_id
                    or metadata["name"] != name
                    or metadata["value_type"] != declaration["type"]
                    or metadata["secret"] != declaration["secret"]
                ):
                    raise ValueError("plugin_variable_identity_changed")
                configuration["variables"][name] = handle(metadata)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            if self.is_mounted:
                self._status(
                    saved
                    + "Complete mapping refresh unavailable. Draft preserved; refresh and review configuration."
                )
            return
        if not self.is_mounted:
            return
        if (
            self.draft is not draft
            or self._input_generation != generation
            or self.query_one("#plugin-review-reference", Input).value != original
        ):
            self._status(
                saved
                + "Newer draft preserved. Review the complete configuration before confirming."
            )
            return
        self._discard_pending_review()
        self.query_one("#plugin-review-confirm", Button).disabled = True
        self.query_one("#plugin-review-action", Select).value = "configure"
        self.query_one("#plugin-review-reference", Input).value = json.dumps(
            configuration
        )
        self.query_one("#plugin-review-reference", Input).focus()
        self._status(
            saved
            + (
                "Review the complete mapping before confirming."
                if action == "saved"
                else "Its current mapping is unavailable; replace it before reviewing configuration."
            )
        )

    async def _execute_test(self, review):
        try:
            outcome = await self.app.console_runtime.execute_plugin_test(review)
            if self.is_mounted:
                self.query_one("#plugin-review-body", Static).update(
                    display_text(str(outcome), limit=16384)
                )
                self._status(
                    "Test returned; inspect failures and cleanup above. No effects were added to the conversation."
                )
        except (OSError, ValueError, RuntimeError) as error:
            if self.is_mounted:
                self._status(
                    f"Test unavailable: {type(error).__name__}. Review current authority and Console session."
                )

    async def _refresh(self):
        if hasattr(self.draft.review, "targets") and not any(
            job.review is self.draft.review for job in self.operations.jobs.values()
        ):
            self._discard_pending_review()
            self.query_one("#plugin-review-confirm", Button).disabled = True
            self._status(
                "Status refreshed. Review current root impact again before confirmation."
            )
        await self.operations.refresh()
        submitted = self.operations.jobs.get(self._job_id)
        installation_id = self.installation_id or getattr(
            submitted, "installation_id", None
        )
        jobs = [
            job
            for job in self.operations.jobs.values()
            if installation_id is not None and job.installation_id == installation_id
        ]
        lines = [
            f"{job.label} · blockers: {job.runtime_status.get('blockers', ())}"
            for job in jobs
        ]
        # A completed uninstall remains visible even when installation details
        # no longer exist or a separate retained-root lookup is unavailable.
        self.query_one("#plugin-review-body", Static).update(
            display_text("\n".join(lines) or "No retained operations.", limit=65536)
        )
        if self.installation_id:
            rows = await self._data_rows()
            for row in rows:
                if row["root"].installation_id == self.installation_id:
                    lines.append(
                        f"Root {row['root'].root_id}: {row['root'].path}\n"
                        f"{row['recovery_reason'] or row['cleanup'] or ('Cleaned — saved data removed' if row['cleaned_absent'] else 'retained')}; users: {row['blockers']}"
                    )
                    if row["recovery_reason"]:
                        lines.append(
                            "Recovery required. Wait for known users or cancel affected work, then refresh. "
                            "Resume only a qualified cleanup. Reboot reconciliation requires a protected clean checkpoint; "
                            "dirty, unknown, legacy or remote-uncertain history cannot be repaired here without host proof."
                        )
        self.query_one("#plugin-review-body", Static).update(
            display_text("\n".join(lines) or "No retained operations.", limit=65536)
        )
        job = self.operations.jobs.get(self._job_id)
        cancellable = bool(job and job.runtime_status.get("cancel_operation"))
        self.query_one("#plugin-review-cancel", Button).disabled = not cancellable
        self.query_one("#plugin-review-cancel-work", Button).disabled = not cancellable
        self.call_after_refresh(self._keep_focused_action_visible)

    @on(Select.Changed, "#plugin-review-dialect")
    def _dialect(self, event):
        if self._loading or not hasattr(self, "draft") or self.installation_id:
            return
        selector = self.query_one("#plugin-review-dialect", Select)
        dialect = None if event.value == "auto" else str(event.value)
        if (
            event.select is not selector
            or event.value != selector.value
            or dialect == self.draft.dialect
        ):
            return
        self._capture()
        self._input_generation += 1
        self.draft.dialect = dialect
        self.draft.review = None
        self._loading = True
        selector.disabled = True
        self.query_one("#plugin-review-confirm", Button).disabled = True
        self.run_worker(self._load(), exclusive=True, group="plugin-review-load")

    @on(Button.Pressed)
    async def _button(self, event):
        action = event.button.id
        if not action or not action.startswith("plugin-review-"):
            return
        event.stop()
        if action == "plugin-review-console":
            from ..Constants import TAB_CHAT
            from ..UI.Navigation.main_navigation import NavigateToScreen

            self.action_close()
            self.app.post_message(NavigateToScreen(TAB_CHAT))
            self.app.notify(
                "Select an existing chat in the captured workspace. Return with F8 and Manage; your exact draft is retained."
            )
            return
        if action == "plugin-review-close":
            self.action_close()
            return
        if action == "plugin-review-discard":
            try:
                self._discard_pending_review()
                if self.producer is not None:
                    from ..Plugins.models import AcquiredSnapshot

                    candidate = (
                        self.producer
                        if isinstance(self.producer, AcquiredSnapshot)
                        else self.producer.inspection
                    )
                    await self.sources.discard_candidate(candidate)
                self.operations.drafts.pop(self.key, None)
                self.dismiss(None)
            except (OSError, ValueError, RuntimeError):
                self._status(
                    "Discard blocked by current custody. Refresh and wait for owned work before retrying."
                )
            return
        if self._busy:
            return
        self._busy = True
        try:
            if action == "plugin-review-prepare":
                if (
                    self.query_one("#plugin-review-action", Select).value
                    == "save-variable"
                ):
                    self._discard_pending_review()
                    self.query_one("#plugin-review-confirm", Button).disabled = True
                    request = (
                        self.draft,
                        self._input_generation,
                        self.query_one("#plugin-review-reference", Input).value,
                    )
                    details = await self.service.management_details(
                        self.installation_id, workspace_id=self.workspace_id
                    )
                    if not json.loads(details["inspection"].variables_json):
                        raise ValueError("plugin_has_no_declared_variables")
                    await self.app.push_screen(
                        PluginVariableDialog(
                            self.operations, self.installation_id, details
                        ),
                        partial(self._variable_result, request=request),
                    )
                    return
                await self._prepare()
            elif action == "plugin-review-confirm" and self.draft.review is not None:
                await self._validate_root_impact()
                await self.service.management_snapshot(self.workspace_id, limit=1)
                if self.draft.action == "save-profile":
                    self.query_one("#plugin-review-confirm", Button).disabled = True
                    self.run_worker(
                        self._save_profile(self.draft.review),
                        exclusive=True,
                        group="plugin-profile-view",
                    )
                    return
                if self.draft.action.startswith("test-"):
                    review = self.draft.review
                    self.query_one("#plugin-review-confirm", Button).disabled = True
                    self.run_worker(
                        self._execute_test(review),
                        exclusive=True,
                        group="plugin-test-view",
                    )
                    return
                job = self.operations.submit(
                    self.draft.action,
                    self.draft.review,
                    installation_id=self.installation_id
                    or self.draft.review.installation_id,
                    workspace_id=self.workspace_id,
                )
                self._job_id = job.receipt.request_id or job.receipt.operation_id
                self.draft.job_id = self._job_id
                self._status(
                    "Submitted. Close leaves this operation running; Refresh reads its owner status."
                )
                self.query_one("#plugin-review-confirm", Button).disabled = True
            elif action == "plugin-review-refresh":
                await self._refresh()
            elif (
                action in {"plugin-review-cancel", "plugin-review-cancel-work"}
                and self._job_id
            ):
                await self.operations.cancel(
                    self._job_id, affected_work=action.endswith("-work")
                )
                await self._refresh()
        except (OSError, ValueError, RuntimeError) as error:
            code = (
                str(error)
                if str(error).replace("_", "").isalnum()
                else type(error).__name__
            )
            self._status(
                f"Action blocked: {code}. Retained draft unchanged; review current state before retrying."
            )
        finally:
            self._busy = False
