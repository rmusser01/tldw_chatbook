"""Explicit plugin management with retained nonsecret drafts and opaque reviews."""

from __future__ import annotations

import json

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
                yield Button("Close", id="plugin-status-close")

    async def on_mount(self):
        self.operations = app_operations(self.app, self.sources.plugins)
        await self._load()

    async def _load(self):
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
                "Mapping references: JSON with connections, tools, models or component_tools. "
                "For tests, select one hook/server and enter synthetic event data below. "
                "Tests require an existing idle Console chat in this workspace and may initialize SessionStart hooks.",
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
                self.details = await self.service.management_details(
                    self.installation_id, workspace_id=self.workspace_id
                )
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

    def _loaded(self):
        self._loading = False
        self.query_one("#plugin-review-dialect", Select).disabled = False
        matches = self.query(f"#{self.draft.focus_id}")
        if matches:
            matches.first().focus()

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
        self.query_one("#plugin-review-confirm", Button).disabled = True
        self._status("Inputs changed. Review again before confirming.")

    def _discard_pending_review(self):
        review = self.draft.review
        if getattr(review, "kind", None) in {"select", "update"} and not any(
            job.review is review for job in self.operations.jobs.values()
        ):
            self.service.cancel_revision(review)
        self.draft.review = None

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
            body = "Exact saved roots:\n" + "\n".join(
                row["path"] for row in review.targets
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
        )

    async def _prepare(self):
        self._discard_pending_review()
        self._capture()
        action = str(self.query_one("#plugin-review-action", Select).value)
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
                "models",
                "component_tools",
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
            self.details = await self.service.management_details(
                self.installation_id, workspace_id=self.workspace_id
            )
            roots = self.details["roots"]
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
        if generation != self._input_generation:
            if getattr(review, "kind", None) in {"select", "update"}:
                self.service.cancel_revision(review)
            raise ValueError("review_inputs_changed")
        self.draft.review, self.draft.action = review, action
        self._render_review()
        self._status(
            "Review ready. Confirm applies these exact inputs; Close does not approve."
        )

    async def _save_profile(self, review):
        try:
            owner = getattr(self.app, "local_mcp_control_service", None)
            if owner is None:
                raise ValueError("mcp_configuration_owner_unavailable")
            result = await self.operations.save_profile(review, owner)
            if self.is_mounted:
                self.query_one("#plugin-review-action", Select).value = "configure"
                self.query_one("#plugin-review-reference", Input).value = json.dumps(
                    {"connections": {review.component_id: result["profile_id"]}}
                )
                self._status(
                    "MCP configuration saved without connecting. Review the filled mapping separately; then connect/discover and review tool mappings."
                )
        except (OSError, ValueError, RuntimeError) as error:
            if self.is_mounted:
                self._status(
                    f"Configuration save unavailable: {type(error).__name__}. No mapping or connection was approved."
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
        if self.installation_id:
            self.details = await self.service.management_details(
                self.installation_id, workspace_id=self.workspace_id
            )
            rows = await self.service.data_status(self.details["roots"])
            for row in rows:
                if row["root"].installation_id == self.installation_id:
                    lines.append(
                        f"Root {row['root'].root_id}: {row['root'].path}\n"
                        f"{row['recovery_reason'] or row['cleanup'] or 'retained'}; users: {row['blockers']}"
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
                await self._prepare()
            elif action == "plugin-review-confirm" and self.draft.review is not None:
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
