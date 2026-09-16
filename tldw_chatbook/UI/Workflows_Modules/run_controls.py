"""Disposable launch controls and a projection of the app-owned workflow session."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Self

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Input, Select, Static, TextArea

from tldw_chatbook.Chat.console_provider_endpoints import first_configured_endpoint
from tldw_chatbook.Chat.provider_readiness import (
    provider_config_key,
    resolve_provider_credential,
)
from tldw_chatbook.UI.Workflows_Modules.library import WorkflowButton, compact_button
from tldw_chatbook.Workflows.models import Revision
from tldw_chatbook.Workflows.session import (
    ModelSelection,
    RunBindings,
    RunSetup,
    RunView,
    SessionError,
    WorkflowSession,
)

SESSION_DISCLOSURE = (
    "Session only: leaving this screen keeps the run; "
    "quitting loses pending review and intermediate results. Saved Notes remain."
)
TERMINAL_STATES = {"completed", "cancelled", "failed", "rejected", "uncertain"}


class SessionButton(WorkflowButton):
    """Snapshot the displayed run when pressed, before its message is queued."""

    run_view: RunView | None = None

    def press(self) -> Self:
        """Deliver the ordinary button event with immutable displayed identity."""
        if self.disabled or not self.display:
            return self
        self._start_active_affect()
        event = Button.Pressed(self)
        event.workflow_view = self.run_view
        self.post_message(event)
        return self


def captured_models(app: Any) -> tuple[ModelSelection, ...]:
    """Capture configured identities and current transport facts once, without discovery."""
    from tldw_chatbook.config import get_runtime_config_snapshot

    settings = get_runtime_config_snapshot().values
    catalog = app.local_llm_provider_catalog_service.list_providers()
    api_settings = settings.get("api_settings", {})
    choices = []
    for record in catalog["providers"]:
        provider = record["name"]
        key = provider_config_key(provider)
        # The catalog's display name/local-runtime label grants no transport authority.
        if key not in {"llama_cpp", "local_llamacpp"}:
            continue
        matches = [
            value
            for name, value in api_settings.items()
            if provider_config_key(name) == key
        ]
        if len(matches) != 1 or not isinstance(matches[0], Mapping):
            continue
        config = matches[0]
        credential, _, _ = resolve_provider_credential(key, config, environ=os.environ)
        if credential or config.get("credential_source") not in {None, "none"}:
            continue
        endpoint = first_configured_endpoint(config)
        if not endpoint:
            continue
        models = tuple(model for model in record["models"] if model and model != "None")
        for model in models or ("",):
            choices.append(
                ModelSelection(
                    provider,
                    endpoint,
                    model,
                    config.get("timeout", 120),
                    tuple(
                        (name, config[name])
                        for name in ("temperature", "top_p", "top_k", "min_p", "seed")
                        if name in config
                    ),
                )
            )
    if not choices:
        raise SessionError("keyless_llamacpp_unavailable")
    return tuple(choices)


class WorkflowRunSetup(ModalScreen[tuple[dict[str, Any], RunSetup] | None]):
    """Collect ordinary inputs and explicit model/file selections; no execution owner."""

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        revision: Revision,
        models: tuple[ModelSelection, ...],
        actor: str,
        protected_paths: tuple[Path, ...],
    ) -> None:
        super().__init__()
        self.revision, self.models = revision, models
        self.actor, self.protected_paths = actor, protected_paths
        document = json.loads(revision.raw_json)
        owned = {
            key
            for requirement in document["metadata"]["tldw_workflow"]
            .get("requirements", {})
            .values()
            for key in requirement["input_keys"]
        }
        self.inputs = {
            key: value
            for key, value in document.get("inputs", {}).items()
            if key not in owned
        }

    def compose(self) -> ComposeResult:
        with Vertical(classes="workflow-run-dialog"):
            yield Static("Run saved revision", classes="workflow-run-heading")
            with VerticalScroll():
                yield Static(
                    f"Workflow {self.revision.workflow_id}\nRevision {self.revision.revision_id}",
                    markup=False,
                )
                yield Static(SESSION_DISCLOSURE, markup=False)
                yield Static("Source · local UTF-8 .txt", markup=False)
                yield Input(id="workflow-source", classes="form-input")
                yield compact_button("Choose file…", "workflow-source-choose")
                yield Static("Configured provider / model", markup=False)
                yield Select(
                    (
                        (
                            f"{model.provider_id} / {model.model or 'Enter model below'}",
                            i,
                        )
                        for i, model in enumerate(self.models)
                    ),
                    value=0,
                    allow_blank=False,
                    id="workflow-model-choice",
                )
                yield Static("Actual model ID (editable)", markup=False)
                yield Input(
                    self.models[0].model, id="workflow-model", classes="form-input"
                )
                yield Static("Note title · Local Note", markup=False)
                yield Input(
                    str(self.inputs.get("note_title", "")),
                    id="workflow-note-title",
                    classes="form-input",
                )
                yield Static("Other ordinary inputs (JSON object)", markup=False)
                ordinary = {k: v for k, v in self.inputs.items() if k != "note_title"}
                yield TextArea(
                    json.dumps(ordinary, ensure_ascii=False),
                    id="workflow-inputs",
                    classes="form-textarea",
                )
                yield Static("", id="workflow-setup-error", markup=False)
            with Horizontal(classes="workflow-run-actions"):
                yield compact_button("Review destinations", "workflow-setup-review")
                yield compact_button("Cancel", "workflow-setup-cancel")

    def on_mount(self) -> None:
        self.query_one("#workflow-source", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Select.Changed, "#workflow-model-choice")
    def model_changed(self, event: Select.Changed) -> None:
        self.query_one("#workflow-model", Input).value = self.models[event.value].model

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "workflow-setup-cancel":
            self.action_cancel()
        elif event.button.id == "workflow-source-choose":
            from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen

            def selected(path):
                if path is not None and self.is_mounted:
                    self.query_one("#workflow-source", Input).value = str(path)
                    self.query_one("#workflow-source", Input).focus()

            self.app.push_screen(
                EnhancedFileOpen(filters=["*.txt"], context="workflow_source"), selected
            )
        elif event.button.id == "workflow-setup-review":
            try:
                values = json.loads(self.query_one("#workflow-inputs", TextArea).text)
                if not isinstance(values, dict):
                    raise TypeError
                values["note_title"] = self.query_one(
                    "#workflow-note-title", Input
                ).value
                selected = self.models[
                    self.query_one("#workflow-model-choice", Select).value
                ]
                model = ModelSelection(
                    selected.provider_id,
                    selected.selected_url,
                    self.query_one("#workflow-model", Input).value,
                    selected.request_timeout_seconds,
                    selected.sampling,
                )
                source = Path(self.query_one("#workflow-source", Input).value)
                if (
                    not source.is_absolute()
                    or not model.model.strip()
                    or not values["note_title"].strip()
                ):
                    raise ValueError
            except (ValueError, TypeError):
                self.query_one("#workflow-setup-error", Static).update(
                    "Choose an absolute file path, a model and Note title; inputs must be a JSON object."
                )
                return
            self.dismiss(
                (values, RunSetup(source, model, self.actor, self.protected_paths))
            )


class WorkflowRunConfirmation(ModalScreen[bool]):
    """Review the session's exact captured bindings before consuming its ticket."""

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(
        self, revision: Revision, binding: RunBindings, note_title: str
    ) -> None:
        super().__init__()
        self.revision, self.binding = revision, binding
        self.note_title = note_title

    def compose(self) -> ComposeResult:
        binding = self.binding
        with Vertical(classes="workflow-run-dialog"):
            yield Static("Confirm run destinations", classes="workflow-run-heading")
            with VerticalScroll():
                yield Static(
                    f"Workflow {self.revision.workflow_id}\nRevision {self.revision.revision_id}\n"
                    f"File: {binding.source}\nProvider: {binding.model.provider_id}\n"
                    f"Model: {binding.model.model}\nSelected: {binding.model.selected_url}\n"
                    f"Concrete endpoint: {binding.dispatch_url}\nKeyless execution\n"
                    f"Local Note: {binding.notes.db_path}\nUser: {binding.notes.user_id}\n"
                    f"Note title: {self.note_title}\n\n" + SESSION_DISCLOSURE,
                    markup=False,
                )
            with Horizontal(classes="workflow-run-actions"):
                yield compact_button("Start run", "workflow-start")
                yield compact_button("Cancel", "workflow-start-cancel")

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "workflow-start")


class WorkflowRunPanel(Vertical):
    """A disposable projection; the app session retains every edit and operation."""

    def __init__(self, session: WorkflowSession, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.session = session
        self.shown: RunView | None = None
        self._release = None

    def compose(self) -> ComposeResult:
        with Collapsible(
            title="Session run", collapsed=True, id="workflow-session-collapse"
        ):
            yield Static("", id="workflow-session-status", markup=False)
            with Horizontal(classes="workflow-run-actions"):
                for label, identifier in (
                    ("Approve once", "workflow-effect-approve"),
                    ("Reject", "workflow-effect-reject"),
                    ("Accept", "workflow-review-accept"),
                    ("Reject", "workflow-review-reject"),
                    ("Cancel run", "workflow-cancel"),
                    ("Open Note", "workflow-open-note"),
                ):
                    yield SessionButton(
                        label, id=identifier, classes="workflow-compact", disabled=True
                    )
            with VerticalScroll(id="workflow-session-content"):
                yield Static("", id="workflow-session-identity", markup=False)
                yield Static("", id="workflow-effect-text", markup=False)
                yield Static("", id="workflow-review-instructions", markup=False)
                yield TextArea("", id="workflow-review-text", classes="form-textarea")
                yield Static(SESSION_DISCLOSURE, markup=False)

    def on_mount(self) -> None:
        self._release = self.session.subscribe(self.refresh_view)
        self.call_after_refresh(self.refresh_view)

    def on_unmount(self) -> None:
        if self._release:
            self._release()

    @on(Collapsible.Toggled)
    def toggled(self, event: Collapsible.Toggled) -> None:
        self.set_class(not event.collapsible.collapsed, "workflow-session-expanded")

    def refresh_view(self) -> None:
        if not self.is_mounted:
            return
        view = self.session.view()
        previous, self.shown = self.shown, view
        self.display = view is not None
        if view is None:
            return
        closing = (
            bool(getattr(self.app, "_quit_in_progress", False))
            or view.message_code == "note_cleanup_failed"
        )
        for button in self.query(SessionButton):
            button.run_view = view
            button.disabled = closing
        if previous is None or previous.run_id != view.run_id:
            self.query_one(Collapsible).collapsed = False
            self.add_class("workflow-session-expanded")
        self.query_one("#workflow-session-status", Static).update(
            f"{view.state.capitalize()} · {view.step_id or 'Starting'}"
            + (f" · {view.message_code}" if view.message_code else "")
        )
        self.query_one("#workflow-session-identity", Static).update(
            f"Workflow {view.workflow_id}\nRevision {view.revision_id}\nRun {view.run_id}"
        )
        self.query_one("#workflow-review-instructions", Static).update(
            view.review_instructions or ""
        )
        effect = view.pending_effect
        self.query_one("#workflow-effect-text", Static).update(
            effect.payload_json if effect else ""
        )
        for identifier in ("workflow-effect-approve", "workflow-effect-reject"):
            self.query_one("#" + identifier).display = view.state == "approval"
            self.query_one("#" + identifier, Button).disabled = (
                closing or effect is None
            )
        review = self.query_one("#workflow-review-text", TextArea)
        review.display = view.state == "review"
        for identifier in ("workflow-review-accept", "workflow-review-reject"):
            self.query_one("#" + identifier).display = view.state == "review"
        self.query_one("#workflow-review-accept", Button).disabled = (
            closing or view.review_text is None
        )
        if view.review_text is not None and review.text != view.review_text:
            review.load_text(view.review_text)
        self.query_one("#workflow-cancel", Button).disabled = (
            closing or view.state in TERMINAL_STATES or view.state == "stopping"
        )
        self.query_one("#workflow-open-note", Button).disabled = (
            closing or not view.note_id
        )

    @on(TextArea.Changed, "#workflow-review-text")
    def edited(self, event: TextArea.Changed) -> None:
        event.stop()
        view = self.shown
        if view and event.text_area.text != view.review_text:
            self.session.update_review(view.run_id, view.step_id, event.text_area.text)

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "workflow-open-note":
            return  # The owning screen performs the existing Library route.
        event.stop()
        view = getattr(event, "workflow_view", None)
        if view is None:
            return
        identifier = event.button.id
        if identifier != "workflow-cancel" and view != self.session.view():
            return  # A queued approval cannot authorize newly displayed content.
        if identifier == "workflow-cancel":
            self.session.cancel(view.run_id)
        elif identifier in {"workflow-review-accept", "workflow-review-reject"}:
            self.session.answer_review(
                view.run_id, view.step_id, accept=identifier.endswith("accept")
            )
        elif (
            identifier in {"workflow-effect-approve", "workflow-effect-reject"}
            and view.pending_effect
        ):
            self.session.answer_effect(
                view.run_id,
                view.step_id,
                view.pending_effect.payload_json,
                approve=identifier.endswith("approve"),
            )
