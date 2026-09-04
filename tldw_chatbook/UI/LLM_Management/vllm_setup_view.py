"""Focused Textual projection for vLLM setup and launch preflight."""

from __future__ import annotations

from dataclasses import replace

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Collapsible, Input, Label, TextArea

from .vllm_connection import VllmConnectionSnapshot

from .vllm_setup import (
    VllmLaunchDraft,
    VllmMode,
    VllmModelSource,
    VllmPreflightResult,
    VllmReadinessState,
    semantic_fingerprint,
)


class VllmSetupView(VerticalScroll):
    """Collect a vLLM draft and render only current preflight evidence."""

    class CheckRequested(Message):
        def __init__(self, draft: VllmLaunchDraft) -> None:
            super().__init__()
            self.draft = draft

    class StartRequested(Message):
        def __init__(self, draft: VllmLaunchDraft) -> None:
            super().__init__()
            self.draft = draft

    class StopRequested(Message):
        """Request settlement of the exact Chatbook-owned process."""

    class RetryRequested(Message):
        """Request a new generation of readiness evidence."""

    class LocalDirectoryBrowseRequested(Message):
        """Request a local model-directory picker without exposing a path globally."""

    class DraftChanged(Message):
        def __init__(self, draft: VllmLaunchDraft) -> None:
            super().__init__()
            self.draft = draft

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("classes", "llm-view-body")
        super().__init__(**kwargs)
        self._draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="",
        )
        self._state = VllmReadinessState.NOT_CONFIGURED
        self._preflight: VllmPreflightResult | None = None
        self._connection: VllmConnectionSnapshot | None = None
        self._rendering = False

    def compose(self) -> ComposeResult:
        yield Label("Set up vLLM", classes="section-title")
        yield Label(
            "Check the selected environment and model before Chatbook starts a local server.",
            classes="description",
        )
        yield Label("Setup incomplete", id="vllm-readiness-state")
        with Horizontal(classes="vllm-mode-actions"):
            yield Button("Start on this computer", id="vllm-start-local-button")
            yield Button(
                "Connect to existing server", id="vllm-connect-existing-button"
            )
        yield Label("Launch mode", classes="section_label")
        yield Label("", id="vllm-mode-summary")
        with Container(id="vllm-local-setup"):
            yield Label("Python environment", classes="inline-label")
            yield Input(
                value=self._draft.python_environment,
                id="vllm-python-environment",
                placeholder="python or /path/to/venv/bin/python",
            )
            yield Label("Model source", classes="inline-label")
            with Horizontal(classes="vllm-source-actions"):
                yield Button(
                    "Hugging Face repository", id="vllm-hugging-face-source-button"
                )
                yield Button(
                    "Local model directory", id="vllm-local-model-source-button"
                )
            yield Input(
                value=self._draft.model_value,
                id="vllm-hf-model",
                placeholder="organization/model",
            )
            yield Input(
                value=self._draft.model_value,
                id="vllm-local-model-directory",
                placeholder="Select a local model directory",
            )
            yield Button(
                "Browse",
                id="vllm-browse-local-model-directory-button",
                classes="browse_button",
            )
            yield Label("Network", classes="section_label")
            yield Label("Bind address", classes="inline-label")
            yield Input(value=self._draft.bind_address, id="vllm-bind-address")
            yield Label("Port", classes="inline-label")
            yield Input(value=str(self._draft.port), id="vllm-port")
        with Container(id="vllm-existing-setup"):
            yield Label("Existing server URL", classes="inline-label")
            yield Input(
                value=self._draft.existing_server_url,
                id="vllm-existing-server-url",
                placeholder="http://127.0.0.1:8000/v1",
            )
        yield Label("", id="vllm-start-blocker", classes="prereq-hint")
        with Horizontal(classes="vllm-action-bar"):
            yield Button("Check setup", id="vllm-check-setup-button")
            yield Button("Start", id="vllm-start-button", disabled=True)
            yield Button("Stop", id="vllm-stop-button", disabled=True)
            yield Button("Retry check", id="vllm-retry-button", disabled=True)
        yield Label("No activity yet.", id="vllm-activity-summary")
        with Collapsible(
            title="Activity details", id="vllm-activity-details", collapsed=True
        ):
            yield Label("No activity yet.", id="vllm-activity-events")
        yield Label("Advanced options", classes="section_label")
        yield TextArea(id="vllm-raw-arguments", classes="additional_args_textarea")

    def apply_state(
        self,
        *,
        draft: VllmLaunchDraft,
        state: VllmReadinessState,
        preflight: VllmPreflightResult | None,
        connection: VllmConnectionSnapshot | None = None,
    ) -> None:
        self._draft = draft
        self._state = state
        self._preflight = preflight
        self._connection = connection
        if self.is_mounted:
            self._render_projection()

    def project_lifecycle(self, *, active: bool, status: str | None = None) -> None:
        """Project app-owned process truth without inventing API readiness."""

        if active:
            state = self._state
            if state not in {
                VllmReadinessState.LAUNCHING,
                VllmReadinessState.LOADING_MODEL,
                VllmReadinessState.READY,
                VllmReadinessState.STOPPING,
            }:
                state = VllmReadinessState.LAUNCHING
        elif self._state in {
            VllmReadinessState.LAUNCHING,
            VllmReadinessState.LOADING_MODEL,
            VllmReadinessState.READY,
            VllmReadinessState.STOPPING,
        }:
            state = VllmReadinessState.NEEDS_ATTENTION
        else:
            state = self._state
        connection = self._connection
        owner = getattr(self.app, "_vllm_connection_owner", None)
        if owner is not None and callable(getattr(owner, "snapshot", None)):
            connection = owner.snapshot()
            state = connection.state
        self.apply_state(
            draft=self._draft,
            state=state,
            preflight=self._preflight,
            connection=connection,
        )

    def on_mount(self) -> None:
        self._render_projection()

    @property
    def draft(self) -> VllmLaunchDraft:
        """Return the current immutable launch candidate."""

        return self._draft

    @property
    def preflight(self) -> VllmPreflightResult | None:
        """Return current preflight evidence, if any."""

        return self._preflight

    def _render_projection(self) -> None:
        self._rendering = True
        try:
            local = self._draft.mode is VllmMode.LOCAL
            source_is_hf = self._draft.model_source is VllmModelSource.HUGGING_FACE
            self.query_one("#vllm-local-setup", Container).display = local
            self.query_one("#vllm-existing-setup", Container).display = not local
            self.query_one("#vllm-hf-model", Input).display = local and source_is_hf
            self.query_one("#vllm-local-model-directory", Input).display = (
                local and not source_is_hf
            )
            self.query_one(
                "#vllm-browse-local-model-directory-button", Button
            ).display = local and not source_is_hf
            self.query_one("#vllm-mode-summary", Label).update(
                "Start on this computer" if local else "Connect to existing server"
            )
            projected_inputs = {
                "#vllm-python-environment": self._draft.python_environment,
                "#vllm-hf-model": self._draft.model_value,
                "#vllm-local-model-directory": self._draft.model_value,
                "#vllm-bind-address": self._draft.bind_address,
                "#vllm-port": str(self._draft.port),
                "#vllm-existing-server-url": self._draft.existing_server_url,
            }
            for selector, value in projected_inputs.items():
                input_widget = self.query_one(selector, Input)
                if input_widget.value != value:
                    with input_widget.prevent(Input.Changed):
                        input_widget.value = value
            arguments = self.query_one("#vllm-raw-arguments", TextArea)
            if arguments.text != self._draft.raw_arguments:
                with arguments.prevent(TextArea.Changed):
                    arguments.text = self._draft.raw_arguments
            is_current_success = (
                self._preflight is not None
                and not self._preflight.issues
                and self._preflight.fingerprint == semantic_fingerprint(self._draft)
                and self._state is VllmReadinessState.READY_TO_START
            )
            start = self.query_one("#vllm-start-button", Button)
            stop = self.query_one("#vllm-stop-button", Button)
            retry = self.query_one("#vllm-retry-button", Button)
            start.disabled = not (local and is_current_success)
            stop.disabled = not local or self._state not in {
                VllmReadinessState.LAUNCHING,
                VllmReadinessState.LOADING_MODEL,
                VllmReadinessState.READY,
                VllmReadinessState.NEEDS_ATTENTION,
            }
            retry.disabled = self._state is not VllmReadinessState.NEEDS_ATTENTION
            self._render_readiness()
            blocker = self.query_one("#vllm-start-blocker", Label)
            if self._preflight and self._preflight.issues:
                issue = self._preflight.issues[0]
                blocker.update(f"{issue.field}: {issue.code.replace('_', ' ')}")
            elif not is_current_success:
                blocker.update("Check setup before Start is available.")
            elif self._preflight is not None and self._preflight.network_exposed:
                blocker.update(
                    "Network exposed: this server accepts non-loopback connections."
                )
            else:
                blocker.update(
                    "Setup checks passed. Start will launch a Chatbook-owned server."
                )
        finally:
            self._rendering = False

    def _render_readiness(self) -> None:
        """Render only allowlisted status and Activity copy."""

        state_copy = {
            VllmReadinessState.NOT_CONFIGURED: "Setup incomplete",
            VllmReadinessState.CHECKING: "Checking setup…",
            VllmReadinessState.READY_TO_START: "Setup checked · Ready to start",
            VllmReadinessState.LAUNCHING: "Launching process…",
            VllmReadinessState.LOADING_MODEL: "Loading model…",
            VllmReadinessState.READY: "Ready",
            VllmReadinessState.STOPPING: "Stopping…",
            VllmReadinessState.NEEDS_ATTENTION: "Needs attention",
        }
        connection = self._connection
        readiness = state_copy[self._state]
        if connection is not None and connection.target is not None:
            readiness = (
                f"Ready at {connection.launch_snapshot.client_api_url}"
                if connection.launch_snapshot is not None
                else "Ready · Existing vLLM server"
            )
        self.query_one("#vllm-readiness-state", Label).update(readiness)

        activity_copy = {
            "cancelled": "Check cancelled",
            "checking": "Checking setup",
            "claim_unavailable": "Server already starting or running",
            "credential_required": "Configured vLLM credentials are required",
            "health_checking": "Checking API health",
            "health_ok": "API health confirmed",
            "health_timeout": "API is not ready yet",
            "invalid_endpoint": "Server URL is invalid",
            "invalid_models_response": "Server model response is invalid",
            "invalidated": "Readiness reset",
            "launch_failed": "Server launch failed",
            "launch_reserved": "Launch reserved",
            "loading_model": "Loading model",
            "model_checking": "Checking served model",
            "model_missing": "Expected chat model is unavailable",
            "process_alive": "Server process is running",
            "process_exited": "Server process exited",
            "ready": "API and model are ready",
            "recomposed": "View changed; readiness reset",
            "screen_detached": "Models screen closed; check cancelled",
            "stopped": "Server stopped",
            "stopping": "Stopping server",
            "target_changed": "Setup changed; readiness reset",
        }
        events = connection.activity if connection is not None else ()
        lines = [
            f"{activity_copy[event.code]} · {event.elapsed_bucket.replace('_', ' ')}"
            + (f" · exit {event.exit_code}" if event.exit_code is not None else "")
            for event in events
        ]
        summary = lines[-1] if lines else "No activity yet."
        self.query_one("#vllm-activity-summary", Label).update(summary)
        self.query_one("#vllm-activity-events", Label).update(
            "\n".join(lines) or summary
        )
        details = self.query_one("#vllm-activity-details", Collapsible)
        if self._state is VllmReadinessState.NEEDS_ATTENTION:
            details.collapsed = False

    def _change_draft(self, **changes: object) -> None:
        self._draft = replace(self._draft, **changes)
        self._preflight = None
        self._state = VllmReadinessState.NOT_CONFIGURED
        self._render_projection()
        self.post_message(self.DraftChanged(self._draft))

    @on(Input.Changed)
    def _on_input_changed(self, event: Input.Changed) -> None:
        if self._rendering:
            return
        field_for_id = {
            "vllm-python-environment": "python_environment",
            "vllm-hf-model": "model_value",
            "vllm-local-model-directory": "model_value",
            "vllm-bind-address": "bind_address",
            "vllm-existing-server-url": "existing_server_url",
        }
        field = field_for_id.get(event.input.id)
        if field:
            self._change_draft(**{field: event.value})
        elif event.input.id == "vllm-port":
            try:
                port = int(event.value)
            except ValueError:
                port = 0
            self._change_draft(port=port)

    @on(TextArea.Changed, "#vllm-raw-arguments")
    def _on_raw_arguments_changed(self, event: TextArea.Changed) -> None:
        if self._rendering:
            return
        self._change_draft(raw_arguments=event.text_area.text)

    @on(Button.Pressed)
    def _on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        match event.button.id:
            case "vllm-start-local-button":
                self._change_draft(mode=VllmMode.LOCAL)
            case "vllm-connect-existing-button":
                self._change_draft(mode=VllmMode.EXISTING)
            case "vllm-hugging-face-source-button":
                self._change_draft(model_source=VllmModelSource.HUGGING_FACE)
            case "vllm-local-model-source-button":
                self._change_draft(model_source=VllmModelSource.LOCAL_DIRECTORY)
            case "vllm-browse-local-model-directory-button":
                self.post_message(self.LocalDirectoryBrowseRequested())
            case "vllm-check-setup-button":
                self.post_message(self.CheckRequested(self._draft))
            case "vllm-start-button":
                self.post_message(self.StartRequested(self._draft))
            case "vllm-stop-button":
                self.post_message(self.StopRequested())
            case "vllm-retry-button":
                self.post_message(self.RetryRequested())
