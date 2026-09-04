"""Focused Textual projection for vLLM setup and launch preflight."""

from __future__ import annotations

from dataclasses import replace

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Label, TextArea

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
        self._rendering = False

    def compose(self) -> ComposeResult:
        yield Label("Set up vLLM", classes="section-title")
        yield Label(
            "Check the selected environment and model before Chatbook starts a local server.",
            classes="description",
        )
        with Horizontal(classes="vllm-mode-actions"):
            yield Button("Start on this computer", id="vllm-start-local-button")
            yield Button("Connect to existing server", id="vllm-connect-existing-button")
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
                yield Button("Hugging Face repository", id="vllm-hugging-face-source-button")
                yield Button("Local model directory", id="vllm-local-model-source-button")
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
        yield Label("Advanced options", classes="section_label")
        yield TextArea(id="vllm-raw-arguments", classes="additional_args_textarea")

    def apply_state(
        self,
        *,
        draft: VllmLaunchDraft,
        state: VllmReadinessState,
        preflight: VllmPreflightResult | None,
    ) -> None:
        self._draft = draft
        self._state = state
        self._preflight = preflight
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
        self.apply_state(draft=self._draft, state=state, preflight=self._preflight)

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
            self.query_one("#vllm-local-model-directory", Input).display = local and not source_is_hf
            self.query_one("#vllm-browse-local-model-directory-button", Button).display = (
                local and not source_is_hf
            )
            self.query_one("#vllm-mode-summary", Label).update(
                "Start on this computer" if local else "Connect to existing server"
            )
            self.query_one("#vllm-python-environment", Input).value = self._draft.python_environment
            self.query_one("#vllm-hf-model", Input).value = self._draft.model_value
            self.query_one("#vllm-local-model-directory", Input).value = self._draft.model_value
            self.query_one("#vllm-bind-address", Input).value = self._draft.bind_address
            self.query_one("#vllm-port", Input).value = str(self._draft.port)
            self.query_one("#vllm-existing-server-url", Input).value = self._draft.existing_server_url
            self.query_one("#vllm-raw-arguments", TextArea).text = self._draft.raw_arguments
            is_current_success = (
                self._preflight is not None
                and not self._preflight.issues
                and self._preflight.fingerprint == semantic_fingerprint(self._draft)
                and self._state is VllmReadinessState.READY_TO_START
            )
            start = self.query_one("#vllm-start-button", Button)
            stop = self.query_one("#vllm-stop-button", Button)
            start.disabled = not (local and is_current_success)
            stop.disabled = self._state not in {
                VllmReadinessState.LAUNCHING,
                VllmReadinessState.LOADING_MODEL,
                VllmReadinessState.READY,
                VllmReadinessState.NEEDS_ATTENTION,
            }
            blocker = self.query_one("#vllm-start-blocker", Label)
            if self._preflight and self._preflight.issues:
                issue = self._preflight.issues[0]
                blocker.update(f"{issue.field}: {issue.code.replace('_', ' ')}")
            elif not is_current_success:
                blocker.update("Check setup before Start is available.")
            elif self._preflight.network_exposed:
                blocker.update("Network exposed: this server accepts non-loopback connections.")
            else:
                blocker.update("Setup checks passed. Start will launch a Chatbook-owned server.")
        finally:
            self._rendering = False

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
