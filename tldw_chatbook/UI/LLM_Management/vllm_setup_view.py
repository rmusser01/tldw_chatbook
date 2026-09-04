"""Focused Textual projection for vLLM setup and launch preflight."""

from __future__ import annotations

import ipaddress
from dataclasses import replace

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Input, Label, Select, TextArea

from .vllm_connection import VllmConnectionSnapshot
from .vllm_profiles import (
    VllmProfileDocumentV1,
    default_vllm_profile,
)
from .vllm_setup import (
    VllmIssue,
    VllmLaunchDraft,
    VllmLaunchSnapshot,
    VllmMode,
    VllmModelSource,
    VllmPreflightResult,
    VllmReadinessState,
    changed_launch_field_labels,
    semantic_fingerprint,
)

_PREFLIGHT_HELP_TARGETS = {
    "python_environment": "vllm-python-environment-help",
    "model_value": "vllm-model-help",
    "bind_address": "vllm-bind-address-help",
    "port": "vllm-port-help",
    "existing_server_url": "vllm-existing-server-url-help",
    "dtype": "vllm-dtype-help",
    "raw_arguments": "vllm-raw-arguments-help",
    "tensor_parallel_size": "vllm-tensor-parallel-size-help",
    "maximum_model_length": "vllm-maximum-model-length-help",
    "gpu_memory_utilization": "vllm-gpu-memory-utilization-help",
    "trust_remote_code": "vllm-trust-remote-code-error",
}
_PREFLIGHT_ISSUE_COPY = {
    "invalid_python_environment": (
        "Choose an absolute Python path or a bare executable name."
    ),
    "missing_python_environment": "Choose a Python interpreter or virtual environment.",
    "python_unavailable": (
        "Python environment not found. Choose an available interpreter "
        "or virtual environment."
    ),
    "vllm_cli_unavailable": (
        "vLLM is not installed in this Python environment. Install it there "
        "or choose another environment."
    ),
    "vllm_import_unavailable": (
        "vLLM cannot be imported from this Python environment. Repair it "
        "or choose another environment."
    ),
    "invalid_hugging_face_model": ("Enter a Hugging Face model as organization/model."),
    "invalid_model_directory": "Choose an existing local model directory.",
    "invalid_bind_address": "Enter an IP address or localhost.",
    "invalid_port": "Enter a port from 1 to 65535.",
    "port_unavailable": "This port is already in use. Choose another port.",
    "invalid_existing_server_url": (
        "Enter an http(s) vLLM server URL without credentials or extra parameters."
    ),
    "invalid_arguments": "Check the Advanced arguments quoting and try again.",
    "arguments_conflict": (
        "Advanced arguments duplicate a managed setting. Remove the duplicate option."
    ),
    "invalid_dtype": "Choose one of the supported dtype values.",
    "invalid_tensor_parallel_size": (
        "The saved tensor parallel size must be a positive whole number."
    ),
    "invalid_maximum_model_length": (
        "The saved maximum model length must be a positive whole number."
    ),
    "invalid_gpu_memory_utilization": (
        "The saved GPU memory utilization must be greater than 0 and at most 1."
    ),
    "invalid_trust_remote_code": "Choose whether remote model code is allowed.",
}


def _preflight_issue_copy(issue: VllmIssue) -> str:
    """Return bounded user-facing recovery copy without internal field keys."""

    return _PREFLIGHT_ISSUE_COPY.get(
        issue.code,
        "Review this setup value and retry the setup check.",
    )


def _draft_network_exposed(draft: VllmLaunchDraft) -> bool:
    """Classify only syntactically valid non-loopback local bind addresses."""

    if draft.mode is not VllmMode.LOCAL or draft.bind_address == "localhost":
        return False
    try:
        return not ipaddress.ip_address(draft.bind_address).is_loopback
    except ValueError:
        return False


class VllmSetupView(VerticalScroll):
    """Collect a vLLM draft and render only current preflight evidence."""

    BINDINGS = [
        Binding("tab", "vllm_focus(1)", show=False, priority=True),
        Binding("shift+tab", "vllm_focus(-1)", show=False, priority=True),
    ]

    _WIDTH_CLASSES = ("vllm-wide", "vllm-medium", "vllm-compact")
    _FOCUS_TARGETS = {
        VllmReadinessState.CHECKING: "vllm-cancel-check",
        VllmReadinessState.READY_TO_START: "vllm-start",
        VllmReadinessState.LAUNCHING: "vllm-stop",
        VllmReadinessState.LOADING_MODEL: "vllm-stop",
        VllmReadinessState.READY: "vllm-use-console",
        VllmReadinessState.NEEDS_ATTENTION: "vllm-recovery-primary",
    }

    DEFAULT_CSS = """
    VllmSetupView > #vllm-local-setup,
    VllmSetupView > #vllm-existing-setup,
    VllmSetupView > #vllm-current-server,
    VllmSetupView > #vllm-next-restart {
        height: auto;
    }
    VllmSetupView > #vllm-next-action,
    VllmSetupView > .vllm-mode-actions,
    VllmSetupView > .vllm-profile-actions,
    VllmSetupView > .vllm-source-actions,
    VllmSetupView > .vllm-console-actions {
        height: 3;
    }
    """

    class CheckRequested(Message):
        def __init__(self, draft: VllmLaunchDraft) -> None:
            super().__init__()
            self.draft = draft

    class CancelCheckRequested(Message):
        def __init__(self, generation: int) -> None:
            super().__init__()
            self.generation = generation

    class StartRequested(Message):
        def __init__(self, draft: VllmLaunchDraft) -> None:
            super().__init__()
            self.draft = draft

    class StopRequested(Message):
        """Request settlement of the exact Chatbook-owned process."""

    class RetryRequested(Message):
        """Request a new generation of readiness evidence."""

    class UseInConsoleRequested(Message):
        """Request session-only adoption of the current verified target."""

    class MakeDefaultRequested(Message):
        """Request Settings prefill for the current verified target."""

    class RestartRequested(Message):
        def __init__(
            self, draft: VllmLaunchDraft, changed_fields: tuple[str, ...]
        ) -> None:
            super().__init__()
            self.draft = draft
            self.changed_fields = changed_fields

    class ProfileSelected(Message):
        def __init__(self, profile_id: str) -> None:
            super().__init__()
            self.profile_id = profile_id

    class CreateProfileRequested(Message):
        def __init__(self, name: str, draft: VllmLaunchDraft) -> None:
            super().__init__()
            self.name = name
            self.draft = draft

    class SaveProfileRequested(Message):
        def __init__(self, profile_id: str, draft: VllmLaunchDraft) -> None:
            super().__init__()
            self.profile_id = profile_id
            self.draft = draft

    class RenameProfileRequested(Message):
        def __init__(self, profile_id: str, name: str) -> None:
            super().__init__()
            self.profile_id = profile_id
            self.name = name

    class DuplicateProfileRequested(Message):
        def __init__(self, profile_id: str) -> None:
            super().__init__()
            self.profile_id = profile_id

    class DeleteProfileRequested(Message):
        def __init__(self, profile_id: str) -> None:
            super().__init__()
            self.profile_id = profile_id

    class LocalDirectoryBrowseRequested(Message):
        """Request a local model-directory picker without exposing a path globally."""

    class PythonEnvironmentBrowseRequested(Message):
        """Request an interpreter picker using the established bounded file dialog."""

    class ExternalModelSelected(Message):
        def __init__(self, model_id: str) -> None:
            super().__init__()
            self.model_id = model_id

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
        initial_profile = default_vllm_profile()
        self._profiles = VllmProfileDocumentV1(
            1, 0, initial_profile.profile_id, (initial_profile,)
        )
        self._current_launch_snapshot: VllmLaunchSnapshot | None = None
        self._runtime_active = False
        self._discovered_model_ids: tuple[str, ...] = ()
        self._credential_configured = False
        self._profiles_ready = False
        self._profile_store_error = False
        self._rendering = False

    def compose(self) -> ComposeResult:
        yield Label("Set up vLLM / OPERATE", classes="section-title")
        yield Label(
            "Check the selected environment and model before Chatbook starts a local server.",
            classes="description",
        )
        yield Label("Setup incomplete", id="vllm-readiness-state")
        with Container(id="vllm-readiness-checklist"):
            yield Label("○ Environment · not checked", id="vllm-check-environment")
            yield Label(
                "○ vLLM installation · not checked", id="vllm-check-installation"
            )
            yield Label("○ Model · choose a model", id="vllm-check-model")
            yield Label("○ Network · not checked", id="vllm-check-network")
        with Horizontal(id="vllm-next-action", classes="vllm-action-bar"):
            yield Button("Check setup", id="vllm-check-setup")
            yield Button("Cancel check", id="vllm-cancel-check")
            yield Button("Start", id="vllm-start", disabled=True)
            yield Button("Stop", id="vllm-stop", disabled=True)
            yield Button("Retry check", id="vllm-recovery-primary", disabled=True)
            yield Button("Restart with draft", id="vllm-restart", disabled=True)
            yield Button("Use in Console", id="vllm-use-console", disabled=True)
        yield Label("▼ more below — scroll", id="vllm-fold-cue")
        with Horizontal(classes="vllm-mode-actions"):
            yield Button("Start on this computer", id="vllm-start-local-button")
            yield Button(
                "Connect to existing server", id="vllm-connect-existing-button"
            )
        yield Label("Launch mode", classes="section_label")
        yield Label("", id="vllm-mode-summary")
        yield Label("Profile", classes="section_label")
        yield Select(
            [(profile.name, profile.profile_id) for profile in self._profiles.profiles],
            value=self._profiles.selected_profile_id,
            allow_blank=False,
            id="vllm-profile-select",
        )
        yield Input(
            value=next(
                profile.name
                for profile in self._profiles.profiles
                if profile.profile_id == self._profiles.selected_profile_id
            ),
            id="vllm-profile-name",
            placeholder="Profile name",
        )
        yield Label(
            "",
            id="vllm-profile-name-help",
            classes="prereq-hint vllm-field-help",
        )
        with Horizontal(classes="vllm-profile-actions"):
            yield Button("New profile", id="vllm-profile-create-button")
            yield Button("Save changes", id="vllm-profile-save-button")
            yield Button("Rename", id="vllm-profile-rename-button")
            yield Button("Duplicate", id="vllm-profile-duplicate-button")
            yield Button("Delete", id="vllm-profile-delete-button")
        with Container(id="vllm-local-setup"):
            yield Label("Python environment", classes="inline-label")
            with Horizontal(classes="vllm-picker-row"):
                yield Input(
                    value=self._draft.python_environment,
                    id="vllm-python-environment",
                    placeholder="python or /path/to/venv/bin/python",
                )
                yield Button(
                    "Browse",
                    id="vllm-browse-python-environment",
                    classes="browse_button",
                )
            yield Label(
                "",
                id="vllm-python-environment-help",
                classes="prereq-hint vllm-field-help",
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
            yield Label("", id="vllm-model-help", classes="prereq-hint vllm-field-help")
            yield Label("Network", classes="section_label")
            yield Label("Bind address", classes="inline-label")
            yield Input(value=self._draft.bind_address, id="vllm-bind-address")
            yield Label(
                "",
                id="vllm-bind-address-help",
                classes="prereq-hint vllm-field-help",
            )
            yield Label("Port", classes="inline-label")
            yield Input(value=str(self._draft.port), id="vllm-port")
            yield Label("", id="vllm-port-help", classes="prereq-hint vllm-field-help")
        with Container(id="vllm-existing-setup"):
            yield Label("Existing server URL", classes="inline-label")
            yield Input(
                value=self._draft.existing_server_url,
                id="vllm-existing-server-url",
                placeholder="http://127.0.0.1:8000/v1",
            )
            yield Label(
                "",
                id="vllm-existing-server-url-help",
                classes="prereq-hint vllm-field-help",
            )
            yield Label(
                "Credential source · not configured",
                id="vllm-credential-status",
                classes="description",
            )
            yield Label("Returned model", classes="inline-label")
            yield Select(
                [],
                allow_blank=True,
                prompt="Check connection, then choose a model",
                id="vllm-existing-model",
            )
            yield Label(
                "Check connection to load available model IDs.",
                id="vllm-existing-model-help",
                classes="prereq-hint",
            )
        yield Label("", id="vllm-profile-help", classes="prereq-hint vllm-field-help")
        yield Label("", id="vllm-start-blocker", classes="prereq-hint")
        with Container(id="vllm-current-server"):
            yield Label("Current server", classes="section-title")
            yield Label("", id="vllm-current-server-summary")
        with Container(id="vllm-next-restart"):
            yield Label("Next restart configuration", classes="section-title")
            yield Label("", id="vllm-next-restart-state")
            yield Label("", id="vllm-next-restart-changes")
        yield Label("No activity yet.", id="vllm-activity-summary")
        activity = Collapsible(
            title="Activity details", id="vllm-activity-details", collapsed=True
        )
        activity._title.id = "vllm-activity-toggle"
        with activity:
            yield Label("No activity yet.", id="vllm-activity-events")
        advanced = Collapsible(
            title="Advanced options", id="vllm-advanced-options", collapsed=True
        )
        advanced._title.id = "vllm-advanced-toggle"
        with advanced:
            yield Label("dtype", classes="inline-label")
            yield Select(
                [
                    ("Auto", ""),
                    ("Half", "half"),
                    ("float16", "float16"),
                    ("bfloat16", "bfloat16"),
                    ("float32", "float32"),
                ],
                value="",
                allow_blank=False,
                id="vllm-dtype",
            )
            yield Label(
                "",
                id="vllm-dtype-help",
                classes="prereq-hint vllm-field-help",
            )
            yield Label("Tensor parallel size", classes="inline-label")
            yield Input(id="vllm-tensor-parallel-size", placeholder="Automatic")
            yield Label(
                "Number of GPUs used together for each model replica.",
                classes="description",
            )
            yield Label(
                "",
                id="vllm-tensor-parallel-size-help",
                classes="prereq-hint vllm-field-help",
            )
            yield Label("Maximum model length", classes="inline-label")
            yield Input(id="vllm-maximum-model-length", placeholder="Model default")
            yield Label(
                "Limits the model context length; larger values use more GPU memory.",
                classes="description",
            )
            yield Label(
                "",
                id="vllm-maximum-model-length-help",
                classes="prereq-hint vllm-field-help",
            )
            yield Label("GPU memory utilization", classes="inline-label")
            yield Input(id="vllm-gpu-memory-utilization", placeholder="vLLM default")
            yield Label(
                "Fraction from greater than 0 through 1; higher values leave less headroom.",
                classes="description",
            )
            yield Label(
                "",
                id="vllm-gpu-memory-utilization-help",
                classes="prereq-hint vllm-field-help",
            )
            yield Button("Trust remote code · Disabled", id="vllm-trust-remote-code")
            yield Label(
                "Disabled is safer. Enable only when you trust the model code source.",
                id="vllm-trust-remote-code-help",
                classes="description",
            )
            yield Label(
                "",
                id="vllm-trust-remote-code-error",
                classes="prereq-hint vllm-field-help",
            )
            arguments = Collapsible(
                title="Advanced arguments",
                id="vllm-advanced-arguments",
                collapsed=True,
            )
            arguments._title.id = "vllm-advanced-arguments-toggle"
            with arguments:
                yield Label(
                    "Launch only · not saved in profiles. Managed and secret flags are rejected.",
                    id="vllm-raw-arguments-scope",
                )
                yield TextArea(
                    id="vllm-raw-arguments", classes="additional_args_textarea"
                )
                yield Label(
                    "",
                    id="vllm-raw-arguments-help",
                    classes="prereq-hint vllm-field-help",
                )
        with Horizontal(classes="vllm-console-actions"):
            yield Button(
                "Make default for new chats",
                id="vllm-make-default",
                disabled=True,
            )
        yield Label(
            "Session only · restart uses your saved provider endpoint.",
            id="vllm-console-scope-copy",
        )

    def apply_state(
        self,
        *,
        draft: VllmLaunchDraft,
        state: VllmReadinessState,
        preflight: VllmPreflightResult | None,
        connection: VllmConnectionSnapshot | None = None,
        current_launch_snapshot: VllmLaunchSnapshot | None = None,
        profiles: VllmProfileDocumentV1 | None = None,
        runtime_active: bool | None = None,
        discovered_model_ids: tuple[str, ...] | None = None,
        credential_configured: bool | None = None,
        profiles_ready: bool | None = None,
        profile_store_error: bool | None = None,
    ) -> None:
        self._draft = draft
        self._state = state
        self._preflight = preflight
        self._connection = connection
        self._current_launch_snapshot = current_launch_snapshot
        if profiles is not None:
            self._profiles = profiles
        if runtime_active is not None:
            self._runtime_active = runtime_active
        if discovered_model_ids is not None:
            self._discovered_model_ids = discovered_model_ids
        if credential_configured is not None:
            self._credential_configured = credential_configured
        if profiles_ready is not None:
            self._profiles_ready = profiles_ready
        if profile_store_error is not None:
            self._profile_store_error = profile_store_error
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
            try:
                from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
                    process_is_running,
                    server_lifecycle_snapshot,
                )

                claim, process = server_lifecycle_snapshot(self.app, "vllm")
                active = bool(
                    active
                    and owner.owns_launch_claim(claim)
                    and (
                        process_is_running(process)
                        or (process is None and not claim.cancel_event.is_set())
                    )
                )
            except (AttributeError, RuntimeError, ValueError):
                active = False
        self.apply_state(
            draft=self._draft,
            state=state,
            preflight=self._preflight,
            connection=connection,
            runtime_active=active,
        )

    def on_mount(self) -> None:
        self._apply_width_class()
        self._render_projection()

    def on_resize(self) -> None:
        """Recompose layout from the allocated vLLM body width."""

        self._apply_width_class()

    def _apply_width_class(self) -> None:
        """Apply exactly one width class from this mounted body's allocation."""

        width = self.size.width
        width_class = (
            "vllm-compact"
            if width <= 55
            else "vllm-medium"
            if width <= 70
            else "vllm-wide"
        )
        for candidate in self._WIDTH_CLASSES:
            self.set_class(candidate == width_class, candidate)
        if self.is_mounted:
            self.call_after_refresh(self._sync_fold_cue)

    def _sync_fold_cue(self) -> None:
        """Paint a cue when the current compact/medium viewport has more work."""

        cue = self.query_one("#vllm-fold-cue", Label)
        cue.display = (
            not self.has_class("vllm-wide")
            and self.virtual_size.height > self.size.height
        )

    def _restore_top_scan_if_unfocused(self) -> None:
        """Keep readiness/action in view unless focus owns a deeper form row."""

        focused = self.app.focused
        if focused is not None and self in focused.ancestors_with_self:
            return
        self.scroll_home(animate=False)

    def _focusable_controls(self) -> tuple[Widget, ...]:
        """Return the visible, enabled Tab order owned by this provider pane."""

        return tuple(
            widget
            for widget in self.query("*").results(Widget)
            if widget.can_focus
            and all(ancestor.display for ancestor in widget.ancestors_with_self)
            and not widget.disabled
        )

    def action_vllm_focus(self, direction: int) -> None:
        """Cycle Tab focus within vLLM, excluding every hidden provider pane."""

        controls = self._focusable_controls()
        if not controls:
            return
        focused = self.app.focused
        try:
            index = controls.index(focused)
        except ValueError:
            index = -1 if direction > 0 else 0
        target = controls[(index + direction) % len(controls)]
        target.focus(scroll_visible=True)

    def focus_state_action(self, state: VllmReadinessState) -> None:
        """Focus the stable action for an explicit lifecycle transition."""

        target_id = self._FOCUS_TARGETS.get(state)
        if target_id is None:
            return
        try:
            target = self.query_one(f"#{target_id}", Button)
        except Exception:
            return
        if (
            all(ancestor.display for ancestor in target.ancestors_with_self)
            and not target.disabled
        ):
            target.focus(scroll_visible=True)

    @property
    def draft(self) -> VllmLaunchDraft:
        """Return the current immutable launch candidate."""

        return self._draft

    @property
    def preflight(self) -> VllmPreflightResult | None:
        """Return current preflight evidence, if any."""

        return self._preflight

    def show_profile_validation_error(
        self, field: str, classification: str | None = None
    ) -> None:
        """Place bounded profile repair beside the exact editable control."""

        targets = {
            "profile": ("vllm-profile-help", "vllm-profile-select"),
            "name": ("vllm-profile-name-help", "vllm-profile-name"),
            "python_environment": (
                "vllm-python-environment-help",
                "vllm-python-environment",
            ),
            "model_source": (
                "vllm-model-help",
                (
                    "vllm-hugging-face-source-button"
                    if self._draft.model_source is VllmModelSource.HUGGING_FACE
                    else "vllm-local-model-source-button"
                ),
            ),
            "model_value": (
                "vllm-model-help",
                (
                    "vllm-hf-model"
                    if self._draft.model_source is VllmModelSource.HUGGING_FACE
                    else "vllm-local-model-directory"
                ),
            ),
            "mode": ("vllm-profile-help", "vllm-start-local-button"),
            "bind_address": ("vllm-bind-address-help", "vllm-bind-address"),
            "port": ("vllm-port-help", "vllm-port"),
            "dtype": ("vllm-dtype-help", "vllm-dtype"),
            "tensor_parallel_size": (
                "vllm-tensor-parallel-size-help",
                "vllm-tensor-parallel-size",
            ),
            "maximum_model_length": (
                "vllm-maximum-model-length-help",
                "vllm-maximum-model-length",
            ),
            "gpu_memory_utilization": (
                "vllm-gpu-memory-utilization-help",
                "vllm-gpu-memory-utilization",
            ),
            "trust_remote_code": (
                "vllm-trust-remote-code-error",
                "vllm-trust-remote-code",
            ),
        }
        help_id, control_id = targets.get(field, targets["profile"])
        copies = {
            "duplicate_name": "Choose a unique profile name.",
            "invalid_name": "Enter a valid unique profile name.",
            "profile_cap": (
                "Profile limit reached. Delete a local profile before creating another."
            ),
            "profile_unavailable": (
                "The selected profile is no longer available. Reload profiles."
            ),
            "local_profiles_only": (
                "Profiles are for local starts. Switch to Start on this computer to edit them."
            ),
            "invalid_model_source": (
                "Choose Hugging Face repository or Local model directory."
            ),
        }
        if classification in copies:
            copy = copies[classification]
        elif field == "model_value":
            issue_code = (
                "invalid_hugging_face_model"
                if self._draft.model_source is VllmModelSource.HUGGING_FACE
                else "invalid_model_directory"
            )
            copy = _preflight_issue_copy(VllmIssue(issue_code, field))
        elif classification is not None:
            copy = _preflight_issue_copy(VllmIssue(classification or "", field))
        elif field == "name":
            copy = "Enter a valid unique profile name."
        elif field == "profile":
            copy = "Profile data needs repair or reload before it can be used."
        else:
            copy = "Repair this profile value before saving."
        if field in {
            "dtype",
            "tensor_parallel_size",
            "maximum_model_length",
            "gpu_memory_utilization",
            "trust_remote_code",
        }:
            self.query_one("#vllm-advanced-options", Collapsible).collapsed = False
        help_label = self.query_one(f"#{help_id}", Label)
        help_label.update(copy)
        help_label.display = True
        control = self.query_one(f"#{control_id}", Widget)
        self.call_after_refresh(control.focus)

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
            check = self.query_one("#vllm-check-setup", Button)
            cancel_check = self.query_one("#vllm-cancel-check", Button)
            check.label = "Check setup" if local else "Check connection"
            self.query_one("#vllm-credential-status", Label).update(
                "Credential source · configured"
                if self._credential_configured
                else "Credential source · not configured"
            )
            external_model = self.query_one("#vllm-existing-model", Select)
            selected_external_model = (
                self._draft.existing_model_id
                if self._draft.existing_model_id in self._discovered_model_ids
                else Select.NULL
            )
            with external_model.prevent(Select.Changed):
                external_model.set_options(
                    [(model_id, model_id) for model_id in self._discovered_model_ids]
                )
                if external_model.value != selected_external_model:
                    external_model.value = selected_external_model
            external_model.disabled = not self._discovered_model_ids
            external_help = self.query_one("#vllm-existing-model-help", Label)
            if self._discovered_model_ids and selected_external_model is Select.NULL:
                external_help.update("Select a returned model to verify it exactly.")
            elif selected_external_model is not Select.NULL:
                external_help.update("Selection requires an exact fresh verification.")
            else:
                external_help.update("Check connection to load available model IDs.")
            projected_inputs = {
                "#vllm-python-environment": self._draft.python_environment,
                "#vllm-hf-model": self._draft.model_value,
                "#vllm-local-model-directory": self._draft.model_value,
                "#vllm-bind-address": self._draft.bind_address,
                "#vllm-port": str(self._draft.port),
                "#vllm-existing-server-url": self._draft.existing_server_url,
                "#vllm-tensor-parallel-size": (
                    ""
                    if self._draft.tensor_parallel_size is None
                    else str(self._draft.tensor_parallel_size)
                ),
                "#vllm-maximum-model-length": (
                    ""
                    if self._draft.maximum_model_length is None
                    else str(self._draft.maximum_model_length)
                ),
                "#vllm-gpu-memory-utilization": (
                    ""
                    if self._draft.gpu_memory_utilization is None
                    else str(self._draft.gpu_memory_utilization)
                ),
            }
            advanced_options = self.query_one("#vllm-advanced-options", Collapsible)
            advanced_options.display = local
            dtype = self.query_one("#vllm-dtype", Select)
            projected_dtype = "" if self._draft.dtype == "auto" else self._draft.dtype
            if dtype.value != projected_dtype:
                with dtype.prevent(Select.Changed):
                    dtype.value = projected_dtype
            trust = self.query_one("#vllm-trust-remote-code", Button)
            trust.label = (
                "Trust remote code · Enabled"
                if self._draft.trust_remote_code
                else "Trust remote code · Disabled"
            )
            profile_select = self.query_one("#vllm-profile-select", Select)
            with profile_select.prevent(Select.Changed):
                profile_select.set_options(
                    [
                        (profile.name, profile.profile_id)
                        for profile in self._profiles.profiles
                    ]
                )
                if profile_select.value != self._profiles.selected_profile_id:
                    profile_select.value = self._profiles.selected_profile_id
            profile_select.disabled = not local or not self._profiles_ready
            selected_profile = next(
                profile
                for profile in self._profiles.profiles
                if profile.profile_id == self._profiles.selected_profile_id
            )
            profile_name = self.query_one("#vllm-profile-name", Input)
            if profile_name.value != selected_profile.name:
                with profile_name.prevent(Input.Changed):
                    profile_name.value = selected_profile.name
            profile_name.disabled = not local or not self._profiles_ready
            for profile_action_id in (
                "vllm-profile-create-button",
                "vllm-profile-save-button",
                "vllm-profile-rename-button",
                "vllm-profile-duplicate-button",
                "vllm-profile-delete-button",
            ):
                self.query_one(f"#{profile_action_id}", Button).disabled = (
                    not local or not self._profiles_ready
                )
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
            start = self.query_one("#vllm-start", Button)
            stop = self.query_one("#vllm-stop", Button)
            retry = self.query_one("#vllm-recovery-primary", Button)
            restart = self.query_one("#vllm-restart", Button)
            use_in_console = self.query_one("#vllm-use-console", Button)
            make_default = self.query_one("#vllm-make-default", Button)
            start.disabled = not (
                local and is_current_success and not self._runtime_active
            )
            stop.disabled = not self._runtime_active
            retry.disabled = (
                not self._profiles_ready
                or self._state is not VllmReadinessState.NEEDS_ATTENTION
            )
            target = self._connection.target if self._connection is not None else None
            token = (
                self._connection.current_token if self._connection is not None else None
            )
            current_target = bool(
                self._profiles_ready
                and self._state is VllmReadinessState.READY
                and self._connection is not None
                and self._connection.state is VllmReadinessState.READY
                and target is not None
                and token is not None
                and target.generation == token.generation
            )
            use_in_console.disabled = not current_target
            make_default.disabled = not current_target
            current_snapshot = self._current_launch_snapshot
            current_container = self.query_one("#vllm-current-server", Container)
            next_container = self.query_one("#vllm-next-restart", Container)
            show_current = self._runtime_active and current_snapshot is not None
            current_container.display = show_current
            next_container.display = show_current
            changed_fields: tuple[str, ...] = ()
            if current_snapshot is not None:
                changed_fields = changed_launch_field_labels(
                    current_snapshot, self._draft
                )
                self.query_one("#vllm-current-server-summary", Label).update(
                    "Current server · "
                    f"{current_snapshot.client_api_url} · "
                    f"{current_snapshot.served_model} · "
                    f"{current_snapshot.display_profile_name}"
                )
            dirty = bool(changed_fields)
            self.query_one("#vllm-next-restart-state", Label).update(
                "Modified for next restart" if dirty else "Matches current server"
            )
            self.query_one("#vllm-next-restart-changes", Label).update(
                "Changed: " + " · ".join(changed_fields) if dirty else "No changes"
            )
            restart.disabled = not (
                show_current and dirty and local and is_current_success
            )
            dirty_restart = bool(show_current and dirty)
            check.label = (
                "Check draft"
                if dirty_restart
                else ("Check setup" if local else "Check connection")
            )
            check.display = (
                not self._runtime_active or dirty_restart
            ) and self._state in {
                VllmReadinessState.NOT_CONFIGURED,
                VllmReadinessState.READY_TO_START,
            }
            check.disabled = self._state is VllmReadinessState.CHECKING
            cancel_check.display = self._state is VllmReadinessState.CHECKING
            cancel_check.disabled = not (
                token is not None
                and self._connection is not None
                and self._connection.state is VllmReadinessState.CHECKING
            )
            start.display = bool(is_current_success and not self._runtime_active)
            stop.display = self._runtime_active
            retry.display = bool(
                self._profiles_ready
                and self._state is VllmReadinessState.NEEDS_ATTENTION
            )
            restart.display = dirty_restart
            use_in_console.display = current_target
            make_default.display = current_target
            visible_primary_actions = sum(
                button.display
                for button in (
                    check,
                    cancel_check,
                    start,
                    stop,
                    retry,
                    restart,
                    use_in_console,
                )
            )
            self.set_class(visible_primary_actions > 1, "vllm-two-actions")
            self._render_readiness()
            blocker = self.query_one("#vllm-start-blocker", Label)
            help_labels = self.query(".vllm-field-help").results(Label)
            for help_label in help_labels:
                help_label.update("")
                help_label.display = False
            if self._profile_store_error:
                profile_help = self.query_one("#vllm-profile-help", Label)
                profile_help.update(
                    "Profile data needs repair or reload before it can be used."
                )
                profile_help.display = True
            elif not self._profiles_ready:
                profile_help = self.query_one("#vllm-profile-help", Label)
                profile_help.update(
                    "Loading saved profiles before readiness can be used."
                )
                profile_help.display = True
            elif not local:
                profile_help = self.query_one("#vllm-profile-help", Label)
                profile_help.update(
                    "Profiles are for local starts. Switch to Start on this computer "
                    "to edit them."
                )
                profile_help.display = True
            if self._preflight and self._preflight.issues:
                issue = self._preflight.issues[0]
                target_id = _PREFLIGHT_HELP_TARGETS.get(issue.field)
                if target_id is not None:
                    if issue.field in {
                        "dtype",
                        "tensor_parallel_size",
                        "maximum_model_length",
                        "gpu_memory_utilization",
                        "trust_remote_code",
                        "raw_arguments",
                    }:
                        advanced_options.collapsed = False
                    if issue.field == "raw_arguments":
                        self.query_one(
                            "#vllm-advanced-arguments", Collapsible
                        ).collapsed = False
                    help_label = self.query_one(f"#{target_id}", Label)
                    help_label.update(_preflight_issue_copy(issue))
                    help_label.display = True
                blocker.update(
                    "Fix the highlighted setup field, then retry the setup check."
                )
            elif dirty_restart and not is_current_success:
                blocker.update(
                    "Network exposed: Check draft before Restart is available."
                    if _draft_network_exposed(self._draft)
                    else "Check draft before Restart is available."
                )
            elif not is_current_success:
                blocker.update("Check setup before Start is available.")
            elif self._preflight is not None and self._preflight.network_exposed:
                blocker.update(
                    "Network exposed: Restart will accept non-loopback connections."
                    if dirty_restart
                    else "Network exposed: Start will accept non-loopback connections."
                )
            else:
                blocker.update(
                    "Setup checks passed. Start will launch a Chatbook-owned server."
                )
            self.call_after_refresh(self._sync_fold_cue)
            self.call_after_refresh(self._restore_top_scan_if_unfocused)
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
        state = (
            self._state if self._profiles_ready else VllmReadinessState.NOT_CONFIGURED
        )
        connection = self._connection if self._profiles_ready else None
        readiness = state_copy[state]
        if (
            self._profiles_ready
            and connection is not None
            and connection.target is not None
        ):
            readiness = (
                f"Ready at {connection.launch_snapshot.client_api_url}"
                if connection.launch_snapshot is not None
                else "Ready · Existing vLLM server"
            )
        self.query_one("#vllm-readiness-state", Label).update(readiness)
        self._render_readiness_checklist()

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
            "models_discovered": "Choose one returned model",
            "process_alive": "Server process is running",
            "process_exited": "Server process exited",
            "preflight_failed": "Setup check needs attention",
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
        if state is VllmReadinessState.NEEDS_ATTENTION:
            details.collapsed = False

    def _render_readiness_checklist(self) -> None:
        """Keep four stable setup checks visible with bounded recovery copy."""

        local = self._draft.mode is VllmMode.LOCAL
        checking = bool(
            self._profiles_ready and self._state is VllmReadinessState.CHECKING
        )
        current_preflight = bool(
            self._profiles_ready
            and self._preflight is not None
            and self._preflight.fingerprint == semantic_fingerprint(self._draft)
        )
        issues = self._preflight.issues if current_preflight and self._preflight else ()
        issue_fields = {issue.field for issue in issues}

        if not local:
            environment = "— Environment · managed by existing server"
            installation = "— vLLM installation · managed by existing server"
        elif checking:
            environment = "… Environment · checking"
            installation = "… vLLM installation · checking"
        elif current_preflight:
            environment = (
                "✕ Environment · choose or repair Python"
                if "python_environment" in issue_fields
                else "✓ Environment · "
                + (self._preflight.python_version or "Python resolved")
            )
            if self._preflight.repair_only:
                installation = "○ vLLM installation · not checked"
            else:
                installation = (
                    "✕ vLLM installation · install or repair vLLM"
                    if any(
                        issue.code
                        in {"vllm_cli_unavailable", "vllm_import_unavailable"}
                        for issue in issues
                    )
                    else "✓ vLLM installation · "
                    + (self._preflight.vllm_version or "vLLM resolved")
                )
        else:
            environment = "○ Environment · not checked"
            installation = "○ vLLM installation · not checked"

        connection = self._connection if self._profiles_ready else None
        if checking:
            model = "… Model · checking"
            network = "… Network · checking"
        elif connection is not None and connection.target is not None:
            model = "✓ Model · exact selection verified"
            network = "✓ Network · API reachable"
        elif not local and connection is not None and connection.discovered_model_ids:
            model = "○ Model · choose one returned model"
            network = "✓ Network · API reachable"
        elif current_preflight:
            model = (
                "✕ Model · choose or repair the model"
                if "model_value" in issue_fields or "model" in issue_fields
                else "✓ Model · selected"
            )
            if self._preflight.repair_only:
                network = "○ Network · not checked"
            elif "bind_address" in issue_fields or "port" in issue_fields:
                network = "✕ Network · repair bind address or port"
            elif self._preflight.network_exposed:
                network = "! Network · reachable beyond this computer"
            else:
                network = "✓ Network · local only"
        else:
            model = (
                "○ Model · choose a model"
                if local and not self._draft.model_value.strip()
                else "○ Model · not checked"
            )
            network = (
                "! Network · draft reaches beyond this computer"
                if _draft_network_exposed(self._draft)
                else "○ Network · not checked"
            )

        for widget_id, copy in (
            ("vllm-check-environment", environment),
            ("vllm-check-installation", installation),
            ("vllm-check-model", model),
            ("vllm-check-network", network),
        ):
            self.query_one(f"#{widget_id}", Label).update(copy)

    def _change_draft(self, **changes: object) -> None:
        candidate = replace(self._draft, **changes)
        if semantic_fingerprint(candidate) == semantic_fingerprint(self._draft):
            return
        self._draft = candidate
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
        elif event.input.id in {
            "vllm-tensor-parallel-size",
            "vllm-maximum-model-length",
        }:
            field = {
                "vllm-tensor-parallel-size": "tensor_parallel_size",
                "vllm-maximum-model-length": "maximum_model_length",
            }[event.input.id]
            try:
                value: object = int(event.value) if event.value.strip() else None
            except ValueError:
                value = event.value
            self._change_draft(**{field: value})
        elif event.input.id == "vllm-gpu-memory-utilization":
            try:
                utilization: object = (
                    float(event.value) if event.value.strip() else None
                )
            except ValueError:
                utilization = event.value
            self._change_draft(gpu_memory_utilization=utilization)

    @on(TextArea.Changed, "#vllm-raw-arguments")
    def _on_raw_arguments_changed(self, event: TextArea.Changed) -> None:
        if self._rendering:
            return
        self._change_draft(raw_arguments=event.text_area.text)

    @on(Select.Changed, "#vllm-profile-select")
    def _on_profile_selected(self, event: Select.Changed) -> None:
        if (
            self._rendering
            or self._draft.mode is not VllmMode.LOCAL
            or not isinstance(event.value, str)
        ):
            return
        self.post_message(self.ProfileSelected(event.value))

    @on(Select.Changed, "#vllm-dtype")
    def _on_dtype_selected(self, event: Select.Changed) -> None:
        if self._rendering or not isinstance(event.value, str):
            return
        self._change_draft(dtype=event.value)

    @on(Select.Changed, "#vllm-existing-model")
    def _on_external_model_selected(self, event: Select.Changed) -> None:
        if (
            self._rendering
            or not isinstance(event.value, str)
            or event.value not in self._discovered_model_ids
        ):
            return
        self.post_message(self.ExternalModelSelected(event.value))

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
            case "vllm-browse-python-environment":
                self.post_message(self.PythonEnvironmentBrowseRequested())
            case "vllm-trust-remote-code":
                self._change_draft(trust_remote_code=not self._draft.trust_remote_code)
            case "vllm-check-setup":
                self.post_message(self.CheckRequested(self._draft))
            case "vllm-cancel-check":
                token = (
                    self._connection.current_token
                    if self._connection is not None
                    else None
                )
                if token is not None:
                    self.post_message(self.CancelCheckRequested(token.generation))
            case "vllm-start":
                self.post_message(self.StartRequested(self._draft))
            case "vllm-stop":
                self.post_message(self.StopRequested())
            case "vllm-recovery-primary":
                self.post_message(self.RetryRequested())
            case "vllm-restart":
                snapshot = self._current_launch_snapshot
                if snapshot is not None:
                    self.post_message(
                        self.RestartRequested(
                            self._draft,
                            changed_launch_field_labels(snapshot, self._draft),
                        )
                    )
            case "vllm-use-console":
                self.post_message(self.UseInConsoleRequested())
            case "vllm-make-default":
                self.post_message(self.MakeDefaultRequested())
            case "vllm-profile-create-button":
                name = self.query_one("#vllm-profile-name", Input).value
                self.post_message(self.CreateProfileRequested(name, self._draft))
            case "vllm-profile-save-button":
                self.post_message(
                    self.SaveProfileRequested(
                        self._profiles.selected_profile_id, self._draft
                    )
                )
            case "vllm-profile-rename-button":
                name = self.query_one("#vllm-profile-name", Input).value
                self.post_message(
                    self.RenameProfileRequested(
                        self._profiles.selected_profile_id, name
                    )
                )
            case "vllm-profile-duplicate-button":
                self.post_message(
                    self.DuplicateProfileRequested(self._profiles.selected_profile_id)
                )
            case "vllm-profile-delete-button":
                self.post_message(
                    self.DeleteProfileRequested(self._profiles.selected_profile_id)
                )
