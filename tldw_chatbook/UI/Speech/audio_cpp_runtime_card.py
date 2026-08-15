"""Pure projection and presentation for the Speech Lab audio.cpp runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Collapsible, RichLog, Static

from tldw_chatbook.TTS import AudioCppRuntimeObservation

from ..Workbench.workbench_state import WorkbenchAction
from .speech_action_strip import SpeechActionStrip

AudioCppRuntimeOperation = Literal[
    "test",
    "sample",
    "clone_generate",
    "restart",
    "shutdown",
]
AudioCppSampleState = Literal["not_attempted", "generating", "failed", "ready"]
AudioCppCloneDraftState = Literal["missing", "processing", "invalid", "ready"]


@dataclass(frozen=True, slots=True)
class AudioCppRuntimeCardObservation:
    """One immutable runtime-and-current-sample view used for action projection."""

    runtime: AudioCppRuntimeObservation
    sample_state: AudioCppSampleState = "not_attempted"
    clone_draft_state: AudioCppCloneDraftState = "missing"


@dataclass(frozen=True, slots=True)
class AudioCppRuntimeAction:
    """One lifecycle action and the reason it may not currently run."""

    operation: AudioCppRuntimeOperation
    label: str
    enabled: bool
    disabled_reason: str = ""
    tooltip: str = ""
    progress_label: str = "Working…"
    post_operation_focus: str = "#tts-test-connection-btn"


@dataclass(frozen=True, slots=True)
class AudioCppRuntimeCardProjection:
    """Bounded copy and controls derived from one coherent observation."""

    primary_status: str
    pending_copy: str
    saved_copy: str
    applied_copy: str
    process_copy: str
    endpoint_copy: str
    capability_copy: str
    catalog_copy: str
    artifact_privacy_copy: str
    primary_action: AudioCppRuntimeAction
    restart_action: AudioCppRuntimeAction
    shutdown_action: AudioCppRuntimeAction
    diagnostics_generation_copy: str
    dropped_diagnostics_copy: str
    diagnostic_lines: tuple[str, ...] = field(repr=False)
    saved_binary_path: str | None = field(repr=False)
    saved_server_json_path: str | None = field(repr=False)
    applied_binary_path: str | None = field(repr=False)
    applied_server_json_path: str | None = field(repr=False)


_BUSY_STATES = frozenset({"starting", "draining", "stopping"})
_LIVE_STATES = frozenset({"running", "unhealthy"})


def _action(
    operation: AudioCppRuntimeOperation,
    label: str,
    enabled: bool,
    reason: str = "",
    *,
    tooltip: str = "",
    progress_label: str = "Working…",
    post_operation_focus: str = "#tts-test-connection-btn",
) -> AudioCppRuntimeAction:
    return AudioCppRuntimeAction(
        operation=operation,
        label=label,
        enabled=enabled,
        disabled_reason="" if enabled else reason,
        tooltip=tooltip or (label if enabled else reason),
        progress_label=progress_label,
        post_operation_focus=post_operation_focus,
    )


def project_audio_cpp_unknown_action() -> AudioCppRuntimeAction:
    """Project the safe primary action used before passive runtime truth arrives.

    Returns:
        A Test Connection action that does not reuse stale runtime evidence.
    """

    return _action(
        "test",
        "Test Connection",
        True,
        tooltip="Read and test the current audio.cpp configuration",
        progress_label="Testing…",
    )


def _primary_status(observation: AudioCppRuntimeObservation) -> str:
    process = observation.process
    if observation.service_closed:
        return "[CLOSED] The TTS service is closed."
    if observation.applied_mode == "external" and process.state in {
        "stopped",
        "unavailable",
    }:
        return "[EXTERNAL] Chatbook will connect to the active external server."
    if process.state == "running":
        if process.tts_capability == "available":
            return "[RUNNING] Managed audio.cpp is ready."
        if process.tts_capability == "not_configured":
            return "[RUNNING] The server is healthy, but TTS is not configured."
        return "[RUNNING] The server is healthy; TTS has not been checked."
    messages = {
        "stopped": "Managed audio.cpp is not running.",
        "starting": "Managed audio.cpp is starting and being tested.",
        "unhealthy": "Managed audio.cpp failed its recent health checks.",
        "draining": "Managed audio.cpp is draining admitted speech work.",
        "stopping": "Managed audio.cpp is shutting down.",
        "unavailable": "Managed audio.cpp is unavailable.",
    }
    return f"[{process.state.upper()}] {messages[process.state]}"


def _pending_copy(observation: AudioCppRuntimeObservation) -> str:
    if not observation.pending_configuration:
        return "Saved and active audio.cpp settings match."
    saved = observation.saved_configuration_generation
    applied = observation.applied_configuration_generation
    if observation.saved_mode == "external" and observation.applied_mode == "managed":
        return (
            f"External mode is saved at generation {saved}; managed generation "
            f"{applied} remains active until settings are applied."
        )
    if observation.saved_mode == "managed" and observation.applied_mode == "external":
        return (
            f"Managed mode is saved at generation {saved}; External generation "
            f"{applied} remains active until the next deliberate audio.cpp action."
        )
    return f"Saved generation {saved} is pending; generation {applied} remains active."


def _runtime_actions(
    card_observation: AudioCppRuntimeCardObservation,
) -> tuple[AudioCppRuntimeAction, AudioCppRuntimeAction, AudioCppRuntimeAction]:
    observation = card_observation.runtime
    state = observation.process.state
    busy_reason = "A managed audio.cpp lifecycle change is already in progress."
    no_child_reason = "No managed audio.cpp server is active."
    external_reason = "External mode does not own a server process."
    closed_reason = "The TTS service is closed."

    if observation.service_closed:
        return (
            _action("test", "Service closed", False, closed_reason),
            _action("restart", "Restart", False, closed_reason),
            _action("shutdown", "Shut down server", False, closed_reason),
        )
    if state in _BUSY_STATES:
        label, operation = {
            "starting": ("Starting & Testing…", "test"),
            "draining": ("Applying Settings…", "restart"),
            "stopping": ("Shutting down…", "shutdown"),
        }[state]
        return (
            _action(operation, label, False, busy_reason),  # type: ignore[arg-type]
            _action("restart", "Restart", False, busy_reason),
            _action("shutdown", "Shut down server", False, busy_reason),
        )

    live_managed = observation.applied_mode == "managed" and state in _LIVE_STATES
    clone_setup_ready = bool(
        observation.clone_setup is not None
        and not observation.pending_configuration
        and live_managed
        and observation.applied_managed_setup_source == "guided"
        and observation.clone_setup.model_id in observation.applied_guided_model_ids
        and observation.tts_capability == "available"
        and observation.catalog_fresh
    )
    if (
        observation.pending_configuration
        and observation.saved_mode == "external"
        and live_managed
    ):
        primary = _action(
            "restart",
            "Apply Settings & Stop Managed Server",
            True,
            tooltip="Apply the saved External settings and stop the managed server",
            progress_label="Applying & Stopping…",
        )
    elif observation.pending_configuration and live_managed:
        primary = _action(
            "restart",
            "Restart & Apply Settings",
            True,
            tooltip="Restart audio.cpp with the latest saved Managed settings",
            progress_label="Restarting & Applying…",
        )
    elif state == "unhealthy" and observation.applied_mode == "managed":
        primary = _action(
            "restart",
            "Restart",
            True,
            tooltip="Restart the unhealthy managed audio.cpp server",
            progress_label="Restarting…",
        )
    elif clone_setup_ready:
        draft_reasons = {
            "missing": "Choose a reference WAV and enter its transcript first.",
            "processing": "Wait for the reference WAV to finish validation.",
            "invalid": "Correct the reference WAV or transcript before generating.",
            "ready": "",
        }
        primary = _action(
            "clone_generate",
            "Create Voice & Generate",
            card_observation.clone_draft_state == "ready",
            draft_reasons[card_observation.clone_draft_state],
            tooltip=(
                "Generate one complete WAV with the staged local voice reference"
                if card_observation.clone_draft_state == "ready"
                else draft_reasons[card_observation.clone_draft_state]
            ),
            progress_label="Creating Voice…",
            post_operation_focus="#audio-play-btn",
        )
    elif observation.saved_mode == "managed" and state in {
        "stopped",
        "unavailable",
    }:
        guided_default = observation.saved_guided_default_model_id
        guided_sample_ready = bool(
            observation.saved_managed_setup_source == "guided"
            and observation.saved_guided_text_ready
            and guided_default is not None
            and guided_default in observation.saved_guided_model_ids
        )
        if observation.clone_setup is not None:
            primary = _action(
                "test",
                "Start & Set Up Voice",
                True,
                tooltip=(
                    "Start audio.cpp and verify the selected clone model before "
                    "setting up its voice reference"
                ),
                progress_label="Starting & Testing…",
                post_operation_focus="#speech-clone-reference-choose",
            )
        elif guided_sample_ready:
            primary = _action(
                "sample",
                "Start & Generate Sample",
                True,
                tooltip=(
                    f"Start audio.cpp, verify {guided_default}, and generate one "
                    "complete WAV"
                ),
                progress_label="Starting & Generating…",
                post_operation_focus="#audio-play-btn",
            )
        else:
            primary = _action(
                "test",
                "Start & Test Connection",
                True,
                progress_label="Starting & Testing…",
            )
    elif (
        observation.pending_configuration
        and observation.saved_mode == "external"
        and not live_managed
    ):
        primary = _action(
            "test",
            "Apply & Test Connection",
            True,
            tooltip="Apply the saved External endpoint and test it",
            progress_label="Applying & Testing…",
        )
    else:
        primary = _action(
            "test",
            "Test Connection",
            True,
            tooltip="Test the active audio.cpp connection and model catalog",
            progress_label="Testing…",
        )

    healthy_guided_sample = bool(
        not observation.pending_configuration
        and live_managed
        and observation.applied_managed_setup_source == "guided"
        and observation.applied_guided_text_ready
        and observation.applied_guided_default_model_id is not None
        and observation.applied_guided_default_model_id
        in observation.applied_guided_model_ids
        and observation.tts_capability == "available"
        and observation.catalog_fresh
    )
    if card_observation.sample_state == "generating" and healthy_guided_sample:
        primary = _action(
            "sample",
            "Generating Sample…",
            False,
            "Sample generation is already in progress.",
            tooltip="Sample generation is already in progress.",
            progress_label="Generating Sample…",
            post_operation_focus="#audio-play-btn",
        )
    elif card_observation.sample_state == "failed" and healthy_guided_sample:
        primary = _action(
            "sample",
            "Retry Sample",
            True,
            tooltip="Generate another complete WAV with the active guided model",
            progress_label="Generating Sample…",
            post_operation_focus="#audio-play-btn",
        )

    duplicate_restart = live_managed and primary.operation == "restart"
    restart_reason = (
        "Use the primary action above to restart the managed server."
        if duplicate_restart
        else external_reason
        if observation.applied_mode == "external"
        else no_child_reason
    )
    restart = _action(
        "restart",
        "Restart",
        live_managed and not duplicate_restart,
        restart_reason,
        tooltip=(
            "Restart the active managed audio.cpp server"
            if live_managed and not duplicate_restart
            else restart_reason
        ),
        progress_label="Restarting…",
    )
    shutdown = _action(
        "shutdown",
        "Shut down server",
        live_managed,
        external_reason if observation.applied_mode == "external" else no_child_reason,
        tooltip=(
            "Shut down the active managed audio.cpp server"
            if live_managed
            else external_reason
            if observation.applied_mode == "external"
            else no_child_reason
        ),
        progress_label="Shutting down…",
    )
    return primary, restart, shutdown


def _configuration_copy(
    observation: AudioCppRuntimeObservation,
    *,
    saved: bool,
) -> str:
    """Return one path-free saved or applied configuration summary."""

    prefix = "Saved" if saved else "Active"
    mode = observation.saved_mode if saved else observation.applied_mode
    generation = (
        observation.saved_configuration_generation
        if saved
        else observation.applied_configuration_generation
    )
    if mode == "external":
        return f"{prefix}: External · generation {generation}"
    source = (
        observation.saved_managed_setup_source
        if saved
        else observation.applied_managed_setup_source
    )
    if source != "guided":
        return f"{prefix}: Managed · Manual server.json · generation {generation}"
    model_ids = (
        observation.saved_guided_model_ids
        if saved
        else observation.applied_guided_model_ids
    )
    default_model = (
        observation.saved_guided_default_model_id
        if saved
        else observation.applied_guided_default_model_id
    )
    model_copy = "1 model" if len(model_ids) == 1 else f"{len(model_ids)} models"
    default_copy = default_model or "None"
    return (
        f"{prefix}: Managed · Guided · {model_copy} · default {default_copy} · "
        f"generation {generation}"
    )


def _artifact_privacy_copy(posture: str) -> str:
    detail = {
        "windows_account_protected": (
            "Account protected · administrators and SYSTEM retain access · "
            "plaintext, not encryption"
        ),
        "posix_owner_only": ("Owner-only filesystem mode · plaintext, not encryption"),
        "unverified": "Not verified — review required",
        "not_applicable": (
            "Not active · protection is applied only for a Guided start"
        ),
    }.get(posture, "Not verified — review required")
    return f"Generated launch privacy: {detail}"


def project_audio_cpp_runtime_card(
    observation: AudioCppRuntimeObservation | AudioCppRuntimeCardObservation,
) -> AudioCppRuntimeCardProjection:
    """Project one service observation without I/O or lifecycle work.

    Args:
        observation: Coherent saved, applied, process, and catalog state.

    Returns:
        Bounded display copy and lifecycle-action state for the runtime card.
    """

    card_observation = (
        observation
        if isinstance(observation, AudioCppRuntimeCardObservation)
        else AudioCppRuntimeCardObservation(runtime=observation)
    )
    runtime = card_observation.runtime
    process = runtime.process
    primary, restart, shutdown = _runtime_actions(card_observation)
    catalog_state = "Fresh" if runtime.catalog_fresh else "Stale"
    catalog_copy = (
        "Catalog: Not checked"
        if runtime.catalog_revision is None
        else f"Catalog: {catalog_state} · revision {runtime.catalog_revision}"
    )
    dropped = process.dropped_diagnostic_lines
    dropped_copy = (
        "No diagnostic lines were dropped."
        if dropped == 0
        else "1 older line was dropped."
        if dropped == 1
        else f"{dropped} older lines were dropped."
    )
    return AudioCppRuntimeCardProjection(
        primary_status=_primary_status(runtime),
        pending_copy=_pending_copy(runtime),
        saved_copy=(_configuration_copy(runtime, saved=True)),
        applied_copy=(_configuration_copy(runtime, saved=False)),
        process_copy=(
            f"Process: {process.state.title()} · generation "
            f"{process.process_generation}"
        ),
        endpoint_copy=(
            f"Active endpoint: {runtime.active_endpoint or process.endpoint or 'None'}"
        ),
        capability_copy=(
            f"TTS capability: {runtime.tts_capability.replace('_', ' ').title()}"
        ),
        catalog_copy=catalog_copy,
        artifact_privacy_copy=_artifact_privacy_copy(
            process.generated_artifact_privacy_posture
        ),
        primary_action=primary,
        restart_action=restart,
        shutdown_action=shutdown,
        diagnostics_generation_copy=(
            f"Process generation: {process.process_generation}"
        ),
        dropped_diagnostics_copy=dropped_copy,
        diagnostic_lines=tuple(
            f"{line.stream.upper()} · {line.text}" for line in process.diagnostics
        ),
        saved_binary_path=runtime.saved_managed_binary_path,
        saved_server_json_path=runtime.saved_managed_server_json_path,
        applied_binary_path=runtime.applied_managed_binary_path,
        applied_server_json_path=runtime.applied_managed_server_json_path,
    )


class AudioCppRuntimeCard(Vertical):
    """Focus-stable runtime summary; the owning pane performs all async work."""

    def __init__(
        self,
        observation: AudioCppRuntimeObservation | None = None,
        **kwargs: object,
    ) -> None:
        classes = str(kwargs.pop("classes", ""))
        super().__init__(classes=f"audio-cpp-runtime-card {classes}".strip(), **kwargs)
        self.observation = observation
        self._rendered_diagnostic_lines: tuple[str, ...] | None = None

    def compose(self) -> ComposeResult:
        """Compose the stable runtime, action, detail, and diagnostic controls.

        Yields:
            Always-mounted children updated in place by runtime observations.
        """

        yield Static(
            "audio.cpp runtime",
            classes="speech-section-head",
            markup=False,
        )
        yield Static(
            "[NOT CHECKED] Waiting for passive runtime status.",
            id="audio-cpp-runtime-status",
            classes="audio-cpp-runtime-status",
            markup=False,
        )
        yield Static(
            "No saved-versus-active observation yet.",
            id="audio-cpp-runtime-pending",
            classes="audio-cpp-runtime-pending",
            markup=False,
        )
        yield SpeechActionStrip(
            (
                WorkbenchAction(
                    id="audio-cpp-runtime-restart",
                    label="Restart",
                    disabled=True,
                ),
                WorkbenchAction(
                    id="audio-cpp-runtime-shutdown",
                    label="Shut down server",
                    disabled=True,
                ),
                WorkbenchAction(
                    id="audio-cpp-runtime-open-settings",
                    label="Global Settings",
                    tooltip="Edit durable audio.cpp setup",
                ),
            ),
            id="audio-cpp-runtime-actions",
        )
        yield Static(
            "Runtime status is still loading.",
            id="audio-cpp-runtime-action-reason",
            classes="audio-cpp-runtime-reason",
            markup=False,
        )
        yield Collapsible(
            Static("Saved: Not observed", id="audio-cpp-runtime-saved", markup=False),
            Static(
                "Active: Not observed", id="audio-cpp-runtime-applied", markup=False
            ),
            Static(
                "Process: Not observed", id="audio-cpp-runtime-process", markup=False
            ),
            Static(
                "Active endpoint: None", id="audio-cpp-runtime-endpoint", markup=False
            ),
            Static(
                "TTS capability: Unknown",
                id="audio-cpp-runtime-capability",
                markup=False,
            ),
            Static(
                "Catalog: Not checked", id="audio-cpp-runtime-catalog", markup=False
            ),
            Static(
                "Generated launch privacy: Not active",
                id="audio-cpp-runtime-artifact-privacy",
                markup=False,
            ),
            Static(
                "Saved binary: None", id="audio-cpp-runtime-saved-binary", markup=False
            ),
            Static(
                "Saved server.json: None",
                id="audio-cpp-runtime-saved-json",
                markup=False,
            ),
            Static(
                "Active binary: None",
                id="audio-cpp-runtime-applied-binary",
                markup=False,
            ),
            Static(
                "Active server.json: None",
                id="audio-cpp-runtime-applied-json",
                markup=False,
            ),
            title="Runtime details",
            id="audio-cpp-runtime-details",
            collapsed=True,
        )
        yield Collapsible(
            Static(
                "Potentially sensitive: child output may contain model or request "
                "details. Focus the log and use Arrow or Page Up/Page Down to review "
                "all lines. The next managed start clears it.",
                id="audio-cpp-diagnostics-warning",
                classes="audio-cpp-diagnostics-warning",
                markup=False,
            ),
            Static(
                "Process generation: Not observed",
                id="audio-cpp-diagnostics-generation",
                markup=False,
            ),
            RichLog(
                max_lines=200,
                min_width=0,
                wrap=True,
                highlight=False,
                markup=False,
                auto_scroll=True,
                id="audio-cpp-diagnostics-lines",
                classes="audio-cpp-diagnostics-lines",
            ),
            Static(
                "No diagnostic lines were dropped.",
                id="audio-cpp-diagnostics-dropped",
                markup=False,
            ),
            title="Recent managed diagnostics",
            id="audio-cpp-runtime-diagnostics",
            collapsed=True,
        )

    def on_mount(self) -> None:
        if self.observation is not None:
            self.apply_observation(self.observation)

    def apply_observation(
        self,
        observation: AudioCppRuntimeObservation | AudioCppRuntimeCardObservation,
    ) -> AudioCppRuntimeCardProjection:
        """Update mounted children in place and return the pure projection.

        Args:
            observation: Coherent saved, applied, process, and catalog state.

        Returns:
            The projection rendered by the card, or available to an unmounted caller.
        """

        projection = project_audio_cpp_runtime_card(observation)
        self.observation = (
            observation.runtime
            if isinstance(observation, AudioCppRuntimeCardObservation)
            else observation
        )
        if not self.is_mounted:
            return projection
        copies = {
            "#audio-cpp-runtime-status": projection.primary_status,
            "#audio-cpp-runtime-pending": projection.pending_copy,
            "#audio-cpp-runtime-saved": projection.saved_copy,
            "#audio-cpp-runtime-applied": projection.applied_copy,
            "#audio-cpp-runtime-process": projection.process_copy,
            "#audio-cpp-runtime-endpoint": projection.endpoint_copy,
            "#audio-cpp-runtime-capability": projection.capability_copy,
            "#audio-cpp-runtime-catalog": projection.catalog_copy,
            "#audio-cpp-runtime-artifact-privacy": (projection.artifact_privacy_copy),
            "#audio-cpp-runtime-saved-binary": (
                f"Saved binary: {projection.saved_binary_path or 'None'}"
            ),
            "#audio-cpp-runtime-saved-json": (
                f"Saved server.json: {projection.saved_server_json_path or 'None'}"
            ),
            "#audio-cpp-runtime-applied-binary": (
                f"Active binary: {projection.applied_binary_path or 'None'}"
            ),
            "#audio-cpp-runtime-applied-json": (
                f"Active server.json: {projection.applied_server_json_path or 'None'}"
            ),
            "#audio-cpp-diagnostics-generation": (
                projection.diagnostics_generation_copy
            ),
            "#audio-cpp-diagnostics-dropped": (projection.dropped_diagnostics_copy),
        }
        for selector, copy in copies.items():
            self.query_one(selector, Static).update(copy)
        diagnostic_lines = projection.diagnostic_lines or (
            "No recent managed diagnostics.",
        )
        if diagnostic_lines != self._rendered_diagnostic_lines:
            diagnostic_log = self.query_one("#audio-cpp-diagnostics-lines", RichLog)
            diagnostic_log.clear()
            for line in diagnostic_lines:
                diagnostic_log.write(line)
            self._rendered_diagnostic_lines = diagnostic_lines
        self._apply_action("#audio-cpp-runtime-restart", projection.restart_action)
        self._apply_action("#audio-cpp-runtime-shutdown", projection.shutdown_action)
        reasons = tuple(
            action.disabled_reason
            for action in (
                projection.primary_action,
                projection.restart_action,
                projection.shutdown_action,
            )
            if action.disabled_reason
        )
        self.query_one("#audio-cpp-runtime-action-reason", Static).update(
            reasons[0] if reasons else "Lifecycle actions are available."
        )
        return projection

    def _apply_action(self, selector: str, action: AudioCppRuntimeAction) -> None:
        try:
            button = self.query_one(selector, Button)
        except NoMatches:
            return
        button.label = action.label
        button.disabled = not action.enabled
        button.tooltip = action.tooltip
