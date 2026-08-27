"""Direct-user raw CLI submission outside the provider prompt queue."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
import threading
from time import monotonic
from typing import Any
import uuid

from ...Chat.console_chat_models import (
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleMessageRole,
    RawCliLifecycleState,
    RawCliPresentation,
)
from ...Chat.console_chat_store import RawCliMarkerTransitionError
from ...Chat.console_message_actions import ConsoleMessageActionService
from ...Tools.raw_cli_executor import (
    MAX_RAW_TIMEOUT_SECONDS,
    MAX_RAW_PREVIEW_BYTES,
    RawCliRequest,
    RawCliResult,
    RawCliStreamEvent,
    validate_raw_cli_request,
)
from ...Widgets.Console.console_composer_bar import (
    ConsoleDraftStash,
    classify_console_raw_draft,
)

_EMPTY_REFUSAL = "Raw CLI command is empty. The exact draft was restored."
_LOCKED_REFUSAL = (
    "Raw CLI is Locked. Unlock it in Settings > Privacy & Security, then arm it "
    "for this launch. The exact draft was restored."
)
_UNARMED_REFUSAL = (
    "Raw CLI is Unlocked but not armed for this launch. Arm it in Settings > "
    "Privacy & Security. The exact draft was restored."
)
_AUTHORITY_CHANGED_REFUSAL = (
    "Raw CLI authority changed before launch. The exact draft was restored."
)
_CONTAINMENT_REFUSAL = (
    "Raw CLI did not launch because authority changed or process containment "
    "could not be established. The exact draft was restored."
)

_RAW_CLI_REPAINT_SECONDS = 0.05
_RAW_CLI_COMPACT_OUTPUT_BYTES = 4 * 1024


def _literal_terminal_text(value: str) -> str:
    """Make terminal controls visible while leaving ordinary text literal."""
    return "".join(
        character
        if character in "\n\t"
        or not (ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F)
        else f"\\x{ord(character):02x}"
        for character in value
    )


def _utf8_prefix(value: str, byte_limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= byte_limit:
        return value, False
    return encoded[:byte_limit].decode("utf-8", errors="ignore"), True


def _bounded_stream_append(
    stdout: str,
    stderr: str,
    stream: str,
    delta: str,
) -> tuple[str, str, bool]:
    """Append within the one combined stdout+stderr preview budget."""
    remaining = MAX_RAW_PREVIEW_BYTES - len(
        stdout.encode("utf-8") + stderr.encode("utf-8")
    )
    if remaining <= 0:
        return stdout, stderr, bool(delta)
    accepted, clipped = _utf8_prefix(delta, remaining)
    if stream == "stdout":
        stdout += accepted
    else:
        stderr += accepted
    return stdout, stderr, clipped


def _raw_cli_output(stdout: str, stderr: str) -> str:
    safe_stdout = _literal_terminal_text(stdout) or "(no output)"
    safe_stderr = _literal_terminal_text(stderr) or "(no output)"
    return f"stdout:\n{safe_stdout}\n\nstderr:\n{safe_stderr}"


def _raw_cli_content(
    presentation: RawCliPresentation,
    stdout: str,
    stderr: str,
) -> tuple[str, str]:
    full_output = _raw_cli_output(stdout, stderr)
    compact_output, clipped = _utf8_prefix(
        full_output,
        _RAW_CLI_COMPACT_OUTPUT_BYTES,
    )
    if clipped:
        compact_output += "\n… output preview clipped; use Full output"
    exit_code = (
        "Pending" if presentation.exit_code is None else str(presentation.exit_code)
    )
    cleanup = {
        None: "Pending",
        True: "Proven",
        False: "Unproven",
    }[presentation.cleanup_proven]
    content = (
        f"Command:\n{_literal_terminal_text(presentation.command)}\n\n"
        f"Caller: {presentation.caller.title()}\n"
        f"Shell: {_literal_terminal_text(presentation.shell)}\n"
        f"CWD: {_literal_terminal_text(presentation.cwd)}\n"
        f"Elapsed: {presentation.elapsed_seconds:.1f}s\n"
        f"Exit code: {exit_code}\n"
        f"Truncated: {'Yes' if presentation.truncated else 'No'}\n"
        f"Cleanup: {cleanup}\n\n"
        f"{compact_output}"
    )
    return content, full_output


def _terminal_lifecycle(result: RawCliResult) -> RawCliLifecycleState:
    if result.terminal_state in {
        "exited",
        "timed_out",
        "cancelled",
        "cleanup_unproven",
    }:
        return result.terminal_state
    return "failed"


def _activity_for_raw_cli(
    lifecycle_state: RawCliLifecycleState,
    exit_code: int | None,
) -> ConsoleActivityPresentation:
    if lifecycle_state == "exited" and exit_code == 0:
        status = "success"
    elif lifecycle_state in {"starting", "running", "stopping"}:
        status = "done"
    elif lifecycle_state in {"timed_out", "cancelled"}:
        status = "blocked"
    else:
        status = "failed"
    return ConsoleActivityPresentation("tool", "Raw CLI", status)


def restore_refused_raw_cli_stash(
    session_id: str | None,
    stash: ConsoleDraftStash,
    *,
    composer: Any | None,
    active_session_id: str | None,
    visible_session_id: str | None,
) -> bool:
    """Restore only when the composer unambiguously belongs to the origin."""
    if composer is None:
        return False
    if active_session_id != session_id or visible_session_id != session_id:
        return False
    composer.restore_stashed_draft(stash)
    return True


class ConsoleRawCliController:
    """Capture and start one trusted user command without touching a provider."""

    def __init__(
        self,
        *,
        raw_cli_runtime: Callable[[], Any],
        active_session_id: Callable[[], str | None],
        persisted_leaf_anchor: Callable[[str], str | None],
        selected_local_root: Callable[[str], Path | None],
        private_scratch_root: Callable[[str], Path],
        refusal_stash_bank: dict[str, list[Any]],
        accepts_raw_cli_refusal_callbacks: Callable[[], bool],
        restore_stash: Callable[[str | None, ConsoleDraftStash], bool],
        append_local_error: Callable[[str | None, str], None],
        append_store_marker: Callable[..., Any],
        update_store_marker: Callable[..., Any],
        agent_runs_db: Callable[[], Any],
        run_log_access: Callable[[], Any],
        start_worker: Callable[..., Any],
        marshal_to_ui: Callable[..., Any],
        schedule_projection: Callable[[str], None] = lambda _session_id: None,
    ) -> None:
        self._raw_cli_runtime = raw_cli_runtime
        self._active_session_id = active_session_id
        self._persisted_leaf_anchor = persisted_leaf_anchor
        self._selected_local_root = selected_local_root
        self._private_scratch_root = private_scratch_root
        self._banked_stashes_by_session = refusal_stash_bank
        self._accepts_raw_cli_refusal_callbacks = (
            accepts_raw_cli_refusal_callbacks
        )
        self._restore_stash = restore_stash
        self._append_local_error = append_local_error
        # Task 7/8 consume these already-wired boundaries; Task 6 must not
        # invent an interim persistence or transcript representation.
        self._append_store_marker = append_store_marker
        self._update_store_marker = update_store_marker
        self._agent_runs_db = agent_runs_db
        self._run_log_access = run_log_access
        self._start_worker = start_worker
        self._marshal_to_ui = marshal_to_ui
        self._schedule_projection = schedule_projection

    def start_user_command(self, stash: ConsoleDraftStash) -> bool:
        """Start one trusted raw stash in its own non-exclusive thread worker."""
        classified = classify_console_raw_draft(stash)
        if classified.kind != "raw":
            return False
        session_id = self._active_session_id()
        if not classified.text.strip():
            return self._refuse(session_id, stash, _EMPTY_REFUSAL)

        runtime = self._raw_cli_runtime()
        if runtime.permitted is not True:
            return self._refuse(session_id, stash, _LOCKED_REFUSAL)
        if runtime.armed is not True:
            return self._refuse(session_id, stash, _UNARMED_REFUSAL)

        if not session_id:
            return self._refuse(
                session_id,
                stash,
                "Raw CLI needs an active Console session. The exact draft was restored.",
            )
        try:
            request = RawCliRequest(
                invocation_id=uuid.uuid4().hex,
                caller="user",
                command=classified.text,
                shell="auto",
                initial_directory=self._submission_root(session_id),
                timeout_seconds=MAX_RAW_TIMEOUT_SECONDS,
                console_session_id=session_id,
                transcript_anchor_id=self._persisted_leaf_anchor(session_id),
            )
            validate_raw_cli_request(request)
        except ValueError as exc:
            return self._refuse(
                session_id, stash, f"Raw CLI refused: {exc}. Draft restored."
            )
        except Exception:  # noqa: BLE001 -- submission snapshot fails locally
            return self._refuse(
                session_id,
                stash,
                "Raw CLI could not capture this Console context. The exact draft "
                "was restored.",
            )

        try:
            self._start_worker(
                partial(self._execute, runtime, request, session_id, stash),
                thread=True,
                exclusive=False,
                name=f"console-raw-cli-{request.invocation_id}",
            )
        except Exception:  # noqa: BLE001 -- failed worker admission is local refusal
            return self._refuse(
                session_id,
                stash,
                "Raw CLI worker could not start. The exact draft was restored.",
            )
        return True

    def _submission_root(self, session_id: str) -> Path:
        selected = self._selected_local_root(session_id)
        root = (
            selected if selected is not None else self._private_scratch_root(session_id)
        )
        return Path(root).resolve()

    def _execute(
        self,
        runtime: Any,
        request: RawCliRequest,
        session_id: str,
        stash: ConsoleDraftStash,
    ) -> None:
        started_at: float | None = None
        stdout = ""
        stderr = ""
        stream_truncated = False
        marker_started = False
        repaint_lock = threading.Lock()
        repaint_timer: threading.Timer | None = None
        pending_repaint: tuple[str, str, bool] | None = None
        terminal = False

        def flush_repaint() -> None:
            nonlocal repaint_timer, pending_repaint
            with repaint_lock:
                repaint_timer = None
                if terminal or pending_repaint is None or started_at is None:
                    return
                pending = pending_repaint
                pending_repaint = None
                self._marshal_to_ui(
                    self._update_running_marker,
                    request,
                    session_id,
                    started_at,
                    *pending,
                )

        def cancel_repaint() -> None:
            nonlocal repaint_timer, pending_repaint, terminal
            with repaint_lock:
                terminal = True
                timer = repaint_timer
                repaint_timer = None
                pending_repaint = None
            if timer is not None:
                timer.cancel()

        def on_registered() -> None:
            nonlocal marker_started
            self._marshal_to_ui(
                self._append_starting_marker,
                request,
                session_id,
            )
            marker_started = True

        def on_started(timestamp: float) -> None:
            nonlocal started_at
            started_at = timestamp
            self._marshal_to_ui(
                self._update_running_marker,
                request,
                session_id,
                timestamp,
                "",
                "",
                False,
            )

        def on_event(event: RawCliStreamEvent) -> None:
            nonlocal stdout, stderr, stream_truncated, repaint_timer
            nonlocal pending_repaint
            with repaint_lock:
                if terminal:
                    return
                stdout, stderr, clipped = _bounded_stream_append(
                    stdout,
                    stderr,
                    event.stream,
                    event.text,
                )
                stream_truncated = stream_truncated or event.truncated or clipped
                pending_repaint = (stdout, stderr, stream_truncated)
                if repaint_timer is not None:
                    return
                repaint_timer = threading.Timer(
                    _RAW_CLI_REPAINT_SECONDS,
                    flush_repaint,
                )
                repaint_timer.daemon = True
                repaint_timer.start()

        try:
            result = runtime.execute(
                request,
                on_event,
                on_registered=on_registered,
                on_started=on_started,
            )
        except Exception:  # noqa: BLE001 -- runtime already owns diagnostic detail
            cancel_repaint()
            if marker_started:
                self._marshal_to_ui(
                    self._fail_running_marker,
                    request,
                    session_id,
                    started_at,
                    stdout,
                    stderr,
                    stream_truncated,
                )
            else:
                self._marshal_to_ui(
                    self._append_execution_error,
                    session_id,
                    "Raw CLI execution failed locally.",
                )
            return
        cancel_repaint()
        if marker_started:
            self._marshal_to_ui(
                self._finish_marker,
                request,
                session_id,
                started_at,
                result,
            )
        if result.terminal_state == "refused":
            self._marshal_to_ui(
                self._refuse,
                session_id,
                stash,
                _AUTHORITY_CHANGED_REFUSAL,
            )
        elif result.terminal_state == "containment_unavailable":
            self._marshal_to_ui(
                self._refuse,
                session_id,
                stash,
                _CONTAINMENT_REFUSAL,
            )

    @staticmethod
    def _marker_id(invocation_id: str) -> str:
        return f"raw-cli-{invocation_id}"

    def _append_starting_marker(
        self,
        request: RawCliRequest,
        session_id: str,
    ) -> None:
        presentation = RawCliPresentation(
            invocation_id=request.invocation_id,
            caller=request.caller,
            lifecycle_state="starting",
            command=request.command,
            shell=request.shell,
            cwd=str(request.initial_directory),
            started_at_monotonic=None,
            elapsed_seconds=0.0,
            exit_code=None,
            truncated=False,
            cleanup_proven=None,
        )
        content, full_output = _raw_cli_content(presentation, "", "")
        self._append_store_marker(
            session_id,
            role=ConsoleMessageRole.TOOL,
            content=content,
            tool_output_full=full_output,
            activity_presentation=_activity_for_raw_cli("starting", None),
            raw_cli_presentation=presentation,
            record_trajectory=False,
            message_id=self._marker_id(request.invocation_id),
        )
        self._schedule_projection(session_id)

    def _update_running_marker(
        self,
        request: RawCliRequest,
        session_id: str,
        started_at: float,
        stdout: str,
        stderr: str,
        truncated: bool,
    ) -> None:
        presentation = RawCliPresentation(
            invocation_id=request.invocation_id,
            caller=request.caller,
            lifecycle_state="running",
            command=request.command,
            shell=request.shell,
            cwd=str(request.initial_directory),
            started_at_monotonic=started_at,
            elapsed_seconds=max(0.0, monotonic() - started_at),
            exit_code=None,
            truncated=bool(truncated),
            cleanup_proven=None,
        )
        content, full_output = _raw_cli_content(presentation, stdout, stderr)
        try:
            self._update_store_marker(
                session_id,
                self._marker_id(request.invocation_id),
                content=content,
                tool_output_full=full_output,
                activity_presentation=_activity_for_raw_cli("running", None),
                raw_cli_presentation=presentation,
            )
        except (KeyError, RawCliMarkerTransitionError):
            return
        self._schedule_projection(session_id)

    def _finish_marker(
        self,
        request: RawCliRequest,
        session_id: str,
        started_at: float | None,
        result: RawCliResult,
    ) -> None:
        lifecycle = _terminal_lifecycle(result)
        presentation = RawCliPresentation(
            invocation_id=request.invocation_id,
            caller=request.caller,
            lifecycle_state=lifecycle,
            command=request.command,
            shell=result.resolved_shell or request.shell,
            cwd=str(result.initial_directory),
            started_at_monotonic=started_at,
            elapsed_seconds=result.elapsed_seconds,
            exit_code=result.exit_code,
            truncated=result.truncated,
            cleanup_proven=result.cleanup_proven,
        )
        content, full_output = _raw_cli_content(
            presentation,
            result.stdout_preview,
            result.stderr_preview,
        )
        try:
            self._update_store_marker(
                session_id,
                self._marker_id(request.invocation_id),
                content=content,
                tool_output_full=full_output,
                activity_presentation=_activity_for_raw_cli(
                    lifecycle,
                    result.exit_code,
                ),
                raw_cli_presentation=presentation,
            )
        except KeyError:
            return
        self._schedule_projection(session_id)

    def _fail_running_marker(
        self,
        request: RawCliRequest,
        session_id: str,
        started_at: float | None,
        stdout: str,
        stderr: str,
        truncated: bool,
    ) -> None:
        result = RawCliResult(
            invocation_id=request.invocation_id,
            caller=request.caller,
            resolved_shell=request.shell,
            initial_directory=request.initial_directory,
            elapsed_seconds=(
                0.0 if started_at is None else max(0.0, monotonic() - started_at)
            ),
            stdout_preview=stdout,
            stderr_preview=stderr,
            record_output="",
            exit_code=None,
            terminal_state="spawn_failed",
            truncated=truncated,
            cleanup_proven=False,
        )
        self._finish_marker(request, session_id, started_at, result)

    def stop_user_command(self, marker: ConsoleChatMessage) -> bool:
        """Stop exactly one active raw marker through its invocation id."""
        action = ConsoleMessageActionService().dispatch("raw-cli-stop", marker)
        raw_cli = marker.raw_cli_presentation
        session_id = self._active_session_id()
        if (
            action.status != "completed"
            or action.target_invocation_id is None
            or raw_cli is None
            or session_id is None
        ):
            return False
        stopping = replace(
            raw_cli,
            lifecycle_state="stopping",
            elapsed_seconds=(
                raw_cli.elapsed_seconds
                if raw_cli.started_at_monotonic is None
                else max(
                    raw_cli.elapsed_seconds,
                    monotonic() - raw_cli.started_at_monotonic,
                )
            ),
        )
        try:
            self._update_store_marker(
                session_id,
                marker.id,
                activity_presentation=_activity_for_raw_cli("stopping", None),
                raw_cli_presentation=stopping,
            )
        except KeyError:
            return False
        self._schedule_projection(session_id)
        self._raw_cli_runtime().cancel(action.target_invocation_id)
        return True

    def _refuse(
        self,
        session_id: str | None,
        stash: ConsoleDraftStash,
        message: str,
    ) -> bool:
        if not self._refusal_callbacks_are_open():
            return False
        restored = self._restore_stash(session_id, stash)
        if not restored and session_id is not None:
            self._banked_stashes_by_session.setdefault(session_id, []).append(stash)
        self._append_local_error(session_id, message)
        return False

    def _append_execution_error(self, session_id: str, message: str) -> None:
        """Append a worker failure only while completion callbacks are open."""
        if self._refusal_callbacks_are_open():
            self._append_local_error(session_id, message)

    def _refusal_callbacks_are_open(self) -> bool:
        """Read the app-owned terminal fence, failing closed on teardown."""
        try:
            return self._accepts_raw_cli_refusal_callbacks() is True
        except Exception:  # noqa: BLE001 -- a broken lifecycle seam is closed
            return False

    def restore_banked_stashes(self, session_id: str, composer: Any) -> int:
        """Prepend exact refused stashes once their origin is reconciled."""
        if not self._refusal_callbacks_are_open():
            return 0
        stashes = self._banked_stashes_by_session.pop(session_id, [])
        for stash in reversed(stashes):
            composer.restore_stashed_draft(stash)
        return len(stashes)


__all__ = ["ConsoleRawCliController", "restore_refused_raw_cli_stash"]
