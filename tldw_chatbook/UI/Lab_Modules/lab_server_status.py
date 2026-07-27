"""Read the app's local-server process handles into displayable rows.

Deliberately pure: it takes any object carrying the five process attributes,
so the Lab status chip and inspector are testable against a fake without
spawning subprocesses or mounting widgets.

The handles live on the app rather than on LLMManagementWindow (see
``app.py``'s ``*_server_process`` attributes), and liveness uses the same
``proc and proc.poll() is None`` idiom as the LLM management event handlers.

Ollama is deliberately absent from :data:`LAB_SERVER_SOURCES`. Its sibling
servers (llama.cpp, Llamafile, vLLM, ONNX, MLX-LM) all assign their process
handle from a dedicated worker via ``app_instance.call_from_thread``. Ollama
has no such assignment anywhere in the codebase -- ``app.ollama_server_process``
is only declared and reset to ``None`` -- so a row for it would always read
"stopped" even while a user-started ``ollama serve`` process is alive: a
wrong row, not a missing one. Worse, ``handle_ollama_start_service_button_pressed``
(``Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py``)
cannot currently start that process at all: it calls
``stream_worker_output_to_log(app, "ollama-log-output")`` with one positional
argument short of that coroutine's three, and separately passes ``cmd`` and
that (broken) call's result as positional args to ``App.run_worker`` --
whose second positional parameter is also named ``name``, which the same
call supplies again by keyword, so Python raises
``TypeError: got multiple values for argument 'name'`` before any subprocess
is created. This is pre-existing and unrelated to the Lab frame (verified
identical on ``origin/dev``; the file has no diff on this branch) -- wiring
Ollama up correctly needs that start handler fixed first, which is a
separate, untested legacy code path and out of this fix's scope. Tracked
for a follow-up task rather than guessed at here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: (app attribute, display name), in the order the inspector lists them.
#: Ollama is intentionally not one of these -- see the module docstring.
LAB_SERVER_SOURCES: tuple[tuple[str, str], ...] = (
    ("llamacpp_server_process", "llama.cpp"),
    ("llamafile_server_process", "Llamafile"),
    ("vllm_server_process", "vLLM"),
    ("onnx_server_process", "ONNX"),
    ("mlx_server_process", "MLX-LM"),
)


@dataclass(frozen=True)
class LabServerRow:
    """One local server's display state.

    Attributes:
        name: Human-readable server name.
        running: Whether its process is currently alive.
    """

    name: str
    running: bool


def _is_running(process: Any) -> bool:
    """Report whether a process handle is alive.

    Args:
        process: A ``subprocess.Popen``-like object, or None.

    Returns:
        True only when the handle exists and ``poll()`` returns None. A
        handle whose ``poll()`` raises counts as stopped: a status chip must
        never take down the screen.
    """
    if process is None:
        return False
    try:
        return process.poll() is None
    except Exception:  # noqa: BLE001 -- a status read must not crash the UI
        return False


def read_server_rows(app: Any) -> tuple[LabServerRow, ...]:
    """Read every known local-server handle off the app.

    Args:
        app: The application (or any object carrying the handles). Missing
            attributes read as stopped, since the app may not have set them.

    Returns:
        One row per entry in :data:`LAB_SERVER_SOURCES`, in that order.
    """
    return tuple(
        LabServerRow(name=name, running=_is_running(getattr(app, attribute, None)))
        for attribute, name in LAB_SERVER_SOURCES
    )


def servers_chip_text(rows: Sequence[LabServerRow]) -> str:
    """Summarise running servers for the status row.

    Args:
        rows: Rows from :func:`read_server_rows`.

    Returns:
        ``"Servers: N running"``, or ``"Servers: none running"`` when none are.
    """
    running = sum(1 for row in rows if row.running)
    if running == 0:
        return "Servers: none running"
    return f"Servers: {running} running"


def server_row_id(name: str) -> str:
    """Return the stable widget id for one server's inspector row.

    Shared by ``compose_lab_inspector`` (which mounts the row) and
    ``lab_inspector_rows`` (which refreshes it in place), so the two never
    drift out of sync on id formatting.

    Args:
        name: The server's display name (``LabServerRow.name``).

    Returns:
        ``"lab-inspector-server-<name>"``, with dots turned to hyphens since
        Textual ids may not contain them (e.g. ``"llama.cpp"``).
    """
    return f"lab-inspector-server-{name.replace('.', '-')}"


def server_row_text(row: LabServerRow) -> str:
    """Render one server's inspector row text.

    Args:
        row: The server's current state.

    Returns:
        e.g. ``"● llama.cpp — running"`` or ``"○ llama.cpp — stopped"``.
    """
    marker = "●" if row.running else "○"
    state = "running" if row.running else "stopped"
    return f"{marker} {row.name} — {state}"
