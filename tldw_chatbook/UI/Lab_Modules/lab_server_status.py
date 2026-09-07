"""Read the app's local-server process handles into displayable rows.

Deliberately pure: it takes any object carrying the five process attributes,
so the Lab status chip and inspector are testable against a fake without
spawning subprocesses or mounting widgets.

The handles live on the app rather than on LLMManagementWindow (see
``app.py``'s ``*_server_process`` attributes), and liveness uses the same
``proc and proc.poll() is None`` idiom as the LLM management event handlers.

Every provider here is tracked the same way: its process handle is published
by ``server_lifecycle.run_server_subprocess`` through
``app.call_from_thread(publish_server_process, ...)``, which sets the
matching ``*_server_process`` attribute on the app.

Ollama was excluded when this module was written, because at the time its
start handler could not spawn a process at all and nothing ever assigned
``app.ollama_server_process`` -- a row for it would have read "stopped"
forever even while a user-started ``ollama serve`` was alive. Dev has since
routed Ollama through the shared lifecycle owner
(``server_lifecycle.SERVER_PROCESS_ATTRS`` includes it, and
``handle_ollama_start_service_button_pressed`` reserves and publishes a
claim like every sibling), so the exclusion outlived its reason and the
Models status list was under-reporting: a user with Ollama running saw a
count that ignored it. See task-886.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: (app attribute, display name), in the order the inspector lists them.
#: Kept in step with ``server_lifecycle.SERVER_PROCESS_ATTRS``; the test
#: suite asserts the two agree so a provider added there cannot go
#: unreported here.
LAB_SERVER_SOURCES: tuple[tuple[str, str], ...] = (
    ("llamacpp_server_process", "llama.cpp"),
    ("llamafile_server_process", "Llamafile"),
    ("ollama_server_process", "Ollama"),
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
