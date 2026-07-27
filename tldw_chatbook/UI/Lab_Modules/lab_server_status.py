"""Read the app's local-server process handles into displayable rows.

Deliberately pure: it takes any object carrying the six process attributes,
so the Lab status chip and inspector are testable against a fake without
spawning subprocesses or mounting widgets.

The handles live on the app rather than on LLMManagementWindow (see
``app.py``'s ``*_server_process`` attributes), and liveness uses the same
``proc and proc.poll() is None`` idiom as the LLM management event handlers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: (app attribute, display name), in the order the inspector lists them.
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
