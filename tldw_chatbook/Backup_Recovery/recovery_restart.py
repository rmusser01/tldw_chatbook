"""A fixed fresh-process handoff to the existing recovery-only interface."""

from __future__ import annotations

import subprocess  # nosec B404 - fixed interpreter and separate argument values.
import sys
from dataclasses import dataclass
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os


@dataclass(frozen=True)
class RecoveryRestart:
    """Nonsecret convenience hints; the child must inspect and review again."""

    archive: Path | None
    target_config: Path

    def __post_init__(self) -> None:
        paths = (self.target_config,) + ((self.archive,) if self.archive else ())
        if any(
            not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts
            for path in paths
        ):
            raise ValueError("invalid_recovery_restart")


_ENTRY = (
    "import sys; from pathlib import Path; "
    "from tldw_chatbook.Backup_Recovery.recovery_restart import RecoveryRestart; "
    "from tldw_chatbook.Backup_Recovery.launcher import minimal_recovery; "
    "raise SystemExit(minimal_recovery('replacement_requested', "
    "restart_request=RecoveryRestart(Path(sys.argv[1]) if sys.argv[1] else None, Path(sys.argv[2]))))"
)


def restart(request: RecoveryRestart) -> None:
    """Replace the successfully stopped normal app with a clean interpreter."""
    from .isolated_restore import _launch_environment

    if type(request) is not RecoveryRestart:
        raise ValueError("invalid_recovery_restart")
    # Fixed interpreter and code; reviewed path hints are separate argv values.
    argv = [
        sys.executable,
        "-c",
        _ENTRY,
        str(request.archive) if request.archive else "",
        str(request.target_config),
    ]
    environment = _launch_environment()
    if os.name == "nt":
        # CreateProcess supplies argv quoting without UCRT execve's environment
        # construction. Forward only standard handles, then release this
        # process's native leases by exiting rather than waiting on the child.
        _child = subprocess.Popen(  # nosec B603 - fixed interpreter, no shell.
            argv, env=environment, stdin=0, stdout=1, stderr=2, close_fds=True
        )
        os._exit(0)
    os.execve(sys.executable, argv, environment)  # nosec B606
