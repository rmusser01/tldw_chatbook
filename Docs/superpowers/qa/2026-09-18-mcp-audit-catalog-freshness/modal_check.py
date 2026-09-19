"""Native modal selector equivalence check in an isolated profile."""

import os
import runpy
import socket
import sys
from pathlib import Path


def main() -> None:
    """Run ROOT TMUX_SOCKET SESSION in an existing native tmux session.

    ROOT is an unused, prepared private profile under /tmp. Arguments, profile
    paths and tmux availability are checked before app startup or output writes.
    The run writes native.log, launch.json and evidence/ under ROOT. It exits 0
    after a successful journey, 1 for a journey failure and 2 for invalid input.

    Raises:
        SystemExit: Status 0 after success, 1 after a journey or application
            failure, or 2 for invalid command-line input or profile paths.
    """
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    sys.path.insert(0, str(repo))
    args = runpy.run_path(str(here.parent / "native_runner_args.py"))[
        "parse_native_args"
    ]()
    root = args.root
    os.environ.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        TLDW_CONFIG_PATH=str(root / "config.toml"),
        XDG_DATA_HOME=str(root / "data"),
        XDG_CONFIG_HOME=str(root / "config"),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    os.environ.pop("NO_COLOR", None)
    attempts = []
    original_connect = socket.socket.connect

    def guard_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            attempts.append("network connect")
            raise RuntimeError("Network is disabled in this disposable UI journey")
        return original_connect(sock, address)

    socket.socket.connect = guard_connect
    try:
        # Defer application imports until validation, private paths and the
        # outbound-network guard are in place. The journey owns grouped imports.
        run = runpy.run_path(str(here / "modal_journey.py"))["run"]
        raise SystemExit(run(args, attempts))
    finally:
        socket.socket.connect = original_connect


if __name__ == "__main__":
    main()
