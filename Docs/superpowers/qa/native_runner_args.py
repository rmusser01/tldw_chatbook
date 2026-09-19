"""CLI boundary for disposable MCP native QA on macOS/Linux."""

import argparse
import os
import runpy
import shutil
from pathlib import Path


def parse_native_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Validate a prepared, unused temporary profile without modifying it.

    Args:
        argv: ROOT TMUX_SOCKET SESSION arguments, or None for the process CLI.
            ROOT must be an existing canonical child of /tmp with a private
            config and existing home/config/data directories. Socket and session
            names use 1–64 ASCII letters, digits, underscores or hyphens and
            start with a letter or digit.

    Returns:
        Arguments with root converted to Path and tmux_path resolved from PATH.

    Raises:
        SystemExit: Exit 2 with usage for invalid input, missing tmux, unsafe
            profile paths or previous evidence. Help exits 0. No app is started
            and no profile or output files are written here.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", metavar="ROOT")
    parser.add_argument("tmux_socket", metavar="TMUX_SOCKET")
    parser.add_argument("session", metavar="SESSION")
    args = parser.parse_args(argv)

    from tldw_chatbook.Utils.input_validation import validate_username
    from tldw_chatbook.Utils.path_validation import (
        validate_canonical_directory,
        validate_path,
    )

    for name in (args.tmux_socket, args.session):
        if (
            name != name.strip()
            or not validate_username(name, min_length=1, max_length=64)
            or not (name[0].isascii() and name[0].isalnum())
        ):
            parser.error("tmux names require 1–64 ASCII identifier characters")
    args.tmux_path = shutil.which("tmux")
    if args.tmux_path is None:
        parser.error("tmux must be installed and available on PATH")
    try:
        root = validate_canonical_directory(args.root)
        base = Path("/tmp").resolve(strict=True)
        validate_path(root, base, redact_paths=True, allow_hidden=True)
        if root == base:
            raise ValueError("Use a dedicated profile below /tmp")
        for child in ("home", "config", "data"):
            directory = validate_canonical_directory(root / child)
            validate_path(directory, root, redact_paths=True, allow_hidden=True)
        if any(
            os.path.lexists(root / name)
            for name in ("native.log", "launch.json", "evidence")
        ):
            raise ValueError("Profile already contains native run output")
        runpy.run_path(
            str(Path(__file__).parent / "2026-09-16-ingest-lifecycle/native_check.py")
        )["validate_profile"](root)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        parser.error(
            "Use an unused canonical profile below /tmp with contained data paths"
        )
    args.root = root
    return args
