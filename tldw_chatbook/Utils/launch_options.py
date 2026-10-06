"""Launch flags shared by ``tldw-cli`` and ``python -m tldw_chatbook.app``.

TASK-34100.16 ([coverage-10], E6 step 1): a second machine is set up by
carrying a finished config.toml, so one launch must be able to name that file
(``--config PATH``) and skip the splash (``--no-splash``), and ``--help`` must
say where the second-machine route is documented.

``--config`` is the ``TLDW_CONFIG_PATH`` environment variable, set for this
process (and so for every process it starts): one source of truth for the
active profile, with the flag winning over an inherited value. It has to be
adopted before the ADR-126 storage fence and before the TASK-34100.4 startup
unlock, because both read the selected profile. That is why this module may
import only the standard library and ``Backup_Recovery.profile_paths`` -- never
config, Textual, loguru or the application.

Neither flag writes to config, and neither touches ``[first_run]``: setup is
never marked completed by a launch flag.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from collections.abc import MutableMapping, Sequence
from typing import Any

from tldw_chatbook.Backup_Recovery.profile_paths import lexical_path

#: The environment variable ``--config`` sets for this launch.
CONFIG_PATH_ENV_VAR = "TLDW_CONFIG_PATH"

#: Where the second-machine route is documented (the guide section's title is
#: also what the Restore view's settings-file message points at). The wheel
#: ships no Docs/, so --help gives the published copy's URL, anchored at the
#: section.
SECOND_MACHINE_SECTION = "Setting up another machine"
SECOND_MACHINE_GUIDE = "Docs/User_Guide/First_Run_Setup.md"
SECOND_MACHINE_URL = (
    "https://github.com/rmusser01/tldw_chatbook/blob/main/"
    f"{SECOND_MACHINE_GUIDE}#{SECOND_MACHINE_SECTION.lower().replace(' ', '-')}"
)

_EPILOG = f"""\
environment:
  {CONFIG_PATH_ENV_VAR}  use this config.toml instead of
                    ~/.config/tldw_cli/config.toml. --config does the same for
                    one launch and wins when both are set.

{SECOND_MACHINE_SECTION}: copy a finished config.toml and launch with
--config PATH (or set {CONFIG_PATH_ENV_VAR}). The copy carries any API keys
saved in it, in plain text unless password encryption is on, so handle it as
a secret. Keys in environment variables are never written to it. See
"{SECOND_MACHINE_SECTION}" in the User Guide:
{SECOND_MACHINE_URL}"""


def _config_file(value: str) -> str:
    """Validate ``--config`` and return it as a lexical absolute path.

    Args:
        value: The command-line value.

    Returns:
        The absolute path, home-expanded, without resolving links -- the same
        spelling ``TLDW_CONFIG_PATH`` is read with.

    Raises:
        argparse.ArgumentTypeError: For an empty value, a folder, a file in a
            folder that does not exist or cannot be written (nothing could
            create it there), a file that cannot be read, or a path that
            cannot be checked. Every refusal is a usage error (exit 2):
            argparse reports nothing else from a ``type=`` function as one,
            so an OSError would escape as a traceback. An existing file need
            not be writable: the startup fence admits a read-only config.
    """
    if not value.strip():
        raise argparse.ArgumentTypeError("needs the path of a config.toml file")
    path = lexical_path(value)
    try:
        status = os.stat(path)
    except FileNotFoundError:
        # A new file is a first launch -- but only where it can be created.
        if not os.path.isdir(path.parent):
            raise argparse.ArgumentTypeError(
                f"{path.parent} does not exist; create that folder or check the path"
            ) from None
        if not os.access(path.parent, os.W_OK | os.X_OK):
            raise argparse.ArgumentTypeError(
                f"cannot create {path}: {path.parent} is not writable"
            ) from None
        return str(path)
    except OSError as error:
        raise argparse.ArgumentTypeError(
            f"cannot read {path}: {error.strerror or error}"
        ) from None
    if stat.S_ISDIR(status.st_mode):
        raise argparse.ArgumentTypeError(
            f"{path} is a folder; give the config.toml file inside it"
        )
    # os.stat succeeds on a file this user cannot open (mode 000), and the
    # launch then ends on a recovery screen with no explanation.
    if not os.access(path, os.R_OK):
        raise argparse.ArgumentTypeError(f"cannot read {path}: permission denied")
    return str(path)


def build_launch_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    """Build the one ``tldw-cli`` argument parser (both entry points use it).

    Args:
        add_help: False for the early ``--config`` pre-parse, which must not
            answer ``--help`` before the recovery fence has run.

    Returns:
        The parser, ending its help with the second-machine epilog.
    """
    parser = argparse.ArgumentParser(
        description="tldw chatbook - A Textual TUI for chatting with LLMs",
        prog="tldw-cli",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=add_help,
    )
    parser.add_argument(
        "--serve", action="store_true", help="Run the application as a web server"
    )
    parser.add_argument(
        "--host", type=str, help="Host address for web server (default: localhost)"
    )
    parser.add_argument("--port", type=int, help="Port for web server (default: 8000)")
    parser.add_argument("--web-title", type=str, help="Title for the web page")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for web server"
    )
    parser.add_argument(
        "--focus",
        action="store_true",
        help="Start chrome-free in the Console (hides nav bar and workbench header)",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        type=_config_file,
        help=(
            f"Use this config.toml for this launch (same as {CONFIG_PATH_ENV_VAR}; "
            "this flag wins when both are set). A missing file is created with "
            "defaults, as on a first launch; its folder must already exist."
        ),
    )
    parser.add_argument(
        "--no-splash",
        action="store_true",
        help=(
            "Skip the splash screen for this launch only (not a --serve browser "
            "session); [splash_screen] in config is not changed"
        ),
    )
    return parser


def config_flag(argv: Sequence[str] | None = None) -> str | None:
    """Return the ``--config`` path given on the command line, if any.

    ``--help`` is ignored here (the full parse answers it after the recovery
    fence); a malformed known flag exits 2 with usage before anything was
    read or written.

    Args:
        argv: Arguments after the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The lexical absolute path, or None when ``--config`` was not given.
    """
    arguments = sys.argv[1:] if argv is None else list(argv)
    known, _unknown = build_launch_parser(add_help=False).parse_known_args(arguments)
    return known.config


def adopt_config_flag(
    argv: Sequence[str] | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> str | None:
    """Export ``--config PATH`` as ``TLDW_CONFIG_PATH`` before the profile is read.

    Runs before the ADR-126 fence and the startup unlock, so both admit and
    unlock the selected file.

    Args:
        argv: Arguments after the program name; defaults to ``sys.argv[1:]``.
        environ: The environment to update; defaults to ``os.environ``.

    Returns:
        The exported path, or None when ``--config`` was not given (the
        environment is then left exactly as it was).
    """
    chosen = config_flag(argv)
    if chosen is not None:
        target = os.environ if environ is None else environ
        target[CONFIG_PATH_ENV_VAR] = chosen
    return chosen


def apply_launch_options(app: Any, args: argparse.Namespace) -> None:
    """Hand the per-launch flags to the app instance (never to config).

    Args:
        app: The ``TldwCli`` instance, before ``run()``.
        args: The parsed launch arguments.
    """
    app._cli_focus_override = bool(args.focus)
    app._cli_no_splash = bool(args.no_splash)
