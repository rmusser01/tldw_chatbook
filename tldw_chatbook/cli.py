"""Lightweight command-line entry point for tldw_chatbook."""

from typing import Any


def main_cli_runner() -> Any:
    """Load and run the full application only when the CLI is invoked.

    Returns:
        The full application runner's return value.
    """

    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--recovery-profile")
    parser.add_argument("--recovery-control-root", type=Path)
    for flag in ("--recovery-profile", "--recovery-control-root"):
        if sum(arg.split("=", 1)[0] == flag for arg in sys.argv[1:]) > 1:
            parser.error("recovery selectors must occur once")
    selected, remaining = parser.parse_known_args()
    if bool(selected.recovery_profile) != bool(selected.recovery_control_root):
        parser.error("both recovery profile and control root are required")
    if selected.recovery_profile:
        from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile

        select_profile(selected.recovery_profile, selected.recovery_control_root)
    sys.argv[1:] = remaining
    from tldw_chatbook.Backup_Recovery.storage_admission import admit_startup

    admit_startup()

    from tldw_chatbook.app import main_cli_runner as app_main_cli_runner

    return app_main_cli_runner()
