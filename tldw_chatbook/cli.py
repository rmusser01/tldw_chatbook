"""Lightweight command-line entry point for tldw_chatbook."""

from typing import Any


def main_cli_runner() -> Any:
    """Load and run the full application only when the CLI is invoked.

    Returns:
        The full application runner's return value.

    Raises:
        SystemExit: With status 1 when a private-path refusal (fail-closed
            by design) escapes the Backup_Recovery preflight, the heavy
            application import, or the runner itself. The plain-language
            diagnostic is emitted to stderr first; no traceback is shown.
    """

    # TASK-21147 (UAT G-7): silence import-time DEBUG/INFO spew BEFORE the
    # heavy import chain that emits it — a cold start's first paint must
    # not be internal debug logs. TLDW_VERBOSE_STARTUP=1 restores it.
    from tldw_chatbook.Utils.startup_logging import quiet_startup_stderr

    quiet_startup_stderr()

    import argparse
    import sys
    from pathlib import Path

    if sys.argv[1:2] == ["recovery"]:
        from tldw_chatbook.Backup_Recovery.launcher import recovery_main

        return recovery_main(sys.argv[2:])

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--recovery-profile")
    parser.add_argument("--recovery-control-root", type=Path)
    parser.add_argument("--recovery-launch-attempt")
    for flag in (
        "--recovery-profile", "--recovery-control-root", "--recovery-launch-attempt"
    ):
        if sum(arg.split("=", 1)[0] == flag for arg in sys.argv[1:]) > 1:
            parser.error("recovery selectors must occur once")
    selected, remaining = parser.parse_known_args()
    if bool(selected.recovery_profile) != bool(selected.recovery_control_root):
        parser.error("both recovery profile and control root are required")
    if selected.recovery_launch_attempt is not None and not selected.recovery_profile:
        parser.error("launch attempt requires recovery profile selectors")
    if selected.recovery_profile:
        from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile

        select_profile(
            selected.recovery_profile, selected.recovery_control_root,
            launch_attempt=selected.recovery_launch_attempt,
        )
    sys.argv[1:] = remaining

    # task-32900: a private-path refusal (fail-closed by design, ADR-029/
    # ADR-127) must not reach a first-time user as a bare traceback.
    # config.py already emitted the plain-language diagnostic before
    # re-raising; this emit covers refusals from later startup stages
    # (preflight, storage admission, the app import itself) and is a no-op
    # when the diagnostic was already printed. SystemExit(1) keeps the
    # packaged command's output to the diagnostic alone.
    from tldw_chatbook.Utils.private_paths import PrivatePathError
    from tldw_chatbook.Utils.startup_errors import (
        emit_private_path_startup_error,
    )

    try:
        from tldw_chatbook.Backup_Recovery.launcher import (
            minimal_recovery,
            startup_preflight,
        )

        reason, password = startup_preflight()
        if reason is not None:
            return minimal_recovery(reason)
        from tldw_chatbook.Backup_Recovery.storage_admission import admit_startup

        admit_startup()
        if password is not None:
            from tldw_chatbook.config import set_encryption_password

            set_encryption_password(password)
            password = None

        from tldw_chatbook.app import main_cli_runner as app_main_cli_runner

        result = app_main_cli_runner()
        from tldw_chatbook.Backup_Recovery.recovery_restart import (
            RecoveryRestart,
            restart,
        )

        if type(result) is RecoveryRestart:
            return restart(result)
        return result
    except PrivatePathError as exc:
        emit_private_path_startup_error(exc)
        raise SystemExit(1) from None
