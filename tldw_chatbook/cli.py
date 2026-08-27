"""Lightweight command-line entry point for tldw_chatbook."""

from typing import Any


def main_cli_runner() -> Any:
    """Load and run the full application only when the CLI is invoked.

    Returns:
        The full application runner's return value.
    """

    # TASK-21147 (UAT G-7): silence import-time DEBUG/INFO spew BEFORE the
    # heavy import chain that emits it — a cold start's first paint must
    # not be internal debug logs. TLDW_VERBOSE_STARTUP=1 restores it.
    from tldw_chatbook.Utils.startup_logging import quiet_startup_stderr

    quiet_startup_stderr()

    from tldw_chatbook.app import main_cli_runner as app_main_cli_runner

    return app_main_cli_runner()
