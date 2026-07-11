"""Lightweight command-line entry point for tldw_chatbook."""

from typing import Any


def main_cli_runner() -> Any:
    """Load and run the full application only when the CLI is invoked."""

    from tldw_chatbook.app import main_cli_runner as app_main_cli_runner

    return app_main_cli_runner()
