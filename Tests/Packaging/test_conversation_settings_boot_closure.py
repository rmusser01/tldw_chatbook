"""Import-closure guard for Conversation Settings-only helpers.

The Console and canonical Settings screens are resident at ``_ui_ready``, but
the Conversation Settings modal, its searchable provider picker, and endpoint
test helper are only needed after the user opens the workflow. Keep those
modules off the first-paint path so the workflow does not consume ADR-097's
zero-headroom UI-ready module ratchet.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_chat_screen_defers_conversation_settings_only_helpers(
    tmp_path: Path,
) -> None:
    """Importing the Console screen leaves settings-only helpers unloaded."""
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)

    code = """
import sys

import tldw_chatbook.UI.Screens.chat_screen  # noqa: F401

deferred = (
    "tldw_chatbook.UI.Screens.settings_endpoint_probe",
    "tldw_chatbook.Widgets.Console.console_provider_picker",
    "tldw_chatbook.Widgets.Console.console_settings_modal",
)
resident = [name for name in deferred if sys.modules.get(name) is not None]
assert not resident, f"Conversation Settings helpers resident at boot: {resident}"

# Anti-vacuity: the deferred contracts remain importable on first use.
from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
    SettingsEndpointProbePurpose,
)
from tldw_chatbook.Widgets.Console import ConsoleSettingsModal
from tldw_chatbook.Widgets.Console.console_provider_picker import (
    ConsoleProviderPicker,
)

assert SettingsEndpointProbePurpose.CHAT_CATALOG.value == "chat_catalog"
assert ConsoleSettingsModal is not None
assert ConsoleProviderPicker is not None
print("CONVERSATION_SETTINGS_BOOT_CLOSURE_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, (
        "Conversation Settings-only helpers must stay off the Console boot path:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "CONVERSATION_SETTINGS_BOOT_CLOSURE_OK" in result.stdout
