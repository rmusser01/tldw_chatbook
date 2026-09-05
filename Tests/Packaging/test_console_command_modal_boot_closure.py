"""Command-only modal implementations stay outside controller construction."""

from pathlib import Path
import subprocess
import sys


def test_command_controllers_defer_style_rewind_and_video_capacity_modals():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys
from tldw_chatbook.UI.Console_Modules import commands, video

for name, class_name in (
    ("console_style_picker_modal", "ConsoleStylePickerModal"),
    ("console_rewind_modal", "ConsoleRewindModal"),
    ("console_video_capacity_modal", "ConsoleVideoCapacityModal"),
):
    module_name = "tldw_chatbook.Widgets.Console." + name
    assert module_name not in sys.modules, module_name
    assert getattr(importlib.import_module(module_name), class_name) is not None
""",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
