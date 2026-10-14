"""Received Console projections preserve the existing lazy boot boundary."""

import json

from Tests.Performance.test_app_import_weight import _run_isolated_python


def test_received_ui_callbacks_do_not_import_turn_authority(tmp_path, record_property):
    result = _run_isolated_python(tmp_path, '''
import json
import sys
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Console_Modules import composer_run_controls
names = (
    "_publish_console_authored_draft",
    "_project_console_received_preparing",
    "_project_console_received_finished",
    "_project_console_received_input",
    "_dispatch_console_draft_send",
)
assert "tldw_chatbook.UI.Console_Modules.turn_admission" not in sys.modules
assert all(getattr(ChatScreen, name) is getattr(composer_run_controls, name) for name in names)
assert all(getattr(ChatScreen, name).__module__ == composer_run_controls.__name__ for name in names)
print(json.dumps({"resident_callback_count": len(names), "turn_authority_imported": False}))
''')
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {"resident_callback_count": 5, "turn_authority_imported": False}
    for name, value in payload.items():
        record_property(name, value)
