"""Installed SQLite validators can load pure models inside a recovery fence."""

import os
import subprocess
import sys


def test_goal_models_do_not_initialize_normal_config(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    code = """
import sys
from tldw_chatbook.Agents.goal_models import GoalRequest
assert 'tldw_chatbook.config' not in sys.modules
from tldw_chatbook.Chat import ChatConversationService
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService as Direct
assert ChatConversationService is Direct
print('pure-goal-import-ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "HOME": str(home),
            "USERPROFILE": str(home),
            "TLDW_CONFIG_PATH": str(home / ".config/tldw_cli/config.toml"),
            "TEMP": str(tmp_path),
            "TMP": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "pure-goal-import-ok" in result.stdout
