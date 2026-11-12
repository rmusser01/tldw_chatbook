"""Recovery contract imports cannot create a runtime/source startup holder."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.bootstrap_profile


def test_workflow_and_goal_values_import_without_config_or_chat_runtime(tmp_path):
    home = tmp_path / "private-home"
    home.mkdir()
    env = os.environ.copy()
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_CONFIG_HOME=str(home / "config"),
        XDG_DATA_HOME=str(home / "data"),
        TLDW_CONFIG_PATH=str(home / "config.toml"),
    )
    script = """
import sys
from tldw_chatbook.Agents.workflow_models import WorkflowRequest
from tldw_chatbook.DB.recovery_goal_schema import validate_goal_rows
from tldw_chatbook.DB.recovery_workflow_schema import validate_workflow_rows
import sqlite3
connection = sqlite3.connect(':memory:')
connection.execute('CREATE TABLE schema_version(version INTEGER)')
connection.execute('INSERT INTO schema_version VALUES (23)')
assert validate_goal_rows(connection) == ()
assert validate_workflow_rows(connection) == ()
connection.close()
assert 'tldw_chatbook.config' not in sys.modules
assert 'tldw_chatbook.Chat' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert list(home.rglob("*")) == []
