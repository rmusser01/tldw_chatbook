"""Four exposure paths from the core review must stay closed.

TASK-32806.7. Each is checked at the level it can be, without the
storage-admission gate this worktree cannot satisfy: the redactor and the
prompt-export temp helper run directly; the connection-test egress gate and
the worktree root are pinned structurally over source.
"""

from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


# --- Path 1: advanced-runner result redaction (runs directly) --------------

def test_a_list_shaped_action_result_is_redacted():
    from tldw_chatbook.UI.MCP_Modules.mcp_inspector import _redact_action_result

    result = [
        {"api_key": "sk-secret-1234567890"},
        {"nested": {"authorization": "Bearer sk-secret-abcdef"}},
        "a plain string",
    ]
    redacted = _redact_action_result(result)
    flat = repr(redacted)
    assert "sk-secret-1234567890" not in flat
    assert "sk-secret-abcdef" not in flat
    # A plain mapping still redacts, and a scalar passes through.
    assert _redact_action_result({"api_key": "sk-xyz"}) != {"api_key": "sk-xyz"}
    assert _redact_action_result("plain") == "plain"


# --- Path 2: prompt exports use mkstemp, not a predictable temp name -------

def test_prompt_export_uses_mkstemp_not_gettempdir():
    source = (REPO / "tldw_chatbook/DB/Prompts_DB.py").read_text()
    # No export builds a name in the shared temp dir any more.
    assert "tempfile.gettempdir()" not in source
    # And three exports reach mkstemp.
    assert source.count("tempfile.mkstemp(") >= 3


# --- Path 3: connection test clears the egress gate, redirects off ---------

def _function_source(module: str, name: str) -> str:
    src = (REPO / module).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"{name} not found in {module}")


def test_connection_test_preflights_egress_and_disables_redirects():
    body = _function_source(
        "tldw_chatbook/Widgets/Settings_Widgets/server_switch_modal.py",
        "_run_connection_test",
    )
    assert "check_url_or_raise_async" in body, "no egress pre-check before the token is sent"
    assert "follow_redirects=False" in body, "redirects not disabled on the token request"
    # The pre-check must precede the client that carries the token.
    assert body.index("check_url_or_raise_async") < body.index("AsyncClient")


# --- Path 4: agent worktrees rooted under the profile data dir -------------

def test_agent_worktrees_are_not_under_shared_temp():
    body = _function_source("tldw_chatbook/Agents/agent_worktree.py", "_worktrees_base")
    assert "tempfile.gettempdir()" not in body, "worktrees still under shared temp"
    assert "get_user_data_dir()" in body, "worktrees not rooted under the profile data dir"
    assert "secure_private_directory" in body, "worktree root not created privately"
