"""Pin current tool ownership while preserving legacy compatibility imports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess  # nosec B404 # fixed child interpreter for import isolation
import sys

import pytest

import tldw_chatbook.Agents.local_tool_provider as local_tool_provider
import tldw_chatbook.config as config
from tldw_chatbook.Agents.library_rag_tool_provider import LibraryRagToolProvider
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider,
    ToolProvider,
    gateable_builtin_tools,
)

LEGACY_NAMES = frozenset({"rag_search", "web_search", "search_notes", "code_audit"})
REPO_ROOT = Path(__file__).resolve().parents[2]


def _catalog_names(provider: ToolProvider) -> set[str]:
    return {entry.name for entry in provider.list_catalog()}


def test_legacy_names_are_absent_and_replacements_have_current_owners(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "get_cli_setting", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        local_tool_provider, "get_cli_setting", lambda *_args, **_kwargs: False
    )

    gateable_names = {entry.tool_name for entry in gateable_builtin_tools()}
    assert LEGACY_NAMES.isdisjoint(gateable_names)
    assert _catalog_names(BuiltinToolProvider()) == {
        "calculator",
        "get_current_datetime",
    }
    assert "web_search" in _catalog_names(LocalToolProvider(workspace_root=tmp_path))
    assert "library_search_notes" in _catalog_names(LibraryToolProvider(object()))
    assert _catalog_names(LibraryRagToolProvider(object())) == {"search_library_rag"}


def test_legacy_compatibility_classes_resolve_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    env = {
        **os.environ,
        "HOME": str(home),
        "TLDW_CONFIG_PATH": str(tmp_path / "config.toml"),
        "TLDW_TEST_MODE": "1",
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    code = """
import json
import sys

blocked_events = {"socket.connect", "socket.getaddrinfo", "sqlite3.connect"}

def block_external_io(event, _args):
    if event in blocked_events:
        raise RuntimeError("external I/O blocked during compatibility import")

sys.addaudithook(block_external_io)

import tldw_chatbook.Tools as Tools

classes = (Tools.WebSearchTool, Tools.RAGSearchTool, Tools.SearchNotesTool)
print(json.dumps(
    {cls.__name__: [cls.__module__, cls.__name__] for cls in classes},
    separators=(",", ":"),
    sort_keys=True,
))
"""

    result = subprocess.run(  # nosec B603 # fixed executable and arguments
        [sys.executable, "-B", "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "RAGSearchTool": [
            "tldw_chatbook.Tools.rag_search_tool",
            "RAGSearchTool",
        ],
        "SearchNotesTool": [
            "tldw_chatbook.Tools.note_management_tools",
            "SearchNotesTool",
        ],
        "WebSearchTool": [
            "tldw_chatbook.Tools.web_search_tool",
            "WebSearchTool",
        ],
    }
