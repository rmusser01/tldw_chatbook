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
_SUBPROCESS_TIMEOUT_SECONDS = 120
FORBIDDEN_AUDIT_EVENT_NAMES = frozenset(
    {
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.gethostname",
        "socket.getnameinfo",
        "socket.sendmsg",
        "socket.sendto",
        "sqlite3.connect",
    }
)
EXPECTED_PROBE_AUDIT_EVENT_NAMES = frozenset(
    {
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.gethostname",
        "socket.getnameinfo",
        "socket.sendmsg",
        "socket.sendto",
        "sqlite3.connect",
    }
)


def _catalog_names(provider: ToolProvider) -> set[str]:
    return {entry.name for entry in provider.list_catalog()}


def test_legacy_names_are_absent_and_replacements_have_current_owners(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pin legacy-name absence and authoritative replacement catalogs.

    Args:
        monkeypatch: Pytest fixture used to disable optional local tools.
        tmp_path: Isolated workspace root for the local provider.
    """
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
    """Resolve compatibility imports without external I/O in a fresh process.

    Args:
        tmp_path: Isolated home and configuration root for the child process.
    """
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
    env.pop("TLDW_TEST_FORCE_FORBIDDEN_AUDIT_EVENT", None)
    code = f"""
import json
import os
import socket
import sys

forbidden_event_names = set({sorted(FORBIDDEN_AUDIT_EVENT_NAMES)!r})
expected_probe_event_names = set({sorted(EXPECTED_PROBE_AUDIT_EVENT_NAMES)!r})
forbidden_events = []

# Prevent urllib3's import-time IPv6 capability probe from opening a socket.
socket.has_ipv6 = False

def block_external_io(event, _args):
    if event in forbidden_event_names:
        forbidden_events.append(event)
        raise RuntimeError("external I/O blocked during compatibility import")

sys.addaudithook(block_external_io)

import tldw_chatbook.Tools as Tools

classes = (Tools.WebSearchTool, Tools.RAGSearchTool, Tools.SearchNotesTool)
if os.environ.get("TLDW_TEST_FORCE_FORBIDDEN_AUDIT_EVENT") == "1":
    for event in sorted(expected_probe_event_names):
        try:
            sys.audit(event, None)
        except RuntimeError:
            pass
if forbidden_events:
    raise RuntimeError("forbidden I/O attempted: " + ", ".join(forbidden_events))
print("compatibility-import-preamble")
print(json.dumps(
    {{cls.__name__: [cls.__module__, cls.__name__] for cls in classes}},
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
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert result.returncode == 0, result.stderr
    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert stdout_lines
    assert json.loads(stdout_lines[-1]) == {
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

    probe_env = {**env, "TLDW_TEST_FORCE_FORBIDDEN_AUDIT_EVENT": "1"}
    probe = subprocess.run(  # nosec B603 # fixed executable and arguments
        [sys.executable, "-B", "-c", code],
        cwd=REPO_ROOT,
        env=probe_env,
        text=True,
        capture_output=True,
        check=False,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
    )
    assert probe.returncode != 0
    assert not probe.stdout
    for event in EXPECTED_PROBE_AUDIT_EVENT_NAMES:
        assert event in probe.stderr
    assert FORBIDDEN_AUDIT_EVENT_NAMES == EXPECTED_PROBE_AUDIT_EVENT_NAMES
