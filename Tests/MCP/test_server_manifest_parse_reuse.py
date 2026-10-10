"""The built-in MCP manifest reuses its parse of ``server.py`` (TASK-33620.15.5).

Measured live on origin/dev: a Console's first send composes the MCP catalog on
the UI loop, and its built-in inventory read ran
``describe_local_mcp_capabilities()``, which parsed this 1,100-line module
three times (tools, resources, prompts), several milliseconds each, inside the
first send's longest loop callback. The parsed tree is now kept while the
file's identity (path, device, inode, size, change times) is unchanged; the
manifest is still built fresh from it on every call.
"""

from __future__ import annotations

import ast
import os
import shutil

import pytest

from tldw_chatbook.MCP import server


@pytest.fixture
def parses(monkeypatch):
    """Count ``ast.parse`` calls from a cold tree."""
    if hasattr(server, "_SERVER_MODULE_AST"):
        monkeypatch.setattr(server, "_SERVER_MODULE_AST", None)
    calls = []
    real = ast.parse

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(server.ast, "parse", counted)
    return calls


def test_an_unchanged_server_module_is_parsed_once(parses):
    """Red on origin/dev: two manifests parsed the file six times."""
    first = server.describe_local_mcp_capabilities()
    second = server.describe_local_mcp_capabilities()

    assert len(parses) == 1, f"parsed server.py {len(parses)} times for two manifests"
    assert first == second


def test_the_kept_tree_builds_the_same_manifest_as_a_fresh_parse(parses, monkeypatch):
    kept = server.describe_local_mcp_capabilities()
    kept["tools"][0]["description"] = "changed by a caller"
    if hasattr(server, "_SERVER_MODULE_AST"):
        monkeypatch.setattr(server, "_SERVER_MODULE_AST", None)

    fresh = server.describe_local_mcp_capabilities()

    assert server.describe_local_mcp_capabilities() == fresh
    assert fresh["tools"][0]["description"] != "changed by a caller"
    assert len(fresh["tools"]) == len(kept["tools"])


def test_a_changed_server_module_is_parsed_again(parses, monkeypatch, tmp_path):
    copy = tmp_path / "server.py"
    shutil.copyfile(server.__file__, copy)
    monkeypatch.setattr(server, "__file__", str(copy))
    before = server.describe_local_mcp_capabilities()
    name = before["tools"][0]["name"]
    source = copy.read_text(encoding="utf-8")
    renamed = source.replace(f"async def {name}(", f"async def {name}_renamed(", 1)
    assert renamed != source
    copy.write_text(renamed, encoding="utf-8")
    stamp = os.stat(copy)
    os.utime(copy, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))

    after = server.describe_local_mcp_capabilities()

    assert after["tools"][0]["name"] == f"{name}_renamed"
    assert len(parses) >= 2
