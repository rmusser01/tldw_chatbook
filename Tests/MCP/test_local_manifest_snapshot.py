"""One capability request observes one source and returns independent output."""

import copy

import pytest

from tldw_chatbook.MCP import server


def test_manifest_reads_source_once_per_request_and_returns_independent_data(
    monkeypatch,
):
    actual_load = server._load_server_module_ast
    reads = []

    def observe_load():
        tree = actual_load()
        reads.append(tree)
        return tree

    monkeypatch.setattr(server, "_load_server_module_ast", observe_load)
    first = server.describe_local_mcp_capabilities()
    expected = copy.deepcopy(first)
    assert first["tools"] and first["resources"] and first["prompts"]
    first["tools"][0]["inputSchema"]["properties"].clear()
    first["resources"].clear()
    first["prompts"][0]["arguments"].clear()
    second = server.describe_local_mcp_capabilities()
    assert second == expected
    assert len(reads) == 2, "each manifest must read and parse source exactly once"
    assert reads[0] is not reads[1], "the next request must observe source afresh"


def test_next_manifest_observes_changed_source_and_propagates_parse_failure(
    tmp_path, monkeypatch
):
    from pathlib import Path

    source = Path(server.__file__).read_text(encoding="utf-8")
    original = "Generate a prompt to summarize a conversation."
    replacement = "Generate a prompt from the changed server source."
    assert source.count(original) == 1
    selected = tmp_path / "server.py"
    selected.write_text(source, encoding="utf-8")
    monkeypatch.setattr(server, "__file__", str(selected))
    first = server.describe_local_mcp_capabilities()
    assert any(item["description"] == original for item in first["prompts"])
    selected.write_text(source.replace(original, replacement), encoding="utf-8")
    second = server.describe_local_mcp_capabilities()
    assert any(item["description"] == replacement for item in second["prompts"])
    assert any(item["description"] == original for item in first["prompts"])
    selected.write_text("def invalid syntax", encoding="utf-8")
    with pytest.raises(SyntaxError):
        server.describe_local_mcp_capabilities()


@pytest.mark.parametrize("section", ["tools", "resources", "prompts"])
def test_standalone_manifest_helpers_keep_their_fresh_no_argument_contract(
    monkeypatch, section
):
    actual_load = server._load_server_module_ast
    reads = []

    def observe_load():
        tree = actual_load()
        reads.append(tree)
        return tree

    helper = getattr(server, "_describe_local_" + section)
    expected = helper()
    monkeypatch.setattr(server, "_load_server_module_ast", observe_load)
    assert helper() == expected
    assert len(reads) == 1
