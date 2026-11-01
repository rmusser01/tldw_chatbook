"""Common virtual listing flags retain confinement across closed workers."""

from __future__ import annotations

import json
import stat

import pytest

from tldw_chatbook.Agents.virtual_cli_provider import VirtualCliProvider
from tldw_chatbook.Tools.virtual_cli_impls import (
    VirtualCliArgumentError,
    VirtualCliRegistry,
    parse_request,
)
from tldw_chatbook.Tools.workspace_tool_executor import (
    WorkspaceToolExecutionError,
    WorkspaceToolExecutor,
)
from tldw_chatbook.Tools.workspace_tool_protocol import (
    WorkspaceProtocolError,
    WorkspaceToolRequest,
)
from tldw_chatbook.Tools.workspace_wire_decode import WireDecodeError, decode_request


@pytest.mark.parametrize(
    "flag", ["-a", "-A", "--all", "--almost-all", "-l", "-la", "-al", "-Al", "-lA"]
)
def test_listing_flags_reach_the_pinned_executor(tmp_path, monkeypatch, flag):
    calls = []

    class Executor:
        def execute(self, operation, arguments, *, intent):
            calls.append((operation, arguments, intent))
            return "worker listing"

    def direct_core(*args, **kwargs):
        pytest.fail("leased listing bypassed pinned worker")

    monkeypatch.setattr(
        "tldw_chatbook.Tools.virtual_cli_impls.list_directory", direct_core
    )
    registry = VirtualCliRegistry(tmp_path, workspace_executor=Executor())
    assert registry.execute("ls", [flag, "docs"]) == "worker listing"
    expected = {"path": "docs"}
    if "l" in flag.lstrip("-") and flag not in {"--all", "--almost-all"}:
        expected["long_format"] = True
    assert calls == [("fs_list", expected, "read")]


@pytest.mark.parametrize(
    "command, argv",
    [
        ("ls", ["--recursive"]),
        ("cat", []),
        ("grep", ["x", "--mode", "bogus"]),
        ("find", []),
        ("stat", []),
        ("git_status", ["a", "b"]),
        ("git_diff", ["--bogus"]),
        ("git_log", ["--count", "zero"]),
        ("git_blame", ["a", "--start", "0"]),
        ("git_branches", ["extra"]),
        ("ls", ["\x00"]),
    ],
)
def test_invalid_arguments_include_the_selected_command_usage(command, argv):
    with pytest.raises(VirtualCliArgumentError) as caught:
        parse_request(command, argv)
    assert f"Usage: {command}" in str(caught.value)


def test_model_schema_discloses_every_supported_usage(tmp_path):
    provider = VirtualCliProvider(workspace_root=tmp_path)
    schema = provider.load_schema("virtual_cli:virtual_cli")
    disclosed = schema.description + json.dumps(schema.parameters)
    for hub in provider.hub_tools():
        assert hub.input_schema["properties"]["argv"]["description"] in disclosed
    for flag in ("-a", "-A", "--all", "--almost-all", "-l", "-la", "-al"):
        assert flag in disclosed
    assert "hidden" in disclosed.lower()
    assert "size" in disclosed.lower()


def test_real_worker_long_listing_includes_metadata_and_excludes_protected_entries(
    tmp_path,
):
    tmp_path = tmp_path / "workspace"
    tmp_path.mkdir()
    (tmp_path / "docs").mkdir()
    visible = tmp_path / "visible.txt"
    visible.write_bytes(b"twelve bytes")
    hidden = tmp_path / ".hidden"
    hidden.write_bytes(b"hidden")
    (tmp_path / "private.txt").write_bytes(b"private")
    executor = WorkspaceToolExecutor(
        tmp_path, user_exclusion_paths=lambda: (tmp_path / "private.txt",)
    )
    registry = VirtualCliRegistry(tmp_path, workspace_executor=executor)
    assert registry.execute("ls", []) == "docs/\n.hidden\nvisible.txt"
    assert registry.execute("ls", ["-a"]) == registry.execute("ls", [])
    expected = {
        f"{stat.filemode(path.lstat().st_mode)} {path.lstat().st_size} {path.name}{'/' if path.is_dir() else ''}"
        for path in (tmp_path / "docs", hidden, visible)
    }
    assert set(registry.execute("ls", ["-la"]).splitlines()) == expected
    assert registry.execute("ls", ["-al"]) == registry.execute("ls", ["-l"])
    assert executor.execute(
        "fs_list", {"path": ".", "long_format": False}, intent="read"
    ) == registry.execute("ls", [])
    with pytest.raises(WorkspaceToolExecutionError):
        registry.execute("ls", ["-l", "../"])


@pytest.mark.parametrize("value", [True, False])
def test_listing_wire_accepts_optional_boolean_in_parent_and_stdlib_decoder(
    tmp_path, value
):
    request = WorkspaceToolExecutor(tmp_path)._build_request(
        "fs_list", {"path": "."}, intent="read"
    )
    payload = json.loads(request.to_bytes())
    payload["arguments"]["long_format"] = value
    raw = json.dumps(payload).encode()
    assert decode_request(raw)["arguments"]["long_format"] is value
    assert WorkspaceToolRequest.from_bytes(raw).arguments["long_format"] is value


@pytest.mark.parametrize("value", [0, 1, "true", None, [], {}])
def test_listing_wire_rejects_non_boolean_long_format(tmp_path, value):
    request = WorkspaceToolExecutor(tmp_path)._build_request(
        "fs_list", {"path": "."}, intent="read"
    )
    payload = json.loads(request.to_bytes())
    payload["arguments"]["long_format"] = value
    raw = json.dumps(payload).encode()
    with pytest.raises(WireDecodeError):
        decode_request(raw)
    with pytest.raises(WorkspaceProtocolError):
        WorkspaceToolRequest.from_bytes(raw)


@pytest.mark.parametrize("argv", [["--unsupported"], ["x"] * 65, [1]])
def test_provider_rejections_include_actionable_ls_usage(tmp_path, argv):
    result = VirtualCliProvider(workspace_root=tmp_path).invoke(
        "virtual_cli", {"command": "ls", "argv": argv}
    )
    assert not result.ok
    assert "Usage: ls" in result.error
    assert "-l" in result.error
