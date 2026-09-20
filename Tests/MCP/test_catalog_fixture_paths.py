"""The executable discovery fixture must stay inside its caller-owned root."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures/stdio_catalog_server.py"
REQUEST = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"


@pytest.mark.parametrize("target", ["state", "trace"])
@pytest.mark.parametrize("escape", ["parent", "absolute", "symlink"])
def test_fixture_rejects_paths_outside_owned_root(
    tmp_path: Path, target: str, escape: str
) -> None:
    """Reject both read and write escapes before processing any wire request."""
    root = tmp_path / "owned"
    root.mkdir()
    state = root / "state.json"
    trace = root / "trace.jsonl"
    state.write_text('{"version": "inside"}')
    outside = tmp_path / "outside.json"
    sentinel = '{"version": "outside"}\n'
    outside.write_text(sentinel)
    if escape == "parent":
        bad = "../outside.json"
    elif escape == "absolute":
        bad = str(outside)
    else:
        alias = root / "alias.json"
        alias.symlink_to(outside)
        bad = str(alias)
    result = subprocess.run(
        [
            sys.executable,
            str(FIXTURE.resolve()),
            bad if target == "state" else str(state),
            bad if target == "trace" else str(trace),
            str(root.resolve()),
        ],
        cwd=root,
        input=REQUEST,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0, result.stdout
    assert result.stdout == ""
    assert outside.read_text() == sentinel
    assert not trace.exists()


@pytest.mark.parametrize("absolute", [True, False])
def test_fixture_serves_valid_paths_beneath_owned_root(
    tmp_path: Path, absolute: bool
) -> None:
    """Validated relative and absolute paths still provide real discovery."""
    state = tmp_path / "state.json"
    trace = tmp_path / "trace.jsonl"
    state.write_text('{"version": "inside"}')
    result = subprocess.run(
        [
            sys.executable,
            str(FIXTURE.resolve()),
            str(state) if absolute else state.name,
            str(trace) if absolute else trace.name,
            str(tmp_path.resolve()),
        ],
        cwd=tmp_path,
        input=REQUEST,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["result"]["tools"][0]["name"] == "inside_tool"
    assert json.loads(trace.read_text())["method"] == "tools/list"
