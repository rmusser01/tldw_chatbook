"""Native QA must reject invalid CLI input before app imports or output writes."""

import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNERS = [
    REPO / "Docs/superpowers/qa/2026-09-18-mcp-lifecycle-cancellation/current-dev/native_check.py",
    REPO / "Docs/superpowers/qa/2026-09-20-mcp-inspector-refresh/native_check.py",
    REPO / "Docs/superpowers/qa/2026-09-19-approval-action-ownership/native_check.py",
    REPO / "Docs/superpowers/qa/2026-09-19-mcp-recovery-catalog/native_check.py",
    REPO / "Docs/superpowers/qa/2026-09-18-mcp-restored-roots/native_check.py",
    REPO / "Docs/superpowers/qa/2026-09-19-mcp-tools-header/native_check.py",
    REPO / "Docs/superpowers/qa/2026-09-18-mcp-audit-catalog-freshness/modal_check.py",
    REPO / "Docs/superpowers/qa/2026-09-18-mcp-audit-catalog-freshness/native_check.py",
    REPO / "Docs/superpowers/qa/2026-09-18-mcp-audit-navigation/native_check.py",
    REPO / "Docs/superpowers/qa/2026-09-19-mcp-session-revocation/native_check.py",
    REPO / "Docs/superpowers/qa/2026-09-19-mcp-rule-actions/native_check.py",
    REPO / "Docs/superpowers/qa/2026-09-19-mcp-permission-navigation/native_check.py",
]


@pytest.mark.parametrize(
    "runner",
    RUNNERS,
    ids=[
        "lifecycle-cancellation",
        "inspector-refresh",
        "approval-actions",
        "recovery-catalog",
        "restored-roots",
        "tools-header",
        "modal-selector",
        "catalog-freshness",
        "audit-navigation",
        "revoke",
        "rule-actions",
        "permission-navigation",
    ],
)
@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["ROOT"],
        ["ROOT", "socket", "session", "extra"],
        ["ROOT", "../socket", "session"],
        ["ROOT", "socket", "session:window"],
        ["ROOT", "socket", "space name"],
        ["ROOT", "socket", "x" * 65],
    ],
)
def test_invalid_cli_exits_with_usage_without_touching_profile(
    tmp_path, arguments, runner
):
    root = tmp_path / "profile"
    root.mkdir()
    sentinel = root / "config.toml"
    sentinel.write_text("# Must not be read or changed for invalid arguments\n")
    arguments = [str(root) if value == "ROOT" else value for value in arguments]
    env = dict(os.environ, HOME=str(root), USERPROFILE=str(root))
    result = subprocess.run(
        [sys.executable, str(runner), *arguments],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "usage:" in result.stderr
    assert "Traceback" not in result.stderr
    assert list(root.iterdir()) == [sentinel]
    assert sentinel.read_text().startswith("# Must not be read")


@pytest.fixture
def prepared_profile(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="mcp-qa-args-", dir="/tmp") as temporary:
        base = Path(temporary).resolve()
        root = base / "profile"
        root.mkdir()
        for child in ("home", "config", "data"):
            (root / child).mkdir()
        (root / "config.toml").write_text(
            f'[paths]\ndata_dir = "{root}/data"\n'
            f'[database]\nUSER_DB_BASE_DIR = "{root}/data"\n'
            f'chat_db_path = "{root}/data/chat.db"\n'
        )
        binary = base / "tmux"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o700)
        monkeypatch.setenv("PATH", str(base))
        yield root, binary


def parse_profile(root, socket_name="review-socket_1", session_name="review"):
    parse = runpy.run_path(str(REPO / "Docs/superpowers/qa/native_runner_args.py"))[
        "parse_native_args"
    ]
    return parse([str(root), socket_name, session_name])


@pytest.mark.parametrize("names", [("socket\n", "review"), ("socket", "review\n")])
def test_tmux_names_reject_trailing_newlines(prepared_profile, names):
    root, _ = prepared_profile
    with pytest.raises(SystemExit) as error:
        parse_profile(root, *names)
    assert error.value.code == 2


def test_prepared_profile_and_path_discovered_tmux_are_returned_without_writes(
    prepared_profile,
):
    root, binary = prepared_profile
    before = sorted(root.rglob("*"))
    environment = dict(os.environ)
    args = parse_profile(root)
    assert args.root == root
    assert args.tmux_path == str(binary)
    assert args.tmux_socket == "review-socket_1"
    assert args.session == "review"
    assert sorted(root.rglob("*")) == before
    assert dict(os.environ) == environment


@pytest.mark.parametrize(
    "unsafe",
    ["outside", "alias", "traversal", "tmp-base", "escaped-home", "escaped-db"],
)
def test_unsafe_profile_is_rejected_before_output(prepared_profile, unsafe):
    root, _ = prepared_profile
    candidate = root
    if unsafe == "outside":
        candidate = REPO
    elif unsafe == "alias":
        candidate = root.parent / "alias"
        candidate.symlink_to(root, target_is_directory=True)
    elif unsafe == "traversal":
        candidate = root / ".." / "profile"
    elif unsafe == "tmp-base":
        candidate = Path("/tmp").resolve()
    elif unsafe == "escaped-home":
        (root / "home").rmdir()
        (root / "home").symlink_to(root.parent, target_is_directory=True)
    else:
        config = root / "config.toml"
        config.write_text(
            config.read_text().replace(f"{root}/data/chat.db", f"{root.parent}/chat.db")
        )
    with pytest.raises(SystemExit) as error:
        parse_profile(candidate)
    assert error.value.code == 2
    assert not (root / "native.log").exists()
    assert not (root / "launch.json").exists()
    assert not (root / "evidence").exists()


@pytest.mark.parametrize(
    "output", ["native.log", "launch.json", "evidence", "dangling-log"]
)
def test_existing_output_is_preserved_and_rejected(prepared_profile, output):
    root, _ = prepared_profile
    if output == "dangling-log":
        target = root / "native.log"
        target.symlink_to(root / "missing")
    else:
        target = root / output
        target.write_text("previous evidence")
    with pytest.raises(SystemExit) as error:
        parse_profile(root)
    assert error.value.code == 2
    if output == "dangling-log":
        assert target.is_symlink()
        assert not target.exists()
    else:
        assert target.read_text() == "previous evidence"


def test_missing_tmux_is_reported_before_output(prepared_profile, monkeypatch, capsys):
    root, _ = prepared_profile
    monkeypatch.setenv("PATH", "")
    with pytest.raises(SystemExit) as error:
        parse_profile(root)
    assert error.value.code == 2
    assert "tmux must be installed" in capsys.readouterr().err
    assert not (root / "native.log").exists()
