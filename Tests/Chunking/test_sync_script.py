# Tests/Chunking/test_sync_script.py
"""Contract tests for the vendoring sync script (spec §5.2, §0 wrong-tree hazard)."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from Helper_Scripts import sync_chunking_engine as sync_helper

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "tldw_chatbook" / "Chunking" / "engine"
SYNC = REPO / "Helper_Scripts" / "sync_chunking_engine.py"
PIN = "385afa951922c8a9dc2002c675bb6cad65e4ac23"

# The network-cloning path is a manual helper feature, never a test fallback.
# Opt in to the mutating sync contract with an explicit pinned local worktree.
SOURCE = os.environ.get("TLDW_SERVER_SYNC_SOURCE")
SYNC_TIMEOUT_SECONDS = 300
SOURCE_CHECK_TIMEOUT_SECONDS = 10


def _validated_source() -> Path:
    if not SOURCE:
        pytest.skip(
            "set TLDW_SERVER_SYNC_SOURCE to a local tldw_server checkout at the pin"
        )

    source = Path(SOURCE).expanduser().resolve()
    if not source.is_dir():
        pytest.skip(f"TLDW_SERVER_SYNC_SOURCE does not exist: {source}")

    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=SOURCE_CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("timed out validating TLDW_SERVER_SYNC_SOURCE")
    if result.returncode != 0:
        pytest.fail(f"TLDW_SERVER_SYNC_SOURCE is not a Git checkout: {result.stderr}")
    if result.stdout.strip() != PIN:
        pytest.fail(
            f"TLDW_SERVER_SYNC_SOURCE must be at pinned commit {PIN}; "
            f"found {result.stdout.strip() or '<no HEAD>'}"
        )
    return source


def _run_sync() -> subprocess.CompletedProcess:
    source = _validated_source()
    cmd = [sys.executable, str(SYNC), "--source", str(source)]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=SYNC_TIMEOUT_SECONDS,
    )


def test_manifest_pins_upstream():
    manifest = tomllib.loads((ENGINE / "VENDOR_MANIFEST.toml").read_text())
    assert manifest["upstream"]["repo"] == "https://github.com/rmusser01/tldw_server.git"
    assert manifest["upstream"]["branch"] == "dev"
    assert manifest["upstream"]["commit"] == PIN
    assert "chunker.py" in " ".join(manifest["files"]["vendored"])
    assert "LICENSE" in manifest["files"]["extra"]
    # GPLv3 §4: the licence text itself must ship with the vendored subtree
    assert "LICENSES/GPL-3.0-only.txt" in manifest["files"]["extra"]
    assert manifest["licence"]["spdx"] == "GPL-3.0-only"


def test_engine_tree_complete():
    for rel in manifest_vendored():
        assert (ENGINE / rel).exists(), f"missing vendored file {rel}"
    for rel in manifest_extra():
        assert (ENGINE / rel).exists(), f"missing extra file {rel}"
    # GPLv3 §4: the shipped extra really is the GPL-3.0 text, not a stub
    gpl = (ENGINE / "LICENSES" / "GPL-3.0-only.txt").read_text(errors="ignore")
    assert "GNU GENERAL PUBLIC LICENSE" in gpl
    assert "Version 3, 29 June 2007" in gpl
    # excluded-by-design files must NOT exist
    for rel in ("templates.py", "template_initialization.py", "auto_planner.py",
                "async_chunker.py", "auto_boundary_assistant.py",
                "strategies/propositions.py", "utils/proposition_eval.py"):
        assert not (ENGINE / rel).exists(), f"deferred file vendored: {rel}"
    # upstream's own __init__ must not be vendored (chatbook-authored instead)
    assert "load_and_log_configs" not in (ENGINE / "__init__.py").read_text()


def manifest_vendored():
    return tomllib.loads((ENGINE / "VENDOR_MANIFEST.toml").read_text())["files"]["vendored"]


def manifest_extra():
    return tomllib.loads((ENGINE / "VENDOR_MANIFEST.toml").read_text())["files"]["extra"]


def test_no_server_imports_remain():
    for py in ENGINE.rglob("*.py"):
        src = py.read_text()
        assert "tldw_Server_API" not in src, f"{py.name} still references upstream package"
        assert "from app.core" not in src, f"{py.name} still references app.core"


def test_sync_idempotent_and_rejects_local_edits():
    r1 = _run_sync()
    assert r1.returncode == 0, r1.stderr
    # second run is a no-op
    r2 = _run_sync()
    assert r2.returncode == 0, r2.stderr
    # local modification → loud failure
    victim = ENGINE / "constants.py"
    original = victim.read_text()
    victim.write_text(original + "\n# local edit\n")
    try:
        r3 = _run_sync()
        assert r3.returncode != 0, "sync must fail loudly on local modifications"
        assert "local modification" in (r3.stderr + r3.stdout).lower()
    finally:
        victim.write_text(original)


def test_sync_skips_before_subprocess_when_configured_source_is_absent(
    monkeypatch, tmp_path
):
    missing_source = tmp_path / "missing-tldw-server"
    calls = []

    monkeypatch.setattr(sys.modules[__name__], "SOURCE", str(missing_source))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(pytest.skip.Exception, match="TLDW_SERVER_SYNC_SOURCE"):
        _run_sync()

    assert calls == []


def test_sync_validates_pin_before_starting_sync(monkeypatch, tmp_path):
    source = tmp_path / "tldw-server"
    source.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="wrong-pin\n", stderr="")

    monkeypatch.setattr(sys.modules[__name__], "SOURCE", str(source))
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(pytest.fail.Exception, match="pinned commit"):
        _run_sync()

    assert len(calls) == 1
    assert calls[0][0][:3] == ["git", "-C", str(source.resolve())]


def test_sync_subprocess_has_bounded_timeout(monkeypatch, tmp_path):
    source = tmp_path / "tldw-server"
    source.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        stdout = f"{PIN}\n" if command[0] == "git" else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(sys.modules[__name__], "SOURCE", str(source))
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _run_sync()

    assert result.returncode == 0
    sync_call = calls[-1]
    assert sync_call[0][-2:] == ["--source", str(source.resolve())]
    assert 0 < sync_call[1]["timeout"] <= 300


@pytest.mark.parametrize("sync_fails", [False, True], ids=["success", "failure"])
def test_owned_temporary_clone_is_removed(monkeypatch, tmp_path, sync_fails):
    owned_clone = tmp_path / "owned-clone"
    subprocess_calls = []

    def fake_mkdtemp(*, prefix):
        assert prefix == "tldw_server_sync_"
        owned_clone.mkdir()
        return str(owned_clone)

    def fake_run(command, **kwargs):
        subprocess_calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    def fake_sync(worktree):
        assert worktree == owned_clone
        if sync_fails:
            raise RuntimeError("injected sync failure")
        return 0

    monkeypatch.setattr(sync_helper.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(sync_helper.subprocess, "run", fake_run)
    monkeypatch.setattr(sync_helper, "_sync_worktree", fake_sync, raising=False)

    if sync_fails:
        with pytest.raises(RuntimeError, match="injected sync failure"):
            sync_helper._run_with_source(None)
    else:
        assert sync_helper._run_with_source(None) == 0

    assert not owned_clone.exists()
    assert len(subprocess_calls) == 2
    assert all(0 < call[1]["timeout"] for call in subprocess_calls)


def test_supplied_source_is_never_removed(monkeypatch, tmp_path):
    supplied_source = tmp_path / "supplied-source"
    supplied_source.mkdir()

    monkeypatch.setattr(sync_helper, "verify_clean", lambda source: None)
    monkeypatch.setattr(sync_helper, "_sync_worktree", lambda source: 0, raising=False)

    assert sync_helper._run_with_source(str(supplied_source)) == 0
    assert supplied_source.exists()
