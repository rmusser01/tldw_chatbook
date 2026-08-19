# Tests/Chunking/test_sync_script.py
"""Contract tests for the vendoring sync script (spec §5.2, §0 wrong-tree hazard)."""
import os, subprocess, sys, tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "tldw_chatbook" / "Chunking" / "engine"
SYNC = REPO / "Helper_Scripts" / "sync_chunking_engine.py"
PIN = "385afa951922c8a9dc2002c675bb6cad65e4ac23"

# The sync flow documents checking out a local worktree at the pin and running
# the script with --source (the no-arg path git-clones the ~532 MiB upstream
# repo and never cleans up its temp dir). Use the local worktree when it is
# available; otherwise exercise the default GitHub-clone path.
SOURCE = os.environ.get("TLDW_SERVER_SYNC_SOURCE", "/tmp/tldw_server_sync")


def _run_sync() -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SYNC)]
    if Path(SOURCE).exists():
        cmd += ["--source", SOURCE]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))


def test_manifest_pins_upstream():
    manifest = tomllib.loads((ENGINE / "VENDOR_MANIFEST.toml").read_text())
    assert manifest["upstream"]["repo"] == "https://github.com/rmusser01/tldw_server.git"
    assert manifest["upstream"]["branch"] == "dev"
    assert manifest["upstream"]["commit"] == PIN
    assert "chunker.py" in " ".join(manifest["files"]["vendored"])
    assert "LICENSE" in manifest["files"]["extra"]
    assert manifest["licence"]["spdx"] == "GPL-3.0-only"


def test_engine_tree_complete():
    for rel in manifest_vendored():
        assert (ENGINE / rel).exists(), f"missing vendored file {rel}"
    # excluded-by-design files must NOT exist
    for rel in ("templates.py", "template_initialization.py", "auto_planner.py",
                "async_chunker.py", "auto_boundary_assistant.py",
                "strategies/propositions.py", "utils/proposition_eval.py"):
        assert not (ENGINE / rel).exists(), f"deferred file vendored: {rel}"
    # upstream's own __init__ must not be vendored (chatbook-authored instead)
    assert "load_and_log_configs" not in (ENGINE / "__init__.py").read_text()


def manifest_vendored():
    return tomllib.loads((ENGINE / "VENDOR_MANIFEST.toml").read_text())["files"]["vendored"]


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
