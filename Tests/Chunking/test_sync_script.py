# Tests/Chunking/test_sync_script.py
"""Contract tests for the vendoring sync script (spec §5.2, §0 wrong-tree hazard)."""
import importlib
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
    # GPLv3 §4: the licence text itself must ship with the vendored subtree
    assert "LICENSES/GPL-3.0-only.txt" in manifest["files"]["extra"]
    assert manifest["licence"]["spdx"] == "GPL-3.0-only"


def test_manifest_templates_vendored_not_excluded():
    """Spec §6.1: vendoring templates.py is a MOVE from `excluded` to
    `vendored` — the name left in both lists would make a sync run ambiguous."""
    vendored = manifest_vendored()
    excluded = manifest_excluded()
    assert "templates.py" in vendored
    assert "templates.py" not in excluded
    # the spec's ambiguity warning, enforced generally: no file in both lists
    assert not (set(vendored) & set(excluded)), \
        f"files in both vendored and excluded: {sorted(set(vendored) & set(excluded))}"


def test_manifest_auto_planner_vendored_not_excluded():
    """Spec §4.1: vendoring auto_planner.py is the same MOVE pattern — never
    in both lists (a sync run would be ambiguous about which list wins)."""
    manifest = tomllib.loads((ENGINE / "VENDOR_MANIFEST.toml").read_text())
    vendored = set(manifest["files"]["vendored"])
    excluded = set(manifest["files"]["excluded"])
    assert "auto_planner.py" in vendored and "auto_planner.py" not in excluded
    assert not (vendored & excluded)  # spec §0.2: never both lists


def test_auto_planner_importable_zero_new_shims():
    """Spec §4.1: auto_planner.py is stdlib-only at the pin, so the synced
    file must carry no _shims reference at all — zero rewritten lines."""
    from tldw_chatbook.Chunking.engine import auto_planner
    from tldw_chatbook.Chunking.engine.auto_planner import plan_auto_chunking
    assert callable(plan_auto_chunking)
    # stdlib-only at the pin — the module must not import _shims at all
    import inspect
    assert "_shims" not in inspect.getsource(auto_planner)


def test_engine_tree_complete():
    # auto-selection task 1 (spec §4.1): vendoring auto_planner.py takes the
    # manifest 36 -> 37 entries (the engine tree goes 37 -> 38 .py files
    # counting the chatbook-authored __init__.py).
    assert len(manifest_vendored()) == 37
    for rel in manifest_vendored():
        assert (ENGINE / rel).exists(), f"missing vendored file {rel}"
    for rel in manifest_extra():
        assert (ENGINE / rel).exists(), f"missing extra file {rel}"
    # GPLv3 §4: the shipped extra really is the GPL-3.0 text, not a stub
    gpl = (ENGINE / "LICENSES" / "GPL-3.0-only.txt").read_text(errors="ignore")
    assert "GNU GENERAL PUBLIC LICENSE" in gpl
    assert "Version 3, 29 June 2007" in gpl
    # templates.py is vendored (spec §6.1) and importable (see below)
    assert (ENGINE / "templates.py").exists()
    # auto_planner.py is vendored (spec §4.1) and importable (see below)
    assert (ENGINE / "auto_planner.py").exists()
    # excluded-by-design files must NOT exist
    for rel in ("template_initialization.py",
                "async_chunker.py", "auto_boundary_assistant.py",
                "strategies/propositions.py", "utils/proposition_eval.py"):
        assert not (ENGINE / rel).exists(), f"deferred file vendored: {rel}"
    # upstream's own __init__ must not be vendored (chatbook-authored instead)
    assert "load_and_log_configs" not in (ENGINE / "__init__.py").read_text()


def test_templates_importable_zero_new_shims():
    """Spec §6.1 import table: templates.py resolves with ZERO new shims.
    Its only server import (is_truthy) lands on #1's `_shims.testing` via the
    sync script's second rewrite rule; everything else is stdlib, loguru, or
    already-vendored relative imports."""
    mod = importlib.import_module("tldw_chatbook.Chunking.engine.templates")
    # the surface chatbook consumes (spec §6.2/§6.3): processor + 2 dataclasses
    assert hasattr(mod, "TemplateProcessor")
    assert hasattr(mod, "TemplateStage")
    assert hasattr(mod, "ChunkingTemplate")
    # proof the rewritten import binds to the existing shim, not a new one
    from tldw_chatbook.Chunking._shims.testing import is_truthy
    assert mod.is_truthy is is_truthy


def manifest_vendored():
    return tomllib.loads((ENGINE / "VENDOR_MANIFEST.toml").read_text())["files"]["vendored"]


def manifest_extra():
    return tomllib.loads((ENGINE / "VENDOR_MANIFEST.toml").read_text())["files"]["extra"]


def manifest_excluded():
    return tomllib.loads((ENGINE / "VENDOR_MANIFEST.toml").read_text())["files"]["excluded"]


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
