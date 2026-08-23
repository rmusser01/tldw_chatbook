# Tests/Chunking/test_sync_script.py
"""Contract tests for the vendoring sync script (spec §5.2, §0 wrong-tree hazard)."""
import importlib
import os, subprocess, sys, tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "tldw_chatbook" / "Chunking" / "engine"
SYNC = REPO / "Helper_Scripts" / "sync_chunking_engine.py"
PIN = "385afa951922c8a9dc2002c675bb6cad65e4ac23"

# The sync flow documents checking out a local worktree at the pin and running
# the script with --source. TASK-19574: there is deliberately no built-in
# fallback location any more (the old default, /tmp/tldw_server_sync, is a
# /private/tmp path -- the standing rule here is never to keep work there,
# since the macOS cleaner has destroyed a worktree in it three times). Point
# TLDW_SERVER_SYNC_SOURCE at a local tldw_server worktree checked out at PIN
# to run test_sync_idempotent_and_rejects_local_edits; when it is unset (or
# the path is gone), that test SKIPS instead of falling through to
# sync_chunking_engine.py's no-arg path -- which git-clones the ~1.0 GiB
# upstream repo per invocation (three times for this one test) and, before
# this task, never cleaned its temp clone up. This module must never itself
# trigger that network clone.
SOURCE = os.environ.get("TLDW_SERVER_SYNC_SOURCE")


def _run_sync() -> subprocess.CompletedProcess:
    assert SOURCE and Path(SOURCE).exists(), (
        "_run_sync() must only be called once the caller has confirmed SOURCE "
        "exists (see the pytest.skip guard in "
        "test_sync_idempotent_and_rejects_local_edits) -- it never falls "
        "through to the script's no-arg network-clone path."
    )
    cmd = [sys.executable, str(SYNC), "--source", SOURCE]
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


def test_manifest_propositions_vendored_not_excluded():
    """2026-08-23 propositions spec §5: the 39th file is a MOVE from
    `excluded` to `vendored` in BOTH the manifest and the sync script's
    VENDORED list — never both lists."""
    vendored = set(manifest_vendored())
    excluded = set(manifest_excluded())
    assert "strategies/propositions.py" in vendored
    assert "strategies/propositions.py" not in excluded
    assert not (vendored & excluded)
    # the move is mirrored in the sync script's own list (single source per
    # tool; the manifest documents, the script acts)
    sync_src = SYNC.read_text()
    assert '"strategies/propositions.py"' in sync_src


def test_propositions_importable_zero_new_shims():
    """2026-08-23 propositions spec §5: the file resolves with ZERO new
    shims — its only server import (prompt_loader) lands on #1's existing
    `_shims/Utils/prompt_loader` via the second rewrite rule; `..base` is
    relative and already vendored."""
    mod = importlib.import_module(
        "tldw_chatbook.Chunking.engine.strategies.propositions")
    from tldw_chatbook.Chunking.engine.strategies.propositions import (
        PropositionChunkingStrategy,
    )
    assert callable(PropositionChunkingStrategy)
    # the rewritten import binds to the existing shim, not a new one
    from tldw_chatbook.Chunking._shims.Utils.prompt_loader import load_prompt
    assert mod.load_prompt is load_prompt
    # ...and no shim may reference the strategy back (zero shims, both ways)
    shims_root = REPO / "tldw_chatbook" / "Chunking" / "_shims"
    for py in shims_root.rglob("*.py"):
        assert "engine.strategies.propositions" not in py.read_text(), \
            f"{py.name} references the propositions strategy"


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
    # propositions vendoring (2026-08-23 spec §5): the manifest goes 37 -> 38
    # entries and the engine tree 38 -> 39 .py files (counting the
    # chatbook-authored __init__.py) — the spec's "39th file".
    assert len(manifest_vendored()) == 38
    assert len([p for p in ENGINE.rglob("*.py")]) == 39
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
    # strategies/propositions.py is vendored (2026-08-23 spec §5) and
    # importable (see below)
    assert (ENGINE / "strategies" / "propositions.py").exists()
    # descope-ruled / not-vendored files must NOT exist (spec §4 ledger)
    for rel in ("template_initialization.py",
                "async_chunker.py", "auto_boundary_assistant.py",
                "utils/proposition_eval.py"):
        assert not (ENGINE / rel).exists(), f"descoped file vendored: {rel}"
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
    if not SOURCE or not Path(SOURCE).exists():
        pytest.skip(
            "TLDW_SERVER_SYNC_SOURCE is not set to an existing local "
            f"tldw_server worktree checked out at pin {PIN}; skipping rather "
            "than falling through to sync_chunking_engine.py's no-arg "
            "network-clone path (TASK-19574 -- this test must never itself "
            "trigger a ~1.0 GiB clone from GitHub). Set up a worktree with, "
            "e.g.: git -C <tldw_server checkout> worktree add "
            f"<dest> {PIN} && TLDW_SERVER_SYNC_SOURCE=<dest> pytest "
            "Tests/Chunking/test_sync_script.py"
        )
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
