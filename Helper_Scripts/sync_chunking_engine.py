#!/usr/bin/env python3
"""Sync the vendored Chunking engine from tldw_server dev @ pinned SHA.

Spec §5.2: idempotent, SHA-verifying, loud on local modifications, never
syncs from an unverified local path.
"""
import argparse, hashlib, subprocess, sys, tempfile
from pathlib import Path

REPO = "https://github.com/rmusser01/tldw_server.git"
BRANCH = "dev"
PIN = "385afa951922c8a9dc2002c675bb6cad65e4ac23"

# Phase-1 file set (spec §5.1); excludes #2/#3/#6-deferred modules and
# upstream's own __init__.py (chatbook-authored instead, §5.1).
VENDORED = [
    "base.py", "chunker.py", "constants.py", "exceptions.py", "error_policy.py",
    "option_utils.py", "regex_safety.py", "security_logger.py",
    "multilingual.py", "llm_context.py",
    "process_text/__init__.py", "process_text/models.py", "process_text/options.py",
    "process_text/preparation.py", "process_text/dispatch.py",
    "process_text/pipeline.py", "process_text/metadata.py",
    "splitters/__init__.py", "splitters/regex.py", "splitters/blingfire.py",
    "strategies/__init__.py", "strategies/words.py", "strategies/sentences.py",
    "strategies/paragraphs.py", "strategies/tokens.py", "strategies/json_xml.py",
    "strategies/ebook_chapters.py", "strategies/ebook_chapters_patch.py",
    "strategies/structure_aware.py", "strategies/code.py", "strategies/code_ast.py",
    "strategies/fixed_size.py", "strategies/semantic.py",
    "strategies/rolling_summarize.py",
    "utils/metrics.py",
]
UPSTREAM_ROOT = "tldw_Server_API/app/core/Chunking"
TARGET_ROOT = Path("tldw_chatbook/Chunking/engine")


def rewrite_imports(src: str) -> str:
    # Mechanical, order matters: the Chunking-specific rule first.
    src = src.replace("tldw_Server_API.app.core.Chunking",
                      "tldw_chatbook.Chunking.engine")
    src = src.replace("tldw_Server_API.app.core",
                      "tldw_chatbook.Chunking._shims")
    # Slashed (filesystem-path) form of the same mapping, e.g. upstream
    # chunker.py's docstring pointer at its own README; keeps the vendored
    # tree free of any `tldw_Server_API` text (spec §0/§5.2, test contract).
    src = src.replace("tldw_Server_API/app/core/Chunking",
                      "tldw_chatbook/Chunking/engine")
    return src


def git_show(worktree: Path, path: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(worktree), "show", f"{PIN}:{UPSTREAM_ROOT}/{path}"],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FATAL: {path} not found at pinned SHA {PIN}: {r.stderr}")
    return r.stdout


def verify_clean(worktree: Path) -> None:
    """Wrong-tree hazard (spec §0): the source must match the pin exactly."""
    r = subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    if r.stdout.strip() != PIN:
        sys.exit(f"FATAL: worktree HEAD {r.stdout.strip()[:8]} != pin {PIN[:8]}; "
                 f"checkout the pinned SHA first (git checkout {PIN})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="Existing tldw_server worktree already at the pinned SHA")
    args = ap.parse_args()

    tmp = None
    if args.source:
        worktree = Path(args.source).resolve()
        verify_clean(worktree)
    else:
        tmp = tempfile.mkdtemp(prefix="tldw_server_sync_")
        worktree = Path(tmp)
        subprocess.run(["git", "clone", "--no-checkout", REPO, str(worktree)], check=True)
        subprocess.run(["git", "-C", str(worktree), "checkout", PIN], check=True)

    # 1. Refuse to overwrite local modifications (loud, spec §5.2)
    for rel in VENDORED + ["__init__.py"]:
        dst = TARGET_ROOT / rel
        if dst.exists():
            upstream = rewrite_imports(git_show(worktree, rel)) if rel != "__init__.py" else dst.read_text()
            if rel == "__init__.py":
                continue  # chatbook-authored, never touched by sync
            if dst.read_text() != upstream:
                sys.exit(f"FATAL: local modification to vendored file {rel}; "
                         f"revert it or move the change to a shim/subclass")

    # 2. Copy + rewrite
    for rel in VENDORED:
        dst = TARGET_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(rewrite_imports(git_show(worktree, rel)))

    # 3. Manifest + licence
    (TARGET_ROOT / "LICENSE").write_bytes(
        subprocess.run(["git", "-C", str(worktree), "show", f"{PIN}:LICENSE"],
                       capture_output=True).stdout)
    print(f"Synced {len(VENDORED)} files from {REPO} @ {PIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
