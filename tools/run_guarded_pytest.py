"""Run explicit pytest targets with normal guards and verified private storage.

Usage (from a normal user terminal):
    python tools/run_guarded_pytest.py PRIVATE_PARENT Tests/TTS/test_example.py

This changes only the child process's temporary-directory environment. It does
not change existing ACLs, profiles, recovery policy, or pytest's conftest loading.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def verify_temp_parent(parent: Path) -> None:
    """Check the native ancestry before creating disposable test storage."""
    from tldw_chatbook.Utils.private_paths import _native_close, _open_verified_parent

    fd, _ = _open_verified_parent(parent / "pytest-probe", missing_leaf_allowed=True)
    _native_close(fd)


def main() -> int:
    """Run targeted tests, then remove only this invocation's owned directory."""
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    if os.environ.get("TLDW_TEST_CONFIG_ROOT"):
        raise RuntimeError("refusing to reuse an inherited test profile")
    parent = Path(sys.argv[1]).resolve(strict=True)
    targets = sys.argv[2:]
    if "--noconftest" in targets or not any(
        arg.startswith("Tests/") for arg in targets
    ):
        raise RuntimeError("explicit guarded test targets required")
    print("Checking proposed temp parent:", parent, flush=True)
    verify_temp_parent(parent)
    print("Native path verification passed.", flush=True)
    temp = Path(tempfile.mkdtemp(prefix="chatbook-pytest-", dir=parent))
    try:
        verify_temp_parent(temp)
        env = {**os.environ, "TMPDIR": str(temp), "TEMP": str(temp), "TMP": str(temp)}
        return subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(temp / "case"),
                "-o",
                "asyncio_default_fixture_loop_scope=function",
                *targets,
            ],
            cwd=REPO,
            env=env,
            check=False,
        ).returncode
    finally:
        resolved_temp = temp.resolve(strict=True)
        if resolved_temp != temp or resolved_temp.parent != parent:
            raise RuntimeError("refusing cleanup outside the verified temp parent")
        shutil.rmtree(resolved_temp)


if __name__ == "__main__":
    raise SystemExit(main())
