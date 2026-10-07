"""Denied launcher admission must stop before the pytest import boundary."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    ROOT / "Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py"
)


def test_denied_startup_stops_before_importing_pytest(tmp_path):
    private = tmp_path / "refused-profile"
    bootstrap = private / "home/.config/tldw_cli/recovery-bootstrap"
    bootstrap.mkdir(mode=0o700, parents=True)
    # Actual invalid recovery evidence exercises the unchanged admission check.
    (bootstrap / "pending-invalid.json").write_text("{}", encoding="utf-8")
    environment = dict(os.environ)
    environment["APPROVAL_REFUSAL_TEST_ROOT"] = str(private)
    environment["APPROVAL_REFUSAL_LAUNCHER"] = str(LAUNCHER)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins, os, runpy, sys, tempfile
from pathlib import Path
root = Path(os.environ['APPROVAL_REFUSAL_TEST_ROOT'])
original_import = builtins.__import__
def observe_import(name, *args, **kwargs):
    if name == 'pytest':
        (root / 'pytest-imported').write_text('unexpected collection import')
        raise SystemExit(77)
    return original_import(name, *args, **kwargs)
builtins.__import__ = observe_import
# Keep the real launcher and admission implementation; supply an owned fixture
# root so its existing startup check sees the actual invalid recovery evidence.
tempfile.mkdtemp = lambda *args, **kwargs: str(root)
sys.argv = [os.environ['APPROVAL_REFUSAL_LAUNCHER']]
runpy.run_path(sys.argv[0], run_name='__main__')
""",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert '"startup_permission": [false, "recovery_scope_uncertain"]' in child.stdout
    assert not (private / "pytest-imported").exists(), child.stdout + child.stderr
    assert child.returncode == 1, child.stdout + child.stderr
