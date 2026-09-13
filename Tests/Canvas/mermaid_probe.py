"""Disposable Node/QuickJS semantic probe; no application imports or guest eval."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def run_mermaid_case(case: dict) -> dict:
    """Execute one bounded case with a fresh VM and owned process environment."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node is required for Mermaid QuickJS verification")
    with tempfile.TemporaryDirectory(prefix="canvas-mermaid-probe-") as directory:
        env = {
            "PATH": str(Path(node).parent),
            "HOME": directory,
            "TMPDIR": directory,
            "XDG_CONFIG_HOME": directory,
            "XDG_DATA_HOME": directory,
            "LANG": "C.UTF-8",
        }
        result = subprocess.run(
            [node, str(Path(__file__).with_suffix(".mjs"))],
            input=json.dumps(case),
            text=True,
            capture_output=True,
            cwd=directory,
            env=env,
            timeout=15,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "Mermaid probe host/API failure: " + result.stderr[-1500:]
            )
        if len(result.stdout.encode()) > 128 * 1024 or result.stderr:
            raise RuntimeError("invalid Mermaid probe output")
        return json.loads(result.stdout)
