"""Serial native ABBA comparison; retain exact subprocess exit and source hashes."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
BASE = "3742efae9428dc10f879606b519215913263ac4e"
SOURCE = "tldw_chatbook/UI/Console_Modules/session.py"
RUNS = (("control1", True), ("after1", False), ("after2", False), ("control2", True))
for tag, original in RUNS:
    command = [
        sys.executable,
        str(OUT / "probe.py"),
        "--ready",
        "--resume-only",
        "--attach-poll",
        "--clock-safe",
        "--tag",
        f"attachment-{tag}",
    ]
    if original:
        command.append("--attachment-baseline")
    receipt = {
        "tag": tag,
        "baseline_revision": BASE if original else None,
        "source_sha256": hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest(),
        "checkout_revision": subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
    }
    started = time.monotonic()
    with (OUT / f"attachment-{tag}.log").open("wb") as log:
        try:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=log, timeout=300
            )
            receipt["exit_code"] = result.returncode
        except subprocess.TimeoutExpired:
            receipt["timeout_seconds"] = 300
    receipt["seconds"] = time.monotonic() - started
    (OUT / f"attachment-{tag}-exit.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt), flush=True)
    if receipt.get("exit_code") != 0:
        raise SystemExit(1)
