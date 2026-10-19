"""Serial ABBA return profiles with exact subprocess exit receipts."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sources = [
    "tldw_chatbook/UI/Screens/chat_screen.py",
    "tldw_chatbook/Utils/textual_css_fastpath.py",
]
for tag, original in (
    ("benchmark", False),
    ("control1", True),
    ("after1", False),
    ("after2", False),
    ("control2", True),
):
    command = [
        sys.executable,
        str(OUT / "probe.py"),
        "--ready",
        "--resume-only",
        "--attach-poll",
        "--clock-safe",
        "--tag",
        f"subject-{tag}",
    ]
    if tag == "benchmark":
        command.append("--subject-benchmark")
    if original:
        command.append("--subject-baseline")
    receipt = {
        "tag": tag,
        "original": original,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sources
        },
    }
    started = time.monotonic()
    with (OUT / f"subject-{tag}.log").open("wb") as log:
        try:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=log, timeout=300
            )
            receipt["exit_code"] = result.returncode
        except subprocess.TimeoutExpired:
            receipt["timeout_seconds"] = 300
    receipt["seconds"] = time.monotonic() - started
    (OUT / f"subject-{tag}-exit.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt), flush=True)
    if receipt.get("exit_code") != 0:
        raise SystemExit(1)
