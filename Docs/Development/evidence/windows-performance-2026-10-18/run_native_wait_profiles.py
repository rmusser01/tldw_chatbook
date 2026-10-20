"""Run serial ABBA comparisons with exact exits and product source receipts."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
CONTROL_BAR = "--control-bar" in sys.argv
MEMORY = "--memory" in sys.argv
PREFIX = "memory" if MEMORY else "control-bar" if CONTROL_BAR else "native-waits"
SOURCES = (
    "tldw_chatbook/UI/Screens/home_screen.py",
    "tldw_chatbook/UI/Screens/chat_screen.py",
    "tldw_chatbook/Utils/textual_css_fastpath.py",
    "tldw_chatbook/UI/Console_Modules/session.py",
    "tldw_chatbook/Chat/console_preparation_reads.py",
    "tldw_chatbook/UI/Console_Modules/config_sync.py",
    "tldw_chatbook/UI/Console_Modules/console_spend_projection.py",
    "tldw_chatbook/UI/Console_Modules/workspace.py",
)
for tag, original in (
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
        f"{PREFIX}-{tag}",
    ]
    if original:
        command.append(
            "--memory-baseline"
            if MEMORY
            else "--control-bar-baseline"
            if CONTROL_BAR
            else "--native-wait-baseline"
        )
    receipt = {
        "tag": tag,
        "original": original,
        "checkout_revision": subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
    }
    started = time.monotonic()
    with (OUT / f"{PREFIX}-{tag}.log").open("wb") as log:
        try:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=log, timeout=300
            )
            receipt["exit_code"] = result.returncode
        except subprocess.TimeoutExpired:
            receipt["timeout_seconds"] = 300
    receipt["seconds"] = time.monotonic() - started
    (OUT / f"{PREFIX}-{tag}-exit.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt), flush=True)
    if receipt.get("exit_code") != 0:
        raise SystemExit(1)
