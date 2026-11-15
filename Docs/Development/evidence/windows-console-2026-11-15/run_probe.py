"""Bounded process launcher for the guarded native Console journey."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("tag")
parser.add_argument("--diagnostic", action="store_true")
parser.add_argument("--cpu-profile", action="store_true")
args = parser.parse_args()
if not args.tag or any(
    c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    for c in args.tag
):
    parser.error("unsafe tag")
name = ("diagnostic" if args.diagnostic else "plain") + "-" + args.tag
command = [
    sys.executable,
    "-X",
    "utf8",
    str(OUT / "probe.py"),
    "--ready",
    "--resume-only",
    "--clock-safe",
    "--attach-poll",
    "--tag",
    args.tag,
]
if args.diagnostic:
    command.append("--return-observe")
if args.cpu_profile:
    if not args.diagnostic:
        parser.error("--cpu-profile requires --diagnostic")
    command.append("--cpu-profile")
head = subprocess.check_output(
    ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
    cwd=ROOT,
    text=True,
).strip()
harness = {
    file: hashlib.sha256((OUT / file).read_bytes()).hexdigest()
    for file in ("probe.py", "return_observer.py", "run_probe.py")
}
css = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((ROOT / "tldw_chatbook/css").rglob("*.tcss"))
}
started = time.monotonic()
timed_out = False
timeout_cleanup = None
with (OUT / (name + ".log")).open("w", encoding="utf-8") as output:
    child = subprocess.Popen(
        command,
        cwd=ROOT,
        env=dict(os.environ, PYTHONUTF8="1"),
        stdout=output,
        stderr=subprocess.STDOUT,
    )
    try:
        exit_code = child.wait(timeout=300)
    except subprocess.TimeoutExpired:
        timed_out = True
        exit_code = None
        # Windows venv python is a redirector. Terminate its owned tree before
        # losing the parent, so its actual interpreter cannot outlive the run.
        timeout_cleanup = {"pid": child.pid}
        try:
            killed = subprocess.run(
                ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            timeout_cleanup.update(
                returncode=killed.returncode, stdout=killed.stdout, stderr=killed.stderr
            )
            timeout_cleanup["root_returncode"] = child.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired) as error:
            timeout_cleanup["error"] = repr(error)
receipt = {
    "command": command,
    "head": head,
    "exit_code": exit_code,
    "timed_out": timed_out,
    "timeout_cleanup": timeout_cleanup,
    "external_deadline_seconds": 300,
    "seconds": time.monotonic() - started,
    "harness_sha256": harness,
    "css_sha256_before": css,
}
receipt["css_sha256_after"] = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((ROOT / "tldw_chatbook/css").rglob("*.tcss"))
}
receipt["css_unchanged"] = receipt["css_sha256_before"] == receipt["css_sha256_after"]
(OUT / (name + "-exit.json")).write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            k: v
            for k, v in receipt.items()
            if k not in ("css_sha256_before", "css_sha256_after", "harness_sha256")
        }
    )
)
if timed_out:
    raise SystemExit(
        "FAILED diagnostic: external deadline killed parent; no normal retirement claim"
    )
if not receipt["css_unchanged"]:
    raise SystemExit("CSS source changed during run")
raise SystemExit(exit_code)
