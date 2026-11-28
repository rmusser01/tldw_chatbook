"""Run a single sealed hosted comparison and qualify only complete cohorts."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
import uuid

from loopback_fixture import require_closed_port, serve

E = Path(__file__).resolve().parent
PREFIX = "postsave-current31-rag-path-"
PROFILE_PARENT = Path("/private/tmp/pr3023-live-current-r2")


def load(path):
    return json.loads(path.read_bytes())


def verify_inputs(seal):
    for name, expected in seal["files"].items():
        assert hashlib.sha256((E / name).read_bytes()).hexdigest() == expected, name


def main():
    assert __debug__ and sys.platform == "darwin"
    manifest = load(E / f"{PREFIX}paired-frozen-chain-r1.json")
    seal = load(E / f"{PREFIX}prelaunch-seal-r1.json")
    source = Path(os.environ["FIRSTSEND_SOURCE"])
    artifacts = Path(os.environ["FIRSTSEND_ARTIFACTS"])
    packet = E / manifest["packet"]
    launch = E / f"{PREFIX}quiet-launch-r1.json"
    watch = E / f"{PREFIX}host-watch-r1.json"
    result = E / f"{PREFIX}paired-qualified-r1.json"
    stub_receipt = E / "loopback-http.json"
    assert not any(path.exists() for path in (packet, launch, watch, result, stub_receipt, PROFILE_PARENT))
    verify_inputs(seal)

    def child(script, *args):
        return subprocess.run([sys.executable, str(E / script), *args], cwd=source).returncode

    try:
        runtime = {
            "macos": platform.mac_ver()[0],
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "logical_cpus": os.cpu_count(),
            "runner_image": os.environ.get("ImageVersion"),
            "transport": "loopback HTTP SSE",
        }
        (E / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        label = "hosted-" + uuid.uuid4().hex
        quiet_status = child(f"{PREFIX}quiet-host-r1.py", "--label", label)
        if quiet_status:
            return quiet_status
        data = (E / f"{PREFIX}quiet-{label}-r1.json").read_bytes()
        quiet = json.loads(data)
        age = datetime.now(timezone.utc) - datetime.fromisoformat(quiet["checked_utc"])
        assert quiet["continuous_quiet_30s"] and 0 <= age.total_seconds() < 30
        for port in seal["ports_unused"]:
            require_closed_port(port)
        with launch.open("xb") as handle:
            handle.write(data)

        with serve(stub_receipt) as requests:
            watch_status = child(f"{PREFIX}host-watch-r1.py")
        if watch_status:
            return watch_status
        observed = load(watch)
        assert observed["measure_exit"] == 0 and observed["overlaps"] == []
        rows = load(packet / "plan-closed.json")["runs"]
        assert len(rows) == len(requests) == 8
        for row, request in zip(rows, requests, strict=True):
            audit = load(packet / f"run{row['number']}/closed-audit.json")
            sent, opened, closed = audit["http"]
            assert request["method"] == "POST" and request["status"] == 200
            assert sent["path"] == "/chat/completions" and sent["stream"] is True
            assert sent["monotonic_ns"] <= request["received_monotonic_ns"] <= opened["monotonic_ns"] <= closed["monotonic_ns"]
        verify_inputs(seal)
        return child(f"{PREFIX}qualify-pairs-r1.py")
    finally:
        # The hosted profiles hold only synthetic fixture inputs. Copy the
        # specific audit inputs and leave browser/process outputs intact.
        for path in sorted(PROFILE_PARENT.glob("pr3023-*")):
            dest = artifacts / "profiles" / path.name
            dest.mkdir(parents=True)
            for name in ("http.jsonl", "app-start.json", "app-returned.json", "config.toml",
                         "data/matched-verification/tldw_cli_app.log"):
                item = path / name
                if item.is_file():
                    shutil.copy2(item, dest / item.name)
        for path in (packet, watch, result, stub_receipt, E / "runtime.json",
                     E / f"{PREFIX}measure-r1.log", *E.glob(f"{PREFIX}quiet-*.json")):
            if path.is_dir():
                shutil.copytree(path, artifacts / path.name)
            elif path.is_file():
                shutil.copy2(path, artifacts / path.name)


if __name__ == "__main__":
    raise SystemExit(main())
