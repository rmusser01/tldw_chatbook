"""Start only a fresh, task-owned networkless Linux probe; default is import-only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


BASE = Path(__file__).resolve().parent
SOURCE = Path("/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/tts-macos-burndown")
MODEL = Path("/private/tmp/kokoro-playback-validation/models/kokoro-v1.0.onnx")
VOICES = Path("/private/tmp/kokoro-playback-validation/models/voices-v1.0.bin")
TAG = "tldw-task32153-onnx-arm64:20260909"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-inference", action="store_true", help="Use only after the exclusive inference slot is granted")
    parser.add_argument("--name", help="New evidence directory name")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 1800:
        parser.error("timeout must be between 1 and 1800 seconds")
    run_id = args.name or ("inference-" if args.run_inference else "preflight-") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        parser.error("name must be one filename component")
    evidence_root = BASE / "evidence"
    evidence_root.mkdir(mode=0o700, exist_ok=True)
    output = evidence_root / run_id
    if output.exists() or (BASE / (run_id + "-launch.json")).exists():
        parser.error("evidence output or launch record already exists")
    name = "tldw-task32153-" + uuid.uuid4().hex[:12]
    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", TAG], text=True))[0]
    if inspected["Config"]["Labels"].get("tldw.validation.task") != "32153":
        raise ValueError("Image does not carry this task's ownership label")
    revision = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    command = ["docker", "run", "--rm", "--name", name, "--label", "tldw.validation.task=32153",
               "--platform", "linux/arm64", "--network", "none", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--cpus", "4",
               "--memory", "6g", "--pids-limit", "256", "--user", f"{os.getuid()}:{os.getgid()}",
               "--tmpfs", "/tmp:rw,nosuid,nodev,exec,size=536870912",
               "--mount", f"type=bind,source={SOURCE},target=/workspace,readonly",
               "--mount", f"type=bind,source={MODEL},target=/assets/kokoro.onnx,readonly",
               "--mount", f"type=bind,source={VOICES},target=/assets/voices.bin,readonly",
               "--mount", f"type=bind,source={BASE},target=/probe,readonly",
               "--mount", f"type=bind,source={evidence_root},target=/evidence",
               "--env", "PYTHONPATH=/workspace:/workspace/packages/tldw_profile_core/src"]
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "FTP_PROXY"):
        command += ["--env", variable + "=", "--env", variable.lower() + "="]
    command += [inspected["Id"], "python", "-B", "/probe/validate_linux_onnx.py",
                "--run-inference" if args.run_inference else "--preflight",
                "--output", "/evidence/" + run_id, "--host-output", str(output), "--source-revision", revision]
    record = {"image_id": inspected["Id"], "image_tag": TAG, "container_name": name,
              "source_root": str(SOURCE), "source_revision": revision, "model": str(MODEL), "voices": str(VOICES),
              "host_output": str(output), "argv": command, "timeout_seconds": args.timeout,
              "mode": "inference" if args.run_inference else "imports-only-preflight"}
    record_path = BASE / (run_id + "-launch.json")
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        result = subprocess.run(command, check=False, timeout=args.timeout)
        record["exit_code"] = result.returncode
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # The unpredictable unique name was just created by this exact invocation.
        subprocess.run(["docker", "stop", "--time", "5", name], check=False)
        record.update(exit_code=1, forced_container_stop=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print("LAUNCH_RECORD", record_path, flush=True)
    return record["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
