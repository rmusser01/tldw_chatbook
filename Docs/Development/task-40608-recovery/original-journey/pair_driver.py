"""Run only explicit frozen, sequential original-journey qualification pairs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import time

HERE = Path(__file__).resolve().parent
ORDER = ((1, "baseline"), (1, "candidate"), (2, "candidate"),
         (2, "baseline"), (3, "baseline"), (3, "candidate"))
JOURNEYS = ("readme", "edit")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def durable(path, value, mode="x"):
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append(directory, **row):
    durable(directory / "order.jsonl", {
        "utc": datetime.now(timezone.utc).isoformat(), "clock_ns": time.perf_counter_ns(), **row,
    }, "a")


def configuration(path):
    settings = read(path)

    def resolve(value):
        return str((path.parent / value).resolve())

    for name in ("output_directory", "private_parent", "observer"):
        settings[name] = resolve(settings[name])
    for side in ("baseline", "candidate"):
        settings[side]["manifest_sha256"] = settings[side]["manifest_sha256"].lower()
        for name in ("directory", "runner"):
            settings[side][name] = resolve(settings[side][name])
    settings["files"] = {resolve(name): value.lower() for name, value in settings["files"].items()}
    settings["config_sha256"] = digest(path)
    return settings


def pins(config_path):
    settings = configuration(config_path)
    manifests = {}
    for name, expected in settings["files"].items():
        if digest(Path(name)) != expected:
            raise RuntimeError("Pinned qualification input changed: " + name)
    for side in ("baseline", "candidate"):
        item = settings[side]
        if type(item["exhaustive_switch"]) is not bool or (side == "candidate" and not item["exhaustive_switch"]):
            raise RuntimeError("Candidate must use the exhaustive runner switch")
        directory = Path(item["directory"])
        manifest_path = directory / "replay-manifest.json"
        if settings["files"].get(str(manifest_path)) != item["manifest_sha256"]:
            raise RuntimeError("Both exact manifests must be explicit configuration pins")
        manifest = manifests[side] = read(manifest_path)
        if manifest["candidate_commit"] != item["commit"]:
            raise RuntimeError("Frozen commit changed")
        if Path(manifest["candidate_directory"]).resolve() != directory:
            raise RuntimeError("Replay moved; rebuild its original import environment")
        if manifest["lab_observer"]["mode"] != "guard":
            raise RuntimeError("Admission-monitoring runs are not timing qualification")
        if manifest["lab_observer"]["route"] != ("local" if side == "baseline" else "process"):
            raise RuntimeError("Frozen route changed")
        for name in ("test_task40608.py", "task40608_trace.py"):
            if digest(directory / "Tests/QA" / name) != digest(HERE / name):
                raise RuntimeError("Original QA/trace is not byte-identical")
        if manifest["original_journey_sha256"] != manifest["candidate_journey_sha256"]:
            raise RuntimeError("Original journey fixture changed")
        if digest(HERE / "summarize.py") != manifest["original_summary_sha256"]:
            raise RuntimeError("Original timing summary changed")
        for name, expected in manifest["tool_sha256"].items():
            if digest(Path(item["runner"]).parent / name) != expected:
                raise RuntimeError("Frozen runner helper changed: " + name)
        if digest(Path(item["runner"])) != manifest["tool_sha256"].get(Path(item["runner"]).name):
            raise RuntimeError("Runner is not pinned by the replay manifest")
        if digest(Path(settings["observer"])) != manifest["candidate_sources"]["task40608_lab_observer.py"]:
            raise RuntimeError("Observer does not match both frozen replays")
    def core(manifest):
        return {key: value for key, value in manifest["candidate_sources"].items()
                if key.startswith("packages/tldw_profile_core/src/")}
    if not core(manifests["baseline"]) or core(manifests["baseline"]) != core(manifests["candidate"]):
        raise RuntimeError("Baseline and candidate must pin the same profile-core package")
    return settings


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_run(directory, side, item, settings, *, require_exhaustive=True):
    metadata = read(directory / "run-metadata.json")
    expected = settings["files"][str(Path(item["directory"]) / "replay-manifest.json")]
    if metadata["exitCode"] != 0 or metadata["manifestSha256"].lower() != expected:
        raise RuntimeError("Original runner failed or used different source")
    if metadata["candidate"] != item["commit"]:
        raise RuntimeError("Runner commit differs from the frozen input")
    if require_exhaustive and item["exhaustive_switch"] and metadata.get("verification") != "exhaustive":
        raise RuntimeError("Candidate lacks exhaustive qualification verification")
    artifacts = list(directory.glob("native-*"))
    if len(artifacts) != 1 or read(artifacts[0] / "receipt.json")["status"] != "passed":
        raise RuntimeError("Original journey did not pass")
    summary = load("original_summary", HERE / "summarize.py")
    observer = load("original_guard", Path(settings["observer"]))
    guard_directory = directory / "lab-observer"
    ui_paths = list(guard_directory.glob("ui-*.json"))
    if len(ui_paths) != 1:
        raise RuntimeError("Expected the original UI guard receipt")
    ui = read(ui_paths[0])
    children = [read(path) for path in guard_directory.glob("backend-*.json")]
    observer.validate_receipts(ui, children, side == "candidate")
    if ui["mode"] != "guard" or ui["exit_code"] != 0:
        raise RuntimeError("Guard-only successful run required")
    guards = {"ui_pid": ui["pid"], "backend_pids": ui["backend_pids"], "mode": ui["mode"]}
    rows = [json.loads(line) for line in (directory / "progress.jsonl").read_text(
        encoding="utf-8-sig").splitlines()]
    times = {}
    for journey in JOURNEYS:
        events = sorted((row for row in rows if row["turn"] == journey), key=lambda row: row["t"])
        original = summary.summarize(rows, journey)
        if original["events"].get("approval_click") != (2 if journey == "readme" else 3):
            raise RuntimeError("Original expected approval count changed")
        first = {}
        for row in events:
            first.setdefault(row["event"], row["t"])
        wait = 0.0
        if "painted_disclosure" in first and "disclosure_click" in first:
            wait += first["disclosure_click"] - first["painted_disclosure"]
        previous = -1.0
        for click in (row for row in events if row["event"] == "approval_click"):
            observed = [row for row in events if previous < row["t"] <= click["t"]
                        and row["event"] in ("painted_approval", "approval_observed")]
            painted = [row for row in observed if row["event"] == "painted_approval"]
            wait += click["t"] - min(painted or observed, key=lambda row: row["t"])["t"]
            previous = click["t"]
        endpoint = "reply_completed" if "reply_completed" in first else "reply_seen"
        elapsed = first[endpoint] - first["send"]
        seen = first["reply_seen"] - first["send"]
        if not all(math.isfinite(value) for value in (wait, elapsed, seen)) or wait < 0 or elapsed <= wait or seen <= wait:
            raise RuntimeError("Original timing boundaries are invalid")
        times[journey] = {
            "application_s": elapsed - wait, "original_completion_endpoint": endpoint,
            "common_reply_seen_application_s": seen - wait,
            "raw_elapsed_s": elapsed, "raw_reply_seen_s": seen,
            "original_observed_human_wait_s": wait, "original_summary": original,
        }
    return {"times": times, "guards": guards}


def report(directory):
    paths = sorted(directory.glob("attempt-*/order.jsonl"))
    if (directory / "order.jsonl").exists():
        paths.insert(0, directory / "order.jsonl")  # Preserve earlier driver ledgers.
    saved = read(directory / "inputs.json") if (directory / "inputs.json").exists() else {}
    expected_hashes = saved.get("files")
    pairs, pair_attempts, attempts = {}, {}, []
    for path in paths:
        rows, malformed = [], False
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                malformed = True
                break
        attempts.append({"ledger": path.relative_to(directory).as_posix(), "entries": len(rows),
                         "truncated": malformed})
        for number in (1, 2, 3):
            positions = [index for index, row in enumerate(rows) if row.get("pair") == number]
            if number in pairs or len(positions) != 4 or positions != list(range(positions[0], positions[0] + 4)):
                continue
            cohort = [rows[index] for index in positions]
            wanted = [side for pair, side in ORDER if pair == number]
            valid = all(
                (start.get("side"), start.get("status")) == (side, "started")
                and (end.get("side"), end.get("status"), end.get("code")) == (side, "finished", 0)
                and end.get("eligible") is True
                and start.get("clock_ns", -1) <= end.get("clock_ns", -1)
                for side, (start, end) in zip(wanted, (cohort[:2], cohort[2:]), strict=True)
            )
            valid = valid and cohort[1]["clock_ns"] <= cohort[2]["clock_ns"]
            if positions[0]:
                valid = valid and rows[positions[0] - 1]["clock_ns"] <= cohort[0]["clock_ns"]
            hashes = cohort[0].get("input_hashes")
            valid = valid and all(row.get("input_hashes" if index % 2 == 0 else "source_hashes_after") == hashes
                                  for index, row in enumerate(cohort))
            valid = valid and (expected_hashes is None or hashes == expected_hashes)
            if valid:
                # A pair is indivisible: both samples come from one uninterrupted
                # driver attempt, never from opposite sides of a reboot/retry.
                pairs[number] = {row["side"]: row["observation"]["times"] for row in cohort[1::2]}
                pair_attempts[number] = path.relative_to(directory).as_posix()
    complete = len(pairs) == 3
    result = {"complete_eligible_sequence": complete, "ledger_entries": sum(a["entries"] for a in attempts),
              "truncated_ledger": any(a["truncated"] for a in attempts), "attempts": attempts,
              "order": [f"{s[0].upper()}{p}" for p, s in ORDER], "qualification_passed": False,
              "completed_pairs": sorted(pairs), "pending_pairs": [n for n in (1, 2, 3) if n not in pairs],
              "pair_attempts": pair_attempts, "interrupted_or_ineligible_attempts_preserved": True}
    if not complete:
        result["reason"] = "Completed pairs stand; retry each incomplete pair in its preset order after native cleanup"
        return result
    result["pairs"] = pairs
    result["journeys"] = {}
    for journey in JOURNEYS:
        values = {side: [pairs[n][side][journey]["application_s"] for n in (1, 2, 3)]
                  for side in ("baseline", "candidate")}
        b, c = median(values["baseline"]), median(values["candidate"])
        common = {side: median(pairs[n][side][journey]["common_reply_seen_application_s"]
                               for n in (1, 2, 3)) for side in ("baseline", "candidate")}
        result["journeys"][journey] = {
            "baseline_median_application_s": b, "candidate_median_application_s": c,
            "median_reduction_pct": 100 * (1 - c / b), "passes_30_percent": 1 - c / b >= 0.30,
            "candidate_application_seconds": values["candidate"],
            "every_candidate_strictly_under_five": all(value < 5.0 for value in values["candidate"]),
            "per_pair_reduction_pct_informational": [100 * (1 - cv / bv) for bv, cv in
                                                      zip(values["baseline"], values["candidate"], strict=True)],
            "common_reply_seen_medians": common,
            "common_reply_seen_median_reduction_pct": 100 * (1 - common["candidate"] / common["baseline"]),
        }
    result["qualification_passed"] = all(
        item["passes_30_percent"] and item["every_candidate_strictly_under_five"]
        for item in result["journeys"].values()
    )
    return result


def prepare_attempt(directory, active, batch, settings, config_path, resume):
    if resume:
        saved = read(directory / "inputs.json")
        if any(saved.get(key) != value for key, value in settings.items()):
            raise RuntimeError("Cannot resume with different source/configuration pins; use a new batch")
    else:
        directory.mkdir()
        durable(directory / "inputs.json", {**settings, "driver_sha256": digest(Path(__file__)),
                                           "inputs_sha256": digest(config_path)})
    previous_bytes = active.read_bytes() if active.exists() else None
    try:
        previous = json.loads(previous_bytes) if previous_bytes is not None else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        if not resume:
            raise RuntimeError("Interrupted active marker requires native cleanup before resume") from None
        previous = {"batch": batch, "unreadable_marker_sha256": digest(active)}
    if previous is not None and (not resume or previous["batch"] != batch):
        raise RuntimeError("A different or still-active batch owns the native sequence")
    result = report(directory)
    if not result["pending_pairs"]:
        if previous is not None:
            active.unlink()
        return None, result
    number = max((int(path.name.split("-")[-1]) for path in directory.glob("attempt-*") if path.is_dir()), default=0) + 1
    attempt = directory / f"attempt-{number:04d}"
    attempt.mkdir()
    owner = {"batch": batch, "driver_pid": os.getpid(), "directory": str(directory),
             "attempt": number, "resumed": resume, "previous_owner": previous,
             "started_utc": datetime.now(timezone.utc).isoformat()}
    durable(attempt / "owner.json", owner)
    if previous_bytes is not None:
        with (attempt / "previous-active.bin").open("xb") as stream:
            stream.write(previous_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    if previous is None:
        durable(active, owner)
    else:
        replacement = attempt / "active-marker.json"
        durable(replacement, owner)
        os.replace(replacement, active)
    return attempt, result


def run(batch, config_path, resume=False):
    settings = pins(config_path)
    root = Path(settings["output_directory"])
    root.mkdir(exist_ok=True)
    directory = root / batch
    active = root / "active-batch.json"
    attempt, prior = prepare_attempt(directory, active, batch, settings, config_path, resume)
    if attempt is None:
        print(json.dumps(prior, indent=2))
        return 0 if prior["qualification_passed"] else 1
    attempt_number = int(attempt.name.split("-")[-1])
    for pair, side in ORDER:
        if pair in prior["completed_pairs"]:
            continue
        item = settings[side]
        label = batch + format(attempt_number, "x") + side[0] + str(pair)
        receipts = Path(item["directory"]) / "replay-runs" / label
        command = [settings["powershell"], "-NoProfile", "-NonInteractive", "-File", item["runner"],
                   "-CandidateDirectory", item["directory"], "-RunLabel", label,
                   "-PrivateParent", settings["private_parent"]]
        if item["exhaustive_switch"]:
            command.append("-Exhaustive")
        append(attempt, pair=pair, side=side, status="started", label=label, command=command,
               receipts=str(receipts), input_hashes=settings["files"], verification="exhaustive")
        code, eligible, observation, error = None, False, None, None
        try:
            if pins(config_path) != settings:
                raise RuntimeError("Frozen pair inputs changed after the batch started")
            with (attempt / (label + ".log")).open("xb") as output:
                completed = subprocess.run(command, cwd=config_path.parent, stdout=output, stderr=subprocess.STDOUT,
                                           env={**os.environ, "PYTEST_PLUGINS": "task40608_lab_observer"}, check=False)
                output.flush()
                os.fsync(output.fileno())
            code = completed.returncode
            if pins(config_path) != settings:
                raise RuntimeError("Frozen pair inputs changed during the original run")
            if code != 0:
                raise RuntimeError("Original exhaustive guarded runner failed")
            observation = inspect_run(receipts, side, item, settings)
            eligible = True
        except BaseException as failure:
            error = repr(failure)
        hashes = {path.relative_to(receipts).as_posix(): digest(path)
                  for path in receipts.rglob("*") if path.is_file()} if receipts.exists() else {}
        append(attempt, pair=pair, side=side, status="finished" if code is not None else "interrupted",
               code=code, eligible=eligible, error=error, observation=observation, artifact_hashes=hashes,
               source_hashes_after={name: digest(Path(name)) for name in settings["files"]})
        if not eligible:
            durable(attempt / "ineligible.json", report(directory))
            raise RuntimeError("Stopped; preserve this attempt and resume its incomplete pair after native cleanup")
    result = report(directory)
    durable(attempt / "summary.json", result)
    active.unlink()  # All six original runners and guard receipts completed positively.
    print(json.dumps(result, indent=2))
    return 0 if result["qualification_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", default="q1")
    parser.add_argument("--resume", action="store_true", help="Resume the same pins after original native-runner cleanup")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9]{2}", args.batch):
        parser.error("Use exactly two lower-case letters/digits, keeping retry labels short")
    config_path = args.config.resolve(strict=True)
    if args.command == "plan":
        print(json.dumps({"inputs": pins(config_path), "order": ORDER, "native_workload_executed": False}, indent=2))
    elif args.command == "report":
        print(json.dumps(report(Path(configuration(config_path)["output_directory"]) / args.batch), indent=2))
    else:
        raise SystemExit(run(args.batch, config_path, args.resume))
