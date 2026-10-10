"""No-App source/process qualification for the whole startup liveness pair.

This exercises real selected interpreters/helpers and independent source manifest
enumeration. The manifest mutation fixture contains source text only; it does
not impersonate or execute an application or native producer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--receipt-root", type=Path)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--driver-pid", type=int)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()
    repo = args.repo.absolute()
    driver = Path(__file__).with_name("run_console_startup_liveness.py")
    child = Path(__file__).with_name("console_startup_liveness_child.py")
    helpers = runpy.run_path(str(driver), run_name="startup_source_process_qualifier")
    helpers["configure_managed_source_roots"](repo)
    assert not any(
        name == "tldw_chatbook" or name.startswith("tldw_chatbook.")
        for name in sys.modules
    ), "Tiny qualifier must not import the App"
    source_paths = (driver, child, Path(__file__))
    if args.child:
        assert args.driver_pid is not None and args.receipt is not None
        assert not args.receipt.exists()
        origins = {}
        for name, expected in (
            ("tldw_chatbook", repo / "tldw_chatbook/__init__.py"),
            ("Tests", repo / "Tests/__init__.py"),
            (
                "tldw_profile_core",
                repo / "packages/tldw_profile_core/src/tldw_profile_core/__init__.py",
            ),
        ):
            spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
            assert spec is not None and spec.origin is not None
            actual = Path(spec.origin).resolve(strict=True)
            assert actual == expected.resolve(strict=True)
            origins[name] = {
                "path": str(actual),
                "raw_sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
                "spec_only_not_executed": True,
            }
        record = {
            "process_identity": helpers["child_process_receipt"](args.driver_pid),
            "written_at": time.perf_counter(),
            "app_imported": False,
            "managed_spec_origins": origins,
            "actual_driver_helper_code_file": str(
                Path(helpers["source_snapshot"].__code__.co_filename).resolve()
            ),
            "raw_helper_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths
            },
        }
        args.receipt.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise SystemExit(args.exit_code)
    assert args.python is not None and args.receipt_root is not None
    receipts = args.receipt_root.absolute()
    assert not receipts.exists(), "Never overwrite prior process/source qualification"
    receipts.mkdir(mode=0o700)
    before = helpers["source_snapshot"](repo, source_paths)
    expected = helpers["selected_executable_identities"](args.python)
    report = {
        "complete": False,
        "launches": [],
        "app_imported": False,
        "source_before": before,
        "expected_executables": expected,
    }
    try:
        # New/deleted/changed source detection is checked on private source-text
        # fixtures, never by editing an installed application/guard file.
        fixture = receipts / "source-manifest-algorithm-fixture"
        for relative in (
            "tldw_chatbook",
            "Tests",
            "packages/tldw_profile_core/src/tldw_profile_core",
        ):
            (fixture / relative).mkdir(parents=True)
            (fixture / relative / "__init__.py").write_text(
                "# source enumeration fixture\n", encoding="utf-8"
            )
        original = helpers["source_snapshot"](fixture, source_paths)
        added = fixture / "Tests/new_source.py"
        added.write_text("# newly enumerated source\n", encoding="utf-8")
        addition_seen = helpers["source_snapshot"](fixture, source_paths) != original
        added.unlink()
        deletion_seen = helpers["source_snapshot"](fixture, source_paths) == original
        changed = fixture / "Tests/__init__.py"
        changed.write_text("# changed source bytes\n", encoding="utf-8")
        changed_seen = helpers["source_snapshot"](fixture, source_paths) != original
        changed.write_text("# source enumeration fixture\n", encoding="utf-8")
        assert addition_seen and deletion_seen and changed_seen
        assert helpers["source_snapshot"](fixture, source_paths) == original
        report["independent_enumeration_controls"] = {
            "added_file_observed": addition_seen,
            "deleted_addition_removed": deletion_seen,
            "changed_bytes_observed": changed_seen,
            "fixture_restored": True,
            "fixture_source_never_executed": True,
        }
        for requested_status in (0, 37):
            receipt = receipts / f"child-exit-{requested_status}.json"
            command = [
                str(args.python),
                "-I",
                "-X",
                "utf8",
                str(Path(__file__)),
                "--child",
                "--repo",
                str(repo),
                "--driver-pid",
                str(os.getpid()),
                "--receipt",
                str(receipt),
                "--exit-code",
                str(requested_status),
            ]
            spawned = time.perf_counter()
            with (receipts / f"child-exit-{requested_status}.log").open(
                "w", encoding="utf-8"
            ) as output:
                process = subprocess.Popen(
                    command, cwd=repo, stdout=output, stderr=subprocess.STDOUT
                )
                try:
                    status = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise
            exited = time.perf_counter()
            assert process.poll() is not None and status == requested_status
            actual = json.loads(receipt.read_text(encoding="utf-8"))
            route = helpers["assert_child_process_ownership"](
                actual["process_identity"], process.pid, os.getpid(), expected
            )
            assert spawned <= actual["written_at"] <= exited
            assert actual["app_imported"] is False
            assert actual["actual_driver_helper_code_file"] == str(driver.resolve())
            assert actual["raw_helper_sha256"] == {
                path.name: row["raw_sha256"]
                for path in source_paths
                for row in (before["probe_helpers"][path.name],)
            }
            rejected = []
            for field in (
                "parent_pid",
                "driver_pid",
                "executable_identity",
                "base_executable_identity",
            ):
                foreign = dict(actual["process_identity"])
                foreign[field] = (
                    -1
                    if field.endswith("pid")
                    else {"path": "foreign", "sha256": "foreign"}
                )
                try:
                    helpers["assert_child_process_ownership"](
                        foreign, process.pid, os.getpid(), expected
                    )
                except AssertionError:
                    rejected.append(field)
            assert rejected == [
                "parent_pid",
                "driver_pid",
                "executable_identity",
                "base_executable_identity",
            ]
            report["launches"].append(
                {
                    "popen_pid": process.pid,
                    "identity": actual,
                    "process_chain": route,
                    "status": status,
                    "requested_status": requested_status,
                    "physically_exited": True,
                    "foreign_receipt_fields_rejected": rejected,
                }
            )
        assert helpers["selected_executable_identities"](args.python) == expected
        report["complete"] = True
    finally:
        after = helpers["source_snapshot"](repo, source_paths)
        report.update(source_after=after, source_unchanged=before == after)
        (receipts / "startup-liveness-qualification.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        assert before == after


if __name__ == "__main__":
    main()
