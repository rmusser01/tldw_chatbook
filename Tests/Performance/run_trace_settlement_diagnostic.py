"""Diagnostic-only source-fenced launcher for the unchanged native probe.

Intended tracked path: Tests/Performance/run_trace_settlement_diagnostic.py.
This standalone draft is not installed and must not be launched without root's
serialized native slot. It imports no application or Tests module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


PLUGIN = "Tests.Performance.console_trace_settlement_witness"
NODE = "Tests/Performance/test_console_native_pause_probe.py::test_native_console_pause_probe"
HELPERS = (
    "Tests.private_profile",
    "Tests.real_profile_guard",
    "Tests.network_guard",
    "Tests.windows_private_fixture_runner",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root: Path) -> dict:
    """Retain every managed app/test Python source and actual source specs."""
    source_roots = (
        root / "tldw_chatbook",
        root / "Tests",
        root / "packages/tldw_profile_core/src/tldw_profile_core",
    )
    records, invalid = {}, []
    for directory in source_roots:
        if not directory.is_dir():
            invalid.append("missing_source_root:" + str(directory))
            continue
        for path in sorted(directory.rglob("*.py")):
            if not path.resolve().is_relative_to(root):
                invalid.append("foreign_managed_python_source:" + str(path))
            try:
                records[path.relative_to(root).as_posix()] = digest(path)
            except OSError as error:
                records[path.relative_to(root).as_posix()] = {
                    "error_category": type(error).__name__
                }
                invalid.append("unreadable_managed_python_source:" + str(path))

    origins = {}
    # Top-level spec lookup does not import the package. PathFinder resolves
    # nested helper/plugin files using the retained package search locations.
    tests = importlib.util.find_spec("Tests")
    search = tests.submodule_search_locations if tests is not None else None
    performance = (
        importlib.machinery.PathFinder.find_spec("Tests.Performance", search)
        if search is not None
        else None
    )
    specifications = {
        "tldw_chatbook": importlib.util.find_spec("tldw_chatbook"),
        "tldw_profile_core": importlib.util.find_spec("tldw_profile_core"),
        **{
            name: importlib.machinery.PathFinder.find_spec(name, search)
            if search is not None
            else None
            for name in HELPERS
        },
        PLUGIN: importlib.machinery.PathFinder.find_spec(
            PLUGIN, performance.submodule_search_locations
        )
        if performance is not None
        else None,
    }
    expected = {
        "tldw_chatbook": root / "tldw_chatbook/__init__.py",
        "tldw_profile_core": root
        / "packages/tldw_profile_core/src/tldw_profile_core/__init__.py",
        **{
            name: root / (name.replace(".", "/") + ".py") for name in (*HELPERS, PLUGIN)
        },
    }
    for name, spec in specifications.items():
        origin = spec.origin if spec is not None else None
        if not isinstance(origin, str) or not Path(origin).is_file():
            origins[name] = {"origin_missing": True}
            invalid.append("missing_installed_source:" + name)
            continue
        path = Path(origin).resolve()
        origins[name] = {
            "origin": str(path),
            "sha256": digest(path),
            "matches_managed_source": path == expected[name].resolve(),
        }
        if not origins[name]["matches_managed_source"]:
            invalid.append("foreign_installed_source:" + name)
    return {
        "python_executable": sys.executable,
        "python_executable_resolved": str(Path(sys.executable).resolve()),
        "python_executable_sha256": digest(Path(sys.executable)),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "python_flags": {
            "isolated": sys.flags.isolated,
            "utf8_mode": sys.flags.utf8_mode,
        },
        "managed_source_sha256": records,
        "actual_installed_source_specs": origins,
        "invalid_source_evidence": invalid,
    }


def child_source_valid(root: Path, before: dict, records: dict) -> bool:
    """Require actual child origins/hashes to match the pre-launch manifest."""
    if records.get("actual_child_installed_sources_unchanged") is not True:
        return False
    for edge in ("before", "after"):
        record = records.get("actual_child_installed_sources_" + edge)
        if not isinstance(record, dict):
            return False
        if (
            Path(record.get("python_executable", "")).resolve()
            != Path(sys.executable).resolve()
            or record.get("python_version") != sys.version
        ):
            return False
        for module in record.get("modules", {}).values():
            if not module.get("loaded"):
                continue  # Honest late/unloaded metadata, never an imported fallback.
            origin = module.get("origin")
            if not isinstance(origin, str):
                return False
            path = Path(origin).resolve()
            if not path.is_relative_to(root):
                return False
            if before["managed_source_sha256"].get(
                path.relative_to(root).as_posix()
            ) != module.get("sha256"):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.repo.resolve(), args.output_base.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path, result_path = (
        output.with_suffix(".source.json"),
        output.with_suffix(".json"),
    )
    xml_path, log_path = output.with_suffix(".xml"), output.with_suffix(".log")
    if any(path.exists() for path in (source_path, result_path, xml_path, log_path)):
        raise RuntimeError(
            "use a fresh diagnostic output base; preserve previous receipts"
        )
    before = snapshot(root)
    command = [
        sys.executable,
        "-m",
        "Tests.windows_private_fixture_runner",
        NODE,
        "-q",
        "--timeout=900",
        "--junitxml=" + str(xml_path),
    ]
    environment = os.environ.copy()
    plugins = [
        name for name in environment.get("PYTEST_PLUGINS", "").split(",") if name
    ]
    if PLUGIN not in plugins:
        plugins.append(PLUGIN)
    environment["PYTEST_PLUGINS"] = ",".join(plugins)
    environment["TLDW_PAUSE_PROBE_RESULT"] = str(result_path)
    environment["PYTHONUTF8"] = "1"
    status, launch_error, started, launched = None, None, time.monotonic(), False
    try:
        if before["invalid_source_evidence"]:
            raise RuntimeError(
                "diagnostic source/origin qualification failed before launch"
            )
        with log_path.open("w", encoding="utf-8") as output_log:
            launched = True
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=output_log,
                stderr=subprocess.STDOUT,
            )
            interruption = None
            while status is None:
                try:
                    status = process.wait()
                except KeyboardInterrupt as error:
                    interruption = error
            if interruption is not None:
                raise interruption  # Original process is positively reaped first.
    except BaseException as error:
        launch_error = type(error).__name__
        raise
    finally:
        after = snapshot(root)
        unchanged = before == after and not after["invalid_source_evidence"]
        child_sources = {}
        child_evidence_error = None
        try:
            observed = json.loads(result_path.read_text(encoding="utf-8"))
            diagnostic = observed["diagnostic_trace_settlement"]
            child_sources = {
                name: diagnostic[name]
                for name in (
                    "actual_child_installed_sources_before",
                    "actual_child_installed_sources_after",
                    "actual_child_installed_sources_unchanged",
                )
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            child_evidence_error = type(error).__name__
        source_valid = (
            unchanged
            and status is not None
            and child_source_valid(root, before, child_sources)
        )
        receipt = {
            "diagnostic_only": True,
            "native_launched": launched,
            "source_root": str(root),
            "source_before": before,
            "source_after": after,
            "source_unchanged": unchanged,
            "diagnostic_source_valid": source_valid,
            "actual_private_child_installed_sources": child_sources,
            "actual_private_child_source_evidence_error": child_evidence_error,
            "original_subprocess_command": command,
            "original_pytest_timeout_seconds": 900,
            "added_parent_timeout": None,
            "opt_in_plugin": PLUGIN,
            "plugin_reaches_original_private_child": True,
            "test_exit_code": status,
            "original_test_process_positively_reaped": status is not None,
            "launcher_error_category": launch_error,
            "elapsed_seconds_diagnostic_only": time.monotonic() - started,
            "original_three_os_budget_route_unmodified": True,
            "original_guards_sql_and_app_calls_unmodified": True,
        }
        source_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
        )
        if not source_valid:
            print(
                "Diagnostic invalid: source/origin drift or missing child source witness; retain .source.json and original failure artifacts."
            )
    return int(status) if source_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
