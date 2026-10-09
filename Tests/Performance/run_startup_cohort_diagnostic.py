"""Opt-in original control-prefix diagnostic; no App import or acceptance changes."""

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
            name: importlib.machinery.PathFinder.find_spec(
                name,
                performance.submodule_search_locations if name == BASE else search,
            )
            if (performance is not None if name == BASE else search is not None)
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
    workflow_path = root / ".github/workflows/console-pause-native-evidence.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    original = workflow_prefix(workflow)
    if original != PREFIX:
        invalid.append("original_workflow_control_prefix_changed")
    return {
        "original_workflow_sha256": digest(workflow_path),
        "original_workflow_control_prefix": original,
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


PLUGIN = "Tests.Performance.console_startup_cohort_witness"
BASE = "Tests.Performance.console_startup_cohort_code_local"
HELPERS = (
    "Tests.private_profile",
    "Tests.real_profile_guard",
    "Tests.network_guard",
    "Tests.windows_private_fixture_runner",
    BASE,
)
PREFIX = (
    "Tests/Utils/test_windows_private_fixture_runner.py",
    "Tests/test_private_profile_failure_report.py",
    "Tests/UI/test_console_shared_context_policy.py",
    "Tests/Backup_Recovery/test_mcp_source_metadata_repetition.py",
    "Tests/Backup_Recovery/test_nested_repository_check_repetition.py",
    "Tests/Backup_Recovery/test_controller_finite_db_interval.py",
    "Tests/Backup_Recovery/test_controller_agent_source_intervals.py",
    "Tests/Backup_Recovery/test_console_settings_worker_lifetime.py",
    "Tests/Chat/test_console_async_mcp_snapshot.py",
    "Tests/Chat/test_console_snapshot_first_import.py",
    "Tests/Chat/test_console_local_review_hook.py",
    "Tests/Tools/test_workspace_root_native_identity.py",
    "Tests/Tools/test_workspace_root_pin.py",
    "Tests/Tools/test_worker_import_closure.py",
    "Tests/MCP/test_console_snapshot_source_contracts.py",
    "Tests/UI/test_character_context_async_ownership.py",
    "Tests/UI/test_console_character_context.py",
    "Tests/UI/test_console_agent_controller.py",
    "Tests/UI/test_console_native_send_pump.py",
    "Tests/UI/test_console_command_origin_chat.py",
    "Tests/UI/test_console_video_send_freeze.py",
    "Tests/Console/test_console_command_draft.py",
    "Tests/UI/test_home_console_resume.py",
    "Tests/UI/test_console_shutdown_recompose.py",
    "Tests/UI/test_chat_screen_resume_handoff_registration.py",
    "Tests/Chat/test_console_generation_actions.py::test_generate_image_handler_restores_draft_when_batch_raises",
    "Tests/UI/test_console_hook_refresh_lifetime.py",
    "Tests/Chat/test_console_hook_admission.py",
    "Tests/UI/test_console_hook_review_send_freeze.py",
    "Tests/UI/test_console_checked_display_scope.py",
    "Tests/Backup_Recovery/test_startup_initializing_publication.py",
    "Tests/Backup_Recovery/test_visual_identity_observation_budget.py",
    "Tests/Backup_Recovery/test_repository_coordinator_io.py",
    "Tests/Backup_Recovery/test_storage_coordinator_native_io.py",
    "Tests/Backup_Recovery/test_storage_scope_publication.py",
    "Tests/Backup_Recovery/test_related_path_admission.py",
    "Tests/Backup_Recovery/test_startup_readmission_continuity.py",
    "Tests/Backup_Recovery/test_console_presentation_cadence.py",
    "Tests/Backup_Recovery/test_console_live_empty_fleet.py",
    "Tests/MCP/test_external_catalog_worker_ownership.py::test_standard_catalog_reads_one_fresh_bundle_off_loop",
    "Tests/MCP/test_external_catalog_worker_ownership.py::test_accepted_catalog_retains_producer_and_native_custody[False]",
)


def workflow_prefix(workflow: str) -> tuple[str, ...]:
    import shlex

    candidates = [
        line.strip()
        for line in workflow.splitlines()
        if "run: python -m Tests.windows_private_fixture_runner Tests/Utils/test_windows_private_fixture_runner.py "
        in line
    ]
    if len(candidates) != 1:
        raise RuntimeError("original control command missing or ambiguous")
    args = shlex.split(candidates[0].split("run: ", 1)[1].split(" -q ", 1)[0])[3:]
    catalog = "Tests/MCP/test_external_catalog_worker_ownership.py"
    index = args.index(catalog)
    return tuple(
        args[:index]
        + [
            catalog + "::test_standard_catalog_reads_one_fresh_bundle_off_loop",
            catalog
            + "::test_accepted_catalog_retains_producer_and_native_custody[False]",
        ]
    )


def control_sources_valid(root: Path, before: dict, observed: dict) -> bool:
    """Check only observed origins; never import a fallback source."""
    for edge in ("before", "after"):
        sources = observed.get("actual_control_process_installed_sources_" + edge)
        if not isinstance(sources, dict):
            return False
        if (
            Path(sources.get("python_executable", "")).resolve()
            != Path(sys.executable).resolve()
            or sources.get("python_version") != sys.version
        ):
            return False
        for source in sources.get("modules", {}).values():
            if not source.get("loaded"):
                continue
            origin = source.get("origin")
            if not isinstance(origin, str):
                return False
            path = Path(origin).resolve()
            if not path.is_relative_to(root) or before["managed_source_sha256"].get(
                path.relative_to(root).as_posix()
            ) != source.get("sha256"):
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
    paths = {
        suffix: output.with_suffix(suffix)
        for suffix in (".source.json", ".witness.json", ".xml", ".log")
    }
    if any(path.exists() for path in paths.values()):
        raise RuntimeError("use a fresh output base; preserve all earlier receipts")
    before = snapshot(root)
    command = [
        sys.executable,
        "-m",
        "Tests.windows_private_fixture_runner",
        "-p",
        PLUGIN,
        *PREFIX,
        "-q",
        "--timeout=180",
        "--junitxml=" + str(paths[".xml"]),
    ]
    environment = os.environ.copy()
    if PLUGIN in environment.get("PYTEST_PLUGINS", "").split(","):
        raise RuntimeError(
            "the control-process-only plugin must not be inherited by private children"
        )
    environment.update(
        TLDW_CI_WITNESS_REPO=str(root),
        TLDW_CI_WITNESS_RECEIPT=str(paths[".witness.json"]),
        PYTHONUTF8="1",
    )
    status, error_category, launched = None, None, False
    started = time.monotonic()
    try:
        if before["invalid_source_evidence"]:
            raise RuntimeError(
                "diagnostic source origin or original workflow qualification failed"
            )
        with paths[".log"].open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            launched = True
            interruption = None
            while status is None:
                try:
                    status = process.wait()
                except KeyboardInterrupt as error:
                    interruption = error
            if interruption is not None:
                raise interruption  # Original control process has been positively reaped.
    except BaseException as error:
        error_category = type(error).__name__
        raise
    finally:
        after = snapshot(root)
        unchanged = before == after and not after["invalid_source_evidence"]
        observed, observer_valid, witness_error = {}, False, None
        try:
            observed = json.loads(paths[".witness.json"].read_text(encoding="utf-8"))
            observer = observed["observer"]
            observer_valid = (
                observed["source_unchanged"] is True
                and observer["original_callable_identities_unchanged"] is True
                and observer["monitoring_global_events_zero"] is True
                and observer["monitoring_owned_and_restored"] is True
                and not observer["source_mismatches"]
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            witness_error = type(error).__name__
        valid = (
            unchanged
            and status is not None
            and observer_valid
            and control_sources_valid(root, before, observed)
        )
        receipt = {
            "diagnostic_only": True,
            "native_launched": launched,
            "source_before": before,
            "source_after": after,
            "source_unchanged": unchanged,
            "diagnostic_source_valid": valid,
            "observer_source_valid": observer_valid,
            "observer_evidence_error_category": witness_error,
            "original_subprocess_command": command,
            "original_control_prefix": PREFIX,
            "original_pytest_timeout_seconds": 180,
            "added_parent_timeout": None,
            "test_exit_code": status,
            "original_test_process_positively_reaped": status is not None,
            "launcher_error_category": error_category,
            "elapsed_seconds_diagnostic_only": time.monotonic() - started,
            "opt_in_code_local_plugin": PLUGIN,
            "plugin_passed_only_to_shared_control_process": True,
            "exceptional_selected_completions_remain_explicit_gaps": True,
            "original_six_acceptance_jobs_unchanged": True,
            "original_guards_app_sql_assertions_deadlines_unchanged": True,
        }
        paths[".source.json"].write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
        )
    return int(status) if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
