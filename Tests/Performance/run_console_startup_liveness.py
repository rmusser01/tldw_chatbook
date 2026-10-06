"""Cold/warm original-App startup liveness with private process/source receipts.

Both launches have one persisted private profile and separate interpreters.
Native I/O counts belong to a separately qualified diagnostic observer; this
liveness route installs no profiler and claims no complete kernel I/O coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time


def executable_identity(path):
    """Record a selected executable's resolved spelling and actual bytes."""
    actual = Path(path).resolve(strict=True)
    assert actual.is_file(), "An executable identity requires an existing file"
    return {
        "path": os.path.normcase(str(actual)),
        "sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
    }


def selected_executable_identities(python):
    """Select the base executable from the real venv configuration, if present."""
    python = Path(python).absolute()
    selected = executable_identity(python)
    candidates = (python.parent / "pyvenv.cfg", python.parent.parent / "pyvenv.cfg")
    config = next((path for path in candidates if path.is_file()), None)
    if config is None:
        base = python
    else:
        values = {}
        for line in config.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key.strip().lower()] = value.strip()
        if os.name == "nt":
            # The 3.12 VENV_REDIRECT route uses pyvenv.cfg home/python.exe,
            # rather than the diagnostic 'executable' configuration entry.
            base = Path(values["home"]) / python.name
        else:
            base = Path(values["executable"])
    return {
        "launched": selected,
        "base": executable_identity(base),
        "venv_configuration": None if config is None else str(config.absolute()),
        "venv_configuration_sha256": None
        if config is None
        else hashlib.sha256(config.read_bytes()).hexdigest(),
    }


def child_process_receipt(driver_pid):
    """Capture the real interpreter and its still-live immediate parent."""
    return {
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "driver_pid": driver_pid,
        "sys_executable": sys.executable,
        "base_executable": sys._base_executable,
        "executable_identity": executable_identity(sys.executable),
        "base_executable_identity": executable_identity(sys._base_executable),
    }


def assert_child_process_ownership(actual, popen_pid, driver_pid, expected):
    """Accept a direct interpreter or the exact Windows redirector child chain."""
    assert actual["driver_pid"] == driver_pid
    assert actual["executable_identity"] == expected["launched"]
    assert actual["base_executable_identity"] == expected["base"]
    if actual["pid"] == popen_pid:
        assert (
            actual["parent_pid"] == driver_pid
        ), "Direct interpreter has a foreign parent"
        return "driver -> interpreter"
    assert os.name == "nt", "Only the qualified Windows redirector may add a process"
    assert expected["venv_configuration"] is not None
    assert (
        actual["parent_pid"] == popen_pid
    ), "Interpreter is not the exact launched redirector's child"
    return "driver -> launched Windows venv redirector -> interpreter"


# Registered TASK-34404 AC11 outcome bounds; original App/Pilot and process
# deadlines stay distinct from these usability acceptance gates.
MAX_COLD_USABLE_SECONDS = 15.0
MAX_WARM_USABLE_SECONDS = 10.0
MAX_MOUNTED_HEARTBEAT_GAP_SECONDS = 0.200
MAX_ORIGINAL_KEY_DELIVERY_SECONDS = 0.500
SOURCE_PREFIXES = ("tldw_chatbook", "Tests", "tldw_profile_core")


def configure_managed_source_roots(repo):
    """Select checkout code explicitly even under the required -I process."""
    sys.path[:0] = [str(repo), str(repo / "packages/tldw_profile_core/src")]


def source_snapshot(repo, helper_paths):
    """Independently enumerate every managed Python namespace for each call."""
    source_roots = (
        repo / "tldw_chatbook",
        repo / "Tests",
        repo / "packages/tldw_profile_core/src/tldw_profile_core",
    )
    records = {}
    for directory in source_roots:
        assert directory.is_dir(), ("missing managed source root", str(directory))
        for path in sorted(directory.rglob("*.py")):
            assert path.resolve(strict=True).is_relative_to(
                repo.resolve(strict=True)
            ), ("foreign managed source", str(path))
            records[path.relative_to(repo).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    helpers = {}
    for path in helper_paths:
        actual = path.resolve(strict=True)
        helpers[path.name] = {
            "path": str(actual),
            "raw_sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
        }
    return {
        "managed_python_raw_sha256": records,
        "probe_helpers": helpers,
        "coverage": "All Python source files independently enumerated under app/Tests/profile_core; dependencies, resources and OS kernel I/O are not this source manifest",
    }


def _namespace_source_receipt(name, module, repo):
    """Accept only a real managed PEP 420 namespace's exact metadata origins."""
    from importlib._bootstrap_external import _NamespacePath
    from importlib.machinery import ModuleSpec, NamespaceLoader
    from types import ModuleType

    assert type(module) is ModuleType, ("non-module namespace", name)
    fields = vars(module)
    spec = fields.get("__spec__")
    assert type(spec) is ModuleSpec, ("missing namespace spec", name)
    loader = fields.get("__loader__")
    assert type(loader) is NamespaceLoader, ("non-namespace loader", name)
    locations = fields.get("__path__")
    assert type(locations) is _NamespacePath, ("non-namespace paths", name)
    assert spec.loader is loader and loader._path is locations
    assert spec.submodule_search_locations is locations
    assert spec.name == name and spec.origin is None and not spec.has_location
    assert spec.cached is None
    assert fields.get("__name__") == name and fields.get("__package__") == name
    assert fields.get("__file__") is None and fields.get("__cached__") is None
    assert fields.get("__doc__") is None
    canonical = {
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "__path__",
    }
    for attribute, value in fields.items():
        if attribute in canonical:
            continue
        child_name = name + "." + attribute
        assert (
            type(value) is ModuleType
            and sys.modules.get(child_name) is value
            and vars(value).get("__name__") == child_name
        ), ("unexpected namespace body", name, attribute)
    actual_root = repo.resolve(strict=True)
    base = (
        actual_root / "packages/tldw_profile_core/src"
        if name == "tldw_profile_core" or name.startswith("tldw_profile_core.")
        else actual_root
    )
    expected = (base / Path(*name.split("."))).resolve(strict=True)
    assert expected.is_relative_to(actual_root) and expected.is_dir()
    # Read the stock path record without triggering a lazy search-path callback.
    recorded = vars(locations).get("_path")
    # Namespace provenance requires concrete stock containers and path strings.
    assert type(recorded) is list and all(type(path) is str for path in recorded)  # noqa: E721
    actual_locations = tuple(Path(path).resolve(strict=True) for path in recorded)
    assert actual_locations == (expected,), ("foreign namespace source", name)
    assert not (expected / "__init__.py").exists(), ("namespace has file body", name)
    return {
        "source_kind": "namespace_package",
        "namespace_search_locations": [expected.relative_to(actual_root).as_posix()],
        "module_actor_id": id(module),
        "spec_actor_id": id(spec),
        "loader_actor_id": id(loader),
        "actual_imported_module": True,
    }


def loaded_source_receipt(repo):
    """Record actual imported managed namespaces and refuse foreign origins."""
    loaded, origins = {}, {}
    actual_root = repo.resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if not any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in SOURCE_PREFIXES
        ):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            origins[name] = _namespace_source_receipt(name, module, actual_root)
            continue
        path = Path(filename).resolve(strict=True)
        assert path.is_relative_to(actual_root), ("foreign loaded source", name)
        key = path.relative_to(actual_root).as_posix()
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        loaded[key] = value
        origins[name] = {
            "source_relative_path": key,
            "raw_sha256": value,
            "module_actor_id": id(module),
            "actual_imported_module": True,
        }
    return loaded, origins


def observed_callable_source(function, repo):
    """Retain the real key route's defining code/file metadata, without tracing."""
    import types

    assert isinstance(function, types.FunctionType)
    code = function.__code__
    filename = Path(code.co_filename).resolve(strict=True)
    assert filename.is_relative_to(repo.resolve(strict=True))
    namespace = sys.modules.get(function.__module__)
    assert namespace is not None and function.__globals__ is vars(namespace)
    return {
        "module": function.__module__,
        "qualified_name": function.__qualname__,
        "code_name": code.co_name,
        "first_line": code.co_firstlineno,
        "source_relative_path": filename.relative_to(
            repo.resolve(strict=True)
        ).as_posix(),
        "raw_sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
        "function_actor_id": id(function),
        "actual_defining_code": True,
    }


def liveness_budget_result(result):
    """Evaluate pre-registered bounds only after original functional shutdown."""
    import math

    assert (
        result.get("mode", "liveness") == "liveness"
        and not result.get("diagnostic_only", False)
        and result.get("budget_acceptance_eligible", True) is True
    ), "Diagnostic timing cannot qualify original startup budgets"
    input_result = result["usable_input"]
    elapsed = input_result["seconds_from_parent_spawn"]
    key_delay = input_result["accepted"] - result["key_posted"]
    posted = result["key_posted"]
    # Clip crossing intervals to the observed mounted input window. Keep its
    # first interval; do not attribute earlier construction or later diagnostic
    # SQL reads to the mounted-loop gate.
    window_end = result["mounted_heartbeat_window_end"]
    mounted_intervals = [
        row
        for row in result["heartbeat_intervals"]
        if row["entered"] < window_end and row["exited"] > posted
    ]
    assert mounted_intervals, "No continuous mounted-loop heartbeat witness"
    heartbeat_gap = max(
        max(0, min(row["exited"], window_end) - max(row["entered"], posted) - 0.02)
        for row in mounted_intervals
    )
    assert all(
        math.isfinite(value) and value >= 0
        for value in (elapsed, key_delay, heartbeat_gap)
    )
    limit = (
        MAX_COLD_USABLE_SECONDS
        if result["launch"] == "cold"
        else MAX_WARM_USABLE_SECONDS
    )
    return {
        "usable_seconds": elapsed,
        "usable_limit_seconds": limit,
        "usable_within_limit": elapsed <= limit,
        "key_delivery_upper_bound_seconds": key_delay,
        "key_delivery_limit_seconds": MAX_ORIGINAL_KEY_DELIVERY_SECONDS,
        "key_delivery_within_limit": True
        if key_delay <= MAX_ORIGINAL_KEY_DELIVERY_SECONDS
        else None,
        "key_delivery_coverage": "Pilot return upper bound; a larger value does not prove a slow original key body",
        "key_body_entry_observer_attached": False,
        "mounted_heartbeat_max_gap_seconds": heartbeat_gap,
        "mounted_heartbeat_limit_seconds": MAX_MOUNTED_HEARTBEAT_GAP_SECONDS,
        "mounted_heartbeat_within_limit": heartbeat_gap
        <= MAX_MOUNTED_HEARTBEAT_GAP_SECONDS,
        "mounted_heartbeat_intervals": len(mounted_intervals),
    }


def timing_diagnostic_result(result, expected_sha256):
    """Accept source/retirement metadata only; never evaluate startup budgets."""
    assert result["mode"] == "timing" and result["diagnostic_only"] is True
    assert (
        result["budget_acceptance_eligible"] is False and result["budgets_pass"] is None
    )
    assert result.get("startup_timing_retirement_error_type") is None
    timing = result["startup_timing"]
    assert timing["diagnostic_only"] is True
    assert (
        timing["budget_acceptance_eligible"] is False and timing["budgets_pass"] is None
    )
    for name in (
        "complete",
        "source_current",
        "monitoring_global_zero",
        "monitoring_masks_owned",
        "monitoring_callbacks_owned",
        "monitoring_masks_cleared_while_active",
        "monitoring_tool_freed",
    ):
        assert timing[name] is True, ("Incomplete startup timing witness", name)
    path = "Tests/Performance/console_startup_timing_witness.py"
    helper = result["executed_probe_helpers"]["timing"]
    assert helper["raw_sha256"] == expected_sha256
    assert helper["start_callable"]["source_relative_path"] == path
    assert helper["start_callable"]["raw_sha256"] == expected_sha256
    assert helper["start_callable"]["actual_defining_code"] is True
    assert result["loaded_source_sha256"][path] == expected_sha256
    return {
        "diagnostic_only": True,
        "budget_acceptance_eligible": False,
        "budgets_pass": None,
        "timing_complete": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("liveness", "timing"), default="liveness")
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    repo = args.repo.absolute()
    pair_root = args.pair_root.absolute()
    receipts = args.receipt_root.absolute()
    assert not pair_root.exists(), "A cold pair must use a new profile"
    assert (
        not receipts.exists()
    ), "Use a fresh receipts directory; preserve every prior failed receipt"
    assert (
        os.environ.get("TLDW_REQUIRE_ELEVATED_CUSTODY") != "1"
    ), "Startup pairs belong on ordinary hosts; preserve elevated custody jobs"
    receipts.mkdir(mode=0o700)
    expected_executables = selected_executable_identities(args.python)
    git_before = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if args.expected_commit is not None:
        assert (
            git_before == args.expected_commit
        ), "Checkout differs from selected CI HEAD"
    configure_managed_source_roots(repo)
    from Tests import real_profile_guard
    from Tests.windows_private_fixture_runner import user_fixture_default_owner

    real_profile_guard.install()
    with user_fixture_default_owner():
        pair_root.mkdir(mode=0o700)
        for name in ("home", "config", "data"):
            (pair_root / name).mkdir(mode=0o700)
        selector = pair_root / "config" / "config.toml"
        # Same existing deterministic startup fixture values; written only cold.
        # Ordinary timers/GC/scheduler/seeders remain enabled.
        selector.write_text(
            '[general]\nusers_name="startup-census"\n'
            "[first_run]\nsetup_completed=true\n"
            "[_first_run]\nsetup_completed=true\n"
            "[splash_screen]\nenabled=false\n"
            "[api_settings.openai]\n"
            'api_key="sk-census-000000000000000000000000000000000000"\n',
            encoding="utf-8",
        )
        selector.chmod(0o600)
    environment = dict(os.environ)
    environment.update(
        HOME=str(pair_root / "home"),
        USERPROFILE=str(pair_root / "home"),
        XDG_CONFIG_HOME=str(pair_root / "config"),
        XDG_DATA_HOME=str(pair_root / "data"),
        TLDW_CONFIG_PATH=str(selector),
        TLDW_STARTUP_PROFILE_ROOT=str(pair_root),
        TLDW_TEST_MODE="1",
        PYTEST_CURRENT_TEST="console_startup_census",
        PYTHONUTF8="1",
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
        TLDW_STARTUP_PATH_HMAC_KEY=secrets.token_hex(32),
        PYTHONPATH=os.pathsep.join((str(Path(__file__).parent), str(repo))),
    )
    helper_paths = (
        Path(__file__),
        Path(__file__).with_name("console_startup_liveness_child.py"),
    )
    if args.mode == "timing":
        helper_paths += (Path(__file__).with_name("console_startup_timing_witness.py"),)
    before = source_snapshot(repo, helper_paths)
    (
        receipts
        / (
            "startup-timing.source-before.json"
            if args.mode == "timing"
            else "startup-liveness.source-before.json"
        )
    ).write_text(json.dumps(before, indent=2), encoding="utf-8")
    pair = {
        "mode": args.mode,
        "diagnostic_only": args.mode == "io",
        "launches": [],
        "complete": False,
        "driver_pid": os.getpid(),
        "git_head_before": git_before,
        "expected_executables": expected_executables,
        "kernel_audit_coverage": "Not attached by this draft; facade/API/audit coverage is explicit",
        "os_page_cache": "Uncontrolled; cold means pristine application profile/new process",
        "forced_timeout_descendant_retirement_proven": False,
        "timeout_retirement_coverage": "Original Popen kill/wait observes the launched process only; Windows redirector interpreter/descendant retirement is unqualified on forced timeout. Normal exits and actual parent chain are verified.",
    }
    if args.mode == "timing":
        pair.update(
            diagnostic_only=True, budget_acceptance_eligible=False, budgets_pass=None
        )
    try:
        for launch in ("cold", "warm"):
            child_receipt = receipts / f"startup-{args.mode}-{launch}.json"
            log = receipts / f"startup-{args.mode}-{launch}.log"
            profile_before = hashlib.sha256(selector.read_bytes()).hexdigest()
            spawned = time.perf_counter()
            command = [
                str(args.python),
                "-I",
                "-X",
                "utf8",
                str(Path(__file__).with_name("console_startup_liveness_child.py")),
                "--repo",
                str(repo),
                "--receipt",
                str(child_receipt),
                "--mode",
                args.mode,
                "--launch",
                launch,
                "--driver-pid",
                str(os.getpid()),
                "--parent-spawn",
                str(spawned),
                "--expected-driver-sha256",
                before["probe_helpers"][Path(__file__).name]["raw_sha256"],
                "--expected-child-sha256",
                before["probe_helpers"]["console_startup_liveness_child.py"][
                    "raw_sha256"
                ],
            ]
            if args.mode == "timing":
                command.extend(
                    (
                        "--expected-timing-sha256",
                        before["probe_helpers"]["console_startup_timing_witness.py"][
                            "raw_sha256"
                        ],
                    )
                )
            with log.open("w", encoding="utf-8") as output:
                child = subprocess.Popen(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                forced = False
                try:
                    status = child.wait(timeout=900)  # Existing private child bound.
                except subprocess.TimeoutExpired:
                    forced = True
                    child.kill()
                    status = child.wait()
            row = {
                "launch": launch,
                "pid": child.pid,
                "parent_spawn": spawned,
                "parent_exit_observed": time.perf_counter(),
                "physically_exited": child.poll() is not None,
                "physically_exited_coverage": "Launched Popen process; normal interpreter exit additionally requires its completed receipt and verified actual process ancestry",
                "exit_code": status,
                "deadline_forced_exit": forced,
                "selected_config_sha256_before": profile_before,
            }
            pair["launches"].append(row)
            assert (
                not forced and status == 0
            ), "Startup child failed; no automatic retry/warm spawn"
            actual = json.loads(child_receipt.read_text(encoding="utf-8"))
            row["process_chain"] = assert_child_process_ownership(
                actual["process_identity"], child.pid, os.getpid(), expected_executables
            )
            assert actual["pid"] == actual["process_identity"]["pid"]
            assert actual["launch"] == launch and actual["complete"]
            assert selected_executable_identities(args.python) == expected_executables
            assert actual["selected_config_sha256_before"] == profile_before
            assert actual["original_app_shutdown_completed"]
            for path, value in actual["loaded_source_sha256"].items():
                assert before["managed_python_raw_sha256"].get(path) == value, (
                    "loaded source differs from selected checkout",
                    path,
                )
            row["result"] = actual
            if args.mode == "timing":
                row["timing_diagnostic"] = timing_diagnostic_result(
                    actual,
                    before["probe_helpers"]["console_startup_timing_witness.py"][
                        "raw_sha256"
                    ],
                )
            else:
                row["budgets"] = liveness_budget_result(actual)
            # A second Popen creates a new interpreter only after the first has
            # exited. The profile is not reset/reseeded and config is not rewritten.
        assert (
            pair["launches"][1]["parent_spawn"]
            > pair["launches"][0]["parent_exit_observed"]
        )
        assert (
            pair["launches"][1]["result"]["child_entered"]
            > pair["launches"][0]["result"]["exited_after_original_shutdown"]
        )
        assert (
            pair["launches"][1]["result"]["selected_config_sha256_before"]
            == pair["launches"][0]["result"]["selected_config_sha256_after"]
        ), "Warm launch did not retain cold selected config"
        assert (
            pair["launches"][1]["result"]["durable_builtin_state"]
            == pair["launches"][0]["result"]["durable_builtin_state"]
        ), "Warm durable builtin state changed"
        pair["complete"] = True
        if args.mode == "timing":
            pair.update(budget_failures=None, budgets_pass=None)
        else:
            pair["budget_failures"] = [
                row["launch"] + ":" + key
                for row in pair["launches"]
                for key in ("usable_within_limit", "mounted_heartbeat_within_limit")
                if row["budgets"][key] is not True
            ]
            pair["budgets_pass"] = not pair["budget_failures"]
        pair["exact_key_gate_coverage"] = (
            "Not attached; exact original insertion timing remains a separate qualified observer requirement"
        )
    finally:
        # Enumerate again independently, including newly added/deleted files.
        after = source_snapshot(repo, helper_paths)
        git_after = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        pair.update(
            source_before=before,
            source_after=after,
            source_unchanged=before == after,
            git_head_after=git_after,
            git_head_unchanged=git_before == git_after,
        )
        (receipts / f"startup-{args.mode}-pair.json").write_text(
            json.dumps(pair, indent=2), encoding="utf-8"
        )
        assert before == after, "Source changed during startup pair"
        assert git_before == git_after, "Git HEAD changed during startup pair"
    if args.mode == "liveness":
        assert pair["budgets_pass"], pair["budget_failures"]


if __name__ == "__main__":
    main()
