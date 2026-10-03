"""Finite, deterministic idle receipt/census contracts; no performance run."""

import asyncio
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest


def harness():
    path = (
        Path(__file__).parents[2]
        / "Helper_Scripts/Benchmarks/backup_admission_idle_benchmark.py"
    )
    assert path.is_file(), "separate real-idle harness is missing"
    spec = importlib.util.spec_from_file_location("idle_probe_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receipt(side, pair, opens):
    task = {"identity": 1, "known": True, "running": True}
    producers = {
        "screen_identity": 2,
        "runtime_identity": 3,
        "maintenance_error_clear": True,
        "monitor": dict(task),
        "trace": dict(task),
        "credential_timer": {
            "identity": 4,
            "known": True,
            "active": True,
            "interval": 0.25,
            "repeat": None,
            "task": dict(task),
        },
        "scheduler": {
            "identity": 5,
            "known": True,
            "running": True,
            "cancelled": False,
            "finished": False,
            "error_clear": True,
            "task": dict(task),
        },
    }
    return {
        "side": side,
        "pair": pair,
        "protocol": "live-idle-v1",
        "platform": "darwin",
        "execution_mode": "source",
        "source": {
            "commit": ("a" if side == "baseline" else "b") * 40,
            "tree": "c" * 40,
            "content_sha256": ("d" if side == "baseline" else "e") * 64,
        },
        "probe_sha256": "f" * 64,
        "containment_sha256": "0" * 64,
        "dependency_sha256": "1" * 64,
        "profile_depth": 7,
        "seed_notes": 8,
        "window_seconds": 60.0,
        "elapsed_ns": 60_000_000_000,
        "settlement": "ui-ready-plus-live-probe-and-five-seconds",
        "ui_ready": True,
        "settled": True,
        "timers_live": True,
        "producer_checks": {
            "before": copy.deepcopy(producers),
            "cutoff": copy.deepcopy(producers),
        },
        "uncovered_process_events": {},
        "retired": True,
        "supervisor_retired": True,
        "exit_code": 0,
        "network_attempts": 0,
        "foreign_source_modules": 0,
        "source_stable": True,
        "installed_stable": True,
        "child_network_coverage": "no-descendants",
        "unresolved_at_cutoff": {"scopes": 0, "probes": 0, "children": 0},
        "window": {
            "os_opens": opens,
            "native_handle_opens": 0,
            "native_acl_reads": 0,
            "probe_starts": 60,
            "probe_completions": 60,
            "probe_errors": 0,
            "probe_overlap": 0,
            "storage_admissions": 240,
            "storage_admission_successes": 240,
            "helper_starts": 0,
            "child_starts": 0,
            "config_entry_attempts": 240,
            "config_entries": 240,
            "config_retirement_attempts": 240,
            "config_retirements": 240,
            "repository_entry_attempts": 60,
            "repository_entries": 60,
            "repository_retirement_attempts": 60,
            "repository_retirements": 60,
        },
        "drain": {"os_opens": 12},
        "outstanding_ownership": {"_holds": 0},
    }


def windows():
    return [
        receipt(side, pair, 600 if side == "baseline" else 300)
        for pair, order in enumerate(
            (("baseline", "final"), ("final", "baseline"), ("baseline", "final"))
        )
        for side in order
    ]


def test_paired_comparison_uses_rates_and_retains_adverse_pairs():
    runs = windows()
    runs[1]["window"]["os_opens"] = 900
    result = harness().compare(runs)
    assert result["qualified"] is True
    assert result["baseline_median_opens_per_second"] == 10
    assert result["final_median_opens_per_second"] == 5
    assert result["final_rates"] == [15, 5, 5]
    assert result["reduction_percent"] == 50


@pytest.mark.parametrize(
    "change", [None, "interval_one", "interval_all", "repeat_one", "repeat_all"]
)
def test_comparison_requires_matching_observed_credential_timer_cadence(change):
    runs = windows()
    for index, run in enumerate(runs):
        for state in run["producer_checks"].values():
            # Process-local identities may differ between independently run apps.
            state["credential_timer"]["identity"] = index + 100
    if change is not None:
        field, affected = change.split("_")
        finals = [run for run in runs if run["side"] == "final"]
        for run in finals[:1] if affected == "one" else finals:
            for state in run["producer_checks"].values():
                state["credential_timer"][field] = 0.5 if field == "interval" else 10
    assert all(
        run["producer_checks"]["before"] == run["producer_checks"]["cutoff"]
        for run in runs
    )
    assert harness().compare(runs)["qualified"] == (change is None)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "order",
        "zero",
        "cleanup",
        "network",
        "foreign",
        "changed",
        "pending",
        "short",
        "failed",
        "timers",
        "readiness",
        "probe",
        "dependencies",
        "denominator",
        "mode",
        "counter_missing",
        "negative",
        "boolean",
        "native_missing",
        "child_network",
        "missing_config",
        "bad_scope_close",
        "nan",
        "missing_ownership",
        "cutoff_state_missing",
        "cutoff_paused",
        "uncovered_launch",
    ],
)
def test_incomplete_or_incompatible_receipts_cannot_qualify(damage):
    runs = windows()
    run = runs[0]
    if damage == "missing":
        runs.pop()
    elif damage == "order":
        runs.reverse()
    elif damage == "zero":
        for row in runs:
            if row["side"] == "baseline":
                row["window"]["os_opens"] = 0
    elif damage == "cleanup":
        run["retired"] = False
    elif damage == "network":
        run["network_attempts"] = 1
    elif damage == "foreign":
        run["foreign_source_modules"] = 1
    elif damage == "changed":
        run["source_stable"] = False
    elif damage == "pending":
        run["unresolved_at_cutoff"]["probes"] = 1
    elif damage == "short":
        run["elapsed_ns"] = 59_000_000_000
    elif damage == "failed":
        run["exit_code"] = -15
    elif damage == "timers":
        run["timers_live"] = False
    elif damage == "readiness":
        run["ui_ready"] = False
    elif damage == "probe":
        run["window"]["probe_errors"] = 1
    elif damage == "dependencies":
        run["dependency_sha256"] = "2" * 64
    elif damage == "denominator":
        run["source"] = copy.deepcopy(runs[1]["source"])
    elif damage == "mode":
        run["execution_mode"] = "installed"
    elif damage == "counter_missing":
        del run["window"]["native_acl_reads"]
    elif damage == "negative":
        run["window"]["os_opens"] = -1
    elif damage == "boolean":
        run["window"]["os_opens"] = True
    elif damage == "native_missing":
        for row in runs:
            row["platform"] = "win32"
    elif damage == "child_network":
        run["child_network_coverage"] = "unresolved"
    elif damage == "missing_config":
        del run["window"]["config_entries"]
    elif damage == "bad_scope_close":
        run["window"]["config_retirements"] = 239
    elif damage == "nan":
        run["elapsed_ns"] = float("nan")
    elif damage == "missing_ownership":
        run["outstanding_ownership"] = {}
    elif damage == "cutoff_state_missing":
        del run["producer_checks"]
    elif damage == "cutoff_paused":
        run["producer_checks"]["cutoff"]["credential_timer"]["active"] = False
    elif damage == "uncovered_launch":
        run["uncovered_process_events"] = {"os.fork": 1}
    assert harness().compare(runs)["qualified"] is False


def test_comparison_reports_unmet_target():
    runs = windows()
    for row in runs:
        if row["side"] == "final":
            row["window"]["os_opens"] = 301
    assert harness().compare(runs)["qualified"] is False


def test_all_thread_audit_keeps_open_identity_and_cutoff_drain_separate(tmp_path):
    module = harness()
    census = module.Census()
    original = os.open
    census.install_audit()
    census.phase = "window"

    def work():
        fd = os.open(tmp_path, os.O_RDONLY)
        os.close(fd)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join(5)
    census.cutoff()
    work()
    census.phase = None
    assert census.counts["window"]["os_opens"] == 1
    assert census.counts["drain"]["os_opens"] == 1
    assert os.open is original


def test_failed_scope_close_and_cross_cutoff_scope_remain_visible():
    module = harness()
    census = module.Census()
    census.phase = "window"

    class Scope:
        def __enter__(self):
            return 7

        def __exit__(self, *error):
            raise OSError("private-error")

    scope = census.scope(Scope, "config")
    assert scope.__enter__() == 7
    assert census.cutoff()["scopes"] == 1
    with pytest.raises(OSError):
        scope.__exit__(None, None, None)
    assert census.active_scopes == 1
    assert census.counts["window"]["config_entries"] == 1
    assert census.counts["drain"]["config_retirement_attempts"] == 1
    assert census.counts["drain"]["config_retirements"] == 0


def test_private_environment_replaces_all_ambient_storage_and_temp(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TLDW_CONFIG_PATH", "/ambient/secret")
    monkeypatch.setenv("TMPDIR", "/ambient/temp")
    monkeypatch.setenv("PYTHONPATH", "/ambient/source")
    environment = harness().private_environment(tmp_path / "profile")
    assert all(
        Path(environment[key]).is_relative_to(tmp_path)
        for key in (
            "HOME",
            "USERPROFILE",
            "XDG_CACHE_HOME",
            "XDG_STATE_HOME",
            "TEMP",
            "TMP",
            "TMPDIR",
            "TLDW_CONFIG_PATH",
        )
    )
    assert "PYTHONPATH" not in environment
    assert environment["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"


def test_installed_join_rejects_shadowed_or_changed_production_bytes(tmp_path):
    module = harness()
    source = tmp_path / "source"
    installed = tmp_path / "installed"
    for root in (source, installed):
        package = root / "tldw_chatbook"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("same")
    import zipfile

    wheel = tmp_path / "native.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tldw_chatbook/__init__.py", "same")
    assert module.installed_join(source, installed, wheel)["joined"] is True
    (installed / "tldw_chatbook/__init__.py").write_text("foreign")
    with pytest.raises(ValueError, match="installed_source_wheel_mismatch"):
        module.installed_join(source, installed, wheel)


def test_live_window_waits_once_and_retains_cutoff_without_stimulus(monkeypatch):
    module = harness()
    census = module.Census()
    ticks = iter((0, 60_500_000_000))
    slept = []

    async def sleep(seconds):
        slept.append(seconds)
        census.bump("os_opens")

    monkeypatch.setattr(module.time, "monotonic_ns", lambda: next(ticks))
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    result = asyncio.run(module.live_window(census))
    assert slept == [60.0]
    assert result["elapsed_ns"] == 60_500_000_000
    assert result["window"]["os_opens"] == 1
    assert census.phase == "drain"


def test_finite_selection_keeps_posix_warm_and_windows_cold_routes_distinct():
    from Tests.Backup_Recovery import run_platform_product as runner

    assert "admission-amortization" in runner._PRODUCT_SELECTIONS
    posix_native, posix = runner.admission_selection("Darwin")
    windows_native, cold = runner.admission_selection("Windows")
    assert "Tests/Utils/test_windows_files.py" in windows_native
    assert all("windows" not in node for node in posix_native)
    assert any(
        "test_warm_admission_reads_current_complete_control_bytes" in node
        for node in posix
    )
    assert not any("test_admission_evidence_reuse.py" in node for node in cold)
    assert any("test_windows_cold_acquisition" in node for node in cold)
    plain = "Tests/ProductionApp/test_backup_restore_end_to_end.py::test_f9_created_archive_restores_and_opens_through_actual_controls[plain]"
    assert posix.count(plain) == cold.count(plain) == 1
    assert not any("[profile-record-edited-in-place-unbound]" in node for node in posix)
    with pytest.raises(ValueError):
        runner.admission_selection("unsupported")


@pytest.mark.parametrize(
    "selected",
    [r"C:\Users\fixture\new\data", '/tmp/a"b/data', "C:\\Users\\fixture-😀\\data"],
)
def test_mcp_fixture_preserves_platform_path_in_toml(selected):
    import tomllib

    from Tests.Backup_Recovery import test_mcp_source_lifetimes as fixtures

    assert hasattr(fixtures, "_fixture_config"), (
        "portable MCP fixture serialization is missing"
    )
    assert (
        tomllib.loads(fixtures._fixture_config(selected))["paths"]["data_dir"]
        == selected
    )


def test_child_guard_and_null_keyring_precede_selected_app_import(tmp_path):
    from Tests.Performance.test_backup_admission_benchmark import probe, source

    module = harness()
    fixed = probe()
    selected = source(fixed, tmp_path / "source")
    (selected / "tldw_chatbook/Backup_Recovery/config_participants.py").write_text(
        "import os, sys, keyring\nfrom keyring.backends.null import Keyring\n"
        "from pathlib import Path\nassert isinstance(keyring.get_keyring(),Keyring)\n"
        "assert 'Tests.network_guard' in sys.modules\n"
        "assert Path(os.environ['HOME']).is_relative_to(Path.cwd().parent)\n"
        "raise RuntimeError('private-content-must-not-escape')\n"
    )
    manifest = json.loads((selected / fixed.MANIFEST).read_text())
    manifest["content_sha256"] = fixed.source_digest(selected)
    (selected / fixed.MANIFEST).write_text(json.dumps(manifest))
    result = module.run_child(selected, tmp_path / "profile", "isolation")
    assert result["error_type"] == "RuntimeError"
    assert result["exit_code"] != 0 and result["network_attempts"] == 0
    assert "private-content" not in json.dumps(result)
    assert result["supervisor_retired"] is True
    with pytest.raises(ValueError, match="idle_receipt_already_exists"):
        module.run_child(selected, tmp_path / "profile", "isolation")


def test_callthrough_instrumentation_counts_errors_and_preserves_suppression(
    monkeypatch,
):
    from contextlib import ExitStack, contextmanager

    from tldw_chatbook.Backup_Recovery import config_participants
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _RepositoryParticipant
    from tldw_chatbook.DB.private_sqlite_process import HelperLease

    module = harness()
    census = module.Census()
    census.phase = "window"

    @contextmanager
    def scope(*args):
        try:
            yield 17
        except ValueError:
            pass

    monkeypatch.setattr(config_participants, "operation", scope)
    monkeypatch.setattr(_RepositoryParticipant, "operation", scope)
    monkeypatch.setattr(storage, "_acquire_storage", lambda *args: 23)
    monkeypatch.setattr(HelperLease, "start", classmethod(lambda *args: 42))

    def failed_probe():
        raise OSError("private-error")

    monkeypatch.setattr(storage, "_local_pause_requested", failed_probe)
    with ExitStack() as stack:
        census.instrument(stack)
        with config_participants.operation():  # noqa: SIM117 - nested scope census
            with config_participants.operation() as value:
                assert value == 17
                raise ValueError()
        with _RepositoryParticipant.operation(object()):
            pass
        assert storage._acquire_storage(None) == 23
        assert HelperLease.start() == 42
        with pytest.raises(OSError):
            storage._local_pause_requested()
    assert census.counts["window"]["config_entries"] == 1
    assert census.counts["window"]["config_retirements"] == 1
    assert census.counts["window"]["repository_entries"] == 1
    assert census.counts["window"]["storage_admissions"] == 1
    assert census.counts["window"]["helper_starts"] == 1
    assert (
        census.counts["window"]["probe_starts"]
        == census.counts["window"]["probe_errors"]
        == 1
    )
    assert census.cutoff()["scopes"] == census.active_probes == 0


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_runner_executes_resolved_selection_with_original_phase_budgets(
    tmp_path, monkeypatch, system
):
    from Tests.Backup_Recovery import run_platform_product as runner

    workspace = tmp_path / "source"
    workspace.mkdir()
    monkeypatch.setattr(runner.platform, "system", lambda: system)
    monkeypatch.setattr(
        runner, "_copy_tracked_source", lambda *args: (workspace, "a" * 64)
    )
    monkeypatch.setattr(runner, "_source_receipt", lambda *args: {"source": "exact"})
    monkeypatch.setattr(
        runner, "_windows_ancestor_receipt", lambda *args: {"native": "pending"}
    )
    monkeypatch.setattr(runner, "_native_identity", lambda *args: {"native": "pending"})
    monkeypatch.setattr(
        runner, "_installed_receipts", lambda *args: [{"wheel_sha256": "b" * 64}]
    )
    seen = []

    def phase(**kwargs):
        seen.append((kwargs["phase"], kwargs["timeout_seconds"], kwargs["noconftest"]))
        return {
            "tests": list(kwargs["tests"]),
            "pytest_returncode": 0,
            "junit": {
                "collected": len(kwargs["tests"]),
                "parse_error": None,
                "failed": [],
                "skipped": [],
            },
        }

    monkeypatch.setattr(runner, "_run_pytest_phase", phase)
    evidence = tmp_path / "evidence"
    assert (
        runner.run(workspace, evidence, product_selection="admission-amortization") == 0
    )
    summary = json.loads((evidence / "artifacts/summary.json").read_text())
    assert seen == [("native", 600, True), ("product", 4800, False)]
    assert summary["product_selection"] == "admission-amortization"
    assert any(
        "test_windows_cold_acquisition" in node for node in summary["tests"]
    ) == (system == "Windows")
    assert any(
        "test_warm_admission_reads_current_complete_control_bytes" in node
        for node in summary["tests"]
    ) == (system == "Darwin")


@pytest.mark.parametrize("route", ["guarded", "descendant", "overflow", "wrong_bytes"])
def test_known_helper_observer_executes_original_entry_checks_and_fails_uncovered(
    tmp_path, route
):
    module = harness()
    selected = tmp_path / "source"
    (selected / "Tests").mkdir(parents=True)
    (selected / "Tests/network_guard.py").write_bytes(
        (Path(__file__).parents[1] / "network_guard.py").read_bytes()
    )
    entry = selected / "tldw_chatbook/DB/private_sqlite_helper_entry.py"
    entry.parent.mkdir(parents=True)
    entry.write_text(
        "import os,sys,socket,subprocess\nfrom pathlib import Path\n"
        "assert sys.flags.isolated and sys.flags.no_site\n"
        "assert len(sys.argv)==1 and Path(sys.argv[0]).resolve()==Path(__file__).resolve()\n"
        "assert os.getppid()==int(os.environ['_TLDW_PRIVATE_SQLITE_PARENT_PID'])\n"
        "assert 'Tests.network_guard' in sys.modules\n"
        "try:socket.create_connection(('127.0.0.1',9))\nexcept OSError:pass\n"
        + (
            "subprocess.run([sys.executable,'-I','-S','-c','pass'],check=True)\n"
            if route == "descendant"
            else ""
        )
        + (
            "for _ in range(5000):\n fd=os.open(os.devnull,os.O_RDONLY);os.close(fd)\n"
            if route == "overflow"
            else "fd=os.open(os.devnull,os.O_RDONLY);os.close(fd)\n"
        )
    )
    receipt_file = tmp_path / "helper.json"
    command = [
        sys.executable,
        "-I",
        "-S",
        str(Path(module.__file__)),
        "--helper-observer",
        "--source",
        str(selected),
        "--entry",
        str(entry),
        "--entry-sha256",
        "0" * 64
        if route == "wrong_bytes"
        else hashlib.sha256(entry.read_bytes()).hexdigest(),
        "--receipt",
        str(receipt_file),
    ]
    result = subprocess.run(
        command,
        env={**os.environ, "_TLDW_PRIVATE_SQLITE_PARENT_PID": str(os.getpid())},
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert receipt_file.exists(), "known helper observer receipt is missing"
    receipt = json.loads(receipt_file.read_text())
    assert b"127.0.0.1" not in receipt_file.read_bytes()
    if route == "wrong_bytes":
        assert result.returncode != 0 and receipt["entry_stable"] is False
    else:
        assert (
            receipt["original_entry_executed"] is True
            and receipt["entry_stable"] is True
        )
        assert any(event["unit"] == "network_attempts" for event in receipt["events"])
        assert receipt["uncovered_descendants"] == (route == "descendant")
        assert receipt["overflow"] == (route == "overflow")
        assert len(receipt["events"]) <= 4096


@pytest.mark.parametrize(
    "failure",
    [None, "missing", "killed", "overflow", "unknown", "negative_time", "oversized"],
)
def test_child_projection_accounts_for_cold_events_at_exact_window_cutoffs(
    tmp_path, failure
):
    from types import SimpleNamespace

    module = harness()
    census = module.Census()
    child = SimpleNamespace(pid=123, poll=lambda: -15 if failure == "killed" else 0)
    census.children.append(child)
    path = tmp_path / "known-helper.json"
    observed = {"receipt": path, "entry_sha256": "a" * 64, "guard_sha256": "b" * 64}
    census.child_receipts.append((child, None if failure == "unknown" else observed))
    data = {
        "schema": 1,
        "pid": 123,
        "parent_pid": os.getpid(),
        "entry_sha256": "a" * 64,
        "guard_sha256": "b" * 64,
        "observer_sha256": hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest(),
        "exit_code": 0,
        "entry_stable": True,
        "original_entry_executed": True,
        "overflow": failure == "overflow",
        "uncovered_descendants": False,
        "windows_native_observed": True,
        "events": [
            {"timestamp_ns": 99, "unit": "os_opens"},
            {"timestamp_ns": 100, "unit": "os_opens"},
            {"timestamp_ns": 199, "unit": "native_handle_opens"},
            {"timestamp_ns": 200, "unit": "os_opens"},
        ],
    }
    if failure == "negative_time":
        data["events"][0]["timestamp_ns"] = -1
    elif failure == "oversized":
        data["events"] = [data["events"][0]] * 4097
    if failure != "missing":
        path.write_text(json.dumps(data))
    measured = {
        "window_started_ns": 100,
        "cutoff_ns": 200,
        "window": {"os_opens": 10, "native_handle_opens": 0},
    }
    result = census.project_children(measured)
    if failure:
        assert result["child_network_coverage"] == "unresolved"
    else:
        assert result["child_network_coverage"] == "joined"
        assert measured["window"] == {"os_opens": 11, "native_handle_opens": 1}
        assert measured["parent_window"]["os_opens"] == 10
        assert (
            census.counts["settlement"]["os_opens"]
            == census.counts["drain"]["os_opens"]
            == 1
        )


@pytest.mark.parametrize("known", [True, False, "same_argv_outside_helper"])
def test_launch_observer_only_wraps_exact_known_helper_argv(
    tmp_path, monkeypatch, known
):
    from contextlib import ExitStack

    from tldw_chatbook.DB import private_sqlite_process

    module = harness()
    source = tmp_path / "source"
    (source / "Tests").mkdir(parents=True)
    (source / "Tests/network_guard.py").write_bytes(
        (Path(__file__).parents[1] / "network_guard.py").read_bytes()
    )
    entry = source / "tldw_chatbook/DB/private_sqlite_helper_entry.py"
    entry.parent.mkdir(parents=True)
    entry.write_text(
        "import os,sys\nfrom pathlib import Path\n"
        "assert sys.flags.isolated and sys.flags.no_site and len(sys.argv)==1\n"
        "assert Path(sys.argv[0]).resolve()==Path(__file__).resolve()\n"
        "assert os.getppid()==int(os.environ['_TLDW_PRIVATE_SQLITE_PARENT_PID'])\n"
        + ("assert 'Tests.network_guard' in sys.modules\n" if known is True else "")
        + "fd=os.open(os.devnull,os.O_RDONLY);os.close(fd)\n"
    )
    monkeypatch.setattr(
        private_sqlite_process,
        "__file__",
        str(entry.with_name("private_sqlite_process.py")),
    )
    monkeypatch.setenv("TLDW_TEST_CONFIG_ROOT", str(tmp_path))
    environment = {**os.environ, "_TLDW_PRIVATE_SQLITE_PARENT_PID": str(os.getpid())}
    census = module.Census()
    census.phase = "window"
    command = (
        [sys.executable, "-I", "-S", str(entry)]
        if known is not False
        else [sys.executable, "-I", "-S", "-c", "pass"]
    )
    started = module.time.monotonic_ns()
    from tldw_chatbook.DB.private_sqlite_process import HelperLease

    def original_start(cls):
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )

    monkeypatch.setattr(HelperLease, "start", classmethod(original_start))
    with ExitStack() as stack:
        census.install_audit()
        census.instrument(stack, source)
        child = (
            HelperLease.start()
            if known is True
            else subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
        )
        output, error = child.communicate(timeout=30)
    measured = {
        "window_started_ns": started,
        "cutoff_ns": module.time.monotonic_ns(),
        "window": dict(census.counts["window"]),
    }
    result = census.project_children(measured)
    assert child.returncode == 0 and output == error == b""
    assert result["child_network_coverage"] == (
        "joined" if known is True else "unresolved"
    )
    if known is True:
        # Keep observer bootstrap costs as well as the entry's intentional open.
        assert measured["window"]["os_opens"] >= 1
        assert measured["parent_window"]["os_opens"] == 0
        assert result["uncovered_process_events"] == {}


@pytest.mark.parametrize("parent", ["actual", "wrong", "missing"])
def test_observer_runs_unchanged_entry_parent_predicate(tmp_path, parent):
    module = harness()
    source = tmp_path / "source"
    (source / "Tests").mkdir(parents=True)
    (source / "Tests/network_guard.py").write_bytes(
        (Path(__file__).parents[1] / "network_guard.py").read_bytes()
    )
    entry = source / "tldw_chatbook/DB/private_sqlite_helper_entry.py"
    entry.parent.mkdir(parents=True)
    entry.write_bytes(
        (
            Path(__file__).parents[2]
            / "tldw_chatbook/DB/private_sqlite_helper_entry.py"
        ).read_bytes()
    )
    marker = tmp_path / "called"
    entry.with_name("private_sqlite_helper.py").write_text(
        "import os,sys\nfrom pathlib import Path\n"
        "def run(parent_pid):\n"
        " assert parent_pid==os.getppid()\n"
        " assert 'Tests.network_guard' in sys.modules\n"
        f" Path({str(marker)!r}).write_text('entry-reached-helper')\n"
        " return 0\n"
    )
    environment = dict(os.environ)
    environment.pop("_TLDW_PRIVATE_SQLITE_PARENT_PID", None)
    if parent != "missing":
        environment["_TLDW_PRIVATE_SQLITE_PARENT_PID"] = str(
            os.getpid() if parent == "actual" else os.getpid() + 1
        )
    receipt = tmp_path / "original-entry.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(Path(module.__file__)),
            "--helper-observer",
            "--source",
            str(source),
            "--entry",
            str(entry),
            "--entry-sha256",
            hashlib.sha256(entry.read_bytes()).hexdigest(),
            "--receipt",
            str(receipt),
        ],
        env=environment,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == (0 if parent == "actual" else 1)
    assert marker.exists() == (parent == "actual")
    assert json.loads(receipt.read_text())["entry_stable"] is True


@pytest.mark.parametrize(
    "change",
    [
        None,
        "monitor",
        "trace",
        "scheduler",
        "scheduler_error",
        "timer_stop",
        "timer_pause",
        "timer_replace",
        "maintenance_error",
    ],
)
def test_live_window_rechecks_actual_producer_state_at_cutoff(monkeypatch, change):
    from types import SimpleNamespace

    from textual.timer import Timer
    from textual.worker import Worker, WorkerState

    module = harness()

    class Node:
        def post_message(self, message):
            pass

    async def run():
        gate = asyncio.Event()
        monitor = asyncio.create_task(gate.wait())
        trace = asyncio.create_task(gate.wait())
        timer = Timer(Node(), 0.25)
        timer._task = asyncio.create_task(gate.wait())
        scheduler = Worker(Node(), lambda: None)
        scheduler._task = asyncio.create_task(gate.wait())
        scheduler.state = WorkerState.RUNNING
        app = SimpleNamespace(
            _backup_maintenance_monitor_task=monitor,
            _backup_maintenance_error=None,
            scheduler_worker=scheduler,
            screen=SimpleNamespace(
                _console_credential_poll_timer=timer,
                _console_runtime_ref=SimpleNamespace(
                    _legacy_trace_maintenance_task=trace
                ),
            ),
        )
        original_sleep = asyncio.sleep

        async def awaited_window(seconds):
            assert seconds == 60.0
            if change in ("monitor", "trace"):
                task = monitor if change == "monitor" else trace
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            elif change == "scheduler":
                scheduler.state = WorkerState.SUCCESS
            elif change == "scheduler_error":
                scheduler._error = RuntimeError("private error")
                scheduler.state = WorkerState.ERROR
            elif change == "timer_stop":
                timer.stop()
            elif change == "timer_pause":
                timer.pause()
            elif change == "timer_replace":
                app.screen._console_credential_poll_timer = Timer(Node(), 0.25)
            elif change == "maintenance_error":
                app._backup_maintenance_error = "admission_state_unavailable"
            await original_sleep(0)

        monkeypatch.setattr(module.asyncio, "sleep", awaited_window)
        tasks = [monitor, trace, timer._task, scheduler._task]
        try:
            result = await module.live_window(module.Census(), app)
            assert result["timers_live"] == (change is None)
            assert set(result["producer_checks"]) == {"before", "cutoff"}
            assert (
                result["producer_checks"]["before"]
                == result["producer_checks"]["cutoff"]
            ) == (change is None)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "installed_bytes",
        "wheel_bytes",
        "wheel_missing",
        "post_join",
        "stat_only",
    ],
)
def test_installed_sql_assets_require_current_source_wheel_bytes_and_stat(
    tmp_path, change
):
    import zipfile

    module = harness()
    source, installed = tmp_path / "source", tmp_path / "installed"
    for root in (source, installed):
        package = root / "tldw_chatbook"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("same")
        (package / "schema75.sql").write_text("SELECT 1;")
    wheel = tmp_path / "native.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tldw_chatbook/__init__.py", "same")
        if change != "wheel_missing":
            archive.writestr(
                "tldw_chatbook/schema75.sql",
                "SELECT 2;" if change == "wheel_bytes" else "SELECT 1;",
            )
    asset = installed / "tldw_chatbook/schema75.sql"
    if change in ("post_join", "stat_only"):
        before = module.installed_join(source, installed, wheel)["installed_files"]
        if change == "post_join":
            asset.write_text("SELECT 2;")
        else:
            info = asset.stat()
            os.utime(asset, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
        assert module.package_files(installed) != before
    else:
        if change == "missing":
            asset.unlink()
        elif change == "installed_bytes":
            asset.write_text("SELECT 2;")
        with pytest.raises((KeyError, ValueError)):
            module.installed_join(source, installed, wheel)


@pytest.mark.parametrize(
    "event",
    ["os.fork", "os.posix_spawn", "os.system", "os.exec", "_winapi.CreateProcess"],
)
def test_parent_refuses_unsupported_creation_event_without_changing_audit_callthrough(
    event,
):
    module = harness()
    census = module.Census()
    census.install_audit()
    sys.audit(event, "metadata-only-control")
    result = census.project_children({})
    assert result["child_network_coverage"] == "unresolved"
    assert result["uncovered_process_events"] == {event: 1}


@pytest.mark.parametrize(
    "event",
    ["os.fork", "os.posix_spawn", "os.system", "os.exec", "_winapi.CreateProcess"],
)
def test_known_helper_refuses_unsupported_creation_event_and_keeps_entry_callthrough(
    tmp_path, event
):
    module = harness()
    source = tmp_path / "source"
    (source / "Tests").mkdir(parents=True)
    (source / "Tests/network_guard.py").write_bytes(
        (Path(__file__).parents[1] / "network_guard.py").read_bytes()
    )
    entry = source / "tldw_chatbook/DB/private_sqlite_helper_entry.py"
    entry.parent.mkdir(parents=True)
    marker = tmp_path / "continued"
    entry.write_text(
        f"import sys\nfrom pathlib import Path\nsys.audit({event!r}, 'metadata-only-control')\nPath({str(marker)!r}).write_text('continued')\n"
    )
    receipt = tmp_path / "helper.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(Path(module.__file__)),
            "--helper-observer",
            "--source",
            str(source),
            "--entry",
            str(entry),
            "--entry-sha256",
            hashlib.sha256(entry.read_bytes()).hexdigest(),
            "--receipt",
            str(receipt),
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    data = json.loads(receipt.read_text())
    assert result.returncode == 0 and marker.read_text() == "continued"
    assert data["uncovered_descendants"] is True
