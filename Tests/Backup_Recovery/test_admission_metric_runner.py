"""Finite trust, receipt and custody controls for Windows metric invocation."""

import asyncio
import copy
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import run_platform_product as runner


@pytest.fixture
def history(tmp_path, monkeypatch):
    """Three tiny actual trees; no application or qualification is executed."""
    workspace = tmp_path / "checkout"
    workspace.mkdir()

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()

    git("init", "-q")
    revisions = []
    for index in range(3):
        (workspace / f"side-{index}.txt").write_text(str(index))
        git("add", ".")
        git(
            "-c",
            "user.name=Metric",
            "-c",
            "user.email=metric@example.invalid",
            "commit",
            "-qm",
            "synthetic source",
        )
        revisions.append(git("rev-parse", "HEAD"))
    monkeypatch.setattr(runner.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        runner, "_ADMISSION_HISTORICAL_REF", revisions[0], raising=False
    )
    return workspace, revisions


def test_metric_preflight_selects_exact_distinct_ancestor_trees(history):
    workspace, refs = history
    assert runner._metric_preflight(workspace, refs[1]) == {
        "historical": refs[0],
        "baseline": refs[1],
        "final": refs[2],
    }


@pytest.mark.parametrize("baseline", ["HEAD", "-x", "a" * 39, "A" * 40])
def test_metric_preflight_refuses_arbitrary_ref_input(history, baseline):
    with pytest.raises(ValueError):
        runner._metric_preflight(history[0], baseline)


@pytest.mark.parametrize("side", [0, 2])
def test_metric_preflight_refuses_changed_denominator(history, side):
    with pytest.raises(ValueError):
        runner._metric_preflight(history[0], history[1][side])


def test_metric_preflight_refuses_dirty_source(history):
    (history[0] / "side-0.txt").write_text("changed")
    with pytest.raises(RuntimeError):
        runner._metric_preflight(history[0], history[1][1])


def test_metric_preflight_refuses_posix_nested_group_claim(history, monkeypatch):
    monkeypatch.setattr(runner.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError):
        runner._metric_preflight(history[0], history[1][1])


def test_exact_source_copy_uses_selected_tree_members(history, tmp_path):
    private = tmp_path / "copy"
    private.mkdir()
    source, digest = runner._copy_tracked_source(
        history[0],
        private,
        revision=history[1][0],
    )
    assert (sorted(path.name for path in source.iterdir()), len(digest)) == (
        ["side-0.txt"],
        64,
    )


def fixed_receipts():
    """Literal original receipt units; target expectations are hand-derived."""
    sources = {
        side: {"commit": digit * 40, "tree": digit * 40, "content_sha256": digit * 64}
        for side, digit in (("historical", "1"), ("final", "2"))
    }
    receipts = {}
    for side in ("historical", "final"):
        child = {
            "exit_code": 0,
            "retired": True,
            "supervisor_retired": True,
            "network_attempts": 0,
            "foreign_source_modules": 0,
            "source_sha256": sources[side]["content_sha256"],
            "outstanding_ownership": {"_holds": 0},
            "live_children_before_reaping": 0,
            "native_handle_opens": 1,
            "native_acl_reads": 1,
            "transaction_boundary_median_ns": 499999,
            "os_opens": 100 if side == "historical" else 20,
        }
        for phase, iterations in (("transaction", 100), ("boot", 3)):
            receipts[f"{side}-{phase}"] = {
                "protocol": "reconstructed-v1",
                "phase": phase,
                "iterations": iterations,
                "platform": "win32",
                "native_windows_measured": True,
                "source": sources[side],
                "probe_sha256": "34278facac896ecc0e4ed8a3319243d3501272e87692a858779c6b449a475428",
                "containment_sha256": "77c5fd81925b9f3377b88fa41782707b4979da9361fd6c5e4d111ceb75c5a6a5",
                "exit_code": 0,
                "seed": copy.deepcopy(child),
                "warmup": copy.deepcopy(child) if phase == "boot" else None,
                "runs": [
                    copy.deepcopy(child) for _ in range(3 if phase == "boot" else 1)
                ],
            }
            receipts[f"{side}-{phase}"]["seed"]["seed_notes"] = 8
            if phase == "transaction":
                receipts[f"{side}-{phase}"]["runs"][0]["seed_notes"] = 8
    return receipts, sources


@pytest.mark.parametrize("boundary,qualified", [(499999, True), (500000, False)])
def test_fixed_complete_boundary_miss_is_retained(boundary, qualified):
    receipts, sources = fixed_receipts()
    receipts["final-transaction"]["runs"][0]["transaction_boundary_median_ns"] = (
        boundary
    )
    summary = runner._metric_fixed_summary(receipts, sources)
    assert (
        summary["qualified"],
        summary["complete_transaction_boundary_median_ns"],
        summary["boot_reduction_percent"],
    ) == (qualified, boundary, 80.0)


@pytest.mark.parametrize(
    "damage", ["short", "cleanup", "network", "native", "source", "zero"]
)
def test_fixed_incomplete_evidence_never_qualifies(damage):
    receipts, sources = fixed_receipts()
    run = receipts["final-boot"]["runs"][0]
    if damage == "short":
        receipts["final-boot"]["runs"].pop()
    elif damage == "cleanup":
        run["supervisor_retired"] = False
    elif damage == "network":
        run["network_attempts"] = 1
    elif damage == "native":
        run["native_acl_reads"] = 0
    elif damage == "source":
        run["source_sha256"] = "f" * 64
    else:
        for run in receipts["historical-boot"]["runs"]:
            run["os_opens"] = 0
    assert runner._metric_fixed_summary(receipts, sources)["qualified"] is False


@pytest.mark.parametrize("empty", [True, False])
def test_outer_supervisor_closes_only_proven_empty_job(tmp_path, empty):
    async def wait_process():
        return 0

    async def discard(_size):
        return b""

    class Controller:
        def __init__(self):
            self.closed = self.killed = False
            self.process = SimpleNamespace(
                wait=wait_process,
                stdout=SimpleNamespace(read=discard),
                stderr=SimpleNamespace(read=discard),
            )

        async def spawn(self, *args, **kwargs):
            return SimpleNamespace(process=self.process)

        async def wait(self, tree, *, timeout):
            return empty

        def kill(self, tree):
            self.killed = True

        def close(self, tree):
            self.closed = True

    controller = Controller()
    result = asyncio.run(
        runner._metric_supervise(
            controller,
            ["fixed-command"],
            tmp_path,
            {},
            time.monotonic() + 10,
        )
    )
    assert (result["supervisor_retired"], controller.closed, controller.killed) == (
        empty,
        empty,
        not empty,
    )


def test_outer_shared_deadline_refuses_launch_before_effects(tmp_path):
    class Controller:
        async def spawn(self, *args, **kwargs):
            pytest.fail("expired phase launched a child")

    result = asyncio.run(
        runner._metric_supervise(
            Controller(),
            ["fixed-command"],
            tmp_path,
            {},
            time.monotonic() - 1,
        )
    )
    assert result["exit_code"] != 0


@pytest.mark.parametrize("retired", [False, True])
def test_partial_receipt_reads_require_positive_outer_retirement(
    tmp_path, monkeypatch, retired
):
    """An unknown tree may still be writing; only settled metadata is read."""
    copy_metric_programs(tmp_path)
    private = tmp_path / "private"
    raw = private / "metric-receipts"
    raw.mkdir(parents=True)
    child = raw / "boot-child.json"
    child.write_text('{"exit_code": 0}\n')
    monkeypatch.setattr(runner, "_metric_preflight", lambda *args: {})
    monkeypatch.setattr(runner, "_create_private_root", lambda *args: private)
    monkeypatch.setattr(runner, "_private_environment", lambda *args: {})
    module = SimpleNamespace(ProcessTreeController=lambda: object())
    module.loader = SimpleNamespace(exec_module=lambda *args: None)
    spec = SimpleNamespace(name="metric_containment", loader=module.loader)
    monkeypatch.setattr(
        runner.importlib.util, "spec_from_file_location", lambda *args: spec
    )
    monkeypatch.setattr(runner.importlib.util, "module_from_spec", lambda *args: module)
    monkeypatch.setitem(runner.sys.modules, "metric_containment", module)

    async def supervise(*args):
        return {"exit_code": 0 if retired else 1, "supervisor_retired": retired}

    monkeypatch.setattr(runner, "_metric_supervise", supervise)
    reads = []
    original = runner._sanitize_file

    def observe(source, destination, **kwargs):
        reads.append(source)
        return original(source, destination, **kwargs)

    monkeypatch.setattr(runner, "_sanitize_file", observe)
    runner._run_admission_metrics(tmp_path, tmp_path / "evidence", "reviewed")
    assert reads == ([child] if retired else [])


@pytest.mark.parametrize(
    "field,value",
    [
        ("containment_sha256", None),
        ("containment_sha256", "0" * 64),
        ("seed", None),
        ("seed", 7),
        ("transaction", None),
        ("transaction", 7),
    ],
)
def test_fixed_schema_requires_approved_containment_and_eight_note_seed(field, value):
    receipts, sources = fixed_receipts()
    if field == "containment_sha256":
        target, key = receipts["final-boot"], field
    else:
        target = (
            receipts["final-boot"]["seed"]
            if field == "seed"
            else receipts["final-transaction"]["runs"][0]
        )
        key = "seed_notes"
    if value is None:
        target.pop(key)
    else:
        target[key] = value
    assert runner._metric_fixed_summary(receipts, sources)["qualified"] is False


def copy_metric_programs(workspace):
    origin = Path(runner.__file__).parents[2]
    for name in (
        "Helper_Scripts/Benchmarks/backup_admission_benchmark.py",
        "Helper_Scripts/Benchmarks/backup_admission_idle_benchmark.py",
        "tldw_chatbook/Notes/git_process_containment.py",
    ):
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin / name, target)


@pytest.fixture
def metric_phase(tmp_path, monkeypatch):
    """Actual phase flow; tiny real source/wheel joins, fake native boundaries."""
    from Tests.Packaging import test_backup_helper_distribution as packaging
    from Tests.Performance.test_backup_admission_idle_benchmark import harness, windows

    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda *args: SimpleNamespace(free=3 * 1024**3)
    )

    workspace, private, artifacts = (
        tmp_path / name for name in ("workspace", "private", "artifacts")
    )
    private.mkdir()
    artifacts.mkdir()
    copy_metric_programs(workspace)
    idle = harness()
    idle.FIXED = workspace / "Helper_Scripts/Benchmarks/backup_admission_benchmark.py"
    refs = {
        side: digit * 40
        for side, digit in (("historical", "1"), ("baseline", "3"), ("final", "2"))
    }
    state = SimpleNamespace(
        workspace=workspace,
        private=private,
        artifacts=artifacts,
        calls=[],
        builds=[],
        snapshots={},
        damage=None,
        imported=[],
    )
    monkeypatch.setattr(runner, "_metric_preflight", lambda *args: refs)
    monkeypatch.setattr(runner, "_run_git", lambda *args: "4" * 40)
    monkeypatch.setattr(packaging, "REPO_ROOT", workspace)

    def source_copy(_workspace, root, *, revision):
        source = root / "source"
        file = source / "tldw_chatbook/fixture.py"
        file.parent.mkdir(parents=True)
        file.write_text("# synthetic " + revision + "\n")
        state.snapshots[revision] = source
        return source, "5" * 64

    monkeypatch.setattr(runner, "_copy_tracked_source", source_copy)
    monkeypatch.setattr(
        runner,
        "_source_receipt",
        lambda *args, **kwargs: {"revision": kwargs["revision"]},
    )

    def build_copy(target):
        source = packaging.REPO_ROOT
        state.builds.append((source, target))
        shutil.copytree(source / "tldw_chatbook", target / "tldw_chatbook")

    def build(source, output):
        output.mkdir()
        wheel = output / "fixture.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.write(
                source / "tldw_chatbook/fixture.py", "tldw_chatbook/fixture.py"
            )
        return wheel

    monkeypatch.setattr(packaging, "_copy_build_source", build_copy)
    monkeypatch.setattr(packaging, "_build_wheel", build)
    loader = SimpleNamespace(exec_module=lambda module: state.imported.append(module))
    monkeypatch.setattr(
        runner.importlib.util,
        "spec_from_file_location",
        lambda *args: SimpleNamespace(loader=loader),
    )
    monkeypatch.setattr(runner.importlib.util, "module_from_spec", lambda *args: idle)

    def execute(command, **kwargs):
        if "pip" in command:
            target = Path(command[command.index("--target") + 1])
            with zipfile.ZipFile(command[-1]) as wheel:
                wheel.extractall(target)
            if state.damage == "install":
                (target / "tldw_chatbook/fixture.py").write_text(
                    "wrong installed bytes"
                )
            return SimpleNamespace(returncode=0)
        state.calls.append(command)
        path = Path(command[command.index("--receipt") + 1])
        if "--phase" in command:
            source = Path(command[command.index("--source") + 1])
            side = "historical" if "historical" in str(source) else "final"
            phase = command[command.index("--phase") + 1]
            receipt = fixed_receipts()[0][f"{side}-{phase}"]
            receipt["source"] = idle.fixed.select_source(source)
            for child in [
                receipt["seed"],
                *receipt["runs"],
                *([receipt["warmup"]] if phase == "boot" else []),
            ]:
                child["source_sha256"] = receipt["source"]["content_sha256"]
            if state.damage == "boundary":
                receipt["runs"][0]["transaction_boundary_median_ns"] = 500000
            if state.damage == "custody":
                receipt["runs"][0]["supervisor_retired"] = False
            if state.damage == "source":
                (source / "tldw_chatbook/fixture.py").write_text("changed source")
            if state.damage in ("idle-live", "containment-live"):
                name = (
                    "Helper_Scripts/Benchmarks/backup_admission_idle_benchmark.py"
                    if state.damage == "idle-live"
                    else "tldw_chatbook/Notes/git_process_containment.py"
                )
                (workspace / name).write_text("changed live program")
            path.write_text(json.dumps(receipt))
            return SimpleNamespace(
                returncode=1 if state.damage == "prerequisite" else 0
            )
        manifests = {
            side: idle.fixed.select_source(state.snapshots[refs[side]])
            for side in ("baseline", "final")
        }
        runs = windows()
        for run in runs:
            run.update(
                source=manifests[run["side"]],
                platform="win32",
                execution_mode="installed",
                probe_sha256="1d4df8e02a0472c66b994ac562cc2aa729e75e9a4cb06a2e7a6617d0221bf440",
                containment_sha256="77c5fd81925b9f3377b88fa41782707b4979da9361fd6c5e4d111ceb75c5a6a5",
            )
            run["window"].update(native_handle_opens=1, native_acl_reads=1)
        if state.damage == "idle-comparison":
            runs[0]["retired"] = False
        if state.damage == "idle-join":
            for run in runs:
                run["probe_sha256"] = "f" * 64
        result = {
            "protocol": "live-idle-v1",
            "sources": manifests,
            "seeds": [],
            "runs": runs,
            "comparison": idle.compare(runs),
        }
        path.write_text(json.dumps(result))
        if state.damage in ("wheel", "installed"):
            wheel = Path(command[command.index("--wheel") + 1])
            target = Path(command[command.index("--installed") + 1])
            (
                wheel
                if state.damage == "wheel"
                else target / "tldw_chatbook/fixture.py"
            ).write_bytes(b"changed artifact")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    return state


@pytest.mark.parametrize("damage", [None, "boundary"])
def test_metric_phase_calls_four_fixed_and_installed_idle_with_separate_builds(
    metric_phase, damage
):
    state = metric_phase
    state.damage = damage
    exit_code = runner._metric_phase(
        state.workspace, state.private, state.artifacts, "3" * 40
    )
    fixed = state.workspace / "Helper_Scripts/Benchmarks/backup_admission_benchmark.py"
    expected = [
        [
            runner.sys.executable,
            str(fixed),
            "--source",
            str(state.private / f"metric-{side}/source"),
            "--phase",
            phase,
            "--iterations",
            count,
            "--receipt",
            str(state.private / f"metric-receipts/{side}-{phase}.json"),
        ]
        for phase, count in (("transaction", "100"), ("boot", "3"))
        for side in ("historical", "final")
    ]
    expected.append(
        [
            runner.sys.executable,
            str(fixed.with_name("backup_admission_idle_benchmark.py")),
            "--source",
            str(state.private / "metric-final/source"),
            "--baseline",
            str(state.private / "metric-baseline/source"),
            "--baseline-description",
            "Reviewed contemporary pre-Task2/3 1Hz/schema75/PERF07/release0.2.3/trace baseline "
            + "3" * 40,
            "--installed",
            str(state.private / "metric-final/installed"),
            "--wheel",
            str(state.private / "metric-final/wheels/fixture.whl"),
            "--baseline-installed",
            str(state.private / "metric-baseline/installed"),
            "--baseline-wheel",
            str(state.private / "metric-baseline/wheels/fixture.whl"),
            "--receipt",
            str(state.private / "metric-receipts/idle.json"),
        ]
    )
    assert (
        exit_code,
        state.calls,
        all(
            build.parent == source.parent
            and build != source
            and not build.is_relative_to(source)
            for source, build in state.builds
        ),
    ) == (int(damage == "boundary"), expected, True)


@pytest.mark.parametrize(
    "damage",
    [
        "custody",
        "prerequisite",
        "source",
        "idle-live",
        "containment-live",
        "install",
        "wheel",
        "installed",
        "idle-comparison",
        "idle-join",
    ],
)
def test_metric_phase_refuses_failure_or_drift_and_preserves_safe_progress(
    metric_phase, damage
):
    state = metric_phase
    state.damage = damage
    exit_code = runner._metric_phase(
        state.workspace, state.private, state.artifacts, "3" * 40
    )
    summary = json.loads((state.artifacts / "metric-summary.json").read_text())
    expected_calls = (
        0
        if damage == "install"
        else 1
        if damage
        in ("custody", "prerequisite", "source", "idle-live", "containment-live")
        else 5
    )
    assert (
        exit_code,
        len(state.calls),
        summary["exit_code"],
        len(summary["calls"]),
        all(call["status"] == "completed" for call in summary["calls"]),
        any(p.suffix == ".log" for p in state.artifacts.iterdir()),
    ) == (1, expected_calls, 1, expected_calls, True, False)


@pytest.mark.parametrize("program", ["idle", "containment"])
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_metric_phase_refuses_live_program_before_import(metric_phase, program, damage):
    state = metric_phase
    name = (
        "Helper_Scripts/Benchmarks/backup_admission_idle_benchmark.py"
        if program == "idle"
        else "tldw_chatbook/Notes/git_process_containment.py"
    )
    path = state.workspace / name
    path.unlink() if damage == "missing" else path.write_text("changed program")
    exit_code = runner._metric_phase(
        state.workspace, state.private, state.artifacts, "3" * 40
    )
    assert (exit_code, state.imported, state.calls) == (1, [], [])


def test_shared_outer_budget_includes_preparation(tmp_path, monkeypatch):
    copy_metric_programs(tmp_path)
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda *args: SimpleNamespace(free=3 * 1024**3)
    )
    clock, deadlines = [100.0], []
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])

    def preflight(*args):
        clock[0] += 20
        return {}

    def private(*args):
        clock[0] += 30
        return tmp_path

    monkeypatch.setattr(runner, "_metric_preflight", preflight)
    monkeypatch.setattr(runner, "_create_private_root", private)
    monkeypatch.setattr(runner, "_private_environment", lambda *args: {})
    module = SimpleNamespace(ProcessTreeController=lambda: object())
    spec = SimpleNamespace(
        name="metric_containment",
        loader=SimpleNamespace(exec_module=lambda *args: None),
    )
    monkeypatch.setattr(
        runner.importlib.util, "spec_from_file_location", lambda *args: spec
    )
    monkeypatch.setattr(runner.importlib.util, "module_from_spec", lambda *args: module)
    monkeypatch.setitem(runner.sys.modules, "metric_containment", module)

    async def supervise(*args):
        deadlines.append((args[-1], args[-1] - clock[0]))
        return {"exit_code": 1, "supervisor_retired": True}

    monkeypatch.setattr(runner, "_metric_supervise", supervise)
    runner._run_admission_metrics(tmp_path, tmp_path / "evidence", "3" * 40)
    assert deadlines == [(4900, 4750)]
