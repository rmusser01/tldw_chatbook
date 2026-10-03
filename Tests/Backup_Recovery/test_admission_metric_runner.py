"""Finite trust, receipt and custody controls for Windows metric invocation."""

import asyncio
import copy
import subprocess
import time
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
                "exit_code": 0,
                "seed": copy.deepcopy(child),
                "warmup": copy.deepcopy(child) if phase == "boot" else None,
                "runs": [
                    copy.deepcopy(child) for _ in range(3 if phase == "boot" else 1)
                ],
            }
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
