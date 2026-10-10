"""TASK-19572: the shape of the one CI check that is meant to be required.

The `Tests` workflow has produced no verdict since 2026-06-26 -- 200-227 minute
runtime against 23-50 merges/day on `dev`, with cancel-in-progress killing every
in-flight run. `derived-artifacts.yml` is the replacement gate: install-free,
~90 s, and safe to mark as a required status check.

These tests pin the properties that make it requireable at all. They are
deliberately shape-only (the workflow cannot be executed here), and every
assertion below corresponds to a way the gate would silently stop gating.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "derived-artifacts.yml"
LANES = (
    "github.event_name == 'pull_request' || (github.event_name == 'workflow_dispatch' && "
    "(inputs.pr != '' || github.ref != 'refs/heads/dev'))"
)
CHECKERS = (
    "tldw_chatbook/css/check_bundle_sync.py",
    "scripts/check_canvas_mermaid_assets.py",
    "scripts/check_profile_owned_path_inventory.py",
    "scripts/check_persistent_diagnostic_inventory.py",
    "scripts/check_backlog_task_ids.py",
    "scripts/check_backlog_task_files.py",
    # TASK-20971. VALID_TABLES['chachanotes'] went stale, was repaired, and
    # went stale again 14.5 hours later; this is its authoring-time half.
    "scripts/check_schema_table_allowlist.py",
    # TASK-21593. Every shipped database index must have an explicit query-plan
    # decision instead of being assumed useful because it exists.
    "scripts/check_index_plan_pins.py",
    # TASK-32800.4. Three of the four P0s in the 2026-09-17 core-runtime review
    # were one defect class with no guard: a synchronous callable handed to
    # run_worker, and a query_one resuming into a removed subtree.
    "scripts/check_textual_worker_contract.py",
    # TASK-32803.1 / ADR-173. Timestamp writers must go through the shared UTC
    # helper: datetime.utcnow() is forbidden and naive datetime.now().isoformat()
    # is ratcheted.
    "scripts/check_timestamp_writers.py",
    # TASK-32908. Tests/UI gated nothing on a PR; a verified-green subset now
    # runs in the ui-fast-lane job, and this checker is what stops that subset
    # from being quietly edited away instead of fixed.
    "scripts/check_ui_pr_gate_census.py",
)


#: Each lane the required aggregate depends on, and its verdict step.
VERDICTS = {
    "pr-fast-lane": "Require successful PR fast lane",
    "ui-fast-lane": "Require successful UI fast lane",
    "console-p0-gate": "Require successful Console P0 regression gate",
}


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _job() -> dict:
    return _workflow()["jobs"]["derived-artifacts"]


def _steps() -> list[dict]:
    return _job()["steps"]


def test_workflow_has_one_fast_prerequisite_and_one_required_aggregator():
    """The added lanes must not create a second branch-protection context.

    TASK-32908 added `ui-fast-lane` beside `pr-fast-lane`. Both are ordinary
    jobs that report their own square; neither is named under branch
    protection. `derived-artifacts` stays the single required context, and
    `needs` + its two verdict steps are what make a red lane fail it -- see
    `test_required_aggregator_fails_when_either_lane_fails`.

    `queue-tick` (the merge queue, spec 2026-10-03) runs after the aggregate and
    is never a required context.
    """
    assert list(_workflow()["jobs"]) == [
        "pr-fast-lane",
        "ui-fast-lane",
        # TASK-33621.27: the Console review's P0 regression tests, moved out
        # of pr-fast-lane when they took it to 33m41s of its 35-min cap.
        "console-p0-gate",
        "derived-artifacts",
        "queue-tick",
    ]


def test_required_aggregator_fails_when_either_lane_fails():
    """A lane that goes red must take the REQUIRED check down with it.

    Without a verdict step a failing lane shows its own red square while
    `Derived artifacts reproduce from their sources` stays green -- the
    "guard nobody is gated on" shape this whole workflow exists to replace.
    """
    job = _workflow()["jobs"]["derived-artifacts"]
    assert job.get("needs") == ["pr-fast-lane", "ui-fast-lane", "console-p0-gate"]
    for lane, step_name in VERDICTS.items():
        verdict = next(step for step in job["steps"] if step.get("name") == step_name)
        # Round 4: `!cancelled()` so one red lane cannot skip the next verdict.
        assert verdict["if"] == (
            f"${{{{ !cancelled() && ({LANES}) && needs.{lane}.result != 'success' }}}}"
        )
        assert "exit 1" in verdict["run"]


def test_ui_fast_lane_runs_the_census_in_serial_round_robin_shards():
    """TASK-32908: the run CI performs must be the run the census was verified
    against.

    The census was verified serially, in file order, on the same minimal
    dependency set pr-fast-lane installs. xdist, a different plugin set, or a
    different order would all be untested configurations for a gate whose
    entire value is that it is green -- and a gate that lands red trains
    people to ignore it.

    TASK-34353: the serial census outgrew the job's 20-minute cap, so it runs
    as parallel round-robin shards picked by the census checker -- each one a
    subsequence of the census, in census order -- never an xdist split.
    """
    job = _workflow()["jobs"]["ui-fast-lane"]

    assert job["if"] == LANES
    strategy = job["strategy"]
    assert strategy["fail-fast"] is False  # one red shard must not hide another
    assert list(strategy["matrix"]) == ["shard"]
    assert len(strategy["matrix"]["shard"]) >= 2

    install = next(
        step
        for step in job["steps"]
        if step.get("name") == "Install fast-lane dependencies"
    )["run"]
    assert "requirements-test.txt" not in install
    assert "pytest-xdist" not in install
    assert ".[" not in install

    run = next(
        step
        for step in job["steps"]
        if step.get("name") == "Run the gated Tests/UI slice"
    )["run"]
    # The census checker reads scripts/ui_pr_gate_census.txt and picks the shard, sized by the matrix; its output goes
    # through a file (so its failure fails the step) and an empty shard is
    # refused (bare `pytest` would collect the whole tree).
    assert "scripts/check_ui_pr_gate_census.py --shard" in run
    assert '"${{ strategy.job-index }}" "${{ strategy.job-total }}"' in run
    assert 'mapfile -t SHARD < "$RUNNER_TEMP/ui-shard.txt"' in run
    assert 'test "${#SHARD[@]}" -gt 0' in run
    assert 'pytest "${SHARD[@]}"' in run
    assert "-n auto" not in run and "--dist" not in run
    assert "-p no:randomly" not in run  # not installed; order is collection order


@pytest.mark.parametrize("total", [1, 2, 3, 4, 5])
def test_ui_gate_shards_cover_the_census_once_each_in_census_order(total):
    """TASK-34353: the shards the UI lane runs are exactly the census.

    Drives the real `--shard` command the workflow runs, for each shard of a
    `total`-way split: every census file lands in exactly one shard, no shard
    is empty, and each shard keeps census order.
    """
    import subprocess
    import sys

    checker = PROJECT_ROOT / "scripts" / "check_ui_pr_gate_census.py"
    census = [
        line.strip()
        for line in (PROJECT_ROOT / "scripts" / "ui_pr_gate_census.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    shards = [
        subprocess.run(
            [sys.executable, str(checker), "--shard", str(index), str(total)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        for index in range(total)
    ]

    assert all(shards)
    assert sorted(path for shard in shards for path in shard) == sorted(census)
    for shard in shards:
        assert shard == [path for path in census if path in shard]


def test_ui_gate_shard_refuses_an_index_outside_the_split():
    """A mistyped matrix must fail the step, not silently gate nothing."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_ui_pr_gate_census",
        PROJECT_ROOT / "scripts" / "check_ui_pr_gate_census.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError):
        module.shard(["a", "b"], 2, 2)


def test_ui_gate_census_is_non_empty_and_every_entry_exists():
    """A census of renamed-away paths collects nothing and still exits 0."""
    census = PROJECT_ROOT / "scripts" / "ui_pr_gate_census.txt"
    entries = [
        line.strip()
        for line in census.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert entries, "the gated Tests/UI census must not be empty"
    assert len(entries) == len(set(entries))
    for entry in entries:
        assert entry.startswith("Tests/UI/")
        # TASK-33621.27: an entry may be one test's node id (`file::test`).
        file_part = entry.split("::", 1)[0]
        assert (PROJECT_ROOT / file_part).is_file(), f"censused file is gone: {entry}"


def test_triggers_are_not_path_filtered():
    """A path-filtered required check never reports and blocks the PR forever.

    GitHub leaves a skipped required check on "Expected - waiting for status to
    be reported", so a docs-only PR would be unmergeable. At ~90 s the job is
    cheap enough to run unconditionally; adding `paths:` is the one edit that
    would brick merges without failing anything.
    """
    triggers = _workflow()[True]  # PyYAML parses the bare `on:` key as True
    assert set(triggers) == {"pull_request", "push", "workflow_dispatch"}
    for event, config in triggers.items():
        assert not (config or {}).get("paths"), f"{event} must not be path-filtered"
        assert not (config or {}).get("paths-ignore"), f"{event} must not path-ignore"


def test_every_checker_runs():
    """Every registered checker is invoked, so one job covers the census."""
    script = "\n".join(step.get("run", "") for step in _steps())
    for checker in CHECKERS:
        assert checker in script, f"{checker} is not run by the required job"


def test_local_preflight_runs_the_same_checker_inventory_in_order():
    """Local authoring checks must not silently omit a required CI checker."""
    preflight = (PROJECT_ROOT / "scripts" / "preflight.sh").read_text(encoding="utf-8")

    positions = [preflight.index(checker) for checker in CHECKERS]

    assert positions == sorted(positions)


def test_checker_steps_survive_an_earlier_failure():
    """One red checker must not hide the others.

    With the default `success()` condition the first failure skips the rest, so
    a burn-down needs one push per checker. `!cancelled()` reports all of the
    drift in a single run while still failing the job.
    """
    checker_steps = [step for step in _steps() if "python " in step.get("run", "")]
    assert len(checker_steps) == len(CHECKERS)
    for step in checker_steps:
        assert "cancelled()" in str(
            step.get("if", "")
        ), f"step {step.get('name')!r} would be skipped after an earlier failure"


def test_job_installs_nothing():
    """Install-free is what keeps this at ~90 s; a pip install re-creates the
    runtime that made the Tests workflow unusable."""
    job = _job()
    assert "pip install" not in yaml.safe_dump(job)
    for step in job["steps"]:
        uses = step.get("uses", "")
        assert not uses.startswith("actions/setup-python") or "cache" not in (
            step.get("with") or {}
        ), "no pip cache is needed when nothing is installed"


def test_derived_artifact_checkers_use_mermaid_builder_python_pin():
    setup = next(
        step for step in _steps() if step.get("uses") == "actions/setup-python@v5"
    )

    assert setup["with"] == {"python-version": "3.12.11"}


def test_required_check_name_is_stable():
    """Renaming this silently detaches branch protection from the job."""
    assert _job()["name"] == "Derived artifacts reproduce from their sources"


def test_dispatch_inputs_name_the_pr_or_the_run_to_wait_for():
    """A manual full-gate dispatch names its PR. A queue kick names none, so the lanes skip; the
    queue's own kick names the run its tick waits for (spec V4: the queue wakes through this file,
    the only queue file on main, so the only one GitHub will dispatch)."""
    dispatch = _workflow()[True]["workflow_dispatch"]
    assert dispatch["inputs"]["pr"] == {"description": "PR number (manual full-gate dispatch)", "required": False,
                                        "type": "string", "default": ""}
    assert dispatch["inputs"]["wait_run"] == {
        "description": "A run the queue tick waits for before deciding (set by the merge queue)",
        "required": False, "type": "string", "default": ""}


def test_queue_tick_runs_after_ci_and_is_never_required():
    jobs = _workflow()["jobs"]
    tick = jobs["queue-tick"]
    assert tick["needs"] == ["derived-artifacts"]
    assert "queue-tick" not in jobs["derived-artifacts"].get("needs", [])
    assert tick["if"].startswith("!cancelled() &&")
    assert "vars.MERGE_QUEUE == 'dry' || vars.MERGE_QUEUE == 'on'" in tick["if"]
    assert "github.event_name == 'workflow_dispatch'" in tick["if"]
    assert "auto_merge" not in tick["if"], (
        "the payload's auto_merge is a trigger-time snapshot; disarm-push-rearm leaves it null, "
        "so gating on it means red author runs never wake the queue"
    )
    assert "github.event.pull_request.head.repo.full_name == github.repository" in tick["if"]
    assert "push" not in tick["if"], "pushes to dev are merge-queue.yml's job"
    assert tick["permissions"] == {"contents": "write", "pull-requests": "write", "actions": "write"}
    assert _workflow()["permissions"] == {"contents": "read"}
    checkout = tick["steps"][0]
    assert checkout["uses"] == "actions/checkout@v4" and checkout["with"] == {
        "ref": "dev",
        "persist-credentials": False,
    }
    assert tick["steps"][1]["run"] == "python3 scripts/merge_queue.py"
    assert tick["steps"][1]["env"] == {"GH_TOKEN": "${{ github.token }}", "MERGE_QUEUE": "${{ vars.MERGE_QUEUE }}",
                                       "WAIT_RUN": "${{ inputs.wait_run }}"}


def test_branch_dispatch_without_pr_runs_the_gate():
    """A no-`pr` dispatch off `dev` must still run the lanes, or the required check

    can be greened with zero tests run (spike F2: a dispatched run counts as the
    PR's required context). Only a no-`pr` dispatch ON `dev` is the cheap manual
    queue kick.
    """
    workflow = _workflow()
    for job_name in VERDICTS:
        assert workflow["jobs"][job_name]["if"] == LANES
    for step_name in VERDICTS.values():
        step = next(
            step
            for step in workflow["jobs"]["derived-artifacts"]["steps"]
            if step.get("name") == step_name
        )
        assert "github.ref != 'refs/heads/dev'" in step["if"]


def _pytest_step_targets() -> list[tuple[str, tuple[str, ...]]]:
    import shlex

    steps = []
    for job_name in ("pr-fast-lane", "console-p0-gate"):
        for step in _workflow()["jobs"][job_name]["steps"]:
            run = step.get("run", "")
            if not run.lstrip().startswith("pytest"):
                continue
            tokens = shlex.split(run.replace("\\\n", " "))
            targets = tuple(t for t in tokens[1:] if t == "Tests" or t.startswith("Tests/"))
            steps.append((f"{job_name}: {step['name']}", targets))
    return steps


def _census_checker():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "census_overlap", PROJECT_ROOT / "scripts" / "check_ui_pr_gate_census.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_pr_lane_pytest_step_has_disjoint_targets():
    """TASK-33621.27 review: pytest collapses overlapping arguments (ADR-103).

    With a file listed whole beside its own node ids, pytest collected 1 of
    its 21 tests and the step stayed green. The census checker refuses that
    for the UI lane; this is the same rule, through the same helper, for
    every pytest step of the PR lanes: no node id beside its whole file, no
    target under a listed directory, nothing listed twice.
    """
    steps = _pytest_step_targets()
    assert len(steps) == 4, [name for name, _ in steps]
    overlapping = _census_checker().overlapping_targets
    for name, targets in steps:
        assert targets, f"{name} lists no targets"
        problems = overlapping(targets)
        assert not problems, f"{name}:\n" + "\n".join(problems)


def test_every_node_id_target_in_a_pr_lane_step_resolves():
    """A renamed test or parametrize id makes pytest exit 4 with "not found"."""
    resolve = _census_checker().resolve_node
    for name, targets in _pytest_step_targets():
        for target in targets:
            file_part, _, node = target.partition("::")
            assert (PROJECT_ROOT / file_part).exists(), f"{name}: {file_part} is gone"
            if node:
                reason = resolve(PROJECT_ROOT / file_part, node)
                assert reason is None, f"{name}: {target}: {reason}"
