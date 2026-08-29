import shlex
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_LEASE_TEST_TARGETS = (
    "Tests/Model_Artifacts/test_operation_leases.py",
    "Tests/Model_Artifacts/test_operation_leases_process.py",
)
def _workflow_text() -> str:
    return (PROJECT_ROOT / ".github" / "workflows" / "test.yml").read_text()


def _nightly_workflow_text() -> str:
    return (PROJECT_ROOT / ".github" / "workflows" / "nightly-deep.yml").read_text()


def _all_tests_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  all-tests:")
    end = workflow.index("  test-summary:", start)
    return workflow[start:end]


def _core_tests_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  core-tests:")
    end = workflow.index("  artifact-lease-spike:", start)
    return workflow[start:end]


def _nightly_deep_job_block() -> str:
    workflow = _nightly_workflow_text()
    start = workflow.index("  nightly-deep:")
    return workflow[start:]


def _textual_minimum_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  textual-minimum:")
    end = workflow.index("  all-tests:", start)
    return workflow[start:end]


def _artifact_lease_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  artifact-lease-spike:")
    shape_start = workflow.find("  artifact-lease-shape:", start)
    end = (
        shape_start
        if shape_start != -1
        else workflow.index("  artifact-lease-gate:", start)
    )
    return workflow[start:end]


def _artifact_lease_shape_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  artifact-lease-shape:")
    end = workflow.index("  artifact-lease-gate:", start)
    return workflow[start:end]


def _artifact_lease_gate_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  artifact-lease-gate:")
    end = workflow.index("  ui-tests:", start)
    return workflow[start:end]


def _ui_tests_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  ui-tests:")
    end = workflow.index("  textual-minimum:", start)
    return workflow[start:end]


def _test_summary_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  test-summary:")
    return workflow[start:]


def _pytest_invocations(block: str) -> list[list[str]]:
    lines = iter(block.splitlines())
    pytest_invocations: list[list[str]] = []

    for raw_line in lines:
        command = raw_line.strip()
        if command != "pytest" and not command.startswith("pytest "):
            continue

        while command.endswith("\\"):
            command = f"{command[:-1].rstrip()} {next(lines).strip()}"
        pytest_invocations.append(shlex.split(command))

    return pytest_invocations


def _assert_artifact_lease_test_targets(block: str) -> None:
    pytest_invocations = _pytest_invocations(block)

    assert len(pytest_invocations) == 1
    test_targets = tuple(
        token.removeprefix("./")
        for token in pytest_invocations[0][1:]
        if token.removeprefix("./") == "Tests"
        or token.removeprefix("./").startswith("Tests/")
    )
    assert test_targets == ARTIFACT_LEASE_TEST_TARGETS


def test_ci_installs_pytest_timeout_for_configured_test_timeouts() -> None:
    requirements = (PROJECT_ROOT / "requirements-test.txt").read_text()

    assert "pytest-timeout" in requirements


def test_ci_installs_distribution_build_dependencies() -> None:
    requirements = (PROJECT_ROOT / "requirements-test.txt").read_text()

    assert "build" in requirements.splitlines()
    assert "setuptools>=77" in requirements.splitlines()


def test_pytest_ui_marker_is_registered_for_ci_marker_selection() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()

    assert '"ui: marks tests as UI/Textual tests"' in pyproject


def test_full_suite_job_is_bounded_and_manual_only() -> None:
    all_tests_job = _all_tests_job_block()

    assert "timeout-minutes:" in all_tests_job
    assert "if: github.event_name == 'workflow_dispatch'" in all_tests_job
    assert "pull_request" not in all_tests_job
    assert "name: Full Test Suite (Manual)" in all_tests_job
    assert "pytest ./Tests/" in all_tests_job


def test_jobs_running_architecture_tests_fetch_pinned_history() -> None:
    """Keep immutable architecture baselines available in every full test job.

    The Wave 6 inventory reads a pinned source blob with ``git show``. GitHub's
    default depth-one checkout does not contain that historical commit.
    """
    for block in (
        _core_tests_job_block(),
        _all_tests_job_block(),
        _nightly_deep_job_block(),
    ):
        checkout_start = block.index("uses: actions/checkout@v4")
        checkout_end = block.index("Set up Python", checkout_start)
        assert "fetch-depth: 0" in block[checkout_start:checkout_end]


def test_ci_exercises_mcp_against_minimum_textual() -> None:
    textual_minimum = _textual_minimum_job_block()
    test_summary = _test_summary_job_block()

    # Floor check must install the declared minimum supported Textual
    # (pinned in pyproject.toml, currently ==8.2.8 per TASK-1353/1362).
    assert 'pip install "textual==8.2.8"' in textual_minimum
    assert "Tests/CI/test_textual_runtime_contract.py" in textual_minimum
    assert "Tests/UI/test_mcp_workbench.py" in textual_minimum
    assert "Tests/UI/test_mcp_tools_mode.py" in textual_minimum
    assert (
        "needs: [core-tests, ui-tests, textual-minimum, artifact-lease-gate]"
        in test_summary
    )


def test_artifact_lease_spike_runs_three_os_on_main_and_manual_events() -> None:
    """The comprehensive workflow retains cross-platform lease evidence."""
    block = _artifact_lease_job_block()

    assert "os: [ubuntu-latest, macos-latest, windows-latest]" in block
    assert "pull_request" not in block
    assert 'python-version: ["3.11"]' in block
    assert "pip install -e ." in block
    assert "pip install -r requirements-test.txt" in block
    assert (
        "- name: Prove cross-platform operation leases\n"
        "      shell: bash\n"
        "      run:" in block
    )
    _assert_artifact_lease_test_targets(block)


def test_artifact_lease_target_check_rejects_unrelated_explicit_test() -> None:
    block = _artifact_lease_job_block()
    mutated = block.replace(
        "Tests/Model_Artifacts/test_operation_leases_process.py -v",
        "Tests/Model_Artifacts/test_operation_leases_process.py \\\n"
        "          Tests/Other/test_unrelated.py -v",
    )

    assert mutated != block
    with pytest.raises(AssertionError):
        _assert_artifact_lease_test_targets(mutated)


def test_ci_shape_regression_runs_in_dedicated_comprehensive_job() -> None:
    """Keep the CI shape regression in its dedicated comprehensive job."""
    workflow = _workflow_text()

    assert "  artifact-lease-shape:" in workflow
    shape = _artifact_lease_shape_job_block()
    install_commands = [
        line.strip()
        for line in shape.splitlines()
        if line.strip().startswith(("pip install ", "python -m pip install "))
    ]

    assert "runs-on: ubuntu-latest" in shape
    assert "github.event_name" not in shape
    assert "uses: actions/checkout@v4" in shape
    assert "uses: actions/setup-python@v5" in shape
    assert 'python-version: "3.11"' in shape
    assert install_commands == ["python -m pip install pytest pytest-timeout"]
    assert _pytest_invocations(shape) == [
        [
            "pytest",
            "Tests/CI/test_github_actions_test_workflow.py",
            "--confcutdir=Tests/CI",
        ]
    ]


def test_artifact_lease_gate_exposes_stable_required_context() -> None:
    gate = _artifact_lease_gate_job_block()
    test_summary = _test_summary_job_block()

    assert "name: Artifact Lease Gate" in gate
    assert "runs-on: ubuntu-latest" in gate
    assert "needs: [artifact-lease-spike, artifact-lease-shape]" in gate
    assert "if: ${{ always() }}" in gate
    assert (
        'if [ "${{ needs.artifact-lease-spike.result }}" != "success" ] || '
        '[ "${{ needs.artifact-lease-shape.result }}" != "success" ]; then' in gate
    )
    assert "exit 1" in gate
    assert "artifact-lease-gate" in test_summary


def test_comprehensive_shards_cover_the_whole_tree_in_parallel() -> None:
    """task-1465: core+ui shards replace the 27-file `-m unit` selection."""
    workflow = _workflow_text()

    assert "  core-tests:" in workflow
    assert "pytest Tests --ignore=Tests/UI" in workflow
    assert workflow.count("-n auto --dist loadscope --max-worker-restart=3") >= 2
    # The fake stratification and its duplicate full-run workflow are gone.
    assert "pytest -m unit" not in workflow
    assert "pytest -m integration" not in workflow
    assert not (PROJECT_ROOT / ".github" / "workflows" / "python-app.yml").exists()


def test_ui_job_is_sharded_to_fit_its_time_budget() -> None:
    """TASK-18608: one job could never finish the UI suite.

    The full Tests/UI directory needs ~5.8 serial-equivalent hours on a
    standard runner; the job budget is 45 minutes, so the unsharded job was
    cancelled at ~11% on every run, on every branch (100+ consecutive red
    runs). The fix is a matrix of pytest-shard slices -- the contract here
    is that the split stays deterministic (pytest-shard, not xdist-only,
    which is per-job randomness), covers every test exactly once across the
    matrix, and each slice still parallelizes internally with xdist.
    """
    ui_job = _ui_tests_job_block()

    assert "pytest-shard" in (PROJECT_ROOT / "requirements-test.txt").read_text()
    assert "--shard-id=${{ matrix.shard }}" in ui_job
    # The ids and the divisor must describe the same complete partition:
    # 12 shards numbered 0..11. A mismatch (e.g. ids 1..12 against
    # num-shards 12, or a stale id list after resizing) would silently
    # duplicate or drop a slice of the suite.
    # Read the ids from the UI JOB, not the whole workflow. This used to
    # search `workflow` for the first "shard: [" -- which was the UI job only
    # because it was then the only sharded job. TASK-21411 sharded core-tests
    # too, and core's ids were suddenly being checked against the UI job's
    # divisor. A partition assertion that can silently start describing a
    # different job is worse than no assertion.
    id_values = _shard_ids(ui_job)
    divisor = int(ui_job.split("--num-shards=")[1].split()[0].rstrip("'\""))
    assert id_values == list(range(divisor)), "shard ids must be 0..N-1"
    assert divisor >= 10, (
        "the UI suite needs ~5.8h serial; a 45-minute job budget needs at "
        "least ~10 deterministic shards (~35 min each) to finish"
    )
    # xdist still parallelizes WITHIN each shard.
    assert "-n auto --dist loadscope" in ui_job
    # Every shard uploads its report, and the names must be per-shard so
    # matrix siblings cannot overwrite each other's artifacts.
    assert "ui-test-results-${{ matrix.shard }}.json" in ui_job
    assert "name: ui-test-results-${{ matrix.shard }}" in ui_job


def _shard_ids(job_block: str) -> list[int]:
    """The `shard: [...]` id list declared inside one job block."""
    line = job_block[job_block.index("shard: [") :].splitlines()[0]
    return [
        int(part.strip())
        for part in line[line.index("[") + 1 : line.rindex("]")].split(",")
    ]


def test_core_tests_job_is_sharded_to_fit_its_time_budget() -> None:
    """TASK-21411: one job could not finish the core suite either.

    Diagnosed from a killed run's own log rather than from the outcome: the
    ubuntu leg started pytest at 15:44:55 and was still emitting progress
    steadily when the 120-minute cap killed it at 17:39, having reached 60%.
    Steady progress at the kill is what separates "too slow" from "hung" --
    the suite needs ~190 minutes on a 4-vCPU runner, so no timeout below
    GitHub's 6-hour ceiling makes one job the right container.

    The contract mirrors the UI job's: a deterministic pytest-shard split
    covering every test exactly once, xdist still parallelizing within each
    slice, and per-shard artifact names so matrix siblings cannot overwrite
    each other's report.
    """
    core = _core_tests_job_block()

    assert "--shard-id=${{ matrix.shard }}" in core
    divisor = int(core.split("--num-shards=")[1].split()[0].rstrip("'\""))
    assert _shard_ids(core) == list(range(divisor)), "shard ids must be 0..N-1"
    assert divisor >= 4, (
        "the core suite needs ~190 minutes on a standard runner; fewer than "
        "~4 shards puts a slice back within reach of the 120-minute cap"
    )
    assert "-n auto --dist loadscope" in core
    assert "core-test-results-${{ matrix.shard }}.json" in core
    assert "name: core-test-results-${{ matrix.shard }}" in core


def test_core_tests_job_does_not_multiply_the_scarcest_runner_pool() -> None:
    """macOS breadth belongs to nightly-deep, not to the core shard matrix.

    macOS runners are the constrained pool here -- queue waits of 42 to 90
    minutes were observed while ubuntu jobs started promptly -- so sharding
    core across them would spend more in queueing than it buys. This pins the
    intent, and pins that the coverage it gives up still exists elsewhere.
    """
    core = _core_tests_job_block()
    assert "runs-on: ubuntu-latest" in core
    assert "macos" not in core
    assert "macos-latest" in _nightly_deep_job_block()


def test_core_tests_job_budget_covers_the_suite() -> None:
    """TASK-18608: 60 minutes stopped being enough for the core suite.

    The ubuntu leg was killed by its own timeout mid-run while the macOS
    leg finished at ~58m, so the budget must sit clearly above the
    observed worst leg, not at it.
    """
    core = _core_tests_job_block()

    line = next(
        candidate.strip()
        for candidate in core.splitlines()
        if candidate.strip().startswith("timeout-minutes:")
    )
    assert int(line.split(":")[1]) >= 120


def test_nightly_deep_runs_the_tiers_the_fast_pr_lane_does_not() -> None:
    """task-1465: serial + thorough + --run-slow + cache-off + breadth, on dev."""
    workflow = _nightly_workflow_text()
    nightly = _nightly_deep_job_block()

    assert "- cron: '30 8 * * *'" in workflow
    assert "workflow_dispatch:" in workflow
    assert "ref: ${{ needs.resolve-dev-sha.outputs.sha }}" in nightly
    assert "--run-slow" in nightly
    assert "TLDW_HYPOTHESIS_PROFILE: thorough" in nightly
    assert 'TLDW_TEST_CSS_CACHE: "0"' in nightly
    assert "-n auto" not in nightly  # serial on purpose: order-regression canary
    assert "windows-latest" in nightly
    assert "macos-latest" in nightly


def test_every_json_report_invocation_omits_log_capture() -> None:
    """Every ``--json-report`` MUST carry ``--json-report-omit log``.

    Incident (2026-08-20, TASK-19160): pytest-json-report captures RAW
    ``logging.LogRecord`` objects per test and attaches them to the report
    (``report._json_report_extra``). The websockets library logs through a
    ``LoggerAdapter`` whose extra carries the LIVE connection object, so when
    a realtime test failed on the runner with DEBUG-level logging left on by
    an earlier test in the same xdist worker, execnet hit
    ``DumpError: can't serialize <class 'websockets...ServerConnection'>``
    while shipping the report — killing the worker and aborting BOTH Core
    jobs with INTERNALERROR. One flaky test nuked the whole job's results.

    Reproduced deterministically (failing realtime test + ``--log-level=DEBUG``
    + xdist + json-report) and verified fixed by the omit flag. Nothing
    consumes the log section: ``generate_test_summary.py`` reads only
    ``tests[].outcome`` and ``call.longrepr``.
    """
    # Join shell continuation lines into logical commands first, so the
    # activation flag and its omit argument may legally sit on different
    # physical lines; then accept every valid spelling of the omit
    # ("--json-report-omit log", "--json-report-omit=log", comma lists).
    logical = (_workflow_text() + _nightly_workflow_text()).replace("\\\n", " ")
    activations = 0
    for command in logical.splitlines():
        tokens = command.split()
        if "--json-report" not in tokens:
            continue
        activations += 1
        omits: list[str] = []
        for index, token in enumerate(tokens):
            if token == "--json-report-omit" and index + 1 < len(tokens):
                omits.extend(tokens[index + 1].split(","))
            elif token.startswith("--json-report-omit="):
                omits.extend(token.split("=", 1)[1].split(","))
        assert "log" in omits, (
            f"test.yml: '--json-report' without log in '--json-report-omit' "
            f"— raw LogRecords in reports crash xdist workers (see "
            f"docstring): {command.strip()[:120]!r}"
        )
    assert activations >= 4, (
        f"expected at least 4 json-report activations (core, UI, full, "
        f"nightly); found {activations} — the pin may have gone inert"
    )


def test_the_test_summary_job_can_actually_fail() -> None:
    """TASK-21411: `needs:` + `if: always()` schedules a job; it does not judge one.

    On run 32647831275 the Test Summary check reported success while all
    twelve UI shards were red -- it ran after them, read their artifacts, and
    never looked at their conclusions. Since this is the check most likely to
    be marked required, a summary that cannot go red is worse than no summary.

    The verdict must also be the last step, after summary generation.
    """
    summary = _test_summary_job_block()

    for job in ("core-tests", "ui-tests", "textual-minimum", "artifact-lease-gate"):
        assert f"needs.{job}.result != 'success'" in summary, (
            f"the summary does not fail when {job} fails"
        )
    assert "exit 1" in summary

    step_names = [
        line.split("- name:", 1)[1].strip()
        for line in summary.splitlines()
        if line.strip().startswith("- name:")
    ]
    assert step_names[-1] == "Require every gated job to have succeeded", (
        "the verdict must be the final step, after the summary is generated; "
        f"steps end with {step_names[-2:]}"
    )
