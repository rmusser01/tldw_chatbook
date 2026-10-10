"""Contracts for the bounded fast PR lane and comprehensive CI cadence."""

from __future__ import annotations

import copy
import shlex
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = PROJECT_ROOT / ".github" / "workflows"
PULL_REQUEST_TYPES = ["opened", "synchronize", "reopened", "ready_for_review"]
PUSH_ONLY_CANCELLATION = (
    "${{ github.event_name == 'push' && github.ref != 'refs/heads/main' }}"
)
LANES = (
    "github.event_name == 'pull_request' || (github.event_name == 'workflow_dispatch' && "
    "(inputs.pr != '' || github.ref != 'refs/heads/dev'))"
)
FAST_LANE_TARGETS = (
    "Tests/CI",
    "Tests/test_smoke.py",
    "Tests/MCP/test_approval_timeout_policy.py",
    "Tests/MCP/test_control_plane_bridge.py",
    "Tests/MCP/test_live_server_request_wiring.py",
    "Tests/Agents/test_execution_capacity.py",
    "Tests/Agents/test_fleet_messages.py",
    "Tests/Agents/test_session_todo_store.py",
    "Tests/DB/test_agent_orchestration_dev_migration.py",
    "Tests/DB/test_agent_run_budget_accounting.py",
    "Tests/DB/test_automatic_work_budget.py",
    "Tests/DB/test_automatic_runtime_owner.py",
    "Tests/DB/test_automatic_wake_attempts.py",
    "Tests/Model_Artifacts/test_operation_leases.py",
    "Tests/Model_Artifacts/test_operation_leases_process.py",
    "Tests/UI/test_mcp_workbench.py",
    "Tests/UI/test_mcp_tools_mode.py",
    "Tests/Widgets/test_detach_safe_text_area.py",
    "Tests/Architecture/test_console_controllers_define_their_self_attributes.py",
    # TASK-34000.1: a Screen that flushes on navigation must answer the quit
    # walk; quit prompts are awaited only through the choke point; the Notes
    # autosave max-wait arithmetic. Static AST scans (one cached tree walk
    # each) and pure unit tests: a few seconds for the lane in total.
    "Tests/Architecture/test_flush_screens_have_quit_hooks.py",
    "Tests/Architecture/test_quit_flow_prompt_choke_point.py",
    "Tests/Library/test_library_note_autosave_max_wait.py",
    # TASK-34000 wave 1a final review (I2): the fast pins for the export seam
    # (atomic write, no-clobber, symlink write-through, the Report and
    # Collections replace checks) and for every sync status publication
    # carrying its time. About 6 s together; nothing else gated them.
    "Tests/Library/test_library_file_export.py",
    "Tests/Architecture/test_notes_sync_snapshot_construction.py",
    # TASK-34000.25: the ``[<title>](media://<uuid>)`` source line a note
    # takes from a Media item is pure and refuses anything but a uuid
    # (under a second).
    "Tests/Library/test_library_media_source_link.py",
    # TASK-34000.25 fix round 2: the compose-time gate lookup reads Textual's
    # ``_pending_children``; a Textual change fails here by name (one bare
    # ``run_test``, about a second).
    "Tests/Library/test_library_notes_canvas_pending_lookup.py",
    # TASK-34000.48: every file-profile comparison and binding commit goes
    # through the newline rule (AST scan of the three sync modules, seconds).
    "Tests/Architecture/test_notes_sync_binding_profile_commits.py",
    # TASK-34000.49: binding.note_version is never a precondition against
    # the live note (AST scan of the executor, under a second).
    "Tests/Architecture/test_notes_sync_version_proxy_sites.py",
    # TASK-34100.5: the first-reply and handoff unit guards -- the direct
    # runtime's unavailable tools never reach the agent catalog, plain arrival
    # words, the missing-revision trace diagnostic, the provider's own reason
    # on a failed first reply, and the cold local first-token window. Pure
    # unit tests, a few seconds together.
    "Tests/Agents/test_mcp_unavailable_builtin_tools.py",
    "Tests/Chat/test_console_arrival_vocabulary.py",
    "Tests/Chat/test_console_trace_missing_revision_diagnostic.py",
    "Tests/Chat/test_first_reply_failure_copy.py",
    "Tests/Chat/test_first_token_window.py",
    "Tests/Architecture/test_surface_swap_guard.py",
    "Tests/Library/test_ingest_analysis_load_settings.py",
    "Tests/test_config_load_settings_table_guard.py",
    "Tests/Architecture/test_study_handler_service_keywords.py",
    # TASK-33621.27: the plain Console P0 regression files -- the Save .md
    # export seam, the trace row sources behind the refused sends, the wizard
    # lifecycle guard, and the compaction failure_reason column and its
    # repository contract. About 18 s together under load.
    "Tests/Console/test_console_markdown_export.py",
    "Tests/Chat/test_console_trace_row_sources.py",
    "Tests/Architecture/test_wizard_lifecycle_guards.py",
    "Tests/DB/test_chachanotes_v74_auxiliary_failure_reason.py",
)
#: The lasting-sync real-stack files (a real database, a real ``.md``, the
#: production runtime). They are ``bootstrap_profile``, so they run in the
#: admission-sensitive invocation, never in the sandboxed one above. Pinned as
#: a subset: that step also holds other teams' files.
NOTES_SYNC_REAL_STACK_TARGETS = (
    "Tests/Notes/test_notes_sync_tail_edit.py",
    "Tests/Notes/test_notes_sync_delete_restore_signal.py",
    "Tests/Notes/test_notes_sync_crlf_single_line.py",
    "Tests/Notes/test_notes_sync_source_moved_settle.py",
    "Tests/Notes/test_notes_sync_watcher_after_heal.py",
    "Tests/Notes/test_notes_sync_version_only_move.py",
    "Tests/Notes/test_notes_sync_resave_window.py",
    "Tests/Notes/test_notes_sync_attention_fence.py",
)
HEAVY_JOB_KEYS = {
    "core-tests",
    "artifact-lease-spike",
    "artifact-lease-shape",
    "artifact-lease-gate",
    "ui-tests",
    "textual-minimum",
    "all-tests",
    "test-summary",
    "backup-platform-windows",
}
STANDALONE_WORKFLOWS = (
    "derived-artifacts.yml",
    "perf-guard.yml",
)


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOW_ROOT / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True))


def _named_step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def _pytest_targets(run: str) -> tuple[str, ...]:
    tokens = shlex.split(run.replace("\\\n", " "))
    return tuple(
        token for token in tokens[1:] if token == "Tests" or token.startswith("Tests/")
    )


def _assert_required_aggregation(workflow: dict) -> None:
    fast = workflow["jobs"]["pr-fast-lane"]
    required = workflow["jobs"]["derived-artifacts"]
    assert not fast.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in fast["steps"])
    assert required["name"] == "Derived artifacts reproduce from their sources"
    # TASK-32908: ui-fast-lane joined the aggregation. Both lanes are ordinary
    # jobs; `derived-artifacts` remains the only branch-protection context, so
    # each lane needs its own verdict step below or a red lane would leave the
    # required check green.
    assert required.get("needs") == ["pr-fast-lane", "ui-fast-lane"]
    assert required["if"] == "${{ always() }}"
    assert not required.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in required["steps"])

    verdict = _named_step(required, "Require successful PR fast lane")
    assert not verdict.get("continue-on-error", False)
    assert verdict["if"] == f"${{{{ ({LANES}) && needs.pr-fast-lane.result != 'success' }}}}"
    assert "needs.pr-fast-lane.result" in verdict["run"]
    assert "exit 1" in verdict["run"]

    ui_verdict = _named_step(required, "Require successful UI fast lane")
    assert not ui_verdict.get("continue-on-error", False)
    assert ui_verdict["if"] == f"${{{{ ({LANES}) && needs.ui-fast-lane.result != 'success' }}}}"
    assert "needs.ui-fast-lane.result" in ui_verdict["run"]
    assert "exit 1" in ui_verdict["run"]

    # The UI lane is bounded the same way the fast lane is: serial jobs,
    # minimal install, its own timeout. TASK-32908 put it in its own job
    # precisely so it cannot eat pr-fast-lane's 30-minute budget; TASK-34353
    # split it into contiguous shards so the census fits that timeout. A
    # matrix job's `needs.<job>.result` is success only when every shard is.
    ui = workflow["jobs"]["ui-fast-lane"]
    assert ui["runs-on"] == "ubuntu-latest"
    assert list(ui["strategy"]["matrix"]) == ["shard"]
    assert ui["timeout-minutes"] <= 20
    assert not ui.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in ui["steps"])
    ui_commands = "\n".join(str(step.get("run", "")) for step in ui["steps"])
    assert ui_commands.count("pip install") == 1


def _assert_fast_lane_invocation(workflow: dict) -> None:
    fast = workflow["jobs"]["pr-fast-lane"]
    run = _named_step(fast, "Run fast PR contract")["run"]
    assert tuple(shlex.split(run.replace("\\\n", " "))) == (
        "pytest",
        *FAST_LANE_TARGETS,
        "--timeout=180",
        "--tb=short",
    )


def test_heavy_tests_run_only_on_main_push_or_manual_dispatch() -> None:
    """Keep comprehensive test fan-out off pull-request and schedule events."""
    workflow = _workflow("test.yml")
    triggers = _triggers(workflow)

    assert set(triggers) == {"push", "workflow_dispatch"}
    assert triggers["push"]["branches"] == ["main"]
    assert set(workflow["jobs"]) == HEAVY_JOB_KEYS
    assert workflow["jobs"]["backup-platform-windows"]["if"] == (
        "github.event_name == 'workflow_dispatch' && inputs.backup_platform_only == true"
    )
    assert workflow["permissions"] == {"contents": "read"}
    assert "createComment" not in (WORKFLOW_ROOT / "test.yml").read_text()


def test_dedicated_nightly_owns_exact_schedule_and_full_tree_matrix() -> None:
    """Pin dedicated nightly ownership, cadence, matrix, and tested commit."""
    workflow = _workflow("nightly-deep.yml")
    triggers = _triggers(workflow)

    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert triggers["schedule"] == [{"cron": "30 8 * * *"}]
    assert set(workflow["jobs"]) == {"resolve-dev-sha", "nightly-deep"}

    resolver = workflow["jobs"]["resolve-dev-sha"]
    assert resolver["outputs"] == {"sha": "${{ steps.resolve.outputs.sha }}"}
    resolver_checkout = next(
        step for step in resolver["steps"] if step.get("uses") == "actions/checkout@v4"
    )
    assert resolver_checkout["with"] == {"ref": "dev"}
    resolve = _named_step(resolver, "Resolve one dev commit for every matrix leg")
    assert resolve["id"] == "resolve"
    assert "git rev-parse HEAD" in resolve["run"]
    assert "GITHUB_OUTPUT" in resolve["run"]

    nightly = workflow["jobs"]["nightly-deep"]
    assert nightly["needs"] == ["resolve-dev-sha"]
    assert nightly["strategy"]["matrix"]["include"] == [
        {"os": "ubuntu-latest", "python-version": "3.12", "io-encoding": "utf-8"},
        {"os": "ubuntu-latest", "python-version": "3.13", "io-encoding": "utf-8"},
        {"os": "macos-latest", "python-version": "3.12", "io-encoding": "utf-8"},
        {"os": "windows-latest", "python-version": "3.12", "io-encoding": "cp1252"},
    ]
    checkout = next(
        step for step in nightly["steps"] if step.get("uses") == "actions/checkout@v4"
    )
    assert checkout["with"] == {
        "ref": "${{ needs.resolve-dev-sha.outputs.sha }}",
        "fetch-depth": 0,
    }
    record = _named_step(nightly, "Record tested dev commit")
    assert "needs.resolve-dev-sha.outputs.sha" in record["run"]
    assert "GITHUB_STEP_SUMMARY" in record["run"]
    run = _named_step(
        nightly, "Run deep suite (serial, thorough, slow tiers, cache-off)"
    )
    assert "pytest ./Tests/" in run["run"]
    assert "--run-slow" in run["run"]
    assert "-n auto" not in run["run"]


def test_fast_lane_is_one_serial_minimal_python_312_job() -> None:
    """Keep the fast lane serial, bounded, and minimally provisioned."""
    fast = _workflow("derived-artifacts.yml")["jobs"]["pr-fast-lane"]

    assert fast["name"] == "PR Fast Lane"
    assert fast["if"] == LANES
    assert fast["runs-on"] == "ubuntu-latest"
    # Roleplay frame B1 (TASK-33910.2): owner decision 2026-10-09, "Raise
    # timeout to 35 (Rec.)" -- 30 -> 35 to fit B1's mounted Roleplay suites in
    # the admission-sensitive step (worst case measured 28m20s).
    assert fast["timeout-minutes"] == 35
    assert "strategy" not in fast
    # TASK-32873: 5 steps -- the admission-sensitive suites run in a
    # separate pytest invocation inside the same job (their
    # keep_bootstrap_profile enrollment poisons sandboxed suites sharing
    # a process).
    assert len(fast["steps"]) == 5

    setup = next(
        step for step in fast["steps"] if step.get("uses") == "actions/setup-python@v5"
    )
    assert setup["with"]["python-version"] == "3.12"

    install = _named_step(fast, "Install fast-lane dependencies")["run"]
    assert shlex.split(install) == [
        "python",
        "-m",
        "pip",
        "install",
        "-e",
        ".",
        "pytest",
        "pytest-asyncio",
        "pytest-timeout",
        "packaging",
    ]
    assert "requirements-test.txt" not in install
    assert ".[" not in install
    all_commands = "\n".join(str(step.get("run", "")) for step in fast["steps"])
    assert all_commands.count("pip install") == 1


def test_fast_lane_target_set_is_exact_and_non_overlapping() -> None:
    """Require the approved exact pytest targets without nested selections."""
    workflow = _workflow("derived-artifacts.yml")
    fast = workflow["jobs"]["pr-fast-lane"]
    run = _named_step(fast, "Run fast PR contract")["run"]
    targets = _pytest_targets(run)

    _assert_fast_lane_invocation(workflow)
    assert targets == FAST_LANE_TARGETS
    for index, target in enumerate(targets):
        target_path = Path(target)
        for other in targets[index + 1 :]:
            other_path = Path(other)
            assert target_path not in other_path.parents
            assert other_path not in target_path.parents


def test_admission_sensitive_step_gates_the_notes_sync_real_stack_files() -> None:
    """Keep the "no silent winner" and no-hold pins on every pull request.

    TASK-34000 wave 1a final review (I2): on a pull request only these lists
    and the UI census run; the rest of ``Tests/Notes`` runs on pushes to
    ``main``. A later PR could otherwise break the released pass, the Recovery
    negative controls or the re-save wait and merge green.
    """
    workflow = _workflow("derived-artifacts.yml")
    fast = workflow["jobs"]["pr-fast-lane"]
    targets = _pytest_targets(
        _named_step(fast, "Run admission-sensitive suites")["run"]
    )

    missing = [name for name in NOTES_SYNC_REAL_STACK_TARGETS if name not in targets]
    assert not missing, f"not gated on pull requests: {missing}"
    assert len(set(targets)) == len(targets)
    for target in targets:
        # TASK-33621.27: a target may be one test's node id (`file::test`).
        path = PROJECT_ROOT / target.split("::", 1)[0]
        assert path.exists(), f"gated target is gone: {target}"
    # Their enrollment poisons sandboxed suites sharing a process (TASK-32873).
    assert not set(targets) & set(FAST_LANE_TARGETS)


def test_required_context_fails_closed_and_keeps_artifact_checks_install_free() -> None:
    """Fail the stable gate closed while preserving install-free diagnostics."""
    workflow = _workflow("derived-artifacts.yml")
    _assert_required_aggregation(workflow)

    required = workflow["jobs"]["derived-artifacts"]
    # Everything after the LAST lane verdict is a derived-artifact checker.
    # TASK-32908 added a second verdict step (the UI lane); anchoring on the
    # first one would have classified it as a checker and demanded
    # `!cancelled()` on a step that must stay conditional on its lane.
    verdict_index = max(
        index
        for index, step in enumerate(required["steps"])
        if str(step.get("name", "")).startswith("Require successful ")
    )
    checker_steps = required["steps"][verdict_index + 1 :]
    assert checker_steps
    assert all(step.get("if") == "${{ !cancelled() }}" for step in checker_steps)
    assert "pip install" not in "\n".join(str(step) for step in required["steps"])


def test_required_aggregation_contract_rejects_missing_prerequisite() -> None:
    """Reject a required gate that no longer depends on the fast lane."""
    mutated = copy.deepcopy(_workflow("derived-artifacts.yml"))
    mutated["jobs"]["derived-artifacts"].pop("needs", None)

    with pytest.raises(AssertionError):
        _assert_required_aggregation(mutated)


def test_required_aggregation_contract_rejects_partial_failure_check() -> None:
    """Reject aggregation that handles failure but accepts other bad results."""
    mutated = copy.deepcopy(_workflow("derived-artifacts.yml"))
    verdict = _named_step(
        mutated["jobs"]["derived-artifacts"], "Require successful PR fast lane"
    )
    verdict["if"] = verdict["if"].replace("!= 'success'", "== 'failure'")

    with pytest.raises(AssertionError):
        _assert_required_aggregation(mutated)


@pytest.mark.parametrize(
    ("job_name", "step_name"),
    [
        ("pr-fast-lane", None),
        ("derived-artifacts", None),
        ("derived-artifacts", "Require successful PR fast lane"),
        ("derived-artifacts", "Generated stylesheets reproduce from their sources"),
    ],
)
def test_required_aggregation_contract_rejects_continue_on_error(
    job_name: str, step_name: str | None
) -> None:
    """Reject error-tolerant jobs or steps in the required gate.

    Args:
        job_name: Workflow job to mutate.
        step_name: Optional named step within the job to mutate.
    """
    mutated = copy.deepcopy(_workflow("derived-artifacts.yml"))
    job = mutated["jobs"][job_name]
    target = job if step_name is None else _named_step(job, step_name)
    target["continue-on-error"] = True

    with pytest.raises(AssertionError):
        _assert_required_aggregation(mutated)


@pytest.mark.parametrize(
    "flag",
    [
        "--collect-only",
        "-k smoke",
        "--ignore=Tests/CI",
        "--deselect=Tests/test_smoke.py",
    ],
)
def test_fast_lane_contract_rejects_selection_suppressing_flags(flag: str) -> None:
    """Reject pytest flags that can turn the exact lane into a subset.

    Args:
        flag: Selection-suppressing argument appended to the pytest command.
    """
    mutated = copy.deepcopy(_workflow("derived-artifacts.yml"))
    run_step = _named_step(mutated["jobs"]["pr-fast-lane"], "Run fast PR contract")
    run_step["run"] += f" {flag}"

    with pytest.raises(AssertionError):
        _assert_fast_lane_invocation(mutated)


def test_focused_guards_keep_dev_pr_and_dev_main_push_coverage() -> None:
    """Retain focused guards on dev PRs and pushes to dev and main."""
    for workflow_name in STANDALONE_WORKFLOWS:
        triggers = _triggers(_workflow(workflow_name))

        assert triggers["pull_request"]["branches"] == ["dev"]
        assert triggers["pull_request"]["types"] == PULL_REQUEST_TYPES
        assert triggers["push"]["branches"] == ["dev", "main"]


def test_pull_request_workflows_are_never_cancelled_in_progress() -> None:
    """Prevent base-branch churn from cancelling pull-request gate runs."""
    for workflow_name in STANDALONE_WORKFLOWS:
        concurrency = _workflow(workflow_name)["concurrency"]
        assert concurrency["cancel-in-progress"] == PUSH_ONLY_CANCELLATION

    heavy = _workflow("test.yml")
    assert "pull_request" not in _triggers(heavy)
    assert heavy["concurrency"]["cancel-in-progress"] == PUSH_ONLY_CANCELLATION


def test_bundle_and_backlog_checks_run_on_push_events() -> None:
    """The bundle and backlog-id checks still run on dev/main pushes.

    With css-bundle-guard/backlog-guard deleted, the required workflow is the
    only place these checks run -- so they must run on dev/main pushes too, not
    only inside the pull-request-only fast lanes.
    """
    workflow = _workflow("derived-artifacts.yml")
    assert {"dev", "main"} <= set(_triggers(workflow)["push"]["branches"])
    steps = workflow["jobs"]["derived-artifacts"]["steps"]
    for script in (
        "tldw_chatbook/css/check_bundle_sync.py",
        "scripts/check_backlog_task_ids.py",
    ):
        matching = [step for step in steps if script in str(step.get("run", ""))]
        assert matching, f"{script} is not run by the required job"
        for step in matching:
            assert "pull_request" not in str(step.get("if", "")), (
                f"{script} must not be pull-request-only"
            )
    assert not (PROJECT_ROOT / ".github/workflows/css-bundle-guard.yml").exists()
    assert not (PROJECT_ROOT / ".github/workflows/backlog-guard.yml").exists()
