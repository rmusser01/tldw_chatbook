"""Behavioral and wiring checks for the opt-in dev/nightly/release policy."""

from __future__ import annotations

import importlib.util
import json
import shlex
from pathlib import Path
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "tier_policy", ROOT / "scripts/ci/tier_policy.py"
)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def test_explicit_sources_select_their_tests_and_unconditional_smoke():
    for source, tests in policy.SUBSYSTEM_TESTS.items():
        assert (ROOT / source).is_dir(), source
        plan = policy.select_changes(
            [("M", source + "ordinary.py"), ("A", "backlog/tasks/task-1.md")], ROOT
        )
        assert not plan["full"]
        assert set(plan["targets"]) == set(policy.BASE_TESTS + tests)


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        "requirements.txt",
        ".github/workflows/derived-artifacts.yml",
        "requirements-dev.txt",
        "setup.py",
        "Tests/conftest.py",
        "Tests/UI/conftest.py",
        "Tests/private_profile.py",
        "Tests/Chat/hooks_v2_owned_rig.py",
        "scripts/ci/tier_policy.py",
        "tldw_chatbook/DB/migrations/123.sql",
        # TASK-42521 review: schema outside DB/migrations, packages, pytest config,
        # guard data and the files no PR lane runs whole.
        "tldw_chatbook/Plugins/migrations/001.sql",
        "tldw_chatbook/Scheduling/db/schedules_db.py",
        "tldw_chatbook/DB/Client_Media_DB_v2.py",
        "packages/tldw_profile_core/src/tldw_profile_core/paths.py",
        "packages/tldw_profile_core/tests/test_paths.py",
        "Tests/UI/pytest.ini",
        "tox.ini",
        "Tests/Architecture/allowlist.tsv",
        "Tests/Performance/boot_budget_snapshots/console.txt",
        "Tests/UI/test_library_shell.py",
        "Tests/Backup_Recovery/test_response_rules_restore.py",
    ],
)
def test_ci_build_schema_and_shared_test_inputs_require_broad(path):
    """TASK-42521: only inputs that can break CI itself, or the schema, stay broad."""
    assert policy.select_changes([("M", path)], ROOT)["full"]


@pytest.mark.parametrize(
    "path",
    [
        "tldw_chatbook/Backup_Recovery/new.py",
        "tldw_chatbook/Utils/path_validation.py",
        "tldw_chatbook/MCP/permissions.py",
        "Docs/QA/fixture.json",
        "Docs/User_Guide/console/agent-runs-and-tools.md",
        "backlog/docs/input.md",
        "scripts/ui_pr_gate_census.txt",
        "unknown.py",
    ],
)
def test_other_changes_run_sanity_checks_only(path):
    """TASK-42521: the nightly and release runs are the broad gates for these."""
    plan = policy.select_changes([("M", path)], ROOT)
    assert not plan["full"]
    assert plan["targets"] == sorted(policy.BASE_TESTS)


def _tree(tmp_path, files):
    """A synthetic checkout: the smoke set, the subsystem smoke files and `files`."""
    for rel in [*policy.BASE_TESTS, *files]:
        path = tmp_path / rel.split("::", 1)[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(files.get(rel, "") if isinstance(files, dict) else "")
    for tests in policy.SUBSYSTEM_TESTS.values():
        for target in tests:
            path = tmp_path / target.split("::", 1)[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return tmp_path


def test_changed_test_files_run_in_the_light_tier(tmp_path):
    root = _tree(
        tmp_path,
        [
            "Tests/Agents/test_a.py",
            "Tests/Backup_Recovery/test_b.py",
            "Tests/CI/test_c.py",
        ],
    )
    plan = policy.select_changes(
        [
            ("M", "Tests/Agents/test_a.py"),
            ("A", "Tests/Backup_Recovery/test_b.py"),
            ("M", "Tests/CI/test_c.py"),
            ("D", "Tests/UI/test_retired.py"),
        ],
        root,
    )
    assert not plan["full"]
    # Tests/CI runs whole in the light job, and a deleted file has nothing to run.
    assert set(plan["targets"]) == set(policy.BASE_TESTS) | {
        "Tests/Agents/test_a.py",
        "Tests/Backup_Recovery/test_b.py",
    }


def test_a_changed_test_file_is_not_narrowed_to_its_own_smoke_nodes(tmp_path):
    """Review C1: pytest collapses `f f::node` to `f::node`, so the file must win."""
    nav = "Tests/UI/test_screen_navigation.py"
    plan = policy.select_changes(
        [("M", "tldw_chatbook/UI/screen.py"), ("M", nav)], _tree(tmp_path, [nav])
    )
    assert nav in plan["targets"]
    assert not [t for t in plan["targets"] if t.startswith(nav + "::")]


def test_an_imported_test_module_brings_its_importers_or_goes_broad(tmp_path):
    """Review I1: a test module other tests import is shared infrastructure."""
    files = {"Tests/UI/test_shared.py": "", "Tests/UI/test_lonely.py": ""}
    for n in range(policy.IMPORTER_CAP):
        files[f"Tests/UI/test_user_{n}.py"] = (
            "from Tests.UI.test_shared import harness\n"
        )
    root = _tree(tmp_path, files)
    plan = policy.select_changes([("M", "Tests/UI/test_shared.py")], root)
    assert not plan["full"]
    assert {f"Tests/UI/test_user_{n}.py" for n in range(policy.IMPORTER_CAP)} <= set(
        plan["targets"]
    )
    assert policy.select_changes([("D", "Tests/UI/test_shared.py")], root)["full"]
    assert not policy.select_changes([("D", "Tests/UI/test_lonely.py")], root)["full"]
    (root / "Tests/UI/test_one_more.py").write_text(
        "from .test_shared import harness\n"
    )
    assert policy.select_changes([("M", "Tests/UI/test_shared.py")], root)["full"]


@pytest.mark.parametrize("status", ["D", "R100", "T", "U"])
def test_structural_changes_cannot_disappear(status):
    assert policy.select_changes(
        [(status, "tldw_chatbook/Widgets/detach_safe_text_area.py")], ROOT
    )["full"]


def test_empty_diff_is_broad_and_missing_target_is_an_error(tmp_path):
    assert policy.select_changes([], ROOT)["full"]
    with pytest.raises(ValueError, match="does not exist"):
        policy.select_changes(
            [("M", "tldw_chatbook/Widgets/detach_safe_text_area.py")], tmp_path
        )


def test_diff_preserves_deletions_and_odd_filenames(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    git("init", "-q")
    git("config", "user.email", "ci@example.invalid")
    git("config", "user.name", "CI test")
    (tmp_path / "old.py").write_text("before")
    git("add", ".")
    git("commit", "-qm", "before")
    base = git("rev-parse", "HEAD")
    (tmp_path / "old.py").rename(tmp_path / "a\nb.py")
    git("add", "-A")
    git("commit", "-qm", "after")
    changes = policy.changed_paths(base, git("rev-parse", "HEAD"))
    assert set(changes) == {("D", "old.py"), ("A", "a\nb.py")}
    with pytest.raises(ValueError):
        policy.changed_paths("--output=/tmp/no", base)


@pytest.mark.parametrize(
    "kind,ref,base,enabled,light",
    [
        ("pull_request", "refs/pull/1/merge", "dev", "true", True),
        ("pull_request", "refs/pull/1/merge", "dev", "", False),
        ("pull_request", "refs/pull/1/merge", "main", "true", False),
        ("push", "refs/heads/dev", "", "true", True),
        ("push", "refs/heads/main", "", "true", False),
        ("workflow_dispatch", "refs/heads/dev", "", "true", False),
    ],
)
def test_event_policy_never_lightens_release_or_manual(
    kind, ref, base, enabled, light, tmp_path, monkeypatch
):
    event = {"before": "a" * 40}
    if kind == "pull_request":
        event["pull_request"] = {"base": {"ref": base, "sha": "a" * 40}}
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event))
    output = tmp_path / "output"
    for key, value in {
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_EVENT_NAME": kind,
        "GITHUB_REF": ref,
        "GITHUB_SHA": "b" * 40,
        "GITHUB_OUTPUT": str(output),
        "TIERED_ENABLED": enabled,
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(
        policy,
        "changed_paths",
        lambda *_: [("M", "tldw_chatbook/Widgets/detach_safe_text_area.py")],
    )
    policy.main()
    result = {
        k: json.loads(v)
        for k, v in (line.split("=", 1) for line in output.read_text().splitlines())
    }
    assert result["light"] is light
    assert result["full"] is not light


def test_release_and_publishing_share_full_tests_and_validated_package():
    release = workflow("release-qualification.yml")["jobs"]
    assert release["tests"]["uses"] == "./.github/workflows/test.yml"
    assert release["package"]["uses"] == "./.github/workflows/package-check.yml"
    assert release["verdict"]["needs"] == ["tests", "package"]
    assert release["verdict"]["if"] == "${{ always() }}"
    verdict = release["verdict"]["steps"][0]["run"]
    for test_result in ("success", "failure", "cancelled", "skipped", ""):
        for package_result in ("success", "failure", "cancelled", "skipped", ""):
            completed = subprocess.run(
                ["bash", "-c", verdict],
                env={"TESTS": test_result, "PACKAGE": package_result},
            )
            assert (completed.returncode == 0) == (
                test_result == package_result == "success"
            )
    publish = workflow("publish-pypi.yml")["jobs"]
    assert publish["qualify"]["uses"] == "./.github/workflows/test.yml"
    assert publish["build"]["uses"] == "./.github/workflows/package-check.yml"
    for name in ("publish-pypi", "publish-testpypi"):
        assert {"build", "qualify"} <= set(publish[name]["needs"])
    # Callers test their own immutable event SHA; no mutable ref checkout.
    for name in ("test.yml", "package-check.yml"):
        for job in workflow(name)["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/checkout@"):
                    assert "ref" not in step.get("with", {})


def test_safe_release_dispatch_qualifies_candidate_without_publishing():
    derived = workflow("derived-artifacts.yml")
    trigger = derived.get("on", derived.get(True))
    assert trigger["workflow_dispatch"]["inputs"]["release"]["type"] == "boolean"
    jobs = derived["jobs"]
    condition = jobs["release-qualification"]["if"]
    assert "github.event_name == 'workflow_dispatch'" in condition
    assert "inputs.release" in condition
    assert (
        jobs["release-qualification"]["uses"]
        == "./.github/workflows/release-qualification.yml"
    )
    verdict = next(
        step
        for step in jobs["derived-artifacts"]["steps"]
        if step.get("name") == "Require complete release qualification"
    )
    assert "inputs.release" in verdict["if"]
    for job in jobs.values():
        assert "publish" not in job.get("uses", "")
        for step in job.get("steps", []):
            assert "pypi-publish" not in step.get("uses", "")


def test_nightly_and_release_dispatch_have_own_concurrency_groups():
    group = workflow("derived-artifacts.yml")["concurrency"]["group"]
    assert group == (
        "${{ github.workflow }}-${{ github.event_name }}-${{ github.ref }}"
        "${{ inputs.nightly && '-nightly' || inputs.release && '-release' || '' }}"
    )


def test_light_gate_is_required_and_uses_hosted_runners():
    jobs = workflow("derived-artifacts.yml")["jobs"]
    assert {"dev-light", "release-qualification"} <= set(
        jobs["derived-artifacts"]["needs"]
    )
    assert jobs["dev-light"]["if"] == "needs.preflight.outputs.light == 'true'"
    assert jobs["dev-light"]["runs-on"] == "ubuntu-latest"
    assert jobs["dev-light"]["timeout-minutes"] < 45
    steps = {s.get("name"): s for s in jobs["derived-artifacts"]["steps"]}
    for name, lane in [
        ("Require selected dev smoke tests", "dev-light"),
        ("Require complete release qualification", "release-qualification"),
    ]:
        assert f"needs.{lane}.result != 'success'" in steps[name]["if"]
        assert "exit 1" in steps[name]["run"]


def test_missing_or_inconsistent_selection_cannot_produce_green_aggregate():
    job = workflow("derived-artifacts.yml")["jobs"]["derived-artifacts"]
    script = next(
        s["run"] for s in job["steps"] if s.get("name") == "Require valid tier decision"
    )
    for full in ("true", "false", "", "null"):
        for light in ("true", "false", "", "null"):
            result = subprocess.run(
                ["bash", "-c", script],
                env={"FULL": full, "LIGHT": light},
                capture_output=True,
            )
            assert (result.returncode == 0) == (
                (full, light) in {("true", "false"), ("false", "true")}
            )


@pytest.mark.parametrize(
    "runs,dispatch",
    [
        ([], True),
        ([{"conclusion": "success", "status": "completed"}], False),
        ([{"conclusion": "failure", "status": "completed"}], True),
        ([{"conclusion": "cancelled", "status": "completed"}], True),
        ([{"conclusion": "skipped", "status": "completed"}], True),
        ([{"conclusion": None, "status": "queued"}], False),
        (
            [{"conclusion": "success", "status": "completed", "head_sha": "b" * 40}],
            True,
        ),
        (
            [
                {
                    "conclusion": "success",
                    "status": "completed",
                    "display_title": "Other run",
                }
            ],
            True,
        ),
    ],
)
def test_nightly_dispatches_changed_sha_and_retries_failed_coverage(runs, dispatch):
    source = workflow("regression-nightly.yml")["jobs"]["dispatch"]["steps"][0]["with"][
        "script"
    ]
    runs = [
        {"head_sha": "a" * 40, "display_title": "Nightly regression", **run}
        for run in runs
    ]
    harness = r"""
    const fs = require('fs');
    const {source, runs} = JSON.parse(fs.readFileSync(0, 'utf8'));
    const calls = [];
    const github = {
      rest: {
        git: {getRef: async args => {calls.push(['ref', args]); return {data: {object: {sha: 'a'.repeat(40)}}};}},
        actions: {
          listWorkflowRuns: 'list-runs',
          createWorkflowDispatch: async args => {calls.push(['dispatch', args]);}
        }
      },
      paginate: async (endpoint, args) => {calls.push(['list', args]); return runs;}
    };
    const run = new (Object.getPrototypeOf(async function(){}).constructor)('github', 'context', 'core', source);
    run(github, {repo: {owner: 'o', repo: 'r'}}, {info: () => {}})
      .then(() => process.stdout.write(JSON.stringify(calls)))
      .catch(err => {console.error(err); process.exit(1);});
    """
    result = subprocess.run(
        ["node", "-e", harness],
        input=json.dumps({"source": source, "runs": runs}),
        text=True,
        capture_output=True,
        check=True,
    )
    calls = json.loads(result.stdout)
    query = next(args for name, args in calls if name == "list")
    assert query["workflow_id"] == "derived-artifacts.yml"
    assert query["head_sha"] == "a" * 40
    assert query["event"] == "workflow_dispatch" and query["branch"] == "dev"
    dispatched = [args for name, args in calls if name == "dispatch"]
    assert bool(dispatched) is dispatch
    if dispatch:
        assert dispatched[0]["ref"] == "dev"
        assert dispatched[0]["inputs"] == {"nightly": "true"}


def test_reusable_suite_has_a_distinct_concurrency_group():
    assert workflow("test.yml")["concurrency"]["group"].startswith("comprehensive-")
    assert not workflow("derived-artifacts.yml")["concurrency"]["group"].startswith(
        "comprehensive-"
    )


@pytest.mark.parametrize(
    "path",
    [
        "tldw_chatbook/Chat/console_approval_service.py",
        "tldw_chatbook/Chat/console_project_instructions.py",
        "tldw_chatbook/UI/Screens/settings_screen.py",
        "tldw_chatbook/Notes/notes_sync_runtime.py",
        "tldw_chatbook/Widgets/credential_dialog.py",
    ],
)
def test_sensitive_names_get_their_subsystem_smoke_not_broad_ci(path):
    """TASK-42521 (owner): sensitive paths no longer force the broad run on the PR."""
    plan = policy.select_changes([("M", path)], ROOT)
    assert not plan["full"]
    subsystem = next(k for k in policy.SUBSYSTEM_TESTS if path.startswith(k))
    assert set(policy.SUBSYSTEM_TESTS[subsystem]) <= set(plan["targets"])


def test_new_modules_are_light_but_removed_application_files_stay_broad():
    assert not policy.select_changes([("A", "tldw_chatbook/Chat/new.py")], ROOT)["full"]
    assert not policy.select_changes([("D", "backlog/tasks/task-1.md")], ROOT)["full"]
    assert policy.select_changes([("D", "tldw_chatbook/Chat/old.py")], ROOT)["full"]


@pytest.mark.parametrize("nightly,full_label", [(True, False), (False, True)])
def test_nightly_qualifies_all_smoke_and_full_label_overrides_selection(
    tmp_path, monkeypatch, nightly, full_label
):
    event = (
        {"inputs": {"nightly": "true"}}
        if nightly
        else {"pull_request": {"base": {"ref": "dev"}, "labels": [{"name": "ci:full"}]}}
    )
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event))
    output = tmp_path / "output"
    for key, value in {
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_EVENT_NAME": "workflow_dispatch" if nightly else "pull_request",
        "GITHUB_REF": "refs/heads/dev",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_OUTPUT": str(output),
        "TIERED_ENABLED": "true" if full_label else "",
    }.items():
        monkeypatch.setenv(key, value)
    policy.main()
    result = {
        k: json.loads(v)
        for k, v in (line.split("=", 1) for line in output.read_text().splitlines())
    }
    assert result["full"] is True
    assert result["light"] is nightly
    if nightly:
        assert set(result["targets"]) == set(policy.BASE_TESTS).union(
            *policy.SUBSYSTEM_TESTS.values()
        )
        job = workflow("derived-artifacts.yml")["jobs"]["derived-artifacts"]
        script = next(
            s["run"]
            for s in job["steps"]
            if s.get("name") == "Require valid tier decision"
        )
        subprocess.run(
            ["bash", "-c", script],
            env={"FULL": "true", "LIGHT": "true", "NIGHTLY": "true"},
            check=True,
        )


def test_required_preflight_runs_the_same_pinned_lint_rules():
    steps = workflow("derived-artifacts.yml")["jobs"]["preflight"]["steps"]
    lint = workflow("lint.yml")["jobs"]
    install = next(s for s in steps if s.get("name") == "Install preflight test tools")
    assert "ruff==0.16.6" in install["run"]
    for job in lint.values():
        commands = [
            s["run"]
            for s in job["steps"]
            if "run" in s and "pip install" not in s["run"]
        ]
        for command in commands:
            matched = [
                s
                for s in steps
                if shlex.split(s.get("run", ""), comments=True)
                == shlex.split(command, comments=True)
            ]
            assert len(matched) == 1
            assert matched[0]["if"] == "${{ !cancelled() }}"


def test_shallow_checkout_fetches_only_the_missing_immutable_base(
    tmp_path, monkeypatch
):
    repo = tmp_path / "source"
    repo.mkdir()

    def git(*args, cwd=repo):
        return subprocess.check_output(
            ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()

    git("init", "-q")
    git("config", "user.email", "ci@example.invalid")
    git("config", "user.name", "CI test")
    base = None
    for value in ("one", "two", "three"):
        (repo / "value.txt").write_text(value)
        git("add", ".")
        git("commit", "-qm", value)
        if base is None:
            base = git("rev-parse", "HEAD")
    shallow = tmp_path / "shallow"
    git("clone", "--depth=1", repo.as_uri(), str(shallow))
    head = git("rev-parse", "HEAD", cwd=shallow)
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", base], cwd=shallow, capture_output=True
        ).returncode
        != 0
    )
    monkeypatch.chdir(shallow)
    assert policy.changed_paths(base, head) == [("M", "value.txt")]
    assert git("rev-parse", "--is-shallow-repository", cwd=shallow) == "true"


runner_spec = importlib.util.spec_from_file_location(
    "run_light", ROOT / "scripts/ci/run_light.py"
)
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)


def test_light_runner_gives_each_file_its_own_process_and_whole_files_win():
    """TASK-42521 review C1/M4: grouping is by file, and `f` beats `f::node`."""
    assert runner.plan(
        [
            "Tests/test_smoke.py",
            "Tests/UI/test_nav.py::test_a",
            "Tests/UI/test_nav.py::test_b",
            "Tests/Chat/test_x.py::test_c",
            "Tests/Chat/test_x.py",
        ]
    ) == [
        ["Tests/CI"],
        ["Tests/test_smoke.py"],
        ["Tests/UI/test_nav.py::test_a", "Tests/UI/test_nav.py::test_b"],
        ["Tests/Chat/test_x.py"],
    ]


@pytest.mark.parametrize("targets", [[], "Tests/x.py", ["../x.py"], ["Docs/x.py"], [3]])
def test_light_runner_refuses_a_missing_or_invalid_selection(targets):
    with pytest.raises(SystemExit, match="Refusing"):
        runner.plan(targets)


def test_light_runner_fails_on_any_red_file_and_warns_on_an_empty_one(
    monkeypatch, capsys
):
    """Review M1/M4: exit 5 warns and passes; any other non-zero fails, all reported."""
    codes = {"Tests/CI": 0, "Tests/a.py": 5, "Tests/b.py": 1, "Tests/c.py": 2}
    calls = []

    def fake_run(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, codes[argv[3]])

    monkeypatch.setenv(
        "CI_TEST_TARGETS", json.dumps(["Tests/a.py", "Tests/b.py", "Tests/c.py"])
    )
    assert runner.main(run=fake_run) == 1
    out = capsys.readouterr().out
    assert "::warning::Tests/a.py collected no tests" in out
    assert "Failed: Tests/b.py, Tests/c.py" in out
    assert all(argv[-2:] == ["--timeout=180", "--tb=short"] for argv in calls)
    monkeypatch.setenv("CI_TEST_TARGETS", json.dumps(["Tests/a.py"]))
    assert runner.main(run=fake_run) == 0


def test_light_job_runs_the_tested_runner_with_the_lanes_plugin_setup():
    job = workflow("derived-artifacts.yml")["jobs"]["dev-light"]
    step = next(
        s
        for s in job["steps"]
        if s.get("name") == "Run CI contracts and selected smoke tests"
    )
    assert step["run"].strip() == "python scripts/ci/run_light.py"
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" not in step.get("env", {})


def test_too_many_selected_files_go_broad(tmp_path):
    """A light run is serial; more files than the cap would not fit its timeout."""
    files = [f"Tests/UI/test_f{n}.py" for n in range(policy.LIGHT_FILE_CAP)]
    root = _tree(tmp_path, files)
    assert policy.select_changes([("M", f) for f in files], root)["full"]
    assert not policy.select_changes([("M", f) for f in files[:-3]], root)["full"]
