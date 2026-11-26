"""Behavioral and wiring checks for the opt-in dev/nightly/release policy."""

from __future__ import annotations

import importlib.util
import json
import random
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
        "Tests/real_profile_guard.py",
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


def _laned(root, *dirs):
    """A PR lane that runs these directories whole (lane directories)."""
    _lane_workflow(root, {"pr-fast-lane": ["pytest " + " ".join(dirs)]})
    return root


def _sampled(plan, *changed):
    """The test files a light plan runs beyond the smoke set and `changed`."""
    assert not plan["full"], plan["reason"]
    return set(plan["targets"]) - set(policy.BASE_TESTS) - set(changed)


def test_a_widely_imported_test_module_runs_a_capped_sample_not_broad(tmp_path):
    """TASK-42626: direct importers first, at most IMPORTER_CAP, deleted or not."""
    shared = "Tests/UI/test_shared.py"
    direct = [f"Tests/UI/test_user_{n}.py" for n in range(policy.IMPORTER_CAP)]
    files = {
        shared: "",
        **{f: "from Tests.UI.test_shared import harness\n" for f in direct},
    }
    # Sorts before every direct importer, but imports the module only through one.
    files["Tests/UI/test_a_caller.py"] = "from Tests.UI.test_user_0 import rig\n"
    root = _laned(_tree(tmp_path, files), "Tests/UI")
    plan = policy.select_changes([("M", shared)], root)
    assert _sampled(plan, shared) == set(direct)
    assert shared in plan["targets"]
    assert plan["reason"] == (
        f"Sanity checks; sampled {policy.IMPORTER_CAP} of {policy.IMPORTER_CAP + 1}+"
        " importers of Tests/UI/test_shared.py a PR lane runs;"
        " nightly and release runs are the broad gates"
    )
    (root / shared).unlink()
    assert _sampled(policy.select_changes([("D", shared)], root)) == set(direct)


def test_the_sample_is_the_same_whatever_order_the_tree_is_walked(
    tmp_path, monkeypatch
):
    """TASK-42626: same diff and tree, same targets, on every machine."""
    shared = "Tests/Chat/test_shared.py"
    users = [f"Tests/Chat/test_user_{n:02d}.py" for n in range(policy.IMPORTER_CAP + 3)]
    files = {shared: "", **{f: "from .test_shared import harness\n" for f in users}}
    root = _laned(_tree(tmp_path, files), "Tests/Chat")
    walk = Path.rglob
    plans = []
    for seed in range(6):
        order = random.Random(seed)

        def shuffled(self, pattern, order=order):
            found = list(walk(self, pattern))
            order.shuffle(found)
            return iter(found)

        monkeypatch.setattr(Path, "rglob", shuffled)
        plans.append(policy.select_changes([("M", shared)], root))
    assert all(plan == plans[0] for plan in plans)
    assert _sampled(plans[0], shared) == set(users[: policy.IMPORTER_CAP])


def test_the_file_cap_keeps_the_prs_own_changed_files_first(tmp_path):
    """TASK-42626: over LIGHT_FILE_CAP, the PR's own files stay and importers go."""
    own = [f"Tests/Zeta/test_f{n:02d}.py" for n in range(policy.LIGHT_FILE_CAP - 4)]
    shared = "Tests/Zeta/test_shared.py"
    users = [f"Tests/Alpha/test_user_{n}.py" for n in range(policy.IMPORTER_CAP)]
    files = {f: "" for f in [*own, shared]}
    files.update({f: "from Tests.Zeta.test_shared import x\n" for f in users})
    root = _tree(tmp_path, files)
    plan = policy.select_changes([("M", f) for f in [*own, shared]], root)
    assert _sampled(plan) == {*own, shared, *users[:3]}
    assert (
        "cut to 20 test files; sampled 3 of 8 importers of Tests/Zeta/test_shared.py"
        in (plan["reason"])
    )
    # The PR's own files alone over the cap: the first LIGHT_FILE_CAP by path.
    many = [f"Tests/UI/test_f{n:02d}.py" for n in range(policy.LIGHT_FILE_CAP + 1)]
    root = _tree(tmp_path, many)
    plan = policy.select_changes([("M", f) for f in reversed(many)], root)
    assert _sampled(plan) == set(many[:-1])
    assert (
        "cut to 20 test files, not running the PR's own Tests/UI/test_f20.py"
        in (plan["reason"])
    )


def test_an_importer_the_light_job_cannot_run_is_skipped_not_broad(tmp_path):
    """TASK-42626: helpers, heavy, deselected, split and push-only importers."""
    base = "Tests/UI/test_base.py"
    imports = "from Tests.UI.test_base import Shell\n"
    skipped = [
        "Tests/UI/harness.py",
        "Tests/UI/test_library_shell.py",
        "Tests/Chat/test_deselected.py",
        "Tests/Chat/test_split.py",
        "Tests/Backup_Recovery/test_slow.py",
    ]
    files = {base: "", "Tests/UI/test_ok.py": imports, **{f: imports for f in skipped}}
    files["Tests/UI/test_via_harness.py"] = "from Tests.UI.harness import rig\n"
    root = _tree(tmp_path, files)
    _lane_workflow(
        root,
        {
            "pr-fast-lane": [
                "pytest Tests/Chat/test_deselected.py "
                '"--deselect=Tests/Chat/test_deselected.py::test_known_red" '
                "Tests/Chat/test_split.py::test_a",
                "pytest Tests/Chat/test_split.py::test_b",
            ]
        },
        other="pytest Tests/Backup_Recovery/test_slow.py -n 4",
    )
    # Only the full run reaches the laned ones, and nothing laned runs instead.
    plan = policy.select_changes([("M", base)], root)
    assert plan["full"] and plan["reason"] == (
        f"Only the full run reaches the laned dependents of {base}: "
        "Tests/Backup_Recovery/test_slow.py, Tests/Chat/test_deselected.py,"
        " Tests/Chat/test_split.py"
    )
    workflow = root / ".github/workflows/derived-artifacts.yml"
    workflow.write_text(
        workflow.read_text().replace(
            "pytest Tests/Chat/test_split.py::test_b",
            "pytest Tests/Chat/test_split.py::test_b Tests/UI/test_ok.py",
        )
    )
    plan = policy.select_changes([("M", base)], root)
    assert _sampled(plan, base) == {
        "Tests/UI/test_ok.py",
        "Tests/UI/test_via_harness.py",
    }
    assert (
        "sampled 2 of 2 importers of Tests/UI/test_base.py,"
        " skipping 4 the light job cannot run" in plan["reason"]
    )
    # The same files changed by the PR itself still go broad.
    for path in skipped[1:]:
        assert policy.select_changes([("M", path)], root)["full"], path


def test_a_shared_helper_samples_its_importers(tmp_path):
    """TASK-42626: a non-test module under Tests/ is sampled like a test module."""
    rig = "Tests/Chat/hooks_rig.py"
    users = [f"Tests/Chat/test_hook_{n:02d}.py" for n in range(policy.IMPORTER_CAP + 2)]
    files = {rig: "", **{f: "from Tests.Chat.hooks_rig import rig\n" for f in users}}
    root = _laned(_tree(tmp_path, files), "Tests/Chat")
    for status in ("M", "D"):
        plan = policy.select_changes([(status, rig)], root)
        assert _sampled(plan) == set(users[: policy.IMPORTER_CAP]), status
        assert f"sampled 8 of 9+ importers of {rig} a PR lane runs" in plan["reason"]
    assert not policy.select_changes([("M", "Tests/Chat/hooks_v2_owned_rig.py")], ROOT)[
        "full"
    ]


_CONFTEST_TREE = {
    "Tests/conftest.py": (
        "from Tests import real_guard as _real_guard, net_guard  # noqa: F401\n"
        "from Tests import (\n    paren_one,\n    paren_two,\n)\n"
        "import Tests.chain.start\n"
        "from tldw_chatbook.Utils import other_rig\n"
        "\n\ndef pytest_configure(config):\n    from Tests.fixtures import doubles\n"
    ),
    "Tests/chain/start.py": "from Tests.chain.middle import step\n",
    "Tests/chain/middle.py": "import Tests.chain.end as end\n",
    "Tests/UI/deep/conftest.py": (
        "from Tests.UI.deep.rig import thing\n\n\ndef recorder():\n"
        "    from Tests.Chat.test_loaded import Recorder\n"
    ),
    "Tests/UI/test_other.py": "from Tests.UI.other_rig import x\n",
}


@pytest.mark.parametrize(
    "path",
    [
        "Tests/real_guard.py",  # `from Tests import a as b, c`
        "Tests/net_guard.py",
        "Tests/paren_two.py",  # a parenthesized list
        "Tests/chain/end.py",  # reached only through two other helpers
        "Tests/fixtures/doubles.py",  # an import indented inside a hook
        "Tests/UI/deep/rig.py",  # a nested conftest's helper
        "Tests/Chat/test_loaded.py",  # a test module a conftest imports
    ],
)
def test_anything_a_conftest_loads_goes_broad(path, tmp_path):
    """Fix round 1: it starts every suite below the conftest, as the conftest does."""
    files = {**_CONFTEST_TREE, path: ""}
    files.update({f: "" for f in ("Tests/UI/other_rig.py", "Tests/chain/end.py")})
    root = _tree(tmp_path, files)
    for status in ("M", "D"):
        if status == "D":
            (root / path).unlink()
        plan = policy.select_changes([(status, path)], root)
        assert plan["full"] and plan["reason"] == f"Loaded by conftest.py: {path}"
    # A helper only tests import, named like a non-Tests import, stays light.
    assert not policy.select_changes([("M", "Tests/UI/other_rig.py")], root)["full"]


def test_the_real_conftests_load_the_profile_harness_and_shared_doubles():
    """Fix round 1: replaces the hand list of the two profile-harness files."""
    plan = policy.select_changes([("M", "Tests/network_guard.py")], ROOT)
    assert plan["full"] and plan["reason"] == (
        "Loaded by conftest.py: Tests/network_guard.py"
    )
    corpus = {
        str(f.relative_to(ROOT)): f.read_text(errors="replace")
        for f in (ROOT / "Tests").rglob("*.py")
    }
    assert {
        "Tests/private_profile.py",
        "Tests/real_profile_guard.py",
        "Tests/textual_test_harness.py",
        "Tests/fixtures/required_doubles.py",
        "Tests/Backup_Recovery/config_test_support.py",
        "Tests/Chat/test_console_skill_substitution.py",
    } <= policy._conftest_loaded(corpus)
    assert "Tests/Chat/hooks_v2_owned_rig.py" not in policy._conftest_loaded(corpus)


def test_conftest_files_still_go_broad(tmp_path):
    for conftest in ("Tests/conftest.py", "Tests/Chat/deep/conftest.py"):
        assert policy.select_changes([("M", conftest)], _tree(tmp_path, []))["full"]


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


def test_application_code_with_its_own_schema_goes_broad(tmp_path):
    """Review I2: feature modules that create or version SQLite tables."""
    root = _tree(tmp_path, [])
    feature = root / "tldw_chatbook/Feature"
    feature.mkdir(parents=True)
    (feature / "repository.py").write_text('SQL = "CREATE TABLE things (id INTEGER)"\n')
    (feature / "view.py").write_text("def render():\n    return 1\n")
    assert policy.select_changes([("M", "tldw_chatbook/Feature/repository.py")], root)[
        "full"
    ]
    assert not policy.select_changes([("M", "tldw_chatbook/Feature/view.py")], root)[
        "full"
    ]


def test_changed_test_data_runs_the_tests_that_read_it(tmp_path):
    """Review M3: a data file selects the test files that name it."""
    root = _tree(
        tmp_path,
        {
            "Tests/UI/shortfall.tsv": "",
            "Tests/UI/test_reader.py": 'DATA = "shortfall.tsv"\n',
            "Tests/UI/test_other.py": "",
        },
    )
    plan = policy.select_changes([("M", "Tests/UI/shortfall.tsv")], root)
    assert not plan["full"]
    assert "Tests/UI/test_reader.py" in plan["targets"]
    assert "Tests/UI/test_other.py" not in plan["targets"]


def test_importers_are_found_through_parentheses_and_transitively(tmp_path):
    """Review N1: multi-line imports and importers of importers both count."""
    root = _tree(
        tmp_path,
        {
            "Tests/Sync/test_base.py": "",
            "Tests/Plugins/test_paren.py": "from Tests.Sync import (\n    test_base as suite,\n)\n",
            "Tests/Chat/test_rig.py": "from Tests.Sync.test_base import Recorder\n",
            "Tests/Chat/test_caller.py": "from Tests.Chat.test_rig import rig\n",
        },
    )
    plan = policy.select_changes([("M", "Tests/Sync/test_base.py")], root)
    assert {
        "Tests/Plugins/test_paren.py",
        "Tests/Chat/test_rig.py",
        "Tests/Chat/test_caller.py",
    } <= set(plan["targets"])


def test_importers_are_resolved_by_module_path_not_by_name(tmp_path):
    """Review I1: a same-named module elsewhere, and prose, are not importers."""
    target = "Tests/UI/test_dup.py"
    files = {target: "", "Tests/Chat/test_dup.py": ""}
    # The Chat module's importers sort first and name the same stem.
    for n in range(policy.IMPORTER_CAP + 1):
        files[f"Tests/Chat/test_c{n}.py"] = "from Tests.Chat.test_dup import X\n"
    importers = {
        "Tests/UI/test_abs.py": "import Tests.UI.test_dup\n",
        "Tests/UI/test_from_pkg.py": "from Tests.UI import (\n    other,\n    test_dup,\n)\n",
        "Tests/UI/test_from_mod.py": "def f():\n    from Tests.UI.test_dup import Y\n",
        "Tests/UI/test_rel.py": "from .test_dup import Z\n",
        "Tests/UI/test_rel_pkg.py": "from . import test_dup\n",
        "Tests/UI/test_bare.py": "from test_dup import W\n",  # beside it
        "Tests/UI/test_broken.py": "def (:\n    test_dup\n",  # does not parse: counts
    }
    files.update(importers)
    files["Tests/UI/test_prose.py"] = '"""Mirrors test_dup."""\nimport test_dup_extra\n'
    files["Tests/Other/test_bare.py"] = "from test_dup import W\n"  # not beside it
    root = _tree(tmp_path, files)
    plan = policy.select_changes([("M", target)], root)
    assert _sampled(plan, target) == set(importers)
    assert plan["reason"] == (
        "Sanity checks; nightly and release runs are the broad gates"
    )


def test_a_child_script_embedded_as_a_string_imports_too(tmp_path):
    """Round 2: tests that run a child script load what the script imports."""
    rig = "Tests/fixture_runner.py"
    importers = {
        "Tests/UI/test_child.py": (
            "SCRIPT = '''\nimport sys\nfrom Tests.fixture_runner import owner\n'''\n"
        ),
        # A template with placeholders does not parse whole: its import lines count.
        "Tests/UI/test_template.py": (
            'CHILD = """\nsys.path.insert(0, {repo!r})\n'
            'from Tests.fixture_runner import owner\n"""\n'
        ),
        "Tests/UI/test_fragments.py": (
            'CODE = "import os; " "from Tests.fixture_runner import owner; owner()"\n'
        ),
    }
    files = {rig: "", **importers}
    files["Tests/UI/test_prose.py"] = (
        '"""From the fixture_runner docs: import none."""\n'
    )
    root = _tree(tmp_path, files)
    assert _sampled(policy.select_changes([("M", rig)], root)) == set(importers)


def test_an_unrunnable_or_helper_importer_takes_no_sample_slot(tmp_path):
    """Review I2/M3(b): the search counts runnable test files only."""
    base = "Tests/Chat/test_base.py"
    imports = "from Tests.Chat.test_base import X\n"
    users = [f"Tests/Chat/test_u{n}.py" for n in range(policy.IMPORTER_CAP + 1)]
    root = _tree(
        tmp_path,
        {
            base: "",
            "Tests/Chat/test_deselected.py": imports,
            **dict.fromkeys(users, imports),
        },
    )
    _lane_workflow(
        root,
        {
            "pr-fast-lane": [
                "pytest Tests/Chat "
                '"--deselect=Tests/Chat/test_deselected.py::test_known_red"'
            ]
        },
    )
    plan = policy.select_changes([("M", base)], root)
    assert _sampled(plan, base) == set(users[: policy.IMPORTER_CAP])
    assert (
        f"sampled 8 of 9+ importers of {base} a PR lane runs,"
        " skipping 1 the light job cannot run" in plan["reason"]
    )
    # Nine helpers that sort first: their importers fill the sample instead.
    rig = "Tests/Core/base_rig.py"
    files = {rig: ""}
    for n in range(policy.IMPORTER_CAP + 1):
        files[f"Tests/Core/a_rig_{n}.py"] = "from Tests.Core.base_rig import X\n"
        files[f"Tests/Core/test_user_{n}.py"] = f"from Tests.Core.a_rig_{n} import X\n"
    root = _laned(_tree(tmp_path / "helpers", files), "Tests/Core")
    plan = policy.select_changes([("M", rig)], root)
    assert len(_sampled(plan)) == policy.IMPORTER_CAP


def test_a_tests_ci_importer_runs_anyway_and_is_not_counted(tmp_path):
    """Review M3(c): Tests/CI runs whole, so it takes no slot from the sample."""
    base = "Tests/UI/test_base.py"
    imports = "from Tests.UI.test_base import X\n"
    users = [f"Tests/UI/test_u{n}.py" for n in range(policy.IMPORTER_CAP)]
    only_ci = "Tests/UI/test_only_ci.py"
    files = {base: "", only_ci: "", "Tests/CI/test_ci_user.py": imports}
    files.update(dict.fromkeys(users, imports))
    files["Tests/CI/test_ci_only_user.py"] = "from Tests.UI.test_only_ci import X\n"
    root = _tree(tmp_path, files)
    plan = policy.select_changes([("M", base)], root)
    assert _sampled(plan, base) == set(users)
    assert (
        plan["reason"] == "Sanity checks; nightly and release runs are the broad gates"
    )
    plan = policy.select_changes([("M", only_ci)], root)
    assert not plan["full"]
    assert (
        plan["reason"] == "Sanity checks; nightly and release runs are the broad gates"
    )
    # Sampling with Tests/CI laned, as on dev: the CI importer (it sorts first)
    # still takes no place among the nine the search counts.
    ninth = "Tests/UI/test_u_last.py"
    (root / ninth).write_text(imports)
    _laned(root, "Tests/CI", "Tests/UI")
    plan = policy.select_changes([("M", base)], root)
    assert _sampled(plan, base) == set(users)
    assert f"sampled 8 of 9+ importers of {base} a PR lane runs" in plan["reason"]


def test_the_search_is_breadth_first(tmp_path):
    """Review M3(d): the first helper's importers come before the second's."""
    rig = "Tests/Core/base_rig.py"
    imports = "from Tests.Core.base_rig import X\n"
    direct = [f"Tests/Core/test_d{n}.py" for n in range(4)]
    files = {rig: "", "Tests/Core/a_rig.py": imports, "Tests/Core/b_rig.py": imports}
    files.update(dict.fromkeys(direct, imports))
    for helper in ("a", "b"):
        for n in range(policy.IMPORTER_CAP):
            files[f"Tests/Core/test_v{helper}{n}.py"] = (
                f"from Tests.Core.{helper}_rig import X\n"
            )
    root = _laned(_tree(tmp_path, files), "Tests/Core")
    plan = policy.select_changes([("M", rig)], root)
    assert _sampled(plan) == {*direct, *(f"Tests/Core/test_va{n}.py" for n in range(4))}


def test_a_data_file_samples_its_readers_and_a_reading_helpers_importers(tmp_path):
    """Review I2/M3(a): heavy readers are skipped and noted; something must run."""
    data = "Tests/UI/golden.json"
    reads = 'DATA = "golden.json"\n'
    files = {
        data: "{}",
        "Tests/UI/loader.py": reads,
        "Tests/UI/test_library_shell.py": reads,
        "Tests/UI/test_reader.py": reads,
        "Tests/Chat/test_uses_loader.py": "from Tests.UI.loader import DATA\n",
        "Tests/UI/heavy_only.tsv": "",
        "Tests/UI/orphan.tsv": "",
        "Tests/UI/orphan_rig.py": 'DATA = "orphan.tsv"\n',
    }
    files["Tests/UI/test_library_shell.py"] += 'MORE = "heavy_only.tsv"\n'
    root = _tree(tmp_path, files)
    # As on dev: a PR lane runs the reader, and a push job the heavy file.
    _lane_workflow(
        root,
        {"pr-fast-lane": ["pytest Tests/UI/test_reader.py"]},
        other="pytest Tests/UI/test_library_shell.py",
    )
    plan = policy.select_changes([("M", data)], root)
    assert _sampled(plan) == {
        "Tests/UI/test_reader.py",
        "Tests/Chat/test_uses_loader.py",
    }
    assert (
        f"sampled 2 of 2 readers of {data}, skipping 1 the light job cannot run"
        in plan["reason"]
    )
    plan = policy.select_changes([("M", "Tests/UI/heavy_only.tsv")], root)
    assert plan["full"] and plan["reason"] == (
        "Only the full run reaches the laned dependents of Tests/UI/heavy_only.tsv:"
        " Tests/UI/test_library_shell.py"
    )
    # A helper nothing imports: nothing runs anywhere, so broad adds nothing.
    plan = policy.select_changes([("M", "Tests/UI/orphan.tsv")], root)
    assert _sampled(plan) == set()
    assert "no PR lane runs a dependent of Tests/UI/orphan.tsv;" in plan["reason"]


def test_the_cap_is_shared_in_turn_and_the_notes_follow_the_cut(tmp_path):
    """Review M1: per-module notes after the 20-file cut; readers capped per file."""
    files = {}
    for m in "abc":
        files[f"Tests/M{m}/test_shared.py"] = ""
        for n in range(policy.IMPORTER_CAP + 1):
            files[f"Tests/M{m}/test_user_{n}.py"] = (
                f"from Tests.M{m}.test_shared import X\n"
            )
    small = [f"Tests/Md/test_user_{n}.py" for n in range(5)]
    files["Tests/Md/test_shared.py"] = ""
    files.update(dict.fromkeys(small, "from Tests.Md.test_shared import X\n"))
    root = _laned(_tree(tmp_path, files), "Tests/Ma", "Tests/Mb", "Tests/Mc")
    changes = [("M", f"Tests/M{m}/test_shared.py") for m in "abcd"]
    plan = policy.select_changes(changes, root)
    assert plan["reason"] == (
        "Sanity checks; 8 sampled files in all;"
        " sampled 3 of 9+ importers of Tests/Ma/test_shared.py a PR lane runs;"
        " sampled 3 of 9+ importers of Tests/Mb/test_shared.py a PR lane runs;"
        " sampled 2 of 9+ importers of Tests/Mc/test_shared.py a PR lane runs;"
        " nightly and release runs are the broad gates"
    )
    # The module under the cap runs all its importers, outside the 8-file budget.
    assert set(small) <= set(plan["targets"])
    assert len(_sampled(plan, *(c[1] for c in changes))) == 5 + 8
    readers = {
        f"Tests/A/test_reader_{n:02d}.py": 'P = "common.json"\n' for n in range(25)
    }
    users = {
        f"Tests/Z/test_user_{n}.py": "from Tests.Z.test_shared import X\n"
        for n in range(3)
    }
    root = _tree(
        tmp_path / "data",
        {"Tests/A/common.json": "{}", "Tests/Z/test_shared.py": "", **readers, **users},
    )
    _laned(root, "Tests/A")
    plan = policy.select_changes(
        [("M", "Tests/A/common.json"), ("M", "Tests/Z/test_shared.py")], root
    )
    assert _sampled(plan, "Tests/Z/test_shared.py") == {
        *sorted(readers)[: policy.IMPORTER_CAP],
        *users,
    }


def test_a_sample_holds_only_dependents_a_pr_lane_runs(tmp_path):
    """Round 3: unlaned files have no known health; the search goes past them."""
    base = "Tests/Mix/test_base.py"
    imports = "from Tests.Mix.test_base import X\n"
    unlaned = [f"Tests/Mix/test_a{n:02d}.py" for n in range(20)]
    laned = [f"Tests/Mix/zlaned/test_z{n}.py" for n in range(policy.IMPORTER_CAP + 1)]
    files = {base: "", **dict.fromkeys(unlaned + laned, imports)}
    root = _laned(_tree(tmp_path, files), "Tests/Mix/zlaned")
    plan = policy.select_changes([("M", base)], root)
    assert _sampled(plan, base) == set(laned[: policy.IMPORTER_CAP])
    assert plan["reason"] == (
        f"Sanity checks; sampled 8 of 9+ importers of {base} a PR lane runs,"
        " skipping 20+ no PR lane runs; nightly and release runs are the broad gates"
    )
    for extra in laned[3:]:
        (root / extra).unlink()
    plan = policy.select_changes([("M", base)], root)
    assert _sampled(plan, base) == set(laned[:3])
    assert (
        f"sampled 3 of 3 importers of {base} a PR lane runs, skipping 20 no PR lane runs"
        in plan["reason"]
    )


def test_broad_only_when_the_full_run_reaches_what_light_cannot(tmp_path):
    """Round 3: a laned dependent the light job cannot run, and none laned to run."""
    base = "Tests/Mix/test_base.py"
    imports = "from Tests.Mix.test_base import X\n"
    unlaned = [f"Tests/Mix/test_u{n}.py" for n in range(policy.IMPORTER_CAP + 1)]
    deselected = "Tests/Mix/test_deselected.py"
    root = _tree(
        tmp_path, {base: "", deselected: imports, **dict.fromkeys(unlaned, imports)}
    )
    _lane_workflow(
        root,
        {"pr-fast-lane": [f'pytest {deselected} "--deselect={deselected}::test_red"']},
    )
    for users in (unlaned, unlaned[:3]):  # sampling, and under the cap
        for gone in set(unlaned) - set(users):
            (root / gone).unlink()
        plan = policy.select_changes([("M", base)], root)
        assert plan["full"] and plan["reason"] == (
            f"Only the full run reaches the laned dependents of {base}: {deselected}"
        )
    # One laned file among them: light, and under the cap all of them run.
    (root / "Tests/Mix/zlaned").mkdir()
    (root / "Tests/Mix/zlaned/test_l.py").write_text(imports)
    _lane_workflow(
        root,
        {
            "pr-fast-lane": [
                f'pytest {deselected} "--deselect={deselected}::test_red"',
                "pytest Tests/Mix/zlaned",
            ]
        },
    )
    plan = policy.select_changes([("M", base)], root)
    assert _sampled(plan, base) == {*unlaned[:3], "Tests/Mix/zlaned/test_l.py"}
    assert "skipping 1 the light job cannot run" in plan["reason"]


def test_no_pr_lane_runs_any_dependent_stays_light_and_says_so(tmp_path):
    """Round 3: broad would run none of them either; only the release run does."""
    base = "Tests/Mix/test_base.py"
    users = [f"Tests/Mix/test_u{n}.py" for n in range(policy.IMPORTER_CAP + 1)]
    root = _tree(
        tmp_path,
        {base: "", **dict.fromkeys(users, "from Tests.Mix.test_base import X\n")},
    )
    plan = policy.select_changes([("M", base)], root)
    assert _sampled(plan, base) == set()
    assert plan["reason"] == (
        f"Sanity checks; no PR lane runs a dependent of {base};"
        " only the release run covers its 9 test files;"
        " nightly and release runs are the broad gates"
    )


def _lane_workflow(root, lanes, other=""):
    """A workflow skeleton: PR lane jobs the required check needs, and a dev-only job."""
    names = list(lanes)
    body = "jobs:\n"
    for name, steps in lanes.items():
        body += f"  {name}:\n    steps:\n"
        for run in steps:
            body += "      - name: step\n        run: |\n          " + run + "\n"
    body += (
        "  library-shell-dev:\n    steps:\n      - run: |\n          " + other + "\n"
    )
    body += (
        f"  derived-artifacts:\n    needs: [preflight, dev-light, {', '.join(names)}]\n"
    )
    path = root / ".github/workflows/derived-artifacts.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_a_file_the_lanes_run_only_by_node_runs_those_nodes(tmp_path):
    """Review N2: never run whole a file no PR lane runs whole; run its lane nodes."""
    root = _tree(tmp_path, ["Tests/UI/test_nodes.py", "Tests/UI/test_whole.py"])
    _lane_workflow(
        root,
        {
            "pr-fast-lane": [
                "pytest Tests/UI/test_nodes.py::test_a \\\n"
                "            'Tests/UI/test_nodes.py::test_b[anthropic-200,000]' \\\n"
                "            Tests/UI/test_whole.py Tests/UI/test_whole.py::test_c"
            ]
        },
    )
    nodes = policy.select_changes([("M", "Tests/UI/test_nodes.py")], root)
    assert not nodes["full"]
    assert {
        "Tests/UI/test_nodes.py::test_a",
        "Tests/UI/test_nodes.py::test_b[anthropic-200,000]",
    } <= set(nodes["targets"])
    assert "Tests/UI/test_nodes.py" not in nodes["targets"]
    whole = policy.select_changes([("M", "Tests/UI/test_whole.py")], root)
    assert "Tests/UI/test_whole.py" in whole["targets"]


def test_census_node_ids_are_read_as_whole_lines(tmp_path):
    """Final review I1: a comma inside a parametrize id must not cut the node id."""
    root = _tree(tmp_path, ["Tests/UI/test_disclosures.py"])
    _lane_workflow(
        root, {"ui-fast-lane": ["python scripts/check_ui_pr_gate_census.py --shard 1"]}
    )
    census = root / "scripts/ui_pr_gate_census.txt"
    census.parent.mkdir(parents=True)
    census.write_text(
        "# header\n\nTests/UI/test_disclosures.py::test_reset[openai-32,000]\n"
    )
    plan = policy.select_changes([("M", "Tests/UI/test_disclosures.py")], root)
    assert "Tests/UI/test_disclosures.py::test_reset[openai-32,000]" in plan["targets"]


def test_every_real_census_entry_is_a_lane_target():
    steps, _, _ = policy._lanes(ROOT)
    census = [
        line.strip()
        for line in (ROOT / policy.UI_CENSUS).read_text().splitlines()
        if line.strip().startswith("Tests/")
    ]
    assert census and all(entry in steps for entry in census)


def test_nightly_only_deselected_and_split_files_go_broad(tmp_path):
    """Final review I2 and the split-step Minor."""
    files = [
        "Tests/Backup_Recovery/test_slow.py",
        "Tests/Chat/test_deselected.py",
        "Tests/Chat/test_split.py",
    ]
    root = _tree(tmp_path, files)
    _lane_workflow(
        root,
        {
            "pr-fast-lane": [
                "pytest Tests/Chat/test_deselected.py "
                '"--deselect=Tests/Chat/test_deselected.py::test_known_red" '
                "Tests/Chat/test_split.py::test_a",
                "pytest Tests/Chat/test_split.py::test_b",
            ]
        },
        other="pytest Tests/Backup_Recovery/test_slow.py -n 4",
    )
    for path in files:
        assert policy.select_changes([("M", path)], root)["full"], path


def test_schema_ddl_is_matched_case_sensitively(tmp_path):
    """Final review Minor: ALTER TABLE and CREATE TRIGGER count; prose does not."""
    root = _tree(tmp_path, [])
    feature = root / "tldw_chatbook/Feature"
    feature.mkdir(parents=True)
    (feature / "keys.py").write_text('SQL = "ALTER TABLE meta ADD COLUMN k"\n')
    (feature / "dictionary.py").write_text('SQL = "CREATE TRIGGER t AFTER INSERT"\n')
    (feature / "notes.py").write_text("# we create table views for the user\n")
    for name, full in [("keys.py", True), ("dictionary.py", True), ("notes.py", False)]:
        plan = policy.select_changes([("M", f"tldw_chatbook/Feature/{name}")], root)
        assert plan["full"] is full, name


def test_only_the_pragma_versions_a_schema_not_a_python_name(tmp_path):
    """Fix round 1: #3769's `user_version` parameter is not schema code."""
    root = _tree(tmp_path, [])
    feature = root / "tldw_chatbook/Feature"
    feature.mkdir(parents=True)
    (feature / "repository.py").write_text(
        "def settle(row, user_version: int):\n"
        '    return row["current_user_version"] == user_version\n'
    )
    (feature / "store.py").write_text('conn.execute("PRAGMA user_version = 3")\n')
    for name, full in [("repository.py", False), ("store.py", True)]:
        plan = policy.select_changes([("M", f"tldw_chatbook/Feature/{name}")], root)
        assert plan["full"] is full, name


def test_a_deleted_data_file_runs_the_tests_that_read_it(tmp_path):
    root = _tree(tmp_path, {"Tests/UI/test_reader.py": 'DATA = "gone.tsv"\n'})
    plan = policy.select_changes([("D", "Tests/UI/gone.tsv")], root)
    assert "Tests/UI/test_reader.py" in plan["targets"]


def test_a_real_node_only_ui_file_runs_exactly_its_census_nodes():
    """Final review: step names must not make `Tests/UI` a lane directory."""
    path = "Tests/UI/test_settings_advanced_disclosures.py"
    census = {
        line.strip()
        for line in (ROOT / policy.UI_CENSUS).read_text().splitlines()
        if line.strip().startswith(path + "::")
    }
    plan = policy.select_changes([("M", path)], ROOT)
    assert census and not plan["full"]
    assert path not in plan["targets"]
    assert {t for t in plan["targets"] if t.startswith(path)} == census


def test_step_names_and_env_are_not_lane_targets(tmp_path):
    root = _tree(tmp_path, ["Tests/UI/test_x.py"])
    workflow = root / ".github/workflows/derived-artifacts.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n  ui-fast-lane:\n    steps:\n"
        "      - name: Run the gated Tests/UI slice\n"
        "        env:\n          TARGET: Tests/UI\n"
        "        run: |\n          echo ok  # Tests/UI in a comment\n"
        "          pytest Tests/UI/test_x.py::test_a\n"
        "  derived-artifacts:\n    needs: [preflight, ui-fast-lane]\n"
    )
    steps, _, _ = policy._lanes(root)
    assert set(steps) == {"Tests/UI/test_x.py::test_a"}


def test_the_plain_word_fallback_drops_quotes_from_deselects():
    words = policy._run_words(
        'name: x\n        run: |\n          pytest it\'s "--deselect=Tests/x.py::n"\n'
    )
    assert "--deselect=Tests/x.py::n" in words
