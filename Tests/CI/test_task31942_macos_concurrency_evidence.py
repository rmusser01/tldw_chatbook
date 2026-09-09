"""Behavioral contract for TASK-31942's narrow macOS evidence workflow."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = (
    REPOSITORY_ROOT / ".github/workflows/task-31942-macos-concurrency-evidence.yml"
)
EXPECTED_NODES = (
    "Tests/TTS/test_profile_repository.py::test_spawned_repositories_open_one_fresh_store_concurrently[0]",
    "Tests/TTS/test_profile_repository.py::test_spawned_repositories_open_one_fresh_store_concurrently[1]",
    "Tests/TTS/test_profile_repository.py::test_spawned_repositories_open_one_fresh_store_concurrently[2]",
    "Tests/TTS/test_profile_repository.py::test_spawned_repositories_resolve_sqlite_constraint_race_safely[0]",
    "Tests/TTS/test_profile_repository.py::test_spawned_repositories_resolve_sqlite_constraint_race_safely[1]",
    "Tests/TTS/test_profile_repository.py::test_spawned_repositories_resolve_sqlite_constraint_race_safely[2]",
    "Tests/TTS/test_profile_repository.py::test_spawned_repositories_resolve_sqlite_constraint_race_safely[3]",
    "Tests/TTS/test_profile_repository.py::test_spawned_repositories_resolve_sqlite_constraint_race_safely[4]",
    "Tests/TTS/test_profile_repository.py::test_spawned_set_delete_race_is_serialized_without_partial_mutation[0]",
    "Tests/TTS/test_profile_repository.py::test_spawned_set_delete_race_is_serialized_without_partial_mutation[1]",
    "Tests/TTS/test_profile_repository.py::test_spawned_set_delete_race_is_serialized_without_partial_mutation[2]",
)


def _workflow() -> dict[str, object]:
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _job() -> dict[str, object]:
    jobs = _workflow()["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {"macos-concurrency-evidence"}
    job = jobs["macos-concurrency-evidence"]
    assert isinstance(job, dict)
    return job


def _step(name: str) -> dict[str, object]:
    steps = _job()["steps"]
    assert isinstance(steps, list)
    matches = [step for step in steps if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _inline_python(step_name: str) -> str:
    command = _step(step_name)["run"]
    assert isinstance(command, str)
    marker = "<<'PY'\n"
    assert command.count(marker) == 1
    body = command.split(marker, 1)[1]
    assert body.endswith("\nPY\n")
    return body[: -len("\nPY\n")]


def _junit_case(node: str, outcome: str = "passed") -> str:
    path, name = node.split("::", 1)
    classname = path.removesuffix(".py").replace("/", ".")
    child = "" if outcome == "passed" else f"<{outcome} />"
    return f"<testcase classname={quoteattr(classname)} name={quoteattr(name)}>{child}</testcase>"


def _junit(nodes: tuple[str, ...] = EXPECTED_NODES) -> str:
    cases = "".join(_junit_case(node) for node in nodes)
    return (
        f'<testsuites><testsuite tests="{len(nodes)}" errors="0" failures="0" '
        f'skipped="0">{cases}</testsuite></testsuites>'
    )


def _validate(tmp_path: Path, xml: str | None) -> subprocess.CompletedProcess[str]:
    junit_path = tmp_path / "evidence.xml"
    if xml is not None:
        junit_path.write_text(xml, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-", str(junit_path)],
        input=_inline_python("Validate exact JUnit evidence"),
        text=True,
        capture_output=True,
        check=False,
    )


def test_workflow_is_one_branch_only_read_only_fresh_macos_job() -> None:
    workflow = _workflow()
    job = _job()

    assert workflow["on"] == {
        "push": {"branches": ["codex/task-31942-macos-concurrency-evidence"]}
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert job["runs-on"] == "macos-15"
    assert job["timeout-minutes"] == "30"
    assert job["defaults"] == {"run": {"shell": "bash"}}
    assert "concurrency" not in workflow

    checkout = _step("Check out the exact tested commit")
    assert checkout == {
        "name": "Check out the exact tested commit",
        "uses": "actions/checkout@v4",
        "with": {
            "ref": "${{ github.sha }}",
            "persist-credentials": "false",
        },
    }
    setup = _step("Set up exact Python")
    assert setup["uses"] == "actions/setup-python@v5"
    assert setup["with"] == {"python-version": "3.12.10"}


def test_workflow_runs_only_the_exact_serial_nodes_and_retains_narrow_evidence() -> (
    None
):
    install = _step("Install existing development dependencies")
    run = _step("Run the eleven concurrency cases once")
    upload = _step("Upload narrow macOS evidence")
    command = run["run"]
    assert isinstance(command, str)

    assert install["run"] == 'python -m pip install -e ".[dev]"'
    assert command.count("python -m pytest") == 1
    for node in EXPECTED_NODES:
        assert command.split().count(node) == 1
    assert command.count("Tests/") == 11
    assert '--junitxml="$RUNNER_TEMP/task-31942-junit.xml"' in command
    assert '| tee "$RUNNER_TEMP/task-31942-pytest.log"' in command
    assert not {"continue-on-error", "if"}.intersection(run)
    assert upload == {
        "name": "Upload narrow macOS evidence",
        "if": "always()",
        "uses": "actions/upload-artifact@v4",
        "with": {
            "name": "task-31942-macos-concurrency-evidence",
            "path": (
                "${{ runner.temp }}/task-31942-metadata.txt\n"
                "${{ runner.temp }}/task-31942-lock-control.json\n"
                "${{ runner.temp }}/task-31942-pytest.log\n"
                "${{ runner.temp }}/task-31942-junit.xml\n"
            ),
            "if-no-files-found": "error",
        },
    }


def test_metadata_and_lock_control_precede_install_and_fail_closed() -> None:
    job = _job()
    steps = job["steps"]
    assert isinstance(steps, list)
    names = [step["name"] for step in steps]
    assert (
        names.index("Record fixed runner metadata")
        < names.index("Require isolated spawn lock control")
        < names.index("Install existing development dependencies")
    )
    assert names.index("Require isolated spawn lock control") < names.index(
        "Run the eleven concurrency cases once"
    )

    metadata = _step("Record fixed runner metadata")["run"]
    control = _step("Require isolated spawn lock control")["run"]
    assert isinstance(metadata, str)
    assert isinstance(control, str)
    assert all(
        field in metadata
        for field in (
            "tested_git_sha=",
            "python=",
            "sqlite=",
            "macos=",
            "architecture=",
        )
    )
    assert "os.environ" not in metadata
    assert "python -I -" in control
    assert 'mp.get_context("spawn").Lock()' in control
    assert "probe.acquire(False)" in control
    assert "probe.release()" in control
    assert "del probe" in control
    assert "gc.collect()" in control
    assert "set -o pipefail" in control
    assert not {"continue-on-error", "if"}.intersection(
        _step("Require isolated spawn lock control")
    )


def test_validator_always_runs_before_upload() -> None:
    job = _job()
    steps = job["steps"]
    assert isinstance(steps, list)
    names = [step["name"] for step in steps]

    assert _step("Validate exact JUnit evidence")["if"] == "always()"
    assert names.index("Validate exact JUnit evidence") < names.index(
        "Upload narrow macOS evidence"
    )


def test_actual_pytest_shell_preserves_failure_through_tee(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/bash\nprintf 'controlled pytest failure\\n'\nexit 17\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["RUNNER_TEMP"] = str(tmp_path)
    command = _step("Run the eleven concurrency cases once")["run"]
    assert isinstance(command, str)

    result = subprocess.run(
        ["bash", "-c", command],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 17
    assert (tmp_path / "task-31942-pytest.log").read_text(encoding="utf-8") == (
        "controlled pytest failure\n"
    )


def test_validator_accepts_only_the_literal_eleven_passes(tmp_path: Path) -> None:
    result = _validate(tmp_path, _junit())

    assert result.returncode == 0
    assert result.stdout.strip() == "validated exact 11 passing cases"


@pytest.mark.parametrize(
    ("label", "xml"),
    (
        ("missing", _junit(EXPECTED_NODES[:-1])),
        (
            "extra",
            _junit(
                EXPECTED_NODES + ("Tests/TTS/test_profile_repository.py::test_extra",)
            ),
        ),
        ("duplicate", _junit(EXPECTED_NODES + (EXPECTED_NODES[0],))),
        (
            "skipped",
            _junit().replace(
                _junit_case(EXPECTED_NODES[0]),
                _junit_case(EXPECTED_NODES[0], "skipped"),
            ),
        ),
        (
            "failure",
            _junit().replace(
                _junit_case(EXPECTED_NODES[0]),
                _junit_case(EXPECTED_NODES[0], "failure"),
            ),
        ),
        (
            "error",
            _junit().replace(
                _junit_case(EXPECTED_NODES[0]), _junit_case(EXPECTED_NODES[0], "error")
            ),
        ),
        ("malformed", "<testsuite><testcase"),
        ("aggregate-error", _junit().replace('errors="0"', 'errors="1"')),
        ("missing-file", None),
    ),
)
def test_validator_rejects_non_exact_evidence(
    tmp_path: Path, label: str, xml: str | None
) -> None:
    result = _validate(tmp_path, xml)

    assert result.returncode != 0, label
    assert result.stderr.strip(), label
