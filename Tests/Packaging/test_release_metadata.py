from __future__ import annotations

import ast
import re
import tomllib
import urllib.error
from pathlib import Path

import pytest
from pydantic import ValidationError

from Packaging.common.dist_path import resolve_dist_dir
from Packaging.common import version as packaging_version
from Packaging import check_pypi_release


REPO_ROOT = Path(__file__).resolve().parents[2]


class _PypiJsonResponse:
    def __init__(self, payload: bytes = b"{") -> None:
        self.payload = payload

    def __enter__(self) -> "_PypiJsonResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        return self.payload


def _package_version_metadata() -> tuple[str, tuple[int, ...]]:
    tree = ast.parse((REPO_ROOT / "tldw_chatbook" / "__init__.py").read_text())
    assignments: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                "__version__",
                "VERSION_TUPLE",
            }:
                assignments[target.id] = ast.literal_eval(node.value)

    package_version_tuple = assignments["VERSION_TUPLE"]
    assert isinstance(package_version_tuple, tuple)

    return (str(assignments["__version__"]), package_version_tuple)


def test_release_version_metadata_stays_in_lockstep() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    project_version = project["version"]
    package_version, package_version_tuple = _package_version_metadata()

    assert package_version == project_version
    assert packaging_version.VERSION == project_version
    assert packaging_version.VERSION_TUPLE == package_version_tuple


def test_pypi_release_scripts_match_packaged_entry_points() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        scripts = tomllib.load(stream)["project"]["scripts"]

    assert scripts == {
        "tldw-cli": "tldw_chatbook.cli:main_cli_runner",
        "tldw-serve": "tldw_chatbook.Web_Server.serve:main",
    }


def test_distribution_output_path_must_be_strictly_inside_repo(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()

    assert resolve_dist_dir("dist", repo_root) == repo_root / "dist"
    assert resolve_dist_dir("./dist", repo_root) == repo_root / "dist"

    unsafe_paths = ("", ".", "./", "..", "../dist", "dist/..", "./dist/..", str(external))
    for unsafe in unsafe_paths:
        with pytest.raises(ValueError):
            resolve_dist_dir(unsafe, repo_root)


def test_distribution_output_path_rejects_symlink_escape(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    link = repo_root / "linked-external"

    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError):
        resolve_dist_dir("linked-external/dist", repo_root)


def test_pypi_release_exists_returns_true_for_existing_version() -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(url: str, timeout: int) -> _PypiJsonResponse:
        captured["url"] = url
        captured["timeout"] = timeout
        return _PypiJsonResponse()

    assert check_pypi_release.release_exists(
        "tldw-chatbook",
        "1.2.3",
        base_url="https://example.test/pypi/",
        urlopen=fake_urlopen,
    )
    assert captured == {
        "url": "https://example.test/pypi/tldw-chatbook/1.2.3/json",
        "timeout": 30,
    }


def test_pypi_release_exists_returns_false_for_missing_version() -> None:
    def fake_urlopen(url: str, timeout: int) -> object:
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    assert not check_pypi_release.release_exists(
        "tldw-chatbook",
        "9.9.9",
        urlopen=fake_urlopen,
    )


def test_pypi_release_exists_reraises_unexpected_http_errors() -> None:
    def fake_urlopen(url: str, timeout: int) -> object:
        raise urllib.error.HTTPError(url, 503, "Unavailable", hdrs=None, fp=None)

    with pytest.raises(urllib.error.HTTPError):
        check_pypi_release.release_exists(
            "tldw-chatbook",
            "1.2.3",
            urlopen=fake_urlopen,
        )


def test_pypi_release_decision_allows_absent_version_newer_than_latest() -> None:
    def fake_urlopen(url: str, timeout: int) -> _PypiJsonResponse:
        if url.endswith("/tldw-chatbook/1.2.4/json"):
            raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
        if url.endswith("/tldw-chatbook/json"):
            return _PypiJsonResponse(b'{"releases": {"1.2.3": [], "not-version": []}}')
        raise AssertionError(url)

    decision = check_pypi_release.release_decision(
        "tldw-chatbook",
        "1.2.4",
        base_url="https://example.test/pypi",
        urlopen=fake_urlopen,
    )

    assert not decision.release_exists
    assert decision.latest_version == "1.2.3"
    assert decision.publish_release


def test_pypi_release_decision_skips_existing_version() -> None:
    def fake_urlopen(url: str, timeout: int) -> _PypiJsonResponse:
        if url.endswith("/tldw-chatbook/1.2.3/json"):
            return _PypiJsonResponse()
        if url.endswith("/tldw-chatbook/json"):
            return _PypiJsonResponse(b'{"releases": {"1.2.3": []}}')
        raise AssertionError(url)

    decision = check_pypi_release.release_decision(
        "tldw-chatbook",
        "1.2.3",
        base_url="https://example.test/pypi",
        urlopen=fake_urlopen,
    )

    assert decision.release_exists
    assert decision.latest_version == "1.2.3"
    assert not decision.publish_release


def test_pypi_release_decision_reports_real_latest_for_existing_stale_candidate() -> None:
    def fake_urlopen(url: str, timeout: int) -> _PypiJsonResponse:
        if url.endswith("/tldw-chatbook/1.2.3/json"):
            return _PypiJsonResponse()
        if url.endswith("/tldw-chatbook/json"):
            return _PypiJsonResponse(b'{"releases": {"1.2.3": [], "1.2.4": []}}')
        raise AssertionError(url)

    decision = check_pypi_release.release_decision(
        "tldw-chatbook",
        "1.2.3",
        base_url="https://example.test/pypi",
        urlopen=fake_urlopen,
    )

    assert decision.release_exists
    assert decision.latest_version == "1.2.4"
    assert not decision.publish_release


def test_pypi_release_decision_skips_absent_version_older_than_latest() -> None:
    def fake_urlopen(url: str, timeout: int) -> _PypiJsonResponse:
        if url.endswith("/tldw-chatbook/1.2.2/json"):
            raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
        if url.endswith("/tldw-chatbook/json"):
            return _PypiJsonResponse(b'{"releases": {"1.2.3": []}}')
        raise AssertionError(url)

    decision = check_pypi_release.release_decision(
        "tldw-chatbook",
        "1.2.2",
        base_url="https://example.test/pypi",
        urlopen=fake_urlopen,
    )

    assert not decision.release_exists
    assert decision.latest_version == "1.2.3"
    assert not decision.publish_release


def test_pypi_release_decision_allows_first_release_for_missing_package() -> None:
    def fake_urlopen(url: str, timeout: int) -> object:
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    decision = check_pypi_release.release_decision(
        "tldw-chatbook",
        "0.1.0",
        base_url="https://example.test/pypi",
        urlopen=fake_urlopen,
    )

    assert not decision.release_exists
    assert decision.latest_version is None
    assert decision.publish_release


def test_pypi_release_payload_shape_is_validated() -> None:
    def fake_urlopen(url: str, timeout: int) -> _PypiJsonResponse:
        return _PypiJsonResponse(b'{"releases": []}')

    with pytest.raises(ValidationError):
        check_pypi_release.latest_release_version(
            "tldw-chatbook",
            base_url="https://example.test/pypi",
            urlopen=fake_urlopen,
        )


def test_github_output_path_is_validated_against_runner_temp(tmp_path: Path) -> None:
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    output_path = runner_temp / "_runner_file_commands" / "set_output"
    output_path.parent.mkdir()
    output_path.touch()
    outside_path = tmp_path / "outside_output"
    outside_path.touch()

    check_pypi_release.write_github_output(
        "release_exists",
        "false",
        output_path=output_path,
        runner_temp=runner_temp,
    )
    assert output_path.read_text() == "release_exists=false\n"

    with pytest.raises(ValueError):
        check_pypi_release.write_github_output(
            "release_exists",
            "false",
            output_path=outside_path,
            runner_temp=runner_temp,
        )


def test_check_pypi_release_cli_uses_workflow_arguments_and_emits_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[tuple[str, int]] = []

    def fake_urlopen(url: str, timeout: int) -> _PypiJsonResponse:
        captured.append((url, timeout))
        if url.endswith("/tldw-chatbook/1.2.3/json"):
            return _PypiJsonResponse()
        if url.endswith("/tldw-chatbook/json"):
            return _PypiJsonResponse(b'{"releases": {"1.2.3": []}}')
        raise AssertionError(url)

    output_path = tmp_path / "_runner_file_commands" / "set_output"
    output_path.parent.mkdir()
    output_path.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))

    assert (
        check_pypi_release.main(
            ["1.2.3", "--base-url", "https://example.test/pypi"],
            urlopen=fake_urlopen,
        )
        == 0
    )

    assert captured == [
        (
            "https://example.test/pypi/tldw-chatbook/1.2.3/json",
            check_pypi_release.PYPI_REQUEST_TIMEOUT_SECONDS,
        ),
        (
            "https://example.test/pypi/tldw-chatbook/json",
            check_pypi_release.PYPI_REQUEST_TIMEOUT_SECONDS,
        ),
    ]
    assert output_path.read_text() == (
        "release_exists=true\n"
        "latest_version=1.2.3\n"
        "publish_release=false\n"
    )
    output = capsys.readouterr().out
    assert "release_exists=true" in output
    assert "publish_release=false" in output


def test_check_pypi_release_cli_skips_stale_version_and_emits_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_urlopen(url: str, timeout: int) -> _PypiJsonResponse:
        if url.endswith("/tldw-chatbook/1.2.2/json"):
            raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
        if url.endswith("/tldw-chatbook/json"):
            return _PypiJsonResponse(b'{"releases": {"1.2.3": []}}')
        raise AssertionError(url)

    output_path = tmp_path / "_runner_file_commands" / "set_output"
    output_path.parent.mkdir()
    output_path.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))

    assert (
        check_pypi_release.main(
            ["1.2.2", "--base-url", "https://example.test/pypi"],
            urlopen=fake_urlopen,
        )
        == 0
    )

    assert output_path.read_text() == (
        "release_exists=false\n"
        "latest_version=1.2.3\n"
        "publish_release=false\n"
    )
    output = capsys.readouterr().out
    assert "release_exists=false" in output
    assert "latest_version=1.2.3" in output
    assert "publish_release=false" in output


def test_check_pypi_release_cli_rejects_invalid_version() -> None:
    called = False

    def fake_urlopen(url: str, timeout: int) -> _PypiJsonResponse:
        nonlocal called
        called = True
        return _PypiJsonResponse()

    with pytest.raises(SystemExit) as excinfo:
        check_pypi_release.main(["not-a-version"], urlopen=fake_urlopen)

    assert excinfo.value.code == 2
    assert not called


def test_check_pypi_release_cli_rejects_reserved_output_names() -> None:
    called = False

    def fake_urlopen(url: str, timeout: int) -> _PypiJsonResponse:
        nonlocal called
        called = True
        return _PypiJsonResponse()

    with pytest.raises(SystemExit) as excinfo:
        check_pypi_release.main(
            ["1.2.3", "--output-name", "release_exists"],
            urlopen=fake_urlopen,
        )

    assert excinfo.value.code == 2
    assert not called


def test_testpypi_publish_requires_protected_dev_ref() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text()

    assert (
        "if: github.event_name == 'workflow_dispatch' && "
        "github.ref == 'refs/heads/dev' && github.ref_protected"
    ) in workflow


def _publish_workflow_text() -> str:
    return (REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text()


def _workflow_job_block(workflow: str, job_name: str) -> str:
    match = re.search(
        rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [\w-]+:\n|\Z)",
        workflow,
        re.M | re.S,
    )
    assert match is not None, f"{job_name} job not found"
    return match.group("body")


def test_production_pypi_publish_requires_protected_main_push() -> None:
    workflow = _publish_workflow_text()
    trigger_block = workflow.split("\npermissions:", maxsplit=1)[0]
    publish_job = _workflow_job_block(workflow, "publish-pypi")

    assert "push:\n    branches:\n      - main" in trigger_block
    assert "tags:" not in trigger_block
    assert "github.event_name == 'push'" in publish_job
    assert "github.ref == 'refs/heads/main'" in publish_job
    assert "github.ref_protected" in publish_job
    assert "refs/tags" not in publish_job
    assert "refs/heads/dev" not in publish_job


def test_production_pypi_publish_checks_version_before_upload() -> None:
    workflow = _publish_workflow_text()
    check_job = _workflow_job_block(workflow, "check_pypi_release")
    publish_job = _workflow_job_block(workflow, "publish-pypi")

    assert 'run: python Packaging/check_pypi_release.py "$RELEASE_VERSION"' in check_job
    assert "Install output path validation dependencies" in check_job
    assert "python -m pip install loguru packaging psutil pydantic" in check_job
    assert "needs: [build, check_pypi_release]" in publish_job
    assert "needs.check_pypi_release.outputs.publish_release == 'true'" in publish_job


def test_release_workflow_serializes_same_ref_runs() -> None:
    workflow = _publish_workflow_text()

    assert (
        "concurrency:\n"
        "  group: ${{ github.workflow }}-${{ github.ref }}\n"
        "  cancel-in-progress: false"
    ) in workflow


def test_release_workflow_uses_validated_github_output_writes() -> None:
    workflow = _publish_workflow_text()
    build_job = _workflow_job_block(workflow, "build")

    assert "version: ${{ steps.release-version.outputs.version }}" in build_job
    assert "from Packaging.check_pypi_release import write_github_output" in build_job
    assert 'write_github_output("version", project_version)' in build_job
    assert "GITHUB_OUTPUT" not in workflow


def test_release_instructions_do_not_publish_production_pypi_from_tags() -> None:
    checked_paths = (
        REPO_ROOT / "Packaging" / "PYPI_README.md",
        REPO_ROOT / "Packaging" / "PYPI_RELEASE.md",
        REPO_ROOT / "Packaging" / "build_release.sh",
    )
    forbidden_lines = (
        "Protected `v*` tags publish to PyPI",
        "PyPI job only runs for protected matching version tags",
        "Publish production PyPI from a protected v",
    )

    for path in checked_paths:
        text = path.read_text()
        for forbidden_line in forbidden_lines:
            assert forbidden_line not in text
