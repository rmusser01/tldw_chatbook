from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import traceback
from pathlib import Path, PureWindowsPath
from types import ModuleType, SimpleNamespace

import pytest
from packaging.requirements import Requirement


REPO_ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION_ROOT = REPO_ROOT / "scripts" / "terminal_qualification"
EVIDENCE_ROOT = (
    REPO_ROOT / "Docs" / "superpowers" / "reviews" / "evidence" / "task-22512"
)
EVIDENCE = EVIDENCE_ROOT / "dependency-qualification.md"
FORMAT_BASELINE = EVIDENCE_ROOT / "format-baseline.json"
RAW_ROOT = EVIDENCE_ROOT / "raw"
README = QUALIFICATION_ROOT / "README.md"

REQUIRED_HEADINGS = (
    "# TASK-22512 dependency qualification",
    "## Scope and decision",
    "## Qualification hosts and commands",
    "## Package versions, hashes, and licenses",
    "## Wheel matrix",
    "## Platform matrix",
    "## API and backend identity",
    "## Parser matrix",
    "## Environment key sets and profile behavior",
    "## I/O, EOF, and output integrity",
    "## Memory and resource bounds",
    "## Raw evidence",
    "## Limitations and fail-closed boundaries",
)

FORMAT_PATHS = (
    "tldw_chatbook/app.py",
    "tldw_chatbook/UI/Screens/settings_screen.py",
    "tldw_chatbook/UI/Screens/chat_screen.py",
    "tldw_chatbook/UI/Console_Modules/left_rail.py",
    "tldw_chatbook/UI/Console_Modules/wiring.py",
    "tldw_chatbook/UI/console_command_provider.py",
    "tldw_chatbook/Widgets/Console/__init__.py",
    "Tests/Chat/test_console_runtime_lifetime.py",
    "Tests/UI/test_console_runtime_ownership.py",
    "Tests/UI/test_settings_raw_cli.py",
    "Tests/UI/test_console_left_rail.py",
    "Tests/UI/test_console_internals_decomposition.py",
    "Tests/UI/test_console_controller_wiring.py",
    "Tests/UI/test_console_workbench_contract.py",
    "Tests/UI/test_console_shell_regions.py",
    "Tests/UI/test_css_bundle_sync_guard.py",
)

MANDATORY_ROWS = (
    "package-pyte-0.8.2",
    "package-regex-2026.4.4",
    "package-pywinpty-3.0.5",
    "parser-shell-captures",
    "parser-powershell-cmd-fixtures",
    "parser-full-screen-programs",
    "parser-unicode-cells",
    "parser-alternate-screen",
    "parser-resize",
    "parser-bracketed-paste",
    "parser-terminal-queries",
    "parser-malformed-controls",
    "parser-incomplete-sequence-bounds",
    "parser-mutable-collections",
    "environment-default-shell",
    "environment-bash",
    "environment-zsh",
    "windows-platform-floor",
    "windows-low-level-api",
    "windows-conpty-only",
    "windows-job-admission-membership",
    "windows-handle-inheritance",
    "windows-one-credit-bounded-read",
    "windows-concurrent-io-close",
    "windows-profile-module-discovery",
    "windows-unicode-alternate-screen",
    "windows-app-crash-descendant-cleanup",
    "windows-eof-output-integrity",
    "four-session-managed-rss",
)

FORBIDDEN_WINDOWS_APIS = (
    "PtyProcess",
    "PtyProcessUnicode",
    "Backend.WinPTY",
    "subprocess.PIPE",
)

LINUX_ROW_ID = "linux-arm64-py312"
MACOS_ROW_IDS = (
    "macos-arm64-py311",
    "macos-arm64-py312",
    "macos-arm64-py313",
    "macos-arm64-py314",
)
WINDOWS_ROW_ID = "win-amd64-py311"
WINDOWS_FAIL_CLOSED_ROWS = {
    "windows-unicode-alternate-screen",
    "windows-eof-output-integrity",
}


def _row_status(evidence: str, row_id: str) -> str:
    pattern = re.compile(
        rf"^\|\s*{re.escape(row_id)}\s*\|\s*MANDATORY\s*\|\s*([^|]+?)\s*\|",
        re.MULTILINE,
    )
    match = pattern.search(evidence)
    assert match is not None, f"missing mandatory qualification row: {row_id}"
    return match.group(1).strip()


def test_dependency_sources_admit_only_qualified_terminal_parser() -> None:
    """Require dependency manifests to admit only the qualified terminal parser."""
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    sources = {
        "pyproject.toml": pyproject["project"]["dependencies"],
        "requirements.txt": (REPO_ROOT / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines(),
    }

    for source, entries in sources.items():
        requirements = [
            requirement
            for entry in entries
            if (candidate := entry.split("#", 1)[0].strip())
            for requirement in (Requirement(candidate),)
        ]
        assert [
            str(requirement)
            for requirement in requirements
            if requirement.name.lower() == "pyte"
        ] == ["pyte==0.8.2"], source
        assert [
            str(requirement)
            for requirement in requirements
            if requirement.name.lower() == "regex"
        ] == ["regex==2026.4.4"], source
        assert all(
            requirement.name.lower() != "pywinpty" for requirement in requirements
        ), source

    optional_requirements = (
        Requirement(entry)
        for entries in pyproject["project"]["optional-dependencies"].values()
        for entry in entries
    )
    assert all(
        requirement.name.lower() != "pywinpty" for requirement in optional_requirements
    )

    evidence = EVIDENCE.read_text(encoding="utf-8")
    compact_evidence = " ".join(evidence.split())
    assert "no Windows dependency is admitted by this artifact." in compact_evidence
    assert {_row_status(evidence, row_id) for row_id in WINDOWS_FAIL_CLOSED_ROWS} == {
        "FAIL_CLOSED"
    }


def test_dependency_qualification_records_all_binding_rows() -> None:
    assert EVIDENCE.is_file(), f"missing evidence artifact: {EVIDENCE}"
    evidence = EVIDENCE.read_text(encoding="utf-8")
    for heading in REQUIRED_HEADINGS:
        assert heading in evidence
    assert "pyte==0.8.2" in evidence
    assert "regex==2026.4.4" in evidence
    assert "pywinpty==3.0.5" in evidence
    assert "textual-terminal source adaptation: none" in evidence
    assert "PENDING" not in evidence
    assert "UNKNOWN" not in evidence

    for row_id in MANDATORY_ROWS:
        assert _row_status(evidence, row_id) in {
            "PASS",
            "FAIL_CLOSED",
            "UNSUPPORTED_FAIL_CLOSED",
        }


def test_package_rows_record_artifact_hash_license_and_wheel_facts() -> None:
    evidence = EVIDENCE.read_text(encoding="utf-8")
    for requirement in (
        "pyte==0.8.2",
        "regex==2026.4.4",
        "pywinpty==3.0.5",
    ):
        assert requirement in evidence, f"missing package evidence: {requirement}"
        package_block = evidence.split(requirement, 1)[1].split("\n\n", 1)[0]
        assert re.search(r"sha256:[0-9a-f]{64}", package_block)
        assert "license:" in package_block.lower()
        assert "wheel:" in package_block.lower()


def test_format_baseline_is_pinned_and_records_all_paths() -> None:
    assert FORMAT_BASELINE.is_file(), f"missing formatter baseline: {FORMAT_BASELINE}"
    payload = json.loads(FORMAT_BASELINE.read_text(encoding="utf-8"))
    for key in (
        "base_sha",
        "base_ref",
        "ruff_version",
        "paths",
        "files",
        "baseline_red_paths",
    ):
        assert key in payload, f"formatter baseline missing key: {key}"
    assert re.fullmatch(r"[0-9a-f]{40}", payload["base_sha"])
    assert payload["base_ref"] == "origin/dev"
    assert payload["ruff_version"].startswith("ruff ")
    assert tuple(payload["paths"]) == FORMAT_PATHS
    assert set(payload["files"]) == set(FORMAT_PATHS)
    for path in FORMAT_PATHS:
        facts = payload["files"][path]
        assert re.fullmatch(r"[0-9a-f]{64}", facts["source_sha256"])
        assert re.fullmatch(r"[0-9a-f]{64}", facts["normalized_diff_sha256"])
    assert set(payload["baseline_red_paths"]).issubset(FORMAT_PATHS)


def test_raw_evidence_names_are_unique_content_free_and_hashed() -> None:
    evidence = EVIDENCE.read_text(encoding="utf-8")
    raw_files = sorted(RAW_ROOT.glob("*/*.json"))
    assert raw_files, "no collected raw qualification rows"
    identities: set[tuple[str, str]] = set()
    names: set[str] = set()
    for raw_file in raw_files:
        payload = json.loads(raw_file.read_text(encoding="utf-8"))
        row_id = payload["row_id"]
        probe = payload["probe"]
        assert raw_file.parent.name == row_id
        assert raw_file.name == f"{row_id}-{probe}.json"
        assert raw_file.name not in names
        names.add(raw_file.name)
        identity = (row_id, probe)
        assert identity not in identities
        identities.add(identity)
        serialized_keys = " ".join(_all_keys(payload)).lower()
        for forbidden_key in (
            "terminal_output",
            "profile_content",
            "environment_values",
            "secret_value",
        ):
            assert forbidden_key not in serialized_keys
        digest = hashlib.sha256(raw_file.read_bytes()).hexdigest()
        assert (
            f"{raw_file.relative_to(EVIDENCE_ROOT).as_posix()} | sha256:{digest}"
            in evidence
        )


def _all_keys(value: object) -> list[str]:
    if isinstance(value, dict):
        keys = [str(key) for key in value]
        for child in value.values():
            keys.extend(_all_keys(child))
        return keys
    if isinstance(value, list):
        keys: list[str] = []
        for child in value:
            keys.extend(_all_keys(child))
        return keys
    return []


def _load_qualification_module(name: str) -> object:
    module_path = QUALIFICATION_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"task22512_{name}", module_path)
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(QUALIFICATION_ROOT))
    previous = sys.modules.get(spec.name)
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        sys.path.remove(str(QUALIFICATION_ROOT))


def test_qualification_scripts_do_not_import_product_code() -> None:
    for script in sorted(QUALIFICATION_ROOT.glob("*.py")):
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not [
            name
            for name in imported
            if name == "tldw_chatbook" or name.startswith("tldw_chatbook.")
        ]


def test_artifact_failure_manifest_remains_schema_valid() -> None:
    common = _load_qualification_module("common")
    payload = common._failure_manifest(
        row_id="macos-arm64-py311",
        requirements=("pyte==0.8.2", "wcwidth>=0.2.14,<1"),
        started="2026-08-29T00:00:00+00:00",
        started_monotonic=time.monotonic(),
        category="artifact_download_failed",
        runtime={"kind": "host"},
    )

    common.validate_content_free(payload)


def test_prepared_platform_facts_use_row_interpreter_identity(tmp_path: Path) -> None:
    common = _load_qualification_module("common")
    row_python = tmp_path / "row-python"
    row_python.symlink_to(sys.executable)

    facts = common._prepared_platform_facts(row_python, tmp_path)

    assert facts["python_executable_name"] == "row-python"
    assert facts["python_version"] == platform.python_version()


def test_windows_probe_contains_no_forbidden_api_path() -> None:
    source = (QUALIFICATION_ROOT / "pywinpty_probe.py").read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_WINDOWS_APIS:
        assert forbidden not in source


def test_probe_clis_are_importable_and_expose_required_arguments() -> None:
    expected_help = {
        "common.py": ("prepare-row", "collect-row", "validate-row"),
        "environment_probe.py": ("--json-out", "--replace", "--shell"),
        "pyte_probe.py": ("--json-out", "--replace", "--artifact-manifest"),
        "pywinpty_probe.py": ("--json-out", "--replace", "--artifact-manifest"),
        "format_ratchet.py": ("snapshot", "verify"),
    }
    for name, fragments in expected_help.items():
        completed = subprocess.run(
            [sys.executable, str(QUALIFICATION_ROOT / name), "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        for fragment in fragments:
            assert fragment in completed.stdout


def _minimal_raw_payload(row_id: str, probe: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "row_id": row_id,
        "probe": probe,
        "status": "PASS",
        "mandatory": True,
        "started_at_utc": "2026-08-29T00:00:00+00:00",
        "completed_at_utc": "2026-08-29T00:00:01+00:00",
        "elapsed_seconds": 1.0,
        "command": {
            "argv": [
                "/usr/bin/python3",
                "qualification.py",
                "--json-out",
                "result.json",
            ],
            "working_directory": "/tmp/qualification",
        },
        "platform": {
            "os": "Darwin",
            "os_release": "24.6.0",
            "os_version": "Darwin Kernel Version 24.6.0",
            "architecture": "arm64",
            "python_implementation": "CPython",
            "python_version": "3.12.11",
            "python_executable_name": "python3.12",
        },
        "measurements": {"current_rss_bytes": None, "peak_rss_bytes": 1},
        "runtime": {"kind": "host"},
        "rows": [],
    }
    if probe == "artifacts":
        payload.update(
            {
                "artifacts": [
                    {
                        "filename": "fixture-1.0-py3-none-any.whl",
                        "kind": "wheel",
                        "license": "MIT",
                        "license_classifiers": [],
                        "license_expression": None,
                        "license_files": [],
                        "name": "fixture",
                        "sha256": "a" * 64,
                        "sha256_after_install": "a" * 64,
                        "sha256_before_install": "a" * 64,
                        "size_bytes": 1,
                        "tags": ["py3-none-any"],
                        "version": "1.0",
                    }
                ],
                "requirements": ["fixture==1.0"],
                "resolved_distributions": [
                    {
                        "name": "fixture",
                        "version": "1.0",
                        "record_file": "fixture-1.0.dist-info/RECORD",
                        "record_file_sha256": "b" * 64,
                        "primary_file": "fixture.py",
                        "primary_file_sha256": "c" * 64,
                    }
                ],
                "rows": [
                    {
                        "id": "artifact-download-hash-offline-install",
                        "mandatory": True,
                        "status": "PASS",
                        "artifact_count": 1,
                    }
                ],
            }
        )
    elif probe.startswith("environment-"):
        shell_name = probe.removeprefix("environment-")
        result_id = (
            "environment-default-shell"
            if shell_name == "default"
            else f"environment-{shell_name}"
        )
        payload.update(
            {
                "initial_keys": ["HOME", "PATH", "TERM"],
                "rows": [
                    {
                        "id": result_id,
                        "mandatory": True,
                        "status": "PASS",
                        "available": True,
                        "initial_key_count": 3,
                        "sensitive_initial_key_count": 0,
                    }
                ],
            }
        )
    elif probe == "pyte":
        payload.update(
            {
                "term": "linux",
                "rows": [
                    {
                        "id": "parser-shell-captures",
                        "mandatory": True,
                        "status": "PASS",
                        "available_count": 2,
                        "captured_byte_count": 128,
                        "captured_count": 2,
                    }
                ],
            }
        )
    elif probe == "pywinpty":
        payload.update(
            {
                "status": "UNSUPPORTED_FAIL_CLOSED",
                "reason_category": "native-windows-host-required",
                "rows": [
                    {
                        "id": "windows-handle-inheritance",
                        "mandatory": True,
                        "status": "UNSUPPORTED_FAIL_CLOSED",
                        "native_execution": False,
                    }
                ],
            }
        )
    return payload


def _collect_payload(
    tmp_path: Path, payload: dict[str, object]
) -> subprocess.CompletedProcess[str]:
    row_dir = tmp_path / "row"
    row_dir.mkdir()
    (row_dir / "probe.json").write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(QUALIFICATION_ROOT / "common.py"),
            "collect-row",
            "--row-dir",
            str(row_dir),
            "--evidence-root",
            str(tmp_path / "raw"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_complete_source_row(row_dir: Path) -> None:
    """Write one complete six-probe source row without collection metadata."""
    retained = RAW_ROOT / "macos-arm64-py312"
    row_dir.mkdir()
    for source in sorted(retained.glob("*.json")):
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload.pop("collection_command")
        payload.pop("generation_id")
        (row_dir / source.name).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


def _write_complete_windows_source_row(row_dir: Path) -> None:
    """Write one complete Windows-family source row from retained full fixtures."""
    retained = RAW_ROOT / "macos-arm64-py312"
    templates = {
        json.loads(path.read_text(encoding="utf-8"))["probe"]: json.loads(
            path.read_text(encoding="utf-8")
        )
        for path in retained.glob("*.json")
    }
    row_dir.mkdir()
    source_probes = {
        "artifacts": "artifacts",
        "environment-default": "environment-default",
        "environment-powershell": "environment-bash",
        "environment-cmd": "environment-bash",
        "pyte": "pyte",
        "pywinpty": "pywinpty",
    }
    for probe, template_probe in source_probes.items():
        payload = json.loads(json.dumps(templates[template_probe]))
        payload.pop("collection_command")
        payload.pop("generation_id")
        payload["row_id"] = "win-amd64-py312"
        payload["probe"] = probe
        payload["platform"] = {
            **payload["platform"],
            "os": "Windows",
            "architecture": "AMD64",
            "python_executable_name": "python.exe",
        }
        if probe.startswith("environment-"):
            shell_name = probe.removeprefix("environment-")
            payload["rows"][0]["id"] = (
                "environment-default-shell"
                if shell_name == "default"
                else f"environment-{shell_name}"
            )
            if shell_name == "default":
                payload["selected_shell_family"] = "powershell"
        (row_dir / f"{probe}.json").write_text(json.dumps(payload), encoding="utf-8")


def _make_stale_legacy_published_row(
    common: ModuleType, *, source: Path, evidence_root: Path
) -> Path:
    _write_complete_source_row(source)
    common.collect_row(row_dir=source, evidence_root=evidence_root, replace=False)
    published = evidence_root / "macos-arm64-py312"
    (published / common.CURRENT_GENERATION_MARKER).unlink()
    pyte_path = next(published.glob("*-pyte.json"))
    payload = json.loads(pyte_path.read_text(encoding="utf-8"))
    bounds = next(
        row
        for row in payload["rows"]
        if row["id"] == "parser-incomplete-sequence-bounds"
    )
    bounds["accepted_control_count"] = bounds.pop("accepted_fixture_count")
    bounds["rejected_class_count"] = bounds.pop("rejected_fixture_count")
    pyte_path.write_text(json.dumps(payload), encoding="utf-8")
    return published


def _published_payloads(row_directory: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(row_directory.glob("*.json"))
    ]


@pytest.mark.parametrize(
    ("platform_os", "expected"),
    (
        (
            "Darwin",
            {
                "artifacts",
                "environment-default",
                "environment-bash",
                "environment-zsh",
                "pyte",
                "pywinpty",
            },
        ),
        (
            "Linux",
            {
                "artifacts",
                "environment-default",
                "environment-bash",
                "environment-zsh",
                "pyte",
                "pywinpty",
            },
        ),
        (
            "Windows",
            {
                "artifacts",
                "environment-default",
                "environment-powershell",
                "environment-cmd",
                "pyte",
                "pywinpty",
            },
        ),
    ),
)
def test_required_collected_probe_set_is_exact_for_platform(
    platform_os: str, expected: set[str]
) -> None:
    common = _load_qualification_module("common")

    assert common.required_collected_probes(platform_os) == frozenset(expected)


def test_required_collected_probe_set_rejects_unknown_platform() -> None:
    common = _load_qualification_module("common")

    with pytest.raises(common.QualificationError, match="platform"):
        common.required_collected_probes("Plan9")


def test_collect_row_accepts_exact_windows_six_probe_generation(
    tmp_path: Path,
) -> None:
    common = _load_qualification_module("common")
    source = tmp_path / "source"
    evidence_root = tmp_path / "raw"
    _write_complete_windows_source_row(source)

    assert (
        common.collect_row(row_dir=source, evidence_root=evidence_root, replace=False)
        == 6
    )

    published = evidence_root / "win-amd64-py312"
    payloads = common.validate_published_row(published, recover=False)
    assert {payload["probe"] for payload in payloads} == {
        "artifacts",
        "environment-default",
        "environment-powershell",
        "environment-cmd",
        "pyte",
        "pywinpty",
    }
    marker = json.loads(
        (published / common.CURRENT_GENERATION_MARKER).read_text(encoding="utf-8")
    )
    assert {entry["name"] for entry in marker["files"]} == {
        f"win-amd64-py312-{probe}.json"
        for probe in {
            "artifacts",
            "environment-default",
            "environment-powershell",
            "environment-cmd",
            "pyte",
            "pywinpty",
        }
    }


def test_collect_row_rejects_posix_probe_family_for_windows_platform(
    tmp_path: Path,
) -> None:
    common = _load_qualification_module("common")
    source = tmp_path / "source"
    _write_complete_source_row(source)
    for path in source.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["row_id"] = "win-amd64-py312"
        payload["platform"] = {
            **payload["platform"],
            "os": "Windows",
            "architecture": "AMD64",
            "python_executable_name": "python.exe",
        }
        if payload["probe"] == "environment-default":
            payload["selected_shell_family"] = "powershell"
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(common.QualificationError, match="sibling probe set"):
        common.collect_row(
            row_dir=source,
            evidence_root=tmp_path / "raw",
            replace=False,
        )

    assert not list((tmp_path / "raw").glob("*/*.json"))


def test_collect_row_rejects_duplicate_identity_without_partial_copy(
    tmp_path: Path,
) -> None:
    row_dir = tmp_path / "row"
    row_dir.mkdir()
    payload = _minimal_raw_payload("macos-arm64-py312", "pyte")
    (row_dir / "first.json").write_text(json.dumps(payload), encoding="utf-8")
    (row_dir / "second.json").write_text(json.dumps(payload), encoding="utf-8")
    evidence_root = tmp_path / "raw"

    completed = subprocess.run(
        [
            sys.executable,
            str(QUALIFICATION_ROOT / "common.py"),
            "collect-row",
            "--row-dir",
            str(row_dir),
            "--evidence-root",
            str(evidence_root),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not list(evidence_root.glob("*/*.json"))


@pytest.mark.parametrize(
    "case",
    (
        "artifacts-object",
        "numeric-initial-key",
        "nested-command",
        "root-row-field",
        "platform-command-field",
        "numeric-working-directory",
    ),
)
def test_collect_row_rejects_wrong_types_and_shape_misplacement(
    tmp_path: Path,
    case: str,
) -> None:
    payload = _minimal_raw_payload("macos-arm64-py312", "pyte")
    if case == "artifacts-object":
        payload["artifacts"] = {}
    elif case == "numeric-initial-key":
        payload["initial_keys"] = [123]
    elif case == "nested-command":
        row = payload["rows"][0]
        assert isinstance(row, dict)
        row["command"] = payload["command"]
    elif case == "root-row-field":
        payload["artifact_count"] = 1
    elif case == "platform-command-field":
        platform = payload["platform"]
        assert isinstance(platform, dict)
        platform["argv"] = ["misplaced"]
    else:
        command = payload["command"]
        assert isinstance(command, dict)
        command["working_directory"] = 123

    completed = _collect_payload(tmp_path, payload)

    assert completed.returncode != 0, case
    assert not list((tmp_path / "raw").glob("*/*.json"))


@pytest.mark.parametrize("case", ("license", "joined-argv", "split-argv"))
def test_collect_row_rejects_sensitive_payloads_in_allowed_fields(
    tmp_path: Path,
    case: str,
) -> None:
    if case == "license":
        payload = _minimal_raw_payload("macos-arm64-py312", "artifacts")
        artifacts = payload["artifacts"]
        assert isinstance(artifacts, list) and isinstance(artifacts[0], dict)
        artifacts[0]["license"] = "credential=correct-horse-battery-staple"
    else:
        payload = _minimal_raw_payload("macos-arm64-py312", "pyte")
        command = payload["command"]
        assert isinstance(command, dict)
        argv = command["argv"]
        assert isinstance(argv, list)
        if case == "joined-argv":
            argv.append("--token correct-horse-battery-staple")
        else:
            argv.extend(("--token", "correct-horse-battery-staple"))

    completed = _collect_payload(tmp_path, payload)

    assert completed.returncode != 0, case
    assert not list((tmp_path / "raw").glob("*/*.json"))


@pytest.mark.parametrize(
    ("probe", "foreign_key", "foreign_value"),
    (
        ("artifacts", "job_handle_non_inheritable", True),
        ("environment-bash", "artifact_count", 1),
        ("pyte", "job_handle_non_inheritable", True),
        ("pyte", "four_session_count", 4),
        ("pyte", "fixture_byte_count", 1),
        ("pywinpty", "captured_count", 1),
        ("pywinpty", "four_session_count", 4),
    ),
)
def test_collect_row_rejects_fields_from_another_probe_or_row_schema(
    tmp_path: Path,
    probe: str,
    foreign_key: str,
    foreign_value: object,
) -> None:
    row_id = "win-amd64-py311" if probe == "pywinpty" else "macos-arm64-py312"
    payload = _minimal_raw_payload(row_id, probe)
    rows = payload["rows"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    rows[0][foreign_key] = foreign_value

    completed = _collect_payload(tmp_path, payload)

    assert completed.returncode != 0, (probe, foreign_key)
    assert not list((tmp_path / "raw").glob("*/*.json"))


@pytest.mark.parametrize(
    "case",
    (
        "artifact-license-bearer-jwt",
        "artifact-license-authorization-bearer",
        "artifact-license-jwt-shape",
        "argv-github-classic-token",
        "argv-github-fine-grained-token",
    ),
)
def test_collect_row_rejects_common_secret_shapes_in_allowed_fields(
    tmp_path: Path,
    case: str,
) -> None:
    jwt = ".".join(
        (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlRlc3QifQ",
            "c2lnbmF0dXJlLWZpeHR1cmU",
        )
    )
    if case.startswith("artifact-license"):
        payload = _minimal_raw_payload("macos-arm64-py312", "artifacts")
        artifacts = payload["artifacts"]
        assert isinstance(artifacts, list) and isinstance(artifacts[0], dict)
        if case == "artifact-license-bearer-jwt":
            artifacts[0]["license"] = f"Bearer {jwt}"
        elif case == "artifact-license-authorization-bearer":
            artifacts[0]["license"] = f"Authorization: Bearer {jwt}"
        else:
            artifacts[0]["license"] = jwt
    else:
        payload = _minimal_raw_payload("macos-arm64-py312", "pyte")
        command = payload["command"]
        assert isinstance(command, dict)
        argv = command["argv"]
        assert isinstance(argv, list)
        if case == "argv-github-classic-token":
            argv.append("gh" + "p_" + "a" * 36)
        else:
            argv.append("github" + "_pat_" + "a" * 40)

    completed = _collect_payload(tmp_path, payload)

    assert completed.returncode != 0, case
    assert not list((tmp_path / "raw").glob("*/*.json"))


def test_command_facts_uses_invoked_venv_interpreter_not_runtime_base(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_qualification_module("common")
    invoked = tmp_path / "row" / "venv" / "bin" / "python"
    runtime_base = Path("/opt/homebrew/Frameworks/Python.app/Contents/MacOS/Python")
    monkeypatch.setattr(module.sys, "executable", str(invoked))
    monkeypatch.setattr(
        module.sys,
        "orig_argv",
        [str(runtime_base), "-B", "probe.py", "--json-out", "row.json"],
    )

    facts = module.command_facts()

    assert facts["argv"] == [
        str(invoked),
        "-B",
        "probe.py",
        "--json-out",
        "row.json",
    ]


@pytest.mark.parametrize(
    ("location", "key"),
    (
        ("root", "renamed_secret"),
        ("row", "diagnostic_note"),
        ("root", "reason_category"),
    ),
)
def test_collect_row_rejects_secret_strings_under_unknown_or_renamed_keys(
    tmp_path: Path,
    location: str,
    key: str,
) -> None:
    row_dir = tmp_path / "row"
    row_dir.mkdir()
    payload = _minimal_raw_payload("macos-arm64-py312", "pyte")
    target = payload if location == "root" else payload["rows"][0]
    assert isinstance(target, dict)
    target[key] = "sk-live-secret-material-must-not-survive"
    (row_dir / "pyte.json").write_text(json.dumps(payload), encoding="utf-8")
    evidence_root = tmp_path / "raw"

    completed = subprocess.run(
        [
            sys.executable,
            str(QUALIFICATION_ROOT / "common.py"),
            "collect-row",
            "--row-dir",
            str(row_dir),
            "--evidence-root",
            str(evidence_root),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not list(evidence_root.glob("*/*.json"))


def test_collected_rows_record_exact_generation_and_collection_argv() -> None:
    evidence = EVIDENCE.read_text(encoding="utf-8")
    assert "<ROW_" not in evidence
    assert "<EXACT-" not in evidence
    raw_files = sorted(RAW_ROOT.glob("*/*.json"))
    assert raw_files
    for raw_file in raw_files:
        payload = json.loads(raw_file.read_text(encoding="utf-8"))
        generated = payload.get("command")
        collected = payload.get("collection_command")
        assert isinstance(generated, dict), raw_file
        assert isinstance(collected, dict), raw_file
        for command in (generated, collected):
            argv = command.get("argv")
            assert isinstance(argv, list) and argv
            assert all(isinstance(item, str) and item for item in argv)
            assert not any(re.search(r"<(?:ROW_|EXACT-)[^>]*>", item) for item in argv)
            working_directory = str(command.get("working_directory"))
            assert (
                Path(working_directory).is_absolute()
                or PureWindowsPath(working_directory).is_absolute()
            )
        if raw_file.parent.name.startswith(("macos-", "linux-")):
            assert "--replace" in collected["argv"]


def test_non_windows_pywinpty_probe_records_complete_fail_closed_rows(
    tmp_path: Path,
) -> None:
    if sys.platform == "win32":
        return
    manifest = tmp_path / "artifacts.json"
    manifest.write_text(
        json.dumps(_minimal_raw_payload("macos-arm64-py312", "artifacts")),
        encoding="utf-8",
    )
    output = tmp_path / "winpty.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(QUALIFICATION_ROOT / "pywinpty_probe.py"),
            "--artifact-manifest",
            str(manifest),
            "--json-out",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert output.is_file(), completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "UNSUPPORTED_FAIL_CLOSED"
    rows = {row["id"]: row["status"] for row in payload["rows"]}
    windows_rows = {row for row in MANDATORY_ROWS if row.startswith("windows-")}
    assert windows_rows <= rows.keys()
    assert {rows[row] for row in windows_rows} == {"UNSUPPORTED_FAIL_CLOSED"}
    assert rows["four-session-managed-rss"] == "UNSUPPORTED_FAIL_CLOSED"


def test_pyte_preparser_bounds_every_incomplete_sequence_class() -> None:
    module = _load_qualification_module("pyte_probe")
    guard = getattr(module, "sequence_is_bounded", lambda _: True)
    rejected = (
        b"\xe2\x82",
        b"\x1b" + b"x" * 16,
        b"\x1b[" + b"1" * 257,
        b"\x1b[" + b"1;" * 33 + b"m",
        b"\x1b[10000m",
        b"\x1b[" + b" " * 17 + b"m",
        b"\x1b]" + b"x" * 4096,
        b"\x1bP" + b"x" * 4096,
        b"\x1b_" + b"x" * 4096,
        b"\x1b^" + b"x" * 4096,
    )
    accepted = (
        "€".encode(),
        b"\x1b[31m",
        b"\x1b[9999m",
        b"\x1b" + b"x" * 15,
        b"\x1b]title\x07",
        b"\x1bPdata\x1b\\",
    )

    for value in rejected:
        assert not guard(value), value[:16]
    for value in accepted:
        assert guard(value), value


def test_pyte_rejects_a_five_digit_csi_numeric_value() -> None:
    module = _load_qualification_module("pyte_probe")

    assert not module.sequence_is_bounded(b"\x1b[10000m")


def test_pyte_rejects_a_seventeen_byte_non_csi_control() -> None:
    module = _load_qualification_module("pyte_probe")

    assert not module.sequence_is_bounded(b"\x1b" + b"x" * 16)


def test_pyte_retains_the_exact_approved_sequence_limit_facts() -> None:
    module = _load_qualification_module("pyte_probe")

    assert module._sequence_limit_facts(
        accepted_fixture_count=6,
        rejected_fixture_count=10,
    ) == {
        "accepted_fixture_count": 6,
        "rejected_fixture_count": 10,
        "control_sequence_byte_limit": 4096,
        "csi_parameter_limit": 32,
        "csi_parameter_digit_limit": 4,
        "csi_parameter_value_limit": 9999,
        "csi_private_intermediate_byte_limit": 16,
        "non_csi_control_byte_limit": 16,
        "string_control_byte_limit": 4096,
    }


def test_pyte_alternate_screen_enters_exits_and_restores_primary_buffer() -> None:
    module = _load_qualification_module("pyte_probe")
    probe = getattr(module, "_alternate_screen_facts", None)
    assert callable(probe), "pyte probe must expose the alternate-screen behavior"

    class FakeScreen:
        def __init__(self, columns: int, lines: int) -> None:
            self.columns = columns
            self.lines = lines
            self.text = ""
            self.cursor = SimpleNamespace(x=0, y=0)
            self.saved_cursor: tuple[int, int] | None = None

        @property
        def display(self) -> list[str]:
            return [self.text.ljust(self.columns)[: self.columns]] + [
                " " * self.columns
            ] * (self.lines - 1)

        def draw(self, value: str) -> None:
            self.text += value
            self.cursor.x += len(value)

        def reset(self) -> None:
            self.text = ""
            self.cursor.x = 0
            self.cursor.y = 0

        def save_cursor(self) -> None:
            self.saved_cursor = (self.cursor.x, self.cursor.y)

        def restore_cursor(self) -> None:
            assert self.saved_cursor is not None
            self.cursor.x, self.cursor.y = self.saved_cursor

        def set_mode(self, *_: int, **__: object) -> None:
            return None

        def reset_mode(self, *_: int, **__: object) -> None:
            return None

        def __getattr__(self, _: str):
            return lambda *args, **kwargs: None

    class FakeStream:
        def __init__(self, listener: object) -> None:
            self.listener = listener

        def feed(self, value: str) -> None:
            if value == "\x1b[?1049h":
                self.listener.set_mode(1049, private=True)
            elif value == "\x1b[?1049l":
                self.listener.reset_mode(1049, private=True)
            else:
                self.listener.draw(value)

    facts = probe(SimpleNamespace(Screen=FakeScreen, Stream=FakeStream))

    assert facts == {
        "control_sequence_count": 2,
        "entered": True,
        "entry_count": 1,
        "exited": True,
        "exit_count": 1,
        "alternate_isolated": True,
        "primary_restored": True,
    }


def test_available_posix_shells_are_mandatory_but_missing_optional_shells_are_not() -> (
    None
):
    module = _load_qualification_module("environment_probe")
    policy = getattr(module, "_shell_is_mandatory", None)
    assert callable(policy), "environment probe must expose one shell policy"
    for shell in ("bash", "zsh"):
        assert policy(shell, available=True, windows=False)
        assert not policy(shell, available=False, windows=False)
    assert policy("default", available=False, windows=False)
    assert policy("powershell", available=False, windows=True)
    assert policy("cmd", available=False, windows=True)


@pytest.mark.parametrize(
    ("available", "expected_path", "expected_family", "lookups"),
    (
        (
            {"pwsh.exe": r"C:\Program Files\PowerShell\7\pwsh.exe"},
            r"C:\Program Files\PowerShell\7\pwsh.exe",
            "powershell",
            ["pwsh.exe"],
        ),
        (
            {"powershell.exe": r"C:\Windows\System32\WindowsPowerShell\powershell.exe"},
            r"C:\Windows\System32\WindowsPowerShell\powershell.exe",
            "powershell",
            ["pwsh.exe", "powershell.exe"],
        ),
        (
            {},
            r"C:\Windows\System32\cmd.exe",
            "cmd",
            ["pwsh.exe", "powershell.exe"],
        ),
    ),
)
def test_windows_default_shell_selection_is_code_owned_and_ordered(
    available: dict[str, str],
    expected_path: str,
    expected_family: str,
    lookups: list[str],
) -> None:
    module = _load_qualification_module("environment_probe")
    observed: list[str] = []

    def locate(name: str, *, path: str | None) -> str | None:
        assert path == r"C:\Windows\System32"
        observed.append(name)
        return available.get(name)

    selection = module._resolve_shell(
        "default",
        {
            "PATH": r"C:\Windows\System32",
            "COMSPEC": r"C:\Windows\System32\cmd.exe",
        },
        windows=True,
        which=locate,
        is_file=lambda path: str(path) == r"C:\Windows\System32\cmd.exe",
    )

    assert selection.path == Path(expected_path)
    assert selection.family == expected_family
    assert observed == lookups


def test_windows_default_probe_records_selected_family_in_current_schema(
    tmp_path: Path,
) -> None:
    common = _load_qualification_module("common")
    payload = _minimal_raw_payload("win-amd64-py312", "environment-default")
    payload["platform"] = {
        **payload["platform"],
        "os": "Windows",
        "architecture": "AMD64",
        "python_executable_name": "python.exe",
    }
    payload["selected_shell_family"] = "powershell"

    common.validate_content_free(payload)

    payload.pop("selected_shell_family")
    with pytest.raises(common.QualificationError, match="selected shell"):
        common.validate_content_free(payload)


def test_cmd_contract_uses_normal_autorun_profile_and_discovery(monkeypatch) -> None:
    module = _load_qualification_module("environment_probe")
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = list(argv)
        observed["bootstrap_setup"] = kwargs["bootstrap_setup"]
        nonce = kwargs["result_nonce"]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(module._result_marker(nonce) + "0,0,0,0\r\n").encode(),
            stderr=b"",
        )

    monkeypatch.setattr(module, "_run_shell_process", fake_run)
    result = module._run_windows_shell("cmd", Path("cmd.exe"), {"TERM": "linux"})

    assert result.get("profile_contract_applicable") is True
    assert result["startup_completed"] is True
    assert result["command_discovery"] is True
    assert result["profile_marker_present"] is True
    assert result["sensitive_key_repopulated_by_profile"] is True
    assert result["default_module_discovery"] is True
    assert result["profile_extended_module_discovery"] is True
    argv = observed["argv"]
    assert isinstance(argv, list)
    assert "/C" not in argv and "/D" not in argv
    setup = observed["bootstrap_setup"]
    assert setup.profile_files == ()
    assert setup.registry_values == (
        (
            r"Software\Microsoft\Command Processor",
            "AutoRun",
            "@set TLDW_QUALIFICATION_PROFILE=1"
            "&@set OPENAI_API_KEY=fixture-restored"
            "&@set TLDW_TASK22512_AUTORUN=1",
        ),
    )
    contract = getattr(module, "_windows_shell_passed", None)
    assert callable(contract)
    assert contract("cmd", result)


def test_powershell_uses_controlled_profile_and_module_fixture(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_qualification_module("environment_probe")
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = list(argv)
        observed["environment"] = dict(kwargs["env"])
        observed["input"] = kwargs["input_bytes"].decode()
        observed["bootstrap_setup"] = kwargs["bootstrap_setup"]
        nonce = kwargs["result_nonce"]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(module._result_marker(nonce) + "0,0,0,0\r\n").encode(),
            stderr=b"",
        )

    monkeypatch.setattr(module, "_run_shell_process", fake_run)
    result = module._run_windows_shell(
        "powershell",
        Path("powershell.exe"),
        {"PATH": str(tmp_path), "TERM": "linux"},
    )

    argv = observed["argv"]
    assert isinstance(argv, list)
    assert "-NoProfile" not in argv
    assert "-NonInteractive" not in argv
    assert "-File" not in argv
    setup = observed["bootstrap_setup"]
    profile_paths = {path for path, _ in setup.profile_files}
    assert "Modules/TldwTask22512Probe/TldwTask22512Probe.psm1" in profile_paths
    assert "Documents/WindowsPowerShell/profile.ps1" in profile_paths
    assert "Documents/PowerShell/profile.ps1" in profile_paths
    assert setup.registry_values == ()
    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert "USERPROFILE" not in environment
    assert "PSMODULEPATH" not in environment
    assert "OPENAI_API_KEY" not in environment
    assert result["profile_marker_present"] is True
    assert result["sensitive_key_repopulated_by_profile"] is True
    assert result["module_discovery"] is True
    assert result["default_module_discovery"] is True
    assert result["profile_extended_module_discovery"] is True
    assert "correct-horse-battery-staple" not in json.dumps(result)
    assert module._windows_shell_passed("powershell", result)


def test_windows_probe_ast_uses_only_explicit_low_level_conpty_construction() -> None:
    source = (QUALIFICATION_ROOT / "pywinpty_probe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    direct_pty_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "winpty"
        and node.func.attr == "PTY"
    ]
    assert direct_pty_calls, "native path must directly construct winpty.PTY"
    assert any(
        isinstance(keyword.value, ast.Attribute)
        and keyword.value.attr == "ConPTY"
        and isinstance(keyword.value.value, ast.Attribute)
        and keyword.value.value.attr == "Backend"
        and isinstance(keyword.value.value.value, ast.Name)
        and keyword.value.value.value.id == "winpty"
        for call in direct_pty_calls
        for keyword in call.keywords
        if keyword.arg == "backend"
    )
    native = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_native_probe"
    )
    assert not any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and any(
            isinstance(item, ast.Constant) and item.value is False
            for item in node.value.elts
        )
        for node in ast.walk(native)
    )


def test_windows_spawn_session_does_not_duplicate_python_application_name(
    monkeypatch,
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    observed: dict[str, object] = {}

    class FakeTerminal:
        pid = 42

        def spawn(self, appname, cmdline, **kwargs) -> bool:
            observed.update({"appname": appname, "cmdline": cmdline, "kwargs": kwargs})
            return True

    terminal = FakeTerminal()
    winpty = SimpleNamespace(
        Backend=SimpleNamespace(ConPTY=0),
        PTY=lambda columns, rows, backend: terminal,
    )
    executable = r"C:\Python311\python.exe"
    monkeypatch.setattr(module.sys, "executable", executable)

    assert module._spawn_session(winpty, "crash-live") is terminal

    assert observed["appname"] == executable
    assert observed["cmdline"] == subprocess.list2cmdline(
        ["-u", "-c", module._fixture_source(), "crash-live"]
    )
    assert observed["kwargs"] == {"cwd": tempfile.gettempdir(), "env": None}


def test_windows_controller_assigns_job_before_releasing_fd_backed_worker(
    monkeypatch,
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    events: list[object] = []
    popen_kwargs: dict[str, object] = {}

    class FakeJob:
        def __init__(self) -> None:
            self.handle = 1
            self._kernel32 = object()
            events.append("job-created")

        def non_inheritable(self) -> bool:
            return True

        def assign(self, process_id: int) -> None:
            events.append(("assigned", process_id))

        def contains(self, process_id: int) -> bool:
            return process_id in {40, 41, 42, 43, 44, 45}

        def process_ids(self) -> list[int]:
            return [40, 41, 42, 43, 44, 45]

        def retain_process_handles(self, process_ids) -> list[int]:
            events.append(("retained", tuple(process_ids)))
            return [101, 102, 103, 104, 105, 106]

        def close(self) -> None:
            events.append("job-closed")
            self.handle = None

    class FakeProcess:
        pid = 40

        def __init__(self, argv, **kwargs) -> None:
            events.append("worker-spawned")
            popen_kwargs.update(kwargs)
            self.finished = False

        def wait(self, timeout: float) -> int:
            events.append(("worker-wait", timeout))
            self.finished = True
            return 1

        def poll(self) -> int | None:
            return 1 if self.finished else None

        def kill(self) -> None:
            raise AssertionError("Job close must terminate the worker")

    worker = _complete_windows_observations()
    monkeypatch.setattr(module, "_WindowsJob", FakeJob)
    monkeypatch.setattr(module, "_create_event", lambda name: name)
    monkeypatch.setattr(module, "_close_handle", lambda handle: None)
    monkeypatch.setattr(module, "_wait_event", lambda handle, timeout: True)
    monkeypatch.setattr(module, "_load_worker_result", lambda path: worker)
    monkeypatch.setattr(
        module,
        "_load_rss_fixture_ids",
        lambda path: [41, 42, 43, 44],
    )
    monkeypatch.setattr(
        module,
        "_aggregate_working_set",
        lambda process_ids: 100 if process_ids == [os.getpid(), 40] else 200,
    )
    monkeypatch.setattr(
        module,
        "_wait_retained_process_handles",
        lambda kernel32, handles, timeout_seconds: (
            events.append(("retained-wait", tuple(handles), timeout_seconds))
            or (True, len(handles))
        ),
    )
    monkeypatch.setattr(
        module,
        "_run_app_crash_supervisor",
        lambda: {
            key: value
            for key, value in worker.items()
            if key.startswith("crash_") or key == "app_crash_observed"
        },
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_set_event",
        lambda handle: events.append(("event-set", handle)),
    )
    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        module.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        0x200,
        raising=False,
    )
    observations = module._default_observations()

    module._run_native_controller(Path("artifacts.json"), observations)

    assert events.index("worker-spawned") < events.index(("assigned", 40))
    start_release = next(
        event
        for event in events
        if isinstance(event, tuple)
        and isinstance(event[1], str)
        and "start" in event[1]
    )
    rss_release = next(
        event
        for event in events
        if isinstance(event, tuple)
        and isinstance(event[1], str)
        and "rss-continue" in event[1]
    )
    assert events.index(("assigned", 40)) < events.index(start_release)
    assert events.index(start_release) < events.index(rss_release)
    assert events.index(rss_release) < events.index("job-closed")
    assert events.index("job-closed") < events.index(
        ("worker-wait", module.CLOSE_TIMEOUT_SECONDS)
    )
    retained = next(
        event for event in events if isinstance(event, tuple) and event[0] == "retained"
    )
    waited = next(
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "retained-wait"
    )
    assert events.index(retained) < events.index("job-closed") < events.index(waited)
    assert popen_kwargs["close_fds"] is True
    assert popen_kwargs["creationflags"] == 0x200
    for name in ("stdin", "stdout", "stderr"):
        assert hasattr(popen_kwargs[name], "fileno")
    assert observations["priority_close_completed"] is True
    assert observations["priority_close_preempted_inflight"] is True
    assert observations["rss_fixture_process_count"] == 4
    assert observations["rss_crash_session_present"] is False
    assert observations["normal_cleanup_expected_process_count"] == 6
    assert observations["normal_cleanup_retained_handle_count"] == 6
    assert observations["normal_cleanup_wait_object_0_count"] == 6
    assert observations["normal_cleanup_all_wait_object_0"] is True


def test_windows_crash_source_enforces_separate_job_handle_ownership() -> None:
    source = (QUALIFICATION_ROOT / "pywinpty_probe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    app = functions["_native_crash_app_controller"]
    supervisor = functions["_run_app_crash_supervisor"]
    crash_worker = functions["_native_crash_worker"]
    normal_worker = functions["_worker_observations"]
    synchronize_opener = functions["_open_synchronize_process_handles"]

    def direct_calls(function: ast.AST, name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]

    assert len(direct_calls(app, "_WindowsJob")) == 1
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.func.attr == "abort"
        for node in ast.walk(app)
    )
    assert not direct_calls(supervisor, "_WindowsJob")
    assert not direct_calls(crash_worker, "_WindowsJob")
    assert direct_calls(supervisor, "_open_synchronize_process_handles")
    assert direct_calls(supervisor, "_wait_retained_process_handles")
    forbidden_job_apis = {"CreateJobObjectW", "OpenJobObjectW", "DuplicateHandle"}
    assert not any(
        (
            isinstance(node, ast.Name)
            and node.id in forbidden_job_apis
            or isinstance(node, ast.Attribute)
            and node.attr in forbidden_job_apis
        )
        for node in ast.walk(supervisor)
    )
    open_process_calls = [
        node
        for node in ast.walk(synchronize_opener)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "OpenProcess"
    ]
    assert len(open_process_calls) == 1
    assert isinstance(open_process_calls[0].args[0], ast.Name)
    assert open_process_calls[0].args[0].id == "SYNCHRONIZE_ACCESS"

    normal_worker_fact_names = {
        target.slice.value
        for node in ast.walk(normal_worker)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id == "observations"
        and isinstance(target.slice, ast.Constant)
        and isinstance(target.slice.value, str)
    }
    assert "app_crash_observed" not in normal_worker_fact_names
    assert {
        "terminal_child_member_before_crash",
        "terminal_grandchild_member_before_crash",
    } <= normal_worker_fact_names
    assert direct_calls(normal_worker, "_request_terminal_crash")


def test_windows_crash_app_accepts_contained_venv_redirector_then_aborts(
    monkeypatch,
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    events: list[object] = []
    written: dict[str, object] = {}
    popen_kwargs: dict[str, object] = {}

    class AbortObserved(BaseException):
        pass

    class FakeJob:
        instances = 0

        def __init__(self) -> None:
            type(self).instances += 1
            self.handle = 1
            events.append("job-created")

        def assign(self, process_id: int) -> None:
            events.append(("worker-assigned", process_id))

        def contains(self, process_id: int) -> bool:
            return process_id in {601, 602, 603, 604}

        def non_inheritable(self) -> bool:
            return True

        def close(self) -> None:
            events.append("job-closed")
            self.handle = None

    class FakeProcess:
        pid = 601

        def __init__(self, argv, **kwargs) -> None:
            del argv
            popen_kwargs.update(kwargs)
            events.append("worker-spawned")

        def poll(self) -> int:
            return 1

    worker_result = {
        "worker_process_id": 604,
        "terminal_process_id": 602,
        "terminal_descendant_process_id": 603,
        "worker_admitted_before_conpty": True,
        "terminal_member": True,
        "terminal_descendant_member": True,
    }
    monkeypatch.setattr(module, "_WindowsJob", FakeJob)
    monkeypatch.setattr(module, "_create_event", lambda name: name)
    monkeypatch.setattr(module, "_close_handle", lambda handle: None)
    monkeypatch.setattr(
        module,
        "_set_event",
        lambda handle: events.append(("worker-released", handle)),
    )
    monkeypatch.setattr(
        module,
        "_wait_event",
        lambda handle, timeout: events.append(("worker-ready", handle)) or True,
    )
    monkeypatch.setattr(
        module,
        "_load_crash_worker_result",
        lambda path: worker_result,
    )
    monkeypatch.setattr(
        module,
        "_stable_job_process_ids",
        lambda job, process_ids, timeout_seconds: (
            events.append(("membership-stable", tuple(process_ids)))
            or [601, 602, 603, 604]
        ),
    )
    monkeypatch.setattr(
        module,
        "_write_crash_result",
        lambda path, payload: (
            written.update(payload) or events.append("result-written")
        ),
    )
    monkeypatch.setattr(
        module,
        "_signal_named_event",
        lambda name: events.append("supervisor-ready"),
    )
    monkeypatch.setattr(
        module,
        "_wait_named_event",
        lambda name, timeout: events.append("supervisor-retained") or True,
    )
    monkeypatch.setattr(
        module.os,
        "abort",
        lambda: events.append("app-abort") or (_ for _ in ()).throw(AbortObserved()),
    )
    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        module.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        0x200,
        raising=False,
    )

    with pytest.raises(AbortObserved):
        module._native_crash_app_controller(
            Path("app.json"),
            Path("worker.json"),
            "supervisor-ready-event",
            "supervisor-retained-event",
        )

    assert FakeJob.instances == 1
    assert events.index("worker-spawned") < events.index(("worker-assigned", 601))
    assert events.index(("worker-assigned", 601)) < next(
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "worker-released"
    )
    assert events.index("result-written") < events.index("supervisor-ready")
    assert events.index("supervisor-ready") < events.index("supervisor-retained")
    assert events.index("supervisor-retained") < events.index("app-abort")
    assert ("membership-stable", (601, 604, 602, 603)) in events
    assert written == {
        "app_process_id": os.getpid(),
        "job_handle_owner_count": 1,
        "job_handle_non_inheritable": True,
        "worker_admitted_before_conpty": True,
        "descendant_set_stable": True,
        "known_descendant_process_ids": [601, 602, 603, 604],
    }
    assert popen_kwargs["close_fds"] is True


def test_windows_crash_supervisor_opens_only_synchronize_authority(
    monkeypatch,
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    open_calls: list[tuple[int, bool, int]] = []

    class FakeOpenProcess:
        argtypes = None
        restype = None

        def __call__(self, access: int, inherit: bool, process_id: int) -> int:
            open_calls.append((access, inherit, process_id))
            return process_id + 1000

    class FakeKernel32:
        OpenProcess = FakeOpenProcess()

        def CloseHandle(self, handle: int) -> bool:
            return True

    wintypes = SimpleNamespace(DWORD=int, BOOL=bool, HANDLE=int)
    kernel32 = FakeKernel32()
    monkeypatch.setattr(module, "_windows_api", lambda: (object(), wintypes, kernel32))

    returned_kernel32, handles = module._open_synchronize_process_handles(
        [601, 602, 603]
    )

    assert returned_kernel32 is kernel32
    assert handles == [1601, 1602, 1603]
    assert open_calls == [
        (0x00100000, False, 601),
        (0x00100000, False, 602),
        (0x00100000, False, 603),
    ]


def test_windows_crash_supervisor_accepts_venv_redirector_before_app_abort(
    monkeypatch,
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    events: list[object] = []
    popen_argv: list[str] = []

    class FakeProcess:
        pid = 500

        def __init__(self, argv, **kwargs) -> None:
            del kwargs
            popen_argv.extend(argv)
            events.append("app-spawned")
            self.returncode: int | None = None

        def wait(self, timeout: float) -> int:
            assert "abort-released" in events
            events.append(("app-wait", timeout))
            self.returncode = 3
            return self.returncode

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            raise AssertionError("ready app/controller must abort itself")

    app_result = {
        "app_process_id": 501,
        "job_handle_owner_count": 1,
        "job_handle_non_inheritable": True,
        "worker_admitted_before_conpty": True,
        "descendant_set_stable": True,
        "known_descendant_process_ids": [601, 602, 603],
    }

    monkeypatch.setattr(
        module,
        "_create_event",
        lambda name: events.append(("event-created", name)) or name,
    )
    monkeypatch.setattr(module, "_close_handle", lambda handle: None)
    monkeypatch.setattr(
        module,
        "_wait_event",
        lambda handle, timeout: events.append(("app-ready", handle, timeout)) or True,
    )
    monkeypatch.setattr(
        module,
        "_load_crash_app_result",
        lambda path: events.append(("result-loaded", path)) or app_result,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_open_synchronize_process_handles",
        lambda process_ids: (
            events.append(("sync-opened", tuple(process_ids))) or "kernel32",
            [701, 702, 703],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_set_event",
        lambda handle: events.append("abort-released"),
    )
    monkeypatch.setattr(
        module,
        "_wait_retained_process_handles",
        lambda kernel32, handles, timeout_seconds: (
            events.append(
                ("descendants-waited", kernel32, tuple(handles), timeout_seconds)
            )
            or (True, 3)
        ),
    )
    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        module.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        0x200,
        raising=False,
    )

    facts = module._run_app_crash_supervisor()

    assert "--native-crash-app-controller" in popen_argv
    assert events.index(("sync-opened", (601, 602, 603))) < events.index(
        "abort-released"
    )
    assert events.index("abort-released") < next(
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "app-wait"
    )
    assert next(
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "app-wait"
    ) < next(
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "descendants-waited"
    )
    assert facts == {
        "app_crash_observed": True,
        "crash_app_process_separate": True,
        "crash_app_sole_job_handle_owner": True,
        "crash_job_handle_non_inheritable": True,
        "crash_worker_admitted_before_conpty": True,
        "crash_descendant_set_stable": True,
        "crash_descendants_ready_before_abort": True,
        "crash_supervisor_job_handle_count": 0,
        "crash_known_descendant_count": 3,
        "crash_supervisor_synchronize_handle_count": 3,
        "crash_wait_object_0_count": 3,
        "crash_all_descendants_wait_object_0": True,
    }


def test_windows_output_credit_caps_measured_pty_chunks_at_64_kib() -> None:
    module = _load_qualification_module("pywinpty_probe")

    class FakeTerminal:
        def read(self, blocking: bool = False) -> str:
            assert blocking
            return "x" * module.UPSTREAM_READ_BUFFER_BYTES

    credit = module._OutputCredit()
    chunk = credit.read(FakeTerminal())

    assert len(chunk.data) == module.UPSTREAM_READ_BUFFER_BYTES
    assert credit.max_chunk_bytes == module.UPSTREAM_READ_BUFFER_BYTES
    assert credit.max_unacknowledged == 1
    assert credit.outstanding == 1
    credit.acknowledge(chunk)
    assert credit.outstanding == 0


def test_windows_output_credit_rejects_a_second_unacknowledged_read() -> None:
    module = _load_qualification_module("pywinpty_probe")
    entered = threading.Event()
    release = threading.Event()

    class FakeTerminal:
        def read(self, blocking: bool = False) -> str:
            assert blocking
            entered.set()
            assert release.wait(timeout=2.0)
            return "x"

    credit = module._OutputCredit()
    terminal = FakeTerminal()
    reader = threading.Thread(target=credit.read, args=(terminal,))
    reader.start()
    assert entered.wait(timeout=2.0)

    with pytest.raises(module.QualificationError, match="already outstanding"):
        credit.read(terminal)

    release.set()
    reader.join(timeout=2.0)
    assert not reader.is_alive()
    assert credit.max_unacknowledged == 1


@pytest.mark.parametrize(
    ("field", "failing_row"),
    (
        ("fresh_worker", "windows-low-level-api"),
        ("worker_std_streams_fd_backed", "windows-low-level-api"),
        ("job_member_count", "windows-job-admission-membership"),
        ("measured_chunk_count", "windows-one-credit-bounded-read"),
        ("max_unacknowledged_credits", "windows-one-credit-bounded-read"),
        ("concurrent_operation_count", "windows-concurrent-io-close"),
        ("inflight_operation_category_count", "windows-concurrent-io-close"),
        ("priority_close_preempted_inflight", "windows-concurrent-io-close"),
        ("quiet_terminal_startup_drained", "windows-concurrent-io-close"),
        ("read_entered", "windows-concurrent-io-close"),
        ("write_completed_at_handoff", "windows-concurrent-io-close"),
        ("read_completed_post_close", "windows-concurrent-io-close"),
        (
            "normal_cleanup_all_wait_object_0",
            "windows-job-admission-membership",
        ),
        (
            "normal_cleanup_expected_process_count",
            "windows-job-admission-membership",
        ),
        (
            "normal_cleanup_retained_handle_count",
            "windows-job-admission-membership",
        ),
        (
            "normal_cleanup_wait_object_0_count",
            "windows-job-admission-membership",
        ),
        ("alternate_isolated", "windows-unicode-alternate-screen"),
        ("primary_restored", "windows-unicode-alternate-screen"),
        ("app_crash_observed", "windows-app-crash-descendant-cleanup"),
        ("crash_app_process_separate", "windows-app-crash-descendant-cleanup"),
        (
            "crash_app_sole_job_handle_owner",
            "windows-app-crash-descendant-cleanup",
        ),
        (
            "crash_descendants_ready_before_abort",
            "windows-app-crash-descendant-cleanup",
        ),
        (
            "crash_all_descendants_wait_object_0",
            "windows-app-crash-descendant-cleanup",
        ),
        ("crash_known_descendant_count", "windows-app-crash-descendant-cleanup"),
        (
            "crash_supervisor_synchronize_handle_count",
            "windows-app-crash-descendant-cleanup",
        ),
        ("crash_wait_object_0_count", "windows-app-crash-descendant-cleanup"),
        ("four_session_count", "four-session-managed-rss"),
        ("rss_measurement_complete", "four-session-managed-rss"),
    ),
)
def test_windows_native_rows_fail_closed_when_required_observation_is_missing(
    field: str,
    failing_row: str,
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    observations = _complete_windows_observations()
    observations[field] = False if isinstance(observations[field], bool) else 0

    rows = {row["id"]: row for row in module._build_native_rows(observations)}

    assert rows[failing_row]["status"] == "FAIL"


def _complete_windows_observations() -> dict[str, object]:
    return {
        "artifact_filename": "pywinpty-3.0.5-cp311-cp311-win_amd64.whl",
        "artifact_sha256": "a" * 64,
        "artifact_size_bytes": 1,
        "artifact_verified_during_probe": True,
        "distribution_version": "3.0.5",
        "primary_file_name": "winpty.pyd",
        "primary_file_sha256": "b" * 64,
        "record_file_name": "pywinpty-3.0.5.dist-info/RECORD",
        "record_file_sha256": "c" * 64,
        "windows_build": 17763,
        "fresh_worker": True,
        "worker_std_streams_fd_backed": True,
        "low_level_api": True,
        "conpty_constructed": True,
        "job_admitted_before_conpty": True,
        "job_membership_complete": True,
        "job_member_count": 6,
        "job_handle_non_inheritable": True,
        "one_credit_max_bytes": 32768,
        "measured_chunk_count": 1,
        "read_api_accepts_size": False,
        "upstream_read_buffer_bytes": 32768,
        "max_unacknowledged_credits": 1,
        "max_concurrent_readers": 1,
        "concurrent_operation_count": 4,
        "inflight_operation_category_count": 1,
        "io_inflight_at_handoff": True,
        "priority_close_preempted_inflight": True,
        "quiet_terminal_startup_drained": True,
        "quiet_terminal_quiescent_before_handoff": True,
        "read_entered": True,
        "write_entered": True,
        "resize_entered": True,
        "cancel_entered": True,
        "read_completed_at_handoff": False,
        "write_completed_at_handoff": True,
        "resize_completed_at_handoff": True,
        "cancel_completed_at_handoff": True,
        "read_completed_post_close": True,
        "write_completed_post_close": False,
        "resize_completed_post_close": False,
        "cancel_completed_post_close": False,
        "write_completed": True,
        "resize_completed": True,
        "cancel_completed": True,
        "priority_close_completed": True,
        "read_returned_after_close": True,
        "write_returned_after_close": False,
        "resize_returned_after_close": False,
        "cancel_returned_after_close": False,
        "normal_cleanup_expected_process_count": 6,
        "normal_cleanup_retained_handle_count": 6,
        "normal_cleanup_wait_object_0_count": 6,
        "normal_cleanup_all_wait_object_0": True,
        "profile_module_discovery": True,
        "default_module_discovery": True,
        "profile_extended_module_discovery": True,
        "unicode_roundtrip": True,
        "alternate_screen": True,
        "alternate_isolated": True,
        "primary_restored": True,
        "app_crash_observed": True,
        "crash_app_process_separate": True,
        "crash_app_sole_job_handle_owner": True,
        "crash_job_handle_non_inheritable": True,
        "crash_worker_admitted_before_conpty": True,
        "crash_descendant_set_stable": True,
        "crash_descendants_ready_before_abort": True,
        "crash_supervisor_job_handle_count": 0,
        "crash_known_descendant_count": 3,
        "crash_supervisor_synchronize_handle_count": 3,
        "crash_wait_object_0_count": 3,
        "crash_all_descendants_wait_object_0": True,
        "terminal_child_crash_observed": True,
        "terminal_child_eof_observed": True,
        "terminal_child_member_before_crash": True,
        "terminal_grandchild_member_before_crash": True,
        "eof_observed": True,
        "output_integrity": True,
        "captured_byte_count": 160000,
        "sequence_complete": True,
        "digest_equal": True,
        "post_exit_drain_bounded": True,
        "missing_eof_bounded": False,
        "four_session_count": 4,
        "rss_measurement_complete": True,
        "four_session_rss_delta_bytes": 256 * 1024 * 1024,
        "rss_controller_process_count": 1,
        "rss_worker_process_count": 1,
        "rss_helper_process_count": 2,
        "rss_fixture_process_count": 4,
        "rss_fixture_processes_excluded": True,
        "rss_ipc_included_in_worker": True,
        "rss_sample_live_session_count": 4,
        "rss_crash_session_present": False,
    }


def test_windows_native_result_schema_can_represent_a_complete_pass() -> None:
    module = _load_qualification_module("pywinpty_probe")
    builder = getattr(module, "_build_native_rows", None)
    assert callable(builder), "native probe must build rows from measured observations"
    observations = _complete_windows_observations()

    rows = builder(observations)

    assert {row["id"] for row in rows} == set(module.WINDOWS_ROWS)
    assert {row["status"] for row in rows} == {"PASS"}
    assert all(row["mandatory"] is True for row in rows)


def test_common_schema_accepts_complete_native_windows_rows() -> None:
    winpty_module = _load_qualification_module("pywinpty_probe")
    common_module = _load_qualification_module("common")
    payload = _minimal_raw_payload("win-amd64-py311", "pywinpty")
    payload["status"] = "PASS"
    payload.pop("reason_category", None)
    payload["rows"] = winpty_module._build_native_rows(_complete_windows_observations())

    common_module.validate_content_free(payload)


def test_terminal_child_crash_facts_cannot_satisfy_app_crash_row() -> None:
    module = _load_qualification_module("pywinpty_probe")
    observations = _complete_windows_observations()
    observations.update(
        {
            "app_crash_observed": False,
            "crash_app_process_separate": False,
            "crash_app_sole_job_handle_owner": False,
            "crash_descendants_ready_before_abort": False,
            "crash_all_descendants_wait_object_0": False,
        }
    )
    assert observations["terminal_child_crash_observed"] is True

    rows = {row["id"]: row for row in module._build_native_rows(observations)}

    assert rows["windows-app-crash-descendant-cleanup"]["status"] == "FAIL"


def test_linux_arm64_python312_row_is_genuine_container_evidence() -> None:
    row_root = RAW_ROOT / LINUX_ROW_ID
    raw_files = sorted(row_root.glob("*.json"))
    assert raw_files, "real Linux ARM64 qualification row has not been collected"
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in raw_files]
    assert {payload["probe"] for payload in payloads} >= {
        "artifacts",
        "environment-default",
        "environment-bash",
        "environment-zsh",
        "pyte",
        "pywinpty",
    }
    for payload in payloads:
        assert payload["platform"]["os"] == "Linux"
        assert payload["platform"]["architecture"] in {"aarch64", "arm64"}
        runtime = payload.get("runtime")
        assert isinstance(runtime, dict)
        assert runtime["kind"] == "docker"
        assert runtime["image"] == "ubuntu:24.04"
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", runtime["image_id"])
        assert re.fullmatch(r"[0-9a-f]{12,64}", runtime["container_id"])
    statuses = {payload["probe"]: payload["status"] for payload in payloads}
    assert statuses["artifacts"] == "PASS"
    assert statuses["environment-default"] == "PASS"
    assert statuses["environment-bash"] == "PASS"
    assert statuses["environment-zsh"] == "PASS"
    assert statuses["pyte"] == "PASS"
    assert statuses["pywinpty"] == "UNSUPPORTED_FAIL_CLOSED"
    evidence = EVIDENCE.read_text(encoding="utf-8")
    assert "ubuntu:24.04" in evidence
    assert LINUX_ROW_ID in evidence
    assert "docker run" in evidence


def test_evidence_names_exact_linux_runtime_and_native_windows_blocker() -> None:
    evidence = EVIDENCE.read_text(encoding="utf-8")
    linux_artifacts = json.loads(
        (RAW_ROOT / LINUX_ROW_ID / f"{LINUX_ROW_ID}-artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    runtime = linux_artifacts["runtime"]

    assert runtime["image"] in evidence
    assert runtime["image_id"] in evidence
    assert runtime["container_id"] in evidence
    assert "Exact machine-recorded argv" in evidence
    assert "Reproduction templates" in evidence
    assert (
        "Status: POSIX delivery qualified; the pinned native Windows boundary "
        "remains" in evidence
    )
    for row_id in MANDATORY_ROWS:
        if row_id.startswith("windows-") or row_id in {
            "package-pywinpty-3.0.5",
            "four-session-managed-rss",
        }:
            expected = "FAIL_CLOSED" if row_id in WINDOWS_FAIL_CLOSED_ROWS else "PASS"
            assert _row_status(evidence, row_id) == expected

    native = json.loads(
        (RAW_ROOT / WINDOWS_ROW_ID / f"{WINDOWS_ROW_ID}-pywinpty.json").read_text(
            encoding="utf-8"
        )
    )
    assert native["status"] == "FAIL"
    assert {
        row["id"] for row in native["rows"] if row["status"] == "FAIL"
    } == WINDOWS_FAIL_CLOSED_ROWS


def test_retained_macos_statuses_match_reported_pass_slices() -> None:
    evidence = EVIDENCE.read_text(encoding="utf-8")
    expected = {
        "artifacts": "PASS",
        "environment-default": "PASS",
        "environment-bash": "PASS",
        "environment-zsh": "PASS",
        "pyte": "PASS",
        "pywinpty": "UNSUPPORTED_FAIL_CLOSED",
    }

    for row_id in MACOS_ROW_IDS:
        payloads = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((RAW_ROOT / row_id).glob("*.json"))
        ]
        assert {payload["probe"]: payload["status"] for payload in payloads} == expected
        assert (
            f"| {row_id} | PASS | PASS | default/Bash/Zsh PASS | "
            "UNSUPPORTED_FAIL_CLOSED |"
        ) in evidence


def test_readme_records_exact_linux_reproduction_privacy_and_ratchet_commands() -> None:
    readme = README.read_text(encoding="utf-8")
    compact = " ".join(readme.split())

    assert "docker run --platform linux/arm64" in compact
    assert "ubuntu:24.04" in readme
    assert (
        "sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea"
        in readme
    )
    assert "--runtime-kind docker" in compact
    assert "--runtime-image-id" in compact
    assert "--runtime-container-id" in compact
    assert (
        "format_ratchet.py verify --head HEAD --baseline "
        "Docs/superpowers/reviews/evidence/task-22512/format-baseline.json"
    ) in compact
    assert "shape-specific allowlists" in readme
    assert "wrong type or placement" in readme
    assert "environment values" in readme
    assert "terminal output" in readme
    assert "profile content" in readme
    assert "secrets" in readme


def test_common_creates_venv_from_selected_interpreter_base_prefix(
    tmp_path: Path,
) -> None:
    python311 = shutil.which("python3.11")
    if python311 is None:
        return
    venv_dir = tmp_path / "venv"
    source = (
        "import pathlib,sys; "
        f"sys.path.insert(0, {str(QUALIFICATION_ROOT)!r}); "
        "import common; "
        "common.create_isolated_venv(pathlib.Path(sys.argv[1]))"
    )

    completed = subprocess.run(
        [python311, "-c", source, str(venv_dir)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    venv_python = venv_dir / "bin" / "python"
    verified = subprocess.run(
        [str(venv_python), "-c", "import sys; assert sys.prefix != sys.base_prefix"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verified.returncode == 0, verified.stderr


def test_windows_output_credit_uses_pty_read_and_requires_explicit_ack() -> None:
    module = _load_qualification_module("pywinpty_probe")

    class FakeTerminal:
        calls: list[bool] = []

        @property
        def fd(self) -> int:
            raise AssertionError("PTY.fd is a process handle, not output")

        def read(self, blocking: bool = False) -> str:
            self.calls.append(blocking)
            return "界"

    terminal = FakeTerminal()
    credit = module._OutputCredit()

    chunk = credit.read(terminal)

    assert chunk.data == "界".encode()
    assert terminal.calls == [True]
    with pytest.raises(module.QualificationError, match="already outstanding"):
        credit.read(terminal)
    credit.acknowledge(chunk)
    assert credit.outstanding == 0
    assert credit.max_chunk_bytes == len("界".encode())
    assert credit.upstream_read_buffer_bytes == 32 * 1024


def test_concurrent_operations_handoff_only_one_blocking_read_after_quiet_drain() -> (
    None
):
    module = _load_qualification_module("pywinpty_probe")
    read_released = threading.Event()
    read_entered = threading.Event()
    trace: list[str] = []

    class WriteTerminal:
        def write(self, value: str) -> int:
            assert value == "W"
            trace.append("write-returned")
            return 1

    class CancelTerminal:
        def cancel_io(self) -> bool:
            trace.append("cancel-returned")
            return True

    class ResizeTerminal:
        def set_size(self, columns: int, rows: int) -> None:
            assert (columns, rows) == (100, 30)
            trace.append("resize-returned")

    class QuietReadTerminal:
        def __init__(self) -> None:
            self.startup = ["startup", "", ""]

        def read(self, blocking: bool = False) -> str:
            if not blocking:
                trace.append("startup-read")
                return self.startup.pop(0) if self.startup else ""
            trace.append("blocking-read-entered")
            read_entered.set()
            assert read_released.wait(timeout=2.0)
            trace.append("blocking-read-returned")
            return ""

        def cancel_io(self) -> bool:
            trace.append("priority-cancel")
            read_released.set()
            return True

    close_observation: dict[str, object] = {}

    def close_action() -> None:
        close_observation.update(
            {
                "read_entered": read_entered.is_set(),
                "sync_returns": {
                    name
                    for name in trace
                    if name in {"write-returned", "resize-returned", "cancel-returned"}
                },
                "startup_reads_before_block": trace.index("blocking-read-entered"),
            }
        )
        trace.append("priority-close")

    result = module._concurrent_operations(
        [WriteTerminal(), CancelTerminal(), ResizeTerminal(), QuietReadTerminal()],
        module._OutputCredit(),
        close_action=close_action,
        timeout=2.0,
        startup_quiet_seconds=0.01,
    )

    assert close_observation["read_entered"] is True
    assert close_observation["sync_returns"] == {
        "write-returned",
        "resize-returned",
        "cancel-returned",
    }
    assert close_observation["startup_reads_before_block"] >= 2
    assert result["quiet_terminal_startup_drained"] is True
    assert result["quiet_terminal_quiescent_before_handoff"] is True
    assert result["concurrent_operation_count"] == 4
    assert result["inflight_operation_category_count"] == 1
    for name in ("read", "write", "resize", "cancel"):
        assert result[f"{name}_entered"] is True
    assert result["read_completed_at_handoff"] is False
    assert result["read_completed_post_close"] is True
    for name in ("write", "resize", "cancel"):
        assert result[f"{name}_completed_at_handoff"] is True
        assert result[f"{name}_completed_post_close"] is False
    assert result["io_inflight_at_handoff"] is True
    assert result["priority_close_preempted_inflight"] is True
    assert trace.index("startup-read") < trace.index("blocking-read-entered")
    assert trace.index("priority-cancel") < trace.index("priority-close")
    assert trace.index("priority-close") < trace.index("blocking-read-returned")


def test_concurrent_operations_reject_nonquiet_startup_before_handoff() -> None:
    module = _load_qualification_module("pywinpty_probe")
    operation_entered = threading.Event()

    class ForbiddenTerminal:
        def write(self, value: str) -> int:
            operation_entered.set()
            return len(value)

        def set_size(self, columns: int, rows: int) -> None:
            operation_entered.set()

        def cancel_io(self) -> bool:
            operation_entered.set()
            return True

    class NoisyTerminal:
        def read(self, blocking: bool = False) -> str:
            assert not blocking
            return "still-starting"

    with pytest.raises(module.QualificationError, match="quiet"):
        module._concurrent_operations(
            [
                ForbiddenTerminal(),
                ForbiddenTerminal(),
                ForbiddenTerminal(),
                NoisyTerminal(),
            ],
            module._OutputCredit(),
            close_action=lambda: None,
            timeout=0.05,
            startup_quiet_seconds=0.01,
        )
    assert operation_entered.is_set() is False


def test_post_exit_drain_records_multibuffer_integrity_and_eof_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    frames, expected = module._sequenced_output(frame_count=4, payload_bytes=40_000)
    captured = b"\x1b[?25l" + expected + b"\x1b[?25h"
    chunks = [
        captured[index : index + 16_000] for index in range(0, len(captured), 16_000)
    ]
    waits: list[tuple[int, float]] = []

    class FakeTerminal:
        fd = 17

        def __init__(self) -> None:
            self.chunks = [chunk.decode() for chunk in chunks]

        def read(self, blocking: bool = False) -> str:
            assert blocking
            if self.chunks:
                return self.chunks.pop(0)
            return ""

        def cancel_io(self) -> bool:
            return True

    monkeypatch.setattr(
        module,
        "_wait_process_handle",
        lambda handle, timeout: waits.append((handle, timeout)) or True,
    )
    facts = module._drain_after_exit(
        FakeTerminal(),
        module._OutputCredit(),
        frames=frames,
        expected=expected,
        deadline_seconds=2.0,
        post_exit_seconds=0.2,
    )

    assert waits == [(FakeTerminal.fd, 2.0)]
    assert facts["captured_byte_count"] == len(captured)
    assert facts["sequence_complete"] is True
    assert facts["digest_equal"] is True
    assert facts["eof_observed"] is True
    assert facts["missing_eof_bounded"] is False
    assert facts["post_exit_drain_bounded"] is True


def test_sequenced_frames_reject_non_vt_bytes_outside_expected_payload() -> None:
    module = _load_qualification_module("pywinpty_probe")
    frames, expected = module._sequenced_output(frame_count=3, payload_bytes=40_000)

    corrupt_streams = (
        b"corrupt-before" + expected,
        frames[0] + b"corrupt-between" + b"".join(frames[1:]),
        expected + b"corrupt-after",
        expected + frames[-1],
        b"\x1b[?25" + expected,
    )

    for captured in corrupt_streams:
        assert module._extract_sequenced_frames(captured, frames) is None

    framed = b"\x1b[?25l" + expected + b"\x1b[?25h"
    assert module._extract_sequenced_frames(framed, frames) == expected


def test_windows_fixture_waits_for_complete_alternate_output_and_conpty_eof() -> None:
    module = _load_qualification_module("pywinpty_probe")
    source = module._fixture_source().encode()
    completion_pattern = getattr(module, "ALTERNATE_COMPLETE_RE", re.compile(b"never"))

    assert completion_pattern.search(source)
    assert b"ALT" not in completion_pattern.pattern
    alternate_payload = getattr(module, "ALTERNATE_PAYLOAD", "").encode()
    assert alternate_payload and alternate_payload in source
    alternate_source = inspect.getsource(module._alternate_facts)
    assert "ALTERNATE_PAYLOAD in alternate" in alternate_source
    assert "ALTERNATE_PAYLOAD not in primary" in alternate_source
    assert getattr(module, "CONPTY_POST_EXIT_DRAIN_SECONDS", 0) > 5.0
    assert (
        module.INTEGRITY_FRAME_COUNT * module.INTEGRITY_PAYLOAD_BYTES
        < module.CONPTY_PIPE_BUFFER_BYTES
    )
    settle_seconds = getattr(module, "INTEGRITY_SETTLE_SECONDS", 0)
    assert 0 < settle_seconds < 1.0
    assert f"time.sleep({settle_seconds!r})" in module._fixture_source()


def test_windows_empty_blocking_read_after_process_exit_is_eof() -> None:
    module = _load_qualification_module("pywinpty_probe")

    class FakeTerminal:
        def read(self, blocking: bool = False) -> str:
            assert blocking
            return ""

        def cancel_io(self) -> bool:
            return True

    assert (
        module._wait_terminal_eof(
            FakeTerminal(),
            module._OutputCredit(),
            timeout=0.1,
        )
        is True
    )


def test_windows_terminal_crash_completes_the_overlapped_console_line_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    writes: list[str] = []
    reads: list[bool] = []
    waits: list[tuple[int, float]] = []

    fake_winpty_error = type("WinptyError", (Exception,), {"__module__": "pywinpty"})

    assert "sys.stdin.buffer.readline()" in module._fixture_source()

    class FakeTerminal:
        fd = 123

        def write(self, value: str) -> int:
            writes.append(value)
            return len(value)

        def read(self, blocking: bool = False) -> str:
            assert blocking is True
            reads.append(blocking)
            raise fake_winpty_error("Standard out reached EOF")

    trigger = getattr(module, "_request_terminal_crash", None)
    assert callable(trigger)
    monkeypatch.setattr(
        module,
        "_wait_process_handle",
        lambda handle, timeout: waits.append((handle, timeout)) or True,
        raising=False,
    )
    facts = trigger(
        FakeTerminal(),
        module._OutputCredit(),
        timeout=0.1,
    )

    assert writes == ["!\r\n", "x"]
    assert waits == [(FakeTerminal.fd, 0.1)]
    assert reads == [True]
    assert facts == {
        "terminal_child_crash_observed": True,
        "terminal_child_eof_observed": True,
    }


def test_windows_grandchild_membership_is_sampled_before_crash_trigger() -> None:
    module = _load_qualification_module("pywinpty_probe")
    source = inspect.getsource(module._worker_observations)

    assert source.index('observations["terminal_grandchild_member_before_crash"]') < (
        source.index("_request_terminal_crash")
    )


def test_windows_native_worker_retains_pty_owners_until_job_cleanup(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    owner = object()
    observed: dict[str, object] = {}

    monkeypatch.setattr(module, "_wait_named_event", lambda *args: True)

    def worker_observations(**kwargs) -> dict[str, object]:
        retained = kwargs["retained_terminals"]
        retained.append(owner)
        observed["retained"] = retained
        return {"fresh_worker": True}

    monkeypatch.setattr(module, "_worker_observations", worker_observations)
    monkeypatch.setattr(module, "_write_worker_result", lambda *args: None)
    monkeypatch.setattr(module, "_signal_named_event", lambda *args: None)

    class StopWorkerLoop(Exception):
        pass

    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(StopWorkerLoop()),
    )

    with pytest.raises(StopWorkerLoop):
        module._native_worker(
            tmp_path / "worker.json",
            "start",
            "ready",
            tmp_path / "rss.json",
            "rss-ready",
            "rss-continue",
        )

    assert observed["retained"] == [owner]


def test_windows_native_worker_fails_closed_and_signals_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    published: dict[str, object] = {}
    signals: list[str] = []

    monkeypatch.setattr(module, "_wait_named_event", lambda *args: True)
    monkeypatch.setattr(
        module,
        "_worker_observations",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )
    monkeypatch.setattr(module, "_fd_is_backed", lambda descriptor: True)
    monkeypatch.setattr(
        module,
        "_write_worker_result",
        lambda path, observations: published.update(observations),
    )
    monkeypatch.setattr(
        module, "_signal_named_event", lambda name: signals.append(name)
    )

    class StopWorkerLoop(Exception):
        pass

    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(StopWorkerLoop()),
    )

    with pytest.raises(StopWorkerLoop):
        module._native_worker(
            tmp_path / "worker.json",
            "start",
            "ready",
            tmp_path / "rss.json",
            "rss-ready",
            "rss-continue",
        )

    assert published["fresh_worker"] is True
    assert published["worker_std_streams_fd_backed"] is True
    assert published["low_level_api"] is False
    assert signals == ["ready"]


def test_windows_owned_process_cleanup_uses_signal_ctrl_break_event(
    monkeypatch,
) -> None:
    common = _load_qualification_module("common")
    sent: list[int] = []

    class FakeProcess:
        pid = 41
        args = ["cmd.exe"]

        def poll(self) -> None:
            return None

        def send_signal(self, requested_signal: int) -> None:
            sent.append(requested_signal)

        def wait(self, timeout: float | None = None) -> int:
            return 0

    class FakeJob:
        def close(self) -> None:
            raise AssertionError("graceful console break must not close the Job")

    monkeypatch.setattr(common.os, "name", "nt")
    monkeypatch.setattr(common.subprocess, "CTRL_BREAK_EVENT", 81, raising=False)
    monkeypatch.setattr(common.signal, "CTRL_BREAK_EVENT", 82, raising=False)

    assert common.terminate_owned_group(
        FakeProcess(), FakeJob(), grace_seconds=0.1
    ) == (True, False)
    assert sent == [82]


def test_alternate_user_process_recognizes_signal_ctrl_break_event(
    monkeypatch,
) -> None:
    common = _load_qualification_module("common")
    generated: list[tuple[int, int]] = []

    class FakeKernel32:
        def GenerateConsoleCtrlEvent(self, requested_signal: int, pid: int) -> bool:
            generated.append((requested_signal, pid))
            return True

    process = object.__new__(common._WindowsHandleProcess)
    process._kernel32 = FakeKernel32()
    process.pid = 43
    monkeypatch.setattr(common.subprocess, "CTRL_BREAK_EVENT", 81, raising=False)
    monkeypatch.setattr(common.signal, "CTRL_BREAK_EVENT", 82, raising=False)
    monkeypatch.setattr(
        process,
        "terminate",
        lambda: (_ for _ in ()).throw(AssertionError("must send console break")),
    )

    process.send_signal(82)

    assert generated == [(82, 43)]


def test_managed_rss_population_excludes_exactly_four_fixture_processes() -> None:
    module = _load_qualification_module("pywinpty_probe")

    managed, facts = module._managed_rss_population(
        controller_pid=10,
        worker_pid=20,
        job_member_ids=[20, 31, 32, 41, 42, 43, 44],
        fixture_process_ids=[41, 42, 43, 44],
    )

    assert managed == [10, 20, 31, 32]
    assert facts == {
        "rss_controller_process_count": 1,
        "rss_worker_process_count": 1,
        "rss_helper_process_count": 2,
        "rss_fixture_process_count": 4,
        "rss_fixture_processes_excluded": True,
        "rss_ipc_included_in_worker": True,
        "rss_sample_live_session_count": 4,
        "rss_crash_session_present": False,
    }


@pytest.mark.parametrize(
    ("wait_results", "expected"),
    (
        ([0, 0], (True, 2)),
        ([0, 258], (False, 1)),
        ([0, 0xFFFFFFFF], (False, 1)),
        ([0, 17], (False, 1)),
    ),
)
def test_retained_process_handles_require_wait_object_0_for_every_descendant(
    wait_results: list[int], expected: tuple[bool, int]
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    calls: list[tuple[int, int]] = []
    remaining_results = iter(wait_results)

    class FakeKernel32:
        def WaitForSingleObject(self, handle: int, timeout_ms: int) -> int:
            calls.append((handle, timeout_ms))
            return next(remaining_results)

        def CloseHandle(self, handle: int) -> bool:
            calls.append((handle, -1))
            return True

    assert (
        module._wait_retained_process_handles(
            FakeKernel32(), [101, 102], timeout_seconds=0.2
        )
        == expected
    )
    assert (101, -1) in calls and (102, -1) in calls


def _passing_normal_cleanup_facts() -> dict[str, object]:
    return {
        "normal_cleanup_expected_process_count": 2,
        "normal_cleanup_retained_handle_count": 2,
        "normal_cleanup_wait_object_0_count": 2,
        "normal_cleanup_all_wait_object_0": True,
    }


@pytest.mark.parametrize(
    "cleanup_facts",
    (
        {
            "normal_cleanup_expected_process_count": 2,
            "normal_cleanup_retained_handle_count": 2,
            "normal_cleanup_wait_object_0_count": 2,
            "normal_cleanup_all_wait_object_0": False,
        },
        {
            "normal_cleanup_expected_process_count": 2,
            "normal_cleanup_retained_handle_count": 2,
            "normal_cleanup_wait_object_0_count": 1,
            "normal_cleanup_all_wait_object_0": True,
        },
    ),
)
def test_native_observations_are_not_published_for_false_or_partial_cleanup(
    cleanup_facts: dict[str, object],
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    published = module._default_observations()
    before = dict(published)
    candidate = dict(published)
    candidate["priority_close_completed"] = True

    with pytest.raises(module.QualificationError, match="normal cleanup"):
        module._commit_native_observations(
            published,
            candidate,
            cleanup_action=lambda: cleanup_facts,
        )

    assert published == before


def test_native_observations_are_not_published_when_cleanup_raises() -> None:
    module = _load_qualification_module("pywinpty_probe")
    published = module._default_observations()
    before = dict(published)
    candidate = dict(published)
    candidate["priority_close_completed"] = True

    def cleanup_action() -> dict[str, object]:
        raise OSError("wait failed")

    with pytest.raises(OSError, match="wait failed"):
        module._commit_native_observations(
            published,
            candidate,
            cleanup_action=cleanup_action,
        )

    assert published == before


def test_normal_cleanup_retains_before_close_and_reaps_exact_wait_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    events: list[object] = []

    class FakeJob:
        handle = 1
        _kernel32 = object()

        def process_ids(self) -> list[int]:
            return [40, 41]

        def retain_process_handles(self, process_ids) -> list[int]:
            events.append(("retained", tuple(process_ids)))
            return [101, 102]

        def close(self) -> None:
            events.append("job-closed")
            self.handle = None

    class FakeProcess:
        def wait(self, timeout: float) -> int:
            events.append(("reaped", timeout))
            return 1

    monkeypatch.setattr(
        module,
        "_wait_retained_process_handles",
        lambda kernel32, handles, timeout_seconds: (
            events.append(("waited", tuple(handles), timeout_seconds)) or (True, 2)
        ),
    )

    facts = module._normal_cleanup_facts(FakeJob(), FakeProcess(), timeout_seconds=0.2)

    assert facts == _passing_normal_cleanup_facts()
    assert events == [
        ("retained", (40, 41)),
        "job-closed",
        ("waited", (101, 102), 0.2),
        ("reaped", 0.2),
    ]


@pytest.mark.parametrize("wait_result", ((False, 2), (True, 1)))
def test_normal_cleanup_rejects_false_or_partial_waits(
    monkeypatch: pytest.MonkeyPatch,
    wait_result: tuple[bool, int],
) -> None:
    module = _load_qualification_module("pywinpty_probe")

    class FakeJob:
        handle = 1
        _kernel32 = object()

        def process_ids(self) -> list[int]:
            return [40, 41]

        def retain_process_handles(self, process_ids) -> list[int]:
            return [101, 102]

        def close(self) -> None:
            self.handle = None

    class FakeProcess:
        def wait(self, timeout: float) -> int:
            return 1

    monkeypatch.setattr(
        module,
        "_wait_retained_process_handles",
        lambda kernel32, handles, timeout_seconds: wait_result,
    )

    with pytest.raises(module.QualificationError, match="normal cleanup"):
        module._normal_cleanup_facts(FakeJob(), FakeProcess(), timeout_seconds=0.2)


def test_normal_cleanup_partial_retention_closes_every_opened_handle() -> None:
    module = _load_qualification_module("pywinpty_probe")
    closed: list[int] = []

    class FakeKernel32:
        def CloseHandle(self, handle: int) -> bool:
            closed.append(handle)
            return handle != 101

    class FakeJob:
        handle = 1
        _kernel32 = FakeKernel32()

        def process_ids(self) -> list[int]:
            return [40, 41, 42]

        def retain_process_handles(self, process_ids) -> list[int]:
            return [101, 102]

        def close(self) -> None:
            self.handle = None

    class FakeProcess:
        def wait(self, timeout: float) -> int:
            return 1

    with pytest.raises(module.QualificationError, match="retained handle"):
        module._normal_cleanup_facts(FakeJob(), FakeProcess(), timeout_seconds=0.2)

    assert closed == [101, 102]


@pytest.mark.parametrize("failure_stage", ("retain", "wait", "reap"))
def test_normal_cleanup_propagates_retention_wait_and_reap_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    module = _load_qualification_module("pywinpty_probe")

    class FakeJob:
        handle = 1
        _kernel32 = object()

        def process_ids(self) -> list[int]:
            return [40, 41]

        def retain_process_handles(self, process_ids) -> list[int]:
            if failure_stage == "retain":
                raise OSError("retain failed")
            return [101, 102]

        def close(self) -> None:
            self.handle = None

    class FakeProcess:
        def wait(self, timeout: float) -> int:
            if failure_stage == "reap":
                raise subprocess.TimeoutExpired("worker", timeout)
            return 1

    def wait_handles(kernel32, handles, timeout_seconds):
        if failure_stage == "wait":
            raise OSError("wait failed")
        return True, 2

    monkeypatch.setattr(module, "_wait_retained_process_handles", wait_handles)

    with pytest.raises((OSError, subprocess.TimeoutExpired)):
        module._normal_cleanup_facts(FakeJob(), FakeProcess(), timeout_seconds=0.2)


def test_windows_shell_result_marker_allows_default_cmd_prompt_only() -> None:
    module = _load_qualification_module("environment_probe")
    nonce = "a" * 32
    marker = module._result_marker(nonce)
    valid = b"noise\r\n" + marker.encode() + b"0,0,0,0\r\n"

    match = module._single_result_match(valid, nonce, windows=True)

    assert match is not None and match.groups() == (b"0", b"0", b"0", b"0")
    prompted = b"noise\r\nC:\\Users\\fixture>" + marker.encode() + b"0,0,0,0\r\n"
    prompt_match = module._single_result_match(prompted, nonce, windows=True)
    assert prompt_match is not None and prompt_match.groups() == (
        b"0",
        b"0",
        b"0",
        b"0",
    )
    assert (
        module._single_result_match(
            valid + marker.encode() + b"0,0,0,0\r\n", nonce, windows=True
        )
        is None
    )
    assert (
        module._single_result_match(
            b"prefix" + marker.encode() + b"0,0,0,0\r\n", nonce, windows=True
        )
        is None
    )


def test_posix_shell_result_marker_accepts_cr_line_boundary_only_once() -> None:
    module = _load_qualification_module("environment_probe")
    nonce = "b" * 32
    marker = module._result_marker(nonce).encode()
    echoed_command = b"printf '" + marker + b"%s,%s,%s'"
    valid = echoed_command + b"\r" + marker + b"0,1,1\r\n"

    match = module._single_result_match(valid, nonce, windows=False)

    assert match is not None and match.groups() == (b"0", b"1", b"1")
    assert (
        module._single_result_match(
            valid + b"\r" + marker + b"0,1,1\r\n",
            nonce,
            windows=False,
        )
        is None
    )


def test_posix_shell_waits_for_interactive_startup_before_command(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX PTY sequencing test")
    module = _load_qualification_module("environment_probe")
    shell = tmp_path / "bash"
    shell.write_text(
        f"#!{sys.executable}\n"
        "import os,re,termios,time,tty\n"
        "os.write(1, b'early\\r\\n')\n"
        "time.sleep(0.05)\n"
        "tty.setraw(0)\n"
        "termios.tcflush(0, termios.TCIFLUSH)\n"
        "os.write(1, b'ready\\r\\n')\n"
        "value=os.read(0, 8192)\n"
        "match=re.search(rb'__TLDW_TASK22512_ENV_[0-9a-f]{32}__', value)\n"
        "os.write(1, match.group(0) + b'0,0,0\\r\\n')\n",
        encoding="utf-8",
    )
    shell.chmod(0o755)

    result = module._run_posix_shell(
        shell,
        {
            "HOME": str(tmp_path),
            "LOGNAME": "fixture",
            "PATH": os.defpath,
            "SHELL": str(shell),
            "TERM": "linux",
            "TMPDIR": str(tmp_path),
            "USER": "fixture",
        },
    )

    assert result["startup_completed"] is True
    assert result["command_discovery"] is True


def test_windows_startup_uses_normal_profile_and_cmd_autorun_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_qualification_module("environment_probe")
    observed: list[tuple[list[str], object]] = []

    def fake_run(argv, **kwargs):
        observed.append((list(argv), kwargs["bootstrap_setup"]))
        nonce = kwargs["result_nonce"]
        marker = module._result_marker(nonce).encode()
        groups = b"0,0,0,0"
        return subprocess.CompletedProcess(
            argv, 0, stdout=marker + groups + b"\r\n", stderr=b""
        )

    monkeypatch.setattr(module, "_run_shell_process", fake_run)

    powershell = module._run_windows_shell(
        "powershell",
        Path("powershell.exe"),
        {"PATH": str(tmp_path), "TERM": "linux"},
    )
    cmd = module._run_windows_shell(
        "cmd", Path("cmd.exe"), {"PATH": str(tmp_path), "TERM": "linux"}
    )

    assert "-NoProfile" not in observed[0][0]
    assert "-NonInteractive" not in observed[0][0]
    assert "/D" not in observed[1][0]
    assert observed[0][1].profile_files
    assert observed[0][1].registry_values == ()
    assert observed[1][1].profile_files == ()
    assert observed[1][1].registry_values
    assert powershell["default_module_discovery"] is True
    assert powershell["profile_extended_module_discovery"] is True
    assert cmd["profile_marker_present"] is True


def test_windows_registry_fixtures_require_verified_disposable_user_context() -> None:
    common = _load_qualification_module("common")
    isolated_root = object()
    interactive_root = object()
    events: list[tuple[object, ...]] = []

    class FakeRegistry:
        def verify_disposable_user(self, username: str) -> bool:
            events.append(("verify", username))
            return username == "tldw-isolated"

        def open_current_user(self) -> object:
            events.append(("open-current-user",))
            return isolated_root

        def set_string(self, root: object, subkey: str, name: str, value: str) -> None:
            assert root is not interactive_root
            events.append(("set", root, subkey, name, value))

        def close_key(self, root: object) -> None:
            events.append(("close", root))

    values = ((r"Software\Microsoft\Command Processor", "AutoRun", "fixture-command"),)

    common._install_disposable_registry_values(
        FakeRegistry(), username="tldw-isolated", values=values
    )

    assert events == [
        ("verify", "tldw-isolated"),
        ("open-current-user",),
        (
            "set",
            isolated_root,
            r"Software\Microsoft\Command Processor",
            "AutoRun",
            "fixture-command",
        ),
        ("close", isolated_root),
    ]

    with pytest.raises(common.QualificationError, match="disposable user"):
        common._install_disposable_registry_values(
            FakeRegistry(), username="interactive-user", values=values
        )


def test_disposable_windows_profile_cleanup_runs_after_probe_crash() -> None:
    common = _load_qualification_module("common")
    events: list[tuple[str, str]] = []

    class FakeProfileApi:
        def create_account(self, username: str, password: str) -> None:
            assert password
            events.append(("create-account", username))

        def create_profile(self, username: str):
            events.append(("create-profile", username))
            return common.DisposableProfileIdentity(
                username=username,
                sid="S-1-5-21-1000",
                profile_path=Path(r"C:\Users\tldw-isolated"),
            )

        def delete_profile(self, identity: object) -> None:
            events.append(("delete-profile", identity.username))

        def delete_account(self, username: str) -> None:
            events.append(("delete-account", username))

    with pytest.raises(RuntimeError, match="probe crash"):
        with common._DisposableWindowsProfile(
            FakeProfileApi(),
            username="tldw-isolated",
            password="ephemeral-password",
        ) as profile:
            assert profile.identity.username == "tldw-isolated"
            raise RuntimeError("probe crash")

    assert events == [
        ("create-account", "tldw-isolated"),
        ("create-profile", "tldw-isolated"),
        ("delete-profile", "tldw-isolated"),
        ("delete-account", "tldw-isolated"),
    ]


def test_native_windows_profile_delete_retries_sharing_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = _load_qualification_module("common")
    attempts = 0
    sleeps: list[float] = []

    class FakeUserEnv:
        def DeleteProfileW(self, sid: str, path: str, machine: object) -> bool:
            nonlocal attempts
            assert sid == "S-1-5-21-1000"
            assert path == r"C:\Users\tldw-isolated"
            assert machine is None
            attempts += 1
            return attempts == 2

    api = object.__new__(common._NativeWindowsProfileApi)
    api.ctypes = SimpleNamespace(
        set_last_error=lambda error: None,
        get_last_error=lambda: 32,
    )
    api.userenv = FakeUserEnv()
    monkeypatch.setattr(common.time, "sleep", sleeps.append)

    api.delete_profile(
        common.DisposableProfileIdentity(
            username="tldw-isolated",
            sid="S-1-5-21-1000",
            profile_path=Path(r"C:\Users\tldw-isolated"),
        )
    )

    assert attempts == 2
    assert sleeps == [0.1]


@pytest.mark.parametrize("account_delete_fails", (False, True), ids=("one", "both"))
def test_disposable_windows_profile_preserves_body_and_cleanup_failures(
    account_delete_fails: bool,
) -> None:
    common = _load_qualification_module("common")
    events: list[tuple[str, str]] = []

    class FailingCleanupProfileApi:
        def create_account(self, username: str, password: str) -> None:
            assert password
            events.append(("create-account", username))

        def create_profile(self, username: str):
            events.append(("create-profile", username))
            return common.DisposableProfileIdentity(
                username=username,
                sid="S-1-5-21-1000",
                profile_path=Path(r"C:\Users\tldw-isolated"),
            )

        def delete_profile(self, identity: object) -> None:
            events.append(("delete-profile", identity.username))
            raise OSError("profile deletion failed")

        def delete_account(self, username: str) -> None:
            events.append(("delete-account", username))
            if account_delete_fails:
                raise OSError("account deletion failed")

    with pytest.raises(ExceptionGroup) as caught:
        with common._DisposableWindowsProfile(
            FailingCleanupProfileApi(),
            username="tldw-isolated",
            password="ephemeral-password",
        ):
            raise RuntimeError("probe body failed")

    diagnostics = "".join(traceback.format_exception(caught.value))
    assert "probe body failed" in diagnostics
    assert "profile deletion failed" in diagnostics
    assert ("account deletion failed" in diagnostics) is account_delete_fails
    assert events == [
        ("create-account", "tldw-isolated"),
        ("create-profile", "tldw-isolated"),
        ("delete-profile", "tldw-isolated"),
        ("delete-account", "tldw-isolated"),
    ]


def test_disposable_windows_profile_precondition_fails_closed_before_launch() -> None:
    common = _load_qualification_module("common")
    events: list[tuple[str, str]] = []

    class UnprivilegedProfileApi:
        def create_account(self, username: str, password: str) -> None:
            assert password
            events.append(("create-account", username))

        def create_profile(self, username: str):
            events.append(("create-profile", username))
            raise PermissionError("account/profile privilege unavailable")

        def delete_account(self, username: str) -> None:
            events.append(("delete-account", username))

    with pytest.raises(common.QualificationError, match="precondition unavailable"):
        with common._DisposableWindowsProfile(
            UnprivilegedProfileApi(),
            username="tldw-isolated",
            password="ephemeral-password",
        ):
            pytest.fail("precondition failure must never release or launch a probe")

    assert events == [
        ("create-account", "tldw-isolated"),
        ("create-profile", "tldw-isolated"),
        ("delete-account", "tldw-isolated"),
    ]


def test_windows_profile_source_never_writes_interactive_hkcu() -> None:
    environment_source = (QUALIFICATION_ROOT / "environment_probe.py").read_text(
        encoding="utf-8"
    )
    common_source = (QUALIFICATION_ROOT / "common.py").read_text(encoding="utf-8")
    forbidden_direct_writes = []
    for source in (environment_source, common_source):
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if node.func.attr not in {"SetValueEx", "CreateKeyEx"} or not node.args:
                continue
            root = node.args[0]
            if isinstance(root, ast.Attribute) and root.attr == "HKEY_CURRENT_USER":
                forbidden_direct_writes.append(node)

    assert not forbidden_direct_writes
    assert "_temporary_registry_value" not in environment_source
    for native_api in (
        "NetUserAdd",
        "CreateProcessWithLogonW",
        "LOGON_WITH_PROFILE",
        "DeleteProfileW",
        "NetUserDel",
        "RegOpenCurrentUser",
        "RegSetValueExW",
    ):
        assert native_api in common_source


def test_windows_profile_launch_names_the_executable_explicitly() -> None:
    common_source = (QUALIFICATION_ROOT / "common.py").read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(common_source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "CreateProcessWithLogonW"
    ]

    assert len(calls) == 1
    application_name = calls[0].args[4]
    assert isinstance(application_name, ast.Subscript)
    assert isinstance(application_name.value, ast.Name)
    assert application_name.value.id == "launch_argv"
    assert isinstance(application_name.slice, ast.Constant)
    assert application_name.slice.value == 0


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-field",
        "missing-row",
        "duplicate-row",
        "root-status",
        "child-status",
        "child-mandatory",
        "native-semantics",
    ),
)
def test_semantic_schema_rejects_incomplete_or_inconsistent_rows(
    mutation: str,
) -> None:
    common = _load_qualification_module("common")
    winpty = _load_qualification_module("pywinpty_probe")
    payload = _minimal_raw_payload("win-amd64-py311", "pywinpty")
    payload["status"] = "PASS"
    payload.pop("reason_category", None)
    payload["rows"] = winpty._build_native_rows(_complete_windows_observations())
    if mutation == "missing-field":
        payload["rows"][0].pop("distribution_version")
    elif mutation == "missing-row":
        payload["rows"].pop()
    elif mutation == "duplicate-row":
        payload["rows"].append(dict(payload["rows"][0]))
    elif mutation == "root-status":
        payload["status"] = "FAIL"
    elif mutation == "child-status":
        payload["rows"][0]["status"] = "FAIL"
    elif mutation == "child-mandatory":
        payload["rows"][0]["mandatory"] = False
    else:
        payload["rows"][0]["native_execution"] = False

    with pytest.raises(common.QualificationError):
        common.validate_content_free(payload)


def test_retained_raw_evidence_matches_current_schema_and_one_generation() -> None:
    common = _load_qualification_module("common")
    expected_rows = {
        "linux-arm64-py312",
        "macos-arm64-py311",
        "macos-arm64-py312",
        "macos-arm64-py313",
        "macos-arm64-py314",
        WINDOWS_ROW_ID,
    }
    row_directories = sorted(path for path in RAW_ROOT.iterdir() if path.is_dir())

    assert {path.name for path in row_directories} == expected_rows
    for row_directory in row_directories:
        assert (row_directory / common.CURRENT_GENERATION_MARKER).is_file()
        published_payloads = common.validate_published_row(row_directory, recover=False)
        raw_files = sorted(row_directory.glob("*.json"))
        assert len(raw_files) == 6, row_directory
        payloads = [
            json.loads(raw_file.read_text(encoding="utf-8")) for raw_file in raw_files
        ]
        assert published_payloads == payloads
        environment_probes = (
            {"environment-default", "environment-powershell", "environment-cmd"}
            if row_directory.name == WINDOWS_ROW_ID
            else {"environment-default", "environment-bash", "environment-zsh"}
        )
        assert {payload["probe"] for payload in payloads} == environment_probes | {
            "artifacts",
            "pyte",
            "pywinpty",
        }
        for payload in payloads:
            assert payload["schema_version"] == common.SCHEMA_VERSION
            common.validate_content_free(payload)
        common.validate_sibling_identity(payloads)


def test_sibling_rows_require_one_generation_platform_runtime_and_row_identity() -> (
    None
):
    common = _load_qualification_module("common")
    first = _minimal_raw_payload("macos-arm64-py312", "environment-bash")
    second = _minimal_raw_payload("macos-arm64-py312", "environment-zsh")
    for payload in (first, second):
        payload["generation_id"] = "a" * 32
    common.validate_sibling_identity([first, second])

    second["runtime"] = {
        "kind": "docker",
        "image": "ubuntu:24.04",
        "image_id": "sha256:" + "b" * 64,
        "container_id": "c" * 12,
    }
    with pytest.raises(common.QualificationError, match="runtime"):
        common.validate_sibling_identity([first, second])

    second["runtime"] = dict(first["runtime"])
    second["generation_id"] = "b" * 32
    with pytest.raises(common.QualificationError, match="generation"):
        common.validate_sibling_identity([first, second])


def _write_passing_windows_environment_sibling(
    tmp_path: Path,
    probe: str,
    *,
    architecture: str = "AMD64",
) -> None:
    result = {
        "startup_completed": True,
        "command_discovery": True,
        "profile_contract_applicable": True,
        "profile_marker_present": True,
        "sensitive_key_repopulated_by_profile": True,
        "module_discovery": True,
        "default_module_discovery": True,
        "profile_extended_module_discovery": True,
        "captured_byte_count": 1,
        "capture_within_bound": True,
    }
    payload = _minimal_raw_payload("win-amd64-py311", probe)
    payload["platform"] = {
        **payload["platform"],
        "os": "Windows",
        "architecture": architecture,
        "python_executable_name": "python.exe",
    }
    payload["actual_startup"] = dict(result)
    payload["synthetic_profile"] = dict(result)
    if probe == "environment-default":
        payload["selected_shell_family"] = "powershell"
    (tmp_path / f"{probe}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_environment_decision_accepts_documented_windows_environment_filenames(
    tmp_path: Path,
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    for probe in (
        "environment-default",
        "environment-powershell",
        "environment-cmd",
    ):
        _write_passing_windows_environment_sibling(tmp_path, probe)

    facts = module._environment_row_facts(tmp_path / "artifacts.json")

    assert facts == {
        "profile_module_discovery": True,
        "default_module_discovery": True,
        "profile_extended_module_discovery": True,
    }


def test_environment_decision_rejects_status_only_or_mismatched_siblings(
    tmp_path: Path,
) -> None:
    module = _load_qualification_module("pywinpty_probe")
    for probe in (
        "environment-default",
        "environment-powershell",
        "environment-cmd",
    ):
        architecture = "x86_64" if probe == "environment-cmd" else "AMD64"
        _write_passing_windows_environment_sibling(
            tmp_path,
            probe,
            architecture=architecture,
        )

    facts = module._environment_row_facts(tmp_path / "artifacts.json")

    assert facts == {
        "profile_module_discovery": False,
        "default_module_discovery": False,
        "profile_extended_module_discovery": False,
    }


def test_artifact_manifest_rehashes_exact_artifact_during_probe(tmp_path: Path) -> None:
    common = _load_qualification_module("common")
    row_dir = tmp_path / "row"
    artifact_dir = row_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "fixture-1.0-py3-none-any.whl"
    artifact.write_bytes(b"before")
    payload = _minimal_raw_payload("macos-arm64-py312", "artifacts")
    item = payload["artifacts"][0]
    assert isinstance(item, dict)
    digest = hashlib.sha256(b"before").hexdigest()
    item.update(
        {
            "filename": artifact.name,
            "sha256": digest,
            "sha256_before_install": digest,
            "sha256_after_install": digest,
            "size_bytes": len(b"before"),
        }
    )
    payload["resolved_distributions"] = [
        {
            "name": "fixture",
            "version": "1.0",
            "record_file": "RECORD",
            "record_file_sha256": "c" * 64,
            "primary_file": "fixture.py",
            "primary_file_sha256": "d" * 64,
        }
    ]
    manifest = row_dir / "artifacts.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    artifact.write_bytes(b"after")

    with pytest.raises(common.QualificationError, match="artifact.*changed"):
        common.artifact_manifest(manifest, required_distribution="fixture")


@pytest.mark.parametrize(
    "mutation",
    ("filename", "size", "version", "record", "primary"),
)
def test_artifact_manifest_rejects_unbound_artifact_or_install_facts(
    tmp_path: Path,
    mutation: str,
) -> None:
    common = _load_qualification_module("common")
    row_dir = tmp_path / "row"
    artifact_dir = row_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "fixture-1.0-py3-none-any.whl"
    artifact.write_bytes(b"artifact")
    digest = hashlib.sha256(b"artifact").hexdigest()
    payload = _minimal_raw_payload("macos-arm64-py312", "artifacts")
    item = payload["artifacts"][0]
    assert isinstance(item, dict)
    item.update(
        {
            "filename": artifact.name,
            "name": "fixture",
            "version": "1.0",
            "sha256": digest,
            "sha256_before_install": digest,
            "sha256_after_install": digest,
            "size_bytes": len(b"artifact"),
        }
    )
    installed = {
        "name": "fixture",
        "version": "1.0",
        "record_file": "fixture-1.0.dist-info/RECORD",
        "record_file_sha256": "c" * 64,
        "primary_file": "fixture.py",
        "primary_file_sha256": "d" * 64,
    }
    payload["resolved_distributions"] = [installed]
    if mutation == "filename":
        item["filename"] = "../fixture.whl"
    elif mutation == "size":
        item["size_bytes"] = len(b"artifact") + 1
    elif mutation == "version":
        installed["version"] = "2.0"
    elif mutation == "record":
        installed["record_file"] = None
    else:
        installed["primary_file_sha256"] = None
    manifest = row_dir / "artifacts.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(common.QualificationError):
        common.artifact_manifest(manifest, required_distribution="fixture")


def test_pyte_probe_binds_artifact_and_installed_record_primary_facts(
    monkeypatch,
) -> None:
    module = _load_qualification_module("pyte_probe")
    installed = {
        "primary_file_name": "pyte/__init__.py",
        "primary_file_sha256": "c" * 64,
        "record_file_name": "pyte-0.8.2.dist-info/RECORD",
        "record_file_sha256": "d" * 64,
    }

    class FakeDistribution:
        version = "0.8.2"

    monkeypatch.setattr(
        module.importlib.metadata,
        "distribution",
        lambda name: FakeDistribution(),
    )
    monkeypatch.setattr(
        module,
        "_installed_file_facts",
        lambda distribution: dict(installed),
    )
    artifact_hash = "a" * 64
    manifest = {
        "artifacts": [
            {
                "filename": "pyte-0.8.2-py3-none-any.whl",
                "name": "pyte",
                "version": "0.8.2",
                "size_bytes": 123,
                "sha256": artifact_hash,
                "sha256_before_install": artifact_hash,
                "sha256_after_install": artifact_hash,
            }
        ],
        "resolved_distributions": [
            {
                "name": "pyte",
                "version": "0.8.2",
                "primary_file": installed["primary_file_name"],
                "primary_file_sha256": installed["primary_file_sha256"],
                "record_file": installed["record_file_name"],
                "record_file_sha256": installed["record_file_sha256"],
            }
        ],
    }

    passed, facts = module._artifact_binding(manifest)

    assert passed is True
    assert facts == {
        "artifact_filename": "pyte-0.8.2-py3-none-any.whl",
        "artifact_sha256": artifact_hash,
        "artifact_size_bytes": 123,
        "artifact_verified_during_probe": True,
        "distribution_version": "0.8.2",
        **installed,
    }
    manifest["resolved_distributions"][0]["record_file_sha256"] = "b" * 64
    assert module._artifact_binding(manifest)[0] is False


def test_full_screen_program_requires_class_marker_and_clean_exit() -> None:
    module = _load_qualification_module("pyte_probe")
    good = module._CaptureResult(
        output=b"\x1b[?1049hEDITOR\x1b[?1049l",
        exit_code=0,
        timed_out=False,
        capture_within_bound=True,
        terminated=False,
        killed=False,
    )
    bad_exit = module._CaptureResult(
        output=good.output,
        exit_code=1,
        timed_out=False,
        capture_within_bound=True,
        terminated=False,
        killed=False,
    )
    arbitrary = module._CaptureResult(
        output=b"nonempty",
        exit_code=0,
        timed_out=False,
        capture_within_bound=True,
        terminated=False,
        killed=False,
    )

    assert module._program_capture_passes("editor", good)
    assert not module._program_capture_passes("editor", bad_exit)
    assert not module._program_capture_passes("editor", arbitrary)


def test_full_screen_pager_accepts_real_line_erase_transition() -> None:
    module = _load_qualification_module("pyte_probe")
    result = module._CaptureResult(
        output=b"\x1b[7mstatus\x1b[0m\x1b[K",
        exit_code=0,
        timed_out=False,
        capture_within_bound=True,
        terminated=False,
        killed=False,
    )

    assert module._program_capture_passes("pager", result)


def test_windows_full_screen_fixtures_satisfy_their_acceptance_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_qualification_module("pyte_probe")
    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module, "_feed", lambda pyte_module, output: None)

    passed, facts = module._program_matrix(None)

    assert passed is True
    assert all(facts["class_pass"].values())


def test_capture_waits_for_interactive_startup_before_sending_input() -> None:
    if os.name == "nt":
        pytest.skip("POSIX PTY sequencing test")
    module = _load_qualification_module("pyte_probe")
    source = (
        "import os,sys,termios,time,tty; "
        "sys.stdout.write('\\x1b[?1049h'); sys.stdout.flush(); "
        "time.sleep(0.05); tty.setraw(0); "
        "termios.tcflush(0, termios.TCIFLUSH); "
        "value=os.read(0,1); sys.exit(0 if value==b'q' else 2)"
    )

    result = module._capture([sys.executable, "-c", source], b"q")

    assert result.exit_code == 0
    assert not result.timed_out
    assert not result.terminated
    assert not result.killed


def test_capture_reaps_clean_exit_after_pty_stream_closes() -> None:
    if os.name == "nt":
        pytest.skip("POSIX PTY sequencing test")
    module = _load_qualification_module("pyte_probe")
    source = "import os,time; os.closerange(0,3); time.sleep(0.2)"

    result = module._capture([sys.executable, "-c", source])

    assert result.exit_code == 0
    assert not result.timed_out
    assert not result.terminated
    assert not result.killed


def test_bounded_runner_bounds_output_and_reaps_whole_process_group(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("native Windows Job behavior is covered by the Windows probe")
    common = _load_qualification_module("common")
    pid_file = tmp_path / "child.pid"
    source = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); "
        "print('x'*100000); time.sleep(30)"
    )

    completed = common.run_bounded(
        [sys.executable, "-c", source, str(pid_file)],
        cwd=tmp_path,
        timeout_seconds=0.5,
        output_limit=1024,
        operation="group-cleanup-test",
    )

    assert completed.timed_out is False
    assert completed.overflowed is True
    assert completed.terminated is True
    assert completed.stored_output_bytes <= 1024
    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("descendant survived bounded-runner process-group cleanup")


def test_windows_bootstrap_is_released_only_after_successful_job_admission(
    tmp_path: Path,
) -> None:
    common = _load_qualification_module("common")
    events: list[object] = []

    class FakeInput:
        def write(self, value: bytes) -> int:
            events.append(("release", value))
            return len(value)

        def flush(self) -> None:
            events.append("flush")

        def close(self) -> None:
            events.append("stdin-close")

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = FakeInput()

        def poll(self) -> None:
            return None

    process = FakeProcess()

    class FakeProfile:
        def launch_waiting_bootstrap(self, argv, **kwargs):
            events.append(("profile-bootstrap-start", tuple(argv), kwargs))
            return process

        def verify_process_identity_profile(self, admitted: object) -> None:
            assert admitted is process
            events.append("profile-verify")

    class FakeJob:
        def assign(self, admitted: object) -> None:
            assert admitted is process
            events.append("job-admit")

        def contains(self, admitted: object) -> bool:
            assert admitted is process
            events.append("job-member")
            return True

    admitted, writer = common._launch_admitted_windows_bootstrap(
        ("real-command.exe", "--fixture"),
        cwd=tmp_path,
        env={"TERM": "linux"},
        input_bytes=b"payload",
        job=FakeJob(),
        profile=FakeProfile(),
    )
    writer.join(timeout=1.0)

    assert admitted is process
    assert not writer.is_alive()
    assert events[0][0] == "profile-bootstrap-start"
    assert "_bounded-bootstrap" in events[0][1]
    assert events[1] == "job-admit"
    assert events[2] == "job-member"
    assert events[3] == "profile-verify"
    assert events[4] == (
        "release",
        common.WINDOWS_BOOTSTRAP_RELEASE + b"payload",
    )


def test_windows_bootstrap_admission_failure_never_releases_or_spawns_child(
    tmp_path: Path,
) -> None:
    common = _load_qualification_module("common")
    writes: list[bytes] = []

    class FakeInput:
        def write(self, value: bytes) -> int:
            writes.append(value)
            return len(value)

        def close(self) -> None:
            return None

    class FakeProcess:
        stdin = FakeInput()

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 1

    class RejectingJob:
        def assign(self, process: object) -> None:
            del process
            raise common.QualificationError("bounded subprocess Job admission failed")

    with pytest.raises(common.QualificationError, match="Job admission"):
        common._launch_admitted_windows_bootstrap(
            ("real-command.exe",),
            cwd=tmp_path,
            env={"TERM": "linux"},
            input_bytes=b"payload",
            job=RejectingJob(),
            popen_factory=lambda argv, **kwargs: FakeProcess(),
        )

    assert writes == []


@pytest.mark.parametrize("failure", ("membership", "identity"))
def test_windows_bootstrap_verification_failure_never_releases(
    failure: str,
    tmp_path: Path,
) -> None:
    common = _load_qualification_module("common")
    writes: list[bytes] = []

    class FakeInput:
        def write(self, value: bytes) -> int:
            writes.append(value)
            return len(value)

        def close(self) -> None:
            return None

    class FakeProcess:
        stdin = FakeInput()

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 1

    process = FakeProcess()

    class FakeProfile:
        def launch_waiting_bootstrap(self, argv, **kwargs):
            del argv, kwargs
            return process

        def verify_process_identity_profile(self, admitted: object) -> None:
            assert admitted is process
            if failure == "identity":
                raise common.QualificationError(
                    "disposable Windows process identity/profile verification failed"
                )

    class FakeJob:
        def assign(self, admitted: object) -> None:
            assert admitted is process

        def contains(self, admitted: object) -> bool:
            assert admitted is process
            return failure != "membership"

    with pytest.raises(common.QualificationError, match="verification failed"):
        common._launch_admitted_windows_bootstrap(
            ("real-command.exe",),
            cwd=tmp_path,
            env={"TERM": "linux"},
            input_bytes=b"payload",
            job=FakeJob(),
            profile=FakeProfile(),
        )

    assert writes == []


def test_windows_bootstrap_refuses_to_spawn_real_command_outside_job() -> None:
    common = _load_qualification_module("common")
    spawned: list[tuple[str, ...]] = []

    result = common._bounded_bootstrap_main(
        ("real-command.exe", "--fast"),
        input_stream=__import__("io").BytesIO(common.WINDOWS_BOOTSTRAP_RELEASE),
        output_stream=__import__("io").BytesIO(),
        error_stream=__import__("io").BytesIO(),
        is_in_job=lambda: False,
        popen_factory=lambda argv, **kwargs: spawned.append(tuple(argv)),
    )

    assert result != 0
    assert spawned == []


def test_current_process_job_check_uses_pointer_width_ctypes_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = _load_qualification_module("common")

    class FakeBool:
        def __init__(self) -> None:
            self.value = 0

    class FakeFunction:
        def __init__(self, name: str) -> None:
            self.name = name
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            if self.name == "GetCurrentProcess":
                return "current-process"
            args[2].value = 1
            return 1

    get_current_process = FakeFunction("GetCurrentProcess")
    is_process_in_job = FakeFunction("IsProcessInJob")
    kernel32 = SimpleNamespace(
        GetCurrentProcess=get_current_process,
        IsProcessInJob=is_process_in_job,
    )
    fake_wintypes = SimpleNamespace(BOOL=FakeBool, HANDLE=object())
    fake_ctypes = ModuleType("ctypes")
    fake_ctypes.WinDLL = lambda *args, **kwargs: kernel32
    fake_ctypes.POINTER = lambda value: ("pointer", value)
    fake_ctypes.byref = lambda value: value
    fake_ctypes.wintypes = fake_wintypes

    monkeypatch.setattr(common.os, "name", "nt")
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    monkeypatch.setitem(sys.modules, "ctypes.wintypes", fake_wintypes)

    assert common._current_process_is_in_job() is True
    assert get_current_process.argtypes == []
    assert get_current_process.restype is fake_wintypes.HANDLE
    assert is_process_in_job.argtypes == [
        fake_wintypes.HANDLE,
        fake_wintypes.HANDLE,
        ("pointer", fake_wintypes.BOOL),
    ]
    assert is_process_in_job.restype is fake_wintypes.BOOL


def _install_fake_windows_api_modules(
    monkeypatch: pytest.MonkeyPatch,
    common: ModuleType,
) -> tuple[ModuleType, SimpleNamespace, dict[str, SimpleNamespace]]:
    class FakeFunction:
        def __init__(self) -> None:
            self.argtypes = None
            self.restype = None

    class FakeLibrary(SimpleNamespace):
        def __getattr__(self, name: str):
            function = FakeFunction()
            setattr(self, name, function)
            return function

    libraries = {
        name: FakeLibrary() for name in ("advapi32", "kernel32", "netapi32", "userenv")
    }
    fake_wintypes = SimpleNamespace(
        BOOL=object(),
        DWORD=object(),
        HANDLE=object(),
        LPCWSTR=object(),
        LPWSTR=object(),
    )
    fake_ctypes = ModuleType("ctypes")
    fake_ctypes.WinDLL = lambda name, **kwargs: libraries[name]
    fake_ctypes.POINTER = lambda value: ("pointer", value)
    fake_ctypes.c_int = object()
    fake_ctypes.c_long = object()
    fake_ctypes.c_ubyte = object()
    fake_ctypes.c_void_p = object()
    fake_ctypes.wintypes = fake_wintypes

    monkeypatch.setattr(common.os, "name", "nt")
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    monkeypatch.setitem(sys.modules, "ctypes.wintypes", fake_wintypes)
    return fake_ctypes, fake_wintypes, libraries


def _assert_explicit_ctypes_signatures(
    library: SimpleNamespace,
    function_names: tuple[str, ...],
) -> None:
    for function_name in function_names:
        function = getattr(library, function_name)
        assert function.argtypes is not None, function_name
        assert function.restype is not None, function_name


def test_native_windows_profile_api_declares_all_ctypes_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = _load_qualification_module("common")
    _, _, libraries = _install_fake_windows_api_modules(monkeypatch, common)

    common._NativeWindowsProfileApi()

    _assert_explicit_ctypes_signatures(
        libraries["advapi32"],
        (
            "LookupAccountNameW",
            "ConvertSidToStringSidW",
            "CreateProcessWithLogonW",
            "OpenProcessToken",
            "GetTokenInformation",
            "RegOpenKeyExW",
            "RegCloseKey",
        ),
    )
    _assert_explicit_ctypes_signatures(
        libraries["kernel32"], ("CloseHandle", "LocalFree")
    )
    _assert_explicit_ctypes_signatures(
        libraries["netapi32"], ("NetUserAdd", "NetUserDel")
    )
    _assert_explicit_ctypes_signatures(
        libraries["userenv"],
        ("CreateProfile", "DeleteProfileW", "GetUserProfileDirectoryW"),
    )


def test_native_windows_create_profile_uses_max_path_buffer() -> None:
    common = _load_qualification_module("common")
    requested_capacities: list[int] = []
    create_profile_calls: list[tuple[object, ...]] = []

    class FakeBuffer:
        value = r"C:\Users\tldw-isolated"

        def __init__(self, capacity: int) -> None:
            self.capacity = capacity

        def __len__(self) -> int:
            return self.capacity

    def create_unicode_buffer(capacity: int) -> FakeBuffer:
        requested_capacities.append(capacity)
        return FakeBuffer(capacity)

    def create_profile(*args: object) -> int:
        create_profile_calls.append(args)
        return 0

    api = object.__new__(common._NativeWindowsProfileApi)
    api.ctypes = SimpleNamespace(create_unicode_buffer=create_unicode_buffer)
    api.userenv = SimpleNamespace(CreateProfile=create_profile)
    api._account_sid = lambda username: "S-1-5-21-1234"

    identity = api.create_profile("tldw-isolated")

    assert requested_capacities == [260]
    assert create_profile_calls[0][3] == 260
    assert identity.username == "tldw-isolated"


def test_native_windows_registry_api_declares_all_ctypes_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = _load_qualification_module("common")
    _, _, libraries = _install_fake_windows_api_modules(monkeypatch, common)

    common._NativeWindowsRegistryApi()

    _assert_explicit_ctypes_signatures(
        libraries["advapi32"],
        (
            "GetUserNameW",
            "RegOpenCurrentUser",
            "RegCreateKeyExW",
            "RegSetValueExW",
            "RegCloseKey",
        ),
    )


def test_windows_handle_process_declares_all_ctypes_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = _load_qualification_module("common")
    _, _, libraries = _install_fake_windows_api_modules(monkeypatch, common)

    common._WindowsHandleProcess(
        args=("command.exe",),
        process_handle=object(),
        pid=123,
        stdin=SimpleNamespace(),
        stdout=SimpleNamespace(),
        stderr=SimpleNamespace(),
    )

    _assert_explicit_ctypes_signatures(
        libraries["kernel32"],
        (
            "WaitForSingleObject",
            "GetExitCodeProcess",
            "TerminateProcess",
            "GenerateConsoleCtrlEvent",
            "CloseHandle",
        ),
    )


def test_windows_handle_process_close_releases_process_handle_once() -> None:
    common = _load_qualification_module("common")
    handle = object()
    closed: list[object] = []
    process = object.__new__(common._WindowsHandleProcess)
    process._handle = handle
    process._kernel32 = SimpleNamespace(
        CloseHandle=lambda value: closed.append(value) or True
    )

    process.close()
    process.close()

    assert closed == [handle]
    assert process._handle is None


def test_windows_bootstrap_payload_dispatches_verified_profile_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    common = _load_qualification_module("common")
    observed: dict[str, object] = {}
    setup = common.WindowsBootstrapSetup(
        profile_files=(("Documents/PowerShell/profile.ps1", "profile"),),
        registry_values=((r"Software\Fixture", "Value", "data"),),
    )

    def fake_bootstrap_environment(overrides: dict[str, str]) -> dict[str, str]:
        observed["overrides"] = overrides
        return {"USERNAME": "tldw-isolated", "TERM": "linux"}

    def fake_bootstrap_main(command, **kwargs) -> int:
        observed["command"] = tuple(command)
        observed.update(kwargs)
        return 17

    monkeypatch.setattr(common, "_bootstrap_environment", fake_bootstrap_environment)
    monkeypatch.setattr(common, "_bounded_bootstrap_main", fake_bootstrap_main)
    argv = common._bootstrap_argv(
        ("real-command.exe", "--fixture"),
        cwd=tmp_path,
        env={"TERM": "linux"},
        setup=setup,
        expected_username="tldw-isolated",
    )

    assert common.main(["_bounded-bootstrap", argv[-1]]) == 17
    assert observed["overrides"] == {"TERM": "linux"}
    assert observed["environment"] == {
        "USERNAME": "tldw-isolated",
        "TERM": "linux",
    }
    assert observed["expected_username"] == "tldw-isolated"
    assert observed["setup"] == setup
    assert observed["command"] == ("real-command.exe", "--fixture")


def test_windows_logon_launch_externalizes_oversized_bootstrap_payload(
    tmp_path: Path,
) -> None:
    common = _load_qualification_module("common")
    inline_argv = common._bootstrap_argv(
        ("real-command.exe", "--fixture"),
        cwd=tmp_path,
        env={"TERM": "linux"},
        setup=common.WindowsBootstrapSetup(
            profile_files=(("Documents/PowerShell/profile.ps1", "x" * 2048),)
        ),
        expected_username="tldw-isolated",
    )

    assert (
        len(subprocess.list2cmdline(inline_argv))
        > common.WINDOWS_LOGON_COMMAND_LINE_LIMIT
    )
    external_argv = common._externalize_windows_bootstrap_argv(inline_argv, tmp_path)

    assert external_argv[2] == "_bounded-bootstrap-file"
    assert Path(external_argv[3]).read_text(encoding="ascii") == inline_argv[3]
    assert (
        len(subprocess.list2cmdline(external_argv))
        <= common.WINDOWS_LOGON_COMMAND_LINE_LIMIT
    )


def test_windows_bootstrap_file_dispatches_encoded_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    common = _load_qualification_module("common")
    payload_path = tmp_path / "bootstrap.payload"
    payload_path.write_text("encoded-payload", encoding="ascii")
    observed: list[str] = []

    def fake_run(encoded: str) -> int:
        observed.append(encoded)
        return 19

    monkeypatch.setattr(common, "_run_bounded_bootstrap_payload", fake_run)

    assert common.main(["_bounded-bootstrap-file", str(payload_path)]) == 19
    assert observed == ["encoded-payload"]


def test_windows_environment_output_overflow_is_an_explicit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_qualification_module("environment_probe")

    def fake_run(argv, **kwargs):
        nonce = kwargs["result_nonce"]
        return SimpleNamespace(
            args=argv,
            returncode=0,
            stdout=(module._result_marker(nonce) + "0,0,0,0\r\n").encode(),
            stderr=b"",
            timed_out=False,
            overflowed=True,
        )

    monkeypatch.setattr(module, "_run_shell_process", fake_run)
    result = module._run_windows_shell("cmd", Path("cmd.exe"), {"TERM": "linux"})

    assert result["output_overflowed"] is True
    assert result["capture_within_bound"] is False
    assert result["startup_completed"] is False
    assert module._windows_shell_passed("cmd", result) is False


def test_bounded_runner_stops_on_output_overflow_without_unbounded_spooling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("host-independent Windows ordering is covered separately")
    common = _load_qualification_module("common")
    output_limit = 32 * 1024
    source = (
        "import os,time; chunk=b'x'*8192; "
        "[(os.write(1,chunk),os.write(2,chunk)) for _ in range(4096)]; "
        "time.sleep(30)"
    )

    def forbid_temporary_output(*args, **kwargs):
        del args, kwargs
        raise AssertionError("bounded output must not use TemporaryFile")

    monkeypatch.setattr(common.tempfile, "TemporaryFile", forbid_temporary_output)
    started = time.monotonic()

    completed = common.run_bounded(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        timeout_seconds=10.0,
        output_limit=output_limit,
        operation="output-overflow-test",
    )

    assert completed.overflowed is True
    assert completed.timed_out is False
    assert completed.terminated is True
    assert len(completed.stdout) + len(completed.stderr) <= output_limit
    assert completed.stored_output_bytes <= output_limit
    assert time.monotonic() - started < 5.0


@pytest.mark.parametrize(
    "failing_channels",
    (("stdout",), ("stderr",), ("stdout", "stderr")),
    ids=("stdout", "stderr", "simultaneous"),
)
def test_bounded_runner_fails_closed_on_output_drain_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failing_channels: tuple[str, ...],
) -> None:
    if os.name == "nt":
        pytest.skip("host-independent Windows ordering is covered separately")
    common = _load_qualification_module("common")
    barrier = threading.Barrier(2) if len(failing_channels) == 2 else None
    terminated = threading.Event()

    class DrainStream:
        def __init__(self, channel: str) -> None:
            self.channel = channel
            self.closed = False

        def read(self, size: int) -> bytes:
            assert size == common.OUTPUT_READ_CHUNK
            if self.channel not in failing_channels:
                return b""
            if barrier is not None:
                barrier.wait(timeout=1.0)
            raise OSError(f"{self.channel} drain exploded")

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self) -> None:
            self.pid = 424242
            self.returncode: int | None = None
            self.stdout = DrainStream("stdout")
            self.stderr = DrainStream("stderr")

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(common.subprocess, "Popen", lambda *args, **kwargs: process)

    def terminate(process_arg, job, *, grace_seconds):
        del job, grace_seconds
        assert process_arg is process
        terminated.set()
        process.returncode = -15
        return True, False

    monkeypatch.setattr(common, "terminate_owned_group", terminate)
    started = time.monotonic()

    with pytest.raises(
        common.QualificationError, match="output_drain_failed"
    ) as caught:
        common.run_bounded(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            timeout_seconds=1.0,
            output_limit=1024,
            operation="reader-failure-test",
        )

    assert terminated.is_set()
    assert time.monotonic() - started < 0.5
    assert process.stdout.closed
    assert process.stderr.closed
    assert isinstance(caught.value.__cause__, OSError)
    assert str(caught.value.__cause__) in {
        f"{channel} drain exploded" for channel in failing_channels
    }


def test_environment_pty_capture_does_not_use_temporary_file() -> None:
    module = _load_qualification_module("environment_probe")
    source = inspect.getsource(module._read_pty)

    assert "TemporaryFile" not in source


def test_pyte_capture_reaps_descendants_and_records_timeout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("native Windows Job behavior is covered by the Windows probe")
    module = _load_qualification_module("pyte_probe")
    monkeypatch.setattr(module, "CAPTURE_SECONDS", 0.2)
    pid_file = tmp_path / "child.pid"
    source = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)"
    )

    result = module._capture([sys.executable, "-c", source, str(pid_file)])

    assert result.timed_out is True
    assert result.terminated is True
    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("PTY capture descendant survived process-group cleanup")


def test_collect_row_rolls_back_when_second_sibling_replace_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    common = _load_qualification_module("common")
    source = tmp_path / "source"
    evidence_root = tmp_path / "raw"
    _write_complete_source_row(source)
    assert (
        common.collect_row(row_dir=source, evidence_root=evidence_root, replace=False)
        == 6
    )
    published = evidence_root / "macos-arm64-py312"
    previous = {path.name: path.read_bytes() for path in published.glob("*.json")}

    original_replace = common.os.replace
    sibling_replacements = 0

    def interrupt_second_sibling(source_path: Path, destination_path: Path) -> None:
        nonlocal sibling_replacements
        destination = Path(destination_path)
        if destination.suffix == ".json" and not destination.name.startswith("."):
            sibling_replacements += 1
            if sibling_replacements == 2:
                raise OSError("injected second sibling replacement failure")
        original_replace(source_path, destination_path)

    monkeypatch.setattr(common.os, "replace", interrupt_second_sibling)

    with pytest.raises(OSError, match="second sibling"):
        common.collect_row(row_dir=source, evidence_root=evidence_root, replace=True)

    assert {path.name: path.read_bytes() for path in published.glob("*.json")} == (
        previous
    )
    payloads = common.validate_published_row(published, recover=False)
    assert len({payload["generation_id"] for payload in payloads}) == 1


def test_collect_row_replaces_stale_pre_marker_generation_fail_closed(
    tmp_path: Path,
) -> None:
    common = _load_qualification_module("common")
    evidence_root = tmp_path / "raw"
    published = _make_stale_legacy_published_row(
        common,
        source=tmp_path / "legacy-source",
        evidence_root=evidence_root,
    )
    source = tmp_path / "current-source"
    _write_complete_source_row(source)

    assert (
        common.collect_row(
            row_dir=source,
            evidence_root=evidence_root,
            replace=True,
        )
        == 6
    )

    payloads = common.validate_published_row(published, recover=False)
    assert len(payloads) == 6
    assert not list(published.glob(f"{common.RECOVERY_DIRECTORY_PREFIX}*"))
    assert not (published / common.PENDING_PUBLICATION_MARKER).exists()


def test_stale_pre_marker_backup_precedes_siblings_and_marker_commits_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    common = _load_qualification_module("common")
    evidence_root = tmp_path / "raw"
    published = _make_stale_legacy_published_row(
        common,
        source=tmp_path / "legacy-source",
        evidence_root=evidence_root,
    )
    previous = {path.name: path.read_bytes() for path in published.glob("*.json")}
    previous_generation = {
        json.loads(value)["generation_id"] for value in previous.values()
    }
    source = tmp_path / "current-source"
    _write_complete_source_row(source)
    original_replace = common.os.replace
    publication_events: list[str] = []

    def observe_replace(source_path: Path, destination_path: Path) -> None:
        destination = Path(destination_path)
        if destination.parent == published and (
            destination.suffix == ".json" and not destination.name.startswith(".")
        ):
            pending = json.loads(
                (published / common.PENDING_PUBLICATION_MARKER).read_text(
                    encoding="utf-8"
                )
            )
            recovery = published / pending["recovery_directory"]
            assert {
                path.name: path.read_bytes() for path in recovery.glob("*.json")
            } == previous
            common._validate_legacy_recovery_generation(recovery)
            publication_events.append(destination.name)
        elif destination == published / common.CURRENT_GENERATION_MARKER:
            assert len(publication_events) == 6
            assert not destination.exists()
            visible_generations = {
                json.loads(path.read_text(encoding="utf-8"))["generation_id"]
                for path in published.glob("*.json")
            }
            assert len(visible_generations) == 1
            assert visible_generations != previous_generation
            publication_events.append(common.CURRENT_GENERATION_MARKER)
        original_replace(source_path, destination_path)

    monkeypatch.setattr(common.os, "replace", observe_replace)

    common.collect_row(row_dir=source, evidence_root=evidence_root, replace=True)

    assert publication_events[-1] == common.CURRENT_GENERATION_MARKER


def test_stale_pre_marker_replacement_rolls_back_complete_previous_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    common = _load_qualification_module("common")
    evidence_root = tmp_path / "raw"
    published = _make_stale_legacy_published_row(
        common,
        source=tmp_path / "legacy-source",
        evidence_root=evidence_root,
    )
    previous = {path.name: path.read_bytes() for path in published.glob("*.json")}
    source = tmp_path / "current-source"
    _write_complete_source_row(source)
    original_replace = common.os.replace
    sibling_replacements = 0

    def interrupt_second_sibling(source_path: Path, destination_path: Path) -> None:
        nonlocal sibling_replacements
        destination = Path(destination_path)
        if destination.suffix == ".json" and not destination.name.startswith("."):
            sibling_replacements += 1
            if sibling_replacements == 2:
                raise OSError("injected stale-generation replacement failure")
        original_replace(source_path, destination_path)

    monkeypatch.setattr(common.os, "replace", interrupt_second_sibling)

    with pytest.raises(OSError, match="stale-generation"):
        common.collect_row(row_dir=source, evidence_root=evidence_root, replace=True)

    assert {path.name: path.read_bytes() for path in published.glob("*.json")} == (
        previous
    )
    assert not (published / common.CURRENT_GENERATION_MARKER).exists()
    assert not (published / common.PENDING_PUBLICATION_MARKER).exists()
    assert not list(published.glob(f"{common.RECOVERY_DIRECTORY_PREFIX}*"))


def test_abrupt_stale_pre_marker_replacement_recovers_exact_legacy_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    common = _load_qualification_module("common")
    evidence_root = tmp_path / "raw"
    published = _make_stale_legacy_published_row(
        common,
        source=tmp_path / "legacy-source",
        evidence_root=evidence_root,
    )
    previous = {path.name: path.read_bytes() for path in published.glob("*.json")}
    source = tmp_path / "current-source"
    _write_complete_source_row(source)
    original_replace = common.os.replace
    sibling_replacements = 0

    def interrupt_second_sibling(source_path: Path, destination_path: Path) -> None:
        nonlocal sibling_replacements
        destination = Path(destination_path)
        if destination.suffix == ".json" and not destination.name.startswith("."):
            sibling_replacements += 1
            if sibling_replacements == 2:
                raise SystemExit("injected abrupt legacy publication death")
        original_replace(source_path, destination_path)

    monkeypatch.setattr(common.os, "replace", interrupt_second_sibling)

    with pytest.raises(SystemExit, match="abrupt legacy"):
        common.collect_row(row_dir=source, evidence_root=evidence_root, replace=True)

    with pytest.raises(common.QualificationError, match="unfinished transaction"):
        common.validate_published_row(published, recover=False)
    monkeypatch.setattr(common.os, "replace", original_replace)
    with pytest.raises(common.QualificationError, match="legacy generation"):
        common.validate_published_row(published, recover=True)

    assert {path.name: path.read_bytes() for path in published.glob("*.json")} == (
        previous
    )
    assert not (published / common.CURRENT_GENERATION_MARKER).exists()
    assert not (published / common.PENDING_PUBLICATION_MARKER).exists()
    assert not list(published.glob(f"{common.RECOVERY_DIRECTORY_PREFIX}*"))


@pytest.mark.parametrize(
    "mutation",
    ("missing", "generation", "platform", "runtime"),
)
def test_stale_pre_marker_rejects_incomplete_or_mixed_legacy_set(
    tmp_path: Path, mutation: str
) -> None:
    common = _load_qualification_module("common")
    evidence_root = tmp_path / "raw"
    published = _make_stale_legacy_published_row(
        common,
        source=tmp_path / "legacy-source",
        evidence_root=evidence_root,
    )
    target = next(published.glob("*-pyte.json"))
    if mutation == "missing":
        target.unlink()
    else:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if mutation == "generation":
            payload["generation_id"] = "f" * 32
        elif mutation == "platform":
            payload["platform"]["architecture"] = "mixed-architecture"
        else:
            payload["runtime"] = {"kind": "mixed-runtime"}
        target.write_text(json.dumps(payload), encoding="utf-8")
    previous = {path.name: path.read_bytes() for path in published.glob("*.json")}
    source = tmp_path / "current-source"
    _write_complete_source_row(source)

    with pytest.raises(common.QualificationError, match="legacy published"):
        common.collect_row(row_dir=source, evidence_root=evidence_root, replace=True)

    assert {path.name: path.read_bytes() for path in published.glob("*.json")} == (
        previous
    )
    assert not (published / common.CURRENT_GENERATION_MARKER).exists()
    assert not (published / common.PENDING_PUBLICATION_MARKER).exists()
    assert not list(published.glob(f"{common.RECOVERY_DIRECTORY_PREFIX}*"))


def test_abrupt_mixed_generation_recovers_previous_complete_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    common = _load_qualification_module("common")
    source = tmp_path / "source"
    evidence_root = tmp_path / "raw"
    _write_complete_source_row(source)
    common.collect_row(row_dir=source, evidence_root=evidence_root, replace=False)
    published = evidence_root / "macos-arm64-py312"
    previous = {path.name: path.read_bytes() for path in published.glob("*.json")}
    previous_generation = {
        payload["generation_id"] for payload in _published_payloads(published)
    }

    original_replace = common.os.replace
    sibling_replacements = 0

    def terminate_during_second_sibling(
        source_path: Path, destination_path: Path
    ) -> None:
        nonlocal sibling_replacements
        destination = Path(destination_path)
        if destination.suffix == ".json" and not destination.name.startswith("."):
            sibling_replacements += 1
            if sibling_replacements == 2:
                raise SystemExit("simulated abrupt process death")
        original_replace(source_path, destination_path)

    with monkeypatch.context() as interruption:
        interruption.setattr(common.os, "replace", terminate_during_second_sibling)
        with pytest.raises(SystemExit, match="abrupt process death"):
            common.collect_row(
                row_dir=source,
                evidence_root=evidence_root,
                replace=True,
            )

    with pytest.raises(common.QualificationError):
        common.validate_published_row(published, recover=False)
    recovered = common.validate_published_row(published, recover=True)

    assert {path.name: path.read_bytes() for path in published.glob("*.json")} == (
        previous
    )
    assert {payload["generation_id"] for payload in recovered} == previous_generation


def test_validator_never_accepts_unrecoverable_mixed_generation(
    tmp_path: Path,
) -> None:
    common = _load_qualification_module("common")
    source = tmp_path / "source"
    evidence_root = tmp_path / "raw"
    _write_complete_source_row(source)
    common.collect_row(row_dir=source, evidence_root=evidence_root, replace=False)
    published = evidence_root / "macos-arm64-py312"
    changed = next(published.glob("*-pyte.json"))
    payload = json.loads(changed.read_text(encoding="utf-8"))
    payload["generation_id"] = "f" * 32
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(common.QualificationError):
        common.validate_published_row(published, recover=True)


@pytest.mark.parametrize("mutation", ("missing", "extra", "duplicate"))
def test_collect_row_rejects_any_nonexact_six_probe_sibling_set(
    tmp_path: Path, mutation: str
) -> None:
    row_dir = tmp_path / "row"
    _write_complete_source_row(row_dir)
    bash_path = next(row_dir.glob("*-environment-bash.json"))
    if mutation == "missing":
        bash_path.unlink()
    elif mutation == "extra":
        payload = json.loads(bash_path.read_text(encoding="utf-8"))
        payload["probe"] = "environment-cmd"
        payload["rows"][0]["id"] = "environment-cmd"
        (row_dir / "extra-environment-cmd.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    else:
        shutil.copyfile(bash_path, row_dir / "duplicate-environment-bash.json")
    evidence_root = tmp_path / "raw"

    completed = subprocess.run(
        [
            sys.executable,
            str(QUALIFICATION_ROOT / "common.py"),
            "collect-row",
            "--row-dir",
            str(row_dir),
            "--evidence-root",
            str(evidence_root),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0, mutation
    assert not list(evidence_root.glob("*/*.json"))
