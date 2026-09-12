"""Run the finite installed backup product qualification on a native runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess  # nosec B404 - fixed local commands and arguments only
import sys
from pathlib import Path
from typing import Iterable

from defusedxml import ElementTree as ET


_PRODUCT_TESTS = (
    "Tests/Utils/test_windows_files.py",
    "Tests/Backup_Recovery/test_complete_roundtrip.py::"
    "test_two_captured_profiles_restore_and_open_with_native_content",
    "Tests/Backup_Recovery/test_f9_replacement_workflow.py::"
    "test_full_f9_replacement_after_explicit_safety_and_credential_review",
    "Tests/Backup_Recovery/test_later_rollback_credential_ui.py::"
    "test_f9_later_rollback_requires_explicit_credential_review",
)
_SYNTHETIC_CREDENTIALS = (
    "test-only-new-safety-password",
    "test-only-later-safety-password",
    "qualification-worker-password",
    "alpha-synthetic-history-api-value",
    "beta-synthetic-history-api-value",
    "test-only",
)
_ALLOWED_ENVIRONMENT = frozenset(
    {
        "CI",
        "COLORTERM",
        "COMSPEC",
        "GITHUB_ACTIONS",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TERM",
        "TZ",
        "WINDIR",
    }
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Write one deterministic JSON receipt."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _run_git(workspace: Path, *arguments: str) -> str:
    """Run a fixed read-only Git query for the source receipt."""
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("Git is unavailable for source identity capture")
    completed = subprocess.run(  # nosec B603 - fixed executable and arguments
        [executable, *arguments],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _source_receipt(workspace: Path) -> dict[str, object]:
    """Identify every tracked source file used to build the installed fixture."""
    tracked = _run_git(
        workspace,
        "ls-files",
        "--",
        "tldw_chatbook",
        "Tests/Backup_Recovery",
        "Tests/ProductionApp",
        "Tests/Utils/test_windows_files.py",
        "Tests/network_guard.py",
        ".github/workflows/test.yml",
    ).splitlines()
    files = {}
    for relative in tracked:
        candidate = workspace / relative
        if candidate.is_file() and not candidate.is_symlink():
            files[relative] = _sha256(candidate)
    package = importlib.metadata.distribution("tldw_chatbook")
    return {
        "schema": 1,
        "git_head": _run_git(workspace, "rev-parse", "HEAD"),
        "git_status": _run_git(workspace, "status", "--porcelain=v1"),
        "distribution_version": package.version,
        "loaded_package": str((workspace / "tldw_chatbook/__init__.py").resolve()),
        "files": files,
    }


def _native_identity(private_root: Path) -> dict[str, object]:
    """Record the production adapter's native identity and operation decisions."""
    receipt: dict[str, object] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "os_name": os.name,
    }
    try:
        from tldw_chatbook.Backup_Recovery import native_files, qualification

        native_root = private_root / "native-identity-root"
        native_root.mkdir(mode=0o700)
        with native_files.pinned_directory(native_root) as descriptor:
            receipt["native_identity"] = qualification.native_identity(descriptor)
        receipt["operations"] = {
            operation: list(qualification.qualified_for(operation, native_root))
            for operation in (
                "publish_new",
                "publish_file",
                "publish_directory",
                "admission",
            )
        }
    except Exception as error:  # preserve a real pre-test platform failure
        receipt["identity_error"] = f"{type(error).__name__}: {error}"
    return receipt


def _private_environment(workspace: Path, private_root: Path) -> dict[str, str]:
    """Build a credential-free, offline environment rooted below runner temp."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _ALLOWED_ENVIRONMENT
    }
    directories = {
        "HOME": private_root / "home",
        "USERPROFILE": private_root / "home",
        "XDG_CONFIG_HOME": private_root / "xdg-config",
        "XDG_DATA_HOME": private_root / "xdg-data",
        "XDG_CACHE_HOME": private_root / "xdg-cache",
        "XDG_STATE_HOME": private_root / "xdg-state",
        "TEMP": private_root / "tmp",
        "TMP": private_root / "tmp",
        "TMPDIR": private_root / "tmp",
    }
    for directory in set(directories.values()):
        directory.mkdir(parents=True, mode=0o700)
    selector = private_root / "xdg-config" / "config.toml"
    selector.write_text(
        '[general]\nusers_name="windows-qualification"\n'
        '[first_run]\nsetup_completed=true\n'
        '[splash_screen]\nenabled=false\n',
        encoding="utf-8",
    )
    environment.update({key: str(path) for key, path in directories.items()})
    environment.update(
        TLDW_CONFIG_PATH=str(selector),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(workspace),
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TRANSFORMERS_OFFLINE="1",
    )
    return environment


def _redact(text: str) -> str:
    """Remove fixed synthetic credentials and password-shaped output fields."""
    for value in _SYNTHETIC_CREDENTIALS:
        text = text.replace(value, "[REDACTED_SYNTHETIC_CREDENTIAL]")
    return re.sub(
        r"(?i)(password(?:_confirm)?\s*[=:]\s*)[^\s,;\]\}]+",
        r"\1[REDACTED]",
        text,
    )


def _sanitize_file(source: Path, destination: Path) -> None:
    """Copy a UTF-8 evidence file while applying the fixed redaction policy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _redact(source.read_text(encoding="utf-8", errors="replace")),
        encoding="utf-8",
    )


def _junit_result(path: Path) -> dict[str, object]:
    """Read counts and skip identities from one JUnit document."""
    document = ET.parse(path)
    cases = [node for node in document.iter() if node.tag.rsplit("}", 1)[-1] == "testcase"]
    skipped = []
    failed = []
    for case in cases:
        identity = "::".join(
            filter(None, (case.attrib.get("classname"), case.attrib.get("name")))
        )
        child_tags = {child.tag.rsplit("}", 1)[-1] for child in case}
        if "skipped" in child_tags:
            skipped.append(identity)
        if child_tags & {"failure", "error"}:
            failed.append(identity)
    return {"collected": len(cases), "skipped": skipped, "failed": failed}


def _installed_receipts(private_root: Path) -> list[dict[str, object]]:
    """Hash every file in each wheel installation built by product tests."""
    results = []
    resolved_private = private_root.resolve()
    for source_receipt in sorted(private_root.rglob("native-package.json")):
        source = json.loads(source_receipt.read_text(encoding="utf-8"))
        installed = Path(source["installed"]).resolve()
        if not installed.is_relative_to(resolved_private):
            raise RuntimeError("installed package escaped the private evidence root")
        files = {
            str(path.relative_to(installed)).replace("\\", "/"): _sha256(path)
            for path in sorted(installed.rglob("*"))
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
        }
        results.append(
            {
                "fixture_receipt": str(source_receipt.relative_to(private_root)).replace("\\", "/"),
                "wheel_sha256": source["sha256"],
                "installed_files": files,
            }
        )
    return results


def _collect_safe_logs(private_root: Path, artifacts: Path) -> int:
    """Collect only text logs; configs, archives and fixture payloads stay private."""
    count = 0
    for source in sorted((private_root / "pytest").rglob("*.log")):
        if source.is_symlink() or not source.is_file():
            continue
        relative = source.relative_to(private_root / "pytest")
        _sanitize_file(source, artifacts / "test-logs" / relative)
        count += 1
    return count


def _artifact_hashes(artifacts: Path) -> dict[str, str]:
    """Hash final artifact files without including the hash receipt itself."""
    return {
        str(path.relative_to(artifacts)).replace("\\", "/"): _sha256(path)
        for path in sorted(artifacts.rglob("*"))
        if path.is_file() and path.name != "artifact-sha256.json"
    }


def run(workspace: Path, evidence_root: Path) -> int:
    """Execute the finite qualification and retain safe failure evidence."""
    workspace = workspace.resolve()
    evidence_root = evidence_root.resolve()
    private_root = evidence_root / "private"
    artifacts = evidence_root / "artifacts"
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    artifacts.mkdir(parents=True, exist_ok=True, mode=0o700)

    _write_json(artifacts / "source-receipt.json", _source_receipt(workspace))
    _write_json(artifacts / "native-identity.json", _native_identity(private_root))
    environment = _private_environment(workspace, private_root)
    raw_log = private_root / "pytest-output.log"
    raw_junit = private_root / "pytest.xml"
    bootstrap = (
        "from Tests.network_guard import install; install(); "
        "import keyring; from keyring.backends.null import Keyring; "
        "keyring.set_keyring(Keyring()); import pytest, sys; "
        "raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    command = [
        sys.executable,
        "-c",
        bootstrap,
        *_PRODUCT_TESTS,
        "-vv",
        "--tb=long",
        "--timeout=300",
        f"--basetemp={private_root / 'pytest'}",
        f"--junitxml={raw_junit}",
    ]
    with raw_log.open("w", encoding="utf-8") as output:
        try:
            completed = subprocess.run(  # nosec B603 - fixed interpreter/tests
                command,
                cwd=workspace,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=80 * 60,
            )
            pytest_returncode = completed.returncode
        except subprocess.TimeoutExpired:
            output.write("\nPLATFORM PRODUCT QUALIFICATION TIMED OUT\n")
            pytest_returncode = 124

    _sanitize_file(raw_log, artifacts / "pytest-output.log")
    junit = {"collected": 0, "skipped": [], "failed": [], "parse_error": None}
    if raw_junit.is_file():
        _sanitize_file(raw_junit, artifacts / "pytest.xml")
        try:
            junit.update(_junit_result(raw_junit))
        except (OSError, ET.ParseError) as error:
            junit["parse_error"] = f"{type(error).__name__}: {error}"
    else:
        junit["parse_error"] = "pytest did not produce JUnit XML"

    installed = _installed_receipts(private_root)
    _write_json(artifacts / "installed-package-receipt.json", installed)
    log_count = _collect_safe_logs(private_root, artifacts)
    effective_failure = bool(
        pytest_returncode
        or junit["parse_error"]
        or junit["skipped"]
        or junit["failed"]
        or junit["collected"] < len(_PRODUCT_TESTS)
        or not installed
    )
    summary = {
        "schema": 1,
        "tests": list(_PRODUCT_TESTS),
        "pytest_returncode": pytest_returncode,
        "effective_returncode": 1 if effective_failure else 0,
        "junit": junit,
        "installed_package_receipts": len(installed),
        "collected_test_logs": log_count,
    }
    _write_json(artifacts / "summary.json", summary)
    _write_json(artifacts / "artifact-sha256.json", _artifact_hashes(artifacts))
    return summary["effective_returncode"]


def main(arguments: Iterable[str] | None = None) -> int:
    """Parse command-line arguments and run the native qualification."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    options = parser.parse_args(arguments)
    return run(options.workspace, options.evidence_root)


if __name__ == "__main__":
    raise SystemExit(main())
