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
import stat
import struct
import subprocess  # nosec B404 - fixed local commands and arguments only
import sys
from collections.abc import Iterable
from pathlib import Path

from defusedxml import ElementTree as ET

_NATIVE_TESTS = ("Tests/Utils/test_windows_files.py",)
_PRODUCT_TESTS = (
    "Tests/DB/test_private_sqlite_windows_descriptor.py",
    (
        "Tests/Backup_Recovery/test_complete_roundtrip.py::"
        "test_two_captured_profiles_restore_and_open_with_native_content"
    ),
    (
        "Tests/Backup_Recovery/test_f9_replacement_workflow.py::"
        "test_full_f9_replacement_after_explicit_safety_and_credential_review"
    ),
    (
        "Tests/Backup_Recovery/test_later_rollback_credential_ui.py::"
        "test_f9_later_rollback_requires_explicit_credential_review"
    ),
    (
        "Tests/Backup_Recovery/test_projection_dependency_lock.py::"
        "test_dependency_union_retains_same_private_regular_lock"
    ),
    (
        "Tests/Backup_Recovery/test_projection_dependency_lock.py::"
        "test_independent_process_refuses_contended_lock_before_publication"
    ),
    (
        "Tests/Backup_Recovery/test_projection_dependency_lock.py::"
        "test_unsafe_existing_lock_is_refused_without_publication"
    ),
    (
        "Tests/Backup_Recovery/test_projection_dependency_lock.py::"
        "test_failed_publication_releases_lock_without_removing_stable_name"
    ),
    (
        "Tests/ProductionApp/test_backup_restore_composition.py::"
        "test_actual_mounted_backup_publishes_verified_archive_after_navigation"
    ),
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
    except Exception as error:  # noqa: BLE001 - preserve platform failure receipt
        receipt["identity_error"] = f"{type(error).__name__}: {error}"
    return receipt


def _windows_ancestor_receipt(workspace: Path, private_root: Path) -> dict[str, object]:
    """Classify native owner/DACL data without exporting SIDs or local paths."""
    receipt: dict[str, object] = {
        "schema": 1,
        "entries": [],
    }
    if os.name != "nt":
        receipt["unsupported"] = "native_windows_required"
        return receipt

    runner_home = Path.home().resolve(strict=True)
    root_candidates = {
        "workspace": workspace,
        "private_root": private_root,
        "runner_home": runner_home,
        "runner_local_temp": runner_home / "AppData" / "Local" / "Temp",
        "system_drive": Path(os.environ.get("SYSTEMDRIVE", runner_home.drive) + "\\"),
    }
    roots = {
        name: selected.resolve(strict=True)
        for name, selected in root_candidates.items()
        if selected.is_dir()
    }
    receipt["roots"] = list(roots)

    import ctypes as C

    from tldw_chatbook.Utils.windows_files import _P, WindowsOS, _native

    windows, native = WindowsOS(), _native()
    principal_aliases: dict[str, str] = {
        native.user_sid: "CURRENT_USER",
        "S-1-5-18": "LOCAL_SYSTEM",
        "S-1-5-32-544": "BUILTIN_ADMINISTRATORS",
        "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464": (
            "TRUSTED_INSTALLER"
        ),
    }

    def principal_alias(sid: str) -> str:
        if sid not in principal_aliases:
            principal_aliases[sid] = f"UNRECOGNIZED_{len(principal_aliases) - 3}"
        return principal_aliases[sid]

    receipt["current_principal"] = principal_alias(native.user_sid)
    paths: list[Path] = []
    for resolved in roots.values():
        for candidate in reversed((resolved, *resolved.parents)):
            if candidate not in paths:
                paths.append(candidate)

    entries = receipt["entries"]
    assert isinstance(entries, list)
    for path in paths:
        roles = {}
        for name, root in roots.items():
            if path == root:
                roles[name] = "self"
            elif path in root.parents:
                roles[name] = f"ancestor_{root.parents.index(path) + 1}"
        entry: dict[str, object] = {"roles": roles}
        descriptor = None
        file_descriptor = None
        try:
            file_descriptor = windows.open(path, windows.O_RDONLY | windows.O_DIRECTORY)
            handle = native.handle(file_descriptor)
            info = windows.fstat(file_descriptor)
            entry.update(
                projected_mode=stat.S_IMODE(info.st_mode),
                projected_mode_octal=oct(stat.S_IMODE(info.st_mode)),
                projected_uid=info.st_uid,
            )

            owner, dacl, descriptor = _P(), _P(), _P()
            result = native.advapi.GetSecurityInfo(
                handle,
                1,
                5,
                C.byref(owner),
                None,
                C.byref(dacl),
                None,
                C.byref(descriptor),
            )
            if result:
                raise C.WinError(result)
            entry["owner_principal"] = principal_alias(native.sid_string(owner))
            aces = []
            if dacl.value:
                count = struct.unpack_from("<H", C.string_at(dacl, 8), 4)[0]
                for index in range(count):
                    ace = _P()
                    native.check(native.advapi.GetAce(dacl, index, C.byref(ace)))
                    kind, flags, length = struct.unpack("<BBH", C.string_at(ace, 4))
                    if length < 8:
                        raise OSError("malformed_windows_acl")
                    mask = struct.unpack("<I", C.string_at(ace.value + 4, 4))[0]
                    trustee_sid = (
                        native.sid_string(ace.value + 8) if kind in {0, 1} else None
                    )
                    aces.append(
                        {
                            "type": kind,
                            "flags": flags,
                            "mask": mask,
                            "trustee_principal": (
                                principal_alias(trustee_sid) if trustee_sid else None
                            ),
                        }
                    )
            entry["aces"] = aces
        except Exception as error:  # noqa: BLE001 - retain per-ancestor outcome
            entry["error"] = {
                "type": type(error).__name__,
                "errno": getattr(error, "errno", None),
                "winerror": getattr(error, "winerror", None),
            }
        finally:
            if descriptor is not None and descriptor.value:
                native.kernel.LocalFree(descriptor)
            if file_descriptor is not None:
                windows.close(file_descriptor)
        entries.append(entry)
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
        "[first_run]\nsetup_completed=true\n"
        "[splash_screen]\nenabled=false\n",
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
    cases = [
        node for node in document.iter() if node.tag.rsplit("}", 1)[-1] == "testcase"
    ]
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


def _run_pytest_phase(
    *,
    workspace: Path,
    private_root: Path,
    artifacts: Path,
    environment: dict[str, str],
    phase: str,
    tests: tuple[str, ...],
    noconftest: bool,
    timeout_seconds: int,
) -> dict[str, object]:
    """Run one fixed pytest phase and retain its sanitized log and JUnit receipt."""
    prefix = "native-" if phase == "native" else ""
    raw_log = private_root / f"{prefix}pytest-output.log"
    raw_junit = private_root / f"{prefix}pytest.xml"
    bootstrap = (
        "from Tests.network_guard import install; install(); "
        "import keyring; from keyring.backends.null import Keyring; "
        "keyring.set_keyring(Keyring()); import pytest, sys; "
        "raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    command = [sys.executable, "-c", bootstrap]
    if noconftest:
        command.append("--noconftest")
    command.extend(
        (
            *tests,
            "-vv",
            "--tb=long",
            "--timeout=300",
            f"--basetemp={private_root / f'{phase}-pytest'}",
            f"--junitxml={raw_junit}",
        )
    )
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
                timeout=timeout_seconds,
            )
            pytest_returncode = completed.returncode
        except subprocess.TimeoutExpired:
            output.write(f"\n{phase.upper()} PYTEST PHASE TIMED OUT\n")
            pytest_returncode = 124

    _sanitize_file(raw_log, artifacts / raw_log.name)
    junit = {"collected": 0, "skipped": [], "failed": [], "parse_error": None}
    if raw_junit.is_file():
        _sanitize_file(raw_junit, artifacts / raw_junit.name)
        try:
            junit.update(_junit_result(raw_junit))
        except (OSError, ET.ParseError) as error:
            junit["parse_error"] = f"{type(error).__name__}: {error}"
    else:
        junit["parse_error"] = "pytest did not produce JUnit XML"
    return {
        "tests": list(tests),
        "pytest_returncode": pytest_returncode,
        "junit": junit,
    }


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
            if path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
        }
        results.append(
            {
                "fixture_receipt": str(
                    source_receipt.relative_to(private_root)
                ).replace("\\", "/"),
                "wheel_sha256": source["sha256"],
                "installed_files": files,
            }
        )
    return results


def _collect_safe_logs(private_root: Path, artifacts: Path) -> int:
    """Collect only text logs; configs, archives and fixture payloads stay private."""
    count = 0
    for phase in ("native-pytest", "product-pytest"):
        phase_root = private_root / phase
        for source in sorted(phase_root.rglob("*.log")):
            if source.is_symlink() or not source.is_file():
                continue
            relative = source.relative_to(phase_root)
            _sanitize_file(source, artifacts / "test-logs" / phase / relative)
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
    try:
        ancestor_receipt = _windows_ancestor_receipt(workspace, private_root)
    except Exception as error:  # noqa: BLE001 - preserve diagnostic failure
        ancestor_receipt = {
            "schema": 1,
            "error": {
                "type": type(error).__name__,
                "errno": getattr(error, "errno", None),
                "winerror": getattr(error, "winerror", None),
            },
        }
    _write_json(artifacts / "windows-ancestor-security.json", ancestor_receipt)
    _write_json(artifacts / "native-identity.json", _native_identity(private_root))
    environment = _private_environment(workspace, private_root)
    native_phase = _run_pytest_phase(
        workspace=workspace,
        private_root=private_root,
        artifacts=artifacts,
        environment=environment,
        phase="native",
        tests=_NATIVE_TESTS,
        noconftest=True,
        timeout_seconds=10 * 60,
    )
    product_phase = _run_pytest_phase(
        workspace=workspace,
        private_root=private_root,
        artifacts=artifacts,
        environment=environment,
        phase="product",
        tests=_PRODUCT_TESTS,
        noconftest=False,
        timeout_seconds=80 * 60,
    )

    installed = _installed_receipts(private_root)
    _write_json(artifacts / "installed-package-receipt.json", installed)
    log_count = _collect_safe_logs(private_root, artifacts)
    phases = {"native": native_phase, "product": product_phase}
    effective_failure = not installed
    for phase in phases.values():
        phase_junit = phase["junit"]
        assert isinstance(phase_junit, dict)
        phase_tests = phase["tests"]
        assert isinstance(phase_tests, list)
        effective_failure |= bool(
            phase["pytest_returncode"]
            or phase_junit["parse_error"]
            or phase_junit["skipped"]
            or phase_junit["failed"]
            or phase_junit["collected"] < len(phase_tests)
        )
    summary = {
        "schema": 2,
        "tests": [*_NATIVE_TESTS, *_PRODUCT_TESTS],
        "phases": phases,
        "effective_returncode": 1 if effective_failure else 0,
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
