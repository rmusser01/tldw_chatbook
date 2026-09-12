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
import tarfile
from collections.abc import Iterable
from pathlib import Path

from defusedxml import ElementTree as ET

_NATIVE_TESTS = ("Tests/Utils/test_windows_files.py",)
_PRODUCT_TESTS = (
    "Tests/DB/test_private_sqlite_windows_descriptor.py",
    (
        "Tests/ProductionApp/test_backup_restore_end_to_end.py::"
        "test_f9_created_archive_restores_and_opens_through_actual_controls"
    ),
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
    "Tests/Backup_Recovery/test_mcp_recovery_native_files.py",
    "Tests/Backup_Recovery/test_skills_recovery_native_files.py",
    "Tests/Backup_Recovery/test_provider_recovery_native_files.py",
    "Tests/Backup_Recovery/test_admission_closed_gate.py",
    "Tests/Backup_Recovery/test_runtime_native_poll.py",
    "Tests/Backup_Recovery/test_prompt_count_admission.py",
    "Tests/Backup_Recovery/test_capture_root_order.py",
    "Tests/Backup_Recovery/test_admission_diagnostics.py",
    "Tests/Backup_Recovery/test_loop_diagnostics.py",
)
_RESTORE_DIAGNOSTIC_TESTS = (
    (
        "Tests/ProductionApp/test_backup_restore_end_to_end.py::"
        "test_f9_created_archive_restores_and_opens_through_actual_controls[plain]"
    ),
)
_PRODUCT_SELECTIONS = {
    "full": _PRODUCT_TESTS,
    "restore-diagnostic": _RESTORE_DIAGNOSTIC_TESTS,
    "plain": _RESTORE_DIAGNOSTIC_TESTS,
    "encrypted": (_PRODUCT_TESTS[1] + "[encrypted]",),
    "encrypted-credentials": (_PRODUCT_TESTS[1] + "[encrypted_credentials]",),
    "roundtrip": _PRODUCT_TESTS[2:3],
    "replacement": _PRODUCT_TESTS[3:4],
    "rollback": _PRODUCT_TESTS[4:5],
    "support": (_PRODUCT_TESTS[0], *_PRODUCT_TESTS[5:]),
}
_SYNTHETIC_CREDENTIALS = (
    "test-only-new-safety-password",
    "test-only-later-safety-password",
    "qualification-worker-password",
    "alpha-synthetic-history-api-value",
    "beta-synthetic-history-api-value",
    "synthetic-f9-secret",
    "private F9 archive passphrase",
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
        encoding="utf-8",
        timeout=30,
    )
    return completed.stdout.rstrip("\r\n")


def _tracked_files(workspace: Path) -> tuple[str, ...]:
    """Return exact tracked path names without Git's display quoting."""
    return tuple(
        relative
        for relative in _run_git(workspace, "ls-files", "-z").split("\0")
        if relative
    )


def _create_private_root(evidence_root: Path) -> Path:
    """Create the test-only private root below a platform-trusted ancestor."""
    if os.name != "nt":
        private_root = evidence_root / "private"
        private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return private_root

    from tldw_chatbook.Utils.windows_files import WindowsOS

    windows = WindowsOS()
    trusted_temp = Path.home() / "AppData" / "Local" / "Temp"
    private_root = trusted_temp / f"tldw-backup-platform-{os.getpid()}"
    windows.mkdir(private_root, 0o700)
    info = windows.stat(private_root)
    if info.st_uid != windows.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise OSError("windows_private_root_not_private")
    return private_root


def _copy_tracked_source(workspace: Path, private_root: Path) -> tuple[Path, str]:
    """Copy exact tracked HEAD bytes into the private runtime without secrets."""
    if _run_git(workspace, "status", "--porcelain=v1"):
        raise RuntimeError("source_checkout_not_clean")
    archive_path = private_root / "tracked-head.tar"
    source_copy = private_root / "source"
    if os.name == "nt":
        from tldw_chatbook.Utils.windows_files import WindowsOS

        WindowsOS().mkdir(source_copy, 0o700)
    else:
        source_copy.mkdir(mode=0o700)

    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("Git is unavailable for tracked source copy")
    subprocess.run(  # nosec B603 - fixed Git archive of the selected HEAD
        [executable, "archive", "--format=tar", "-o", str(archive_path), "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        timeout=60,
    )
    tracked = set(_tracked_files(workspace))
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        archived = {member.name.rstrip("/") for member in members if member.isfile()}
        if archived != tracked or any(
            not (member.isfile() or member.isdir()) for member in members
        ):
            raise RuntimeError("tracked_source_archive_mismatch")
        archive.extractall(source_copy, filter="data")
    copied = {
        str(path.relative_to(source_copy)).replace("\\", "/")
        for path in source_copy.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if copied != tracked:
        raise RuntimeError("tracked_source_copy_mismatch")
    return source_copy, _sha256(archive_path)


def _source_receipt(
    workspace: Path, source_copy: Path, archive_sha256: str
) -> dict[str, object]:
    """Identify and hash every tracked file in the private execution copy."""
    tracked = _tracked_files(workspace)
    files = {}
    for relative in tracked:
        candidate = source_copy / relative
        if candidate.is_file() and not candidate.is_symlink():
            files[relative] = _sha256(candidate)
    if len(files) != len(tracked):
        raise RuntimeError("source_receipt_file_count_mismatch")
    package = importlib.metadata.distribution("tldw_chatbook")
    return {
        "schema": 2,
        "git_head": _run_git(workspace, "rev-parse", "HEAD"),
        "git_status": _run_git(workspace, "status", "--porcelain=v1"),
        "distribution_version": package.version,
        "execution_source": "private_tracked_head_copy",
        "source_archive_sha256": archive_sha256,
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
    entries: list[dict[str, object]] = []
    receipt: dict[str, object] = {
        "schema": 1,
        "entries": entries,
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
        "S-1-3-4": "OWNER_RIGHTS",
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
        PYTHONIOENCODING="utf-8",
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        PYTHONUTF8="1",
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


def _sanitize_file(
    source: Path, destination: Path, *, private_root: Path | None = None
) -> None:
    """Copy a UTF-8 evidence file while applying the fixed redaction policy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = _redact(source.read_text(encoding="utf-8", errors="replace"))
    if private_root is not None:
        replacements = (
            (str(private_root), "[PRIVATE_ROOT]"),
            (str(Path.home()), "[RUNNER_HOME]"),
        )
        for path_value, replacement in replacements:
            variants = {
                path_value,
                path_value.replace("\\", "/"),
                path_value.replace("/", "\\"),
            }
            variants.update(
                value.replace("\\", "\\\\") for value in tuple(variants)
            )
            for value in sorted(variants, key=len, reverse=True):
                content = content.replace(value, replacement)
    destination.write_text(
        content,
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
            "--timeout=2400",
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

    _sanitize_file(raw_log, artifacts / raw_log.name, private_root=private_root)
    junit = {"collected": 0, "skipped": [], "failed": [], "parse_error": None}
    if raw_junit.is_file():
        _sanitize_file(raw_junit, artifacts / raw_junit.name, private_root=private_root)
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
            visible_relative = Path(
                *(
                    f"_dot_{part[1:]}" if part.startswith(".") else part
                    for part in relative.parts
                )
            )
            destination = artifacts / "test-logs" / phase / visible_relative
            if destination.exists():
                raise RuntimeError("safe_log_artifact_name_collision")
            _sanitize_file(
                source,
                destination,
                private_root=private_root,
            )
            count += 1
    return count


def _artifact_hashes(artifacts: Path) -> dict[str, str]:
    """Hash final artifact files without including the hash receipt itself."""
    return {
        str(path.relative_to(artifacts)).replace("\\", "/"): _sha256(path)
        for path in sorted(artifacts.rglob("*"))
        if path.is_file() and path.name != "artifact-sha256.json"
    }


def run(
    workspace: Path, evidence_root: Path, *, product_selection: str = "full"
) -> int:
    """Execute the finite qualification and retain safe failure evidence."""
    product_tests = _PRODUCT_SELECTIONS[product_selection]
    workspace = workspace.resolve()
    evidence_root = evidence_root.resolve()
    artifacts = evidence_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_root = _create_private_root(evidence_root)
    source_copy, archive_sha256 = _copy_tracked_source(workspace, private_root)

    _write_json(
        artifacts / "source-receipt.json",
        _source_receipt(workspace, source_copy, archive_sha256),
    )
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
    environment = _private_environment(source_copy, private_root)
    native_phase = _run_pytest_phase(
        workspace=source_copy,
        private_root=private_root,
        artifacts=artifacts,
        environment=environment,
        phase="native",
        tests=_NATIVE_TESTS,
        noconftest=True,
        timeout_seconds=10 * 60,
    )
    product_phase = _run_pytest_phase(
        workspace=source_copy,
        private_root=private_root,
        artifacts=artifacts,
        environment=environment,
        phase="product",
        tests=product_tests,
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
        phase_tests = phase["tests"]
        if not isinstance(phase_junit, dict) or not isinstance(phase_tests, list):
            raise TypeError("invalid_internal_phase_receipt")
        effective_failure |= bool(
            phase["pytest_returncode"]
            or phase_junit["parse_error"]
            or phase_junit["skipped"]
            or phase_junit["failed"]
            or phase_junit["collected"] < len(phase_tests)
        )
    summary = {
        "schema": 2,
        "product_selection": product_selection,
        "tests": [*_NATIVE_TESTS, *product_tests],
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
    parser.add_argument(
        "--product-selection",
        choices=tuple(_PRODUCT_SELECTIONS),
        default="full",
    )
    options = parser.parse_args(arguments)
    return run(
        options.workspace,
        options.evidence_root,
        product_selection=options.product_selection,
    )


if __name__ == "__main__":
    raise SystemExit(main())
