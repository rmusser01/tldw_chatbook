"""Content-free automated qualification for speculative duplex voice."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping

from Packaging.compute_voice_source_digest import compute_voice_source_digest
from Packaging.qualify_installed_voice_aec import qualify_installed_voice_aec
from Packaging.speculative_voice_history_gate import run_completed_pair_history_gate
from Packaging.speculative_voice_latency_gate import (
    run_speculative_voice_latency_gate,
)


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PATH_LIST = _ROOT / "Packaging/speculative_voice_source_paths.txt"
_CORPUS_MANIFEST = _ROOT / "Tests/Audio/fixtures/voice_aec/manifest.json"
_OWNER_INVENTORY = (
    _ROOT / "Docs/Development/TTS/speculative-voice-durable-owner-inventory.md"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40,64}")
_OWNER_ROW = re.compile(r"^\| `([^`]+)` \|")
_FORBIDDEN_CONTENT_KEYS = {
    "audio",
    "capture_body",
    "credential",
    "device_name",
    "pcm",
    "request_body",
    "response",
    "response_text",
    "transcript",
}
_LIFECYCLE_TESTS = (
    "Tests/Chat/test_console_voice_attempts.py",
    "Tests/Chat/test_console_voice_effect_barrier.py",
    "Tests/Chat/test_console_voice_capture.py",
    "Tests/integration/test_speculative_voice_pipeline.py",
)
_DURABLE_OWNER_TEST = "Tests/Chat/test_console_voice_ephemerality.py"


class VoiceQualificationError(RuntimeError):
    """Raised when qualification evidence is invalid or cannot be produced."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_content_free(value: object, *, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise VoiceQualificationError(
                    "qualification report keys must be strings"
                )
            if key.casefold() in _FORBIDDEN_CONTENT_KEYS:
                raise VoiceQualificationError(
                    f"content-bearing qualification field is forbidden: {path}.{key}"
                )
            _validate_content_free(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_content_free(item, path=f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise VoiceQualificationError("qualification report value is not JSON-safe")


def assemble_automated_report(
    *,
    source_tree_digest: str,
    interpreter: Mapping[str, object],
    companion: Mapping[str, object],
    corpus: Mapping[str, object],
    latency_distributions: Mapping[str, object],
    completed_pair_history_gate: Mapping[str, object],
    lifecycle: Mapping[str, object],
    durable_owners: Mapping[str, object],
    git_revision: str | None = None,
) -> dict[str, object]:
    """Assemble one schema-stable report from independently passing gates."""

    if not _SHA256.fullmatch(source_tree_digest):
        raise VoiceQualificationError("source-tree digest must be lowercase SHA-256")
    if git_revision is not None and not _GIT_REVISION.fullmatch(git_revision):
        raise VoiceQualificationError("Git revision provenance is invalid")
    sections = {
        "companion": dict(companion),
        "corpus": dict(corpus),
        "latency_distributions": dict(latency_distributions),
        "completed_pair_history_gate": dict(completed_pair_history_gate),
        "lifecycle": dict(lifecycle),
        "durable_owners": dict(durable_owners),
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "scenario": "automated",
        "source_tree_digest": source_tree_digest,
        "interpreter": dict(interpreter),
        **sections,
        "passed": all(section.get("passed") is True for section in sections.values()),
    }
    if git_revision is not None:
        report["git_revision"] = git_revision
    _validate_content_free(report)
    return report


def assemble_soak_report(
    *,
    scenario: str,
    source_tree_digest: str,
    interpreter: Mapping[str, object],
    soak: Mapping[str, object],
    git_revision: str | None = None,
) -> dict[str, object]:
    """Assemble one source-bound duplex or cancellation soak report."""

    if scenario not in {"duplex-soak", "cancellation-soak"}:
        raise VoiceQualificationError("unknown soak qualification scenario")
    if not _SHA256.fullmatch(source_tree_digest):
        raise VoiceQualificationError("source-tree digest must be lowercase SHA-256")
    if git_revision is not None and not _GIT_REVISION.fullmatch(git_revision):
        raise VoiceQualificationError("Git revision provenance is invalid")
    soak_section = dict(soak)
    report: dict[str, object] = {
        "schema_version": 1,
        "scenario": scenario,
        "source_tree_digest": source_tree_digest,
        "interpreter": dict(interpreter),
        "soak": soak_section,
        "passed": soak_section.get("passed") is True,
    }
    if git_revision is not None:
        report["git_revision"] = git_revision
    _validate_content_free(report)
    return report


def _wheel_digest_from_direct_url(distribution: Any) -> str:
    try:
        raw = distribution.read_text("direct_url.json")
        document = json.loads(raw) if raw else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VoiceQualificationError(
            "installed companion wheel digest is unavailable"
        ) from exc
    if not isinstance(document, Mapping):
        raise VoiceQualificationError("installed companion wheel digest is unavailable")
    archive = document.get("archive_info")
    if not isinstance(archive, Mapping):
        raise VoiceQualificationError("installed companion wheel digest is unavailable")
    hashes = archive.get("hashes")
    digest = hashes.get("sha256") if isinstance(hashes, Mapping) else None
    if digest is None:
        legacy = archive.get("hash")
        if isinstance(legacy, str) and legacy.startswith("sha256="):
            digest = legacy.removeprefix("sha256=")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise VoiceQualificationError("installed companion wheel digest is unavailable")
    return digest


def installed_companion_identity(
    *,
    aec_report: Mapping[str, object],
    wheel_path: Path | None = None,
    distribution_reader: Callable[[str], Any] = metadata.distribution,
) -> dict[str, object]:
    """Read version/provenance from the installed wheel and bind its exact bytes."""

    try:
        distribution = distribution_reader("tldw-voice-aec")
        installed_version = distribution.version
        provenance_path = Path(
            distribution.locate_file("tldw_voice_aec/provenance/UPSTREAM.json")
        )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VoiceQualificationError(
            "installed companion provenance is unavailable"
        ) from exc
    report_version = aec_report.get("companion_version")
    extension_digest = aec_report.get("extension_sha256")
    upstream_commit = (
        provenance.get("commit") if isinstance(provenance, Mapping) else None
    )
    if (
        not isinstance(installed_version, str)
        or installed_version != report_version
        or not isinstance(extension_digest, str)
        or not _SHA256.fullmatch(extension_digest)
        or not isinstance(upstream_commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", upstream_commit)
    ):
        raise VoiceQualificationError("installed companion identity is invalid")

    if wheel_path is None:
        wheel_digest = _wheel_digest_from_direct_url(distribution)
    else:
        wheel = Path(wheel_path)
        if not wheel.is_file() or wheel.is_symlink() or wheel.suffix != ".whl":
            raise VoiceQualificationError("companion wheel path is invalid")
        wheel_digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    return {
        "version": installed_version,
        "upstream_commit": upstream_commit,
        "wheel_sha256": wheel_digest,
        "extension_sha256": extension_digest,
        "passed": True,
    }


def _interpreter_identity() -> dict[str, object]:
    return {
        "implementation": platform.python_implementation().casefold(),
        "version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


def _git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    return (
        revision
        if result.returncode == 0 and _GIT_REVISION.fullmatch(revision)
        else None
    )


def _pytest_gate(
    root: Path,
    tests: tuple[str, ...],
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, object]:
    suite = {"tests": list(tests)}
    try:
        result = runner(
            (sys.executable, "-m", "pytest", "-q", *tests),
            cwd=root,
            capture_output=True,
            check=False,
            timeout=900,
        )
        exit_code = int(result.returncode)
        error_class = None
    except Exception as exc:
        exit_code = -1
        error_class = type(exc).__name__
    report: dict[str, object] = {
        "suite_sha256": _canonical_sha256(suite),
        "test_path_count": len(tests),
        "exit_code": exit_code,
        "passed": exit_code == 0,
    }
    if error_class is not None:
        report["error_class"] = error_class
    return report


def _durable_owner_report(
    root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, object]:
    inventory_path = root / _OWNER_INVENTORY.relative_to(_ROOT)
    inventory_bytes = inventory_path.read_bytes()
    owner_ids = sorted(
        match.group(1)
        for line in inventory_bytes.decode("utf-8").splitlines()
        if (match := _OWNER_ROW.match(line)) is not None
    )
    probe = _pytest_gate(root, (_DURABLE_OWNER_TEST,), runner=runner)
    core = {
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "owner_ids_sha256": _canonical_sha256(owner_ids),
        "owner_count": len(owner_ids),
        "probe": probe,
    }
    return {
        **core,
        "report_sha256": _canonical_sha256(core),
        "passed": len(owner_ids) == 14 and probe["passed"] is True,
    }


def run_automated_qualification(
    *,
    source_tree_digest: str,
    root: Path = _ROOT,
    wheel_path: Path | None = None,
    pytest_runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, object]:
    """Execute all local automated gates against one clean source identity."""

    repository = Path(root).resolve()
    actual_digest = compute_voice_source_digest(
        root=repository,
        path_list=repository / _SOURCE_PATH_LIST.relative_to(_ROOT),
    )
    if actual_digest != source_tree_digest:
        raise VoiceQualificationError("source-tree digest does not match listed source")

    aec_report = qualify_installed_voice_aec(
        repository / _CORPUS_MANIFEST.relative_to(_ROOT)
    )
    companion = installed_companion_identity(
        aec_report=aec_report,
        wheel_path=wheel_path,
    )
    corpus = {
        key: value
        for key, value in aec_report.items()
        if key not in {"companion_version", "extension_sha256"}
    }
    latency = run_speculative_voice_latency_gate()
    with tempfile.TemporaryDirectory(prefix="tldw-voice-history-") as directory:
        history = run_completed_pair_history_gate(Path(directory))
    lifecycle = _pytest_gate(repository, _LIFECYCLE_TESTS, runner=pytest_runner)
    durable_owners = _durable_owner_report(repository, runner=pytest_runner)
    return assemble_automated_report(
        source_tree_digest=source_tree_digest,
        git_revision=_git_revision(repository),
        interpreter=_interpreter_identity(),
        companion=companion,
        corpus=corpus,
        latency_distributions=latency,
        completed_pair_history_gate=history,
        lifecycle=lifecycle,
        durable_owners=durable_owners,
    )


def run_soak_qualification(
    *,
    scenario: str,
    minutes: int,
    root: Path = _ROOT,
    expected_source_tree_digest: str | None = None,
) -> dict[str, object]:
    """Run one wall-clock soak and bind it to the exact clean source tree."""

    if scenario not in {"duplex-soak", "cancellation-soak"}:
        raise VoiceQualificationError("unknown soak qualification scenario")
    if type(minutes) is not int or not 1 <= minutes <= 240:
        raise VoiceQualificationError("soak minutes must be between 1 and 240")
    repository = Path(root).resolve()
    source_tree_digest = compute_voice_source_digest(
        root=repository,
        path_list=repository / _SOURCE_PATH_LIST.relative_to(_ROOT),
    )
    if (
        expected_source_tree_digest is not None
        and expected_source_tree_digest != source_tree_digest
    ):
        raise VoiceQualificationError("source-tree digest does not match listed source")

    from Packaging.speculative_voice_soak import (  # avoid a schema import cycle
        run_cancellation_soak,
        run_duplex_soak,
    )

    runner = run_duplex_soak if scenario == "duplex-soak" else run_cancellation_soak
    soak = asyncio.run(runner(duration_seconds=minutes * 60.0))
    return assemble_soak_report(
        scenario=scenario,
        source_tree_digest=source_tree_digest,
        git_revision=_git_revision(repository),
        interpreter=_interpreter_identity(),
        soak=soak,
    )


def write_qualification_report(path: Path, report: Mapping[str, object]) -> None:
    """Write canonical JSON without embedding local paths or content."""

    _validate_content_free(report)
    output = Path(path)
    if output.exists() and output.is_symlink():
        raise VoiceQualificationError("qualification output cannot be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(report) + b"\n")


def main(argv: list[str] | None = None) -> int:
    """Run one local qualification scenario and write its safe report."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=("automated", "duplex-soak", "cancellation-soak"),
        required=True,
    )
    parser.add_argument("--source-tree-digest")
    parser.add_argument("--companion-wheel", type=Path)
    parser.add_argument("--minutes", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.scenario == "automated":
            if args.source_tree_digest is None:
                parser.error("--source-tree-digest is required for automated")
            if args.minutes is not None:
                parser.error("--minutes is only valid for soak scenarios")
            report = run_automated_qualification(
                source_tree_digest=args.source_tree_digest,
                wheel_path=args.companion_wheel,
            )
        else:
            if args.minutes is None:
                parser.error("--minutes is required for soak scenarios")
            if args.companion_wheel is not None:
                parser.error("--companion-wheel is only valid for automated")
            report = run_soak_qualification(
                scenario=args.scenario,
                minutes=args.minutes,
                expected_source_tree_digest=args.source_tree_digest,
            )
        write_qualification_report(args.output, report)
    except Exception as exc:
        parser.error(f"qualification failed: {type(exc).__name__}")
    return 0 if report["passed"] is True else 1


__all__ = [
    "VoiceQualificationError",
    "assemble_automated_report",
    "assemble_soak_report",
    "installed_companion_identity",
    "main",
    "run_automated_qualification",
    "run_soak_qualification",
    "write_qualification_report",
]
