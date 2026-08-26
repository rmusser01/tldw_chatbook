#!/usr/bin/env python3
"""Provisioned Windows audio.cpp lifecycle UAT.

The harness consumes user-provided inputs in place. It never downloads,
copies, imports, or installs an audio.cpp executable.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
import logging
import platform
from pathlib import Path
import struct
import sys
import wave
from typing import Literal, Mapping


Status = Literal["pass", "fail", "partial", "inaudible"]


@dataclass(frozen=True, slots=True)
class ExpectedPackage:
    """Path-free exact identity expected from one selected package."""

    recipe_id: str
    recipe_revision: int
    package_variant: str
    model_id: str

    def as_evidence(self) -> dict[str, str | int]:
        return {
            "model_id": self.model_id,
            "package_variant": self.package_variant,
            "recipe_id": self.recipe_id,
            "recipe_revision": self.recipe_revision,
        }


@dataclass(frozen=True, slots=True)
class UATEvidence:
    """Sanitized stable evidence written for the operator and review."""

    status: Status
    objective: Literal["pass", "fail"]
    architecture: str
    python: str
    checks: tuple[str, ...]
    text_package: Mapping[str, str | int] | None
    clone_package: Mapping[str, str | int] | None
    text_wav: Mapping[str, int] | None
    clone_wav: Mapping[str, int] | None
    audible: bool | None
    cleanup: Literal["pass", "fail"]


def validate_host_contract(
    system: str,
    windows_version: tuple[int, int],
    python_version: tuple[int, int],
    architecture: str,
) -> str:
    """Return the admitted architecture or reject the unsupported host."""

    normalized = architecture.casefold()
    admitted = {
        "amd64": "x64",
        "x64": "x64",
        "x86_64": "x64",
        "x86": "x86",
        "i386": "x86",
        "i686": "x86",
    }.get(normalized)
    if (
        system.casefold() != "windows"
        or windows_version < (10, 0)
        or python_version < (3, 12)
        or admitted is None
    ):
        raise ValueError("unsupported Windows audio.cpp UAT host")
    return admitted


def validate_wav(payload: bytes) -> dict[str, int]:
    """Validate a complete PCM WAV and return bounded structural evidence."""

    try:
        with wave.open(io.BytesIO(payload), "rb") as source:
            channels = source.getnchannels()
            frames = source.getnframes()
            sample_rate = source.getframerate()
            sample_width = source.getsampwidth()
            compression = source.getcomptype()
    except (EOFError, OSError, wave.Error):
        raise ValueError("generated WAV is invalid") from None
    if (
        channels not in (1, 2)
        or frames < 1
        or not 8_000 <= sample_rate <= 192_000
        or sample_width != 2
        or compression != "NONE"
    ):
        raise ValueError("generated WAV is invalid")
    return {
        "channels": channels,
        "frames": frames,
        "sample_rate_hz": sample_rate,
    }


def compare_package(
    expected: ExpectedPackage,
    actual: Mapping[str, str | int],
) -> dict[str, str | int]:
    """Require one exact package identity without exposing its root."""

    evidence = expected.as_evidence()
    if dict(actual) != evidence:
        raise ValueError("audio.cpp package identity did not match")
    return evidence


def final_status(*, journey: Status, audible: bool, cleanup: str) -> Status:
    """Combine objective, human, and cleanup evidence conservatively."""

    if cleanup != "pass":
        return "fail"
    if journey != "pass":
        return journey
    return "pass" if audible else "inaudible"


def write_evidence(path: Path, evidence: UATEvidence) -> None:
    """Atomically write only the fixed path-free evidence schema."""

    payload = json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _host_architecture() -> str:
    version = sys.getwindowsversion()  # type: ignore[attr-defined]
    machine = "x86" if struct.calcsize("P") == 4 else platform.machine()
    return validate_host_contract(
        platform.system(),
        (version.major, version.minor),
        (sys.version_info.major, sys.version_info.minor),
        machine,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--text-package-root", type=Path, required=True)
    parser.add_argument("--clone-package-root", type=Path, required=True)
    parser.add_argument("--clone-reference-wav", type=Path, required=True)
    parser.add_argument("--clone-reference-text", required=True)
    parser.add_argument("--text-recipe-id", required=True)
    parser.add_argument("--text-recipe-revision", type=int, required=True)
    parser.add_argument("--text-package-variant", required=True)
    parser.add_argument("--text-model-id", required=True)
    parser.add_argument("--clone-recipe-id", required=True)
    parser.add_argument("--clone-recipe-revision", type=int, required=True)
    parser.add_argument("--clone-package-variant", required=True)
    parser.add_argument("--clone-model-id", required=True)
    parser.add_argument("--clone-artifact-id", required=True)
    parser.add_argument("--clone-artifact-revision", required=True)
    parser.add_argument("--clone-artifact-variant", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument(
        "--audible",
        choices=("yes", "no"),
        help="Finalize an existing objective result after human playback.",
    )
    return parser


def _expected_package(args: argparse.Namespace, prefix: str) -> ExpectedPackage:
    return ExpectedPackage(
        recipe_id=getattr(args, f"{prefix}_recipe_id"),
        recipe_revision=getattr(args, f"{prefix}_recipe_revision"),
        package_variant=getattr(args, f"{prefix}_package_variant"),
        model_id=getattr(args, f"{prefix}_model_id"),
    )


def _scan_package(root: Path, expected: ExpectedPackage):
    from tldw_chatbook.TTS.audio_cpp_package_scanner import (
        AudioCppScanOutcome,
        scan_audio_cpp_package_root,
    )

    scan = scan_audio_cpp_package_root(root)
    candidates = tuple(
        candidate
        for discovery in scan.discoveries
        for candidate in discovery.match.candidates
        if candidate.recipe.recipe_id == expected.recipe_id
    )
    if scan.outcome is not AudioCppScanOutcome.COMPLETE or len(candidates) != 1:
        raise ValueError("audio.cpp package scan was not exact")
    candidate = candidates[0]
    compare_package(
        expected,
        {
            "model_id": expected.model_id,
            "package_variant": candidate.recipe.package_variant,
            "recipe_id": candidate.recipe.recipe_id,
            "recipe_revision": candidate.recipe.recipe_revision,
        },
    )
    return candidate


async def _response_bytes(response) -> bytes:
    try:
        return b"".join([chunk async for chunk in response.byte_stream])
    finally:
        await response.aclose()


async def _production_journey(
    args: argparse.Namespace,
    *,
    architecture: str,
) -> tuple[dict[str, int], dict[str, int], tuple[str, ...]]:
    logging.disable(logging.CRITICAL)
    try:
        from loguru import logger

        logger.remove()
    except ImportError:
        pass

    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactRef,
        ArtifactRemovalAvailability,
        ModelArtifactService,
    )
    from tldw_chatbook.Model_Artifacts.store import managed_model_artifact_root
    from tldw_chatbook.TTS.adapter_types import (
        TTSRequest,
        _new_admitted_audio_cpp_clone_request,
    )
    from tldw_chatbook.TTS.adapters.audio_cpp import AudioCppAdapter
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        audio_cpp_curated_entries,
    )
    from tldw_chatbook.TTS.audio_cpp_guided_config import (
        AudioCppBackendPreference,
        AudioCppManagedArtifactIdentity,
        AudioCppManagedSetupSource,
        AudioCppSettingsConfig,
    )
    from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
    from tldw_chatbook.TTS.audio_cpp_guided_launch import _validate_binary
    from tldw_chatbook.TTS.audio_cpp_supervisor import AudioCppSupervisor
    from tldw_chatbook.TTS.profile_reference_materialization import (
        TTSCloneReferenceMaterializer,
    )
    from tldw_chatbook.TTS.profile_reference_types import (
        CanonicalTTSCloneReference,
    )

    binary = await asyncio.to_thread(
        _validate_binary,
        str(args.server_binary),
        system="windows",
        architecture=architecture,
    )
    if binary is None:
        raise ValueError("audio.cpp binary review failed")

    text_expected = _expected_package(args, "text")
    clone_expected = _expected_package(args, "clone")
    text_candidate = await asyncio.to_thread(
        _scan_package, args.text_package_root, text_expected
    )
    await asyncio.to_thread(_scan_package, args.clone_package_root, clone_expected)

    managed_reference = ArtifactRef(
        args.clone_artifact_id,
        args.clone_artifact_revision,
        args.clone_artifact_variant,
    )
    catalog_entries = tuple(
        item
        for item in audio_cpp_curated_entries()
        if item[0].reference == managed_reference
        and item[0].model_id == clone_expected.model_id
    )
    if len(catalog_entries) != 1:
        raise ValueError("audio.cpp managed package identity did not match")
    descriptor, _sources = catalog_entries[0]
    if descriptor.reference != managed_reference:
        raise ValueError("audio.cpp managed package identity did not match")

    service = ModelArtifactService(managed_model_artifact_root())
    installed = await asyncio.to_thread(
        service.install,
        descriptor,
        args.clone_package_root,
        declared_files_only=True,
    )
    if installed != managed_reference:
        raise ValueError("audio.cpp managed package identity did not match")
    root_lease = await asyncio.to_thread(
        service.acquire_installed_root, managed_reference
    )
    try:
        managed_root = root_lease.handle.paths[0][1]
    finally:
        await asyncio.to_thread(root_lease.close)
    managed_identity = AudioCppManagedArtifactIdentity(
        artifact_id=managed_reference.artifact_id,
        revision=managed_reference.revision,
        variant=managed_reference.variant,
    )
    clone_candidate = await asyncio.to_thread(
        _scan_package, managed_root, clone_expected
    )
    text_package = text_candidate.accept(public_model_id=text_expected.model_id)
    clone_package = clone_candidate.accept(
        public_model_id=clone_expected.model_id,
        managed_artifact=managed_identity,
    )

    settings = AudioCppSettingsConfig(
        mode="managed",
        managed_setup_source=AudioCppManagedSetupSource.GUIDED,
        guided_binary_path=str(binary),
        guided_packages=(text_package, clone_package),
        guided_default_model_id=text_expected.model_id,
        guided_backend_preference=AudioCppBackendPreference.CPU,
    )
    supervisor = AudioCppSupervisor()
    adapter = AudioCppAdapter(
        config=AudioCppConfig.from_mapping(settings.to_mapping()),
        supervisor=supervisor,
        guided_settings=settings,
    )
    materializer = TTSCloneReferenceMaterializer(args.runtime_root / "clone")
    checks = [
        "host",
        "binary_review_no_launch",
        "local_package_exact",
        "managed_package_exact_root",
        "guided_save_no_launch",
    ]
    if supervisor.snapshot().state != "stopped":
        raise RuntimeError("guided save started a process")

    text_path = args.runtime_root / "text.wav"
    clone_path = args.runtime_root / "clone.wav"
    try:
        text_response = await adapter.synthesize(
            TTSRequest(
                provider_id="audio_cpp",
                model_id=text_expected.model_id,
                text="Windows audio.cpp text lifecycle verification.",
                voice=None,
                response_format="wav",
            )
        )
        text_audio = await _response_bytes(text_response)
        text_wav = validate_wav(text_audio)
        text_path.write_bytes(text_audio)
        checks.extend(("generated_json", "health_catalog", "text_wav"))

        reference_audio = args.clone_reference_wav.read_bytes()
        reference_info = validate_wav(reference_audio)
        canonical_reference = CanonicalTTSCloneReference(
            wav_bytes=reference_audio,
            reference_text=args.clone_reference_text,
            sha256=sha256(reference_audio).hexdigest(),
            byte_length=len(reference_audio),
            duration_ms=max(
                1,
                round(
                    reference_info["frames"] * 1000 / reference_info["sample_rate_hz"]
                ),
            ),
            sample_rate_hz=reference_info["sample_rate_hz"],
            channels=reference_info["channels"],
            sample_encoding="pcm_s16le",
        )
        clone_request = TTSRequest(
            provider_id="audio_cpp",
            model_id=clone_expected.model_id,
            text="Windows audio.cpp clone lifecycle verification.",
            voice=None,
            response_format="wav",
        )
        capability = adapter.admit_clone_capability(clone_request)
        materialization = await materializer.materialize(canonical_reference)
        admitted = _new_admitted_audio_cpp_clone_request(
            request=clone_request,
            materialization=materialization,
            capability=capability,
            provider_revision=adapter._catalog.revision,
            applied_provider_generation=0,
        )
        clone_response = await adapter.synthesize_clone(admitted)
        clone_audio = await _response_bytes(clone_response)
        clone_wav = validate_wav(clone_audio)
        clone_path.write_bytes(clone_audio)
        checks.extend(("clone_materialization", "clone_wav"))

        availability = await asyncio.to_thread(
            service.probe_removal_availability, managed_reference
        )
        if availability is not ArtifactRemovalAvailability.BUSY:
            raise RuntimeError("live managed package removal was not blocked")
        checks.append("live_removal_blocked")

        await supervisor.stop()
        cancelled_start = asyncio.create_task(adapter._refresh_catalog(force=True))
        await asyncio.sleep(0)
        cancelled_start.cancel()
        try:
            await cancelled_start
        except asyncio.CancelledError:
            pass
        await adapter._refresh_catalog(force=True)
        if supervisor.snapshot().state != "running":
            raise RuntimeError("managed restart did not recover")
        checks.extend(("cancelled_start_settled", "restart_recovery"))

        generation = supervisor._generation
        if generation is None:
            raise RuntimeError("managed process ownership was unavailable")
        generation.owned.process.kill()
        await generation.owned.process.wait()
        for _ in range(100):
            if supervisor.snapshot().state != "running":
                break
            await asyncio.sleep(0.05)
        await adapter._refresh_catalog(force=True)
        if supervisor.snapshot().state != "running":
            raise RuntimeError("managed crash recovery did not converge")
        checks.append("crash_recovery")
    finally:
        await adapter.close()
        await materializer.close()
        await supervisor.close()
        await supervisor.wait_closed()

    availability = await asyncio.to_thread(
        service.probe_removal_availability, managed_reference
    )
    if availability is not ArtifactRemovalAvailability.AVAILABLE:
        raise RuntimeError("managed package lease remained after shutdown")
    authority = await asyncio.to_thread(
        service.acquire_removal_authority, managed_reference
    )
    try:
        await asyncio.to_thread(authority.commit)
    finally:
        await asyncio.to_thread(authority.close)
    checks.extend(("removal_after_stop", "shutdown_clean"))
    return text_wav, clone_wav, tuple(checks)


def run_journey(args: argparse.Namespace, *, architecture: str) -> UATEvidence:
    """Run the retained production lifecycle and return only bounded evidence."""

    try:
        text_wav, clone_wav, checks = asyncio.run(
            _production_journey(args, architecture=architecture)
        )
    except Exception:
        return UATEvidence(
            status="fail",
            objective="fail",
            architecture=architecture,
            python=f"{sys.version_info.major}.{sys.version_info.minor}",
            checks=(),
            text_package=None,
            clone_package=None,
            text_wav=None,
            clone_wav=None,
            audible=None,
            cleanup="fail",
        )
    return UATEvidence(
        status="partial",
        objective="pass",
        architecture=architecture,
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
        checks=checks,
        text_package=_expected_package(args, "text").as_evidence(),
        clone_package={
            **_expected_package(args, "clone").as_evidence(),
            "artifact_id": args.clone_artifact_id,
            "artifact_revision": args.clone_artifact_revision,
            "artifact_variant": args.clone_artifact_variant,
        },
        text_wav=text_wav,
        clone_wav=clone_wav,
        audible=None,
        cleanup="pass",
    )


def main(argv: list[str] | None = None) -> int:
    """Run or finalize the provisioned UAT without printing private inputs."""

    args = _parser().parse_args(argv)
    try:
        architecture = _host_architecture()
    except ValueError:
        print("Windows audio.cpp UAT: unsupported")
        return 2
    if args.audible is not None:
        current = json.loads(args.result_file.read_text(encoding="utf-8"))
        status = final_status(
            journey=current["objective"],
            audible=args.audible == "yes",
            cleanup=current["cleanup"],
        )
        current["audible"] = args.audible == "yes"
        current["status"] = status
        args.result_file.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Windows audio.cpp UAT: {status}")
        return 0 if status == "pass" else 1

    # The production journey is deliberately imported only after the native
    # host contract passes, keeping ordinary tests and unsupported hosts inert.
    evidence = run_journey(args, architecture=architecture)
    write_evidence(args.result_file, evidence)
    print(f"Windows audio.cpp UAT: {evidence.status}")
    return 0 if evidence.objective == "pass" and evidence.cleanup == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
