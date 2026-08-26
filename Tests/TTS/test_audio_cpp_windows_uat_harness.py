from __future__ import annotations

import json
from argparse import Namespace
import struct
from pathlib import Path

import pytest


def _wav(*, frames: int = 160) -> bytes:
    data = b"\x00\x00" * frames
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 16_000, 32_000, 2, 16)
    body = b"WAVE" + fmt + struct.pack("<4sI", b"data", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_host_contract_accepts_only_supported_windows_tuple() -> None:
    from scripts.uat_audio_cpp_windows import validate_host_contract

    assert validate_host_contract("Windows", (10, 0), (3, 12), "AMD64") == "x64"
    assert validate_host_contract("Windows", (10, 0), (3, 12), "x86") == "x86"
    for values in (
        ("Linux", (10, 0), (3, 12), "AMD64"),
        ("Windows", (6, 3), (3, 12), "AMD64"),
        ("Windows", (10, 0), (3, 11), "AMD64"),
        ("Windows", (10, 0), (3, 12), "ARM64"),
    ):
        with pytest.raises(ValueError, match="unsupported"):
            validate_host_contract(*values)


def test_command_requires_every_provisioned_input() -> None:
    from scripts.uat_audio_cpp_windows import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args([])


def test_structural_wav_validation_is_bounded() -> None:
    from scripts.uat_audio_cpp_windows import validate_wav

    evidence = validate_wav(_wav())
    assert evidence == {
        "channels": 1,
        "frames": 160,
        "sample_rate_hz": 16_000,
    }
    with pytest.raises(ValueError, match="invalid"):
        validate_wav(b"PRIVATE_PATH_NOT_A_WAV")


def test_identity_comparison_is_exact_and_path_free() -> None:
    from scripts.uat_audio_cpp_windows import ExpectedPackage, compare_package

    expected = ExpectedPackage(
        recipe_id="recipe.one",
        recipe_revision=2,
        package_variant="variant_one",
        model_id="model-one",
    )
    actual = expected.as_evidence()
    assert compare_package(expected, actual) == actual
    with pytest.raises(ValueError, match="identity") as caught:
        compare_package(expected, {**actual, "recipe_revision": 3})
    assert "PRIVATE" not in str(caught.value)


def test_evidence_json_has_fixed_schema_and_no_private_inputs(tmp_path: Path) -> None:
    from scripts.uat_audio_cpp_windows import UATEvidence, write_evidence

    private = tmp_path / "PRIVATE_USER" / "server.exe"
    evidence = UATEvidence(
        status="partial",
        objective="pass",
        architecture="x64",
        python="3.12",
        checks=("host", "binary_review"),
        text_package=None,
        clone_package=None,
        text_wav=None,
        clone_wav=None,
        audible=None,
        cleanup="pass",
    )
    target = tmp_path / "evidence.json"
    write_evidence(target, evidence)
    payload = target.read_text(encoding="utf-8")
    assert str(private) not in payload
    assert json.loads(payload) == {
        "architecture": "x64",
        "audible": None,
        "checks": ["host", "binary_review"],
        "cleanup": "pass",
        "clone_package": None,
        "clone_wav": None,
        "objective": "pass",
        "python": "3.12",
        "status": "partial",
        "text_package": None,
        "text_wav": None,
    }


def test_final_status_requires_audible_confirmation_and_clean_teardown() -> None:
    from scripts.uat_audio_cpp_windows import final_status

    assert final_status(journey="pass", audible=True, cleanup="pass") == "pass"
    assert final_status(journey="pass", audible=False, cleanup="pass") == "inaudible"
    assert final_status(journey="pass", audible=True, cleanup="fail") == "fail"
    assert final_status(journey="partial", audible=True, cleanup="pass") == "partial"


def test_objective_journey_reports_only_bounded_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import uat_audio_cpp_windows as harness

    async def completed(_args: Namespace, *, architecture: str):
        assert architecture == "x64"
        return (
            {"channels": 1, "frames": 100, "sample_rate_hz": 16_000},
            {"channels": 1, "frames": 200, "sample_rate_hz": 16_000},
            ("host", "shutdown_clean"),
        )

    monkeypatch.setattr(harness, "_production_journey", completed)
    result = harness.run_journey(
        Namespace(
            text_recipe_id="text.recipe",
            text_recipe_revision=1,
            text_package_variant="text_variant",
            text_model_id="text-model",
            clone_recipe_id="clone.recipe",
            clone_recipe_revision=2,
            clone_package_variant="clone_variant",
            clone_model_id="clone-model",
            clone_artifact_id="clone-artifact",
            clone_artifact_revision="a" * 40,
            clone_artifact_variant="clone_variant",
        ),
        architecture="x64",
    )
    assert result.status == "partial"
    assert result.objective == "pass"
    assert result.cleanup == "pass"
    assert result.checks == ("host", "shutdown_clean")


def test_objective_journey_fails_closed_without_private_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import uat_audio_cpp_windows as harness

    async def failed(_args: Namespace, *, architecture: str):
        del architecture
        raise RuntimeError("PRIVATE_MACHINE_PATH")

    monkeypatch.setattr(harness, "_production_journey", failed)
    result = harness.run_journey(Namespace(), architecture="x86")
    assert result.status == "fail"
    assert result.objective == "fail"
    assert result.cleanup == "fail"
    assert "PRIVATE" not in repr(result)


def test_powershell_wrapper_has_no_machine_defaults_and_restores_environment() -> None:
    script = (
        Path(__file__).parents[2] / "scripts" / "uat_audio_cpp_windows.ps1"
    ).read_text(encoding="utf-8")
    assert "[Parameter(Mandatory = $true)]" in script
    assert "$env:TLDW_CONFIG_PATH" in script
    assert "$env:XDG_CONFIG_HOME" in script
    assert "$env:XDG_DATA_HOME" in script
    assert "finally" in script
    assert "Read-Host" in script
    assert "System.Media.SoundPlayer" in script
    for forbidden in ("C:\\Users\\", "Program Files", 'audiocpp_server.exe"'):
        assert forbidden not in script


def test_harness_never_copies_or_installs_the_server_executable() -> None:
    script = (
        Path(__file__).parents[2] / "scripts" / "uat_audio_cpp_windows.py"
    ).read_text(encoding="utf-8")
    assert "copy2(args.server_binary" not in script
    assert "install(args.server_binary" not in script
