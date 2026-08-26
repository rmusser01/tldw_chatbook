"""Package-profile contract for the cross-platform Parakeet ONNX runtime."""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PARAKEET_CPU_REQUIREMENT = "onnx-asr[cpu]==0.12.0"
PARAKEET_PROFILES = (
    "audio",
    "video",
    "media_processing",
    "transcription_parakeet",
    "transcription_parakeet_onnx",
    "all-tools",
)
ACCELERATOR_RUNTIME_MARKERS = (
    "onnx-asr[gpu]",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-openvino",
)


def _optional_dependencies() -> dict[str, list[str]]:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return pyproject["project"]["optional-dependencies"]


@pytest.mark.parametrize("profile", PARAKEET_PROFILES)
def test_parakeet_profiles_pin_the_cpu_runtime(profile: str) -> None:
    """Every supported media profile installs the same reviewed CPU runtime."""
    assert PARAKEET_CPU_REQUIREMENT in _optional_dependencies()[profile]


@pytest.mark.parametrize("profile", PARAKEET_PROFILES)
def test_parakeet_profiles_do_not_mix_accelerator_runtimes(profile: str) -> None:
    """CPU profiles must not combine mutually exclusive ONNX distributions."""
    requirements = "\n".join(_optional_dependencies()[profile]).lower()
    assert not any(marker in requirements for marker in ACCELERATOR_RUNTIME_MARKERS)
