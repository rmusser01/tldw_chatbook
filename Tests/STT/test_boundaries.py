from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest
import tldw_chatbook.STT as stt


_FORBIDDEN_IMPORTS = (
    "onnxruntime",
    "onnx_asr",
    "faster_whisper",
    "parakeet_mlx",
    "lightning_whisper_mlx",
    "nemo",
    "torch",
    "transformers",
    "httpx",
    "requests",
    "tldw_chatbook.Local_Ingestion.transcription_service",
)
# The application package has its own pre-existing imports, including Textual.
# This expanded inventory therefore applies to modules added by STT itself,
# while the original planned boundary above remains an absolute assertion.
_INCREMENTAL_FORBIDDEN_IMPORTS = (
    *_FORBIDDEN_IMPORTS,
    "tldw_chatbook.config",
    "tldw_chatbook.DB",
    "tldw_chatbook.Model_Artifacts",
    "textual",
    "loguru",
    "rich",
    "pydantic",
    "toml",
    "keyring",
    "aiofiles",
    "jinja2",
    "portalocker",
)
_EXPECTED_EXPORTS = (
    "MAX_BUFFER_AUDIO_BYTES",
    "BufferAudioSource",
    "CancellationGranularity",
    "CancellationToken",
    "ExecutionDevice",
    "FileAudioSource",
    "InputKind",
    "LanguageInputMode",
    "PipelineCapabilities",
    "PrivacyRequirements",
    "ProducedCapabilities",
    "ProgressSink",
    "TimestampGranularity",
    "TranscriptionPhase",
    "TranscriptionProgress",
    "TranscriptionProvenance",
    "TranscriptionRequest",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscriptionTask",
    "TranscriptionTimings",
    "TranscriptionWarningCode",
)


@pytest.fixture(autouse=True)
def isolate_test_environment() -> None:
    """Keep this import-boundary test independent of application config."""


def test_package_exports_only_deliberate_contract_values() -> None:
    assert tuple(stt.__all__) == _EXPECTED_EXPORTS
    assert all(getattr(stt, name) is not None for name in stt.__all__)


def test_contract_import_boundary_covers_application_and_optional_dependencies() -> None:
    required_prefixes = {
        "tldw_chatbook.config",
        "tldw_chatbook.DB",
        "tldw_chatbook.Model_Artifacts",
        "textual",
        "loguru",
        "rich",
        "pydantic",
        "toml",
        "keyring",
        "aiofiles",
        "jinja2",
        "portalocker",
    }

    assert required_prefixes <= set(_INCREMENTAL_FORBIDDEN_IMPORTS)


def test_contracts_import_without_runtime_or_legacy_dependencies() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys

        import tldw_chatbook

        baseline_modules = set(sys.modules)
        import tldw_chatbook.STT.contracts

        print(
            json.dumps(
                {
                    "all": sorted(sys.modules),
                    "incremental": sorted(set(sys.modules) - baseline_modules),
                }
            )
        )
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    imported = json.loads(completed.stdout)
    imported_modules = set(imported["all"])
    unexpectedly_imported = {
        forbidden
        for forbidden in _FORBIDDEN_IMPORTS
        if any(
            module == forbidden or module.startswith(f"{forbidden}.")
            for module in imported_modules
        )
    }
    assert unexpectedly_imported == set()

    incremental_modules = set(imported["incremental"])
    incrementally_imported = {
        forbidden
        for forbidden in _INCREMENTAL_FORBIDDEN_IMPORTS
        if any(
            module == forbidden or module.startswith(f"{forbidden}.")
            for module in incremental_modules
        )
    }
    assert incrementally_imported == set()
