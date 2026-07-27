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


def test_contracts_import_without_runtime_or_legacy_dependencies() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys

        import tldw_chatbook.STT.contracts

        print(json.dumps(sorted(sys.modules)))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    imported_modules = set(json.loads(completed.stdout))
    unexpectedly_imported = {
        forbidden
        for forbidden in _FORBIDDEN_IMPORTS
        if any(
            module == forbidden or module.startswith(f"{forbidden}.")
            for module in imported_modules
        )
    }
    assert unexpectedly_imported == set()
