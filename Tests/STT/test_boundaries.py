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
# This expanded inventory therefore applies to imports attempted and modules
# added by STT itself, while the original boundary remains an absolute check.
_EXPANDED_FORBIDDEN_IMPORTS = (
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
    "TRANSCRIPTION_FAILURE_CONTRACT",
    "BufferAudioSource",
    "CancellationGranularity",
    "CancellationToken",
    "DeviceFailureOrigin",
    "DeviceRetryPolicy",
    "ExecutionDevice",
    "FileAudioSource",
    "InputKind",
    "LanguageInputMode",
    "PipelineCapabilities",
    "PrivacyRequirements",
    "ProducedCapabilities",
    "ProgressSink",
    "TimestampGranularity",
    "TranscriptionFailure",
    "TranscriptionFailureCode",
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


def test_contract_import_boundary_covers_application_and_optional_dependencies() -> (
    None
):
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

    assert required_prefixes <= set(_EXPANDED_FORBIDDEN_IMPORTS)


def test_contracts_import_without_runtime_or_legacy_dependencies() -> None:
    script = textwrap.dedent(
        """
        import builtins
        import importlib.util
        import json
        import sys

        import tldw_chatbook

        baseline_modules = set(sys.modules)
        attempted_imports = set()
        original_import = builtins.__import__

        def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
            absolute_name = name
            package = globals.get("__package__") if globals is not None else None
            if level and package:
                absolute_name = importlib.util.resolve_name(
                    f"{'.' * level}{name}",
                    package,
                )
            attempted_imports.add(absolute_name)
            attempted_imports.update(
                f"{absolute_name}.{requested_name}"
                for requested_name in fromlist or ()
                if requested_name != "*"
            )
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = recording_import
        try:
            # Prove preloaded direct and from-list imports remain observable.
            import rich
            from tldw_chatbook import __version__

            hook_probe_recorded = {
                "rich",
                "tldw_chatbook.__version__",
            } <= attempted_imports
            attempted_imports.clear()
            import tldw_chatbook.STT.contracts
        finally:
            builtins.__import__ = original_import

        print(
            json.dumps(
                {
                    "all": sorted(sys.modules),
                    "attempted": sorted(attempted_imports),
                    "hook_probe_recorded": hook_probe_recorded,
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
    assert imported["hook_probe_recorded"]

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

    attempted_imports = set(imported["attempted"])
    forbidden_attempts = {
        forbidden
        for forbidden in _EXPANDED_FORBIDDEN_IMPORTS
        if any(
            module == forbidden or module.startswith(f"{forbidden}.")
            for module in attempted_imports
        )
    }
    assert forbidden_attempts == set()

    incremental_modules = set(imported["incremental"])
    incrementally_imported = {
        forbidden
        for forbidden in _EXPANDED_FORBIDDEN_IMPORTS
        if any(
            module == forbidden or module.startswith(f"{forbidden}.")
            for module in incremental_modules
        )
    }
    assert incrementally_imported == set()
