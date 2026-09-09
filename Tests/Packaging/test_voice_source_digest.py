"""Non-self-referential source identity for speculative voice evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

from Packaging.compute_voice_source_digest import (
    VoiceSourceDigestError,
    compute_voice_source_digest,
)
from tldw_chatbook.Audio import voice_process_lifetime


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_PATH_LIST = _PROJECT_ROOT / "Packaging/speculative_voice_source_paths.txt"
_REQUIRED_CURRENT_SCOPE = {
    "native/voice_aec/vendor/webrtc/system_wrappers/source/cpu_features.cc",
    "tldw_chatbook/DB/migrations/chachanotes_v68_to_v69_console_trace_source_pin.sql",
    "Tests/Chat/test_console_runtime_lazy_voice.py",
    "native/voice_aec/patches/0002-include-stddef-for-clockdrift-detector.patch",
    "native/voice_aec/patches/0003-include-memory-for-reverb-model-estimator.patch",
    "Docs/superpowers/specs/2026-09-05-speculative-voice-provider-readiness-design.md",
    "Docs/superpowers/plans/2026-09-05-speculative-voice-provider-readiness.md",
    "Tests/Chat/test_console_voice_preflight.py",
    "Tests/UI/test_console_voice_provider_preflight.py",
    "tldw_chatbook/Chat/console_voice_preflight.py",
    "Docs/superpowers/specs/2026-09-04-speculative-voice-native-callback-design.md",
    "Docs/superpowers/plans/2026-09-04-speculative-voice-native-callback.md",
    "native/voice_aec/src/duplex_bindings.cpp",
    "native/voice_aec/src/duplex_bridge.cpp",
    "native/voice_aec/src/duplex_bridge.h",
    "Tests/Audio/fakes/native_duplex_driver.cpp",
    "Tests/Audio/fakes/native_duplex_helpers.py",
    "Tests/Audio/test_native_duplex_bridge.py",
    "Tests/Audio/test_native_duplex_stream.py",
    "Tests/Chat/test_console_speculative_voice_session.py",
    "Tests/UI/test_console_voice_native_callback.py",
    "tldw_chatbook/Audio/native_duplex_stream.py",
    "tldw_chatbook/__init__.py",
    "Docs/superpowers/plans/2026-09-02-live-physical-voice-qualification-and-isolated-headset.md",
    "Docs/superpowers/specs/2026-09-02-live-physical-voice-qualification-and-isolated-headset-design.md",
    "Packaging/assets/voice_physical_reference.json",
    "Packaging/assets/voice_physical_reference.wav",
    "Packaging/compute_voice_source_digest.py",
    "Packaging/generate_voice_qualification_manifest.py",
    "Packaging/physical_voice_runner.py",
    "Packaging/qualify_speculative_voice.py",
    "Packaging/speculative_voice_history_gate.py",
    "Packaging/speculative_voice_latency_gate.py",
    "Packaging/speculative_voice_python_paths.txt",
    "Packaging/speculative_voice_soak.py",
    "Packaging/speculative_voice_source_paths.txt",
    "Packaging/voice_physical_report.schema.json",
    "Packaging/voice_physical_reports.py",
    "Packaging/voice_qualification_manifest.schema.json",
    "scripts/qualify_physical_voice.py",
    "scripts/qualify_speculative_voice.py",
    "Tests/Chat/test_console_voice_ephemerality.py",
    "Tests/Audio/test_acoustic_isolation.py",
    "Tests/Chat/test_console_exchange_capture.py",
    "Tests/Chat/test_console_trace_call_lifecycle.py",
    "Tests/DB/test_chachanotes_v56_semantic_trace_migration.py",
    "Tests/DB/test_chachanotes_v57_semantic_mutation_guard_migration.py",
    "Tests/UI/test_console_runtime_ownership.py",
    "Tests/Packaging/test_qualify_speculative_voice.py",
    "Tests/Packaging/test_physical_voice_runner.py",
    "Tests/Packaging/test_speculative_voice_lint_scope.py",
    "Tests/Packaging/test_speculative_voice_soak.py",
    "Tests/Packaging/test_voice_physical_reports.py",
    "Tests/Packaging/test_voice_qualification_manifest.py",
    "Tests/Packaging/test_voice_source_digest.py",
    "Tests/Performance/test_speculative_voice_latency.py",
    "Tests/Packaging/fixtures/speculative_voice_physical/safe_isolated_full_duplex.json",
    "tldw_chatbook/Chat/console_prepared_request.py",
    "tldw_chatbook/Chat/console_speculative_voice_session.py",
    "tldw_chatbook/Chat/console_turn_preparation.py",
    "tldw_chatbook/TTS/TTS_Generation.py",
    "tldw_chatbook/Widgets/Console/console_capture_policy_dialog.py",
    "tldw_chatbook/Audio/acoustic_isolation.py",
}
_VOICE_PROCESS_RUNTIME_SOURCES = (
    "__init__.py",
    "Audio/__init__.py",
    "Audio/acoustic_isolation.py",
    "Audio/aec_backend.py",
    "Audio/duplex_contracts.py",
    "Audio/duplex_transport.py",
    "Audio/native_duplex_stream.py",
    "Audio/parakeet_voice_worker.py",
    "Audio/rolling_transcript.py",
    "Audio/voice_phrase_sequencer.py",
    "Audio/voice_preprocessor.py",
    "Audio/voice_process_core.py",
    "Audio/voice_process_entry.py",
    "Audio/voice_process_io.py",
    "Audio/voice_process_lifetime.py",
    "Audio/voice_process_protocol.py",
    "Audio/voice_process_types.py",
    "Audio/voice_transcription.py",
    "Audio/voice_turn_coordinator.py",
    "Chat/__init__.py",
    "Chat/console_provider_gateway.py",
    "Chat/console_speculative_voice.py",
    "Chat/console_speculative_voice_session.py",
    "Chat/console_turn_context.py",
    "Chat/console_voice_attempts.py",
    "Chat/console_voice_eligibility.py",
    "Chat/console_voice_preflight.py",
    "Chat/console_voice_process.py",
    "Chat/console_voice_process_effects.py",
    "Chat/console_voice_promotion.py",
    "Chat/console_voice_supervisor.py",
    "Chat/console_voice_trace_gateway.py",
    "Chat/console_voice_trace_promotion.py",
    "Chat/console_voice_tts_bridge.py",
    "Chat/console_voice_worker.py",
    "Chat/voice_phrase_sequencer.py",
    "Local_Ingestion/__init__.py",
    "Local_Ingestion/transcription_service.py",
    "STT/__init__.py",
    "STT/contracts.py",
    "STT/executor.py",
    "STT/executor_process_tree.py",
    "STT/executor_worker.py",
    "STT/parakeet_dispatch.py",
    "STT/parakeet_onnx.py",
    "TTS/__init__.py",
    "TTS/adapter_types.py",
    "TTS/audio_cpp_contract.py",
    "TTS/pcm_stream.py",
    "Utils/__init__.py",
    "Utils/fd_protection.py",
    "Utils/local_stt_providers.py",
    "Utils/persistent_diagnostics.py",
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "--quiet")
    (root / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "corpus.json").write_text("{}\n", encoding="utf-8")
    paths = root / "paths.txt"
    paths.write_text("paths.txt\nruntime.py\ncorpus.json\n", encoding="utf-8")
    _git(root, "add", "runtime.py", "corpus.json", "paths.txt")
    _git(
        root,
        "-c",
        "user.name=Voice Test",
        "-c",
        "user.email=voice-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    return root, paths


def _expected(root: Path, paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            hashlib.sha256((root / relative).read_bytes()).hexdigest().encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _listed_project_paths() -> set[str]:
    return {
        line.strip()
        for line in _PROJECT_PATH_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _listed_project_path_lines() -> list[str]:
    return [
        line.strip()
        for line in _PROJECT_PATH_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _copied_project_source_repository(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project-source"
    paths = root / "Packaging/speculative_voice_source_paths.txt"
    for relative in _listed_project_paths():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_PROJECT_ROOT / relative, destination)
    _git(root, "init", "--quiet")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Voice Test",
        "-c",
        "user.email=voice-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "copy project source",
    )
    return root, paths


def test_project_source_list_names_only_present_regular_files() -> None:
    listed = _listed_project_paths()
    runtime_sources = {
        f"tldw_chatbook/{relative}" for relative in _VOICE_PROCESS_RUNTIME_SOURCES
    }

    assert _listed_project_path_lines() == sorted(listed)
    assert _REQUIRED_CURRENT_SCOPE <= listed
    assert (
        voice_process_lifetime._RUNTIME_SOURCE_PATHS == _VOICE_PROCESS_RUNTIME_SOURCES
    )
    assert runtime_sources <= listed
    assert not {
        relative
        for relative in listed
        if not (_PROJECT_ROOT / relative).is_file()
        or (_PROJECT_ROOT / relative).is_symlink()
    }


@pytest.mark.parametrize("relative", _VOICE_PROCESS_RUNTIME_SOURCES)
def test_voice_process_identity_fingerprints_composed_runtime_source(
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    before = voice_process_lifetime.source_identity().source
    target = (_PROJECT_ROOT / "tldw_chatbook" / relative).resolve()
    original_read_bytes = Path.read_bytes

    def changed_source(path: Path) -> bytes:
        contents = original_read_bytes(path)
        return contents + b"\n" if path.resolve() == target else contents

    monkeypatch.setattr(Path, "read_bytes", changed_source)

    assert voice_process_lifetime.source_identity().source != before


def test_digest_uses_canonical_sorted_path_nul_file_hash_lf(tmp_path: Path) -> None:
    root, paths = _repository(tmp_path)

    assert compute_voice_source_digest(root=root, path_list=paths) == _expected(
        root,
        ("paths.txt", "runtime.py", "corpus.json"),
    )


def test_source_path_authority_must_include_itself(tmp_path: Path) -> None:
    root, paths = _repository(tmp_path)
    paths.write_text("runtime.py\ncorpus.json\n", encoding="utf-8")
    _git(root, "add", "paths.txt")
    _git(
        root,
        "-c",
        "user.name=Voice Test",
        "-c",
        "user.email=voice-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "omit list",
    )

    with pytest.raises(VoiceSourceDigestError, match="must include itself"):
        compute_voice_source_digest(root=root, path_list=paths)


def test_committed_path_list_comment_changes_digest(tmp_path: Path) -> None:
    root, paths = _repository(tmp_path)
    before = compute_voice_source_digest(root=root, path_list=paths)
    paths.write_text(
        "# qualification scope\npaths.txt\nruntime.py\ncorpus.json\n",
        encoding="utf-8",
    )
    _git(root, "add", "paths.txt")
    _git(
        root,
        "-c",
        "user.name=Voice Test",
        "-c",
        "user.email=voice-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "document list",
    )

    assert compute_voice_source_digest(root=root, path_list=paths) != before


def test_committed_code_change_changes_digest(tmp_path: Path) -> None:
    root, paths = _repository(tmp_path)
    before = compute_voice_source_digest(root=root, path_list=paths)
    (root / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(root, "add", "runtime.py")
    _git(
        root,
        "-c",
        "user.name=Voice Test",
        "-c",
        "user.email=voice-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "change",
    )

    assert compute_voice_source_digest(root=root, path_list=paths) != before


@pytest.mark.parametrize(
    "relative",
    [
        "Packaging/assets/voice_physical_reference.json",
        "tldw_chatbook/Audio/acoustic_isolation.py",
    ],
)
def test_live_qualification_source_change_changes_project_digest(
    tmp_path: Path,
    relative: str,
) -> None:
    assert relative in _listed_project_paths()
    root, paths = _copied_project_source_repository(tmp_path)
    before = compute_voice_source_digest(root=root, path_list=paths)
    target = root / relative
    target.write_bytes(target.read_bytes() + b"\n")
    _git(root, "add", relative)
    _git(
        root,
        "-c",
        "user.name=Voice Test",
        "-c",
        "user.email=voice-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "mutate live qualification source",
    )

    assert compute_voice_source_digest(root=root, path_list=paths) != before


def test_unlisted_evidence_does_not_change_digest_or_require_cleanliness(
    tmp_path: Path,
) -> None:
    root, paths = _repository(tmp_path)
    before = compute_voice_source_digest(root=root, path_list=paths)
    evidence = root / "Artifacts" / "voice_qualification" / "automated"
    evidence.mkdir(parents=True)
    (evidence / "report.json").write_text('{"passed": true}\n', encoding="utf-8")

    assert compute_voice_source_digest(root=root, path_list=paths) == before


def test_regenerated_rollout_authority_does_not_change_digest_or_require_cleanliness(
    tmp_path: Path,
) -> None:
    root, paths = _repository(tmp_path)
    authority = root / "tldw_chatbook" / "Audio"
    authority.mkdir(parents=True)
    manifest = authority / "voice_qualification_manifest.json"
    build_identity = authority / "voice_build_identity.json"
    manifest.write_text('{"qualified": false}\n', encoding="utf-8")
    build_identity.write_text('{"digest": "first"}\n', encoding="utf-8")
    before = compute_voice_source_digest(root=root, path_list=paths)

    manifest.write_text('{"qualified": true}\n', encoding="utf-8")
    build_identity.write_text('{"digest": "second"}\n', encoding="utf-8")

    assert compute_voice_source_digest(root=root, path_list=paths) == before


@pytest.mark.parametrize("state", ["dirty", "untracked", "missing"])
def test_digest_rejects_non_exact_listed_source_state(
    tmp_path: Path,
    state: str,
) -> None:
    root, paths = _repository(tmp_path)
    if state == "dirty":
        (root / "runtime.py").write_text("VALUE = 3\n", encoding="utf-8")
    elif state == "untracked":
        (root / "new.py").write_text("VALUE = 4\n", encoding="utf-8")
        paths.write_text(
            "paths.txt\nruntime.py\ncorpus.json\nnew.py\n", encoding="utf-8"
        )
        _git(root, "add", "paths.txt")
        _git(
            root,
            "-c",
            "user.name=Voice Test",
            "-c",
            "user.email=voice-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "list untracked",
        )
    else:
        paths.write_text(
            "paths.txt\nruntime.py\ncorpus.json\nmissing.py\n", encoding="utf-8"
        )
        _git(root, "add", "paths.txt")
        _git(
            root,
            "-c",
            "user.name=Voice Test",
            "-c",
            "user.email=voice-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "list missing",
        )

    with pytest.raises(VoiceSourceDigestError, match=state):
        compute_voice_source_digest(root=root, path_list=paths)


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_digest_rejects_hidden_git_index_flags(
    tmp_path: Path,
    index_flag: str,
) -> None:
    root, paths = _repository(tmp_path)
    _git(root, "update-index", index_flag, "runtime.py")

    with pytest.raises(VoiceSourceDigestError, match="hidden Git index flag"):
        compute_voice_source_digest(root=root, path_list=paths)


@pytest.mark.parametrize(
    "forbidden",
    [
        "Artifacts/voice_qualification/automated/report.json",
        "tldw_chatbook/Audio/voice_qualification_manifest.json",
        "tldw_chatbook/Audio/voice_build_identity.json",
        ".git/HEAD",
    ],
)
def test_digest_rejects_evidence_or_authority_paths(
    tmp_path: Path,
    forbidden: str,
) -> None:
    root, paths = _repository(tmp_path)
    paths.write_text(f"paths.txt\nruntime.py\n{forbidden}\n", encoding="utf-8")

    with pytest.raises(VoiceSourceDigestError, match="excluded"):
        compute_voice_source_digest(root=root, path_list=paths)


@pytest.mark.parametrize("unsafe", ["C:/escape.py", r"runtime\\escape.py"])
def test_digest_rejects_nonportable_or_windows_absolute_paths(
    tmp_path: Path,
    unsafe: str,
) -> None:
    root, paths = _repository(tmp_path)
    paths.write_text(f"paths.txt\n{unsafe}\n", encoding="utf-8")

    with pytest.raises(VoiceSourceDigestError, match="invalid source path"):
        compute_voice_source_digest(root=root, path_list=paths)
