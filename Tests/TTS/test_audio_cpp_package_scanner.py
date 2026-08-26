"""Bounded read-only discovery for guided audio.cpp packages."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import os
import stat
import struct
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

NEW_SCANNER_FIXTURES = {
    "qwen3_tts_1_7b_base_q8_0": (("qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf", "gguf"),),
    "omnivoice_safetensors": (
        ("config.json", "json"),
        ("tokenizer.json", "json"),
        ("tokenizer_config.json", "json"),
        ("audio_tokenizer/config.json", "json"),
        ("audio_tokenizer/preprocessor_config.json", "json"),
        ("model.safetensors", "safetensors"),
        ("audio_tokenizer/model.safetensors", "safetensors"),
    ),
    "voxcpm2_safetensors": (
        ("config.json", "json"),
        ("tokenizer_config.json", "json"),
        ("tokenizer.json", "json"),
        ("special_tokens_map.json", "json"),
        ("model.safetensors", "safetensors"),
        ("audiovae.safetensors", "safetensors"),
    ),
    "index_tts2_safetensors": (
        ("config.yaml", "other"),
        ("bpe.model", "other"),
        ("w2v-bert-2.0/config.json", "json"),
        ("w2v-bert-2.0/preprocessor_config.json", "json"),
        ("bigvgan/config.json", "json"),
        ("qwen0.6bemo4-merge/config.json", "json"),
        ("qwen0.6bemo4-merge/generation_config.json", "json"),
        ("qwen0.6bemo4-merge/tokenizer.json", "json"),
        ("qwen0.6bemo4-merge/tokenizer_config.json", "json"),
        ("qwen0.6bemo4-merge/vocab.json", "json"),
        ("qwen0.6bemo4-merge/merges.txt", "other"),
        ("gpt.safetensors", "safetensors"),
        ("s2mel.safetensors", "safetensors"),
        ("feat1.safetensors", "safetensors"),
        ("feat2.safetensors", "safetensors"),
        ("wav2vec2bert_stats.safetensors", "safetensors"),
        ("w2v-bert-2.0/model.safetensors", "safetensors"),
        ("semantic_codec_model.safetensors", "safetensors"),
        ("campplus.safetensors", "safetensors"),
        ("bigvgan/model.safetensors", "safetensors"),
        ("qwen0.6bemo4-merge/model.safetensors", "safetensors"),
    ),
    "glm_tts_q8_0": (("Text to audio (TTS)/GLM-TTS_Q8.gguf", "gguf"),),
}


def _api():
    from tldw_chatbook.TTS.audio_cpp_package_scanner import (  # noqa: F401
        AudioCppPackageScanError,
        AudioCppScanIssueCode,
        AudioCppScanLimit,
        AudioCppScanLimits,
        AudioCppScanOutcome,
        _is_reparse_or_symlink,
        scan_audio_cpp_package_root,
        scan_audio_cpp_package_root_async,
    )
    from tldw_chatbook.TTS.audio_cpp_recipes import AudioCppMatchState  # noqa: F401

    return locals()


def _write_gguf(path: Path, *, version: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<I", version))


def _write_safetensors(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", 2) + b"{}")


def _write_recipe_fixture(root: Path, package_variant: str) -> None:
    for relative_path, kind in NEW_SCANNER_FIXTURES[package_variant]:
        target = root / relative_path
        if kind == "gguf":
            _write_gguf(target)
        elif kind == "safetensors":
            _write_safetensors(target)
        elif kind == "json":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}", encoding="utf-8")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture")


def _managed_identity(**changes: str):
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        AUDIO_CPP_ARTIFACT_COMMIT,
    )
    from tldw_chatbook.TTS.audio_cpp_guided_config import (
        AudioCppManagedArtifactIdentity,
    )

    values = {
        "artifact_id": "audio-cpp-supertonic-3-orig",
        "revision": AUDIO_CPP_ARTIFACT_COMMIT,
        "variant": "orig",
    }
    values.update(changes)
    return AudioCppManagedArtifactIdentity(**values)


def _managed_root_evidence(root: Path):
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    initial = scanner.scan_audio_cpp_package_root(root)
    candidate = tuple(
        candidate
        for discovery in initial.discoveries
        for candidate in discovery.match.candidates
    )
    assert len(candidate) == 1
    return initial, candidate[0]


def test_exact_gguf_package_is_discovered_from_only_the_selected_root(
    tmp_path: Path,
) -> None:
    api = _api()
    root = tmp_path / "selected"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")

    result = api["scan_audio_cpp_package_root"](root, request_revision=7)

    assert result.outcome is api["AudioCppScanOutcome"].COMPLETE
    assert result.request_revision == 7
    assert result.visited_entries == 1
    assert len(result.discoveries) == 1
    discovery = result.discoveries[0]
    assert discovery.match.state is api["AudioCppMatchState"].EXACT
    assert discovery.match.candidates[0].recipe.package_variant == ("supertonic_3_orig")
    assert discovery.description.safe_name == "selected"


def test_nested_package_root_is_derived_without_scanning_its_parent_siblings(
    tmp_path: Path,
) -> None:
    api = _api()
    selected = tmp_path / "selected"
    selected.mkdir()
    _write_gguf(
        selected / "PocketTTS-GGUF" / "spanish" / "pocket-tts-spanish-q8_0.gguf"
    )
    unrelated = tmp_path / "outside"
    unrelated.mkdir()
    _write_gguf(unrelated / "supertonic-3-orig.gguf")

    result = api["scan_audio_cpp_package_root"](selected)

    assert len(result.discoveries) == 1
    candidate = result.discoveries[0].match.candidates[0]
    assert candidate.recipe.package_variant == "pocket_tts_spanish_q8_0"
    assert candidate.safe_name == "spanish"
    assert "outside" not in repr(result)


def test_multiple_exact_variants_in_one_root_are_preserved_as_ambiguous(
    tmp_path: Path,
) -> None:
    api = _api()
    root = tmp_path / "variants"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    _write_gguf(root / "supertonic-3-q8_0.gguf")

    result = api["scan_audio_cpp_package_root"](root)

    assert len(result.discoveries) == 1
    match = result.discoveries[0].match
    assert match.state is api["AudioCppMatchState"].AMBIGUOUS
    assert {item.recipe.package_variant for item in match.candidates} == {
        "supertonic_3_orig",
        "supertonic_3_q8_0",
    }


def test_bad_gguf_version_is_recognizable_but_never_selected(tmp_path: Path) -> None:
    api = _api()
    root = tmp_path / "bad-version"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf", version=2)

    result = api["scan_audio_cpp_package_root"](root)

    assert len(result.discoveries) == 1
    assert result.discoveries[0].match.state is api["AudioCppMatchState"].INCOMPLETE
    assert result.discoveries[0].match.candidates == ()


def test_selected_root_is_validated_before_any_filesystem_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    calls: list[str] = []

    def reject_path(*_args, **_kwargs) -> Path:
        calls.append("validate")
        raise ValueError("private validator detail")

    def reject_probe(*_args, **_kwargs):
        calls.append("lstat")
        raise OSError("private filesystem detail")

    monkeypatch.setattr(scanner, "validate_path_simple", reject_path, raising=False)
    monkeypatch.setattr(scanner.os, "lstat", reject_probe)

    with pytest.raises(
        scanner.AudioCppPackageScanError,
        match="Selected audio.cpp package root is unavailable",
    ):
        scanner.scan_audio_cpp_package_root(tmp_path / "selected")

    assert calls == ["validate"]


def test_missing_no_follow_open_support_fails_closed_without_opening_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner
    from tldw_chatbook.TTS.audio_cpp_recipes import AudioCppMatchState

    root = tmp_path / "selected"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    real_open = scanner.os.open
    opened: list[Path] = []

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):
        opened.append(Path(path))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.delattr(scanner.os, "O_NOFOLLOW", raising=False)
    monkeypatch.setattr(scanner.os, "open", recording_open)

    result = scanner.scan_audio_cpp_package_root(root)

    assert opened == []
    assert result.outcome is scanner.AudioCppScanOutcome.PARTIAL
    assert result.discoveries[0].match.state is AudioCppMatchState.INCOMPLETE
    assert result.issues == (
        scanner.AudioCppScanIssue(
            scanner.AudioCppScanIssueCode.NO_FOLLOW_UNAVAILABLE,
            "supertonic-3-orig.gguf",
        ),
    )


def test_multifile_safetensors_layout_requires_every_companion(tmp_path: Path) -> None:
    api = _api()
    root = tmp_path / "supertonic-3"
    (root / "config").mkdir(parents=True)
    (root / "config" / "tts.json").write_text("{}", encoding="utf-8")
    (root / "config" / "unicode_indexer.json").write_text("{}", encoding="utf-8")
    _write_safetensors(root / "ggml" / "supertonic.safetensors")

    complete = api["scan_audio_cpp_package_root"](root)
    (root / "config" / "unicode_indexer.json").unlink()
    incomplete = api["scan_audio_cpp_package_root"](root)

    assert complete.discoveries[0].match.state is api["AudioCppMatchState"].EXACT
    assert incomplete.discoveries[0].match.state is (
        api["AudioCppMatchState"].INCOMPLETE
    )


@pytest.mark.parametrize(
    "package_variant",
    (
        "qwen3_tts_1_7b_base_q8_0",
        "omnivoice_safetensors",
        "voxcpm2_safetensors",
        "index_tts2_safetensors",
        "glm_tts_q8_0",
    ),
)
def test_new_recipe_layouts_are_exact_through_the_bounded_scanner(
    tmp_path: Path,
    package_variant: str,
) -> None:
    api = _api()
    root = tmp_path / package_variant
    root.mkdir()
    _write_recipe_fixture(root, package_variant)

    exact = api["scan_audio_cpp_package_root"](root)
    candidate = next(
        candidate
        for discovery in exact.discoveries
        for candidate in discovery.match.candidates
        if candidate.recipe.package_variant == package_variant
    )
    (root / NEW_SCANNER_FIXTURES[package_variant][-1][0]).unlink()
    incomplete = api["scan_audio_cpp_package_root"](root)

    exact_matches = tuple(
        discovery
        for discovery in exact.discoveries
        if discovery.match.state is api["AudioCppMatchState"].EXACT
    )
    assert len(exact_matches) == 1
    assert exact_matches[0].match.candidates[0] is candidate
    assert all(
        candidate.recipe.package_variant != package_variant
        for discovery in incomplete.discoveries
        for candidate in discovery.match.candidates
    )


def test_new_qwen_extra_variant_signal_is_ambiguous_not_selected(
    tmp_path: Path,
) -> None:
    api = _api()
    root = tmp_path / "qwen-conflict"
    root.mkdir()
    _write_gguf(root / "qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf")
    _write_gguf(root / "qwen3-tts-12hz-1.7b-base-bf16.gguf")

    result = api["scan_audio_cpp_package_root"](root)

    exact = next(
        discovery.match
        for discovery in result.discoveries
        if discovery.match.state is api["AudioCppMatchState"].AMBIGUOUS
    )
    assert {candidate.recipe.package_variant for candidate in exact.candidates} == {
        "qwen3_tts_1_7b_base_q8_0",
        "qwen3_tts_1_7b_base_bf16",
    }


def test_unrelated_generic_model_gguf_is_not_classified_as_vietneu(
    tmp_path: Path,
) -> None:
    api = _api()
    root = tmp_path / "unrelated-model"
    root.mkdir()
    _write_gguf(root / "model.gguf")

    result = api["scan_audio_cpp_package_root"](root)

    assert all(
        candidate.recipe.package_variant != "vietneu_tts_v3_turbo_q8_0"
        for discovery in result.discoveries
        for candidate in discovery.match.candidates
    )


def test_entry_and_depth_limits_return_partial_instead_of_absent(
    tmp_path: Path,
) -> None:
    api = _api()
    limits_type = api["AudioCppScanLimits"]
    limit = api["AudioCppScanLimit"]
    outcome = api["AudioCppScanOutcome"]
    root = tmp_path / "limited"
    root.mkdir()
    _write_gguf(root / "a" / "supertonic-3-orig.gguf")
    _write_gguf(root / "b" / "supertonic-3-q8_0.gguf")

    entry_limited = api["scan_audio_cpp_package_root"](
        root,
        limits=limits_type(max_entries=1),
    )
    depth_limited = api["scan_audio_cpp_package_root"](
        root,
        limits=limits_type(max_depth=0),
    )

    assert entry_limited.outcome is outcome.PARTIAL
    assert limit.ENTRIES in entry_limited.limits_reached
    assert depth_limited.outcome is outcome.PARTIAL
    assert limit.DEPTH in depth_limited.limits_reached


def test_candidate_and_result_limits_are_finite_and_truthful(tmp_path: Path) -> None:
    api = _api()
    limits_type = api["AudioCppScanLimits"]
    limit = api["AudioCppScanLimit"]
    root = tmp_path / "many"
    root.mkdir()
    for index, filename in enumerate(
        ("supertonic-3-orig.gguf", "supertonic-3-q8_0.gguf")
    ):
        _write_gguf(root / str(index) / filename)

    candidate_limited = api["scan_audio_cpp_package_root"](
        root,
        limits=limits_type(max_candidate_roots=1),
    )
    result_limited = api["scan_audio_cpp_package_root"](
        root,
        limits=limits_type(max_results=1),
    )

    assert limit.CANDIDATE_ROOTS in candidate_limited.limits_reached
    assert len(candidate_limited.discoveries) <= 1
    assert limit.RESULTS in result_limited.limits_reached
    assert len(result_limited.discoveries) == 1


def test_metadata_byte_limits_block_selection_without_reading_weights(
    tmp_path: Path,
) -> None:
    api = _api()
    limits_type = api["AudioCppScanLimits"]
    limit = api["AudioCppScanLimit"]
    root = tmp_path / "metadata-limited"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")

    result = api["scan_audio_cpp_package_root"](
        root,
        limits=limits_type(
            max_metadata_bytes_per_file=4,
            max_metadata_bytes_total=4,
        ),
    )

    assert result.metadata_bytes_read <= 4
    assert limit.METADATA_PER_FILE in result.limits_reached
    assert result.discoveries[0].match.state is api["AudioCppMatchState"].INCOMPLETE


def test_total_and_individual_time_limits_use_deterministic_work_fences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    api = _api()
    limits_type = api["AudioCppScanLimits"]
    limit = api["AudioCppScanLimit"]
    root = tmp_path / "timed"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    moments = iter((0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    monkeypatch.setattr(scanner, "_monotonic", lambda: next(moments, 1.0))

    result = scanner.scan_audio_cpp_package_root(
        root,
        limits=limits_type(max_entry_seconds=0.5, max_total_seconds=100.0),
    )

    assert limit.ENTRY_TIME in result.limits_reached
    assert result.outcome is api["AudioCppScanOutcome"].PARTIAL


def test_total_time_limit_stops_before_scanning_more_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    api = _api()
    root = tmp_path / "total-time-limited"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    moments = iter((0.0, 1.0))
    monkeypatch.setattr(scanner, "_monotonic", lambda: next(moments, 1.0))

    result = scanner.scan_audio_cpp_package_root(
        root,
        limits=api["AudioCppScanLimits"](max_total_seconds=0.5),
    )

    assert result.outcome is api["AudioCppScanOutcome"].PARTIAL
    assert result.limits_reached == (api["AudioCppScanLimit"].TOTAL_TIME,)
    assert result.visited_entries == 0


def test_total_time_limit_downgrades_a_candidate_after_one_slow_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    api = _api()
    root = tmp_path / "slow-final-entry"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    moments = iter((0.0, 0.0, 0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr(scanner, "_monotonic", lambda: next(moments, 1.0))

    result = scanner.scan_audio_cpp_package_root(
        root,
        limits=api["AudioCppScanLimits"](
            max_entry_seconds=100.0,
            max_total_seconds=0.5,
        ),
    )

    assert result.outcome is api["AudioCppScanOutcome"].PARTIAL
    assert result.limits_reached == (api["AudioCppScanLimit"].TOTAL_TIME,)
    assert result.discoveries[0].match.state is api["AudioCppMatchState"].INCOMPLETE


def test_pre_cancelled_scan_publishes_no_candidate(tmp_path: Path) -> None:
    api = _api()
    root = tmp_path / "cancelled"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    cancellation = threading.Event()
    cancellation.set()

    result = api["scan_audio_cpp_package_root"](
        root,
        cancellation_event=cancellation,
    )

    assert result.outcome is api["AudioCppScanOutcome"].CANCELLED
    assert result.discoveries == ()
    assert result.visited_entries == 0


@pytest.mark.asyncio
async def test_async_scan_runs_the_sync_scanner_off_the_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "off-loop"
    root.mkdir()
    expected = scanner.scan_audio_cpp_package_root(root)
    loop_thread = threading.get_ident()
    observed: list[int] = []

    def fake_scan(*_args, **_kwargs):
        observed.append(threading.get_ident())
        return expected

    monkeypatch.setattr(scanner, "scan_audio_cpp_package_root", fake_scan)

    result = await scanner.scan_audio_cpp_package_root_async(root)

    assert result is expected
    assert observed and observed[0] != loop_thread


@pytest.mark.asyncio
async def test_cancelling_async_scan_signals_the_worker_and_drops_late_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "async-cancel"
    root.mkdir()
    started = threading.Event()
    stopped = threading.Event()
    real_scan = scanner.scan_audio_cpp_package_root

    def slow_scan(*_args, cancellation_event=None, **_kwargs):
        assert cancellation_event is not None
        started.set()
        while not cancellation_event.is_set():
            time.sleep(0.001)
        stopped.set()
        return real_scan(root)

    monkeypatch.setattr(scanner, "scan_audio_cpp_package_root", slow_scan)
    task = asyncio.create_task(scanner.scan_audio_cpp_package_root_async(root))
    await asyncio.to_thread(started.wait, 1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(stopped.wait, 1.0)


def test_local_candidate_accept_call_and_serialized_shape_stay_unchanged(
    tmp_path: Path,
) -> None:
    root = tmp_path / "local-package"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    _, candidate = _managed_root_evidence(root)

    accepted = candidate.accept()

    assert accepted.managed_artifact is None
    assert "managed_artifact" not in accepted.model_dump(mode="json")


def test_candidate_accepts_only_its_exact_managed_artifact_mapping(
    tmp_path: Path,
) -> None:
    root = tmp_path / "managed-package"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    _, candidate = _managed_root_evidence(root)
    identity = _managed_identity()

    accepted = candidate.accept(managed_artifact=identity)

    assert accepted.managed_artifact == identity


def test_repeated_managed_accepts_reuse_the_strict_offline_manifest(
    tmp_path: Path,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_artifact_catalog as catalog

    root = tmp_path / "managed-package"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    _, candidate = _managed_root_evidence(root)
    catalog._cached_audio_cpp_artifact_source_manifest.cache_clear()
    try:
        candidate.accept(managed_artifact=_managed_identity())
        candidate.accept(managed_artifact=_managed_identity())

        cache_info = catalog._cached_audio_cpp_artifact_source_manifest.cache_info()
        assert cache_info.misses == 1
        assert cache_info.hits == 1
    finally:
        catalog._cached_audio_cpp_artifact_source_manifest.cache_clear()


def _exception_graph_text(error: BaseException) -> str:
    nodes: list[BaseException] = []
    pending: BaseException | None = error
    while pending is not None and pending not in nodes:
        nodes.append(pending)
        pending = pending.__cause__ or pending.__context__
    return "\n".join(
        [*(str(node) for node in nodes), "".join(traceback.format_exception(error))]
    )


def test_candidate_manifest_failure_is_private_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_artifact_catalog as catalog

    root = tmp_path / "managed-package"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    _, candidate = _managed_root_evidence(root)
    real_loader = catalog.load_audio_cpp_artifact_source_manifest
    private_canary = "/private/managed/store/secret-manifest.json"
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(f"cannot read {private_canary}")
        return real_loader()

    catalog._cached_audio_cpp_artifact_source_manifest.cache_clear()
    monkeypatch.setattr(catalog, "load_audio_cpp_artifact_source_manifest", fail_once)
    try:
        with pytest.raises(
            ValueError,
            match="audio.cpp managed artifact does not match recipe",
        ) as raised:
            candidate.accept(managed_artifact=_managed_identity())

        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
        assert private_canary not in _exception_graph_text(raised.value)
        assert candidate.accept(managed_artifact=_managed_identity()).managed_artifact
        assert calls == 2
    finally:
        catalog._cached_audio_cpp_artifact_source_manifest.cache_clear()


def test_managed_scanner_manifest_failure_is_private_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner
    import tldw_chatbook.TTS.audio_cpp_artifact_catalog as catalog

    root = tmp_path / "managed-package"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    real_loader = catalog.load_audio_cpp_artifact_source_manifest
    private_canary = "/private/managed/store/secret-manifest.json"
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(f"cannot read {private_canary}")
        return real_loader()

    catalog._cached_audio_cpp_artifact_source_manifest.cache_clear()
    monkeypatch.setattr(catalog, "load_audio_cpp_artifact_source_manifest", fail_once)
    scan_kwargs = {
        "expected_managed_artifact": _managed_identity(),
        "expected_canonical_root": str(root.resolve()),
    }
    try:
        with pytest.raises(
            scanner.AudioCppPackageScanError,
            match="Managed audio.cpp package no longer matches its installed identity",
        ) as raised:
            scanner.scan_audio_cpp_package_root(root, **scan_kwargs)

        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
        assert private_canary not in _exception_graph_text(raised.value)
        assert scanner.scan_audio_cpp_package_root(root, **scan_kwargs).discoveries
        assert calls == 2
    finally:
        catalog._cached_audio_cpp_artifact_source_manifest.cache_clear()


@pytest.mark.parametrize(
    "control_flow",
    (KeyboardInterrupt(), SystemExit(), asyncio.CancelledError()),
)
def test_manifest_control_flow_exceptions_are_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    control_flow: BaseException,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_artifact_catalog as catalog
    from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY

    def interrupt():
        raise control_flow

    recipe = AUDIO_CPP_RECIPE_REGISTRY.for_package("supertonic_3_orig")
    identity = _managed_identity()
    catalog._cached_audio_cpp_artifact_source_manifest.cache_clear()
    monkeypatch.setattr(catalog, "load_audio_cpp_artifact_source_manifest", interrupt)
    try:
        with pytest.raises(type(control_flow)):
            catalog.audio_cpp_artifact_identity_matches_recipe(
                recipe_id=recipe.recipe_id,
                recipe_revision=recipe.recipe_revision,
                package_variant=recipe.package_variant,
                recipe_artifact_ids=recipe.model_library_artifact_ids,
                recipe_precision=recipe.precision,
                artifact_id=identity.artifact_id,
                revision=identity.revision,
                variant=identity.variant,
            )
    finally:
        catalog._cached_audio_cpp_artifact_source_manifest.cache_clear()


@pytest.mark.parametrize(
    "changes",
    (
        {"artifact_id": "audio-cpp-supertonic-3-f16"},
        {"revision": "a" * 40},
        {"variant": "f16"},
    ),
)
def test_candidate_rejects_managed_artifact_recipe_disagreement(
    tmp_path: Path,
    changes: dict[str, str],
) -> None:
    root = tmp_path / "managed-mismatch"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    _, candidate = _managed_root_evidence(root)

    with pytest.raises(ValueError, match="managed artifact does not match") as raised:
        candidate.accept(managed_artifact=_managed_identity(**changes))

    assert str(root) not in str(raised.value)


@pytest.mark.parametrize(
    "recipe_changes",
    (
        {"recipe_revision": 2},
        {"package_variant": "supertonic_3_orig_drifted"},
    ),
)
def test_candidate_rejects_managed_artifact_manifest_package_key_drift(
    tmp_path: Path,
    recipe_changes: dict[str, object],
) -> None:
    root = tmp_path / "managed-recipe-drift"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    _, candidate = _managed_root_evidence(root)
    candidate = replace(candidate, recipe=replace(candidate.recipe, **recipe_changes))

    with pytest.raises(ValueError, match="managed artifact does not match"):
        candidate.accept(managed_artifact=_managed_identity())


@pytest.mark.asyncio
async def test_managed_exact_root_contract_returns_one_exact_candidate(
    tmp_path: Path,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "managed-exact"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    expected_root = str(root.resolve())
    identity = _managed_identity()

    result = await scanner.scan_audio_cpp_package_root_async(
        root,
        expected_managed_artifact=identity,
        expected_canonical_root=expected_root,
    )
    candidates = tuple(
        candidate
        for discovery in result.discoveries
        for candidate in discovery.match.candidates
    )

    assert result.outcome is scanner.AudioCppScanOutcome.COMPLETE
    assert len(result.discoveries) == len(candidates) == 1
    assert result.canonical_root == candidates[0].canonical_root == expected_root
    assert result.canonical_root_identity == candidates[0].canonical_root_identity
    assert candidates[0].accept(managed_artifact=identity).managed_artifact == identity


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "expected_managed_artifact": object(),
            "expected_canonical_root": "/tmp/root",
        },
        {"expected_managed_artifact": None, "expected_canonical_root": "/tmp/root"},
    ),
)
def test_managed_exact_root_contract_must_be_complete_and_typed(
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "typed-contract"
    root.mkdir()

    with pytest.raises((TypeError, ValueError)):
        scanner.scan_audio_cpp_package_root(root, **kwargs)


@pytest.mark.parametrize("raised_error", (OSError, ValueError, TypeError))
@pytest.mark.parametrize("use_async", (False, True))
@pytest.mark.asyncio
async def test_hostile_managed_root_pathlike_is_normalized_without_private_context(
    tmp_path: Path,
    raised_error: type[Exception],
    use_async: bool,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    private_canary = "/private/managed/store/secret-root"

    class HostilePath:
        def __fspath__(self):
            raise raised_error(private_canary)

    root = tmp_path / "managed"
    root.mkdir()
    kwargs = {
        "expected_managed_artifact": _managed_identity(),
        "expected_canonical_root": HostilePath(),
    }

    with pytest.raises(
        TypeError,
        match="audio.cpp managed canonical root is required",
    ) as raised:
        if use_async:
            await scanner.scan_audio_cpp_package_root_async(root, **kwargs)
        else:
            scanner.scan_audio_cpp_package_root(root, **kwargs)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert private_canary not in _exception_graph_text(raised.value)


@pytest.mark.parametrize(
    "control_flow",
    (KeyboardInterrupt(), SystemExit(), asyncio.CancelledError()),
)
def test_hostile_managed_root_pathlike_preserves_control_flow(
    tmp_path: Path,
    control_flow: BaseException,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    class InterruptingPath:
        def __fspath__(self):
            raise control_flow

    root = tmp_path / "managed"
    root.mkdir()

    with pytest.raises(type(control_flow)):
        scanner.scan_audio_cpp_package_root(
            root,
            expected_managed_artifact=_managed_identity(),
            expected_canonical_root=InterruptingPath(),
        )


@pytest.mark.asyncio
async def test_managed_exact_root_contract_rejects_wrong_canonical_root(
    tmp_path: Path,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "managed-root"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    other_root = tmp_path / "other-root"
    other_root.mkdir()

    with pytest.raises(scanner.AudioCppPackageScanError):
        await scanner.scan_audio_cpp_package_root_async(
            root,
            expected_managed_artifact=_managed_identity(),
            expected_canonical_root=str(other_root.resolve()),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    (
        {"artifact_id": "audio-cpp-supertonic-3-f16"},
        {"revision": "a" * 40},
        {"variant": "f16"},
    ),
)
async def test_managed_exact_root_contract_rejects_wrong_artifact_ref_axis(
    tmp_path: Path,
    changes: dict[str, str],
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "private-managed-axis"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")

    with pytest.raises(
        scanner.AudioCppPackageScanError,
        match="Managed audio.cpp package no longer matches its installed identity",
    ) as raised:
        await scanner.scan_audio_cpp_package_root_async(
            root,
            expected_managed_artifact=_managed_identity(**changes),
            expected_canonical_root=str(root.resolve()),
        )

    assert "private-managed-axis" not in str(raised.value)


@pytest.mark.asyncio
async def test_managed_exact_root_contract_rejects_sibling_candidate(
    tmp_path: Path,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    selected = tmp_path / "selected"
    selected.mkdir()
    _write_gguf(selected / "sibling" / "supertonic-3-orig.gguf")

    with pytest.raises(scanner.AudioCppPackageScanError):
        await scanner.scan_audio_cpp_package_root_async(
            selected,
            expected_managed_artifact=_managed_identity(),
            expected_canonical_root=str(selected.resolve()),
        )


@pytest.mark.asyncio
async def test_managed_exact_root_contract_rejects_multiple_candidates(
    tmp_path: Path,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "multiple"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    _write_gguf(root / "supertonic-3-f16.gguf")

    with pytest.raises(scanner.AudioCppPackageScanError):
        await scanner.scan_audio_cpp_package_root_async(
            root,
            expected_managed_artifact=_managed_identity(),
            expected_canonical_root=str(root.resolve()),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "recipe_changes",
    (
        {"model_library_artifact_ids": ()},
        {"recipe_revision": 2},
        {"package_variant": "supertonic_3_orig_drifted"},
    ),
)
async def test_managed_exact_root_contract_rejects_recipe_mapping_drift(
    tmp_path: Path,
    recipe_changes: dict[str, object],
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner
    from tldw_chatbook.TTS.audio_cpp_recipes import (
        AUDIO_CPP_RECIPE_REGISTRY,
        AudioCppRecipeRegistry,
    )

    root = tmp_path / "recipe-drift"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    recipe = AUDIO_CPP_RECIPE_REGISTRY.for_package("supertonic_3_orig")
    drifted = replace(recipe, **recipe_changes)

    with pytest.raises(scanner.AudioCppPackageScanError):
        await scanner.scan_audio_cpp_package_root_async(
            root,
            registry=AudioCppRecipeRegistry((drifted,)),
            expected_managed_artifact=_managed_identity(),
            expected_canonical_root=str(root.resolve()),
        )


def test_managed_exact_root_contract_revalidates_root_after_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "private-substituted-root"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    expected_root = str(root.resolve())
    displaced = tmp_path / "displaced-root"
    real_scandir = scanner._scandir

    class SubstitutingIterator:
        def __init__(self) -> None:
            self._iterator = real_scandir(root)
            self._substituted = False

        def __iter__(self):
            return self

        def __next__(self):
            try:
                return next(self._iterator)
            except StopIteration:
                if not self._substituted:
                    self._iterator.close()
                    root.rename(displaced)
                    root.mkdir()
                    _write_gguf(root / "supertonic-3-orig.gguf")
                    self._substituted = True
                raise

        def close(self) -> None:
            self._iterator.close()

    monkeypatch.setattr(
        scanner,
        "_scandir",
        lambda path: (
            SubstitutingIterator() if Path(path) == root else real_scandir(path)
        ),
    )

    with pytest.raises(scanner.AudioCppPackageScanError) as raised:
        scanner.scan_audio_cpp_package_root(
            root,
            expected_managed_artifact=_managed_identity(),
            expected_canonical_root=expected_root,
        )

    assert "private-substituted-root" not in str(raised.value)


def test_managed_exact_root_contract_preserves_pre_cancellation(
    tmp_path: Path,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "managed-cancelled"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    cancellation = threading.Event()
    cancellation.set()

    result = scanner.scan_audio_cpp_package_root(
        root,
        cancellation_event=cancellation,
        expected_managed_artifact=_managed_identity(),
        expected_canonical_root=str(root.resolve()),
    )

    assert result.outcome is scanner.AudioCppScanOutcome.CANCELLED
    assert result.discoveries == ()


def test_nested_symlink_and_windows_reparse_points_are_never_traversed(
    tmp_path: Path,
) -> None:
    api = _api()
    issue = api["AudioCppScanIssueCode"]
    root = tmp_path / "selected"
    outside = tmp_path / "private-outside"
    root.mkdir()
    outside.mkdir()
    _write_gguf(outside / "supertonic-3-orig.gguf")
    (root / "escape").symlink_to(outside, target_is_directory=True)

    result = api["scan_audio_cpp_package_root"](root)

    assert result.discoveries == ()
    assert issue.SYMLINK_SKIPPED in {item.code for item in result.issues}
    reparse = SimpleNamespace(
        st_mode=stat.S_IFDIR,
        st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
    )
    assert api["_is_reparse_or_symlink"](reparse)


def test_queued_directory_replaced_by_symlink_is_not_traversed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    api = _api()
    root = tmp_path / "selected"
    queued = root / "queued"
    outside = tmp_path / "private-outside"
    queued.mkdir(parents=True)
    outside.mkdir()
    _write_gguf(outside / "supertonic-3-orig.gguf")
    real_scandir = scanner._scandir

    class ReplacingIterator:
        def __init__(self) -> None:
            self._iterator = real_scandir(root)
            self._replaced = False

        def __iter__(self):
            return self

        def __next__(self):
            try:
                return next(self._iterator)
            except StopIteration:
                if not self._replaced:
                    self._iterator.close()
                    queued.rmdir()
                    queued.symlink_to(outside, target_is_directory=True)
                    self._replaced = True
                raise

        def close(self) -> None:
            self._iterator.close()

    def replacing_scandir(path):
        return ReplacingIterator() if Path(path) == root else real_scandir(path)

    monkeypatch.setattr(scanner, "_scandir", replacing_scandir)

    result = scanner.scan_audio_cpp_package_root(root)

    assert result.discoveries == ()
    assert api["AudioCppScanIssueCode"].SOURCE_CHANGED in {
        item.code for item in result.issues
    }
    assert str(outside) not in repr(result)


def test_directory_replaced_during_scandir_open_is_not_traversed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    api = _api()
    root = tmp_path / "selected"
    queued = root / "queued"
    outside = tmp_path / "private-outside"
    queued.mkdir(parents=True)
    outside.mkdir()
    _write_gguf(outside / "supertonic-3-orig.gguf")
    real_scandir = scanner._scandir
    replaced = False

    def replacing_scandir(path):
        nonlocal replaced
        if Path(path) == queued and not replaced:
            queued.rmdir()
            queued.symlink_to(outside, target_is_directory=True)
            replaced = True
        return real_scandir(path)

    monkeypatch.setattr(scanner, "_scandir", replacing_scandir)

    result = scanner.scan_audio_cpp_package_root(root)

    assert result.discoveries == ()
    assert api["AudioCppScanIssueCode"].SOURCE_CHANGED in {
        item.code for item in result.issues
    }
    assert str(outside) not in repr(result)


def test_top_level_symlink_requires_explicit_disclosure(tmp_path: Path) -> None:
    api = _api()
    target = tmp_path / "target"
    target.mkdir()
    _write_gguf(target / "supertonic-3-orig.gguf")
    selected = tmp_path / "selected-link"
    selected.symlink_to(target, target_is_directory=True)

    with pytest.raises(api["AudioCppPackageScanError"], match="symlink"):
        api["scan_audio_cpp_package_root"](selected)
    result = api["scan_audio_cpp_package_root"](
        selected,
        allow_root_symlink=True,
    )

    assert result.root_was_symlink
    assert result.discoveries[0].match.state is api["AudioCppMatchState"].EXACT


def test_selected_root_with_symlinked_ancestor_requires_disclosure(
    tmp_path: Path,
) -> None:
    api = _api()
    target_parent = tmp_path / "target-parent"
    package = target_parent / "package"
    package.mkdir(parents=True)
    _write_gguf(package / "supertonic-3-orig.gguf")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(target_parent, target_is_directory=True)
    selected = linked_parent / "package"

    with pytest.raises(api["AudioCppPackageScanError"], match="symlink"):
        api["scan_audio_cpp_package_root"](selected)

    result = api["scan_audio_cpp_package_root"](
        selected,
        allow_root_symlink=True,
    )

    assert result.root_was_symlink
    assert result.discoveries[0].match.state is api["AudioCppMatchState"].EXACT


def test_permission_failure_is_isolated_and_uses_only_safe_bounded_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    api = _api()
    root = tmp_path / "selected-private-root"
    blocked = root / "blocked-private-child"
    root.mkdir()
    blocked.mkdir()
    real_scandir = scanner._scandir

    def permission_scandir(path):
        if Path(path).name == blocked.name:
            raise PermissionError("PRIVATE FULL PATH SHOULD NOT ESCAPE")
        return real_scandir(path)

    monkeypatch.setattr(scanner, "_scandir", permission_scandir)

    result = scanner.scan_audio_cpp_package_root(root)

    assert result.outcome is api["AudioCppScanOutcome"].PERMISSION_LIMITED
    assert len(result.issues) == 1
    assert result.issues[0].safe_name == blocked.name
    assert str(tmp_path) not in repr(result)
    assert "PRIVATE FULL PATH" not in repr(result)


def test_unrelated_permission_failure_does_not_poison_complete_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    api = _api()
    root = tmp_path / "selected"
    blocked = root / "blocked-sibling"
    blocked.mkdir(parents=True)
    _write_gguf(root / "supertonic-3-orig.gguf")
    real_scandir = scanner._scandir

    def permission_scandir(path):
        if Path(path).name == blocked.name:
            raise PermissionError("PRIVATE SIBLING PATH SHOULD NOT ESCAPE")
        return real_scandir(path)

    monkeypatch.setattr(scanner, "_scandir", permission_scandir)

    result = scanner.scan_audio_cpp_package_root(root)

    assert result.outcome is api["AudioCppScanOutcome"].PERMISSION_LIMITED
    assert result.discoveries[0].match.state is api["AudioCppMatchState"].EXACT
    assert "PRIVATE SIBLING PATH" not in repr(result)


def test_unreadable_sibling_reports_partial_without_poisoning_complete_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    api = _api()
    root = tmp_path / "selected"
    blocked = root / "unreadable-sibling"
    blocked.mkdir(parents=True)
    _write_gguf(root / "supertonic-3-orig.gguf")
    real_scandir = scanner._scandir

    def unreadable_scandir(path):
        if Path(path).name == blocked.name:
            raise OSError("PRIVATE SIBLING PATH SHOULD NOT ESCAPE")
        return real_scandir(path)

    monkeypatch.setattr(scanner, "_scandir", unreadable_scandir)

    result = scanner.scan_audio_cpp_package_root(root)

    assert result.outcome is api["AudioCppScanOutcome"].PARTIAL
    assert result.discoveries[0].match.state is api["AudioCppMatchState"].EXACT
    assert "PRIVATE SIBLING PATH" not in repr(result)


def test_zero_metadata_companion_is_opened_to_prove_readability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner
    from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY

    api = _api()
    root = tmp_path / "pocket-tts"
    recipe = AUDIO_CPP_RECIPE_REGISTRY.for_package("pocket_tts_english_safetensors")
    for signal in recipe.required_files:
        path = root / signal.relative_path
        if signal.kind.value == "safetensors":
            _write_safetensors(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"tokenizer")
    real_open = scanner.os.open

    def deny_tokenizer(path, flags, mode=0o777, *, dir_fd=None):
        if Path(path).name == "tokenizer.model":
            raise PermissionError("PRIVATE TOKENIZER PATH SHOULD NOT ESCAPE")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(scanner.os, "open", deny_tokenizer)

    result = scanner.scan_audio_cpp_package_root(root)

    discovery = result.discoveries[0]
    tokenizer = next(
        item
        for item in discovery.description.files
        if item.relative_path.endswith("tokenizer.model")
    )
    assert not tokenizer.readable
    assert discovery.match.state is api["AudioCppMatchState"].PERMISSION_LIMITED
    assert "PRIVATE TOKENIZER PATH" not in repr(result)


def test_permission_failure_during_directory_iteration_is_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    api = _api()
    root = tmp_path / "selected-private-root"
    root.mkdir()

    class FailingIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise PermissionError("PRIVATE ITERATOR PATH SHOULD NOT ESCAPE")

        def close(self) -> None:
            return None

    monkeypatch.setattr(scanner, "_scandir", lambda _path: FailingIterator())

    result = scanner.scan_audio_cpp_package_root(root)

    assert result.outcome is api["AudioCppScanOutcome"].PERMISSION_LIMITED
    assert result.issues == (
        scanner.AudioCppScanIssue(
            api["AudioCppScanIssueCode"].PERMISSION_DENIED,
            root.name,
        ),
    )
    assert str(tmp_path) not in repr(result)
    assert "PRIVATE ITERATOR PATH" not in repr(result)


def test_directory_iterator_close_failure_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    api = _api()
    root = tmp_path / "selected-private-root"
    root.mkdir()

    class CloseFailingIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self) -> None:
            raise OSError("PRIVATE CLOSE PATH SHOULD NOT ESCAPE")

    monkeypatch.setattr(scanner, "_scandir", lambda _path: CloseFailingIterator())

    result = scanner.scan_audio_cpp_package_root(root)

    assert result.outcome is api["AudioCppScanOutcome"].COMPLETE
    assert result.issues == (
        scanner.AudioCppScanIssue(
            api["AudioCppScanIssueCode"].UNREADABLE,
            root.name,
        ),
    )
    assert str(tmp_path) not in repr(result)
    assert "PRIVATE CLOSE PATH" not in repr(result)


def test_descriptor_close_failure_invalidates_file_evidence_without_escaping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    api = _api()
    root = tmp_path / "selected-private-root"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    real_close = scanner.os.close

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError("PRIVATE DESCRIPTOR PATH SHOULD NOT ESCAPE")

    monkeypatch.setattr(scanner.os, "close", close_then_fail)

    result = scanner.scan_audio_cpp_package_root(root)

    assert result.outcome is api["AudioCppScanOutcome"].PARTIAL
    assert result.discoveries[0].match.state is api["AudioCppMatchState"].INCOMPLETE
    assert result.issues == (
        scanner.AudioCppScanIssue(
            api["AudioCppScanIssueCode"].UNREADABLE,
            "supertonic-3-orig.gguf",
        ),
    )
    assert str(tmp_path) not in repr(result)
    assert "PRIVATE DESCRIPTOR PATH" not in repr(result)


def test_unknown_and_issue_detail_is_sanitized_and_capped(tmp_path: Path) -> None:
    api = _api()
    limits_type = api["AudioCppScanLimits"]
    root = tmp_path / "unknowns"
    root.mkdir()
    for index in range(5):
        (root / f"unknown-{index}\nprivate.gguf").write_bytes(b"not gguf")

    result = api["scan_audio_cpp_package_root"](
        root,
        limits=limits_type(max_unknown_names=2, max_issues=2),
    )

    assert len(result.unknown_names) == 2
    assert result.unknown_names_truncated
    assert all("\n" not in name and len(name) <= 128 for name in result.unknown_names)
    assert str(tmp_path) not in repr(result)


@pytest.mark.parametrize(
    "changes",
    (
        {"max_depth": -1},
        {"max_entries": 0},
        {"max_candidate_roots": 0},
        {"max_results": 0},
        {"max_metadata_bytes_per_file": 0},
        {"max_metadata_bytes_total": 0},
        {"max_entry_seconds": 0},
        {"max_total_seconds": 0},
        {"max_issues": 0},
        {"max_unknown_names": 0},
    ),
)
def test_all_scanner_limits_are_finite_positive_bounds(
    changes: dict[str, object],
) -> None:
    limits_type = _api()["AudioCppScanLimits"]

    with pytest.raises(ValueError):
        limits_type(**changes)


def test_guided_foundation_has_no_process_network_or_model_write_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket
    import subprocess
    import urllib.request

    import httpx
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner
    from tldw_chatbook.TTS.audio_cpp_guided_config import AudioCppSettingsConfig
    from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY

    root = tmp_path / "selected-model"
    root.mkdir()
    model_file = root / "supertonic-3-orig.gguf"
    _write_gguf(model_file)

    def snapshot() -> tuple[bytes, tuple[int, int, int, int, int]]:
        info = model_file.stat()
        return model_file.read_bytes(), (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
        )

    before = snapshot()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("guided foundation attempted a forbidden side effect")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(httpx, "Client", forbidden)
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)

    real_open = scanner.os.open
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def read_only_open(path, flags, mode=0o777, *, dir_fd=None):
        assert flags & write_flags == 0
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(scanner.os, "open", read_only_open)

    scan = scanner.scan_audio_cpp_package_root(root)
    candidate = scan.discoveries[0].match.candidates[0]
    accepted = candidate.accept()
    settings = AudioCppSettingsConfig.from_mapping(
        {
            "mode": "managed",
            "managed_setup_source": "guided",
            "guided_binary_path": "/opt/homebrew/bin/audiocpp_server",
            "guided_packages": [accepted.model_dump(mode="json")],
            "guided_default_model_id": accepted.public_model_id,
        }
    )

    recipe = AUDIO_CPP_RECIPE_REGISTRY.validate_accepted(settings.guided_packages[0])

    assert recipe.recipe_id == accepted.recipe_id
    assert before == snapshot()


class _FakeWindowsScanHandle:
    def __init__(
        self,
        path: Path,
        *,
        identity: object,
        close_failures: int = 0,
    ) -> None:
        self.path = path
        self.identity = identity
        self.close_failures = close_failures
        self.closed = False

    def read(self, count: int, *, offset: int = 0) -> bytes:
        return self.path.read_bytes()[offset : offset + count]

    def close(self) -> None:
        from tldw_chatbook.TTS.windows_artifact_fs import WindowsArtifactError

        if self.close_failures:
            self.close_failures -= 1
            raise WindowsArtifactError("cleanup_failed", cleanup_owner=self)  # type: ignore[arg-type]
        self.closed = True


class _FakeWindowsScanFilesystem:
    def __init__(self) -> None:
        self.directory_opens: list[Path] = []
        self.file_opens: list[Path] = []
        self.handles: list[_FakeWindowsScanHandle] = []
        self.before_file_open = None
        self.directory_generation: dict[Path, int] = {}
        self.close_failures = 0

    @staticmethod
    def _identity(path: Path, generation: int = 0):
        from tldw_chatbook.TTS.windows_artifact_fs import WindowsFileIdentity

        info = path.stat()
        value = (info.st_ino + generation).to_bytes(16, "little", signed=False)
        return WindowsFileIdentity(
            volume_serial_number=info.st_dev,
            file_id=value,
            kind="directory" if path.is_dir() else "file",
            reparse_tag=0,
        )

    def pin_directory_no_reparse(self, path: Path) -> _FakeWindowsScanHandle:
        selected = Path(path)
        self.directory_opens.append(selected)
        identity = self._identity(
            selected,
            self.directory_generation.get(selected, 0),
        )
        handle = _FakeWindowsScanHandle(
            selected,
            identity=identity,
            close_failures=self.close_failures,
        )
        self.close_failures = 0
        self.handles.append(handle)
        return handle

    def open_file_no_reparse(
        self, path: Path, *, writable: bool = False
    ) -> _FakeWindowsScanHandle:
        assert not writable
        selected = Path(path)
        if self.before_file_open is not None:
            self.before_file_open(selected)
        self.file_opens.append(selected)
        handle = _FakeWindowsScanHandle(selected, identity=self._identity(selected))
        self.handles.append(handle)
        return handle


def test_windows_scan_pins_selected_root_and_reads_exact_file_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "Voice 模型"
    root.mkdir()
    model = root / "supertonic-3-orig.gguf"
    _write_gguf(model)
    filesystem = _FakeWindowsScanFilesystem()
    monkeypatch.setattr(scanner, "_windows_artifact_filesystem", filesystem)

    result = scanner.scan_audio_cpp_package_root(root)

    assert result.outcome is scanner.AudioCppScanOutcome.COMPLETE
    assert result.discoveries[0].match.state.value == "exact"
    assert filesystem.directory_opens.count(root.resolve()) >= 1
    assert filesystem.file_opens == [model.resolve()]
    assert all(handle.closed for handle in filesystem.handles)
    assert scanner.AudioCppScanIssueCode.NO_FOLLOW_UNAVAILABLE not in {
        issue.code for issue in result.issues
    }


def test_windows_scan_rejects_file_substitution_before_handle_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "selected"
    root.mkdir()
    model = root / "supertonic-3-orig.gguf"
    _write_gguf(model)
    filesystem = _FakeWindowsScanFilesystem()

    def replace(path: Path) -> None:
        replacement = path.with_suffix(".replacement")
        _write_gguf(replacement)
        path.unlink()
        replacement.rename(path)
        filesystem.before_file_open = None

    filesystem.before_file_open = replace
    monkeypatch.setattr(scanner, "_windows_artifact_filesystem", filesystem)

    result = scanner.scan_audio_cpp_package_root(root)

    assert result.outcome is scanner.AudioCppScanOutcome.PARTIAL
    assert scanner.AudioCppScanIssueCode.SOURCE_CHANGED in {
        issue.code for issue in result.issues
    }
    assert not result.discoveries or result.discoveries[0].match.state.value != "exact"


def test_windows_scan_rejects_directory_identity_change_before_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "selected"
    nested = root / "nested"
    nested.mkdir(parents=True)
    _write_gguf(nested / "supertonic-3-orig.gguf")
    filesystem = _FakeWindowsScanFilesystem()
    original_pin = filesystem.pin_directory_no_reparse

    def pin(path: Path) -> _FakeWindowsScanHandle:
        selected = Path(path)
        if selected == nested and selected in filesystem.directory_opens:
            filesystem.directory_generation[selected] = 1
        return original_pin(selected)

    filesystem.pin_directory_no_reparse = pin  # type: ignore[method-assign]
    monkeypatch.setattr(scanner, "_windows_artifact_filesystem", filesystem)

    result = scanner.scan_audio_cpp_package_root(root)

    assert result.outcome is scanner.AudioCppScanOutcome.PARTIAL
    assert result.discoveries == ()
    assert scanner.AudioCppScanIssueCode.SOURCE_CHANGED in {
        issue.code for issue in result.issues
    }


def test_windows_scan_casefold_collision_is_never_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "selected"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    filesystem = _FakeWindowsScanFilesystem()
    monkeypatch.setattr(scanner, "_windows_artifact_filesystem", filesystem)
    real_entry = next(os.scandir(root))
    duplicate = SimpleNamespace(name="SUPERTONIC-3-ORIG.GGUF")
    monkeypatch.setattr(
        scanner, "_scandir", lambda _path: iter((real_entry, duplicate))
    )

    result = scanner.scan_audio_cpp_package_root(root)

    assert result.outcome is scanner.AudioCppScanOutcome.PARTIAL
    assert not result.discoveries or result.discoveries[0].match.state.value != "exact"


def test_windows_scan_close_failure_exposes_one_retry_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "selected"
    root.mkdir()
    _write_gguf(root / "supertonic-3-orig.gguf")
    filesystem = _FakeWindowsScanFilesystem()
    filesystem.close_failures = 2
    monkeypatch.setattr(scanner, "_windows_artifact_filesystem", filesystem)

    with pytest.raises(scanner.AudioCppPackageScanError) as raised:
        scanner.scan_audio_cpp_package_root(root)

    cleanup = raised.value.take_cleanup_owner()
    assert cleanup is not None
    assert raised.value.take_cleanup_owner() is None
    cleanup.close()
    assert cleanup.closed


@pytest.mark.asyncio
async def test_windows_scan_cancellation_waits_for_exact_handle_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_package_scanner as scanner

    root = tmp_path / "selected"
    root.mkdir()
    model = root / "supertonic-3-orig.gguf"
    _write_gguf(model)
    started = threading.Event()
    release = threading.Event()
    filesystem = _FakeWindowsScanFilesystem()
    original_open = filesystem.open_file_no_reparse

    def open_blocking(path: Path, *, writable: bool = False) -> _FakeWindowsScanHandle:
        handle = original_open(path, writable=writable)
        original_read = handle.read

        def read(count: int, *, offset: int = 0) -> bytes:
            started.set()
            assert release.wait(2.0)
            return original_read(count, offset=offset)

        handle.read = read  # type: ignore[method-assign]
        return handle

    filesystem.open_file_no_reparse = open_blocking  # type: ignore[method-assign]
    monkeypatch.setattr(scanner, "_windows_artifact_filesystem", filesystem)
    task = asyncio.create_task(scanner.scan_audio_cpp_package_root_async(root))
    assert await asyncio.to_thread(started.wait, 1.0)

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(handle.closed for handle in filesystem.handles)
