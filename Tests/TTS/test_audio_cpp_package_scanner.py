"""Bounded read-only discovery for guided audio.cpp packages."""

from __future__ import annotations

import asyncio
import os
import stat
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


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
