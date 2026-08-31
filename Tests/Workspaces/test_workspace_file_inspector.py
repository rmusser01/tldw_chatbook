"""Real-filesystem contracts for the read-only Workspace Files service."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import threading

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces.file_inspector import (
    DirectoryStatus,
    FileReadKind,
    FilterStatus,
    WorkspaceFileInspector,
    ScopeCaptureError,
    safe_filesystem_text,
)
from tldw_chatbook.Workspaces.models import (
    DEFAULT_WORKSPACE_ID,
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceRecord,
    WorkspaceRuntimeBinding,
)
import tldw_chatbook.Workspaces.file_inspector as file_inspector


def _registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="file-inspector")
    )
    registry.create_workspace(workspace_id="ws-a", name="Workspace A")
    registry.create_workspace(workspace_id="ws-b", name="Workspace B")
    return registry


def _scope(tmp_path: Path) -> tuple[WorkspaceFileInspector, LocalWorkspaceRegistryService, Path, object]:
    registry = _registry(tmp_path)
    root = tmp_path / "workspace"
    root.mkdir()
    binding = registry.add_folder_binding("ws-a", root)
    inspector = WorkspaceFileInspector(registry)
    return inspector, registry, root, inspector.capture_binding("ws-a", binding.binding_id)


def test_capture_is_immutable_and_every_operation_revalidates_current_registry(
    tmp_path: Path,
) -> None:
    """Catch a service that treats an opening snapshot as continuing authority."""
    inspector, registry, root, scope = _scope(tmp_path)
    (root / "visible.txt").write_text("visible")

    assert inspector.list_directory(scope).status is DirectoryStatus.COMPLETE
    registry.archive_workspace("ws-a")

    result = inspector.list_directory(scope)

    assert result.status is DirectoryStatus.FAILED
    assert result.error_code == "workspace_unavailable"


def test_scope_rejects_removed_retargeted_foreign_default_and_nonlocal_bindings(
    tmp_path: Path,
) -> None:
    """Catch stale binding IDs silently selecting a current or default root."""
    inspector, registry, root, scope = _scope(tmp_path)
    binding = registry.list_folder_bindings("ws-a")[0]
    registry.remove_runtime_binding(binding.binding_id)
    assert inspector.list_directory(scope).error_code == "binding_changed"

    root.mkdir(exist_ok=True)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    replacement_binding = registry.add_folder_binding("ws-a", replacement)
    retargeted_scope = inspector.capture_binding("ws-a", replacement_binding.binding_id)
    registry.save_runtime_binding(
        type(replacement_binding)(
            workspace_id="ws-b",
            binding_id=replacement_binding.binding_id,
            binding_kind=replacement_binding.binding_kind,
            label=replacement_binding.label,
            locator=replacement_binding.locator,
            status=replacement_binding.status,
            metadata=replacement_binding.metadata,
            created_at=replacement_binding.created_at,
        )
    )
    assert inspector.list_directory(retargeted_scope).error_code == "binding_changed"


def test_list_and_read_reject_component_escape_links_special_files_and_vcs(
    tmp_path: Path,
) -> None:
    """Catch path resolution that follows a rendered label or a symlink."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    (root / "safe.txt").write_text("ok")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (root / "link.txt").symlink_to(outside)
    linked_dir = tmp_path / "linked-dir"
    linked_dir.mkdir()
    (linked_dir / "nested.txt").write_text("outside")
    (root / "linked-dir").symlink_to(linked_dir, target_is_directory=True)
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("private")
    fifo = root / "pipe"
    os.mkfifo(fifo)

    page = inspector.list_directory(scope)
    labels = {entry.display_name for entry in page.entries}

    assert "safe.txt" in labels
    assert "link.txt" not in labels
    assert "linked-dir" not in labels
    assert ".git" not in labels
    assert "pipe" not in labels
    assert inspector.read_file(scope, ("..", "outside.txt")).error_code == "invalid_path"
    assert inspector.read_file(scope, (str(outside),)).error_code == "invalid_path"
    assert inspector.read_file(scope, ("link.txt",)).error_code == "unsafe_target"
    assert inspector.read_file(scope, (".git", "config")).error_code == "excluded_path"


def test_safe_filesystem_text_is_one_way_safe_while_raw_parts_remain_authority(
    tmp_path: Path,
) -> None:
    """Catch display formatting that loses bytes or leaves terminal controls live."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    name = "Readme[bold] ✓\n\t\x1b\u202e"
    (root / name).write_text("safe")

    page = inspector.list_directory(scope)
    entry = next(item for item in page.entries if item.raw_parts == (name,))

    assert safe_filesystem_text("Résumé [literal]") == "Résumé [literal]"
    assert entry.display_name == "Readme[bold] ✓\\n\\t\\x1b\\u202e"
    assert entry.raw_parts == (name,)
    assert inspector.read_file(scope, entry.raw_parts).kind is FileReadKind.TEXT
    assert safe_filesystem_text(bytes([0xFF]).decode("utf-8", "surrogateescape")) == "\\xff"


def test_directory_pages_are_deterministic_bounded_and_revision_pinned(tmp_path: Path) -> None:
    """Catch unbounded or mixed-revision directory pagination."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    (root / "z-dir").mkdir()
    (root / "a-dir").mkdir()
    (root / ".dotfile").write_text("dot")
    for index in range(205):
        (root / f"file-{index:03}.txt").write_text(str(index))
    (root / ".git").mkdir()
    (root / "node_modules").mkdir()

    first = inspector.list_directory(scope)
    second = inspector.list_directory(scope, continuation=first.continuation)

    assert first.status is DirectoryStatus.PARTIAL
    assert len(first.entries) == 200
    assert [item.raw_parts[0] for item in first.entries[:2]] == ["a-dir", "z-dir"]
    assert any(item.raw_parts == (".dotfile",) for item in first.entries)
    assert first.excluded_vcs_count == 1
    assert first.excluded_cache_count == 1
    assert second.status is DirectoryStatus.COMPLETE
    assert len(second.entries) == 8
    (root / "new.txt").write_text("new")
    assert inspector.list_directory(scope, continuation=first.continuation).error_code == "invalid_page"


def test_filter_is_literal_selected_scope_bounded_and_reports_exclusions(tmp_path: Path) -> None:
    """Catch recursive filtering that crosses bindings, follows links, or hides bounds."""
    inspector, registry, root, scope = _scope(tmp_path)
    (root / "nested").mkdir()
    (root / "nested" / "Needle[one].txt").write_text("one")
    (root / ".git").mkdir()
    (root / ".git" / "needle.txt").write_text("hidden")
    other = tmp_path / "other"
    other.mkdir()
    (other / "needle.txt").write_text("other")
    registry.add_folder_binding("ws-b", other)
    (root / "external").symlink_to(other, target_is_directory=True)
    for index in range(501):
        (root / f"needle-{index:03}.txt").write_text("x")

    result = inspector.filter_paths(scope, "needle[")
    truncated = inspector.filter_paths(scope, "needle-")
    excluded = inspector.filter_paths(scope, "hidden")

    assert result.status is FilterStatus.COMPLETE
    assert [match.raw_parts for match in result.matches] == [("nested", "Needle[one].txt")]
    assert truncated.status is FilterStatus.TRUNCATED
    assert len(truncated.matches) == 500
    assert "500" in truncated.status_copy
    assert excluded.status is FilterStatus.ONLY_EXCLUDED
    assert excluded.excluded_count >= 1
    assert excluded.excluded_locations_unsearched is True
    assert inspector.filter_debounce_ms == 150


def test_filter_reports_cancellation_and_does_not_descend_symlink_directories(
    tmp_path: Path,
) -> None:
    """Catch a filter that keeps walking after cancellation or follows aliases."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    for index in range(20):
        (root / f"match-{index}").write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "match-outside").write_text("x")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls > 4

    result = inspector.filter_paths(scope, "match", is_cancelled=cancelled)

    assert result.status is FilterStatus.CANCELLED
    assert all("outside" not in "/".join(item.raw_parts) for item in result.matches)


def test_read_classifies_safe_bom_invalid_control_and_oversized_files(tmp_path: Path) -> None:
    """Catch unsafe previews and accidental reads of metadata-only files."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    (root / "bom.txt").write_bytes(b"\xef\xbb\xbfHello")
    (root / "invalid.bin").write_bytes(b"\xff\x00")
    (root / "control.txt").write_text("hi\x1bthere")
    huge = root / "huge.bin"
    with huge.open("wb") as handle:
        handle.truncate(8 * 1024 * 1024 + 1)

    assert inspector.read_file(scope, ("bom.txt",)).text == "Hello"
    assert inspector.read_file(scope, ("invalid.bin",)).kind is FileReadKind.INVALID_UTF8
    control = inspector.read_file(scope, ("control.txt",))
    assert control.kind is FileReadKind.CONTROL_TEXT
    assert control.text == "hi\\x1bthere"
    assert inspector.read_file(scope, ("huge.bin",)).kind is FileReadKind.METADATA_ONLY


def test_large_utf8_reads_are_paged_on_character_boundaries_and_revision_pinned(
    tmp_path: Path,
) -> None:
    """Catch byte-sliced UTF-8 pages or pages mixed after an external replacement."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    payload = "é" * 100_000 + "🙂" * 100_001
    (root / "large.txt").write_text(payload)

    first = inspector.read_file(scope, ("large.txt",))
    second = inspector.read_file(
        scope,
        ("large.txt",),
        page_offset=first.next_page_offset,
        expected_revision=first.revision,
    )

    assert first.kind is FileReadKind.PAGED
    assert first.character_range == (0, 100_000)
    assert first.text == "é" * 100_000
    assert second.character_range == (100_000, 200_000)
    assert second.text == "🙂" * 100_000
    assert len(inspector.cached_page_offsets(scope, ("large.txt",))) <= 3
    replacement = root / "replacement.txt"
    replacement.write_text(payload)
    os.replace(replacement, root / "large.txt")
    changed = inspector.read_file(
        scope,
        ("large.txt",),
        page_offset=second.next_page_offset,
        expected_revision=first.revision,
    )
    assert changed.kind is FileReadKind.REVISION_CHANGED


def test_paged_cache_rechecks_a_strong_content_revision_before_serving_hits(
    tmp_path: Path,
) -> None:
    """Same inode/size/mtime replacement content must not revive a cached page."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    path = root / "large.txt"
    path.write_text("a" * 200_001, encoding="utf-8")
    first = inspector.read_file(scope, ("large.txt",))
    assert first.kind is FileReadKind.PAGED
    original = path.stat()
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"b" * 200_001)
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    restored = path.stat()
    assert (restored.st_ino, restored.st_size, restored.st_mtime_ns) == (
        original.st_ino,
        original.st_size,
        original.st_mtime_ns,
    )
    changed = inspector.read_file(
        scope, ("large.txt",), page_offset=0, expected_revision=first.revision
    )
    assert changed.kind is FileReadKind.REVISION_CHANGED
    assert changed.text == ""


def test_root_replacement_and_disappeared_file_fail_closed(tmp_path: Path) -> None:
    """Catch root/file replacement races that retain an old rendered selection."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    file_path = root / "gone.txt"
    file_path.write_text("gone")
    file_path.unlink()
    assert inspector.read_file(scope, ("gone.txt",)).error_code == "missing_target"

    replacement = tmp_path / "replacement-root"
    replacement.mkdir()
    shutil.rmtree(root)
    replacement.rename(root)
    assert inspector.list_directory(scope).error_code == "binding_changed"


def test_read_rejects_root_symlink_swap_after_scope_revalidation(
    tmp_path: Path, monkeypatch
) -> None:
    """Catch a root pathname reopened after validation and redirected by a swap."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    (root / "visible.txt").write_text("inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "visible.txt").write_text("secret")
    real_open = file_inspector.os.open
    swapped = False

    def swap_before_root_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and Path(path) == root:
            swapped = True
            shutil.rmtree(root)
            root.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(file_inspector.os, "open", swap_before_root_open)

    result = inspector.read_file(scope, ("visible.txt",))

    assert swapped is True
    assert result.kind is FileReadKind.FAILED
    assert result.text == ""


def test_read_rejects_intermediate_directory_symlink_swap_after_root_open(
    tmp_path: Path, monkeypatch
) -> None:
    """Catch intermediate components reopened from a root-derived pathname."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    nested = root / "nested"
    nested.mkdir()
    (nested / "visible.txt").write_text("inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "visible.txt").write_text("secret")
    real_open = file_inspector.os.open
    swapped = False

    def swap_before_intermediate_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and path == "nested" and kwargs.get("dir_fd") is not None:
            swapped = True
            nested.rename(root / "previous-nested")
            nested.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(file_inspector.os, "open", swap_before_intermediate_open)

    result = inspector.read_file(scope, ("nested", "visible.txt"))

    assert swapped is True
    assert result.kind is FileReadKind.FAILED
    assert result.text == ""


def test_filter_streams_progress_and_returns_honest_unsearched_exclusion_state(
    tmp_path: Path,
) -> None:
    """Catch eager filtering and an unrelated query reported as excluded-only."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    (root / "ordinary.txt").write_text("x")
    (root / ".git").mkdir()
    (root / ".git" / "needle.txt").write_text("hidden")
    progress = []

    unrelated = inspector.filter_paths(
        scope, "unrelated", on_progress=progress.append
    )
    excluded_name = inspector.filter_paths(scope, "needle", on_progress=progress.append)

    assert unrelated.status is FilterStatus.ONLY_EXCLUDED
    assert unrelated.excluded_locations_unsearched is True
    assert excluded_name.status is FilterStatus.ONLY_EXCLUDED
    assert excluded_name.excluded_locations_unsearched is True
    assert progress[-1].visited_entries >= 1


def test_filter_checks_the_visit_cap_before_a_huge_directory_is_materialized(
    tmp_path: Path, monkeypatch
) -> None:
    """Catch list(scandir(...)) exhausting a huge directory before cancellation."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    for index in range(10):
        (root / f"entry-{index}").write_text("x")
    monkeypatch.setattr(file_inspector, "FILTER_VISIT_LIMIT", 3)

    result = inspector.filter_paths(scope, "entry")

    assert result.status is FilterStatus.PARTIAL
    assert result.visited_entries == 3
    assert result.progress.visited_entries == 3


def test_directory_continuation_is_service_issued_and_validated(tmp_path: Path) -> None:
    """Catch caller-forged, negative, or cross-service continuation offsets."""
    inspector, registry, root, scope = _scope(tmp_path)
    for index in range(205):
        (root / f"entry-{index}").write_text("x")
    first = inspector.list_directory(scope)
    assert first.continuation is not None
    forged = replace(first.continuation, offset=-200)
    other_service = WorkspaceFileInspector(registry)

    assert inspector.list_directory(scope, continuation=forged).error_code == "invalid_page"
    assert other_service.list_directory(scope, continuation=first.continuation).error_code == "invalid_page"


def test_large_page_cache_serves_hits_and_evicts_nonadjacent_pages(tmp_path: Path, monkeypatch) -> None:
    """Catch a nominal page cache that always decodes and retains stale pages."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    (root / "large.txt").write_text("é" * 450_000)
    first = inspector.read_file(scope, ("large.txt",))
    calls = 0
    real_decoder = file_inspector._decode_text_page

    def count_decodes(descriptor, page_offset):
        nonlocal calls
        calls += 1
        return real_decoder(descriptor, page_offset)

    monkeypatch.setattr(file_inspector, "_decode_text_page", count_decodes)
    hit = inspector.read_file(scope, ("large.txt",), page_offset=0, expected_revision=first.revision)
    second = inspector.read_file(scope, ("large.txt",), page_offset=100_000, expected_revision=first.revision)
    fourth = inspector.read_file(scope, ("large.txt",), page_offset=300_000, expected_revision=first.revision)

    assert hit.text == first.text
    assert calls == 2
    assert second.kind is FileReadKind.PAGED
    assert fourth.kind is FileReadKind.PAGED
    assert inspector.cached_page_offsets(scope, ("large.txt",)) == (300_000,)


def test_service_operations_do_not_start_threads_or_workers(tmp_path: Path, monkeypatch) -> None:
    """Catch future service work escaping the modal's owned worker lanes."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    (root / "file.txt").write_text("x")
    starts = []
    monkeypatch.setattr(threading.Thread, "start", lambda thread: starts.append(thread))

    inspector.list_directory(scope)
    inspector.filter_paths(scope, "file")
    inspector.read_file(scope, ("file.txt",))

    assert starts == []


def test_filter_only_excluded_never_claims_the_hidden_name_matched(tmp_path: Path) -> None:
    """Catch excluded-only state being replaced by an ambiguous empty result."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    (root / ".git").mkdir()
    (root / ".git" / "hidden-name.txt").write_text("x")

    result = inspector.filter_paths(scope, "hidden-name")

    assert result.status is FilterStatus.ONLY_EXCLUDED
    assert result.matches == ()
    assert result.excluded_locations_unsearched is True


def test_continuation_store_is_bounded_and_tokens_are_one_shot(tmp_path: Path) -> None:
    """Catch unbounded continuation retention and replayed pagination tokens."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    for index in range(205):
        (root / f"entry-{index}").write_text("x")
    pages = [inspector.list_directory(scope) for _ in range(20)]
    token = pages[-1].continuation
    assert token is not None

    assert inspector.continuation_count <= 8
    assert inspector.list_directory(scope, continuation=token).status is DirectoryStatus.COMPLETE
    assert inspector.list_directory(scope, continuation=token).error_code == "invalid_page"


def test_filter_closes_directory_descriptor_when_scandir_construction_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """Catch a descriptor leak before filter iteration starts."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    (root / "file.txt").write_text("x")
    real_close = file_inspector.os.close
    closed = []

    def track_close(descriptor):
        closed.append(descriptor)
        return real_close(descriptor)

    monkeypatch.setattr(file_inspector.os, "close", track_close)
    monkeypatch.setattr(file_inspector.os, "scandir", lambda descriptor: (_ for _ in ()).throw(OSError()))

    before = len(closed)
    for _ in range(5):
        result = inspector.filter_paths(scope, "file")
        assert result.error_code == "directory_unavailable"

    assert len(closed) - before >= 10


def test_capture_rejects_real_default_workspace_identity(tmp_path: Path) -> None:
    """Catch Default workspace scopes being treated as filesystem authority."""
    root = tmp_path / "root"
    root.mkdir()
    binding = WorkspaceRuntimeBinding(
        workspace_id=DEFAULT_WORKSPACE_ID, binding_id="default-binding",
        binding_kind=RuntimeBindingKind.LOCAL_FILESYSTEM, label="root",
        locator=str(root), status=RuntimeBindingStatus.READY,
    )

    class Registry:
        def get_workspace(self, workspace_id):
            return WorkspaceRecord(workspace_id=DEFAULT_WORKSPACE_ID, name="Default")
        def get_runtime_binding(self, binding_id):
            return binding

    with pytest.raises(ScopeCaptureError, match="workspace_unavailable"):
        WorkspaceFileInspector(Registry()).capture_binding(DEFAULT_WORKSPACE_ID, "default-binding")


def test_capture_rejects_nonlocal_runtime_binding(tmp_path: Path) -> None:
    """Catch container/runtime bindings entering the local filesystem service."""
    registry = _registry(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    binding = WorkspaceRuntimeBinding(
        workspace_id="ws-a", binding_id="container-binding",
        binding_kind=RuntimeBindingKind.CONTAINER, label="container",
        locator=str(root), status=RuntimeBindingStatus.READY,
    )
    registry.save_runtime_binding(binding)

    with pytest.raises(ScopeCaptureError, match="binding_changed"):
        WorkspaceFileInspector(registry).capture_binding("ws-a", binding.binding_id)


def test_read_and_filter_revalidate_fresh_registry_mutations(tmp_path: Path) -> None:
    """Catch stale read/filter scope use after independent registry revocation."""
    inspector, registry, root, scope = _scope(tmp_path)
    (root / "file.txt").write_text("secret")
    registry.archive_workspace("ws-a")

    read = inspector.read_file(scope, ("file.txt",))
    filtered = inspector.filter_paths(scope, "file")

    assert read.kind is FileReadKind.FAILED
    assert read.text == ""
    assert filtered.status is FilterStatus.FAILED
    assert filtered.matches == ()


def test_unconsumed_continuation_detects_directory_revision_change(tmp_path: Path) -> None:
    """Catch a fresh page token mixing entries after its directory changed."""
    inspector, _registry_service, root, scope = _scope(tmp_path)
    for index in range(205):
        (root / f"entry-{index}").write_text("x")
    first = inspector.list_directory(scope)
    assert first.continuation is not None
    (root / "revision-change").write_text("x")

    result = inspector.list_directory(scope, continuation=first.continuation)

    assert result.error_code == "directory_changed"
