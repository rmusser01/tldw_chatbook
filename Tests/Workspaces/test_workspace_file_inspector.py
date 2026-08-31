"""Real-filesystem contracts for the read-only Workspace Files service."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces.file_inspector import (
    DirectoryStatus,
    FileReadKind,
    FilterStatus,
    WorkspaceFileInspector,
    safe_filesystem_text,
)


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
    assert inspector.list_directory(scope, continuation=first.continuation).error_code == "directory_changed"


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
