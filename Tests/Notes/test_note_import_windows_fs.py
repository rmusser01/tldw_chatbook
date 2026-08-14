"""Host-independent contracts for the Windows one-time import filesystem seam."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest

from tldw_chatbook.Notes import note_import_discovery
from tldw_chatbook.Notes.note_import_discovery import (
    ImportSelectionError,
    VerifiedSourceReadError,
    discover_import_sources,
    read_discovered_source,
)
from tldw_chatbook.Notes.note_import_plan_models import ImportBounds

_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


def _bounds(**overrides: int) -> ImportBounds:
    values = {
        "max_files": 50,
        "max_file_bytes": 1_000_000,
        "max_total_bytes": 5_000_000,
        "max_depth": 8,
        "max_entries": 100,
    }
    values.update(overrides)
    return ImportBounds(**values)


class FakeWindowsFilesystem:
    """Read-only host wrapper with injectable Windows race/reparse behavior."""

    def __init__(self) -> None:
        self.reparse_paths: set[str] = set()
        self.lstat_error_paths: set[str] = set()
        self.open_flags: list[int] = []
        self.inheritable_values: list[bool] = []
        self.read_calls = 0
        self.open_error = False
        self.fstat_identity_mismatch = False
        self.before_open: Callable[[Path], None] | None = None
        self.before_scandir: Callable[[Path], None] | None = None
        self.after_first_read: Callable[[], None] | None = None
        self.after_scandir: Callable[[Path], None] | None = None
        self.scanned_paths: list[Path] = []

    def _key(self, path: Path) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    def mark_reparse(self, path: Path) -> None:
        self.reparse_paths.add(self._key(path))

    def absolute(self, path: Path) -> Path:
        return Path(os.path.abspath(os.fspath(path)))

    def lstat(self, path: Path) -> os.stat_result | SimpleNamespace:
        if self._key(path) in self.lstat_error_paths:
            raise OSError(f"PRIVATE WINDOWS PATH {path}")
        metadata = os.lstat(path)
        if self._key(path) not in self.reparse_paths:
            return metadata
        return SimpleNamespace(
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_mode=metadata.st_mode,
            st_size=metadata.st_size,
            st_mtime_ns=metadata.st_mtime_ns,
            st_ctime_ns=metadata.st_ctime_ns,
            st_file_attributes=_REPARSE_ATTRIBUTE,
        )

    def scandir(self, path: Path) -> object:
        self.scanned_paths.append(path)
        if self.before_scandir is not None:
            self.before_scandir(path)
        iterator = os.scandir(path)
        callback = self.after_scandir
        if callback is None:
            return iterator

        class ScandirWithExitHook:
            def __enter__(self) -> os.ScandirIterator[str]:
                return iterator.__enter__()

            def __exit__(self, *args: object) -> None:
                try:
                    iterator.__exit__(*args)
                finally:
                    callback(path)

        return ScandirWithExitHook()

    def open(self, path: Path, flags: int) -> int:
        self.open_flags.append(flags)
        if self.open_error:
            raise OSError(f"PRIVATE WINDOWS OPEN PATH {path}")
        if self.before_open is not None:
            callback = self.before_open
            self.before_open = None
            callback(path)
        return os.open(path, flags)

    def set_inheritable(self, descriptor: int, inheritable: bool) -> None:
        self.inheritable_values.append(inheritable)
        os.set_inheritable(descriptor, inheritable)

    def fstat(self, descriptor: int) -> os.stat_result | SimpleNamespace:
        metadata = os.fstat(descriptor)
        if not self.fstat_identity_mismatch:
            return metadata
        return SimpleNamespace(
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino + 1,
            st_mode=metadata.st_mode,
            st_size=metadata.st_size,
            st_mtime_ns=metadata.st_mtime_ns,
            st_ctime_ns=metadata.st_ctime_ns,
        )

    def read(self, descriptor: int, count: int) -> bytes:
        self.read_calls += 1
        chunk = os.read(descriptor, count)
        if chunk and self.after_first_read is not None:
            callback = self.after_first_read
            self.after_first_read = None
            callback()
        return chunk

    def close(self, descriptor: int) -> None:
        os.close(descriptor)


class GuardedScandirFilesystem(FakeWindowsFilesystem):
    """Fail if bounded discovery asks the directory iterator for a third entry."""

    def __init__(self) -> None:
        super().__init__()
        self.entries_yielded = 0

    def scandir(self, path: Path) -> object:
        iterator = os.scandir(path)
        filesystem = self

        class GuardedIterator:
            def __enter__(self) -> Self:
                iterator.__enter__()
                return self

            def __exit__(self, *args: object) -> None:
                iterator.__exit__(*args)

            def __iter__(self) -> GuardedIterator:
                return self

            def __next__(self) -> os.DirEntry[str]:
                filesystem.entries_yielded += 1
                if filesystem.entries_yielded > 2:
                    raise AssertionError("directory iterator exceeded max_entries")
                return next(iterator)

        return GuardedIterator()


def _force_windows_adapter(
    monkeypatch: pytest.MonkeyPatch,
    filesystem: FakeWindowsFilesystem,
) -> None:
    monkeypatch.setattr(
        note_import_discovery,
        "_platform_uses_windows_adapter",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        note_import_discovery,
        "_windows_filesystem",
        lambda: filesystem,
        raising=False,
    )


def test_windows_adapter_discovers_and_reads_selected_file_without_posix_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "note.md"
    source.write_bytes(b"Body")
    filesystem = FakeWindowsFilesystem()
    _force_windows_adapter(monkeypatch, filesystem)
    monkeypatch.delattr(note_import_discovery.os, "O_NOFOLLOW", raising=False)

    discovery = discover_import_sources([source], _bounds())
    content = read_discovered_source(discovery.candidates[0], _bounds())

    assert discovery.root_label is None
    assert discovery.failures == ()
    assert discovery.entry_count == 1
    assert discovery.total_bytes == 4
    assert len(discovery.candidates[0].parent_identities) == len(source.parts) - 1
    assert content == b"Body"
    assert len(filesystem.open_flags) == 1
    assert all(
        flags & (os.O_WRONLY | os.O_RDWR) == 0 for flags in filesystem.open_flags
    )
    assert filesystem.inheritable_values == [False]


def test_windows_adapter_recursively_discovers_and_reads_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "Project"
    child = root / "child"
    child.mkdir(parents=True)
    (root / "a.md").write_bytes(b"A")
    (child / "b.md").write_bytes(b"B")
    filesystem = FakeWindowsFilesystem()
    _force_windows_adapter(monkeypatch, filesystem)

    discovery = discover_import_sources([root], _bounds())
    contents = tuple(
        read_discovered_source(candidate, _bounds())
        for candidate in discovery.candidates
    )

    assert discovery.root_label == "Project"
    assert discovery.entry_count == 3
    assert discovery.failures == ()
    assert tuple(
        candidate.source.display_path for candidate in discovery.candidates
    ) == ("Project/a.md", "Project/child/b.md")
    assert contents == (b"A", b"B")


@pytest.mark.parametrize("reparse_location", ["root", "parent", "leaf"])
def test_windows_adapter_rejects_selected_reparse_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reparse_location: str,
) -> None:
    root = tmp_path / "Project"
    root.mkdir()
    source = root / "note.md"
    source.write_text("Body", encoding="utf-8")
    selected = root if reparse_location == "root" else source
    reparse_path = root if reparse_location in {"root", "parent"} else source
    filesystem = FakeWindowsFilesystem()
    filesystem.mark_reparse(reparse_path)
    _force_windows_adapter(monkeypatch, filesystem)

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([selected], _bounds())

    assert raised.value.reason_code == "selected_symlink"
    assert str(root) not in str(raised.value)


def test_windows_adapter_skips_nested_reparse_without_descending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "Project"
    linked = root / "linked"
    linked.mkdir(parents=True)
    (linked / "secret.md").write_text("private", encoding="utf-8")
    filesystem = FakeWindowsFilesystem()
    filesystem.mark_reparse(linked)
    _force_windows_adapter(monkeypatch, filesystem)

    discovery = discover_import_sources([root], _bounds())

    assert discovery.candidates == ()
    assert len(discovery.failures) == 1
    assert discovery.failures[0].display_path == "Project/linked"
    assert discovery.failures[0].reason_code == "nested_symlink"


def test_windows_adapter_rejects_identity_swap_after_open_before_any_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "note.txt"
    source.write_text("original", encoding="utf-8")
    filesystem = FakeWindowsFilesystem()
    _force_windows_adapter(monkeypatch, filesystem)
    discovery = discover_import_sources([source], _bounds())

    def replace_leaf(_path: Path) -> None:
        filesystem.fstat_identity_mismatch = True

    filesystem.before_open = replace_leaf

    with pytest.raises(VerifiedSourceReadError) as raised:
        read_discovered_source(discovery.candidates[0], _bounds())

    assert raised.value.reason_code == "source_changed"
    assert filesystem.read_calls == 0


@pytest.mark.parametrize("changed_component", ["leaf", "parent"])
def test_windows_adapter_rejects_lexical_changes_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_component: str,
) -> None:
    parent = tmp_path / "selected"
    parent.mkdir()
    source = parent / "note.txt"
    source.write_text("original", encoding="utf-8")
    filesystem = FakeWindowsFilesystem()
    _force_windows_adapter(monkeypatch, filesystem)
    discovery = discover_import_sources([source], _bounds())

    if changed_component == "leaf":
        filesystem.after_first_read = lambda: filesystem.mark_reparse(source)
    else:

        def replace_parent() -> None:
            filesystem.mark_reparse(parent)

        filesystem.after_first_read = replace_parent

    with pytest.raises(VerifiedSourceReadError) as raised:
        read_discovered_source(discovery.candidates[0], _bounds())

    assert raised.value.reason_code == "source_changed"
    assert filesystem.read_calls >= 1


def test_windows_adapter_rechecks_directory_identity_around_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "Project"
    root.mkdir()
    (root / "note.md").write_text("Body", encoding="utf-8")
    filesystem = FakeWindowsFilesystem()

    def replace_root(path: Path) -> None:
        if path == root:
            filesystem.mark_reparse(root)

    filesystem.after_scandir = replace_root
    _force_windows_adapter(monkeypatch, filesystem)

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([root], _bounds())

    assert raised.value.reason_code == "selection_changed"


def test_windows_adapter_rechecks_ancestor_chain_before_nested_descent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "Project"
    parent = root / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    (child / "note.md").write_text("Body", encoding="utf-8")
    filesystem = FakeWindowsFilesystem()

    def replace_ancestor(path: Path) -> None:
        if path == parent:
            filesystem.mark_reparse(root)

    filesystem.before_scandir = replace_ancestor
    _force_windows_adapter(monkeypatch, filesystem)

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([root], _bounds())

    assert raised.value.reason_code == "selection_changed"
    assert child not in filesystem.scanned_paths


@pytest.mark.parametrize(
    ("setup_kind", "bounds_overrides", "reason_code"),
    [
        ("large_file", {"max_file_bytes": 4}, "max_file_bytes_exceeded"),
        ("many_files", {"max_files": 1}, "max_files_exceeded"),
        (
            "total_bytes",
            {"max_file_bytes": 5, "max_total_bytes": 5},
            "max_total_bytes_exceeded",
        ),
        ("many_entries", {"max_entries": 1}, "max_entries_exceeded"),
        ("deep_tree", {"max_depth": 0}, "max_depth_exceeded"),
    ],
)
def test_windows_adapter_preserves_discovery_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setup_kind: str,
    bounds_overrides: dict[str, int],
    reason_code: str,
) -> None:
    root = tmp_path / "Project"
    root.mkdir()
    if setup_kind == "large_file":
        selected = root / "large.md"
        selected.write_bytes(b"12345")
    elif setup_kind in {"many_files", "many_entries", "total_bytes"}:
        (root / "one.md").write_text("one", encoding="utf-8")
        (root / "two.md").write_text("two", encoding="utf-8")
        selected = root
    else:
        child = root / "child"
        child.mkdir()
        (child / "note.md").write_text("Body", encoding="utf-8")
        selected = root
    filesystem = FakeWindowsFilesystem()
    _force_windows_adapter(monkeypatch, filesystem)

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([selected], _bounds(**bounds_overrides))

    assert raised.value.reason_code == reason_code


def test_windows_adapter_applies_entry_bound_while_iterating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "Project"
    root.mkdir()
    for name in ("one.md", "two.md", "three.md"):
        (root / name).write_text(name, encoding="utf-8")
    filesystem = GuardedScandirFilesystem()
    _force_windows_adapter(monkeypatch, filesystem)

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([root], _bounds(max_entries=1))

    assert raised.value.reason_code == "max_entries_exceeded"
    assert filesystem.entries_yielded == 2


def test_windows_adapter_normalizes_os_errors_without_private_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "PRIVATE-note.md"
    source.write_text("Body", encoding="utf-8")
    filesystem = FakeWindowsFilesystem()
    filesystem.lstat_error_paths.add(filesystem._key(source))
    _force_windows_adapter(monkeypatch, filesystem)

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([source], _bounds())

    assert raised.value.reason_code == "selection_unreadable"
    assert "PRIVATE" not in str(raised.value)
    assert str(source) not in str(raised.value)


def test_windows_adapter_normalizes_read_errors_without_private_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "PRIVATE-note.md"
    source.write_text("Body", encoding="utf-8")
    filesystem = FakeWindowsFilesystem()
    _force_windows_adapter(monkeypatch, filesystem)
    discovery = discover_import_sources([source], _bounds())
    filesystem.open_error = True

    with pytest.raises(VerifiedSourceReadError) as raised:
        read_discovered_source(discovery.candidates[0], _bounds())

    assert raised.value.reason_code == "source_unavailable"
    assert "PRIVATE" not in str(raised.value)
    assert str(source) not in str(raised.value)


@pytest.mark.skipif(
    os.name != "nt" or not hasattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT"),
    reason="requires native Windows reparse metadata",
)
def test_native_windows_selected_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Windows symlink creation unavailable: {type(error).__name__}")

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([alias], _bounds())

    assert raised.value.reason_code == "selected_symlink"
