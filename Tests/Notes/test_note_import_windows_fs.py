"""Host-independent contracts for the Windows one-time import filesystem seam."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest

from tldw_chatbook.Notes import note_import_discovery, note_import_windows_fs
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
        self.require_pins = False
        self.pin_calls = 0
        self.fail_pin_call: int | None = None
        self.live_pins: dict[int, Path] = {}
        self.closed_pins: list[int] = []
        self.open_file_paths: dict[int, Path] = {}
        self.unsafe_open_calls = 0
        self.scandir_without_pins = 0
        self.read_without_pins = 0
        self.fail_open_file = False
        self.scandir_exception: BaseException | None = None
        self.read_exception: BaseException | None = None
        self.path_stat_overrides: dict[str, dict[str, int]] = {}
        self.handle_stat_overrides: dict[str, dict[str, int]] = {}
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

    def override_handle_stat(self, path: Path, **fields: int) -> None:
        self.handle_stat_overrides[self._key(path)] = fields

    def override_path_stat(self, path: Path, **fields: int) -> None:
        self.path_stat_overrides[self._key(path)] = fields

    def _directory_chain(self, path: Path, *, include_leaf: bool) -> tuple[Path, ...]:
        absolute = self.absolute(path)
        components = absolute.parts[1:] if include_leaf else absolute.parts[1:-1]
        current = Path(absolute.anchor)
        chain = [current]
        for component in components:
            current /= component
            chain.append(current)
        return tuple(chain)

    def _has_pinned_chain(self, path: Path, *, include_leaf: bool) -> bool:
        live_paths = {self._key(value) for value in self.live_pins.values()}
        return all(
            self._key(component) in live_paths
            for component in self._directory_chain(path, include_leaf=include_leaf)
        )

    def absolute(self, path: Path) -> Path:
        return Path(os.path.abspath(os.fspath(path)))

    def lstat(self, path: Path) -> os.stat_result | SimpleNamespace:
        key = self._key(path)
        if key in self.lstat_error_paths:
            raise OSError(f"PRIVATE WINDOWS PATH {path}")
        metadata = os.lstat(path)
        overrides = self.path_stat_overrides.get(key, {})
        if key not in self.reparse_paths and not overrides:
            return metadata
        return SimpleNamespace(
            st_dev=overrides.get("st_dev", metadata.st_dev),
            st_ino=overrides.get("st_ino", metadata.st_ino),
            st_mode=overrides.get("st_mode", metadata.st_mode),
            st_size=overrides.get("st_size", metadata.st_size),
            st_mtime_ns=overrides.get("st_mtime_ns", metadata.st_mtime_ns),
            st_ctime_ns=overrides.get("st_ctime_ns", metadata.st_ctime_ns),
            st_file_attributes=(_REPARSE_ATTRIBUTE if key in self.reparse_paths else 0),
        )

    def scandir(self, path: Path) -> object:
        self.scanned_paths.append(path)
        if self.require_pins and not self._has_pinned_chain(path, include_leaf=True):
            self.scandir_without_pins += 1
            raise AssertionError("scandir called without a fully pinned chain")
        if self.scandir_exception is not None:
            raise self.scandir_exception
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
        self.unsafe_open_calls += 1
        return self.open_file_no_reparse(path)

    def pin_directory_no_reparse(self, path: Path) -> int:
        self.pin_calls += 1
        if self.fail_pin_call == self.pin_calls:
            raise OSError("PRIVATE PIN FAILURE")
        token = -(self.pin_calls + 10_000)
        self.live_pins[token] = path
        return token

    def open_file_no_reparse(self, path: Path) -> int:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        self.open_flags.append(flags)
        if self.require_pins and not self._has_pinned_chain(path, include_leaf=False):
            raise AssertionError("file opened without a fully pinned parent chain")
        if self.open_error or self.fail_open_file:
            raise OSError(f"PRIVATE WINDOWS OPEN PATH {path}")
        if self.before_open is not None:
            callback = self.before_open
            self.before_open = None
            callback(path)
        descriptor = os.open(path, flags)
        self.open_file_paths[descriptor] = path
        return descriptor

    def set_inheritable(self, descriptor: int, inheritable: bool) -> None:
        self.inheritable_values.append(inheritable)
        os.set_inheritable(descriptor, inheritable)

    def fstat(self, descriptor: int) -> os.stat_result | SimpleNamespace:
        if descriptor in self.live_pins:
            path = self.live_pins[descriptor]
            metadata = self.lstat(path)
        else:
            path = self.open_file_paths[descriptor]
            metadata = os.fstat(descriptor)
        overrides = dict(self.handle_stat_overrides.get(self._key(path), {}))
        if self.fstat_identity_mismatch:
            overrides["st_ino"] = metadata.st_ino + 1
        if not overrides:
            return metadata
        return SimpleNamespace(
            st_dev=overrides.get("st_dev", metadata.st_dev),
            st_ino=overrides.get("st_ino", metadata.st_ino),
            st_mode=overrides.get("st_mode", metadata.st_mode),
            st_size=overrides.get("st_size", metadata.st_size),
            st_mtime_ns=overrides.get("st_mtime_ns", metadata.st_mtime_ns),
            st_ctime_ns=overrides.get("st_ctime_ns", metadata.st_ctime_ns),
        )

    def read(self, descriptor: int, count: int) -> bytes:
        self.read_calls += 1
        path = self.open_file_paths[descriptor]
        if self.require_pins and not self._has_pinned_chain(path, include_leaf=False):
            self.read_without_pins += 1
            raise AssertionError("read called without a fully pinned parent chain")
        if self.read_exception is not None:
            raise self.read_exception
        chunk = os.read(descriptor, count)
        if chunk and self.after_first_read is not None:
            callback = self.after_first_read
            self.after_first_read = None
            callback()
        return chunk

    def close(self, descriptor: int) -> None:
        if descriptor in self.live_pins:
            self.live_pins.pop(descriptor)
            self.closed_pins.append(descriptor)
            return
        self.open_file_paths.pop(descriptor, None)
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
    assert len(filesystem.open_flags) == 2
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


def test_windows_adapter_keeps_all_ancestor_pins_live_for_scan_and_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "Project"
    child = root / "child"
    child.mkdir(parents=True)
    source = child / "note.md"
    source.write_text("Body", encoding="utf-8")
    filesystem = FakeWindowsFilesystem()
    filesystem.require_pins = True
    _force_windows_adapter(monkeypatch, filesystem)

    discovery = discover_import_sources([root], _bounds())
    content = read_discovered_source(discovery.candidates[0], _bounds())

    assert content == b"Body"
    assert filesystem.scandir_without_pins == 0
    assert filesystem.read_without_pins == 0
    assert filesystem.unsafe_open_calls == 0
    assert filesystem.pin_calls > 0
    assert filesystem.live_pins == {}
    assert len(filesystem.closed_pins) == filesystem.pin_calls


@pytest.mark.parametrize("selection_kind", ["file", "root", "nested_directory"])
def test_windows_adapter_accepts_path_handle_changed_ns_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection_kind: str,
) -> None:
    root = tmp_path / "Project"
    nested = root / "nested"
    nested.mkdir(parents=True)
    source = nested / "note.md"
    source.write_text("Body", encoding="utf-8")
    selected = source if selection_kind == "file" else root
    overridden = {
        "file": source,
        "root": root,
        "nested_directory": nested,
    }[selection_kind]
    metadata = os.lstat(overridden)
    filesystem = FakeWindowsFilesystem()
    filesystem.override_handle_stat(
        overridden,
        st_ctime_ns=metadata.st_ctime_ns + 1,
    )
    _force_windows_adapter(monkeypatch, filesystem)

    discovery = discover_import_sources([selected], _bounds())
    contents = tuple(
        read_discovered_source(candidate, _bounds())
        for candidate in discovery.candidates
    )

    assert contents == (b"Body",)


def test_windows_adapter_rejects_changed_ns_mutation_between_handle_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "note.md"
    source.write_text("Body", encoding="utf-8")
    metadata = os.lstat(source)
    filesystem = FakeWindowsFilesystem()
    filesystem.override_handle_stat(
        source,
        st_ctime_ns=metadata.st_ctime_ns + 1,
    )
    _force_windows_adapter(monkeypatch, filesystem)
    discovery = discover_import_sources([source], _bounds())

    filesystem.after_first_read = lambda: filesystem.override_handle_stat(
        source,
        st_ctime_ns=metadata.st_ctime_ns + 2,
    )

    with pytest.raises(VerifiedSourceReadError) as raised:
        read_discovered_source(discovery.candidates[0], _bounds())

    assert raised.value.reason_code == "source_changed"
    assert filesystem.read_calls >= 1


@pytest.mark.parametrize(
    "field",
    ["st_dev", "st_ino", "type", "st_mode", "st_size", "st_mtime_ns"],
)
@pytest.mark.parametrize("check_phase", ["selected_root", "reread_file"])
def test_windows_adapter_rejects_stable_path_handle_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    check_phase: str,
) -> None:
    root = tmp_path / "Project"
    root.mkdir()
    source = root / "note.md"
    source.write_text("Body", encoding="utf-8")
    target = root if check_phase == "selected_root" else source
    metadata = os.lstat(target)
    if field == "type":
        mismatched = stat.S_IFREG if stat.S_ISDIR(metadata.st_mode) else stat.S_IFDIR
        override = {"st_mode": mismatched | stat.S_IMODE(metadata.st_mode)}
    elif field == "st_mode":
        override = {field: metadata.st_mode ^ stat.S_IXUSR}
    else:
        override = {field: getattr(metadata, field) + 1}
    filesystem = FakeWindowsFilesystem()
    _force_windows_adapter(monkeypatch, filesystem)

    if check_phase == "selected_root":
        filesystem.override_handle_stat(target, **override)
        with pytest.raises(ImportSelectionError) as raised:
            discover_import_sources([root], _bounds())
        assert raised.value.reason_code == "selection_changed"
    else:
        discovery = discover_import_sources([source], _bounds())
        filesystem.override_handle_stat(target, **override)
        with pytest.raises(VerifiedSourceReadError) as raised:
            read_discovered_source(discovery.candidates[0], _bounds())
        assert raised.value.reason_code == "source_changed"
        assert filesystem.read_calls == 0


def test_windows_adapter_rejects_nested_directory_path_handle_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "Project"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "note.md").write_text("Body", encoding="utf-8")
    metadata = os.lstat(nested)
    filesystem = FakeWindowsFilesystem()
    filesystem.override_handle_stat(nested, st_ino=metadata.st_ino + 1)
    _force_windows_adapter(monkeypatch, filesystem)

    discovery = discover_import_sources([root], _bounds())

    assert discovery.candidates == ()
    assert tuple(failure.reason_code for failure in discovery.failures) == (
        "nested_unavailable",
    )


@pytest.mark.parametrize(
    "selection_kind",
    ["selected_file", "selected_root", "nested_directory"],
)
def test_windows_adapter_rejects_zero_inode_path_handle_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection_kind: str,
) -> None:
    root = tmp_path / "Project"
    nested = root / "nested"
    nested.mkdir(parents=True)
    source = nested / "note.md"
    source.write_text("Body", encoding="utf-8")
    selected = source if selection_kind == "selected_file" else root
    target = {
        "selected_file": source,
        "selected_root": root,
        "nested_directory": nested,
    }[selection_kind]
    filesystem = FakeWindowsFilesystem()
    filesystem.override_path_stat(target, st_ino=0)
    filesystem.override_handle_stat(target, st_ino=0)
    _force_windows_adapter(monkeypatch, filesystem)

    if selection_kind == "nested_directory":
        discovery = discover_import_sources([selected], _bounds())
        assert discovery.candidates == ()
        assert tuple(failure.reason_code for failure in discovery.failures) == (
            "nested_unavailable",
        )
    else:
        with pytest.raises(ImportSelectionError) as raised:
            discover_import_sources([selected], _bounds())
        assert raised.value.reason_code == "selection_changed"


def test_windows_adapter_rejects_zero_inode_on_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "note.md"
    source.write_text("Body", encoding="utf-8")
    filesystem = FakeWindowsFilesystem()
    _force_windows_adapter(monkeypatch, filesystem)
    discovery = discover_import_sources([source], _bounds())
    candidate = replace(
        discovery.candidates[0],
        identity=replace(discovery.candidates[0].identity, inode=0),
    )
    filesystem.override_path_stat(source, st_ino=0)
    filesystem.override_handle_stat(source, st_ino=0)

    with pytest.raises(VerifiedSourceReadError) as raised:
        read_discovered_source(candidate, _bounds())

    assert raised.value.reason_code == "source_changed"
    assert filesystem.read_calls == 0


def test_windows_adapter_accepts_matching_zero_device_with_nonzero_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "note.md"
    source.write_text("Body", encoding="utf-8")
    assert os.lstat(source).st_ino != 0
    filesystem = FakeWindowsFilesystem()
    filesystem.override_path_stat(source, st_dev=0)
    filesystem.override_handle_stat(source, st_dev=0)
    _force_windows_adapter(monkeypatch, filesystem)

    discovery = discover_import_sources([source], _bounds())
    content = read_discovered_source(discovery.candidates[0], _bounds())

    assert content == b"Body"


def test_windows_adapter_closes_partial_directory_pins_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "Project"
    root.mkdir()
    (root / "note.md").write_text("Body", encoding="utf-8")
    filesystem = FakeWindowsFilesystem()
    filesystem.fail_pin_call = 3
    _force_windows_adapter(monkeypatch, filesystem)

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([root], _bounds())

    assert raised.value.reason_code == "selection_unreadable"
    assert filesystem.live_pins == {}
    assert len(filesystem.closed_pins) == 2


def test_windows_adapter_closes_pins_after_unexpected_scan_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "Project"
    root.mkdir()
    filesystem = FakeWindowsFilesystem()
    filesystem.scandir_exception = RuntimeError("PRIVATE UNEXPECTED SCAN")
    _force_windows_adapter(monkeypatch, filesystem)

    with pytest.raises(RuntimeError, match="PRIVATE UNEXPECTED SCAN"):
        discover_import_sources([root], _bounds())

    assert filesystem.pin_calls > 0
    assert filesystem.live_pins == {}
    assert len(filesystem.closed_pins) == filesystem.pin_calls


def test_windows_adapter_closes_parent_pins_when_leaf_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "note.md"
    source.write_text("Body", encoding="utf-8")
    filesystem = FakeWindowsFilesystem()
    _force_windows_adapter(monkeypatch, filesystem)
    discovery = discover_import_sources([source], _bounds())
    pins_before_read = filesystem.pin_calls
    filesystem.fail_open_file = True

    with pytest.raises(VerifiedSourceReadError) as raised:
        read_discovered_source(discovery.candidates[0], _bounds())

    assert raised.value.reason_code == "source_unavailable"
    assert filesystem.pin_calls > pins_before_read
    assert filesystem.live_pins == {}
    assert len(filesystem.closed_pins) == filesystem.pin_calls


def test_windows_adapter_closes_leaf_and_parent_pins_on_unexpected_read_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "note.md"
    source.write_text("Body", encoding="utf-8")
    filesystem = FakeWindowsFilesystem()
    _force_windows_adapter(monkeypatch, filesystem)
    discovery = discover_import_sources([source], _bounds())
    filesystem.read_exception = RuntimeError("PRIVATE UNEXPECTED READ")

    with pytest.raises(RuntimeError, match="PRIVATE UNEXPECTED READ"):
        read_discovered_source(discovery.candidates[0], _bounds())

    assert filesystem.pin_calls > 0
    assert filesystem.live_pins == {}
    assert filesystem.open_file_paths == {}
    assert len(filesystem.closed_pins) == filesystem.pin_calls


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


def test_production_windows_filesystem_is_native_only_on_windows() -> None:
    native_type = getattr(
        note_import_windows_fs,
        "NativeWindowsReadOnlyFilesystem",
        None,
    )
    unavailable_type = getattr(
        note_import_windows_fs,
        "UnavailableWindowsReadOnlyFilesystem",
        None,
    )

    assert native_type is not None
    assert unavailable_type is not None
    expected_type = native_type if os.name == "nt" else unavailable_type
    assert isinstance(note_import_windows_fs.OS_WINDOWS_FILESYSTEM, expected_type)


@pytest.mark.parametrize(
    ("directory", "expected_share_mode"),
    [(False, 0x00000001), (True, 0x00000001 | 0x00000002)],
)
def test_native_windows_share_mode_is_scoped_by_handle_kind(
    directory: bool,
    expected_share_mode: int,
) -> None:
    share_mode_builder = getattr(
        note_import_windows_fs,
        "_native_windows_share_mode",
        None,
    )

    assert share_mode_builder is not None
    share_mode = share_mode_builder(directory=directory)
    assert share_mode == expected_share_mode
    assert share_mode & 0x00000004 == 0


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


@pytest.mark.skipif(
    os.name != "nt",
    reason="requires native Windows sharing and rename semantics",
)
def test_native_windows_directory_pin_denies_validation_scan_rename(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Project"
    root.mkdir()
    (root / "inside.md").write_text("inside", encoding="utf-8")
    moved = tmp_path / "moved-Project"
    native_type = note_import_windows_fs.NativeWindowsReadOnlyFilesystem

    class RenameAttemptFilesystem(native_type):  # type: ignore[misc, valid-type]
        rename_denied = False

        def scandir(self, path: Path) -> object:
            try:
                path.rename(moved)
            except OSError:
                self.rename_denied = True
            return super().scandir(path)

    filesystem = RenameAttemptFilesystem()

    discovery = note_import_windows_fs.discover_import_sources(
        [root],
        _bounds(),
        filesystem=filesystem,
    )

    assert filesystem.rename_denied
    assert not moved.exists()
    assert tuple(
        candidate.source.display_path for candidate in discovery.candidates
    ) == ("Project/inside.md",)


@pytest.mark.skipif(
    os.name != "nt",
    reason="requires native Windows sharing semantics",
)
def test_native_windows_source_handle_denies_concurrent_writer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "note.md"
    source.write_text("Body", encoding="utf-8")
    filesystem = note_import_windows_fs.NativeWindowsReadOnlyFilesystem()
    descriptor = filesystem.open_file_no_reparse(source)
    try:
        with pytest.raises(OSError):
            writer = os.open(source, os.O_WRONLY | getattr(os, "O_BINARY", 0))
            os.close(writer)
    finally:
        filesystem.close(descriptor)


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_windows_descriptor_cleanup_propagates_interruptions_after_closing_all(
    interruption: type[BaseException],
) -> None:
    """Windows cleanup closes every handle and then preserves an interruption."""

    class InterruptingFilesystem(FakeWindowsFilesystem):
        def close(self, descriptor: int) -> None:
            super().close(descriptor)
            if descriptor == 2:
                raise interruption()

    filesystem = InterruptingFilesystem()
    filesystem.live_pins = {1: Path("first"), 2: Path("second")}

    with pytest.raises(interruption):
        note_import_windows_fs._close_filesystem_descriptors(filesystem, (1, 2))

    assert filesystem.closed_pins == [2, 1]
    assert filesystem.live_pins == {}
