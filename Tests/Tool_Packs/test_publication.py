"""Security regressions for captured Tool Pack archive publication."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import stat

import pytest

from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventoryAdapter,
    PermissionInventoryRegistry,
)
from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.Tool_Packs import export as export_module
from tldw_chatbook.Tool_Packs import publication as publication_module
from tldw_chatbook.Tool_Packs.export import (
    ToolPackExportService,
    ToolPackExportSnapshot,
)
from tldw_chatbook.Tool_Packs.publication import (
    CapturedToolPackDestination,
    ToolPackPublicationPrimitives,
    publish_tool_pack,
)


class _Adapter(PermissionInventoryAdapter):
    namespace = "local:docs"
    complete = True

    def snapshot(self) -> tuple[object, ...]:
        return ()


class _Store:
    def read_snapshot_strict(self) -> object:
        from tldw_chatbook.MCP.permission_store import PermissionStoreSnapshot
        from types import MappingProxyType

        return PermissionStoreSnapshot(
            payload=MappingProxyType(
                {
                    "schema_version": 1,
                    "profiles": {"default": {"global_default": "deny", "servers": {}}},
                }
            ),
            generation="sha256:" + "0" * 64,
            file_exists=True,
        )


@pytest.fixture
def snapshot(monkeypatch: pytest.MonkeyPatch) -> ToolPackExportSnapshot:
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    registry.register(_Adapter())
    monkeypatch.setattr(
        export_module, "capture_v1_inventory", lambda value: value.capture()
    )
    return (
        ToolPackExportService(_Store(), registry)
        .capture(profile_id="default", display_name="Default", suggested_id="default")
        .snapshot
    )


def test_missing_destination_appearing_after_capture_is_refused(
    tmp_path: Path, snapshot: ToolPackExportSnapshot
) -> None:
    """Removing destination revalidation would replace a file the user never chose."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    destination.write_bytes(b"appeared")

    with pytest.raises(
        ToolPackError, match=r"^tool_pack\.export\.destination_changed$"
    ):
        publish_tool_pack(snapshot, captured, overwrite=False)

    assert destination.read_bytes() == b"appeared"


def test_publish_writes_a_complete_archive_to_an_absent_destination(
    tmp_path: Path, snapshot: ToolPackExportSnapshot
) -> None:
    """Skipping atomic publication would expose incomplete deterministic archives."""
    destination = tmp_path / "research.tldw-tool-pack"

    result = publish_tool_pack(
        snapshot, CapturedToolPackDestination.capture(destination)
    )

    assert result.committed is True
    assert result.durability_uncertain is False
    assert result.archive_sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_capture_uses_the_path_returned_by_central_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "selected.tldw-tool-pack"
    normalized = tmp_path / "normalized.tldw-tool-pack"
    calls: list[tuple[Path, Path, bool]] = []

    def validate_path(path: Path, base: Path, *, redact_paths: bool) -> Path:
        calls.append((path, base, redact_paths))
        return normalized

    monkeypatch.setattr(
        publication_module, "validate_path", validate_path, raising=False
    )

    captured = CapturedToolPackDestination.capture(selected)

    assert calls == [(selected, selected.parent, True)]
    assert captured.path == normalized


def test_capture_rejects_an_invalid_path_returned_by_central_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "selected.tldw-tool-pack"
    normalized = tmp_path / "normalized.txt"
    monkeypatch.setattr(
        publication_module, "validate_path", lambda *_args, **_kwargs: normalized
    )

    with pytest.raises(ToolPackError, match=r"destination_invalid$"):
        CapturedToolPackDestination.capture(selected)


def test_existing_destination_fails_closed_even_with_the_exact_overwrite_token(
    tmp_path: Path, snapshot: ToolPackExportSnapshot
) -> None:
    """POSIX replace cannot atomically compare the incumbent with captured evidence."""
    destination = tmp_path / "research.tldw-tool-pack"
    destination.write_bytes(b"incumbent")
    captured = CapturedToolPackDestination.capture(destination)

    with pytest.raises(ToolPackError, match=r"publication_unsupported$"):
        publish_tool_pack(
            snapshot,
            captured,
            overwrite=True,
            overwrite_token=captured.overwrite_token,
        )

    assert destination.read_bytes() == b"incumbent"


def test_target_appearing_at_atomic_publish_boundary_is_not_replaced(
    tmp_path: Path,
    snapshot: ToolPackExportSnapshot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    primitives = ToolPackPublicationPrimitives.current()
    real_link = os.link

    def create_incumbent_then_link(*args: object, **kwargs: object) -> None:
        destination.write_bytes(b"appeared at boundary")
        real_link(*args, **kwargs)

    monkeypatch.setattr(publication_module.os, "link", create_incumbent_then_link)

    with pytest.raises(ToolPackError, match=r"destination_changed$"):
        publish_tool_pack(snapshot, captured, primitives=primitives)

    assert destination.read_bytes() == b"appeared at boundary"


def test_overwrite_refuses_an_incumbent_rewritten_in_place_after_capture(
    tmp_path: Path, snapshot: ToolPackExportSnapshot
) -> None:
    """An old overwrite token must not authorize different bytes on the same inode."""
    destination = tmp_path / "research.tldw-tool-pack"
    destination.write_bytes(b"incumbent")
    captured = CapturedToolPackDestination.capture(destination)
    captured_identity = (destination.stat().st_dev, destination.stat().st_ino)
    destination.write_bytes(b"rewritten!")

    with pytest.raises(
        ToolPackError, match=r"^tool_pack\.export\.publication_unsupported$"
    ):
        publish_tool_pack(
            snapshot,
            captured,
            overwrite=True,
            overwrite_token=captured.overwrite_token,
        )

    assert (destination.stat().st_dev, destination.stat().st_ino) == captured_identity
    assert destination.read_bytes() == b"rewritten!"


@pytest.mark.parametrize("kind", ("symlink", "directory"))
def test_capture_rejects_nonregular_destinations(tmp_path: Path, kind: str) -> None:
    """Following special targets could write outside the picker-confirmed boundary."""
    destination = tmp_path / "research.tldw-tool-pack"
    if kind == "symlink":
        target = tmp_path / "private"
        target.write_bytes(b"private")
        destination.symlink_to(target)
    else:
        destination.mkdir()

    with pytest.raises(ToolPackError, match=r"destination_invalid$"):
        CapturedToolPackDestination.capture(destination)


@pytest.mark.parametrize("phase", ("archive_fsynced", "before_replace"))
def test_target_substitution_before_replace_is_refused(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, phase: str
) -> None:
    """Removing the final target check could replace a target selected after capture."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)

    def substitute(event: str) -> None:
        if event == phase:
            destination.write_bytes(b"replacement")

    with pytest.raises(ToolPackError, match=r"destination_changed$"):
        publish_tool_pack(
            snapshot,
            captured,
            phase_hook=substitute,
        )

    assert destination.read_bytes() == b"replacement"
    assert not list(tmp_path.glob(".*.tmp"))


def test_parent_substitution_is_refused_and_owned_temp_is_removed(
    tmp_path: Path, snapshot: ToolPackExportSnapshot
) -> None:
    """Using a path after its parent moves could publish into an attacker directory."""
    parent = tmp_path / "exports"
    parent.mkdir()
    destination = parent / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    displaced = tmp_path / "displaced"

    def substitute(event: str) -> None:
        if event == "archive_fsynced":
            os.rename(parent, displaced)
            parent.mkdir()

    with pytest.raises(ToolPackError, match=r"destination_changed$"):
        publish_tool_pack(snapshot, captured, phase_hook=substitute)

    assert not destination.exists()
    assert not list(displaced.glob(".*.tmp"))


def test_failed_parent_identity_check_closes_its_opened_descriptor(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed parent recheck must not leak the descriptor opened for publication."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    primitives = ToolPackPublicationPrimitives.current()
    real_open = os.open
    real_close = os.close
    real_lstat = os.lstat
    opened: list[int] = []
    closed: list[int] = []

    def record_open(*args: object, **kwargs: object) -> int:
        descriptor = real_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def fail_parent_lstat(
        path: object, *args: object, **kwargs: object
    ) -> os.stat_result:
        if path == destination.parent:
            raise OSError("private parent identity failure")
        return real_lstat(path, *args, **kwargs)

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.open", record_open)
    monkeypatch.setattr(
        "tldw_chatbook.Tool_Packs.publication.os.lstat", fail_parent_lstat
    )
    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.close", record_close)

    with pytest.raises(ToolPackError, match=r"destination_changed$"):
        publish_tool_pack(snapshot, captured, primitives=primitives)

    assert opened == closed


def test_cancellation_before_replace_preserves_destination_and_removes_temp(
    tmp_path: Path, snapshot: ToolPackExportSnapshot
) -> None:
    """Ignoring cancellation could commit an export after its caller abandoned it."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)

    with pytest.raises(ToolPackError, match=r"cancelled$"):
        publish_tool_pack(snapshot, captured, cancelled=lambda: True)

    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_missing_secure_primitive_fails_before_any_destination_mutation(
    tmp_path: Path, snapshot: ToolPackExportSnapshot
) -> None:
    """A fallback without no-follow protections must not publish at all."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    unavailable = replace(ToolPackPublicationPrimitives.current(), nofollow=False)

    with pytest.raises(ToolPackError, match=r"publication_unsupported$"):
        publish_tool_pack(snapshot, captured, primitives=unavailable)

    assert not destination.exists()


def test_missing_descriptor_relative_link_is_unsupported_before_mutation(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual no-replace primitive must accept directory descriptors."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)

    def link_without_directory_descriptors(source: object, target: object) -> None:
        raise AssertionError(f"unexpected publication: {source!r}, {target!r}")

    monkeypatch.setattr(
        "tldw_chatbook.Tool_Packs.publication.os.link",
        link_without_directory_descriptors,
    )

    with pytest.raises(
        ToolPackError, match=r"^tool_pack\.export\.publication_unsupported$"
    ):
        publish_tool_pack(snapshot, captured)

    assert not destination.exists()


def test_private_temp_has_mode_0600_and_shares_the_destination_parent(
    tmp_path: Path, snapshot: ToolPackExportSnapshot
) -> None:
    """A globally readable or cross-directory temporary would leak or lose atomicity."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    observed: list[Path] = []
    observed_mode: list[int] = []

    def inspect(event: str) -> None:
        if event == "archive_fsynced":
            observed.extend(tmp_path.glob(".*.tmp"))
            observed_mode.extend(stat.S_IMODE(item.stat().st_mode) for item in observed)

    with pytest.raises(ToolPackError, match=r"cancelled$"):
        publish_tool_pack(
            snapshot, captured, cancelled=lambda: True, phase_hook=inspect
        )

    assert len(observed) == 1
    assert observed[0].parent == destination.parent
    assert observed_mode == [0o600]


def test_file_fsync_precedes_link_and_parent_fsync(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linking before archive fsync could make a crash publish a partial archive."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    primitives = ToolPackPublicationPrimitives.current()
    events: list[str] = []
    real_fsync = os.fsync
    real_link = os.link

    def record_fsync(descriptor: int) -> None:
        events.append("fsync")
        real_fsync(descriptor)

    def record_link(
        source: object,
        target: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        events.append("link")
        real_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.fsync", record_fsync)
    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.link", record_link)

    publish_tool_pack(snapshot, captured, primitives=primitives)

    assert events == ["fsync", "link", "fsync"]


def test_cleanup_never_unlinks_a_substituted_temp(
    tmp_path: Path, snapshot: ToolPackExportSnapshot
) -> None:
    """Deleting a name without identity verification could remove another file."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    replacement: Path | None = None

    def substitute(event: str) -> None:
        nonlocal replacement
        if event == "archive_fsynced":
            temporary = next(tmp_path.glob(".*.tmp"))
            issued = tmp_path / "issued"
            os.rename(temporary, issued)
            replacement = temporary
            replacement.write_bytes(b"unrelated")

    with pytest.raises(ToolPackError, match=r"publication_failed$"):
        publish_tool_pack(snapshot, captured, phase_hook=substitute)

    assert replacement is not None and replacement.read_bytes() == b"unrelated"
    assert not destination.exists()


def test_parent_fsync_failure_reconciles_exact_new_archive_as_committed(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Treating a post-replace fsync error as no commit would misreport publication."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    real_fsync = os.fsync
    calls = 0

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("private parent failure")
        real_fsync(descriptor)

    monkeypatch.setattr(
        "tldw_chatbook.Tool_Packs.publication.os.fsync", fail_parent_fsync
    )

    result = publish_tool_pack(snapshot, captured)

    assert result.committed is True
    assert result.durability_uncertain is True
    assert result.archive_sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_post_link_cleanup_failure_is_durability_uncertain(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed link must not be misreported when staging-name cleanup fails."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    primitives = ToolPackPublicationPrimitives.current()
    real_unlink = os.unlink

    def fail_temporary_unlink(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        if isinstance(path, str) and path.startswith("."):
            raise OSError("private staging cleanup failure")
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(publication_module.os, "unlink", fail_temporary_unlink)

    with pytest.raises(
        ToolPackError, match=r"^tool_pack\.export\.durability_uncertain$"
    ):
        publish_tool_pack(snapshot, captured, primitives=primitives)

    assert destination.is_file()
    assert destination.read_bytes().startswith(b"PK")
    assert len(list(tmp_path.glob(".*.tmp"))) == 1


def test_parent_close_error_is_not_exposed_after_a_durable_publication(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Descriptor cleanup must not leak private filesystem errors after commit."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    real_close = os.close
    calls = 0

    def fail_parent_close(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("private parent close failure")
        real_close(descriptor)

    monkeypatch.setattr(
        "tldw_chatbook.Tool_Packs.publication.os.close", fail_parent_close
    )

    result = publish_tool_pack(snapshot, captured)

    assert result.committed is True
    assert result.durability_uncertain is False
    assert destination.is_file()


def test_post_replace_parent_substitution_is_durability_uncertain(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path-based reconciliation must not treat a substituted parent as old state."""
    parent = tmp_path / "exports"
    parent.mkdir()
    destination = parent / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    displaced = tmp_path / "displaced"
    real_fsync = os.fsync
    calls = 0

    def substitute_parent_then_fail(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            os.rename(parent, displaced)
            parent.mkdir()
            raise OSError("private parent failure")
        real_fsync(descriptor)

    monkeypatch.setattr(
        "tldw_chatbook.Tool_Packs.publication.os.fsync", substitute_parent_then_fail
    )

    with pytest.raises(
        ToolPackError, match=r"^tool_pack\.export\.durability_uncertain$"
    ):
        publish_tool_pack(snapshot, captured)

    assert (displaced / destination.name).is_file()
    assert not destination.exists()


def test_parent_substitution_at_link_boundary_is_durability_uncertain(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful link must still attest that the captured pathname names it."""
    parent = tmp_path / "exports"
    parent.mkdir()
    destination = parent / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    displaced = tmp_path / "displaced"
    primitives = ToolPackPublicationPrimitives.current()
    real_link = os.link

    def substitute_parent_then_link(
        source: object,
        target: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        os.rename(parent, displaced)
        parent.mkdir()
        real_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Tool_Packs.publication.os.link", substitute_parent_then_link
    )

    with pytest.raises(
        ToolPackError, match=r"^tool_pack\.export\.durability_uncertain$"
    ):
        publish_tool_pack(snapshot, captured, primitives=primitives)

    assert (displaced / destination.name).is_file()
    assert not destination.exists()


def test_failed_link_reports_publication_failure_when_destination_remains_absent(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed link is non-committed only when exact absence is reconciled."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    primitives = ToolPackPublicationPrimitives.current()

    def fail_link(
        _source: object,
        _target: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        del src_dir_fd, dst_dir_fd, follow_symlinks
        raise OSError("private link failure")

    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.link", fail_link)

    with pytest.raises(
        ToolPackError, match=r"^tool_pack\.export\.publication_failed$"
    ) as caught:
        publish_tool_pack(snapshot, captured, primitives=primitives)

    assert "private link failure" not in str(caught.value)
    assert not destination.exists()


def test_ambiguous_post_link_state_reports_durability_uncertain(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A third post-replace state cannot safely be reported as old or committed."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    primitives = ToolPackPublicationPrimitives.current()
    real_link = os.link

    def link_then_mutate(
        source: object,
        target: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        real_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        destination.write_bytes(b"third state")
        raise OSError("private link failure")

    monkeypatch.setattr(
        "tldw_chatbook.Tool_Packs.publication.os.link", link_then_mutate
    )

    with pytest.raises(
        ToolPackError, match=r"^tool_pack\.export\.durability_uncertain$"
    ) as caught:
        publish_tool_pack(snapshot, captured, primitives=primitives)

    assert str(destination) not in str(caught.value)


def test_post_link_reconciliation_requires_the_published_temp_identity(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-byte third file must not be mistaken for this publication's commit."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    primitives = ToolPackPublicationPrimitives.current()
    real_link = os.link
    real_replace = os.replace
    real_fsync = os.fsync
    fsync_calls = 0

    def link_then_substitute(
        source: object,
        target: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        real_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        clone = tmp_path / "same-bytes-clone"
        clone.write_bytes(destination.read_bytes())
        real_replace(clone, destination)

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("private parent failure")
        real_fsync(descriptor)

    monkeypatch.setattr(
        "tldw_chatbook.Tool_Packs.publication.os.link", link_then_substitute
    )
    monkeypatch.setattr(
        "tldw_chatbook.Tool_Packs.publication.os.fsync", fail_parent_fsync
    )

    with pytest.raises(
        ToolPackError, match=r"^tool_pack\.export\.durability_uncertain$"
    ):
        publish_tool_pack(snapshot, captured, primitives=primitives)
