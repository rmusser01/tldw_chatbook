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
from tldw_chatbook.Tool_Packs.export import ToolPackExportService, ToolPackExportSnapshot
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
                {"schema_version": 1, "profiles": {"default": {"global_default": "deny", "servers": {}}}}
            ),
            generation="sha256:" + "0" * 64,
            file_exists=True,
        )


@pytest.fixture
def snapshot(monkeypatch: pytest.MonkeyPatch) -> ToolPackExportSnapshot:
    registry = PermissionInventoryRegistry(current_permission_namespaces=lambda: {"local:docs"})
    registry.register(_Adapter())
    monkeypatch.setattr(export_module, "capture_v1_inventory", lambda value: value.capture())
    return ToolPackExportService(_Store(), registry).capture(
        profile_id="default", display_name="Default", suggested_id="default"
    ).snapshot


def test_missing_destination_appearing_after_capture_is_refused(
    tmp_path: Path, snapshot: ToolPackExportSnapshot
) -> None:
    """Removing destination revalidation would replace a file the user never chose."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    destination.write_bytes(b"appeared")

    with pytest.raises(ToolPackError, match=r"^tool_pack\.export\.destination_changed$"):
        publish_tool_pack(snapshot, captured, overwrite=False)

    assert destination.read_bytes() == b"appeared"


def test_publish_writes_a_complete_archive_to_an_absent_destination(
    tmp_path: Path, snapshot: ToolPackExportSnapshot
) -> None:
    """Skipping atomic publication would expose incomplete deterministic archives."""
    destination = tmp_path / "research.tldw-tool-pack"

    result = publish_tool_pack(snapshot, CapturedToolPackDestination.capture(destination))

    assert result.committed is True
    assert result.durability_uncertain is False
    assert result.archive_sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_overwrite_requires_the_exact_token_for_the_captured_incumbent(
    tmp_path: Path, snapshot: ToolPackExportSnapshot
) -> None:
    """Accepting any confirmation could overwrite a different destination file."""
    destination = tmp_path / "research.tldw-tool-pack"
    destination.write_bytes(b"incumbent")
    captured = CapturedToolPackDestination.capture(destination)

    for supplied in (None, "wrong"):
        with pytest.raises(ToolPackError, match=r"destination_changed$"):
            publish_tool_pack(snapshot, captured, overwrite=True, overwrite_token=supplied)
        assert destination.read_bytes() == b"incumbent"

    result = publish_tool_pack(
        snapshot, captured, overwrite=True, overwrite_token=captured.overwrite_token
    )

    assert result.committed is True
    assert destination.read_bytes() != b"incumbent"


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
    destination.write_bytes(b"incumbent")
    captured = CapturedToolPackDestination.capture(destination)

    def substitute(event: str) -> None:
        if event == phase:
            replacement = tmp_path / f"replacement-{phase}"
            replacement.write_bytes(b"replacement")
            os.replace(replacement, destination)

    with pytest.raises(ToolPackError, match=r"destination_changed$"):
        publish_tool_pack(
            snapshot,
            captured,
            overwrite=True,
            overwrite_token=captured.overwrite_token,
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

    def fail_parent_lstat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        if path == destination.parent:
            raise OSError("private parent identity failure")
        return real_lstat(path, *args, **kwargs)

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.open", record_open)
    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.lstat", fail_parent_lstat)
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
        publish_tool_pack(snapshot, captured, cancelled=lambda: True, phase_hook=inspect)

    assert len(observed) == 1
    assert observed[0].parent == destination.parent
    assert observed_mode == [0o600]


def test_file_fsync_precedes_replace_and_parent_fsync(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing before archive fsync could make a crash publish a partial archive."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def record_fsync(descriptor: int) -> None:
        events.append("fsync")
        real_fsync(descriptor)

    def record_replace(*args: object, **kwargs: object) -> None:
        events.append("replace")
        real_replace(*args, **kwargs)

    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.fsync", record_fsync)
    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.replace", record_replace)

    publish_tool_pack(snapshot, captured)

    assert events == ["fsync", "replace", "fsync"]


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

    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.fsync", fail_parent_fsync)

    result = publish_tool_pack(snapshot, captured)

    assert result.committed is True
    assert result.durability_uncertain is True
    assert result.archive_sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()


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
        if calls == 2:
            raise OSError("private parent close failure")
        real_close(descriptor)

    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.close", fail_parent_close)

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

    with pytest.raises(ToolPackError, match=r"^tool_pack\.export\.durability_uncertain$"):
        publish_tool_pack(snapshot, captured)

    assert (displaced / destination.name).is_file()
    assert not destination.exists()


def test_failed_replace_reports_publication_failure_only_when_exact_old_target_remains(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reporting old preservation without exact reconciliation would hide a commit race."""
    destination = tmp_path / "research.tldw-tool-pack"
    destination.write_bytes(b"incumbent")
    captured = CapturedToolPackDestination.capture(destination)

    def fail_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError("private replace failure")

    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.replace", fail_replace)

    with pytest.raises(ToolPackError, match=r"^tool_pack\.export\.publication_failed$") as caught:
        publish_tool_pack(
            snapshot, captured, overwrite=True, overwrite_token=captured.overwrite_token
        )

    assert "private replace failure" not in str(caught.value)
    assert destination.read_bytes() == b"incumbent"


def test_ambiguous_post_replace_state_reports_durability_uncertain(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A third post-replace state cannot safely be reported as old or committed."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    real_replace = os.replace

    def replace_then_mutate(*args: object, **kwargs: object) -> None:
        real_replace(*args, **kwargs)
        destination.write_bytes(b"third state")
        raise OSError("private replace failure")

    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.replace", replace_then_mutate)

    with pytest.raises(ToolPackError, match=r"^tool_pack\.export\.durability_uncertain$") as caught:
        publish_tool_pack(snapshot, captured)

    assert str(destination) not in str(caught.value)


def test_post_replace_reconciliation_requires_the_replaced_temp_identity(
    tmp_path: Path, snapshot: ToolPackExportSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-byte third file must not be mistaken for this publication's commit."""
    destination = tmp_path / "research.tldw-tool-pack"
    captured = CapturedToolPackDestination.capture(destination)
    real_replace = os.replace
    real_fsync = os.fsync
    fsync_calls = 0

    def replace_then_substitute(*args: object, **kwargs: object) -> None:
        real_replace(*args, **kwargs)
        clone = tmp_path / "same-bytes-clone"
        clone.write_bytes(destination.read_bytes())
        real_replace(clone, destination)

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("private parent failure")
        real_fsync(descriptor)

    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.replace", replace_then_substitute)
    monkeypatch.setattr("tldw_chatbook.Tool_Packs.publication.os.fsync", fail_parent_fsync)

    with pytest.raises(ToolPackError, match=r"^tool_pack\.export\.durability_uncertain$"):
        publish_tool_pack(snapshot, captured)
