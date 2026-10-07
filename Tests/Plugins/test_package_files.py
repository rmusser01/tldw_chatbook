"""Exercise capture and materialization against actual filesystem resources."""

import os
import stat

import pytest

from tldw_chatbook.Plugins.inspection import inspect_package


def test_materialization_preserves_bytes_mode_and_digest(native_package, tmp_path):
    from tldw_chatbook.Plugins.package_files import materialize_package

    root = native_package()
    script = root / "run"
    script.write_bytes(b"must never execute\n")
    script.chmod(0o755)
    before = inspect_package(root)
    destination = tmp_path / "snapshot"
    result = materialize_package(root, destination)
    assert (destination / "run").read_bytes() == b"must never execute\n"
    assert (destination / "run").stat().st_mode & stat.S_IXUSR
    assert result.content_digest == before.content_digest
    assert result.content_digest == inspect_package(destination).content_digest
    script.chmod(0o644)
    assert inspect_package(root).content_digest != before.content_digest
    with pytest.raises(ValueError, match="destination_exists"):
        materialize_package(root, destination)
    assert (destination / "run").read_bytes() == b"must never execute\n"


def test_internal_file_link_materializes_as_regular_file(native_package, tmp_path):
    from tldw_chatbook.Plugins.package_files import materialize_package

    root = native_package()
    (root / "alias").symlink_to("skills/review/SKILL.md")
    result = materialize_package(root, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot" / "alias").is_symlink()
    assert (tmp_path / "snapshot" / "alias").read_bytes() == (
        root / "skills/review/SKILL.md"
    ).read_bytes()
    assert "internal_link_materialized" in {d.code for d in result.diagnostics}
    assert (
        result.content_digest == inspect_package(tmp_path / "snapshot").content_digest
    )


@pytest.mark.parametrize("kind", ["external", "directory", "fifo", "broken"])
def test_unsafe_entries_abort_materialization_with_no_destination(
    kind, native_package, tmp_path
):
    from tldw_chatbook.Plugins.package_files import materialize_package

    root = native_package()
    path = root / "unsafe"
    if kind == "external":
        outside = tmp_path / "secret"
        outside.write_text("private")
        path.symlink_to(outside)
    elif kind == "directory":
        path.symlink_to(root / "skills", target_is_directory=True)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.symlink_to("absent")
    with pytest.raises(ValueError):
        materialize_package(root, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()
    path.unlink()
    assert materialize_package(root, tmp_path / "snapshot").content_digest


def test_escaping_skill_has_narrow_inspection_boundary(native_package, tmp_path):
    root = native_package()
    outside = tmp_path / "outside"
    outside.write_text("secret")
    child = root / "skills" / "escape"
    child.mkdir()
    (child / "SKILL.md").symlink_to(outside)
    result = inspect_package(root)
    assert not result.rejected
    assert set(result.inventory) == {"skill:review"}
    assert not result.inventory["skill:review"].activation_blockers
    assert result.content_digest is None
    assert "package_path_escape" in {d.code for d in result.diagnostics}


@pytest.mark.parametrize(
    "limit,value",
    [
        ("MAX_FILES", 1),
        ("MAX_FILE_BYTES", 20),
        ("MAX_PACKAGE_BYTES", 100),
        ("MAX_PATH_DEPTH", 2),
    ],
)
def test_streamed_snapshot_limits_abort(
    limit, value, native_package, tmp_path, monkeypatch
):
    from tldw_chatbook.Plugins import package_files

    root = native_package()
    monkeypatch.setattr(package_files, limit, value)
    with pytest.raises(ValueError, match="limit"):
        package_files.materialize_package(root, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_case_collision_is_rejected_before_destination_write(native_package, tmp_path):
    from tldw_chatbook.Plugins.package_files import materialize_package

    root = native_package()
    # APFS may be case-insensitive: differently cased ancestor spellings are also ambiguous.
    (root / "CON.txt").write_text("reserved on Windows")
    with pytest.raises(ValueError, match="package_path_invalid"):
        materialize_package(root, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_destination_ancestor_link_and_nested_destination_are_rejected(
    native_package, tmp_path
):
    from tldw_chatbook.Plugins.package_files import materialize_package

    root = native_package()
    (tmp_path / "alias").symlink_to(root, target_is_directory=True)
    for destination in (tmp_path / "alias" / "snapshot", root / "snapshot"):
        with pytest.raises(ValueError):
            materialize_package(root, destination)
    assert not (root / "snapshot").exists()


def test_link_provenance_is_distinct_from_materialized_content(
    native_package, tmp_path
):
    from tldw_chatbook.Plugins.package_files import materialize_package

    root = native_package()
    (root / "alias").symlink_to("skills/review/SKILL.md")
    original = inspect_package(root)
    destination = tmp_path / "snapshot"
    copied = materialize_package(root, destination)
    on_disk = inspect_package(destination)
    assert copied.source_digest == original.source_digest
    assert copied.source_digest != on_disk.source_digest
    assert copied.content_digest == on_disk.content_digest
    assert copied.effective_digest == on_disk.effective_digest
    assert copied.materialized_identity == str(destination)
    assert copied.link_targets == {"alias": "skills/review/SKILL.md"}
    with pytest.raises(TypeError):
        copied.link_targets["alias"] = "other"


def test_case_and_unicode_collisions_are_detected_on_capture(native_package, tmp_path):
    from tldw_chatbook.Plugins.package_files import materialize_package

    # A two-entry collision requires a filesystem that can represent both.
    # Hosts that prohibit the pair only qualify the lexical comparison here.
    for first, second in (("Case", "case"), ("\u00e9", "e\u0301")):
        root = native_package()
        (root / first).write_text("one")
        try:
            fd = os.open(root / second, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # The host already prevents this collision. Lexical rejection is
            # separately exercised below without claiming a two-file fixture.
            from tldw_chatbook.Plugins.package_files import collision_key

            assert collision_key(first) == collision_key(second)
            continue
        os.close(fd)
        with pytest.raises(ValueError, match="collision"):
            materialize_package(root, tmp_path / f"snapshot-{first}")


def test_destination_write_does_not_follow_replaced_directory(
    native_package, tmp_path, monkeypatch
):
    from tldw_chatbook.Plugins import package_files

    root = native_package()
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "snapshot"
    original_open = os.open
    attacked = False

    def swap(name, flags, *args, **kwargs):
        nonlocal attacked
        if not attacked and flags & os.O_CREAT and destination.exists():
            target = destination / "skills" / "review"
            if target.exists():
                target.rmdir()
                target.symlink_to(outside, target_is_directory=True)
                attacked = True
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(package_files.os, "open", swap)
    # Preserve capability classification of the wrapper so failure must reach
    # the actual path substitution rather than a platform precondition.
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {swap})
    with pytest.raises(ValueError):
        package_files.materialize_package(root, destination)
    assert attacked
    assert not list(outside.iterdir())
    assert not destination.exists()


def test_file_count_limit_counts_files_without_charging_parent_directories(
    native_package, tmp_path, monkeypatch
):
    from tldw_chatbook.Plugins import package_files

    root = native_package()
    monkeypatch.setattr(package_files, "MAX_FILES", 2)
    assert package_files.materialize_package(root, tmp_path / "snapshot").content_digest


@pytest.mark.parametrize("directories", [False, True])
def test_rejected_entries_consume_traversal_and_diagnostic_budgets(
    native_package, tmp_path, monkeypatch, directories
):
    from tldw_chatbook.Plugins import package_files

    monkeypatch.setattr(package_files, "MAX_FILES", 2)
    control = inspect_package(native_package())
    assert not control.rejected and control.content_digest
    root = tmp_path / "invalid-names"
    root.mkdir()
    for index in range(8):
        path = root / f"CON.{index}"
        if directories:
            path.mkdir()
        else:
            path.write_text("invalid portable spelling")
    result = inspect_package(root)
    assert result.rejected
    expected = (
        "package_diagnostics_limit" if directories else "package_file_count_limit"
    )
    assert expected in result.activation_blockers
    assert len(result.diagnostics) <= 2
    with pytest.raises(ValueError, match=expected):
        package_files.materialize_package(root, tmp_path / "snapshot-invalid")
    assert not (tmp_path / "snapshot-invalid").exists()


def test_stat_failures_consume_visited_entry_budget(native_package, monkeypatch):
    from tldw_chatbook.Plugins import package_files

    root = native_package()
    monkeypatch.setattr(package_files, "MAX_FILES", 2)
    assert not inspect_package(root).rejected
    real_scandir = os.scandir
    visits = 0

    class UnreadableEntry:
        name = "unreadable"

        def stat(self, **kwargs):
            raise OSError("controlled unreadable entry")

    class UnreadableScan:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            nonlocal visits
            for _ in range(100):
                visits += 1
                yield UnreadableEntry()

    def unreadable_scan(descriptor):
        return (
            UnreadableScan()
            if isinstance(descriptor, int)
            else real_scandir(descriptor)
        )

    monkeypatch.setattr(package_files.os, "scandir", unreadable_scan)
    result = inspect_package(root)
    assert result.rejected
    assert "package_file_count_limit" in result.activation_blockers
    assert visits == 3
