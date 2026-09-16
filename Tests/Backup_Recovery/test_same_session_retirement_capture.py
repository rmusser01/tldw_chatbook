"""Private capture keeps native exclusions across journal-proved retirements."""

import sys
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_held_sqlite_rollback import (
    replacement_case,
    run_capture,
)
from tldw_chatbook.Backup_Recovery import bootstrap, crypto, publication, replacement
from tldw_chatbook.Backup_Recovery.control_records import (
    UNBOUND_NAMESPACE,
    admission_authority,
)
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.storage_admission import copy_capture_file
from tldw_chatbook.Utils.platform_files import os as native_os


def private_file(path, data=b"local source"):
    path.write_bytes(data)
    path.chmod(0o600)
    return path


@pytest.mark.parametrize("held", [False, True])
def test_ordinary_capture_refuses_arbitrary_missing_registry_root(tmp_path, held):
    source = private_file(tmp_path / "source")
    unrelated = private_file(tmp_path / "unrelated")
    stage = tmp_path / "private"
    stage.mkdir(mode=0o700)
    authority = admission_authority(tmp_path / "bootstrap")
    authority.register("source", (source,))
    authority.register("unrelated", (unrelated,))
    names = ("source", "unrelated") if held else ("source",)
    limits = ArchiveLimits()
    with authority.maintenance(names, 3) as session:
        unrelated.unlink()
        with (
            pytest.raises(FileNotFoundError),
            session._capture_bound_sources((), stage, limits, limits.expanded_bytes),
        ):
            pytest.fail("An ordinary session cannot qualify an arbitrary disappearance")
    assert source.read_bytes() == b"local source"


@pytest.mark.parametrize("location", ["original", "current"])
def test_alias_retarget_preserves_original_and_current_staging_exclusions(
    tmp_path, location
):
    original = tmp_path / "original"
    current = tmp_path / "current"
    for directory in (original, current):
        directory.mkdir(mode=0o700)
        private_file(directory / "data")
    alias = tmp_path / "alias"
    alias.symlink_to(original, target_is_directory=True)
    authority = admission_authority(tmp_path / "bootstrap")
    if sys.platform == "win32":
        before = bootstrap._registry(tmp_path / "bootstrap")
        with pytest.raises(OSError, match="windows_reparse_point_refused"):
            authority.register("alias", (alias,))
        assert bootstrap._registry(tmp_path / "bootstrap") == before
        assert (original / "data").read_bytes() == b"local source"
        assert (current / "data").read_bytes() == b"local source"
        return
    authority.register("alias", (alias,))
    limits = ArchiveLimits()
    with authority.maintenance(("alias",), 3) as session:
        assert (original, True) in session._publication_roots
        alias.unlink()
        alias.symlink_to(current, target_is_directory=True)
        with (
            pytest.raises(
                bootstrap.RecoveryRequired, match="capture_staging_overlaps_source"
            ),
            session._capture_bound_sources(
                (),
                original if location == "original" else current,
                limits,
                limits.expanded_bytes,
            ),
        ):
            pytest.fail("Retargeting cannot clear either native staging exclusion")


@pytest.mark.parametrize("change", ["missing", "identity", "foreign"])
def test_private_capture_still_refuses_unproved_actual_source_identity(
    tmp_path, change
):
    source = private_file(tmp_path / "source")
    foreign = private_file(tmp_path / "foreign")
    stage = tmp_path / "private"
    stage.mkdir(mode=0o700)
    authority = admission_authority(tmp_path / "bootstrap")
    authority.register("source", (source,))
    limits = ArchiveLimits()
    info = native_os.stat(source)
    selected = ((source, info.st_dev, info.st_ino),)
    with (
        authority.maintenance(("source",), 3) as session,
        session._capture_bound_sources(selected, stage, limits, limits.expanded_bytes),
    ):
        if change == "missing":
            source.unlink()
        elif change == "identity":
            private_file(tmp_path / "new-source").replace(source)
        with pytest.raises((FileNotFoundError, bootstrap.RecoveryRequired)):
            copy_capture_file(
                "config",
                foreign if change == "foreign" else source,
                stage / "copy",
                Event(),
                max_bytes=1024,
            )
        assert not (stage / "copy").exists()


@pytest.mark.parametrize("damage", [None, "unrelated", "retained"])
def test_same_session_missing_root_needs_actual_durable_retirement_proof(
    tmp_path, monkeypatch, helper_resource_root, damage
):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    unrelated = private_file(tmp_path / "unrelated")

    def enroll_sidecar(live):
        authority = admission_authority(tmp_path / "bootstrap")
        authority.register("selected.shm", (live / "research.db-shm",))
        authority.register("unrelated", (unrelated,))
        return ()

    with replacement_case(tmp_path, monkeypatch, extras=enroll_sidecar) as case:
        candidate, plan, journal, session, source, _ = case
        sidecar = Path(str(source) + "-shm")
        assert sidecar.exists()
        archived = run_capture(case, tmp_path)
        publication.publish_candidate(
            candidate, plan, journal, archived, session=session
        )
        assert not sidecar.exists()
        with journal._locked(exclusive=False) as parent:
            before = journal._records(parent)
        prepared = publication._Prepared.model_validate(
            next(row.evidence for row in before if row.event == "prepared")
        )
        retired = next(row for row in prepared.artifacts if row.target == str(sidecar))
        if damage == "unrelated":
            unrelated.unlink()
        elif damage == "retained":
            Path(retired.retained).write_bytes(b"changed retained evidence")
        names = tuple(sorted((*prepared.publication.namespaces, UNBOUND_NAMESPACE)))
        if damage is not None:
            with pytest.raises(ValueError, match="recovery_root_unverified"):
                replacement._recovery_staging_roots(
                    journal,
                    session._control,
                    names,
                    session._all_roots,
                )
        else:
            assert sidecar in replacement._recovery_staging_roots(
                journal,
                session._control,
                names,
                session._all_roots,
            )
        with journal._locked(exclusive=False) as parent:
            assert journal._records(parent) == before
