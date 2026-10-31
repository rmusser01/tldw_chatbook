"""Detached acquisition binds the reviewed bytes without modifying the source."""

import hashlib
import os
import subprocess

import pytest


def test_acquisition_retains_all_files_and_source_bytes(tmp_path):
    from tldw_chatbook.Backup_Recovery.detached_source import (
        acquire_source,
        review_source,
    )

    source = tmp_path / "source ü"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "note").write_bytes(b"retained note")
    (source / "empty").mkdir()
    before = hashlib.sha256((source / "nested" / "note").read_bytes()).hexdigest()
    review = review_source(source)
    destination = tmp_path / "acquired"
    assert acquire_source(review, destination) == destination
    assert (destination / "nested" / "note").read_bytes() == b"retained note"
    assert (destination / "empty").is_dir()
    assert (
        hashlib.sha256((source / "nested" / "note").read_bytes()).hexdigest() == before
    )


@pytest.mark.parametrize("change", ["bytes", "add", "remove", "replace"])
def test_changed_review_is_refused_before_acquisition(tmp_path, change):
    from tldw_chatbook.Backup_Recovery.detached_source import (
        acquire_source,
        review_source,
    )

    source = tmp_path / "source"
    source.mkdir()
    member = source / "note"
    member.write_bytes(b"old")
    review = review_source(source)
    if change == "bytes":
        member.write_bytes(b"new")
    elif change == "add":
        (source / "new").write_bytes(b"new")
    elif change == "remove":
        member.unlink()
    else:
        member.unlink()
        member.write_bytes(b"old")
    destination = tmp_path / "acquired"
    with pytest.raises(ValueError, match="detached_source_changed"):
        acquire_source(review, destination)
    assert not destination.exists()


def test_hardlinked_source_is_refused(tmp_path):
    from tldw_chatbook.Backup_Recovery.detached_source import review_source

    source = tmp_path / "source"
    source.mkdir()
    (source / "one").write_bytes(b"private")
    os.link(source / "one", source / "two")
    with pytest.raises(ValueError, match="detached_source_unsafe"):
        review_source(source)


@pytest.mark.skipif(os.name != "nt", reason="native Windows junction boundary")
def test_native_junction_cannot_escape_source(tmp_path):
    from tldw_chatbook.Backup_Recovery.detached_source import review_source

    source = tmp_path / "source"
    target = tmp_path / "outside"
    source.mkdir()
    target.mkdir()
    (target / "retained").write_bytes(b"outside copied source")
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(source / "linked"), str(target)],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="detached_source_unsafe"):
        review_source(source)
    assert (target / "retained").read_bytes() == b"outside copied source"


def test_acquisition_refuses_destination_overlapping_source(tmp_path):
    from tldw_chatbook.Backup_Recovery.detached_source import (
        acquire_source,
        review_source,
    )

    source = tmp_path / "source"
    source.mkdir()
    (source / "note").write_bytes(b"retained")
    review = review_source(source)
    with pytest.raises(ValueError, match="detached_destination_overlap"):
        acquire_source(review, source / "staging")
    assert not (source / "staging").exists()


def test_installed_work_root_can_hold_acquisition_without_entering_review(tmp_path):
    from tldw_chatbook.Backup_Recovery.detached_source import (
        acquire_source,
        review_source,
    )

    source = tmp_path / "Data"
    source.mkdir()
    work = source / ".portable-relocation"
    work.mkdir()
    (source / "note").write_bytes(b"retained")
    review = review_source(source, excluded_roots=(".portable-relocation",))
    destination = work / "acquired"
    acquire_source(review, destination)
    assert (destination / "note").read_bytes() == b"retained"
    assert not (destination / ".portable-relocation").exists()


def test_change_during_copy_refuses_the_candidate(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import detached_source

    source = tmp_path / "source"
    source.mkdir()
    member = source / "note"
    member.write_bytes(b"retained")
    review = detached_source.review_source(source)
    original_hash = detached_source._hash

    def change_after_copy(fd, size, destination=None):
        digest = original_hash(fd, size, destination)
        if destination is not None:
            member.write_bytes(b"changed during copy")
        return digest

    monkeypatch.setattr(detached_source, "_hash", change_after_copy)
    with pytest.raises(ValueError, match="detached_source_changed"):
        detached_source.acquire_source(review, tmp_path / "acquired")
    assert member.read_bytes() == b"changed during copy"


def test_growing_file_is_bounded_before_eof(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import detached_source

    source = tmp_path / "source"
    source.mkdir()
    member = source / "note"
    member.write_bytes(b"old")
    inode = detached_source.os.stat(member).st_ino
    original = detached_source.os.read
    reads = 0

    def growing_read(fd, size):
        nonlocal reads
        chunk = original(fd, size)
        if detached_source.os.fstat(fd).st_ino == inode:
            reads += 1
            if reads > 3:
                raise RuntimeError("unbounded growing source")
            with member.open("ab") as output:
                output.write(b"growth")
        return chunk

    monkeypatch.setattr(detached_source.os, "read", growing_read)
    with pytest.raises(ValueError, match="detached_source_changed"):
        detached_source.review_source(source)
    assert reads <= 2


def test_nested_transient_cache_is_not_read_or_copied(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import detached_source

    source = tmp_path / "Data"
    cache = source / "home" / "AppData" / "Local"
    cache.mkdir(parents=True)
    (cache / "locked-driver-cache").write_bytes(b"transient")
    retained = cache.parent / "Roaming"
    retained.mkdir()
    (retained / "saved").write_bytes(b"must remain visible")
    inode = detached_source.os.stat(cache / "locked-driver-cache").st_ino
    original = detached_source._hash

    def refuse_cache_read(fd, size, destination=None):
        assert detached_source.os.fstat(fd).st_ino != inode
        return original(fd, size, destination)

    monkeypatch.setattr(detached_source, "_hash", refuse_cache_read)
    review = detached_source.review_source(
        source, excluded_roots=("home/AppData/Local",)
    )
    (cache / "locked-driver-cache").write_bytes(b"changed while GUI is open")
    destination = tmp_path / "acquired"
    detached_source.acquire_source(review, destination)
    assert not (destination / "home" / "AppData" / "Local").exists()
    assert (destination / "home" / "AppData" / "Roaming" / "saved").read_bytes() == (
        b"must remain visible"
    )


@pytest.mark.skipif(os.name != "nt", reason="native Windows junction boundary")
def test_nested_transient_exclusion_still_refuses_a_junction(tmp_path):
    from tldw_chatbook.Backup_Recovery.detached_source import review_source

    source = tmp_path / "Data"
    (source / "home" / "AppData").mkdir(parents=True)
    target = tmp_path / "outside"
    target.mkdir()
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(source / "home/AppData/Local"), str(target)],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="detached_source_unsafe"):
        review_source(source, excluded_roots=("home/AppData/Local",))
