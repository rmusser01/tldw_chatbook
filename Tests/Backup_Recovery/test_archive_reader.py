"""Temporary-file adversarial recovery container checks."""

from threading import Event
import pytest
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.archive_reader import acquire


def test_input_limit_applies_before_archive_parsing(tmp_path):
    source = tmp_path / "oversized"
    source.write_bytes(b"x" * 1025)
    with pytest.raises(ValueError, match="input_limit"):
        acquire(
            source, tmp_path / "work", ArchiveLimits(input_bytes=1024), None, Event()
        )


import hashlib
import json
import stat
import zipfile


def manifest(data=b"hello"):
    return dict(
        format_version=1,
        producer_version="0.1",
        captured_at="2026-09-10T00:00:00Z",
        profile_ids=["profile"],
        owners=[dict(owner_id="notes", schema_version=1, capabilities=[])],
        directories=[
            dict(
                logical_id="root",
                root_id="root",
                parent_id=None,
                relative_path="",
                metadata=dict(version=1, mtime_ns=0, mode=448),
            )
        ],
        files=[
            dict(
                logical_id="file",
                root_id="root",
                parent_id="root",
                relative_path="note.txt",
                owner_id="notes",
                payload="payload/1",
                size=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        ],
        dependency_groups=[dict(group_id="main", members=["file"], complete=True)],
        consistency="coherent",
        exclusions=[],
        credential_policy="exclude",
        required_capabilities=[],
        report=dict(version=1, lines=["Recovery archive"]),
        relocations=[],
    )


def archive(
    tmp_path, doc=None, data=b"hello", extra=None, compression=zipfile.ZIP_STORED
):
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w", compression=compression) as z:
        z.writestr("manifest.json", json.dumps(doc or manifest(data)))
        z.writestr("payload/1", data)
        if extra:
            z.writestr(*extra)
    return source


def test_sealed_copy_survives_original_replacement_and_rejects_staged_mutation(
    tmp_path,
):
    from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed

    source = archive(tmp_path)
    sealed = acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())
    source.write_bytes(b"replaced")
    assert verify_sealed(sealed).format_version == 1
    assert sealed.path.stat().st_mode & 0o077 == 0
    sealed.path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="sealed_changed"):
        verify_sealed(sealed)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(format_version=2),
        lambda d: d.update(format_version=True),
        lambda d: d.update(unknown="bad"),
        lambda d: d["files"][0].update(relative_path="../outside"),
        lambda d: d["files"][0].update(sha256="0" * 64),
        lambda d: d["report"].update(lines=["\x1b[31msecret"]),
        lambda d: d["report"].update(lines=["[link=https://evil]click[/link]"]),
        lambda d: d["files"][0].update(parent_id="missing"),
        lambda d: d["dependency_groups"][0].update(members=["missing"]),
        lambda d: d.update(
            relocations=[dict(logical_id="root", locator="/private/home")]
        ),
    ],
)
def test_rejects_invalid_manifest_and_integrity(tmp_path, mutation):
    doc = manifest()
    mutation(doc)
    source = archive(tmp_path, doc)
    with pytest.raises(ValueError):
        acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())
    assert not list((tmp_path / "work").glob("**/*"))


@pytest.mark.parametrize(
    "limits",
    [
        ArchiveLimits(manifest_bytes=20),
        ArchiveLimits(member_bytes=4),
        ArchiveLimits(expanded_bytes=4),
        ArchiveLimits(members=2),
        ArchiveLimits(path_bytes=5),
    ],
)
def test_independent_archive_limits(tmp_path, limits):
    source = archive(tmp_path)
    with pytest.raises(ValueError):
        acquire(source, tmp_path / "work", limits, None, Event())


@pytest.mark.parametrize(
    "extra",
    [
        ("../outside", b"x"),
        ("payload/1", b"x"),
        ("PAYLOAD/1", b"x"),
        ("unexplained", b"x"),
    ],
)
def test_rejects_ambiguous_or_unexplained_members(tmp_path, extra):
    source = archive(tmp_path, extra=extra)
    with pytest.raises(ValueError):
        acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())


def test_cancellation_does_not_leave_staging(tmp_path):
    source = archive(tmp_path)
    cancel = Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        acquire(source, tmp_path / "work", ArchiveLimits(), None, cancel)
    assert not (tmp_path / "work").exists()


def test_high_compression_requires_expanded_byte_review(tmp_path):
    source = archive(tmp_path, data=b"0" * 100_000, compression=zipfile.ZIP_DEFLATED)
    with pytest.raises(ValueError, match="compression_review_required"):
        acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())


def test_symlink_member_and_truncated_container_rejected(tmp_path):
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    source = archive(tmp_path, extra=(info, b"outside"))
    with pytest.raises(ValueError):
        acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())
    source.write_bytes(source.read_bytes()[:-10])
    with pytest.raises(ValueError):
        acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())


def test_encrypted_reader_authenticates_and_applies_custom_output_quota(
    tmp_path, helper_resource_root, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import crypto

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    source = archive(tmp_path)
    encrypted = tmp_path / "encrypted.age"
    crypto.transform(source, encrypted, password=b"pass", decrypt=False, cancel=Event())
    limits = ArchiveLimits(decrypted_bytes=4096)
    sealed = acquire(encrypted, tmp_path / "work", limits, b"pass", Event())
    assert sealed.path.read_bytes() == source.read_bytes()
    with pytest.raises(crypto.CryptoError):
        acquire(
            encrypted,
            tmp_path / "small",
            ArchiveLimits(decrypted_bytes=100),
            b"pass",
            Event(),
        )
    assert not list((tmp_path / "small").glob("**/*"))
    with pytest.raises(crypto.CryptoError):
        acquire(encrypted, tmp_path / "wrong", limits, b"wrong", Event())
    assert not list((tmp_path / "wrong").glob("**/*"))


@pytest.mark.parametrize(
    "mutation", ["local_method", "overlap", "central_count", "prefix", "suffix"]
)
def test_rejects_inconsistent_zip_structures(tmp_path, mutation):
    import struct

    source = archive(tmp_path)
    raw = bytearray(source.read_bytes())
    if mutation == "local_method":
        struct.pack_into("<H", raw, 8, 8)
    elif mutation == "overlap":
        first = raw.index(b"PK\x01\x02")
        second = raw.index(b"PK\x01\x02", first + 4)
        struct.pack_into("<I", raw, second + 42, 0)
    elif mutation == "central_count":
        struct.pack_into("<HH", raw, len(raw) - 14, 65534, 65534)
    elif mutation == "prefix":
        raw = b"unexplained" + raw
    else:
        raw += b"unexplained"
    source.write_bytes(raw)
    with pytest.raises(ValueError):
        acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())


def test_explicit_empty_directory_and_partial_groups_are_inert(tmp_path):
    doc = manifest()
    doc["consistency"] = "partial"
    doc["dependency_groups"][0]["complete"] = False
    doc["directories"].append(
        dict(
            logical_id="empty",
            root_id="root",
            parent_id="root",
            relative_path="empty",
            metadata=dict(version=1, mtime_ns=0, mode=448),
        )
    )
    sealed = acquire(
        archive(tmp_path, doc), tmp_path / "work", ArchiveLimits(), None, Event()
    )
    from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed

    assert verify_sealed(sealed).directories[1].relative_path == "empty"
    assert not (tmp_path / "work" / "empty").exists()


def test_crypto_rejects_invalid_local_budget_before_helper_lookup(tmp_path):
    from tldw_chatbook.Backup_Recovery.crypto import transform, CryptoError

    with pytest.raises(CryptoError, match="invalid_budget"):
        transform(
            tmp_path / "missing",
            tmp_path / "out",
            password=b"pass",
            decrypt=True,
            cancel=Event(),
            output_limit=0,
        )


def test_preview_manifest_cannot_be_substituted(tmp_path):
    from dataclasses import replace
    from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed

    sealed = acquire(
        archive(tmp_path), tmp_path / "work", ArchiveLimits(), None, Event()
    )
    doc = json.loads(sealed.manifest_bytes)
    doc["profile_ids"] = ["different"]
    with pytest.raises(ValueError, match="sealed_changed"):
        verify_sealed(replace(sealed, manifest_bytes=json.dumps(doc).encode()))


def test_force_zip64_member_round_trip(tmp_path):
    source = tmp_path / "zip64.zip"
    with zipfile.ZipFile(source, "w") as z:
        z.writestr("manifest.json", json.dumps(manifest()))
        with z.open("payload/1", "w", force_zip64=True) as out:
            out.write(b"hello")
    sealed = acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())
    assert sealed.path.read_bytes() == source.read_bytes()


def test_source_mutation_during_copy_cleans_operation(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from tldw_chatbook.Backup_Recovery import archive_reader as reader

    source = archive(tmp_path)
    original = reader.create_private_file

    @contextmanager
    def mutate_after_copy(path):
        with original(path) as fd:
            yield fd
        source.write_bytes(b"different source")

    monkeypatch.setattr(reader, "create_private_file", mutate_after_copy)
    with pytest.raises(ValueError, match="source_changed"):
        acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())
    assert not list((tmp_path / "work").glob("**/*"))


def test_space_admission_precedes_copy(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tldw_chatbook.Backup_Recovery import archive_reader as reader

    source = archive(tmp_path)
    monkeypatch.setattr(
        reader, "shutil", SimpleNamespace(disk_usage=lambda _: SimpleNamespace(free=0))
    )
    with pytest.raises(ValueError, match="insufficient_space"):
        acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())
    assert not (tmp_path / "work").exists()


def test_high_compression_review_is_bound_to_digest_and_expanded_bytes(tmp_path):
    from dataclasses import replace
    from tldw_chatbook.Backup_Recovery.archive_reader import CompressionReviewRequired

    source = archive(tmp_path, data=b"0" * 100_000, compression=zipfile.ZIP_DEFLATED)
    with pytest.raises(CompressionReviewRequired) as refusal:
        acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())
    reviewed = replace(
        ArchiveLimits(),
        reviewed_digest=refusal.value.digest,
        reviewed_expanded_bytes=refusal.value.expanded_bytes,
    )
    assert (
        acquire(source, tmp_path / "work", reviewed, None, Event()).digest
        == refusal.value.digest
    )
    with pytest.raises(ValueError, match="expanded_limit"):
        acquire(
            source,
            tmp_path / "work",
            replace(reviewed, expanded_bytes=99),
            None,
            Event(),
        )
    archive(tmp_path, data=b"1" * 100_000, compression=zipfile.ZIP_DEFLATED)
    with pytest.raises(CompressionReviewRequired):
        acquire(source, tmp_path / "work", reviewed, None, Event())


def test_zip64_end_records_round_trip(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(zipfile, "ZIP_FILECOUNT_LIMIT", 1)
        source = archive(tmp_path)
    sealed = acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())
    assert sealed.path.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("member", ["manifest.json", "payload/1"])
@pytest.mark.parametrize(
    "defect", ["extra_output", "trailing_compressed", "truncated_stream"]
)
def test_complete_deflate_stream_must_match_declared_member(tmp_path, member, defect):
    import struct
    import zlib

    entries = [
        ("manifest.json", json.dumps(manifest()).encode()),
        ("payload/1", b"hello"),
    ]
    local = bytearray()
    central = bytearray()
    for name, declared in entries:
        actual = (
            declared + b"x" * 100_000
            if name == member and defect == "extra_output"
            else declared
        )
        compressor = zlib.compressobj(wbits=-15)
        compressed = compressor.compress(actual) + compressor.flush()
        if name == member and defect == "trailing_compressed":
            compressed += b"unexplained"
        if name == member and defect == "truncated_stream":
            compressed = compressed[:-1]
        encoded = name.encode()
        crc = zlib.crc32(declared)
        offset = len(local)
        local += struct.pack(
            "<4s5H3I2H",
            b"PK\x03\x04",
            20,
            0,
            8,
            0,
            0,
            crc,
            len(compressed),
            len(declared),
            len(encoded),
            0,
        )
        local += encoded + compressed
        central += struct.pack(
            "<4s6H3I5H2I",
            b"PK\x01\x02",
            20,
            20,
            0,
            8,
            0,
            0,
            crc,
            len(compressed),
            len(declared),
            len(encoded),
            0,
            0,
            0,
            0,
            0,
            offset,
        )
        central += encoded
    source = tmp_path / "malformed.zip"
    source.write_bytes(
        local
        + central
        + struct.pack(
            "<4s4H2IH", b"PK\x05\x06", 0, 0, 2, 2, len(central), len(local), 0
        )
    )
    with pytest.raises(ValueError):
        acquire(
            source,
            tmp_path / "work",
            ArchiveLimits(member_bytes=2000, expanded_bytes=2000),
            None,
            Event(),
        )
    assert not list((tmp_path / "work").glob("**/*"))


def test_compression_review_retry_inside_handler_retains_sealed_bytes(tmp_path):
    from dataclasses import replace
    from tldw_chatbook.Backup_Recovery.archive_reader import (
        CompressionReviewRequired,
        verify_sealed,
    )

    source = archive(tmp_path, data=b"0" * 100_000, compression=zipfile.ZIP_DEFLATED)
    try:
        acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())
    except CompressionReviewRequired as review:
        limits = replace(
            ArchiveLimits(),
            reviewed_digest=review.digest,
            reviewed_expanded_bytes=review.expanded_bytes,
        )
        sealed = acquire(source, tmp_path / "work", limits, None, Event())
        assert sealed.path.exists()
        assert verify_sealed(sealed).format_version == 1
    else:
        pytest.fail("compression review was not requested")


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"hello",
        b"".join(hashlib.sha256(str(i).encode()).digest() for i in range(8192)),
    ],
)
def test_complete_deflate_validation_accepts_empty_and_multiple_input_chunks(
    tmp_path, data
):
    source = archive(tmp_path, data=data, compression=zipfile.ZIP_DEFLATED)
    sealed = acquire(source, tmp_path / "work", ArchiveLimits(), None, Event())
    assert sealed.path.read_bytes() == source.read_bytes()
