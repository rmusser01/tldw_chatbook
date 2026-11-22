"""Hostile archive and byte-preservation tests for Tool Pack inspection."""

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import warnings
import zipfile

import pytest

from tldw_chatbook.Tool_Packs import importer as importer_module
from tldw_chatbook.Tool_Packs.catalog_snapshot import PermissionInventorySnapshot
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.Tool_Packs.contracts import (
    MAX_JSON_NODES,
    MAX_PROFILE_BYTES,
    PROFILE_PATH,
    TOOL_PACK_SCHEMA,
    PortableToolRule,
    ToolPackError,
    canonical_json_bytes,
    portable_contract_sha256,
)
from tldw_chatbook.Tool_Packs.importer import ToolPackImportService

from .test_importer import _Store, _archive, _inventory, _tool


def test_import_safety_api_is_available() -> None:
    assert ToolPackImportService


def _service(
    monkeypatch: pytest.MonkeyPatch,
    inventory: PermissionInventorySnapshot,
    fixed_now: datetime,
) -> ToolPackImportService:
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _value: inventory
    )
    return ToolPackImportService(
        _Store(), inventory, lambda _profile_id: False, now=lambda: fixed_now
    )


@pytest.fixture
def fixed_now() -> datetime:
    return datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def _canonical_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.flag_bits = 0
    info.external_attr = 0o100644 << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def _members(path: Path) -> list[tuple[zipfile.ZipInfo, bytes]]:
    with zipfile.ZipFile(path) as reader:
        return [(copy.copy(info), reader.read(info)) for info in reader.infolist()]


def _write_members(
    path: Path,
    members: list[tuple[zipfile.ZipInfo, bytes]],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as writer:
        for info, data in members:
            writer.writestr(info, data)


def _valid_archive(tmp_path: Path) -> tuple[Path, object]:
    tool = _tool()
    return (
        _archive(
            tmp_path / "valid.tldw-tool-pack",
            rules=(
                PortableToolRule(
                    "mcp",
                    "local:docs",
                    "search",
                    "allow",
                    portable_contract_sha256(tool),
                ),
            ),
        ),
        tool,
    )


def _replace_payload_with_valid_manifest(path: Path, payload: bytes) -> None:
    members = _members(path)
    manifest = json.loads(members[0][1])
    manifest["files"][0]["size"] = len(payload)
    manifest["files"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_without_digest = dict(manifest)
    manifest_without_digest.pop("content_digest")
    manifest["content_digest"] = hashlib.sha256(
        TOOL_PACK_SCHEMA.encode("ascii")
        + b"\0"
        + canonical_json_bytes(manifest_without_digest)
        + b"\0"
        + payload
    ).hexdigest()
    _write_members(
        path,
        [
            (_canonical_info("tool-pack.json"), canonical_json_bytes(manifest)),
            (_canonical_info(PROFILE_PATH), payload),
        ],
    )


def test_rejects_noncanonical_member_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    tool = _tool()
    archive = _archive(
        tmp_path / "timestamp.tldw-tool-pack",
        rules=(
            PortableToolRule(
                "mcp", "local:docs", "search", "allow", portable_contract_sha256(tool)
            ),
        ),
    )
    with zipfile.ZipFile(archive) as reader:
        members = [(copy.copy(info), reader.read(info)) for info in reader.infolist()]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as writer:
        for info, data in members:
            info.date_time = (1981, 1, 1, 0, 0, 0)
            writer.writestr(info, data)

    with pytest.raises(ToolPackError) as raised:
        _service(monkeypatch, _inventory(tool), fixed_now).inspect_archive(
            archive, destination_id="research"
        )

    assert raised.value.category == "archive_invalid"


@pytest.mark.parametrize(
    "hostile_name",
    [
        "../tool-pack.json",
        "/tool-pack.json",
        "C:/tool-pack.json",
        "profile\\profile.json",
        "./tool-pack.json",
        "profile/../profile.json",
        "CON",
        "nested.zip",
    ],
)
def test_rejects_hostile_and_nested_member_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
    hostile_name: str,
) -> None:
    archive, tool = _valid_archive(tmp_path)
    members = _members(archive)
    members[0][0].filename = hostile_name
    _write_members(archive, members)

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.archive_invalid$"):
        _service(monkeypatch, _inventory(tool), fixed_now).inspect_archive(
            archive, destination_id="research"
        )


def test_rejects_nul_member_name(tmp_path: Path) -> None:
    archive, _tool_value = _valid_archive(tmp_path)
    raw = archive.read_bytes().replace(b"tool-pack.json", b"tool-pack\x00json")
    archive.write_bytes(raw)

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.archive_invalid$"):
        importer_module._read_document(raw)


@pytest.mark.parametrize("duplicate", ["tool-pack.json", "TOOL-PACK.JSON"])
def test_rejects_exact_and_casefold_member_duplicates(
    tmp_path: Path,
    duplicate: str,
) -> None:
    archive, _tool_value = _valid_archive(tmp_path)
    members = _members(archive)
    members.append((_canonical_info(duplicate), b"{}\n"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _write_members(archive, members)

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.archive_invalid$"):
        importer_module._read_document(archive.read_bytes())


@pytest.mark.parametrize(
    "name,external_attr,extra",
    [
        ("profile/profile.json", stat.S_IFLNK << 16, b""),
        ("profile/profile.json", stat.S_IFCHR << 16, b""),
        ("profile/profile.json", stat.S_IFIFO << 16, b""),
        ("profile/profile.json", 0o100644 << 16, b"\x0d\x00\x00\x00"),
        ("profile/", (stat.S_IFDIR | 0o755) << 16, b""),
    ],
)
def test_rejects_links_hardlink_metadata_nonregular_and_directories(
    tmp_path: Path,
    name: str,
    external_attr: int,
    extra: bytes,
) -> None:
    archive, _tool_value = _valid_archive(tmp_path)
    members = _members(archive)
    members[1][0].filename = name
    members[1][0].external_attr = external_attr
    members[1][0].extra = extra
    _write_members(archive, members)

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.archive_invalid$"):
        importer_module._read_document(archive.read_bytes())


@pytest.mark.parametrize("flag", [0x1, 0x8])
def test_rejects_encryption_and_data_descriptor_flags(
    tmp_path: Path,
    flag: int,
) -> None:
    archive, _tool_value = _valid_archive(tmp_path)
    raw = bytearray(archive.read_bytes())
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        cursor = 0
        while (cursor := raw.find(signature, cursor)) >= 0:
            value = int.from_bytes(raw[cursor + offset : cursor + offset + 2], "little")
            raw[cursor + offset : cursor + offset + 2] = (value | flag).to_bytes(
                2, "little"
            )
            cursor += 4

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.archive_invalid$"):
        importer_module._read_document(bytes(raw))


def test_rejects_compressed_members(tmp_path: Path) -> None:
    archive, _tool_value = _valid_archive(tmp_path)
    members = _members(archive)
    for info, _data in members:
        info.compress_type = zipfile.ZIP_DEFLATED
    _write_members(archive, members, compression=zipfile.ZIP_DEFLATED)

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.archive_invalid$"):
        importer_module._read_document(archive.read_bytes())


def test_rejects_archive_and_member_comments(tmp_path: Path) -> None:
    archive, _tool_value = _valid_archive(tmp_path)
    members = _members(archive)
    members[0][0].comment = b"comment"
    _write_members(archive, members)

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.archive_invalid$"):
        importer_module._read_document(archive.read_bytes())


def test_rejects_duplicate_and_noncanonical_json(tmp_path: Path) -> None:
    archive, _tool_value = _valid_archive(tmp_path)
    members = _members(archive)
    members[0] = (
        _canonical_info("tool-pack.json"),
        members[0][1].replace(b'"schema":', b'"schema":"duplicate","schema":', 1),
    )
    _write_members(archive, members)

    with pytest.raises(ToolPackError) as raised:
        importer_module._read_document(archive.read_bytes())
    assert raised.value.category == "manifest_invalid"


@pytest.mark.parametrize(
    "payload",
    [
        b"{\n",
        b'{"schema":"tldw.tool-profile/v1","schema":"duplicate"}\n',
        b'{"fallbacks": [], "schema":"tldw.tool-profile/v1","tools":[]}\n',
    ],
)
def test_rejects_malformed_duplicate_and_noncanonical_payload_json(
    tmp_path: Path,
    payload: bytes,
) -> None:
    archive, _tool_value = _valid_archive(tmp_path)
    _replace_payload_with_valid_manifest(archive, payload)

    with pytest.raises(ToolPackError) as raised:
        importer_module._read_document(archive.read_bytes())
    assert raised.value.category == "payload_invalid"

    members = _members(_valid_archive(tmp_path)[0])
    members[0] = (_canonical_info("tool-pack.json"), members[0][1] + b" ")
    _write_members(archive, members)
    with pytest.raises(ToolPackError) as raised:
        importer_module._read_document(archive.read_bytes())
    assert raised.value.category == "manifest_invalid"


@pytest.mark.parametrize("field,bad_value", [("size", 1), ("sha256", "0" * 64)])
def test_rejects_payload_size_and_digest_mismatch(
    tmp_path: Path,
    field: str,
    bad_value: object,
) -> None:
    archive, _tool_value = _valid_archive(tmp_path)
    members = _members(archive)
    manifest = json.loads(members[0][1])
    manifest["files"][0][field] = bad_value
    members[0] = (
        _canonical_info("tool-pack.json"),
        canonical_json_bytes(manifest),
    )
    _write_members(archive, members)

    with pytest.raises(ToolPackError) as raised:
        importer_module._read_document(archive.read_bytes())
    assert raised.value.category == "manifest_invalid"


def test_rejects_content_digest_mismatch(tmp_path: Path) -> None:
    archive, _tool_value = _valid_archive(tmp_path)
    members = _members(archive)
    manifest = json.loads(members[0][1])
    manifest["content_digest"] = "0" * 64
    members[0] = (_canonical_info("tool-pack.json"), canonical_json_bytes(manifest))
    _write_members(archive, members)

    with pytest.raises(ToolPackError) as raised:
        importer_module._read_document(archive.read_bytes())
    assert raised.value.category == "manifest_invalid"


def test_rejects_archive_member_depth_and_node_limits(tmp_path: Path) -> None:
    too_large = tmp_path / "too-large.tldw-tool-pack"
    too_large.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    with pytest.raises(ToolPackError) as raised:
        importer_module._read_regular_archive(too_large)
    assert raised.value.category == "too_large"

    archive, _tool_value = _valid_archive(tmp_path)
    members = _members(archive)
    members[1] = (_canonical_info(PROFILE_PATH), b"x" * (MAX_PROFILE_BYTES + 1))
    _write_members(archive, members)
    with pytest.raises(ToolPackError) as raised:
        importer_module._read_document(archive.read_bytes())
    assert raised.value.category == "too_large"

    for payload in (
        (b'{"x":' * 13) + b"null" + (b"}" * 13) + b"\n",
        b'{"x":[' + b",".join([b"null"] * (MAX_JSON_NODES + 1)) + b"]}\n",
    ):
        archive, _tool_value = _valid_archive(tmp_path)
        _replace_payload_with_valid_manifest(archive, payload)
        with pytest.raises(ToolPackError) as raised:
            importer_module._read_document(archive.read_bytes())
        assert raised.value.category == "payload_invalid"


def test_rejects_symlink_source_without_following_it(tmp_path: Path) -> None:
    target, _tool_value = _valid_archive(tmp_path)
    link = tmp_path / "link.tldw-tool-pack"
    link.symlink_to(target)

    with pytest.raises(ToolPackError) as raised:
        importer_module._read_regular_archive(link)
    assert raised.value.category == "archive_invalid"


def test_inspection_rejects_non_tool_pack_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    archive, tool = _valid_archive(tmp_path)
    renamed = archive.with_suffix(".zip")
    archive.rename(renamed)

    with pytest.raises(ToolPackError) as raised:
        _service(monkeypatch, _inventory(tool), fixed_now).inspect_archive(
            renamed, destination_id="research"
        )
    assert raised.value.category == "archive_invalid"


def test_rejects_archive_path_substitution_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, _tool_value = _valid_archive(tmp_path)
    replacement = tmp_path / "replacement.tldw-tool-pack"
    replacement.write_bytes(archive.read_bytes())
    real_read = importer_module.os.read
    swapped = False

    def substituting_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        data = real_read(descriptor, count)
        if not swapped:
            swapped = True
            os.replace(replacement, archive)
        return data

    monkeypatch.setattr(importer_module.os, "read", substituting_read)

    with pytest.raises(ToolPackError) as raised:
        importer_module._read_regular_archive(archive)
    assert raised.value.category == "archive_invalid"


# The archive reader on native Windows. None of these tests is platform-skipped:
# Windows has no O_NOFOLLOW, its CRT opens files in text mode unless O_BINARY
# is passed, and its fstat and stat disagree about st_ctime. Each of those once
# made every Windows import fail (or fail at random).

# The creation-or-change timestamp the reader includes in a file's identity.
_IDENTITY_TIME_FIELD = "st_birthtime_ns" if os.name == "nt" else "st_ctime_ns"


class _StatView:
    """An ``os.stat`` result with selected fields replaced."""

    def __init__(self, real: os.stat_result, **overrides: int) -> None:
        self._real = real
        self._overrides = overrides

    def __getattr__(self, name: str) -> object:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._real, name)


def test_inspects_a_valid_archive_on_every_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    archive, tool = _valid_archive(tmp_path)
    raw = archive.read_bytes()

    review = _service(monkeypatch, _inventory(tool), fixed_now).inspect_archive(
        archive, destination_id="research"
    )

    assert review.archive_sha256 == hashlib.sha256(raw).hexdigest()
    assert importer_module._read_regular_archive(archive) == raw


def test_inspects_an_archive_rewritten_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    # On Windows a rewrite moves the ChangeTime that fstat reports as st_ctime,
    # but not the CreationTime that stat reports as st_ctime.
    archive, tool = _valid_archive(tmp_path)
    raw = archive.read_bytes()
    time.sleep(0.05)  # Longer than one filesystem clock tick.
    with archive.open("r+b") as handle:
        handle.write(raw)
        handle.truncate()

    review = _service(monkeypatch, _inventory(tool), fixed_now).inspect_archive(
        archive, destination_id="research"
    )

    assert review.archive_sha256 == hashlib.sha256(raw).hexdigest()


def test_reads_archive_bytes_exactly_including_crlf_and_ctrl_z(
    tmp_path: Path,
) -> None:
    raw = b"PK\x03\x04" + bytes(range(256)) * 4 + b"\r\n\x1a\r\n"
    path = tmp_path / "binary.tldw-tool-pack"
    path.write_bytes(raw)

    assert importer_module._read_regular_archive(path) == raw


def test_inspects_a_valid_archive_whose_bytes_contain_ctrl_z(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    # A CRC or length byte of 0x1A ends a text-mode read early; a few percent
    # of real archives contain one. Search deterministically for such a pack.
    for index in range(2000):
        tool = _tool(name=f"search-{index}")
        archive = _archive(
            tmp_path / "ctrl-z.tldw-tool-pack",
            rules=(
                PortableToolRule(
                    "mcp",
                    "local:docs",
                    tool.name,
                    "allow",
                    portable_contract_sha256(tool),
                ),
            ),
        )
        raw = archive.read_bytes()
        if b"\x1a" in raw:
            break
    assert b"\x1a" in raw

    review = _service(monkeypatch, _inventory(tool), fixed_now).inspect_archive(
        archive, destination_id="research"
    )

    assert review.archive_sha256 == hashlib.sha256(raw).hexdigest()
    assert len(review.matched) == 1


@pytest.mark.parametrize(
    "field",
    [
        None,
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        pytest.param(_IDENTITY_TIME_FIELD, id="identity_time"),
    ],
)
def test_path_identity_must_match_the_descriptor_that_was_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str | None,
) -> None:
    # Unlike the substitution test above, this reaches the identity comparison
    # on every platform: Windows refuses that test's os.replace of an open file.
    archive, _tool_value = _valid_archive(tmp_path)
    raw = archive.read_bytes()
    real_stat = importer_module.os.stat
    calls: list[tuple[object, bool]] = []

    def drifted_stat(path: object, *, follow_symlinks: bool = True) -> object:
        calls.append((path, follow_symlinks))
        real = real_stat(path, follow_symlinks=follow_symlinks)
        if field is None:
            return real
        return _StatView(real, **{field: getattr(real, field) + 1})

    monkeypatch.setattr(importer_module.os, "stat", drifted_stat)

    if field is None:
        assert importer_module._read_regular_archive(archive) == raw
    else:
        with pytest.raises(ToolPackError) as raised:
            importer_module._read_regular_archive(archive)
        assert raised.value.category == "archive_invalid"
    assert calls == [(archive, False)]


def test_windows_identity_ignores_st_ctime_but_posix_compares_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, _tool_value = _valid_archive(tmp_path)
    raw = archive.read_bytes()
    real_stat = importer_module.os.stat

    def ctime_shifted_stat(path: object, *, follow_symlinks: bool = True) -> object:
        real = real_stat(path, follow_symlinks=follow_symlinks)
        return _StatView(real, st_ctime_ns=real.st_ctime_ns + 1)

    monkeypatch.setattr(importer_module.os, "stat", ctime_shifted_stat)

    if os.name == "nt":
        # Windows stat's st_ctime is CreationTime while fstat's is ChangeTime,
        # so st_ctime cannot be part of the identity there.
        assert importer_module._read_regular_archive(archive) == raw
    else:
        with pytest.raises(ToolPackError) as raised:
            importer_module._read_regular_archive(archive)
        assert raised.value.category == "archive_invalid"


def test_posix_host_without_o_nofollow_refuses_before_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, _tool_value = _valid_archive(tmp_path)

    def refusing_open(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("the reader opened the archive")

    # Scoped to the one call: a process-wide os.name of "posix" breaks pathlib
    # on Windows, including in fixture teardown.
    with monkeypatch.context() as patch:
        patch.delattr(importer_module.os, "O_NOFOLLOW", raising=False)
        patch.setattr(importer_module.os, "name", "posix")
        patch.setattr(importer_module.os, "open", refusing_open)
        with pytest.raises(ToolPackError) as raised:
            importer_module._read_regular_archive(archive)
    assert raised.value.category == "archive_invalid"


class _WindowsOsView:
    """``os`` as the importer sees it on native Windows, simulated on any host.

    There is no ``O_NOFOLLOW``, and ``fstat`` reports ChangeTime as
    ``st_ctime`` while ``stat`` reports CreationTime. Both report the same
    ``st_birthtime_ns`` (POSIX hosts lack it; a constant stands in).
    """

    name = "nt"

    def __init__(self, birthtime_drift: int) -> None:
        self._drift = birthtime_drift

    def __getattr__(self, name: str) -> object:
        if name == "O_NOFOLLOW":
            raise AttributeError(name)
        return getattr(os, name)

    def fstat(self, descriptor: int) -> _StatView:
        real = os.fstat(descriptor)
        return _StatView(
            real,
            st_ctime_ns=real.st_ctime_ns + 1,
            st_birthtime_ns=getattr(real, "st_birthtime_ns", 0),
        )

    def stat(self, path: object, *, follow_symlinks: bool = True) -> _StatView:
        real = os.stat(path, follow_symlinks=follow_symlinks)
        return _StatView(
            real, st_birthtime_ns=getattr(real, "st_birthtime_ns", 0) + self._drift
        )


@pytest.mark.parametrize("birthtime_drift", [0, 1], ids=["same-file", "drifted"])
def test_windows_os_semantics_read_the_archive_and_keep_the_identity_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    birthtime_drift: int,
) -> None:
    # Runs the nt branches on POSIX CI too, which the tests above reach only on
    # a Windows host.
    archive, _tool_value = _valid_archive(tmp_path)
    raw = archive.read_bytes()
    monkeypatch.setattr(importer_module, "os", _WindowsOsView(birthtime_drift))

    if birthtime_drift == 0:
        assert importer_module._read_regular_archive(archive) == raw
    else:
        with pytest.raises(ToolPackError) as raised:
            importer_module._read_regular_archive(archive)
        assert raised.value.category == "archive_invalid"


@pytest.mark.parametrize("raw", [b"{", b'{"schema_version":99}'])
def test_invalid_store_bytes_are_unchanged_and_legacy_recovery_is_never_called(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
    raw: bytes,
) -> None:
    archive, tool = _valid_archive(tmp_path)
    permission_path = tmp_path / "permissions.json"
    permission_path.write_bytes(raw)
    before = permission_path.read_bytes()
    inventory = _inventory(tool)
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _value: inventory
    )

    def forbidden_load(_store: object) -> object:
        raise AssertionError("legacy load must never be called")

    monkeypatch.setattr(MCPPermissionStore, "load", forbidden_load)
    service = ToolPackImportService(
        MCPPermissionStore(permission_path),
        inventory,
        lambda _profile_id: False,
        now=lambda: fixed_now,
    )

    with pytest.raises(ToolPackError) as raised:
        service.inspect_archive(archive, destination_id="research")

    assert raised.value.category == "store_invalid"
    assert permission_path.read_bytes() == before
    assert not list(permission_path.parent.glob("*.bak"))
    assert str(tmp_path) not in str(raised.value)
