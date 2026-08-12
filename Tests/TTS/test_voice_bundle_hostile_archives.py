from __future__ import annotations

import json
import struct
import unicodedata
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest
from hypothesis import given, strategies as st

from tldw_chatbook.TTS.voice_bundle_codec import (
    EXPECTED_MEMBER_ORDER,
    TTSVoiceBundleError,
    inspect_clone_voice_bundle,
)
import tldw_chatbook.TTS.voice_bundle_codec as codec

from .test_voice_bundle_codec import canonical_bundle
from tldw_chatbook.TTS.voice_bundle_codec import encode_clone_voice_bundle


def _valid() -> bytes:
    return encode_clone_voice_bundle(canonical_bundle())


def _central_offsets(payload: bytes) -> list[int]:
    eocd = len(payload) - 22
    directory = struct.unpack_from("<L", payload, eocd + 16)[0]
    offsets: list[int] = []
    cursor = directory
    while cursor < eocd:
        assert payload[cursor : cursor + 4] == b"PK\x01\x02"
        offsets.append(cursor)
        name, extra, comment = struct.unpack_from("<3H", payload, cursor + 28)
        cursor += 46 + name + extra + comment
    return offsets


def _local_offset(payload: bytes, central: int) -> int:
    return struct.unpack_from("<L", payload, central + 42)[0]


def _patch_u16(payload: bytes, offset: int, value: int) -> bytes:
    changed = bytearray(payload)
    struct.pack_into("<H", changed, offset, value)
    return bytes(changed)


def _patch_u32(payload: bytes, offset: int, value: int) -> bytes:
    changed = bytearray(payload)
    struct.pack_into("<L", changed, offset, value)
    return bytes(changed)


def _patch_flags(payload: bytes, flags: int, *, central_only: bool = False) -> bytes:
    changed = payload
    for central in _central_offsets(payload):
        changed = _patch_u16(changed, central + 8, flags)
        if not central_only:
            changed = _patch_u16(changed, _local_offset(payload, central) + 6, flags)
    return changed


def _repack(
    *,
    compression: int = ZIP_STORED,
    create_system: int = 3,
    create_version: int = 20,
    extract_version: int = 20,
    external_attr: int = 0o100600 << 16,
    member_extra: bytes = b"",
    member_comment: bytes = b"",
    archive_comment: bytes = b"",
) -> bytes:
    source = _valid()
    with ZipFile(BytesIO(source)) as archive:
        contents = {name: archive.read(name) for name in EXPECTED_MEMBER_ORDER}
    target = BytesIO()
    with ZipFile(target, "w", compression=compression, allowZip64=False) as archive:
        archive.comment = archive_comment
        for name in EXPECTED_MEMBER_ORDER:
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = compression
            info.create_system = create_system
            info.create_version = create_version
            info.extract_version = extract_version
            info.external_attr = external_attr
            info.internal_attr = 0
            info.extra = member_extra
            info.comment = member_comment
            archive.writestr(info, contents[name])
    return target.getvalue()


def _members(payload: bytes | None = None) -> dict[str, bytes]:
    with ZipFile(BytesIO(payload or _valid())) as archive:
        return {name: archive.read(name) for name in EXPECTED_MEMBER_ORDER}


def _pack_members(values: dict[str, bytes], *, compression: int = ZIP_STORED) -> bytes:
    target = BytesIO()
    with ZipFile(target, "w", compression=compression, allowZip64=False) as archive:
        for name in EXPECTED_MEMBER_ORDER:
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = compression
            info.create_system = 3
            info.create_version = 20
            info.extract_version = 20
            info.external_attr = 0o100600 << 16
            archive.writestr(info, values[name])
    return target.getvalue()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


@pytest.mark.parametrize(
    ("compression", "needed", "created", "system", "attributes"),
    [
        (ZIP_STORED, 10, 10, 3, 0o100600 << 16),
        (ZIP_STORED, 10, 20, 3, 0o100600 << 16),
        (ZIP_STORED, 20, 10, 3, 0o100600 << 16),
        (ZIP_STORED, 20, 20, 3, 0o100644 << 16),
        (ZIP_STORED, 10, 10, 0, 0x20),
        (ZIP_STORED, 10, 20, 0, 0x20),
        (ZIP_STORED, 20, 10, 0, 0x20),
        (ZIP_STORED, 20, 20, 0, 0x20),
        (ZIP_DEFLATED, 20, 10, 3, 0o100400 << 16),
        (ZIP_DEFLATED, 20, 20, 3, 0o100400 << 16),
        (ZIP_DEFLATED, 20, 10, 0, 0x20),
        (ZIP_DEFLATED, 20, 20, 0, 0x20),
    ],
)
@pytest.mark.parametrize("flags", [0, 1 << 11])
def test_accepts_exact_interoperable_metadata_matrix(
    compression: int,
    needed: int,
    created: int,
    system: int,
    attributes: int,
    flags: int,
) -> None:
    payload = _repack(
        compression=compression,
        create_system=system,
        create_version=created,
        extract_version=needed,
        external_attr=attributes,
    )
    payload = _patch_flags(payload, flags)

    assert inspect_clone_voice_bundle(payload) == canonical_bundle()


def _replace_final_member_name(payload: bytes, replacement: bytes) -> bytes:
    assert len(replacement) == len(b"reference.txt")
    changed = bytearray(payload)
    central = _central_offsets(payload)[-1]
    local = _local_offset(payload, central)
    changed[central + 46 : central + 46 + len(replacement)] = replacement
    changed[local + 30 : local + 30 + len(replacement)] = replacement
    return bytes(changed)


def _local_name_disagreement(payload: bytes) -> bytes:
    changed = bytearray(payload)
    central = _central_offsets(payload)[-1]
    local = _local_offset(payload, central)
    changed[local + 30 : local + 43] = b"REFERENCE.txt"
    return bytes(changed)


def _overlap(payload: bytes) -> bytes:
    central = _central_offsets(payload)
    return _patch_u32(payload, central[1] + 42, _local_offset(payload, central[0]))


def _invalid_needed(payload: bytes) -> bytes:
    changed = payload
    for central in _central_offsets(payload):
        changed = _patch_u16(changed, central + 6, 45)
        changed = _patch_u16(changed, _local_offset(payload, central) + 4, 45)
    return changed


def _invalid_created(payload: bytes) -> bytes:
    changed = payload
    for central in _central_offsets(payload):
        changed = _patch_u16(changed, central + 4, (3 << 8) | 45)
    return changed


def _invalid_unix_mode(payload: bytes) -> bytes:
    changed = payload
    for central in _central_offsets(payload):
        changed = _patch_u32(changed, central + 38, 0o120777 << 16)
    return changed


def _invalid_dos_attributes(payload: bytes) -> bytes:
    changed = payload
    for central in _central_offsets(payload):
        changed = _patch_u16(changed, central + 4, 20)
        changed = _patch_u32(changed, central + 38, 0x10)
    return changed


def _multipart(payload: bytes) -> bytes:
    eocd = len(payload) - 22
    return _patch_u16(payload, eocd + 4, 1)


def _zip64_sentinel(payload: bytes) -> bytes:
    central = _central_offsets(payload)[0]
    return _patch_u32(payload, central + 24, 0xFFFFFFFF)


HOSTILE_MUTATORS = (
    lambda value: _patch_flags(value, 1),
    lambda value: _patch_flags(value, 1 << 3),
    lambda value: _patch_flags(value, 1 << 5),
    lambda value: _patch_flags(value, 1 << 6),
    lambda value: _patch_flags(value, 1 << 13),
    lambda value: _patch_flags(value, 1 << 14),
    _invalid_needed,
    _invalid_created,
    _invalid_unix_mode,
    _invalid_dos_attributes,
    lambda _value: _repack(member_extra=b"\x01\x00\x00\x00"),
    lambda _value: _repack(member_comment=b"comment"),
    lambda _value: _repack(archive_comment=b"comment"),
    _multipart,
    _zip64_sentinel,
    lambda value: _replace_final_member_name(value, b"reference.wav"),
    lambda value: _replace_final_member_name(value, b"REFERENCE.txt"),
    lambda value: _replace_final_member_name(value, b"../erence.txt"),
    lambda value: _replace_final_member_name(value, b"/eference.txt"),
    lambda value: _replace_final_member_name(value, b"reference\\txt"),
    lambda value: _replace_final_member_name(value, b"C:ference.txt"),
    _local_name_disagreement,
    _overlap,
    lambda value: b"prefix" + value,
    lambda value: value + b"trailing",
)


@pytest.mark.parametrize("mutator", HOSTILE_MUTATORS)
def test_hostile_archive_structure_is_rejected(mutator) -> None:
    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(mutator(_valid()))


@given(
    st.integers(min_value=0, max_value=0xFFFF).filter(
        lambda value: value not in (0, 1 << 11)
    )
)
def test_every_other_general_purpose_flag_combination_is_rejected(flags: int) -> None:
    with pytest.raises(TTSVoiceBundleError):
        inspect_clone_voice_bundle(_patch_flags(_valid(), flags))


@given(
    st.binary(min_size=len(b"reference.txt"), max_size=len(b"reference.txt")).filter(
        lambda value: value != b"reference.txt"
    )
)
def test_every_non_allowlisted_generated_member_name_is_rejected(name: bytes) -> None:
    with pytest.raises(TTSVoiceBundleError):
        inspect_clone_voice_bundle(_replace_final_member_name(_valid(), name))


@given(
    st.integers(min_value=0, max_value=255).filter(lambda value: value not in (10, 20))
)
def test_generated_unsupported_needed_versions_are_rejected(version: int) -> None:
    payload = _valid()
    changed = payload
    for central in _central_offsets(payload):
        changed = _patch_u16(changed, central + 6, version)
        changed = _patch_u16(changed, _local_offset(payload, central) + 4, version)
    with pytest.raises(TTSVoiceBundleError):
        inspect_clone_voice_bundle(changed)


@given(st.integers(min_value=1, max_value=4096))
def test_generated_declared_size_disagreement_is_rejected(delta: int) -> None:
    payload = _valid()
    central = _central_offsets(payload)[-1]
    local = _local_offset(payload, central)
    declared = struct.unpack_from("<L", payload, central + 24)[0] + delta
    changed = _patch_u32(payload, central + 24, declared)
    changed = _patch_u32(changed, local + 22, declared)
    with pytest.raises(TTSVoiceBundleError):
        inspect_clone_voice_bundle(changed)


@pytest.mark.parametrize(
    "mutate_manifest",
    [
        lambda value: {**value, "unexpected": True},
        lambda value: {**value, "schema_version": 2},
        lambda value: {**value, "schema_version": True},
        lambda value: {**value, "schema_version": None},
        lambda value: {**value, "schema_version": "1"},
        lambda value: {**value, "schema_version": 1.0},
        lambda value: {**value, "bundle_format": "other"},
        lambda value: {
            **value,
            "declaration": {
                "plaintext_sensitive_data_acknowledged": False,
                "version": 1,
            },
        },
        lambda value: {
            **value,
            "dependency": {**value["dependency"], "model_id": "different"},
        },
        lambda value: {
            **value,
            "reference": {**value["reference"], "duration_ms": 999},
        },
        lambda value: {
            **value,
            "entries": {
                **value["entries"],
                "reference.wav": {
                    **value["entries"]["reference.wav"],
                    "sha256": "0" * 64,
                },
            },
        },
        lambda value: {
            **value,
            "entries": {
                **value["entries"],
                "reference.txt": {
                    **value["entries"]["reference.txt"],
                    "size": 999,
                },
            },
        },
    ],
)
def test_manifest_schema_dependency_facts_and_checksums_are_exact(
    mutate_manifest,
) -> None:
    values = _members()
    manifest = json.loads(values["manifest.json"])
    values["manifest.json"] = _canonical_json(mutate_manifest(manifest))

    with pytest.raises(TTSVoiceBundleError):
        inspect_clone_voice_bundle(_pack_members(values))


@pytest.mark.parametrize(
    "mutate_profile",
    [
        lambda value: {**value, "unexpected": True},
        lambda value: {**value, "schema_version": 2},
        lambda value: {**value, "schema_version": True},
        lambda value: {**value, "schema_version": None},
        lambda value: {**value, "schema_version": "1"},
        lambda value: {**value, "schema_version": 1.0},
        lambda value: {**value, "profile_id": "0123456789ab4cde8fab0123456789ab"},
        lambda value: {**value, "name": " leading"},
        lambda value: {**value, "provider_id": "openai"},
        lambda value: {**value, "response_format": "mp3"},
        lambda value: {**value, "speed": 1},
        lambda value: {**value, "options": {"private": "value"}},
    ],
)
def test_profile_schema_is_exact_even_when_manifest_checksum_is_updated(
    mutate_profile,
) -> None:
    values = _members()
    profile = _canonical_json(mutate_profile(json.loads(values["profile.json"])))
    values["profile.json"] = profile
    manifest = json.loads(values["manifest.json"])
    manifest["entries"]["profile.json"] = {
        "sha256": __import__("hashlib").sha256(profile).hexdigest(),
        "size": len(profile),
    }
    values["manifest.json"] = _canonical_json(manifest)

    with pytest.raises(TTSVoiceBundleError):
        inspect_clone_voice_bundle(_pack_members(values))


@pytest.mark.parametrize("member", ["manifest.json", "profile.json"])
@pytest.mark.parametrize(
    ("version", "expected_code"),
    [
        (None, "bundle_invalid"),
        (True, "bundle_invalid"),
        ("1", "bundle_invalid"),
        (1.0, "bundle_invalid"),
        (2, "unsupported_bundle"),
    ],
)
def test_schema_version_error_taxonomy_is_exact(
    member: str,
    version: object,
    expected_code: str,
) -> None:
    values = _members()
    changed = json.loads(values[member])
    changed["schema_version"] = version
    values[member] = _canonical_json(changed)
    if member == "profile.json":
        manifest = json.loads(values["manifest.json"])
        manifest["entries"]["profile.json"] = {
            "sha256": __import__("hashlib").sha256(values[member]).hexdigest(),
            "size": len(values[member]),
        }
        values["manifest.json"] = _canonical_json(manifest)

    with pytest.raises(TTSVoiceBundleError) as caught:
        inspect_clone_voice_bundle(_pack_members(values))

    assert caught.value.code == expected_code


@pytest.mark.parametrize(
    "member_payload",
    [
        b'{"schema_version":NaN}\n',
        b'{"schema_version":1,"schema_version":1}\n',
        b"\xef\xbb\xbf{}\n",
        b"\xff\n",
        b"{}",
        b'{"a":{"b":{"c":{"d":{"e":1}}}}}\n',
    ],
)
def test_malformed_nonfinite_duplicate_deep_or_noncanonical_json_is_rejected(
    member_payload: bytes,
) -> None:
    values = _members()
    values["manifest.json"] = member_payload

    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(_pack_members(values))


def test_metadata_and_streaming_quota_guards_return_bounded_limit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript_size = len(_members()["reference.txt"])
    monkeypatch.setitem(codec._MEMBER_LIMITS, "reference.txt", transcript_size - 1)

    with pytest.raises(TTSVoiceBundleError) as caught:
        inspect_clone_voice_bundle(_valid())

    assert caught.value.code == "bundle_limit_exceeded"


def test_stream_counter_does_not_trust_declared_member_size() -> None:
    payload = _repack(compression=ZIP_DEFLATED)
    central = _central_offsets(payload)[-1]
    local = _local_offset(payload, central)
    declared = struct.unpack_from("<L", payload, central + 24)[0] - 1
    payload = _patch_u32(payload, central + 24, declared)
    payload = _patch_u32(payload, local + 22, declared)

    with pytest.raises(TTSVoiceBundleError) as caught:
        inspect_clone_voice_bundle(payload)

    assert caught.value.code == "bundle_invalid"


def _refresh_entry(manifest: dict, name: str, payload: bytes) -> None:
    manifest["entries"][name] = {
        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
        "size": len(payload),
    }


@pytest.mark.parametrize(
    "transcript",
    [
        b"",
        b" leading",
        b"trailing ",
        b"\xef\xbb\xbftext",
        b"control\x00text",
        b"\xff",
    ],
)
def test_transcript_must_be_exact_bounded_canonical_utf8(transcript: bytes) -> None:
    values = _members()
    values["reference.txt"] = transcript
    manifest = json.loads(values["manifest.json"])
    _refresh_entry(manifest, "reference.txt", transcript)
    values["manifest.json"] = _canonical_json(manifest)

    with pytest.raises(TTSVoiceBundleError):
        inspect_clone_voice_bundle(_pack_members(values))


def test_wav_must_be_exact_canonical_shape_even_with_matching_checksum() -> None:
    values = _members()
    original = values["reference.wav"]
    noncanonical = original[:12] + b"JUNK\x00\x00\x00\x00" + original[12:]
    noncanonical = (
        noncanonical[:4] + struct.pack("<I", len(noncanonical) - 8) + noncanonical[8:]
    )
    values["reference.wav"] = noncanonical
    manifest = json.loads(values["manifest.json"])
    _refresh_entry(manifest, "reference.wav", noncanonical)
    manifest["reference"]["byte_length"] = len(noncanonical)
    values["manifest.json"] = _canonical_json(manifest)

    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(_pack_members(values))


def test_crc_corruption_is_rejected_before_semantic_admission() -> None:
    payload = _valid()
    central = _central_offsets(payload)[-1]
    local = _local_offset(payload, central)
    changed = _patch_u32(payload, central + 16, 0)
    changed = _patch_u32(changed, local + 14, 0)

    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(changed)


@pytest.mark.parametrize(
    "limit_name",
    ["MAX_BUNDLE_EXPANSION_RATIO", "MAX_BUNDLE_UNCOMPRESSED_BYTES"],
)
def test_ratio_and_aggregate_limits_are_checked_from_metadata(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
) -> None:
    monkeypatch.setattr(codec, limit_name, 0)

    with pytest.raises(TTSVoiceBundleError) as caught:
        inspect_clone_voice_bundle(_valid())

    assert caught.value.code == "bundle_limit_exceeded"


def test_codec_never_calls_general_archive_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("general extraction used")

    monkeypatch.setattr(ZipFile, "extract", forbidden)
    monkeypatch.setattr(ZipFile, "extractall", forbidden)

    assert inspect_clone_voice_bundle(_valid()) == canonical_bundle()


@pytest.mark.parametrize(
    ("central_field", "value"),
    [
        (36, 1),  # central-directory start disk
        (36, 0xFFFF),
        (34, 1),  # internal attributes
    ],
)
def test_central_multipart_and_internal_attribute_values_are_rejected(
    central_field: int,
    value: int,
) -> None:
    payload = _valid()
    changed = payload
    for central in _central_offsets(payload):
        changed = _patch_u16(changed, central + central_field, value)

    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(changed)


@pytest.mark.parametrize("mode", [0o040700, 0o010600, 0o020600, 0o100600 | 0o4000])
def test_unix_directories_special_files_and_special_bits_are_rejected(
    mode: int,
) -> None:
    payload = _valid()
    changed = payload
    for central in _central_offsets(payload):
        changed = _patch_u32(changed, central + 38, mode << 16)

    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(changed)


def test_unknown_create_system_is_rejected() -> None:
    payload = _valid()
    changed = payload
    for central in _central_offsets(payload):
        changed = _patch_u16(changed, central + 4, (1 << 8) | 20)

    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(changed)


def test_local_and_central_metadata_must_agree() -> None:
    payload = _valid()
    central = _central_offsets(payload)[0]
    local = _local_offset(payload, central)
    changed = _patch_u16(payload, local + 10, 1)

    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(changed)


def test_layout_parser_itself_owns_local_name_agreement() -> None:
    with pytest.raises(Exception):
        codec._validate_layout(_local_name_disagreement(_valid()))


def test_invalid_dos_timestamp_is_rejected() -> None:
    payload = _valid()
    changed = payload
    for central in _central_offsets(payload):
        local = _local_offset(payload, central)
        changed = _patch_u16(changed, central + 12, 0)
        changed = _patch_u16(changed, central + 14, 0)
        changed = _patch_u16(changed, local + 10, 0)
        changed = _patch_u16(changed, local + 12, 0)

    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(changed)


def _deflate_with_trailing_member_junk() -> bytes:
    payload = _repack(compression=ZIP_DEFLATED)
    directory = struct.unpack_from("<L", payload, len(payload) - 6)[0]
    central = _central_offsets(payload)[-1]
    local = _local_offset(payload, central)
    compressed = struct.unpack_from("<L", payload, central + 20)[0]
    name_length, extra_length = struct.unpack_from("<2H", payload, local + 26)
    data_end = local + 30 + name_length + extra_length + compressed
    assert data_end == directory
    junk = b"TRAILING-COMPRESSED-JUNK"
    changed = bytearray(payload[:data_end] + junk + payload[data_end:])
    shifted_central = central + len(junk)
    shifted_eocd = len(changed) - 22
    struct.pack_into("<L", changed, local + 18, compressed + len(junk))
    struct.pack_into("<L", changed, shifted_central + 20, compressed + len(junk))
    struct.pack_into("<L", changed, shifted_eocd + 16, directory + len(junk))
    return bytes(changed)


def test_deflate_stream_rejects_trailing_raw_data_with_consistent_layout() -> None:
    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(_deflate_with_trailing_member_junk())


def test_deflate_does_not_delegate_decompression_to_zipfile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _repack(compression=ZIP_DEFLATED)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ZipFile.open must not decompress admitted members")

    monkeypatch.setattr(ZipFile, "open", forbidden)

    assert inspect_clone_voice_bundle(payload) == canonical_bundle()


@pytest.mark.parametrize(
    "voice_id", ["unsafe\x00voice", "unsafe\nvoice", "unsafe\u202evoice"]
)
def test_bundle_voice_id_rejects_control_and_format_characters(
    voice_id: str,
) -> None:
    values = _members()
    profile = json.loads(values["profile.json"])
    profile["voice_id"] = voice_id
    profile_bytes = _canonical_json(profile)
    values["profile.json"] = profile_bytes
    manifest = json.loads(values["manifest.json"])
    _refresh_entry(manifest, "profile.json", profile_bytes)
    values["manifest.json"] = _canonical_json(manifest)

    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(_pack_members(values))


def test_bundle_voice_id_preserves_safe_unicode_without_normalizing() -> None:
    voice_id = "Cafe\u0301"
    assert unicodedata.normalize("NFC", voice_id) != voice_id
    values = _members()
    profile = json.loads(values["profile.json"])
    profile["voice_id"] = voice_id
    profile_bytes = _canonical_json(profile)
    values["profile.json"] = profile_bytes
    manifest = json.loads(values["manifest.json"])
    _refresh_entry(manifest, "profile.json", profile_bytes)
    values["manifest.json"] = _canonical_json(manifest)

    inspected = inspect_clone_voice_bundle(_pack_members(values))

    assert inspected.profile.draft.voice_id == voice_id


def _with_dependency(
    *,
    recipe_id: str,
    recipe_revision: int,
    model_id: str,
    voice_id: str | None,
) -> bytes:
    values = _members()
    profile = json.loads(values["profile.json"])
    profile["model_id"] = model_id
    profile["voice_id"] = voice_id
    profile_bytes = _canonical_json(profile)
    values["profile.json"] = profile_bytes
    manifest = json.loads(values["manifest.json"])
    manifest["dependency"] = {
        "model_id": model_id,
        "provider_id": "audio_cpp",
        "recipe_id": recipe_id,
        "recipe_revision": recipe_revision,
    }
    _refresh_entry(manifest, "profile.json", profile_bytes)
    values["manifest.json"] = _canonical_json(manifest)
    return _pack_members(values)


@pytest.mark.parametrize(
    ("recipe_id", "revision", "model_id", "voice_id", "accepted"),
    [
        (
            "audio-cpp-0.5.1.supertonic.supertonic_3_orig",
            1,
            "supertonic-3-orig",
            None,
            False,
        ),
        (
            "audio-cpp-0.5.1.pocket_tts.pocket_tts_english_q8_0",
            2,
            "pocket-tts-english-q8-0",
            None,
            True,
        ),
        (
            "audio-cpp-0.5.1.pocket_tts.pocket_tts_english_q8_0",
            2,
            "pocket-tts-english-q8-0",
            "amy",
            False,
        ),
        (
            "audio-cpp-0.5.1.pocket_tts.pocket_tts_english_safetensors",
            1,
            "pocket-tts-english-safetensors",
            None,
            True,
        ),
        (
            "audio-cpp-0.5.1.pocket_tts.pocket_tts_english_safetensors",
            1,
            "pocket-tts-english-safetensors",
            "amy",
            False,
        ),
        ("future-recipe", 1, "future-model", "amy", True),
        (
            "audio-cpp-0.5.1.supertonic.supertonic_3_orig",
            2,
            "future-supertonic",
            "amy",
            True,
        ),
    ],
)
def test_exact_installed_recipe_voice_reference_policy_is_purely_enforced(
    recipe_id: str,
    revision: int,
    model_id: str,
    voice_id: str | None,
    accepted: bool,
) -> None:
    payload = _with_dependency(
        recipe_id=recipe_id,
        recipe_revision=revision,
        model_id=model_id,
        voice_id=voice_id,
    )
    if accepted:
        assert (
            inspect_clone_voice_bundle(payload).recipe_requirement.recipe_id
            == recipe_id
        )
    else:
        with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
            inspect_clone_voice_bundle(payload)


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b"", "bundle_invalid"),
        (_pack_members({**_members(), "reference.txt": b""}), "bundle_invalid"),
    ],
)
def test_empty_archive_or_member_is_malformed(payload: bytes, code: str) -> None:
    with pytest.raises(TTSVoiceBundleError) as caught:
        inspect_clone_voice_bundle(payload)
    assert caught.value.code == code


def test_only_true_archive_upper_bound_uses_limit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid()
    monkeypatch.setattr(codec, "MAX_BUNDLE_ARCHIVE_BYTES", len(payload) - 1)
    with pytest.raises(TTSVoiceBundleError) as caught:
        inspect_clone_voice_bundle(payload)
    assert caught.value.code == "bundle_limit_exceeded"
