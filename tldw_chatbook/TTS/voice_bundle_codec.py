"""Deterministic clone-voice bundles with a hostile ZIP admission boundary."""

from __future__ import annotations

import asyncio
import json
import re
import stat
import struct
import unicodedata
import zlib
from binascii import crc32
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from io import BytesIO
from typing import BinaryIO, Final, Literal, NoReturn, cast
from uuid import UUID
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY
from tldw_chatbook.TTS.profile_portability import PortableTTSProfile
from tldw_chatbook.TTS.profile_reference_audio import validate_canonical_reference_wav
from tldw_chatbook.TTS.profile_reference_types import (
    CanonicalTTSCloneReference,
    TTSCloneRecipeRequirement,
    validate_reference_text,
)
from tldw_chatbook.TTS.profile_types import TTSProfileDraft

EXPECTED_MEMBER_ORDER: Final[tuple[str, str, str, str]] = (
    "manifest.json",
    "profile.json",
    "reference.wav",
    "reference.txt",
)
MAX_BUNDLE_ARCHIVE_BYTES = 40 * 1024 * 1024
MAX_BUNDLE_UNCOMPRESSED_BYTES = 33 * 1024 * 1024
MAX_BUNDLE_EXPANSION_RATIO = 100
_MEMBER_LIMITS: Final[dict[str, int]] = {
    "manifest.json": 64 * 1024,
    "profile.json": 16 * 1024,
    "reference.wav": 32 * 1024 * 1024,
    "reference.txt": 16 * 1024,
}
_EOCD = struct.Struct("<4s4H2LH")
_CENTRAL = struct.Struct("<4s6H3L5H2L")
_LOCAL = struct.Struct("<4s5H3L2H")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_JSON_CHUNK_BYTES = 64 * 1024

TTSVoiceBundleErrorCode = Literal[
    "acknowledgement_required",
    "bundle_invalid",
    "bundle_limit_exceeded",
    "cleanup_failed",
    "destination_changed",
    "operation_failed",
    "source_changed",
    "stale_inspection",
    "unsupported_bundle",
    "unsupported_platform",
]
_ControlFlowCode = Literal["cancelled", "keyboard_interrupt", "system_exit"]
_FailureCode = TTSVoiceBundleErrorCode | _ControlFlowCode
_CONTROL_FLOW_CODES = frozenset({"cancelled", "keyboard_interrupt", "system_exit"})
_ERROR_CODES = frozenset(
    {
        "acknowledgement_required",
        "bundle_invalid",
        "bundle_limit_exceeded",
        "cleanup_failed",
        "destination_changed",
        "operation_failed",
        "source_changed",
        "stale_inspection",
        "unsupported_bundle",
        "unsupported_platform",
    }
)


class TTSVoiceBundleError(Exception):
    """Bounded archive error which never retains collaborator detail."""

    __slots__ = ("code",)

    def __init__(self, code: TTSVoiceBundleErrorCode) -> None:
        if type(code) is not str or code not in _ERROR_CODES:
            code = "bundle_invalid"
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True, repr=False)
class TTSCloneVoiceBundle:
    """Exact validated profile, dependency, and private clone reference."""

    profile: PortableTTSProfile
    reference: CanonicalTTSCloneReference
    recipe_requirement: TTSCloneRecipeRequirement

    def __post_init__(self) -> None:
        if (
            type(self.profile) is not PortableTTSProfile
            or type(self.reference) is not CanonicalTTSCloneReference
            or type(self.recipe_requirement) is not TTSCloneRecipeRequirement
            or self.profile.draft.provider_id != "audio_cpp"
            or self.profile.draft.model_id != self.recipe_requirement.model_id
            or not _voice_id_is_safe(self.profile.draft.voice_id)
            or not _installed_recipe_admits(
                self.recipe_requirement,
                has_voice=self.profile.draft.voice_id is not None,
            )
        ):
            raise TTSVoiceBundleError("bundle_invalid")
        try:
            metadata = validate_canonical_reference_wav(self.reference.wav_bytes)
        except Exception:
            raise TTSVoiceBundleError("bundle_invalid") from None
        if (
            metadata.byte_length != self.reference.byte_length
            or metadata.duration_ms != self.reference.duration_ms
            or metadata.sample_rate_hz != self.reference.sample_rate_hz
            or metadata.channels != self.reference.channels
            or metadata.sample_encoding != self.reference.sample_encoding
        ):
            raise TTSVoiceBundleError("bundle_invalid")

    def __repr__(self) -> str:
        return "TTSCloneVoiceBundle(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class TTSVoiceBundleSinks:
    """Caller-owned, nontransactional destinations for validated member bytes.

    A sink failure may leave partial data in any destination. The portability
    service must discard its fresh staging destinations after failure; the codec
    does not roll them back.
    """

    manifest_json: BinaryIO
    profile_json: BinaryIO
    reference_wav: BinaryIO
    reference_txt: BinaryIO

    def __repr__(self) -> str:
        return "TTSVoiceBundleSinks(<private>)"

    def for_member(self, name: str) -> BinaryIO:
        return {
            "manifest.json": self.manifest_json,
            "profile.json": self.profile_json,
            "reference.wav": self.reference_wav,
            "reference.txt": self.reference_txt,
        }[name]


@dataclass(frozen=True, slots=True)
class _MemberLayout:
    name: str
    flags: int
    compression: int
    crc: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int
    data_offset: int
    data_end: int


def _voice_id_is_safe(value: str | None) -> bool:
    if value is None:
        return True
    if type(value) is not str or not value or len(value) > 256:
        return False
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        return False
    for character in value:
        code_point = ord(character)
        if (
            unicodedata.category(character) in {"Cc", "Cf", "Cs"}
            or 0xFDD0 <= code_point <= 0xFDEF
            or code_point & 0xFFFF in (0xFFFE, 0xFFFF)
        ):
            return False
    return True


def _installed_recipe_admits(
    requirement: TTSCloneRecipeRequirement,
    *,
    has_voice: bool,
) -> bool:
    for recipe in AUDIO_CPP_RECIPE_REGISTRY.recipes:
        if (
            recipe.recipe_id == requirement.recipe_id
            and recipe.recipe_revision == requirement.recipe_revision
        ):
            return recipe.admits_voice_reference(
                has_voice=has_voice,
                has_reference=True,
            )
    return True


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


def _profile_payload(source: TTSCloneVoiceBundle) -> dict[str, object]:
    draft = source.profile.draft
    return {
        "model_id": draft.model_id,
        "name": draft.display_name,
        "options": {},
        "profile_id": str(source.profile.profile_id),
        "provider_id": "audio_cpp",
        "response_format": "wav",
        "schema_version": 1,
        "speed": 1.0,
        "voice_id": draft.voice_id,
    }


def _manifest_payload(
    source: TTSCloneVoiceBundle,
    members: dict[str, bytes],
) -> dict[str, object]:
    reference = source.reference
    requirement = source.recipe_requirement
    return {
        "bundle_format": "tldw_chatbook.clone_voice_bundle",
        "declaration": {
            "plaintext_sensitive_data_acknowledged": True,
            "version": 1,
        },
        "dependency": {
            "model_id": requirement.model_id,
            "provider_id": "audio_cpp",
            "recipe_id": requirement.recipe_id,
            "recipe_revision": requirement.recipe_revision,
        },
        "entries": {
            name: {"sha256": sha256(payload).hexdigest(), "size": len(payload)}
            for name, payload in members.items()
        },
        "reference": {
            "byte_length": reference.byte_length,
            "channels": reference.channels,
            "duration_ms": reference.duration_ms,
            "sample_encoding": reference.sample_encoding,
            "sample_rate_hz": reference.sample_rate_hz,
        },
        "schema_version": 1,
    }


def _encode_bundle(source: TTSCloneVoiceBundle) -> bytes:
    if type(source) is not TTSCloneVoiceBundle:
        raise TTSVoiceBundleError("bundle_invalid")
    profile = _canonical_json(_profile_payload(source))
    members = {
        "profile.json": profile,
        "reference.txt": source.reference.reference_text.encode("utf-8"),
        "reference.wav": source.reference.wav_bytes,
    }
    manifest = _canonical_json(_manifest_payload(source, members))
    payloads = {
        "manifest.json": manifest,
        "profile.json": profile,
        "reference.wav": source.reference.wav_bytes,
        "reference.txt": source.reference.reference_text.encode("utf-8"),
    }
    target = BytesIO()
    with ZipFile(target, "w", compression=ZIP_STORED, allowZip64=False) as archive:
        for name in EXPECTED_MEMBER_ORDER:
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_STORED
            info.flag_bits = 0
            info.extract_version = 20
            info.create_version = 20
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            info.internal_attr = 0
            archive.writestr(info, payloads[name])
    return target.getvalue()


def _strict_json(payload: bytes) -> dict[str, object]:
    if payload.startswith(b"\xef\xbb\xbf") or not payload.endswith(b"\n"):
        raise ValueError
    text = payload.decode("utf-8", errors="strict")

    def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError
            value[key] = item
        return value

    value = json.loads(
        text,
        object_pairs_hook=exact_object,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
    )
    if (
        type(value) is not dict
        or _json_container_depth(value) > 4
        or _canonical_json(value) != payload
    ):
        raise ValueError
    return cast(dict[str, object], value)


def _json_container_depth(value: object) -> int:
    items: Iterable[object]
    if type(value) is dict:
        items = cast(dict[str, object], value).values()
    elif type(value) is list:
        items = cast(list[object], value)
    else:
        return 0
    return 1 + max((_json_container_depth(item) for item in items), default=0)


def _metadata_is_accepted(
    *,
    created: int,
    needed: int,
    flags: int,
    compression: int,
    modified_time: int,
    modified_date: int,
    internal_attributes: int,
    external_attributes: int,
) -> bool:
    create_system = created >> 8
    create_version = created & 0xFF
    if (
        flags not in (0, 1 << 11)
        or compression not in (ZIP_STORED, ZIP_DEFLATED)
        or create_version not in (10, 20)
        or internal_attributes != 0
    ):
        return False
    if compression == ZIP_STORED:
        if needed not in (10, 20):
            return False
    elif needed != 20:
        return False
    try:
        datetime(
            1980 + ((modified_date >> 9) & 0x7F),
            (modified_date >> 5) & 0x0F,
            modified_date & 0x1F,
            (modified_time >> 11) & 0x1F,
            (modified_time >> 5) & 0x3F,
            (modified_time & 0x1F) * 2,
        )
    except ValueError:
        return False
    dos_attributes = external_attributes & 0xFFFF
    if dos_attributes & 0x18:
        return False
    if create_system == 0:
        return True
    if create_system != 3:
        return False
    mode = (external_attributes >> 16) & 0xFFFF
    return stat.S_ISREG(mode) and mode & 0o7000 == 0


def _validate_layout(payload: bytes) -> tuple[_MemberLayout, ...]:
    if not payload:
        raise ValueError
    if len(payload) > MAX_BUNDLE_ARCHIVE_BYTES:
        raise TTSVoiceBundleError("bundle_limit_exceeded")
    if len(payload) < _EOCD.size:
        raise ValueError
    eocd_offset = len(payload) - _EOCD.size
    (
        signature,
        disk_number,
        directory_disk,
        disk_members,
        total_members,
        directory_size,
        directory_offset,
        comment_length,
    ) = _EOCD.unpack_from(payload, eocd_offset)
    if (
        signature != b"PK\x05\x06"
        or disk_number != 0
        or directory_disk != 0
        or disk_members != len(EXPECTED_MEMBER_ORDER)
        or total_members != len(EXPECTED_MEMBER_ORDER)
        or comment_length != 0
        or directory_size in (0xFFFFFFFF,)
        or directory_offset in (0xFFFFFFFF,)
        or directory_offset + directory_size != eocd_offset
    ):
        raise ValueError

    cursor = directory_offset
    layouts: list[_MemberLayout] = []
    compressed_total = 0
    uncompressed_total = 0
    for expected_name in EXPECTED_MEMBER_ORDER:
        if cursor + _CENTRAL.size > eocd_offset:
            raise ValueError
        (
            central_signature,
            created,
            needed,
            flags,
            compression,
            modified_time,
            modified_date,
            crc,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            member_comment_length,
            member_disk,
            internal_attributes,
            external_attributes,
            local_offset,
        ) = _CENTRAL.unpack_from(payload, cursor)
        name_start = cursor + _CENTRAL.size
        name_end = name_start + name_length
        record_end = name_end + extra_length + member_comment_length
        if (
            central_signature != b"PK\x01\x02"
            or record_end > eocd_offset
            or extra_length != 0
            or member_comment_length != 0
            or member_disk != 0
            or compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_offset == 0xFFFFFFFF
            or payload[name_start:name_end] != expected_name.encode("ascii")
            or not _metadata_is_accepted(
                created=created,
                needed=needed,
                flags=flags,
                compression=compression,
                modified_time=modified_time,
                modified_date=modified_date,
                internal_attributes=internal_attributes,
                external_attributes=external_attributes,
            )
        ):
            raise ValueError
        if compressed_size <= 0 or uncompressed_size <= 0:
            raise ValueError
        if (
            uncompressed_size > _MEMBER_LIMITS[expected_name]
            or compressed_size > MAX_BUNDLE_ARCHIVE_BYTES
            or uncompressed_size > compressed_size * MAX_BUNDLE_EXPANSION_RATIO
            or (compression == ZIP_STORED and compressed_size != uncompressed_size)
        ):
            raise TTSVoiceBundleError("bundle_limit_exceeded")

        if local_offset + _LOCAL.size > directory_offset:
            raise ValueError
        (
            local_signature,
            local_needed,
            local_flags,
            local_compression,
            local_time,
            local_date,
            local_crc,
            local_compressed_size,
            local_uncompressed_size,
            local_name_length,
            local_extra_length,
        ) = _LOCAL.unpack_from(payload, local_offset)
        local_name_start = local_offset + _LOCAL.size
        local_name_end = local_name_start + local_name_length
        data_offset = local_name_end + local_extra_length
        data_end = data_offset + compressed_size
        if (
            local_signature != b"PK\x03\x04"
            or local_extra_length != 0
            or data_end > directory_offset
            or local_needed != needed
            or local_flags != flags
            or local_compression != compression
            or local_time != modified_time
            or local_date != modified_date
            or local_crc != crc
            or local_compressed_size != compressed_size
            or local_uncompressed_size != uncompressed_size
            or payload[local_name_start:local_name_end] != expected_name.encode("ascii")
        ):
            raise ValueError
        layouts.append(
            _MemberLayout(
                name=expected_name,
                flags=flags,
                compression=compression,
                crc=crc,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_offset=local_offset,
                data_offset=data_offset,
                data_end=data_end,
            )
        )
        compressed_total += compressed_size
        uncompressed_total += uncompressed_size
        cursor = record_end

    if cursor != eocd_offset:
        raise ValueError
    if (
        compressed_total > MAX_BUNDLE_ARCHIVE_BYTES
        or uncompressed_total > MAX_BUNDLE_UNCOMPRESSED_BYTES
    ):
        raise TTSVoiceBundleError("bundle_limit_exceeded")
    expected_offset = 0
    for layout in layouts:
        if layout.local_offset != expected_offset:
            raise ValueError
        expected_offset = layout.data_end
    if expected_offset != directory_offset:
        raise ValueError
    return tuple(layouts)


def _default_sinks() -> TTSVoiceBundleSinks:
    return TTSVoiceBundleSinks(
        manifest_json=BytesIO(),
        profile_json=BytesIO(),
        reference_wav=BytesIO(),
        reference_txt=BytesIO(),
    )


def _stream_member(
    payload: bytes,
    layout: _MemberLayout,
    sink: BinaryIO,
    limit: int,
) -> bytes:
    sink.seek(0)
    sink.truncate(0)
    total = 0
    checksum = 0

    def write(chunk: bytes) -> None:
        nonlocal checksum, total
        if chunk:
            total += len(chunk)
            if total > limit:
                raise TTSVoiceBundleError("bundle_limit_exceeded")
            if total > layout.uncompressed_size:
                raise ValueError
            written = sink.write(chunk)
            if written is not None and written != len(chunk):
                raise ValueError
            checksum = crc32(chunk, checksum)

    raw = memoryview(payload)[layout.data_offset : layout.data_end]
    if len(raw) != layout.compressed_size:
        raise ValueError
    if layout.compression == ZIP_STORED:
        for offset in range(0, len(raw), _JSON_CHUNK_BYTES):
            write(bytes(raw[offset : offset + _JSON_CHUNK_BYTES]))
    else:
        decompressor = zlib.decompressobj(-15)
        consumed = 0
        while consumed < len(raw):
            chunk = bytes(raw[consumed : consumed + _JSON_CHUNK_BYTES])
            consumed += len(chunk)
            pending = chunk
            while pending:
                output = decompressor.decompress(
                    pending,
                    min(_JSON_CHUNK_BYTES, limit + 1 - total),
                )
                pending = decompressor.unconsumed_tail
                write(output)
                if decompressor.unused_data:
                    raise ValueError
                if decompressor.eof and (pending or consumed != len(raw)):
                    raise ValueError
        if (
            not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
        ):
            raise ValueError
    if total != layout.uncompressed_size or checksum & 0xFFFFFFFF != layout.crc:
        raise ValueError
    sink.flush()
    sink.seek(0)
    result = sink.read(limit + 1)
    if type(result) is not bytes or len(result) != total:
        raise ValueError
    return result


def _exact_object(
    value: object,
    keys: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict or frozenset(cast(dict[object, object], value)) != keys:
        raise ValueError
    return cast(dict[str, object], value)


def _validate_profile(value: dict[str, object]) -> PortableTTSProfile:
    profile = _exact_object(
        value,
        frozenset(
            {
                "model_id",
                "name",
                "options",
                "profile_id",
                "provider_id",
                "response_format",
                "schema_version",
                "speed",
                "voice_id",
            }
        ),
    )
    if type(profile["schema_version"]) is not int:
        raise ValueError
    if profile["schema_version"] != 1:
        raise TTSVoiceBundleError("unsupported_bundle")
    profile_id = profile["profile_id"]
    if type(profile_id) is not str:
        raise ValueError
    parsed_id = UUID(profile_id)
    if str(parsed_id) != profile_id:
        raise ValueError
    voice_id = profile["voice_id"]
    if (
        profile["provider_id"] != "audio_cpp"
        or profile["response_format"] != "wav"
        or type(profile["speed"]) is not float
        or profile["speed"] != 1.0
        or type(profile["options"]) is not dict
        or bool(profile["options"])
        or (voice_id is not None and type(voice_id) is not str)
        or type(profile["name"]) is not str
        or type(profile["model_id"]) is not str
    ):
        raise ValueError
    result = PortableTTSProfile(
        profile_id=parsed_id,
        draft=TTSProfileDraft(
            display_name=cast(str, profile["name"]),
            provider_id="audio_cpp",
            model_id=cast(str, profile["model_id"]),
            voice_id=cast(str | None, voice_id),
            response_format="wav",
            speed=1.0,
            options={},
        ),
    )
    if result.draft.display_name != profile["name"]:
        raise ValueError
    return result


def _validate_manifest(
    value: dict[str, object],
    members: dict[str, bytes],
    profile: PortableTTSProfile,
    metadata: object,
) -> TTSCloneRecipeRequirement:
    manifest = _exact_object(
        value,
        frozenset(
            {
                "bundle_format",
                "declaration",
                "dependency",
                "entries",
                "reference",
                "schema_version",
            }
        ),
    )
    if type(manifest["schema_version"]) is not int:
        raise ValueError
    if manifest["schema_version"] != 1:
        raise TTSVoiceBundleError("unsupported_bundle")
    if manifest["bundle_format"] != "tldw_chatbook.clone_voice_bundle":
        raise TTSVoiceBundleError("unsupported_bundle")
    declaration = _exact_object(
        manifest["declaration"],
        frozenset({"plaintext_sensitive_data_acknowledged", "version"}),
    )
    if (
        declaration["plaintext_sensitive_data_acknowledged"] is not True
        or type(declaration["version"]) is not int
        or declaration["version"] != 1
    ):
        raise ValueError
    dependency = _exact_object(
        manifest["dependency"],
        frozenset({"model_id", "provider_id", "recipe_id", "recipe_revision"}),
    )
    if dependency["provider_id"] != "audio_cpp":
        raise ValueError
    requirement = TTSCloneRecipeRequirement(
        recipe_id=cast(str, dependency["recipe_id"]),
        recipe_revision=cast(int, dependency["recipe_revision"]),
        model_id=cast(str, dependency["model_id"]),
    )
    if requirement.model_id != profile.draft.model_id:
        raise ValueError

    entries = _exact_object(
        manifest["entries"],
        frozenset({"profile.json", "reference.wav", "reference.txt"}),
    )
    for name in ("profile.json", "reference.wav", "reference.txt"):
        declared = _exact_object(entries[name], frozenset({"sha256", "size"}))
        digest = declared["sha256"]
        size = declared["size"]
        if (
            type(digest) is not str
            or _SHA256.fullmatch(digest) is None
            or type(size) is not int
            or size != len(members[name])
            or digest != sha256(members[name]).hexdigest()
        ):
            raise ValueError

    reference = _exact_object(
        manifest["reference"],
        frozenset(
            {
                "byte_length",
                "channels",
                "duration_ms",
                "sample_encoding",
                "sample_rate_hz",
            }
        ),
    )
    facts = {
        "byte_length": getattr(metadata, "byte_length"),
        "channels": getattr(metadata, "channels"),
        "duration_ms": getattr(metadata, "duration_ms"),
        "sample_encoding": getattr(metadata, "sample_encoding"),
        "sample_rate_hz": getattr(metadata, "sample_rate_hz"),
    }
    if any(
        type(reference[key]) is not type(expected) for key, expected in facts.items()
    ):
        raise ValueError
    if reference != facts:
        raise ValueError
    return requirement


def _decode_bundle(
    payload: bytes,
    sinks: TTSVoiceBundleSinks | None,
) -> TTSCloneVoiceBundle:
    if sinks is None:
        destinations = _default_sinks()
    else:
        if type(sinks) is not TTSVoiceBundleSinks:
            raise ValueError
        destinations = sinks
        _validate_sinks(destinations)
    layouts = _validate_layout(payload)
    members = {
        layout.name: _stream_member(
            payload,
            layout,
            destinations.for_member(layout.name),
            _MEMBER_LIMITS[layout.name],
        )
        for layout in layouts
    }
    profile_data = _strict_json(members["profile.json"])
    manifest = _strict_json(members["manifest.json"])
    transcript_bytes = members["reference.txt"]
    if transcript_bytes.startswith(b"\xef\xbb\xbf"):
        raise ValueError
    transcript = transcript_bytes.decode("utf-8", errors="strict")
    if validate_reference_text(transcript) != transcript:
        raise ValueError
    metadata = validate_canonical_reference_wav(members["reference.wav"])
    profile = _validate_profile(profile_data)
    requirement = _validate_manifest(manifest, members, profile, metadata)
    reference = CanonicalTTSCloneReference(
        wav_bytes=members["reference.wav"],
        reference_text=transcript,
        sha256=sha256(members["reference.wav"]).hexdigest(),
        byte_length=metadata.byte_length,
        duration_ms=metadata.duration_ms,
        sample_rate_hz=metadata.sample_rate_hz,
        channels=metadata.channels,
        sample_encoding=metadata.sample_encoding,
    )
    return TTSCloneVoiceBundle(
        profile=profile,
        reference=reference,
        recipe_requirement=requirement,
    )


def _validate_sinks(sinks: TTSVoiceBundleSinks) -> None:
    destinations = (
        sinks.manifest_json,
        sinks.profile_json,
        sinks.reference_wav,
        sinks.reference_txt,
    )
    if len({id(destination) for destination in destinations}) != len(destinations):
        raise ValueError
    methods = (
        "flush",
        "read",
        "readable",
        "seek",
        "seekable",
        "truncate",
        "write",
        "writable",
    )
    for destination in destinations:
        if any(not callable(getattr(destination, method, None)) for method in methods):
            raise ValueError
        if (
            destination.readable() is not True
            or destination.seekable() is not True
            or destination.writable() is not True
        ):
            raise ValueError


def _classify_and_sever(error: BaseException) -> _FailureCode:
    if type(error) is asyncio.CancelledError:
        code: _FailureCode = "cancelled"
    elif type(error) is KeyboardInterrupt:
        code = "keyboard_interrupt"
    elif type(error) is SystemExit:
        code = "system_exit"
    elif type(error) is TTSVoiceBundleError:
        code = error.code
    else:
        code = "bundle_invalid"
    BaseException.__setattr__(error, "__traceback__", None)
    BaseException.__setattr__(error, "__cause__", None)
    BaseException.__setattr__(error, "__context__", None)
    return code


def _encode_outcome(source: TTSCloneVoiceBundle) -> bytes | _FailureCode:
    try:
        return _encode_bundle(source)
    except BaseException as error:
        code = _classify_and_sever(error)
        return code if code in _CONTROL_FLOW_CODES else "bundle_invalid"


def _inspect_outcome(
    payload: bytes,
) -> TTSCloneVoiceBundle | _FailureCode:
    try:
        if type(payload) is not bytes:
            raise ValueError
        return _decode_bundle(payload, None)
    except BaseException as error:
        return _classify_and_sever(error)


def _inspect_outcome_with_sinks(
    payload: bytes,
    sinks: TTSVoiceBundleSinks,
) -> TTSCloneVoiceBundle | _FailureCode:
    try:
        if type(payload) is not bytes or type(sinks) is not TTSVoiceBundleSinks:
            raise ValueError
        return _decode_bundle(payload, sinks)
    except BaseException as error:
        return _classify_and_sever(error)


def _raise_failure(code: _FailureCode) -> NoReturn:
    if code == "cancelled":
        raise asyncio.CancelledError() from None
    if code == "keyboard_interrupt":
        raise KeyboardInterrupt() from None
    if code == "system_exit":
        raise SystemExit() from None
    raise TTSVoiceBundleError(code) from None


def encode_clone_voice_bundle(source: TTSCloneVoiceBundle) -> bytes:
    """Return the deterministic four-member ZIP for one canonical clone voice."""

    outcome = _encode_outcome(source)
    del source
    if type(outcome) is str:
        _raise_failure(outcome)
    return cast(bytes, outcome)


def inspect_clone_voice_bundle(
    payload: bytes,
    *,
    sinks: TTSVoiceBundleSinks | None = None,
) -> TTSCloneVoiceBundle:
    """Validate one hostile bundle without extracting archive paths."""

    outcome = (
        _inspect_outcome(payload)
        if sinks is None
        else _inspect_outcome_with_sinks(payload, sinks)
    )
    del payload, sinks
    if type(outcome) is str:
        _raise_failure(outcome)
    return cast(TTSCloneVoiceBundle, outcome)


__all__ = [
    "EXPECTED_MEMBER_ORDER",
    "TTSCloneVoiceBundle",
    "TTSVoiceBundleError",
    "TTSVoiceBundleSinks",
    "encode_clone_voice_bundle",
    "inspect_clone_voice_bundle",
]
