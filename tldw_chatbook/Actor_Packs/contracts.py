"""Pure ``tldw.actor-pack/v1`` validation contracts.

This module deliberately knows nothing about ZIP files or host filesystem paths.
Export and import code consume this in-memory contract in later tasks.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import uuid
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


ACTOR_PACK_SCHEMA = "tldw.actor-pack/v1"
ACTOR_PAYLOAD_SCHEMA = "tldw.actor/v1"
MAX_MANIFEST_BYTES = 1 * 1024 * 1024
MAX_ACTOR_BYTES = 2 * 1024 * 1024
MAX_PORTRAIT_BYTES = 25 * 1024 * 1024
MAX_MEMBER_BYTES = 50 * 1024 * 1024
MAX_TOTAL_BYTES = 768 * 1024 * 1024
MAX_FILES = 4096
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 20_000
MAX_JSON_STRING = 4096
MAX_PORTRAIT_DIMENSION = 4096
MAX_PORTRAIT_PIXELS = MAX_PORTRAIT_DIMENSION * MAX_PORTRAIT_DIMENSION

ZIP_COMPRESSION = zipfile.ZIP_STORED
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_CREATE_SYSTEM = 3
ZIP_GENERAL_PURPOSE_FLAGS = 0
ZIP_EXTERNAL_ATTR = 0o100644 << 16

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SEGMENT_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{number}" for number in range(1, 10)}
    | {f"lpt{number}" for number in range(1, 10)}
)
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "actor",
        "sections",
        "producer",
        "license",
        "provenance",
        "required_features",
        "files",
        "content_digest",
    }
)
_ACTOR_KEYS = frozenset({"kind", "portable_uuid", "payload", "portrait"})
_SECTION_KEYS = frozenset({"kind", "manifest"})
_FILE_KEYS = frozenset({"path", "bytes", "sha256"})
_KNOWN_REQUIRED_FEATURES = frozenset(
    {"shared-visual-identity/v1", "persona-runtime/sprite-frames-v1"}
)
_CHARACTER_FIELDS = frozenset(
    {
        "name",
        "description",
        "personality",
        "scenario",
        "post_history_instructions",
        "first_message",
        "message_example",
        "creator_notes",
        "system_prompt",
        "alternate_greetings",
        "tags",
        "creator",
        "character_version",
        "extensions",
    }
)
_PERSONA_FIELDS = frozenset(
    {
        "name",
        "description",
        "archetype_key",
        "mode",
        "system_prompt",
        "is_active",
        "personality_traits",
        "use_persona_state_context_default",
        "voice_defaults",
        "setup",
    }
)
_PERSONA_VOICE_FIELDS = frozenset(
    {
        "stt_language",
        "stt_model",
        "tts_provider",
        "tts_voice",
        "confirmation_mode",
        "voice_chat_trigger_phrases",
        "auto_resume",
        "barge_in",
        "auto_commit_enabled",
        "vad_threshold",
        "min_silence_ms",
        "turn_stop_secs",
        "min_utterance_secs",
    }
)
_PERSONA_SETUP_FIELDS = frozenset(
    {
        "status",
        "version",
        "run_id",
        "current_step",
        "completed_steps",
        "completed_at",
        "last_test_type",
    }
)
_LOCAL_ONLY_ACTOR_FIELDS = frozenset(
    {
        "id",
        "record_id",
        "client_id",
        "version",
        "deleted",
        "backend",
        "created_at",
        "last_modified",
        "image",
        "avatar",
        "character_card_id",
    }
)


class ActorPackValidationError(ValueError):
    """Fixed-category Actor Pack validation failure."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class ActorPackFile:
    """One declared member's immutable public metadata."""

    path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ActorPackSection:
    """One typed optional visual section reference."""

    kind: str
    manifest_path: str


@dataclass(frozen=True, slots=True)
class ActorPackDocument:
    """Validated, path-free Actor Pack metadata."""

    schema: str
    actor_kind: str
    portable_uuid: str
    payload_path: str
    portrait_path: str
    sections: tuple[ActorPackSection, ...]
    files: tuple[ActorPackFile, ...]
    content_digest: str


def canonical_json_bytes(value: object) -> bytes:
    """Return the V1 canonical UTF-8 JSON representation."""

    _validate_json_tree(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, UnicodeError, ValueError):
        raise ActorPackValidationError("actor_pack_json_invalid") from None


def canonical_member_order(paths: object) -> tuple[str, ...]:
    """Return the only valid deterministic archive member order."""

    if type(paths) not in {list, tuple}:
        raise ActorPackValidationError("actor_pack_path_invalid")
    canonical = tuple(
        _path(path) if path != "actor-pack.json" else path for path in paths
    )
    if len(canonical) != len(set(canonical)):
        raise ActorPackValidationError("actor_pack_path_invalid")
    return tuple(sorted(canonical, key=lambda path: (path != "actor-pack.json", path)))


def build_file_inventory(files: Mapping[str, bytes]) -> tuple[ActorPackFile, ...]:
    """Build sorted immutable size/digest metadata for declared bytes."""

    supplied = _supplied_files(files)
    return tuple(
        ActorPackFile(path, len(data), hashlib.sha256(data).hexdigest())
        for path, data in sorted(supplied.items())
    )


def canonicalize_actor_payload(
    actor_kind: str,
    portable_uuid: str,
    actor_record: Mapping[str, Any],
) -> bytes:
    """Project one local actor record into canonical portable JSON."""

    if actor_kind not in {"character", "persona"} or not isinstance(
        actor_record, Mapping
    ):
        raise ActorPackValidationError("actor_pack_actor_invalid")
    identity = _portable_uuid(portable_uuid)
    allowed = _CHARACTER_FIELDS if actor_kind == "character" else _PERSONA_FIELDS
    keys = set(actor_record)
    if not keys <= allowed | _LOCAL_ONLY_ACTOR_FIELDS:
        raise ActorPackValidationError("actor_pack_actor_invalid")
    projected = {
        key: actor_record[key]
        for key in sorted(keys & allowed)
        if actor_record[key] is not None
    }
    payload = {
        "schema": ACTOR_PAYLOAD_SCHEMA,
        "actor_kind": actor_kind,
        "portable_uuid": identity,
        "data": projected,
    }
    encoded = canonical_json_bytes(payload)
    _validate_actor_payload(encoded, actor_kind=actor_kind, portable_uuid=identity)
    return encoded


def actor_pack_content_digest(manifest: Mapping[str, Any]) -> str:
    """Hash canonical top-level data while excluding only its digest field."""

    if not isinstance(manifest, Mapping):
        raise ActorPackValidationError("actor_pack_manifest_invalid")
    materialized = dict(manifest)
    materialized.pop("content_digest", None)
    return hashlib.sha256(canonical_json_bytes(materialized)).hexdigest()


def validate_actor_pack_document(
    manifest: Mapping[str, Any], files: Mapping[str, bytes]
) -> ActorPackDocument:
    """Validate one in-memory Actor Pack document.

    Args:
        manifest: Parsed root manifest.
        files: Declared member bytes, excluding ``actor-pack.json``.

    Returns:
        Immutable public metadata with no actor content or member bytes.

    Raises:
        ActorPackValidationError: A fixed, path-free validation category.
    """

    try:
        return _validate_actor_pack_document(manifest, files)
    except ActorPackValidationError:
        raise
    except (
        MemoryError,
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        raise ActorPackValidationError("actor_pack_manifest_invalid") from None


def validate_actor_pack_manifest(manifest: Mapping[str, Any]) -> ActorPackDocument:
    """Validate the bounded root contract without materializing member bytes."""

    try:
        return _validate_actor_pack_manifest(manifest)
    except ActorPackValidationError:
        raise
    except (
        MemoryError,
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        raise ActorPackValidationError("actor_pack_manifest_invalid") from None


def _validate_actor_pack_document(
    manifest: Mapping[str, Any], files: Mapping[str, bytes]
) -> ActorPackDocument:
    document = _validate_actor_pack_manifest(manifest)
    supplied = _supplied_files(files)
    if {item.path for item in document.files} != set(supplied):
        raise ActorPackValidationError("actor_pack_inventory_invalid")
    for item in document.files:
        data = supplied[item.path]
        if (
            item.byte_count != len(data)
            or item.sha256 != hashlib.sha256(data).hexdigest()
        ):
            raise ActorPackValidationError("actor_pack_inventory_mismatch")
    _validate_actor_payload(
        supplied[document.payload_path],
        actor_kind=document.actor_kind,
        portable_uuid=document.portable_uuid,
    )
    portrait = supplied[document.portrait_path]
    if len(portrait) > MAX_PORTRAIT_BYTES or not _validate_portrait(
        document.portrait_path, portrait
    ):
        raise ActorPackValidationError("actor_pack_portrait_invalid")
    for section in document.sections:
        if section.manifest_path not in supplied:
            raise ActorPackValidationError("actor_pack_section_invalid")
    return document


def _validate_actor_pack_manifest(manifest: Mapping[str, Any]) -> ActorPackDocument:
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
        raise ActorPackValidationError("actor_pack_manifest_invalid")
    if manifest.get("schema") != ACTOR_PACK_SCHEMA:
        raise ActorPackValidationError("actor_pack_schema_unsupported")
    if len(canonical_json_bytes(manifest)) > MAX_MANIFEST_BYTES:
        raise ActorPackValidationError("actor_pack_manifest_invalid")

    required_features = manifest.get("required_features")
    if (
        type(required_features) is not list
        or any(type(item) is not str or not item for item in required_features)
        or len(required_features) != len(set(required_features))
        or not set(required_features) <= _KNOWN_REQUIRED_FEATURES
    ):
        raise ActorPackValidationError("actor_pack_feature_unsupported")

    actor = manifest.get("actor")
    if not isinstance(actor, Mapping) or set(actor) != _ACTOR_KEYS:
        raise ActorPackValidationError("actor_pack_manifest_invalid")
    actor_kind = actor.get("kind")
    if actor_kind not in {"character", "persona"}:
        raise ActorPackValidationError("actor_pack_actor_invalid")
    portable_uuid = _portable_uuid(actor.get("portable_uuid"))
    payload_path = _path(actor.get("payload"))
    portrait_path = _path(actor.get("portrait"))
    if payload_path != "actor/actor.json" or not portrait_path.startswith(
        "actor/portrait."
    ):
        raise ActorPackValidationError("actor_pack_actor_invalid")

    sections = _sections(manifest.get("sections"), actor_kind)
    inventory = _inventory(manifest.get("files"))
    if sum(item.byte_count for item in inventory) > MAX_TOTAL_BYTES:
        raise ActorPackValidationError("actor_pack_inventory_invalid")

    digest = manifest.get("content_digest")
    if type(digest) is not str or not _SHA256_RE.fullmatch(digest):
        raise ActorPackValidationError("actor_pack_manifest_invalid")
    if digest != actor_pack_content_digest(manifest):
        raise ActorPackValidationError("actor_pack_digest_mismatch")

    return ActorPackDocument(
        schema=ACTOR_PACK_SCHEMA,
        actor_kind=actor_kind,
        portable_uuid=portable_uuid,
        payload_path=payload_path,
        portrait_path=portrait_path,
        sections=sections,
        files=inventory,
        content_digest=digest,
    )


def _portable_uuid(value: object) -> str:
    if type(value) is not str:
        raise ActorPackValidationError("actor_pack_actor_invalid")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError):
        raise ActorPackValidationError("actor_pack_actor_invalid") from None
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122 or str(parsed) != value:
        raise ActorPackValidationError("actor_pack_actor_invalid")
    return value


def _path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value or ":" in value:
        raise ActorPackValidationError("actor_pack_path_invalid")
    parts = value.split("/")
    if len(parts) < 2:
        raise ActorPackValidationError("actor_pack_path_invalid")
    for part in parts:
        if (
            not _SEGMENT_RE.fullmatch(part)
            or part in {".", ".."}
            or part.endswith((".", " "))
            or part.split(".", 1)[0] in _DEVICE_NAMES
        ):
            raise ActorPackValidationError("actor_pack_path_invalid")
    return value


def _sections(value: object, actor_kind: str) -> tuple[ActorPackSection, ...]:
    if type(value) is not list or len(value) > 2:
        raise ActorPackValidationError("actor_pack_section_invalid")
    records: list[ActorPackSection] = []
    kinds: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != _SECTION_KEYS:
            raise ActorPackValidationError("actor_pack_section_invalid")
        kind = raw.get("kind")
        if kind not in {"shared-visual-identity", "persona-runtime"} or kind in kinds:
            raise ActorPackValidationError("actor_pack_section_invalid")
        if kind == "persona-runtime" and actor_kind != "persona":
            raise ActorPackValidationError("actor_pack_section_invalid")
        manifest_path = _path(raw.get("manifest"))
        expected = f"{kind}/manifest.json"
        if manifest_path != expected:
            raise ActorPackValidationError("actor_pack_section_invalid")
        kinds.add(kind)
        records.append(ActorPackSection(kind, manifest_path))
    return tuple(records)


def _inventory(value: object) -> tuple[ActorPackFile, ...]:
    if type(value) is not list or not 2 <= len(value) <= MAX_FILES:
        raise ActorPackValidationError("actor_pack_inventory_invalid")
    records: list[ActorPackFile] = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != _FILE_KEYS:
            raise ActorPackValidationError("actor_pack_inventory_invalid")
        if raw.get("path") == "actor-pack.json":
            raise ActorPackValidationError("actor_pack_inventory_invalid")
        path = _path(raw.get("path"))
        byte_count = raw.get("bytes")
        sha256 = raw.get("sha256")
        if (
            type(byte_count) is not int
            or not 0 < byte_count <= MAX_MEMBER_BYTES
            or type(sha256) is not str
            or not _SHA256_RE.fullmatch(sha256)
        ):
            raise ActorPackValidationError("actor_pack_inventory_invalid")
        records.append(ActorPackFile(path, byte_count, sha256))
    if tuple(item.path for item in records) != tuple(
        sorted(item.path for item in records)
    ) or len({item.path for item in records}) != len(records):
        raise ActorPackValidationError("actor_pack_inventory_invalid")
    return tuple(records)


def _supplied_files(files: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(files, Mapping) or not 2 <= len(files) <= MAX_FILES:
        raise ActorPackValidationError("actor_pack_inventory_invalid")
    supplied: dict[str, bytes] = {}
    for raw_path, raw_data in files.items():
        path = _path(raw_path)
        if path == "actor-pack.json" or type(raw_data) is not bytes or not raw_data:
            raise ActorPackValidationError("actor_pack_inventory_invalid")
        if path in supplied:
            raise ActorPackValidationError("actor_pack_inventory_invalid")
        supplied[path] = raw_data
    return supplied


def _validate_actor_payload(
    data: bytes, *, actor_kind: str, portable_uuid: str
) -> None:
    if len(data) > MAX_ACTOR_BYTES:
        raise ActorPackValidationError("actor_pack_actor_invalid")
    try:
        payload = json.loads(data)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ActorPackValidationError("actor_pack_actor_invalid") from None
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != data:
        raise ActorPackValidationError("actor_pack_actor_invalid")
    if set(payload) != {"schema", "actor_kind", "portable_uuid", "data"}:
        raise ActorPackValidationError("actor_pack_actor_invalid")
    if (
        payload.get("schema") != ACTOR_PAYLOAD_SCHEMA
        or payload.get("actor_kind") != actor_kind
        or payload.get("portable_uuid") != portable_uuid
        or not isinstance(payload.get("data"), Mapping)
    ):
        raise ActorPackValidationError("actor_pack_actor_invalid")
    actor_data = payload["data"]
    allowed = _CHARACTER_FIELDS if actor_kind == "character" else _PERSONA_FIELDS
    if not set(actor_data) <= allowed or type(actor_data.get("name")) is not str:
        raise ActorPackValidationError("actor_pack_actor_invalid")
    if not actor_data["name"].strip() or len(actor_data["name"]) > 200:
        raise ActorPackValidationError("actor_pack_actor_invalid")
    _validate_actor_fields(actor_kind, actor_data)


def _validate_actor_fields(actor_kind: str, actor_data: Mapping[str, Any]) -> None:
    """Require values accepted by the local mutation boundary."""

    try:
        if actor_kind == "persona":
            nullable_text = {
                "description",
                "archetype_key",
                "system_prompt",
            }
            required_text = {"name", "personality_traits"}
            if any(
                actor_data[key] is not None and type(actor_data[key]) is not str
                for key in actor_data.keys() & nullable_text
            ) or any(
                type(actor_data[key]) is not str
                for key in actor_data.keys() & required_text
            ):
                raise ValueError
            if "mode" in actor_data and actor_data["mode"] not in {
                "session_scoped",
                "persistent_scoped",
            }:
                raise ValueError
            for key in actor_data.keys() & {
                "is_active",
                "use_persona_state_context_default",
            }:
                if type(actor_data[key]) is not bool:
                    raise ValueError
            for key in actor_data.keys() & {"voice_defaults", "setup"}:
                if type(actor_data[key]) is not dict:
                    raise ValueError
            if "voice_defaults" in actor_data:
                _validate_persona_voice_defaults(actor_data["voice_defaults"])
            if "setup" in actor_data:
                _validate_persona_setup(actor_data["setup"])
            return
        text_fields = _CHARACTER_FIELDS - {
            "alternate_greetings",
            "tags",
            "extensions",
        }
        if any(
            type(actor_data[key]) is not str for key in actor_data.keys() & text_fields
        ):
            raise ValueError
        for key in actor_data.keys() & {"alternate_greetings", "tags"}:
            value = actor_data[key]
            if type(value) is not list or any(type(item) is not str for item in value):
                raise ValueError
        if "extensions" in actor_data:
            extensions = actor_data["extensions"]
            if (
                type(extensions) is not dict
                or "actor_pack_persona_portrait_owner" in extensions
            ):
                raise ValueError
    except (ImportError, TypeError, ValueError):
        raise ActorPackValidationError("actor_pack_actor_invalid") from None


def _validate_persona_voice_defaults(value: Mapping[str, Any]) -> None:
    if not set(value) <= _PERSONA_VOICE_FIELDS:
        raise ValueError
    for key in value.keys() & {
        "stt_language",
        "stt_model",
        "tts_provider",
        "tts_voice",
    }:
        if value[key] is not None and type(value[key]) is not str:
            raise ValueError
    if "confirmation_mode" in value and value["confirmation_mode"] not in {
        None,
        "always",
        "destructive_only",
        "never",
    }:
        raise ValueError
    if "voice_chat_trigger_phrases" in value:
        phrases = value["voice_chat_trigger_phrases"]
        if type(phrases) is not list or any(type(item) is not str for item in phrases):
            raise ValueError
    for key in value.keys() & {
        "auto_resume",
        "barge_in",
        "auto_commit_enabled",
    }:
        if value[key] is not None and type(value[key]) is not bool:
            raise ValueError
    bounds = {
        "vad_threshold": (0.0, 1.0),
        "turn_stop_secs": (0.05, 10.0),
        "min_utterance_secs": (0.0, 10.0),
    }
    for key in value.keys() & bounds.keys():
        number = value[key]
        if number is not None and (
            type(number) not in {int, float}
            or not bounds[key][0] <= number <= bounds[key][1]
        ):
            raise ValueError
    silence = value.get("min_silence_ms")
    if silence is not None and (
        type(silence) is not int or not 50 <= silence <= 10_000
    ):
        raise ValueError


def _validate_persona_setup(value: Mapping[str, Any]) -> None:
    if not set(value) <= _PERSONA_SETUP_FIELDS:
        raise ValueError
    if value.get("status", "not_started") not in {
        "not_started",
        "in_progress",
        "completed",
    }:
        raise ValueError
    version = value.get("version", 1)
    if type(version) is not int or version < 1:
        raise ValueError
    run_id = value.get("run_id")
    if run_id is not None and (type(run_id) is not str or not 1 <= len(run_id) <= 200):
        raise ValueError
    steps = {"archetype", "persona", "voice", "commands", "safety", "test"}
    if value.get("current_step", "persona") not in steps:
        raise ValueError
    completed = value.get("completed_steps", [])
    if type(completed) is not list or any(item not in steps for item in completed):
        raise ValueError
    completed_at = value.get("completed_at")
    if completed_at is not None and type(completed_at) is not str:
        raise ValueError
    if value.get("last_test_type") not in {None, "dry_run", "live_session"}:
        raise ValueError


def _looks_like_raster(path: str, data: bytes) -> bool:
    suffix = path.rsplit(".", 1)[-1]
    if suffix == "png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix in {"jpg", "jpeg"}:
        return data.startswith(b"\xff\xd8\xff")
    if suffix == "gif":
        return data.startswith((b"GIF87a", b"GIF89a"))
    if suffix == "webp":
        return data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    return False


def _validate_portrait(path: str, data: bytes) -> bool:
    if not _looks_like_raster(path, data):
        return False
    # PIL stays function-local (TASK-21103): this module is in the app's
    # import closure via Actor_Packs.persona_coordinator, and a module-level
    # PIL import would put it back on the boot path.
    from PIL import Image, UnidentifiedImageError

    try:
        from PIL.Image import DecompressionBombError
    except ImportError:  # pragma: no cover - Pillow is a required project dependency.
        DecompressionBombError = OSError  # type: ignore[misc, assignment]
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            expected = {
                "png": "PNG",
                "jpg": "JPEG",
                "jpeg": "JPEG",
                "gif": "GIF",
                "webp": "WEBP",
            }[path.rsplit(".", 1)[-1]]
            if (
                image.format != expected
                or type(width) is not int
                or type(height) is not int
                or width < 1
                or height < 1
                or width > MAX_PORTRAIT_DIMENSION
                or height > MAX_PORTRAIT_DIMENSION
                or width * height > MAX_PORTRAIT_PIXELS
            ):
                return False
            image.verify()
    except (
        DecompressionBombError,
        KeyError,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        ValueError,
    ):
        return False
    return True


def validate_actor_portrait(path: str, data: bytes) -> None:
    """Validate one bounded raster portrait using the Actor Pack contract."""

    if (
        type(path) is not str
        or type(data) is not bytes
        or not data
        or len(data) > MAX_PORTRAIT_BYTES
        or not _validate_portrait(path, data)
    ):
        raise ActorPackValidationError("actor_pack_portrait_invalid")


def validate_actor_payload(data: bytes, *, actor_kind: str, portable_uuid: str) -> None:
    """Validate one bounded canonical actor payload independently."""

    _validate_actor_payload(
        data,
        actor_kind=actor_kind,
        portable_uuid=_portable_uuid(portable_uuid),
    )


def _validate_json_tree(value: object) -> None:
    nodes = 0
    active: set[int] = set()

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise ActorPackValidationError("actor_pack_json_invalid")
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            if not -(2**63) <= item <= 2**63 - 1:
                raise ActorPackValidationError("actor_pack_json_invalid")
            return
        if type(item) is float:
            if item != item or item in {float("inf"), float("-inf")}:
                raise ActorPackValidationError("actor_pack_json_invalid")
            return
        if type(item) is str:
            if len(item) > MAX_JSON_STRING:
                raise ActorPackValidationError("actor_pack_json_invalid")
            return
        if type(item) not in {dict, list}:
            raise ActorPackValidationError("actor_pack_json_invalid")
        marker = id(item)
        if marker in active:
            raise ActorPackValidationError("actor_pack_json_invalid")
        active.add(marker)
        try:
            if type(item) is dict:
                for key, child in item.items():
                    if type(key) is not str or not key or len(key) > 128:
                        raise ActorPackValidationError("actor_pack_json_invalid")
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
        finally:
            active.remove(marker)

    visit(value, 0)


__all__ = [
    "ACTOR_PACK_SCHEMA",
    "ACTOR_PAYLOAD_SCHEMA",
    "ActorPackDocument",
    "ActorPackFile",
    "ActorPackSection",
    "ActorPackValidationError",
    "ZIP_COMPRESSION",
    "ZIP_CREATE_SYSTEM",
    "ZIP_EXTERNAL_ATTR",
    "ZIP_GENERAL_PURPOSE_FLAGS",
    "ZIP_TIMESTAMP",
    "actor_pack_content_digest",
    "build_file_inventory",
    "canonical_json_bytes",
    "canonical_member_order",
    "canonicalize_actor_payload",
    "validate_actor_portrait",
    "validate_actor_payload",
    "validate_actor_pack_document",
    "validate_actor_pack_manifest",
]
