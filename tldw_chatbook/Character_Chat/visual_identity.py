"""Visual Identity expression, manifest, and immutable asset contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from importlib import resources
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from PIL import Image, UnidentifiedImageError

from tldw_chatbook.config import get_user_data_dir

CANONICAL_EXPRESSION_SLOTS = (
    "neutral",
    "happy",
    "excited",
    "sad",
    "angry",
    "thinking",
    "confused",
    "surprised",
)
CUSTOM_EXPRESSION_PREFIX = "custom:"

EXPRESSION_ALIASES = {
    "default": "neutral",
    "normal": "neutral",
    "calm": "neutral",
    "joy": "happy",
    "joyful": "happy",
    "cheerful": "happy",
    "hype": "excited",
    "thrilled": "excited",
    "upset": "sad",
    "sorrowful": "sad",
    "mad": "angry",
    "annoyed": "angry",
    "furious": "angry",
    "anger": "angry",
    "thoughtful": "thinking",
    "pondering": "thinking",
    "unsure": "confused",
    "puzzled": "confused",
    "shocked": "surprised",
    "astonished": "surprised",
}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_expression_key(value: str) -> str | None:
    """Normalize a user-facing expression label into a canonical or custom key."""
    if not isinstance(value, str):
        return None

    raw_value = value.strip()
    if not raw_value:
        return None

    if raw_value.lower().startswith(CUSTOM_EXPRESSION_PREFIX):
        custom_part = raw_value[len(CUSTOM_EXPRESSION_PREFIX) :]
        custom_key = _sanitize_expression_token(custom_part)
        return f"{CUSTOM_EXPRESSION_PREFIX}{custom_key}" if custom_key else None

    normalized = _sanitize_expression_token(raw_value)
    if not normalized:
        return None
    if normalized in CANONICAL_EXPRESSION_SLOTS:
        return normalized
    alias = EXPRESSION_ALIASES.get(normalized)
    if alias is not None:
        return alias
    return f"{CUSTOM_EXPRESSION_PREFIX}{normalized}"


def normalize_expression_filename(filename: str) -> str | None:
    """Normalize a source filename stem into a canonical or custom expression key."""
    if not isinstance(filename, str):
        return None

    basename = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if "." in basename:
        basename = basename.rsplit(".", 1)[0]
    normalized = _sanitize_expression_token(basename)
    if not normalized:
        return None
    if normalized in CANONICAL_EXPRESSION_SLOTS:
        return normalized
    alias = EXPRESSION_ALIASES.get(normalized)
    if alias is not None:
        return alias
    return f"{CUSTOM_EXPRESSION_PREFIX}{normalized}"


def is_custom_expression_key(value: str) -> bool:
    """Return whether a value normalizes to a custom expression key."""
    normalized = normalize_expression_key(value)
    return normalized is not None and normalized.startswith(CUSTOM_EXPRESSION_PREFIX)


def display_label_for_expression_key(value: str) -> str:
    """Build a human-readable label for a canonical, alias, or custom expression key."""
    normalized = normalize_expression_key(value)
    if normalized is None:
        return ""
    if normalized.startswith(CUSTOM_EXPRESSION_PREFIX):
        normalized = normalized[len(CUSTOM_EXPRESSION_PREFIX) :]
    return normalized.replace("_", " ").title()


def _sanitize_expression_token(value: str) -> str:
    normalized = _NON_ALNUM_RE.sub("_", value.strip().lower())
    return normalized.strip("_")


SAMIRA_REACTION_LABELS = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "neutral",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "thinking",
    "speaking",
    "error",
)

SAMIRA_EXPRESSION_KEYS = {
    "admiration": "custom:admiration",
    "amusement": "custom:amusement",
    "anger": "angry",
    "annoyance": "custom:annoyance",
    "approval": "custom:approval",
    "caring": "custom:caring",
    "confusion": "confused",
    "curiosity": "custom:curiosity",
    "desire": "custom:desire",
    "disappointment": "custom:disappointment",
    "disapproval": "custom:disapproval",
    "disgust": "custom:disgust",
    "embarrassment": "custom:embarrassment",
    "excitement": "excited",
    "fear": "custom:fear",
    "gratitude": "custom:gratitude",
    "grief": "custom:grief",
    "joy": "happy",
    "love": "custom:love",
    "nervousness": "custom:nervousness",
    "neutral": "neutral",
    "optimism": "custom:optimism",
    "pride": "custom:pride",
    "realization": "custom:realization",
    "relief": "custom:relief",
    "remorse": "custom:remorse",
    "sadness": "sad",
    "surprise": "surprised",
    "thinking": "thinking",
    "speaking": "custom:speaking",
    "error": "custom:error",
}

SAMIRA_PACK_ID = "tldw.builtin.samira.reactions"
SAMIRA_MANIFEST_SCHEMA_ID = "tldw.visual_identity_pack/v1"
SAMIRA_LICENSE = "AGPL-3.0-or-later"
SAMIRA_DEFAULT_EXPRESSION_KEY = "neutral"
SAMIRA_SERVER_COMMIT = "385afa951922c8a9dc2002c675bb6cad65e4ac23"

SAMIRA_MAX_REACTION_BYTES = 1024 * 1024
SAMIRA_MAX_REACTIONS_BYTES = 16 * 1024 * 1024
SAMIRA_MAX_DIRECTORY_BYTES = 20 * 1024 * 1024

_LOWER_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_LICENSE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*\Z")
_EXPECTED_IMAGE_FORMATS = {
    "image/gif": "GIF",
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
_USER_SOURCE_KINDS = frozenset({"manual"})


@dataclass(frozen=True, slots=True)
class VisualIdentityManifestAsset:
    """One immutable asset declared by a validated pack manifest."""

    expression_key: str
    original_label: str
    display_label: str
    storage_relpath: str
    content_type: str
    bytes: int
    width: int
    height: int
    sha256: str
    is_animated: bool
    frame_count: int
    duration_ms: int | None

    @property
    def relative_filename(self) -> str:
        """Return the digest-facing relative filename."""
        return self.storage_relpath


@dataclass(frozen=True, slots=True)
class VisualIdentityManifest:
    """Validated immutable subset of a Visual Identity pack manifest."""

    schema_id: str
    pack_id: str
    title: str
    license: str
    default_expression_key: str
    source_server_commit: str | None
    pack_content_sha256: str
    assets: tuple[VisualIdentityManifestAsset, ...]


@dataclass(frozen=True, slots=True)
class LoadedVisualIdentityAsset:
    """Verified bytes for one selected Visual Identity asset."""

    asset: VisualIdentityManifestAsset
    data: bytes


def compute_pack_content_sha256(
    manifest: Mapping[str, Any] | VisualIdentityManifest,
) -> str:
    """Compute the canonical content digest for a Visual Identity pack.

    Args:
        manifest: Raw or validated manifest data. Only content-bearing fields
            are projected into the canonical payload.

    Returns:
        Lowercase SHA-256 of compact canonical UTF-8 JSON bytes.

    Raises:
        TypeError: If required payload fields cannot be read.
        ValueError: If canonical JSON contains a non-finite numeric value.
    """
    canonical_json = _canonical_pack_content_json(manifest)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _canonical_pack_content_json(
    manifest: Mapping[str, Any] | VisualIdentityManifest,
) -> str:
    """Return the exact canonical JSON hashed for pack content identity."""
    if isinstance(manifest, VisualIdentityManifest):
        schema_id = manifest.schema_id
        pack_id = manifest.pack_id
        default_expression_key = manifest.default_expression_key
        license_id = manifest.license
        assets: Any = manifest.assets
    else:
        schema_id = manifest["schema_id"]
        pack_id = manifest["pack_id"]
        default_expression_key = manifest["default_expression_key"]
        license_id = manifest["license"]
        assets = manifest["assets"]

    projected_assets = sorted(
        (_content_asset_payload(asset) for asset in assets),
        key=lambda asset: asset["original_label"],
    )
    payload = {
        "schema_id": schema_id,
        "pack_id": pack_id,
        "default_expression_key": default_expression_key,
        "license": license_id,
        "assets": projected_assets,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def validate_visual_identity_manifest(
    data: Mapping[str, Any],
    *,
    require_samira_bundle: bool = False,
    directory_bytes: int | None = None,
) -> VisualIdentityManifest:
    """Validate a Visual Identity pack manifest without reading asset bytes.

    Args:
        data: Parsed manifest mapping.
        require_samira_bundle: Require the exact bundled Samira contract.
        directory_bytes: Optional byte count for the complete supplied pack
            directory. The Samira bundle enforces its directory cap when set.

    Returns:
        Frozen validated manifest data.

    Raises:
        ValueError: With a stable validation category when data is invalid.
    """
    if not isinstance(data, Mapping):
        raise ValueError("visual_identity_manifest_invalid")

    try:
        schema_id = _nonempty_string(data.get("schema_id"))
        pack_id = _nonempty_string(data.get("pack_id"))
        license_id = _nonempty_string(data.get("license"))
        default_expression_key = _valid_expression_key(
            data.get("default_expression_key")
        )
        title = _nonempty_string(data.get("title", pack_id))
        source_commit_value = data.get("source_server_commit")
        source_server_commit = (
            None
            if source_commit_value is None
            else _nonempty_string(source_commit_value)
        )
        digest = _lower_sha256(data.get("pack_content_sha256"))
        if schema_id != SAMIRA_MANIFEST_SCHEMA_ID or not _LICENSE_RE.fullmatch(
            license_id
        ):
            raise ValueError
        raw_assets = data.get("assets")
        if not isinstance(raw_assets, list) or not raw_assets:
            raise ValueError
        assets = tuple(_validate_manifest_asset(asset) for asset in raw_assets)
        _require_unique_assets(assets)
        if default_expression_key not in {asset.expression_key for asset in assets}:
            raise ValueError
        _validate_directory_bytes(directory_bytes)
    except (KeyError, TypeError, ValueError):
        raise ValueError("visual_identity_manifest_invalid") from None

    manifest = VisualIdentityManifest(
        schema_id=schema_id,
        pack_id=pack_id,
        title=title,
        license=license_id,
        default_expression_key=default_expression_key,
        source_server_commit=source_server_commit,
        pack_content_sha256=digest,
        assets=assets,
    )
    if compute_pack_content_sha256(manifest) != digest:
        raise ValueError("visual_identity_digest_mismatch")

    if require_samira_bundle:
        _validate_samira_manifest(manifest, directory_bytes=directory_bytes)
    return manifest


def load_visual_identity_asset(
    asset: VisualIdentityManifestAsset,
    *,
    source_kind: str,
    user_data_dir: str | Path | None = None,
) -> LoadedVisualIdentityAsset:
    """Read and verify one selected immutable Visual Identity asset.

    Args:
        asset: Validated asset metadata.
        source_kind: Owning pack source, either ``builtin`` or profile-owned
            ``manual``.
        user_data_dir: Injectable profile data root for user-owned sources.

    Returns:
        Frozen asset metadata and verified bytes.

    Raises:
        ValueError: With a stable category for unsafe, unavailable, or corrupt
            asset data.
    """
    if not isinstance(asset, VisualIdentityManifestAsset):
        raise ValueError("visual_identity_manifest_invalid")
    if source_kind != "builtin" and source_kind not in _USER_SOURCE_KINDS:
        raise ValueError("visual_identity_source_kind_unsupported")

    parts = _safe_relative_parts(asset.storage_relpath)
    if source_kind == "builtin":
        try:
            data = (
                resources.files("tldw_chatbook").joinpath("assets", *parts).read_bytes()
            )
        except (OSError, TypeError):
            raise ValueError("visual_identity_asset_unavailable") from None
    else:
        profile_root = (
            Path(user_data_dir) if user_data_dir is not None else get_user_data_dir()
        )
        resolved_profile_root = profile_root.resolve(strict=False)
        lexical_assets_root = resolved_profile_root / "visual_identities"
        resolved_assets_root = lexical_assets_root.resolve(strict=False)
        if not resolved_assets_root.is_relative_to(resolved_profile_root):
            raise ValueError("visual_identity_path_invalid")
        candidate = resolved_assets_root.joinpath(*parts).resolve(strict=False)
        if not candidate.is_relative_to(resolved_assets_root):
            raise ValueError("visual_identity_path_invalid")
        try:
            data = candidate.read_bytes()
        except OSError:
            raise ValueError("visual_identity_asset_unavailable") from None

    if len(data) != asset.bytes:
        raise ValueError("visual_identity_asset_size_mismatch")
    if hashlib.sha256(data).hexdigest() != asset.sha256:
        raise ValueError("visual_identity_asset_sha256_mismatch")
    return LoadedVisualIdentityAsset(asset=asset, data=data)


def validate_visual_identity_assets(
    manifest: VisualIdentityManifest,
    *,
    source_kind: str,
    user_data_dir: str | Path | None = None,
    directory_bytes: int | None = None,
) -> tuple[LoadedVisualIdentityAsset, ...]:
    """Load and completely validate every asset in a candidate pack.

    Args:
        manifest: Previously validated manifest.
        source_kind: Owning pack source kind.
        user_data_dir: Injectable profile data root for user-owned sources.
        directory_bytes: Optional complete supplied-directory byte count.

    Returns:
        Every loaded asset in manifest order.

    Raises:
        ValueError: With a stable category for manifest, budget, byte, or image
            validation failures.
    """
    if not isinstance(manifest, VisualIdentityManifest):
        raise ValueError("visual_identity_manifest_invalid")
    try:
        _validate_directory_bytes(directory_bytes)
    except ValueError:
        raise ValueError("visual_identity_manifest_invalid") from None
    if manifest.pack_id == SAMIRA_PACK_ID:
        _validate_samira_manifest(manifest, directory_bytes=directory_bytes)

    loaded_assets = tuple(
        load_visual_identity_asset(
            asset,
            source_kind=source_kind,
            user_data_dir=user_data_dir,
        )
        for asset in manifest.assets
    )
    for loaded in loaded_assets:
        _validate_image_bytes(loaded)
    return loaded_assets


def _content_asset_payload(
    asset: Mapping[str, Any] | VisualIdentityManifestAsset,
) -> dict[str, Any]:
    if isinstance(asset, VisualIdentityManifestAsset):
        return {
            "expression_key": asset.expression_key,
            "original_label": asset.original_label,
            "relative_filename": asset.storage_relpath,
            "content_type": asset.content_type,
            "bytes": asset.bytes,
            "width": asset.width,
            "height": asset.height,
            "sha256": asset.sha256,
        }
    relpath = asset.get("storage_relpath", asset.get("relative_filename"))
    return {
        "expression_key": asset["expression_key"],
        "original_label": asset["original_label"],
        "relative_filename": relpath,
        "content_type": asset["content_type"],
        "bytes": asset["bytes"],
        "width": asset["width"],
        "height": asset["height"],
        "sha256": asset["sha256"],
    }


def _nonempty_string(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError
    return value


def _positive_int(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError
    return value


def _lower_sha256(value: Any) -> str:
    if not isinstance(value, str) or _LOWER_SHA256_RE.fullmatch(value) is None:
        raise ValueError
    return value


def _valid_expression_key(value: Any) -> str:
    key = _nonempty_string(value)
    if normalize_expression_key(key) != key:
        raise ValueError
    return key


def _validate_manifest_asset(data: Any) -> VisualIdentityManifestAsset:
    if not isinstance(data, Mapping):
        raise ValueError
    expression_key = _valid_expression_key(data.get("expression_key"))
    original_label = _nonempty_string(data.get("original_label"))
    display_label = _nonempty_string(data.get("display_label"))
    storage_value = data.get("storage_relpath", data.get("relative_filename"))
    if (
        "storage_relpath" in data
        and "relative_filename" in data
        and data["storage_relpath"] != data["relative_filename"]
    ):
        raise ValueError
    storage_relpath = _nonempty_string(storage_value)
    _safe_relative_parts(storage_relpath)
    content_type = _nonempty_string(data.get("content_type"))
    if content_type not in _EXPECTED_IMAGE_FORMATS:
        raise ValueError
    byte_count = _positive_int(data.get("bytes"))
    width = _positive_int(data.get("width"))
    height = _positive_int(data.get("height"))
    sha256 = _lower_sha256(data.get("sha256"))
    is_animated = data.get("is_animated")
    if type(is_animated) is not bool:
        raise ValueError
    frame_count = _positive_int(data.get("frame_count"))
    duration_ms = data.get("duration_ms")
    if is_animated:
        if frame_count <= 1 or type(duration_ms) is not int or duration_ms <= 0:
            raise ValueError
    elif frame_count != 1 or duration_ms is not None:
        raise ValueError

    return VisualIdentityManifestAsset(
        expression_key=expression_key,
        original_label=original_label,
        display_label=display_label,
        storage_relpath=storage_relpath,
        content_type=content_type,
        bytes=byte_count,
        width=width,
        height=height,
        sha256=sha256,
        is_animated=is_animated,
        frame_count=frame_count,
        duration_ms=duration_ms,
    )


def _require_unique_assets(assets: tuple[VisualIdentityManifestAsset, ...]) -> None:
    expression_keys = {asset.expression_key for asset in assets}
    original_labels = {asset.original_label for asset in assets}
    if len(expression_keys) != len(assets) or len(original_labels) != len(assets):
        raise ValueError


def _safe_relative_parts(value: str) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
        or PurePosixPath(value).is_absolute()
    ):
        raise ValueError("visual_identity_path_invalid")
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("visual_identity_path_invalid")
    return tuple(components)


def _validate_directory_bytes(directory_bytes: int | None) -> None:
    if directory_bytes is not None and (
        type(directory_bytes) is not int or directory_bytes < 0
    ):
        raise ValueError


def _validate_samira_manifest(
    manifest: VisualIdentityManifest,
    *,
    directory_bytes: int | None,
) -> None:
    labels = tuple(asset.original_label for asset in manifest.assets)
    mappings = {asset.original_label: asset.expression_key for asset in manifest.assets}
    exact_contract = (
        manifest.schema_id == SAMIRA_MANIFEST_SCHEMA_ID
        and manifest.pack_id == SAMIRA_PACK_ID
        and manifest.license == SAMIRA_LICENSE
        and manifest.default_expression_key == SAMIRA_DEFAULT_EXPRESSION_KEY
        and manifest.source_server_commit == SAMIRA_SERVER_COMMIT
        and labels == SAMIRA_REACTION_LABELS
        and mappings == SAMIRA_EXPRESSION_KEYS
        and all(
            asset.storage_relpath
            == f"characters/samira/expressions/{asset.original_label}.webp"
            and asset.content_type == "image/webp"
            and asset.width == 1024
            and asset.height == 1024
            and not asset.is_animated
            and asset.frame_count == 1
            and asset.duration_ms is None
            for asset in manifest.assets
        )
    )
    if not exact_contract:
        raise ValueError("visual_identity_samira_contract_invalid")
    if any(asset.bytes > SAMIRA_MAX_REACTION_BYTES for asset in manifest.assets):
        raise ValueError("visual_identity_budget_exceeded")
    if sum(asset.bytes for asset in manifest.assets) > SAMIRA_MAX_REACTIONS_BYTES:
        raise ValueError("visual_identity_budget_exceeded")
    if directory_bytes is not None and directory_bytes > SAMIRA_MAX_DIRECTORY_BYTES:
        raise ValueError("visual_identity_budget_exceeded")


def _validate_image_bytes(loaded: LoadedVisualIdentityAsset) -> None:
    asset = loaded.asset
    try:
        with Image.open(BytesIO(loaded.data)) as image:
            image_format = image.format
            image_size = image.size
            frame_count = max(int(getattr(image, "n_frames", 1) or 1), 1)
            is_animated = bool(getattr(image, "is_animated", False)) or frame_count > 1
            duration_ms = (
                _image_duration_ms(image, frame_count) if is_animated else None
            )
            image.verify()
    except (OSError, UnidentifiedImageError, ValueError):
        raise ValueError("visual_identity_asset_decode_invalid") from None

    try:
        expected_format = _EXPECTED_IMAGE_FORMATS[asset.content_type]
    except KeyError:
        raise ValueError("visual_identity_asset_format_mismatch") from None
    if image_format != expected_format:
        raise ValueError("visual_identity_asset_format_mismatch")
    if image_size != (asset.width, asset.height):
        raise ValueError("visual_identity_asset_dimensions_mismatch")
    if (
        frame_count != asset.frame_count
        or is_animated != asset.is_animated
        or duration_ms != asset.duration_ms
    ):
        raise ValueError("visual_identity_asset_frame_mismatch")


def _image_duration_ms(image: Image.Image, frame_count: int) -> int:
    duration_ms = 0
    for frame_index in range(frame_count):
        image.seek(frame_index)
        duration_ms += int(image.info.get("duration") or 0)
    return duration_ms
