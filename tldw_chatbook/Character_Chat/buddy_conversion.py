"""Reviewed, one-time Buddy timelines converted into ordinary character assets."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from tldw_chatbook.Actor_Packs.activation import ActorPackActivationResult
    from tldw_chatbook.Persona_Visual.snapshot import BuddySnapshot

MAX_CONVERSION_RGBA_BYTES = 64 * 1024 * 1024
_CONVERSION_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class BuddyMapping:
    source_state: str
    expression_key: str
    fallback: bool
    frame_count: int


@dataclass(frozen=True, slots=True)
class BuddyExpression:
    source_state: str
    expression_key: str
    data: bytes = field(repr=False)
    metadata: Mapping[str, Any]
    fallback: bool
    source_asset_sha256: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BuddyConversion:
    snapshot: BuddySnapshot = field(repr=False)
    expressions: tuple[BuddyExpression, ...]
    portrait: bytes = field(repr=False)
    warnings: tuple[str, ...]
    converted_at: str


def _manifest(snapshot: BuddySnapshot):
    from tldw_chatbook.Persona_Visual.validation import validate_persona_visual_manifest

    return validate_persona_visual_manifest(
        json.loads(snapshot.manifest_json),
        {
            asset.metadata.asset_key: (asset.metadata.width, asset.metadata.height)
            for asset in snapshot.assets
        },
    )


def _selections(snapshot: BuddySnapshot) -> dict[str, tuple[Any, bool, int]]:
    from tldw_chatbook.Persona_Visual.contracts import (
        REQUIRED_STATES,
        _select_static_frame,
        resolve_manifest_state,
    )

    manifest = _manifest(snapshot)
    selections = {}
    used = set()
    states = dict.fromkeys(
        (
            *REQUIRED_STATES,
            *manifest.states,
            *manifest.fallbacks,
            *manifest.state_catalog,
        )
    )
    for state in states:
        selection = resolve_manifest_state(manifest, state)
        if selection is not None:
            selections[state] = (
                selection.animation,
                selection.resolved_state != state,
                selection.static.frame_index,
            )
            used.add(selection.animation_id)
    for key, animation in manifest.animations.items():
        if key not in used:
            label = f"animation:{key}"
            while label in selections:
                label = f"animation:{label}"
            selections[label] = (
                animation,
                False,
                _select_static_frame(animation).frame_index,
            )
    return selections


def suggest_buddy_mappings(snapshot: BuddySnapshot) -> tuple[BuddyMapping, ...]:
    """Retain each supported source sequence without guessing emotions for actions."""
    from .visual_identity import normalize_expression_key

    rows = []
    for source, (animation, fallback, _) in _selections(snapshot).items():
        if len(source.encode("utf-8")) > 128 or any(
            c.isspace() or ord(c) < 32 or ord(c) == 127 for c in source
        ):
            raise ValueError("buddy_conversion_source_state_invalid")
        label = source.removeprefix("animation:")
        key = "neutral" if source == "idle" else normalize_expression_key(label)
        if key is None:
            raise ValueError("buddy_conversion_mapping_invalid")
        rows.append(BuddyMapping(source, key, fallback, len(animation.frames)))
    return tuple(rows)


def _geometry(animation: Any, assets: Mapping[str, Any]) -> tuple[int, int, int]:
    widths, heights = [], []
    source_pixels = 0
    for frame in animation.frames:
        asset = assets[frame.asset_id].metadata
        widths.append(frame.region.width if frame.region else asset.width)
        heights.append(frame.region.height if frame.region else asset.height)
        source_pixels = max(source_pixels, asset.width * asset.height)
    return max(widths), max(heights), source_pixels


def _raster(
    frame: Any, animation: Any, assets: Mapping[str, Any], size: tuple[int, int]
):
    from PIL import Image

    asset = assets[frame.asset_id]
    if hashlib.sha256(asset.data).hexdigest() != asset.metadata.sha256:
        raise ValueError("buddy_conversion_source_invalid")
    with Image.open(BytesIO(asset.data)) as source:
        if source.size != (asset.metadata.width, asset.metadata.height):
            raise ValueError("buddy_conversion_source_invalid")
        source.seek(
            0
        )  # Native Persona Visual frame selection never expands nested animation.
        source.load()
        raster = source.convert("RGBA")
    try:
        if frame.region is not None:
            region = frame.region
            cropped = raster.crop(
                (region.x, region.y, region.x + region.width, region.y + region.height)
            )
            raster.close()
            raster = cropped
        alignment = animation.alignment
        x, y = (alignment.x, alignment.y) if alignment is not None else (0.5, 0.5)
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        canvas.paste(
            raster,
            (round((size[0] - raster.width) * x), round((size[1] - raster.height) * y)),
        )
        return canvas
    finally:
        raster.close()


class _BoundedOutput(BytesIO):
    """Enforce the expression byte limit while Pillow writes encoded output."""

    def write(self, data):
        from .visual_identity import MAX_EXPRESSION_ASSET_BYTES

        if self.tell() + len(data) > MAX_EXPRESSION_ASSET_BYTES:
            raise ValueError("buddy_conversion_budget_exceeded")
        return super().write(data)


def _png(image: Any) -> bytes:
    output = _BoundedOutput()
    image.save(output, format="PNG")
    return output.getvalue()


def _same_visible_pixels(left: Any, right: Any) -> bool:
    """Require exact alpha and RGB wherever pixels have nonzero coverage."""
    from PIL import ImageChops

    if left.size != right.size:
        return False
    with ImageChops.difference(left, right) as difference:
        with difference.getchannel("A") as alpha_difference:
            if alpha_difference.getbbox() is not None:
                return False
        with (
            left.getchannel("A") as alpha,
            alpha.point(lambda value: 255 if value else 0) as mask,
        ):
            for channel in ("R", "G", "B"):
                with (
                    difference.getchannel(channel) as values,
                    ImageChops.multiply(values, mask) as visible,
                ):
                    if visible.getbbox() is not None:
                        return False
    return True


def _verified_timeline(
    data: bytes, frames: list[Any], durations: list[int], loop: bool
) -> bool:
    """Compare visible intervals, allowing an encoder to merge identical frames."""
    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        if image.n_frames == 1:
            with image.convert("RGBA") as visible:
                for frame in frames:
                    if not _same_visible_pixels(visible, frame):
                        raise ValueError("buddy_conversion_animation_unfaithful")
            return False
        if image.info.get("loop") != (0 if loop else 1):
            raise ValueError("buddy_conversion_animation_unfaithful")
        source_index = 0
        source_remaining = durations[0]
        total = 0
        for index in range(image.n_frames):
            image.seek(index)
            image.load()
            duration = image.info.get("duration")
            if type(duration) is not int or duration <= 0:
                raise ValueError("buddy_conversion_animation_unfaithful")
            total += duration
            with image.convert("RGBA") as visible:
                remaining = duration
                while remaining:
                    if source_index >= len(frames):
                        raise ValueError("buddy_conversion_animation_unfaithful")
                    if not _same_visible_pixels(visible, frames[source_index]):
                        raise ValueError("buddy_conversion_animation_unfaithful")
                    elapsed = min(remaining, source_remaining)
                    remaining -= elapsed
                    source_remaining -= elapsed
                    if source_remaining == 0:
                        source_index += 1
                        if source_index < len(frames):
                            source_remaining = durations[source_index]
        if total != sum(durations) or source_index != len(frames):
            raise ValueError("buddy_conversion_animation_unfaithful")
    return True


def _encode(
    animation: Any, assets: Mapping[str, Any], *, animate: bool, retained_bytes: int = 0
) -> tuple[bytes, str | None]:
    from PIL import features

    width, height, source_pixels = _geometry(animation, assets)
    count = len(animation.frames) if animate else 1
    # Source decoding, retained frames, encoder copies and sequential verification.
    if (
        retained_bytes + 4 * (source_pixels * 3 + width * height * (2 * count + 6))
        > MAX_CONVERSION_RGBA_BYTES
    ):
        raise ValueError("buddy_conversion_budget_exceeded")
    frames = []
    try:
        first = _raster(animation.frames[0], animation, assets, (width, height))
        frames.append(first)
        if not animate or len(animation.frames) == 1:
            return _png(first), None
        if not features.check("webp"):
            return (
                _png(first),
                "Animated WebP encoding is unavailable; this expression uses its first frame.",
            )
        durations = [
            frame.duration_ms
            if frame.duration_ms is not None
            else 1000 / animation.frame_rate
            for frame in animation.frames
        ]
        if any(not float(value).is_integer() or value <= 0 for value in durations):
            return (
                _png(first),
                "This frame rate cannot be represented exactly in milliseconds; this expression uses its first frame.",
            )
        durations = [int(value) for value in durations]
        for frame in animation.frames[1:]:
            frames.append(_raster(frame, animation, assets, (width, height)))
        output = _BoundedOutput()
        try:
            first.save(
                output,
                format="WEBP",
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0 if animation.loop else 1,
                lossless=True,
                exact=True,
                method=4,
            )
            data = output.getvalue()
            if not _verified_timeline(data, frames, durations, animation.loop):
                return _png(first), None
            return data, None
        except (OSError, ValueError):
            return (
                _png(first),
                "The encoder could not preserve this animation exactly; this expression uses its first frame.",
            )
    finally:
        for frame in frames:
            frame.close()


def _metadata(data: bytes, *, decoded_pixels_before: int = 0) -> dict[str, Any]:
    from .visual_identity import MAX_EXPRESSION_ASSET_BYTES, _inspect_image_bytes

    if len(data) > MAX_EXPRESSION_ASSET_BYTES:
        raise ValueError("buddy_conversion_budget_exceeded")
    format_name, (width, height), count, animated, duration, _ = _inspect_image_bytes(
        data, decoded_pixels_before=decoded_pixels_before
    )
    return {
        "content_type": {"PNG": "image/png", "WEBP": "image/webp"}[format_name],
        "width": width,
        "height": height,
        "frame_count": count,
        "is_animated": animated,
        "duration_ms": duration,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def convert_buddy(
    snapshot: BuddySnapshot,
    mappings: Mapping[str, str | None] | None = None,
    *,
    animate: bool = True,
    portrait_state: str = "idle",
    portrait_frame: int | None = None,
) -> BuddyConversion:
    """Convert only reviewed mappings; source state and actor ownership stay separate."""
    from .visual_identity import (
        MAX_EXPRESSION_PACK_ASSETS,
        MAX_EXPRESSION_TOTAL_BYTES,
        normalize_expression_key,
    )

    if type(animate) is not bool or not snapshot.is_current():
        raise ValueError("buddy_conversion_stale")
    selections = _selections(snapshot)
    expected = {
        row.source_state: row.expression_key for row in suggest_buddy_mappings(snapshot)
    }
    mapping = expected if mappings is None else dict(mappings)
    if set(mapping) != set(expected):
        raise ValueError("buddy_conversion_mapping_incomplete")
    chosen = []
    for source, label in mapping.items():
        if label is None:
            continue
        key = normalize_expression_key(label)
        if key is None:
            raise ValueError("buddy_conversion_mapping_invalid")
        chosen.append((source, key))
    keys = [key for _, key in chosen]
    if len(keys) != len(set(keys)):
        raise ValueError("buddy_conversion_mapping_collision")
    if not chosen or "neutral" not in keys:
        raise ValueError("buddy_conversion_neutral_required")
    if len(chosen) > MAX_EXPRESSION_PACK_ASSETS:
        raise ValueError("buddy_conversion_budget_exceeded")
    if portrait_state not in selections:
        raise ValueError("buddy_conversion_portrait_invalid")
    assets = {asset.metadata.asset_key: asset for asset in snapshot.assets}
    expressions, warnings = [], []
    total_bytes = decoded_pixels = 0
    with _CONVERSION_LOCK:
        for source, key in chosen:
            animation, fallback, _ = selections[source]
            data, warning = _encode(
                animation, assets, animate=animate, retained_bytes=total_bytes
            )
            metadata = _metadata(data, decoded_pixels_before=decoded_pixels)
            total_bytes += len(data)
            decoded_pixels += (
                metadata["width"] * metadata["height"] * metadata["frame_count"]
            )
            if total_bytes > min(MAX_EXPRESSION_TOTAL_BYTES, MAX_CONVERSION_RGBA_BYTES):
                raise ValueError("buddy_conversion_budget_exceeded")
            if warning:
                warnings.append(f"{source}: {warning}")
            expressions.append(
                BuddyExpression(
                    source,
                    key,
                    data,
                    MappingProxyType(metadata),
                    fallback,
                    tuple(
                        sorted(
                            {
                                assets[frame.asset_id].metadata.sha256
                                for frame in animation.frames
                            }
                        )
                    ),
                )
            )
        animation, _, default_frame = selections[portrait_state]
        selected = default_frame if portrait_frame is None else portrait_frame
        if type(selected) is not int or not 0 <= selected < len(animation.frames):
            raise ValueError("buddy_conversion_portrait_invalid")
        width, height, source_pixels = _geometry(animation, assets)
        if (
            total_bytes + 4 * (source_pixels * 3 + width * height * 8)
            > MAX_CONVERSION_RGBA_BYTES
        ):
            raise ValueError("buddy_conversion_budget_exceeded")
        with _raster(
            animation.frames[selected], animation, assets, (width, height)
        ) as portrait:
            portrait_bytes = _png(portrait)
        _metadata(portrait_bytes)
    if not snapshot.is_current():
        raise ValueError("buddy_conversion_stale")
    return BuddyConversion(
        snapshot,
        tuple(expressions),
        portrait_bytes,
        tuple(warnings),
        datetime.now(UTC).isoformat(timespec="seconds"),
    )


def publish_buddy_character(
    conversion: BuddyConversion,
    *,
    name: str,
    personality: str = "",
    first_message: str = "",
    db: Any,
    local_service: Any,
    profile_root: Path,
    authority_guard: Callable[[], bool],
) -> ActorPackActivationResult:
    """Use native Actor Pack activation for one atomic, independently owned character."""
    from tldw_chatbook.Actor_Packs.activation import ActorPackActivationService
    from tldw_chatbook.Actor_Packs.contracts import (
        canonical_json_bytes,
        canonicalize_actor_payload,
    )
    from tldw_chatbook.Actor_Packs.export import (
        ActorPackExportFile,
        ActorPackExportSection,
        ActorPackExportSnapshot,
        write_actor_pack_archive,
    )
    from tldw_chatbook.Actor_Packs.importer import (
        ActorPackImportError,
        ActorPackImportService,
    )
    from tldw_chatbook.Actor_Packs.persona_coordinator import (
        PersonaActorPackCoordinator,
    )
    from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
    from tldw_chatbook.Utils.private_paths import secure_private_directory

    from .artwork_attribution import encode_artwork_attribution
    from .visual_identity import (
        compute_pack_content_sha256,
        validate_visual_identity_manifest,
    )

    if (
        not isinstance(conversion, BuddyConversion)
        or type(name) is not str
        or not name.strip()
    ):
        raise ValueError("buddy_conversion_character_invalid")

    def require_current() -> None:
        try:
            current = authority_guard() is True and conversion.snapshot.is_current()
        except Exception:  # noqa: BLE001 - fail closed at the authority boundary
            current = False
        if not current:
            raise ActorPackImportError("actor_pack_import_review_stale")

    require_current()
    portable_uuid = str(uuid4())
    artwork = dict(conversion.snapshot.artwork)
    fields = {
        "name": name.strip(),
        "personality": personality,
        "first_message": first_message,
    }
    if artwork.get("creator"):
        fields["creator"] = artwork["creator"]
    payload = canonicalize_actor_payload("character", portable_uuid, fields)
    raw_assets, files, contexts = [], [], {}
    for index, expression in enumerate(
        sorted(conversion.expressions, key=lambda value: value.expression_key)
    ):
        metadata = _metadata(expression.data)
        if dict(expression.metadata) != metadata:
            raise ValueError("buddy_conversion_asset_changed")
        filename = f"shared-visual-identity/assets/asset-{index + 1:04d}." + (
            "webp" if metadata["is_animated"] else "png"
        )
        raw_assets.append(
            {
                **metadata,
                "expression_key": expression.expression_key,
                "original_label": expression.source_state,
                "display_label": expression.source_state,
                "storage_relpath": filename,
            }
        )
        files.append(ActorPackExportFile(filename, metadata["sha256"], expression.data))
        contexts[expression.expression_key] = (
            metadata["sha256"],
            {
                "tldw/artwork": {**artwork, "output_sha256": metadata["sha256"]},
                "tldw/buddy_conversion": {
                    "version": 1,
                    "source_sha256": conversion.snapshot.source_sha256,
                    "source_state": expression.source_state,
                    "source_asset_sha256": list(expression.source_asset_sha256),
                    "converted_at": conversion.converted_at,
                    "fallback": expression.fallback,
                    "output_sha256": metadata["sha256"],
                },
            },
        )
    # The compact native license token is independent of complete carried source terms.
    license_token = artwork.get("license") or "unspecified"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+-]*", license_token):
        license_token = "unspecified"
    manifest = {
        "schema_id": "tldw.visual_identity_pack/v1",
        "pack_id": f"buddy.character.{portable_uuid}",
        "title": name.strip(),
        "license": license_token,
        "default_expression_key": "neutral",
        "assets": raw_assets,
    }
    manifest["pack_content_sha256"] = compute_pack_content_sha256(manifest)
    validate_visual_identity_manifest(manifest)
    carrier = encode_artwork_attribution({"tldw/artwork": artwork}, contexts)
    export = ActorPackExportSnapshot(
        actor_kind="character",
        actor_revision=1,
        portable_uuid=portable_uuid,
        identity_version=1,
        portrait_name="portrait.png",
        portrait_sha256=hashlib.sha256(conversion.portrait).hexdigest(),
        local_actor_id="conversion",
        actor_payload=payload,
        portrait_bytes=conversion.portrait,
        sections=(
            ActorPackExportSection(
                "shared-visual-identity",
                "shared-visual-identity/manifest.json",
                (),
                license_token,
                None,
                canonical_json_bytes(manifest),
                tuple(files),
                artwork_attribution=carrier,
            ),
        ),
    )

    class GuardedImporter(ActorPackImportService):
        def revalidate_review(self, review, **kwargs):
            super().revalidate_review(review, **kwargs)
            require_current()  # Also called inside the existing outer SQLite transaction.

    class GuardedActivation(ActorPackActivationService):
        def _activate_shared_visual(self, *args, **kwargs):
            super()._activate_shared_visual(*args, **kwargs)
            require_current()  # Final check after writes, before the transaction commits.

    root = Path(profile_root) / "buddy-conversion"
    if not secure_private_directory(
        root, create=True, application_owned=True
    ).verified_private:
        raise ValueError("buddy_conversion_staging_unavailable")
    repository = ActorPackRepository(db)
    candidate = Path(tempfile.mkdtemp(prefix="convert-", dir=root))
    result = None
    try:
        archive_path = candidate / "character.tldw-actor-pack"
        with archive_path.open("w+b") as stream:
            write_actor_pack_archive(export, stream)
        importer = GuardedImporter(
            repository,
            staging_root=candidate / "staging",
            profile_root=Path(profile_root),
            local_service=local_service,
        )
        review = importer.inspect_archive(archive_path)
        service = GuardedActivation(
            db,
            local_service,
            repository,
            PersonaActorPackCoordinator(repository, local_service),
            importer,
        )
        result = service.activate(review, "create_new")
    finally:
        try:
            shutil.rmtree(candidate)
        except OSError:
            # Cleanup cannot turn a committed character into an apparent failure.
            if result is not None:
                result = replace(result, cleanup_pending=True)
            else:
                from tldw_chatbook.Actor_Packs.activation import (
                    ActorPackActivationError,
                )

                raise ActorPackActivationError(
                    "buddy_conversion_failed_cleanup_pending", cleanup_pending=True
                ) from None
    return result
