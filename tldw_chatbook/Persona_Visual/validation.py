"""Strict metadata-only validation for Persona Visual sprite manifests."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Mapping
from types import MappingProxyType
from typing import Any

from .contracts import (
    ALLOWED_STATE_CATALOG_KINDS,
    ALLOWED_TRIGGER_SOURCES,
    MAX_CUSTOM_STATES,
    MAX_FALLBACK_DEPTH,
    MAX_FRAME_DURATION_MS,
    MAX_FRAMES_PER_ANIMATION,
    MAX_TRIGGER_DURATION_MS,
    MAX_TRIGGERS,
    MIN_FRAME_DURATION_MS,
    MIN_TRIGGER_DURATION_MS,
    REQUIRED_STATES,
    RESERVED_STATES,
    UNSUPPORTED_CAPABILITY_REASON,
    PersonaVisualAlignment,
    PersonaVisualAnimation,
    PersonaVisualCatalogEntry,
    PersonaVisualFrame,
    PersonaVisualManifest,
    PersonaVisualManifestError,
    PersonaVisualRegion,
    PersonaVisualTrigger,
    inspect_persona_visual_capability,
)


_CUSTOM_STATE_PATTERN = re.compile(r"^[a-z][a-z0-9_.:-]{0,95}$")
_UNSAFE_STATE_PREFIXES = (
    "env:",
    "file:",
    "ftp:",
    "http:",
    "https:",
    "proc:",
    "ssh:",
)
_UNSAFE_STATE_MARKERS = (
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "bearer_token",
    "client_secret",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
)
_ROOT_FIELDS = {
    "renderer_type",
    "manifest_version",
    "states",
    "animations",
    "fallbacks",
    "authored_triggers",
    "state_catalog",
}
_ANIMATION_FIELDS = {
    "frames",
    "asset_ids",
    "frame_rate",
    "loop",
    "alignment",
    "preview_frame",
    "preview_asset_id",
}


def validate_persona_visual_manifest(
    payload: object,
    known_assets: Collection[str] | Mapping[str, tuple[int, int] | None],
    *,
    activate: bool = True,
) -> PersonaVisualManifest:
    """Validate the pinned sprite-frame manifest-v1 semantic subset.

    ``known_assets`` supplies identifiers and, optionally, dimensions. The validator
    never opens a file, path, or URI.
    """

    if type(activate) is not bool:
        raise PersonaVisualManifestError()
    document = _load_document(payload)
    renderer_type = document.get("renderer_type")
    manifest_version = document.get("manifest_version")
    if not isinstance(renderer_type, str) or type(manifest_version) is not int:
        raise PersonaVisualManifestError()
    if not inspect_persona_visual_capability(renderer_type, manifest_version).supported:
        raise PersonaVisualManifestError(UNSUPPORTED_CAPABILITY_REASON)

    _object(
        document, allowed=_ROOT_FIELDS, required={"renderer_type", "manifest_version"}
    )
    asset_ids, asset_dimensions = _known_assets(known_assets)
    catalog = _catalog(document.get("state_catalog", {}))
    allowed_states = frozenset(RESERVED_STATES) | catalog.keys()
    animations = _animations(
        document.get("animations", {}),
        asset_ids=asset_ids,
        asset_dimensions=asset_dimensions,
    )
    states = _states(
        document.get("states", {}),
        allowed_states=allowed_states,
        animations=animations,
    )
    fallbacks = _fallbacks(
        document.get("fallbacks", {}),
        allowed_states=allowed_states,
    )
    triggers = _triggers(
        document.get("authored_triggers", []),
        allowed_states=allowed_states,
    )
    resolved = {
        state: animation_id
        for state in REQUIRED_STATES
        if (animation_id := _resolve_animation(state, states, fallbacks))
    }
    if activate and len(resolved) != len(REQUIRED_STATES):
        raise PersonaVisualManifestError()

    return PersonaVisualManifest(
        renderer_type=renderer_type,
        manifest_version=manifest_version,
        states=_freeze(states),
        animations=_freeze(animations),
        fallbacks=_freeze(fallbacks),
        triggers=tuple(triggers),
        state_catalog=_freeze(catalog),
        resolved_required_states=_freeze(resolved),
    )


def _load_document(payload: object) -> dict[str, Any]:
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError:
            raise PersonaVisualManifestError() from None
    if isinstance(payload, str):
        try:
            payload = json.loads(
                payload,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (TypeError, ValueError, RecursionError):
            raise PersonaVisualManifestError() from None
    try:
        _json_value(payload)
    except RecursionError:
        raise PersonaVisualManifestError() from None
    if type(payload) is not dict:
        raise PersonaVisualManifestError()
    return payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError


def _json_value(value: object) -> None:
    if value is None or type(value) in {str, int, bool}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise PersonaVisualManifestError()
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise PersonaVisualManifestError()
            _json_value(item)
        return
    raise PersonaVisualManifestError()


def _object(
    value: object,
    *,
    allowed: set[str] | None = None,
    required: set[str] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise PersonaVisualManifestError()
    if (allowed is not None and not set(value).issubset(allowed)) or (
        required is not None and not required.issubset(value)
    ):
        raise PersonaVisualManifestError()
    return value


def _known_assets(
    known_assets: Collection[str] | Mapping[str, tuple[int, int] | None],
) -> tuple[frozenset[str], dict[str, tuple[int, int]]]:
    dimensions: dict[str, tuple[int, int]] = {}
    if isinstance(known_assets, Mapping):
        ids = set(known_assets)
        for asset_id, size in known_assets.items():
            if not isinstance(asset_id, str) or not asset_id:
                raise PersonaVisualManifestError()
            if size is None:
                continue
            if (
                not isinstance(size, (tuple, list))
                or len(size) != 2
                or type(size[0]) is not int
                or type(size[1]) is not int
                or size[0] <= 0
                or size[1] <= 0
            ):
                raise PersonaVisualManifestError()
            dimensions[asset_id] = (size[0], size[1])
    elif isinstance(known_assets, (set, frozenset, list, tuple)):
        if any(
            not isinstance(asset_id, str) or not asset_id for asset_id in known_assets
        ):
            raise PersonaVisualManifestError()
        ids = set(known_assets)
    else:
        raise PersonaVisualManifestError()
    return frozenset(ids), dimensions


def _catalog(value: object) -> dict[str, PersonaVisualCatalogEntry]:
    entries = _object(value)
    if len(entries) > MAX_CUSTOM_STATES:
        raise PersonaVisualManifestError()
    result: dict[str, PersonaVisualCatalogEntry] = {}
    for state_id, raw_entry in entries.items():
        _custom_state_id(state_id)
        entry = _object(
            raw_entry,
            allowed={"label", "kind", "description", "tags"},
            required={"label", "kind"},
        )
        label, kind = entry["label"], entry["kind"]
        if (
            not isinstance(label, str)
            or not label.strip()
            or len(label) > 80
            or _control(label)
            or kind not in ALLOWED_STATE_CATALOG_KINDS
        ):
            raise PersonaVisualManifestError()
        description = entry.get("description")
        if description is not None and (
            not isinstance(description, str)
            or len(description) > 280
            or _control(description)
        ):
            raise PersonaVisualManifestError()
        tags = entry.get("tags", [])
        if type(tags) is not list or len(tags) > 16:
            raise PersonaVisualManifestError()
        if any(
            not isinstance(tag, str)
            or not tag.strip()
            or len(tag) > 32
            or _control(tag)
            for tag in tags
        ):
            raise PersonaVisualManifestError()
        result[state_id] = PersonaVisualCatalogEntry(
            label=label,
            kind=kind,
            description=description,
            tags=tuple(tags),
        )
    return result


def _custom_state_id(state_id: object) -> None:
    if not isinstance(state_id, str) or not _CUSTOM_STATE_PATTERN.fullmatch(state_id):
        raise PersonaVisualManifestError()
    if state_id in RESERVED_STATES or state_id.startswith(_UNSAFE_STATE_PREFIXES):
        raise PersonaVisualManifestError()
    compact = re.sub(r"[._:-]+", "_", state_id)
    if any(marker in compact for marker in _UNSAFE_STATE_MARKERS):
        raise PersonaVisualManifestError()


def _control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _animations(
    value: object,
    *,
    asset_ids: frozenset[str],
    asset_dimensions: Mapping[str, tuple[int, int]],
) -> dict[str, PersonaVisualAnimation]:
    result: dict[str, PersonaVisualAnimation] = {}
    for animation_id, raw_animation in _object(value).items():
        if not isinstance(animation_id, str) or not animation_id:
            raise PersonaVisualManifestError()
        animation = _object(raw_animation, allowed=_ANIMATION_FIELDS)
        frames_value = animation.get("frames")
        if frames_value is None:
            legacy_ids = animation.get("asset_ids")
            if type(legacy_ids) is not list:
                raise PersonaVisualManifestError()
            frames_value = [{"asset_id": asset_id} for asset_id in legacy_ids]
        if (
            type(frames_value) is not list
            or not 1 <= len(frames_value) <= MAX_FRAMES_PER_ANIMATION
        ):
            raise PersonaVisualManifestError()
        frames = tuple(
            _frame(frame, asset_ids=asset_ids, dimensions=asset_dimensions)
            for frame in frames_value
        )
        frame_rate = animation.get("frame_rate", 1)
        loop = animation.get("loop", True)
        if (
            not _number(frame_rate)
            or not 1 <= frame_rate <= 60
            or type(loop) is not bool
        ):
            raise PersonaVisualManifestError()
        preview_frame = animation.get("preview_frame")
        if preview_frame is not None and (
            type(preview_frame) is not int or not 0 <= preview_frame < len(frames)
        ):
            raise PersonaVisualManifestError()
        preview_asset_id = animation.get("preview_asset_id")
        if preview_asset_id is not None and (
            not isinstance(preview_asset_id, str)
            or preview_asset_id not in {frame.asset_id for frame in frames}
        ):
            raise PersonaVisualManifestError()
        result[animation_id] = PersonaVisualAnimation(
            frames=frames,
            frame_rate=float(frame_rate),
            loop=loop,
            alignment=_alignment(animation.get("alignment")),
            preview_frame=preview_frame,
            preview_asset_id=preview_asset_id,
        )
    return result


def _alignment(value: object) -> PersonaVisualAlignment | None:
    if value is None:
        return None
    alignment = _object(value, allowed={"x", "y"}, required={"x", "y"})
    x, y = alignment["x"], alignment["y"]
    if not _number(x) or not _number(y) or not 0 <= x <= 1 or not 0 <= y <= 1:
        raise PersonaVisualManifestError()
    return PersonaVisualAlignment(float(x), float(y))


def _frame(
    value: object,
    *,
    asset_ids: frozenset[str],
    dimensions: Mapping[str, tuple[int, int]],
) -> PersonaVisualFrame:
    frame = _object(
        value,
        allowed={"asset_id", "duration_ms", "region"},
        required={"asset_id"},
    )
    asset_id = frame["asset_id"]
    if not isinstance(asset_id, str) or not asset_id or asset_id not in asset_ids:
        raise PersonaVisualManifestError()
    duration_ms = frame.get("duration_ms")
    if duration_ms is not None and (
        type(duration_ms) is not int
        or not MIN_FRAME_DURATION_MS <= duration_ms <= MAX_FRAME_DURATION_MS
    ):
        raise PersonaVisualManifestError()
    return PersonaVisualFrame(
        asset_id,
        duration_ms,
        _region(frame.get("region"), dimensions.get(asset_id)),
    )


def _region(
    value: object,
    dimensions: tuple[int, int] | None,
) -> PersonaVisualRegion | None:
    if value is None:
        return None
    region = _object(
        value,
        allowed={"x", "y", "width", "height"},
        required={"x", "y", "width", "height"},
    )
    x, y, width, height = (region[key] for key in ("x", "y", "width", "height"))
    if (
        any(type(part) is not int for part in (x, y, width, height))
        or x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or dimensions
        and (x + width > dimensions[0] or y + height > dimensions[1])
    ):
        raise PersonaVisualManifestError()
    return PersonaVisualRegion(x, y, width, height)


def _number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _states(
    value: object,
    *,
    allowed_states: Collection[str],
    animations: Mapping[str, PersonaVisualAnimation],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for state, raw_mapping in _object(value).items():
        if state not in allowed_states:
            raise PersonaVisualManifestError()
        mapping = _object(
            raw_mapping,
            allowed={"animation_id"},
            required={"animation_id"},
        )
        animation_id = mapping["animation_id"]
        if not isinstance(animation_id, str) or animation_id not in animations:
            raise PersonaVisualManifestError()
        result[state] = animation_id
    return result


def _fallbacks(
    value: object,
    *,
    allowed_states: Collection[str],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for state, chain in _object(value).items():
        if state not in allowed_states or type(chain) is not list:
            raise PersonaVisualManifestError()
        if any(
            not isinstance(candidate, str) or candidate not in allowed_states
            for candidate in chain
        ):
            raise PersonaVisualManifestError()
        result[state] = tuple(chain)

    visiting: set[str] = set()
    depths: dict[str, int] = {}

    def depth(state: str) -> int:
        if state in visiting:
            raise PersonaVisualManifestError()
        if state in depths:
            return depths[state]
        visiting.add(state)
        child_depth = max(
            (depth(candidate) for candidate in result.get(state, ())), default=0
        )
        visiting.remove(state)
        depths[state] = child_depth + 1
        return depths[state]

    for state in result:
        if depth(state) > MAX_FALLBACK_DEPTH:
            raise PersonaVisualManifestError()
    return result


def _triggers(
    value: object,
    *,
    allowed_states: Collection[str],
) -> list[PersonaVisualTrigger]:
    if type(value) is not list or len(value) > MAX_TRIGGERS:
        raise PersonaVisualManifestError()
    fields = {"id", "source", "match", "state", "duration_ms", "priority"}
    result: list[PersonaVisualTrigger] = []
    for raw_trigger in value:
        trigger = _object(raw_trigger, allowed=fields, required=fields)
        trigger_id, source, match = trigger["id"], trigger["source"], trigger["match"]
        state = trigger["state"]
        duration_ms, priority = trigger["duration_ms"], trigger["priority"]
        if (
            not isinstance(trigger_id, str)
            or not trigger_id.strip()
            or source not in ALLOWED_TRIGGER_SOURCES
            or not isinstance(match, str)
            or not match.strip()
            or not isinstance(state, str)
            or state not in allowed_states
            or type(duration_ms) is not int
            or not MIN_TRIGGER_DURATION_MS <= duration_ms <= MAX_TRIGGER_DURATION_MS
            or type(priority) is not int
            or not 0 <= priority <= 100
        ):
            raise PersonaVisualManifestError()
        result.append(
            PersonaVisualTrigger(
                trigger_id,
                source,
                match,
                state,
                duration_ms,
                priority,
            )
        )
    return result


def _resolve_animation(
    state: str,
    states: Mapping[str, str],
    fallbacks: Mapping[str, tuple[str, ...]],
    seen: frozenset[str] = frozenset(),
) -> str | None:
    if state in seen:
        return None
    if animation_id := states.get(state):
        return animation_id
    for candidate in fallbacks.get(state, ()):
        if animation_id := _resolve_animation(
            candidate,
            states,
            fallbacks,
            seen | {state},
        ):
            return animation_id
    return None


def _freeze(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


__all__ = ["validate_persona_visual_manifest"]
