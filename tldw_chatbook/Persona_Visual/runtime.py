"""Path-free runtime resolution for immutable Persona Visual graphs."""

from __future__ import annotations

import hashlib
import re
from io import BytesIO
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from os import PathLike
from typing import Protocol

from PIL import Image

from .assets import (
    PersonaVisualAsset,
    PersonaVisualAssetError,
    PersonaVisualAssetMetadata,
    load_persona_visual_asset,
)
from .contracts import (
    ALLOWED_ASSET_MIME_TYPES,
    MAX_ASSET_COUNT,
    MAX_ASSET_DIMENSION,
    MAX_ASSET_TOTAL_BYTES,
    MAX_FALLBACK_DEPTH,
    PersonaVisualAlignment,
    PersonaVisualAnimation,
    PersonaVisualFrame,
    PersonaVisualManifest,
    PersonaVisualRegion,
)
from .repository import (
    PersonaVisualAssetRecord,
    PersonaVisualGraph,
    PersonaVisualIdentity,
    PersonaVisualRepository,
)


STATE_FALLBACK_REASON = "persona_visual_state_fallback"
IDLE_UNAVAILABLE_REASON = "persona_visual_idle_unavailable"
UNAVAILABLE_REASON = "persona_visual_unavailable"
GRAPH_INVALID_REASON = "persona_visual_graph_invalid"
RUNTIME_FAILED_REASON = "persona_visual_runtime_failed"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STATE = re.compile(r"[a-z][a-z0-9_.:-]{0,95}\Z")
_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_RUNTIME_CANDIDATES = 2 * (256 + 9)
_MAX_PORTRAIT_BYTES = 25 * 1024 * 1024
_RASTER_FORMATS = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
    "image/gif": "GIF",
}
_ASSET_MISS = object()


class PersonaVisualAssetLoader(Protocol):
    """Load one asset through an injected private-storage boundary."""

    def __call__(
        self,
        identity: PersonaVisualIdentity,
        asset: PersonaVisualAssetRecord,
        selected_frame: int,
    ) -> PersonaVisualAsset: ...


@dataclass(frozen=True, slots=True)
class PersonaVisualPortrait:
    """A caller-supplied, already-validated path-free Persona portrait."""

    portrait_id: str
    revision: int
    mime_type: str
    sha256: str
    data: bytes = field(repr=False)
    selected_frame: int = 0


@dataclass(frozen=True, slots=True)
class PersonaVisualCacheAsset:
    """One immutable asset reference contributing to a cache identity."""

    asset_id: int
    asset_key: str
    sha256: str
    manifest_frame_index: int
    selected_frame: int


@dataclass(frozen=True, slots=True)
class PersonaVisualCacheIdentity:
    """Complete path-free identity for a resolved render or stable fallback."""

    graph: PersonaVisualIdentity | None
    requested_state: str
    resolved_state: str | None
    animation_id: str | None
    reduced_motion: bool
    assets: tuple[PersonaVisualCacheAsset, ...]
    portrait_id: str | None = None
    portrait_revision: int | None = None
    portrait_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class PersonaVisualResolvedFrame:
    """One path-free, validated manifest frame ready for a renderer."""

    asset_id: int
    asset_key: str
    sha256: str
    data: bytes = field(repr=False)
    duration_ms: int | None
    region: PersonaVisualRegion | None
    manifest_frame_index: int
    selected_frame: int


@dataclass(frozen=True, slots=True)
class _LoadedAnimation:
    frames: tuple[PersonaVisualResolvedFrame, ...]
    static_reason: str | None
    frame_rate: float
    loop: bool
    alignment: PersonaVisualAlignment | None
    animate: bool


@dataclass(frozen=True, slots=True)
class PersonaVisualResolution:
    """A complete animation, portrait fallback, or stable unavailable result."""

    source: str
    reason: str | None
    requested_state: str
    resolved_state: str | None
    animation_id: str | None
    frames: tuple[PersonaVisualResolvedFrame, ...]
    frame_rate: float | None
    loop: bool
    alignment: PersonaVisualAlignment | None
    animate: bool
    static_reason: str | None
    portrait: PersonaVisualPortrait | None
    cache_identity: PersonaVisualCacheIdentity


def resolve_active_persona_visual(
    repository: PersonaVisualRepository,
    persona_id: str,
    profile_root: PathLike[str] | str,
    requested_state: str,
    *,
    portrait: PersonaVisualPortrait | None = None,
    reduced_motion: bool = False,
) -> PersonaVisualResolution:
    """Resolve one persisted active graph through its private storage bridge."""

    try:
        graph = repository.get_active_persona_pack(persona_id)
    except Exception:
        public_state = (
            requested_state
            if type(requested_state) is str
            and _STATE.fullmatch(requested_state) is not None
            else "invalid"
        )
        return _fallback_result(
            None,
            public_state,
            portrait,
            reduced_motion if type(reduced_motion) is bool else False,
            RUNTIME_FAILED_REASON,
            (),
        )

    def load(
        identity: PersonaVisualIdentity,
        asset: PersonaVisualAssetRecord,
        selected_frame: int,
    ) -> PersonaVisualAsset:
        storage_key = repository._get_active_asset_storage_key(identity, asset)
        return load_persona_visual_asset(
            profile_root,
            storage_key=storage_key,
            metadata=_asset_metadata(asset),
            selected_frame=selected_frame,
        )

    return resolve_persona_visual(
        graph,
        requested_state,
        asset_loader=load,
        portrait=portrait,
        reduced_motion=reduced_motion,
    )


def resolve_persona_visual(
    graph: PersonaVisualGraph | None,
    requested_state: str,
    *,
    asset_loader: PersonaVisualAssetLoader,
    portrait: PersonaVisualPortrait | None = None,
    reduced_motion: bool = False,
) -> PersonaVisualResolution:
    """Resolve one active graph without accepting or returning private storage data."""

    valid_requested_state = (
        type(requested_state) is str and _STATE.fullmatch(requested_state) is not None
    )
    public_requested_state = requested_state if valid_requested_state else "invalid"
    if graph is None:
        return _fallback_result(
            None,
            public_requested_state,
            portrait,
            reduced_motion if type(reduced_motion) is bool else False,
            IDLE_UNAVAILABLE_REASON,
            (),
        )
    if not valid_requested_state or type(reduced_motion) is not bool:
        return _fallback_result(
            None,
            public_requested_state,
            portrait,
            reduced_motion if type(reduced_motion) is bool else False,
            GRAPH_INVALID_REASON,
            (),
        )
    try:
        identity, manifest, assets = _validated_graph(graph)
        candidates = _candidate_states(manifest, requested_state)
    except Exception:
        return _fallback_result(
            None,
            requested_state,
            portrait,
            reduced_motion,
            GRAPH_INVALID_REASON,
            (),
        )

    attempted: list[PersonaVisualCacheAsset] = []
    memo: dict[tuple[PersonaVisualIdentity, PersonaVisualAssetRecord, int], object] = {}
    for state in candidates:
        animation_id = manifest.states.get(state)
        if animation_id is None:
            continue
        animation = manifest.animations.get(animation_id)
        if animation is None:
            return _fallback_result(
                None,
                requested_state,
                portrait,
                reduced_motion,
                GRAPH_INVALID_REASON,
                tuple(attempted),
            )
        try:
            loaded = _load_animation(
                identity,
                animation,
                assets,
                asset_loader,
                reduced_motion,
                attempted,
                memo,
            )
        except Exception:
            return _fallback_result(
                identity,
                requested_state,
                portrait,
                reduced_motion,
                RUNTIME_FAILED_REASON,
                tuple(attempted),
            )
        if loaded is None:
            continue
        reason = None if state == requested_state else STATE_FALLBACK_REASON
        return PersonaVisualResolution(
            source="persona_visual",
            reason=reason,
            requested_state=requested_state,
            resolved_state=state,
            animation_id=animation_id,
            frames=loaded.frames,
            frame_rate=loaded.frame_rate,
            loop=loaded.loop,
            alignment=loaded.alignment,
            animate=loaded.animate,
            static_reason=loaded.static_reason,
            portrait=None,
            cache_identity=_cache_identity(
                identity,
                requested_state,
                state,
                animation_id,
                reduced_motion,
                tuple(attempted),
                None,
            ),
        )

    return _fallback_result(
        identity,
        requested_state,
        portrait,
        reduced_motion,
        IDLE_UNAVAILABLE_REASON,
        tuple(attempted),
    )


def _validated_graph(
    graph: PersonaVisualGraph | None,
) -> tuple[
    PersonaVisualIdentity,
    PersonaVisualManifest,
    Mapping[str, PersonaVisualAssetRecord],
]:
    if type(graph) is not PersonaVisualGraph:
        raise ValueError
    identity = graph.identity
    if (
        type(identity) is not PersonaVisualIdentity
        or graph.pack.status != "active"
        or graph.binding.status != "active"
        or graph.version.renderer_type != "sprite_frames"
        or graph.version.manifest_version != 1
        or graph.pack.id != identity.pack_id
        or graph.pack.revision != identity.pack_revision
        or graph.version.id != identity.pack_version_id
        or graph.version.pack_id != identity.pack_id
        or graph.version.version_number != identity.version_number
        or graph.version.manifest_sha256 != identity.manifest_sha256
        or graph.binding.id != identity.binding_id
        or graph.binding.revision != identity.binding_version
        or graph.binding.persona_id != identity.persona_id
        or graph.binding.persona_revision != identity.persona_revision
        or graph.binding.pack_id != identity.pack_id
        or graph.binding.active_version_id != identity.pack_version_id
        or _SHA256.fullmatch(identity.manifest_sha256) is None
        or type(graph.version.manifest) is not PersonaVisualManifest
    ):
        raise ValueError
    identity = replace(identity)
    records: dict[str, PersonaVisualAssetRecord] = {}
    record_ids: set[int] = set()
    total_bytes = 0
    if type(graph.assets) is not tuple or len(graph.assets) > MAX_ASSET_COUNT:
        raise ValueError
    for record in graph.assets:
        if (
            type(record) is not PersonaVisualAssetRecord
            or type(record.id) is not int
            or record.id <= 0
            or record.id in record_ids
            or record.asset_key in records
            or record.pack_id != identity.pack_id
            or record.pack_version_id != identity.pack_version_id
            or _SHA256.fullmatch(record.sha256) is None
            or type(record.byte_count) is not int
            or record.byte_count <= 0
        ):
            raise ValueError
        record = replace(record)
        total_bytes += record.byte_count
        if total_bytes > MAX_ASSET_TOTAL_BYTES:
            raise ValueError
        record_ids.add(record.id)
        records[record.asset_key] = record
    manifest = graph.version.manifest
    for animation in manifest.animations.values():
        if type(animation) is not PersonaVisualAnimation:
            raise ValueError
        for frame in animation.frames:
            if type(frame) is not PersonaVisualFrame:
                raise ValueError
    _reject_fallback_cycles(manifest)
    return identity, manifest, records


def _reject_fallback_cycles(manifest: PersonaVisualManifest) -> None:
    done: set[str] = set()

    def visit(state: str, active: frozenset[str] = frozenset()) -> None:
        if state in active or len(active) >= MAX_FALLBACK_DEPTH:
            raise ValueError
        if state in done:
            return
        for child in manifest.fallbacks.get(state, ()):
            visit(child, active | {state})
        done.add(state)

    for state in manifest.fallbacks:
        visit(state)


def _candidate_states(
    manifest: PersonaVisualManifest,
    requested_state: str,
) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for root in (requested_state, "idle"):
        stack = [root]
        while stack:
            state = stack.pop()
            if state in seen:
                continue
            seen.add(state)
            ordered.append(state)
            if len(ordered) > _MAX_RUNTIME_CANDIDATES:
                raise ValueError
            children = manifest.fallbacks.get(state, ())
            stack.extend(reversed(children))
    return tuple(ordered)


def _load_animation(
    identity: PersonaVisualIdentity,
    animation: PersonaVisualAnimation,
    assets: Mapping[str, PersonaVisualAssetRecord],
    asset_loader: PersonaVisualAssetLoader,
    reduced_motion: bool,
    attempted: list[PersonaVisualCacheAsset],
    memo: dict[tuple[PersonaVisualIdentity, PersonaVisualAssetRecord, int], object],
) -> _LoadedAnimation | None:
    alignment = _snapshot_alignment(animation.alignment)
    frame_rate = animation.frame_rate
    loop = animation.loop
    frame_snapshots = tuple(
        (index, frame.asset_id, frame.duration_ms, _snapshot_region(frame.region))
        for index, frame in enumerate(animation.frames)
    )
    if reduced_motion:
        index, static_reason = _static_index(animation)
        selected = (frame_snapshots[index],)
    else:
        static_reason = None
        selected = frame_snapshots
    animate = not reduced_motion and (
        len(frame_snapshots) > 1
        or any(
            (record.frame_count or 1) > 1
            for _, asset_id, _, _ in frame_snapshots
            if (record := assets.get(asset_id)) is not None
        )
    )
    loaded_frames: list[PersonaVisualResolvedFrame] = []
    for manifest_index, asset_id, duration_ms, region in selected:
        record = assets.get(asset_id)
        if record is None:
            return None
        cache_asset = PersonaVisualCacheAsset(
            asset_id=record.id,
            asset_key=record.asset_key,
            sha256=record.sha256,
            manifest_frame_index=manifest_index,
            selected_frame=0,
        )
        attempted.append(cache_asset)
        memo_key = (identity, record, 0)
        loaded = memo.get(memo_key)
        if loaded is _ASSET_MISS:
            return None
        if loaded is None:
            try:
                loaded = asset_loader(replace(identity), replace(record), 0)
            except PersonaVisualAssetError:
                memo[memo_key] = _ASSET_MISS
                return None
            _attest_loaded_asset(loaded, record)
            memo[memo_key] = loaded
        if type(loaded) is not PersonaVisualAsset:
            raise ValueError
        loaded_frames.append(
            PersonaVisualResolvedFrame(
                asset_id=record.id,
                asset_key=record.asset_key,
                sha256=record.sha256,
                data=loaded.data,
                duration_ms=duration_ms,
                region=region,
                manifest_frame_index=manifest_index,
                selected_frame=loaded.selected_frame,
            )
        )
    return _LoadedAnimation(
        tuple(loaded_frames), static_reason, frame_rate, loop, alignment, animate
    )


def _snapshot_alignment(
    alignment: PersonaVisualAlignment | None,
) -> PersonaVisualAlignment | None:
    if alignment is None:
        return None
    if type(alignment) is not PersonaVisualAlignment:
        raise ValueError
    return replace(alignment)


def _snapshot_region(region: PersonaVisualRegion | None) -> PersonaVisualRegion | None:
    if region is None:
        return None
    if type(region) is not PersonaVisualRegion:
        raise ValueError
    return replace(region)


def _static_index(animation: PersonaVisualAnimation) -> tuple[int, str]:
    if animation.preview_frame is not None:
        return animation.preview_frame, "preview_frame"
    if animation.preview_asset_id is not None:
        for index, frame in enumerate(animation.frames):
            if frame.asset_id == animation.preview_asset_id:
                return index, "preview_asset_id"
        raise ValueError
    return 0, "first_frame"


def _attest_loaded_asset(
    loaded: object,
    record: PersonaVisualAssetRecord,
) -> None:
    if (
        type(loaded) is not PersonaVisualAsset
        or type(loaded.selected_frame) is not int
        or loaded.selected_frame != 0
    ):
        raise ValueError
    expected = _asset_metadata(record)
    if (
        loaded.metadata != expected
        or type(loaded.data) is not bytes
        or len(loaded.data) != record.byte_count
        or hashlib.sha256(loaded.data).hexdigest() != record.sha256
    ):
        raise ValueError


def _asset_metadata(record: PersonaVisualAssetRecord) -> PersonaVisualAssetMetadata:
    return PersonaVisualAssetMetadata(
        asset_key=record.asset_key,
        role=record.role,
        mime_type=record.mime_type,
        byte_count=record.byte_count,
        sha256=record.sha256,
        width=record.width,
        height=record.height,
        frame_count=record.frame_count,
        duration_ms=record.duration_ms,
    )


def _fallback_result(
    identity: PersonaVisualIdentity | None,
    requested_state: str,
    portrait: PersonaVisualPortrait | None,
    reduced_motion: bool,
    reason: str,
    assets: tuple[PersonaVisualCacheAsset, ...],
) -> PersonaVisualResolution:
    valid_portrait = _validated_portrait(portrait)
    source = "persona_portrait" if valid_portrait is not None else "unavailable"
    if source == "unavailable" and reason == IDLE_UNAVAILABLE_REASON:
        reason = UNAVAILABLE_REASON
    return PersonaVisualResolution(
        source=source,
        reason=reason,
        requested_state=requested_state,
        resolved_state=None,
        animation_id=None,
        frames=(),
        frame_rate=None,
        loop=False,
        alignment=None,
        animate=False,
        static_reason=None,
        portrait=valid_portrait,
        cache_identity=_cache_identity(
            identity,
            requested_state,
            None,
            None,
            reduced_motion,
            assets,
            valid_portrait,
        ),
    )


def _validated_portrait(
    portrait: PersonaVisualPortrait | None,
) -> PersonaVisualPortrait | None:
    if portrait is None:
        return None
    try:
        if type(portrait) is not PersonaVisualPortrait:
            return None
        snapshot = replace(portrait)
        if (
            type(snapshot.portrait_id) is not str
            or _OPAQUE_ID.fullmatch(snapshot.portrait_id) is None
            or type(snapshot.revision) is not int
            or snapshot.revision < 0
            or type(snapshot.mime_type) is not str
            or snapshot.mime_type not in ALLOWED_ASSET_MIME_TYPES
            or type(snapshot.sha256) is not str
            or _SHA256.fullmatch(snapshot.sha256) is None
            or type(snapshot.data) is not bytes
            or not snapshot.data
            or len(snapshot.data) > _MAX_PORTRAIT_BYTES
            or hashlib.sha256(snapshot.data).hexdigest() != snapshot.sha256
            or type(snapshot.selected_frame) is not int
            or snapshot.selected_frame != 0
        ):
            return None
        snapshot.portrait_id.encode("utf-8")
        with Image.open(BytesIO(snapshot.data)) as image:
            if (
                image.format != _RASTER_FORMATS[snapshot.mime_type]
                or image.width < 1
                or image.height < 1
                or image.width > MAX_ASSET_DIMENSION
                or image.height > MAX_ASSET_DIMENSION
            ):
                return None
            image.seek(0)
            image.load()
        return snapshot
    except Exception:
        return None


def _cache_identity(
    identity: PersonaVisualIdentity | None,
    requested_state: str,
    resolved_state: str | None,
    animation_id: str | None,
    reduced_motion: bool,
    assets: tuple[PersonaVisualCacheAsset, ...],
    portrait: PersonaVisualPortrait | None,
) -> PersonaVisualCacheIdentity:
    return PersonaVisualCacheIdentity(
        graph=identity,
        requested_state=requested_state,
        resolved_state=resolved_state,
        animation_id=animation_id,
        reduced_motion=reduced_motion,
        assets=assets,
        portrait_id=portrait.portrait_id if portrait else None,
        portrait_revision=portrait.revision if portrait else None,
        portrait_sha256=portrait.sha256 if portrait else None,
    )


__all__ = [
    "GRAPH_INVALID_REASON",
    "IDLE_UNAVAILABLE_REASON",
    "RUNTIME_FAILED_REASON",
    "STATE_FALLBACK_REASON",
    "UNAVAILABLE_REASON",
    "PersonaVisualAssetLoader",
    "PersonaVisualCacheAsset",
    "PersonaVisualCacheIdentity",
    "PersonaVisualPortrait",
    "PersonaVisualResolution",
    "PersonaVisualResolvedFrame",
    "resolve_active_persona_visual",
    "resolve_persona_visual",
]
