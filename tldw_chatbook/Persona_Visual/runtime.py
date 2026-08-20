"""Path-free runtime resolution for immutable Persona Visual graphs."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from .assets import PersonaVisualAsset, PersonaVisualAssetMetadata
from .contracts import (
    ALLOWED_ASSET_MIME_TYPES,
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
)


STATE_FALLBACK_REASON = "persona_visual_state_fallback"
IDLE_UNAVAILABLE_REASON = "persona_visual_idle_unavailable"
UNAVAILABLE_REASON = "persona_visual_unavailable"
GRAPH_INVALID_REASON = "persona_visual_graph_invalid"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STATE = re.compile(r"[a-z][a-z0-9_.:-]{0,95}\Z")
_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_RUNTIME_CANDIDATES = 2 * (256 + 9)
_MAX_PORTRAIT_BYTES = 25 * 1024 * 1024


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
            graph,
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
            graph,
            requested_state,
            portrait,
            reduced_motion,
            GRAPH_INVALID_REASON,
            (),
        )

    attempted: list[PersonaVisualCacheAsset] = []
    for state in candidates:
        animation_id = manifest.states.get(state)
        if animation_id is None:
            continue
        animation = manifest.animations.get(animation_id)
        if animation is None:
            return _fallback_result(
                graph,
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
            )
        except Exception:
            return _fallback_result(
                graph,
                requested_state,
                portrait,
                reduced_motion,
                GRAPH_INVALID_REASON,
                tuple(attempted),
            )
        if loaded is None:
            continue
        frames, static_reason = loaded
        reason = None if state == requested_state else STATE_FALLBACK_REASON
        return PersonaVisualResolution(
            source="persona_visual",
            reason=reason,
            requested_state=requested_state,
            resolved_state=state,
            animation_id=animation_id,
            frames=frames,
            frame_rate=animation.frame_rate,
            loop=animation.loop,
            alignment=animation.alignment,
            animate=not reduced_motion
            and (
                len(animation.frames) > 1
                or any(
                    (assets[frame.asset_id].frame_count or 1) > 1
                    for frame in animation.frames
                )
            ),
            static_reason=static_reason,
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
        graph,
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
    records: dict[str, PersonaVisualAssetRecord] = {}
    record_ids: set[int] = set()
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
        ):
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
) -> tuple[tuple[PersonaVisualResolvedFrame, ...], str | None] | None:
    if reduced_motion:
        index, static_reason = _static_index(animation)
        selected = ((index, animation.frames[index]),)
    else:
        static_reason = None
        selected = tuple(enumerate(animation.frames))
    loaded_frames: list[PersonaVisualResolvedFrame] = []
    for manifest_index, frame in selected:
        record = assets.get(frame.asset_id)
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
        try:
            loaded = asset_loader(identity, record, 0)
            _attest_loaded_asset(loaded, record)
        except Exception:
            return None
        loaded_frames.append(
            PersonaVisualResolvedFrame(
                asset_id=record.id,
                asset_key=record.asset_key,
                sha256=record.sha256,
                data=loaded.data,
                duration_ms=frame.duration_ms,
                region=frame.region,
                manifest_frame_index=manifest_index,
            )
        )
    return tuple(loaded_frames), static_reason


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
    if type(loaded) is not PersonaVisualAsset or loaded.selected_frame != 0:
        raise ValueError
    expected = PersonaVisualAssetMetadata(
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
    if (
        loaded.metadata != expected
        or type(loaded.data) is not bytes
        or len(loaded.data) != record.byte_count
        or hashlib.sha256(loaded.data).hexdigest() != record.sha256
    ):
        raise ValueError


def _fallback_result(
    graph: PersonaVisualGraph | None,
    requested_state: str,
    portrait: PersonaVisualPortrait | None,
    reduced_motion: bool,
    reason: str,
    assets: tuple[PersonaVisualCacheAsset, ...],
) -> PersonaVisualResolution:
    identity = _safe_identity(graph)
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
        if (
            type(portrait) is not PersonaVisualPortrait
            or type(portrait.portrait_id) is not str
            or _OPAQUE_ID.fullmatch(portrait.portrait_id) is None
            or type(portrait.revision) is not int
            or portrait.revision < 0
            or type(portrait.mime_type) is not str
            or portrait.mime_type not in ALLOWED_ASSET_MIME_TYPES
            or type(portrait.sha256) is not str
            or _SHA256.fullmatch(portrait.sha256) is None
            or type(portrait.data) is not bytes
            or not portrait.data
            or len(portrait.data) > _MAX_PORTRAIT_BYTES
            or hashlib.sha256(portrait.data).hexdigest() != portrait.sha256
        ):
            return None
        portrait.portrait_id.encode("utf-8")
        portrait.mime_type.encode("utf-8")
        return portrait
    except (UnicodeError, ValueError):
        return None


def _safe_identity(graph: PersonaVisualGraph | None) -> PersonaVisualIdentity | None:
    if (
        type(graph) is PersonaVisualGraph
        and type(graph.identity) is PersonaVisualIdentity
    ):
        return graph.identity
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
    "STATE_FALLBACK_REASON",
    "UNAVAILABLE_REASON",
    "PersonaVisualAssetLoader",
    "PersonaVisualCacheAsset",
    "PersonaVisualCacheIdentity",
    "PersonaVisualPortrait",
    "PersonaVisualResolution",
    "PersonaVisualResolvedFrame",
    "resolve_persona_visual",
]
