"""Immutable Persona Visual models, capabilities, and state selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


RESERVED_STATES = (
    "idle",
    "wake_armed",
    "listening",
    "thinking",
    "speaking",
    "tool_running",
    "approval_needed",
    "error",
    "offline",
)
REQUIRED_STATES = ("idle", "listening", "thinking", "speaking", "error")
ALLOWED_TRIGGER_SOURCES = (
    "live_state",
    "tool_category",
    "mcp_runtime",
    "tool_name",
)
ALLOWED_STATE_CATALOG_KINDS = (
    "tool_variant",
    "reaction",
    "live_variant",
    "mcp_runtime",
    "mood",
    "pack_private",
)
ALLOWED_ASSET_MIME_TYPES = (
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
)
ALLOWED_ASSET_ROLES = (
    "frame",
    "still_pose",
    "sprite_sheet",
    "preview",
    "generated_candidate",
)
ALLOWED_ASSET_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")

MAX_FRAMES_PER_ANIMATION = 240
MAX_CUSTOM_STATES = 256
MAX_TRIGGERS = 512
MAX_FALLBACK_DEPTH = 8
MIN_FRAME_DURATION_MS = 16
MAX_FRAME_DURATION_MS = 30_000
MIN_TRIGGER_DURATION_MS = 100
MAX_TRIGGER_DURATION_MS = 30_000
MAX_ASSET_COUNT = 256
MAX_ASSET_TOTAL_BYTES = 100 * 1024 * 1024
MAX_ASSET_DIMENSION = 4096

INVALID_MANIFEST_REASON = "persona_visual_manifest_invalid"
UNSUPPORTED_CAPABILITY_REASON = "persona_visual_capability_unsupported"


class PersonaVisualManifestError(ValueError):
    """A path-free, stable Persona Visual contract error."""

    __slots__ = ("category",)

    def __init__(self, category: str = INVALID_MANIFEST_REASON) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class PersonaVisualCapability:
    """Activation capability and exact raster metadata limits."""

    renderer_type: str | None
    manifest_version: int | None
    supported: bool
    activatable: bool
    reason: str | None
    allowed_asset_roles: tuple[str, ...] = ()
    allowed_mime_types: tuple[str, ...] = ()
    allowed_extensions: tuple[str, ...] = ()
    max_file_count: int | None = None
    max_total_bytes: int | None = None
    max_texture_width: int | None = None
    max_texture_height: int | None = None


@dataclass(frozen=True, slots=True)
class PersonaVisualRegion:
    """A rectangular frame region in asset pixels."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class PersonaVisualAlignment:
    """Normalized animation alignment."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class PersonaVisualFrame:
    """One immutable animation frame."""

    asset_id: str
    duration_ms: int | None = None
    region: PersonaVisualRegion | None = None


@dataclass(frozen=True, slots=True)
class PersonaVisualAnimation:
    """One normalized sprite-frame animation."""

    frames: tuple[PersonaVisualFrame, ...]
    frame_rate: float = 1
    loop: bool = True
    alignment: PersonaVisualAlignment | None = None
    preview_frame: int | None = None
    preview_asset_id: str | None = None


@dataclass(frozen=True, slots=True)
class PersonaVisualTrigger:
    """An authored operational-state trigger."""

    id: str
    source: str
    match: str
    state: str
    duration_ms: int
    priority: int


@dataclass(frozen=True, slots=True)
class PersonaVisualCatalogEntry:
    """Metadata declaring one safe custom state."""

    label: str
    kind: str
    description: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PersonaVisualStaticSelection:
    """The deterministic still frame for previews or reduced motion."""

    frame_index: int
    frame: PersonaVisualFrame
    reason: str


@dataclass(frozen=True, slots=True)
class PersonaVisualStateSelection:
    """A resolved runtime state and its immutable render selection."""

    requested_state: str
    resolved_state: str
    animation_id: str
    animation: PersonaVisualAnimation
    static: PersonaVisualStaticSelection
    animate: bool


@dataclass(frozen=True, slots=True)
class PersonaVisualManifest:
    """Validated and deeply immutable sprite-frame manifest metadata."""

    renderer_type: str
    manifest_version: int
    states: Mapping[str, str]
    animations: Mapping[str, PersonaVisualAnimation]
    fallbacks: Mapping[str, tuple[str, ...]]
    triggers: tuple[PersonaVisualTrigger, ...]
    state_catalog: Mapping[str, PersonaVisualCatalogEntry]
    resolved_required_states: Mapping[str, str]


_SPRITE_CAPABILITY = PersonaVisualCapability(
    renderer_type="sprite_frames",
    manifest_version=1,
    supported=True,
    activatable=True,
    reason=None,
    allowed_asset_roles=ALLOWED_ASSET_ROLES,
    allowed_mime_types=ALLOWED_ASSET_MIME_TYPES,
    allowed_extensions=ALLOWED_ASSET_EXTENSIONS,
    max_file_count=MAX_ASSET_COUNT,
    max_total_bytes=MAX_ASSET_TOTAL_BYTES,
    max_texture_width=MAX_ASSET_DIMENSION,
    max_texture_height=MAX_ASSET_DIMENSION,
)


def inspect_persona_visual_capability(
    renderer_type: object,
    manifest_version: object,
) -> PersonaVisualCapability:
    """Return stable local capability metadata without validating a manifest."""

    if renderer_type == "sprite_frames" and type(manifest_version) is int:
        if manifest_version == 1:
            return _SPRITE_CAPABILITY
        known_renderer: str | None = "sprite_frames"
    elif renderer_type == "live2d" and type(manifest_version) is int:
        known_renderer = "live2d"
    else:
        known_renderer = None
    return PersonaVisualCapability(
        renderer_type=known_renderer,
        manifest_version=manifest_version if type(manifest_version) is int else None,
        supported=False,
        activatable=False,
        reason=UNSUPPORTED_CAPABILITY_REASON,
    )


def resolve_manifest_state(
    manifest: PersonaVisualManifest,
    requested_state: str,
    *,
    reduced_motion: bool = False,
) -> PersonaVisualStateSelection | None:
    """Resolve manifest fallbacks, then idle, and select a deterministic still."""

    if not isinstance(manifest, PersonaVisualManifest):
        raise PersonaVisualManifestError()
    if not isinstance(requested_state, str) or type(reduced_motion) is not bool:
        raise PersonaVisualManifestError()

    memo: dict[str, str | None] = {}
    resolved = _resolve_state_name(
        requested_state,
        states=manifest.states,
        fallbacks=manifest.fallbacks,
        memo=memo,
    )
    if resolved is None and requested_state != "idle":
        resolved = _resolve_state_name(
            "idle",
            states=manifest.states,
            fallbacks=manifest.fallbacks,
            memo=memo,
        )
    if resolved is None:
        return None

    animation_id = manifest.states[resolved]
    animation = manifest.animations[animation_id]
    return PersonaVisualStateSelection(
        requested_state=requested_state,
        resolved_state=resolved,
        animation_id=animation_id,
        animation=animation,
        static=_select_static_frame(animation),
        animate=not reduced_motion and len(animation.frames) > 1,
    )


def _resolve_state_name(
    state: str,
    *,
    states: Mapping[str, str],
    fallbacks: Mapping[str, tuple[str, ...]],
    memo: dict[str, str | None],
    seen: frozenset[str] = frozenset(),
) -> str | None:
    if state in seen:
        return None
    if state in memo:
        return memo[state]
    if state in states:
        memo[state] = state
        return state
    next_seen = seen | {state}
    for candidate in fallbacks.get(state, ()):
        if resolved := _resolve_state_name(
            candidate,
            states=states,
            fallbacks=fallbacks,
            memo=memo,
            seen=next_seen,
        ):
            memo[state] = resolved
            return resolved
    memo[state] = None
    return None


def _select_static_frame(
    animation: PersonaVisualAnimation,
) -> PersonaVisualStaticSelection:
    if animation.preview_frame is not None:
        index, reason = animation.preview_frame, "preview_frame"
    elif animation.preview_asset_id is not None:
        index = next(
            index
            for index, frame in enumerate(animation.frames)
            if frame.asset_id == animation.preview_asset_id
        )
        reason = "preview_asset_id"
    else:
        index, reason = 0, "first_frame"
    return PersonaVisualStaticSelection(index, animation.frames[index], reason)


__all__ = [
    "ALLOWED_ASSET_EXTENSIONS",
    "ALLOWED_ASSET_MIME_TYPES",
    "ALLOWED_ASSET_ROLES",
    "ALLOWED_STATE_CATALOG_KINDS",
    "ALLOWED_TRIGGER_SOURCES",
    "MAX_ASSET_COUNT",
    "MAX_ASSET_DIMENSION",
    "MAX_ASSET_TOTAL_BYTES",
    "MAX_CUSTOM_STATES",
    "MAX_FALLBACK_DEPTH",
    "MAX_FRAME_DURATION_MS",
    "MAX_FRAMES_PER_ANIMATION",
    "MAX_TRIGGER_DURATION_MS",
    "MAX_TRIGGERS",
    "MIN_FRAME_DURATION_MS",
    "MIN_TRIGGER_DURATION_MS",
    "REQUIRED_STATES",
    "RESERVED_STATES",
    "PersonaVisualAlignment",
    "PersonaVisualAnimation",
    "PersonaVisualCapability",
    "PersonaVisualCatalogEntry",
    "PersonaVisualFrame",
    "PersonaVisualManifest",
    "PersonaVisualManifestError",
    "PersonaVisualRegion",
    "PersonaVisualStateSelection",
    "PersonaVisualStaticSelection",
    "PersonaVisualTrigger",
    "inspect_persona_visual_capability",
    "resolve_manifest_state",
]
