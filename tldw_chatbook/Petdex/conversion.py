"""Explicit Petdex atlas interpretation into native sprite-sheet frame regions.

Classic layout facts are pinned to crafter-station/petdex commit 5d1844be,
src/lib/pet-states.ts and sprite-atlas.ts. No upstream code or art is bundled.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from PIL import Image

from tldw_chatbook.Persona_Visual.assets import PersonaVisualAssetMetadata
from tldw_chatbook.Persona_Visual.contracts import (
    MAX_FRAME_DURATION_MS,
    MIN_FRAME_DURATION_MS,
)
from tldw_chatbook.Persona_Visual.snapshot import BuddyAssetSnapshot, BuddySnapshot
from tldw_chatbook.Persona_Visual.validation import validate_persona_visual_manifest

from .sources import PetdexSource


@dataclass(frozen=True, slots=True)
class PetdexState:
    name: str
    row: int
    frames: int
    duration_ms: int
    loop: bool = True


@dataclass(frozen=True, slots=True)
class PetdexInspection:
    version: int
    rows: int
    cell_width: int
    cell_height: int
    states: tuple[PetdexState, ...]
    mapping_source: str
    warnings: tuple[str, ...]


CLASSIC_STATES = tuple(
    PetdexState(name, row, count, duration)
    for row, (name, count, duration) in enumerate(
        (
            ("idle", 6, 1100),
            ("running-right", 8, 1060),
            ("running-left", 8, 1060),
            ("waving", 4, 700),
            ("jumping", 5, 840),
            ("failed", 8, 1220),
            ("waiting", 6, 1010),
            ("running", 6, 820),
            ("review", 6, 1030),
        )
    )
)
_REQUIRED = ("idle", "thinking", "error", "listening", "speaking")
_DEFAULT_SOURCES = {
    "idle": "idle",
    "thinking": "review",
    "error": "failed",
    "listening": "waiting",
    "speaking": "speaking",
}
_STATE_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")


def _validated_states(
    states: Sequence[PetdexState], rows: int
) -> tuple[PetdexState, ...]:
    if not isinstance(states, (tuple, list)) or not 1 <= len(states) <= rows:
        raise ValueError("petdex_states_invalid")
    names, occupied = set(), set()
    result = []
    for state in states:
        if (
            not isinstance(state, PetdexState)
            or type(state.name) is not str
            or not _STATE_NAME.fullmatch(state.name)
        ):
            raise ValueError("petdex_states_invalid")
        if (
            type(state.row) is not int
            or not 0 <= state.row < rows
            or type(state.frames) is not int
            or not 1 <= state.frames <= 8
            or type(state.duration_ms) is not int
            or type(state.loop) is not bool
        ):
            raise ValueError("petdex_states_invalid")
        if state.name in names or state.row in occupied:
            raise ValueError("petdex_states_ambiguous")
        duration, remainder = divmod(state.duration_ms, state.frames)
        if (
            duration < MIN_FRAME_DURATION_MS
            or duration + bool(remainder) > MAX_FRAME_DURATION_MS
        ):
            raise ValueError("petdex_timing_invalid")
        names.add(state.name)
        occupied.add(state.row)
        result.append(state)
    return tuple(result)


def _declared_states(value, rows: int) -> tuple[PetdexState, ...]:
    if type(value) is not list:
        raise ValueError("petdex_states_invalid")
    result = []
    for raw in value:
        if (
            type(raw) is not dict
            or not {"name", "row", "frames", "duration_ms"} <= set(raw)
            or not set(raw) <= {"name", "row", "frames", "duration_ms", "loop"}
        ):
            raise ValueError("petdex_states_invalid")
        result.append(PetdexState(**raw))
    return _validated_states(result, rows)


def inspect_petdex(source: PetdexSource) -> PetdexInspection:
    """Recognize exact cell geometry; never infer eleven-row state semantics."""
    if not isinstance(source, PetdexSource) or not source.is_current():
        raise ValueError("petdex_source_stale")
    metadata = json.loads(source.metadata_json)
    version = metadata["spriteVersionNumber"]
    rows = {1: 9, 2: 11}.get(version)
    if rows is None:
        raise ValueError("petdex_version_unsupported")
    with Image.open(io.BytesIO(source.image_bytes)) as image:
        width, height = image.size
    if width % 8 or height % rows or width * (rows * 208) != height * (8 * 192):
        raise ValueError("petdex_atlas_geometry_invalid")
    if "states" in metadata:
        states = _declared_states(metadata["states"], rows)
        origin = "declared"
    elif version == 1:
        states = CLASSIC_STATES
        origin = "pinned-classic"
    else:
        states = ()
        origin = "manual-required"
    warnings = []
    if not states:
        warnings.append(
            "This eleven-row atlas needs an explicit row, frame-count and loop-duration map before import."
        )
    if states and "speaking" not in {state.name for state in states}:
        warnings.append(
            "Speaking uses the idle fallback unless you choose another source state."
        )
    return PetdexInspection(
        version, rows, width // 8, height // rows, states, origin, tuple(warnings)
    )


def build_petdex_archive(
    source: PetdexSource,
    *,
    states: Sequence[PetdexState] | None = None,
    mappings: Mapping[str, str | None] | None = None,
) -> bytes:
    """Build an unpublished native archive from explicit, validated atlas semantics."""
    from tldw_chatbook.Persona_Visual.export import build_native_buddy_archive

    inspected = inspect_petdex(source)
    chosen = (
        inspected.states
        if states is None
        else _validated_states(states, inspected.rows)
    )
    if not chosen:
        raise ValueError("petdex_mapping_required")
    names = {state.name for state in chosen}
    selected = {
        key: value if value in names else None
        for key, value in _DEFAULT_SOURCES.items()
    }
    if mappings is not None:
        if not isinstance(mappings, Mapping) or set(mappings) != set(_REQUIRED):
            raise ValueError("petdex_mapping_invalid")
        selected = dict(mappings)
    if any(value is not None and type(value) is not str for value in selected.values()):
        raise ValueError("petdex_mapping_invalid")
    if selected["idle"] is None or any(
        value is not None and value not in names for value in selected.values()
    ):
        raise ValueError("petdex_mapping_invalid")
    animations = {}
    for state in chosen:
        duration, remainder = divmod(state.duration_ms, state.frames)
        animations[state.name] = {
            "loop": state.loop,
            "frames": [
                {
                    "asset_id": "sheet",
                    "duration_ms": duration + int(column < remainder),
                    "region": {
                        "x": column * inspected.cell_width,
                        "y": state.row * inspected.cell_height,
                        "width": inspected.cell_width,
                        "height": inspected.cell_height,
                    },
                }
                for column in range(state.frames)
            ],
        }
    manifest = {
        "renderer_type": "sprite_frames",
        "manifest_version": 1,
        "animations": animations,
        "states": {
            **{
                key: {"animation_id": value}
                for key, value in selected.items()
                if value is not None
            },
            **{
                f"custom:{state.name}": {"animation_id": state.name} for state in chosen
            },
        },
        "fallbacks": {
            key: ["idle"] for key, value in selected.items() if value is None
        },
        "state_catalog": {
            f"custom:{state.name}": {"label": state.name, "kind": "reaction"}
            for state in chosen
        },
        "authored_triggers": [],
    }
    width, height = inspected.cell_width * 8, inspected.cell_height * inspected.rows
    validate_persona_visual_manifest(manifest, {"sheet": (width, height)})
    with Image.open(io.BytesIO(source.image_bytes)) as image:
        mime = {"PNG": "image/png", "WEBP": "image/webp"}[image.format]
    asset = BuddyAssetSnapshot(
        PersonaVisualAssetMetadata(
            "sheet",
            "sprite_sheet",
            mime,
            len(source.image_bytes),
            hashlib.sha256(source.image_bytes).hexdigest(),
            width,
            height,
            1,
            None,
        ),
        source.image_bytes,
    )
    snapshot = BuddySnapshot(
        title=source.title,
        manifest_json=json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        assets=(asset,),
        artwork=dict(source.artwork),
        source_sha256=source.source_sha256,
        _guard=source.is_current,
        description=source.description,
    )
    output = build_native_buddy_archive(
        snapshot,
        source_context={
            "source_id": source.source_sha256,
            "mapping_source": inspected.mapping_source
            if states is None
            else "user-reviewed",
        },
    )
    if not source.is_current():
        raise ValueError("petdex_source_stale")
    return output
