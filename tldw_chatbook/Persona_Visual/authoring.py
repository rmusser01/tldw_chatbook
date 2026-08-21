"""Isolated, path-free Persona Visual authoring drafts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath

from .assets import (
    PersonaVisualAssetError,
    PersonaVisualAssetMetadata,
    validate_persona_visual_asset_set,
)
from .contracts import (
    ALLOWED_STATE_CATALOG_KINDS,
    PersonaVisualManifest,
    PersonaVisualManifestError,
    RESERVED_STATES,
)
from .publication import (
    PersonaVisualPublicationAssetSource,
    PersonaVisualPublicationSnapshot,
)
from .repository import (
    PersonaVisualAssetRecord,
    PersonaVisualGraph,
    PersonaVisualIdentity,
)
from .validation import validate_persona_visual_manifest


_INVALID = "persona_visual_draft_invalid"
_INCOMPLETE = "persona_visual_draft_incomplete"
_MAX_TITLE = 256
_MAX_DESCRIPTION = 4096
_MAX_PERSONA_ID = 200
_MAX_SOURCE_KEY = 512
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PersonaVisualAuthoringError(ValueError):
    """Stable path-free authoring failure."""

    __slots__ = ("category",)

    def __init__(self, category: str = _INVALID) -> None:
        self.category = category if category in {_INVALID, _INCOMPLETE} else _INVALID
        super().__init__(self.category)


@dataclass(frozen=True, slots=True)
class PersonaVisualDraftAsset:
    """One immutable authoring source and its path-free metadata."""

    source_storage_key: str = field(repr=False)
    metadata: PersonaVisualAssetMetadata


@dataclass(frozen=True, slots=True)
class PersonaVisualAuthoringDraft:
    """One isolated draft whose active identity remains unchanged until Save."""

    persona_id: str
    persona_revision: int
    expected_identity: PersonaVisualIdentity | None
    title: str
    description: str
    source_kind: str
    source_context: tuple[tuple[str, str], ...]
    manifest_json: str
    assets: tuple[PersonaVisualDraftAsset, ...]
    revision: int = 0


@dataclass(frozen=True, slots=True)
class PersonaVisualDraftRow:
    """Path-free metadata for one Workbench state row."""

    state: str
    label: str
    custom: bool
    configured: bool
    animation_id: str | None
    asset_key: str | None


@dataclass(frozen=True, slots=True)
class PersonaVisualDraftInventory:
    """Path-free draft validation and state inventory."""

    rows: tuple[PersonaVisualDraftRow, ...]
    asset_count: int
    activatable: bool
    validation_reason: str | None


def create_persona_visual_draft(
    *,
    persona_id: str,
    persona_revision: int,
    title: str,
    description: str = "",
) -> PersonaVisualAuthoringDraft:
    """Create an empty first-publication draft for a saved local Persona."""

    manifest = {
        "renderer_type": "sprite_frames",
        "manifest_version": 1,
        "states": {},
        "animations": {},
        "fallbacks": {},
        "state_catalog": {},
        "authored_triggers": [],
    }
    draft = PersonaVisualAuthoringDraft(
        persona_id=_text(persona_id, _MAX_PERSONA_ID),
        persona_revision=_nonnegative_int(persona_revision),
        expected_identity=None,
        title=_text(title, _MAX_TITLE),
        description=_text(description, _MAX_DESCRIPTION, allow_empty=True),
        source_kind="manual",
        source_context=(),
        manifest_json=_canonical_json(manifest),
        assets=(),
    )
    return _validated_draft(draft)


def create_persona_visual_import_draft(
    *,
    persona_id: str,
    persona_revision: int,
    expected_identity: PersonaVisualIdentity | None,
    title: str,
    description: str,
    manifest_json: str,
    assets: tuple[PersonaVisualDraftAsset, ...],
) -> PersonaVisualAuthoringDraft:
    """Create a validated review draft from already-confined imported sources."""

    return _validated_draft(
        PersonaVisualAuthoringDraft(
            persona_id=persona_id,
            persona_revision=persona_revision,
            expected_identity=expected_identity,
            title=title,
            description=description,
            source_kind="imported",
            source_context=(("provenance", "untrusted-import"),),
            manifest_json=manifest_json,
            assets=assets,
        )
    )


def persona_visual_draft_from_graph(
    graph: PersonaVisualGraph,
    *,
    source_storage_keys: Mapping[str, str],
) -> PersonaVisualAuthoringDraft:
    """Snapshot one active graph plus caller-confined materialized source keys."""

    try:
        if type(graph) is not PersonaVisualGraph or not isinstance(
            source_storage_keys, Mapping
        ):
            raise ValueError
        if type(graph.identity) is not PersonaVisualIdentity:
            raise ValueError
        records = tuple(graph.assets)
        expected_keys = {record.asset_key for record in records}
        if set(source_storage_keys) != expected_keys:
            raise ValueError
        assets = tuple(
            PersonaVisualDraftAsset(
                _source_key(source_storage_keys[record.asset_key]),
                _metadata_from_record(record),
            )
            for record in records
        )
        draft = PersonaVisualAuthoringDraft(
            persona_id=graph.identity.persona_id,
            persona_revision=graph.identity.persona_revision,
            expected_identity=replace(graph.identity),
            title=graph.pack.title,
            description=graph.pack.description,
            source_kind=graph.pack.source_kind,
            source_context=(),
            manifest_json=_canonical_json(_manifest_document(graph.version.manifest)),
            assets=assets,
        )
        return _validated_draft(draft)
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        PersonaVisualManifestError,
    ):
        raise PersonaVisualAuthoringError() from None


def add_persona_visual_custom_state(
    draft: PersonaVisualAuthoringDraft,
    *,
    state: str,
    label: str,
    kind: str,
) -> PersonaVisualAuthoringDraft:
    """Return a new draft with one unconfigured safe custom state."""

    draft = _validated_draft(draft)
    if (
        type(state) is not str
        or state in RESERVED_STATES
        or type(label) is not str
        or type(kind) is not str
        or kind not in ALLOWED_STATE_CATALOG_KINDS
    ):
        raise PersonaVisualAuthoringError()
    document = _draft_document(draft)
    catalog = dict(document["state_catalog"])
    if state in catalog:
        raise PersonaVisualAuthoringError()
    catalog[state] = {"label": label, "kind": kind}
    document["state_catalog"] = catalog
    return _changed_draft(draft, document, draft.assets)


def replace_persona_visual_draft_state(
    draft: PersonaVisualAuthoringDraft,
    *,
    state: str,
    asset: PersonaVisualDraftAsset,
) -> PersonaVisualAuthoringDraft:
    """Return a new draft mapping one known state to one replacement still."""

    draft = _validated_draft(draft)
    asset = _validated_asset(asset)
    document = _draft_document(draft)
    catalog = document["state_catalog"]
    if type(state) is not str or (
        state not in RESERVED_STATES and state not in catalog
    ):
        raise PersonaVisualAuthoringError()
    if any(
        existing.metadata.asset_key == asset.metadata.asset_key
        for existing in draft.assets
    ):
        raise PersonaVisualAuthoringError()
    animations = dict(document["animations"])
    animations[state] = {
        "frames": [{"asset_id": asset.metadata.asset_key}],
        "preview_asset_id": asset.metadata.asset_key,
        "frame_rate": 1,
    }
    states = dict(document["states"])
    states[state] = {"animation_id": state}
    document["animations"] = animations
    document["states"] = states
    by_key = {item.metadata.asset_key: item for item in draft.assets}
    by_key[asset.metadata.asset_key] = asset
    return _changed_draft(draft, document, tuple(by_key.values()), prune=True)


def clear_persona_visual_draft_state(
    draft: PersonaVisualAuthoringDraft,
    *,
    state: str,
) -> PersonaVisualAuthoringDraft:
    """Return a new draft with one direct state mapping removed."""

    draft = _validated_draft(draft)
    document = _draft_document(draft)
    catalog = document["state_catalog"]
    if type(state) is not str or (
        state not in RESERVED_STATES and state not in catalog
    ):
        raise PersonaVisualAuthoringError()
    states = dict(document["states"])
    states.pop(state, None)
    document["states"] = states
    return _changed_draft(draft, document, draft.assets, prune=True)


def inspect_persona_visual_draft(
    draft: PersonaVisualAuthoringDraft,
) -> PersonaVisualDraftInventory:
    """Return stable validation and row metadata without loading any asset bytes."""

    draft = _validated_draft(draft)
    document = _draft_document(draft)
    manifest = _validate_document(document, draft.assets, activate=False)
    rows = tuple(
        _inventory_row(state, manifest, custom=False) for state in RESERVED_STATES
    ) + tuple(
        _inventory_row(state, manifest, custom=True) for state in manifest.state_catalog
    )
    try:
        _validate_document(document, draft.assets, activate=True)
    except PersonaVisualManifestError:
        activatable = False
        reason = _INCOMPLETE
    else:
        activatable = True
        reason = None
    return PersonaVisualDraftInventory(
        rows=rows,
        asset_count=len(draft.assets),
        activatable=activatable,
        validation_reason=reason,
    )


def persona_visual_draft_publication_snapshot(
    draft: PersonaVisualAuthoringDraft,
) -> PersonaVisualPublicationSnapshot:
    """Convert one activatable draft to the existing immutable publish request."""

    draft = _validated_draft(draft)
    try:
        _validate_document(_draft_document(draft), draft.assets, activate=True)
    except PersonaVisualManifestError:
        raise PersonaVisualAuthoringError(_INCOMPLETE) from None
    return PersonaVisualPublicationSnapshot(
        persona_id=draft.persona_id,
        persona_revision=draft.persona_revision,
        title=draft.title,
        description=draft.description,
        source_kind=draft.source_kind,
        source_context=draft.source_context,
        manifest_json=draft.manifest_json,
        assets=tuple(
            PersonaVisualPublicationAssetSource(
                source_storage_key=asset.source_storage_key,
                metadata=asset.metadata,
            )
            for asset in draft.assets
        ),
        expected_identity=draft.expected_identity,
    )


def _changed_draft(
    draft: PersonaVisualAuthoringDraft,
    document: dict[str, object],
    assets: tuple[PersonaVisualDraftAsset, ...],
    *,
    prune: bool = False,
) -> PersonaVisualAuthoringDraft:
    if prune:
        document, assets = _pruned(document, assets)
    try:
        _validate_document(document, assets, activate=False)
    except PersonaVisualManifestError:
        raise PersonaVisualAuthoringError() from None
    return _validated_draft(
        replace(
            draft,
            manifest_json=_canonical_json(document),
            assets=assets,
            revision=draft.revision + 1,
        )
    )


def _pruned(
    document: dict[str, object],
    assets: tuple[PersonaVisualDraftAsset, ...],
) -> tuple[dict[str, object], tuple[PersonaVisualDraftAsset, ...]]:
    states = document["states"]
    animations = document["animations"]
    used_animations = {
        state["animation_id"]
        for state in states.values()
        if isinstance(state, dict) and type(state.get("animation_id")) is str
    }
    kept_animations = {
        key: value for key, value in animations.items() if key in used_animations
    }
    used_assets: set[str] = set()
    for animation in kept_animations.values():
        if not isinstance(animation, dict):
            continue
        for frame in animation.get("frames", ()):
            if isinstance(frame, dict) and type(frame.get("asset_id")) is str:
                used_assets.add(frame["asset_id"])
        preview = animation.get("preview_asset_id")
        if type(preview) is str:
            used_assets.add(preview)
    document["animations"] = kept_animations
    return document, tuple(
        asset for asset in assets if asset.metadata.asset_key in used_assets
    )


def _inventory_row(
    state: str,
    manifest: PersonaVisualManifest,
    *,
    custom: bool,
) -> PersonaVisualDraftRow:
    animation_id = manifest.states.get(state)
    asset_key = None
    if animation_id is not None:
        animation = manifest.animations[animation_id]
        asset_key = animation.preview_asset_id or animation.frames[0].asset_id
    label = (
        manifest.state_catalog[state].label
        if custom
        else state.replace("_", " ").title()
    )
    return PersonaVisualDraftRow(
        state=state,
        label=label,
        custom=custom,
        configured=animation_id is not None,
        animation_id=animation_id,
        asset_key=asset_key,
    )


def _validated_draft(value: object) -> PersonaVisualAuthoringDraft:
    try:
        if type(value) is not PersonaVisualAuthoringDraft:
            raise ValueError
        _text(value.persona_id, _MAX_PERSONA_ID)
        _nonnegative_int(value.persona_revision)
        if value.expected_identity is not None:
            identity = _validated_identity(value.expected_identity)
            if (
                identity.persona_id != value.persona_id
                or identity.persona_revision != value.persona_revision
            ):
                raise ValueError
        _text(value.title, _MAX_TITLE)
        _text(value.description, _MAX_DESCRIPTION, allow_empty=True)
        _text(value.source_kind, 64)
        if type(value.source_context) is not tuple:
            raise ValueError
        for item in value.source_context:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not str
            ):
                raise ValueError
        if type(value.manifest_json) is not str or type(value.assets) is not tuple:
            raise ValueError
        _nonnegative_int(value.revision)
        assets = tuple(_validated_asset(asset) for asset in value.assets)
        if len({asset.metadata.asset_key for asset in assets}) != len(assets):
            raise ValueError
        if assets:
            normalized = validate_persona_visual_asset_set(
                tuple(asset.metadata for asset in assets)
            )
            assets = tuple(
                PersonaVisualDraftAsset(asset.source_storage_key, metadata)
                for asset, metadata in zip(assets, normalized)
            )
        document = _load_canonical_document(value.manifest_json)
        _validate_document(document, assets, activate=False)
        return replace(
            value,
            expected_identity=(
                replace(value.expected_identity)
                if value.expected_identity is not None
                else None
            ),
            assets=assets,
        )
    except (
        TypeError,
        ValueError,
        UnicodeError,
        PersonaVisualAssetError,
        PersonaVisualManifestError,
    ):
        raise PersonaVisualAuthoringError() from None


def _validated_asset(value: object) -> PersonaVisualDraftAsset:
    if type(value) is not PersonaVisualDraftAsset:
        raise PersonaVisualAuthoringError()
    try:
        source_key = _source_key(value.source_storage_key)
        metadata = value.metadata
        if type(metadata) is not PersonaVisualAssetMetadata:
            raise ValueError
        metadata = validate_persona_visual_asset_set((metadata,))[0]
        return PersonaVisualDraftAsset(source_key, metadata)
    except (TypeError, ValueError, UnicodeError, PersonaVisualAssetError):
        raise PersonaVisualAuthoringError() from None


def _validated_identity(value: object) -> PersonaVisualIdentity:
    if type(value) is not PersonaVisualIdentity:
        raise ValueError
    _text(value.persona_id, _MAX_PERSONA_ID)
    _nonnegative_int(value.persona_revision)
    for field_name in (
        "binding_id",
        "binding_version",
        "pack_id",
        "pack_revision",
        "pack_version_id",
        "version_number",
    ):
        field_value = getattr(value, field_name)
        if type(field_value) is not int or field_value <= 0:
            raise ValueError
    if (
        type(value.manifest_sha256) is not str
        or _SHA256.fullmatch(value.manifest_sha256) is None
    ):
        raise ValueError
    return replace(value)


def _validate_document(
    document: object,
    assets: tuple[PersonaVisualDraftAsset, ...],
    *,
    activate: bool,
) -> PersonaVisualManifest:
    dimensions = {
        asset.metadata.asset_key: (asset.metadata.width, asset.metadata.height)
        for asset in assets
    }
    return validate_persona_visual_manifest(
        document,
        dimensions,
        activate=activate,
    )


def _draft_document(draft: PersonaVisualAuthoringDraft) -> dict[str, object]:
    return _load_canonical_document(draft.manifest_json)


def _load_canonical_document(value: str) -> dict[str, object]:
    document = json.loads(value)
    if type(document) is not dict or _canonical_json(document) != value:
        raise ValueError
    return document


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _manifest_document(manifest: PersonaVisualManifest) -> dict[str, object]:
    if type(manifest) is not PersonaVisualManifest:
        raise ValueError
    animations: dict[str, object] = {}
    for animation_id, animation in manifest.animations.items():
        record: dict[str, object] = {
            "frames": [
                {
                    "asset_id": frame.asset_id,
                    **(
                        {"duration_ms": frame.duration_ms}
                        if frame.duration_ms is not None
                        else {}
                    ),
                    **(
                        {
                            "region": {
                                "x": frame.region.x,
                                "y": frame.region.y,
                                "width": frame.region.width,
                                "height": frame.region.height,
                            }
                        }
                        if frame.region is not None
                        else {}
                    ),
                }
                for frame in animation.frames
            ],
            "frame_rate": animation.frame_rate,
            "loop": animation.loop,
        }
        if animation.alignment is not None:
            record["alignment"] = {
                "x": animation.alignment.x,
                "y": animation.alignment.y,
            }
        if animation.preview_frame is not None:
            record["preview_frame"] = animation.preview_frame
        if animation.preview_asset_id is not None:
            record["preview_asset_id"] = animation.preview_asset_id
        animations[animation_id] = record
    return {
        "renderer_type": manifest.renderer_type,
        "manifest_version": manifest.manifest_version,
        "states": {
            state: {"animation_id": animation_id}
            for state, animation_id in manifest.states.items()
        },
        "animations": animations,
        "fallbacks": {
            state: list(fallbacks) for state, fallbacks in manifest.fallbacks.items()
        },
        "state_catalog": {
            state: {
                "label": entry.label,
                "kind": entry.kind,
                **(
                    {"description": entry.description}
                    if entry.description is not None
                    else {}
                ),
                **({"tags": list(entry.tags)} if entry.tags else {}),
            }
            for state, entry in manifest.state_catalog.items()
        },
        "authored_triggers": [
            {
                "id": trigger.id,
                "source": trigger.source,
                "match": trigger.match,
                "state": trigger.state,
                "duration_ms": trigger.duration_ms,
                "priority": trigger.priority,
            }
            for trigger in manifest.triggers
        ],
    }


def _metadata_from_record(record: object) -> PersonaVisualAssetMetadata:
    if type(record) is not PersonaVisualAssetRecord:
        raise ValueError
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


def _source_key(value: object) -> str:
    value = _text(value, _MAX_SOURCE_KEY)
    if "\\" in value or value.startswith("/"):
        raise ValueError
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError
    return path.as_posix()


def _text(value: object, maximum: int, *, allow_empty: bool = False) -> str:
    if (
        type(value) is not str
        or len(value) > maximum
        or (not value and not allow_empty)
    ):
        raise ValueError
    value.encode("utf-8")
    return value


def _nonnegative_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError
    return value


__all__ = [
    "PersonaVisualAuthoringDraft",
    "PersonaVisualAuthoringError",
    "PersonaVisualDraftAsset",
    "PersonaVisualDraftInventory",
    "PersonaVisualDraftRow",
    "add_persona_visual_custom_state",
    "clear_persona_visual_draft_state",
    "create_persona_visual_draft",
    "create_persona_visual_import_draft",
    "inspect_persona_visual_draft",
    "persona_visual_draft_from_graph",
    "persona_visual_draft_publication_snapshot",
    "replace_persona_visual_draft_state",
]
