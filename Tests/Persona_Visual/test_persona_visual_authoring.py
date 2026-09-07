"""Tests for isolated Persona Visual Workbench drafts."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest

from tldw_chatbook.Persona_Visual.assets import PersonaVisualAssetMetadata
from tldw_chatbook.Persona_Visual.authoring import (
    PersonaVisualAuthoringError,
    PersonaVisualDraftAsset,
    add_persona_visual_custom_state,
    clear_persona_visual_draft_state,
    create_persona_visual_draft,
    inspect_persona_visual_draft,
    persona_visual_draft_from_graph,
    persona_visual_draft_publication_snapshot,
    replace_persona_visual_draft_state,
)
from tldw_chatbook.Persona_Visual.contracts import RESERVED_STATES
from tldw_chatbook.Persona_Visual.repository import (
    PersonaVisualAssetRecord,
    PersonaVisualBindingRecord,
    PersonaVisualGraph,
    PersonaVisualIdentity,
    PersonaVisualPackRecord,
    PersonaVisualVersionRecord,
)
from tldw_chatbook.Persona_Visual.validation import validate_persona_visual_manifest


def _manifest() -> dict[str, object]:
    return {
        "renderer_type": "sprite_frames",
        "manifest_version": 1,
        "states": {
            state: {"animation_id": "idle-loop"}
            for state in ("idle", "listening", "thinking", "speaking", "error")
        },
        "animations": {
            "idle-loop": {
                "frames": [{"asset_id": "idle"}],
                "frame_rate": 2,
                "preview_asset_id": "idle",
            }
        },
        "fallbacks": {},
        "state_catalog": {},
        "authored_triggers": [],
    }


def _metadata(
    asset_key: str,
    *,
    marker: bytes | None = None,
) -> PersonaVisualAssetMetadata:
    data = marker or asset_key.encode("utf-8")
    return PersonaVisualAssetMetadata(
        asset_key=asset_key,
        role="frame",
        mime_type="image/png",
        byte_count=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        width=4,
        height=5,
        frame_count=1,
    )


def _graph() -> PersonaVisualGraph:
    manifest = validate_persona_visual_manifest(_manifest(), {"idle": (4, 5)})
    identity = PersonaVisualIdentity(
        persona_id="local-persona-7",
        persona_revision=9,
        binding_id=11,
        binding_version=3,
        pack_id=13,
        pack_revision=4,
        pack_version_id=17,
        version_number=2,
        manifest_sha256="a" * 64,
    )
    return PersonaVisualGraph(
        identity=identity,
        pack=PersonaVisualPackRecord(
            id=13,
            title="Operator states",
            description="Local operational art",
            status="active",
            source_kind="manual",
            created_at="2026-08-20 12:00:00",
            updated_at="2026-08-20 12:00:00",
            revision=4,
        ),
        version=PersonaVisualVersionRecord(
            id=17,
            pack_id=13,
            version_number=2,
            renderer_type="sprite_frames",
            manifest_version=1,
            manifest=manifest,
            manifest_sha256="a" * 64,
            created_at="2026-08-20 12:00:00",
        ),
        binding=PersonaVisualBindingRecord(
            id=11,
            persona_id="local-persona-7",
            persona_revision=9,
            pack_id=13,
            active_version_id=17,
            status="active",
            created_at="2026-08-20 12:00:00",
            updated_at="2026-08-20 12:00:00",
            revision=3,
        ),
        assets=(
            PersonaVisualAssetRecord(
                id=19,
                pack_id=13,
                pack_version_id=17,
                asset_key="idle",
                role="frame",
                mime_type="image/png",
                byte_count=4,
                sha256=hashlib.sha256(b"idle").hexdigest(),
                width=4,
                height=5,
                frame_count=1,
                duration_ms=None,
                created_at="2026-08-20 12:00:00",
            ),
        ),
    )


def _active_draft():
    return persona_visual_draft_from_graph(
        _graph(),
        source_storage_keys={"idle": "active/idle.png"},
    )


def test_active_graph_becomes_an_immutable_isolated_draft() -> None:
    graph = _graph()

    draft = persona_visual_draft_from_graph(
        graph,
        source_storage_keys={"idle": "active/idle.png"},
    )

    assert draft.persona_id == graph.identity.persona_id
    assert draft.persona_revision == graph.identity.persona_revision
    assert draft.expected_identity == graph.identity
    assert draft.title == graph.pack.title
    assert draft.revision == 0
    assert draft.assets[0].metadata.asset_key == "idle"
    assert "active/idle.png" not in repr(draft)
    with pytest.raises(FrozenInstanceError):
        draft.title = "mutated"  # type: ignore[misc]


def test_inventory_always_lists_nine_baselines_then_safe_custom_states() -> None:
    draft = add_persona_visual_custom_state(
        _active_draft(),
        state="deep_focus",
        label="Deep focus",
        kind="mood",
    )

    inventory = inspect_persona_visual_draft(draft)

    assert tuple(row.state for row in inventory.rows[:9]) == RESERVED_STATES
    assert inventory.rows[-1].state == "deep_focus"
    assert inventory.rows[-1].label == "Deep focus"
    assert inventory.rows[-1].custom is True
    assert inventory.rows[-1].configured is False
    assert inventory.activatable is True
    assert inventory.asset_count == 1
    assert inventory.validation_reason is None


@pytest.mark.parametrize(
    "state",
    ("Idle", "../private", "secret_key", "idle", "x" * 97),
)
def test_custom_state_rejects_unsafe_reserved_or_private_slugs(state: str) -> None:
    with pytest.raises(
        PersonaVisualAuthoringError, match="^persona_visual_draft_invalid$"
    ):
        add_persona_visual_custom_state(
            _active_draft(),
            state=state,
            label="Custom",
            kind="mood",
        )


def test_replace_returns_new_draft_and_preserves_authoritative_draft() -> None:
    original = _active_draft()
    replacement = PersonaVisualDraftAsset("uploads/speaking.png", _metadata("speaking"))

    changed = replace_persona_visual_draft_state(
        original,
        state="speaking",
        asset=replacement,
    )

    original_inventory = inspect_persona_visual_draft(original)
    changed_inventory = inspect_persona_visual_draft(changed)
    assert original.revision == 0
    assert changed.revision == 1
    assert tuple(asset.metadata.asset_key for asset in original.assets) == ("idle",)
    assert tuple(asset.metadata.asset_key for asset in changed.assets) == (
        "idle",
        "speaking",
    )
    assert (
        next(
            row for row in original_inventory.rows if row.state == "speaking"
        ).asset_key
        == "idle"
    )
    assert (
        next(row for row in changed_inventory.rows if row.state == "speaking").asset_key
        == "speaking"
    )


def test_replace_cannot_reuse_an_asset_key_owned_by_other_states() -> None:
    original = _active_draft()

    with pytest.raises(
        PersonaVisualAuthoringError,
        match="^persona_visual_draft_invalid$",
    ):
        replace_persona_visual_draft_state(
            original,
            state="speaking",
            asset=PersonaVisualDraftAsset(
                "uploads/different-idle.png",
                _metadata("idle", marker=b"different"),
            ),
        )

    assert inspect_persona_visual_draft(original).activatable is True


def test_clear_required_state_is_reviewable_but_not_publishable() -> None:
    original = _active_draft()

    changed = clear_persona_visual_draft_state(original, state="error")
    inventory = inspect_persona_visual_draft(changed)

    assert changed.revision == 1
    assert (
        next(row for row in inventory.rows if row.state == "error").configured is False
    )
    assert inventory.activatable is False
    assert inventory.validation_reason == "persona_visual_draft_incomplete"
    assert inspect_persona_visual_draft(original).activatable is True
    with pytest.raises(
        PersonaVisualAuthoringError,
        match="^persona_visual_draft_incomplete$",
    ):
        persona_visual_draft_publication_snapshot(changed)


def test_clear_prunes_only_assets_and_animations_no_longer_referenced() -> None:
    replaced = replace_persona_visual_draft_state(
        _active_draft(),
        state="speaking",
        asset=PersonaVisualDraftAsset(
            "uploads/speaking.png",
            _metadata("speaking"),
        ),
    )

    cleared = clear_persona_visual_draft_state(replaced, state="speaking")

    assert tuple(asset.metadata.asset_key for asset in cleared.assets) == ("idle",)
    assert "speaking" not in cleared.manifest_json


def test_empty_draft_captures_first_publication_authority() -> None:
    draft = create_persona_visual_draft(
        persona_id="new-local-persona",
        persona_revision=1,
        title="New operational states",
    )

    assert draft.expected_identity is None
    assert draft.assets == ()
    assert inspect_persona_visual_draft(draft).activatable is False
    assert (
        tuple(row.state for row in inspect_persona_visual_draft(draft).rows)
        == RESERVED_STATES
    )


def test_publication_snapshot_is_canonical_and_keeps_exact_authority() -> None:
    original = _active_draft()
    changed = replace_persona_visual_draft_state(
        original,
        state="speaking",
        asset=PersonaVisualDraftAsset(
            "uploads/speaking.png",
            _metadata("speaking"),
        ),
    )

    snapshot = persona_visual_draft_publication_snapshot(changed)

    assert snapshot.persona_id == original.persona_id
    assert snapshot.persona_revision == original.persona_revision
    assert snapshot.expected_identity == original.expected_identity
    assert snapshot.title == original.title
    assert snapshot.manifest_json == changed.manifest_json
    assert tuple(source.source_storage_key for source in snapshot.assets) == (
        "active/idle.png",
        "uploads/speaking.png",
    )


def test_draft_rejects_missing_or_extra_source_keys() -> None:
    graph = _graph()

    for sources in ({}, {"idle": "idle.png", "extra": "extra.png"}):
        with pytest.raises(
            PersonaVisualAuthoringError,
            match="^persona_visual_draft_invalid$",
        ):
            persona_visual_draft_from_graph(graph, source_storage_keys=sources)


def test_draft_revalidates_hostile_frozen_metadata_and_identity() -> None:
    draft = _active_draft()
    object.__setattr__(draft.assets[0].metadata, "byte_count", -1)

    with pytest.raises(
        PersonaVisualAuthoringError,
        match="^persona_visual_draft_invalid$",
    ):
        inspect_persona_visual_draft(draft)

    draft = _active_draft()
    hostile_identity = replace(draft.expected_identity)
    object.__setattr__(hostile_identity, "binding_id", ["private/path"])
    object.__setattr__(draft, "expected_identity", hostile_identity)

    with pytest.raises(
        PersonaVisualAuthoringError,
        match="^persona_visual_draft_invalid$",
    ):
        persona_visual_draft_publication_snapshot(draft)
