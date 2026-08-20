"""Tests for path-free Persona Visual operational-state resolution."""

from __future__ import annotations

import hashlib
from dataclasses import fields, replace
from pathlib import Path
from types import MappingProxyType

import pytest

import tldw_chatbook.Persona_Visual.runtime as runtime_module
from tldw_chatbook.Persona_Visual.assets import (
    PersonaVisualAsset,
    PersonaVisualAssetError,
    PersonaVisualAssetMetadata,
)
from tldw_chatbook.Persona_Visual.repository import (
    PersonaVisualAssetRecord,
    PersonaVisualBindingRecord,
    PersonaVisualGraph,
    PersonaVisualIdentity,
    PersonaVisualPackRecord,
    PersonaVisualVersionRecord,
)
from tldw_chatbook.Persona_Visual.runtime import (
    PersonaVisualPortrait,
    resolve_persona_visual,
)
from tldw_chatbook.Persona_Visual.validation import validate_persona_visual_manifest


_WORKTREE = Path(__file__).resolve().parents[2]
_NOW = "2026-08-20 20:00:00"


def _manifest_payload() -> dict[str, object]:
    return {
        "renderer_type": "sprite_frames",
        "manifest_version": 1,
        "states": {
            "idle": {"animation_id": "idle-animation"},
            "listening": {"animation_id": "listening-animation"},
            "thinking": {"animation_id": "idle-animation"},
            "speaking": {"animation_id": "idle-animation"},
            "error": {"animation_id": "idle-animation"},
            "tool.notes": {"animation_id": "custom-animation"},
        },
        "animations": {
            "idle-animation": {
                "frames": [{"asset_id": "idle"}],
                "frame_rate": 1,
                "loop": True,
            },
            "listening-animation": {
                "frames": [
                    {"asset_id": "listen-1", "duration_ms": 80},
                    {"asset_id": "listen-2", "duration_ms": 120},
                ],
                "frame_rate": 12,
                "loop": True,
                "preview_frame": 1,
                "preview_asset_id": "listen-1",
            },
            "custom-animation": {
                "frames": [{"asset_id": "custom"}],
                "frame_rate": 4,
                "loop": False,
            },
        },
        "fallbacks": {
            "tool.missing": ["tool.middle"],
            "tool.middle": ["tool.notes"],
            "tool.broken": ["listening"],
        },
        "state_catalog": {
            "tool.notes": {"label": "Notes", "kind": "tool_variant"},
            "tool.missing": {"label": "Missing", "kind": "tool_variant"},
            "tool.middle": {"label": "Middle", "kind": "tool_variant"},
            "tool.broken": {"label": "Broken", "kind": "tool_variant"},
        },
        "authored_triggers": [],
    }


def _asset_record(
    asset_id: int,
    asset_key: str,
    *,
    pack_id: int = 31,
    version_id: int = 41,
    data: bytes | None = None,
) -> PersonaVisualAssetRecord:
    payload = data if data is not None else f"bytes:{asset_key}".encode()
    return PersonaVisualAssetRecord(
        id=asset_id,
        pack_id=pack_id,
        pack_version_id=version_id,
        asset_key=asset_key,
        role="frame",
        mime_type="image/png",
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        width=4,
        height=5,
        frame_count=1,
        duration_ms=None,
        created_at=_NOW,
    )


def _graph(
    *,
    persona_id: str = "persona-1",
    persona_revision: int = 7,
    binding_id: int = 11,
    binding_version: int = 13,
    pack_id: int = 31,
    pack_revision: int = 17,
    version_id: int = 41,
    version_number: int = 19,
    manifest_sha256: str = "a" * 64,
    asset_id_offset: int = 0,
    asset_digest_override: str | None = None,
    idle_frame_count: int = 1,
    manifest: object | None = None,
) -> PersonaVisualGraph:
    assets = tuple(
        _asset_record(
            index + asset_id_offset, key, pack_id=pack_id, version_id=version_id
        )
        for index, key in enumerate(("idle", "listen-1", "listen-2", "custom"), 101)
    )
    if asset_digest_override is not None:
        assets = (replace(assets[0], sha256=asset_digest_override), *assets[1:])
    if idle_frame_count != 1:
        assets = (replace(assets[0], frame_count=idle_frame_count), *assets[1:])
    validated = manifest or validate_persona_visual_manifest(
        _manifest_payload(),
        {asset.asset_key: (asset.width, asset.height) for asset in assets},
    )
    identity = PersonaVisualIdentity(
        persona_id=persona_id,
        persona_revision=persona_revision,
        binding_id=binding_id,
        binding_version=binding_version,
        pack_id=pack_id,
        pack_revision=pack_revision,
        pack_version_id=version_id,
        version_number=version_number,
        manifest_sha256=manifest_sha256,
    )
    return PersonaVisualGraph(
        identity=identity,
        pack=PersonaVisualPackRecord(
            id=pack_id,
            title="Operator states",
            description="",
            status="active",
            source_kind="manual",
            created_at=_NOW,
            updated_at=_NOW,
            revision=pack_revision,
        ),
        version=PersonaVisualVersionRecord(
            id=version_id,
            pack_id=pack_id,
            version_number=version_number,
            renderer_type="sprite_frames",
            manifest_version=1,
            manifest=validated,  # type: ignore[arg-type]
            manifest_sha256=manifest_sha256,
            created_at=_NOW,
        ),
        binding=PersonaVisualBindingRecord(
            id=binding_id,
            persona_id=persona_id,
            persona_revision=persona_revision,
            pack_id=pack_id,
            active_version_id=version_id,
            status="active",
            created_at=_NOW,
            updated_at=_NOW,
            revision=binding_version,
        ),
        assets=assets,
    )


class RecordingLoader:
    def __init__(self, *, fail: set[str] | None = None) -> None:
        self.fail = fail or set()
        self.calls: list[tuple[PersonaVisualIdentity, int, str, int]] = []

    def __call__(
        self,
        identity: PersonaVisualIdentity,
        record: PersonaVisualAssetRecord,
        selected_frame: int,
    ) -> PersonaVisualAsset:
        self.calls.append((identity, record.id, record.asset_key, selected_frame))
        if record.asset_key in self.fail:
            raise PersonaVisualAssetError()
        data = f"bytes:{record.asset_key}".encode()
        metadata = PersonaVisualAssetMetadata(
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
        return PersonaVisualAsset(metadata=metadata, data=data, selected_frame=0)


def _portrait() -> PersonaVisualPortrait:
    data = b"portrait-bytes"
    return PersonaVisualPortrait(
        portrait_id="portrait-1",
        revision=3,
        mime_type="image/png",
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
    )


def test_runtime_imports_from_the_assigned_worktree() -> None:
    assert Path(runtime_module.__file__).resolve().is_relative_to(_WORKTREE)


@pytest.mark.parametrize(
    ("requested", "resolved", "animation_id", "reason"),
    [
        ("listening", "listening", "listening-animation", None),
        ("tool.notes", "tool.notes", "custom-animation", None),
        (
            "tool.missing",
            "tool.notes",
            "custom-animation",
            "persona_visual_state_fallback",
        ),
        ("unknown.state", "idle", "idle-animation", "persona_visual_state_fallback"),
    ],
)
def test_resolves_direct_custom_multihop_and_missing_states(
    requested: str,
    resolved: str,
    animation_id: str,
    reason: str | None,
) -> None:
    loader = RecordingLoader()

    result = resolve_persona_visual(_graph(), requested, asset_loader=loader)

    assert result.source == "persona_visual"
    assert result.requested_state == requested
    assert result.resolved_state == resolved
    assert result.animation_id == animation_id
    assert result.reason == reason
    assert result.portrait is None
    assert result.frames


def test_unusable_animation_tries_manifest_fallback_then_stops() -> None:
    loader = RecordingLoader(fail={"listen-2"})

    result = resolve_persona_visual(_graph(), "tool.broken", asset_loader=loader)

    assert result.source == "persona_visual"
    assert result.resolved_state == "idle"
    assert result.reason == "persona_visual_state_fallback"
    assert [call[2] for call in loader.calls] == ["listen-1", "listen-2", "idle"]
    assert [frame.asset_key for frame in result.frames] == ["idle"]


def test_unusable_requested_animation_uses_healthy_manifest_fallback() -> None:
    payload = _manifest_payload()
    payload["fallbacks"]["listening"] = ["tool.notes"]  # type: ignore[index]
    graph = _graph(
        manifest=validate_persona_visual_manifest(
            payload,
            {key: (4, 5) for key in ("idle", "listen-1", "listen-2", "custom")},
        )
    )
    loader = RecordingLoader(fail={"listen-2"})

    result = resolve_persona_visual(graph, "listening", asset_loader=loader)

    assert result.resolved_state == "tool.notes"
    assert [call[2] for call in loader.calls] == ["listen-1", "listen-2", "custom"]
    assert [frame.asset_key for frame in result.frames] == ["custom"]


def test_missing_requested_asset_record_uses_healthy_manifest_fallback() -> None:
    payload = _manifest_payload()
    payload["fallbacks"]["listening"] = ["tool.notes"]  # type: ignore[index]
    graph = _graph(
        manifest=validate_persona_visual_manifest(
            payload,
            {key: (4, 5) for key in ("idle", "listen-1", "listen-2", "custom")},
        )
    )
    graph = replace(
        graph,
        assets=tuple(asset for asset in graph.assets if asset.asset_key != "listen-2"),
    )
    loader = RecordingLoader()

    result = resolve_persona_visual(graph, "listening", asset_loader=loader)

    assert result.source == "persona_visual"
    assert result.resolved_state == "tool.notes"
    assert result.reason == "persona_visual_state_fallback"
    assert [call[2] for call in loader.calls] == ["listen-1", "custom"]


@pytest.mark.parametrize("portrait", [_portrait(), None])
def test_invalid_idle_falls_back_to_portrait_or_unavailable(
    portrait: PersonaVisualPortrait | None,
) -> None:
    loader = RecordingLoader(fail={"idle", "listen-1", "custom"})

    result = resolve_persona_visual(
        _graph(),
        "unknown.state",
        asset_loader=loader,
        portrait=portrait,
    )

    assert result.source == ("persona_portrait" if portrait else "unavailable")
    assert result.reason == (
        "persona_visual_idle_unavailable" if portrait else "persona_visual_unavailable"
    )
    assert result.frames == ()
    assert result.portrait == portrait


@pytest.mark.parametrize("portrait", [_portrait(), None])
def test_missing_idle_asset_record_falls_back_to_portrait_or_unavailable(
    portrait: PersonaVisualPortrait | None,
) -> None:
    graph = _graph()
    graph = replace(
        graph,
        assets=tuple(asset for asset in graph.assets if asset.asset_key != "idle"),
    )

    result = resolve_persona_visual(
        graph,
        "unknown.state",
        asset_loader=RecordingLoader(),
        portrait=portrait,
    )

    assert result.source == ("persona_portrait" if portrait else "unavailable")
    assert result.reason == (
        "persona_visual_idle_unavailable" if portrait else "persona_visual_unavailable"
    )


def test_stored_fallback_cycle_fails_closed_without_exposing_details() -> None:
    original = _graph().version.manifest
    corrupt = replace(
        original,
        fallbacks=MappingProxyType(
            {
                **original.fallbacks,
                "tool.missing": ("tool.middle",),
                "tool.middle": ("tool.missing",),
            }
        ),
    )
    portrait = _portrait()
    loader = RecordingLoader()

    result = resolve_persona_visual(
        _graph(manifest=corrupt),
        "tool.missing",
        asset_loader=loader,
        portrait=portrait,
    )

    assert result.source == "persona_portrait"
    assert result.reason == "persona_visual_graph_invalid"
    assert loader.calls == []
    assert not any(
        "path" in field.name or "storage" in field.name for field in fields(result)
    )


def test_animation_loads_all_frames_and_never_returns_a_partial_animation() -> None:
    healthy = resolve_persona_visual(
        _graph(), "listening", asset_loader=RecordingLoader()
    )
    failed = resolve_persona_visual(
        _graph(),
        "listening",
        asset_loader=RecordingLoader(fail={"listen-2"}),
    )

    assert healthy.animate is True
    assert healthy.frame_rate == 12
    assert [frame.duration_ms for frame in healthy.frames] == [80, 120]
    assert [frame.asset_key for frame in healthy.frames] == ["listen-1", "listen-2"]
    assert failed.resolved_state == "idle"
    assert [frame.asset_key for frame in failed.frames] == ["idle"]


def test_intrinsically_animated_asset_is_not_mistaken_for_a_still() -> None:
    result = resolve_persona_visual(
        _graph(idle_frame_count=3), "idle", asset_loader=RecordingLoader()
    )

    assert result.animate is True
    assert [frame.asset_key for frame in result.frames] == ["idle"]


def test_reduced_motion_uses_preview_frame_before_preview_asset_and_only_loads_it() -> (
    None
):
    loader = RecordingLoader()

    result = resolve_persona_visual(
        _graph(), "listening", asset_loader=loader, reduced_motion=True
    )

    assert result.animate is False
    assert [frame.asset_key for frame in result.frames] == ["listen-2"]
    assert [call[2] for call in loader.calls] == ["listen-2"]
    assert result.static_reason == "preview_frame"


def test_reduced_motion_uses_preview_asset_then_frame_zero() -> None:
    payload = _manifest_payload()
    animation = payload["animations"]["listening-animation"]  # type: ignore[index]
    animation.pop("preview_frame")
    graph = _graph(
        manifest=validate_persona_visual_manifest(
            payload,
            {key: (4, 5) for key in ("idle", "listen-1", "listen-2", "custom")},
        )
    )
    preview = resolve_persona_visual(
        graph, "listening", asset_loader=RecordingLoader(), reduced_motion=True
    )

    animation.pop("preview_asset_id")
    graph = _graph(
        manifest=validate_persona_visual_manifest(
            payload,
            {key: (4, 5) for key in ("idle", "listen-1", "listen-2", "custom")},
        )
    )
    first = resolve_persona_visual(
        graph, "listening", asset_loader=RecordingLoader(), reduced_motion=True
    )

    assert [frame.asset_key for frame in preview.frames] == ["listen-1"]
    assert preview.static_reason == "preview_asset_id"
    assert [frame.asset_key for frame in first.frames] == ["listen-1"]
    assert first.static_reason == "first_frame"


def test_healthy_candidate_does_not_load_fallback_or_idle_assets() -> None:
    loader = RecordingLoader()

    resolve_persona_visual(_graph(), "listening", asset_loader=loader)

    assert [call[2] for call in loader.calls] == ["listen-1", "listen-2"]


def test_absent_active_graph_uses_normal_portrait_fallback_reason() -> None:
    loader = RecordingLoader()

    result = resolve_persona_visual(
        None, "idle", asset_loader=loader, portrait=_portrait()
    )

    assert result.source == "persona_portrait"
    assert result.reason == "persona_visual_idle_unavailable"
    assert loader.calls == []


@pytest.mark.parametrize("graph", [None, _graph()])
def test_invalid_requested_state_is_normalized_before_public_results(
    graph: PersonaVisualGraph | None,
) -> None:
    private_marker = "/Users/example/.config/private/state"

    result = resolve_persona_visual(
        graph,
        private_marker,
        asset_loader=RecordingLoader(),
        portrait=_portrait(),
    )

    assert result.requested_state == "invalid"
    assert result.cache_identity.requested_state == "invalid"
    assert private_marker not in repr(result)


@pytest.mark.parametrize(
    "variant",
    [
        {"persona_id": "persona-2"},
        {"persona_revision": 8},
        {"binding_id": 12},
        {"binding_version": 14},
        {"pack_id": 32},
        {"pack_revision": 18},
        {"version_id": 42},
        {"version_number": 20},
        {"manifest_sha256": "b" * 64},
    ],
)
def test_cache_identity_changes_for_each_graph_identity_field(
    variant: dict[str, object],
) -> None:
    baseline = resolve_persona_visual(
        _graph(), "idle", asset_loader=RecordingLoader()
    ).cache_identity
    changed = resolve_persona_visual(
        _graph(**variant),  # type: ignore[arg-type]
        "idle",
        asset_loader=RecordingLoader(),
    ).cache_identity

    assert changed != baseline
    assert changed.graph != baseline.graph


@pytest.mark.parametrize(
    "variant",
    [{"asset_id_offset": 100}, {"asset_digest_override": "c" * 64}],
)
def test_cache_identity_changes_for_asset_id_and_digest(
    variant: dict[str, object],
) -> None:
    baseline = resolve_persona_visual(
        _graph(), "idle", asset_loader=RecordingLoader()
    ).cache_identity
    changed = resolve_persona_visual(
        _graph(**variant),  # type: ignore[arg-type]
        "idle",
        asset_loader=RecordingLoader(),
    ).cache_identity

    assert changed.assets != baseline.assets


def test_cache_identity_changes_for_requested_resolved_asset_and_motion_fields() -> (
    None
):
    graph = _graph()
    idle = resolve_persona_visual(graph, "idle", asset_loader=RecordingLoader())
    missing = resolve_persona_visual(
        graph, "unknown.state", asset_loader=RecordingLoader()
    )
    listening = resolve_persona_visual(
        graph, "listening", asset_loader=RecordingLoader()
    )
    reduced = resolve_persona_visual(
        graph,
        "listening",
        asset_loader=RecordingLoader(),
        reduced_motion=True,
    )

    assert (
        len(
            {
                idle.cache_identity,
                missing.cache_identity,
                listening.cache_identity,
                reduced.cache_identity,
            }
        )
        == 4
    )
    assert idle.cache_identity.requested_state == "idle"
    assert missing.cache_identity.resolved_state == "idle"
    assert listening.cache_identity.assets != reduced.cache_identity.assets
    assert reduced.cache_identity.reduced_motion is True


def test_loader_result_must_attest_exact_record_and_digest() -> None:
    graph = _graph()

    def wrong_loader(
        identity: PersonaVisualIdentity,
        record: PersonaVisualAssetRecord,
        selected_frame: int,
    ) -> PersonaVisualAsset:
        del identity, selected_frame
        result = RecordingLoader()(graph.identity, record, 0)
        return replace(
            result,
            metadata=replace(result.metadata, asset_key="different"),
        )

    result = resolve_persona_visual(
        graph, "idle", asset_loader=wrong_loader, portrait=_portrait()
    )

    assert result.source == "persona_portrait"
    assert result.reason == "persona_visual_idle_unavailable"


def test_loader_exception_details_and_raster_bytes_are_absent_from_repr() -> None:
    private_marker = "/Users/example/.config/private/idle.png"

    def leaking_loader(
        identity: PersonaVisualIdentity,
        record: PersonaVisualAssetRecord,
        selected_frame: int,
    ) -> PersonaVisualAsset:
        del identity, record, selected_frame
        raise RuntimeError(private_marker)

    failed = resolve_persona_visual(
        _graph(), "idle", asset_loader=leaking_loader, portrait=_portrait()
    )
    healthy = resolve_persona_visual(_graph(), "idle", asset_loader=RecordingLoader())

    assert private_marker not in repr(failed)
    assert "portrait-bytes" not in repr(failed)
    assert "bytes:idle" not in repr(healthy)


def test_public_results_reasons_and_cache_identity_are_path_free_and_immutable() -> (
    None
):
    result = resolve_persona_visual(
        _graph(), "listening", asset_loader=RecordingLoader()
    )

    for value in (result, result.cache_identity, *result.frames):
        assert hasattr(type(value), "__slots__")
        assert not any(
            "path" in field.name or "storage" in field.name for field in fields(value)
        )
    assert "/" not in repr(result)
    with pytest.raises(Exception):
        result.frames = ()  # type: ignore[misc]
