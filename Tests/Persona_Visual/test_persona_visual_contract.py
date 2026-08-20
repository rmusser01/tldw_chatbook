from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Persona_Visual import (
    ALLOWED_ASSET_EXTENSIONS,
    ALLOWED_ASSET_MIME_TYPES,
    ALLOWED_STATE_CATALOG_KINDS,
    ALLOWED_TRIGGER_SOURCES,
    MAX_ASSET_COUNT,
    MAX_ASSET_DIMENSION,
    MAX_ASSET_TOTAL_BYTES,
    REQUIRED_STATES,
    RESERVED_STATES,
    PersonaVisualManifestError,
    inspect_persona_visual_capability,
    resolve_manifest_state,
    validate_persona_visual_manifest,
)


KNOWN_ASSETS = {
    "asset-idle": (128, 128),
    "asset-wake": (128, 128),
    "asset-listen": (128, 128),
    "asset-think": (256, 128),
    "asset-speak": (128, 128),
    "asset-tool": (128, 128),
    "asset-approval": (128, 128),
    "asset-error": (128, 128),
    "asset-offline": (128, 128),
    "asset-custom": (128, 128),
}


PINNED_VALID_MANIFEST = {
    "manifest_version": 1,
    "renderer_type": "sprite_frames",
    "states": {
        "idle": {"animation_id": "idle"},
        "wake_armed": {"animation_id": "wake"},
        "listening": {"animation_id": "listen"},
        "thinking": {"animation_id": "think"},
        "speaking": {"animation_id": "speak"},
        "tool_running": {"animation_id": "tool"},
        "approval_needed": {"animation_id": "approval"},
        "error": {"animation_id": "error"},
        "offline": {"animation_id": "offline"},
        "tool.notes_search": {"animation_id": "custom"},
    },
    "animations": {
        "idle": {"asset_ids": ["asset-idle"], "frame_rate": 1, "loop": True},
        "wake": {"asset_ids": ["asset-wake"], "frame_rate": 8, "loop": True},
        "listen": {
            "asset_ids": ["asset-listen"],
            "frame_rate": 8,
            "alignment": {"x": 0.5, "y": 1},
            "loop": True,
        },
        "think": {
            "frames": [
                {
                    "asset_id": "asset-think",
                    "duration_ms": 120,
                    "region": {"x": 0, "y": 0, "width": 128, "height": 128},
                },
                {
                    "asset_id": "asset-think",
                    "duration_ms": 240,
                    "region": {"x": 128, "y": 0, "width": 128, "height": 128},
                },
            ],
            "frame_rate": 8,
            "loop": True,
            "preview_frame": 1,
            "preview_asset_id": "asset-think",
        },
        "speak": {"asset_ids": ["asset-speak"], "frame_rate": 12, "loop": True},
        "tool": {"asset_ids": ["asset-tool"], "frame_rate": 8, "loop": True},
        "approval": {
            "asset_ids": ["asset-approval"],
            "frame_rate": 1,
            "loop": False,
        },
        "error": {"asset_ids": ["asset-error"], "frame_rate": 1, "loop": False},
        "offline": {
            "asset_ids": ["asset-offline"],
            "frame_rate": 1,
            "loop": False,
        },
        "custom": {
            "asset_ids": ["asset-custom"],
            "frame_rate": 8,
            "loop": True,
        },
    },
    "fallbacks": {"tool.notes_search": ["tool_running", "thinking", "idle"]},
    "state_catalog": {
        "tool.notes_search": {
            "label": "Searching notes",
            "kind": "tool_variant",
            "description": "Shown while the notes search tool runs.",
            "tags": ["tool", "notes"],
        }
    },
    "authored_triggers": [
        {
            "id": "notes-search-tool",
            "source": "tool_name",
            "match": "notes.search",
            "state": "tool.notes_search",
            "duration_ms": 2400,
            "priority": 80,
        }
    ],
}


def _manifest() -> dict[str, object]:
    return deepcopy(PINNED_VALID_MANIFEST)


def _remove_fixture_custom_state(payload: dict[str, object]) -> None:
    payload["states"].pop("tool.notes_search", None)  # type: ignore[union-attr]
    payload["fallbacks"].pop("tool.notes_search", None)  # type: ignore[union-attr]
    payload["authored_triggers"] = []


def _assert_invalid(payload: object, *, known_assets: object = KNOWN_ASSETS) -> None:
    with pytest.raises(PersonaVisualManifestError) as exc_info:
        validate_persona_visual_manifest(payload, known_assets)  # type: ignore[arg-type]
    assert exc_info.value.category == "persona_visual_manifest_invalid"
    assert str(exc_info.value) == "persona_visual_manifest_invalid"


def test_pinned_capability_and_manifest_resolve_all_reserved_states() -> None:
    capability = inspect_persona_visual_capability("sprite_frames", 1)
    assert capability.supported is True
    assert capability.activatable is True
    assert capability.reason is None
    assert (
        capability.allowed_mime_types
        == ALLOWED_ASSET_MIME_TYPES
        == (
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
        )
    )
    assert (
        capability.allowed_extensions
        == ALLOWED_ASSET_EXTENSIONS
        == (
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".gif",
        )
    )
    assert capability.max_file_count == MAX_ASSET_COUNT == 256
    assert capability.max_total_bytes == MAX_ASSET_TOTAL_BYTES == 100 * 1024 * 1024
    assert capability.max_texture_width == MAX_ASSET_DIMENSION == 4096
    assert capability.max_texture_height == 4096

    manifest = validate_persona_visual_manifest(PINNED_VALID_MANIFEST, KNOWN_ASSETS)
    assert manifest.renderer_type == "sprite_frames"
    assert manifest.manifest_version == 1
    assert RESERVED_STATES == (
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
    assert REQUIRED_STATES == ("idle", "listening", "thinking", "speaking", "error")
    assert all(resolve_manifest_state(manifest, state) for state in RESERVED_STATES)
    assert all(resolve_manifest_state(manifest, state) for state in REQUIRED_STATES)


@pytest.mark.parametrize(
    ("renderer_type", "manifest_version"),
    [("live2d", 2), ("sprite_frames", 2), ("static_image", 1), ("unknown", 99)],
)
def test_unsupported_capability_is_stable_and_not_activatable(
    renderer_type: str,
    manifest_version: int,
) -> None:
    result = inspect_persona_visual_capability(renderer_type, manifest_version)
    assert result.supported is False
    assert result.activatable is False
    assert result.reason == "persona_visual_capability_unsupported"

    payload = {"renderer_type": renderer_type, "manifest_version": manifest_version}
    with pytest.raises(PersonaVisualManifestError) as exc_info:
        validate_persona_visual_manifest(payload, set(), activate=False)
    assert exc_info.value.category == "persona_visual_capability_unsupported"
    assert str(exc_info.value) == "persona_visual_capability_unsupported"


def test_contract_objects_and_nested_collections_are_immutable() -> None:
    manifest = validate_persona_visual_manifest(PINNED_VALID_MANIFEST, KNOWN_ASSETS)
    selection = resolve_manifest_state(manifest, "thinking", reduced_motion=True)
    assert selection is not None

    with pytest.raises(FrozenInstanceError):
        manifest.renderer_type = "live2d"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.animations["idle"] = manifest.animations["think"]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        selection.animate = True  # type: ignore[misc]
    assert isinstance(manifest.animations["think"].frames, tuple)
    assert hasattr(type(manifest), "__slots__")


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"renderer_type":"sprite_frames","renderer_type":"live2d","manifest_version":1}',
        '{"renderer_type":"sprite_frames","manifest_version":1,"states":{"idle":{"animation_id":"a","animation_id":"b"}}}',
        '{"renderer_type":"sprite_frames","manifest_version":1,"frame_rate":NaN}',
        '{"renderer_type":"sprite_frames","manifest_version":1,"frame_rate":Infinity}',
    ],
)
def test_json_input_rejects_non_objects_duplicates_and_nonstandard_numbers(
    payload: str,
) -> None:
    _assert_invalid(payload)


def test_json_bytes_are_parsed_without_mutating_the_input() -> None:
    encoded = json.dumps(PINNED_VALID_MANIFEST).encode("utf-8")
    manifest = validate_persona_visual_manifest(encoded, KNOWN_ASSETS)
    assert manifest.renderer_type == "sprite_frames"
    assert encoded == json.dumps(PINNED_VALID_MANIFEST).encode("utf-8")


def test_deeply_nested_json_returns_the_fixed_manifest_error() -> None:
    nested_json = '{"nested":' * 2_000 + "null" + "}" * 2_000

    _assert_invalid(nested_json)


@pytest.mark.parametrize(
    "state_id",
    [
        "tool.search",
        "reaction:happy",
        "live_variant-1",
        "mcp_runtime:ready",
        "mood_calm",
    ],
)
def test_safe_custom_state_grammar_is_accepted(state_id: str) -> None:
    payload = _manifest()
    _remove_fixture_custom_state(payload)
    payload["state_catalog"] = {
        state_id: {"label": "Safe state", "kind": "pack_private"}
    }
    payload["states"][state_id] = {"animation_id": "custom"}  # type: ignore[index]
    manifest = validate_persona_visual_manifest(payload, KNOWN_ASSETS)
    assert resolve_manifest_state(manifest, state_id).resolved_state == state_id  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "state_id",
    [
        "Tool.Search",
        "tool/search",
        "1tool",
        "x" * 97,
        "http:avatar",
        "https:avatar",
        "file:avatar",
        "env:avatar",
        "tool.api-key",
        "tool.client_secret",
        "private-key.preview",
    ],
)
def test_invalid_or_unsafe_custom_state_ids_are_rejected(state_id: str) -> None:
    payload = _manifest()
    payload["state_catalog"] = {
        state_id: {"label": "Unsafe state", "kind": "pack_private"}
    }
    _assert_invalid(payload)


@pytest.mark.parametrize("reserved_state", RESERVED_STATES)
def test_reserved_names_cannot_be_redeclared_in_state_catalog(
    reserved_state: str,
) -> None:
    payload = _manifest()
    payload["state_catalog"] = {
        reserved_state: {"label": "Collision", "kind": "reaction"}
    }
    _assert_invalid(payload)


def test_all_pinned_catalog_kinds_and_exact_text_bounds_are_accepted() -> None:
    payload = _manifest()
    _remove_fixture_custom_state(payload)
    payload["state_catalog"] = {
        f"custom.{index}": {
            "label": "L" * 80,
            "kind": kind,
            "description": "D" * 280,
            "tags": ["T" * 32] * 16,
        }
        for index, kind in enumerate(ALLOWED_STATE_CATALOG_KINDS)
    }
    manifest = validate_persona_visual_manifest(payload, KNOWN_ASSETS)
    assert {entry.kind for entry in manifest.state_catalog.values()} == set(
        ALLOWED_STATE_CATALOG_KINDS
    )


@pytest.mark.parametrize(
    "entry",
    [
        {"label": "", "kind": "reaction"},
        {"label": "L" * 81, "kind": "reaction"},
        {"label": "Bad\x7fLabel", "kind": "reaction"},
        {"label": "Good", "kind": "unknown"},
        {"label": "Good", "kind": "reaction", "description": "D" * 281},
        {"label": "Good", "kind": "reaction", "description": "Bad\ntext"},
        {"label": "Good", "kind": "reaction", "tags": ["tag"] * 17},
        {"label": "Good", "kind": "reaction", "tags": ["T" * 33]},
        {"label": "Good", "kind": "reaction", "tags": ["Bad\ttag"]},
    ],
)
def test_catalog_kinds_text_and_tag_bounds_are_strict(entry: dict[str, object]) -> None:
    payload = _manifest()
    payload["state_catalog"] = {"reaction.boundary": entry}
    _assert_invalid(payload)


def test_catalog_custom_state_limit_is_exact() -> None:
    payload = _manifest()
    _remove_fixture_custom_state(payload)
    payload["state_catalog"] = {
        f"custom.s{index}": {"label": f"State {index}", "kind": "pack_private"}
        for index in range(256)
    }
    validate_persona_visual_manifest(payload, KNOWN_ASSETS)
    payload["state_catalog"]["custom.overflow"] = {  # type: ignore[index]
        "label": "Overflow",
        "kind": "pack_private",
    }
    _assert_invalid(payload)


def test_legacy_asset_ids_normalize_to_frames() -> None:
    manifest = validate_persona_visual_manifest(PINNED_VALID_MANIFEST, KNOWN_ASSETS)
    animation = manifest.animations["idle"]
    assert [frame.asset_id for frame in animation.frames] == ["asset-idle"]
    assert animation.frame_rate == 1
    assert animation.loop is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame_rate", 0),
        ("frame_rate", 61),
        ("frame_rate", True),
        ("alignment", {"x": -0.01, "y": 0.5}),
        ("alignment", {"x": 0.5, "y": 1.01}),
        ("alignment", {"x": float("nan"), "y": 0.5}),
        ("loop", "yes"),
    ],
)
def test_animation_rate_alignment_and_loop_are_bounded(
    field: str, value: object
) -> None:
    payload = _manifest()
    payload["animations"]["idle"][field] = value  # type: ignore[index]
    _assert_invalid(payload)


@pytest.mark.parametrize(
    "frame",
    [
        {"asset_id": "missing"},
        {"asset_id": "asset-think", "duration_ms": 15},
        {"asset_id": "asset-think", "duration_ms": 30001},
        {"asset_id": "asset-think", "duration_ms": True},
        {
            "asset_id": "asset-think",
            "region": {"x": -1, "y": 0, "width": 1, "height": 1},
        },
        {
            "asset_id": "asset-think",
            "region": {"x": 0, "y": 0, "width": 0, "height": 1},
        },
        {
            "asset_id": "asset-think",
            "region": {"x": 200, "y": 0, "width": 128, "height": 128},
        },
    ],
)
def test_frame_assets_duration_and_regions_are_validated(
    frame: dict[str, object],
) -> None:
    payload = _manifest()
    payload["animations"]["thinking"] = {"frames": [frame], "frame_rate": 8}  # type: ignore[index]
    _assert_invalid(payload)


def test_frame_region_is_accepted_when_dimensions_are_not_known() -> None:
    payload = _manifest()
    payload["animations"]["thinking"] = {  # type: ignore[index]
        "frames": [
            {
                "asset_id": "asset-think",
                "region": {"x": 200, "y": 0, "width": 128, "height": 128},
            }
        ]
    }
    manifest = validate_persona_visual_manifest(payload, set(KNOWN_ASSETS))
    assert manifest.animations["thinking"].frames[0].region.x == 200  # type: ignore[union-attr]


def test_frame_count_limit_is_exact() -> None:
    payload = _manifest()
    payload["animations"]["thinking"] = {  # type: ignore[index]
        "frames": [{"asset_id": "asset-think"}] * 240
    }
    validate_persona_visual_manifest(payload, KNOWN_ASSETS)
    payload["animations"]["thinking"]["frames"].append(  # type: ignore[index,union-attr]
        {"asset_id": "asset-think"}
    )
    _assert_invalid(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("preview_frame", -1),
        ("preview_frame", 2),
        ("preview_frame", True),
        ("preview_asset_id", "asset-idle"),
    ],
)
def test_preview_selection_fields_must_reference_animation_frames(
    field: str,
    value: object,
) -> None:
    payload = _manifest()
    payload["animations"]["think"][field] = value  # type: ignore[index]
    _assert_invalid(payload)


def test_static_and_reduced_motion_selection_use_preview_precedence() -> None:
    manifest = validate_persona_visual_manifest(PINNED_VALID_MANIFEST, KNOWN_ASSETS)
    selection = resolve_manifest_state(manifest, "thinking", reduced_motion=True)
    assert selection is not None
    assert selection.requested_state == "thinking"
    assert selection.resolved_state == "thinking"
    assert selection.animation_id == "think"
    assert selection.animate is False
    assert selection.static.frame_index == 1
    assert selection.static.frame.region.x == 128  # type: ignore[union-attr]
    assert selection.static.reason == "preview_frame"

    payload = _manifest()
    payload["animations"]["think"].pop("preview_frame")  # type: ignore[index]
    payload["animations"]["think"]["frames"][0]["asset_id"] = "asset-idle"  # type: ignore[index]
    payload["animations"]["think"]["preview_asset_id"] = "asset-think"  # type: ignore[index]
    manifest = validate_persona_visual_manifest(payload, KNOWN_ASSETS)
    selection = resolve_manifest_state(manifest, "thinking", reduced_motion=False)
    assert selection is not None
    assert selection.animate is True
    assert selection.static.frame_index == 1
    assert selection.static.reason == "preview_asset_id"


def test_fallback_resolution_is_recursive_then_uses_idle() -> None:
    payload = _manifest()
    del payload["states"]["wake_armed"]  # type: ignore[index]
    payload["fallbacks"]["wake_armed"] = ["tool_running"]  # type: ignore[index]
    manifest = validate_persona_visual_manifest(payload, KNOWN_ASSETS)
    selection = resolve_manifest_state(manifest, "wake_armed")
    assert selection is not None
    assert selection.resolved_state == "tool_running"

    selection = resolve_manifest_state(manifest, "undeclared.runtime.state")
    assert selection is not None
    assert selection.resolved_state == "idle"


def test_fallback_depth_limit_accepts_eight_nodes_and_rejects_nine() -> None:
    accepted = _manifest()
    accepted["fallbacks"] = {
        "idle": ["wake_armed"],
        "wake_armed": ["listening"],
        "listening": ["thinking"],
        "thinking": ["speaking"],
        "speaking": ["tool_running"],
        "tool_running": ["approval_needed"],
        "approval_needed": ["error"],
    }
    validate_persona_visual_manifest(accepted, KNOWN_ASSETS)

    accepted["fallbacks"]["error"] = ["offline"]  # type: ignore[index]
    _assert_invalid(accepted)


def test_fallback_depth_cannot_be_bypassed_by_a_previsited_shared_descendant() -> None:
    payload = _manifest()
    payload["fallbacks"] = {
        "error": ["offline"],
        "idle": ["wake_armed"],
        "wake_armed": ["listening"],
        "listening": ["thinking"],
        "thinking": ["speaking"],
        "speaking": ["tool_running"],
        "tool_running": ["approval_needed"],
        "approval_needed": ["error"],
    }

    _assert_invalid(payload)


@pytest.mark.parametrize(
    "fallbacks",
    [
        {"thinking": ["missing"]},
        {"thinking": ["tool_running"], "tool_running": ["thinking"]},
        {"missing": ["idle"]},
    ],
)
def test_fallback_targets_sources_and_cycles_are_validated(
    fallbacks: dict[str, list[str]],
) -> None:
    payload = _manifest()
    payload["fallbacks"] = fallbacks
    _assert_invalid(payload)


@pytest.mark.parametrize("source", ALLOWED_TRIGGER_SOURCES)
def test_all_exact_trigger_sources_are_accepted(source: str) -> None:
    payload = _manifest()
    payload["authored_triggers"] = [
        {
            "id": f"trigger-{source}",
            "source": source,
            "match": "notes.search",
            "state": "tool.notes_search",
            "duration_ms": 100,
            "priority": 100,
        }
    ]
    manifest = validate_persona_visual_manifest(payload, KNOWN_ASSETS)
    assert manifest.triggers[0].source == source


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", ""),
        ("id", " "),
        ("source", "tool"),
        ("match", ""),
        ("match", " "),
        ("state", "missing"),
        ("duration_ms", 99),
        ("duration_ms", 30001),
        ("duration_ms", True),
        ("priority", -1),
        ("priority", 101),
        ("priority", True),
    ],
)
def test_trigger_fields_are_exact_and_bounded(field: str, value: object) -> None:
    payload = _manifest()
    payload["authored_triggers"][0][field] = value  # type: ignore[index]
    _assert_invalid(payload)


def test_trigger_count_limit_is_exact() -> None:
    payload = _manifest()
    payload["authored_triggers"] = [
        {
            "id": f"trigger-{index}",
            "source": "live_state",
            "match": "thinking",
            "state": "thinking",
            "duration_ms": 100,
            "priority": 0,
        }
        for index in range(512)
    ]
    validate_persona_visual_manifest(payload, KNOWN_ASSETS)
    payload["authored_triggers"].append(deepcopy(payload["authored_triggers"][0]))  # type: ignore[union-attr,index]
    _assert_invalid(payload)


def test_required_states_must_resolve_only_for_activation() -> None:
    payload = {
        "renderer_type": "sprite_frames",
        "manifest_version": 1,
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {"idle": {"asset_ids": ["asset-idle"]}},
    }
    manifest = validate_persona_visual_manifest(payload, {"asset-idle"}, activate=False)
    assert resolve_manifest_state(manifest, "idle") is not None
    _assert_invalid(payload, known_assets={"asset-idle"})


@pytest.mark.parametrize(
    "mutation",
    [
        ("root", "future_field", True),
        ("animation", "future_field", True),
        ("frame", "future_field", True),
        ("trigger", "future_field", True),
        ("catalog", "future_field", True),
    ],
)
def test_unapproved_manifest_fields_are_rejected(
    mutation: tuple[str, str, object],
) -> None:
    payload = _manifest()
    target, field, value = mutation
    if target == "root":
        payload[field] = value
    elif target == "animation":
        payload["animations"]["idle"][field] = value  # type: ignore[index]
    elif target == "frame":
        payload["animations"]["think"]["frames"][0][field] = value  # type: ignore[index]
    elif target == "trigger":
        payload["authored_triggers"][0][field] = value  # type: ignore[index]
    else:
        payload["state_catalog"]["tool.notes_search"][field] = value  # type: ignore[index]
    _assert_invalid(payload)


def test_errors_never_include_user_asset_ids_or_exception_details() -> None:
    private_asset = "/Users/alice/.secrets/api_key.png"
    payload = _manifest()
    payload["animations"]["idle"]["asset_ids"] = [private_asset]  # type: ignore[index]
    with pytest.raises(PersonaVisualManifestError) as exc_info:
        validate_persona_visual_manifest(payload, KNOWN_ASSETS)
    assert str(exc_info.value) == "persona_visual_manifest_invalid"
    assert private_asset not in str(exc_info.value)


@pytest.mark.parametrize("case", ["fallback", "trigger", "known_assets"])
def test_malformed_nested_values_still_return_the_fixed_error(case: str) -> None:
    payload = _manifest()
    known_assets: object = KNOWN_ASSETS
    if case == "fallback":
        payload["fallbacks"] = {"thinking": [["idle"]]}
    elif case == "trigger":
        payload["authored_triggers"][0]["state"] = ["thinking"]  # type: ignore[index]
    else:
        known_assets = [["asset-idle"]]
    _assert_invalid(payload, known_assets=known_assets)
