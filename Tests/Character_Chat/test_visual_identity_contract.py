"""Frozen expression and manifest contracts for local Visual Identity packs."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import inspect
import json
import math

import pytest

import tldw_chatbook.Character_Chat.visual_identity as visual_identity
from tldw_chatbook.Character_Chat.visual_identity import (
    CANONICAL_EXPRESSION_SLOTS,
    CUSTOM_EXPRESSION_PREFIX,
    EXPRESSION_ALIASES,
    SAMIRA_DEFAULT_EXPRESSION_KEY,
    SAMIRA_EXPRESSION_KEYS,
    SAMIRA_LICENSE,
    SAMIRA_MANIFEST_SCHEMA_ID,
    SAMIRA_PACK_ID,
    SAMIRA_REACTION_LABELS,
    SAMIRA_SERVER_COMMIT,
    compute_pack_content_sha256,
    display_label_for_expression_key,
    is_custom_expression_key,
    normalize_expression_filename,
    normalize_expression_key,
    validate_visual_identity_manifest,
)


SERVER_CANONICAL_SLOTS = (
    "neutral",
    "happy",
    "excited",
    "sad",
    "angry",
    "thinking",
    "confused",
    "surprised",
)

SERVER_ALIAS_FIXTURES = {
    "default": "neutral",
    "normal": "neutral",
    "calm": "neutral",
    "joy": "happy",
    "joyful": "happy",
    "cheerful": "happy",
    "hype": "excited",
    "thrilled": "excited",
    "upset": "sad",
    "sorrowful": "sad",
    "mad": "angry",
    "annoyed": "angry",
    "furious": "angry",
    "anger": "angry",
    "thoughtful": "thinking",
    "pondering": "thinking",
    "unsure": "confused",
    "puzzled": "confused",
    "shocked": "surprised",
    "astonished": "surprised",
}

EXPECTED_SAMIRA_LABELS = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "neutral",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "thinking",
    "speaking",
    "error",
)

EXPECTED_SAMIRA_KEYS = {
    "admiration": "custom:admiration",
    "amusement": "custom:amusement",
    "anger": "angry",
    "annoyance": "custom:annoyance",
    "approval": "custom:approval",
    "caring": "custom:caring",
    "confusion": "confused",
    "curiosity": "custom:curiosity",
    "desire": "custom:desire",
    "disappointment": "custom:disappointment",
    "disapproval": "custom:disapproval",
    "disgust": "custom:disgust",
    "embarrassment": "custom:embarrassment",
    "excitement": "excited",
    "fear": "custom:fear",
    "gratitude": "custom:gratitude",
    "grief": "custom:grief",
    "joy": "happy",
    "love": "custom:love",
    "nervousness": "custom:nervousness",
    "neutral": "neutral",
    "optimism": "custom:optimism",
    "pride": "custom:pride",
    "realization": "custom:realization",
    "relief": "custom:relief",
    "remorse": "custom:remorse",
    "sadness": "sad",
    "surprise": "surprised",
    "thinking": "thinking",
    "speaking": "custom:speaking",
    "error": "custom:error",
}

VALID_SAMIRA_DIRECTORY_BYTES = 18 * 1024 * 1024
MAX_EXPRESSION_ASSET_BYTES = 25 * 1024 * 1024
MAX_EXPRESSION_IMAGE_DIMENSION = 4096
MAX_EXPRESSION_FRAME_COUNT = 512
MAX_EXPRESSION_PACK_ASSETS = 128
MAX_EXPRESSION_TOTAL_BYTES = 256 * 1024 * 1024
MAX_EXPRESSION_ASSET_DECODED_PIXELS = MAX_EXPRESSION_IMAGE_DIMENSION**2 * 4
MAX_EXPRESSION_PACK_DECODED_PIXELS = MAX_EXPRESSION_IMAGE_DIMENSION**2 * 16


def _asset(
    label: str,
    expression_key: str,
    *,
    byte_count: int = 100,
    sha256: str = "a" * 64,
    storage_relpath: str | None = None,
    is_animated: bool = False,
    frame_count: int = 1,
    duration_ms: int | None = None,
) -> dict[str, object]:
    return {
        "expression_key": expression_key,
        "original_label": label,
        "display_label": label.replace("_", " ").title(),
        "storage_relpath": storage_relpath
        or f"characters/samira/expressions/{label}.webp",
        "content_type": "image/webp",
        "bytes": byte_count,
        "width": 1024,
        "height": 1024,
        "sha256": sha256,
        "is_animated": is_animated,
        "frame_count": frame_count,
        "duration_ms": duration_ms,
        "generation_notes": "excluded from the content digest",
    }


def _manifest_data(
    assets: list[dict[str, object]],
    *,
    pack_id: str = "user.example.pack",
    license_id: str = "MIT",
    default_expression_key: str | None = None,
) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_id": SAMIRA_MANIFEST_SCHEMA_ID,
        "pack_id": pack_id,
        "title": "Example pack",
        "license": license_id,
        "default_expression_key": default_expression_key
        or str(assets[0]["expression_key"]),
        "source_server_commit": SAMIRA_SERVER_COMMIT,
        "generation_provenance": {"tool": "excluded"},
        "assets": assets,
    }
    data["pack_content_sha256"] = compute_pack_content_sha256(data)
    return data


def _samira_manifest_data() -> dict[str, object]:
    return _manifest_data(
        [
            _asset(label, EXPECTED_SAMIRA_KEYS[label])
            for label in EXPECTED_SAMIRA_LABELS
        ],
        pack_id=SAMIRA_PACK_ID,
        license_id=SAMIRA_LICENSE,
        default_expression_key=SAMIRA_DEFAULT_EXPRESSION_KEY,
    )


def _many_assets(count: int, *, byte_count: int = 1) -> list[dict[str, object]]:
    return [
        _asset(
            f"slot_{index:03d}",
            f"custom:slot_{index:03d}",
            byte_count=byte_count,
            sha256=f"{index:064x}",
            storage_relpath=f"packs/example/slot_{index:03d}.webp",
        )
        for index in range(count)
    ]


def test_server_expression_constants_are_frozen() -> None:
    assert CANONICAL_EXPRESSION_SLOTS == SERVER_CANONICAL_SLOTS
    assert CUSTOM_EXPRESSION_PREFIX == "custom:"
    assert EXPRESSION_ALIASES == SERVER_ALIAS_FIXTURES


def test_pinned_normalization_block_has_explicit_source_markers() -> None:
    source = inspect.getsource(visual_identity)
    start = source.index(
        "# Begin pinned server normalization block (byte-for-byte from):"
    )
    canonical = source.index("CANONICAL_EXPRESSION_SLOTS =")
    end = source.index("# End pinned server normalization block.")
    samira = source.index("SAMIRA_REACTION_LABELS =")

    assert start < canonical < end < samira


@pytest.mark.parametrize("value", SERVER_CANONICAL_SLOTS)
def test_canonical_expression_key_parity(value: str) -> None:
    assert normalize_expression_key(value) == value


@pytest.mark.parametrize(("value", "expected"), SERVER_ALIAS_FIXTURES.items())
def test_every_server_alias_has_frozen_parity(value: str, expected: str) -> None:
    assert normalize_expression_key(value) == expected
    assert normalize_expression_filename(f"{value}.WEBP") == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" custom:Quiet Focus ", "custom:quiet_focus"),
        ("Quiet focus?!", "custom:quiet_focus"),
        ("custom:---", None),
        ("  ", None),
        (None, None),
        (42, None),
        (b"joy", None),
        (object(), None),
    ],
)
def test_expression_key_custom_punctuation_empty_and_non_string_parity(
    value: object, expected: str | None
) -> None:
    assert normalize_expression_key(value) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (r"C:\portraits\JOY.PNG", "happy"),
        ("/tmp/My Weird.Face.webp", "custom:my_weird_face"),
        ("archive.tar.gz", "custom:archive_tar"),
        ("crème brûlée.webp", "custom:cr_me_br_l_e"),
        (".png", None),
        ("", None),
        (None, None),
        (7, None),
    ],
)
def test_expression_filename_path_punctuation_empty_and_non_string_parity(
    filename: object, expected: str | None
) -> None:
    assert normalize_expression_filename(filename) == expected  # type: ignore[arg-type]


def test_generic_normalization_does_not_gain_samira_only_aliases() -> None:
    assert normalize_expression_key("excitement") == "custom:excitement"
    assert normalize_expression_key("sadness") == "custom:sadness"
    assert normalize_expression_key("confusion") == "custom:confusion"
    assert normalize_expression_key("surprise") == "custom:surprise"
    assert normalize_expression_filename("excitement.webp") == "custom:excitement"
    assert normalize_expression_filename("sadness.webp") == "custom:sadness"
    assert normalize_expression_filename("confusion.webp") == "custom:confusion"
    assert normalize_expression_filename("surprise.webp") == "custom:surprise"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("custom:quiet_focus", "Quiet Focus"),
        ("quiet-focus", "Quiet Focus"),
        ("joy", "Happy"),
        ("confused", "Confused"),
        ("", ""),
        (None, ""),
    ],
)
def test_display_label_parity(value: object, expected: str) -> None:
    assert display_label_for_expression_key(value) == expected  # type: ignore[arg-type]


def test_custom_key_detection_parity() -> None:
    assert is_custom_expression_key("quiet focus") is True
    assert is_custom_expression_key("joy") is False
    assert is_custom_expression_key(1) is False  # type: ignore[arg-type]


def test_samira_inventory_mapping_and_contract_constants_are_exact() -> None:
    assert SAMIRA_REACTION_LABELS == EXPECTED_SAMIRA_LABELS
    assert SAMIRA_EXPRESSION_KEYS == EXPECTED_SAMIRA_KEYS
    assert SAMIRA_PACK_ID == "tldw.builtin.samira.reactions"
    assert SAMIRA_MANIFEST_SCHEMA_ID == "tldw.visual_identity_pack/v1"
    assert SAMIRA_LICENSE == "AGPL-3.0-or-later"
    assert SAMIRA_DEFAULT_EXPRESSION_KEY == "neutral"
    assert SAMIRA_SERVER_COMMIT == "385afa951922c8a9dc2002c675bb6cad65e4ac23"


def test_samira_mapping_is_one_explicit_31_entry_literal() -> None:
    module = ast.parse(inspect.getsource(visual_identity))
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SAMIRA_EXPRESSION_KEYS"
            for target in node.targets
        )
    )

    assert isinstance(assignment.value, ast.Dict)
    assert len(assignment.value.keys) == len(EXPECTED_SAMIRA_KEYS) == 31


def test_content_digest_uses_the_frozen_literal_payload() -> None:
    data = _manifest_data(
        [
            _asset("neutral", "neutral", byte_count=202, sha256="b" * 64),
            _asset(
                "admiration",
                "custom:admiration",
                byte_count=101,
                sha256="a" * 64,
            ),
        ],
        pack_id=SAMIRA_PACK_ID,
        license_id=SAMIRA_LICENSE,
        default_expression_key="neutral",
    )
    data["pack_content_sha256"] = "excluded"
    data["generation_notes"] = {"ignored": math.nan}

    expected_payload = (
        '{"assets":[{"bytes":101,"content_type":"image/webp",'
        '"expression_key":"custom:admiration","height":1024,'
        '"original_label":"admiration","relative_filename":'
        '"characters/samira/expressions/admiration.webp","sha256":'
        '"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"width":1024},{"bytes":202,"content_type":"image/webp",'
        '"expression_key":"neutral","height":1024,"original_label":"neutral",'
        '"relative_filename":"characters/samira/expressions/neutral.webp",'
        '"sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        '"width":1024}],"default_expression_key":"neutral","license":'
        '"AGPL-3.0-or-later","pack_id":"tldw.builtin.samira.reactions",'
        '"schema_id":"tldw.visual_identity_pack/v1"}'
    )

    assert visual_identity._canonical_pack_content_json(data) == expected_payload
    assert (
        compute_pack_content_sha256(data)
        == "413003b098b0992b2aeed6119d1d3808597af324280b3dea982efa8e001817c1"
    )


def test_content_digest_rejects_nan_in_content_fields() -> None:
    data = _manifest_data([_asset("neutral", "neutral")])
    data["assets"][0]["bytes"] = math.nan  # type: ignore[index]

    with pytest.raises(ValueError):
        compute_pack_content_sha256(data)


def test_general_manifest_accepts_a_validated_user_subset() -> None:
    data = _manifest_data(
        [
            _asset(
                "neutral",
                "neutral",
                storage_relpath="packs/example/v2/neutral.webp",
            ),
            _asset(
                "thinking",
                "thinking",
                storage_relpath="packs/example/v2/thinking.webp",
            ),
        ]
    )

    manifest = validate_visual_identity_manifest(data)

    assert manifest.pack_id == "user.example.pack"
    assert tuple(asset.original_label for asset in manifest.assets) == (
        "neutral",
        "thinking",
    )
    assert manifest.assets[0].storage_relpath == "packs/example/v2/neutral.webp"
    assert manifest.assets[0].relative_filename == "packs/example/v2/neutral.webp"
    assert not hasattr(manifest, "__dict__")
    assert not hasattr(manifest.assets[0], "__dict__")
    with pytest.raises(FrozenInstanceError):
        manifest.pack_id = "changed"  # type: ignore[misc]


def test_general_manifest_accepts_exact_pinned_server_boundaries() -> None:
    static_asset = _asset(
        "neutral",
        "neutral",
        byte_count=MAX_EXPRESSION_ASSET_BYTES,
    )
    static_asset["width"] = MAX_EXPRESSION_IMAGE_DIMENSION
    static_asset["height"] = MAX_EXPRESSION_IMAGE_DIMENSION
    animated_asset = _asset(
        "animated",
        "custom:animated",
        byte_count=1,
        sha256="b" * 64,
        is_animated=True,
        frame_count=MAX_EXPRESSION_FRAME_COUNT,
        duration_ms=1,
    )
    animated_asset["width"] = 1
    animated_asset["height"] = 1
    data = _manifest_data([static_asset, animated_asset])

    manifest = validate_visual_identity_manifest(data)

    assert manifest.assets[0].bytes == MAX_EXPRESSION_ASSET_BYTES
    assert manifest.assets[1].frame_count == MAX_EXPRESSION_FRAME_COUNT


def test_general_limit_constants_match_the_pinned_server_contract() -> None:
    assert visual_identity.MAX_EXPRESSION_ASSET_BYTES == MAX_EXPRESSION_ASSET_BYTES
    assert (
        visual_identity.MAX_EXPRESSION_IMAGE_DIMENSION == MAX_EXPRESSION_IMAGE_DIMENSION
    )
    assert visual_identity.MAX_EXPRESSION_FRAME_COUNT == MAX_EXPRESSION_FRAME_COUNT
    assert visual_identity.MAX_EXPRESSION_PACK_ASSETS == MAX_EXPRESSION_PACK_ASSETS
    assert visual_identity.MAX_EXPRESSION_TOTAL_BYTES == MAX_EXPRESSION_TOTAL_BYTES
    assert (
        visual_identity.MAX_EXPRESSION_ASSET_DECODED_PIXELS
        == MAX_EXPRESSION_ASSET_DECODED_PIXELS
    )
    assert (
        visual_identity.MAX_EXPRESSION_PACK_DECODED_PIXELS
        == MAX_EXPRESSION_PACK_DECODED_PIXELS
    )


def _full_dimension_animation(label: str, frame_count: int = 4):
    asset = _asset(
        label,
        f"custom:{label}",
        is_animated=True,
        frame_count=frame_count,
        duration_ms=1,
    )
    asset["width"] = MAX_EXPRESSION_IMAGE_DIMENSION
    asset["height"] = MAX_EXPRESSION_IMAGE_DIMENSION
    return asset


def test_general_manifest_accepts_exact_decoded_pixel_boundaries() -> None:
    assets = [_full_dimension_animation(f"animated_{index}") for index in range(4)]
    data = _manifest_data(assets)

    manifest = validate_visual_identity_manifest(data)

    assert len(manifest.assets) == 4
    assert (
        manifest.assets[0].width
        * manifest.assets[0].height
        * manifest.assets[0].frame_count
        == MAX_EXPRESSION_ASSET_DECODED_PIXELS
    )


def test_general_manifest_rejects_per_asset_decoded_pixel_work_over_limit() -> None:
    data = _manifest_data([_full_dimension_animation("animated", frame_count=5)])

    with pytest.raises(ValueError, match="^visual_identity_budget_exceeded$"):
        validate_visual_identity_manifest(data)


def test_general_manifest_rejects_cumulative_decoded_pixel_work_over_limit() -> None:
    assets = [_full_dimension_animation(f"animated_{index}") for index in range(5)]
    data = _manifest_data(assets)

    with pytest.raises(ValueError, match="^visual_identity_budget_exceeded$"):
        validate_visual_identity_manifest(data)


@pytest.mark.parametrize(
    "field",
    ["bytes", "width", "height", "frame_count"],
)
def test_general_manifest_rejects_values_over_pinned_server_boundaries(
    field: str,
) -> None:
    asset = _asset("animated", "custom:animated")
    if field == "bytes":
        asset[field] = MAX_EXPRESSION_ASSET_BYTES + 1
    elif field in {"width", "height"}:
        asset[field] = MAX_EXPRESSION_IMAGE_DIMENSION + 1
    else:
        asset.update(
            is_animated=True,
            frame_count=MAX_EXPRESSION_FRAME_COUNT + 1,
            duration_ms=1,
        )
    data = _manifest_data([asset])

    with pytest.raises(ValueError, match="^visual_identity_budget_exceeded$"):
        validate_visual_identity_manifest(data)


def test_general_manifest_rejects_too_many_assets_before_io() -> None:
    data = _manifest_data(_many_assets(MAX_EXPRESSION_PACK_ASSETS + 1))

    with pytest.raises(ValueError, match="^visual_identity_budget_exceeded$"):
        validate_visual_identity_manifest(data)


def test_general_manifest_rejects_aggregate_bytes_over_pinned_server_boundary() -> None:
    assets = _many_assets(10, byte_count=MAX_EXPRESSION_ASSET_BYTES)
    assets.append(
        _asset(
            "remainder",
            "custom:remainder",
            byte_count=6 * 1024 * 1024 + 1,
            sha256="f" * 64,
            storage_relpath="packs/example/remainder.webp",
        )
    )
    assert sum(int(asset["bytes"]) for asset in assets) == (
        MAX_EXPRESSION_TOTAL_BYTES + 1
    )
    data = _manifest_data(assets)

    with pytest.raises(ValueError, match="^visual_identity_budget_exceeded$"):
        validate_visual_identity_manifest(data)


@pytest.mark.parametrize(
    ("mutation", "category"),
    [
        (lambda data: data.update(schema_id=""), "visual_identity_manifest_invalid"),
        (lambda data: data.update(pack_id=""), "visual_identity_manifest_invalid"),
        (
            lambda data: data.update(license="not a license!"),
            "visual_identity_manifest_invalid",
        ),
        (
            lambda data: data.update(default_expression_key="custom:missing"),
            "visual_identity_manifest_invalid",
        ),
        (
            lambda data: data["assets"][0].update(expression_key=""),
            "visual_identity_manifest_invalid",
        ),
        (
            lambda data: data["assets"][0].update(original_label=""),
            "visual_identity_manifest_invalid",
        ),
        (
            lambda data: data["assets"][0].update(content_type="text/plain"),
            "visual_identity_manifest_invalid",
        ),
        (
            lambda data: data["assets"][0].update(content_type="image/x-webp"),
            "visual_identity_manifest_invalid",
        ),
        (
            lambda data: data["assets"][0].update(bytes=0),
            "visual_identity_manifest_invalid",
        ),
        (
            lambda data: data["assets"][0].update(width=-1),
            "visual_identity_manifest_invalid",
        ),
        (
            lambda data: data["assets"][0].update(height=True),
            "visual_identity_manifest_invalid",
        ),
        (
            lambda data: data["assets"][0].update(sha256="A" * 64),
            "visual_identity_manifest_invalid",
        ),
        (
            lambda data: data["assets"][0].update(is_animated=False, frame_count=2),
            "visual_identity_manifest_invalid",
        ),
        (
            lambda data: data["assets"][0].update(
                is_animated=True, frame_count=2, duration_ms=None
            ),
            "visual_identity_manifest_invalid",
        ),
    ],
)
def test_general_manifest_rejects_invalid_shapes(mutation, category: str) -> None:
    data = _manifest_data([_asset("neutral", "neutral")])
    mutation(data)
    data["pack_content_sha256"] = compute_pack_content_sha256(data)

    with pytest.raises(ValueError, match=f"^{category}$"):
        validate_visual_identity_manifest(data)


@pytest.mark.parametrize("duplicate_field", ["expression_key", "original_label"])
def test_general_manifest_rejects_duplicate_keys_and_labels(
    duplicate_field: str,
) -> None:
    first = _asset("neutral", "neutral", sha256="a" * 64)
    second = _asset("thinking", "thinking", sha256="b" * 64)
    second[duplicate_field] = first[duplicate_field]
    data = _manifest_data([first, second])

    with pytest.raises(ValueError, match="^visual_identity_manifest_invalid$"):
        validate_visual_identity_manifest(data)


def test_general_manifest_rejects_digest_mismatch() -> None:
    data = _manifest_data([_asset("neutral", "neutral")])
    data["pack_content_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="^visual_identity_digest_mismatch$"):
        validate_visual_identity_manifest(data)


def test_general_manifest_rejects_lone_surrogate_without_raw_error() -> None:
    data = _manifest_data([_asset("neutral", "neutral")])
    data["assets"][0]["original_label"] = "\ud800"  # type: ignore[index]

    with pytest.raises(ValueError) as error:
        validate_visual_identity_manifest(data)

    assert str(error.value) == "visual_identity_manifest_invalid"
    assert error.value.__cause__ is None
    assert "Unicode" not in str(error.value)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "",
        ".",
        "/absolute.webp",
        "../escape.webp",
        "packs/../escape.webp",
        r"packs\escape.webp",
        "packs/escape\x00.webp",
        "packs//escape.webp",
        "packs/./escape.webp",
    ],
)
def test_general_manifest_rejects_every_unsafe_relative_filename(
    unsafe_path: str,
) -> None:
    asset = _asset("neutral", "neutral")
    asset["storage_relpath"] = unsafe_path
    data = _manifest_data([asset])

    with pytest.raises(ValueError, match="^visual_identity_manifest_invalid$"):
        validate_visual_identity_manifest(data)


def test_samira_manifest_requires_exact_inventory_mapping_and_contract() -> None:
    manifest = validate_visual_identity_manifest(
        _samira_manifest_data(),
        require_samira_bundle=True,
        directory_bytes=VALID_SAMIRA_DIRECTORY_BYTES,
    )

    assert (
        tuple(asset.original_label for asset in manifest.assets)
        == EXPECTED_SAMIRA_LABELS
    )
    assert {
        asset.original_label: asset.expression_key for asset in manifest.assets
    } == (EXPECTED_SAMIRA_KEYS)


@pytest.mark.parametrize(
    ("mutation", "category"),
    [
        (lambda data: data["assets"].pop(), "visual_identity_samira_contract_invalid"),
        (
            lambda data: data["assets"][0].update(expression_key="custom:wrong"),
            "visual_identity_samira_contract_invalid",
        ),
        (
            lambda data: data.update(default_expression_key="happy"),
            "visual_identity_samira_contract_invalid",
        ),
        (
            lambda data: data.update(schema_id="example/v1"),
            "visual_identity_manifest_invalid",
        ),
        (
            lambda data: data.update(pack_id="user.samira"),
            "visual_identity_samira_contract_invalid",
        ),
        (
            lambda data: data.update(license="MIT"),
            "visual_identity_samira_contract_invalid",
        ),
        (
            lambda data: data.update(source_server_commit="0" * 40),
            "visual_identity_samira_contract_invalid",
        ),
    ],
)
def test_samira_manifest_rejects_contract_drift(mutation, category: str) -> None:
    data = _samira_manifest_data()
    mutation(data)
    data["pack_content_sha256"] = compute_pack_content_sha256(data)

    with pytest.raises(ValueError, match=f"^{category}$"):
        validate_visual_identity_manifest(
            data,
            require_samira_bundle=True,
            directory_bytes=VALID_SAMIRA_DIRECTORY_BYTES,
        )


def test_samira_manifest_enforces_per_reaction_budget() -> None:
    data = _samira_manifest_data()
    data["assets"][0]["bytes"] = 1024 * 1024 + 1  # type: ignore[index]
    data["pack_content_sha256"] = compute_pack_content_sha256(data)

    with pytest.raises(ValueError, match="^visual_identity_budget_exceeded$"):
        validate_visual_identity_manifest(
            data,
            require_samira_bundle=True,
            directory_bytes=VALID_SAMIRA_DIRECTORY_BYTES,
        )


def test_samira_manifest_enforces_reaction_aggregate_budget() -> None:
    data = _samira_manifest_data()
    for asset in data["assets"]:  # type: ignore[union-attr]
        asset["bytes"] = 600_000
    data["pack_content_sha256"] = compute_pack_content_sha256(data)

    with pytest.raises(ValueError, match="^visual_identity_budget_exceeded$"):
        validate_visual_identity_manifest(
            data,
            require_samira_bundle=True,
            directory_bytes=VALID_SAMIRA_DIRECTORY_BYTES,
        )


def test_samira_manifest_enforces_supplied_directory_budget() -> None:
    data = _samira_manifest_data()

    with pytest.raises(ValueError, match="^visual_identity_budget_exceeded$"):
        validate_visual_identity_manifest(
            data,
            require_samira_bundle=True,
            directory_bytes=20 * 1024 * 1024 + 1,
        )


def test_reserved_samira_pack_id_is_strict_even_in_general_mode() -> None:
    data = _samira_manifest_data()
    data["assets"][0]["expression_key"] = "custom:wrong"  # type: ignore[index]
    data["pack_content_sha256"] = compute_pack_content_sha256(data)

    with pytest.raises(ValueError, match="^visual_identity_samira_contract_invalid$"):
        validate_visual_identity_manifest(
            data,
            directory_bytes=VALID_SAMIRA_DIRECTORY_BYTES,
        )


@pytest.mark.parametrize("require_samira_bundle", [False, True])
def test_bundled_samira_validation_requires_measured_directory_bytes(
    require_samira_bundle: bool,
) -> None:
    data = _samira_manifest_data()
    if not require_samira_bundle:
        data["pack_id"] = SAMIRA_PACK_ID
        data["pack_content_sha256"] = compute_pack_content_sha256(data)

    with pytest.raises(ValueError, match="^visual_identity_directory_bytes_required$"):
        validate_visual_identity_manifest(
            data,
            require_samira_bundle=require_samira_bundle,
        )


def test_parse_manifest_json_accepts_strict_utf8_object() -> None:
    data = _manifest_data([_asset("neutral", "neutral")])

    parsed = visual_identity.parse_visual_identity_manifest_json(
        json.dumps(data).encode("utf-8")
    )

    assert parsed.pack_id == "user.example.pack"


def test_parse_manifest_json_rejects_escaped_lone_surrogate_without_raw_error() -> None:
    data = _manifest_data([_asset("neutral", "neutral")])
    data["title"] = "\ud800"
    raw = json.dumps(data)
    assert r"\ud800" in raw

    with pytest.raises(ValueError) as error:
        visual_identity.parse_visual_identity_manifest_json(raw)

    assert str(error.value) == "visual_identity_manifest_invalid"
    assert error.value.__cause__ is None
    assert r"\ud800" not in str(error.value)


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_id":"first","schema_id":"second"}',
        '{"assets":[{"bytes":1,"bytes":2}]}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '["not", "an", "object"]',
        b"\xff",
        42,
    ],
)
def test_parse_manifest_json_rejects_ambiguous_or_invalid_json_without_content_leak(
    raw: object,
) -> None:
    with pytest.raises(ValueError) as error:
        visual_identity.parse_visual_identity_manifest_json(raw)  # type: ignore[arg-type]

    assert str(error.value) == "visual_identity_manifest_json_invalid"
    assert error.value.__cause__ is None
    assert repr(raw) not in str(error.value)
