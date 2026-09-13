"""Conversion lineage is public, bounded and bound to one output image."""

import json

import pytest

from tldw_chatbook.Character_Chat.artwork_attribution import (
    artwork_context,
    decode_artwork_attribution,
    encode_artwork_attribution,
)

NAMESPACE = "tldw/buddy_conversion"
DIGEST = "a" * 64


def lineage(**changes):
    return {
        "version": 1,
        "source_sha256": "b" * 64,
        "source_state": "idle",
        "source_asset_sha256": ["c" * 64, "d" * 64],
        "converted_at": "2026-09-07T12:30:00+00:00",
        "fallback": False,
        "output_sha256": DIGEST,
        **changes,
    }


def test_conversion_only_carrier_is_v2_and_preserves_exact_public_record():
    context = {NAMESPACE: lineage(), "persona_id": "private", "local_path": "/secret"}
    assert artwork_context(context, expected_sha256=DIGEST) == {NAMESPACE: lineage()}
    data = encode_artwork_attribution({}, {"neutral": (DIGEST, context)})
    assert decode_artwork_attribution(data, {"neutral": DIGEST}) == {
        "version": 2,
        "pack": None,
        "assets": {},
        "conversions": {"neutral": lineage()},
    }
    assert b"private" not in data and b"secret" not in data


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"version": 2},
        {"local_id": 1},
        {"source_sha256": "A" * 64},
        {"source_sha256": "a" * 63},
        {"source_state": ""},
        {"source_state": None},
        {"source_state": 1},
        {"source_state": "two words"},
        {"source_state": "é" * 65},
        {"source_state": "x" * 129},
        {"source_state": "idle\x00"},
        {"source_asset_sha256": []},
        {"source_asset_sha256": ["d" * 64, "c" * 64]},
        {"source_asset_sha256": ["c" * 64, "c" * 64]},
        {"source_asset_sha256": [f"{i:064x}" for i in range(257)]},
        {"source_asset_sha256": [1]},
        {"fallback": 0},
        {"converted_at": "2026-09-07T12:30:00Z"},
        {"converted_at": "2026-09-07T12:30:00+01:00"},
        {"converted_at": "2026-02-30T12:30:00+00:00"},
        {"converted_at": "2026-09-07T12:30:00.000000+00:00"},
        {"output_sha256": "e" * 64},
    ],
)
def test_invalid_lineage_fails_closed(changes):
    with pytest.raises(ValueError):
        artwork_context({NAMESPACE: lineage(**changes)}, expected_sha256=DIGEST)


def test_pack_lineage_is_rejected_and_asset_count_union_is_bounded():
    with pytest.raises(ValueError):
        artwork_context({NAMESPACE: lineage()})
    assets = {f"custom:a{i}": (DIGEST, {NAMESPACE: lineage()}) for i in range(129)}
    with pytest.raises(ValueError):
        encode_artwork_attribution({}, assets)


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(version=1),
        lambda value: value.update(conversions={}),
        lambda value: value["conversions"]["neutral"].update(output_sha256="e" * 64),
        lambda value: value["conversions"].update(missing=lineage()),
    ],
)
def test_malformed_v2_carrier_rejected(change):
    value = {
        "version": 2,
        "pack": None,
        "assets": {},
        "conversions": {"neutral": lineage()},
    }
    change(value)
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        decode_artwork_attribution(data, {"neutral": DIGEST})
