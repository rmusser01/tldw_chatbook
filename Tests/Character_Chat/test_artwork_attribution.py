"""The public carrier excludes profile authority and never truncates notices."""

import json

import pytest

from tldw_chatbook.Character_Chat.artwork_attribution import (
    artwork_context,
    decode_artwork_attribution,
    encode_artwork_attribution,
)


def record(**changes):
    return {
        "version": 1,
        "creator": "tldw-project",
        "license": None,
        "source_url": "https://example.com/artwork",
        "notices": "Original notice\n" * 400,
        **changes,
    }


def test_only_known_public_records_leave_profile_and_notices_are_exact():
    pack = record()
    asset = record(output_sha256="a" * 64)
    encoded = encode_artwork_attribution(
        {"tldw/artwork": pack, "local_path": "/private/profile", "token": "secret"},
        {"neutral": ("a" * 64, {"tldw/artwork": asset, "retained_asset_id": 1})},
    )
    result = decode_artwork_attribution(encoded, {"neutral": "a" * 64})
    assert result == {"version": 1, "pack": pack, "assets": {"neutral": asset}}
    assert b"private" not in encoded and b"secret" not in encoded
    assert len(result["pack"]["notices"]) > 4096
    assert encode_artwork_attribution({}, {}) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"version": 2},
        {"extra": "private"},
        {"notices": "x" * 65537},
        {"creator": "é" * 257},
        {"source_url": "file:///private/foo"},
        {"source_url": "https://user:password@example.com/x"},
        {"source_url": "https://example.com/x?token=secret"},
        {"source_url": "https://localhost/x"},
        {"source_url": "https://127.0.0.1/x"},
        {"source_url": "https://[::1]/x"},
        {"source_url": "https://host.local/x"},
        {"source_url": "https://example.com:8080/x"},
        {"source_url": "https://example.com/#token"},
    ],
)
def test_invalid_public_records_fail_closed(changes):
    with pytest.raises(ValueError, match="artwork_attribution_invalid"):
        artwork_context({"tldw/artwork": record(**changes)})


def test_asset_record_is_bound_to_exact_output_bytes():
    with pytest.raises(ValueError):
        artwork_context(
            {"tldw/artwork": record(output_sha256="a" * 64)}, expected_sha256="b" * 64
        )
    with pytest.raises(ValueError):
        artwork_context({"tldw/artwork": record()}, expected_sha256="a" * 64)


@pytest.mark.parametrize(
    "data", [b"x" * (512 * 1024 + 1), b'{"version":1,"version":1}', b"[]"]
)
def test_invalid_or_oversized_carrier_rejected(data):
    with pytest.raises(ValueError):
        decode_artwork_attribution(data, {})


def test_asset_count_and_total_notice_budget_are_independent():
    assets = {
        f"custom:a{i}": ("a" * 64, {"tldw/artwork": record(output_sha256="a" * 64)})
        for i in range(129)
    }
    with pytest.raises(ValueError):
        encode_artwork_attribution({}, assets)
    large = record(notices="x" * 65536, output_sha256="a" * 64)
    with pytest.raises(ValueError):
        encode_artwork_attribution(
            {}, {f"custom:a{i}": ("a" * 64, {"tldw/artwork": large}) for i in range(9)}
        )


def test_carrier_rejects_unknown_expressions_and_noncanonical_json():
    data = encode_artwork_attribution(
        {}, {"neutral": ("a" * 64, {"tldw/artwork": record(output_sha256="a" * 64)})}
    )
    with pytest.raises(ValueError):
        decode_artwork_attribution(data, {"happy": "a" * 64})
    with pytest.raises(ValueError):
        decode_artwork_attribution(
            json.dumps(json.loads(data), indent=2).encode(), {"neutral": "a" * 64}
        )
