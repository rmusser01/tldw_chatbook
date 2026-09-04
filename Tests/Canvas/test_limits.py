"""Behavioral tests for the Canvas V1 runtime contract ceilings."""

from __future__ import annotations

import base64

import pytest

from tldw_chatbook.Canvas.limits import (
    CanvasLimitError,
    CanvasLimits,
    decode_data_url,
    json_depth,
    utf8_byte_length,
    validate_asset_payloads,
    validate_count,
    validate_json_value,
    validate_unique_identifiers,
    validate_utf8_text,
)
from tldw_chatbook.Canvas.models import (
    CanvasBridgeRequest,
    CanvasCompatibilityIssue,
    CanvasRenderPlan,
    CanvasRuntimeFailure,
    RenderAsset,
    RenderNode,
)


def test_runtime_limits_have_the_documented_canvas_v1_values() -> None:
    limits = CanvasLimits()

    assert limits.html_bytes == 512 * 1024
    assert limits.asset_bytes == 1024 * 1024
    assert limits.aggregate_asset_bytes == 4 * 1024 * 1024
    assert limits.dom_nodes == 5_000
    assert limits.css_rules == 2_000
    assert limits.script_bytes == 256 * 1024
    assert limits.runtime_memory_bytes == 32 * 1024 * 1024
    assert limits.stack_bytes == 512 * 1024
    assert limits.startup_milliseconds == 250
    assert limits.event_milliseconds == 50
    assert limits.patches_per_event == 1_000
    assert limits.submit_payload_bytes == 16 * 1024
    assert limits.json_depth == 16
    assert limits.download_payload_bytes == 10 * 1024 * 1024


def test_utf8_validation_counts_encoded_bytes_at_the_exact_boundary() -> None:
    assert utf8_byte_length("é") == 2
    assert validate_utf8_text("é", limit=2, field_name="html") == 2

    with pytest.raises(CanvasLimitError, match="html exceeds 1 UTF-8 bytes"):
        validate_utf8_text("é", limit=1, field_name="html")


def test_utf8_validation_rejects_malformed_unicode() -> None:
    with pytest.raises(CanvasLimitError, match="valid Unicode"):
        utf8_byte_length("\ud800")


def test_json_depth_accepts_the_exact_limit_and_rejects_one_level_more() -> None:
    value: object = "leaf"
    for _ in range(16):
        value = [value]

    assert json_depth(value) == 16
    validate_json_value(value, max_depth=16, field_name="submit payload")

    with pytest.raises(CanvasLimitError, match="submit payload exceeds JSON depth 16"):
        validate_json_value([value], max_depth=16, field_name="submit payload")


def test_json_depth_rejects_non_finite_numbers_and_unsupported_values() -> None:
    with pytest.raises(CanvasLimitError, match="finite"):
        validate_json_value(float("nan"), max_depth=16, field_name="submit payload")

    with pytest.raises(CanvasLimitError, match="JSON-compatible"):
        validate_json_value({"bad": {1, 2}}, max_depth=16, field_name="submit payload")


def test_decoded_data_asset_size_uses_decoded_base64_bytes_at_boundary() -> None:
    encoded = base64.b64encode(b"abc").decode("ascii")
    asset = decode_data_url(f"data:text/plain;base64,{encoded}", field_name="asset")

    assert asset.mime_type == "text/plain"
    assert asset.data == b"abc"
    validate_asset_payloads([asset], per_asset_limit=3, aggregate_limit=3)

    with pytest.raises(CanvasLimitError, match="asset exceeds 2 decoded bytes"):
        validate_asset_payloads([asset], per_asset_limit=2, aggregate_limit=3)


def test_asset_validation_rejects_aggregate_size_over_exact_boundary() -> None:
    first = decode_data_url("data:text/plain;base64,YWI=", field_name="asset")
    second = decode_data_url("data:text/plain;base64,Y2Q=", field_name="asset")

    validate_asset_payloads([first, second], per_asset_limit=2, aggregate_limit=4)

    with pytest.raises(CanvasLimitError, match="aggregate assets exceed 3 decoded bytes"):
        validate_asset_payloads([first, second], per_asset_limit=2, aggregate_limit=3)


def test_data_url_rejects_malformed_base64_and_unknown_parameters() -> None:
    with pytest.raises(CanvasLimitError, match="valid base64"):
        decode_data_url("data:text/plain;base64,%%%", field_name="asset")

    with pytest.raises(CanvasLimitError, match="unsupported data URL parameter"):
        decode_data_url("data:text/plain;charset=utf-8,hello", field_name="asset")


@pytest.mark.parametrize(
    ("field_name", "count", "limit"),
    [
        ("DOM nodes", 5_000, 5_000),
        ("CSS rules", 2_000, 2_000),
        ("script bytes", 256 * 1024, 256 * 1024),
    ],
)
def test_count_validation_accepts_exact_ceiling(
    field_name: str, count: int, limit: int
) -> None:
    assert validate_count(count, limit=limit, field_name=field_name) == count


@pytest.mark.parametrize(
    ("field_name", "count", "limit"),
    [
        ("DOM nodes", 5_001, 5_000),
        ("CSS rules", 2_001, 2_000),
        ("script bytes", 256 * 1024 + 1, 256 * 1024),
    ],
)
def test_count_validation_rejects_one_over_ceiling(
    field_name: str, count: int, limit: int
) -> None:
    with pytest.raises(CanvasLimitError, match=f"{field_name} exceeds {limit}"):
        validate_count(count, limit=limit, field_name=field_name)


def test_count_validation_rejects_negative_boolean_and_huge_counts() -> None:
    for count in (-1, True, 2**128):
        with pytest.raises(CanvasLimitError):
            validate_count(count, limit=5_000, field_name="DOM nodes")

    with pytest.raises(CanvasLimitError, match="html_bytes exceeds the supported integer range"):
        CanvasLimits(html_bytes=2**128)


def test_duplicate_opaque_identifiers_fail_closed() -> None:
    assert validate_unique_identifiers(("node-1", "node-2"), field_name="node IDs") == (
        "node-1",
        "node-2",
    )

    with pytest.raises(CanvasLimitError, match="node IDs contains a duplicate identifier"):
        validate_unique_identifiers(("node-1", "node-1"), field_name="node IDs")


def test_canvas_contract_records_are_immutable_slotted_and_validate_wire_messages() -> None:
    issue = CanvasCompatibilityIssue(code="unsupported-tag", message="Unsupported tag")
    node = RenderNode(node_id="node-1", tag="main")
    asset = RenderAsset(asset_id="asset-1", mime_type="text/plain", data=b"x")
    plan = CanvasRenderPlan(
        runtime_profile="canvas-v1",
        root=node,
        assets=(asset,),
        compatibility_issues=(issue,),
    )
    failure = CanvasRuntimeFailure(code="timeout", message="Event exceeded limit")

    assert plan.root.node_id == "node-1"
    assert failure.code == "timeout"
    assert not hasattr(plan, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        plan.runtime_profile = "other"  # type: ignore[misc]

    request = CanvasBridgeRequest.from_wire(
        {"version": "canvas-v1", "request_id": "request-1", "kind": "submit", "value": {"ok": True}}
    )
    assert request.kind == "submit"
    assert request.value == {"ok": True}

    with pytest.raises(ValueError, match="unknown fields"):
        CanvasBridgeRequest.from_wire(
            {
                "version": "canvas-v1",
                "request_id": "request-1",
                "kind": "submit",
                "value": "ok",
                "unexpected": True,
            }
        )

    with pytest.raises(CanvasLimitError, match="unsupported Canvas bridge request version"):
        CanvasBridgeRequest(
            version="canvas-v2",  # type: ignore[arg-type]
            request_id="request-1",
            kind="submit",
            value="ok",
        )
