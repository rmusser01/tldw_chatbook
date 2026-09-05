"""Behavioral tests for the Canvas V1 runtime contract ceilings."""

from __future__ import annotations

import base64
import hashlib

import pytest

from tldw_chatbook.Canvas.limits import (
    MAX_WIRE_INTEGER,
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
    CanvasSourceIdentity,
    RenderAsset,
    RenderNode,
)


@pytest.mark.parametrize(
    ("mime_type", "extension", "signature"),
    [
        ("image/png", ".png", b"\x89PNG\r\n\x1a\n"),
        ("image/jpeg", ".jpg", b"\xff\xd8\xff"),
        ("image/gif", ".gif", b"GIF89a"),
        ("image/webp", ".webp", b"RIFF\x00\x00\x00\x00WEBP"),
    ],
)
def test_generated_raster_download_requires_declared_binary_signature(
    mime_type: str,
    extension: str,
    signature: bytes,
) -> None:
    accepted = CanvasBridgeRequest.from_wire(
        {
            "version": "canvas-v1",
            "request_id": "download-signed-image",
            "kind": "download",
            "value": {
                "filename": f"pixel{extension}",
                "mime_type": mime_type,
                "data": "data:"
                + mime_type
                + ";base64,"
                + base64.b64encode(signature).decode("ascii"),
            },
        }
    )
    assert accepted.download_payload().data == signature

    forged = base64.b64encode(b"<html><script>bad()</script>").decode("ascii")
    with pytest.raises(ValueError, match="signature"):
        CanvasBridgeRequest.from_wire(
            {
                "version": "canvas-v1",
                "request_id": "download-forged-image",
                "kind": "download",
                "value": {
                    "filename": f"pixel{extension}",
                    "mime_type": mime_type,
                    "data": f"data:{mime_type};base64,{forged}",
                },
            }
        )


@pytest.mark.parametrize(
    "filename",
    ["\nreport.txt", "report\tfinal.txt", "report.txt\r"],
)
def test_generated_download_rejects_raw_filename_controls_before_trimming(
    filename: str,
) -> None:
    with pytest.raises(ValueError, match="unsafe characters"):
        CanvasBridgeRequest.from_wire(
            {
                "version": "canvas-v1",
                "request_id": "download-control-name",
                "kind": "download",
                "value": {
                    "filename": filename,
                    "mime_type": "text/plain",
                    "data": "safe",
                },
            }
        )


def test_runtime_limits_have_the_documented_canvas_v1_values() -> None:
    limits = CanvasLimits()

    assert limits.html_bytes == 512 * 1024
    assert limits.asset_bytes == 1024 * 1024
    assert limits.aggregate_asset_bytes == 4 * 1024 * 1024
    assert limits.dom_nodes == 1_800
    assert limits.css_rules == 900
    assert limits.script_bytes == 256 * 1024
    assert limits.runtime_memory_bytes == 32 * 1024 * 1024
    assert limits.stack_bytes == 512 * 1024
    assert limits.startup_milliseconds == 250
    assert limits.event_milliseconds == 50
    assert limits.patches_per_event == 500
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


@pytest.mark.parametrize(
    "assets",
    [(), (decode_data_url("data:text/plain;base64,YQ==", field_name="asset"),)],
)
@pytest.mark.parametrize("field", ["per_asset_limit", "aggregate_limit"])
@pytest.mark.parametrize("invalid", [-1, MAX_WIRE_INTEGER + 1, True])
def test_asset_validation_checks_safe_wire_limits_before_iteration(
    assets, field: str, invalid: int
) -> None:
    limits = {"per_asset_limit": 1, "aggregate_limit": 1}
    limits[field] = invalid

    with pytest.raises(CanvasLimitError):
        validate_asset_payloads(assets, **limits)


def test_data_url_rejects_malformed_base64_and_unknown_parameters() -> None:
    with pytest.raises(CanvasLimitError, match="valid base64"):
        decode_data_url("data:text/plain;base64,%%%", field_name="asset")

    with pytest.raises(CanvasLimitError, match="unsupported data URL parameter"):
        decode_data_url("data:text/plain;charset=utf-8,hello", field_name="asset")


@pytest.mark.parametrize(
    ("field_name", "count", "limit"),
    [
        ("DOM nodes", 1_800, 1_800),
        ("CSS rules", 900, 900),
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
        ("DOM nodes", 1_801, 1_800),
        ("CSS rules", 901, 900),
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


def test_source_identity_preserves_exact_utf8_size_and_full_lowercase_sha256() -> None:
    source = "<!doctype html><main>é</main>"

    identity = CanvasSourceIdentity.from_source(source)

    assert identity.source_bytes == len(source.encode("utf-8"))
    assert identity.sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert identity.sha256 == identity.sha256.lower()
    assert len(identity.sha256) == 64
    identity.verify_source(source)

    with pytest.raises(CanvasLimitError, match="does not match source"):
        identity.verify_source("<!doctype html><main>different</main>")

    with pytest.raises(TypeError):
        CanvasSourceIdentity(source_bytes=identity.source_bytes, sha256=identity.sha256)  # type: ignore[call-arg]


def test_render_plan_rejects_aggregate_text_over_html_ceiling() -> None:
    source_identity = CanvasSourceIdentity.from_source("<main></main>")
    child_text = "x" * (300 * 1024)
    root = RenderNode(
        node_id="root",
        tag="main",
        children=(
            RenderNode(node_id="first", tag="p", text=child_text),
            RenderNode(node_id="second", tag="p", text=child_text),
        ),
    )

    with pytest.raises(CanvasLimitError, match="render plan text exceeds 524288 UTF-8 bytes"):
        CanvasRenderPlan(runtime_profile="canvas-v1", source_identity=source_identity, root=root)


def test_render_plan_aggregate_text_accepts_the_exact_html_ceiling() -> None:
    source_identity = CanvasSourceIdentity.from_source("<main></main>")
    root = RenderNode(node_id="r", tag="x", text="x" * (512 * 1024 - 2))

    plan = CanvasRenderPlan(runtime_profile="canvas-v1", source_identity=source_identity, root=root)

    assert plan.source_identity == source_identity


def test_canvas_contract_records_are_immutable_slotted_and_validate_wire_messages() -> None:
    issue = CanvasCompatibilityIssue(code="unsupported-tag", message="Unsupported tag")
    node = RenderNode(node_id="node-1", tag="main")
    asset = RenderAsset(asset_id="asset-1", mime_type="text/plain", data=b"x")
    plan = CanvasRenderPlan(
        runtime_profile="canvas-v1",
        source_identity=CanvasSourceIdentity.from_source("<main></main>"),
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


def test_submit_text_is_preserved_and_structured_submit_is_canonical_json() -> None:
    text = CanvasBridgeRequest.from_wire(
        {
            "version": "canvas-v1",
            "request_id": "submit-text",
            "kind": "submit",
            "value": "  exact\ntext  ",
        }
    )
    structured = CanvasBridgeRequest.from_wire(
        {
            "version": "canvas-v1",
            "request_id": "submit-json",
            "kind": "submit",
            "value": {"z": 1, "a": [True, None]},
        }
    )

    assert text.submit_text() == "  exact\ntext  "
    assert structured.submit_text() == '{"a":[true,null],"z":1}'


def test_submit_request_rejects_bytes_depth_nonfinite_and_cycles() -> None:
    base = {
        "version": "canvas-v1",
        "request_id": "submit-refused",
        "kind": "submit",
    }
    with pytest.raises(ValueError, match="submit payload exceeds 3 UTF-8 bytes"):
        CanvasBridgeRequest.from_wire(
            {**base, "value": "four"},
            limits=CanvasLimits(submit_payload_bytes=3),
        )
    with pytest.raises(ValueError, match="numbers must be finite"):
        CanvasBridgeRequest.from_wire({**base, "value": float("inf")})

    too_deep: object = "leaf"
    for _ in range(17):
        too_deep = [too_deep]
    with pytest.raises(ValueError, match="exceeds JSON depth 16"):
        CanvasBridgeRequest.from_wire({**base, "value": too_deep})

    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="must not contain a cycle"):
        CanvasBridgeRequest.from_wire({**base, "value": cycle})


def test_generated_download_accepts_only_closed_passive_schema_and_decodes_images() -> None:
    text = CanvasBridgeRequest.from_wire(
        {
            "version": "canvas-v1",
            "request_id": "download-text",
            "kind": "download",
            "value": {
                "filename": " report.csv ",
                "mime_type": "text/csv",
                "data": "name,value\nalpha,1\n",
            },
        }
    ).download_payload()
    image = CanvasBridgeRequest.from_wire(
        {
            "version": "canvas-v1",
            "request_id": "download-image",
            "kind": "download",
            "value": {
                "filename": "pixel.png",
                "mime_type": "image/png",
                "data": "data:image/png;base64,iVBORw0KGgo=",
            },
        }
    ).download_payload()

    assert text.filename == "report.csv"
    assert text.data == b"name,value\nalpha,1\n"
    assert text.text_preview == "name,value\nalpha,1\n"
    assert image.filename == "pixel.png"
    assert image.data == b"\x89PNG\r\n\x1a\n"
    assert image.text_preview is None


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (
            {"filename": "../report.txt", "mime_type": "text/plain", "data": "x"},
            "path separators",
        ),
        (
            {"filename": "CON.backup.txt", "mime_type": "text/plain", "data": "x"},
            "reserved",
        ),
        (
            {"filename": "report:final.txt", "mime_type": "text/plain", "data": "x"},
            "unsafe characters",
        ),
        (
            {"filename": "report.html", "mime_type": "text/plain", "data": "x"},
            "extension does not match",
        ),
        (
            {"filename": "report.svg", "mime_type": "image/svg+xml", "data": "x"},
            "passive V1 MIME",
        ),
        (
            {
                "filename": "report.txt",
                "mime_type": "text/plain",
                "data": "x",
                "extra": True,
            },
            "unknown fields",
        ),
        (
            {
                "filename": "pixel.png",
                "mime_type": "image/png",
                "data": "data:image/jpeg;base64,iVBORw0KGgo=",
            },
            "MIME type does not match",
        ),
    ],
)
def test_generated_download_rejects_active_ambiguous_or_mismatched_values(
    value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CanvasBridgeRequest.from_wire(
            {
                "version": "canvas-v1",
                "request_id": "download-refused",
                "kind": "download",
                "value": value,
            }
        )


def test_generated_download_enforces_decoded_bytes_not_base64_text_size() -> None:
    with pytest.raises(ValueError, match="decoded bytes"):
        CanvasBridgeRequest.from_wire(
            {
                "version": "canvas-v1",
                "request_id": "download-too-large",
                "kind": "download",
                "value": {
                    "filename": "pixel.png",
                    "mime_type": "image/png",
                    "data": "data:image/png;base64,iVBORw0KGgo=",
                },
            },
            limits=CanvasLimits(download_payload_bytes=3),
        )
