"""Strict shared shape boundary for untrusted Canvas bridge requests."""

from collections.abc import Iterator, Mapping
from types import MappingProxyType

import pytest
from pydantic import ValidationError as PydanticValidationError

from tldw_chatbook.Utils import input_validation


def _shared_validator():
    validator = getattr(input_validation, "validate_canvas_bridge_wire", None)
    assert callable(validator), "shared Canvas bridge wire validator is required"
    return validator


class _OverfullNonIterableMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError(f"overfull mapping must not be read: {key}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("overfull mapping must not be iterated")

    def __len__(self) -> int:
        return 5


def test_shared_canvas_bridge_wire_accepts_mapping_without_coercion_and_freezes():
    request_id = "PRIVATE_REQUEST_ID_CANARY"
    value = {"PRIVATE_VALUE_CANARY": True}
    envelope = _shared_validator()(
        MappingProxyType(
            {
                "version": "canvas-v1",
                "request_id": request_id,
                "kind": "submit",
                "value": value,
            }
        )
    )

    assert envelope.version == "canvas-v1"
    assert envelope.request_id == request_id
    assert envelope.kind == "submit"
    assert envelope.value == value
    assert request_id not in repr(envelope)
    assert "PRIVATE_VALUE_CANARY" not in repr(envelope)
    with pytest.raises(PydanticValidationError):
        envelope.kind = "download"


def test_shared_canvas_bridge_wire_rejects_overfull_mapping_before_iteration():
    with pytest.raises(ValueError, match="unknown fields") as failed:
        _shared_validator()(_OverfullNonIterableMapping())

    assert "overfull mapping" not in str(failed.value)
    assert failed.value.__cause__ is None


@pytest.mark.parametrize(
    ("wire", "reason"),
    [
        (
            {
                "version": "canvas-v1",
                "request_id": "request-extra",
                "kind": "submit",
                "value": "safe",
                "PRIVATE_UNKNOWN_CANARY": "PRIVATE_VALUE_CANARY",
            },
            "unknown fields",
        ),
        (
            {
                "version": "canvas-v1",
                "request_id": "request-missing",
                "kind": "submit",
            },
            "missing fields",
        ),
        (
            {
                "version": 1,
                "request_id": "request-wrong-type",
                "kind": "submit",
                "value": "safe",
            },
            "fields are invalid",
        ),
        (
            {
                "version": "canvas-v1",
                "request_id": 7,
                "kind": "submit",
                "value": "safe",
            },
            "fields are invalid",
        ),
        (
            {
                "version": "canvas-v1",
                "request_id": "request-wrong-kind-type",
                "kind": b"submit",
                "value": "safe",
            },
            "fields are invalid",
        ),
    ],
)
def test_shared_canvas_bridge_wire_rejects_shape_with_fixed_private_errors(
    wire,
    reason,
):
    with pytest.raises(ValueError, match=reason) as failed:
        _shared_validator()(wire)

    rendered = f"{failed.value!s} {failed.value!r}"
    assert "PRIVATE_UNKNOWN_CANARY" not in rendered
    assert "PRIVATE_VALUE_CANARY" not in rendered
    assert failed.value.__cause__ is None
    assert failed.value.__suppress_context__ is True
