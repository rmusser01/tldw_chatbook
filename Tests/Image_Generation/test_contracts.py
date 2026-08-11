from collections.abc import Mapping
from typing import get_args, get_origin, get_type_hints

import pytest


def test_request_and_result_dataclasses():
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
    req = ImageGenRequest(
        backend="swarmui", prompt="a red dragon", negative_prompt=None,
        width=512, height=512, steps=20, cfg_scale=7.0, seed=-1,
        sampler=None, model=None, format="png", extra_params={},
    )
    assert req.backend == "swarmui"
    assert req.reference_image is None  # default
    assert req.cancel_event is None
    res = ImageGenResult(content=b"\x89PNG", content_type="image/png", bytes_len=4)
    assert res.bytes_len == 4
    assert res.effective_params is None


def test_image_result_effective_params_is_an_optional_mapping():
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenResult

    params = {"operation": "edit", "steps": 20}
    result = ImageGenResult(
        content=b"\x89PNG",
        content_type="image/png",
        bytes_len=4,
        effective_params=params,
    )
    annotation = get_type_hints(ImageGenResult)["effective_params"]
    outer_args = get_args(annotation)
    mapping_member = next(member for member in outer_args if get_origin(member) is Mapping)
    key_type, value_type = get_args(mapping_member)

    assert result.effective_params is params
    assert len(outer_args) == 2
    assert type(None) in outer_args
    assert key_type is str
    assert len(get_args(value_type)) == 5
    assert set(get_args(value_type)) == {str, int, float, bool, type(None)}


def test_image_generation_cancellation_and_comfyui_errors_are_typed_and_sanitized():
    from tldw_chatbook.Image_Generation.exceptions import (
        ComfyUIImageEditError,
        ImageGenerationCancelled,
        ImageGenerationError,
    )

    assert issubclass(ImageGenerationCancelled, ImageGenerationError)
    error = ComfyUIImageEditError("source_upload")
    assert isinstance(error, ImageGenerationError)
    assert error.phase == "source_upload"
    assert str(error) == "The source image could not be uploaded. Please try again."
    with pytest.raises(ValueError, match="unknown image-edit failure phase"):
        ComfyUIImageEditError("server-body-or-path")

def test_resolved_reference_image_defined_locally():
    # Must be defined in capabilities.py, NOT imported from reference_images (which we dropped)
    from tldw_chatbook.Image_Generation.capabilities import ResolvedReferenceImage
    r = ResolvedReferenceImage(
        file_id=1, filename=None, mime_type="image/png",
        width=None, height=None, bytes_len=3, content=b"abc", temp_path=None,
    )
    assert r.mime_type == "image/png"

def test_adapter_is_structural_protocol():
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenerationAdapter
    from typing import Protocol
    # It is a Protocol; a duck-typed object with name/supported_formats/generate satisfies it structurally.
    assert issubclass(ImageGenerationAdapter, Protocol)
