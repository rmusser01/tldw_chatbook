def test_request_and_result_dataclasses():
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
    req = ImageGenRequest(
        backend="swarmui", prompt="a red dragon", negative_prompt=None,
        width=512, height=512, steps=20, cfg_scale=7.0, seed=-1,
        sampler=None, model=None, format="png", extra_params={},
    )
    assert req.backend == "swarmui"
    assert req.reference_image is None  # default
    res = ImageGenResult(content=b"\x89PNG", content_type="image/png", bytes_len=4)
    assert res.bytes_len == 4

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
