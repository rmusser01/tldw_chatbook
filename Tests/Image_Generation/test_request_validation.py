from io import BytesIO

import pytest
from PIL import Image

from tldw_chatbook.Image_Generation.capabilities import ResolvedReferenceImage

@pytest.fixture
def rv():
    from tldw_chatbook.Image_Generation import request_validation as m
    return m

def _codes(issues):
    return {i.path for i in issues}

def _messages(issues):
    return {i.message for i in issues}


def _image_bytes(*, image_format="PNG", mode="RGB", size=(2, 2)):
    buffer = BytesIO()
    Image.new(mode, size).save(buffer, format=image_format)
    return buffer.getvalue()


def _ref(**overrides):
    content = _image_bytes()
    defaults = dict(
        file_id=1,
        filename="ref.png",
        mime_type="image/png",
        width=2,
        height=2,
        bytes_len=len(content),
        content=content,
        temp_path=None,
    )
    defaults.update(overrides)
    return ResolvedReferenceImage(**defaults)

def test_valid_request_has_no_issues(rv):
    ok = {"backend": "swarmui", "prompt": "cat", "width": 512, "height": 512, "steps": 20, "cfg_scale": 7.0, "extra_params": {}}
    assert rv.validate_image_generation_request(ok) == []

def test_oversize_dimensions_flagged(rv):
    bad = {"backend": "swarmui", "prompt": "cat", "width": 9000, "height": 9000, "extra_params": {}}
    issues = rv.validate_image_generation_request(bad)
    assert any("width" in p for p in _codes(issues))

def test_negative_cfg_scale_flagged(rv):
    bad = {"backend": "swarmui", "prompt": "cat", "cfg_scale": -1.0, "extra_params": {}}
    assert any("cfg_scale" in p for p in _codes(rv.validate_image_generation_request(bad)))

def test_extra_params_not_in_allowlist_flagged(rv):
    bad = {"backend": "swarmui", "prompt": "cat", "extra_params": {"totally_unknown": 1}}
    issues = rv.validate_image_generation_request(bad)
    assert any("extra_params" in p for p in _codes(issues))


def test_reference_image_absent_key_no_issues(rv):
    # Existing callers that never populate "reference_image" at all must be
    # completely unaffected by the new choke-point checks.
    ok = {"backend": "swarmui", "prompt": "cat", "extra_params": {}}
    assert rv.validate_image_generation_request(ok) == []


def test_reference_image_none_no_issues(rv):
    ok = {"backend": "swarmui", "prompt": "cat", "extra_params": {}, "reference_image": None}
    assert rv.validate_image_generation_request(ok) == []


@pytest.mark.parametrize(
    "backend",
    ["stable_diffusion_cpp", "swarmui", "openrouter", "novita", "together", "modelstudio"],
)
def test_reference_image_refused_for_legacy_backends(rv, backend):
    bad = {"backend": backend, "prompt": "cat", "extra_params": {}, "reference_image": _ref()}
    issues = rv.validate_image_generation_request(bad)
    assert issues == [
        rv.ImageGenerationValidationIssue(
            code="image_params_invalid",
            message=f"backend {backend!r} does not support reference images",
            path="reference_image",
        )
    ]


@pytest.mark.parametrize("backend", ["fal", "gemini"])
def test_reference_image_accepted_for_new_backends(rv, backend):
    ok = {"backend": backend, "prompt": "cat", "extra_params": {}, "reference_image": _ref()}
    assert rv.validate_image_generation_request(ok) == []


def test_reference_image_webp_accepted(rv):
    content = _image_bytes(image_format="WEBP")
    ok = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(
            filename="ref.webp",
            mime_type="image/webp",
            content=content,
            bytes_len=len(content),
        ),
    }
    assert rv.validate_image_generation_request(ok) == []


def test_reference_image_gif_refused(rv):
    bad = {"backend": "fal", "prompt": "cat", "extra_params": {}, "reference_image": _ref(mime_type="image/gif")}
    issues = rv.validate_image_generation_request(bad)
    assert "reference image mime 'image/gif' is not supported (png/jpeg/webp)" in _messages(issues)


def test_reference_image_oversize_refused(rv):
    # Real oversized content, with bytes_len reported honestly -- the size
    # cap must fire based on the actual content, not merely a claimed field.
    image = _image_bytes()
    big_content = image + b"x" * (rv.IMAGE_GEN_REFERENCE_MAX_BYTES + 1 - len(image))
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=big_content, bytes_len=len(big_content)),
    }
    issues = rv.validate_image_generation_request(bad)
    assert "reference image exceeds the 10MB limit" in _messages(issues)


def test_reference_image_oversized_content_with_lying_bytes_len_refused(rv):
    # task-686 choke-point hardening: a constructor that reports a tiny
    # bytes_len while content is actually oversized must NOT bypass the cap
    # -- the cap validates len(content), never the caller-supplied bytes_len.
    image = _image_bytes()
    big_content = image + b"x" * (rv.IMAGE_GEN_REFERENCE_MAX_BYTES + 1 - len(image))
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=big_content, bytes_len=4),
    }
    issues = rv.validate_image_generation_request(bad)
    assert "reference image exceeds the 10MB limit" in _messages(issues)


def test_reference_image_at_exact_cap_not_refused(rv):
    image = _image_bytes()
    ok_content = image + b"x" * (rv.IMAGE_GEN_REFERENCE_MAX_BYTES - len(image))
    ok = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=ok_content, bytes_len=len(ok_content)),
    }
    assert rv.validate_image_generation_request(ok) == []


def test_reference_image_no_content_refused(rv):
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=None, temp_path="/tmp/ref.png"),
    }
    issues = rv.validate_image_generation_request(bad)
    assert "reference image has no content bytes" in _messages(issues)


def test_reference_image_empty_content_refused(rv):
    # task-686 choke-point hardening: content == b"" is refused with the
    # same issue string as content is None, not treated as a (tiny) valid
    # payload.
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=b"", bytes_len=0),
    }
    issues = rv.validate_image_generation_request(bad)
    assert "reference image has no content bytes" in _messages(issues)


def test_reference_image_multiple_problems_all_reported_no_content_variant(rv):
    # Unsupported backend + bad mime + no content, all at once -- the checks
    # must not short-circuit each other. (Oversize and no-content are now
    # mutually exclusive states of the same `content` field -- see the
    # oversize variant below for the sibling case.)
    bad = {
        "backend": "swarmui",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(
            mime_type="image/gif",
            content=None,
            temp_path="/tmp/ref.gif",
        ),
    }
    issues = rv.validate_image_generation_request(bad)
    messages = _messages(issues)
    assert "backend 'swarmui' does not support reference images" in messages
    assert "reference image mime 'image/gif' is not supported (png/jpeg/webp)" in messages
    assert "reference image has no content bytes" in messages
    assert len(issues) == 3


def test_reference_image_multiple_problems_all_reported_oversize_variant(rv):
    # Sibling of the above: unsupported backend + bad mime + oversize
    # content, all at once.
    image = _image_bytes()
    big_content = image + b"x" * (rv.IMAGE_GEN_REFERENCE_MAX_BYTES + 1 - len(image))
    bad = {
        "backend": "swarmui",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(
            mime_type="image/gif",
            content=big_content,
            bytes_len=len(big_content),
        ),
    }
    issues = rv.validate_image_generation_request(bad)
    messages = _messages(issues)
    assert "backend 'swarmui' does not support reference images" in messages
    assert "reference image mime 'image/gif' is not supported (png/jpeg/webp)" in messages
    assert "reference image exceeds the 10MB limit" in messages
    assert len(issues) == 3


def test_reference_image_declared_mime_must_match_signature(rv):
    content = _image_bytes(image_format="JPEG")
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=content, bytes_len=len(content)),
    }

    assert "reference image mime does not match image content" in _messages(
        rv.validate_image_generation_request(bad)
    )


def test_reference_image_truncated_decode_is_refused(rv):
    encoded = _image_bytes(size=(8, 8))
    content = encoded[: len(encoded) // 2]
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=content, bytes_len=len(content), width=8, height=8),
    }

    assert "reference image could not be decoded" in _messages(
        rv.validate_image_generation_request(bad)
    )


def test_reference_image_decompression_bomb_warning_is_refused(rv, monkeypatch):
    content = _image_bytes(size=(2, 2))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 2)
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=content, bytes_len=len(content)),
    }

    assert "reference image exceeds safe decode limits" in _messages(
        rv.validate_image_generation_request(bad)
    )


def test_reference_image_unsupported_mode_is_refused(rv):
    content = _image_bytes(image_format="TIFF", mode="I")
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=content, bytes_len=len(content)),
    }

    assert "reference image mode is not supported" in _messages(
        rv.validate_image_generation_request(bad)
    )


@pytest.mark.parametrize("dimension", [0, 9000])
def test_reference_image_declared_dimensions_are_bounded(rv, dimension):
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(width=dimension),
    }

    assert "reference image width out of range" in _messages(
        rv.validate_image_generation_request(bad)
    )


def test_reference_image_decoded_pixel_cap_is_enforced(rv):
    from types import SimpleNamespace

    content = _image_bytes(size=(3, 3))
    config = SimpleNamespace(
        max_prompt_length=10_000,
        max_width=10,
        max_height=10,
        max_pixels=8,
        max_steps=100,
    )
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(
            content=content,
            bytes_len=len(content),
            width=3,
            height=3,
        ),
    }

    assert "reference image dimensions exceed max pixels" in _messages(
        rv.validate_image_generation_request(bad, config=config)
    )


def test_comfyui_reference_temp_path_is_refused_without_opening(rv, monkeypatch):
    def fail_open(*args, **kwargs):
        raise AssertionError("reference path must never be opened")

    monkeypatch.setattr("builtins.open", fail_open)
    bad = {
        "backend": "comfyui",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=None, temp_path="sentinel-file-path"),
    }

    assert "ComfyUI reference image must use in-memory content" in _messages(
        rv.validate_image_generation_request(bad)
    )
