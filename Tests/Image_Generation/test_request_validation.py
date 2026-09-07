import threading
import warnings
import zlib
from io import BytesIO

import pytest
from PIL import Image, PngImagePlugin

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


def _png_with_dimensions(width, height):
    encoded = bytearray(_image_bytes())
    encoded[16:20] = width.to_bytes(4, "big")
    encoded[20:24] = height.to_bytes(4, "big")
    encoded[29:33] = (zlib.crc32(encoded[12:29]) & 0xFFFFFFFF).to_bytes(4, "big")
    return bytes(encoded)


class _ExplosiveBytes(bytes):
    def __bool__(self):
        raise AssertionError("bytes subclass truthiness must not be evaluated")

    def __len__(self):
        raise AssertionError("bytes subclass length must not be evaluated")


class _LyingBytes(bytes):
    def __len__(self):
        return 1


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


@pytest.mark.parametrize("content", ["not-bytes", object()])
def test_reference_image_non_bytes_content_is_refused(rv, content):
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=content, bytes_len=1),
    }

    issues = rv.validate_image_generation_request(bad)

    assert issues == [
        rv.ImageGenerationValidationIssue(
            code="image_params_invalid",
            message="reference image content must be bytes",
            path="reference_image",
        )
    ]


def test_reference_image_plain_builtin_bytes_remain_valid(rv):
    content = _image_bytes()
    ok = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=content, bytes_len=len(content)),
    }

    assert type(content) is bytes
    assert rv.validate_image_generation_request(ok) == []


@pytest.mark.parametrize("content_factory", [memoryview, lambda value: _ExplosiveBytes(value)])
def test_reference_image_requires_exact_builtin_bytes(rv, content_factory):
    content = content_factory(_image_bytes())
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=content, bytes_len=1),
    }

    issues = rv.validate_image_generation_request(bad)

    assert issues == [
        rv.ImageGenerationValidationIssue(
            code="image_params_invalid",
            message="reference image content must be bytes",
            path="reference_image",
        )
    ]


def test_lying_bytes_subclass_cannot_bypass_byte_cap(rv, monkeypatch):
    builtin_content = _image_bytes()
    monkeypatch.setattr(rv, "IMAGE_GEN_REFERENCE_MAX_BYTES", len(builtin_content))
    content = _LyingBytes(builtin_content + b"padding")
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=content, bytes_len=1),
    }

    issues = rv.validate_image_generation_request(bad)

    assert issues == [
        rv.ImageGenerationValidationIssue(
            code="image_params_invalid",
            message="reference image content must be bytes",
            path="reference_image",
        )
    ]


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
    buffer = BytesIO()
    Image.effect_noise((64, 64), 100).convert("RGB").save(buffer, format="PNG")
    content = buffer.getvalue()[:-25]
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=content, bytes_len=len(content), width=64, height=64),
    }

    assert "reference image could not be decoded" in _messages(
        rv.validate_image_generation_request(bad)
    )


def test_reference_image_decompression_bomb_error_is_sanitized(rv):
    assert Image.MAX_IMAGE_PIXELS is not None
    width = 100_000
    height = (int(Image.MAX_IMAGE_PIXELS) * 3 // width) + 1
    content = _png_with_dimensions(width, height)
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=content, bytes_len=len(content)),
    }

    assert "reference image exceeds safe decode limits" in _messages(
        rv.validate_image_generation_request(bad)
    )


def test_reference_image_warning_band_ceiling_is_sanitized_before_load(
    rv,
    monkeypatch,
):
    from types import SimpleNamespace

    warning_ceiling = rv.PILLOW_DECOMPRESSION_WARNING_MAX_PIXELS
    assert warning_ceiling == Image.MAX_IMAGE_PIXELS
    assert warning_ceiling > 0
    width = 100_000
    height = warning_ceiling // width + 1
    decoded_pixels = width * height
    assert warning_ceiling < decoded_pixels <= 2 * warning_ceiling
    content = _png_with_dimensions(width, height)
    config = SimpleNamespace(
        max_prompt_length=10_000,
        max_width=width + 1,
        max_height=height + 1,
        max_pixels=decoded_pixels + 1,
        max_steps=100,
    )
    load_calls = []

    def spy_load(*args, **kwargs):
        load_calls.append(True)

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", spy_load)
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(
            content=content,
            bytes_len=len(content),
            width=None,
            height=None,
        ),
    }

    issues = rv.validate_image_generation_request(bad, config=config)

    assert load_calls == []
    assert issues == [
        rv.ImageGenerationValidationIssue(
            code="image_params_invalid",
            message="reference image exceeds safe decode limits",
            path="reference_image",
        )
    ]


def test_reference_image_external_bomb_warning_is_sanitized(rv):
    from types import SimpleNamespace

    warning_ceiling = Image.MAX_IMAGE_PIXELS
    assert type(warning_ceiling) is int
    width = 100_000
    height = warning_ceiling // width + 1
    decoded_pixels = width * height
    content = _png_with_dimensions(width, height)
    config = SimpleNamespace(
        max_prompt_length=10_000,
        max_width=width + 1,
        max_height=height + 1,
        max_pixels=decoded_pixels + 1,
        max_steps=100,
    )
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(
            content=content,
            bytes_len=len(content),
            width=None,
            height=None,
        ),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        issues = rv.validate_image_generation_request(bad, config=config)

    assert issues == [
        rv.ImageGenerationValidationIssue(
            code="image_params_invalid",
            message="reference image exceeds safe decode limits",
            path="reference_image",
        )
    ]


def test_reference_validation_does_not_mutate_warning_filters_during_concurrent_warning(
    rv,
    monkeypatch,
):
    content = _image_bytes()
    entered_load = threading.Event()
    release_load = threading.Event()
    original_load = PngImagePlugin.PngImageFile.load
    worker_errors = []
    worker_issues = []

    def blocking_load(image, *args, **kwargs):
        entered_load.set()
        if not release_load.wait(2):
            raise AssertionError("validation load did not resume")
        return original_load(image, *args, **kwargs)

    def validate_reference():
        try:
            worker_issues.extend(
                rv.validate_image_generation_request(
                    {
                        "backend": "fal",
                        "prompt": "cat",
                        "extra_params": {},
                        "reference_image": _ref(content=content, bytes_len=len(content)),
                    }
                )
            )
        except BaseException as exc:
            worker_errors.append(exc)

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", blocking_load)
    original_filters = list(warnings.filters)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", Image.DecompressionBombWarning)
        expected_filters = list(warnings.filters)
        thread = threading.Thread(target=validate_reference)
        thread.start()
        assert entered_load.wait(2)
        filters_changed = warnings.filters != expected_filters
        unrelated_error = None
        try:
            warnings.warn("unrelated Pillow warning", Image.DecompressionBombWarning)
        except BaseException as exc:
            unrelated_error = exc
        finally:
            release_load.set()
            thread.join(2)

        assert not thread.is_alive()
        assert filters_changed is False
        assert unrelated_error is None
        assert worker_errors == []
        assert worker_issues == []
        assert len(caught) == 1
        assert warnings.filters == expected_filters

    assert warnings.filters == original_filters


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


@pytest.mark.parametrize(
    ("field", "dimension"),
    [("width", 0), ("width", 9000), ("height", 0), ("height", 9000)],
    ids=["width-zero", "width-over-cap", "height-zero", "height-over-cap"],
)
def test_reference_image_declared_dimensions_are_bounded(rv, field, dimension):
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(**{field: dimension}),
    }

    assert f"reference image {field} out of range" in _messages(
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


def _assert_header_cap_precedes_load(
    rv,
    monkeypatch,
    *,
    size,
    max_width,
    max_height,
    max_pixels,
    expected_message,
):
    from types import SimpleNamespace

    content = _image_bytes(size=size)
    config = SimpleNamespace(
        max_prompt_length=10_000,
        max_width=max_width,
        max_height=max_height,
        max_pixels=max_pixels,
        max_steps=100,
    )
    load_calls = []
    original_load = PngImagePlugin.PngImageFile.load

    def spy_load(image, *args, **kwargs):
        load_calls.append(True)
        return original_load(image, *args, **kwargs)

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", spy_load)
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(
            content=content,
            bytes_len=len(content),
            width=None,
            height=None,
        ),
    }

    issues = rv.validate_image_generation_request(bad, config=config)

    assert expected_message in _messages(issues)
    assert load_calls == []


def test_reference_image_over_width_cap_is_rejected_before_decode_load(rv, monkeypatch):
    _assert_header_cap_precedes_load(
        rv,
        monkeypatch,
        size=(4, 2),
        max_width=3,
        max_height=10,
        max_pixels=100,
        expected_message="reference image width out of range",
    )


def test_reference_image_over_height_cap_is_rejected_before_decode_load(rv, monkeypatch):
    _assert_header_cap_precedes_load(
        rv,
        monkeypatch,
        size=(2, 4),
        max_width=10,
        max_height=3,
        max_pixels=100,
        expected_message="reference image height out of range",
    )


def test_reference_image_over_pixel_cap_is_rejected_before_decode_load(rv, monkeypatch):
    _assert_header_cap_precedes_load(
        rv,
        monkeypatch,
        size=(4, 4),
        max_width=10,
        max_height=10,
        max_pixels=15,
        expected_message="reference image dimensions exceed max pixels",
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
